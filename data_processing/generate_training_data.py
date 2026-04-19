#!/usr/bin/env python3
"""
Training Data Generator
-----------------------
Takes parsed question-solution JSON pairs and calls a frontier LLM API
(Claude 3.5 Sonnet, GPT-4o, or local Ollama) to synthesize fine-tuning data.

Produces:
    - brain_train.json : sharegpt format with multimodal messages (scenes-array schema)
    - coder_train.json : sharegpt format with validated Manim CE code

Usage:
    python generate_training_data.py \
        --parsed-dir ../data/parsed \
        --output-dir ../data/training \
        --api claude                    # or "openai" or "ollama"

Prerequisites:
    Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests


# ─── Load env ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


# ─── Constants ──────────────────────────────────────────────────────────────

SAVE_EVERY = 10  # Checkpoint save interval


# ─── System Prompts ─────────────────────────────────────────────────────────

BRAIN_SYSTEM_PROMPT = """You are an expert JEE (Joint Entrance Examination) teacher and visual explainer.
Given a question (with optional diagram description), you must solve it and structure the explanation into discrete visual scenes for animation.

OUTPUT FORMAT: You MUST respond with ONLY a valid JSON object.
The JSON schema MUST be:
{
  "question_type": "physics | math | chemistry",
  "topic": "specific topic name",
  "difficulty": "easy | medium | hard",
  "scenes": [
    {
      "scene_id": "scene_01",
      "duration_estimate_sec": 7.5,
      "narration": "Teacher-style narration for this scene",
      "visual_type": "one of: equation_transform, axes_plot, free_body, projectile, circuit, ray_diagram, geometric, wave, energy_diagram, vector",
      "visual_params": {
        "description": "What to draw",
        "labels": ["label1", "label2"],
        "values": {}
      },
      "requires_codegen": false
    }
  ],
  "final_answer": "Final numerical or symbolic answer with units"
}

Rules:
- Break complex problems into 3-8 scenes
- Each scene should be a self-contained visual step
- Set requires_codegen=true ONLY for scenes that need custom Manim code beyond templates
- Be precise with physics. Show all intermediate calculations across scenes.
- Duration estimates should total 30-90 seconds for the full explanation."""

CODER_SYSTEM_PROMPT = """You are an expert Manim Community Edition (v0.18+) code generator.
Given a JSON schema describing a physics/math visualization, write complete, runnable Manim Python code.

Rules:
1. Use `from manim import *` — Community Edition only
2. Class must inherit from `Scene` and implement `construct(self)`
3. Include proper animations: Create, Write, FadeIn, Transform, etc.
4. Add LaTeX labels for all equations and values
5. Use color coding: RED for forces, BLUE for velocity, GREEN for displacement, YELLOW for energy
6. Include a title card and step-by-step animation matching the narration
7. Code must be FULLY SELF-CONTAINED — no external imports besides manim
8. Target 30-60 seconds of animation

Output ONLY the Python code, no markdown fences or explanation."""


# ─── API Clients ─────────────────────────────────────────────────────────────


def call_claude(system: str, user_message: str, max_retries: int = 3) -> Optional[str]:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return None


def call_openai(system: str, user_message: str, max_retries: int = 3) -> Optional[str]:
    """Call OpenAI GPT-4o API."""
    try:
        import openai
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return None


def call_ollama(
    system: str, user_message: str, model: str = "qwen2.5-coder:7b",
    force_json: bool = False,
) -> Optional[str]:
    """Call local Ollama API."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": user_message,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }
    if force_json:
        payload["format"] = "json"
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def call_api(
    system: str, user_message: str, api: str = "claude",
    force_json: bool = False,
) -> Optional[str]:
    """Unified API caller."""
    if api == "claude":
        return call_claude(system, user_message)
    elif api == "openai":
        return call_openai(system, user_message)
    elif api == "ollama":
        return call_ollama(system, user_message, force_json=force_json)
    else:
        raise ValueError(f"Unknown API: {api}")


# ─── Manim Validation ────────────────────────────────────────────────────────


def _validate_manim_syntax(code: str) -> tuple[bool, str]:
    """Fast syntax-only check (no rendering needed)."""
    try:
        compile(code, "<manim_scene>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def _validate_manim_render(code: str, timeout: int = 120) -> tuple[bool, str]:
    """Full render validation — tries Docker first, falls back to local manim."""
    tmp_dir = Path(__file__).parent.parent / "data" / "_tmp_manim"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=str(tmp_dir)
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Try Docker first
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{tmp_path}:/scene.py:ro",
                "manimcommunity/manim:stable",
                "manim",
                "render",
                "/scene.py",
                "-ql",
                "--disable_caching",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        # Docker failed — try local manim
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["manim", "render", tmp_path, "-ql", "--disable_caching"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(tmp_dir),
        )
        success = result.returncode == 0
        return success, "" if success else result.stderr[-500:]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def validate_manim_code(code: str, full_render: bool = False) -> tuple[bool, str]:
    """Validate Manim code: syntax check + optional full render."""
    ok, err = _validate_manim_syntax(code)
    if not ok:
        return False, err
    if full_render:
        return _validate_manim_render(code)
    return True, ""


# ─── Checkpoint / Resume ────────────────────────────────────────────────────


def _load_checkpoint(output_file: Path) -> tuple[list, set]:
    """Load existing data and return (data_list, processed_question_ids)."""
    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        processed = set()
        for entry in data:
            # Build a key from metadata
            meta = entry.get("metadata", {})
            key = f"{meta.get('institute', '')}_{meta.get('question_number', '')}_{meta.get('source', '')}"
            processed.add(key)
        print(f"  Resuming from checkpoint: {len(data)} entries already processed")
        return data, processed
    return [], set()


def _question_key(question: dict) -> str:
    """Build a unique key for a question to track processing."""
    return f"{question.get('institute', '')}_{question.get('question_number', '')}_{question.get('source', '')}"


def _save_checkpoint(data: list, output_file: Path):
    """Save data to file as a checkpoint."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Data Generation ────────────────────────────────────────────────────────

import re as _re


def _try_parse_brain_json(response: str) -> Optional[dict]:
    """Try to parse brain model JSON response with multiple fallback strategies.

    Returns parsed dict with 'scenes' array, or None if parsing fails.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Strip markdown fences
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Strategy 1: Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "scenes" in parsed:
            return parsed
        if isinstance(parsed, dict):
            return parsed  # Accept even without scenes
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON object with regex
    obj_match = _re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: Fix common issues (trailing commas)
    fixed = _re.sub(r',\s*([}\]])', r'\1', text)
    open_braces = fixed.count('{') - fixed.count('}')
    fixed += '}' * max(0, open_braces)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return None


def generate_brain_entry(
    question: dict,
    solution: Optional[dict],
    api: str,
    image_path: Optional[str] = None,
) -> Optional[dict]:
    """Generate a brain training entry in sharegpt format.

    If image_path is provided, creates a multimodal user message (for VL models).
    """
    user_text = (
        f"Question #{question['question_number']} ({question['institute'].title()}):\n"
    )
    user_text += question["text"]
    if solution:
        user_text += f"\n\nReference solution:\n{solution['solution_text']}"

    response = call_api(BRAIN_SYSTEM_PROMPT, user_text, api, force_json=True)
    if response is None:
        return None

    # Validate that the response is valid JSON with the scenes array
    parsed = _try_parse_brain_json(response)
    if parsed is None:
        print("  ⚠ Response not valid JSON, retrying...")
        response = call_api(BRAIN_SYSTEM_PROMPT, user_text, api, force_json=True)
        if response:
            parsed = _try_parse_brain_json(response)
            if parsed is None:
                print("  ✗ Retry also invalid JSON. Skipping.")
                return None
        else:
            return None

    # ── Build sharegpt format ──
    # Multimodal user message if image is available
    if image_path and os.path.exists(image_path):
        user_content = [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    return {
        "messages": [
            {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "question_number": question["question_number"],
            "institute": question["institute"],
            "source": question["source"],
        },
    }


def _strip_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM output."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python") :].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def generate_coder_entry(
    brain_entry: dict, api: str, full_render: bool = False, max_attempts: int = 2
) -> Optional[dict]:
    """Generate a coder training entry with Manim validation.

    Retries up to max_attempts times. On retry, feeds the error back to the LLM
    so it can fix the code instead of blindly regenerating.
    """
    # Get the assistant response (brain output) from sharegpt messages
    brain_output = ""
    for msg in brain_entry.get("messages", []):
        if msg["role"] == "assistant":
            brain_output = msg["content"]
            break

    user_msg = (
        "Create a Manim animation for this physics/math visualization:\n\n"
        f"{brain_output}"
    )

    last_code = ""
    last_error = ""

    for attempt in range(max_attempts):
        # On retry, feed the error back so the LLM can fix it
        if attempt > 0 and last_code and last_error:
            retry_msg = (
                f"The following Manim code has an error:\n```python\n{last_code}\n```\n"
                f"Error: {last_error}\n\n"
                "Fix the code and return ONLY the corrected Python code."
            )
            response = call_api(CODER_SYSTEM_PROMPT, retry_msg, api)
        else:
            response = call_api(CODER_SYSTEM_PROMPT, user_msg, api)

        if response is None:
            continue

        code = _strip_code_fences(response)

        # ── Quality Gate: validate before accepting ──
        valid, error = validate_manim_code(code, full_render=full_render)
        if valid:
            return {
                "messages": [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": brain_output},
                    {"role": "assistant", "content": code},
                ],
                "metadata": brain_entry.get("metadata", {}),
            }
        else:
            last_code = code
            last_error = error
            print(
                f"  ⚠ Attempt {attempt + 1}/{max_attempts} failed validation: "
                f"{error[:100]}"
            )
            time.sleep(1)

    return None


def match_questions_solutions(questions: list, solutions: list) -> list[tuple]:
    """Match questions to their solutions by question number."""
    sol_map = {s["question_number"]: s for s in solutions}
    pairs = []
    for q in questions:
        sol = sol_map.get(q["question_number"])
        pairs.append((q, sol))
    return pairs


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate training data via LLM API")
    parser.add_argument(
        "--parsed-dir", default=str(Path(__file__).parent.parent / "data" / "parsed")
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent.parent / "data" / "training")
    )
    parser.add_argument(
        "--api", choices=["claude", "openai", "ollama"], default="claude"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max questions to process (0=all)"
    )
    parser.add_argument(
        "--skip-coder", action="store_true", help="Skip coder data generation"
    )
    parser.add_argument(
        "--validate-render",
        action="store_true",
        help="Full Manim render validation (slower but catches runtime errors)",
    )
    parser.add_argument(
        "--images-dir",
        default="",
        help="Dir with page images (PNG) for multimodal brain training. "
        "Files should be named like chaitanya_1_p0.png matching the PDF stem.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing checkpoint — start fresh",
    )

    args = parser.parse_args()
    parsed_dir = Path(args.parsed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir) if args.images_dir else None

    # Load parsed data
    all_questions = []
    all_solutions = []
    for institute in ["chaitanya", "narayana"]:
        q_file = parsed_dir / f"{institute}_questions.json"
        s_file = parsed_dir / f"{institute}_solutions.json"
        if q_file.exists():
            with open(q_file) as f:
                all_questions.extend(json.load(f))
        if s_file.exists():
            with open(s_file) as f:
                all_solutions.extend(json.load(f))

    if not all_questions:
        print(
            "ERROR: No parsed questions found. Run parse_pdfs.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = match_questions_solutions(all_questions, all_solutions)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"Processing {len(pairs)} question-solution pairs using {args.api} API\n")

    # ── Generate brain data (with checkpoint/resume) ──
    brain_file = output_dir / "brain_train.json"

    if args.no_resume:
        brain_data = []
        processed_keys = set()
    else:
        brain_data, processed_keys = _load_checkpoint(brain_file)

    for i, (q, s) in enumerate(pairs):
        key = _question_key(q)
        if key in processed_keys:
            continue  # Already processed in previous run

        print(
            f"[Brain {i + 1}/{len(pairs)}] Q#{q['question_number']} ({q['institute']})"
        )

        # Find page image if available
        img_path = None
        if images_dir:
            stem = Path(q.get("source", "")).stem
            # Try common naming patterns
            for pattern in [f"{stem}_*.png", f"{stem}.png"]:
                matches = list(images_dir.glob(pattern))
                if matches:
                    img_path = str(matches[0])
                    break

        entry = generate_brain_entry(q, s, args.api, image_path=img_path)
        if entry:
            brain_data.append(entry)
            processed_keys.add(key)
            print(f"  ✓ Generated")
        else:
            print(f"  ✗ Failed")

        # Checkpoint save every N entries
        if (i + 1) % SAVE_EVERY == 0:
            _save_checkpoint(brain_data, brain_file)
            print(f"  💾 Checkpoint saved ({len(brain_data)} entries)")

        time.sleep(0.5)  # Rate limit courtesy

    # Final save
    _save_checkpoint(brain_data, brain_file)
    print(f"\nSaved {len(brain_data)} brain entries → {brain_file}")

    # ── Generate coder data (with checkpoint/resume) ──
    if not args.skip_coder:
        coder_file = output_dir / "coder_train.json"

        if args.no_resume:
            coder_data = []
            coder_processed = set()
        else:
            coder_data, coder_processed = _load_checkpoint(coder_file)

        coder_failed = 0
        for i, entry in enumerate(brain_data):
            meta = entry.get("metadata", {})
            key = _question_key(meta)
            if key in coder_processed:
                continue

            print(
                f"[Coder {i + 1}/{len(brain_data)}] Q#{meta.get('question_number', '?')}"
            )
            coder_entry = generate_coder_entry(
                entry, args.api, full_render=args.validate_render
            )
            if coder_entry:
                coder_data.append(coder_entry)
                coder_processed.add(key)
                print(f"  ✓ Validated & accepted")
            else:
                coder_failed += 1
                print(f"  ✗ Failed validation — excluded from dataset")

            # Checkpoint
            if (i + 1) % SAVE_EVERY == 0:
                _save_checkpoint(coder_data, coder_file)
                print(f"  💾 Checkpoint saved ({len(coder_data)} entries)")

            time.sleep(0.5)

        _save_checkpoint(coder_data, coder_file)
        print(f"\nSaved {len(coder_data)} coder entries → {coder_file}")
        if coder_failed:
            print(f"  ⚠ {coder_failed} entries failed validation and were excluded")

    print("\n✅ Training data generation complete!")


if __name__ == "__main__":
    main()
