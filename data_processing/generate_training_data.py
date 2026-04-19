#!/usr/bin/env python3
"""
Training Data Generator
-----------------------
Takes parsed question-solution JSON pairs and calls a frontier LLM API
(Claude 3.5 Sonnet or GPT-4o) to synthesize fine-tuning data.

Produces:
    - brain_train.json : scenes-array schema matching orchestrator BrainOutput
    - coder_train.json : validated Manim CE code (only renders that compile)

Usage:
    python generate_training_data.py \
        --parsed-dir ../data/parsed \
        --output-dir ../data/training \
        --api claude                    # or "openai"

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

# ─── Load env ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


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


def call_api(system: str, user_message: str, api: str = "claude") -> Optional[str]:
    """Unified API caller."""
    if api == "claude":
        return call_claude(system, user_message)
    elif api == "openai":
        return call_openai(system, user_message)
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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
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
            cwd="/tmp",
        )
        success = result.returncode == 0
        return success, "" if success else result.stderr[-500:]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def validate_manim_code(code: str, full_render: bool = False) -> tuple[bool, str]:
    """Validate Manim code: syntax check + optional full render."""
    ok, err = _validate_manim_syntax(code)
    if not ok:
        return False, err
    if full_render:
        return _validate_manim_render(code)
    return True, ""


# ─── Data Generation ────────────────────────────────────────────────────────


def generate_brain_entry(
    question: dict, solution: Optional[dict], api: str
) -> Optional[dict]:
    """Generate a brain training entry from a question-solution pair."""
    user_msg = (
        f"Question #{question['question_number']} ({question['institute'].title()}):\n"
    )
    user_msg += question["text"]
    if solution:
        user_msg += f"\n\nReference solution:\n{solution['solution_text']}"

    response = call_api(BRAIN_SYSTEM_PROMPT, user_msg, api)
    if response is None:
        return None

    # Validate that the response is valid JSON with the scenes array
    try:
        parsed = json.loads(response)
        if "scenes" not in parsed or not isinstance(parsed["scenes"], list):
            print("  ⚠ Response missing 'scenes' array, retrying...")
            response = call_api(BRAIN_SYSTEM_PROMPT, user_msg, api)
            if response:
                parsed = json.loads(response)
                if "scenes" not in parsed or not isinstance(parsed["scenes"], list):
                    print("  ✗ Retry also missing 'scenes'. Skipping.")
                    return None
            else:
                return None
    except json.JSONDecodeError:
        print("  ⚠ Response not valid JSON, retrying...")
        response = call_api(BRAIN_SYSTEM_PROMPT, user_msg, api)
        if response:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                print("  ✗ Retry also invalid JSON. Skipping.")
                return None
        else:
            return None

    return {
        "system": BRAIN_SYSTEM_PROMPT,
        "instruction": question["text"],
        "output": response,
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

    Retries up to max_attempts times if the generated code fails validation.
    Only validated, compiling code enters the training set.
    """
    user_msg = (
        "Create a Manim animation for this physics/math visualization:\n\n"
        f"{brain_entry['output']}"
    )

    for attempt in range(max_attempts):
        response = call_api(CODER_SYSTEM_PROMPT, user_msg, api)
        if response is None:
            continue

        code = _strip_code_fences(response)

        # ── Quality Gate: validate before accepting ──
        valid, error = validate_manim_code(code, full_render=full_render)
        if valid:
            return {
                "system": CODER_SYSTEM_PROMPT,
                "instruction": brain_entry["output"],
                "output": code,
                "metadata": brain_entry.get("metadata", {}),
            }
        else:
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
    parser.add_argument("--api", choices=["claude", "openai"], default="claude")
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

    args = parser.parse_args()
    parsed_dir = Path(args.parsed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Generate brain data ──
    brain_data = []
    for i, (q, s) in enumerate(pairs):
        print(
            f"[Brain {i + 1}/{len(pairs)}] Q#{q['question_number']} ({q['institute']})"
        )
        entry = generate_brain_entry(q, s, args.api)
        if entry:
            brain_data.append(entry)
            print(f"  ✓ Generated")
        else:
            print(f"  ✗ Failed")
        time.sleep(0.5)  # Rate limit courtesy

    brain_file = output_dir / "brain_train.json"
    with open(brain_file, "w") as f:
        json.dump(brain_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(brain_data)} brain entries → {brain_file}")

    # ── Generate coder data ──
    if not args.skip_coder:
        coder_data = []
        coder_failed = 0
        for i, entry in enumerate(brain_data):
            print(
                f"[Coder {i + 1}/{len(brain_data)}] Q#{entry['metadata']['question_number']}"
            )
            coder_entry = generate_coder_entry(
                entry, args.api, full_render=args.validate_render
            )
            if coder_entry:
                coder_data.append(coder_entry)
                print(f"  ✓ Validated & accepted")
            else:
                coder_failed += 1
                print(f"  ✗ Failed validation — excluded from dataset")
            time.sleep(0.5)

        coder_file = output_dir / "coder_train.json"
        with open(coder_file, "w") as f:
            json.dump(coder_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(coder_data)} coder entries → {coder_file}")
        if coder_failed:
            print(f"  ⚠ {coder_failed} entries failed validation and were excluded")

    print("\n✅ Training data generation complete!")


if __name__ == "__main__":
    main()
