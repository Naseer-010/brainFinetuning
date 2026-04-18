#!/usr/bin/env python3
"""
Training Data Generator
-----------------------
Takes parsed question-solution JSON pairs and calls a frontier LLM API
(Claude 3.5 Sonnet or GPT-4o) to synthesize fine-tuning data.

Produces:
    - brain_train.json : {system, instruction, output} with visual_type/narration/final_answer
    - coder_train.json : {system, instruction, output} with Manim Python code

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
import sys
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

BRAIN_SYSTEM_PROMPT = """You are an expert JEE Physics/Math visual reasoning engine.
Given a question (with optional diagram description), produce a JSON object with:
{
    "visual_type": "string — one of: free_body_diagram, circuit, graph, geometric, optics, wave, projectile, energy_diagram, vector, other",
    "key_concepts": ["list of physics/math concepts involved"],
    "step_by_step": ["ordered reasoning steps"],
    "narration": "A clear, teacher-like explanation suitable for text-to-speech narration of an educational video",
    "final_answer": "The final answer with units"
}
Be precise with physics. Show all intermediate calculations in step_by_step."""

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


def generate_coder_entry(brain_entry: dict, api: str) -> Optional[dict]:
    """Generate a coder training entry from a brain output."""
    user_msg = (
        "Create a Manim animation for this physics/math visualization:\n\n"
        f"{brain_entry['output']}"
    )

    response = call_api(CODER_SYSTEM_PROMPT, user_msg, api)
    if response is None:
        return None

    # Strip markdown fences if present
    code = response.strip()
    if code.startswith("```python"):
        code = code[len("```python") :].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return {
        "system": CODER_SYSTEM_PROMPT,
        "instruction": brain_entry["output"],
        "output": code,
        "metadata": brain_entry.get("metadata", {}),
    }


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
        for i, entry in enumerate(brain_data):
            print(
                f"[Coder {i + 1}/{len(brain_data)}] Q#{entry['metadata']['question_number']}"
            )
            coder_entry = generate_coder_entry(entry, args.api)
            if coder_entry:
                coder_data.append(coder_entry)
                print(f"  ✓ Generated")
            else:
                print(f"  ✗ Failed")
            time.sleep(0.5)

        coder_file = output_dir / "coder_train.json"
        with open(coder_file, "w") as f:
            json.dump(coder_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(coder_data)} coder entries → {coder_file}")

    print("\n✅ Training data generation complete!")


if __name__ == "__main__":
    main()
