#!/usr/bin/env python3
"""
PDF Parsing Pipeline
--------------------
Extracts questions and solutions from Chaitanya & Narayana coaching PDFs.

Strategy (dual-mode):
  1. VISION MODE (recommended): Renders each PDF page as a PNG image and uses
     a vision LLM (Claude/GPT-4o) to extract structured Q&A. Reliable for
     complex JEE multi-column layouts, math, and tabular answer keys.
  2. TEXT MODE (fallback): Uses MinerU or PyMuPDF for text extraction with
     regex-based parsing. Fast but unreliable on complex layouts.

Usage:
    # Vision-LLM mode (recommended)
    python parse_pdfs.py --mode vision --api claude

    # Text extraction mode (faster, less reliable)
    python parse_pdfs.py --mode text

    # Custom I/O dirs
    python parse_pdfs.py --input-dir ../data --output-dir ../data/parsed
"""

import argparse
import base64
import json
import os
import sys
import re
from pathlib import Path
from typing import Optional

# ─── PyMuPDF (required for both modes — page rendering + text fallback) ─────
try:
    import fitz  # PyMuPDF
except ImportError:
    print(
        "ERROR: PyMuPDF required. Install with: pip install PyMuPDF",
        file=sys.stderr,
    )
    sys.exit(1)

# ─── Optional: MinerU for better text extraction ───────────────────────────
MINERU_AVAILABLE = False
try:
    from magic_pdf.data.data_reader_writer import FileBasedDataReader
    from magic_pdf.pipe.UNIPipe import UNIPipe

    MINERU_AVAILABLE = True
except ImportError:
    pass


# ─── Load env ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


# ─── Vision LLM Extraction Prompt ──────────────────────────────────────────

VISION_EXTRACT_PROMPT = """You are an expert at reading JEE (Indian engineering entrance exam) question papers.

Analyze this page image from a {institute} coaching institute {doc_type} PDF.

Extract ALL questions (or solutions) visible on this page as a JSON array.
Each item should have:
{{
  "question_number": <int>,
  "text": "<full question text including all options A/B/C/D, LaTeX math as $...$ or $$...$$>",
  "subject": "<physics | chemistry | math | unknown>",
  "has_diagram": <true/false>,
  "diagram_description": "<describe the diagram if present, empty string otherwise>"
}}

For SOLUTION pages, use this schema instead:
{{
  "question_number": <int>,
  "solution_text": "<full solution including working, formulas, final answer>",
  "final_answer": "<just the answer: number, option letter, or expression>"
}}

Rules:
- Extract EVERY question/solution visible on the page — do not skip any
- Preserve mathematical notation in LaTeX format
- For multi-column layouts, read left column first, then right
- For tabular answer keys, extract each number-answer mapping
- Question numbering: Physics usually 1-25 or 26-50, Chemistry 26-50 or 51-75, Math 51-75 or 76-100
- Output ONLY the JSON array, no other text"""


# ─── API Clients for Vision ─────────────────────────────────────────────────


def call_claude_vision(
    system: str, image_b64: str, user_text: str, max_retries: int = 3
) -> Optional[str]:
    """Call Anthropic Claude with a base64 image."""
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                import time

                time.sleep(2**attempt)
    return None


def call_openai_vision(
    system: str, image_b64: str, user_text: str, max_retries: int = 3
) -> Optional[str]:
    """Call OpenAI GPT-4o with a base64 image."""
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
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                import time

                time.sleep(2**attempt)
    return None


# ─── Page Image Extraction ──────────────────────────────────────────────────


def render_page_image(pdf_path: str, page_num: int, output_dir: str) -> tuple[str, str]:
    """Render a PDF page as PNG and return (image_path, base64_string)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
    pix = page.get_pixmap(matrix=mat)

    img_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_p{page_num}.png")
    pix.save(img_path)

    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    doc.close()
    return img_path, img_b64


# ─── Vision Mode: Extract via LLM ──────────────────────────────────────────


def extract_via_vision(
    pdf_path: str,
    doc_type: str,
    institute: str,
    output_dir: str,
    api: str = "claude",
) -> list[dict]:
    """Extract Q&A from a PDF by rendering pages and sending to a vision LLM."""
    doc = fitz.open(pdf_path)
    all_items = []

    print(f"  Parsing {doc.page_count} pages via {api} vision API...")

    for page_num in range(doc.page_count):
        print(f"    Page {page_num + 1}/{doc.page_count}...", end=" ")

        img_path, img_b64 = render_page_image(pdf_path, page_num, output_dir)

        prompt = VISION_EXTRACT_PROMPT.format(
            institute=institute.title(),
            doc_type=doc_type,
        )

        if api == "claude":
            response = call_claude_vision(
                "Extract structured data from JEE exam pages.",
                img_b64,
                prompt,
            )
        elif api == "openai":
            response = call_openai_vision(
                "Extract structured data from JEE exam pages.",
                img_b64,
                prompt,
            )
        else:
            print(f"Unsupported vision API: {api}")
            continue

        if response is None:
            print("FAILED")
            continue

        # Parse the JSON response
        try:
            # Strip markdown fences if present
            text = response.strip()
            if text.startswith("```json"):
                text = text[len("```json") :].strip()
            if text.startswith("```"):
                text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

            items = json.loads(text)
            if not isinstance(items, list):
                items = [items]

            # Tag each item with source metadata
            for item in items:
                item["source"] = Path(pdf_path).name
                item["institute"] = institute
                item["page"] = page_num
                item["page_image"] = img_path

            all_items.extend(items)
            print(f"→ {len(items)} items")

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            continue

        import time

        time.sleep(0.5)  # Rate limit

    doc.close()
    return all_items


# ─── Text Mode: Regex-based extraction (fallback) ──────────────────────────


def parse_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (or MinerU if available)."""
    if MINERU_AVAILABLE:
        try:
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_path)
            pipe = UNIPipe(pdf_bytes, [], image_writer=None)
            pipe.pipe_classify()
            pipe.pipe_analyze()
            pipe.pipe_parse()
            return pipe.pipe_mk_markdown("", drop_mode="none")
        except Exception as e:
            print(f"    MinerU failed ({e}), falling back to PyMuPDF")

    doc = fitz.open(pdf_path)
    text_blocks = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text_blocks.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(text_blocks)


def extract_questions_text(
    raw_text: str, source_file: str, institute: str
) -> list[dict]:
    """Regex-based question extraction (fallback for text mode)."""
    questions = []
    pattern = r"(?:Q\.?\s*)?(\d+)\s*[.)]\s*(.*?)(?=(?:Q\.?\s*)?\d+\s*[.)]|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for num, text in matches:
        text = text.strip()
        if len(text) > 10:
            questions.append(
                {
                    "question_number": int(num),
                    "text": text,
                    "source": source_file,
                    "institute": institute,
                }
            )
    return questions


def extract_solutions_text(
    raw_text: str, source_file: str, institute: str
) -> list[dict]:
    """Regex-based solution extraction (fallback for text mode)."""
    solutions = []
    pattern = r"(?:Sol\.?\s*|Ans\.?\s*|Solution\s*)?(\d+)\s*[.):\s]\s*(.*?)(?=(?:Sol\.?\s*|Ans\.?\s*|Solution\s*)?\d+\s*[.):\s]|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for num, text in matches:
        text = text.strip()
        if len(text) > 5:
            solutions.append(
                {
                    "question_number": int(num),
                    "solution_text": text,
                    "source": source_file,
                    "institute": institute,
                }
            )
    return solutions


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def process_vision_mode(input_dir: str, output_dir: str, api: str):
    """Process all PDFs using vision-LLM extraction."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    questions_dir = input_path / "questions"
    solutions_dir = input_path / "solutions"

    results = {
        "chaitanya_questions": [],
        "chaitanya_solutions": [],
        "narayana_questions": [],
        "narayana_solutions": [],
    }

    # Parse question PDFs
    if questions_dir.exists():
        for pdf_file in sorted(questions_dir.glob("*.pdf")):
            institute = (
                "chaitanya" if "chaitanya" in pdf_file.name.lower() else "narayana"
            )
            print(f"\n  📄 {pdf_file.name} (questions, {institute})")
            items = extract_via_vision(
                str(pdf_file),
                "question paper",
                institute,
                str(images_dir),
                api,
            )
            results[f"{institute}_questions"].extend(items)
            print(f"    Total: {len(items)} questions extracted")

    # Parse solution PDFs
    if solutions_dir.exists():
        for pdf_file in sorted(solutions_dir.glob("*.pdf")):
            institute = (
                "chaitanya" if "chaitanya" in pdf_file.name.lower() else "narayana"
            )
            print(f"\n  📄 {pdf_file.name} (solutions, {institute})")
            items = extract_via_vision(
                str(pdf_file),
                "solution/answer key",
                institute,
                str(images_dir),
                api,
            )
            results[f"{institute}_solutions"].extend(items)
            print(f"    Total: {len(items)} solutions extracted")

    return results


def process_text_mode(input_dir: str, output_dir: str):
    """Process all PDFs using text extraction + regex (fallback)."""
    input_path = Path(input_dir)

    results = {
        "chaitanya_questions": [],
        "chaitanya_solutions": [],
        "narayana_questions": [],
        "narayana_solutions": [],
    }

    questions_dir = input_path / "questions"
    solutions_dir = input_path / "solutions"

    if questions_dir.exists():
        for pdf_file in sorted(questions_dir.glob("*.pdf")):
            institute = (
                "chaitanya" if "chaitanya" in pdf_file.name.lower() else "narayana"
            )
            print(f"  📄 {pdf_file.name} (text mode)")
            raw_text = parse_pdf_text(str(pdf_file))
            items = extract_questions_text(raw_text, pdf_file.name, institute)
            results[f"{institute}_questions"].extend(items)
            print(f"    → {len(items)} questions")

    if solutions_dir.exists():
        for pdf_file in sorted(solutions_dir.glob("*.pdf")):
            institute = (
                "chaitanya" if "chaitanya" in pdf_file.name.lower() else "narayana"
            )
            print(f"  📄 {pdf_file.name} (text mode)")
            raw_text = parse_pdf_text(str(pdf_file))
            items = extract_solutions_text(raw_text, pdf_file.name, institute)
            results[f"{institute}_solutions"].extend(items)
            print(f"    → {len(items)} solutions")

    return results


def save_results(results: dict, output_dir: str):
    """Save parsed results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for key, data in results.items():
        out_file = output_path / f"{key}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file} ({len(data)} entries)")

    total = sum(len(v) for v in results.values())
    print(f"\n{'=' * 50}")
    print(f"  Total extracted: {total} entries")
    print(f"  Output dir: {output_path}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Parse coaching institute PDFs")
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).parent.parent / "data"),
        help="Directory containing questions/ and solutions/ folders",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "data" / "parsed"),
        help="Output directory for parsed JSON files",
    )
    parser.add_argument(
        "--mode",
        choices=["vision", "text"],
        default="vision",
        help="Extraction mode: 'vision' (LLM-based, reliable) or 'text' (regex, fast)",
    )
    parser.add_argument(
        "--api",
        choices=["claude", "openai"],
        default="claude",
        help="Vision API to use (only for --mode vision)",
    )

    args = parser.parse_args()
    print(f"PDF Parser (mode: {args.mode})")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}\n")

    if args.mode == "vision":
        results = process_vision_mode(args.input_dir, args.output_dir, args.api)
    else:
        results = process_text_mode(args.input_dir, args.output_dir)

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
