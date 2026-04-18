#!/usr/bin/env python3
"""
PDF Parsing Pipeline
--------------------
Extracts questions and solutions from Chaitanya & Narayana coaching PDFs
using MinerU (or fallback OCR). Outputs structured JSON for training data generation.

Prerequisites:
    pip install -r requirements.txt

Usage:
    python parse_pdfs.py --input-dir ../data --output-dir ../data/parsed

Output:
    ../data/parsed/chaitanya_questions.json
    ../data/parsed/chaitanya_solutions.json
    ../data/parsed/narayana_questions.json
    ../data/parsed/narayana_solutions.json
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Optional

# ─── Try MinerU first, fallback to PyMuPDF ──────────────────────────────────
PARSER_BACKEND = None

try:
    from magic_pdf.data.data_reader_writer import (
        FileBasedDataWriter,
        FileBasedDataReader,
    )
    from magic_pdf.pipe.UNIPipe import UNIPipe

    PARSER_BACKEND = "mineru"
except ImportError:
    pass

if PARSER_BACKEND is None:
    try:
        import fitz  # PyMuPDF

        PARSER_BACKEND = "pymupdf"
    except ImportError:
        pass

if PARSER_BACKEND is None:
    print(
        "ERROR: No PDF parser found. Install one of:\n"
        "  pip install magic-pdf[full]     # MinerU (recommended)\n"
        "  pip install PyMuPDF             # PyMuPDF (fallback)\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ─── Parsing Backends ───────────────────────────────────────────────────────


def parse_with_mineru(pdf_path: str, output_dir: str) -> str:
    """Parse PDF using MinerU — best for scientific docs with LaTeX math."""
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)

    # MinerU pipeline
    model_json = []  # Let MinerU auto-detect layout
    pipe = UNIPipe(pdf_bytes, model_json, image_writer=None)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    md_content = pipe.pipe_mk_markdown("", drop_mode="none")
    return md_content


def parse_with_pymupdf(pdf_path: str) -> str:
    """Fallback: Parse PDF using PyMuPDF text extraction."""
    doc = fitz.open(pdf_path)
    text_blocks = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text_blocks.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(text_blocks)


def parse_pdf(pdf_path: str, output_dir: str) -> str:
    """Parse a PDF file using the best available backend."""
    print(f"  Parsing: {pdf_path} (backend: {PARSER_BACKEND})")
    if PARSER_BACKEND == "mineru":
        return parse_with_mineru(pdf_path, output_dir)
    else:
        return parse_with_pymupdf(pdf_path)


# ─── Institute-Specific Question Extraction ─────────────────────────────────


def extract_chaitanya_questions(raw_text: str, source_file: str) -> list[dict]:
    """
    Chaitanya papers use tabular answer key grids.
    Extract questions by matching 'Q.N' or numbered patterns.
    """
    questions = []
    # Pattern: numbered questions — "1.", "1)", "Q.1", "Q1." etc.
    pattern = r"(?:Q\.?\s*)?(\d+)\s*[.)]\s*(.*?)(?=(?:Q\.?\s*)?\d+\s*[.)]|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for num, text in matches:
        text = text.strip()
        if len(text) > 10:  # Filter noise
            questions.append(
                {
                    "question_number": int(num),
                    "text": text,
                    "source": source_file,
                    "institute": "chaitanya",
                }
            )

    return questions


def extract_narayana_questions(raw_text: str, source_file: str) -> list[dict]:
    """
    Narayana papers use sequential, detailed solution blocks.
    Extract questions similarly but expect different formatting.
    """
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
                    "institute": "narayana",
                }
            )

    return questions


def extract_solutions(raw_text: str, source_file: str, institute: str) -> list[dict]:
    """Extract solution/answer key entries from solution PDFs."""
    solutions = []
    # Try to match numbered solutions
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


def process_all_pdfs(input_dir: str, output_dir: str):
    """Process all PDFs in the data directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    questions_dir = input_path / "questions"
    solutions_dir = input_path / "solutions"

    results = {
        "chaitanya_questions": [],
        "chaitanya_solutions": [],
        "narayana_questions": [],
        "narayana_solutions": [],
    }

    # ── Parse question PDFs ──
    if questions_dir.exists():
        for pdf_file in sorted(questions_dir.glob("*.pdf")):
            raw_text = parse_pdf(str(pdf_file), str(output_path))

            if "chaitanya" in pdf_file.name.lower():
                extracted = extract_chaitanya_questions(raw_text, pdf_file.name)
                results["chaitanya_questions"].extend(extracted)
                print(f"    → Extracted {len(extracted)} Chaitanya questions")
            elif "narayana" in pdf_file.name.lower():
                extracted = extract_narayana_questions(raw_text, pdf_file.name)
                results["narayana_questions"].extend(extracted)
                print(f"    → Extracted {len(extracted)} Narayana questions")

    # ── Parse solution PDFs ──
    if solutions_dir.exists():
        for pdf_file in sorted(solutions_dir.glob("*.pdf")):
            raw_text = parse_pdf(str(pdf_file), str(output_path))

            institute = (
                "chaitanya" if "chaitanya" in pdf_file.name.lower() else "narayana"
            )
            extracted = extract_solutions(raw_text, pdf_file.name, institute)
            results[f"{institute}_solutions"].extend(extracted)
            print(f"    → Extracted {len(extracted)} {institute} solutions")

    # ── Save results ──
    for key, data in results.items():
        out_file = output_path / f"{key}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file} ({len(data)} entries)")

    # ── Summary ──
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

    args = parser.parse_args()
    print(f"PDF Parser (backend: {PARSER_BACKEND})")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}\n")
    process_all_pdfs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
