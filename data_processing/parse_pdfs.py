#!/usr/bin/env python3
"""
PDF Parsing Pipeline (Hybrid Text + Vision)
--------------------------------------------
Extracts questions and solutions from Chaitanya & Narayana coaching PDFs.

Strategy (hybrid):
  1. AUTO-DETECT whether each PDF is text-extractable or image-only (scanned).
  2. TEXT-EXTRACTABLE PDFs → institute-specific regex parsers (fast, accurate).
  3. IMAGE-ONLY PDFs → Qwen2.5-VL via Ollama with format:json enforcement.
  4. KEY SHEETS → text-parse structured number-answer pairs first, fall back to vision.

Supports three modes:
  - hybrid (default): auto-detects and uses best approach per PDF
  - vision: forces vision-LLM on all pages
  - text: forces text-only regex extraction

Usage:
    # Recommended: hybrid mode with local Ollama
    python parse_pdfs.py --mode hybrid --api ollama

    # Force vision-only (slower)
    python parse_pdfs.py --mode vision --api ollama

    # Force text-only (faster but misses image PDFs)
    python parse_pdfs.py --mode text

    # Custom I/O dirs
    python parse_pdfs.py --input-dir ../data --output-dir ../data/parsed
"""

import argparse
import base64
import json
import os
import re
import sys
import time
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

# ─── Load env ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY: Robust JSON parsing from LLM responses
# ═══════════════════════════════════════════════════════════════════════════════


def parse_llm_json(response: str) -> Optional[list]:
    """Parse JSON from LLM response with multiple fallback strategies.

    Returns a list of dicts, or None if all strategies fail.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Strategy 1: Strip markdown fences
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Strategy 2: Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Check if it wraps a list under a known key
            for key in ("questions", "items", "data", "results", "solutions", "answers"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]
        return None
    except json.JSONDecodeError:
        pass

    # Strategy 3: Extract JSON array with regex
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 4: Extract JSON object with regex
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict):
                for key in ("questions", "items", "data", "results", "solutions", "answers"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
        except json.JSONDecodeError:
            pass

    # Strategy 5: Try fixing common issues
    fixed = text
    # Remove trailing commas before ] or }
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # Add missing closing brackets
    open_brackets = fixed.count('[') - fixed.count(']')
    open_braces = fixed.count('{') - fixed.count('}')
    fixed += '}' * max(0, open_braces)
    fixed += ']' * max(0, open_brackets)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Strategy 6: Try to find multiple JSON objects and parse them individually
    objects = re.findall(r'\{[^{}]*\}', text)
    if objects:
        results = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                results.append(obj)
            except json.JSONDecodeError:
                continue
        if results:
            return results

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY: Auto-detect text vs image PDFs
# ═══════════════════════════════════════════════════════════════════════════════


def is_text_extractable(pdf_path: str, threshold: int = 100) -> bool:
    """Check if a PDF has extractable text (vs scanned images).

    Returns True if the average characters per page exceeds threshold.
    Some PDFs have headers but no question content — so we skip the first
    2 pages (usually cover/syllabus) and check the content pages.
    """
    doc = fitz.open(pdf_path)
    content_pages = list(range(min(2, doc.page_count), doc.page_count))
    if not content_pages:
        content_pages = list(range(doc.page_count))

    total_chars = 0
    for pn in content_pages:
        text = doc[pn].get_text("text").strip()
        # Filter out header/footer noise
        lines = [l for l in text.split('\n') if len(l.strip()) > 5]
        total_chars += sum(len(l) for l in lines)

    avg_chars = total_chars / max(1, len(content_pages))
    doc.close()
    return avg_chars > threshold


def detect_institute(filename: str) -> str:
    """Detect institute from filename."""
    name = filename.lower()
    if "chaitanya" in name:
        return "chaitanya"
    elif "narayana" in name:
        return "narayana"
    return "unknown"


def detect_doc_type(filename: str, folder: str) -> str:
    """Detect document type from filename and folder."""
    name = filename.lower()
    if "key" in name or "sol" in name or folder == "solutions":
        return "solutions"
    return "questions"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT PARSING: Chaitanya Questions
# ═══════════════════════════════════════════════════════════════════════════════


def parse_chaitanya_questions_text(pdf_path: str) -> list[dict]:
    """Parse Chaitanya question PDFs using text extraction.

    Chaitanya format:
    - Section headers: MATHEMATICS, PHYSICS, CHEMISTRY
    - Question numbers: 1. 2. 3. ... (numbers with period)
    - Options: 1) 2) 3) 4) or (1) (2) (3) (4)
    - Math: 1-25, Physics: 26-50, Chemistry: 51-75
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n\n"
    doc.close()

    questions = []
    current_subject = "unknown"

    # Detect subject headers
    subject_map = {
        "MATHEMATICS": "math",
        "PHYSICS": "physics",
        "CHEMISTRY": "chemistry",
    }

    # Split into pages/sections and track subject
    lines = full_text.split('\n')
    current_q_num = None
    current_q_text = []
    in_section_header = False

    for line in lines:
        stripped = line.strip()

        # Detect subject changes
        for header, subj in subject_map.items():
            if header in stripped.upper() and len(stripped) < 80:
                current_subject = subj
                break

        # Detect question start: "1." or "1 ." at beginning of meaningful content
        q_match = re.match(r'^(\d+)\s*[.)]\s*(.*)', stripped)
        if q_match:
            q_num = int(q_match.group(1))
            q_start_text = q_match.group(2).strip()

            # Must be a reasonable question number (1-75)
            if 1 <= q_num <= 75:
                # Save previous question if exists
                if current_q_num is not None and current_q_text:
                    text = ' '.join(current_q_text).strip()
                    if len(text) > 15:  # Filter out noise
                        questions.append({
                            "question_number": current_q_num,
                            "text": _clean_text(text),
                            "subject": _subject_from_qnum_chaitanya(current_q_num, current_subject),
                            "has_diagram": False,
                            "source": Path(pdf_path).name,
                            "institute": "chaitanya",
                        })

                current_q_num = q_num
                current_q_text = [q_start_text] if q_start_text else []
                continue

        # Skip headers and noise
        if _is_header_line(stripped):
            continue

        # Accumulate question text
        if current_q_num is not None and stripped:
            current_q_text.append(stripped)

    # Save last question
    if current_q_num is not None and current_q_text:
        text = ' '.join(current_q_text).strip()
        if len(text) > 15:
            questions.append({
                "question_number": current_q_num,
                "text": _clean_text(text),
                "subject": _subject_from_qnum_chaitanya(current_q_num, current_subject),
                "has_diagram": False,
                "source": Path(pdf_path).name,
                "institute": "chaitanya",
            })

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT PARSING: Narayana Questions
# ═══════════════════════════════════════════════════════════════════════════════


def parse_narayana_questions_text(pdf_path: str) -> list[dict]:
    """Parse Narayana question PDFs using text extraction.

    Narayana format:
    - Section headers: PHYSICS, CHEMISTRY, MATHEMATICS
    - Question numbers: 1. 2. 3. ... (numbers with period)
    - Options: A) B) C) D) or (A) (B) (C) (D)
    - Physics: 1-25, Chemistry: 26-50, Math: 51-75
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n\n"
    doc.close()

    questions = []
    current_subject = "unknown"
    subject_map = {
        "MATHEMATICS": "math",
        "PHYSICS": "physics",
        "CHEMISTRY": "chemistry",
    }

    lines = full_text.split('\n')
    current_q_num = None
    current_q_text = []

    for line in lines:
        stripped = line.strip()

        # Detect subject changes
        for header, subj in subject_map.items():
            if header in stripped.upper() and len(stripped) < 80:
                current_subject = subj
                break

        # Detect question start
        q_match = re.match(r'^(\d+)\s*[.)]\s*(.*)', stripped)
        if q_match:
            q_num = int(q_match.group(1))
            q_start_text = q_match.group(2).strip()

            if 1 <= q_num <= 75:
                # Save previous question
                if current_q_num is not None and current_q_text:
                    text = ' '.join(current_q_text).strip()
                    if len(text) > 15:
                        questions.append({
                            "question_number": current_q_num,
                            "text": _clean_text(text),
                            "subject": _subject_from_qnum_narayana(current_q_num, current_subject),
                            "has_diagram": False,
                            "source": Path(pdf_path).name,
                            "institute": "narayana",
                        })

                current_q_num = q_num
                current_q_text = [q_start_text] if q_start_text else []
                continue

        if _is_header_line(stripped):
            continue

        if current_q_num is not None and stripped:
            current_q_text.append(stripped)

    # Save last question
    if current_q_num is not None and current_q_text:
        text = ' '.join(current_q_text).strip()
        if len(text) > 15:
            questions.append({
                "question_number": current_q_num,
                "text": _clean_text(text),
                "subject": _subject_from_qnum_narayana(current_q_num, current_subject),
                "has_diagram": False,
                "source": Path(pdf_path).name,
                "institute": "narayana",
            })

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT PARSING: Chaitanya Key/Solutions
# ═══════════════════════════════════════════════════════════════════════════════


def parse_chaitanya_key_text(pdf_path: str) -> list[dict]:
    """Parse Chaitanya key/solution PDFs using text extraction.

    Chaitanya key format:
    - Page 1: KEY SHEET with number-answer pairs (two lines: number, then answer)
    - Page 2+: SOLUTION with worked solutions
    - Answers for MCQ (Q1-20, Q26-45, Q51-70): single digit 1-4
    - Answers for numerical (Q21-25, Q46-50, Q71-75): integer values
    """
    doc = fitz.open(pdf_path)
    solutions = []

    # ── Phase 1: Parse the key sheet (usually page 1) ──
    key_answers = {}
    current_subject = "unknown"
    subject_map = {"MATHEMATICS": "math", "PHYSICS": "physics", "CHEMISTRY": "chemistry"}

    for page_num in range(min(2, doc.page_count)):  # Key is on first 1-2 pages
        text = doc[page_num].get_text("text")
        if "KEY" not in text.upper():
            continue

        lines = text.split('\n')
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Track subject
            for header, subj in subject_map.items():
                if header in stripped.upper() and len(stripped) < 40:
                    current_subject = subj

            # Look for number-answer pairs: number on one line, answer on next
            num_match = re.match(r'^(\d+)\s*\.?\s*$', stripped)
            if num_match:
                q_num = int(num_match.group(1))
                if 1 <= q_num <= 75 and i + 1 < len(lines):
                    answer_line = lines[i + 1].strip()
                    # Answer could be a single digit (1-4) or a number
                    ans_match = re.match(r'^(\d+)\s*$', answer_line)
                    if ans_match:
                        key_answers[q_num] = ans_match.group(1)
                        i += 2
                        continue

            # Also handle "1. 3" format (number and answer on same line)
            inline_match = re.match(r'^(\d+)\s*[.)]\s*(\d+)\s*$', stripped)
            if inline_match:
                q_num = int(inline_match.group(1))
                if 1 <= q_num <= 75:
                    key_answers[q_num] = inline_match.group(2)

            i += 1

    # ── Phase 2: Parse worked solutions (remaining pages) ──
    solution_texts = {}
    current_q_num = None
    current_sol_text = []
    current_subject = "unknown"

    for page_num in range(doc.page_count):
        text = doc[page_num].get_text("text")

        # Skip pure key-sheet pages
        if page_num < 2 and "KEY" in text.upper() and "SOLUTION" not in text.upper():
            continue

        for line in text.split('\n'):
            stripped = line.strip()

            for header, subj in subject_map.items():
                if header in stripped.upper() and len(stripped) < 40:
                    current_subject = subj

            # Detect solution start
            sol_match = re.match(r'^(\d+)\s*[.)]\s*(.*)', stripped)
            if sol_match:
                q_num = int(sol_match.group(1))
                sol_start = sol_match.group(2).strip()

                if 1 <= q_num <= 75:
                    if current_q_num is not None and current_sol_text:
                        text_combined = ' '.join(current_sol_text).strip()
                        if len(text_combined) > 5:
                            solution_texts[current_q_num] = text_combined

                    current_q_num = q_num
                    current_sol_text = [sol_start] if sol_start else []
                    continue

            if _is_header_line(stripped):
                continue

            if current_q_num is not None and stripped:
                current_sol_text.append(stripped)

    # Save last solution
    if current_q_num is not None and current_sol_text:
        text_combined = ' '.join(current_sol_text).strip()
        if len(text_combined) > 5:
            solution_texts[current_q_num] = text_combined

    doc.close()

    # ── Merge key answers + solution texts ──
    all_q_nums = set(key_answers.keys()) | set(solution_texts.keys())
    for q_num in sorted(all_q_nums):
        entry = {
            "question_number": q_num,
            "final_answer": key_answers.get(q_num, ""),
            "solution_text": _clean_text(solution_texts.get(q_num, "")),
            "subject": _subject_from_qnum_chaitanya(q_num, "unknown"),
            "source": Path(pdf_path).name,
            "institute": "chaitanya",
        }
        solutions.append(entry)

    return solutions


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT PARSING: Narayana Key/Solutions
# ═══════════════════════════════════════════════════════════════════════════════


def parse_narayana_key_text(pdf_path: str) -> list[dict]:
    """Parse Narayana key/solution PDFs using text extraction.

    Narayana key format:
    - Page 1: KEYSHEET with tabular grid (numbers in row, letters below)
    - Page 2+: Solutions with worked solutions
    - MCQ answers: A/B/C/D letters
    - Numerical answers: integer values
    """
    doc = fitz.open(pdf_path)
    solutions = []

    # ── Phase 1: Parse the key sheet ──
    key_answers = {}
    current_subject = "unknown"
    subject_map = {"MATHEMATICS": "math", "PHYSICS": "physics", "CHEMISTRY": "chemistry"}

    for page_num in range(min(2, doc.page_count)):
        text = doc[page_num].get_text("text")
        if "KEY" not in text.upper():
            continue

        lines = [l.strip() for l in text.split('\n') if l.strip()]

        i = 0
        while i < len(lines):
            # Track subject
            for header, subj in subject_map.items():
                if header in lines[i].upper() and len(lines[i]) < 40:
                    current_subject = subj

            # Narayana uses tabular layout: row of numbers, then row of answers
            # e.g.  1   2   3   4   5   6   7   8   9   10
            #        D   A   D   D   C   B   D   D   B   C
            nums_in_line = re.findall(r'\b(\d+)\b', lines[i])
            if nums_in_line and len(nums_in_line) >= 3:
                # Check if all are reasonable question numbers
                try:
                    q_nums = [int(n) for n in nums_in_line]
                except ValueError:
                    i += 1
                    continue

                if all(1 <= n <= 75 for n in q_nums):
                    # Look for answer row below
                    if i + 1 < len(lines):
                        answer_line = lines[i + 1]
                        # MCQ answers: letters
                        answers = re.findall(r'\b([A-Da-d])\b', answer_line)
                        if len(answers) == len(q_nums):
                            for q_num, ans in zip(q_nums, answers):
                                key_answers[q_num] = ans.upper()
                            i += 2
                            continue
                        # Numerical answers: numbers
                        num_answers = re.findall(r'\b(\d+)\b', answer_line)
                        if len(num_answers) == len(q_nums):
                            for q_num, ans in zip(q_nums, num_answers):
                                key_answers[q_num] = ans
                            i += 2
                            continue

            # Also handle inline "num answer" pairs
            inline_match = re.match(r'^(\d+)\s+([A-Da-d])\s*$', lines[i])
            if inline_match:
                q_num = int(inline_match.group(1))
                if 1 <= q_num <= 75:
                    key_answers[q_num] = inline_match.group(2).upper()

            i += 1

    # ── Phase 2: Parse worked solutions ──
    solution_texts = {}
    current_q_num = None
    current_sol_text = []

    for page_num in range(doc.page_count):
        text = doc[page_num].get_text("text")
        if page_num < 2 and "KEY" in text.upper() and "SOLUTION" not in text.upper():
            continue

        for line in text.split('\n'):
            stripped = line.strip()

            for header, subj in subject_map.items():
                if header in stripped.upper() and len(stripped) < 40:
                    current_subject = subj

            sol_match = re.match(r'^(\d+)\s*[.)]\s*(.*)', stripped)
            if sol_match:
                q_num = int(sol_match.group(1))
                sol_start = sol_match.group(2).strip()

                if 1 <= q_num <= 75:
                    if current_q_num is not None and current_sol_text:
                        text_combined = ' '.join(current_sol_text).strip()
                        if len(text_combined) > 5:
                            solution_texts[current_q_num] = text_combined

                    current_q_num = q_num
                    current_sol_text = [sol_start] if sol_start else []
                    continue

            if _is_header_line(stripped):
                continue

            if current_q_num is not None and stripped:
                current_sol_text.append(stripped)

    if current_q_num is not None and current_sol_text:
        text_combined = ' '.join(current_sol_text).strip()
        if len(text_combined) > 5:
            solution_texts[current_q_num] = text_combined

    doc.close()

    # ── Merge ──
    all_q_nums = set(key_answers.keys()) | set(solution_texts.keys())
    for q_num in sorted(all_q_nums):
        entry = {
            "question_number": q_num,
            "final_answer": key_answers.get(q_num, ""),
            "solution_text": _clean_text(solution_texts.get(q_num, "")),
            "subject": _subject_from_qnum_narayana(q_num, "unknown"),
            "source": Path(pdf_path).name,
            "institute": "narayana",
        }
        solutions.append(entry)

    return solutions


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def _clean_text(text: str) -> str:
    """Clean garbled Unicode characters from PDF text extraction."""
    # Remove common garbled sequences from PyMuPDF
    text = re.sub(r'[∩Ç¿∩Ç⌐∩Ç½∩Ç╜∩Ç¡∩Ç╝∩Çá∩é╡∩â⌐∩â╣∩â½∩â╗∩âÄ∩â₧∩ü£∩ü░∩üí∩üó∩üä∩â▓∩âª∩â╢∩âº∩â╖∩â¿∩â╕]+', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_header_line(line: str) -> bool:
    """Check if a line is a header/footer that should be skipped."""
    line_upper = line.upper()
    skip_patterns = [
        "SRI CHAITANYA", "NARAYANA", "IIT ACADEMY", "JEE ACADEMY",
        "PAGE |", "PAGE ", "SECTION-I", "SECTION-II", "SECTION –",
        "SINGLE CORRECT ANSWER", "MULTIPLE CHOICE", "NUMERICAL VALUE",
        "MARKING SCHEME", "MAX MARKS", "MAX.MARKS",
        "ICON CENTRAL", "A RIGHT CHOICE", "THE PERFECT",
        "GTM-", "IGTM-", "JEE-MAIN", "JEE MAIN",
        "DATE:", "TIME:", "GRAND TEST",
        "SEC:", "APEX", "CO-SC",
    ]
    for pattern in skip_patterns:
        if pattern in line_upper:
            return True
    # Skip very short lines that are just noise
    if len(line.strip()) < 3:
        return True
    return False


def _subject_from_qnum_chaitanya(q_num: int, detected: str) -> str:
    """Map question number to subject for Chaitanya (Math 1-25, Phys 26-50, Chem 51-75)."""
    if detected != "unknown":
        return detected
    if 1 <= q_num <= 25:
        return "math"
    elif 26 <= q_num <= 50:
        return "physics"
    elif 51 <= q_num <= 75:
        return "chemistry"
    return "unknown"


def _subject_from_qnum_narayana(q_num: int, detected: str) -> str:
    """Map question number to subject for Narayana (Phys 1-25, Chem 26-50, Math 51-75)."""
    if detected != "unknown":
        return detected
    if 1 <= q_num <= 25:
        return "physics"
    elif 26 <= q_num <= 50:
        return "chemistry"
    elif 51 <= q_num <= 75:
        return "math"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  VISION: Ollama API Client
# ═══════════════════════════════════════════════════════════════════════════════


def call_ollama_vision(
    system: str,
    image_b64: str,
    user_text: str,
    model: str = "qwen2.5vl:7b",
    max_retries: int = 3,
) -> Optional[str]:
    """Call local Ollama API with vision support.

    Uses format:json to force valid JSON output and increased timeout.
    """
    import requests

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user_text,
                "images": [image_b64],
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        },
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            if content and content.strip():
                return content
            print(f"    ⚠ Empty response (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"    ⚠ Ollama API error (attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return None


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
                time.sleep(2 ** attempt)
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
                time.sleep(2 ** attempt)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  VISION: Page Rendering + Extraction
# ═══════════════════════════════════════════════════════════════════════════════


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


# ─── Simplified Vision Prompts (optimized for local 7B models) ──────────────

VISION_QUESTION_PROMPT = """Look at this JEE exam page image. Extract ALL questions visible.

Return a JSON object with a "questions" array. Each question:
{{"questions": [
  {{
    "question_number": <int>,
    "text": "<full question text including all options>",
    "subject": "<physics|chemistry|math>",
    "has_diagram": <true/false>
  }}
]}}

Rules:
- Include ALL options (A/B/C/D or 1/2/3/4) in the text field
- Use LaTeX for math: $x^2$, $\\frac{{a}}{{b}}$
- Read left column first, then right column
- Do NOT skip any questions"""

VISION_KEY_PROMPT = """Look at this answer key page. Extract ALL question-answer pairs.

Return a JSON object with an "answers" array:
{{"answers": [
  {{
    "question_number": <int>,
    "answer": "<answer: letter A/B/C/D, number, or expression>"
  }}
]}}

Extract EVERY question-answer mapping on this page."""

VISION_SOLUTION_PROMPT = """Look at this solution page from a JEE exam. Extract the worked solutions.

Return a JSON object with a "solutions" array:
{{"solutions": [
  {{
    "question_number": <int>,
    "solution_text": "<full solution including steps and formulas>",
    "final_answer": "<just the final answer>"
  }}
]}}

Include ALL solutions visible on this page."""


def extract_page_via_vision(
    pdf_path: str,
    page_num: int,
    page_type: str,
    institute: str,
    output_dir: str,
    api: str = "ollama",
) -> list[dict]:
    """Extract questions/solutions from a single PDF page using vision LLM."""
    img_path, img_b64 = render_page_image(pdf_path, page_num, output_dir)

    system = "You extract structured data from Indian JEE exam papers. Output valid JSON only."

    if page_type == "question":
        prompt = VISION_QUESTION_PROMPT
    elif page_type == "key":
        prompt = VISION_KEY_PROMPT
    else:
        prompt = VISION_SOLUTION_PROMPT

    # Call the appropriate API
    if api == "ollama":
        response = call_ollama_vision(system, img_b64, prompt)
    elif api == "claude":
        response = call_claude_vision(system, img_b64, prompt)
    elif api == "openai":
        response = call_openai_vision(system, img_b64, prompt)
    else:
        print(f"    Unsupported API: {api}")
        return []

    if response is None:
        return []

    # Parse JSON response
    items = parse_llm_json(response)
    if items is None:
        print(f"    ✗ Could not parse JSON from response ({len(response)} chars)")
        # Save raw response for debugging
        debug_path = os.path.join(output_dir, f"debug_{Path(pdf_path).stem}_p{page_num}.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(response)
        return []

    # Tag each item with metadata
    for item in items:
        item["source"] = Path(pdf_path).name
        item["institute"] = institute
        item["page"] = page_num
        item["page_image"] = img_path

    return items


def extract_pdf_via_vision(
    pdf_path: str,
    doc_type: str,
    institute: str,
    output_dir: str,
    api: str = "ollama",
) -> list[dict]:
    """Extract all pages of a PDF using vision LLM."""
    doc = fitz.open(pdf_path)
    all_items = []

    print(f"  Parsing {doc.page_count} pages via {api} vision...")

    for page_num in range(doc.page_count):
        # Skip cover/syllabus pages (usually first 1-2 pages)
        page_text = doc[page_num].get_text("text").strip()

        # Detect page type
        if doc_type == "solutions":
            if page_num == 0 and "KEY" in page_text.upper():
                page_type = "key"
            else:
                page_type = "solution"
        else:
            page_type = "question"

        # Skip pages with too little content (cover pages, blank pages)
        if len(page_text) < 30 and page_type == "question":
            # Check if the page has substantial visual content
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
            # If the page is mostly white, skip it
            samples = pix.samples
            if len(set(samples[:1000])) < 10:  # Very uniform = blank
                print(f"    Page {page_num + 1}/{doc.page_count}... [blank, skipping]")
                continue

        print(f"    Page {page_num + 1}/{doc.page_count}...", end=" ")

        items = extract_page_via_vision(
            pdf_path, page_num, page_type, institute, output_dir, api
        )

        if items:
            all_items.extend(items)
            print(f"→ {len(items)} items")
        else:
            print("→ 0 items")

        time.sleep(0.5)  # Rate limit

    doc.close()
    return all_items


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def process_pdf(
    pdf_path: str,
    doc_type: str,
    institute: str,
    output_dir: str,
    mode: str = "hybrid",
    api: str = "ollama",
) -> list[dict]:
    """Process a single PDF file using the appropriate strategy.

    Args:
        pdf_path: Path to the PDF file
        doc_type: "questions" or "solutions"
        institute: "chaitanya" or "narayana"
        output_dir: Directory for output images/debug files
        mode: "hybrid", "vision", or "text"
        api: "ollama", "claude", or "openai"
    """
    filename = Path(pdf_path).name

    # Determine extraction strategy
    if mode == "vision":
        use_vision = True
    elif mode == "text":
        use_vision = False
    else:  # hybrid
        use_vision = not is_text_extractable(pdf_path)
        strategy = "VISION (image-only PDF)" if use_vision else "TEXT (text-extractable)"
        print(f"  Strategy: {strategy}")

    if use_vision:
        return extract_pdf_via_vision(pdf_path, doc_type, institute, output_dir, api)
    else:
        # Use institute-specific text parsers
        if doc_type == "questions":
            if institute == "chaitanya":
                return parse_chaitanya_questions_text(pdf_path)
            else:
                return parse_narayana_questions_text(pdf_path)
        else:  # solutions
            if institute == "chaitanya":
                return parse_chaitanya_key_text(pdf_path)
            else:
                return parse_narayana_key_text(pdf_path)


def process_all(input_dir: str, output_dir: str, mode: str, api: str) -> dict:
    """Process all PDFs in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "chaitanya_questions": [],
        "chaitanya_solutions": [],
        "narayana_questions": [],
        "narayana_solutions": [],
    }

    # Process questions
    questions_dir = input_path / "questions"
    if questions_dir.exists():
        for pdf_file in sorted(questions_dir.glob("*.pdf")):
            institute = detect_institute(pdf_file.name)
            print(f"\n  📄 {pdf_file.name} (questions, {institute})")
            items = process_pdf(
                str(pdf_file), "questions", institute,
                str(images_dir), mode, api,
            )
            results[f"{institute}_questions"].extend(items)
            print(f"    ✓ Total: {len(items)} questions extracted")

    # Process solutions
    solutions_dir = input_path / "solutions"
    if solutions_dir.exists():
        for pdf_file in sorted(solutions_dir.glob("*.pdf")):
            institute = detect_institute(pdf_file.name)
            print(f"\n  📄 {pdf_file.name} (solutions, {institute})")
            items = process_pdf(
                str(pdf_file), "solutions", institute,
                str(images_dir), mode, api,
            )
            results[f"{institute}_solutions"].extend(items)
            print(f"    ✓ Total: {len(items)} solutions extracted")

    return results


def save_results(results: dict, output_dir: str):
    """Save parsed results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for key, data in results.items():
        out_file = output_path / f"{key}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file} ({len(data)} entries)")

    total = sum(len(v) for v in results.values())
    print(f"\n{'=' * 60}")
    print(f"  Total extracted: {total} entries")
    for key, data in results.items():
        if data:
            print(f"    {key}: {len(data)}")
    print(f"  Output dir: {output_path}")
    print(f"{'=' * 60}")


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
        choices=["hybrid", "vision", "text"],
        default="hybrid",
        help="Extraction mode: 'hybrid' (auto-detect), 'vision' (LLM), or 'text' (regex)",
    )
    parser.add_argument(
        "--api",
        choices=["claude", "openai", "ollama"],
        default="ollama",
        help="Vision API to use (for vision/hybrid mode)",
    )

    args = parser.parse_args()
    print(f"PDF Parser (mode: {args.mode}, api: {args.api})")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}\n")

    results = process_all(args.input_dir, args.output_dir, args.mode, args.api)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
