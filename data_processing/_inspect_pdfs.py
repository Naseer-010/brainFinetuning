#!/usr/bin/env python3
"""Quick inspection of all PDFs to understand their structure."""
import fitz
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

data_dir = Path(__file__).parent.parent / "data"

for subfolder in ["questions", "solutions"]:
    folder = data_dir / subfolder
    if not folder.exists():
        continue
    for pdf_file in sorted(folder.glob("*.pdf")):
        doc = fitz.open(str(pdf_file))
        print(f"\n{'='*80}")
        print(f"FILE: {pdf_file.name}  |  Folder: {subfolder}  |  Pages: {doc.page_count}")
        print(f"{'='*80}")
        
        # Show first 3 pages
        for page_num in range(min(3, doc.page_count)):
            page = doc[page_num]
            text = page.get_text("text")
            print(f"\n--- Page {page_num+1} (chars: {len(text)}) ---")
            print(text[:2000])
            print("...[truncated]" if len(text) > 2000 else "")
        
        doc.close()
