"""Stage 2: Clean. Remove artefacts without destroying structure."""

from __future__ import annotations

import re
from collections import Counter

from .models import Document, Page


def clean(doc: Document) -> Document:
    """Strip headers and footers that repeat on every page; keep headings and
    section boundaries, because chunking and citations depend on them.
    """
    if not doc.pages:
        return doc

    # Pass 1: Identify repeating header/footer candidates across multi-page documents
    cleaned_pages: list[Page] = []
    if len(doc.pages) > 2:
        first_lines = Counter[str]()
        last_lines = Counter[str]()
        for page in doc.pages:
            lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
            if lines:
                first_lines[lines[0]] += 1
                if len(lines) > 1:
                    last_lines[lines[-1]] += 1

        total = len(doc.pages)
        # Lines appearing on more than 50% of pages as first or last lines are recurring headers/footers
        repeat_headers = {ln for ln, count in first_lines.items() if count / total >= 0.50}
        repeat_footers = {ln for ln, count in last_lines.items() if count / total >= 0.50}
    else:
        repeat_headers = set()
        repeat_footers = set()

    for page in doc.pages:
        raw_lines = page.text.splitlines()
        filtered_lines: list[str] = []

        for idx, line in enumerate(raw_lines):
            stripped = line.strip()

            # Skip repeating page headers at top
            if idx <= 2 and stripped in repeat_headers:
                continue

            # Skip repeating footers at bottom
            if idx >= len(raw_lines) - 3 and stripped in repeat_footers:
                continue

            # Skip standalone page numbering patterns (e.g. "Page 14 of 58", "12", "- 12 -")
            if re.fullmatch(r"(?:Page\s+)?-?\s*\d+\s*(?:of\s+\d+)?-?", stripped, re.I):
                continue

            # Deliberately PRESERVE:
            # 1. Headings (Markdown # or Section/Clause/Rule labels, uppercase headings)
            # 2. Numbered clauses like 1.21, 5.4
            # 3. Table rows starting with |
            # 4. Normal paragraph text

            # Clean null bytes and weird unicode artifacts
            cleaned_line = line.replace("\x00", "").replace("\x0c", "").rstrip()
            filtered_lines.append(cleaned_line)

        # Collapse excess empty lines (preserve double newlines for paragraph breaks)
        cleaned_text = "\n".join(filtered_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

        cleaned_pages.append(
            Page(
                source=page.source,
                page=page.page,
                text=cleaned_text,
                ocr=page.ocr,
            )
        )

    return Document(
        source=doc.source,
        doc_version=doc.doc_version,
        superseded=doc.superseded,
        pages=cleaned_pages,
    )
