"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page."""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path

from .models import Document, Page

log = logging.getLogger("rahbar.indexer.extract")


class _SimpleHTMLTextExtractor(HTMLParser):
    """Extract plain text and tabular structure from HTML documents."""

    def __init__(self) -> None:
        super().__init__()
        self.pieces: list[str] = []
        self._in_table = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.pieces.append("\n\n### ")
        elif tag in ("p", "div", "br", "li"):
            self.pieces.append("\n")
        elif tag == "table":
            self._in_table = True
            self.pieces.append("\n\n")
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)
        else:
            self.pieces.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th"):
            self._in_cell = False
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
        elif tag == "tr":
            if self._current_row:
                self.pieces.append("| " + " | ".join(self._current_row) + " |\n")
        elif tag == "table":
            self._in_table = False
            self.pieces.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.pieces.append("\n")

    def get_text(self) -> str:
        raw = "".join(self.pieces)
        # normalize blank lines
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _extract_pdf(path: Path) -> list[Page]:
    """Extract text from a PDF document page-by-page.

    If a page contains no text layer, OCR is attempted using pytesseract and pdf2image.
    """
    import pdfplumber
    from pypdf import PdfReader

    pages: list[Page] = []
    source = path.name

    # Try extraction with pdfplumber first for better layout and table handling
    try:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # Also check tables on this page to ensure tabular numbers are kept
                tables = page.extract_tables()
                if tables:
                    table_lines = []
                    for t in tables:
                        for row in t:
                            cells = [str(c or "").strip() for c in row]
                            if any(cells):
                                table_lines.append("| " + " | ".join(cells) + " |")
                    if table_lines:
                        text = text + "\n\n" + "\n".join(table_lines)

                is_ocr = False
                if len(text.strip()) < 10:
                    # No extractable text; attempt OCR
                    ocr_text = _attempt_ocr(path, idx)
                    if ocr_text:
                        text = ocr_text
                        is_ocr = True
                    else:
                        is_ocr = True  # Scanned page detected and flagged
                pages.append(Page(source=source, page=idx, text=text, ocr=is_ocr))
    except Exception as exc:
        log.warning("pdfplumber failed for %s, falling back to pypdf: %s", path.name, exc)
        reader = PdfReader(str(path))
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            is_ocr = False
            if len(text.strip()) < 10:
                ocr_text = _attempt_ocr(path, idx)
                if ocr_text:
                    text = ocr_text
                    is_ocr = True
                else:
                    is_ocr = True
            pages.append(Page(source=source, page=idx, text=text, ocr=is_ocr))

    return pages


def _attempt_ocr(path: Path, page_num: int) -> str | None:
    """Attempt OCR using pdf2image and pytesseract on the specified 1-based page number."""
    try:
        import pdf2image
        import pytesseract

        images = pdf2image.convert_from_path(str(path), first_page=page_num, last_page=page_num)
        if images:
            text = pytesseract.image_to_string(images[0])
            if text and text.strip():
                return text.strip()
    except Exception as exc:
        log.info("OCR not executed for %s page %d (%s)", path.name, page_num, exc)
    return None


def _extract_html(path: Path) -> list[Page]:
    """Extract text from an HTML document."""
    html_content = path.read_text(encoding="utf-8", errors="replace")
    parser = _SimpleHTMLTextExtractor()
    parser.feed(html_content)
    text = parser.get_text()
    return [Page(source=path.name, page=1, text=text, ocr=False)]


def _extract_markdown(path: Path) -> list[Page]:
    """Extract text from Markdown or plain text document."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Strip front matter if present
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    return [Page(source=path.name, page=1, text=text.strip(), ocr=False)]


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    Detect pages with no text layer and OCR them; mark those pages ocr=True.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"document not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf(p)
    elif suffix in (".html", ".htm"):
        pages = _extract_html(p)
    elif suffix in (".md", ".txt"):
        pages = _extract_markdown(p)
    else:
        # Fallback reading as text
        pages = [Page(source=p.name, page=1, text=p.read_text(encoding="utf-8", errors="replace"), ocr=False)]

    return Document(
        source=p.name,
        doc_version=doc_version,
        superseded=superseded,
        pages=pages,
    )
