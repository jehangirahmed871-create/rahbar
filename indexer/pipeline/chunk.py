"""Stage 3: Chunk. Structure-aware, with overlap."""

from __future__ import annotations

import re

from .models import Chunk, Document

HEADING_PATTERNS = [
    re.compile(r"^#{1,4}\s+(.+)$"),
    re.compile(r"^(?:Section|Clause|Rule|Article)\s+([0-9]+(?:\.[0-9]+)*\b.*)$", re.I),
    re.compile(r"^([0-9]+(?:\.[0-9]+)+\.?\s+[A-Za-z].*)$"),
    re.compile(r"^([0-9]+\.\s+[A-Za-z][\w\s]{2,50})$"),
]


def _slugify(val: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", val.strip().lower()).strip("-")
    return cleaned[:40] or "general"


def _is_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 120:
        return None
    for pattern in HEADING_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def _token_count(text: str) -> int:
    """Approximate token count based on whitespace splitting (standard ~1.3 tokens per word)."""
    return len(text.split())


def _split_into_sentences(text: str) -> list[str]:
    """Split paragraph text into sentences preserving punctuation."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk(doc: Document, size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Structure-aware chunker with overlap.

    Splits on section headings, then paragraphs, then sentences, keeping each chunk under
    `size` tokens with `overlap` tokens shared between neighbours.
    """
    chunks: list[Chunk] = []
    chunk_idx = 0
    current_section = f"General ({doc.source})"

    for page in doc.pages:
        page_num = page.page
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]

        for para in paragraphs:
            # Check if this paragraph or its first line is a section heading
            first_line = para.splitlines()[0].strip()
            detected_heading = _is_heading(first_line)
            if detected_heading:
                current_section = detected_heading
                # If paragraph only had heading, advance
                remainder = "\n".join(para.splitlines()[1:]).strip()
                if not remainder:
                    continue
                para = remainder

            para_tokens = _token_count(para)
            if para_tokens <= size:
                # Small enough to be a standalone unit or merged
                chunk_id = f"{doc.source}#{_slugify(current_section)}#p{page_num}#c{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=f"[{current_section}]\n{para}",
                        source=doc.source,
                        section=current_section,
                        page=page_num,
                        doc_version=doc.doc_version,
                        superseded=doc.superseded,
                    )
                )
                chunk_idx += 1
            else:
                # Structure-aware splitting: break paragraph by sentences
                sentences = _split_into_sentences(para)
                window: list[str] = []
                window_tokens = 0

                for sent in sentences:
                    sent_tokens = _token_count(sent)
                    if window_tokens + sent_tokens > size and window:
                        chunk_text = " ".join(window)
                        chunk_id = f"{doc.source}#{_slugify(current_section)}#p{page_num}#c{chunk_idx}"
                        chunks.append(
                            Chunk(
                                id=chunk_id,
                                text=f"[{current_section}]\n{chunk_text}",
                                source=doc.source,
                                section=current_section,
                                page=page_num,
                                doc_version=doc.doc_version,
                                superseded=doc.superseded,
                            )
                        )
                        chunk_idx += 1

                        # Carry forward overlap sentences
                        overlap_tokens = 0
                        overlap_window: list[str] = []
                        for s in reversed(window):
                            st = _token_count(s)
                            if overlap_tokens + st <= overlap:
                                overlap_window.insert(0, s)
                                overlap_tokens += st
                            else:
                                break
                        window = overlap_window
                        window_tokens = overlap_tokens

                    window.append(sent)
                    window_tokens += sent_tokens

                if window:
                    chunk_text = " ".join(window)
                    chunk_id = f"{doc.source}#{_slugify(current_section)}#p{page_num}#c{chunk_idx}"
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=f"[{current_section}]\n{chunk_text}",
                            source=doc.source,
                            section=current_section,
                            page=page_num,
                            doc_version=doc.doc_version,
                            superseded=doc.superseded,
                        )
                    )
                    chunk_idx += 1

    return chunks
