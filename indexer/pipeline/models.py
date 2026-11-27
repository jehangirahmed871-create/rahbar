"""The shapes passed between stages. Metadata fields are required (section 3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Page:
    """Text of one page, as extraction produced it."""

    source: str              # file name, e.g. Academic_Rules_Handbook_2026.pdf
    page: int                # 1-based
    text: str
    ocr: bool = False        # True when this page had no text layer and was OCR'd


@dataclass
class Document:
    source: str
    doc_version: str         # e.g. "2026-06"; from corpus/sources.yaml
    superseded: bool         # True for an older version of a policy
    pages: list[Page] = field(default_factory=list)
    title: str = ""
    document_id: str = ""
    text_truncated: bool = False
    content_hash: str = ""
    extraction_error: str = ""
    source_path: str = ""


@dataclass
class Chunk:
    id: str                  # stable: same document + same position gives the same id
    text: str
    source: str
    section: str             # the heading this chunk sits under, e.g. "5.4 Academic Warning"
    page: int
    doc_version: str
    superseded: bool
    document_id: str = ""
    document_version_id: str = ""
    document_hash: str = ""
    title: str = ""
    page_count: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0

    def metadata(self) -> dict[str, str | int | bool]:
        """Chroma metadata values must be str, int, float or bool."""
        return {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "doc_version": self.doc_version,
            "superseded": self.superseded,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "document_hash": self.document_hash,
            "title": self.title,
            "page_count": self.page_count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
