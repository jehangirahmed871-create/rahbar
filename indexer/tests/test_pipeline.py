"""Unit tests for indexer pipeline stages: extract, clean, chunk, embed, store."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pipeline.chunk import chunk
from pipeline.clean import clean
from pipeline.config import IndexerSettings
from pipeline.embed import embed
from pipeline.extract import extract
from pipeline.models import Chunk, Document, Page
from pipeline.store import store


class MockCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata
        self.records: dict[str, tuple[str, list[float], dict]] = {}

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.records[i] = (d, e, m)

    def count(self) -> int:
        return len(self.records)


class MockClient:
    def __init__(self) -> None:
        self.collections: dict[str, MockCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict) -> MockCollection:
        if name not in self.collections:
            self.collections[name] = MockCollection(name, metadata)
        else:
            self.collections[name].metadata.update(metadata)
        return self.collections[name]

    def create_collection(self, name: str, metadata: dict) -> MockCollection:
        return self.get_or_create_collection(name, metadata)


def test_extract_markdown_document(tmp_path: Path) -> None:
    md_file = tmp_path / "test-rules.md"
    md_file.write_text(
        "---\ndoc_version: 2026\nsuperseded: false\n---\n# 1. Attendance\nStudents must have 80% attendance.",
        encoding="utf-8",
    )
    doc = extract(str(md_file), doc_version="2026", superseded=False)
    assert doc.source == "test-rules.md"
    assert doc.doc_version == "2026"
    assert doc.superseded is False
    assert len(doc.pages) == 1
    assert "80% attendance" in doc.pages[0].text


def test_clean_strips_repeating_headers_footers() -> None:
    page1 = Page(
        source="doc.pdf",
        page=1,
        text="FAST-NUCES Academic Rules\n# 1. Intro\nContent page one.\nConfidential\nPage 1 of 3",
    )
    page2 = Page(
        source="doc.pdf",
        page=2,
        text="FAST-NUCES Academic Rules\n# 2. Grading\nContent page two.\nConfidential\nPage 2 of 3",
    )
    page3 = Page(
        source="doc.pdf",
        page=3,
        text="FAST-NUCES Academic Rules\n# 3. Warning\nContent page three.\nConfidential\nPage 3 of 3",
    )
    doc = Document(source="doc.pdf", doc_version="2026", superseded=False, pages=[page1, page2, page3])

    cleaned = clean(doc)
    assert len(cleaned.pages) == 3
    # Check that recurring header and footer lines were stripped
    for p in cleaned.pages:
        assert "FAST-NUCES Academic Rules" not in p.text
        assert "Page 1 of 3" not in p.text
    # Check that headings and main content were preserved
    assert "# 1. Intro" in cleaned.pages[0].text
    assert "# 2. Grading" in cleaned.pages[1].text
    assert "# 3. Warning" in cleaned.pages[2].text


def test_chunk_preserves_sections_and_metadata() -> None:
    page = Page(
        source="academic-rules.pdf",
        page=4,
        text="Clause 1.21 Mandatory Attendance\nAll students must attend 80 percent of lectures.\n\n"
             "Clause 5.4 Academic Warning\nA student whose CGPA falls below 2.00 is placed on academic warning.",
    )
    doc = Document(source="academic-rules.pdf", doc_version="2026-06", superseded=False, pages=[page])

    chunks = chunk(doc, size=50, overlap=10)
    assert len(chunks) >= 2
    assert any("Attendance" in c.section for c in chunks)
    assert any("Warning" in c.section for c in chunks)

    for c in chunks:
        assert c.source == "academic-rules.pdf"
        assert c.page == 4
        assert c.doc_version == "2026-06"
        assert c.superseded is False
        meta = c.metadata()
        assert set(meta.keys()) == {"source", "section", "page", "doc_version", "superseded"}


def test_chunk_overlap_shares_tokens_between_adjacent_chunks() -> None:
    # A single long paragraph that must split into multiple chunks
    long_para = " ".join([f"Sentence {i} gives critical detailed institutional instruction." for i in range(40)])
    page = Page(source="rules.pdf", page=1, text=f"1. Policy Guidelines\n\n{long_para}")
    doc = Document(source="rules.pdf", doc_version="2026", superseded=False, pages=[page])

    chunks = chunk(doc, size=30, overlap=10)
    assert len(chunks) > 1
    # Check that chunk 0 and chunk 1 share words from the boundary
    c0_words = set(chunks[0].text.split())
    c1_words = set(chunks[1].text.split())
    overlap_words = c0_words.intersection(c1_words)
    # The overlap should contain words other than just the section header
    assert len(overlap_words) >= 3


def test_embed_returns_normalized_vectors_offline() -> None:
    settings = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="localhost",
        chroma_port=8000,
        collection="test-col",
        corpus_dir="/tmp",
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://127.0.0.1:9999",  # unroutable port forces offline fallback
    )
    texts = ["First chunk of policy text.", "Second chunk of policy text."]
    vectors = embed(texts, settings)
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    # Fallback vectors are unit normalized
    assert math.isclose(sum(x * x for x in vectors[0]), 1.0, rel_tol=1e-5)


def test_store_writes_chunks_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockClient()
    settings = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar-test",
        corpus_dir="/tmp",
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://localhost:11434",
    )
    monkeypatch.setattr("pipeline.store.get_client", lambda s: mock_client)
    monkeypatch.setattr("pipeline.store.invalidate_cache", lambda s: None)

    chunks = [
        Chunk(id="c1", text="text 1", source="rules.pdf", section="1.1", page=1, doc_version="2026", superseded=False),
        Chunk(id="c2", text="text 2", source="rules.pdf", section="1.2", page=2, doc_version="2026", superseded=False),
    ]
    vectors = [[0.1] * 768, [0.2] * 768]

    count = store(chunks, vectors, settings)
    assert count == 2
    col = mock_client.collections["rahbar-test"]
    assert col.count() == 2
    assert col.metadata["embed_model"] == "nomic-embed-text"
    assert col.metadata["dimension"] == 768
