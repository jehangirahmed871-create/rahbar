"""Unit tests for indexer pipeline stages: extract, clean, chunk, embed, store."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from pipeline.chunk import chunk
from pipeline.clean import clean
from pipeline.config import IndexerSettings
from pipeline.embed import embed
from pipeline.extract import extract
from pipeline.models import Chunk, Document, Page
from pipeline.store import plan_incremental, store
from pipeline.verify import CHECK_NAMES, verify_documents


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

    def get(self, where=None, include=None) -> dict:
        rows = self.records.items()
        if where and "document_id" in where:
            rows = [
                (chunk_id, row) for chunk_id, row in rows
                if row[2].get("document_id") == where["document_id"]
            ]
        return {
            "ids": [chunk_id for chunk_id, _ in rows],
            "documents": [row[0] for _, row in rows],
            "metadatas": [row[2] for _, row in rows],
        }

    def delete(self, ids) -> None:
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

    def update(self, ids, metadatas) -> None:
        for chunk_id, metadata in zip(ids, metadatas):
            text, vector, _old_metadata = self.records[chunk_id]
            self.records[chunk_id] = (text, vector, metadata)


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

    def get_collection(self, name: str) -> MockCollection:
        return self.collections[name]


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
        assert set(meta.keys()) == {
            "source", "section", "page", "doc_version", "superseded",
            "document_id", "document_version_id", "document_hash", "title", "page_count",
            "chunk_size", "chunk_overlap",
        }


def test_incremental_store_skips_unchanged_chunks_and_removes_deleted_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr("pipeline.store.get_client", lambda _settings: mock_client)
    monkeypatch.setattr("pipeline.store.invalidate_cache", lambda _settings: None)
    item = Chunk(
        id="rules#intro#p1#c0", text="academic policy", source="rules.md", section="Intro",
        page=1, doc_version="2026", superseded=False, document_id="rules",
        document_version_id="rules@2026", document_hash="hash-a", title="Rules",
        page_count=1, chunk_size=400, chunk_overlap=50,
    )
    other = Chunk(
        id="fees#intro#p1#c0", text="tuition payment", source="fees.md", section="Intro",
        page=1, doc_version="2026", superseded=False, document_id="fees",
        document_version_id="fees@2026", document_hash="hash-fees", title="Fees",
        page_count=1, chunk_size=400, chunk_overlap=50,
    )

    assert plan_incremental([item, other], settings) == [item, other]
    assert store([item, other], [[1.0, 0.0], [0.0, 1.0]], settings,
                 {"rules@2026", "fees@2026"}) == 2
    assert plan_incremental([item, other], settings) == []
    assert store([], [], settings, {"rules@2026", "fees@2026"}) == 0
    assert mock_client.collections["rahbar-test"].count() == 2

    changed = Chunk(
        **{**item.__dict__, "text": "updated academic policy", "document_hash": "hash-b"}
    )
    assert plan_incremental([changed, other], settings) == [changed]
    assert store([changed], [[1.0, 1.0]], settings,
                 {"rules@2026", "fees@2026"}) == 1
    rows = mock_client.collections["rahbar-test"].records
    assert rows[item.id][0] == "updated academic policy"
    assert rows[other.id][0] == "tuition payment"

    assert store([], [], settings, {"rules@2026"}) == 0
    assert mock_client.collections["rahbar-test"].count() == 1
    assert store([], [], settings, set()) == 0
    assert mock_client.collections["rahbar-test"].count() == 0


def test_verification_reports_all_checks_and_rejects_a_near_duplicate() -> None:
    settings = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar-test",
        corpus_dir="/tmp",
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://localhost:11434",
        near_duplicate_threshold=0.9,
    )
    documents = [
        Document(
            source="rules-2023.md", doc_version="2023", superseded=True,
            pages=[Page(source="rules-2023.md", page=1, text="A complete policy document with enough text.")],
            title="Rules", document_id="rules",
        ),
        Document(
            source="rules-2026.md", doc_version="2026", superseded=False,
            pages=[Page(source="rules-2026.md", page=1, text="A newer and complete policy document.")],
            title="Rules", document_id="rules",
        ),
        Document(
            source="near.md", doc_version="2026", superseded=False,
            pages=[Page(source="near.md", page=1, text="A near duplicate policy document.")],
            title="Near copy", document_id="near",
        ),
        Document(
            source="missing-title.md", doc_version="2026", superseded=False,
            pages=[Page(source="missing-title.md", page=1, text="Text is present but metadata is incomplete.")],
            document_id="missing-title",
        ),
    ]

    def embedder(texts: list[str], _settings: IndexerSettings) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    records = verify_documents(documents, settings, embedder=embedder)
    assert set(records[0].checks) == set(CHECK_NAMES)
    assert records[0].checks["recency"].passed
    assert records[1].checks["recency"].passed
    assert not records[2].checks["uniqueness"].passed
    assert "embedding cosine similarity" in records[2].checks["uniqueness"].reason
    assert not records[3].checks["completeness"].passed


def test_pipeline_quarantines_batch_b_and_is_incremental(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pipeline import embed as embed_module
    from pipeline import run as run_module
    from pipeline import verify as verify_module
    from pipeline.embed import _fallback_vector

    source_dir = Path(__file__).resolve().parents[2] / "eval" / "fixtures" / "batch-b"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for source in source_dir.glob("*.md"):
        shutil.copyfile(source, corpus_dir / source.name)

    settings = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="unused",
        chroma_port=8000,
        collection="rahbar-test",
        corpus_dir=str(corpus_dir),
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://127.0.0.1:9999",
        quarantine_dir=str(tmp_path / "quarantine"),
        archive_dir=str(tmp_path / "archive"),
        verification_dir=str(tmp_path / "verification"),
    )
    client = MockClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda _settings: client)
    monkeypatch.setattr("pipeline.store.invalidate_cache", lambda _settings: None)
    monkeypatch.setattr(run_module, "_wait_for_chroma", lambda _settings: None)
    monkeypatch.setattr(
        verify_module,
        "embed",
        lambda texts, _settings: [_fallback_vector(text, 64) for text in texts],
    )
    chunk_embedding_calls: list[list[str]] = []

    def embed_chunks(texts: list[str], _settings: IndexerSettings) -> list[list[float]]:
        chunk_embedding_calls.append(texts)
        return [_fallback_vector(text, 64) for text in texts]

    monkeypatch.setattr(embed_module, "embed", embed_chunks)

    first_count = run_module.run_pipeline(settings)
    assert first_count > 0
    assert chunk_embedding_calls[-1]
    reports = list((tmp_path / "verification").glob("verification-*.json"))
    assert len(reports) == 1
    import json

    report = json.loads(reports[0].read_text(encoding="utf-8"))
    checks_by_source = {item["source"]: item["checks"] for item in report["documents"]}
    assert checks_by_source["exact-duplicate-b.md"]["uniqueness"]["status"] == "fail"
    assert checks_by_source["near-duplicate-b.md"]["uniqueness"]["status"] == "fail"
    assert checks_by_source["missing-title.md"]["completeness"]["status"] == "fail"
    assert checks_by_source["truncated-document.md"]["completeness"]["status"] == "fail"
    assert list((tmp_path / "quarantine").rglob("*.md"))
    assert (tmp_path / "archive" / "undergraduate-academic-rules" / "2026" /
            "academic-rules-2026.md").is_file()

    second_count = run_module.run_pipeline(settings)
    assert second_count == 0
    assert chunk_embedding_calls[-1] == []
    assert len(chunk_embedding_calls) == 2

    rules_collection = client.collections["rahbar-test"]
    rules = rules_collection.get(where={"document_id": "undergraduate-academic-rules"})
    assert {metadata["doc_version"] for metadata in rules["metadatas"]} == {"2026", "2027"}
    assert store([], [], settings, {"undergraduate-academic-rules@2026",
                                    "undergraduate-academic-rules@2027"}) == 0
    from pipeline.store import restore_version

    restore_version("undergraduate-academic-rules", "2026", settings)
    restored = rules_collection.get(where={"document_id": "undergraduate-academic-rules"})
    status_by_version = {metadata["doc_version"]: metadata["superseded"]
                         for metadata in restored["metadatas"]}
    assert status_by_version == {"2026": False, "2027": True}

    removed_source = corpus_dir / "fee-payment.md"
    removed_source.unlink()
    run_module.run_pipeline(settings)
    remaining = rules_collection.get()
    assert all(metadata.get("document_id") != "fee-payment-schedule"
               for metadata in remaining["metadatas"])


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
