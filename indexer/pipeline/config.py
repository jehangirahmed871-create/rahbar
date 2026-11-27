"""Indexer configuration, from the same ConfigMap the API reads (section 4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerSettings:
    embed_model: str        # EMBED_MODEL: must equal the API's value (section 4.4)
    chroma_host: str
    chroma_port: int
    collection: str
    corpus_dir: str         # where the documents and _index.json sit inside the Job's pod
    chunk_size: int
    chunk_overlap: int
    quarantine_dir: str = ""
    archive_dir: str = ""
    verification_dir: str = ""
    near_duplicate_threshold: float = 0.95


def _parse_port(val: str | int | None, default: int) -> int:
    s = str(val or default).strip()
    if s.startswith("tcp://"):
        s = s.split(":")[-1]
    try:
        return int(s)
    except ValueError:
        return default


def get_settings() -> IndexerSettings:
    return IndexerSettings(
        embed_model=os.environ.get("EMBED_MODEL", "offline-feature-hash-v1"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=_parse_port(os.environ.get("CHROMA_PORT", "8000"), 8000),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        corpus_dir=os.environ.get("CORPUS_DIR", "/corpus"),
        chunk_size=_parse_port(os.environ.get("CHUNK_SIZE", "400"), 400),
        chunk_overlap=_parse_port(os.environ.get("CHUNK_OVERLAP", "50"), 50),
        quarantine_dir=os.environ.get("QUARANTINE_DIR", os.path.join(
            os.environ.get("CORPUS_DIR", "/corpus"), "quarantine")),
        archive_dir=os.environ.get("ARCHIVE_DIR", os.path.join(
            os.environ.get("CORPUS_DIR", "/corpus"), "archive")),
        verification_dir=os.environ.get("VERIFICATION_DIR", os.path.join(
            os.environ.get("CORPUS_DIR", "/corpus"), "verification")),
        near_duplicate_threshold=float(os.environ.get("NEAR_DUPLICATE_THRESHOLD", "0.95")),
    )
