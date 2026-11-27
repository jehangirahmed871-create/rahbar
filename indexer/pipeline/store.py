"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

from __future__ import annotations

import logging
import os
from typing import Any

from .config import IndexerSettings
from .models import Chunk

log = logging.getLogger("rahbar.indexer.store")


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    # Local mode: use PersistentClient (no Docker/network needed)
    local_dir = os.environ.get("CHROMA_LOCAL_DIR", "")
    if local_dir:
        log.info("Using local PersistentClient at %s", local_dir)
        return chromadb.PersistentClient(
            path=local_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def invalidate_cache(settings: IndexerSettings) -> None:
    """Invalidate semantic cache upon re-index so old stale answers are not served."""
    redis_host = os.environ.get("REDIS_HOST", "redis")
    port_val = str(os.environ.get("REDIS_PORT", "6379")).strip()
    if port_val.startswith("tcp://"):
        port_val = port_val.split(":")[-1]
    try:
        redis_port = int(port_val)
    except ValueError:
        redis_port = 6379
    try:
        import redis
    except ImportError as exc:
        log.info("Redis cache invalidation skipped; redis dependency unavailable: %s", exc)
        return
    try:
        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)
        r.flushdb()
        log.info("Flushed and invalidated semantic cache in Redis")
    except (redis.RedisError, OSError) as exc:
        log.info("Redis cache invalidation skipped or unreachable: %s", exc)


def store(
    chunks: list[Chunk],
    vectors: list[list[float]],
    settings: IndexerSettings,
    desired_document_version_ids: set[str] | None = None,
) -> int:
    """Replace changed document versions and delete chunks no longer in the corpus."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks received with {len(vectors)} vectors")

    client = get_client(settings)
    dim = len(vectors[0]) if vectors else 768
    collection = client.get_or_create_collection(
        name=settings.collection,
        metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model, "dimension": dim},
    )
    if getattr(collection, "metadata", {}).get("embed_model") != settings.embed_model:
        collection.modify(metadata={
            "embed_model": settings.embed_model,
            "dimension": dim,
        })
    desired = desired_document_version_ids
    if desired is None:
        desired = {chunk.document_version_id for chunk in chunks}
    changed = {chunk.document_version_id for chunk in chunks}
    current = collection.get(include=["metadatas"])
    delete_ids: list[str] = []
    for chunk_id, metadata in zip(current.get("ids", []), current.get("metadatas", [])):
        version_id = str((metadata or {}).get("document_version_id", ""))
        if not version_id or version_id not in desired or version_id in changed:
            delete_ids.append(str(chunk_id))
    if delete_ids:
        collection.delete(ids=delete_ids)

    total = len(chunks)
    for offset in range(0, total, 100):
        batch_chunks = chunks[offset : offset + 100]
        batch_vectors = vectors[offset : offset + 100]
        collection.upsert(
            ids=[item.id for item in batch_chunks],
            documents=[item.text for item in batch_chunks],
            embeddings=batch_vectors,
            metadatas=[item.metadata() for item in batch_chunks],
        )

    log.info(
        "Stored %d changed chunks in collection %s with model %s (dim=%d); deleted %d stale chunks",
        total, settings.collection, settings.embed_model, dim, len(delete_ids),
    )
    if total or delete_ids:
        invalidate_cache(settings)
    return total


def plan_incremental(chunks: list[Chunk], settings: IndexerSettings) -> list[Chunk]:
    """Return only chunks whose text, metadata, or chunking configuration changed."""
    client = get_client(settings)
    collection = client.get_or_create_collection(
        name=settings.collection,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": settings.embed_model,
            "dimension": 768,
        },
    )
    stored = collection.get(include=["documents", "metadatas"])
    records: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}
    for chunk_id, text, metadata in zip(
        stored.get("ids", []),
        stored.get("documents", []),
        stored.get("metadatas", []),
    ):
        if metadata:
            version_id = str(metadata.get("document_version_id", ""))
            if version_id:
                records.setdefault(version_id, {})[str(chunk_id)] = (str(text), dict(metadata))

    desired: dict[str, list[Chunk]] = {}
    for item in chunks:
        desired.setdefault(item.document_version_id, []).append(item)

    collection_model = (getattr(collection, "metadata", None) or {}).get("embed_model")
    changed: list[Chunk] = []
    for version_id, expected_chunks in desired.items():
        existing = records.get(version_id, {})
        unchanged = (
            collection_model == settings.embed_model
            and len(existing) == len(expected_chunks)
            and all(
                (stored := existing.get(expected.id)) is not None
                and stored[0] == expected.text
                and stored[1].get("document_hash") == expected.document_hash
                and stored[1].get("source") == expected.source
                and stored[1].get("document_id") == expected.document_id
                and stored[1].get("chunk_size") == expected.chunk_size
                and stored[1].get("chunk_overlap") == expected.chunk_overlap
                and stored[1].get("doc_version") == expected.doc_version
                and stored[1].get("title") == expected.title
                and stored[1].get("page_count") == expected.page_count
                and stored[1].get("superseded") == expected.superseded
                for expected in expected_chunks
            )
        )
        if unchanged:
            log.info("document unchanged; skipping chunk embeddings",
                     extra={"document_version_id": version_id, "chunks": len(expected_chunks)})
        else:
            changed.extend(expected_chunks)
    return changed


def restore_version(document_id: str, version: str, settings: IndexerSettings) -> int:
    """Restore one archived document version for retrieval."""
    client = get_client(settings)
    collection = client.get_collection(settings.collection)
    result = collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])
    if not ids:
        raise ValueError(f"no indexed versions found for document_id={document_id}")
    found_version = False
    restored: list[dict[str, Any]] = []
    for metadata in metadatas:
        updated = dict(metadata or {})
        if str(updated.get("doc_version", "")) == version:
            found_version = True
            updated["superseded"] = False
        else:
            updated["superseded"] = True
        restored.append(updated)
    if not found_version:
        raise ValueError(f"version {version} is not indexed for document_id={document_id}")
    collection.update(ids=ids, metadatas=restored)
    invalidate_cache(settings)
    log.info("document version restored", extra={
        "document_id": document_id,
        "doc_version": version,
        "chunks_updated": len(ids),
    })
    return len(ids)
