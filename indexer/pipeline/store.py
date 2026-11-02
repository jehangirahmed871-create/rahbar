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

    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def invalidate_cache(settings: IndexerSettings) -> None:
    """Invalidate semantic cache upon re-index so old stale answers are not served."""
    redis_host = os.environ.get("REDIS_HOST", "redis")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    try:
        import redis

        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)
        r.flushdb()
        log.info("Flushed and invalidated semantic cache in Redis")
    except Exception as exc:
        log.info("Redis cache invalidation skipped or unreachable: %s", exc)


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count.
    """
    if not chunks:
        log.warning("No chunks to store.")
        return 0

    client = get_client(settings)
    dim = len(vectors[0]) if vectors and len(vectors) > 0 else 768

    col_metadata = {
        "hnsw:space": "cosine",
        "embed_model": settings.embed_model,
        "dimension": dim,
    }

    try:
        # If collection exists, update metadata; or create if not present
        collection = client.get_or_create_collection(
            name=settings.collection,
            metadata=col_metadata,
        )
    except Exception:
        collection = client.create_collection(
            name=settings.collection,
            metadata=col_metadata,
        )

    # Batch upsert chunks to Chroma in batches of 100
    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]
        collection.upsert(
            ids=[c.id for c in batch_chunks],
            documents=[c.text for c in batch_chunks],
            embeddings=batch_vectors,
            metadatas=[c.metadata() for c in batch_chunks],
        )

    log.info("Stored %d chunks in collection %s with model %s (dim=%d)",
             total, settings.collection, settings.embed_model, dim)

    # Invalidate semantic cache on index completion
    invalidate_cache(settings)

    return total
