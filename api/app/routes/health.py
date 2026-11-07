"""Liveness and readiness (section 4.3). They answer different questions."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, Response

log = logging.getLogger("rahbar.health")
router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    """Readiness: returns 200 only if Chroma is reachable, the collection is non-empty,
    Redis is reachable, and the collection metadata matches EMBED_MODEL.
    Otherwise returns 503 naming the failed dependency.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return {"status": "ready", "checks": {"state": "uninitialized"}}

    checks: dict[str, Any] = {}

    # If running with stub retriever, stubs are inherently ready
    if settings.retriever == "stub":
        checks["retriever"] = "stub"
        return {"status": "ready", "checks": checks}

    # 1. Chroma & Collection Check
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(settings.collection)
        chunk_count = collection.count()

        # Non-empty collection check
        if chunk_count == 0:
            response.status_code = 503
            return {
                "status": "not_ready",
                "failed_dependency": "chroma",
                "error": "Chroma collection is empty",
                "count": 0,
            }
        checks["chroma_chunks"] = chunk_count

        # 2. Embedding-Model Mismatch Guardrail
        col_meta = collection.metadata or {}
        indexed_model = col_meta.get("embed_model")
        if indexed_model and indexed_model != settings.embed_model:
            response.status_code = 503
            log.error(
                "Embedding model mismatch: Chroma has %s != API EMBED_MODEL %s",
                indexed_model,
                settings.embed_model,
            )
            return {
                "status": "not_ready",
                "failed_dependency": "embedding_model",
                "error": f"embed_model mismatch: collection '{indexed_model}' does not match config '{settings.embed_model}'",
            }
        checks["embed_model"] = settings.embed_model

    except Exception as exc:
        response.status_code = 503
        return {
            "status": "not_ready",
            "failed_dependency": "chroma",
            "error": f"Chroma unreachable: {exc}",
        }

    # 3. Redis Reachability Check (section 4.3)
    if getattr(settings, "redis_host", None):
        try:
            import redis

            r = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=1.0)
            r.ping()
            checks["redis"] = "connected"
        except Exception as exc:
            response.status_code = 503
            log.warning("Redis reachability check failed: %s", exc)
            return {
                "status": "not_ready",
                "failed_dependency": "redis",
                "error": f"Redis unreachable: {exc}",
            }

    return {"status": "ready", "checks": checks}
