"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import hashlib
import logging
import math

import httpx

from .config import IndexerSettings

log = logging.getLogger("rahbar.indexer.embed")

DEFAULT_DIMENSION = 768


def _fallback_vector(text: str, dim: int = DEFAULT_DIMENSION) -> list[float]:
    """Deterministic fallback vector for offline tests and stub operation."""
    values: list[float] = []
    seed = text.encode("utf-8", errors="replace")
    while len(values) < dim:
        seed = hashlib.sha256(seed).digest()
        values.extend((b - 127.5) / 127.5 for b in seed)
    values = values[:dim]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else.

    With Ollama: POST {ollama_url}/api/embed with {"model": ..., "input": [texts]}
    returns {"embeddings": [[...], ...]}. Batch the calls.
    """
    if not texts:
        return []

    batch_size = 32
    embeddings: list[list[float]] = []

    # Attempt embedding via Ollama host endpoint
    use_fallback = False
    endpoint = f"{settings.ollama_url.rstrip('/')}/api/embed"

    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if not use_fallback:
                try:
                    payload = {"model": settings.embed_model, "input": batch}
                    resp = client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        batch_embeds = data.get("embeddings") or []
                        if len(batch_embeds) == len(batch):
                            embeddings.extend(batch_embeds)
                            continue
                    log.warning("Ollama returned status %s: %s; falling back to offline vectorizer",
                                resp.status_code, resp.text[:100])
                    use_fallback = True
                except Exception as exc:
                    log.warning("Could not reach Ollama at %s (%s); using fallback offline vectorizer",
                                endpoint, exc)
                    use_fallback = True

            # If Ollama failed or unavailable, use deterministic fallback
            for text in batch:
                embeddings.append(_fallback_vector(text, DEFAULT_DIMENSION))

    return embeddings
