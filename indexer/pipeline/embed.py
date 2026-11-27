"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from itertools import pairwise

from .config import IndexerSettings

log = logging.getLogger("rahbar.indexer.embed")

DEFAULT_DIMENSION = 768
EMBEDDER_ID = "offline-feature-hash-v1"


def _fallback_vector(text: str, dim: int = DEFAULT_DIMENSION) -> list[float]:
    """Deterministic lexical feature vector for offline tests and local operation."""
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    features: Counter[str] = Counter(tokens)
    features.update(f"{left} {right}" for left, right in pairwise(tokens))
    values = [0.0] * dim
    for feature, weight in features.items():
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign * weight
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Create deterministic local feature-hashing vectors without a model service."""
    if not texts:
        return []
    if settings.embed_model != EMBEDDER_ID:
        raise ValueError(
            f"offline embedder requires EMBED_MODEL={EMBEDDER_ID}, got {settings.embed_model}"
        )
    log.info("Embedding %d chunks with %s", len(texts), EMBEDDER_ID)
    return [_fallback_vector(text, DEFAULT_DIMENSION) for text in texts]
