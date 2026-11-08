"""Semantic cache implementation for RAHBAR (section 3.3).

Stores question embeddings and verified responses in Redis with an in-memory fallback.
Validates cosine similarity against a configurable threshold.
Strictly implements never-cache rules for follow-ups, abstentions, and ungrounded responses.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from ..config import Settings
from ..schemas import AskResponse, Citation, RetrievedRef

log = logging.getLogger("rahbar.cache")


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class SemanticCache:
    """Semantic cache with cosine similarity matching and strict safety gates."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.threshold = settings.cache_threshold
        self._memory_cache: list[dict[str, Any]] = []
        self._redis = None

        try:
            import redis

            client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                socket_timeout=1.5,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            log.info("Semantic cache connected to Redis at %s:%d", settings.redis_host, settings.redis_port)
        except Exception as exc:
            log.info("Redis not reachable (%s); operating with in-memory semantic cache", exc)

    def is_followup(self, question: str) -> bool:
        """Detect conversational follow-up indicators that depend on prior context."""
        lowered = question.lower().strip()
        starters = ("what about", "and then", "why is that", "can they", "does it", "what else")
        return any(lowered.startswith(s) for s in starters) or len(question.split()) < 3

    def never_cache(self, response: AskResponse, is_followup: bool = False) -> bool:
        """Guardrail: returns True if this turn must NEVER be saved in cache.

        Abstentions, follow-ups, ungrounded responses, or low-confidence outputs
        must never be stored.
        """
        if is_followup:
            return True
        if not response.answered_from_context:
            return True
        if response.confidence == "low":
            return True
        if not response.citations:
            return True
        return False

    def get(self, question: str, vector: list[float], is_followup: bool = False) -> AskResponse | None:
        """Query semantic cache for an entry with cosine similarity >= threshold."""
        if is_followup or self.is_followup(question):
            return None

        candidates = self._load_entries()
        best_match: dict[str, Any] | None = None
        best_score = -1.0

        for entry in candidates:
            cached_vector = entry.get("vector")
            if not cached_vector:
                continue
            sim = cosine_similarity(vector, cached_vector)
            if sim > best_score:
                best_score = sim
                best_match = entry

        if best_match and best_score >= self.threshold:
            log.info("Semantic cache HIT (similarity=%.4f >= threshold=%.2f)", best_score, self.threshold)
            resp_data = best_match["response"]
            return AskResponse(
                request_id=resp_data.get("request_id", "cached"),
                answer=resp_data["answer"],
                citations=[Citation(**c) for c in resp_data.get("citations", [])],
                answered_from_context=resp_data.get("answered_from_context", True),
                confidence=resp_data.get("confidence", "high"),
                retrieved=[RetrievedRef(**r) for r in resp_data.get("retrieved", [])],
                provider=f"{resp_data.get('provider', 'cached')} (cache)",
            )

        return None

    def store(self, question: str, vector: list[float], response: AskResponse, is_followup: bool = False) -> bool:
        """Save a verified turn into the semantic cache if safe."""
        if self.never_cache(response, is_followup=is_followup):
            log.info("Turn not cached due to never_cache safety policy (abstention / followup / low-confidence)")
            return False

        entry = {
            "question": question,
            "vector": vector,
            "response": response.model_dump(),
        }

        if self._redis:
            try:
                import uuid

                key = f"rahbar:cache:{uuid.uuid4().hex[:12]}"
                self._redis.set(key, json.dumps(entry), ex=86400 * 7)  # 7-day TTL
                return True
            except Exception as exc:
                log.warning("Redis store failed (%s), writing to in-memory fallback", exc)

        self._memory_cache.append(entry)
        return True

    def invalidate_cache(self) -> None:
        """Expire and flush all cached items upon re-indexing."""
        self._memory_cache.clear()
        if self._redis:
            try:
                self._redis.flushdb()
                log.info("Flushed semantic cache in Redis")
            except Exception as exc:
                log.warning("Could not flush Redis cache: %s", exc)

    def _load_entries(self) -> list[dict[str, Any]]:
        if self._redis:
            try:
                entries: list[dict[str, Any]] = []
                keys = self._redis.keys("rahbar:cache:*")
                for k in keys:
                    raw = self._redis.get(k)
                    if raw:
                        entries.append(json.loads(raw))
                return entries
            except Exception as exc:
                log.warning("Error reading Redis cache keys: %s", exc)
        return self._memory_cache
