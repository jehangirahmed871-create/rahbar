"""The real retriever against ChromaDB (sections 3.2 and 4.4)."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from itertools import pairwise

import httpx

from ..config import Settings
from ..schemas import Passage

log = logging.getLogger("rahbar.retrieval.chroma")

_STOPWORDS = {"a", "an", "and", "are", "be", "for", "from", "how", "in", "is", "of", "on", "or", "the", "to", "what", "when", "which", "who", "with"}


def _terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9][a-z0-9.-]{1,}", text.lower()) if word not in _STOPWORDS}


def _get_query_vector(text: str, _settings: Settings, dim: int = 768) -> list[float]:
    """Create the same deterministic feature-hash vector as the indexer, locally."""
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    features: Counter[str] = Counter(tokens)
    features.update(f"{left} {right}" for left, right in pairwise(tokens))
    values = [0.0] * dim
    for feature, weight in features.items():
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings

        # Local mode: use PersistentClient (no Docker/network needed)
        local_dir = os.environ.get("CHROMA_LOCAL_DIR", "")
        if local_dir:
            log.info("Using local PersistentClient at %s", local_dir)
            self.client = chromadb.PersistentClient(
                path=local_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def retrieve(self, question: str, k: int) -> list[Passage]:
        """Query Chroma for top k chunks, filtering out superseded policy documents
        unless explicitly requested by user inquiry.
        """
        from chromadb.errors import ChromaError

        try:
            collection = self.client.get_collection(self.settings.collection)
        except (ChromaError, httpx.HTTPError, OSError) as exc:
            log.error("Failed to access collection %s: %s", self.settings.collection, exc)
            return []

        # Failure Mode E Guardrail: Exclude superseded documents unless user explicitly asks
        lower_q = question.lower()
        allow_superseded = "2023" in lower_q or "superseded" in lower_q or "old version" in lower_q or "previous policy" in lower_q
        where_filter = None if allow_superseded else {"superseded": False}

        try:
            all_rows = collection.get(
                where=where_filter,  # type: ignore[arg-type]
                include=["documents", "metadatas"],
            )
        except (ChromaError, httpx.HTTPError, OSError, ValueError) as exc:
            log.error("Lexical Chroma retrieval failed: %s", exc)
            return []

        query_terms = _terms(question)
        candidates: list[Passage] = []
        for chunk_id, text, meta in zip(
            all_rows.get("ids") or [],
            all_rows.get("documents") or [],
            all_rows.get("metadatas") or [],
        ):
            chunk_terms = _terms(str(text))
            if not query_terms or not chunk_terms:
                continue
            overlap = len(query_terms & chunk_terms) / len(query_terms)
            if overlap <= 0:
                continue
            candidates.append(Passage(
                chunk_id=str(chunk_id), text=str(text),
                source=str(meta.get("source", "unknown")),
                section=str(meta.get("section", "General")),
                page=int(str(meta["page"])) if meta.get("page") is not None else None,
                score=round(min(1.0, overlap), 4),
                document_id=str(meta.get("document_id", "")) or None,
                doc_version=str(meta.get("doc_version", "")) or None,
            ))
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:k]
