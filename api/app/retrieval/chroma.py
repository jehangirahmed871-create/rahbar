"""The real retriever against ChromaDB (sections 3.2 and 4.4)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re

import httpx

from ..config import Settings
from ..schemas import Passage

log = logging.getLogger("rahbar.retrieval.chroma")

_STOPWORDS = {"a", "an", "and", "are", "be", "for", "from", "how", "in", "is", "of", "on", "or", "the", "to", "what", "when", "which", "who", "with"}


def _terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9][a-z0-9.-]{1,}", text.lower()) if word not in _STOPWORDS}


def _get_query_vector(text: str, settings: Settings, dim: int = 768) -> list[float]:
    """Embed query using Ollama or deterministic fallback vector."""
    try:
        endpoint = f"{settings.ollama_url.rstrip('/')}/api/embed"
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(endpoint, json={"model": settings.embed_model, "input": [text]})
            if resp.status_code == 200:
                data = resp.json()
                embeds = data.get("embeddings") if isinstance(data, dict) else None
                if (
                    isinstance(embeds, list)
                    and embeds
                    and isinstance(embeds[0], list)
                    and embeds[0]
                ):
                    return embeds[0]
    except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.info("Query embedding fallback used (%s)", exc)

    # Deterministic fallback vector
    values: list[float] = []
    seed = text.encode("utf-8", errors="replace")
    while len(values) < dim:
        seed = hashlib.sha256(seed).digest()
        values.extend((b - 127.5) / 127.5 for b in seed)
    values = values[:dim]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


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

        query_vector = _get_query_vector(question, self.settings)

        # Failure Mode E Guardrail: Exclude superseded documents unless user explicitly asks
        lower_q = question.lower()
        allow_superseded = "2023" in lower_q or "superseded" in lower_q or "old version" in lower_q or "previous policy" in lower_q
        where_filter = None if allow_superseded else {"superseded": False}

        try:
            results = collection.query(
                query_embeddings=[query_vector],  # type: ignore[arg-type]
                n_results=k,
                where=where_filter,  # type: ignore[arg-type]
                include=["documents", "metadatas", "distances"],
            )
        except (ChromaError, httpx.HTTPError, OSError, ValueError) as exc:
            # Never silently drop the superseded-document filter. A malformed or
            # unsupported query is a safe empty retrieval, not permission to return
            # obsolete policy text.
            log.error("Chroma query failed with required metadata filter: %s", exc)
            return []

        passages: list[Passage] = []
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance; convert to similarity: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            passages.append(
                Passage(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    source=str(meta.get("source", "unknown")),
                    section=str(meta.get("section", "General")),
                    page=int(str(meta["page"])) if meta.get("page") is not None else None,
                    score=round(similarity, 4),
                    document_id=str(meta.get("document_id", "")) or None,
                    doc_version=str(meta.get("doc_version", "")) or None,
                )
            )

        # Offline backstop: when Ollama is unavailable the deterministic hash
        # vector is not semantic. Score the indexed text lexically so exact
        # FAST terms (CGPA, fees, clause numbers) still retrieve grounded data.
        if passages and max((p.score for p in passages), default=0.0) < self.settings.relevance_floor:
            try:
                all_rows = collection.get(
                    where=where_filter,  # type: ignore[arg-type]
                    include=["documents", "metadatas"],
                )
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
                if candidates:
                    candidates.sort(key=lambda item: item.score, reverse=True)
                    passages = candidates[:k]
            except (ChromaError, httpx.HTTPError, OSError, ValueError) as exc:
                log.warning("Lexical retrieval backstop failed: %s", exc)

        return passages
