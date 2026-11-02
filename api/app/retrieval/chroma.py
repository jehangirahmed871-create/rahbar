"""The real retriever against ChromaDB (sections 3.2 and 4.4)."""

from __future__ import annotations

import hashlib
import logging
import math

import httpx

from ..config import Settings
from ..schemas import Passage

log = logging.getLogger("rahbar.retrieval.chroma")


def _get_query_vector(text: str, settings: Settings, dim: int = 768) -> list[float]:
    """Embed query using Ollama or deterministic fallback vector."""
    try:
        endpoint = f"{settings.ollama_url.rstrip('/')}/api/embed"
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(endpoint, json={"model": settings.embed_model, "input": [text]})
            if resp.status_code == 200:
                data = resp.json()
                embeds = data.get("embeddings") or []
                if embeds and len(embeds[0]) > 0:
                    return embeds[0]
    except Exception as exc:
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
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def retrieve(self, question: str, k: int) -> list[Passage]:
        """Query Chroma for top k chunks, filtering out superseded policy documents
        unless explicitly requested by user inquiry.
        """
        try:
            collection = self.client.get_collection(self.settings.collection)
        except Exception as exc:
            log.error("Failed to access collection %s: %s", self.settings.collection, exc)
            return []

        query_vector = _get_query_vector(question, self.settings)

        # Failure Mode E Guardrail: Exclude superseded documents unless user explicitly asks
        lower_q = question.lower()
        allow_superseded = "2023" in lower_q or "superseded" in lower_q or "old version" in lower_q or "previous policy" in lower_q
        where_filter = None if allow_superseded else {"superseded": False}

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.warning("Chroma query failed with where filter (%s); querying without filter", exc)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

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
                    page=int(meta["page"]) if meta.get("page") is not None else None,
                    score=round(similarity, 4),
                )
            )

        return passages
