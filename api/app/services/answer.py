"""Turns a question into an AskResponse. This is where the query pipeline lives."""

from __future__ import annotations

import logging

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..schemas import Answer, AskResponse, RetrievedRef
from .cache import SemanticCache

log = logging.getLogger("rahbar.answer")

_cache_instance: SemanticCache | None = None


def get_cache(settings: Settings) -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(settings)
    return _cache_instance


def answer_question(
    question: str, retriever: Retriever, provider: LLMProvider, settings: Settings
) -> AskResponse:
    """Execute complete 8-step production query pipeline (sections 3.2-3.4):
    1. Validation: (3 to 500 characters, handled by FastAPI AskRequest)
    2. Cache check: query semantic cache (skip if follow-up)
    3. Retrieval: top-k chunks with superseded filter
    4. Relevance floor check: honest abstention if below floor
    5. Prompt augmentation with evidence
    6. Constrained generation at low temperature
    7. Validation with retry and safe fallback
    8. Cache storage for verified grounded answers
    """
    rid = request_id_var.get()
    cache = get_cache(settings)

    # Deterministic query vector for cache evaluation
    from ..retrieval.chroma import _get_query_vector

    query_vector = _get_query_vector(question, settings)

    # Step 2: Semantic cache check
    cached = cache.get(question, query_vector)
    if cached:
        log.info("Serving answer from semantic cache", extra={"request_id": rid, "cached": True})
        return cached

    # Step 3: Retrieval
    passages = retriever.retrieve(question, k=settings.top_k)

    # Step 4: Relevance Floor & Honest Abstention
    max_score = max((p.score for p in passages), default=0.0)
    is_abstention = False

    if not passages or max_score < settings.relevance_floor:
        log.info(
            "Query relevance below floor (max_score=%.4f < floor=%.2f); abstaining",
            max_score,
            settings.relevance_floor,
            extra={"request_id": rid, "abstention": True},
        )
        is_abstention = True
        answer = Answer(
            reasoning=f"Max retrieval similarity {max_score:.4f} is below the relevance floor {settings.relevance_floor:.2f}.",
            answer="The provided corpus documents do not contain information regarding this topic.",
            citations=[],
            answered_from_context=False,
            confidence="low",
        )
    else:
        # Step 5 & 6 & 7: Augmented Generation with Validation & Retry
        answer = provider.generate(question, passages)

    # Step 7b: Citation Integrity Check
    retrieved_refs = {(p.source.lower(), p.section) for p in passages}
    valid_citations = [
        c for c in answer.citations if (c.source.lower(), c.section) in retrieved_refs
    ]

    log.info(
        "answered",
        extra={
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 4) for p in passages],
            "max_score": round(max_score, 4),
            "answered_from_context": answer.answered_from_context,
            "abstention": is_abstention,
        },
    )

    response = AskResponse(
        request_id=rid,
        answer=answer.answer,
        citations=valid_citations,
        answered_from_context=answer.answered_from_context,
        confidence=answer.confidence,
        retrieved=[
            RetrievedRef(
                chunk_id=p.chunk_id, source=p.source, section=p.section, page=p.page, score=p.score
            )
            for p in passages
        ],
        provider=provider.name,
    )

    # Step 8: Safe Cache Storage (never cache abstentions or ungrounded turns)
    if not is_abstention and answer.answered_from_context:
        cache.store(question, query_vector, response)

    return response
