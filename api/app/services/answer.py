"""Turns a question into an AskResponse. This is where the query pipeline lives."""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..routes.metrics import (
    ABSTENTIONS,
    CACHE_HITS,
    CACHE_LOOKUPS,
    FALLBACKS,
    GENERATION_LATENCY,
    INJECTION_ATTEMPTS,
    RETRIEVAL_LATENCY,
    TOKENS_IN,
    TOKENS_OUT,
    TOP1_SIMILARITY,
)
from ..schemas import Answer, AskResponse, RetrievedRef
from .cache import SemanticCache
from .tracing import write_trace

log = logging.getLogger("rahbar.answer")

_cache_instance: SemanticCache | None = None
_INJECTION_MARKERS = (
    re.compile(r"\bignore\s+(?:all\s+)?previous\s+(?:instructions|rules)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous\s+)?instructions\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(?:the\s+)?system\s+prompt\b", re.IGNORECASE),
)


def _is_injection(text: str) -> bool:
    return any(marker.search(text) for marker in _INJECTION_MARKERS)


def _trace_passage(passage: Any) -> dict[str, Any]:
    return {
        "chunk_id": passage.chunk_id,
        "document_id": passage.document_id,
        "document_version": passage.doc_version,
        "source": passage.source,
        "section": passage.section,
        "page": passage.page,
        "score": passage.score,
        "text": passage.text,
    }


def _trace_prompt(provider: LLMProvider, question: str, passages: list[Any]) -> str | None:
    builder = getattr(provider, "build_prompt", None)
    if not callable(builder):
        return None
    return str(builder(question, passages))


def _write_request_trace(
    *,
    trace_id: str,
    timestamp: str,
    question: str,
    settings: Settings,
    retriever: Retriever,
    provider: LLMProvider,
    raw_passages: list[Any],
    passages: list[Any],
    prompt: str | None,
    answer: Answer | None,
    response: AskResponse | None,
    guardrail_actions: list[dict[str, str]],
    latency: dict[str, float],
    cache_status: str,
) -> None:
    import hashlib

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt is not None else None
    old_policy_query = any(term in question.lower() for term in ("2023", "superseded", "old version", "previous policy"))
    trace = {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "user_query": question,
        "rewritten_query": None,
        "embedding_model": {"name": settings.embed_model, "version": settings.embed_model},
        "collection": {"name": settings.collection, "version": None},
        "filters_applied": {} if old_policy_query else {"superseded": False},
        "retrieved": [_trace_passage(passage) for passage in raw_passages],
        "generation_context": [_trace_passage(passage) for passage in passages],
        "reranked_order": [passage.chunk_id for passage in passages],
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "model": {
            "provider": provider.name,
            "name": settings.llm_model,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
        },
        "response": response.model_dump() if response is not None else None,
        "answer": answer.model_dump() if answer is not None else None,
        "citations": [citation.model_dump() for citation in answer.citations] if answer else [],
        "guardrail_actions": guardrail_actions,
        "latency_seconds": latency,
        "cache": cache_status,
        "retriever": retriever.name,
    }
    write_trace(trace, settings.trace_dir)


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
    trace_id = uuid.uuid4().hex
    timestamp = datetime.now(UTC).isoformat()
    request_started = time.perf_counter()
    latency: dict[str, float] = {}
    guardrail_actions: list[dict[str, str]] = []
    query_injection = _is_injection(question)
    if query_injection:
        INJECTION_ATTEMPTS.inc()
        guardrail_actions.append({
            "source": "query",
            "action": "flagged as untrusted; excluded from semantic cache lookup",
        })
    cache = get_cache(settings)

    # Deterministic query vector for cache evaluation
    from ..retrieval.chroma import _get_query_vector

    embedding_started = time.perf_counter()
    query_vector = _get_query_vector(question, settings)
    latency["query_embedding"] = time.perf_counter() - embedding_started

    # Step 2: Semantic cache check
    CACHE_LOOKUPS.inc()
    is_followup = cache.is_followup(question)
    cache_started = time.perf_counter()
    cached = None if query_injection else cache.get(question, query_vector, is_followup=is_followup)
    latency["cache_lookup"] = time.perf_counter() - cache_started
    if cached:
        CACHE_HITS.inc()
        log.info("Serving answer from semantic cache", extra={"request_id": rid, "cached": True})
        cached = cached.model_copy(update={"request_id": rid, "trace_id": trace_id})
        latency["total"] = time.perf_counter() - request_started
        _write_request_trace(
            trace_id=trace_id,
            timestamp=timestamp,
            question=question,
            settings=settings,
            retriever=retriever,
            provider=provider,
            raw_passages=[],
            passages=[],
            prompt=None,
            answer=None,
            response=cached,
            guardrail_actions=guardrail_actions,
            latency=latency,
            cache_status="hit",
        )
        return cached

    # Step 3: Retrieval
    retrieval_started = time.perf_counter()
    raw_passages = retriever.retrieve(question, k=settings.top_k)
    RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_started)
    latency["retrieval"] = time.perf_counter() - retrieval_started
    TOKENS_IN.inc(max(1, len(question.split())))
    if raw_passages:
        TOP1_SIMILARITY.observe(raw_passages[0].score)

    passages = []
    flagged_chunk_ids: list[str] = []
    for passage in raw_passages:
        if _is_injection(passage.text):
            flagged_chunk_ids.append(passage.chunk_id)
        else:
            passages.append(passage)
    if flagged_chunk_ids:
        INJECTION_ATTEMPTS.inc()
        guardrail_actions.append({
            "source": "retrieved_context",
            "action": "removed instruction-like passages before prompt construction",
            "chunk_ids": ",".join(flagged_chunk_ids),
        })

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
        ABSTENTIONS.inc()
        answer = Answer(
            reasoning=f"Max retrieval similarity {max_score:.4f} is below the relevance floor {settings.relevance_floor:.2f}.",
            answer="The provided corpus documents do not contain information regarding this topic.",
            citations=[],
            answered_from_context=False,
            confidence="low",
        )
    else:
        # Step 5 & 6 & 7: Augmented Generation with Validation & Retry
        prompt = _trace_prompt(provider, question, passages)
        generation_started = time.perf_counter()
        answer = provider.generate(question, passages)
        GENERATION_LATENCY.observe(time.perf_counter() - generation_started)
        latency["generation"] = time.perf_counter() - generation_started
        TOKENS_OUT.inc(max(1, len(answer.answer.split())))
        if not answer.answered_from_context:
            FALLBACKS.inc()
            ABSTENTIONS.inc()

    if is_abstention:
        prompt = None

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
                chunk_id=p.chunk_id,
                source=p.source,
                section=p.section,
                page=p.page,
                score=p.score,
                document_id=p.document_id,
                doc_version=p.doc_version,
            )
            for p in passages
        ],
        provider=provider.name,
        trace_id=trace_id,
    )

    # Step 8: Safe Cache Storage (never cache abstentions or ungrounded turns)
    if not is_abstention and answer.answered_from_context:
        cache.store(question, query_vector, response, is_followup=is_followup)

    latency["total"] = time.perf_counter() - request_started
    _write_request_trace(
        trace_id=trace_id,
        timestamp=timestamp,
        question=question,
        settings=settings,
        retriever=retriever,
        provider=provider,
        raw_passages=raw_passages,
        passages=passages,
        prompt=prompt,
        answer=answer,
        response=response,
        guardrail_actions=guardrail_actions,
        latency=latency,
        cache_status="miss",
    )
    return response
