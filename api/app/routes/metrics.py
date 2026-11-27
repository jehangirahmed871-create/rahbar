"""Prometheus metrics at /metrics (section 4.8).

Request count and latency are wired up already. Add your own the same way:
retrieval latency, generation latency, cache hits, abstentions, fallbacks, tokens.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

router = APIRouter()

REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
RETRIEVAL_LATENCY = Histogram("rahbar_retrieval_seconds", "Chroma retrieval latency")
GENERATION_LATENCY = Histogram("rahbar_generation_seconds", "LLM generation latency")
CACHE_LOOKUPS = Counter("rahbar_cache_lookups_total", "Semantic cache lookups")
CACHE_HITS = Counter("rahbar_cache_hits_total", "Semantic cache hits")
ABSTENTIONS = Counter("rahbar_abstentions_total", "Answers abstained from")
FALLBACKS = Counter("rahbar_fallbacks_total", "Provider fallback answers")
TOKENS_IN = Counter("rahbar_tokens_in_total", "Approximate input tokens")
TOKENS_OUT = Counter("rahbar_tokens_out_total", "Approximate output tokens")
INJECTION_ATTEMPTS = Counter(
    "rahbar_injection_attempts_total", "Prompt-injection attempts detected in query or retrieved context"
)
TOP1_SIMILARITY = Histogram(
    "rahbar_top1_similarity",
    "Top-1 retrieval similarity distribution",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
