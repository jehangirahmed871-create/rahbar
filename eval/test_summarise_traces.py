from __future__ import annotations

from datetime import UTC, datetime

from summarise_traces import summarize_traces


def test_trace_summary_measures_window_metrics_and_guardrails() -> None:
    now = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
    traces = [
        {
            "timestamp": "2026-11-25T11:50:00+00:00",
            "retrieved": [{"score": 0.8}],
            "response": {"answered_from_context": True},
            "cache": "miss",
            "latency_seconds": {"retrieval": 0.2, "generation": 0.8},
            "guardrail_actions": [],
        },
        {
            "timestamp": "2026-11-25T11:55:00+00:00",
            "retrieved": [{"score": 0.6}],
            "response": {"answered_from_context": False},
            "cache": "hit",
            "latency_seconds": {"retrieval": 0.4, "generation": 0.0},
            "guardrail_actions": [{"source": "query", "action": "flagged"}],
        },
        {
            "timestamp": "2026-11-24T11:00:00+00:00",
            "retrieved": [{"score": 0.1}],
            "response": {"answered_from_context": False},
            "cache": "miss",
            "latency_seconds": {"retrieval": 9.0},
            "guardrail_actions": [{"source": "query", "action": "flagged"}],
        },
    ]

    summary = summarize_traces(traces, since_minutes=60, now=now)

    assert summary["trace_count"] == 2
    assert summary["top1_similarity"]["mean"] == 0.7
    assert summary["abstention_rate"] == 0.5
    assert summary["cache_hit_rate"] == 0.5
    assert summary["injection_attempts_flagged"] == 1
    assert summary["stage_latency_p95_seconds"]["retrieval"] > 0.3


def test_trace_summary_handles_no_matching_records() -> None:
    summary = summarize_traces([], since_minutes=15)
    assert summary["trace_count"] == 0
    assert summary["cache_hit_rate"] is None
