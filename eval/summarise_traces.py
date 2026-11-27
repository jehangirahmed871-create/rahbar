#!/usr/bin/env python3
"""Summarize query traces over a rolling time window."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"trace timestamp must include timezone: {value!r}")
    return parsed.astimezone(UTC)


def load_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: trace must be a JSON object")
        traces.append(row)
    return traces


def summarize_traces(
    traces: list[dict[str, Any]],
    since_minutes: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    if since_minutes <= 0:
        raise ValueError("since_minutes must be greater than zero")
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = current_time - timedelta(minutes=since_minutes)
    selected = [
        trace for trace in traces
        if _parse_timestamp(str(trace.get("timestamp", ""))) >= cutoff
    ]
    if not selected:
        return {
            "window_minutes": since_minutes,
            "trace_count": 0,
            "top1_similarity": {"count": 0, "mean": None, "p50": None, "p95": None},
            "abstention_rate": None,
            "cache_hit_rate": None,
            "stage_latency_p95_seconds": {},
            "injection_attempts_flagged": 0,
        }

    top1_scores = [
        float(retrieved[0]["score"])
        for trace in selected
        if (retrieved := trace.get("retrieved") or [])
        and retrieved[0].get("score") is not None
    ]
    answered = [
        trace["response"]["answered_from_context"]
        for trace in selected
        if isinstance(trace.get("response"), dict)
        and "answered_from_context" in trace["response"]
    ]
    cache_statuses = [str(trace.get("cache", "")).casefold() for trace in selected]
    cache_statuses = [status for status in cache_statuses if status in {"hit", "miss"}]
    stage_values: dict[str, list[float]] = {}
    for trace in selected:
        for stage, duration in (trace.get("latency_seconds") or {}).items():
            stage_values.setdefault(str(stage), []).append(float(duration))
    flagged_attempts = sum(
        len(trace.get("guardrail_actions") or [])
        for trace in selected
    )

    return {
        "window_minutes": since_minutes,
        "window_start": cutoff.isoformat(),
        "window_end": current_time.isoformat(),
        "trace_count": len(selected),
        "top1_similarity": {
            "count": len(top1_scores),
            "mean": statistics.mean(top1_scores) if top1_scores else None,
            "p50": _percentile(top1_scores, 0.50),
            "p95": _percentile(top1_scores, 0.95),
        },
        "abstention_rate": (
            sum(not value for value in answered) / len(answered) if answered else None
        ),
        "cache_hit_rate": (
            sum(status == "hit" for status in cache_statuses) / len(cache_statuses)
            if cache_statuses else None
        ),
        "stage_latency_p95_seconds": {
            stage: _percentile(values, 0.95)
            for stage, values in sorted(stage_values.items())
        },
        "injection_attempts_flagged": flagged_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("traces") / "queries.jsonl")
    parser.add_argument("--since-minutes", type=float, default=1440)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = summarize_traces(load_traces(args.traces), args.since_minutes)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"trace summary failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"\nTrace summary written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
