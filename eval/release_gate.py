#!/usr/bin/env python3
"""Compare baseline and candidate evaluation reports against release thresholds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _mean(values: list[float], name: str) -> float:
    if not values:
        raise ValueError(f"evaluation report is missing metric values for {name}")
    return statistics.mean(values)


def extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    runs = report.get("runs")
    embedding_runs = report.get("embedding_runs")
    if not isinstance(runs, list) or not isinstance(embedding_runs, list):
        raise TypeError("evaluation report must contain runs and embedding_runs arrays")

    summaries = [run.get("summary", {}) for run in runs if isinstance(run, dict)]
    retrieval = [summary.get("retrieval", {}) for summary in summaries]
    answers = [summary.get("answers", {}) for summary in summaries]
    embedding_gaps = [
        float(run["paraphrase_unrelated_gap"])
        for run in embedding_runs
        if isinstance(run, dict) and "paraphrase_unrelated_gap" in run
    ]
    return {
        "recall_at_5": _mean(
            [float(group["recall_at_5"]) for group in retrieval if "recall_at_5" in group],
            "retrieval.recall_at_5",
        ),
        "embedding_gap": _mean(embedding_gaps, "embedding.paraphrase_unrelated_gap"),
        "faithfulness": _mean(
            [float(group["faithfulness_expected_answer_proxy"])
             for group in answers if "faithfulness_expected_answer_proxy" in group],
            "answers.faithfulness_expected_answer_proxy",
        ),
        "citation_validity": _mean(
            [float(group["citation_validity"]) for group in answers if "citation_validity" in group],
            "answers.citation_validity",
        ),
        "abstention_correctness": _mean(
            [float(group["abstention_correctness"])
             for group in answers if "abstention_correctness" in group],
            "answers.abstention_correctness",
        ),
        "latency_p95_seconds": _mean(
            [float(group["latency_p95_seconds"])
             for group in answers if "latency_p95_seconds" in group],
            "answers.latency_p95_seconds",
        ),
    }


def evaluate_gate(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    baseline = extract_metrics(baseline_report)
    candidate = extract_metrics(candidate_report)
    required = {
        "max_recall5_drop",
        "max_embedding_gap_drop",
        "min_faithfulness",
        "min_citation_validity",
        "max_latency_p95_seconds",
    }
    missing = required - thresholds.keys()
    if missing:
        raise ValueError(f"threshold configuration is missing: {', '.join(sorted(missing))}")

    comparisons = [
        (
            "recall_at_5",
            candidate["recall_at_5"] >= baseline["recall_at_5"] - float(thresholds["max_recall5_drop"]),
            (
                f"candidate recall@5 {candidate['recall_at_5']:.4f}; "
                f"baseline {baseline['recall_at_5']:.4f}; "
                f"maximum drop {float(thresholds['max_recall5_drop']):.4f}"
            ),
        ),
        (
            "embedding_gap",
            candidate["embedding_gap"] >= baseline["embedding_gap"]
            - float(thresholds["max_embedding_gap_drop"]),
            (
                f"candidate embedding gap {candidate['embedding_gap']:.4f}; "
                f"baseline {baseline['embedding_gap']:.4f}; maximum drop "
                f"{float(thresholds['max_embedding_gap_drop']):.4f}"
            ),
        ),
        (
            "faithfulness",
            candidate["faithfulness"] >= float(thresholds["min_faithfulness"]),
            (
                f"faithfulness proxy {candidate['faithfulness']:.4f}; "
                f"minimum {float(thresholds['min_faithfulness']):.4f}"
            ),
        ),
        (
            "citation_validity",
            candidate["citation_validity"] >= float(thresholds["min_citation_validity"]),
            (
                f"citation validity {candidate['citation_validity']:.4f}; "
                f"minimum {float(thresholds['min_citation_validity']):.4f}"
            ),
        ),
        (
            "abstention_correctness",
            candidate["abstention_correctness"] >= baseline["abstention_correctness"],
            (
                f"abstention correctness {candidate['abstention_correctness']:.4f}; "
                f"baseline {baseline['abstention_correctness']:.4f}"
            ),
        ),
        (
            "latency_p95_seconds",
            candidate["latency_p95_seconds"] <= float(thresholds["max_latency_p95_seconds"]),
            (
                f"latency p95 {candidate['latency_p95_seconds']:.4f}s; "
                f"maximum {float(thresholds['max_latency_p95_seconds']):.4f}s"
            ),
        ),
    ]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "decision": "pass" if all(passed for _name, passed, _detail in comparisons) else "fail",
        "baseline": baseline,
        "candidate": candidate,
        "thresholds": thresholds,
        "checks": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in comparisons
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_gate(
            load_json(args.baseline),
            load_json(args.candidate),
            load_json(args.thresholds),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"release gate could not run: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
