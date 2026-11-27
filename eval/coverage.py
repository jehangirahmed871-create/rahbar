#!/usr/bin/env python3
"""Measure live-query coverage against the golden question set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))

from pipeline.config import get_settings  # noqa: E402, RUF100
from pipeline.embed import embed  # noqa: E402, RUF100


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(row)
    return rows


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read coverage config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"coverage config {path} must contain a JSON object")
    return value


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def analyze_coverage(
    golden: list[dict[str, Any]],
    live_queries: list[dict[str, Any]],
    vectors: list[list[float]],
    threshold: float,
) -> dict[str, Any]:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be between -1 and 1")
    if not golden:
        raise ValueError("golden question set is empty")
    if not live_queries:
        raise ValueError("live query set is empty")
    expected_vectors = len(golden) + len(live_queries)
    if len(vectors) != expected_vectors:
        raise ValueError(f"expected {expected_vectors} vectors, received {len(vectors)}")

    golden_ids = [str(row.get("id", "")) for row in golden]
    live_ids = [str(row.get("id", "")) for row in live_queries]
    if any(not value for value in golden_ids + live_ids):
        raise ValueError("all golden questions and live queries need an id")
    if len(set(golden_ids)) != len(golden_ids) or len(set(live_ids)) != len(live_ids):
        raise ValueError("question and live-query IDs must be unique within their sets")

    golden_vectors = vectors[:len(golden)]
    live_vectors = vectors[len(golden):]
    per_query: list[dict[str, Any]] = []
    for query, vector in zip(live_queries, live_vectors):
        scores = [cosine_similarity(vector, golden_vector) for golden_vector in golden_vectors]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best_score = scores[best_index]
        per_query.append({
            "id": query["id"],
            "question": query["question"],
            "topic": query.get("topic", "unspecified"),
            "expected_coverage": query.get("expected_coverage", "unspecified"),
            "nearest_golden_id": golden_ids[best_index],
            "similarity": best_score,
            "covered": best_score >= threshold,
        })

    in_domain_queries = [
        query for query in per_query
        if query["expected_coverage"] != "out_of_domain"
    ]
    if not in_domain_queries:
        raise ValueError("at least one in-domain live query is required")

    uncovered_by_topic: dict[str, list[str]] = {}
    uncovered_in_domain_by_topic: dict[str, list[str]] = {}
    for query in per_query:
        if not query["covered"]:
            uncovered_by_topic.setdefault(query["topic"], []).append(query["id"])
            if query["expected_coverage"] != "out_of_domain":
                uncovered_in_domain_by_topic.setdefault(query["topic"], []).append(query["id"])
    covered_count = sum(bool(query["covered"]) for query in in_domain_queries)
    overall_covered_count = sum(bool(query["covered"]) for query in per_query)
    out_of_domain_matches = [
        query["id"] for query in per_query
        if query["expected_coverage"] == "out_of_domain" and query["covered"]
    ]
    return {
        "threshold": threshold,
        "golden_question_count": len(golden),
        "live_query_count": len(live_queries),
        "covered_count": covered_count,
        "in_domain_query_count": len(in_domain_queries),
        "coverage_rate": covered_count / len(in_domain_queries),
        "overall_covered_count": overall_covered_count,
        "overall_coverage_rate": overall_covered_count / len(live_queries),
        "out_of_domain_above_threshold_count": len(out_of_domain_matches),
        "out_of_domain_above_threshold_ids": out_of_domain_matches,
        "similarity_mean": statistics.mean(query["similarity"] for query in per_query),
        "per_query": per_query,
        "uncovered_by_topic": uncovered_by_topic,
        "uncovered_in_domain_by_topic": uncovered_in_domain_by_topic,
    }


def run(
    golden_path: Path,
    live_path: Path,
    threshold: float,
) -> dict[str, Any]:
    all_golden = load_jsonl(golden_path)
    golden = [
        row for row in all_golden
        if row.get("type") != "abstain" and not row.get("must_abstain", False)
    ]
    live_queries = load_jsonl(live_path)
    texts = [
        str(row.get("question", "")).strip()
        for row in [*golden, *live_queries]
    ]
    if any(not text for text in texts):
        raise ValueError("all golden questions and live queries need non-empty question text")

    settings = get_settings()
    report = analyze_coverage(golden, live_queries, embed(texts, settings), threshold)
    report.update({
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model_configured": settings.embed_model,
        "golden_file": str(golden_path),
        "abstention_golden_rows_excluded": len(all_golden) - len(golden),
        "live_queries_file": str(live_path),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "eval" / "coverage-thresholds.json")
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    parser.add_argument("--live", type=Path, default=ROOT / "eval" / "live_queries.jsonl")
    parser.add_argument("--threshold", type=float,
                        help="cosine threshold calibrated from the measured query set")
    parser.add_argument("--minimum-coverage", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        threshold = (
            args.threshold if args.threshold is not None
            else float(config["similarity_threshold"])
        )
        minimum_coverage = (
            args.minimum_coverage if args.minimum_coverage is not None
            else float(config["minimum_coverage"])
        )
        if not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError("minimum coverage must be between 0 and 1")
        report = run(args.golden, args.live, threshold)
        report["minimum_coverage"] = minimum_coverage
        report["threshold_justification"] = config["threshold_justification"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"coverage analysis failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nCoverage results written to {args.out}")
    if report["coverage_rate"] < minimum_coverage:
        print(
            f"\nCOVERAGE GATE FAILED: {report['coverage_rate']:.1%} is below "
            f"{minimum_coverage:.1%}"
        )
        return 1
    print(f"\nCoverage gate passed: {report['coverage_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
