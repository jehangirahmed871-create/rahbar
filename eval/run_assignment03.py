#!/usr/bin/env python3
"""Run the Assignment 03 embedding, retrieval, and answer evaluation suite."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))

from pipeline.config import get_settings as get_indexer_settings
from pipeline.embed import embed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(row)
    return rows


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _is_relevant(expected: dict[str, Any], retrieved: dict[str, Any]) -> bool:
    expected_id = expected.get("document_id")
    retrieved_id = retrieved.get("document_id")
    if expected_id and retrieved_id:
        identity_matches = expected_id == retrieved_id
    else:
        identity_matches = (
            str(expected.get("source", "")).casefold()
            == str(retrieved.get("source", "")).casefold()
        )
    if not identity_matches:
        return False
    expected_page = expected.get("page")
    return expected_page is None or retrieved.get("page") == expected_page


def relevant_ranks(question: dict[str, Any], retrieved: list[dict[str, Any]]) -> list[int]:
    relevant = question.get("relevant") or []
    ranks: list[int] = []
    for expected in relevant:
        rank = next(
            (index for index, item in enumerate(retrieved, 1) if _is_relevant(expected, item)),
            None,
        )
        if rank is not None:
            ranks.append(rank)
    return ranks


def recall_at_k(question: dict[str, Any], retrieved: list[dict[str, Any]], k: int) -> float | None:
    relevant = question.get("relevant") or []
    if not relevant:
        return None
    found = sum(
        any(_is_relevant(expected, item) for item in retrieved[:k])
        for expected in relevant
    )
    return found / len(relevant)


def mean_reciprocal_rank(question: dict[str, Any], retrieved: list[dict[str, Any]]) -> float | None:
    ranks = relevant_ranks(question, retrieved)
    return 1.0 / min(ranks) if ranks else (None if not question.get("relevant") else 0.0)


def ndcg_at_k(question: dict[str, Any], retrieved: list[dict[str, Any]], k: int = 5) -> float | None:
    relevant = question.get("relevant") or []
    if not relevant:
        return None
    gains = [
        1.0 if any(_is_relevant(expected, item) for expected in relevant) else 0.0
        for item in retrieved[:k]
    ]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def citation_validity(response: dict[str, Any]) -> float | None:
    citations = response.get("citations") or []
    if not citations:
        return None
    retrieved = response.get("retrieved") or []
    valid = sum(
        any(
            citation.get("source") == item.get("source")
            and citation.get("section") == item.get("section")
            and (citation.get("page") is None or citation.get("page") == item.get("page"))
            for item in retrieved
        )
        for citation in citations
    )
    return valid / len(citations)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _macro_average(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else 0.0


def _answer_support_proxy(question: dict[str, Any], response: dict[str, Any]) -> bool | None:
    if question.get("must_abstain"):
        return None
    expected = _normalise(str(question.get("expected_answer", "")))
    answer = _normalise(str(response.get("answer", "")))
    if not expected:
        return None
    return expected in answer


def summarise_rows(
    questions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    answerable = [
        (question, by_id[question["id"]])
        for question in questions
        if not question.get("must_abstain") and question["id"] in by_id
    ]
    abstain_questions = [
        (question, by_id[question["id"]])
        for question in questions
        if question.get("must_abstain") and question["id"] in by_id
    ]
    citations = [
        score
        for _question, row in answerable
        if (score := citation_validity(row["response"])) is not None
    ]
    support = [
        score
        for question, row in answerable
        if (score := _answer_support_proxy(question, row["response"])) is not None
    ]
    return {
        "retrieval": {
            "recall_at_3": _macro_average([
                recall_at_k(q, row["response"].get("retrieved", []), 3)
                for q, row in answerable
            ]),
            "recall_at_5": _macro_average([
                recall_at_k(q, row["response"].get("retrieved", []), 5)
                for q, row in answerable
            ]),
            "mrr": _macro_average([
                mean_reciprocal_rank(q, row["response"].get("retrieved", []))
                for q, row in answerable
            ]),
            "ndcg_at_5": _macro_average([
                ndcg_at_k(q, row["response"].get("retrieved", []), 5)
                for q, row in answerable
            ]),
        },
        "answers": {
            "faithfulness_expected_answer_proxy": statistics.mean(support) if support else 0.0,
            "faithfulness_proxy_questions": len(support),
            "abstention_correctness": (
                sum(not row["response"].get("answered_from_context", True)
                    for _question, row in abstain_questions) / len(abstain_questions)
                if abstain_questions else 1.0
            ),
            "citation_validity": statistics.mean(citations) if citations else 1.0,
            "latency_p50_seconds": _percentile([row["seconds"] for row in rows], 0.50),
            "latency_p95_seconds": _percentile([row["seconds"] for row in rows], 0.95),
        },
        "counts": {
            "questions": len(rows),
            "answerable": len(answerable),
            "must_abstain": len(abstain_questions),
        },
    }


def _typo(question: str) -> str:
    for word in question.split():
        if len(word) >= 6 and word.isalpha():
            return question.replace(word, word[:2] + word[1:], 1)
    return question


def _ask(client: httpx.Client, api: str, question: str, request_id: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = client.post(
        f"{api.rstrip('/')}/api/ask",
        json={"question": question},
        headers={"X-Request-ID": request_id},
    )
    response.raise_for_status()
    return response.json(), time.perf_counter() - started


def run_api_suite(
    client: httpx.Client,
    api: str,
    questions: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    random.seed(seed)
    clean_rows: list[dict[str, Any]] = []
    noisy_rows: list[dict[str, Any]] = []
    for question in questions:
        response, elapsed = _ask(
            client, api, question["question"], f"eval-{seed}-{question['id']}"
        )
        clean_rows.append({
            "id": question["id"],
            "response": response,
            "seconds": elapsed,
        })
        noisy_question = _typo(question["question"])
        if noisy_question != question["question"]:
            noisy_response, noisy_elapsed = _ask(
                client, api, noisy_question, f"noise-{seed}-{question['id']}"
            )
            noisy_rows.append({
                "id": question["id"],
                "response": noisy_response,
                "seconds": noisy_elapsed,
            })
        print(f"  {question['id']:<5} {elapsed:>6.2f}s {question['question'][:65]}")
    return clean_rows, noisy_rows


def measure_embeddings(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    settings = get_indexer_settings()
    texts = [text for pair in pairs for text in (pair["text_a"], pair["text_b"])]
    started = time.perf_counter()
    vectors = embed(texts, settings)
    elapsed = time.perf_counter() - started
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding count mismatch: {len(vectors)} vectors for {len(texts)} texts")
    similarities = {
        pair["id"]: _cosine(vectors[index * 2], vectors[index * 2 + 1])
        for index, pair in enumerate(pairs)
    }
    labels: dict[str, list[float]] = {"paraphrase": [], "unrelated": []}
    for pair in pairs:
        labels[pair["kind"]].append(similarities[pair["id"]])
    if len(labels["paraphrase"]) < 15 or len(labels["unrelated"]) < 15:
        raise ValueError("embedding evaluation needs at least 15 paraphrase and 15 unrelated pairs")

    sample = pairs[0]["text_a"]
    repeated = embed([sample, sample], settings)
    largest_difference = max(
        (abs(left - right) for left, right in zip(repeated[0], repeated[1])),
        default=0.0,
    )
    paraphrase_mean = statistics.mean(labels["paraphrase"])
    unrelated_mean = statistics.mean(labels["unrelated"])
    return {
        "model": settings.embed_model,
        "pair_counts": {kind: len(values) for kind, values in labels.items()},
        "mean_cosine_similarity": {
            kind: statistics.mean(values) for kind, values in labels.items()
        },
        "paraphrase_unrelated_gap": paraphrase_mean - unrelated_mean,
        "embedding_seconds_per_text": elapsed / len(texts),
        "determinism_largest_absolute_difference": largest_difference,
        "pair_scores": similarities,
    }


def _metric_deltas(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in ("retrieval", "answers"):
        for key, value in first[group].items():
            other = second[group].get(key)
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                values[f"{group}.{key}"] = other - value
    return values


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nMetric summary")
    print(f"  Recall@3:             {summary['retrieval']['recall_at_3']:.4f}")
    print(f"  Recall@5:             {summary['retrieval']['recall_at_5']:.4f}")
    print(f"  MRR:                  {summary['retrieval']['mrr']:.4f}")
    print(f"  nDCG@5:               {summary['retrieval']['ndcg_at_5']:.4f}")
    print(f"  Abstention correctness: {summary['answers']['abstention_correctness']:.4f}")
    print(f"  Citation validity:    {summary['answers']['citation_validity']:.4f}")
    print(f"  Latency p50/p95:      {summary['answers']['latency_p50_seconds']:.3f}s/"
          f"{summary['answers']['latency_p95_seconds']:.3f}s")
    print("  Faithfulness metric:  expected-answer/citation proxy, not a human judgment")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="API base URL, e.g. http://localhost:8000")
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    parser.add_argument("--embedding-pairs", type=Path,
                        default=ROOT / "eval" / "embedding_pairs.jsonl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--output-file", type=Path,
                        help="write the report to this exact path instead of a timestamped file")
    parser.add_argument("--baseline-report", type=Path,
                        help="compare this candidate run against a saved baseline report")
    parser.add_argument("--thresholds", type=Path,
                        help="JSON release-gate thresholds; required with --baseline-report")
    parser.add_argument("--gate-out", type=Path,
                        help="optional path for the separate release-gate decision JSON")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--host", help="optional Host header for an Ingress")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to measure stochasticity")
    if bool(args.baseline_report) != bool(args.thresholds):
        parser.error("--baseline-report and --thresholds must be provided together")

    baseline_report = None
    gate_thresholds = None
    if args.baseline_report:
        from release_gate import load_json

        try:
            baseline_report = load_json(args.baseline_report)
            gate_thresholds = load_json(args.thresholds)
        except ValueError as exc:
            parser.error(str(exc))

    questions = load_jsonl(args.golden)
    pairs = load_jsonl(args.embedding_pairs)
    embedding_runs = [measure_embeddings(pairs) for _ in range(args.repeats)]
    headers = {"Host": args.host} if args.host else {}
    runs = []
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        for run_number in range(args.repeats):
            seed = args.seed
            print(f"\nEvaluation run {run_number + 1}/{args.repeats} (seed={seed})")
            clean_rows, noisy_rows = run_api_suite(client, args.api, questions, seed)
            summary = summarise_rows(questions, clean_rows)
            noisy_summary = summarise_rows(questions, noisy_rows)
            summary["noise_test"] = {
                "typo_queries": len(noisy_rows),
                "metric_deltas": {
                    key: noisy_summary[group][key] - value
                    for group in ("retrieval",)
                    for key, value in summary[group].items()
                },
            }
            _print_summary(summary)
            runs.append({
                "run": run_number + 1,
                "seed": seed,
                "summary": summary,
                "rows": clean_rows,
                "noise_rows": noisy_rows,
            })

    deltas = _metric_deltas(runs[0]["summary"], runs[1]["summary"])
    embedding_deltas = {
        "paraphrase_mean_cosine": (
            embedding_runs[1]["mean_cosine_similarity"]["paraphrase"]
            - embedding_runs[0]["mean_cosine_similarity"]["paraphrase"]
        ),
        "unrelated_mean_cosine": (
            embedding_runs[1]["mean_cosine_similarity"]["unrelated"]
            - embedding_runs[0]["mean_cosine_similarity"]["unrelated"]
        ),
        "paraphrase_unrelated_gap": (
            embedding_runs[1]["paraphrase_unrelated_gap"]
            - embedding_runs[0]["paraphrase_unrelated_gap"]
        ),
    }
    synthetic_questions = [item for item in questions if item.get("origin") == "synthetic"]
    human_questions = [item for item in questions if item.get("origin") == "human"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "golden_set": str(args.golden),
        "golden_set_review": {
            "total": len(questions),
            "human_origin": len(human_questions),
            "synthetic_origin": len(synthetic_questions),
            "synthetic_reviewed_by_human": sum(
                bool(item.get("reviewed_by_human")) for item in synthetic_questions
            ),
            "synthetic_review_gap_to_required_8": max(
                0,
                8 - sum(bool(item.get("reviewed_by_human")) for item in synthetic_questions),
            ),
        },
        "embedding_runs": embedding_runs,
        "stochasticity_embedding_metric_deltas_run2_minus_run1": embedding_deltas,
        "runs": runs,
        "stochasticity_metric_deltas_run2_minus_run1": deltas,
    }
    gate_result = None
    if baseline_report is not None and gate_thresholds is not None:
        from release_gate import evaluate_gate

        gate_result = evaluate_gate(baseline_report, report, gate_thresholds)
        report["release_gate"] = gate_result

    output = args.output_file
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = args.results_dir / f"assignment03-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults written to {output}")
    if gate_result is not None:
        if args.gate_out:
            args.gate_out.parent.mkdir(parents=True, exist_ok=True)
            args.gate_out.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")
        print(f"Release gate: {gate_result['decision'].upper()}")
        return 0 if gate_result["decision"] == "pass" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
