#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.

Given: talking to the API, reading the golden set, writing results.json, the exit
code. Implemented evaluation functions: recall_at_k, citation_validity, summarise, gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    response.raise_for_status()
    return response.json()


def _normalize_str(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _section_matches(rel_sec: str, ret_sec: str) -> bool:
    r1, r2 = _normalize_str(rel_sec), _normalize_str(ret_sec)
    return (r1 in r2 or r2 in r1) if (r1 and r2) else False


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions).
    """
    if not relevant:
        return None
    top_k = retrieved[:k]
    hits = 0
    for rel in relevant:
        rel_src = rel.get("source", "").lower().strip()
        rel_sec = rel.get("section", "").strip()
        found = False
        for ret in top_k:
            ret_src = ret.get("source", "").lower().strip()
            ret_sec = ret.get("section", "").strip()
            if rel_src == ret_src and _section_matches(rel_sec, ret_sec):
                found = True
                break
        if found:
            hits += 1
    return hits / len(relevant)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations.
    """
    citations = response.get("citations") or []
    if not citations:
        return None
    retrieved = response.get("retrieved") or []
    valid = 0
    for c in citations:
        c_src = c.get("source", "").lower().strip()
        c_sec = c.get("section", "").strip()
        matched = False
        for r in retrieved:
            r_src = r.get("source", "").lower().strip()
            r_sec = r.get("section", "").strip()
            if c_src == r_src and _section_matches(c_sec, r_sec):
                matched = True
                break
        if matched:
            valid += 1
    return valid / len(citations)


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json.
    """
    r3_scores: list[float] = []
    r5_scores: list[float] = []
    r5_by_type: dict[str, list[float]] = {}
    citation_scores: list[float] = []
    abstain_correct = 0
    total_must_abstain = 0

    for r in rows:
        q_type = r.get("type", "ordinary")
        must_abstain = r.get("must_abstain", False)
        relevant = r.get("relevant") or []
        resp = r.get("response") or {}
        retrieved = resp.get("retrieved") or []

        if must_abstain:
            total_must_abstain += 1
            if not resp.get("answered_from_context", True):
                abstain_correct += 1
        else:
            r3 = recall_at_k(relevant, retrieved, 3)
            r5 = recall_at_k(relevant, retrieved, 5)
            if r3 is not None:
                r3_scores.append(r3)
            if r5 is not None:
                r5_scores.append(r5)
                r5_by_type.setdefault(q_type, []).append(r5)

        cv = citation_validity(resp)
        if cv is not None:
            citation_scores.append(cv)

    avg_r3 = round(sum(r3_scores) / len(r3_scores), 4) if r3_scores else 0.0
    avg_r5 = round(sum(r5_scores) / len(r5_scores), 4) if r5_scores else 0.0
    avg_cv = round(sum(citation_scores) / len(citation_scores), 4) if citation_scores else 1.0
    abs_pct = round(abstain_correct / total_must_abstain, 4) if total_must_abstain > 0 else 1.0

    breakdown = {t: round(sum(sc) / len(sc), 4) for t, sc in sorted(r5_by_type.items())}

    return {
        "recall_at_3": avg_r3,
        "recall_at_5": avg_r5,
        "recall_at_5_by_type": breakdown,
        "citation_validity": avg_cv,
        "abstention_correctness": abs_pct,
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures: list[str] = []
    if baseline:
        base_r5 = baseline.get("recall_at_5", 0.0)
        curr_r5 = results.get("recall_at_5", 0.0)
        if curr_r5 < base_r5 - MAX_RECALL5_DROP:
            failures.append(
                f"recall@5 dropped by {round(base_r5 - curr_r5, 4):.2%}, exceeding allowable {MAX_RECALL5_DROP:.0%}"
            )

    curr_abs = results.get("abstention_correctness", 0.0)
    if curr_abs < REQUIRED_ABSTENTION:
        failures.append(
            f"abstention correctness is {curr_abs:.1%}; expected {REQUIRED_ABSTENTION:.0%}"
        )

    curr_cv = results.get("citation_validity", 0.0)
    if curr_cv < MIN_CITATION_VALIDITY:
        failures.append(
            f"citation validity is {curr_cv:.1%}; expected >= {MIN_CITATION_VALIDITY:.0%}"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            response = ask(client, args.api, q["question"])
            rows.append({
                "id": q["id"], "type": q.get("type", "ordinary"),
                "must_abstain": bool(q.get("must_abstain", False)),
                "relevant": q.get("relevant") or [],
                "expect": q.get("expect"),
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    try:
        results = summarise(rows)
        baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
        failures = gate(results, baseline)
    except NotImplementedError as exc:
        print(f"\nnot implemented yet: {exc}")
        return 2

    results["golden"] = str(args.golden)
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
