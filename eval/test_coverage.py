from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import coverage as coverage_module
import pytest
from coverage import analyze_coverage, load_jsonl, run

HERE = Path(__file__).resolve().parent


def test_live_queries_have_required_domain_split() -> None:
    queries = load_jsonl(HERE / "live_queries.jsonl")
    assert len(queries) >= 40
    assert sum(query["expected_coverage"] == "in_domain" for query in queries) >= 28
    assert sum(query["expected_coverage"] == "out_of_domain" for query in queries) >= 12


def test_coverage_uses_best_golden_match() -> None:
    golden = [{"id": "g1"}, {"id": "g2"}]
    live = [
        {"id": "l1", "question": "covered", "topic": "policy", "expected_coverage": "in_domain"},
        {"id": "l2", "question": "also covered", "topic": "policy", "expected_coverage": "in_domain"},
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
        [0.0, 1.0],
    ]

    report = analyze_coverage(golden, live, vectors, threshold=0.75)

    assert report["covered_count"] == 2
    assert report["coverage_rate"] == 1.0
    assert report["per_query"][0]["nearest_golden_id"] == "g1"
    assert report["uncovered_by_topic"] == {}


def test_out_of_domain_matches_do_not_inflate_coverage_rate() -> None:
    golden = [{"id": "g1"}, {"id": "g2"}]
    live = [
        {"id": "in", "question": "in domain", "topic": "policy", "expected_coverage": "in_domain"},
        {"id": "out", "question": "out of domain", "topic": "travel", "expected_coverage": "out_of_domain"},
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
        [0.0, 1.0],
    ]

    report = analyze_coverage(golden, live, vectors, threshold=0.9)

    assert report["covered_count"] == 0
    assert report["in_domain_query_count"] == 1
    assert report["coverage_rate"] == 0.0
    assert report["overall_covered_count"] == 1
    assert report["overall_coverage_rate"] == 0.5
    assert report["out_of_domain_above_threshold_ids"] == ["out"]
    assert report["uncovered_by_topic"] == {"policy": ["in"]}
    assert report["uncovered_in_domain_by_topic"] == {"policy": ["in"]}


def test_coverage_excludes_abstention_golden_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        "\n".join([
            json.dumps({"id": "answerable", "question": "policy answer", "type": "answer"}),
            json.dumps({"id": "abstain", "question": "uncovered query", "type": "abstain"}),
        ]),
        encoding="utf-8",
    )
    live_path = tmp_path / "live.jsonl"
    live_path.write_text(
        json.dumps({
            "id": "live-1",
            "question": "policy answer",
            "topic": "policy",
            "expected_coverage": "in_domain",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        coverage_module,
        "get_settings",
        lambda: SimpleNamespace(embed_model="test-model"),
    )
    monkeypatch.setattr(
        coverage_module,
        "embed",
        lambda texts, settings: [[1.0, 0.0] for _ in texts],
    )

    report = run(golden_path, live_path, threshold=0.9)

    assert report["golden_question_count"] == 1
    assert report["abstention_golden_rows_excluded"] == 1
    assert report["coverage_rate"] == 1.0


def test_coverage_flags_low_similarity_and_invalid_vector_count() -> None:
    golden = [{"id": "g1"}]
    live = [{"id": "l1", "question": "uncovered", "topic": "travel"}]
    report = analyze_coverage(golden, live, [[1.0, 0.0], [0.0, 1.0]], threshold=0.8)
    assert report["coverage_rate"] == 0.0
    assert report["uncovered_by_topic"] == {"travel": ["l1"]}

    with pytest.raises(ValueError, match="expected 2 vectors"):
        analyze_coverage(golden, live, [[1.0, 0.0]], threshold=0.8)
