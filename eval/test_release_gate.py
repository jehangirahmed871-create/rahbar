from __future__ import annotations

import pytest
from release_gate import evaluate_gate


def _report(
    recall5: float = 0.9,
    embedding_gap: float = 0.4,
    faithfulness: float = 0.9,
    citations: float = 1.0,
    abstention: float = 1.0,
    latency: float = 1.0,
) -> dict:
    return {
        "embedding_runs": [{"paraphrase_unrelated_gap": embedding_gap}],
        "runs": [
            {
                "summary": {
                    "retrieval": {"recall_at_5": recall5},
                    "answers": {
                        "faithfulness_expected_answer_proxy": faithfulness,
                        "citation_validity": citations,
                        "abstention_correctness": abstention,
                        "latency_p95_seconds": latency,
                    },
                }
            }
        ],
    }


@pytest.fixture
def thresholds() -> dict:
    return {
        "max_recall5_drop": 0.02,
        "max_embedding_gap_drop": 0.02,
        "min_faithfulness": 0.8,
        "min_citation_validity": 0.95,
        "max_latency_p95_seconds": 30.0,
    }


def test_release_gate_passes_candidate_within_all_thresholds(thresholds: dict) -> None:
    result = evaluate_gate(_report(), _report(recall5=0.89), thresholds)
    assert result["decision"] == "pass"
    assert all(check["passed"] for check in result["checks"])


def test_release_gate_rejects_bad_candidate(thresholds: dict) -> None:
    result = evaluate_gate(
        _report(),
        _report(recall5=0.5, faithfulness=0.3, citations=0.8, latency=31.0),
        thresholds,
    )
    assert result["decision"] == "fail"
    assert {check["name"] for check in result["checks"] if not check["passed"]} == {
        "recall_at_5",
        "faithfulness",
        "citation_validity",
        "latency_p95_seconds",
    }


def test_changing_threshold_changes_release_decision_without_code_change(
    thresholds: dict,
) -> None:
    candidate = _report(recall5=0.89)
    assert evaluate_gate(_report(), candidate, thresholds)["decision"] == "pass"
    stricter = {**thresholds, "max_recall5_drop": 0.005}
    result = evaluate_gate(_report(), candidate, stricter)
    assert result["decision"] == "fail"
    assert not result["checks"][0]["passed"]


def test_release_gate_rejects_missing_metric_data(thresholds: dict) -> None:
    with pytest.raises(ValueError, match="missing metric values"):
        evaluate_gate({"runs": [], "embedding_runs": []}, _report(), thresholds)
