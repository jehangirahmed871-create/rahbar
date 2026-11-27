from __future__ import annotations

from pathlib import Path

from run_assignment03 import (
    load_jsonl,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    summarise_rows,
)
from show_trace import format_trace, load_trace

HERE = Path(__file__).resolve().parent


def test_golden_set_meets_count_and_ground_truth_shapes() -> None:
    questions = load_jsonl(HERE / "golden.jsonl")
    answerable = [question for question in questions if not question["must_abstain"]]
    must_abstain = [question for question in questions if question["must_abstain"]]
    assert len(questions) >= 25
    assert len(answerable) >= 15
    assert len(must_abstain) >= 5
    assert sum(question["type"] == "hard" for question in questions) >= 5
    assert all(question.get("origin") in {"human", "synthetic"} for question in questions)
    assert sum(question["origin"] == "human" for question in questions) >= 8
    assert all(question.get("relevant") for question in answerable)
    assert all(
        item.get("source")
        and (item.get("page") is None or item.get("page") >= 1)
        for question in answerable for item in question["relevant"]
    )


def test_embedding_pair_set_has_required_minimum() -> None:
    pairs = load_jsonl(HERE / "embedding_pairs.jsonl")
    assert sum(pair["kind"] == "paraphrase" for pair in pairs) >= 15
    assert sum(pair["kind"] == "unrelated" for pair in pairs) >= 15


def test_retrieval_metrics_use_document_page_relevance() -> None:
    question = {
        "relevant": [
            {"document_id": "rules", "source": "rules.pdf", "page": 2},
            {"document_id": "fees", "source": "fees.pdf", "page": 1},
        ]
    }
    retrieved = [
        {"document_id": "rules", "source": "rules.pdf", "page": 2},
        {"document_id": "rules", "source": "rules.pdf", "page": 3},
        {"document_id": "fees", "source": "fees.pdf", "page": 1},
    ]
    assert recall_at_k(question, retrieved, 1) == 0.5
    assert recall_at_k(question, retrieved, 3) == 1.0
    assert mean_reciprocal_rank(question, retrieved) == 1.0
    assert ndcg_at_k(question, retrieved, 3) > 0.9


def test_summary_computes_abstention_latency_and_citation_validity() -> None:
    questions = [
        {"id": "a", "must_abstain": False, "relevant": [], "expected_answer": "rule"},
        {"id": "b", "must_abstain": True, "relevant": []},
    ]
    rows = [
        {
            "id": "a",
            "seconds": 0.1,
            "response": {
                "answer": "The rule applies.",
                "answered_from_context": True,
                "citations": [{"source": "rules.md", "section": "1.1", "page": 1}],
                "retrieved": [{"source": "rules.md", "section": "1.1", "page": 1}],
            },
        },
        {
            "id": "b",
            "seconds": 0.3,
            "response": {"answer": "Not covered.", "answered_from_context": False,
                         "citations": [], "retrieved": []},
        },
    ]
    summary = summarise_rows(questions, rows)
    assert summary["answers"]["abstention_correctness"] == 1.0
    assert summary["answers"]["citation_validity"] == 1.0
    assert summary["answers"]["latency_p50_seconds"] == 0.2


def test_trace_tool_reconstructs_context_and_prompt(tmp_path: Path) -> None:
    trace_file = tmp_path / "queries.jsonl"
    trace_file.write_text(
        '{"trace_id":"trace-1","user_query":"What is the rule?",'
        '"cache":"miss","generation_context":[{"chunk_id":"c1","source":"rules.md",'
        '"document_version":"2026","page":1,"section":"Attendance","text":"80 percent"}],'
        '"prompt":"Use the evidence: 80 percent","prompt_hash":"abc"}\n',
        encoding="utf-8",
    )
    trace = load_trace(trace_file, "trace-1")
    rendered = format_trace(trace)
    assert "Use the evidence: 80 percent" in rendered
    assert "80 percent" in rendered
    assert "rules.md" in rendered
