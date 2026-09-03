import os
import sys
import json

import pytest

sys.path.append(os.getcwd())

from evaluation.reporting import aggregate_scores
from evaluation.score_predictions import score_files, score_prediction


def test_score_rag_prediction() -> None:
    """RAG scoring should include retrieval, answer, and answerability metrics."""
    record = {
        "id": "r1",
        "source_dataset": "conditionalqa",
        "answerable": True,
        "reference_answers": ["within 10 days"],
        "gold_doc_ids": ["d1"],
        "gold_evidence_doc_ids": ["d1"]
    }
    prediction = {
        "id": "r1",
        "success": True,
        "steps": [{"evidence": [{"id": "d1"}]}],
        "evidence": [{"id": "d1"}],
        "answer": "Within 10 days."
    }
    score = score_prediction(record, prediction)
    assert score["metrics"]["doc_recall@1"] == 1.0
    assert score["metrics"]["evidence_recall@1"] == 1.0
    assert score["metrics"]["answer_exact_match"] == 1.0
    assert score["metrics"]["answerability_accuracy"] == 1.0
    assert score["metrics"]["answerability_coverage"] == 1.0


def test_failed_rag_prediction_is_not_scored_as_correct_refusal() -> None:
    """Exclude failed inference from answerability accuracy and expose zero coverage."""
    record = {
        "id": "r1",
        "source_dataset": "conditionalqa",
        "answerable": False,
        "reference_answers": [],
        "gold_doc_ids": [],
        "gold_evidence_doc_ids": []
    }
    prediction = {
        "id": "r1",
        "success": False,
        "answer": None,
        "evidence": [],
        "errors": [{"stage": "batch", "error": "failed"}]
    }
    metrics = score_prediction(record, prediction)["metrics"]
    assert metrics["answerability_coverage"] == 0.0
    assert "answerability_accuracy" not in metrics


def test_score_multihop_prediction() -> None:
    """Multi-hop scoring should align plans, steps, and per-hop retrieval."""
    plan = {
        "queries": [
            {"id": "q1", "query": "Who is X", "depends_on": []},
            {
                "id": "q2",
                "query": "Where is {{q1.answer}}",
                "depends_on": ["q1"]
            }
        ]
    }
    record = {
        "id": "m1",
        "source_dataset": "musique",
        "hop_count": 2,
        "answerable": True,
        "reference_answers": ["Lisbon"],
        "reference_plan": plan,
        "reference_steps": [
            {
                "id": "q1",
                "query": "Who is X",
                "depends_on": [],
                "answer": "Y",
                "answer_aliases": [],
                "gold_doc_id": "d1"
            },
            {
                "id": "q2",
                "query": "Where is {{q1.answer}}",
                "depends_on": ["q1"],
                "answer": "Lisbon",
                "answer_aliases": [],
                "gold_doc_id": "d2"
            }
        ]
    }
    prediction = {
        "id": "m1",
        "success": True,
        "planner": {"parse_success": True, "schema_valid": True, "parsed_plan": plan},
        "steps": [
            {"answer": "Y", "evidence": [{"id": "d1"}]},
            {"answer": "Lisbon", "evidence": [{"id": "d2"}]}
        ],
        "answer": "Lisbon"
    }
    score = score_prediction(record, prediction)
    metrics = score["metrics"]
    assert metrics["schema_valid"] == 1.0
    assert metrics["hop_count_accuracy"] == 1.0
    assert metrics["dependency_exact_match"] == 1.0
    assert metrics["joint_recall@1"] == 1.0
    assert metrics["final_answer_exact_match"] == 1.0
    assert score["per_step_scores"][0]["retrieval"]["mrr"] == 1.0
    assert score["per_step_scores"][1]["intermediate_answer_exact_match"] == 1.0


def test_invalid_schema_plan_is_scored_without_step_attribute_errors() -> None:
    """Keep parse success while assigning zero plan scores to malformed query items."""
    record = {
        "id": "m-invalid",
        "source_dataset": "musique",
        "hop_count": 2,
        "answerable": True,
        "reference_answers": ["answer"],
        "reference_plan": {
            "queries": [
                {"id": "q1", "query": "first", "depends_on": []},
                {"id": "q2", "query": "second", "depends_on": ["q1"]}
            ]
        },
        "reference_steps": [
            {
                "id": "q1",
                "query": "first",
                "depends_on": [],
                "answer": "middle",
                "answer_aliases": [],
                "gold_doc_id": "d1"
            },
            {
                "id": "q2",
                "query": "second",
                "depends_on": ["q1"],
                "answer": "answer",
                "answer_aliases": [],
                "gold_doc_id": "d2"
            }
        ]
    }
    prediction = {
        "id": "m-invalid",
        "success": False,
        "planner": {
            "parse_success": True,
            "schema_valid": False,
            "parsed_plan": {"queries": ["bad"]}
        },
        "steps": [],
        "answer": None
    }

    score = score_prediction(record, prediction)

    assert score["metrics"]["json_parse"] == 1.0
    assert score["metrics"]["schema_valid"] == 0.0
    assert score["metrics"]["hop_count_accuracy"] == 0.0
    assert score["metrics"]["query_exact_match"] == 0.0


def test_score_files_writes_all_reports_and_strata(tmp_path) -> None:
    """File scoring should deterministically write all three report artifacts."""
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "output"
    record = {
        "id": "r1",
        "source_dataset": "conditionalqa",
        "answerable": False,
        "reference_answers": [],
        "gold_doc_ids": [],
        "gold_evidence_doc_ids": []
    }
    prediction = {"id": "r1", "success": True, "answer": None, "evidence": []}
    dataset_path.write_text(json.dumps(record) + "\n", encoding = "utf-8")
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding = "utf-8")
    summary = score_files(dataset_path, predictions_path, output_dir)
    assert summary["count"] == 1
    assert summary["by_answerability"]["false"]["count"] == 1
    assert "doc_recall@1" not in summary["overall"]
    assert "evidence_recall@1" not in summary["overall"]
    assert (output_dir / "scores.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert aggregate_scores([])["overall"] == {}

    with pytest.raises(FileExistsError, match = "--force"):
        score_files(dataset_path, predictions_path, output_dir)
    forced_summary = score_files(
        dataset_path,
        predictions_path,
        output_dir,
        force = True
    )
    assert forced_summary["count"] == 1
