import os
import sys

sys.path.append(os.getcwd())

from evaluation.metrics import (
    token_f1,
    exact_match,
    retrieval_metrics,
    dependency_metrics,
    best_reference_scores,
    evidence_recall_metrics
)


def test_answer_metrics_normalize_and_choose_best_reference() -> None:
    """Answer metrics should normalize text and select aliases."""
    assert exact_match("The Lisbon!", "lisbon") == 1.0
    assert token_f1("central Atlantic", "central Atlantic Ocean") == 0.8
    assert best_reference_scores("ITF", ["International Tennis Federation", "ITF"]) == {
        "exact_match": 1.0,
        "token_f1": 1.0
    }


def test_retrieval_metrics_deduplicate_and_measure_joint_recall() -> None:
    """Repeated hits should not inflate recall and all Gold docs define joint recall."""
    result = retrieval_metrics(["a", "a", "x", "b"], ["a", "b"])
    assert result["recall@1"] == 0.5
    assert result["recall@5"] == 1.0
    assert result["joint_recall@1"] == 0.0
    assert result["joint_recall@5"] == 1.0
    assert result["mrr"] == 1.0


def test_evidence_recall_preserves_repeated_document_mappings() -> None:
    """Every evidence annotation should count even when documents repeat."""
    result = evidence_recall_metrics(["a"], ["a", "a", "b"])
    assert result["recall@1"] == 2 / 3


def test_dependency_metrics_handle_empty_and_partial_graphs() -> None:
    """Dependency scoring should support independent and partially correct plans."""
    independent = {"queries": [{"id": "q1", "query": "abc", "depends_on": []}]}
    assert dependency_metrics(independent, independent)["dependency_f1"] == 1.0
    predicted = {
        "queries": [
            {"id": "q1", "query": "one", "depends_on": []},
            {"id": "q2", "query": "two", "depends_on": ["q1"]},
            {"id": "q3", "query": "three", "depends_on": ["q1"]}
        ]
    }
    reference = {
        "queries": [
            {"id": "q1", "query": "one", "depends_on": []},
            {"id": "q2", "query": "two", "depends_on": ["q1"]},
            {"id": "q3", "query": "three", "depends_on": ["q2"]}
        ]
    }
    result = dependency_metrics(predicted, reference)
    assert result["dependency_exact_match"] == 0.0
    assert result["dependency_precision"] == 0.5
    assert result["dependency_recall"] == 0.5
    assert result["dependency_f1"] == 0.5
