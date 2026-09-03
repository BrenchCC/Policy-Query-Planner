import os
import sys
from typing import Any

import pytest

# Add project root to Python path / 将项目根目录加入 Python 路径
sys.path.append(os.getcwd())

from data_preprocess.build_datasets import (
    build_rag_eval_records,
    build_multihop_eval_records
)
from data_preprocess.schemas import (
    PLANNER_PLAN_SCHEMA,
    evaluation_schema_hash,
    validate_rag_eval_record,
    validate_multihop_eval_record
)


def test_evaluation_schema_hash_tracks_planner_contract(monkeypatch) -> None:
    """Change the schema digest whenever the normalized Planner contract changes."""
    original_hash = evaluation_schema_hash()
    monkeypatch.setitem(PLANNER_PLAN_SCHEMA, "title", "changed contract")
    assert evaluation_schema_hash() != original_hash


def conditionalqa_record() -> dict[str, Any]:
    """Create one clean ConditionalQA development record.

    Returns:
        Minimal ConditionalQA record accepted by the evaluation builder.
    """
    return {
        "id": "dev-1",
        "split": "dev",
        "url": "https://example.test/policy",
        "title": "Example policy",
        "scenario": "I need help.",
        "question": "What can I claim?",
        "not_answerable": False,
        "answers": [["A grant", ["if eligible"]]],
        "evidences": ["You can claim a grant if eligible."],
        "gold_doc_ids": ["policy_doc1"],
        "unresolved_evidence": []
    }


def test_build_rag_eval_record_with_aligned_evidence() -> None:
    """Build a policy record with preserved groups and aligned evidence IDs."""
    documents = {
        "https://example.test/policy": [
            {
                "id": "policy_doc1",
                "text": "You can claim a grant if eligible."
            }
        ]
    }
    records = build_rag_eval_records([conditionalqa_record()], documents)
    assert records[0]["reference_answers"] == ["A grant"]
    assert records[0]["answer_groups"] == [["A grant", ["if eligible"]]]
    assert records[0]["gold_evidence_doc_ids"] == ["policy_doc1"]
    validate_rag_eval_record(records[0], {"policy_doc1"})


def musique_record() -> dict[str, Any]:
    """Create one clean MuSiQue development record.

    Returns:
        Minimal two-hop MuSiQue record accepted by the evaluation builder.
    """
    return {
        "id": "2hop__1_2",
        "split": "dev",
        "question": "Where was the author born?",
        "answer": "London",
        "answer_aliases": ["London, England"],
        "answerable": True,
        "hop_count": 2,
        "question_decomposition": [
            {
                "step": 1,
                "source_id": 1,
                "question": "Book >> author",
                "answer": "Writer",
                "paragraph_support_idx": 0,
                "gold_doc_id": "musique_doc1"
            },
            {
                "step": 2,
                "source_id": 2,
                "question": "#1 >> birthplace",
                "answer": "London",
                "paragraph_support_idx": 1,
                "gold_doc_id": "musique_doc2"
            }
        ],
        "paragraph_doc_ids": ["musique_doc1", "musique_doc2"],
        "gold_doc_ids": ["musique_doc1", "musique_doc2"]
    }


def test_build_multihop_eval_record_with_aligned_steps() -> None:
    """Build dependency-aware plan and matching step annotations."""
    record = build_multihop_eval_records([musique_record()])[0]
    assert record["reference_answers"] == ["London", "London, England"]
    assert record["reference_plan"]["queries"][1]["depends_on"] == ["q1"]
    assert "{{q1.answer}}" in record["reference_steps"][1]["query"]
    validate_multihop_eval_record(record, {"musique_doc1", "musique_doc2"})


def test_reject_eval_record_with_missing_document() -> None:
    """Reject a Gold document absent from the selected knowledge base."""
    record = build_multihop_eval_records([musique_record()])[0]
    with pytest.raises(ValueError, match = "missing"):
        validate_multihop_eval_record(record, {"musique_doc1"})


def test_reject_eval_step_plan_mismatch() -> None:
    """Reject reference steps whose dependency annotation differs from the plan."""
    record = build_multihop_eval_records([musique_record()])[0]
    record["reference_steps"][1]["depends_on"] = []
    with pytest.raises(ValueError, match = "does not align"):
        validate_multihop_eval_record(record)


def test_reject_rag_evidence_alignment_mismatch() -> None:
    """Reject policy evaluation records lacking one document per evidence."""
    documents = {
        "https://example.test/policy": [
            {
                "id": "policy_doc1",
                "text": "You can claim a grant if eligible."
            }
        ]
    }
    record = build_rag_eval_records([conditionalqa_record()], documents)[0]
    record["gold_evidence_doc_ids"] = []
    with pytest.raises(ValueError, match = "must align"):
        validate_rag_eval_record(record)


def test_reject_rag_answer_group_mismatch() -> None:
    """Reject flattened references that diverge from preserved answer groups."""
    documents = {
        "https://example.test/policy": [
            {
                "id": "policy_doc1",
                "text": "You can claim a grant if eligible."
            }
        ]
    }
    record = build_rag_eval_records([conditionalqa_record()], documents)[0]
    record["reference_answers"] = ["A different answer"]
    with pytest.raises(ValueError, match = "must match answer_groups"):
        validate_rag_eval_record(record)
