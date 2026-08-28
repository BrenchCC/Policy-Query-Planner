from typing import Any

import pytest

from data_preprocess.schemas import (
    serialize_plan,
    validate_planner_plan,
    validate_multihop_sft_record,
    validate_grpo_record
)


def test_single_query_plan() -> None:
    """Accept a valid one-query planner response."""
    value = serialize_plan(
        [
            {
                "id": "q1",
                "query": "paternity leave eligibility",
                "depends_on": []
            }
        ]
    )
    assert validate_planner_plan(value)["queries"][0]["id"] == "q1"


def test_dependency_placeholder() -> None:
    """Accept matching ordered dependencies and placeholders."""
    plan = {
        "queries": [
            {"id": "q1", "query": "policy owner", "depends_on": []},
            {
                "id": "q2",
                "query": "{{q1.answer}} eligibility rules",
                "depends_on": ["q1"]
            }
        ]
    }
    assert len(validate_planner_plan(plan)["queries"]) == 2


def test_reject_forward_dependency() -> None:
    """Reject references to a future query."""
    plan = {
        "queries": [
            {
                "id": "q1",
                "query": "{{q2.answer}} eligibility rules",
                "depends_on": ["q2"]
            },
            {"id": "q2", "query": "policy owner", "depends_on": []}
        ]
    }
    with pytest.raises(ValueError, match = "earlier query"):
        validate_planner_plan(plan)


def test_reject_nonsequential_query_ids() -> None:
    """Reject query IDs that skip an intermediate position."""
    plan = {
        "queries": [
            {"id": "q1", "query": "first fact", "depends_on": []},
            {"id": "q3", "query": "second fact", "depends_on": []}
        ]
    }
    with pytest.raises(ValueError, match = "sequential"):
        validate_planner_plan(plan)


def multihop_sft_record() -> dict[str, Any]:
    """Create one valid multi-hop cold-start record.

    Returns:
        Valid multi-hop cold-start record.
    """
    return {
        "id": "sft_musique_cold_start_example",
        "instruction": "Decompose the question.",
        "input": "Question:\nWhere is the birthplace located?",
        "output": serialize_plan(
            [
                {"id": "q1", "query": "person birthplace", "depends_on": []},
                {
                    "id": "q2",
                    "query": "{{q1.answer}} administrative region",
                    "depends_on": ["q1"]
                }
            ]
        ),
        "system": "Return valid JSON only.",
        "source_dataset": "musique",
        "source_id": "example",
        "sample_type": "musique_multihop_cold_start",
        "namespace": "musique_aux",
        "hop_count": 2
    }


def test_validate_multihop_sft_record() -> None:
    """Accept a cold-start record with strict metadata and matching hops."""
    validate_multihop_sft_record(multihop_sft_record())


def test_reject_multihop_sft_reference_metadata() -> None:
    """Reject GRPO reference metadata from the supervised cold-start file."""
    record = multihop_sft_record()
    record["gold_doc_ids"] = ["doc1", "doc2"]
    with pytest.raises(ValueError, match = "fields"):
        validate_multihop_sft_record(record)


def test_reject_multihop_sft_wrong_sample_type() -> None:
    """Reject a cold-start record with incorrect provenance metadata."""
    record = multihop_sft_record()
    record["sample_type"] = "wrong"
    with pytest.raises(ValueError, match = "sample_type"):
        validate_multihop_sft_record(record)


def test_reject_grpo_hop_count_mismatch() -> None:
    """Reject a GRPO record whose plan length differs from hop_count."""
    record = {
        **multihop_sft_record(),
        "id": "grpo_example",
        "hop_count": 3,
        "reference_answer": "region",
        "answer_aliases": [],
        "hop_answers": ["place", "region", "country"],
        "gold_doc_ids": ["doc1", "doc2", "doc3"]
    }
    with pytest.raises(ValueError, match = "query count"):
        validate_grpo_record(record)
