import json
from pathlib import Path

import pytest

from data_preprocess import build_datasets
from data_preprocess.common import read_jsonl, write_jsonl, stable_record_hash
from data_preprocess.generate_with_ark import (
    sort_response_file,
    validate_generated_plan,
    validate_grpo_annotation
)
from data_preprocess.prompts import build_prompt
from data_preprocess.clean_data import best_evidence_chunk, chunk_policy_document
from data_preprocess.build_datasets import (
    convert_musique_plan,
    make_rejected_query,
    make_rejected_multihop_plan
)


def test_policy_chunking_and_evidence_mapping() -> None:
    """Build a policy chunk and map source HTML evidence to it."""
    document = {
        "title": "Paternity Leave",
        "url": "https://example.test/paternity",
        "contents": [
            "<h1>Eligibility</h1>",
            "<p>You must work for your employer for at least 26 weeks.</p>"
        ]
    }
    chunks = chunk_policy_document(document, ["train"])
    chunk_id, score = best_evidence_chunk(
        "<p>You must work for your employer for at least 26 weeks.</p>",
        chunks
    )
    assert chunk_id == chunks[0]["id"]
    assert score > 1.0


def test_musique_plan_conversion() -> None:
    """Convert MuSiQue references into explicit planner dependencies."""
    record = {
        "question_decomposition": [
            {"step": 1, "question": "The Collegian >> owned by"},
            {"step": 2, "question": "When was #1 founded?"}
        ]
    }
    plan = json.loads(convert_musique_plan(record))
    assert plan["queries"][1]["depends_on"] == ["q1"]
    assert "{{q1.answer}}" in plan["queries"][1]["query"]


def test_musique_plan_preserves_hash_number_in_entity() -> None:
    """Keep a hash-prefixed title literal while converting real dependencies."""
    record = {
        "question_decomposition": [
            {"step": 1, "question": "#9 Dream >> performer"},
            {"step": 2, "question": "#1 >> father"}
        ]
    }
    plan = json.loads(convert_musique_plan(record))
    assert plan["queries"][0]["query"] == "#9 Dream performer"
    assert plan["queries"][0]["depends_on"] == []
    assert plan["queries"][1]["depends_on"] == ["q1"]


def test_rejected_query_differs() -> None:
    """Create a plausible negative that differs from the chosen query."""
    record = {"question": "Did he qualify?", "title": "Paternity Leave"}
    chosen = "Paternity Leave eligibility after 26 weeks employment"
    rejected = make_rejected_query(record, chosen, "constraint_omission", "wrong topic")
    assert rejected != chosen


@pytest.mark.parametrize(
    "error_type",
    [
        "step_omission",
        "broken_dependency",
        "redundant_step",
        "overly_broad_step",
        "relation_omission"
    ]
)
def test_multihop_rejected_plan_is_valid_and_different(error_type: str) -> None:
    """Create valid hard negatives for every multi-hop planning error.

    Args:
        error_type: Multi-hop negative category under test.
    """
    chosen = json.dumps(
        {
            "queries": [
                {"id": "q1", "query": "person birthplace", "depends_on": []},
                {
                    "id": "q2",
                    "query": "{{q1.answer}} administrative region",
                    "depends_on": ["q1"]
                }
            ]
        },
        separators = (",", ":")
    )
    rejected = make_rejected_multihop_plan(chosen, error_type)
    assert rejected != chosen
    assert json.loads(rejected)["queries"]


def test_generation_prompt_contains_schema() -> None:
    """Render a complete SFT generation prompt."""
    prompt = build_prompt(
        "sft",
        {
            "title": "Paternity Leave",
            "scenario": "I have worked for two months.",
            "question": "Do I qualify?"
        }
    )
    assert "Return exactly one JSON object" in prompt
    assert '"queries"' in prompt


def test_generation_request_hash_changes_with_prompt() -> None:
    """Fingerprint the complete request while excluding its own hash field."""
    request = {"id": "api_example", "prompt": "first prompt"}
    first_hash = stable_record_hash(request)
    request["request_hash"] = first_hash
    assert stable_record_hash(request) == first_hash
    request["prompt"] = "changed prompt"
    assert stable_record_hash(request) != first_hash


def test_sort_response_file_uses_request_order(tmp_path: Path) -> None:
    """Normalize concurrent response output to the request queue order.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    path = tmp_path / "responses.jsonl"
    requests = [{"id": "first"}, {"id": "second"}, {"id": "third"}]
    write_jsonl(
        path,
        [
            {"id": "third", "status": "success"},
            {"id": "first", "status": "invalid"},
            {"id": "second", "status": "success"},
            {"id": "first", "status": "success"}
        ]
    )
    sort_response_file(path, requests)
    records = read_jsonl(path)
    assert [record["id"] for record in records] == ["first", "second", "third"]
    assert records[0]["status"] == "success"


def test_grpo_answer_leakage_checks_each_reference_answer() -> None:
    """Reject a synthetic policy plan that copies its reference answer."""
    request = {
        "stage": "grpo",
        "source_question": "What support is available?",
        "evidence_items": [
            {"text": "The relevant office is the council.", "gold_doc_id": "doc1"},
            {"text": "The council offers a disabled facilities grant.", "gold_doc_id": "doc2"}
        ]
    }
    response = json.dumps(
        {
            "scenario": "A resident needs an accessibility adaptation.",
            "question": "Which support can the responsible office provide?",
            "plan": {
                "queries": [
                    {"id": "q1", "query": "responsible office council", "depends_on": []},
                    {
                        "id": "q2",
                        "query": "disabled facilities grant from {{q1.answer}}",
                        "depends_on": ["q1"]
                    }
                ]
            },
            "hop_answers": ["council", "disabled facilities grant"],
            "hop_evidence_indices": [0, 1],
            "reference_answer": "disabled facilities grant"
        }
    )
    with pytest.raises(ValueError, match = "leaks"):
        validate_generated_plan(request, response)


def test_validate_grounded_grpo_annotation() -> None:
    """Accept grounded synthetic policy multi-hop metadata."""
    request = {
        "stage": "grpo",
        "source_question": "What support is available?",
        "evidence_items": [
            {"text": "The responsible public body is the local council.", "gold_doc_id": "doc1"},
            {"text": "A disabled facilities grant can pay for adaptations.", "gold_doc_id": "doc2"}
        ]
    }
    response = json.dumps(
        {
            "scenario": "A resident needs an accessibility adaptation.",
            "question": "What funding is offered by the responsible public body?",
            "plan": {
                "queries": [
                    {"id": "q1", "query": "responsible public body", "depends_on": []},
                    {
                        "id": "q2",
                        "query": "funding offered by {{q1.answer}} for adaptations",
                        "depends_on": ["q1"]
                    }
                ]
            },
            "hop_answers": ["local council", "disabled facilities grant"],
            "hop_evidence_indices": [0, 1],
            "reference_answer": "grant"
        }
    )
    annotation = validate_grpo_annotation(request, response)
    assert annotation["gold_doc_ids"] == ["doc1", "doc2"]


def test_multihop_allocation_is_deterministic_and_disjoint(monkeypatch) -> None:
    """Allocate stable disjoint cold-start and GRPO records by hop.

    Args:
        monkeypatch: Pytest fixture used to reduce the production hop minimum.
    """
    monkeypatch.setattr(build_datasets, "GRPO_HOP_MINIMUM", 1)
    records = []
    for hop_count in [2, 3, 4]:
        for index in range(10):
            records.append(
                {
                    "id": f"{hop_count}hop_{index}",
                    "question": f"Unique {hop_count} hop question {index}",
                    "hop_count": hop_count
                }
            )
    quotas = {2: 2, 3: 2, 4: 2}
    first = build_datasets.allocate_multihop_records(
        records,
        quotas,
        {2: 1, 3: 1, 4: 1},
        6,
        42
    )
    second = build_datasets.allocate_multihop_records(
        records,
        quotas,
        {2: 1, 3: 1, 4: 1},
        6,
        42
    )
    first_cold, first_dpo, first_grpo, first_allocations = first
    second_cold, second_dpo, second_grpo, second_allocations = second
    assert [record["id"] for record in first_cold] == [
        record["id"]
        for record in second_cold
    ]
    assert [record["id"] for record in first_grpo] == [
        record["id"]
        for record in second_grpo
    ]
    assert [record["id"] for record in first_dpo] == [
        record["id"]
        for record in second_dpo
    ]
    assert first_allocations == second_allocations
    assert len(first_cold) == 6
    assert len(first_dpo) == 3
    assert len(first_grpo) == 6
    assert {
        hop_count: sum(record["hop_count"] == hop_count for record in first_cold)
        for hop_count in [2, 3, 4]
    } == quotas
    assert not {record["id"] for record in first_cold}.intersection(
        record["id"] for record in first_grpo
    )
    assert not {record["id"] for record in first_dpo}.intersection(
        record["id"] for record in first_grpo
    )
