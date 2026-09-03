import os
import re
import sys
import json
import logging
import hashlib
from typing import Any

import jsonschema

# Add project root to Python path / 将项目根目录加入 Python 路径
sys.path.append(os.getcwd())

from data_preprocess.common import normalize_text

logger = logging.getLogger(__name__)

EVAL_SCHEMA_VERSION = "1.0"

PLANNER_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries"],
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "query", "depends_on"],
                "properties": {
                    "id": {"type": "string", "pattern": "^q[1-4]$"},
                    "query": {"type": "string", "minLength": 3},
                    "depends_on": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^q[1-4]$"}
                    }
                }
            }
        }
    }
}

RAG_EVAL_FIELDS = {
    "id",
    "input",
    "system",
    "scenario",
    "metadata",
    "question",
    "source_id",
    "answerable",
    "namespace",
    "instruction",
    "answer_groups",
    "gold_doc_ids",
    "schema_version",
    "gold_evidences",
    "source_dataset",
    "reference_answers",
    "gold_evidence_doc_ids"
}
MULTIHOP_EVAL_FIELDS = {
    "id",
    "input",
    "system",
    "metadata",
    "question",
    "source_id",
    "answerable",
    "namespace",
    "hop_count",
    "instruction",
    "gold_doc_ids",
    "reference_plan",
    "schema_version",
    "reference_steps",
    "source_dataset",
    "reference_answers"
}
MULTIHOP_REFERENCE_STEP_FIELDS = {
    "id",
    "query",
    "answer",
    "depends_on",
    "gold_doc_id",
    "answer_aliases"
}


def evaluation_schema_hash() -> str:
    """Hash the normalized evaluation and planner schema contracts.

    Returns:
        SHA256 digest of the public evaluation schema definition.
    """
    definition = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "planner_plan_schema": PLANNER_PLAN_SCHEMA,
        "rag": {
            "fields": sorted(RAG_EVAL_FIELDS),
            "source_dataset": "conditionalqa",
            "namespace": "policy"
        },
        "multihop": {
            "fields": sorted(MULTIHOP_EVAL_FIELDS),
            "reference_step_fields": sorted(MULTIHOP_REFERENCE_STEP_FIELDS),
            "hop_counts": [2, 3, 4],
            "source_dataset": "musique",
            "namespace": "musique_aux"
        }
    }
    serialized = json.dumps(
        definition,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def serialize_plan(queries: list[dict[str, Any]]) -> str:
    """Serialize a planner query list as compact JSON.

    Args:
        queries: Ordered query objects.

    Returns:
        Compact JSON planner response.
    """
    plan = {"queries": queries}
    validate_planner_plan(plan)
    return json.dumps(plan, ensure_ascii = False, separators = (",", ":"))


def validate_planner_plan(value: str | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate a planner output and dependency graph.

    Args:
        value: JSON string or parsed planner object.

    Returns:
        Validated planner object.

    Raises:
        ValueError: If JSON, fields, ordering, or placeholders are invalid.
    """
    if isinstance(value, str):
        try:
            plan = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid planner JSON: {error}") from error
    else:
        plan = value
    try:
        jsonschema.validate(instance = plan, schema = PLANNER_PLAN_SCHEMA)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Planner schema validation failed: {error.message}") from error

    seen_ids = set()
    expected_ids = [f"q{index}" for index in range(1, len(plan["queries"]) + 1)]
    actual_ids = [query_item["id"] for query_item in plan["queries"]]
    if actual_ids != expected_ids:
        raise ValueError(f"Planner query IDs must be sequential: {expected_ids}")
    for query_item in plan["queries"]:
        query_id = query_item["id"]
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query ID: {query_id}")
        dependencies = query_item["depends_on"]
        for dependency in dependencies:
            if dependency not in seen_ids:
                raise ValueError(f"Dependency {dependency} must refer to an earlier query")
            placeholder = "{{" + dependency + ".answer}}"
            if placeholder not in query_item["query"]:
                raise ValueError(f"Dependent query {query_id} must contain {placeholder}")
        placeholders = re.findall(r"\{\{(q[1-4])\.answer\}\}", query_item["query"])
        if set(placeholders) != set(dependencies):
            raise ValueError(f"Query {query_id} placeholders must match depends_on")
        seen_ids.add(query_id)
    return plan


def validate_sft_record(record: dict[str, Any]) -> None:
    """Validate one Alpaca SFT record.

    Args:
        record: SFT record to validate.

    Raises:
        ValueError: If required fields or planner output are invalid.
    """
    required_fields = ["id", "instruction", "input", "output", "system"]
    for field in required_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"SFT field {field} must be a non-empty string")
    validate_planner_plan(record["output"])


def validate_multihop_sft_record(record: dict[str, Any]) -> None:
    """Validate one supervised multi-hop cold-start record.

    Args:
        record: Multi-hop SFT cold-start record to validate.

    Raises:
        ValueError: If provenance, metadata, or plan length is invalid.
    """
    allowed_fields = {
        "id",
        "input",
        "output",
        "system",
        "hop_count",
        "instruction",
        "source_id",
        "namespace",
        "sample_type",
        "source_dataset"
    }
    if set(record) != allowed_fields:
        raise ValueError("Multi-hop SFT fields must match the cold-start contract")
    validate_sft_record(record)
    if not record["id"].startswith("sft_musique_cold_start_"):
        raise ValueError("Multi-hop SFT ID must use the cold-start prefix")
    if not isinstance(record.get("source_id"), str) or not record["source_id"].strip():
        raise ValueError("Multi-hop SFT source_id must be non-empty")
    if record.get("source_dataset") != "musique":
        raise ValueError("Multi-hop SFT source_dataset must be musique")
    if record.get("sample_type") != "musique_multihop_cold_start":
        raise ValueError("Multi-hop SFT sample_type is invalid")
    if record.get("namespace") != "musique_aux":
        raise ValueError("Multi-hop SFT namespace must be musique_aux")
    hop_count = record.get("hop_count")
    if hop_count not in {2, 3, 4}:
        raise ValueError("Multi-hop SFT hop_count must be 2, 3, or 4")
    plan = validate_planner_plan(record["output"])
    if len(plan["queries"]) != hop_count:
        raise ValueError("Multi-hop SFT query count must match hop_count")


def validate_dpo_record(record: dict[str, Any]) -> None:
    """Validate one LLaMA-Factory preference record.

    Args:
        record: DPO preference record to validate.

    Raises:
        ValueError: If fields or preference outputs are invalid.
    """
    required_fields = [
        "id",
        "instruction",
        "input",
        "chosen",
        "rejected",
        "system"
    ]
    for field in required_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"DPO field {field} must be a non-empty string")
    if record["chosen"] == record["rejected"]:
        raise ValueError("DPO chosen and rejected outputs must differ")
    validate_planner_plan(record["chosen"])
    validate_planner_plan(record["rejected"])
    task_type = record.get("task_type")
    hop_count = record.get("hop_count")
    chosen_query_count = len(validate_planner_plan(record["chosen"])["queries"])
    if task_type == "single_hop":
        if hop_count != 1 or chosen_query_count != 1:
            raise ValueError("Single-hop DPO records must contain one chosen query")
    elif task_type == "multi_hop":
        if hop_count not in {2, 3, 4} or chosen_query_count != hop_count:
            raise ValueError("Multi-hop DPO chosen query count must match hop_count")
        if record.get("source_dataset") != "musique":
            raise ValueError("Multi-hop DPO records must use MuSiQue")
        if record.get("namespace") != "musique_aux":
            raise ValueError("Multi-hop DPO namespace must be musique_aux")
    else:
        raise ValueError("DPO task_type must be single_hop or multi_hop")


def validate_grpo_record(record: dict[str, Any]) -> None:
    """Validate one GRPO-compatible record.

    Args:
        record: GRPO record to validate.

    Raises:
        ValueError: If prompt, plan, or reference metadata are invalid.
    """
    validate_sft_record(record)
    hop_count = record.get("hop_count")
    if hop_count not in {1, 2, 3, 4}:
        raise ValueError("GRPO hop_count must be 1, 2, 3, or 4")
    plan = validate_planner_plan(record["output"])
    if len(plan["queries"]) != hop_count:
        raise ValueError("GRPO query count must match hop_count")
    if not isinstance(record.get("reference_answer"), str) or not record["reference_answer"].strip():
        raise ValueError("GRPO reference_answer must be non-empty")
    if record.get("namespace") not in {"musique_aux", "policy"}:
        raise ValueError("GRPO namespace is invalid")
    task_type = record.get("task_type")
    expected_task_type = "single_hop" if hop_count == 1 else "multi_hop"
    if task_type != expected_task_type:
        raise ValueError("GRPO task_type must match hop_count")
    hop_answers = record.get("hop_answers")
    gold_doc_ids = record.get("gold_doc_ids")
    if not isinstance(hop_answers, list):
        raise ValueError("GRPO hop_answers must be a list")
    if not isinstance(gold_doc_ids, list) or not gold_doc_ids:
        raise ValueError("GRPO gold_doc_ids must be a non-empty list")
    if not all(isinstance(answer, str) and answer.strip() for answer in hop_answers):
        raise ValueError("GRPO hop_answers must contain non-empty strings")
    if not all(isinstance(doc_id, str) and doc_id.strip() for doc_id in gold_doc_ids):
        raise ValueError("GRPO gold_doc_ids must contain non-empty strings")
    if record["namespace"] == "musique_aux":
        if record.get("source_dataset") != "musique":
            raise ValueError("MuSiQue GRPO namespace requires MuSiQue provenance")
        if hop_count == 1:
            raise ValueError("MuSiQue GRPO records must be multi-hop")
        if len(hop_answers) != hop_count:
            raise ValueError("MuSiQue GRPO hop_answers must match hop_count")
        if len(gold_doc_ids) != hop_count:
            raise ValueError("MuSiQue GRPO gold_doc_ids must match hop_count")
    elif record.get("source_dataset") != "conditionalqa":
        raise ValueError("Policy GRPO namespace requires ConditionalQA provenance")
    elif hop_count == 1:
        if len(gold_doc_ids) < 1:
            raise ValueError("Single-hop policy GRPO records require gold documents")
    else:
        if record.get("sample_type") != "conditionalqa_synthetic_multihop":
            raise ValueError("Policy multi-hop GRPO sample_type is invalid")
        if len(hop_answers) != hop_count:
            raise ValueError("Policy multi-hop GRPO hop_answers must match hop_count")
        evidence_indices = record.get("hop_evidence_indices")
        if (
            not isinstance(evidence_indices, list)
            or len(evidence_indices) != hop_count
            or not all(isinstance(index, int) for index in evidence_indices)
            or len(set(evidence_indices)) != hop_count
        ):
            raise ValueError("Policy multi-hop GRPO evidence indices must match unique hops")
        if len(gold_doc_ids) != hop_count or len(set(gold_doc_ids)) != hop_count:
            raise ValueError("Policy multi-hop GRPO requires one distinct Gold document per hop")
        if not any(query["depends_on"] for query in plan["queries"]):
            raise ValueError("Policy multi-hop GRPO requires at least one dependency")


def validate_knowledge_record(record: dict[str, Any]) -> None:
    """Validate one knowledge-base JSONL record.

    Args:
        record: Knowledge-base record to validate.

    Raises:
        ValueError: If required fields are missing or empty.
    """
    required_fields = [
        "id",
        "text",
        "title",
        "source",
        "source_dataset",
        "namespace",
        "content_hash"
    ]
    for field in required_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Knowledge-base field {field} must be a non-empty string")


def validate_eval_record_common(
    record: dict[str, Any],
    allowed_fields: set[str],
    knowledge_doc_ids: set[str] | None = None
) -> None:
    """Validate fields shared by model-ready evaluation records.

    Args:
        record: Evaluation record to validate.
        allowed_fields: Exact fields allowed by the dataset contract.
        knowledge_doc_ids: Optional set of valid knowledge-base document IDs.

    Raises:
        ValueError: If fields, values, or Gold document references are invalid.
    """
    if set(record) != allowed_fields:
        raise ValueError("Evaluation fields must exactly match the dataset contract")
    string_fields = [
        "id",
        "input",
        "system",
        "question",
        "source_id",
        "namespace",
        "instruction",
        "source_dataset"
    ]
    for field in string_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Evaluation field {field} must be a non-empty string")
    if record.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError("Evaluation schema_version is unsupported")
    if not isinstance(record.get("answerable"), bool):
        raise ValueError("Evaluation answerable must be boolean")
    reference_answers = record.get("reference_answers")
    if not isinstance(reference_answers, list):
        raise ValueError("Evaluation reference_answers must be a list")
    if not all(isinstance(answer, str) and answer.strip() for answer in reference_answers):
        raise ValueError("Evaluation reference_answers must contain non-empty strings")
    if record["answerable"] and not reference_answers:
        raise ValueError("Answerable evaluation records require reference answers")
    gold_doc_ids = record.get("gold_doc_ids")
    if not isinstance(gold_doc_ids, list):
        raise ValueError("Evaluation gold_doc_ids must be a list")
    if (
        not all(isinstance(doc_id, str) and doc_id.strip() for doc_id in gold_doc_ids)
        or len(gold_doc_ids) != len(set(gold_doc_ids))
    ):
        raise ValueError("Evaluation gold_doc_ids must be unique non-empty strings")
    if not isinstance(record.get("metadata"), dict):
        raise ValueError("Evaluation metadata must be an object")
    if knowledge_doc_ids is not None:
        missing_doc_ids = set(gold_doc_ids) - knowledge_doc_ids
        if missing_doc_ids:
            raise ValueError(f"Gold documents are missing from the knowledge base: {missing_doc_ids}")


def validate_rag_eval_record(
    record: dict[str, Any],
    knowledge_doc_ids: set[str] | None = None
) -> None:
    """Validate one model-ready policy RAG evaluation record.

    Args:
        record: Policy RAG evaluation record.
        knowledge_doc_ids: Optional set of valid policy document IDs.

    Raises:
        ValueError: If schema or evidence alignment is invalid.
    """
    validate_eval_record_common(record, RAG_EVAL_FIELDS, knowledge_doc_ids)
    if record["source_dataset"] != "conditionalqa" or record["namespace"] != "policy":
        raise ValueError("Policy RAG evaluation provenance is invalid")
    if not isinstance(record.get("scenario"), str):
        raise ValueError("Policy RAG scenario must be a string")
    answer_groups = record.get("answer_groups")
    if not isinstance(answer_groups, list):
        raise ValueError("Policy RAG answer_groups must be a list")
    for answer_group in answer_groups:
        if (
            not isinstance(answer_group, list)
            or len(answer_group) != 2
            or not isinstance(answer_group[0], str)
            or not answer_group[0].strip()
            or not isinstance(answer_group[1], list)
            or not all(
                isinstance(condition, str) and condition.strip()
                for condition in answer_group[1]
            )
        ):
            raise ValueError("Policy RAG answer_groups contain an invalid answer annotation")
    gold_evidences = record.get("gold_evidences")
    evidence_doc_ids = record.get("gold_evidence_doc_ids")
    if not isinstance(gold_evidences, list) or not isinstance(evidence_doc_ids, list):
        raise ValueError("Policy RAG evidence fields must be lists")
    if len(gold_evidences) != len(evidence_doc_ids):
        raise ValueError("Policy RAG evidence and document mappings must align")
    if not all(isinstance(value, str) and value.strip() for value in gold_evidences):
        raise ValueError("Policy RAG evidences must be non-empty strings")
    if not all(isinstance(value, str) and value.strip() for value in evidence_doc_ids):
        raise ValueError("Policy RAG evidence document IDs must be non-empty strings")
    if set(evidence_doc_ids) != set(record["gold_doc_ids"]):
        raise ValueError("Policy RAG aggregate and per-evidence Gold documents must match")
    primary_answers = list(dict.fromkeys(normalize_text(group[0]) for group in answer_groups))
    if primary_answers != record["reference_answers"]:
        raise ValueError("Policy RAG reference answers must match answer_groups")
    if record["answerable"] and (not answer_groups or not gold_evidences):
        raise ValueError("Answerable policy RAG records require answers and evidence")
    if not record["answerable"] and (
        answer_groups
        or gold_evidences
        or evidence_doc_ids
        or record["reference_answers"]
        or record["gold_doc_ids"]
    ):
        raise ValueError("Unanswerable policy RAG records cannot contain Gold annotations")
    if knowledge_doc_ids is not None and not set(evidence_doc_ids).issubset(knowledge_doc_ids):
        raise ValueError("Policy RAG evidence references a missing knowledge document")


def validate_multihop_eval_record(
    record: dict[str, Any],
    knowledge_doc_ids: set[str] | None = None
) -> None:
    """Validate one model-ready multi-hop planner evaluation record.

    Args:
        record: Multi-hop planner evaluation record.
        knowledge_doc_ids: Optional set of valid MuSiQue document IDs.

    Raises:
        ValueError: If plan, steps, or Gold annotations are inconsistent.
    """
    validate_eval_record_common(record, MULTIHOP_EVAL_FIELDS, knowledge_doc_ids)
    if record["source_dataset"] != "musique" or record["namespace"] != "musique_aux":
        raise ValueError("Multi-hop evaluation provenance is invalid")
    if not record["answerable"]:
        raise ValueError("MuSiQue multi-hop evaluation records must be answerable")
    hop_count = record.get("hop_count")
    if hop_count not in {2, 3, 4}:
        raise ValueError("Multi-hop evaluation hop_count must be 2, 3, or 4")
    plan = validate_planner_plan(record.get("reference_plan"))
    steps = record.get("reference_steps")
    if not isinstance(steps, list) or len(steps) != hop_count:
        raise ValueError("Multi-hop reference_steps must match hop_count")
    if len(plan["queries"]) != hop_count:
        raise ValueError("Multi-hop reference plan must match hop_count")
    for plan_query, step in zip(plan["queries"], steps):
        if not isinstance(step, dict) or set(step) != MULTIHOP_REFERENCE_STEP_FIELDS:
            raise ValueError("Multi-hop reference step fields are invalid")
        if any(step.get(field) != plan_query[field] for field in ["id", "query", "depends_on"]):
            raise ValueError("Multi-hop reference step does not align with reference plan")
        if not isinstance(step.get("answer"), str) or not step["answer"].strip():
            raise ValueError("Multi-hop step answer must be non-empty")
        if not isinstance(step.get("answer_aliases"), list):
            raise ValueError("Multi-hop step answer_aliases must be a list")
        if not all(isinstance(alias, str) and alias.strip() for alias in step["answer_aliases"]):
            raise ValueError("Multi-hop step aliases must be non-empty strings")
        if not isinstance(step.get("gold_doc_id"), str) or not step["gold_doc_id"].strip():
            raise ValueError("Multi-hop step gold_doc_id must be non-empty")
    step_doc_ids = [step["gold_doc_id"] for step in steps]
    if step_doc_ids != record["gold_doc_ids"]:
        raise ValueError("Multi-hop step documents must align with gold_doc_ids")
    final_references = list(
        dict.fromkeys([steps[-1]["answer"], *steps[-1]["answer_aliases"]])
    )
    if final_references != record["reference_answers"]:
        raise ValueError("Multi-hop final references must match the final step")
