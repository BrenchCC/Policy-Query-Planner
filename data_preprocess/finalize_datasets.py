import os
import sys
import json
import logging
import argparse
from collections import Counter
from typing import Any

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import (
    read_jsonl,
    write_json,
    write_jsonl,
    normalized_key
)
from data_preprocess.build_datasets import GRPO_INSTRUCTION, stratified_sample
from data_preprocess.config import (
    INTERIM_ROOT,
    PROCESSED_ROOT,
    REQUEST_ROOT,
    RESPONSE_ROOT,
    RANDOM_SEED,
    GRPO_POLICY_SINGLE_COUNT,
    GRPO_POLICY_MULTIHOP_COUNT,
    GRPO_GENERAL_MULTIHOP_COUNT
)
from data_preprocess.prompts import PLANNER_SYSTEM_PROMPT
from data_preprocess.schemas import validate_sft_record, validate_dpo_record, validate_grpo_record

logger = logging.getLogger(__name__)

STAGES = ["sft", "dpo", "grpo"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Merge validated API outputs into datasets")
    parser.add_argument("--stage", choices = STAGES + ["all"], required = True)
    return parser.parse_args()


def successful_responses(stage: str) -> list[dict[str, Any]]:
    """Load successful responses for one generation stage.

    Args:
        stage: Generation stage name.

    Returns:
        Successful response records, or an empty list when absent.
    """
    path = RESPONSE_ROOT / f"{stage}_responses.jsonl"
    if not path.exists():
        return []
    request_path = REQUEST_ROOT / f"{stage}_requests.jsonl"
    if not request_path.exists():
        return []
    request_hashes = {
        record["id"]: record.get("request_hash", "")
        for record in read_jsonl(request_path)
    }
    return [
        record
        for record in read_jsonl(path)
        if record["status"] == "success"
        and record.get("stage") == stage
        and record.get("request_hash", "") == request_hashes.get(record["id"])
    ]


def finalize_sft() -> dict[str, int]:
    """Replace local policy SFT plans with validated API plans.

    Returns:
        Merge statistics.
    """
    responses = successful_responses("sft")
    replacements = {record["source_id"]: record["parsed_output"] for record in responses}
    records = read_jsonl(PROCESSED_ROOT / "train" / "sft_train.jsonl")
    replaced = 0
    for record in records:
        if record["source_dataset"] == "conditionalqa" and record["source_id"] in replacements:
            record["output"] = replacements[record["source_id"]]
            validate_sft_record(record)
            replaced += 1
    write_jsonl(PROCESSED_ROOT / "train" / "sft_train_api_augmented.jsonl", records)
    return {"available": len(responses), "replaced": replaced, "total": len(records)}


def finalize_dpo() -> dict[str, int]:
    """Replace local policy DPO negatives with validated API negatives.

    Returns:
        Merge statistics.
    """
    responses = successful_responses("dpo")
    replacements = {
        record["target_record_id"]: record["parsed_output"]
        for record in responses
        if "target_record_id" in record
    }
    records = read_jsonl(PROCESSED_ROOT / "train" / "dpo_train.jsonl")
    replaced = 0
    for record in records:
        if record["id"] in replacements:
            record["rejected"] = replacements[record["id"]]
            validate_dpo_record(record)
            replaced += 1
    write_jsonl(PROCESSED_ROOT / "train" / "dpo_train_api_augmented.jsonl", records)
    return {"available": len(responses), "replaced": replaced, "total": len(records)}


def finalize_grpo() -> dict[str, int]:
    """Create the general multi-hop, policy multi-hop, and policy single mix.

    Returns:
        Merge statistics.

    Raises:
        ValueError: If the required number of grounded policy plans is unavailable.
    """
    responses = sorted(successful_responses("grpo"), key = lambda record: record["source_id"])
    conditional_records = {
        record["id"]: record
        for record in read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
    }
    policy_records = []
    for response in responses:
        if response["source_id"] not in conditional_records:
            continue
        source = conditional_records[response["source_id"]]
        plan = json.loads(response["parsed_output"])
        if len(plan["queries"]) not in {2, 3, 4}:
            continue
        record = {
            "id": "grpo_policy_multihop_" + response["id"].removeprefix("api_grpo_"),
            "instruction": GRPO_INSTRUCTION,
            "input": (
                f"Scenario:\n{response['generated_scenario']}\n\n"
                f"Question:\n{response['generated_question']}"
            ),
            "output": response["parsed_output"],
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "conditionalqa",
            "source_id": source["id"],
            "namespace": "policy",
            "hop_count": len(plan["queries"]),
            "task_type": "multi_hop",
            "reference_answer": response["reference_answer"],
            "answer_aliases": [],
            "hop_answers": response["hop_answers"],
            "hop_evidence_indices": response["hop_evidence_indices"],
            "gold_doc_ids": response["gold_doc_ids"],
            "sample_type": "conditionalqa_synthetic_multihop",
            "source_question": source["question"],
            "variant_index": response["variant_index"]
        }
        try:
            validate_grpo_record(record)
        except ValueError:
            continue
        policy_records.append(record)
    unique_policy_records = {}
    for record in policy_records:
        question_key = normalized_key(record["input"])
        unique_policy_records.setdefault(question_key, record)
    policy_records = list(unique_policy_records.values())
    if len(policy_records) < GRPO_POLICY_MULTIHOP_COUNT:
        raise ValueError(
            "GRPO finalization requires "
            f"{GRPO_POLICY_MULTIHOP_COUNT} grounded policy multi-hop responses; "
            f"found {len(policy_records)}"
        )
    policy_records = stratified_sample(
        policy_records,
        GRPO_POLICY_MULTIHOP_COUNT,
        lambda record: record["hop_count"],
        RANDOM_SEED + 21
    )

    baseline = read_jsonl(PROCESSED_ROOT / "train" / "grpo_train.jsonl")
    single_baseline = [record for record in baseline if record["task_type"] == "single_hop"]
    multi_baseline = [record for record in baseline if record["task_type"] == "multi_hop"]
    if len(single_baseline) != GRPO_POLICY_SINGLE_COUNT:
        raise ValueError("Unexpected policy single-hop GRPO baseline count")
    selected_general = stratified_sample(
        multi_baseline,
        GRPO_GENERAL_MULTIHOP_COUNT,
        lambda record: record["hop_count"],
        RANDOM_SEED + 20
    )
    mixed_records = selected_general + policy_records + single_baseline
    random_order = sorted(
        mixed_records,
        key = lambda record: (record["source_dataset"], record["id"])
    )
    mixed_records = stratified_sample(
        random_order,
        len(random_order),
        lambda record: record["task_type"],
        RANDOM_SEED + 22
    )
    write_jsonl(
        PROCESSED_ROOT / "train" / "grpo_train_mixed.jsonl",
        mixed_records
    )
    return {
        "available": len(responses),
        "general_multihop_records": len(selected_general),
        "policy_multihop_records": len(policy_records),
        "policy_single_records": len(single_baseline),
        "total": len(mixed_records)
    }


def main() -> None:
    """Finalize selected API-enhanced datasets."""
    args = parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]
    functions = {"sft": finalize_sft, "dpo": finalize_dpo, "grpo": finalize_grpo}
    summary = {}
    for stage in stages:
        try:
            summary[stage] = functions[stage]()
        except ValueError as error:
            summary[stage] = {"status": "not_ready", "error": str(error)}
            logger.warning("Skipping %s finalization: %s", stage, error)
    summary["output_counts"] = {
        key: value
        for key, value in Counter(
            path.name
            for path in (PROCESSED_ROOT / "train").glob("*_api_augmented.jsonl")
        ).items()
    }
    write_json(INTERIM_ROOT / "api_finalization_summary.json", summary)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
