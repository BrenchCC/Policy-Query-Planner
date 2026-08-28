import os
import sys
import json
import logging
import argparse
from typing import Any

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import (
    read_jsonl,
    write_json,
    write_jsonl,
    normalize_text,
    stable_record_hash
)
from data_preprocess.config import INTERIM_ROOT, PROCESSED_ROOT, REQUEST_ROOT, PLANNER_SYSTEM_PROMPT
from data_preprocess.prompts import build_prompt

logger = logging.getLogger(__name__)

STAGES = ["sft", "dpo", "grpo"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Prepare optional Ark generation queues")
    parser.add_argument(
        "--stage",
        choices = STAGES + ["all"],
        required = True,
        help = "Generation queue to prepare"
    )
    return parser.parse_args()


def flatten_answers(answers: list[Any]) -> str:
    """Flatten ConditionalQA answer structures for prompt metadata.

    Args:
        answers: Raw ConditionalQA answer list.

    Returns:
        Semicolon-separated reference answers.
    """
    values = []
    for answer in answers:
        if isinstance(answer, list) and answer:
            value = normalize_text(answer[0])
            if value and value not in values:
                values.append(value)
    return "; ".join(values) or "not answerable"


def prepare_sft_requests() -> list[dict[str, Any]]:
    """Prepare ConditionalQA domain rewrite requests.

    Returns:
        SFT generation request records.
    """
    records = read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
    requests = []
    for record in records:
        payload = {
            "title": record["title"],
            "scenario": record["scenario"],
            "question": record["question"]
        }
        requests.append(
            {
                "id": "api_sft_" + record["id"],
                "stage": "sft",
                "source_dataset": "conditionalqa",
                "source_id": record["id"],
                "system": PLANNER_SYSTEM_PROMPT,
                "prompt": build_prompt("sft", payload),
                "payload": payload
            }
        )
    return requests


def prepare_dpo_requests() -> list[dict[str, Any]]:
    """Prepare hard-negative preference requests.

    Returns:
        DPO generation request records.
    """
    records = [
        record
        for record in read_jsonl(PROCESSED_ROOT / "train" / "dpo_train.jsonl")
        if record["source_dataset"] == "conditionalqa"
    ]
    requests = []
    for record in records:
        payload = {
            "user_input": record["input"],
            "chosen": record["chosen"],
            "error_type": record["error_type"]
        }
        requests.append(
            {
                "id": "api_" + record["id"],
                "stage": "dpo",
                "source_dataset": "conditionalqa",
                "source_id": record["source_id"],
                "target_record_id": record["id"],
                "system": PLANNER_SYSTEM_PROMPT,
                "prompt": build_prompt("dpo", payload),
                "payload": payload
            }
        )
    return requests


def prepare_grpo_requests() -> list[dict[str, Any]]:
    """Prepare multi-evidence ConditionalQA decomposition requests.

    Returns:
        GRPO generation request records.
    """
    records = [
        record
        for record in read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
        if len(record["evidences"]) >= 2 and not record["not_answerable"]
    ]
    requests = []
    for record in records:
        answer = flatten_answers(record["answers"])
        payload = {
            "title": record["title"],
            "scenario": record["scenario"],
            "question": record["question"],
            "evidence": "\n".join(f"- {value}" for value in record["evidences"])
        }
        requests.append(
            {
                "id": "api_grpo_" + record["id"],
                "stage": "grpo",
                "source_dataset": "conditionalqa",
                "source_id": record["id"],
                "system": PLANNER_SYSTEM_PROMPT,
                "prompt": build_prompt("grpo", payload),
                "payload": payload,
                "reference_answer": answer,
                "gold_doc_ids": record["gold_doc_ids"],
                "evidences": record["evidences"]
            }
        )
    return requests


def prepare_stage(stage: str) -> int:
    """Prepare one stage and write its JSONL queue.

    Args:
        stage: Generation stage name.

    Returns:
        Number of request records written.
    """
    builders = {
        "sft": prepare_sft_requests,
        "dpo": prepare_dpo_requests,
        "grpo": prepare_grpo_requests
    }
    records = builders[stage]()
    for record in records:
        record["request_hash"] = stable_record_hash(record)
    count = write_jsonl(REQUEST_ROOT / f"{stage}_requests.jsonl", records)
    logger.info("Prepared %d %s requests", count, stage)
    return count


def main() -> None:
    """Prepare selected optional generation queues."""
    args = parse_args()
    selected_stages = STAGES if args.stage == "all" else [args.stage]
    counts = {stage: prepare_stage(stage) for stage in selected_stages}
    summary_path = REQUEST_ROOT / "request_summary.json"
    existing = {}
    if summary_path.exists():
        with summary_path.open("r", encoding = "utf-8") as file:
            existing = json.load(file)
    existing.update(counts)
    write_json(summary_path, existing)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
