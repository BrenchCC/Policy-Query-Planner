import os
import sys
import json
import logging
import argparse
from collections import defaultdict
from typing import Any

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import (
    read_jsonl,
    write_json,
    write_jsonl,
    stable_record_hash
)
from data_preprocess.config import INTERIM_ROOT, PROCESSED_ROOT, REQUEST_ROOT
from data_preprocess.prompts import build_prompt, PLANNER_SYSTEM_PROMPT
from data_preprocess.clean_data import best_evidence_chunk

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
    """Prepare policy single-hop and MuSiQue multi-hop negative requests.

    Returns:
        DPO generation request records.
    """
    records = [
        record
        for record in read_jsonl(PROCESSED_ROOT / "train" / "dpo_train.jsonl")
        if record["source_dataset"] in {"conditionalqa", "musique"}
    ]
    requests = []
    for record in records:
        payload = {
            "user_input": record["input"],
            "chosen": record["chosen"],
            "error_type": record["error_type"],
            "task_type": record["task_type"],
            "hop_count": record["hop_count"]
        }
        requests.append(
            {
                "id": "api_" + record["id"],
                "stage": "dpo",
                "source_dataset": record["source_dataset"],
                "source_id": record["source_id"],
                "target_record_id": record["id"],
                "system": PLANNER_SYSTEM_PROMPT,
                "prompt": build_prompt("dpo", payload),
                "payload": payload
            }
        )
    return requests


def prepare_grpo_requests() -> list[dict[str, Any]]:
    """Prepare grounded synthetic policy multi-hop generation requests.

    Returns:
        GRPO generation request records.
    """
    policy_chunks_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in read_jsonl(PROCESSED_ROOT / "knowledge_base" / "policy.jsonl"):
        policy_chunks_by_url[chunk["url"]].append(chunk)
    records = read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
    requests = []
    for record in records:
        if record["not_answerable"] or len(record["evidences"]) < 2:
            continue
        evidence_items = []
        for evidence in record["evidences"]:
            doc_id, score = best_evidence_chunk(
                evidence,
                policy_chunks_by_url[record["url"]]
            )
            if doc_id is None:
                continue
            evidence_items.append(
                {
                    "text": evidence,
                    "gold_doc_id": doc_id,
                    "mapping_score": score
                }
            )
        distinct_doc_ids = {
            item["gold_doc_id"]
            for item in evidence_items
        }
        if len(evidence_items) < 2 or len(distinct_doc_ids) < 2:
            continue
        variant_count = 2 if len(distinct_doc_ids) >= 3 else 1
        for variant_index in range(1, variant_count + 1):
            payload = {
                "title": record["title"],
                "scenario": record["scenario"],
                "question": record["question"],
                "variant_index": variant_index,
                "evidence": "\n".join(
                    f"[{index}] {item['text']}"
                    for index, item in enumerate(evidence_items)
                )
            }
            requests.append(
                {
                    "id": f"api_grpo_{record['id']}_v{variant_index}",
                    "stage": "grpo",
                    "source_dataset": "conditionalqa",
                    "source_id": record["id"],
                    "variant_index": variant_index,
                    "system": PLANNER_SYSTEM_PROMPT,
                    "prompt": build_prompt("grpo", payload),
                    "payload": payload,
                    "source_question": record["question"],
                    "evidence_items": evidence_items
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
