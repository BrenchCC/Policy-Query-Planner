import os
import sys
import logging
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Callable

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import (
    read_json,
    read_jsonl,
    sha256_file,
    normalized_key,
    stable_record_hash
)
from data_preprocess.config import (
    RAW_ROOT,
    INTERIM_ROOT,
    PROCESSED_ROOT,
    REQUEST_ROOT,
    TARGET_COUNTS,
    GRPO_HOP_MINIMUM,
    GRPO_SINGLE_COUNT,
    DPO_SINGLE_SOURCE_QUOTAS,
    DPO_MULTIHOP_HOP_QUOTAS,
    EXPECTED_QRECC_COUNTS,
    QRECC_ARCHIVE_SHA256,
    MULTIHOP_COLD_START_HOP_QUOTAS,
    EXPECTED_CONDITIONALQA_COUNTS,
    EVIDENCE_MAPPING_THRESHOLD
)
from data_preprocess.schemas import (
    validate_sft_record,
    validate_dpo_record,
    validate_multihop_sft_record,
    validate_grpo_record,
    validate_knowledge_record
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Validate all project data artifacts")
    parser.add_argument(
        "--stage",
        choices = ["raw", "base", "train", "api", "all"],
        default = "all"
    )
    return parser.parse_args()


def assert_unique_ids(records: list[dict[str, Any]], label: str) -> None:
    """Assert that record IDs are unique.

    Args:
        records: Records containing an id field.
        label: Human-readable dataset label.

    Raises:
        ValueError: If any duplicate ID exists.
    """
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs detected in {label}")


def validate_raw() -> None:
    """Validate raw downloads, checksums, and source counts."""
    manifest = read_json(RAW_ROOT / "download_manifest.json")
    for entry in manifest["files"]:
        path = Path(__file__).resolve().parents[1] / entry["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
    qrecc_archive = RAW_ROOT / "qrecc" / "archives" / "qrecc_data.zip"
    if sha256_file(qrecc_archive) != QRECC_ARCHIVE_SHA256:
        raise ValueError("QReCC pinned checksum mismatch")

    documents = read_json(RAW_ROOT / "conditionalqa" / "documents.json")
    if len(documents) != EXPECTED_CONDITIONALQA_COUNTS["documents"]:
        raise ValueError("Unexpected ConditionalQA document count")
    for split in ["train", "dev", "test_no_answer"]:
        records = read_json(RAW_ROOT / "conditionalqa" / f"{split}.json")
        if len(records) != EXPECTED_CONDITIONALQA_COUNTS[split]:
            raise ValueError(f"Unexpected ConditionalQA {split} count")
    for split in ["train", "test"]:
        records = read_json(RAW_ROOT / "qrecc" / "original" / f"qrecc_{split}.json")
        if len(records) != EXPECTED_QRECC_COUNTS[split]:
            raise ValueError(f"Unexpected QReCC {split} count")
    logger.info("Raw source validation passed")


def validate_base() -> None:
    """Validate knowledge bases, cleaning coverage, and official benchmarks."""
    for file_name in ["policy.jsonl", "musique_aux.jsonl"]:
        records = read_jsonl(PROCESSED_ROOT / "knowledge_base" / file_name)
        assert_unique_ids(records, file_name)
        for record in records:
            validate_knowledge_record(record)
    summary = read_json(INTERIM_ROOT / "cleaning_summary.json")
    mapping_rate = summary["conditionalqa"]["evidence_mapping_rate"]
    if mapping_rate < EVIDENCE_MAPPING_THRESHOLD:
        raise ValueError(f"Evidence mapping rate below threshold: {mapping_rate:.4f}")

    expected_eval_counts = {
        "conditionalqa_dev.jsonl": 285,
        "conditionalqa_test_blind.jsonl": 804,
        "qrecc_test.jsonl": 16451,
        "musique_dev.jsonl": 2417
    }
    for file_name, expected_count in expected_eval_counts.items():
        records = read_jsonl(PROCESSED_ROOT / "eval" / file_name)
        if len(records) != expected_count:
            raise ValueError(f"Unexpected benchmark count for {file_name}")
        assert_unique_ids(records, file_name)
    logger.info("Knowledge-base and benchmark validation passed")


def validate_records(
    path: Path,
    expected_count: int,
    validator: Callable[[dict[str, Any]], None]
) -> list[dict[str, Any]]:
    """Validate count, IDs, and schema for one JSONL dataset.

    Args:
        path: Dataset JSONL path.
        expected_count: Exact expected record count.
        validator: Record-level validation function.

    Returns:
        Validated records.
    """
    records = read_jsonl(path)
    if len(records) != expected_count:
        raise ValueError(f"Unexpected count for {path}: {len(records)}")
    assert_unique_ids(records, path.name)
    for record in records:
        validator(record)
        if "reward" in record or "reward_weights" in record:
            raise ValueError(f"Reward design leaked into data: {record['id']}")
    return records


def validate_train() -> None:
    """Validate train schemas, exact counts, distributions, and split leakage."""
    sft_records = validate_records(
        PROCESSED_ROOT / "train" / "sft_train.jsonl",
        TARGET_COUNTS["sft"],
        validate_sft_record
    )
    cold_start_records = validate_records(
        PROCESSED_ROOT / "train" / "sft_multihop_cold_start.jsonl",
        TARGET_COUNTS["sft_multihop_cold_start"],
        validate_multihop_sft_record
    )
    dpo_records = validate_records(
        PROCESSED_ROOT / "train" / "dpo_train.jsonl",
        TARGET_COUNTS["dpo"],
        validate_dpo_record
    )
    grpo_records = validate_records(
        PROCESSED_ROOT / "train" / "grpo_train.jsonl",
        TARGET_COUNTS["grpo"],
        validate_grpo_record
    )

    held_out_ids = set()
    for file_name in [
        "conditionalqa_dev.jsonl",
        "conditionalqa_test_blind.jsonl",
        "qrecc_test.jsonl",
        "musique_dev.jsonl"
    ]:
        held_out_ids.update(
            record["id"]
            for record in read_jsonl(PROCESSED_ROOT / "eval" / file_name)
        )
    for record in sft_records + cold_start_records + dpo_records + grpo_records:
        if record["source_id"] in held_out_ids:
            raise ValueError(f"Benchmark leakage detected: {record['id']}")

    qrecc_train = {
        record["id"]: record
        for record in read_jsonl(INTERIM_ROOT / "qrecc_train.jsonl")
    }
    qrecc_test_keys = {
        (
            normalized_key(record["question"]),
            normalized_key(record["rewrite"]),
            normalized_key(record["answer"])
        )
        for record in read_jsonl(INTERIM_ROOT / "qrecc_test.jsonl")
    }
    for record in sft_records + dpo_records:
        if record["source_dataset"] != "qrecc":
            continue
        source = qrecc_train[record["source_id"]]
        content_key = (
            normalized_key(source["question"]),
            normalized_key(source["rewrite"]),
            normalized_key(source["answer"])
        )
        if content_key in qrecc_test_keys:
            raise ValueError(f"QReCC content leakage detected: {record['id']}")

    musique_train = {
        record["id"]: record
        for record in read_jsonl(INTERIM_ROOT / "musique_train.jsonl")
    }
    musique_dev_question_keys = {
        normalized_key(record["question"])
        for record in read_jsonl(INTERIM_ROOT / "musique_dev.jsonl")
    }
    multihop_dpo_records = [
        record
        for record in dpo_records
        if record["task_type"] == "multi_hop"
    ]
    for record in cold_start_records + multihop_dpo_records + grpo_records:
        if record["source_dataset"] != "musique":
            continue
        if record["source_id"] not in musique_train:
            raise ValueError(f"MuSiQue source is not in train: {record['id']}")
        question_key = normalized_key(musique_train[record["source_id"]]["question"])
        if question_key in musique_dev_question_keys:
            raise ValueError(f"MuSiQue content leakage detected: {record['id']}")

    cold_source_ids = {record["source_id"] for record in cold_start_records}
    dpo_multihop_source_ids = {record["source_id"] for record in multihop_dpo_records}
    grpo_source_ids = {
        record["source_id"]
        for record in grpo_records
        if record["task_type"] == "multi_hop"
    }
    if (
        cold_source_ids.intersection(dpo_multihop_source_ids)
        or cold_source_ids.intersection(grpo_source_ids)
        or dpo_multihop_source_ids.intersection(grpo_source_ids)
    ):
        raise ValueError("Multi-hop SFT, DPO, and GRPO source IDs overlap")
    cold_question_keys = {
        normalized_key(musique_train[source_id]["question"])
        for source_id in cold_source_ids
    }
    grpo_question_keys = {
        normalized_key(musique_train[source_id]["question"])
        for source_id in grpo_source_ids
    }
    dpo_multihop_question_keys = {
        normalized_key(musique_train[source_id]["question"])
        for source_id in dpo_multihop_source_ids
    }
    if (
        cold_question_keys.intersection(dpo_multihop_question_keys)
        or cold_question_keys.intersection(grpo_question_keys)
        or dpo_multihop_question_keys.intersection(grpo_question_keys)
    ):
        raise ValueError("Multi-hop SFT, DPO, and GRPO question fingerprints overlap")

    cold_hop_counts = {}
    for record in cold_start_records:
        hop_count = record["hop_count"]
        cold_hop_counts[hop_count] = cold_hop_counts.get(hop_count, 0) + 1
    if cold_hop_counts != MULTIHOP_COLD_START_HOP_QUOTAS:
        raise ValueError(f"Unexpected cold-start hop distribution: {cold_hop_counts}")
    dpo_task_counts = Counter(record["task_type"] for record in dpo_records)
    if dpo_task_counts != {"single_hop": 2500, "multi_hop": 2500}:
        raise ValueError(f"Unexpected DPO task distribution: {dpo_task_counts}")
    dpo_multihop_hop_counts = Counter(record["hop_count"] for record in multihop_dpo_records)
    if dpo_multihop_hop_counts != DPO_MULTIHOP_HOP_QUOTAS:
        raise ValueError(f"Unexpected multi-hop DPO distribution: {dpo_multihop_hop_counts}")
    dpo_single_source_counts = Counter(
        record["source_dataset"]
        for record in dpo_records
        if record["task_type"] == "single_hop"
    )
    if dpo_single_source_counts != DPO_SINGLE_SOURCE_QUOTAS:
        raise ValueError(f"Unexpected single-hop DPO sources: {dpo_single_source_counts}")

    grpo_task_counts = Counter(record["task_type"] for record in grpo_records)
    expected_grpo_tasks = {
        "single_hop": GRPO_SINGLE_COUNT,
        "multi_hop": TARGET_COUNTS["grpo"] - GRPO_SINGLE_COUNT
    }
    if grpo_task_counts != expected_grpo_tasks:
        raise ValueError(f"Unexpected GRPO task distribution: {grpo_task_counts}")
    grpo_hop_counts = {}
    for record in grpo_records:
        hop_count = record["hop_count"]
        grpo_hop_counts[hop_count] = grpo_hop_counts.get(hop_count, 0) + 1
    below_minimum = any(
        grpo_hop_counts.get(hop_count, 0) < GRPO_HOP_MINIMUM
        for hop_count in [2, 3, 4]
    )
    if below_minimum:
        raise ValueError(f"GRPO hop minimum is not satisfied: {grpo_hop_counts}")

    error_counts = {}
    for record in dpo_records:
        error_counts[record["error_type"]] = error_counts.get(record["error_type"], 0) + 1
    if len(error_counts) != 10 or set(error_counts.values()) != {500}:
        raise ValueError(f"DPO error categories are not balanced: {error_counts}")

    dpo_policy_ids = {
        record["source_id"]
        for record in dpo_records
        if record["source_dataset"] == "conditionalqa"
    }
    grpo_single_policy_ids = {
        record["source_id"]
        for record in grpo_records
        if record["task_type"] == "single_hop"
    }
    if dpo_policy_ids.intersection(grpo_single_policy_ids):
        raise ValueError("Single-hop DPO and GRPO policy sources overlap")
    dataset_info_path = PROCESSED_ROOT / "dataset_info.json"
    if not dataset_info_path.exists():
        raise FileNotFoundError("Missing LLaMA-Factory dataset_info.json")
    dataset_info = read_json(dataset_info_path)
    if "policy_query_multihop_sft_cold_start" not in dataset_info:
        raise ValueError("Missing multi-hop cold-start dataset registration")

    augmented_path = PROCESSED_ROOT / "train" / "grpo_train_domain_augmented.jsonl"
    if augmented_path.exists():
        augmented_records = validate_records(
            augmented_path,
            TARGET_COUNTS["grpo"],
            validate_grpo_record
        )
        augmented_musique = [
            record
            for record in augmented_records
            if record["namespace"] == "musique_aux"
        ]
        augmented_policy = [
            record
            for record in augmented_records
            if record["namespace"] == "policy"
        ]
        if len(augmented_musique) != 3000 or len(augmented_policy) != 2000:
            raise ValueError("Unexpected domain-augmented GRPO mixture")
        augmented_source_ids = {record["source_id"] for record in augmented_musique}
        if not augmented_source_ids.issubset(grpo_source_ids):
            raise ValueError("Domain-augmented MuSiQue records are outside the GRPO pool")
        if augmented_source_ids.intersection(cold_source_ids):
            raise ValueError("Domain-augmented GRPO overlaps cold-start data")
    logger.info("Training data validation passed")


def validate_api() -> None:
    """Validate optional generation request queues."""
    expected_counts = {"sft": 2338, "dpo": 3000, "grpo": 1811}
    for stage, expected_count in expected_counts.items():
        records = read_jsonl(REQUEST_ROOT / f"{stage}_requests.jsonl")
        if len(records) != expected_count:
            raise ValueError(f"Unexpected {stage} request count: {len(records)}")
        assert_unique_ids(records, f"{stage} requests")
        for record in records:
            if not record["prompt"].strip() or not record["system"].strip():
                raise ValueError(f"Empty API prompt: {record['id']}")
            if record.get("request_hash") != stable_record_hash(record):
                raise ValueError(f"Invalid API request hash: {record['id']}")
    logger.info("Optional API request validation passed")


def main() -> None:
    """Run selected validation stages."""
    args = parse_args()
    functions = {
        "raw": validate_raw,
        "base": validate_base,
        "train": validate_train,
        "api": validate_api
    }
    stages = list(functions) if args.stage == "all" else [args.stage]
    for stage in stages:
        logger.info("-" * 60)
        logger.info("Validating stage: %s", stage)
        logger.info("-" * 60)
        functions[stage]()
    logger.info("All selected validations passed")


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
