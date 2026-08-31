import os
import re
import sys
import json
import math
import random
import logging
import argparse
from collections import Counter, defaultdict
from typing import Any, Callable, Hashable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import (
    read_jsonl,
    write_json,
    write_jsonl,
    normalize_text,
    normalized_key
)
from data_preprocess.config import (
    INTERIM_ROOT,
    PROCESSED_ROOT,
    RANDOM_SEED,
    TARGET_COUNTS,
    GRPO_HOP_MINIMUM,
    GRPO_SINGLE_COUNT,
    GRPO_MULTIHOP_COUNT,
    DPO_SINGLE_SOURCE_QUOTAS,
    DPO_MULTIHOP_HOP_QUOTAS,
    MULTIHOP_COLD_START_HOP_QUOTAS,
    PLANNER_SYSTEM_PROMPT
)
from data_preprocess.schemas import (
    serialize_plan,
    validate_sft_record,
    validate_dpo_record,
    validate_multihop_sft_record,
    validate_grpo_record
)

logger = logging.getLogger(__name__)

SFT_INSTRUCTION = (
    "Rewrite the current question as a standalone retrieval query. Return a JSON query plan only "
    "and do not answer the question."
)
GRPO_INSTRUCTION = (
    "Create the minimum sufficient retrieval query plan. Use one standalone query when it is "
    "sufficient; otherwise use ordered queries with explicit dependencies. Return JSON only and "
    "do not answer the question."
)
DPO_INSTRUCTION = GRPO_INSTRUCTION
ERROR_TYPES = [
    "unresolved_reference",
    "entity_omission",
    "constraint_omission",
    "overly_broad",
    "wrong_context"
]
MULTIHOP_ERROR_TYPES = [
    "step_omission",
    "broken_dependency",
    "redundant_step",
    "overly_broad_step",
    "relation_omission"
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Build final training and evaluation datasets")
    parser.add_argument("--force", action = "store_true", help = "Replace existing outputs")
    return parser.parse_args()


def stratified_sample(
    records: list[dict[str, Any]],
    target_count: int,
    key_function: Callable[[dict[str, Any]], Hashable],
    seed: int
) -> list[dict[str, Any]]:
    """Sample records while preserving a categorical distribution.

    Args:
        records: Candidate records.
        target_count: Exact number of requested records.
        key_function: Function mapping a record to a stratum.
        seed: Deterministic random seed.

    Returns:
        Deterministically sampled records.

    Raises:
        ValueError: If the candidate pool is too small.
    """
    if target_count > len(records):
        raise ValueError(f"Cannot sample {target_count} records from {len(records)} candidates")
    random_generator = random.Random(seed)
    groups: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key_function(record)].append(record)
    for group in groups.values():
        random_generator.shuffle(group)

    total = len(records)
    allocations = {}
    fractions = []
    for key, group in groups.items():
        exact = target_count * len(group) / total
        base = min(len(group), math.floor(exact))
        allocations[key] = base
        fractions.append((exact - base, str(key), key))
    remaining = target_count - sum(allocations.values())
    fractions.sort(reverse = True)
    while remaining:
        progress = False
        for _, _, key in fractions:
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                remaining -= 1
                progress = True
                if remaining == 0:
                    break
        if not progress:
            raise ValueError("Unable to complete stratified allocation")

    sampled = []
    for key in sorted(groups, key = str):
        sampled.extend(groups[key][:allocations[key]])
    random_generator.shuffle(sampled)
    return sampled


def format_qrecc_input(record: dict[str, Any]) -> str:
    """Format a QReCC record as an Alpaca input.

    Args:
        record: Clean QReCC record.

    Returns:
        Conversation history and current question.
    """
    if record["history_text"]:
        return (
            f"Conversation history:\n{record['history_text']}\n\n"
            f"Current question:\n{record['question']}"
        )
    return f"Current question:\n{record['question']}"


def qrecc_content_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Build a split-independent QReCC leakage fingerprint.

    Args:
        record: Clean QReCC record.

    Returns:
        Normalized question, rewrite, and answer tuple.
    """
    return (
        normalized_key(record["question"]),
        normalized_key(record["rewrite"]),
        normalized_key(record["answer"])
    )


def single_query_plan(query: str) -> str:
    """Create a serialized one-query planner response.

    Args:
        query: Standalone retrieval query.

    Returns:
        Serialized planner response.
    """
    return serialize_plan(
        [
            {
                "id": "q1",
                "query": normalize_text(query),
                "depends_on": []
            }
        ]
    )


def fit_policy_keyword_model(
    records: list[dict[str, Any]]
) -> tuple[TfidfVectorizer, Any]:
    """Fit a deterministic keyword extractor over policy inputs.

    Args:
        records: ConditionalQA training records.

    Returns:
        Fitted vectorizer and sparse scenario matrix.
    """
    scenario_texts = [record["scenario"] for record in records]
    vectorizer = TfidfVectorizer(
        stop_words = "english",
        ngram_range = (1, 2),
        max_features = 12000,
        min_df = 2
    )
    matrix = vectorizer.fit_transform(scenario_texts)
    return vectorizer, matrix


def build_policy_query(
    record: dict[str, Any],
    vectorizer: TfidfVectorizer,
    row: Any
) -> str:
    """Construct an answer-free policy retrieval query.

    Args:
        record: ConditionalQA training record.
        vectorizer: Fitted policy TF-IDF vectorizer.
        row: Sparse TF-IDF row for the record scenario.

    Returns:
        Compact retrieval query containing prompt-visible constraints.
    """
    feature_names = vectorizer.get_feature_names_out()
    if row.nnz:
        ranked_positions = np.argsort(row.data)[::-1][:6]
        keywords = [feature_names[row.indices[position]] for position in ranked_positions]
    else:
        keywords = []
    query_parts = [record["title"], record["question"]] + keywords
    words = normalize_text(" ".join(query_parts)).split()
    return " ".join(words[:45])


def build_sft_records(
    qrecc_train: list[dict[str, Any]],
    conditional_train: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build the exact 20K SFT mixture.

    Args:
        qrecc_train: Clean QReCC training records.
        conditional_train: Clean ConditionalQA training records.

    Returns:
        SFT records and policy record ID to chosen plan mapping.
    """
    nontrivial = [record for record in qrecc_train if record["nontrivial_rewrite"]]
    no_op = [record for record in qrecc_train if not record["nontrivial_rewrite"]]
    selected_nontrivial = stratified_sample(
        nontrivial,
        14130,
        lambda record: (record["conversation_source"], record["context_turns"]),
        RANDOM_SEED
    )
    selected_no_op = stratified_sample(
        no_op,
        3532,
        lambda record: (record["conversation_source"], record["context_turns"]),
        RANDOM_SEED + 1
    )
    sft_records = []
    for source_type, selected_records in [
        ("qrecc_nontrivial", selected_nontrivial),
        ("qrecc_no_op", selected_no_op)
    ]:
        for record in selected_records:
            sft_record = {
                "id": "sft_" + record["id"],
                "instruction": SFT_INSTRUCTION,
                "input": format_qrecc_input(record),
                "output": single_query_plan(record["rewrite"]),
                "system": PLANNER_SYSTEM_PROMPT,
                "source_dataset": "qrecc",
                "source_id": record["id"],
                "sample_type": source_type
            }
            validate_sft_record(sft_record)
            sft_records.append(sft_record)

    vectorizer, matrix = fit_policy_keyword_model(conditional_train)
    policy_plans = {}
    for index, record in enumerate(conditional_train):
        query = build_policy_query(record, vectorizer, matrix[index])
        plan = single_query_plan(query)
        policy_plans[record["id"]] = plan
        sft_record = {
            "id": "sft_conditionalqa_" + record["id"],
            "instruction": SFT_INSTRUCTION,
            "input": f"Scenario:\n{record['scenario']}\n\nQuestion:\n{record['question']}",
            "output": plan,
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "conditionalqa",
            "source_id": record["id"],
            "sample_type": "policy_domain"
        }
        validate_sft_record(sft_record)
        sft_records.append(sft_record)

    random.Random(RANDOM_SEED).shuffle(sft_records)
    if len(sft_records) != TARGET_COUNTS["sft"]:
        raise ValueError(f"Unexpected SFT count: {len(sft_records)}")
    return sft_records, policy_plans


def remove_named_entity(query: str) -> str:
    """Remove a likely entity from a query.

    Args:
        query: Correct retrieval query.

    Returns:
        Query with a central-looking token removed.
    """
    words = query.split()
    candidates = [
        index
        for index, word in enumerate(words)
        if index > 0 and word[:1].isupper() and len(re.sub(r"\W", "", word)) > 2
    ]
    if not candidates:
        candidates = sorted(range(len(words)), key = lambda index: len(words[index]), reverse = True)
    if candidates and len(words) > 3:
        words.pop(candidates[0])
    return " ".join(words)


def make_rejected_query(
    record: dict[str, Any],
    chosen_query: str,
    error_type: str,
    wrong_context_query: str
) -> str:
    """Create a deterministic difficult negative query.

    Args:
        record: Source QReCC or ConditionalQA record.
        chosen_query: Correct standalone query.
        error_type: Required negative category.
        wrong_context_query: Plausible query from an incorrect context.

    Returns:
        Rejected retrieval query.
    """
    words = chosen_query.split()
    if error_type == "unresolved_reference":
        rejected = record.get("question", chosen_query)
    elif error_type == "entity_omission":
        rejected = remove_named_entity(chosen_query)
    elif error_type == "constraint_omission":
        without_numbers = re.sub(r"\b\d+(?:\.\d+)?\b", "", chosen_query)
        rejected_words = normalize_text(without_numbers).split()
        if rejected_words == words or len(rejected_words) < 3:
            keep_count = max(3, math.ceil(len(words) * 0.7))
            rejected_words = words[:keep_count]
        rejected = " ".join(rejected_words)
    elif error_type == "overly_broad":
        title = record.get("title", "")
        rejected = f"General information about {title}" if title else "General background information"
    elif error_type == "wrong_context":
        rejected = wrong_context_query
    else:
        raise ValueError(f"Unsupported DPO error type: {error_type}")
    rejected = normalize_text(rejected)
    if len(rejected) < 3 or rejected == normalize_text(chosen_query):
        rejected = normalize_text("General information " + " ".join(words[:3]))
    return rejected


def build_dpo_records(
    qrecc_train: list[dict[str, Any]],
    conditional_train: list[dict[str, Any]],
    policy_plans: dict[str, str],
    multihop_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the exact 5K single-hop and multi-hop preference mixture.

    Args:
        qrecc_train: Clean QReCC training records.
        conditional_train: Clean ConditionalQA training records.
        policy_plans: Correct ConditionalQA plan mapping.
        multihop_records: Disjoint MuSiQue records for multi-hop preference learning.

    Returns:
        Validated DPO records.
    """
    qrecc_candidates = [record for record in qrecc_train if record["nontrivial_rewrite"]]
    selected_qrecc = stratified_sample(
        qrecc_candidates,
        DPO_SINGLE_SOURCE_QUOTAS["qrecc"],
        lambda record: (record["conversation_source"], record["context_turns"]),
        RANDOM_SEED + 2
    )
    selected_policy = stratified_sample(
        conditional_train,
        DPO_SINGLE_SOURCE_QUOTAS["conditionalqa"],
        lambda record: (record["not_answerable"], min(len(record["evidences"]), 3)),
        RANDOM_SEED + 3
    )

    dpo_records = []
    for index, record in enumerate(selected_qrecc):
        error_type = ERROR_TYPES[index % len(ERROR_TYPES)]
        chosen_query = record["rewrite"]
        prior_question = record["context"][-2] if len(record["context"]) >= 2 else "related topic"
        rejected_query = make_rejected_query(record, chosen_query, error_type, prior_question)
        dpo_record = {
            "id": "dpo_" + record["id"],
            "instruction": DPO_INSTRUCTION,
            "input": format_qrecc_input(record),
            "chosen": single_query_plan(chosen_query),
            "rejected": single_query_plan(rejected_query),
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "qrecc",
            "source_id": record["id"],
            "task_type": "single_hop",
            "hop_count": 1,
            "error_type": error_type
        }
        validate_dpo_record(dpo_record)
        dpo_records.append(dpo_record)

    policy_titles = [record["title"] for record in selected_policy]
    for index, record in enumerate(selected_policy):
        error_type = ERROR_TYPES[index % len(ERROR_TYPES)]
        chosen_plan = policy_plans[record["id"]]
        chosen_query = json_query(chosen_plan)
        wrong_title = policy_titles[(index + 1) % len(policy_titles)]
        rejected_query = make_rejected_query(
            record,
            chosen_query,
            error_type,
            f"{wrong_title} {record['question']}"
        )
        dpo_record = {
            "id": "dpo_conditionalqa_" + record["id"],
            "instruction": DPO_INSTRUCTION,
            "input": f"Scenario:\n{record['scenario']}\n\nQuestion:\n{record['question']}",
            "chosen": chosen_plan,
            "rejected": single_query_plan(rejected_query),
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "conditionalqa",
            "source_id": record["id"],
            "task_type": "single_hop",
            "hop_count": 1,
            "error_type": error_type
        }
        validate_dpo_record(dpo_record)
        dpo_records.append(dpo_record)

    for index, record in enumerate(multihop_records):
        error_type = MULTIHOP_ERROR_TYPES[index % len(MULTIHOP_ERROR_TYPES)]
        chosen_plan = convert_musique_plan(record)
        dpo_record = {
            "id": "dpo_musique_" + record["id"],
            "instruction": DPO_INSTRUCTION,
            "input": f"Question:\n{record['question']}",
            "chosen": chosen_plan,
            "rejected": make_rejected_multihop_plan(chosen_plan, error_type),
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "musique",
            "source_id": record["id"],
            "task_type": "multi_hop",
            "namespace": "musique_aux",
            "hop_count": record["hop_count"],
            "error_type": error_type
        }
        validate_dpo_record(dpo_record)
        dpo_records.append(dpo_record)

    random.Random(RANDOM_SEED + 4).shuffle(dpo_records)
    if len(dpo_records) != TARGET_COUNTS["dpo"]:
        raise ValueError(f"Unexpected DPO count: {len(dpo_records)}")
    return dpo_records


def make_rejected_multihop_plan(chosen_plan: str, error_type: str) -> str:
    """Create one structurally valid but weaker multi-hop plan.

    Args:
        chosen_plan: Correct dependency-aware plan.
        error_type: Requested multi-hop planning error.

    Returns:
        Serialized rejected plan with one primary planning defect.
    """
    queries = json.loads(chosen_plan)["queries"]
    rejected_queries = [dict(query) for query in queries]
    if error_type == "step_omission":
        rejected_queries = rejected_queries[:-1]
    elif error_type == "broken_dependency":
        target = next(
            (query for query in rejected_queries if query["depends_on"]),
            rejected_queries[-1]
        )
        target["query"] = re.sub(
            r"\{\{q[1-4]\.answer\}\}",
            "the intermediate entity",
            target["query"]
        )
        target["depends_on"] = []
    elif error_type == "redundant_step":
        rejected_queries[-1] = {
            "id": rejected_queries[-1]["id"],
            "query": rejected_queries[0]["query"],
            "depends_on": []
        }
    elif error_type == "overly_broad_step":
        rejected_queries[-1] = {
            "id": rejected_queries[-1]["id"],
            "query": "General background information about the topic",
            "depends_on": []
        }
    elif error_type == "relation_omission":
        target = rejected_queries[-1]
        plain_query = re.sub(r"\{\{q[1-4]\.answer\}\}", "the entity", target["query"])
        words = plain_query.split()
        target["query"] = " ".join(words[:max(3, math.ceil(len(words) * 0.6))])
        target["depends_on"] = []
    else:
        raise ValueError(f"Unsupported multi-hop DPO error type: {error_type}")
    rejected_plan = serialize_plan(rejected_queries)
    if rejected_plan == chosen_plan:
        raise ValueError("Multi-hop rejected plan must differ from the chosen plan")
    return rejected_plan


def json_query(plan: str) -> str:
    """Read the first query string from a serialized plan.

    Args:
        plan: Serialized one-query planner response.

    Returns:
        Retrieval query text.
    """
    return json.loads(plan)["queries"][0]["query"]


def musique_question_key(record: dict[str, Any]) -> str:
    """Build a normalized MuSiQue question fingerprint.

    Args:
        record: Clean MuSiQue record.

    Returns:
        Normalized question fingerprint.
    """
    return normalized_key(record["question"])


def proportional_allocations(
    groups: dict[int, list[dict[str, Any]]],
    target_count: int,
    minimum_per_hop: int
) -> dict[int, int]:
    """Allocate a target across hop groups with minimum coverage.

    Args:
        groups: Remaining records grouped by hop count.
        target_count: Exact number of records to allocate.
        minimum_per_hop: Minimum records required from every hop group.

    Returns:
        Exact per-hop allocation.

    Raises:
        ValueError: If the groups cannot satisfy the requested allocation.
    """
    hop_counts = [2, 3, 4]
    minimum_total = minimum_per_hop * len(hop_counts)
    if target_count < minimum_total:
        raise ValueError("Target count is smaller than the per-hop minimum total")
    for hop_count in hop_counts:
        if len(groups[hop_count]) < minimum_per_hop:
            raise ValueError(f"Insufficient {hop_count}-hop records for minimum allocation")

    allocations = {hop_count: minimum_per_hop for hop_count in hop_counts}
    remaining_target = target_count - minimum_total
    capacities = {
        hop_count: len(groups[hop_count]) - minimum_per_hop
        for hop_count in hop_counts
    }
    total_capacity = sum(capacities.values())
    if remaining_target > total_capacity:
        raise ValueError("Insufficient records for proportional allocation")
    if remaining_target == 0:
        return allocations

    fractions = []
    for hop_count in hop_counts:
        exact = remaining_target * capacities[hop_count] / total_capacity
        base = min(capacities[hop_count], math.floor(exact))
        allocations[hop_count] += base
        fractions.append((exact - base, hop_count))
    unallocated = target_count - sum(allocations.values())
    fractions.sort(key = lambda item: (-item[0], item[1]))
    while unallocated:
        progress = False
        for _, hop_count in fractions:
            if allocations[hop_count] < len(groups[hop_count]):
                allocations[hop_count] += 1
                unallocated -= 1
                progress = True
                if unallocated == 0:
                    break
        if not progress:
            raise ValueError("Unable to finish proportional allocation")
    return allocations


def allocate_multihop_records(
    records: list[dict[str, Any]],
    cold_start_quotas: dict[int, int],
    dpo_quotas: dict[int, int],
    grpo_target_count: int,
    seed: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, int]
]:
    """Jointly allocate disjoint SFT, DPO, and GRPO MuSiQue records.

    Args:
        records: MuSiQue training records.
        cold_start_quotas: Exact 2/3/4-hop cold-start quotas.
        dpo_quotas: Exact 2/3/4-hop DPO quotas.
        grpo_target_count: Exact requested GRPO count.
        seed: Deterministic sampling seed.

    Returns:
        Cold-start records, DPO records, GRPO records, and GRPO hop allocations.

    Raises:
        ValueError: If inputs are duplicated or capacity is insufficient.
    """
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids = set()
    seen_question_keys = set()
    for record in records:
        if record["hop_count"] not in {2, 3, 4}:
            continue
        source_id = record["id"]
        question_key = musique_question_key(record)
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate MuSiQue source ID: {source_id}")
        seen_source_ids.add(source_id)
        if question_key in seen_question_keys:
            continue
        seen_question_keys.add(question_key)
        groups[record["hop_count"]].append(record)

    cold_start_records = []
    dpo_records = []
    remaining_groups: dict[int, list[dict[str, Any]]] = {}
    for hop_count in [2, 3, 4]:
        random.Random(seed + hop_count).shuffle(groups[hop_count])
        cold_count = cold_start_quotas[hop_count]
        dpo_count = dpo_quotas[hop_count]
        required_count = cold_count + dpo_count + GRPO_HOP_MINIMUM
        if len(groups[hop_count]) < required_count:
            raise ValueError(f"Insufficient {hop_count}-hop MuSiQue records")
        cold_start_records.extend(groups[hop_count][:cold_count])
        dpo_records.extend(groups[hop_count][cold_count:cold_count + dpo_count])
        remaining_groups[hop_count] = groups[hop_count][cold_count + dpo_count:]

    grpo_allocations = proportional_allocations(
        remaining_groups,
        grpo_target_count,
        GRPO_HOP_MINIMUM
    )
    grpo_records = []
    for hop_count in [2, 3, 4]:
        grpo_records.extend(remaining_groups[hop_count][:grpo_allocations[hop_count]])

    random.Random(seed + 10).shuffle(cold_start_records)
    random.Random(seed + 11).shuffle(dpo_records)
    random.Random(seed + 12).shuffle(grpo_records)
    cold_source_ids = {record["id"] for record in cold_start_records}
    dpo_source_ids = {record["id"] for record in dpo_records}
    grpo_source_ids = {record["id"] for record in grpo_records}
    if (
        cold_source_ids.intersection(dpo_source_ids)
        or cold_source_ids.intersection(grpo_source_ids)
        or dpo_source_ids.intersection(grpo_source_ids)
    ):
        raise ValueError("SFT, DPO, and GRPO multi-hop source IDs overlap")
    cold_question_keys = {musique_question_key(record) for record in cold_start_records}
    dpo_question_keys = {musique_question_key(record) for record in dpo_records}
    grpo_question_keys = {musique_question_key(record) for record in grpo_records}
    if (
        cold_question_keys.intersection(dpo_question_keys)
        or cold_question_keys.intersection(grpo_question_keys)
        or dpo_question_keys.intersection(grpo_question_keys)
    ):
        raise ValueError("SFT, DPO, and GRPO multi-hop question fingerprints overlap")
    return cold_start_records, dpo_records, grpo_records, grpo_allocations


def convert_musique_plan(record: dict[str, Any]) -> str:
    """Convert MuSiQue decomposition notation to planner JSON.

    Args:
        record: Clean MuSiQue record.

    Returns:
        Serialized dependency-aware query plan.
    """
    queries = []
    for step in record["question_decomposition"]:
        step_number = int(step["step"])
        query = step["question"].replace(">>", " ")
        # Only earlier step numbers are dependencies; titles such as "#9 Dream" stay literal.
        # 仅将前序步骤编号视为依赖，避免把作品名中的 #9 误解析为 q9。
        references = sorted(
            {
                int(value)
                for value in re.findall(r"#(\d+)", query)
                if 1 <= int(value) < step_number
            }
        )
        dependencies = []
        for reference in references:
            dependency = f"q{reference}"
            dependencies.append(dependency)
            query = query.replace(f"#{reference}", "{{" + dependency + ".answer}}")
        queries.append(
            {
                "id": f"q{step_number}",
                "query": normalize_text(query),
                "depends_on": dependencies
            }
        )
    return serialize_plan(queries)


def build_multihop_sft_records(
    selected_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build supervised multi-hop cold-start records.

    Args:
        selected_records: Disjoint MuSiQue records selected for cold-start.

    Returns:
        Validated multi-hop SFT cold-start records.
    """
    cold_start_records = []
    for record in selected_records:
        cold_start_record = {
            "id": "sft_musique_cold_start_" + record["id"],
            "instruction": GRPO_INSTRUCTION,
            "input": f"Question:\n{record['question']}",
            "output": convert_musique_plan(record),
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "musique",
            "source_id": record["id"],
            "sample_type": "musique_multihop_cold_start",
            "namespace": "musique_aux",
            "hop_count": record["hop_count"]
        }
        validate_multihop_sft_record(cold_start_record)
        cold_start_records.append(cold_start_record)
    if len(cold_start_records) != TARGET_COUNTS["sft_multihop_cold_start"]:
        raise ValueError(f"Unexpected multi-hop SFT count: {len(cold_start_records)}")
    return cold_start_records


def flatten_reference_answers(answers: list[Any]) -> str:
    """Flatten nested ConditionalQA answers into one reference string.

    Args:
        answers: Nested answer annotations.

    Returns:
        Deduplicated semicolon-separated reference answers.
    """
    values = []
    for answer in answers:
        if isinstance(answer, list) and answer:
            value = normalize_text(str(answer[0]))
            if value and value not in values:
                values.append(value)
    return "; ".join(values) or "not answerable"


def select_grpo_single_records(
    conditional_train: list[dict[str, Any]],
    excluded_source_ids: set[str]
) -> list[dict[str, Any]]:
    """Select single-hop policy regularizers outside the DPO policy subset.

    Args:
        conditional_train: Clean ConditionalQA training records.
        excluded_source_ids: Source IDs already used for policy DPO.

    Returns:
        Deterministically stratified single-hop GRPO source records.
    """
    candidates = [
        record
        for record in conditional_train
        if record["id"] not in excluded_source_ids and record["gold_doc_ids"]
    ]
    return stratified_sample(
        candidates,
        GRPO_SINGLE_COUNT,
        lambda record: (record["not_answerable"], min(len(record["evidences"]), 3)),
        RANDOM_SEED + 13
    )


def build_grpo_records(
    selected_records: list[dict[str, Any]],
    single_policy_records: list[dict[str, Any]],
    policy_plans: dict[str, str]
) -> list[dict[str, Any]]:
    """Build a 1K single-hop plus 4K multi-hop GRPO-ready mixture.

    Args:
        selected_records: Disjoint MuSiQue records selected for GRPO.
        single_policy_records: ConditionalQA records used to prevent over-decomposition.
        policy_plans: Correct one-query policy plan mapping.

    Returns:
        Validated GRPO-compatible records.
    """
    grpo_records = []
    for record in selected_records:
        grpo_record = {
            "id": "grpo_" + record["id"],
            "instruction": GRPO_INSTRUCTION,
            "input": f"Question:\n{record['question']}",
            "output": convert_musique_plan(record),
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "musique",
            "source_id": record["id"],
            "namespace": "musique_aux",
            "hop_count": record["hop_count"],
            "task_type": "multi_hop",
            "reference_answer": record["answer"],
            "answer_aliases": record["answer_aliases"],
            "hop_answers": [step["answer"] for step in record["question_decomposition"]],
            "gold_doc_ids": record["gold_doc_ids"]
        }
        validate_grpo_record(grpo_record)
        grpo_records.append(grpo_record)

    for record in single_policy_records:
        grpo_record = {
            "id": "grpo_single_policy_" + record["id"],
            "instruction": GRPO_INSTRUCTION,
            "input": f"Scenario:\n{record['scenario']}\n\nQuestion:\n{record['question']}",
            "output": policy_plans[record["id"]],
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "conditionalqa",
            "source_id": record["id"],
            "namespace": "policy",
            "hop_count": 1,
            "task_type": "single_hop",
            "reference_answer": flatten_reference_answers(record["answers"]),
            "answer_aliases": [],
            "hop_answers": [],
            "gold_doc_ids": record["gold_doc_ids"]
        }
        validate_grpo_record(grpo_record)
        grpo_records.append(grpo_record)
    random.Random(RANDOM_SEED + 14).shuffle(grpo_records)
    if len(grpo_records) != TARGET_COUNTS["grpo"]:
        raise ValueError(f"Unexpected GRPO count: {len(grpo_records)}")
    return grpo_records


def export_evaluation_sets() -> dict[str, int]:
    """Export normalized official benchmark splits without resampling.

    Returns:
        Output file to record count mapping.
    """
    mappings = {
        "conditionalqa_dev.jsonl": INTERIM_ROOT / "conditionalqa_dev.jsonl",
        "conditionalqa_test_blind.jsonl": INTERIM_ROOT / "conditionalqa_test_no_answer.jsonl",
        "qrecc_test.jsonl": INTERIM_ROOT / "qrecc_test.jsonl",
        "musique_dev.jsonl": INTERIM_ROOT / "musique_dev.jsonl"
    }
    counts = {}
    for file_name, source_path in mappings.items():
        records = read_jsonl(source_path)
        write_jsonl(PROCESSED_ROOT / "eval" / file_name, records)
        counts[file_name] = len(records)
    return counts


def write_dataset_info() -> None:
    """Write LLaMA-Factory dataset registration metadata."""
    dataset_info = {
        "policy_query_sft": {
            "file_name": "train/sft_train.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system"
            }
        },
        "policy_query_dpo": {
            "file_name": "train/dpo_train.jsonl",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system"
            }
        },
        "policy_query_multihop_sft_cold_start": {
            "file_name": "train/sft_multihop_cold_start.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system"
            }
        },
        "policy_query_grpo_reference": {
            "file_name": "train/grpo_train.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system"
            }
        },
        "policy_query_grpo_domain_augmented": {
            "file_name": "train/grpo_train_domain_augmented.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system"
            }
        }
    }
    write_json(PROCESSED_ROOT / "dataset_info.json", dataset_info)


def main() -> None:
    """Build final train and benchmark datasets."""
    args = parse_args()
    summary_path = INTERIM_ROOT / "dataset_build_summary.json"
    if summary_path.exists() and not args.force:
        logger.info("Dataset outputs already exist; use --force to rebuild")
        return
    logger.info("=" * 80)
    logger.info("Building SFT, DPO, GRPO, and benchmark datasets")
    logger.info("=" * 80)
    qrecc_train = read_jsonl(INTERIM_ROOT / "qrecc_train.jsonl")
    qrecc_test = read_jsonl(INTERIM_ROOT / "qrecc_test.jsonl")
    qrecc_test_keys = {qrecc_content_key(record) for record in qrecc_test}
    qrecc_train_before_filter = len(qrecc_train)
    qrecc_train = [
        record
        for record in qrecc_train
        if qrecc_content_key(record) not in qrecc_test_keys
    ]
    conditional_train = read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
    musique_train = read_jsonl(INTERIM_ROOT / "musique_train.jsonl")

    sft_records, policy_plans = build_sft_records(qrecc_train, conditional_train)
    cold_sources, dpo_multihop_sources, grpo_sources, grpo_allocations = (
        allocate_multihop_records(
            musique_train,
            MULTIHOP_COLD_START_HOP_QUOTAS,
            DPO_MULTIHOP_HOP_QUOTAS,
            GRPO_MULTIHOP_COUNT,
            RANDOM_SEED + 5
        )
    )
    dpo_records = build_dpo_records(
        qrecc_train,
        conditional_train,
        policy_plans,
        dpo_multihop_sources
    )
    dpo_policy_source_ids = {
        record["source_id"]
        for record in dpo_records
        if record["source_dataset"] == "conditionalqa"
    }
    grpo_single_sources = select_grpo_single_records(
        conditional_train,
        dpo_policy_source_ids
    )
    cold_start_records = build_multihop_sft_records(cold_sources)
    grpo_records = build_grpo_records(grpo_sources, grpo_single_sources, policy_plans)

    write_jsonl(PROCESSED_ROOT / "train" / "sft_train.jsonl", sft_records)
    write_jsonl(
        PROCESSED_ROOT / "train" / "sft_multihop_cold_start.jsonl",
        cold_start_records
    )
    write_jsonl(PROCESSED_ROOT / "train" / "dpo_train.jsonl", dpo_records)
    write_jsonl(PROCESSED_ROOT / "train" / "grpo_train.jsonl", grpo_records)
    eval_counts = export_evaluation_sets()
    write_dataset_info()

    summary = {
        "sft_count": len(sft_records),
        "qrecc_train_exact_benchmark_duplicates_removed": (
            qrecc_train_before_filter - len(qrecc_train)
        ),
        "sft_source_counts": dict(Counter(record["sample_type"] for record in sft_records)),
        "multihop_cold_start_count": len(cold_start_records),
        "multihop_cold_start_hop_counts": dict(
            Counter(str(record["hop_count"]) for record in cold_start_records)
        ),
        "dpo_count": len(dpo_records),
        "dpo_source_counts": dict(Counter(record["source_dataset"] for record in dpo_records)),
        "dpo_task_counts": dict(Counter(record["task_type"] for record in dpo_records)),
        "dpo_hop_counts": dict(Counter(str(record["hop_count"]) for record in dpo_records)),
        "dpo_error_counts": dict(Counter(record["error_type"] for record in dpo_records)),
        "grpo_count": len(grpo_records),
        "grpo_task_counts": dict(Counter(record["task_type"] for record in grpo_records)),
        "grpo_hop_counts": dict(Counter(str(record["hop_count"]) for record in grpo_records)),
        "grpo_hop_allocations": {
            str(hop_count): count
            for hop_count, count in grpo_allocations.items()
        },
        "multihop_stage_source_overlap": 0,
        "multihop_stage_question_overlap": 0,
        "eval_counts": eval_counts
    }
    write_json(summary_path, summary)
    logger.info(
        "Built SFT=%d, cold-start=%d, DPO=%d, GRPO=%d",
        len(sft_records),
        len(cold_start_records),
        len(dpo_records),
        len(grpo_records)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
