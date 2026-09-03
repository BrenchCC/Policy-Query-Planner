import re
import string
import unicodedata
from collections import Counter
from typing import Any


def safe_mean(values: list[float | int | bool]) -> float:
    """Return the arithmetic mean or zero for an empty list.

    Args:
        values: Numeric values to average.
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def normalize_answer(value: Any) -> str:
    """Normalize an answer for deterministic exact and token matching.

    Args:
        value: Answer-like value to normalize.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = "".join(" " if character in string.punctuation else character for character in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: Any, reference: Any) -> float:
    """Compute normalized exact match for one reference.

    Args:
        prediction: Predicted answer.
        reference: Gold answer.
    """
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: Any, reference: Any) -> float:
    """Compute bag-of-token F1 for one reference.

    Args:
        prediction: Predicted answer.
        reference: Gold answer.
    """
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return float(2 * precision * recall / (precision + recall))


def best_reference_scores(prediction: Any, references: list[Any]) -> dict[str, float]:
    """Score an answer against its best matching reference.

    Args:
        prediction: Predicted answer.
        references: Acceptable Gold answers and aliases.
    """
    usable = [reference for reference in references if reference is not None]
    if not usable:
        return {"exact_match": 0.0, "token_f1": 0.0}
    return {
        "exact_match": max(exact_match(prediction, reference) for reference in usable),
        "token_f1": max(token_f1(prediction, reference) for reference in usable)
    }


def _unique(values: list[Any]) -> list[str]:
    """Convert identifiers to a stable unique list.

    Args:
        values: Identifier-like values.
    """
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        identifier = str(value)
        if identifier and identifier not in seen:
            result.append(identifier)
            seen.add(identifier)
    return result


def retrieval_metrics(
    retrieved_doc_ids: list[Any],
    gold_doc_ids: list[Any],
    ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Compute document recall, hit, joint recall, and reciprocal rank.

    Args:
        retrieved_doc_ids: Ranked retrieved document identifiers.
        gold_doc_ids: Required Gold document identifiers.
        ks: Rank cutoffs to evaluate.
    """
    retrieved = _unique(retrieved_doc_ids)
    gold = set(_unique(gold_doc_ids))
    scores: dict[str, float] = {}
    for cutoff in ks:
        hits = gold.intersection(retrieved[:cutoff])
        scores[f"recall@{cutoff}"] = len(hits) / len(gold) if gold else 0.0
        scores[f"hit@{cutoff}"] = float(bool(hits))
        scores[f"joint_recall@{cutoff}"] = float(bool(gold) and hits == gold)
    scores["mrr"] = next(
        (1.0 / rank for rank, doc_id in enumerate(retrieved, start = 1) if doc_id in gold),
        0.0
    )
    return scores


def evidence_recall_metrics(
    retrieved_doc_ids: list[Any],
    gold_evidence_doc_ids: list[Any],
    ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Measure coverage of individual Gold evidence annotations.

    Unlike document recall, repeated document mappings are deliberately retained:
    two Gold evidence spans in one retrieved document count as two covered spans.

    Args:
        retrieved_doc_ids: Ranked retrieved document identifiers.
        gold_evidence_doc_ids: One document identifier or identifier list per evidence.
        ks: Rank cutoffs to evaluate.
    """
    retrieved = _unique(retrieved_doc_ids)
    mappings = []
    for value in gold_evidence_doc_ids:
        candidates = value if isinstance(value, list) else [value]
        mappings.append(set(_unique(candidates)))
    scores = {}
    for cutoff in ks:
        retrieved_at_k = set(retrieved[:cutoff])
        covered = sum(bool(mapping.intersection(retrieved_at_k)) for mapping in mappings)
        scores[f"recall@{cutoff}"] = covered / len(mappings) if mappings else 0.0
    return scores


def dependency_edges(plan: dict[str, Any] | None) -> set[tuple[str, str]]:
    """Extract directed dependency edges from a planner object.

    Args:
        plan: Planner dictionary containing an ordered queries list.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("queries"), list):
        return set()
    edges = set()
    for query in plan["queries"]:
        if not isinstance(query, dict):
            continue
        query_id = str(query.get("id", ""))
        dependencies = query.get("depends_on", [])
        if not isinstance(dependencies, list):
            continue
        edges.update((str(dependency), query_id) for dependency in dependencies)
    return edges


def dependency_metrics(
    predicted_plan: dict[str, Any] | None,
    reference_plan: dict[str, Any] | None
) -> dict[str, float]:
    """Compare predicted and Gold dependency graphs.

    Args:
        predicted_plan: Predicted planner object.
        reference_plan: Gold planner object.
    """
    predicted = dependency_edges(predicted_plan)
    reference = dependency_edges(reference_plan)
    overlap = len(predicted.intersection(reference))
    precision = overlap / len(predicted) if predicted else float(not reference)
    recall = overlap / len(reference) if reference else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "dependency_exact_match": float(predicted == reference),
        "dependency_precision": precision,
        "dependency_recall": recall,
        "dependency_f1": f1
    }


def answerability_accuracy(
    predicted_answerable: bool | None,
    reference_answerable: bool
) -> float:
    """Compare a predicted answerability label with the reference.

    Args:
        predicted_answerable: Inferred or explicit prediction.
        reference_answerable: Gold answerability label.
    """
    return float(predicted_answerable is not None and predicted_answerable == reference_answerable)
