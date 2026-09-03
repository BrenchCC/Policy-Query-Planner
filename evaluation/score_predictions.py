import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

# Add project root to Python path / 将项目根目录加入 Python 路径
sys.path.append(os.getcwd())

from data_preprocess.schemas import validate_planner_plan
from evaluation.io import MAX_JSONL_RECORDS, read_bounded_jsonl
from evaluation.judge import DualJudge, load_dotenv
from evaluation.metrics import (
    safe_mean,
    token_f1,
    exact_match,
    retrieval_metrics,
    dependency_metrics,
    best_reference_scores,
    evidence_recall_metrics,
    answerability_accuracy
)
from evaluation.reporting import write_reports

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse scoring command-line options.

    Args:
        argv: Optional argument list used by tests.
    """
    parser = argparse.ArgumentParser(description = "Score saved RAG and planner predictions")
    parser.add_argument("--dataset", required = True, help = "Gold evaluation JSONL")
    parser.add_argument("--predictions", required = True, help = "Prediction JSONL")
    parser.add_argument("--output-dir", required = True, help = "Report output directory")
    parser.add_argument("--judge-cache", help = "Judge vote JSONL cache")
    parser.add_argument("--skip-judge", action = "store_true", help = "Compute deterministic metrics only")
    parser.add_argument("--limit", type = int, help = "Optional sample limit")
    parser.add_argument("--force", action = "store_true", help = "Replace existing reports")
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= MAX_JSONL_RECORDS:
        parser.error(f"--limit must be between 1 and {MAX_JSONL_RECORDS}")
    return args


def _nested(value: Any, *path: str, default: Any = None) -> Any:
    """Read a nested mapping path safely.

    Args:
        value: Root object.
        path: Mapping keys.
        default: Missing-path result.
    """
    current = value
    for name in path:
        if not isinstance(current, dict) or name not in current:
            return default
        current = current[name]
    return current


def _predicted_plan(prediction: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a parsed plan from supported runtime shapes.

    Args:
        prediction: Runtime output record.
    """
    candidates = [
        prediction.get("parsed_plan"),
        prediction.get("query_plan"),
        _nested(prediction, "planner", "parsed_plan")
    ]
    return next((value for value in candidates if isinstance(value, dict)), None)


def _steps(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract step results from supported runtime shapes.

    Args:
        prediction: Runtime output record.
    """
    value = prediction.get("steps", prediction.get("step_results", []))
    return value if isinstance(value, list) else []


def _evidence_id(item: Any) -> str | None:
    """Extract a document identifier from a retrieval hit.

    Args:
        item: Retrieval result in flattened or nested form.
    """
    if not isinstance(item, dict):
        return None
    for key in ("id", "doc_id", "document_id", "chunk_id"):
        if item.get(key) is not None:
            return str(item[key])
    record = item.get("record")
    return _evidence_id(record) if isinstance(record, dict) else None


def _doc_ids(evidence: Any) -> list[str]:
    """Extract ranked identifiers from an evidence list.

    Args:
        evidence: Runtime evidence value.
    """
    if not isinstance(evidence, list):
        return []
    return [identifier for item in evidence if (identifier := _evidence_id(item))]


def _references(record: dict[str, Any]) -> list[Any]:
    """Extract final answer references and aliases.

    Args:
        record: Gold evaluation record.
    """
    references = record.get("reference_answers", [])
    if not isinstance(references, list):
        references = [references]
    return references


def _answer(prediction: dict[str, Any]) -> Any:
    """Extract the final answer from a runtime prediction.

    Args:
        prediction: Runtime output record.
    """
    return prediction.get("final_answer", prediction.get("answer"))


def _infer_answerable(prediction: dict[str, Any]) -> bool | None:
    """Infer answerability when the runtime did not emit an explicit label.

    Args:
        prediction: Runtime output record.
    """
    explicit = prediction.get("answerable", prediction.get("predicted_answerable"))
    if type(explicit) is bool:
        return explicit
    answer = _answer(prediction)
    if not isinstance(answer, str) or not answer.strip():
        return False
    normalized = answer.lower()
    refusal_markers = (
        "cannot be confirmed",
        "insufficient evidence",
        "not enough information",
        "无法确认",
        "证据不足"
    )
    return not any(marker in normalized for marker in refusal_markers)


def _prefix(values: dict[str, float], prefix: str) -> dict[str, float]:
    """Prefix metric names.

    Args:
        values: Metric mapping.
        prefix: Metric namespace.
    """
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _system_metrics(prediction: dict[str, Any]) -> dict[str, float]:
    """Extract inference reliability, cost, and latency metrics.

    Args:
        prediction: Runtime output record.
    """
    steps = _steps(prediction)
    errors = prediction.get("errors", [])
    if not isinstance(errors, list):
        errors = [errors] if errors else []
    metrics = {
        "inference_success": float(bool(prediction.get("success", not errors))),
        "query_count": float(len(steps) or len((_predicted_plan(prediction) or {}).get("queries", []))),
        "fallback_rate": float(bool(prediction.get("fallback", prediction.get("rewrite_fallback", False)))),
        "error_rate": float(bool(errors or prediction.get("error")))
    }
    usage = prediction.get("usage", {})
    total_usage = usage.get("total", usage) if isinstance(usage, dict) else {}
    if isinstance(total_usage, dict):
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(total_usage.get(name), (int, float)):
                metrics[name] = float(total_usage[name])
    latency = prediction.get("latency_seconds", prediction.get("timing", {}))
    if isinstance(latency, (int, float)):
        metrics["latency_total_seconds"] = float(latency)
    elif isinstance(latency, dict):
        for name, value in latency.items():
            if isinstance(value, (int, float)):
                metrics[f"latency_{name}_seconds"] = float(value)
    return metrics


def _rag_metrics(record: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float]:
    """Compute deterministic policy RAG metrics.

    Args:
        record: Gold RAG record.
        prediction: Runtime prediction.
    """
    evidence = prediction.get("evidence", prediction.get("aggregated_evidence", []))
    retrieved = _doc_ids(evidence)
    gold_doc_ids = record.get("gold_doc_ids", [])
    metrics = {}
    if gold_doc_ids:
        metrics.update(_prefix(retrieval_metrics(retrieved, gold_doc_ids), "doc"))
        evidence_gold = record.get("gold_evidence_doc_ids", gold_doc_ids)
        metrics.update(_prefix(evidence_recall_metrics(retrieved, evidence_gold), "evidence"))
    reference_answerable = bool(record.get("answerable", not record.get("not_answerable", False)))
    predicted_answerable = _infer_answerable(prediction)
    answerability_covered = bool(prediction.get("success")) and predicted_answerable is not None
    metrics["answerability_coverage"] = float(answerability_covered)
    if answerability_covered:
        metrics["answerability_accuracy"] = answerability_accuracy(
            predicted_answerable,
            reference_answerable
        )
    if reference_answerable:
        metrics.update(_prefix(best_reference_scores(_answer(prediction), _references(record)), "answer"))
    return metrics


def _multihop_metrics(
    record: dict[str, Any],
    prediction: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute deterministic multi-hop planner and answer metrics.

    Args:
        record: Gold multi-hop record.
        prediction: Runtime prediction.

    Returns:
        Aggregate metrics and aligned per-step diagnostic scores.
    """
    planner = prediction.get("planner", {})
    parsed_plan = _predicted_plan(prediction)
    raw_output = prediction.get("raw_output", _nested(prediction, "planner", "raw_output"))
    parse_success = prediction.get("parse_success", _nested(prediction, "planner", "parse_success"))
    if parse_success is None:
        parse_success = parsed_plan is not None
        if parsed_plan is None and isinstance(raw_output, str):
            try:
                parsed_plan = json.loads(raw_output)
                parse_success = True
            except json.JSONDecodeError:
                pass
    schema_valid = prediction.get("schema_valid", _nested(prediction, "planner", "schema_valid"))
    if schema_valid is None:
        try:
            parsed_plan = validate_planner_plan(parsed_plan) if parsed_plan is not None else None
            schema_valid = parsed_plan is not None
        except ValueError:
            schema_valid = False
    reference_plan = record.get("reference_plan")
    predicted_queries = []
    if bool(schema_valid) and isinstance(parsed_plan, dict):
        queries = parsed_plan.get("queries")
        if isinstance(queries, list) and all(isinstance(query, dict) for query in queries):
            predicted_queries = queries
    metrics = {
        "json_parse": float(bool(parse_success)),
        "schema_valid": float(bool(schema_valid)),
        "hop_count_accuracy": float(len(predicted_queries) == record.get("hop_count"))
    }
    scored_plan = parsed_plan if bool(schema_valid) else None
    metrics.update(dependency_metrics(scored_plan, reference_plan))

    reference_steps = record.get("reference_steps", [])
    predicted_steps = _steps(prediction)
    query_exact = []
    query_f1 = []
    intermediate_exact = []
    intermediate_f1 = []
    hop_retrieval: dict[str, list[float]] = {}
    joint_by_cutoff: dict[int, list[float]] = {1: [], 5: [], 10: []}
    per_step_scores = []
    for index, reference_step in enumerate(reference_steps):
        predicted_query = predicted_queries[index].get("query", "") if index < len(predicted_queries) else ""
        query_exact.append(exact_match(predicted_query, reference_step.get("query", "")))
        query_f1.append(token_f1(predicted_query, reference_step.get("query", "")))
        predicted_step = predicted_steps[index] if index < len(predicted_steps) else {}
        retrieved = _doc_ids(predicted_step.get("evidence", []))
        retrieval = retrieval_metrics(retrieved, [reference_step.get("gold_doc_id")])
        for name, value in retrieval.items():
            hop_retrieval.setdefault(name, []).append(value)
        for cutoff in joint_by_cutoff:
            joint_by_cutoff[cutoff].append(retrieval[f"joint_recall@{cutoff}"])
        references = [reference_step.get("answer")]
        aliases = reference_step.get("answer_aliases", [])
        if isinstance(aliases, list):
            references.extend(aliases)
        answer_scores = best_reference_scores(predicted_step.get("answer"), references)
        intermediate_exact.append(answer_scores["exact_match"])
        intermediate_f1.append(answer_scores["token_f1"])
        per_step_scores.append(
            {
                "id": reference_step.get("id"),
                "status": predicted_step.get("status", "missing"),
                "query_exact_match": query_exact[-1],
                "query_token_f1": query_f1[-1],
                "gold_doc_id": reference_step.get("gold_doc_id"),
                "retrieval": retrieval,
                "intermediate_answer_exact_match": answer_scores["exact_match"],
                "intermediate_answer_token_f1": answer_scores["token_f1"]
            }
        )
    metrics["query_exact_match"] = safe_mean(query_exact)
    metrics["query_token_f1"] = safe_mean(query_f1)
    metrics["intermediate_answer_exact_match"] = safe_mean(intermediate_exact)
    metrics["intermediate_answer_token_f1"] = safe_mean(intermediate_f1)
    for name, values in hop_retrieval.items():
        metrics[f"hop_{name}"] = safe_mean(values)
    for cutoff, values in joint_by_cutoff.items():
        metrics[f"joint_recall@{cutoff}"] = float(bool(values) and all(values))
    metrics.update(_prefix(best_reference_scores(_answer(prediction), _references(record)), "final_answer"))
    return metrics, per_step_scores


def score_prediction(
    record: dict[str, Any],
    prediction: dict[str, Any],
    judge: DualJudge | None = None
) -> dict[str, Any]:
    """Score one matched Gold record and prediction.

    Args:
        record: Gold evaluation record.
        prediction: Runtime prediction.
        judge: Optional configured semantic Judge.
    """
    is_multihop = record.get("source_dataset") == "musique" or "hop_count" in record
    task_type = "multihop" if is_multihop else "rag"
    metrics = _system_metrics(prediction)
    per_step_scores = []
    if is_multihop:
        multihop_metrics, per_step_scores = _multihop_metrics(record, prediction)
        metrics.update(multihop_metrics)
    else:
        metrics.update(_rag_metrics(record, prediction))
    result: dict[str, Any] = {
        "id": record.get("id"),
        "source_dataset": record.get("source_dataset"),
        "hop_count": record.get("hop_count"),
        "answerable": record.get("answerable"),
        "metrics": metrics
    }
    if is_multihop:
        result["per_step_scores"] = per_step_scores
    if judge is not None:
        judge_result = judge.evaluate(record, prediction, task_type)
        result["judge"] = judge_result
        metrics["judge_coverage"] = judge_result["judge_coverage"]
        metrics["judge_unscored_rate"] = float(judge_result["judge_unscored"])
        metrics["judge_disagreement_rate"] = judge_result["judge_disagreement_rate"]
        for name, value in judge_result["consensus"].items():
            if value is not None:
                metrics[f"judge_{name}"] = float(value)
    return result


def score_files(
    dataset_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    judge: DualJudge | None = None,
    limit: int | None = None,
    force: bool = False
) -> dict[str, Any]:
    """Match, score, and report two JSONL files.

    Args:
        dataset_path: Gold evaluation JSONL.
        predictions_path: Runtime prediction JSONL.
        output_dir: Report destination.
        judge: Optional semantic Judge.
        limit: Optional number of Gold records to score.
        force: Whether existing report artifacts may be replaced.
    """
    records = read_bounded_jsonl(dataset_path, limit = limit)
    predictions = read_bounded_jsonl(predictions_path)
    by_id = {}
    for prediction in predictions:
        identifier = prediction.get("id")
        if identifier in by_id:
            raise ValueError(f"Duplicate prediction ID: {identifier}")
        by_id[identifier] = prediction
    missing = [record.get("id") for record in records if record.get("id") not in by_id]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise ValueError(f"Missing predictions for {len(missing)} records: {preview}")
    scores = [score_prediction(record, by_id[record["id"]], judge = judge) for record in records]
    return write_reports(scores, output_dir, force = force)


def main() -> None:
    """Run the scoring CLI."""
    args = parse_args()
    judge = None
    if not args.skip_judge:
        load_dotenv()
        judge_cache = args.judge_cache or str(Path(args.output_dir) / "judge_cache.jsonl")
        client = OpenAI(
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url = (
                os.environ.get("LLM_BASE_URL")
                or os.environ.get("LLM_API_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            ),
            timeout = 300.0,
            max_retries = 0
        )
        judge = DualJudge(client = client, cache_path = judge_cache)
    summary = score_files(
        dataset_path = args.dataset,
        predictions_path = args.predictions,
        output_dir = args.output_dir,
        judge = judge,
        limit = args.limit,
        force = args.force
    )
    logger.info("Scored %s samples into %s", summary["count"], args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
