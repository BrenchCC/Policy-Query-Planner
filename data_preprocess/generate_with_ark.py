import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def load_local_environment() -> None:
    """Load ignored project .env values without overriding process variables."""
    environment_path = Path(__file__).resolve().parents[1] / ".env"
    if environment_path.exists():
        for raw_line in environment_path.read_text(encoding = "utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.removeprefix("export ").split("=", 1)
            name = name.strip()
            value = value.strip().strip("\"'")
            if name and value:
                os.environ.setdefault(name, value)
    if os.environ.get("LLM_BASE_URL"):
        os.environ.setdefault("LLM_API_BASE_URL", os.environ["LLM_BASE_URL"])
    if os.environ.get("MODEL"):
        os.environ.setdefault("LLM_ENDPOINT", os.environ["MODEL"])


load_local_environment()

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.ark_llm_call import call_llm_on_volcengine
from data_preprocess.common import extract_json_object, normalized_key, read_jsonl
from data_preprocess.config import LLM_ENDPOINT, REQUEST_ROOT, RESPONSE_ROOT
from data_preprocess.schemas import validate_planner_plan

logger = logging.getLogger(__name__)


class ExcludeHttpConsoleFilter(logging.Filter):
    """Hide verbose HTTP client INFO records from the terminal only."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a log record should appear in the terminal.

        Args:
            record: Candidate logging record.

        Returns:
            False for httpx and httpcore records, otherwise True.
        """
        return not record.name.startswith(("httpx", "httpcore"))


def configure_logging(stage: str) -> Path:
    """Configure complete file logs and concise terminal logs.

    Args:
        stage: Generation stage used in the log filename.

    Returns:
        Path of the append-only full log file.
    """
    log_root = Path(__file__).resolve().parents[1] / "logs"
    log_root.mkdir(parents = True, exist_ok = True)
    log_path = log_root / f"generate_with_ark_{stage}.log"
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler = logging.FileHandler(log_path, encoding = "utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ExcludeHttpConsoleFilter())
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return log_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Run optional Ark data generation")
    parser.add_argument("--stage", choices = ["sft", "dpo", "grpo"], required = True)
    parser.add_argument("--dry-run", action = "store_true", help = "Inspect requests without API calls")
    parser.add_argument("--resume", action = "store_true", help = "Skip completed request IDs")
    parser.add_argument("--limit", type = int, default = None, help = "Maximum requests to process")
    parser.add_argument("--workers", type = int, default = 4, help = "Concurrent API workers")
    parser.add_argument(
        "--max-retries",
        type = int,
        default = 2,
        help = "Retries after an API or validation failure"
    )
    parser.add_argument(
        "--log-every",
        type = int,
        default = 10,
        help = "Log aggregate progress every N completed requests"
    )
    parser.add_argument(
        "--reasoning-option",
        default = None,
        help = "Optional Ark thinking mode"
    )
    return parser.parse_args()


def load_completed_requests(path: Path) -> dict[str, str]:
    """Load successfully completed request IDs and hashes.

    Args:
        path: Response JSONL path.

    Returns:
        Request ID to request hash mapping.
    """
    if not path.exists():
        return {}
    return {
        record["id"]: record.get("request_hash", "")
        for record in read_jsonl(path)
        if record.get("status") == "success"
    }


def reference_answer_fragments(value: str) -> list[str]:
    """Split reference answers into normalized leakage-check fragments.

    Args:
        value: Semicolon-separated reference answers.

    Returns:
        Non-trivial normalized answer fragments.
    """
    return [
        fragment
        for fragment in (
            normalized_key(part)
            for part in value.split(";")
        )
        if len(fragment) > 4 and fragment != "notanswerable"
    ]


def validate_grpo_annotation(
    request: dict[str, Any],
    response_text: str
) -> dict[str, Any]:
    """Validate one synthetic policy multi-hop annotation.

    Args:
        request: GRPO generation request with indexed evidence.
        response_text: Raw model response text.

    Returns:
        Validated annotation with a parsed planner plan.

    Raises:
        ValueError: If structure, grounding, dependencies, or leakage checks fail.
    """
    annotation = extract_json_object(response_text)
    required_fields = {
        "scenario",
        "question",
        "plan",
        "hop_answers",
        "hop_evidence_indices",
        "reference_answer"
    }
    if set(annotation) != required_fields:
        raise ValueError("Generated GRPO annotation fields do not match the contract")
    for field in ["scenario", "question", "reference_answer"]:
        if not isinstance(annotation[field], str) or not annotation[field].strip():
            raise ValueError(f"Generated GRPO {field} must be non-empty")
    if normalized_key(annotation["question"]) == normalized_key(request["source_question"]):
        raise ValueError("Generated GRPO question must differ from the source question")

    plan = validate_planner_plan(annotation["plan"])
    hop_count = len(plan["queries"])
    if hop_count not in {2, 3, 4}:
        raise ValueError("Generated GRPO plan must contain two to four queries")
    if not any(query["depends_on"] for query in plan["queries"]):
        raise ValueError("Generated GRPO plan must contain at least one dependency")

    hop_answers = annotation["hop_answers"]
    evidence_indices = annotation["hop_evidence_indices"]
    if not isinstance(hop_answers, list) or len(hop_answers) != hop_count:
        raise ValueError("Generated GRPO hop_answers must match the query count")
    if not isinstance(evidence_indices, list) or len(evidence_indices) != hop_count:
        raise ValueError("Generated GRPO evidence indices must match the query count")
    if not all(isinstance(index, int) for index in evidence_indices):
        raise ValueError("Generated GRPO evidence indices must be integers")
    if len(set(evidence_indices)) != hop_count:
        raise ValueError("Generated GRPO hops must use distinct evidence indices")

    evidence_items = request["evidence_items"]
    selected_doc_ids = []
    for answer, evidence_index in zip(hop_answers, evidence_indices):
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Generated GRPO hop answers must be non-empty strings")
        if not 0 <= evidence_index < len(evidence_items):
            raise ValueError("Generated GRPO evidence index is out of range")
        evidence_item = evidence_items[evidence_index]
        answer_key = normalized_key(answer)
        if not answer_key or answer_key not in normalized_key(evidence_item["text"]):
            raise ValueError("Generated GRPO hop answer is not grounded in its evidence")
        selected_doc_ids.append(evidence_item["gold_doc_id"])
    if len(set(selected_doc_ids)) != hop_count:
        raise ValueError("Generated GRPO hops must use distinct policy chunks")

    selected_evidence = " ".join(
        evidence_items[index]["text"]
        for index in evidence_indices
    )
    reference_key = normalized_key(annotation["reference_answer"])
    final_hop_key = normalized_key(hop_answers[-1])
    if not reference_key or reference_key not in normalized_key(selected_evidence):
        raise ValueError("Generated GRPO reference answer is not grounded in selected evidence")
    if reference_key not in final_hop_key and final_hop_key not in reference_key:
        raise ValueError("Generated GRPO reference answer must match the final hop answer")
    serialized_plan = json.dumps(plan, ensure_ascii = False, separators = (",", ":"))
    serialized_key = normalized_key(serialized_plan)
    leaked_answers = [
        answer
        for answer in reference_answer_fragments(annotation["reference_answer"])
        if answer in serialized_key
    ]
    if leaked_answers:
        raise ValueError("Generated plan leaks the reference answer")
    annotation["plan"] = plan
    annotation["gold_doc_ids"] = selected_doc_ids
    return annotation


def validate_generated_plan(
    request: dict[str, Any],
    response_text: str
) -> str:
    """Parse and validate one generated planner response.

    Args:
        request: Source generation request.
        response_text: Raw model response text.

    Returns:
        Compact serialized planner response.

    Raises:
        ValueError: If JSON, schema, preference, or leakage checks fail.
    """
    if response_text == "dummy_result":
        raise ValueError("Ark helper returned dummy_result")
    if request["stage"] == "grpo":
        annotation = validate_grpo_annotation(request, response_text)
        return json.dumps(
            annotation["plan"],
            ensure_ascii = False,
            separators = (",", ":")
        )
    plan = extract_json_object(response_text)
    validate_planner_plan(plan)
    serialized = json.dumps(plan, ensure_ascii = False, separators = (",", ":"))
    if request["stage"] == "dpo" and serialized == request["payload"]["chosen"]:
        raise ValueError("Generated rejected plan equals the chosen plan")
    if request["stage"] == "dpo" and request["payload"]["task_type"] == "single_hop":
        if len(plan["queries"]) != 1:
            raise ValueError("Generated single-hop DPO negative must contain one query")
    return serialized


def build_response_record(
    request: dict[str, Any],
    reasoning: str | None,
    response_text: str,
    prompt_tokens: int | str,
    completion_tokens: int | str
) -> dict[str, Any]:
    """Build one validated or rejected API response record.

    Args:
        request: Source generation request.
        reasoning: Optional model reasoning content.
        response_text: Raw model response.
        prompt_tokens: Prompt token usage.
        completion_tokens: Completion token usage.

    Returns:
        Structured response record with validation status.
    """
    record = {
        "id": request["id"],
        "stage": request["stage"],
        "source_id": request["source_id"],
        "request_hash": request["request_hash"],
        "status": "success",
        "parsed_output": "",
        "raw_response": response_text,
        "reasoning_content": reasoning or "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": ""
    }
    if "target_record_id" in request:
        record["target_record_id"] = request["target_record_id"]
    try:
        record["parsed_output"] = validate_generated_plan(request, response_text)
        if request["stage"] == "grpo":
            annotation = validate_grpo_annotation(request, response_text)
            record.update(
                {
                    "generated_scenario": annotation["scenario"],
                    "generated_question": annotation["question"],
                    "hop_answers": annotation["hop_answers"],
                    "hop_evidence_indices": annotation["hop_evidence_indices"],
                    "reference_answer": annotation["reference_answer"],
                    "gold_doc_ids": annotation["gold_doc_ids"],
                    "variant_index": request["variant_index"]
                }
            )
    except ValueError as error:
        record["status"] = "invalid"
        record["error"] = str(error)
    return record


def generate_response(
    request: dict[str, Any],
    reasoning_option: str | None,
    max_retries: int
) -> dict[str, Any]:
    """Generate and validate one response with bounded retries.

    Args:
        request: Prepared generation request.
        reasoning_option: Optional Ark thinking mode.
        max_retries: Number of retries after the initial attempt.

    Returns:
        Final validated or invalid response record.
    """
    response_record = {}
    for attempt in range(max_retries + 1):
        reasoning, result, prompt_tokens, completion_tokens = call_llm_on_volcengine(
            request["prompt"],
            LLM_ENDPOINT,
            system_prompt = request["system"],
            stream = False,
            reasoning_option = reasoning_option
        )
        response_record = build_response_record(
            request,
            reasoning,
            result,
            prompt_tokens,
            completion_tokens
        )
        response_record["attempts"] = attempt + 1
        if response_record["status"] == "success":
            return response_record
        if attempt < max_retries:
            time.sleep(min(2 ** attempt, 8))
    return response_record


def token_value(value: int | str) -> int:
    """Convert an optional API token count to an integer.

    Args:
        value: Integer token count or an empty string.

    Returns:
        Parsed token count, or zero when unavailable.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def sort_response_file(path: Path, requests: list[dict[str, Any]]) -> None:
    """Rewrite responses in request order while preserving obsolete records.

    Args:
        path: Response JSONL path to normalize.
        requests: Complete ordered request queue.
    """
    if not path.exists():
        return
    all_records = read_jsonl(path)
    latest_records = {
        record["id"]: record
        for record in all_records
    }
    request_ids = {request["id"] for request in requests}
    ordered_records = [
        latest_records[request["id"]]
        for request in requests
        if request["id"] in latest_records
    ]
    ordered_records.extend(
        record
        for record_id, record in latest_records.items()
        if record_id not in request_ids
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding = "utf-8") as file:
        for record in ordered_records:
            file.write(json.dumps(record, ensure_ascii = False) + "\n")
    temporary_path.replace(path)


def run_dry_run(stage: str, requests: list[dict[str, Any]], limit: int | None) -> None:
    """Log queue details without making an API call.

    Args:
        stage: Generation stage.
        requests: Prepared generation requests.
        limit: Optional request limit.
    """
    selected = requests[:limit] if limit is not None else requests
    logger.info("Dry run stage=%s requests=%d", stage, len(selected))
    if selected:
        logger.info("First request ID: %s", selected[0]["id"])
        logger.info("First prompt preview:\n%s", selected[0]["prompt"][:1200])


def main() -> None:
    """Run or inspect one optional Ark generation stage."""
    args = parse_args()
    log_path = configure_logging(args.stage)
    logger.info("Full log file: %s", log_path)
    request_path = REQUEST_ROOT / f"{args.stage}_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"Prepare requests first: {request_path}")
    requests = read_jsonl(request_path)
    if args.dry_run:
        run_dry_run(args.stage, requests, args.limit)
        return
    if not LLM_ENDPOINT:
        raise RuntimeError("Missing LLM_ENDPOINT in environment")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")

    response_path = RESPONSE_ROOT / f"{args.stage}_responses.jsonl"
    if response_path.exists() and not args.resume:
        raise FileExistsError("Response file exists; use --resume to continue safely")
    completed_requests = load_completed_requests(response_path) if args.resume else {}
    stale_ids = [
        request["id"]
        for request in requests
        if request["id"] in completed_requests
        and completed_requests[request["id"]] != request.get("request_hash", "")
    ]
    if stale_ids:
        logger.warning(
            "Regenerating %d stale responses whose request hashes changed",
            len(stale_ids)
        )
    pending = [
        request
        for request in requests
        if completed_requests.get(request["id"]) != request.get("request_hash", "")
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    response_path.parent.mkdir(parents = True, exist_ok = True)
    success_count = 0
    invalid_count = 0
    prompt_token_count = 0
    completion_token_count = 0
    logger.info(
        "Starting stage=%s pending=%d workers=%d max_retries=%d",
        args.stage,
        len(pending),
        args.workers,
        args.max_retries
    )
    with response_path.open("a", encoding = "utf-8") as file:
        with ThreadPoolExecutor(max_workers = args.workers) as executor:
            futures = {
                executor.submit(
                    generate_response,
                    request,
                    args.reasoning_option,
                    args.max_retries
                ): (index, request)
                for index, request in enumerate(pending)
            }
            buffered_records = {}
            next_write_index = 0
            progress = tqdm(total = len(futures), desc = f"Generating {args.stage}")
            for completed_count, future in enumerate(as_completed(futures), start = 1):
                request_index, request = futures[future]
                try:
                    response_record = future.result()
                except Exception as error:
                    logger.exception("Worker failed for request %s", request["id"])
                    response_record = build_response_record(request, None, "dummy_result", "", "")
                    response_record["error"] = f"Worker failed: {error}"
                    response_record["attempts"] = 0
                buffered_records[request_index] = response_record
                while next_write_index in buffered_records:
                    ordered_record = buffered_records.pop(next_write_index)
                    file.write(json.dumps(ordered_record, ensure_ascii = False) + "\n")
                    file.flush()
                    next_write_index += 1
                if response_record["status"] == "success":
                    success_count += 1
                else:
                    invalid_count += 1
                prompt_token_count += token_value(response_record["prompt_tokens"])
                completion_token_count += token_value(response_record["completion_tokens"])
                progress.update(1)
                if completed_count % args.log_every == 0 or completed_count == len(futures):
                    logger.info(
                        "Progress stage=%s completed=%d/%d success=%d invalid=%d "
                        "prompt_tokens=%d completion_tokens=%d",
                        args.stage,
                        completed_count,
                        len(futures),
                        success_count,
                        invalid_count,
                        prompt_token_count,
                        completion_token_count
                    )
            progress.close()
    sort_response_file(response_path, requests)
    logger.info("Response file normalized to request order: %s", response_path)


if __name__ == "__main__":
    main()
