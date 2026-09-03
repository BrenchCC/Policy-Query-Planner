import re
import json
import time
import logging
from typing import Any

from data_preprocess.schemas import validate_planner_plan
from retrieval.rag_pipeline import AnswerGenerator, _usage_dict

logger = logging.getLogger(__name__)

CHAIN_RRF_K = 60
PLANNER_MAX_TOKENS = 1024


def _strip_code_fence(content: str) -> str:
    """Remove one optional Markdown code fence from model output.

    Args:
        content: Raw model response text.

    Returns:
        Text that can be parsed as JSON.
    """
    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags = re.DOTALL)
    return match.group(1).strip() if match else stripped


def _clean_intermediate_answer(content: str) -> str:
    """Remove generated evidence citations before dependency substitution.

    Args:
        content: Evidence-grounded intermediate answer.

    Returns:
        Answer text safe to insert into the next retrieval query.
    """
    without_citations = re.sub(r"\s*\[(?:\d+\s*,?\s*)+\]", "", content)
    return without_citations.strip()


def _sum_usage(usages: list[dict[str, int]]) -> dict[str, int]:
    """Sum token counters reported by several model requests.

    Args:
        usages: OpenAI-compatible usage dictionaries.

    Returns:
        Summed counters, omitting keys that were never reported.
    """
    result: dict[str, int] = {}
    for usage in usages:
        for name, value in usage.items():
            result[name] = result.get(name, 0) + int(value)
    return result


def _fuse_evidence_rankings(
    rankings: list[tuple[str, list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    """Fuse per-step evidence rankings into one chain-level RRF order.

    Args:
        rankings: Ordered step identifiers and their ranked evidence hits.

    Returns:
        Deduplicated evidence with chain-level scores and step provenance.
    """
    entries: dict[str, dict[str, Any]] = {}
    first_seen = 0
    for step_id, evidence in rankings:
        for rank, item in enumerate(evidence, start = 1):
            document_id = item.get("id")
            key = str(document_id) if document_id else json.dumps(
                item,
                ensure_ascii = False,
                sort_keys = True
            )
            if key not in entries:
                entries[key] = {
                    "item": dict(item),
                    "score": 0.0,
                    "first_seen": first_seen,
                    "query_ids": []
                }
                first_seen += 1
            entry = entries[key]
            entry["score"] += 1.0 / (CHAIN_RRF_K + rank)
            if step_id not in entry["query_ids"]:
                entry["query_ids"].append(step_id)

    ordered = sorted(
        entries.values(),
        key = lambda entry: (-entry["score"], entry["first_seen"])
    )
    fused = []
    for entry in ordered:
        item = entry["item"]
        item["chain_rrf_score"] = entry["score"]
        item["query_ids"] = entry["query_ids"]
        fused.append(item)
    return fused


class PlannerGenerator:
    """Generate strict 1-to-4-step plans with an OpenAI-compatible client."""

    def __init__(self, client: Any, model: str) -> None:
        """Configure the planner.

        Args:
            client: OpenAI-compatible chat client.
            model: Planner endpoint or model identifier.
        """
        if not model.strip():
            raise ValueError("planner model cannot be empty")
        self.client = client
        self.model = model

    def generate(self, record: dict[str, Any]) -> dict[str, Any]:
        """Generate and validate one evaluation record's query plan.

        Args:
            record: Evaluation record containing system, instruction, and input.

        Returns:
            Raw output, parsed plan, validation flags, usage, latency, and error.
        """
        started_at = time.perf_counter()
        result: dict[str, Any] = {
            "raw_output": None,
            "parsed_plan": None,
            "parse_success": False,
            "schema_valid": False,
            "usage": {},
            "latency_seconds": 0.0,
            "error": None
        }
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role": "system", "content": record["system"]},
                    {
                        "role": "user",
                        "content": f"{record['instruction']}\n\n{record['input']}"
                    }
                ],
                temperature = 0,
                max_tokens = PLANNER_MAX_TOKENS
            )
            result["usage"] = _usage_dict(response)
            content = response.choices[0].message.content
            result["raw_output"] = content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Planner model returned empty content")
            try:
                parsed = json.loads(_strip_code_fence(content))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid planner JSON: {error}") from error
            result["parse_success"] = True
            result["parsed_plan"] = parsed
            result["parsed_plan"] = validate_planner_plan(parsed)
            result["schema_valid"] = True
        except Exception as error:
            result["error"] = str(error)
        finally:
            result["latency_seconds"] = time.perf_counter() - started_at
        return result


class ChainExecutor:
    """Execute validated query plans one step at a time without fallback."""

    def __init__(
        self,
        retriever: Any,
        planner: PlannerGenerator,
        answer_generator: AnswerGenerator
    ) -> None:
        """Configure a strict multi-hop evaluation executor.

        Args:
            retriever: Namespace-specific retriever exposing search.
            planner: General query-plan generator.
            answer_generator: Evidence-grounded intermediate/final answer generator.
        """
        self.retriever = retriever
        self.planner = planner
        self.answer_generator = answer_generator

    @staticmethod
    def _question_with_scenario(record: dict[str, Any]) -> str:
        """Build the final-answer prompt question.

        Args:
            record: Evaluation record with a question and optional scenario.

        Returns:
            Question text containing scenario context when supplied.
        """
        scenario = str(record.get("scenario") or "").strip()
        question = str(record["question"]).strip()
        if not scenario:
            return question
        return f"Scenario:\n{scenario}\n\nQuestion:\n{question}"

    @staticmethod
    def _embedding_model(retriever: Any) -> str | None:
        """Read an embedding model identifier from a retriever when available.

        Args:
            retriever: Active retrieval component.

        Returns:
            Model identifier or None for a generic test retriever.
        """
        manifest = getattr(retriever, "vector_manifest", {})
        return manifest.get("model") if isinstance(manifest, dict) else None

    def run(
        self,
        record: dict[str, Any],
        top_k: int = 10,
        candidate_k: int = 50
    ) -> dict[str, Any]:
        """Generate and execute one record with strict failure propagation.

        Args:
            record: Model-ready evaluation record.
            top_k: Evidence items retained for each hop.
            candidate_k: Candidate count used by each retrieval channel.

        Returns:
            Complete prediction and diagnostic record.
        """
        started_at = time.perf_counter()
        planner_result = self.planner.generate(record)
        result: dict[str, Any] = {
            "id": record["id"],
            "source_dataset": record.get("source_dataset"),
            "source_id": record.get("source_id"),
            "namespace": record.get("namespace"),
            "question": record["question"],
            "planner": planner_result,
            "steps": [],
            "evidence": [],
            "answer": None,
            "success": False,
            "fallback": False,
            "errors": [],
            "models": {
                "planner": self.planner.model,
                "response": self.answer_generator.model,
                "embedding": self._embedding_model(self.retriever)
            },
            "usage": {
                "planner": planner_result["usage"],
                "intermediate": {},
                "final": {},
                "total": {}
            },
            "latency_seconds": {
                "planner": planner_result["latency_seconds"],
                "retrieval": 0.0,
                "intermediate_answers": 0.0,
                "final_answer": 0.0,
                "total": 0.0
            }
        }
        if not planner_result["schema_valid"]:
            result["errors"].append(
                {"stage": "planner", "error": planner_result["error"]}
            )
            result["usage"]["total"] = _sum_usage([planner_result["usage"]])
            result["latency_seconds"]["total"] = time.perf_counter() - started_at
            return result

        answers: dict[str, str] = {}
        statuses: dict[str, str] = {}
        evidence_rankings: list[tuple[str, list[dict[str, Any]]]] = []
        intermediate_usages = []
        for planned_step in planner_result["parsed_plan"]["queries"]:
            step_started_at = time.perf_counter()
            step_id = planned_step["id"]
            dependencies = list(planned_step["depends_on"])
            step_result: dict[str, Any] = {
                "id": step_id,
                "query": planned_step["query"],
                "depends_on": dependencies,
                "resolved_query": None,
                "status": "pending",
                "evidence": [],
                "answer": None,
                "raw_answer": None,
                "usage": {},
                "latency_seconds": {
                    "retrieval": 0.0,
                    "answer": 0.0,
                    "total": 0.0
                },
                "error": None
            }
            failed_dependencies = [
                dependency
                for dependency in dependencies
                if statuses.get(dependency) != "completed"
            ]
            if failed_dependencies:
                step_result["status"] = "skipped"
                step_result["error"] = (
                    "Dependencies did not complete: " + ", ".join(failed_dependencies)
                )
                statuses[step_id] = "skipped"
                result["errors"].append(
                    {"stage": "step", "step_id": step_id, "error": step_result["error"]}
                )
                step_result["latency_seconds"]["total"] = (
                    time.perf_counter() - step_started_at
                )
                result["steps"].append(step_result)
                continue

            resolved_query = planned_step["query"]
            for dependency in dependencies:
                placeholder = "{{" + dependency + ".answer}}"
                resolved_query = resolved_query.replace(placeholder, answers[dependency])
            step_result["resolved_query"] = resolved_query
            current_stage = "retrieval"
            stage_started_at = time.perf_counter()
            try:
                hits = self.retriever.search(
                    resolved_query,
                    top_k = top_k,
                    candidate_k = candidate_k
                )
                retrieval_latency = time.perf_counter() - stage_started_at
                step_result["latency_seconds"]["retrieval"] = retrieval_latency
                result["latency_seconds"]["retrieval"] += retrieval_latency
                evidence = [
                    hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
                    for hit in hits
                ]
                step_result["evidence"] = evidence
                evidence_rankings.append((step_id, evidence))

                current_stage = "intermediate_answer"
                stage_started_at = time.perf_counter()
                raw_answer, answer_usage = self.answer_generator.generate(
                    resolved_query,
                    evidence
                )
                answer = _clean_intermediate_answer(raw_answer)
                if not answer:
                    raise ValueError("Intermediate answer is empty after citation removal")
                answer_latency = time.perf_counter() - stage_started_at
                step_result["latency_seconds"]["answer"] = answer_latency
                result["latency_seconds"]["intermediate_answers"] += answer_latency
                step_result["raw_answer"] = raw_answer
                step_result["answer"] = answer
                step_result["usage"] = answer_usage
                step_result["status"] = "completed"
                answers[step_id] = answer
                statuses[step_id] = "completed"
                intermediate_usages.append(answer_usage)
            except Exception as error:
                failure_latency = time.perf_counter() - stage_started_at
                if current_stage == "retrieval":
                    step_result["latency_seconds"]["retrieval"] = failure_latency
                    result["latency_seconds"]["retrieval"] += failure_latency
                else:
                    step_result["latency_seconds"]["answer"] = failure_latency
                    result["latency_seconds"]["intermediate_answers"] += failure_latency
                step_result["status"] = "failed"
                step_result["error"] = f"{current_stage}: {error}"
                statuses[step_id] = "failed"
                result["errors"].append(
                    {
                        "stage": current_stage,
                        "step_id": step_id,
                        "error": str(error)
                    }
                )
            step_result["latency_seconds"]["total"] = time.perf_counter() - step_started_at
            result["steps"].append(step_result)

        aggregated_evidence = _fuse_evidence_rankings(evidence_rankings)
        result["evidence"] = aggregated_evidence
        result["usage"]["intermediate"] = _sum_usage(intermediate_usages)
        all_steps_completed = bool(result["steps"]) and all(
            step["status"] == "completed" for step in result["steps"]
        )
        if all_steps_completed:
            try:
                final_started_at = time.perf_counter()
                answer, final_usage = self.answer_generator.generate(
                    self._question_with_scenario(record),
                    aggregated_evidence
                )
                result["latency_seconds"]["final_answer"] = (
                    time.perf_counter() - final_started_at
                )
                result["answer"] = answer
                result["usage"]["final"] = final_usage
                result["success"] = True
            except Exception as error:
                result["errors"].append({"stage": "final_answer", "error": str(error)})

        result["usage"]["total"] = _sum_usage(
            [
                result["usage"]["planner"],
                result["usage"]["intermediate"],
                result["usage"]["final"]
            ]
        )
        result["latency_seconds"]["total"] = time.perf_counter() - started_at
        return result
