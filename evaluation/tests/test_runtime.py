import os
import sys
import json
from types import SimpleNamespace

# Add project root to Python path / 将项目根目录加入 Python 路径
sys.path.append(os.getcwd())

from evaluation.runtime import ChainExecutor, PlannerGenerator
from evaluation.runtime import _clean_intermediate_answer, _fuse_evidence_rankings
from retrieval.models import RetrievalHit


class FakeChatCompletions:
    """Return queued planner responses and record requests."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        """Return one OpenAI-compatible completion.

        Args:
            kwargs: Chat-completion request fields.

        Returns:
            Response-like object with deterministic usage.
        """
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices = [SimpleNamespace(message = SimpleNamespace(content = content))],
            usage = SimpleNamespace(
                prompt_tokens = 3,
                completion_tokens = 2,
                total_tokens = 5
            )
        )


class FakeRetriever:
    """Return deterministic evidence or fail for selected query text."""

    def __init__(self, failing_queries: set[str] | None = None) -> None:
        self.vector_manifest = {"namespace": "musique_aux", "model": "embed-model"}
        self.failing_queries = failing_queries or set()
        self.calls = []

    def search(self, query, top_k, candidate_k):
        """Return one query-specific retrieval hit."""
        self.calls.append((query, top_k, candidate_k))
        if query in self.failing_queries:
            raise RuntimeError(f"retrieval failed: {query}")
        document_id = "shared-doc" if "second" in query else "first-doc"
        return [
            RetrievalHit(
                record = {"id": document_id, "title": query, "text": "Evidence"},
                rrf_score = 0.03,
                query_ids = ["q1"]
            )
        ]


class FakeAnswerGenerator:
    """Generate deterministic intermediate and final answers."""

    def __init__(self, failing_queries: set[str] | None = None) -> None:
        self.model = "answer-model"
        self.failing_queries = failing_queries or set()
        self.calls = []

    def generate(self, query, evidence):
        """Return an answer based on the query or raise a configured error."""
        self.calls.append((query, evidence))
        if query in self.failing_queries:
            raise RuntimeError(f"answer failed: {query}")
        if query == "first entity":
            return "Alpha", {"total_tokens": 7}
        if query == "Alpha second fact":
            return "Beta", {"total_tokens": 8}
        return "Final answer", {"total_tokens": 9}


def make_record() -> dict:
    """Create one model-ready multi-hop evaluation record."""
    return {
        "schema_version": "1.0",
        "id": "eval-1",
        "source_dataset": "musique",
        "source_id": "source-1",
        "namespace": "musique_aux",
        "system": "System prompt",
        "instruction": "Create a plan",
        "input": "Question: Find the answer",
        "question": "Find the answer",
        "reference_answers": ["Final answer"],
        "gold_doc_ids": ["first-doc", "shared-doc"],
        "answerable": True,
        "metadata": {},
        "hop_count": 2,
        "reference_plan": {"queries": []},
        "reference_steps": []
    }


def make_planner(content: str) -> tuple[PlannerGenerator, FakeChatCompletions]:
    """Create a planner around one queued response."""
    completions = FakeChatCompletions([content])
    client = SimpleNamespace(chat = SimpleNamespace(completions = completions))
    return PlannerGenerator(client = client, model = "planner-model"), completions


def test_planner_generator_uses_record_prompts_and_validates_plan() -> None:
    """Preserve the dataset prompt and accept fenced valid multi-hop JSON."""
    plan = {
        "queries": [
            {"id": "q1", "query": "first entity", "depends_on": []},
            {
                "id": "q2",
                "query": "{{q1.answer}} second fact",
                "depends_on": ["q1"]
            }
        ]
    }
    planner, completions = make_planner(f"```json\n{json.dumps(plan)}\n```")

    result = planner.generate(make_record())

    assert result["parse_success"] is True
    assert result["schema_valid"] is True
    assert result["parsed_plan"] == plan
    assert result["usage"]["total_tokens"] == 5
    messages = completions.calls[0]["messages"]
    assert messages[0]["content"] == "System prompt"
    assert messages[1]["content"] == "Create a plan\n\nQuestion: Find the answer"


def test_planner_generator_distinguishes_json_and_schema_failures() -> None:
    """Expose parse and schema validity separately for scoring."""
    invalid_json_planner, _ = make_planner("not-json")
    invalid_schema_planner, _ = make_planner('{"queries": []}')

    invalid_json = invalid_json_planner.generate(make_record())
    invalid_schema = invalid_schema_planner.generate(make_record())

    assert invalid_json["parse_success"] is False
    assert invalid_json["schema_valid"] is False
    assert invalid_schema["parse_success"] is True
    assert invalid_schema["schema_valid"] is False


def test_planner_generator_records_empty_model_output_as_parse_failure() -> None:
    """Treat an empty Planner response as a visible parse failure."""
    planner, _ = make_planner("")

    result = planner.generate(make_record())

    assert result["raw_output"] == ""
    assert result["parse_success"] is False
    assert result["schema_valid"] is False
    assert result["error"]


def test_chain_evidence_uses_cross_step_rrf_ranking() -> None:
    """Interleave equal-ranked evidence instead of placing every first-step hit first."""
    fused = _fuse_evidence_rankings(
        [
            ("q1", [{"id": "a"}, {"id": "b"}]),
            ("q2", [{"id": "c"}, {"id": "d"}])
        ]
    )

    assert [item["id"] for item in fused] == ["a", "c", "b", "d"]
    assert fused[1]["query_ids"] == ["q2"]
    assert fused[0]["chain_rrf_score"] > fused[2]["chain_rrf_score"]


def test_intermediate_answer_citations_do_not_leak_into_next_query() -> None:
    """Remove evidence citation markers while preserving the raw diagnostic answer."""
    assert _clean_intermediate_answer("Steve Hillage [1]") == "Steve Hillage"
    assert _clean_intermediate_answer("Miquette [1, 2]") == "Miquette"


def test_chain_executor_resolves_dependencies_and_generates_final_answer() -> None:
    """Retrieve and answer each hop before replacing dependent placeholders."""
    plan = {
        "queries": [
            {"id": "q1", "query": "first entity", "depends_on": []},
            {
                "id": "q2",
                "query": "{{q1.answer}} second fact",
                "depends_on": ["q1"]
            }
        ]
    }
    planner, _ = make_planner(json.dumps(plan))
    retriever = FakeRetriever()
    answers = FakeAnswerGenerator()
    executor = ChainExecutor(
        retriever = retriever,
        planner = planner,
        answer_generator = answers
    )

    result = executor.run(make_record(), top_k = 5, candidate_k = 12)

    assert [call[0] for call in retriever.calls] == ["first entity", "Alpha second fact"]
    assert result["steps"][1]["resolved_query"] == "Alpha second fact"
    assert result["steps"][1]["answer"] == "Beta"
    assert result["answer"] == "Final answer"
    assert result["success"] is True
    assert result["fallback"] is False
    assert result["models"]["embedding"] == "embed-model"
    assert result["usage"]["total"]["total_tokens"] == 29


def test_chain_executor_skips_dependents_after_failure_without_fallback() -> None:
    """Do not execute dependent or final answers after a failed hop."""
    plan = {
        "queries": [
            {"id": "q1", "query": "first entity", "depends_on": []},
            {
                "id": "q2",
                "query": "{{q1.answer}} second fact",
                "depends_on": ["q1"]
            },
            {"id": "q3", "query": "independent query", "depends_on": []}
        ]
    }
    planner, _ = make_planner(json.dumps(plan))
    retriever = FakeRetriever(failing_queries = {"first entity"})
    answers = FakeAnswerGenerator()
    executor = ChainExecutor(
        retriever = retriever,
        planner = planner,
        answer_generator = answers
    )

    result = executor.run(make_record())

    assert [step["status"] for step in result["steps"]] == [
        "failed",
        "skipped",
        "completed"
    ]
    assert [call[0] for call in retriever.calls] == ["first entity", "independent query"]
    assert result["answer"] is None
    assert result["success"] is False
    assert result["fallback"] is False


def test_chain_executor_propagates_intermediate_answer_failure() -> None:
    """Skip dependent hops when evidence exists but intermediate answering fails."""
    plan = {
        "queries": [
            {"id": "q1", "query": "first entity", "depends_on": []},
            {
                "id": "q2",
                "query": "{{q1.answer}} second fact",
                "depends_on": ["q1"]
            }
        ]
    }
    planner, _ = make_planner(json.dumps(plan))
    retriever = FakeRetriever()
    answers = FakeAnswerGenerator(failing_queries = {"first entity"})
    executor = ChainExecutor(
        retriever = retriever,
        planner = planner,
        answer_generator = answers
    )

    result = executor.run(make_record())

    assert [step["status"] for step in result["steps"]] == ["failed", "skipped"]
    assert result["errors"][0]["stage"] == "intermediate_answer"
    assert result["answer"] is None
    assert result["success"] is False


def test_chain_executor_resolves_multiple_dependencies() -> None:
    """Replace every declared upstream answer in a multi-parent query."""
    plan = {
        "queries": [
            {"id": "q1", "query": "first entity", "depends_on": []},
            {"id": "q2", "query": "independent query", "depends_on": []},
            {
                "id": "q3",
                "query": "{{q1.answer}} and {{q2.answer}} second fact",
                "depends_on": ["q1", "q2"]
            }
        ]
    }
    planner, _ = make_planner(json.dumps(plan))
    retriever = FakeRetriever()
    answers = FakeAnswerGenerator()
    executor = ChainExecutor(
        retriever = retriever,
        planner = planner,
        answer_generator = answers
    )

    result = executor.run(make_record())

    assert result["steps"][2]["resolved_query"] == "Alpha and Final answer second fact"
    assert result["steps"][2]["status"] == "completed"
    assert result["success"] is True


def test_chain_executor_stops_before_retrieval_for_invalid_plan() -> None:
    """Return planner diagnostics without silently querying the original question."""
    planner, _ = make_planner("not-json")
    retriever = FakeRetriever()
    executor = ChainExecutor(
        retriever = retriever,
        planner = planner,
        answer_generator = FakeAnswerGenerator()
    )

    result = executor.run(make_record())

    assert retriever.calls == []
    assert result["steps"] == []
    assert result["errors"][0]["stage"] == "planner"
    assert result["fallback"] is False
