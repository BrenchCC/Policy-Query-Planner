import os
import sys
import json
from types import SimpleNamespace

# Add project root to Python path
sys.path.append(os.getcwd())

from retrieval.models import QueryPlan, QueryStep, RetrievalHit
from retrieval.rag_pipeline import AnswerGenerator, QueryRewriter, RAGPipeline
from retrieval.rag_pipeline import plan_from_json


class FakeChatCompletions:
    """Return queued chat responses and record requests."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        """Return the next configured response.

        Args:
            kwargs: OpenAI-compatible chat request fields.

        Returns:
            Response-like chat object.
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
    """Return one deterministic fused result."""

    def __init__(self) -> None:
        self.vector_manifest = {"namespace": "policy", "model": "embedding-model"}
        self.calls = []

    def search_many(self, queries, top_k, candidate_k):
        """Record a search and return one hit."""
        self.calls.append((queries, top_k, candidate_k))
        return [
            RetrievalHit(
                record = {
                    "id": "doc-1",
                    "title": "Policy",
                    "text": "Grounded fact.",
                    "source": "source-1"
                },
                rrf_score = 0.03,
                bm25_rank = 1,
                embedding_rank = 2,
                query_ids = ["q1"]
            )
        ]


def test_query_rewriter_uses_history_and_query_model() -> None:
    """Use the configured model and QReCC-compatible history format."""
    completions = FakeChatCompletions(
        ['{"queries":[{"id":"q1","query":"standalone query","depends_on":[]}]}']
    )
    rewriter = QueryRewriter(
        client = SimpleNamespace(chat = SimpleNamespace(completions = completions)),
        model = "query-model"
    )

    plan, usage = rewriter.rewrite("What about it?", history = "User: Policy A")

    assert plan.queries[0].query == "standalone query"
    assert completions.calls[0]["model"] == "query-model"
    assert "Conversation history:\nUser: Policy A" in completions.calls[0]["messages"][1]["content"]
    assert usage["total_tokens"] == 5


def test_pipeline_falls_back_and_returns_answer_with_evidence() -> None:
    """Continue with the original query after invalid rewrite output."""
    completions = FakeChatCompletions(["not json", "Supported answer [1]"])
    client = SimpleNamespace(chat = SimpleNamespace(completions = completions))
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        retriever = retriever,
        rewriter = QueryRewriter(client = client, model = "query-model"),
        answer_generator = AnswerGenerator(client = client, model = "response-model")
    )

    result = pipeline.run("Original question", top_k = 1, candidate_k = 2)

    assert result["rewrite_fallback"] is True
    assert result["executed_queries"] == ["Original question"]
    assert result["answer"] == "Supported answer [1]"
    assert result["evidence"][0]["id"] == "doc-1"
    assert result["models"]["query"] == "query-model"
    assert completions.calls[1]["model"] == "response-model"
    assert "[1]" in completions.calls[1]["messages"][1]["content"]


def test_run_plan_supports_independent_multi_query_interface() -> None:
    """Execute a caller-supplied independent multi-query plan."""
    completions = FakeChatCompletions(["Answer [1]"])
    client = SimpleNamespace(chat = SimpleNamespace(completions = completions))
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        retriever = retriever,
        rewriter = QueryRewriter(client = client, model = "query-model"),
        answer_generator = AnswerGenerator(client = client, model = "response-model")
    )
    plan = QueryPlan(
        queries = (
            QueryStep(id = "q1", query = "first"),
            QueryStep(id = "q2", query = "second")
        )
    )

    result = pipeline.run_plan("Question", plan, top_k = 1, candidate_k = 2)

    assert result["executed_queries"] == ["first", "second"]
    assert len(retriever.calls[0][0]) == 2


def test_plan_from_json_preserves_dependency_contract() -> None:
    """Preserve dependent plan steps for a future multi-hop executor."""
    plan = plan_from_json(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "query": "office name", "depends_on": []},
                    {
                        "id": "q2",
                        "query": "{{q1.answer}} eligibility",
                        "depends_on": ["q1"]
                    }
                ]
            }
        )
    )

    assert plan.queries[1].depends_on == ("q1",)
