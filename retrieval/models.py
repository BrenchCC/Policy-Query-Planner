from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen = True)
class QueryStep:
    """Represent one query in a potentially multi-query retrieval plan.

    Args:
        id: Stable query identifier such as q1.
        query: Search text, optionally containing dependency placeholders.
        depends_on: Earlier query identifiers required by this step.
    """

    id: str
    query: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen = True)
class QueryPlan:
    """Represent an ordered retrieval plan.

    Args:
        queries: Ordered query steps.
    """

    queries: tuple[QueryStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the plan to its public JSON-compatible shape."""
        return {
            "queries": [
                {
                    "id": step.id,
                    "query": step.query,
                    "depends_on": list(step.depends_on)
                }
                for step in self.queries
            ]
        }


@dataclass
class RetrievalHit:
    """Store one fused retrieval result and its provenance.

    Args:
        record: Knowledge-base metadata for the document.
        rrf_score: Reciprocal-rank-fusion score.
        bm25_rank: Optional one-based BM25 rank.
        bm25_score: Optional raw SQLite BM25 score.
        embedding_rank: Optional one-based dense rank.
        embedding_score: Optional cosine similarity.
        query_ids: Query steps that retrieved this document.
    """

    record: dict[str, Any]
    rrf_score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    embedding_rank: int | None = None
    embedding_score: float | None = None
    query_ids: list[str] = field(default_factory = list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the hit to a JSON-compatible dictionary."""
        serialized = asdict(self)
        value = serialized.pop("record")
        value.update(serialized)
        return value
