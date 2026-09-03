import re
import json
import time
import sqlite3
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from data_preprocess.common import read_jsonl, sha256_file
from retrieval.models import QueryStep, RetrievalHit

logger = logging.getLogger(__name__)


def _embed_query(
    client: Any,
    query: str,
    model: str,
    dimensions: int,
    max_retries: int = 2
) -> np.ndarray:
    """Embed and normalize one query without importing PyTorch.

    Args:
        client: OpenAI-compatible client.
        query: Query text sent to the embedding endpoint.
        model: Embedding model name.
        dimensions: Required dense vector dimension.
        max_retries: Number of retries after the initial request.

    Returns:
        One ordered unit-length float32 query vector.
    """
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = client.embeddings.create(
                model = model,
                input = [query],
                dimensions = dimensions,
                encoding_format = "float"
            )
            break
        except Exception:
            if attempt >= max_retries:
                raise
            delay = float(2 ** attempt)
            logger.warning(
                "Embedding request failed; retrying attempt=%d/%d delay=%.1fs",
                attempt + 1,
                max_retries,
                delay
            )
            time.sleep(delay)
    if response is None:
        raise RuntimeError("Embedding request completed without a response")
    rows = sorted(response.data, key = lambda item: item.index)
    if [item.index for item in rows] != [0]:
        raise ValueError("Embedding response index does not match the query")
    vector = np.asarray([rows[0].embedding], dtype = np.float32)
    if vector.shape != (1, dimensions):
        raise ValueError(
            f"Embedding response shape {vector.shape} does not match (1, {dimensions})"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Embedding response contains non-finite values")
    norm = np.linalg.norm(vector, axis = 1, keepdims = True)
    if np.any(norm == 0):
        raise ValueError("Embedding response contains a zero vector")
    vector /= norm
    return vector


def _fts_query(query: str) -> str:
    """Convert free text into a safe disjunctive FTS5 query.

    Args:
        query: User or rewritten search text.

    Returns:
        FTS5 expression containing quoted unique terms.
    """
    terms = []
    seen = set()
    for term in re.findall(r"\w+", query.lower(), flags = re.UNICODE):
        if term not in seen:
            seen.add(term)
            terms.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(terms)


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    rrf_k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked document lists using reciprocal rank fusion.

    Args:
        rankings: Ranked lists of document index and raw score pairs.
        rrf_k: Positive rank-offset constant.

    Returns:
        Document indices and fused scores in deterministic order.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    for ranking in rankings:
        for rank, (document_index, _) in enumerate(ranking, start = 1):
            scores[document_index] = scores.get(document_index, 0.0) + 1.0 / (
                rrf_k + rank
            )
            best_ranks[document_index] = min(best_ranks.get(document_index, rank), rank)
    return sorted(
        scores.items(),
        key = lambda item: (-item[1], best_ranks[item[0]], item[0])
    )


class HybridRetriever:
    """Search one namespace with BM25, dense vectors, and RRF fusion."""

    def __init__(
        self,
        namespace_root: Path,
        embedding_client: Any,
        candidate_k: int = 50,
        rrf_k: int = 60,
        show_progress: bool = False
    ) -> None:
        """Load and validate retrieval artifacts.

        Args:
            namespace_root: Directory containing vector and BM25 artifacts.
            embedding_client: OpenAI-compatible client with embeddings.create.
            candidate_k: Default candidates retrieved from each channel.
            rrf_k: Reciprocal-rank-fusion offset.
            show_progress: Whether to display retrieval progress on stderr.
        """
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.namespace_root = namespace_root
        self.embedding_client = embedding_client
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.show_progress = show_progress
        self.metadata_path = namespace_root / "metadata.jsonl"
        self.database_path = namespace_root / "bm25.sqlite3"
        self.records = read_jsonl(self.metadata_path)
        self.vector_manifest = json.loads(
            (namespace_root / "manifest.json").read_text(encoding = "utf-8")
        )
        self.bm25_manifest = json.loads(
            (namespace_root / "bm25_manifest.json").read_text(encoding = "utf-8")
        )
        self.index = faiss.read_index(str(namespace_root / "index.faiss"))
        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        """Ensure sparse, dense, and metadata artifacts are aligned."""
        record_count = len(self.records)
        if self.index.ntotal != record_count:
            raise ValueError("FAISS index and metadata counts do not match")
        if int(self.vector_manifest["indexed_count"]) != record_count:
            raise ValueError("Vector manifest and metadata counts do not match")
        if int(self.bm25_manifest["document_count"]) != record_count:
            raise ValueError("BM25 manifest and metadata counts do not match")
        if self.bm25_manifest["metadata_sha256"] != sha256_file(self.metadata_path):
            raise ValueError("BM25 index was built from different metadata")
        if self.index.d != int(self.vector_manifest["dimensions"]):
            raise ValueError("FAISS dimensions and vector manifest do not match")

    def _search_bm25(self, query: str, candidate_k: int) -> list[tuple[int, float]]:
        """Search SQLite FTS5 and return zero-based document indices."""
        expression = _fts_query(query)
        if not expression:
            return []
        connection = sqlite3.connect(f"file:{self.database_path}?mode=ro", uri = True)
        try:
            rows = connection.execute(
                "SELECT rowid, bm25(chunks, 0.0, 2.0, 1.0) AS score "
                "FROM chunks WHERE chunks MATCH ? ORDER BY score ASC, rowid ASC LIMIT ?",
                (expression, candidate_k)
            ).fetchall()
        finally:
            connection.close()
        return [(int(row_id) - 1, float(score)) for row_id, score in rows]

    def _search_embedding(self, query: str, candidate_k: int) -> list[tuple[int, float]]:
        """Embed and search one query against the FAISS index."""
        vectors = _embed_query(
            client = self.embedding_client,
            query = query,
            model = self.vector_manifest["model"],
            dimensions = int(self.vector_manifest["dimensions"]),
            max_retries = 2
        )
        distances, indices = self.index.search(
            np.ascontiguousarray(vectors, dtype = np.float32),
            min(candidate_k, len(self.records))
        )
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], distances[0])
            if index >= 0
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int | None = None
    ) -> list[RetrievalHit]:
        """Run two-channel retrieval and fuse one query.

        Args:
            query: Standalone retrieval query.
            top_k: Number of fused documents to return.
            candidate_k: Optional per-channel candidate count override.

        Returns:
            Fused retrieval hits.
        """
        step = QueryStep(id = "q1", query = query)
        return self.search_many([step], top_k = top_k, candidate_k = candidate_k)

    def search_many(
        self,
        queries: list[QueryStep],
        top_k: int = 10,
        candidate_k: int | None = None
    ) -> list[RetrievalHit]:
        """Retrieve and fuse multiple resolved query steps.

        Args:
            queries: Query steps whose placeholders have already been resolved.
            top_k: Number of fused documents to return.
            candidate_k: Optional per-channel candidate count override.

        Returns:
            Documents fused across every query and retrieval channel.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not queries:
            raise ValueError("queries cannot be empty")
        selected_candidate_k = candidate_k or self.candidate_k
        if selected_candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        rankings = []
        channel_details: dict[int, dict[str, Any]] = {}
        with tqdm(
            total = len(queries) * 2 + 1,
            desc = "Hybrid retrieval",
            unit = "stage",
            dynamic_ncols = True,
            disable = not self.show_progress
        ) as progress:
            for step in queries:
                if step.depends_on:
                    raise ValueError(
                        f"Query {step.id} still has unresolved dependencies: {step.depends_on}"
                    )
                progress.set_postfix_str(f"{step.id}: BM25")
                bm25_results = self._search_bm25(step.query, selected_candidate_k)
                progress.update(1)
                progress.set_postfix_str(f"{step.id}: Embedding")
                embedding_results = self._search_embedding(step.query, selected_candidate_k)
                progress.update(1)
                rankings.extend([bm25_results, embedding_results])
                for channel, results in (
                    ("bm25", bm25_results),
                    ("embedding", embedding_results)
                ):
                    for rank, (document_index, score) in enumerate(results, start = 1):
                        details = channel_details.setdefault(
                            document_index,
                            {"query_ids": []}
                        )
                        if step.id not in details["query_ids"]:
                            details["query_ids"].append(step.id)
                        rank_key = f"{channel}_rank"
                        score_key = f"{channel}_score"
                        if rank_key not in details or rank < details[rank_key]:
                            details[rank_key] = rank
                            details[score_key] = score
            progress.set_postfix_str("RRF fusion")
            fused = reciprocal_rank_fusion(rankings, rrf_k = self.rrf_k)
            progress.update(1)
        hits = []
        for document_index, rrf_score in fused[:top_k]:
            details = channel_details[document_index]
            hits.append(
                RetrievalHit(
                    record = self.records[document_index],
                    rrf_score = rrf_score,
                    bm25_rank = details.get("bm25_rank"),
                    bm25_score = details.get("bm25_score"),
                    embedding_rank = details.get("embedding_rank"),
                    embedding_score = details.get("embedding_score"),
                    query_ids = details["query_ids"]
                )
            )
        return hits
