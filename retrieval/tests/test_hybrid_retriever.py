import os
import sys
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import sha256_file, write_jsonl
from retrieval.bm25_store import build_bm25_store
from retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from retrieval.models import QueryStep


class FakeEmbeddings:
    """Return a fixed unit query vector."""

    def create(self, **kwargs):
        """Create one embedding response.

        Args:
            kwargs: OpenAI-compatible embedding request fields.

        Returns:
            Response-like embedding object.
        """
        return SimpleNamespace(
            data = [SimpleNamespace(index = 0, embedding = [1.0, 0.0])],
            usage = SimpleNamespace(total_tokens = 1)
        )


def test_retrieval_import_does_not_load_torch() -> None:
    """Keep online FAISS retrieval isolated from PyTorch's OpenMP runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import retrieval.hybrid_retriever; print('torch' in sys.modules)"
        ],
        check = True,
        capture_output = True,
        text = True
    )

    assert result.stdout.strip() == "False"


def test_rag_cli_import_does_not_load_torch() -> None:
    """Keep the complete online CLI isolated from PyTorch's OpenMP runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import retrieval.run_rag; print('torch' in sys.modules)"
        ],
        check = True,
        capture_output = True,
        text = True
    )

    assert result.stdout.strip() == "False"


def make_store(tmp_path: Path) -> Path:
    """Create aligned dense and sparse test artifacts.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Namespace store path.
    """
    namespace_root = tmp_path / "policy"
    namespace_root.mkdir()
    records = [
        {
            "id": "doc-1",
            "title": "Child Tax Credit",
            "text": "Existing Working Tax Credit claims may continue.",
            "source": "source-1",
            "namespace": "policy"
        },
        {
            "id": "doc-2",
            "title": "Universal Credit",
            "text": "Apply for Universal Credit when a new tax credit claim is unavailable.",
            "source": "source-2",
            "namespace": "policy"
        },
        {
            "id": "doc-3",
            "title": "Pension Credit",
            "text": "Support for people over State Pension age.",
            "source": "source-3",
            "namespace": "policy"
        }
    ]
    metadata_path = namespace_root / "metadata.jsonl"
    write_jsonl(metadata_path, records)
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0]
        ],
        dtype = np.float32
    )
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    faiss.write_index(index, str(namespace_root / "index.faiss"))
    (namespace_root / "manifest.json").write_text(
        json.dumps(
            {
                "namespace": "policy",
                "indexed_count": 3,
                "model": "test-embedding",
                "dimensions": 2
            }
        ),
        encoding = "utf-8"
    )
    build_bm25_store(namespace_root)
    assert sha256_file(metadata_path)
    return namespace_root


def test_reciprocal_rank_fusion_deduplicates_and_uses_one_based_ranks() -> None:
    """Sum reciprocal ranks across channels and order deterministically."""
    fused = reciprocal_rank_fusion(
        [
            [(0, 10.0), (1, 9.0)],
            [(1, 0.9), (2, 0.8)]
        ],
        rrf_k = 60
    )

    assert [item[0] for item in fused] == [1, 0, 2]
    assert fused[0][1] == 1 / 62 + 1 / 61


def test_hybrid_search_returns_channel_provenance(tmp_path) -> None:
    """Fuse BM25 and embedding results with aligned metadata."""
    retriever = HybridRetriever(
        namespace_root = make_store(tmp_path),
        embedding_client = SimpleNamespace(embeddings = FakeEmbeddings()),
        candidate_k = 3
    )

    hits = retriever.search("Universal Credit claim", top_k = 2)

    assert len(hits) == 2
    assert hits[0].record["id"] in {"doc-1", "doc-2"}
    assert hits[0].rrf_score > 0
    assert hits[0].query_ids == ["q1"]
    assert any(hit.bm25_rank is not None for hit in hits)
    assert all(hit.embedding_rank is not None for hit in hits)


def test_hybrid_search_displays_tqdm_progress_when_enabled(tmp_path, capsys) -> None:
    """Report BM25, embedding, and RRF completion without polluting stdout."""
    retriever = HybridRetriever(
        namespace_root = make_store(tmp_path),
        embedding_client = SimpleNamespace(embeddings = FakeEmbeddings()),
        candidate_k = 3,
        show_progress = True
    )

    retriever.search("Universal Credit claim", top_k = 2)
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "Hybrid retrieval" in captured.err
    assert "3/3" in captured.err


def test_search_many_fuses_resolved_query_steps(tmp_path) -> None:
    """Expose a working interface for independent multi-query plans."""
    retriever = HybridRetriever(
        namespace_root = make_store(tmp_path),
        embedding_client = SimpleNamespace(embeddings = FakeEmbeddings()),
        candidate_k = 3
    )

    hits = retriever.search_many(
        [
            QueryStep(id = "q1", query = "Child Tax Credit"),
            QueryStep(id = "q2", query = "Universal Credit")
        ],
        top_k = 3
    )

    assert len(hits) == 3
    assert any(hit.query_ids == ["q1", "q2"] for hit in hits)


def test_search_many_rejects_unresolved_dependencies(tmp_path) -> None:
    """Require future multi-hop executors to resolve placeholders explicitly."""
    retriever = HybridRetriever(
        namespace_root = make_store(tmp_path),
        embedding_client = SimpleNamespace(embeddings = FakeEmbeddings()),
        candidate_k = 3
    )

    try:
        retriever.search_many(
            [
                QueryStep(
                    id = "q2",
                    query = "{{q1.answer}} eligibility",
                    depends_on = ("q1",)
                )
            ]
        )
    except ValueError as error:
        assert "unresolved dependencies" in str(error)
    else:
        raise AssertionError("Expected unresolved dependency error")
