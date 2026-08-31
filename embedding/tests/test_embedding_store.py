import os
import sys
import json
import time
import threading
import subprocess
from types import SimpleNamespace

import faiss
import pytest
import torch
import numpy as np

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import read_jsonl, write_jsonl
from embedding.embedding_store import build_namespace_store
from embedding.embedding_store import embed_texts
from embedding.embedding_store import format_embedding_text


def test_format_embedding_text_joins_title_and_body() -> None:
    """Join the knowledge title and body with one blank line."""
    record = {"title": "Policy title", "text": "Policy body"}

    assert format_embedding_text(record) == "Policy title\n\nPolicy body"


class FakeEmbeddings:
    """Return deterministic embedding responses for tests."""

    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        """Return response rows in reverse index order.

        Args:
            kwargs: OpenAI-compatible embedding request fields.

        Returns:
            Response-like object with embeddings and token usage.
        """
        self.calls.append(kwargs)
        return SimpleNamespace(
            data = [
                SimpleNamespace(index = 1, embedding = [0.0, 3.0]),
                SimpleNamespace(index = 0, embedding = [4.0, 0.0])
            ],
            usage = SimpleNamespace(total_tokens = 7)
        )


class FlakyEmbeddings(FakeEmbeddings):
    """Fail one request before returning a valid response."""

    def create(self, **kwargs):
        """Raise once, then delegate to the deterministic response.

        Args:
            kwargs: OpenAI-compatible embedding request fields.

        Returns:
            Response-like object after the first attempt.
        """
        if not self.calls:
            self.calls.append(kwargs)
            raise RuntimeError("temporary failure")
        return super().create(**kwargs)


class ConcurrentEmbeddings:
    """Track concurrent calls and return text-dependent vectors."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, **kwargs):
        """Delay the first batch so completion order differs from input order.

        Args:
            kwargs: OpenAI-compatible embedding request fields.

        Returns:
            Response-like object containing one embedding.
        """
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        text = kwargs["input"][0]
        if text.startswith("Title 0"):
            time.sleep(0.05)
            vector = [1.0, 0.0]
        else:
            time.sleep(0.01)
            vector = [0.0, 1.0]
        with self.lock:
            self.active -= 1
        return SimpleNamespace(
            data = [SimpleNamespace(index = 0, embedding = vector)],
            usage = SimpleNamespace(total_tokens = 1)
        )


class DynamicEmbeddings:
    """Return one deterministic vector for every input row."""

    def __init__(self, fail_call: int | None = None) -> None:
        self.call_count = 0
        self.fail_call = fail_call

    def create(self, **kwargs):
        """Return ordered vectors, optionally failing one numbered call.

        Args:
            kwargs: OpenAI-compatible embedding request fields.

        Returns:
            Response-like object with one row per input.
        """
        self.call_count += 1
        if self.call_count == self.fail_call:
            raise RuntimeError("interrupted batch")
        data = [
            SimpleNamespace(
                index = index,
                embedding = [1.0, 0.0] if "Title 0" in text else [0.0, 1.0]
            )
            for index, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(
            data = data,
            usage = SimpleNamespace(total_tokens = len(data))
        )


def test_embed_texts_restores_order_and_normalizes_vectors() -> None:
    """Restore response order and create unit-length float32 vectors."""
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings = embeddings)

    vectors, token_count = embed_texts(
        client = client,
        texts = ["first", "second"],
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        max_retries = 0
    )

    assert embeddings.calls == [
        {
            "model": "qwen3.7-text-embedding",
            "input": ["first", "second"],
            "dimensions": 2,
            "encoding_format": "float"
        }
    ]
    assert vectors.dtype == np.float32
    assert np.array_equal(vectors, np.array([[1.0, 0.0], [0.0, 1.0]], dtype = np.float32))
    assert token_count == 7


def test_embed_texts_retries_transient_failure() -> None:
    """Retry a failed embedding request without changing its payload."""
    embeddings = FlakyEmbeddings()
    client = SimpleNamespace(embeddings = embeddings)

    vectors, _ = embed_texts(
        client = client,
        texts = ["first", "second"],
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        max_retries = 1,
        sleep_fn = lambda _: None
    )

    assert len(embeddings.calls) == 2
    assert vectors.shape == (2, 2)


def test_embed_texts_rejects_response_dimension_mismatch() -> None:
    """Reject vectors whose dense dimension differs from the request."""
    response = SimpleNamespace(
        data = [SimpleNamespace(index = 0, embedding = [1.0])],
        usage = SimpleNamespace(total_tokens = 1)
    )
    client = SimpleNamespace(
        embeddings = SimpleNamespace(create = lambda **_: response)
    )

    with pytest.raises(ValueError, match = "shape"):
        embed_texts(
            client = client,
            texts = ["text"],
            model = "qwen3.7-text-embedding",
            dimensions = 2,
            max_retries = 0
        )


def test_build_namespace_store_writes_pth_faiss_and_metadata(tmp_path) -> None:
    """Persist aligned tensor, index, metadata, and manifest artifacts."""
    records = [
        {
            "id": "policy_1",
            "title": "First",
            "text": "Body one",
            "source": "source:1",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": "hash-1"
        },
        {
            "id": "policy_2",
            "title": "Second",
            "text": "Body two",
            "source": "source:2",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": "hash-2"
        }
    ]
    source_path = tmp_path / "policy.jsonl"
    output_root = tmp_path / "vector_store"
    write_jsonl(source_path, records)
    client = SimpleNamespace(embeddings = FakeEmbeddings())

    manifest = build_namespace_store(
        namespace = "policy",
        source_path = source_path,
        output_root = output_root,
        client = client,
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        batch_size = 2,
        workers = 1,
        max_retries = 0,
        limit = None,
        resume = False,
        force = False
    )

    namespace_root = output_root / "policy"
    index = faiss.read_index(str(namespace_root / "index.faiss"))
    search_code = (
        "import json, sys; import faiss; import numpy as np; "
        "index = faiss.read_index(sys.argv[1]); "
        "queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype = np.float32); "
        "print(json.dumps(index.search(queries, 1)[1][:, 0].tolist()))"
    )
    search_result = subprocess.run(
        [sys.executable, "-c", search_code, str(namespace_root / "index.faiss")],
        check = True,
        capture_output = True,
        text = True
    )
    vectors = torch.load(namespace_root / "vectors.pth", weights_only = True)
    persisted_manifest = json.loads(
        (namespace_root / "manifest.json").read_text(encoding = "utf-8")
    )

    assert vectors.dtype == torch.float32
    assert tuple(vectors.shape) == (2, 2)
    assert read_jsonl(namespace_root / "metadata.jsonl") == records
    assert index.ntotal == 2
    assert json.loads(search_result.stdout) == [0, 1]
    assert manifest == persisted_manifest
    assert manifest["completed_count"] == 2
    assert manifest["indexed_count"] == 2
    assert manifest["total_tokens"] == 7


def test_build_namespace_store_resumes_without_duplicate_rows(tmp_path) -> None:
    """Continue from a prefix limit without embedding completed records again."""
    records = [
        {
            "id": f"policy_{index}",
            "title": f"Title {index}",
            "text": f"Body {index}",
            "source": f"source:{index}",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": f"hash-{index}"
        }
        for index in range(4)
    ]
    source_path = tmp_path / "policy.jsonl"
    output_root = tmp_path / "vector_store"
    write_jsonl(source_path, records)

    build_namespace_store(
        namespace = "policy",
        source_path = source_path,
        output_root = output_root,
        client = SimpleNamespace(embeddings = FakeEmbeddings()),
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        batch_size = 2,
        workers = 1,
        max_retries = 0,
        limit = 2,
        resume = False,
        force = False
    )
    resumed_embeddings = FakeEmbeddings()
    manifest = build_namespace_store(
        namespace = "policy",
        source_path = source_path,
        output_root = output_root,
        client = SimpleNamespace(embeddings = resumed_embeddings),
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        batch_size = 2,
        workers = 1,
        max_retries = 0,
        limit = 4,
        resume = True,
        force = False
    )

    namespace_root = output_root / "policy"
    vectors = torch.load(namespace_root / "vectors.pth", weights_only = True)

    assert len(resumed_embeddings.calls) == 1
    assert resumed_embeddings.calls[0]["input"] == [
        "Title 2\n\nBody 2",
        "Title 3\n\nBody 3"
    ]
    assert read_jsonl(namespace_root / "metadata.jsonl") == records
    assert tuple(vectors.shape) == (4, 2)
    assert manifest["completed_count"] == 4
    assert manifest["indexed_count"] == 4


def test_build_namespace_store_runs_batches_concurrently_in_source_order(tmp_path) -> None:
    """Use multiple workers while preserving source row alignment."""
    records = [
        {
            "id": f"policy_{index}",
            "title": f"Title {index}",
            "text": f"Body {index}",
            "source": f"source:{index}",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": f"hash-{index}"
        }
        for index in range(2)
    ]
    source_path = tmp_path / "policy.jsonl"
    output_root = tmp_path / "vector_store"
    write_jsonl(source_path, records)
    embeddings = ConcurrentEmbeddings()

    build_namespace_store(
        namespace = "policy",
        source_path = source_path,
        output_root = output_root,
        client = SimpleNamespace(embeddings = embeddings),
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        batch_size = 1,
        workers = 2,
        max_retries = 0,
        limit = None,
        resume = False,
        force = False
    )

    vectors = torch.load(
        output_root / "policy" / "vectors.pth",
        weights_only = True
    )

    assert embeddings.max_active == 2
    assert torch.equal(
        vectors,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype = torch.float32)
    )


def test_build_namespace_store_resumes_failed_run_with_new_batch_size(tmp_path) -> None:
    """Recover committed checkpoint shards independently of the new batch size."""
    records = [
        {
            "id": f"policy_{index}",
            "title": f"Title {index}",
            "text": f"Body {index}",
            "source": f"source:{index}",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": f"hash-{index}"
        }
        for index in range(6)
    ]
    source_path = tmp_path / "policy.jsonl"
    output_root = tmp_path / "vector_store"
    write_jsonl(source_path, records)
    failing_embeddings = DynamicEmbeddings(fail_call = 2)

    with pytest.raises(RuntimeError, match = "interrupted batch"):
        build_namespace_store(
            namespace = "policy",
            source_path = source_path,
            output_root = output_root,
            client = SimpleNamespace(embeddings = failing_embeddings),
            model = "qwen3.7-text-embedding",
            dimensions = 2,
            batch_size = 2,
            workers = 1,
            max_retries = 0,
            limit = None,
            resume = False,
            force = False
        )

    assert failing_embeddings.call_count == 2

    manifest = build_namespace_store(
        namespace = "policy",
        source_path = source_path,
        output_root = output_root,
        client = SimpleNamespace(embeddings = DynamicEmbeddings()),
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        batch_size = 1,
        workers = 1,
        max_retries = 0,
        limit = None,
        resume = True,
        force = False
    )

    namespace_root = output_root / "policy"
    vectors = torch.load(namespace_root / "vectors.pth", weights_only = True)

    assert tuple(vectors.shape) == (6, 2)
    assert read_jsonl(namespace_root / "metadata.jsonl") == records
    assert manifest["completed_count"] == 6


def test_build_namespace_store_requires_explicit_existing_output_action(tmp_path) -> None:
    """Refuse to replace an existing namespace without resume or force."""
    records = [
        {
            "id": "policy_1",
            "title": "Title",
            "text": "Body",
            "source": "source:1",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": "hash-1"
        },
        {
            "id": "policy_2",
            "title": "Title two",
            "text": "Body two",
            "source": "source:2",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": "hash-2"
        }
    ]
    source_path = tmp_path / "policy.jsonl"
    output_root = tmp_path / "vector_store"
    write_jsonl(source_path, records)
    arguments = {
        "namespace": "policy",
        "source_path": source_path,
        "output_root": output_root,
        "client": SimpleNamespace(embeddings = FakeEmbeddings()),
        "model": "qwen3.7-text-embedding",
        "dimensions": 2,
        "batch_size": 2,
        "workers": 1,
        "max_retries": 0,
        "limit": None,
        "resume": False,
        "force": False
    }
    build_namespace_store(**arguments)

    with pytest.raises(FileExistsError, match = "--resume or --force"):
        build_namespace_store(**arguments)


def test_build_namespace_store_rejects_changed_source_on_resume(tmp_path) -> None:
    """Reject resume after the source JSONL content changes."""
    records = [
        {
            "id": "policy_1",
            "title": "Title",
            "text": "Body",
            "source": "source:1",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": "hash-1"
        },
        {
            "id": "policy_2",
            "title": "Title two",
            "text": "Body two",
            "source": "source:2",
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "content_hash": "hash-2"
        }
    ]
    source_path = tmp_path / "policy.jsonl"
    output_root = tmp_path / "vector_store"
    write_jsonl(source_path, records)
    build_namespace_store(
        namespace = "policy",
        source_path = source_path,
        output_root = output_root,
        client = SimpleNamespace(embeddings = FakeEmbeddings()),
        model = "qwen3.7-text-embedding",
        dimensions = 2,
        batch_size = 2,
        workers = 1,
        max_retries = 0,
        limit = None,
        resume = False,
        force = False
    )
    records[0]["text"] = "Changed body"
    write_jsonl(source_path, records)

    with pytest.raises(ValueError, match = "source_sha256"):
        build_namespace_store(
            namespace = "policy",
            source_path = source_path,
            output_root = output_root,
            client = SimpleNamespace(embeddings = FakeEmbeddings()),
            model = "qwen3.7-text-embedding",
            dimensions = 2,
            batch_size = 2,
            workers = 1,
            max_retries = 0,
            limit = None,
            resume = True,
            force = False
        )
