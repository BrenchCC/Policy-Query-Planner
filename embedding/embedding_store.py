import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import torch
import numpy as np
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import read_jsonl, sha256_file
from data_preprocess.prompts import EMBEDDING_TEXT_TEMPLATE

logger = logging.getLogger(__name__)

TEXT_TEMPLATE_ID = EMBEDDING_TEXT_TEMPLATE.replace("\n", "\\n")


def format_embedding_text(record: dict[str, Any]) -> str:
    """Build the text sent to the embedding model.

    Args:
        record: Knowledge-base record containing title and text fields.

    Returns:
        Title and body separated by one blank line.
    """
    return EMBEDDING_TEXT_TEMPLATE.format(
        title = record["title"],
        text = record["text"]
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file.

    Args:
        path: Destination JSON path.
        value: JSON-serializable dictionary.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding = "utf-8") as file:
        json.dump(value, file, ensure_ascii = False, indent = 2)
        file.write("\n")
    temporary_path.replace(path)


def _write_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    """Write a tensor through a sibling temporary file.

    Args:
        path: Destination PTH path.
        tensor: CPU tensor to serialize.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary_path)
    temporary_path.replace(path)


def _write_index_atomic(path: Path, vectors: torch.Tensor) -> None:
    """Build and atomically persist an exact inner-product index.

    Args:
        path: Destination FAISS index path.
        vectors: Ordered unit-length float32 vectors.
    """
    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    if vectors.shape[0]:
        index.add(np.ascontiguousarray(vectors.numpy(), dtype = np.float32))
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary_path))
    temporary_path.replace(path)


def _validate_manifest(
    manifest: dict[str, Any],
    namespace: str,
    source_hash: str,
    model: str,
    dimensions: int
) -> None:
    """Ensure an existing checkpoint matches the requested build.

    Args:
        manifest: Existing checkpoint manifest.
        namespace: Requested knowledge namespace.
        source_hash: Current source JSONL SHA256.
        model: Requested embedding model.
        dimensions: Requested embedding dimension.

    Raises:
        ValueError: If an immutable build field changed.
    """
    expected = {
        "namespace": namespace,
        "source_sha256": source_hash,
        "model": model,
        "dimensions": dimensions,
        "text_template": TEXT_TEMPLATE_ID
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "Cannot resume because manifest fields changed: " + ", ".join(mismatches)
        )


def _truncate_metadata(path: Path, completed_count: int) -> None:
    """Trim metadata rows beyond the committed manifest boundary.

    Args:
        path: Metadata JSONL path.
        completed_count: Number of committed rows to retain.
    """
    records = read_jsonl(path) if path.exists() else []
    if len(records) < completed_count:
        raise ValueError("Metadata contains fewer rows than the manifest checkpoint")
    if len(records) == completed_count:
        return
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding = "utf-8") as file:
        for record in records[:completed_count]:
            file.write(json.dumps(record, ensure_ascii = False) + "\n")
    temporary_path.replace(path)


def embed_texts(
    client: Any,
    texts: list[str],
    model: str,
    dimensions: int,
    max_retries: int,
    sleep_fn: Callable[[float], None] = time.sleep
) -> tuple[np.ndarray, int]:
    """Embed and normalize one ordered text batch.

    Args:
        client: OpenAI-compatible client.
        texts: Ordered input texts.
        model: Embedding model name.
        dimensions: Required dense vector dimension.
        max_retries: Number of retries after the initial request.
        sleep_fn: Delay function used between retries.

    Returns:
        Ordered unit vectors and reported input token count.

    Raises:
        ValueError: If the response shape, indices, or values are invalid.
    """
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = client.embeddings.create(
                model = model,
                input = texts,
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
            sleep_fn(delay)
    if response is None:
        raise RuntimeError("Embedding request completed without a response")
    rows = sorted(response.data, key = lambda item: item.index)
    indices = [item.index for item in rows]
    if indices != list(range(len(texts))):
        raise ValueError("Embedding response indices do not match the input batch")
    vectors = np.asarray([item.embedding for item in rows], dtype = np.float32)
    if vectors.shape != (len(texts), dimensions):
        raise ValueError(
            f"Embedding response shape {vectors.shape} does not match "
            f"({len(texts)}, {dimensions})"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding response contains non-finite values")
    norms = np.linalg.norm(vectors, axis = 1, keepdims = True)
    if np.any(norms == 0):
        raise ValueError("Embedding response contains a zero vector")
    vectors /= norms
    usage = getattr(response, "usage", None)
    token_count = int(getattr(usage, "total_tokens", 0) or 0)
    return vectors, token_count


def build_namespace_store(
    namespace: str,
    source_path: Path,
    output_root: Path,
    client: Any,
    model: str,
    dimensions: int,
    batch_size: int,
    workers: int,
    max_retries: int,
    limit: int | None,
    resume: bool,
    force: bool,
    show_progress: bool = False
) -> dict[str, Any]:
    """Build or resume one namespace-specific embedding store.

    Args:
        namespace: Knowledge namespace name.
        source_path: Ordered knowledge-base JSONL path.
        output_root: Parent directory for namespace artifacts.
        client: OpenAI-compatible client.
        model: Embedding model name.
        dimensions: Dense embedding dimension.
        batch_size: Texts submitted per API request.
        workers: Requested API worker count.
        max_retries: Retries after each initial API request.
        limit: Optional source-prefix target count.
        resume: Whether to continue an existing checkpoint.
        force: Whether to remove an existing namespace store.
        show_progress: Whether to display batch completion progress.

    Returns:
        Final persisted manifest.

    Raises:
        FileExistsError: If output exists without resume or force.
        ValueError: If arguments, source records, or checkpoint are invalid.
    """
    if batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if resume and force:
        raise ValueError("resume and force cannot be used together")

    records = read_jsonl(source_path)
    invalid_namespaces = [
        record.get("id", "<missing>")
        for record in records
        if record.get("namespace") != namespace
    ]
    if invalid_namespaces:
        raise ValueError(
            f"Source contains records outside namespace {namespace}: "
            f"{invalid_namespaces[:3]}"
        )
    target_count = len(records) if limit is None else min(limit, len(records))
    namespace_root = output_root / namespace
    if namespace_root.exists() and force:
        shutil.rmtree(namespace_root)
    if namespace_root.exists() and not resume:
        raise FileExistsError(
            f"Embedding store already exists: {namespace_root}; use --resume or --force"
        )
    namespace_root.mkdir(parents = True, exist_ok = True)

    manifest_path = namespace_root / "manifest.json"
    metadata_path = namespace_root / "metadata.jsonl"
    vectors_path = namespace_root / "vectors.pth"
    index_path = namespace_root / "index.faiss"
    checkpoint_root = namespace_root / ".checkpoints"
    checkpoint_root.mkdir(exist_ok = True)
    source_hash = sha256_file(source_path)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
        _validate_manifest(
            manifest = manifest,
            namespace = namespace,
            source_hash = source_hash,
            model = model,
            dimensions = dimensions
        )
    else:
        manifest = {
            "namespace": namespace,
            "source_path": str(source_path.resolve()),
            "source_sha256": source_hash,
            "source_count": len(records),
            "model": model,
            "dimensions": dimensions,
            "text_template": TEXT_TEMPLATE_ID,
            "completed_count": 0,
            "indexed_count": 0,
            "total_tokens": 0
        }
        _write_json_atomic(manifest_path, manifest)

    completed_count = int(manifest["completed_count"])
    indexed_count = int(manifest["indexed_count"])
    if completed_count > target_count:
        raise ValueError(
            f"Existing checkpoint has {completed_count} rows, above requested limit {target_count}"
        )
    _truncate_metadata(metadata_path, completed_count)

    tensor_parts = []
    if indexed_count:
        if not vectors_path.exists():
            raise ValueError("Manifest references a missing vectors.pth artifact")
        indexed_vectors = torch.load(vectors_path, map_location = "cpu", weights_only = True)
        if (
            indexed_vectors.ndim != 2
            or indexed_vectors.shape[0] < indexed_count
            or indexed_vectors.shape[1] != dimensions
        ):
            raise ValueError("vectors.pth shape does not match the manifest")
        tensor_parts.append(
            indexed_vectors[:indexed_count].to(dtype = torch.float32)
        )
    checkpoint_cursor = indexed_count
    for checkpoint_path in sorted(checkpoint_root.glob("*.pth")):
        checkpoint_start = int(checkpoint_path.stem)
        if checkpoint_start < indexed_count:
            continue
        if checkpoint_start >= completed_count:
            checkpoint_path.unlink()
            continue
        if checkpoint_start != checkpoint_cursor:
            raise ValueError(
                f"Embedding checkpoints are not contiguous at row {checkpoint_cursor}"
            )
        checkpoint_tensor = torch.load(
            checkpoint_path,
            map_location = "cpu",
            weights_only = True
        )
        if checkpoint_tensor.ndim != 2 or checkpoint_tensor.shape[1] != dimensions:
            raise ValueError(f"Invalid embedding checkpoint shape: {checkpoint_path}")
        checkpoint_cursor += int(checkpoint_tensor.shape[0])
        if checkpoint_cursor > completed_count:
            raise ValueError("Embedding checkpoint exceeds the manifest boundary")
        tensor_parts.append(checkpoint_tensor.to(dtype = torch.float32))
    if checkpoint_cursor != completed_count:
        raise ValueError(
            f"Missing embedding checkpoints after row {checkpoint_cursor}"
        )

    def run_batch(start: int) -> tuple[int, int, list[dict[str, Any]], torch.Tensor, int]:
        """Embed one source slice in a worker thread.

        Args:
            start: Inclusive source record offset.

        Returns:
            Source bounds, records, normalized tensor, and token count.
        """
        end = min(start + batch_size, target_count)
        batch_records = records[start:end]
        batch_vectors, token_count = embed_texts(
            client = client,
            texts = [format_embedding_text(record) for record in batch_records],
            model = model,
            dimensions = dimensions,
            max_retries = max_retries
        )
        return start, end, batch_records, torch.from_numpy(batch_vectors.copy()), token_count

    starts = list(range(completed_count, target_count, batch_size))
    buffered_batches = {}
    next_start = completed_count
    with metadata_path.open("a", encoding = "utf-8") as metadata_file:
        with ThreadPoolExecutor(max_workers = workers) as executor:
            start_iterator = iter(starts)
            futures = {}
            for _ in range(min(workers, len(starts))):
                start = next(start_iterator)
                futures[executor.submit(run_batch, start)] = start
            progress = tqdm(
                total = len(starts),
                desc = f"Embedding {namespace}",
                disable = not show_progress
            )
            try:
                while futures:
                    future = next(as_completed(futures))
                    futures.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                    buffered_batches[result[0]] = result
                    progress.update(1)
                    while next_start in buffered_batches:
                        start, end, batch_records, batch_tensor, token_count = (
                            buffered_batches.pop(next_start)
                        )
                        checkpoint_path = checkpoint_root / f"{start:09d}.pth"
                        _write_tensor_atomic(checkpoint_path, batch_tensor)
                        tensor_parts.append(batch_tensor)
                        for record in batch_records:
                            metadata_file.write(json.dumps(record, ensure_ascii = False) + "\n")
                        metadata_file.flush()
                        manifest["completed_count"] = end
                        manifest["total_tokens"] = (
                            int(manifest["total_tokens"]) + token_count
                        )
                        _write_json_atomic(manifest_path, manifest)
                        next_start = end
                    next_pending_start = next(start_iterator, None)
                    if next_pending_start is not None:
                        pending_future = executor.submit(run_batch, next_pending_start)
                        futures[pending_future] = next_pending_start
            finally:
                progress.close()

    if tensor_parts:
        vectors = torch.cat(tensor_parts, dim = 0)
    else:
        vectors = torch.empty((0, dimensions), dtype = torch.float32)
    if tuple(vectors.shape) != (target_count, dimensions):
        raise ValueError("Checkpoint tensors do not match the requested target count")
    _write_tensor_atomic(vectors_path, vectors)
    _write_index_atomic(index_path, vectors)
    manifest["completed_count"] = target_count
    manifest["indexed_count"] = target_count
    _write_json_atomic(manifest_path, manifest)
    shutil.rmtree(checkpoint_root)
    return manifest
