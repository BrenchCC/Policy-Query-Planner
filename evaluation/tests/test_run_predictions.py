import os
import sys
import json
from pathlib import Path

import pytest

# Add project root to Python path / 将项目根目录加入 Python 路径
sys.path.append(os.getcwd())

from data_preprocess.common import read_jsonl, write_jsonl
from evaluation.run_predictions import parse_args, run_predictions
from evaluation.run_predictions import prediction_manifest_path


class FakeExecutor:
    """Return deterministic predictions and optionally raise per sample."""

    def __init__(self, calls: list[str], failing_ids: set[str] | None = None) -> None:
        self.calls = calls
        self.failing_ids = failing_ids or set()

    def run(self, record, top_k, candidate_k):
        """Record one call and return a minimal prediction."""
        self.calls.append(record["id"])
        if record["id"] in self.failing_ids:
            raise RuntimeError("sample failure")
        return {
            "id": record["id"],
            "success": True,
            "top_k": top_k,
            "candidate_k": candidate_k
        }


def make_dataset(path: Path, count: int = 3) -> None:
    """Write a small model-ready evaluation dataset."""
    write_jsonl(
        path,
        [
            {
                "id": f"eval-{index}",
                "namespace": "policy",
                "question": f"Question {index}"
            }
            for index in range(count)
        ]
    )


def test_parse_args_supports_required_prediction_interface() -> None:
    """Expose documented paths, model, retrieval limits, and resume mode."""
    args = parse_args(
        [
            "--dataset",
            "eval.jsonl",
            "--planner-model",
            "planner",
            "--output",
            "predictions.jsonl",
            "--top-k",
            "5",
            "--candidate-k",
            "20",
            "--limit",
            "2",
            "--resume"
        ]
    )

    assert args.dataset == Path("eval.jsonl")
    assert args.planner_model == "planner"
    assert args.top_k == 5
    assert args.candidate_k == 20
    assert args.limit == 2
    assert args.resume is True


@pytest.mark.parametrize(
    "option,value",
    [
        ("--top-k", "101"),
        ("--candidate-k", "1001"),
        ("--limit", "100001")
    ]
)
def test_parse_args_rejects_unbounded_resource_values(option, value) -> None:
    """Reject CLI values that can request unreasonable local or API resources."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                "eval.jsonl",
                "--planner-model",
                "planner",
                "--output",
                "predictions.jsonl",
                option,
                value
            ]
        )


def test_batch_continues_after_failure_and_writes_manifest(tmp_path) -> None:
    """Record one failed sample without terminating later predictions."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    make_dataset(dataset_path)
    calls = []

    manifest = run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor(calls, {"eval-1"}),
        top_k = 5,
        candidate_k = 20,
        show_progress = False
    )

    predictions = read_jsonl(output_path)
    assert calls == ["eval-0", "eval-1", "eval-2"]
    assert len(predictions) == 3
    assert predictions[1]["success"] is False
    assert predictions[2]["success"] is True
    assert manifest["completed_count"] == 3
    assert manifest["failed_count"] == 1
    assert manifest["complete"] is True
    assert prediction_manifest_path(output_path).exists()


def test_resume_skips_completed_ids_and_rejects_configuration_change(tmp_path) -> None:
    """Continue compatible output and reject a different retrieval configuration."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    make_dataset(dataset_path)
    first_calls = []
    run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor(first_calls),
        limit = 2,
        show_progress = False
    )
    resume_calls = []
    run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor(resume_calls),
        limit = 2,
        resume = True,
        show_progress = False
    )

    assert first_calls == ["eval-0", "eval-1"]
    assert resume_calls == []
    assert len(read_jsonl(output_path)) == 2

    try:
        run_predictions(
            dataset_path = dataset_path,
            output_path = output_path,
            planner_model = "different-planner",
            executor_factory = lambda namespace: FakeExecutor([]),
            limit = 2,
            resume = True,
            show_progress = False
        )
    except ValueError as error:
        assert "configuration hash" in str(error)
    else:
        raise AssertionError("Expected an incompatible resume error")


def test_force_replaces_output_and_limit_is_hashed(tmp_path) -> None:
    """Replace an old run only when force is explicit."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    make_dataset(dataset_path)
    run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor([]),
        limit = 1,
        show_progress = False
    )

    replacement_calls = []
    manifest = run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor(replacement_calls),
        limit = 2,
        force = True,
        show_progress = False
    )

    assert replacement_calls == ["eval-0", "eval-1"]
    assert len(read_jsonl(output_path)) == 2
    assert manifest["configuration"]["limit"] == 2


def test_resume_detects_dataset_content_change(tmp_path) -> None:
    """Protect an output from resuming against modified dataset bytes."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    make_dataset(dataset_path, count = 1)
    run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor([]),
        show_progress = False
    )
    records = read_jsonl(dataset_path)
    records[0]["question"] = "Changed"
    write_jsonl(dataset_path, records)

    try:
        run_predictions(
            dataset_path = dataset_path,
            output_path = output_path,
            planner_model = "planner",
            executor_factory = lambda namespace: FakeExecutor([]),
            resume = True,
            show_progress = False
        )
    except ValueError as error:
        assert "configuration hash" in str(error)
    else:
        raise AssertionError("Expected dataset hash protection")


@pytest.mark.parametrize(
    "overrides,error_name",
    [
        ({"top_k": 101}, "top_k"),
        ({"candidate_k": 1001}, "candidate_k"),
        ({"limit": 100001}, "limit")
    ]
)
def test_run_predictions_rejects_resource_values_above_limits(
    tmp_path,
    overrides,
    error_name
) -> None:
    """Apply resource bounds to direct API use as well as the CLI."""
    dataset_path = tmp_path / "eval.jsonl"
    make_dataset(dataset_path)
    with pytest.raises(ValueError, match = error_name):
        run_predictions(
            dataset_path = dataset_path,
            output_path = tmp_path / "predictions.jsonl",
            planner_model = "planner",
            executor_factory = lambda namespace: FakeExecutor([]),
            **overrides,
            show_progress = False
        )


def test_resume_rejects_changed_retrieval_manifest(tmp_path) -> None:
    """Fingerprint manifest contents and embedding service in the resume hash."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    vector_root = tmp_path / "vectors"
    namespace_root = vector_root / "policy"
    namespace_root.mkdir(parents = True)
    (namespace_root / "manifest.json").write_text(
        json.dumps({"model": "embedding-a", "indexed_count": 3}),
        encoding = "utf-8"
    )
    (namespace_root / "bm25_manifest.json").write_text(
        json.dumps({"document_count": 3}),
        encoding = "utf-8"
    )
    for name in ["index.faiss", "metadata.jsonl", "bm25.sqlite3"]:
        (namespace_root / name).write_text(name, encoding = "utf-8")
    make_dataset(dataset_path)
    common = {
        "dataset_path": dataset_path,
        "output_path": output_path,
        "planner_model": "planner",
        "executor_factory": lambda namespace: FakeExecutor([]),
        "vector_store_root": vector_root,
        "embedding_service_identity": "https://embedding.example/v1",
        "show_progress": False
    }
    manifest = run_predictions(**common)
    assert manifest["configuration"]["retrieval_assets"]["policy"][
        "embedding_model"
    ] == "embedding-a"
    assert "embedding_service_identity" not in manifest["configuration"]
    assert manifest["configuration"]["retrieval_assets"]["policy"][
        "runtime_asset_sha256"
    ]["index.faiss"]
    assert len(manifest["configuration"]["embedding_service_identity_sha256"]) == 64
    assert len(
        manifest["configuration"]["retrieval_assets"]["policy"][
            "bm25_manifest_sha256"
        ]
    ) == 64

    (namespace_root / "manifest.json").write_text(
        json.dumps({"model": "embedding-b", "indexed_count": 3}),
        encoding = "utf-8"
    )
    with pytest.raises(ValueError, match = "configuration hash"):
        run_predictions(**common, resume = True)


def test_resume_rejects_changed_runtime_retrieval_asset(tmp_path) -> None:
    """Reject resume when an index changes without a manifest update."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    vector_root = tmp_path / "vectors"
    namespace_root = vector_root / "policy"
    namespace_root.mkdir(parents = True)
    for name, content in {
        "manifest.json": json.dumps({"model": "embedding-a"}),
        "bm25_manifest.json": json.dumps({"document_count": 3}),
        "index.faiss": "index-a",
        "metadata.jsonl": "{}\n",
        "bm25.sqlite3": "sqlite-a"
    }.items():
        (namespace_root / name).write_text(content, encoding = "utf-8")
    make_dataset(dataset_path)
    common = {
        "dataset_path": dataset_path,
        "output_path": output_path,
        "planner_model": "planner",
        "executor_factory": lambda namespace: FakeExecutor([]),
        "vector_store_root": vector_root,
        "embedding_service_identity": "https://embedding.example/v1",
        "show_progress": False
    }
    run_predictions(**common)
    (namespace_root / "index.faiss").write_text("index-b", encoding = "utf-8")

    with pytest.raises(ValueError, match = "configuration hash"):
        run_predictions(**common, resume = True)


def test_resume_rejects_prediction_ids_outside_selected_dataset(tmp_path) -> None:
    """Do not silently retain contaminating prediction rows during resume."""
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    make_dataset(dataset_path, count = 1)
    run_predictions(
        dataset_path = dataset_path,
        output_path = output_path,
        planner_model = "planner",
        executor_factory = lambda namespace: FakeExecutor([]),
        show_progress = False
    )
    with output_path.open("a", encoding = "utf-8") as handle:
        handle.write(json.dumps({"id": "foreign-id", "success": True}) + "\n")

    with pytest.raises(ValueError, match = "outside the selected dataset"):
        run_predictions(
            dataset_path = dataset_path,
            output_path = output_path,
            planner_model = "planner",
            executor_factory = lambda namespace: FakeExecutor([]),
            resume = True,
            show_progress = False
        )
