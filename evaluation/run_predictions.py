import os
import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from tqdm import tqdm

# Add project root to Python path / 将项目根目录加入 Python 路径
sys.path.append(os.getcwd())

from data_preprocess.common import sha256_file, write_json
from data_preprocess.config import PROCESSED_ROOT
from evaluation.io import MAX_JSONL_RECORDS, read_bounded_jsonl
from evaluation.runtime import ChainExecutor, PlannerGenerator
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.rag_pipeline import AnswerGenerator

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_ROOT = PROCESSED_ROOT / "vector_store"
PREDICTION_PROTOCOL_VERSION = "1.1"
MAX_TOP_K = 100
MAX_CANDIDATE_K = 1000
MAX_LIMIT = MAX_JSONL_RECORDS


def load_local_environment(environment_path: Path) -> None:
    """Load dotenv-style values without adding another dependency.

    Args:
        environment_path: Local ignored environment file.
    """
    if not environment_path.exists():
        return
    for raw_line in environment_path.read_text(encoding = "utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and value:
            os.environ.setdefault(name, value)


def _required_environment(name: str) -> str:
    """Read one required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Non-empty environment value.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name} in environment")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse batch-prediction command-line arguments.

    Args:
        argv: Optional argument list used by tests.

    Returns:
        Parsed prediction arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Run strict RAG and multi-hop planner evaluation predictions"
    )
    parser.add_argument("--dataset", type = Path, required = True)
    parser.add_argument("--planner-model", required = True)
    parser.add_argument("--output", type = Path, required = True)
    parser.add_argument("--top-k", type = int, default = 10)
    parser.add_argument("--candidate-k", type = int, default = 50)
    parser.add_argument("--limit", type = int, default = None)
    parser.add_argument(
        "--vector-store-root",
        type = Path,
        default = DEFAULT_VECTOR_ROOT
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action = "store_true")
    mode.add_argument("--force", action = "store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= MAX_TOP_K:
        parser.error(f"--top-k must be between 1 and {MAX_TOP_K}")
    if not 1 <= args.candidate_k <= MAX_CANDIDATE_K:
        parser.error(f"--candidate-k must be between 1 and {MAX_CANDIDATE_K}")
    if args.limit is not None and not 1 <= args.limit <= MAX_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_LIMIT}")
    return args


def prediction_manifest_path(output_path: Path) -> Path:
    """Return the sidecar manifest path for one prediction JSONL file.

    Args:
        output_path: Prediction JSONL destination.

    Returns:
        Sidecar JSON path.
    """
    return Path(str(output_path) + ".manifest.json")


def _configuration_hash(configuration: dict[str, Any]) -> str:
    """Hash prediction parameters with stable JSON serialization.

    Args:
        configuration: Complete reproducibility configuration.

    Returns:
        SHA256 hexadecimal digest.
    """
    serialized = json.dumps(
        configuration,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_unique_ids(records: list[dict[str, Any]]) -> None:
    """Reject records that would collide during resume.

    Args:
        records: Selected evaluation records.

    Raises:
        ValueError: If an ID is absent, empty, or duplicated.
    """
    seen_ids = set()
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Every evaluation record must have a non-empty string id")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate evaluation record id: {record_id}")
        seen_ids.add(record_id)


def _retrieval_asset_configuration(
    records: list[dict[str, Any]],
    vector_store_root: Path,
    embedding_service_identity: str
) -> dict[str, Any]:
    """Fingerprint retrieval manifests and embedding service identity.

    Args:
        records: Selected evaluation records containing namespaces.
        vector_store_root: Root containing namespace retrieval artifacts.
        embedding_service_identity: Stable embedding API service identifier.

    Returns:
        Reproducibility configuration included in the prediction hash.
    """
    if not embedding_service_identity.strip():
        raise ValueError("embedding_service_identity cannot be empty")
    namespace_values = [record.get("namespace") for record in records]
    if not namespace_values or any(
        not isinstance(value, str) or not value.strip() for value in namespace_values
    ):
        raise ValueError("Every evaluation record must have a non-empty namespace")
    namespaces = sorted(set(namespace_values))
    assets = {}
    for namespace in namespaces:
        namespace_root = vector_store_root / namespace
        vector_manifest_path = namespace_root / "manifest.json"
        bm25_manifest_path = namespace_root / "bm25_manifest.json"
        vector_manifest = json.loads(vector_manifest_path.read_text(encoding = "utf-8"))
        runtime_asset_names = [
            "index.faiss",
            "manifest.json",
            "metadata.jsonl",
            "bm25.sqlite3",
            "bm25_manifest.json"
        ]
        assets[namespace] = {
            "embedding_model": vector_manifest.get("model"),
            "vector_manifest_sha256": sha256_file(vector_manifest_path),
            "bm25_manifest_sha256": sha256_file(bm25_manifest_path),
            "runtime_asset_sha256": {
                name: sha256_file(namespace_root / name)
                for name in runtime_asset_names
            }
        }
    return {
        "embedding_service_identity_sha256": hashlib.sha256(
            embedding_service_identity.encode("utf-8")
        ).hexdigest(),
        "retrieval_assets": assets
    }


def _failed_prediction(record: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Create a serializable batch-level failure without stopping the run.

    Args:
        record: Evaluation record that failed unexpectedly.
        error: Raised execution exception.

    Returns:
        Minimal prediction preserving the sample ID and error.
    """
    return {
        "id": record["id"],
        "source_dataset": record.get("source_dataset"),
        "source_id": record.get("source_id"),
        "namespace": record.get("namespace"),
        "question": record.get("question"),
        "planner": None,
        "steps": [],
        "evidence": [],
        "answer": None,
        "success": False,
        "fallback": False,
        "errors": [{"stage": "batch", "error": str(error)}],
        "models": {},
        "usage": {},
        "latency_seconds": {}
    }


def run_predictions(
    dataset_path: Path,
    output_path: Path,
    planner_model: str,
    executor_factory: Callable[[str], ChainExecutor],
    top_k: int = 10,
    candidate_k: int = 50,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
    extra_configuration: dict[str, Any] | None = None,
    vector_store_root: Path | None = None,
    embedding_service_identity: str | None = None,
    show_progress: bool = True
) -> dict[str, Any]:
    """Run resumable predictions while protecting incompatible outputs.

    Args:
        dataset_path: Model-ready evaluation JSONL file.
        output_path: Prediction JSONL destination.
        planner_model: Planner endpoint or model identifier.
        executor_factory: Namespace-to-executor factory.
        top_k: Evidence items retained per hop.
        candidate_k: Candidates requested per retrieval channel.
        limit: Optional leading sample limit.
        resume: Continue a compatible partial output.
        force: Replace any prior output and manifest.
        extra_configuration: Additional model/store settings included in the run hash.
        vector_store_root: Optional root whose retrieval manifests must be fingerprinted.
        embedding_service_identity: Embedding service identity included with asset hashes.
        show_progress: Whether to render a tqdm progress bar.

    Returns:
        Final run manifest.
    """
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if not 1 <= candidate_k <= MAX_CANDIDATE_K:
        raise ValueError(f"candidate_k must be between 1 and {MAX_CANDIDATE_K}")
    if limit is not None and not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    if resume and force:
        raise ValueError("resume and force are mutually exclusive")

    records = read_bounded_jsonl(dataset_path, limit = limit)
    _validate_unique_ids(records)
    retrieval_configuration = {}
    if vector_store_root is not None:
        retrieval_configuration = _retrieval_asset_configuration(
            records,
            vector_store_root,
            embedding_service_identity or ""
        )
    manifest_path = prediction_manifest_path(output_path)
    configuration = {
        "protocol_version": PREDICTION_PROTOCOL_VERSION,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "planner_model": planner_model,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "limit": limit,
        **retrieval_configuration,
        **(extra_configuration or {})
    }
    configuration_hash = _configuration_hash(configuration)
    completed_ids = set()

    if force:
        if output_path.exists():
            output_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    elif resume:
        if output_path.exists() != manifest_path.exists():
            raise ValueError("Resume requires both prediction output and its manifest")
        if manifest_path.exists():
            prior_manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
            if prior_manifest.get("configuration_hash") != configuration_hash:
                raise ValueError("Prediction manifest configuration hash does not match")
            previous_predictions = read_bounded_jsonl(output_path)
            _validate_unique_ids(previous_predictions)
            selected_ids = {record["id"] for record in records}
            unknown_ids = {
                prediction["id"]
                for prediction in previous_predictions
                if prediction["id"] not in selected_ids
            }
            if unknown_ids:
                raise ValueError(
                    "Prediction output contains IDs outside the selected dataset: "
                    + ", ".join(sorted(unknown_ids)[:5])
                )
            completed_ids = {record["id"] for record in previous_predictions}
    elif output_path.exists() or manifest_path.exists():
        raise FileExistsError("Prediction output exists; use --resume or --force")

    output_path.parent.mkdir(parents = True, exist_ok = True)
    manifest = {
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "selected_count": len(records),
        "completed_count": len(completed_ids),
        "failed_count": 0,
        "complete": False
    }
    if output_path.exists():
        manifest["failed_count"] = sum(
            not bool(prediction.get("success"))
            for prediction in read_bounded_jsonl(output_path)
        )
    write_json(manifest_path, manifest)

    executors: dict[str, ChainExecutor] = {}
    pending_records = [record for record in records if record["id"] not in completed_ids]
    with output_path.open("a", encoding = "utf-8") as output_file:
        for record in tqdm(
            pending_records,
            desc = "Evaluation predictions",
            unit = "sample",
            dynamic_ncols = True,
            disable = not show_progress
        ):
            try:
                namespace = record.get("namespace")
                if not isinstance(namespace, str) or not namespace.strip():
                    raise ValueError("Evaluation record namespace must be non-empty")
                if namespace not in executors:
                    executors[namespace] = executor_factory(namespace)
                prediction = executors[namespace].run(
                    record,
                    top_k = top_k,
                    candidate_k = candidate_k
                )
            except Exception as error:
                logger.exception("Prediction failed for %s: %s", record["id"], error)
                prediction = _failed_prediction(record, error)
            output_file.write(json.dumps(prediction, ensure_ascii = False) + "\n")
            output_file.flush()
            manifest["completed_count"] += 1
            manifest["failed_count"] += int(not bool(prediction.get("success")))
            write_json(manifest_path, manifest)

    manifest["complete"] = manifest["completed_count"] == len(records)
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    """Build live clients and execute one batch prediction run."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root / ".env")
    llm_client = OpenAI(
        api_key = _required_environment("LLM_API_KEY"),
        base_url = _required_environment("LLM_BASE_URL"),
        timeout = 300.0,
        max_retries = 2
    )
    embedding_client = OpenAI(
        api_key = _required_environment("EMBEDDING_API_KEY"),
        base_url = _required_environment("EMBEDDING_BASE_URL"),
        timeout = 300.0,
        max_retries = 0
    )
    response_model = _required_environment("RESPONSE_MODEL")
    planner = PlannerGenerator(client = llm_client, model = args.planner_model)
    answer_generator = AnswerGenerator(client = llm_client, model = response_model)

    def executor_factory(namespace: str) -> ChainExecutor:
        """Create one executor for a knowledge-base namespace.

        Args:
            namespace: Evaluation namespace.

        Returns:
            Configured strict chain executor.
        """
        retriever = HybridRetriever(
            namespace_root = args.vector_store_root / namespace,
            embedding_client = embedding_client,
            candidate_k = args.candidate_k,
            show_progress = False
        )
        return ChainExecutor(
            retriever = retriever,
            planner = planner,
            answer_generator = answer_generator
        )

    manifest = run_predictions(
        dataset_path = args.dataset,
        output_path = args.output,
        planner_model = args.planner_model,
        executor_factory = executor_factory,
        top_k = args.top_k,
        candidate_k = args.candidate_k,
        limit = args.limit,
        resume = args.resume,
        force = args.force,
        extra_configuration = {
            "response_model": response_model,
            "vector_store_root": str(args.vector_store_root.resolve())
        },
        vector_store_root = args.vector_store_root,
        embedding_service_identity = _required_environment("EMBEDDING_BASE_URL")
    )
    print(json.dumps(manifest, ensure_ascii = False, indent = 2))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
