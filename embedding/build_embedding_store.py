import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Any

from openai import OpenAI

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.config import PROCESSED_ROOT
from embedding.embedding_store import build_namespace_store

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "vector_store"


def load_local_environment(environment_path: Path) -> None:
    """Load dotenv-style values without overriding process variables.

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse embedding-store command-line arguments.

    Args:
        argv: Optional argument list used by tests.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Build Qwen embedding tensors and namespace-specific FAISS indexes"
    )
    parser.add_argument(
        "--namespace",
        choices = ["policy", "musique_aux", "all"],
        default = "all"
    )
    parser.add_argument("--batch-size", type = int, default = 20)
    parser.add_argument("--workers", type = int, default = 4)
    parser.add_argument("--max-retries", type = int, default = 2)
    parser.add_argument("--limit", type = int, default = None)
    parser.add_argument("--resume", action = "store_true")
    parser.add_argument("--force", action = "store_true")
    parser.add_argument(
        "--output-root",
        type = Path,
        default = DEFAULT_OUTPUT_ROOT
    )
    return parser.parse_args(argv)


def read_embedding_settings() -> dict[str, Any]:
    """Read and validate embedding API settings from the environment.

    Returns:
        OpenAI client and embedding model settings.

    Raises:
        RuntimeError: If an API credential or base URL is missing.
        ValueError: If the requested dimension is invalid.
    """
    api_key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    base_url = os.environ.get("EMBEDDING_BASE_URL", "").strip()
    model = os.environ.get("EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
    dimensions = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))
    if not api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY in environment")
    if not base_url:
        raise RuntimeError("Missing EMBEDDING_BASE_URL in environment")
    if not model:
        raise RuntimeError("Missing EMBEDDING_MODEL in environment")
    if dimensions not in {256, 512, 768, 1024, 1536, 2048, 2560}:
        raise ValueError("EMBEDDING_DIMENSIONS is not supported by the embedding model")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "dimensions": dimensions
    }


def main() -> None:
    """Build embedding artifacts for one or both knowledge namespaces."""
    args = parse_args()
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore2").setLevel(logging.WARNING)
    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root / ".env")
    settings = read_embedding_settings()
    client = OpenAI(
        api_key = settings["api_key"],
        base_url = settings["base_url"],
        timeout = 300.0,
        max_retries = 0
    )
    source_paths = {
        "policy": PROCESSED_ROOT / "knowledge_base" / "policy.jsonl",
        "musique_aux": PROCESSED_ROOT / "knowledge_base" / "musique_aux.jsonl"
    }
    namespaces = list(source_paths) if args.namespace == "all" else [args.namespace]
    logger.info(
        "Starting embedding build namespaces=%s model=%s dimensions=%d output=%s",
        namespaces,
        settings["model"],
        settings["dimensions"],
        args.output_root
    )
    for namespace in namespaces:
        manifest = build_namespace_store(
            namespace = namespace,
            source_path = source_paths[namespace],
            output_root = args.output_root,
            client = client,
            model = settings["model"],
            dimensions = settings["dimensions"],
            batch_size = args.batch_size,
            workers = args.workers,
            max_retries = args.max_retries,
            limit = args.limit,
            resume = args.resume,
            force = args.force,
            show_progress = True
        )
        logger.info(
            "Completed namespace=%s rows=%d tokens=%d",
            namespace,
            manifest["completed_count"],
            manifest["total_tokens"]
        )


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
