import os
import sys
import json
import logging
import argparse
from pathlib import Path

from openai import OpenAI

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.config import PROCESSED_ROOT
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.rag_pipeline import AnswerGenerator, QueryRewriter, RAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_ROOT = PROCESSED_ROOT / "vector_store"


def load_local_environment(environment_path: Path) -> None:
    """Load dotenv-style values without importing the PyTorch build path.

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
    """Parse RAG command-line arguments.

    Args:
        argv: Optional argument list used by tests.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Run hybrid retrieval-augmented generation")
    parser.add_argument("--query", required = True)
    parser.add_argument("--history", default = None)
    parser.add_argument(
        "--namespace",
        choices = ["policy", "musique_aux"],
        default = "policy"
    )
    parser.add_argument("--candidate-k", type = int, default = 50)
    parser.add_argument("--top-k", type = int, default = 10)
    parser.add_argument("--vector-store-root", type = Path, default = DEFAULT_VECTOR_ROOT)
    return parser.parse_args(argv)


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


def main() -> None:
    """Load configured models and print one structured RAG result."""
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
    retriever = HybridRetriever(
        namespace_root = args.vector_store_root / args.namespace,
        embedding_client = embedding_client,
        candidate_k = args.candidate_k,
        show_progress = True
    )
    pipeline = RAGPipeline(
        retriever = retriever,
        rewriter = QueryRewriter(
            client = llm_client,
            model = _required_environment("QUERY_MODEL")
        ),
        answer_generator = AnswerGenerator(
            client = llm_client,
            model = _required_environment("RESPONSE_MODEL")
        )
    )
    result = pipeline.run(
        query = args.query,
        history = args.history,
        top_k = args.top_k,
        candidate_k = args.candidate_k
    )
    print(json.dumps(result, ensure_ascii = False, indent = 2))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
