import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(os.getcwd())

from embedding.build_embedding_store import load_local_environment
from embedding.build_embedding_store import parse_args
from embedding.build_embedding_store import read_embedding_settings


def test_read_embedding_settings_uses_expected_environment(monkeypatch) -> None:
    """Read the OpenAI-compatible embedding settings from standard names."""
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")

    settings = read_embedding_settings()

    assert settings == {
        "api_key": "test-key",
        "base_url": "https://embedding.example/v1",
        "model": "qwen3.7-text-embedding",
        "dimensions": 1024
    }


def test_parse_args_supports_smoke_and_resume_options() -> None:
    """Expose namespace, prefix limit, resume, and output controls."""
    args = parse_args(
        [
            "--namespace",
            "policy",
            "--limit",
            "4",
            "--resume",
            "--output-root",
            "/tmp/vector-store"
        ]
    )

    assert args.namespace == "policy"
    assert args.limit == 4
    assert args.resume is True
    assert args.output_root == Path("/tmp/vector-store")
    assert args.batch_size == 20
    assert args.workers == 4


def test_load_local_environment_preserves_process_overrides(tmp_path, monkeypatch) -> None:
    """Load ignored dotenv values without replacing explicit process values."""
    environment_path = tmp_path / ".env"
    environment_path.write_text(
        "EMBEDDING_API_KEY=file-key\nEMBEDDING_MODEL='file-model'\n",
        encoding = "utf-8"
    )
    monkeypatch.setenv("EMBEDDING_API_KEY", "process-key")
    monkeypatch.delenv("EMBEDDING_MODEL", raising = False)

    load_local_environment(environment_path)

    assert os.environ["EMBEDDING_API_KEY"] == "process-key"
    assert os.environ["EMBEDDING_MODEL"] == "file-model"
