import os
import sys
import json
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import Any

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import read_jsonl, sha256_file
from data_preprocess.config import PROCESSED_ROOT

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_ROOT = PROCESSED_ROOT / "vector_store"
TOKENIZER = "unicode61"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a BM25 manifest atomically.

    Args:
        path: Destination manifest path.
        manifest: JSON-compatible manifest.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding = "utf-8") as file:
        json.dump(manifest, file, ensure_ascii = False, indent = 2)
        file.write("\n")
    temporary_path.replace(path)


def build_bm25_store(namespace_root: Path, force: bool = False) -> dict[str, Any]:
    """Build a persistent SQLite FTS5 BM25 index beside a FAISS store.

    Args:
        namespace_root: Directory containing metadata.jsonl.
        force: Whether to replace an existing BM25 index.

    Returns:
        Persisted BM25 manifest.
    """
    metadata_path = namespace_root / "metadata.jsonl"
    database_path = namespace_root / "bm25.sqlite3"
    manifest_path = namespace_root / "bm25_manifest.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {metadata_path}")
    if (database_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(
            f"BM25 store already exists in {namespace_root}; use --force to rebuild"
        )
    records = read_jsonl(metadata_path)
    temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "document_id UNINDEXED, title, text, tokenize='unicode61')"
        )
        connection.executemany(
            "INSERT INTO chunks(rowid, document_id, title, text) VALUES (?, ?, ?, ?)",
            [
                (
                    index + 1,
                    record["id"],
                    record.get("title", ""),
                    record.get("text", "")
                )
                for index, record in enumerate(records)
            ]
        )
        connection.commit()
    finally:
        connection.close()
    temporary_path.replace(database_path)
    manifest = {
        "namespace": records[0].get("namespace") if records else namespace_root.name,
        "metadata_sha256": sha256_file(metadata_path),
        "document_count": len(records),
        "tokenizer": TOKENIZER,
        "title_weight": 2.0,
        "text_weight": 1.0
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse BM25 build arguments.

    Args:
        argv: Optional argument list used by tests.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Build SQLite FTS5 BM25 indexes")
    parser.add_argument(
        "--namespace",
        choices = ["policy", "musique_aux", "all"],
        default = "all"
    )
    parser.add_argument("--vector-store-root", type = Path, default = DEFAULT_VECTOR_ROOT)
    parser.add_argument("--force", action = "store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Build BM25 indexes for the selected namespaces."""
    args = parse_args()
    namespaces = ["policy", "musique_aux"] if args.namespace == "all" else [args.namespace]
    for namespace in namespaces:
        manifest = build_bm25_store(
            namespace_root = args.vector_store_root / namespace,
            force = args.force
        )
        logger.info(
            "Built BM25 namespace=%s documents=%d",
            namespace,
            manifest["document_count"]
        )


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
