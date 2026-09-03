import json
from pathlib import Path
from typing import Any

MAX_JSONL_RECORDS = 100000
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024


def read_bounded_jsonl(
    path: str | Path,
    limit: int | None = None,
    max_records: int = MAX_JSONL_RECORDS
) -> list[dict[str, Any]]:
    """Read a bounded prefix of an object-only UTF-8 JSONL file.

    Args:
        path: Input JSONL path.
        limit: Optional number of non-empty records to return.
        max_records: Hard record cap when no smaller limit is supplied.

    Returns:
        Parsed JSON object records.
    """
    if limit is not None and not 1 <= limit <= max_records:
        raise ValueError(f"limit must be between 1 and {max_records}")
    effective_limit = limit if limit is not None else max_records
    records = []
    with Path(path).open("rb") as handle:
        line_number = 0
        while True:
            if limit is not None and len(records) == effective_limit:
                break
            line = handle.readline(MAX_JSONL_LINE_BYTES + 1)
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            if len(line) > MAX_JSONL_LINE_BYTES:
                raise ValueError(
                    f"JSONL line exceeds {MAX_JSONL_LINE_BYTES} bytes: {path}:{line_number}"
                )
            if len(records) == effective_limit:
                raise ValueError(f"JSONL file exceeds {max_records} records: {path}")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record on {path}:{line_number} must be an object")
            records.append(value)
    return records
