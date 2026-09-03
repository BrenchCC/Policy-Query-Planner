import os
import sys
import json

import pytest

sys.path.append(os.getcwd())

from evaluation.io import read_bounded_jsonl


def test_bounded_jsonl_stops_at_explicit_limit(tmp_path) -> None:
    """Read only the requested prefix without loading the rest of the file."""
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps({"id": index}) + "\n" for index in range(3)),
        encoding = "utf-8"
    )

    records = read_bounded_jsonl(path, limit = 2)

    assert [record["id"] for record in records] == [0, 1]


def test_bounded_jsonl_rejects_default_record_cap(tmp_path) -> None:
    """Fail explicitly when an unrestricted input exceeds its hard record cap."""
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps({"id": index}) + "\n" for index in range(3)),
        encoding = "utf-8"
    )

    with pytest.raises(ValueError, match = "exceeds 2 records"):
        read_bounded_jsonl(path, max_records = 2)
