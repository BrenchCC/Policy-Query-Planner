import re
import json
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Iterable

import requests
from lxml import html

logger = logging.getLogger(__name__)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a path.

    Args:
        path: File path whose parent directory must exist.
    """
    path.parent.mkdir(parents = True, exist_ok = True)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON value.
    """
    with path.open("r", encoding = "utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    """Write a JSON value with stable formatting.

    Args:
        path: Destination JSON file path.
        value: JSON-serializable value.
    """
    ensure_parent(path)
    with path.open("w", encoding = "utf-8") as file:
        json.dump(value, file, ensure_ascii = False, indent = 2)
        file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory.

    Args:
        path: Source JSONL file path.

    Returns:
        Parsed records in file order.
    """
    records = []
    with path.open("r", encoding = "utf-8") as file:
        for line_number, line in enumerate(file, start = 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as UTF-8 JSONL.

    Args:
        path: Destination JSONL file path.
        records: JSON-serializable record iterable.

    Returns:
        Number of records written.
    """
    ensure_parent(path)
    count = 0
    with path.open("w", encoding = "utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii = False, sort_keys = False) + "\n")
            count += 1
    return count


def normalize_text(value: str) -> str:
    """Normalize whitespace and common Unicode punctuation.

    Args:
        value: Text to normalize.

    Returns:
        Normalized text.
    """
    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"'
    }
    text = str(value or "")
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def strip_html(value: str) -> tuple[str, str]:
    """Extract tag and visible text from one HTML fragment.

    Args:
        value: HTML fragment from a source document.

    Returns:
        Lowercase root tag and normalized visible text.
    """
    fragment = html.fragment_fromstring(value, create_parent = "div")
    children = list(fragment)
    tag = str(children[0].tag).lower() if children else "text"
    text = normalize_text(fragment.text_content())
    return tag, text


def normalized_key(value: str) -> str:
    """Create a comparison key that ignores punctuation and case.

    Args:
        value: Text used for comparison.

    Returns:
        Alphanumeric lowercase comparison key.
    """
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())


def stable_hash(*values: str, length: int = 16) -> str:
    """Create a stable hexadecimal identifier.

    Args:
        values: Ordered values included in the identifier.
        length: Number of hexadecimal characters to return.

    Returns:
        Stable hexadecimal identifier.
    """
    digest = hashlib.sha256("\u241f".join(values).encode("utf-8")).hexdigest()
    return digest[:length]


def stable_record_hash(record: dict[str, Any]) -> str:
    """Create a stable hash for a generation request.

    Args:
        record: Request record whose content should be fingerprinted.

    Returns:
        Full SHA256 request fingerprint.
    """
    payload = {
        key: value
        for key, value in record.items()
        if key != "request_hash"
    }
    serialized = json.dumps(
        payload,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":")
    )
    return stable_hash(serialized, length = 64)


def sha256_file(path: Path) -> str:
    """Calculate the SHA256 checksum of a file.

    Args:
        path: File to checksum.

    Returns:
        Hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, force: bool = False) -> None:
    """Download a file atomically with retry-safe temporary storage.

    Args:
        url: Source HTTP URL.
        path: Destination file path.
        force: Whether to replace an existing destination.
    """
    if path.exists() and not force:
        logger.info("Skipping existing file: %s", path)
        return
    ensure_parent(path)
    with requests.get(url, stream = True, timeout = 120) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir = path.parent, delete = False) as file:
            temporary_path = Path(file.name)
            for chunk in response.iter_content(chunk_size = 1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary_path.replace(path)


def extract_json_object(value: str) -> dict[str, Any]:
    """Extract one JSON object from a model response.

    Args:
        value: Raw model response.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no valid JSON object can be parsed.
    """
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags = re.IGNORECASE)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object")
        try:
            result = json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"Model response contains invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object")
    return result
