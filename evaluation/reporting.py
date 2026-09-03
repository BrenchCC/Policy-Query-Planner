import os
import json
import tempfile
from pathlib import Path
from typing import Any

from evaluation.metrics import safe_mean


def aggregate_scores(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample metrics overall and by important strata.

    Args:
        items: Per-sample score objects.
    """
    return {
        "count": len(items),
        "overall": _average_metrics(items),
        "by_dataset": _group(items, "source_dataset"),
        "by_hop_count": _group(items, "hop_count", skip_none = True),
        "by_answerability": _group(items, "answerable", skip_none = True)
    }


def _average_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    """Average every available numeric metric without treating missing as zero.

    Args:
        items: Per-sample score objects.
    """
    names = sorted(
        {
            name
            for item in items
            for name, value in item.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, complex)
        }
    )
    return {
        name: safe_mean(
            [
                float(item["metrics"][name])
                for item in items
                if isinstance(item.get("metrics", {}).get(name), (int, float))
            ]
        )
        for name in names
    }


def _group(
    items: list[dict[str, Any]],
    field: str,
    skip_none: bool = False
) -> dict[str, Any]:
    """Aggregate samples by a top-level field.

    Args:
        items: Per-sample score objects.
        field: Grouping field.
        skip_none: Whether to omit missing group labels.
    """
    values: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        value = item.get(field)
        if value is None and skip_none:
            continue
        key = str(value).lower() if isinstance(value, bool) else str(value)
        values.setdefault(key, []).append(item)
    return {
        key: {"count": len(group), "metrics": _average_metrics(group)}
        for key, group in sorted(values.items())
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact human-readable evaluation report.

    Args:
        summary: Aggregated summary dictionary.
    """
    lines = [
        "# Evaluation Report",
        "",
        f"Samples: {summary.get('count', 0)}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |"
    ]
    for name, value in summary.get("overall", {}).items():
        lines.append(f"| {name} | {value:.6f} |")
    section_names = (
        ("by_dataset", "By Dataset"),
        ("by_hop_count", "By Hop Count"),
        ("by_answerability", "By Answerability")
    )
    for key, title in section_names:
        groups = summary.get(key, {})
        if not groups:
            continue
        lines.extend(["", f"## {title}", ""])
        for label, group in groups.items():
            lines.extend(
                [
                    f"### {label} (n={group['count']})",
                    "",
                    "| Metric | Value |",
                    "| --- | ---: |"
                ]
            )
            for name, value in group["metrics"].items():
                lines.append(f"| {name} | {value:.6f} |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 report artifact.

    Args:
        path: Final report path.
        content: Complete text written to the artifact.
    """
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode = "w",
            encoding = "utf-8",
            dir = path.parent,
            prefix = f".{path.name}.",
            delete = False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_reports(
    scores: list[dict[str, Any]],
    output_dir: str | Path,
    force: bool = False
) -> dict[str, Any]:
    """Write JSONL scores, JSON summary, and Markdown report.

    Args:
        scores: Per-sample score objects.
        output_dir: Destination directory.
        force: Whether existing report artifacts may be replaced.
    """
    destination = Path(output_dir)
    destination.mkdir(parents = True, exist_ok = True)
    scores_path = destination / "scores.jsonl"
    summary_path = destination / "summary.json"
    report_path = destination / "report.md"
    report_paths = [scores_path, summary_path, report_path]
    existing_paths = [path for path in report_paths if path.exists()]
    if existing_paths and not force:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(f"Evaluation reports exist; use --force to replace: {names}")

    scores_content = "".join(
        json.dumps(score, ensure_ascii = False) + "\n"
        for score in scores
    )
    summary = aggregate_scores(scores)
    _atomic_write_text(scores_path, scores_content)
    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii = False, indent = 2) + "\n",
    )
    _atomic_write_text(report_path, render_markdown(summary))
    return summary
