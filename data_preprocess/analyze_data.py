import os
import sys
import json
import logging
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import read_json, read_jsonl, write_json
from data_preprocess.config import INTERIM_ROOT, PROCESSED_ROOT, REPORT_ROOT

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Create lightweight dataset EDA")
    parser.add_argument("--force", action = "store_true", help = "Replace existing reports")
    return parser.parse_args()


def save_figure(path: Path) -> None:
    """Save and close the current Matplotlib figure.

    Args:
        path: Destination PNG path.
    """
    path.parent.mkdir(parents = True, exist_ok = True)
    plt.tight_layout()
    plt.savefig(path, dpi = 160, bbox_inches = "tight")
    plt.close()


def describe_lengths(values: list[int]) -> dict[str, float]:
    """Summarize a list of text lengths.

    Args:
        values: Integer length values.

    Returns:
        Count, mean, median, p95, minimum, and maximum.
    """
    array = np.asarray(values, dtype = np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max())
    }


def plot_policy_lengths(records: list[dict[str, Any]], figure_root: Path) -> dict[str, float]:
    """Plot policy chunk character lengths.

    Args:
        records: Policy knowledge-base records.
        figure_root: Figure output directory.

    Returns:
        Character-length statistics.
    """
    lengths = [len(record["text"]) for record in records]
    sns.histplot(lengths, bins = 40)
    plt.title("ConditionalQA policy chunk lengths")
    plt.xlabel("Characters")
    save_figure(figure_root / "policy_chunk_lengths.png")
    return describe_lengths(lengths)


def plot_qrecc(qrecc_train: list[dict[str, Any]], figure_root: Path) -> dict[str, Any]:
    """Plot QReCC history depth and rewrite-length changes.

    Args:
        qrecc_train: Clean QReCC training records.
        figure_root: Figure output directory.

    Returns:
        QReCC EDA statistics.
    """
    turns = [record["context_turns"] for record in qrecc_train]
    deltas = [len(record["rewrite"].split()) - len(record["question"].split()) for record in qrecc_train]
    sns.countplot(x = turns, color = "#4c78a8")
    plt.title("QReCC conversation history depth")
    plt.xlabel("Previous turns")
    plt.ylabel("Records")
    save_figure(figure_root / "qrecc_history_turns.png")

    sns.histplot(deltas, bins = 41)
    plt.xlim(-20, 40)
    plt.title("QReCC rewrite word-count delta")
    plt.xlabel("Rewrite words minus question words")
    save_figure(figure_root / "qrecc_rewrite_delta.png")
    return {
        "count": len(qrecc_train),
        "nontrivial_count": sum(record["nontrivial_rewrite"] for record in qrecc_train),
        "source_counts": dict(Counter(record["conversation_source"] for record in qrecc_train)),
        "history_turn_counts": dict(Counter(str(value) for value in turns)),
        "rewrite_delta": describe_lengths(deltas)
    }


def plot_conditionalqa(
    conditional_train: list[dict[str, Any]],
    figure_root: Path
) -> dict[str, Any]:
    """Plot ConditionalQA evidence and answer distributions.

    Args:
        conditional_train: Clean ConditionalQA training records.
        figure_root: Figure output directory.

    Returns:
        ConditionalQA EDA statistics.
    """
    evidence_counts = [len(record["evidences"]) for record in conditional_train]
    clipped = [min(value, 10) for value in evidence_counts]
    sns.countplot(x = clipped, color = "#f58518")
    plt.title("ConditionalQA evidence count per question")
    plt.xlabel("Evidence strings (10 means 10+)")
    plt.ylabel("Questions")
    save_figure(figure_root / "conditionalqa_evidence_counts.png")
    return {
        "count": len(conditional_train),
        "multi_evidence_count": sum(value >= 2 for value in evidence_counts),
        "unanswerable_count": sum(record["not_answerable"] for record in conditional_train),
        "evidence_count_distribution": dict(Counter(str(value) for value in evidence_counts))
    }


def plot_training_mix(
    sft_records: list[dict[str, Any]],
    cold_start_records: list[dict[str, Any]],
    dpo_records: list[dict[str, Any]],
    grpo_records: list[dict[str, Any]],
    figure_root: Path
) -> dict[str, Any]:
    """Plot final train-stage sizes and subtype distributions.

    Args:
        sft_records: Final SFT records.
        cold_start_records: Multi-hop SFT cold-start records.
        dpo_records: Final DPO records.
        grpo_records: Final GRPO records.
        figure_root: Figure output directory.

    Returns:
        Final mixture statistics.
    """
    stage_frame = pd.DataFrame(
        {
            "stage": ["Base SFT", "Cold-start SFT", "DPO", "GRPO"],
            "records": [
                len(sft_records),
                len(cold_start_records),
                len(dpo_records),
                len(grpo_records)
            ]
        }
    )
    sns.barplot(data = stage_frame, x = "stage", y = "records", hue = "stage", legend = False)
    plt.title("Final training dataset sizes")
    save_figure(figure_root / "training_stage_sizes.png")

    cold_start_hop_counts = Counter(
        str(record["hop_count"])
        for record in cold_start_records
    )
    cold_start_hop_frame = pd.DataFrame(
        {
            "hop_count": sorted(cold_start_hop_counts),
            "records": [
                cold_start_hop_counts[key]
                for key in sorted(cold_start_hop_counts)
            ]
        }
    )
    sns.barplot(
        data = cold_start_hop_frame,
        x = "hop_count",
        y = "records",
        hue = "hop_count",
        legend = False
    )
    plt.title("Multi-hop SFT cold-start distribution")
    save_figure(figure_root / "cold_start_hop_distribution.png")

    dpo_hop_counts = Counter(str(record["hop_count"]) for record in dpo_records)
    dpo_hop_frame = pd.DataFrame(
        {
            "hop_count": sorted(dpo_hop_counts),
            "records": [dpo_hop_counts[key] for key in sorted(dpo_hop_counts)]
        }
    )
    sns.barplot(
        data = dpo_hop_frame,
        x = "hop_count",
        y = "records",
        hue = "hop_count",
        legend = False
    )
    plt.title("DPO single-hop and multi-hop distribution")
    save_figure(figure_root / "dpo_hop_distribution.png")

    grpo_hop_counts = Counter(str(record["hop_count"]) for record in grpo_records)
    hop_frame = pd.DataFrame(
        {
            "hop_count": sorted(grpo_hop_counts),
            "records": [grpo_hop_counts[key] for key in sorted(grpo_hop_counts)]
        }
    )
    sns.barplot(data = hop_frame, x = "hop_count", y = "records", hue = "hop_count", legend = False)
    plt.title("GRPO hop distribution")
    save_figure(figure_root / "grpo_hop_distribution.png")
    return {
        "sft_count": len(sft_records),
        "sft_mix": dict(Counter(record["sample_type"] for record in sft_records)),
        "cold_start_count": len(cold_start_records),
        "cold_start_hop_mix": dict(cold_start_hop_counts),
        "dpo_count": len(dpo_records),
        "dpo_mix": dict(Counter(record["source_dataset"] for record in dpo_records)),
        "dpo_task_mix": dict(Counter(record["task_type"] for record in dpo_records)),
        "dpo_hop_mix": dict(dpo_hop_counts),
        "dpo_error_mix": dict(Counter(record["error_type"] for record in dpo_records)),
        "grpo_count": len(grpo_records),
        "grpo_task_mix": dict(Counter(record["task_type"] for record in grpo_records)),
        "grpo_source_task_mix": dict(
            Counter(
                f"{record['source_dataset']}:{record['task_type']}"
                for record in grpo_records
            )
        ),
        "grpo_hop_mix": dict(grpo_hop_counts)
    }


def write_markdown_report(summary: dict[str, Any]) -> None:
    """Write a concise Chinese EDA report.

    Args:
        summary: Structured EDA summary.
    """
    cleaning = summary["cleaning"]
    final_mix = summary["final_training_mix"]
    report = f"""# 数据清洗与 EDA 报告

## 数据概览

- ConditionalQA 原始文档：{cleaning['conditionalqa']['raw_document_count']} 篇。
- 政策知识库：{cleaning['conditionalqa']['policy_chunk_count']} 个 chunk。
- ConditionalQA evidence 映射率：{cleaning['conditionalqa']['evidence_mapping_rate']:.2%}。
- MuSiQue 辅助知识库：{cleaning['musique']['knowledge_record_count']} 条去重段落。
- QReCC train：{cleaning['qrecc']['train']['clean_count']} 条，其中非平凡改写 {cleaning['qrecc']['train']['nontrivial_rewrite_count']} 条。

## 训练数据

- SFT：{final_mix['sft_count']} 条，构成为 `{json.dumps(final_mix['sft_mix'], ensure_ascii = False)}`。
- 多跳 SFT 冷启动：{final_mix['cold_start_count']} 条，hop 分布为 `{json.dumps(final_mix['cold_start_hop_mix'], ensure_ascii = False)}`。
- DPO：{final_mix['dpo_count']} 条，来源为 `{json.dumps(final_mix['dpo_mix'], ensure_ascii = False)}`。
- DPO 单跳/多跳构成为 `{json.dumps(final_mix['dpo_task_mix'], ensure_ascii = False)}`，hop 分布为 `{json.dumps(final_mix['dpo_hop_mix'], ensure_ascii = False)}`。
- DPO 十类单因素困难负样本严格均衡：`{json.dumps(final_mix['dpo_error_mix'], ensure_ascii = False)}`。
- GRPO：{final_mix['grpo_count']} 条，来源与任务构成为 `{json.dumps(final_mix['grpo_source_task_mix'], ensure_ascii = False)}`，hop 分布为 `{json.dumps(final_mix['grpo_hop_mix'], ensure_ascii = False)}`。

## 质量观察

- 官方 benchmark split 未参与训练抽样。
- 政策原始语料包含21篇未被官方问题 split 引用的文档；它们保留在知识库中，用于模拟真实检索噪声。
- 仅有1条 ConditionalQA heading-only evidence 未映射到正文 chunk，已记录在 `data/interim/gold_coverage_issues.json`。
- QReCC no-op 样本被保留，用于抑制模型对本来已独立的问题进行过度改写。
- MuSiQue 多跳 SFT、DPO 与 GRPO 候选池按来源 ID 和问题指纹严格隔离；领域多跳另做逐跳证据与生成问题校验。
- DPO 与 GRPO 都保留单跳样本，用于抑制不必要拆分和奖励投机。
- MuSiQue 使用单独 namespace，训练或评测时必须选择对应知识库。

## 图表

- `figures/policy_chunk_lengths.png`
- `figures/qrecc_history_turns.png`
- `figures/qrecc_rewrite_delta.png`
- `figures/conditionalqa_evidence_counts.png`
- `figures/training_stage_sizes.png`
- `figures/cold_start_hop_distribution.png`
- `figures/dpo_hop_distribution.png`
- `figures/grpo_hop_distribution.png`
"""
    path = REPORT_ROOT / "eda_report.md"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(report, encoding = "utf-8")


def main() -> None:
    """Run lightweight EDA and write figures and reports."""
    args = parse_args()
    summary_path = REPORT_ROOT / "eda_summary.json"
    if summary_path.exists() and not args.force:
        logger.info("EDA outputs already exist; use --force to rebuild")
        return
    sns.set_theme(style = "whitegrid")
    figure_root = REPORT_ROOT / "figures"
    policy_records = read_jsonl(PROCESSED_ROOT / "knowledge_base" / "policy.jsonl")
    qrecc_train = read_jsonl(INTERIM_ROOT / "qrecc_train.jsonl")
    conditional_train = read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
    sft_records = read_jsonl(PROCESSED_ROOT / "train" / "sft_train.jsonl")
    cold_start_records = read_jsonl(
        PROCESSED_ROOT / "train" / "sft_multihop_cold_start.jsonl"
    )
    dpo_records = read_jsonl(PROCESSED_ROOT / "train" / "dpo_train.jsonl")
    mixed_grpo_path = PROCESSED_ROOT / "train" / "grpo_train_mixed.jsonl"
    grpo_path = (
        mixed_grpo_path
        if mixed_grpo_path.exists()
        else PROCESSED_ROOT / "train" / "grpo_train.jsonl"
    )
    grpo_records = read_jsonl(grpo_path)
    summary = {
        "cleaning": read_json(INTERIM_ROOT / "cleaning_summary.json"),
        "policy_chunk_lengths": plot_policy_lengths(policy_records, figure_root),
        "qrecc": plot_qrecc(qrecc_train, figure_root),
        "conditionalqa": plot_conditionalqa(conditional_train, figure_root),
        "final_training_mix": plot_training_mix(
            sft_records,
            cold_start_records,
            dpo_records,
            grpo_records,
            figure_root
        )
    }
    write_json(summary_path, summary)
    write_markdown_report(summary)
    logger.info("Wrote EDA report and %d figures", len(list(figure_root.glob("*.png"))))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
