# Policy Query Planner Data Pipeline

本项目构建英国公共政策 Query Planner 所需的知识库、官方评测集、单跳 SFT、多跳 SFT 冷启动、DPO、GRPO-ready 数据，以及可选的火山方舟生成接口。当前阶段不执行模型训练，也不实现 reward。

## Environment

```bash
pip install -r requirements.txt
```

API 配置只通过环境变量读取，不要将密钥写入代码：

```bash
export LLM_API_KEY="your-key"
export LLM_API_BASE_URL="your-base-url"
export LLM_ENDPOINT="your-endpoint"
```

## Local Pipeline

以下命令不会调用外部大模型：

```bash
python data_preprocess/download_data.py
python data_preprocess/clean_data.py
python data_preprocess/build_datasets.py --force
python data_preprocess/analyze_data.py --force
python data_preprocess/prepare_generation.py --stage all
python data_preprocess/validate_datasets.py --stage all
pytest -q data_preprocess/tests
```

下载脚本固定上游提交并记录 SHA256。重复执行时，校验成功的文件会被跳过；只有显式指定 `--force` 才会重新下载。

## Optional API Generation

先使用 dry-run 检查请求，不产生 API 费用：

```bash
python data_preprocess/generate_with_ark.py --stage sft --dry-run
python data_preprocess/generate_with_ark.py --stage dpo --dry-run
python data_preprocess/generate_with_ark.py --stage grpo --dry-run
```

配置环境变量后，可由用户手动执行：

```bash
python data_preprocess/generate_with_ark.py --stage sft --resume
python data_preprocess/generate_with_ark.py --stage dpo --resume
python data_preprocess/generate_with_ark.py --stage grpo --resume
python data_preprocess/finalize_datasets.py --stage all
```

API 输出只用于替换本地基线样本。原始响应、解析结果、token 用量和失败原因会分别保存，不会覆盖原始数据。

## Main Outputs

- `data/processed/knowledge_base/policy.jsonl`：ConditionalQA 政策知识库。
- `data/processed/knowledge_base/musique_aux.jsonl`：MuSiQue 辅助知识库。
- `data/processed/eval/conditionalqa_dev.jsonl`：主要带标签 benchmark。
- `data/processed/eval/conditionalqa_test_blind.jsonl`：官方无答案测试集。
- `data/processed/eval/qrecc_test.jsonl`：官方 Query Rewrite 测试集。
- `data/processed/eval/musique_dev.jsonl`：官方多跳开发集。
- `data/processed/train/sft_train.jsonl`：20,000 条单跳 Alpaca SFT 基础数据。
- `data/processed/train/sft_multihop_cold_start.jsonl`：2,000 条 MuSiQue 多跳 SFT 冷启动数据，其中 2/3/4-hop 分别为 1,000/600/400 条。
- `data/processed/train/dpo_train.jsonl`：5,000条 LLaMA-Factory preference 数据。
- `data/processed/train/grpo_train.jsonl`：从冷启动集之外的 MuSiQue 样本中抽取的 5,000 条多跳 GRPO-ready 数据。
- `data/processed/train/grpo_train_domain_augmented.jsonl`：API 完成后的领域增强版本。
- `data/processed/dataset_info.json`：LLaMA-Factory 数据注册文件。

MuSiQue 冷启动集与 GRPO-ready 集同时进行来源 ID 和规范化问题指纹隔离，二者零重叠，避免监督阶段提前看到强化学习样本。

## Training and Evaluation Plan

计划训练路径为：

```text
基础模型
  ↓
20K 单跳基础 SFT
  ↓
2K MuSiQue 多跳 SFT 冷启动
  ↓
共享冷启动检查点
  ├─ DPO：优化单跳查询偏好与约束保真
  └─ GRPO：优化多跳规划与检索收益
```

统一评测基础 SFT、冷启动、DPO 和 GRPO 四个检查点，用于分别衡量多跳冷启动的独立贡献，以及两种后训练方法相对同一共享检查点的增益。所有指标在训练与正式评测完成前均标记为待测。

## Reproducibility

所有抽样使用固定随机种子。官方开发集和测试集不会参与训练数据或 API 请求构造。清洗、抽样、Schema 校验和 benchmark 泄漏检查都由测试覆盖。
