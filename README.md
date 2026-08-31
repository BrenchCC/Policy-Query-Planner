# Policy Query Planner Data Pipeline

本项目构建英国公共政策 Query Planner 所需的知识库、官方评测集、分层 SFT、单跳/多跳 DPO、单跳/多跳 GRPO-ready 数据，以及可选的火山方舟生成接口。当前阶段不执行模型训练；GRPO reward 和四卡训练参数已经完成方案设计，尚未实现和产生实验结果。

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
bash scripts/run_api_generation.sh test
bash scripts/run_api_generation.sh full
```

`test` 模式默认每阶段调用 2 条；`full` 模式依次完成 SFT、DPO、GRPO，并自动执行数据合并和全量校验。可以通过 `WORKERS`、`MAX_RETRIES`、`LOG_EVERY`、`TEST_LIMIT` 和 `TEST_WORKERS` 覆盖默认参数，例如 `WORKERS=8 bash scripts/run_api_generation.sh full`。

生成脚本使用 `tqdm` 展示实时进度。完整日志分别追加写入 `logs/generate_with_ark_<stage>.log`；终端会过滤 `httpx/httpcore` 的逐请求 INFO，只显示阶段进度、成功/失败统计和 Token 用量。API 输出只用于替换本地基线样本，原始响应、解析结果、Token 用量和失败原因会分别保存，不会覆盖原始数据。

## Main Outputs

- `data/processed/knowledge_base/policy.jsonl`：ConditionalQA 政策知识库。
- `data/processed/knowledge_base/musique_aux.jsonl`：MuSiQue 辅助知识库。
- `data/processed/eval/conditionalqa_dev.jsonl`：主要带标签 benchmark。
- `data/processed/eval/conditionalqa_test_blind.jsonl`：官方无答案测试集。
- `data/processed/eval/qrecc_test.jsonl`：官方 Query Rewrite 测试集。
- `data/processed/eval/musique_dev.jsonl`：官方多跳开发集。
- `data/processed/train/sft_train.jsonl`：20,000 条单跳 Alpaca SFT 基础数据。
- `data/processed/train/sft_multihop_cold_start.jsonl`：2,000 条 MuSiQue 多跳 SFT 冷启动数据，其中 2/3/4-hop 分别为 1,000/600/400 条。
- `data/processed/train/dpo_train.jsonl`：5,000 条 preference 数据；2,500 条单跳、2,500 条多跳，1/2/3/4-hop 分别为 2,500/1,250/1,000/250。
- `data/processed/train/grpo_train.jsonl`：5,000 条 GRPO-ready 数据；1,000 条单跳、4,000 条多跳，1/2/3/4-hop 分别为 1,000/2,586/910/504。
- `data/processed/train/grpo_train_domain_augmented.jsonl`：API 完成后的领域增强版本。
- `data/processed/dataset_info.json`：LLaMA-Factory 数据注册文件。

MuSiQue 多跳 SFT、DPO 与 GRPO-ready 三个集合同时进行来源 ID 和规范化问题指纹隔离，任意两组均零重叠。单跳 DPO 与单跳 GRPO 的 ConditionalQA 来源也保持零重叠。

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
  ├─ DPO：用单跳/多跳偏好对强化约束保真、步骤完整性与依赖正确性
  └─ GRPO：用 20% 单跳 + 80% 多跳优化检索收益并抑制过度拆分
```

统一评测基础 SFT、冷启动、DPO 和 GRPO 四个检查点，用于分别衡量多跳冷启动的独立贡献，以及两种后训练方法相对同一共享检查点的增益。所有指标在训练与正式评测完成前均标记为待测。

GRPO 计划使用 `Qwen/Qwen3-4B-Instruct-2507`，执行候选计划后，以冻结回复模型对 Gold 答案 token 的长度归一化概率构造二值奖励；同时按 `task_type` 和参考 `hop_count` 对不必要拆分、冗余查询及超出预期跳数的计划进行门控。自研 Trainer、veRL、4 张 A100 的显存分配和超参数估算见 [GRPO 算法与训练配置设计](docs/grpo_algorithm_design.md)。

DPO 与 GRPO 的 Planner Prompt 不直接透露样本是单跳还是多跳，而是统一要求生成“最小充分计划”：一跳足够时只输出一条 Query，确实需要多步证据时才建立依赖。`task_type` 和 `hop_count` 仅用于数据校验、分层采样和 reward 门控。

## Reproducibility

所有抽样使用固定随机种子。官方开发集和测试集不会参与训练数据或 API 请求构造。清洗、抽样、Schema 校验和 benchmark 泄漏检查都由测试覆盖。
