# Policy Query Planner Data Pipeline

本项目构建英国公共政策 Query Planner 所需的知识库、官方评测集、分层 SFT、单跳/多跳 DPO、单跳/多跳 GRPO-ready 数据，以及可选的火山方舟生成接口。当前阶段不执行模型训练；GRPO reward 和四卡训练参数已经完成方案设计，尚未实现和产生实验结果。

## Environment

```bash
pip install -r requirements.txt
```

API 配置只通过环境变量读取，不要将密钥写入代码：

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="your-base-url"
export LLM_ENDPOINT="your-generation-endpoint"
export QUERY_MODEL="your-planner-model"
export RESPONSE_MODEL="your-answer-model"
export JUDGE_MODEL_1="your-first-judge-model"
export JUDGE_MODEL_2="your-second-judge-model"
export EMBEDDING_API_KEY="your-embedding-key"
export EMBEDDING_BASE_URL="https://your-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export EMBEDDING_MODEL="qwen3.7-text-embedding"
export EMBEDDING_DIMENSIONS="1024"
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

## Embedding Index Build

入库使用 `title + "\n\n" + text` 生成 1024 维向量，并为 `policy` 和 `musique_aux` 分别构建 L2 归一化后的 FAISS `IndexFlatIP` 索引。建议先把少量烟测产物写到临时目录：

```bash
python embedding/build_embedding_store.py --namespace all --limit 4 --output-root /tmp/policy-query-planner-vector-smoke
python embedding/build_embedding_store.py --namespace all --limit 8 --resume --output-root /tmp/policy-query-planner-vector-smoke
```

确认 API 和断点续跑正常后，再执行正式全量入库：

```bash
python embedding/build_embedding_store.py --namespace all
```

每个 namespace 生成 `vectors.pth`、`index.faiss`、`metadata.jsonl` 和 `manifest.json`。`vectors.pth` 是按源 JSONL 顺序排列的 `float32` PyTorch Tensor；manifest 记录源文件 SHA256、模型配置、已完成条数、索引条数和 Token 用量。已有产物必须显式使用 `--resume` 续跑或使用 `--force` 重建。

## Hybrid RAG

混合检索先使用 BM25 和 Embedding 分别召回候选，再通过 RRF 合并排名。已有 FAISS 索引无需重建，只需为相同 namespace 补建一次 SQLite FTS5 BM25 索引：

```bash
python retrieval/bm25_store.py --namespace all
```

确保 `.env` 包含 `LLM_API_KEY`、`LLM_BASE_URL`、`QUERY_MODEL`、`RESPONSE_MODEL` 和上述 `EMBEDDING_*` 配置，然后执行单条 RAG 查询：

```bash
python retrieval/run_rag.py \
  --namespace policy \
  --query "Can I make a new Child Tax Credit claim?"
```

CLI 使用 `tqdm` 在 stderr 展示 BM25、Embedding 与 RRF 融合进度，并向 stdout 输出结构化 JSON，包括改写计划、RRF 证据、BM25/Embedding 排名、带 `[1]` 格式引用的答案和 Token 用量。Query 改写失败时自动回退原始问题；答案生成失败时仍保留已召回证据和错误信息。Python 接口默认不显示进度，可在构造 `HybridRetriever` 时设置 `show_progress = True`。

Python 接口支持直接执行已经解析的独立多查询计划：

```python
from retrieval import QueryPlan, QueryStep

plan = QueryPlan(
    queries = (
        QueryStep(id = "q1", query = "Child Tax Credit eligibility"),
        QueryStep(id = "q2", query = "Universal Credit alternative")
    )
)
result = pipeline.run_plan("Which benefit can I claim?", plan)
```

`QueryStep.depends_on` 与 `{{qN.answer}}` 占位符协议也会保留。批量评测使用 `evaluation.runtime.ChainExecutor` 逐跳完成检索、中间答案生成和依赖替换；`run_plan` 仍只接收已经解析依赖的 Query。

## Evaluation

数据构建命令会在保留官方 benchmark 文件的同时生成两套模型就绪评测集及 `eval_manifest.json`：

- `rag_policy_eval.jsonl`：285 条 ConditionalQA Dev，用于政策 RAG 检索、回答与拒答评测。
- `multihop_planner_eval.jsonl`：2,417 条 MuSiQue Dev，用于 2/3/4-hop 查询计划和真实逐跳执行评测。

先按检查点生成预测。下面以小规模烟测为例；移除 `--limit` 后执行完整评测：

```bash
python evaluation/run_predictions.py \
  --dataset data/processed/eval/rag_policy_eval.jsonl \
  --planner-model "$QUERY_MODEL" \
  --output data/evaluation_runs/rag_policy/predictions.jsonl \
  --limit 10 \
  --force

python evaluation/run_predictions.py \
  --dataset data/processed/eval/multihop_planner_eval.jsonl \
  --planner-model "$QUERY_MODEL" \
  --output data/evaluation_runs/multihop_planner/predictions.jsonl \
  --limit 10 \
  --force
```

预测文件可重复离线评分。默认使用两个不同的 `JUDGE_MODEL_1` 和 `JUDGE_MODEL_2` 各投 3 票，并将 Judge 缓存写入报告目录；完整评分最多产生每样本 6 次 Judge 请求，建议先用 `--limit` 烟测成本。增加 `--skip-judge` 可只计算确定性指标：

```bash
python evaluation/score_predictions.py \
  --dataset data/processed/eval/rag_policy_eval.jsonl \
  --predictions data/evaluation_runs/rag_policy/predictions.jsonl \
  --output-dir data/evaluation_runs/rag_policy/report \
  --limit 10

python evaluation/score_predictions.py \
  --dataset data/processed/eval/multihop_planner_eval.jsonl \
  --predictions data/evaluation_runs/multihop_planner/predictions.jsonl \
  --output-dir data/evaluation_runs/multihop_planner/report \
  --limit 10
```

每个报告目录包含逐样本 `scores.jsonl`、汇总 `summary.json`、可读的 `report.md` 和可断点复用的 `judge_cache.jsonl`。预测 sidecar manifest 固定数据哈希、模型和检索参数；兼容续跑必须使用 `--resume`，参数变化后需要显式 `--force`。

## Model-Assisted Generation

先使用 dry-run 检查请求，不产生 API 费用：

```bash
python data_preprocess/generate_with_ark.py --stage sft --dry-run
python data_preprocess/generate_with_ark.py --stage dpo --dry-run
python data_preprocess/generate_with_ark.py --stage grpo --dry-run
```

SFT 和 DPO 的 API 增强是可选项；生成正式的领域混合 GRPO-ready 数据则需要执行 GRPO Teacher 调用。配置环境变量后，可由用户手动执行：

```bash
bash scripts/run_api_generation.sh test
bash scripts/run_api_generation.sh full
```

如果只需要重新生成 GRPO 混合集，可以单独执行：

```bash
python data_preprocess/prepare_generation.py --stage grpo
python data_preprocess/generate_with_ark.py --stage grpo --workers 8 --max-retries 2 --resume
python data_preprocess/finalize_datasets.py --stage grpo
python data_preprocess/validate_datasets.py --stage train
```

`test` 模式默认每阶段调用 2 条；`full` 模式依次完成 SFT、DPO、GRPO，并自动执行数据合并和全量校验。可以通过 `WORKERS`、`MAX_RETRIES`、`LOG_EVERY`、`TEST_LIMIT` 和 `TEST_WORKERS` 覆盖默认参数，例如 `WORKERS=8 bash scripts/run_api_generation.sh full`。

生成脚本使用 `tqdm` 展示实时进度。并发任务完成后先按请求序号进入缓冲区，再按原始请求队列顺序写入响应 JSONL；阶段结束时还会对包含历史断点结果的文件做一次顺序规范化。完整日志分别追加写入 `logs/generate_with_ark_<stage>.log`；终端会过滤 `httpx/httpcore` 的逐请求 INFO，只显示阶段进度、成功/失败统计和 Token 用量。API 输出用于替换 SFT/DPO 基线标签，并为 GRPO 合成经过证据校验的政策领域多跳样本。原始响应、解析结果、Token 用量和失败原因会分别保存，不会覆盖本地基线数据。

## Main Outputs

- `data/processed/knowledge_base/policy.jsonl`：ConditionalQA 政策知识库。
- `data/processed/knowledge_base/musique_aux.jsonl`：MuSiQue 辅助知识库。
- `data/processed/eval/conditionalqa_dev.jsonl`：主要带标签 benchmark。
- `data/processed/eval/conditionalqa_test_blind.jsonl`：官方无答案测试集。
- `data/processed/eval/qrecc_test.jsonl`：官方 Query Rewrite 测试集。
- `data/processed/eval/musique_dev.jsonl`：官方多跳开发集。
- `data/processed/eval/rag_policy_eval.jsonl`：模型就绪的 ConditionalQA RAG 评测集。
- `data/processed/eval/multihop_planner_eval.jsonl`：模型就绪的 MuSiQue 多跳 Planner 评测集。
- `data/processed/eval/eval_manifest.json`：评测集版本、数量、分层分布与 SHA256。
- `data/processed/train/sft_train.jsonl`：20,000 条单跳 Alpaca SFT 基础数据。
- `data/processed/train/sft_multihop_cold_start.jsonl`：2,000 条 MuSiQue 多跳 SFT 冷启动数据，其中 2/3/4-hop 分别为 1,000/600/400 条。
- `data/processed/train/dpo_train.jsonl`：5,000 条 preference 数据；2,500 条单跳、2,500 条多跳，1/2/3/4-hop 分别为 2,500/1,250/1,000/250。
- `data/processed/train/grpo_train.jsonl`：本地候选池；包含 1,000 条政策单跳与 4,000 条 MuSiQue 通用多跳，用于无 API 基线和后续混合抽样。
- `data/processed/train/grpo_train_mixed.jsonl`：正式的 5,000 条混合 GRPO-ready 数据；由 3,000 条 MuSiQue 通用多跳、1,000 条模型合成且证据校验通过的 ConditionalQA 领域多跳、1,000 条 ConditionalQA 领域单跳组成，比例为 60%/20%/20%。
- `data/processed/dataset_info.json`：LLaMA-Factory 数据注册文件。

MuSiQue 多跳 SFT、DPO 与 GRPO 候选池进行来源 ID 和规范化问题指纹隔离，任意两组均零重叠。单跳 DPO 与领域单跳 GRPO 的 ConditionalQA 来源也保持零重叠。合成的领域多跳问题必须不同于原始问题，并通过计划依赖、逐跳答案、证据索引、Gold chunk 和答案泄漏校验。

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
  └─ GRPO：60% 通用多跳 + 20% 领域多跳 + 20% 领域单跳
```

统一评测基础 SFT、冷启动、DPO 和 GRPO 四个检查点，用于分别衡量多跳冷启动的独立贡献，以及两种后训练方法相对同一共享检查点的增益。所有指标在训练与正式评测完成前均标记为待测。

GRPO 计划使用 `Qwen/Qwen3-4B-Instruct-2507`。总奖励包含 10% 的分层格式塑形和 90% 的端到端语义效果：格式部分检查 JSON、Schema、编号与依赖；每个可执行计划实际完成逐跳检索、子答案生成与最终答案生成。子答案正确率用于过程塑形，最终答案通过规则与独立 Judge 检查正确、完整、证据支持和矛盾，并作为语义一票否决；同时按 `task_type` 和参考 `hop_count` 对不必要拆分、冗余查询及超出预期跳数的计划进行硬门控。完整方案见 [GRPO 算法与训练配置设计](docs/grpo_algorithm_design.md)。

DPO 与 GRPO 的 Planner Prompt 不直接透露样本是单跳还是多跳，而是统一要求生成“最小充分计划”：一跳足够时只输出一条 Query，确实需要多步证据时才建立依赖。`task_type` 和 `hop_count` 仅用于数据校验、分层采样和 reward 门控。

领域多跳不是把原始 ConditionalQA 问题机械拆成多条 Query。Teacher 根据至少两个独立政策 Gold chunk 合成新场景和新问题，同时输出私有的 `hop_answers` 与证据索引；流水线要求每跳答案能在对应 evidence 中找到、每跳使用不同 Gold chunk、至少存在一条真实依赖，并阻止最终答案泄漏进 Planner 计划。Teacher 的私有标注只进入 reward 侧，Planner 训练输入仍然只有 `system + instruction + input`。

## Reproducibility

所有抽样使用固定随机种子。官方开发集和测试集不会参与训练数据或 API 请求构造。清洗、抽样、Schema 校验和 benchmark 泄漏检查都由测试覆盖。
