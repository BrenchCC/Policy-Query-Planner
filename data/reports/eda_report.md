# 数据清洗与 EDA 报告

## 数据概览

- ConditionalQA 原始文档：652 篇。
- 政策知识库：12552 个 chunk。
- ConditionalQA evidence 映射率：99.99%。
- MuSiQue 辅助知识库：117528 条去重段落。
- QReCC train：63501 条，其中非平凡改写 56685 条。

## 训练数据

- SFT：20000 条，构成为 `{"qrecc_nontrivial": 14130, "policy_domain": 2338, "qrecc_no_op": 3532}`。
- 多跳 SFT 冷启动：2000 条，hop 分布为 `{"2": 1000, "4": 400, "3": 600}`。
- DPO：5000 条，来源为 `{"qrecc": 2000, "musique": 2500, "conditionalqa": 500}`。
- DPO 单跳/多跳构成为 `{"single_hop": 2500, "multi_hop": 2500}`，hop 分布为 `{"1": 2500, "2": 1250, "3": 1000, "4": 250}`。
- DPO 十类单因素困难负样本严格均衡：`{"entity_omission": 500, "broken_dependency": 500, "step_omission": 500, "redundant_step": 500, "unresolved_reference": 500, "overly_broad": 500, "constraint_omission": 500, "overly_broad_step": 500, "wrong_context": 500, "relation_omission": 500}`。
- GRPO：5000 条，单跳/多跳构成为 `{"single_hop": 1000, "multi_hop": 4000}`，hop 分布为 `{"1": 1000, "3": 910, "2": 2586, "4": 504}`。

## 质量观察

- 官方 benchmark split 未参与训练抽样。
- 政策原始语料包含21篇未被官方问题 split 引用的文档；它们保留在知识库中，用于模拟真实检索噪声。
- 仅有1条 ConditionalQA heading-only evidence 未映射到正文 chunk，已记录在 `data/interim/gold_coverage_issues.json`。
- QReCC no-op 样本被保留，用于抑制模型对本来已独立的问题进行过度改写。
- 多跳 SFT、DPO 与 GRPO-ready 数据按来源 ID 和问题指纹严格隔离。
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
