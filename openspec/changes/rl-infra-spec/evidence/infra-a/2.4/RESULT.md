# 2.4 结论（按预登记计划 3b26fde 的判据）

原始数据在 `sweep-result.json`（由 `analyze.py` 生成），各次运行的 tape/launch.log/modal 记录在对应子目录。所有运行均为 Modal 4×H100!（型号断言通过），Qwen3-0.6B LoRA、GRPO、strict-avg（1 个 learner）、6 轮，seed 17/29。

| 配置 | profile | s17 中位轮时 (s) | s29 中位轮时 (s) | 全池 GPU·h（s17 / s29） | 备用卡 |
|---|---|---|---|---|---|
| T2R2 | partitioned-serial | 89.3 | 86.9¹ | 1.75 / 1.41 | 0 |
| T1R3 | partitioned-serial | 74.5 | 75.5 | 1.33 / 1.30 | 0 |
| C4（默认兼容，共置 4 卡） | colocated-serial（参考，不参与比较） | 109.7 | 110.1 | 1.51 / 1.54 | 0 |
| T3R1 | — | 不合法（DP=3 不能整除 32 个样本），未运行 | | | |

¹ T2R2 s29 的第一次运行因 syncer 连接中断失败（基础设施），数据不计入。重跑（1e21296）与其他运行的代码差异只在 TCP keepalive 和 launcher 无进展超时，不涉及训练计算路径。

- **同 profile 比较（partitioned-serial）**：两个 seed 下 T1R3 的中位轮时都比 T2R2 低 ≥10%（s17 低 16.6%，s29 低 13.1%），按判据 **T1R3 为本池的最佳固定配置**。
- **收益面**：T1R3 的 trainer DP=1，但有 3 个 rollout engine；在这个小模型、短回复的负载下，rollout 侧并行的收益大于 trainer DP=2 的收益。各分段耗时见 `sweep-result.json` 的 `phase_s`（按相邻 phase 事件的时间差累加，只作描述，不作判据）。
- **默认兼容配置**：C4 共置约 110 s/轮。它属于不同 profile，只作为参考记录，不与分区配置比较收益。
- **净收益边**：本 change 尚未实现任何运行中切换（E1/E3 未完成），无法测量"切换收益 − 切换成本"。按计划与验收原文，结论为 **尚无净收益边**。固定配置之间的差异不等于存在可用的切换边。
- 失败与重跑记录：T2R2 s17 的第一次运行因 DistOpt 能力缺失失败（修复为 66afb1e/8f2c801，经冒烟验证）；T2R2 s29 的第一次运行因基础设施失败（修复为 12cde71 + `--rl-stall-timeout`）。两次重跑均经主 agent 同意，依据写在计划末尾。
