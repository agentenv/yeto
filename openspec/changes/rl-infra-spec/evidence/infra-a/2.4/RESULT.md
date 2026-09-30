# 2.4 结论（按预登记计划 3b26fde 的判据）

原始数据在 `sweep-result.json`（由 `analyze.py` 生成），各次运行的 tape/launch.log/modal 记录在对应子目录。所有运行均为 Modal 4×H100!（型号断言通过），Qwen3-0.6B LoRA、GRPO、strict-avg（1 个 learner）、6 轮，seed 17/29。

| 配置 | profile | s17 中位轮时 (s) | s29 中位轮时 (s) | 全池 GPU·h（s17 / s29） | 备用卡 |
|---|---|---|---|---|---|
| T2R2 | partitioned-serial | 89.3 | 86.9¹ | 1.75 / 1.41 | 0 |
| T1R3 | partitioned-serial | 74.5 | 75.5 | 1.33 / 1.30 | 0 |
| C4（默认兼容，共置 4 卡） | colocated-serial（参考，不参与比较） | 109.7 | 110.1 | 1.51 / 1.54 | 0 |
| T3R1 | — | 不合法（DP=3 不能整除 32 个样本），未运行 | | | |

¹ T2R2 s29 的第一次运行因 syncer 连接中断失败（基础设施），数据不计入。**更正（独立审查）**：重跑用的 1e21296 与其他 5 次运行用的 ebcdad9 之间，除 openspec 外的真实差异为 16 个文件、+878/−42 行（`git diff --stat ebcdad9 1e21296 -- . ':!openspec'`），并不只是网络修复。其中运行时代码的差异：
   - `yeto/protocol.py`：TCP keepalive；
   - `yeto/launcher.py`（约 289 行）与 `yeto/cli.py`：island 失败退出与 `--rl-stall-timeout`；
   - `yeto/rl/engine/driver.py`：tool_wait/submitted 字段写入 `rl_round_trained`，以及 `_load_sampler` 线程（只在 observe 开启时运行）；
   - `miles_adapter/rollout.py`、`rollout_meta_hook.py`、`ports.py`：新增元数据字段（tool_wait_seconds 仅在非零时写入，submitted/aborted_in_flight_groups），以及 router 探针（只在 observe 开启时调用）；
   - `yeto/rl/event_echo.py`。
   以上改动都不在训练与生成的计算路径上。本次运行未开启 observe，tape 中没有 `rl_load_sample`（0 条），说明采样线程未运行。元数据只多一次在 hook 内的求和，以及一次读取 data_source offset。T1R3 在 s29 上的优势为 13.1%，这些改动带来的影响不足以抵消这一差距。其余差异为测试与 docs。


- **适用范围**：这里的"最佳固定配置"只对本次的 4 卡池（T2R2/T1R3）成立，不代表 8 卡池 P62/P44 的结论。
- **同 profile 比较（partitioned-serial）**：两个 seed 下 T1R3 的中位轮时都比 T2R2 低 ≥10%（s17 低 16.6%，s29 低 13.1%），按判据 **T1R3 为本池的最佳固定配置**。
- **收益面**：T1R3 的 trainer DP=1，但有 3 个 rollout engine；在这个小模型、短回复的负载下，rollout 侧并行的收益大于 trainer DP=2 的收益。各分段耗时见 `sweep-result.json` 的 `phase_s`（按相邻 phase 事件的时间差累加，只作描述，不作判据）。
- **默认兼容配置**：C4 共置约 110 s/轮。它属于不同 profile，只作为参考记录，不与分区配置比较收益。
- **净收益边**：本 change 尚未实现任何运行中切换（E1/E3 未完成），无法测量"切换收益 − 切换成本"。按计划与验收原文，结论为 **尚无净收益边**。固定配置之间的差异不等于存在可用的切换边。
- 失败与重跑记录：T2R2 s17 的第一次运行因 DistOpt 能力缺失失败（修复为 66afb1e/8f2c801，经冒烟验证）；T2R2 s29 的第一次运行因基础设施失败（修复为 12cde71 + `--rl-stall-timeout`）。两次重跑均经主 agent 同意，依据写在计划末尾。

## 补记（独立审查）
- T2R2 两个 seed 的 train 分段累计差异较大：s17 为 91 s，s29 为 61 s（s29 是重跑的那次）。原因**未查明**。该分段按相邻 phase 事件的时间差累加，会把阶段间等待混入，只作描述；判据用的是逐轮 wall 中位数，不受影响。
- `plan.md` 中关于"launcher 关闭阶段误判"的规则提交于 83438f6（2026-09-29 20:44:31 UTC），比 T2R2 s17 的启动时间（20:43:45 UTC）晚 46 秒。该次运行退出码为 0，这条规则没有被用上。
