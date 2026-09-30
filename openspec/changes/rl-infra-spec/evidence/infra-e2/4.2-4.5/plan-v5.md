# 4.2–4.5 GPU 验证计划 v5（INFRA-E2，2026-09-30；取代 plan-v4.md，旧版保留）

## 变更（运行前，主 agent 裁定 2026-09-30；判据文字不变）
- **C1/C2 从共置改为 fixed-partition**：
  - C1：Qwen3-0.6B，trainer 1 卡（DP1）+ rollout 1 卡，H100!×2；
  - C2：Qwen3-0.6B，trainer 2 卡（DP2，DistOpt）+ rollout 1 卡，H100!×3。
  - 其余配置不变：LoRA r16、dropout 0.05、GBS 16、seed 1234、确定性设置。launcher 参数为 `--rl-placement fixed-partition --rl-rollout-gpus 1`。
- **理由**：
  - fork `RayWorkerManager.start_cells` 在启动 cell 前执行 `_assert_bundles_free`：只要与任何仍在运行的 cell 共用 bundle，就抛 `BundleInUseError`。2f23a0fc 与 5c1b49eb 都有这段检查。
  - 共置时 trainer 与 SGLang rollout cell 共用 bundle，同形重建的新 trainer 无法启动。C1 第 3 次运行中重建连续两次失败，证据见 `gpu-c1-attempt3/`。
  - 生产弹性路径本来就拒绝共置，4.4 的重建也跑在 fixed-partition 上；在共置上验证重建，等于验证生产不支持的配置。
- **C3 已核对**：本来就是 fixed-partition（T2R1S0，H100!×3），不变。
- **harness 按布局区分重新发布**：
  - fixed-partition：arm A 重新发布同一 policy；arm B 走生产路径 `driver.rebuild_trainer`（恢复、policy hash 核对、重新发布给所有成员）。两边对称。
  - 共置：两边都不重新发布。本版计划不再使用共置。
- 镜像、Miles 提交、模型 revision、运行顺序与 plan-v4 相同。

## 卡数、时长与费用（Modal H100 约 $3.95/卡·h；预估 = 实测节奏外推，最坏 = 95 min 硬停）
| run | 卡 | 预估时长 | 预估 | 最坏 |
|---|---|---|---|---|
| c1 | 2 | 15 min | $2.0 | $12.5 |
| c2 | 3 | 18 min | $3.6 | $18.8 |
| c3-b1、c3-rb、c3-rbold、c3-f1、c3-f2、c3-f3、c3-f5、c3-f6a、c3-f6b（9 个） | 各 3 | 各约 15–20 min | 各约 $3.3，合计约 $30 | 各 $18.8 |
| **整组** | | | **约 $35.5** | — |

执行规则：每个 run 启动前核对"B2 已花 + 本次最坏 ≤ $80"；实际累计接近 $40 时先停下报告。
