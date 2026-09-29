# algo1b-g1g 计划：clip 上界的梯度判据（新实验，实验前提交，事后不改）

与 g1e 的关系：这是一个新实验，**不是**对 g1e 事后改换判据。g1e 的结论（未能证明生效）保持不变。本实验换用 seed 18，其余与 g1e 相同。

## 设计
- A `clip_sym`：eps_clip 0.001（Miles 把 eps_clip_high 设为同值）。B `clip_hi`：eps_clip 0.001，eps_clip_high 10。
- 每轮 2 个 optimizer step，lr 1e-4，**seed 18**，3 轮。Qwen3-0.6B@c1899de2，每轮 4 组 × 8 条，response 384。sandbox harness 与 g1e 相同。
- **为什么 grad_norm 的差异可以归因于上界**：A 与 B 只有 eps_clip_high 一个参数不同，Miles 中 eps_clip_high 只在 `compute_policy_loss` 的 clamp 上界处使用（`losses.py:211`、`math_utils.py:263`，全仓库只有这一个调用点）。第 1 步为 on-policy，ratio≡1，两组的 loss 与更新完全相同（用第 1 步 grad_norm 逐位相等核验），所以第 2 步的输入与 ratio 也完全相同。在这个前提下，第 2 步梯度只要不同，唯一可能的来源就是上界。pg_clipfrac 不作为判据，理由见 `2026-09-29-clipfrac-offline/report.md`：它与 loss 不一致。

## 事先固定的判据
- 前提：两个 run 都 rc=0、3 轮完成、无 invariant 错误（freeze_gc 良性链不算）。
- 配对有效：第 1 轮第 1 步的 train/grad_norm 逐位相等，且第 1 步的 rollout/raw_reward 相同。否则判"对照无效"，不声明。
- **clip_higher 生效**：配对有效，并且第 1 轮第 2 步的 train/grad_norm 在 A、B 之间**不相等**。若相等，判"未能证明生效"，不声明。
- 结果按预登记如实交付；只有 harness 或环境问题可以在修复后重跑。

## 资源与回收
与 g1e 相同：Modal Sandbox `H100!`×1，app `algo1b-g1g`；sandbox timeout 10800 秒，watchdog 11100 秒，每个 exec 1800 秒，本地 `timeout 11400`，EXIT trap；结束后执行 `modal app stop algo1b-g1g`。预计约 25 分钟，≤ $3。在 g1f 结束后再运行。
