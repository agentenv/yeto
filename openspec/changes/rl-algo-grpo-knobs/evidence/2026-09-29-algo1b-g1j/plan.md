# algo1b-g1j 计划：clip_higher 的最终实验（唯一一次；实验前提交，事后不改）

主 agent 决定：这是 clip_higher 唯一一次最终实验，无论结果如何都按预登记交付，此后不再为 clip_higher 另立实验。

## 设计
- 每轮 **3 个 optimizer step**（OPT_STEPS=3，每轮 32 条样本分成 3 个 step），使第 2、3 步成为 off-policy。**seed 固定为 17**，与 g1b/g1e 一致，不做挑选或更换。lr 1e-4，3 轮。
- A `clip_sym`：eps_clip 0.001（eps_clip_high 取同值）。B `clip_hi`：eps_clip 0.001，eps_clip_high 10。其余与 g1e/g1g 完全相同（sandbox harness，Qwen3-0.6B@c1899de2，每轮 4 组 × 8 条，response 384）。
- 归因：A 与 B 只有 eps_clip_high 一个参数不同，它只在 `compute_policy_loss` 的 clamp 上界处使用（`losses.py:211`、`math_utils.py:263`）。第 1 步 on-policy 且配对逐位相等时，此后的梯度只要出现差异，就只能来自上界。

## 事先固定的判据
- 前提：两个 run 都 rc=0、3 轮完成、无 invariant 错误（freeze_gc 良性链不算）。
- 配对有效：第 1 轮第 1 步 train/grad_norm 逐位相等，且第 1 步 rollout/raw_reward 相同。否则判"对照无效"。
- **clip_higher 生效**：配对有效，并且第 1 轮**第 3 步**的 train/grad_norm 在 A、B 之间**不相等**。
- 若第 1 轮第 1、2 步的梯度都为 0，导致第 3 步仍是 on-policy、上界无法触发，判"**未能证明生效**"（这**不是**无效运行，不重跑）。若第 3 步相等，同样判"未能证明生效"。这两种情况都不声明。
- 只有 harness 或环境问题（第一个训练 step 之前失败）才可以在修复后重跑。

## 资源与回收
Modal Sandbox `H100!`×1（运行前断言型号），app `algo1b-g1j`；sandbox timeout 10800 秒，独立 watchdog 11100 秒，每个 exec 1800 秒，本地 `timeout 11400`，EXIT trap；结束后执行 `modal app stop algo1b-g1j`。预计约 30 分钟，≤ $3。在 g1h、g1i 结束后再运行。

## 运行前修订（单独提交，发生在任何运行之前）
- 每轮 4 组 × 8 条 = 32 条样本，不能被 3 个 optimizer step 整除（run_config 算出的 global_batch 为 32//3=10，Miles 要求整除）。因此改为**每轮 6 组 × 8 条 = 48 条，每个 step 16 条**。这是唯一的偏离，其余不变：seed 17、lr 1e-4、eps 设定、3 轮、判据。之所以不用 3 组，是为了让每个 step 有足够的非零方差组，减少"第 1、2 步梯度为 0"的风险；这是在运行前做出的配置选择，没有看过任何结果。
