# Proposal

## Why

Decoupled RL 岛的学习率会在训练中途**静默衰减到 0**，此后每一轮都在空转：梯度照常算出，参数却不再变化。2026-09-29 的 head 模式两岛 decoupled 运行（`yeto-hp929d`，`--total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2`）中，island 的 `train/lr-pg_0` 依次为 7.5e-6、5e-6、2.5e-6，之后 18 次记录全是 0.0。第 6 轮起 `grad_norm` 在 0.18–0.5 之间，`delta_l2_norm` 却是 0；syncer 从第 7 个外层步起 `global_delta_norm` 全为 0，最终 cut 两岛 hash 一致，run 以 rc=0 结束，没有任何环节报警。

根因在学习率调度的推导方式。Miles 用 `train_iters = num_rollout × rollout_batch_size × n_samples_per_prompt / global_batch_size` 推出总步数，`lr_decay_iters` 默认等于它，`lr_decay_style` 默认 `linear`，衰减终点为 `min_lr = 0`。yeto 传入的 `--num-rollout` 是全局轮数 `global_rounds`，这在 strict-avg 下正确：每个全局轮恰好一个本地轮。decoupled 则会一直跑到 syncer 发出最终 cut 为止（run-until-stop），本地步数没有事先可知的上限，只要超过 `global_rounds × optimizer_steps`，学习率就停在 0。

legacy 路径（agentenv/miles fork）与 ports 路径（upstream Miles）的推导完全相同，2026-09-29 的 decoupled 等价性实验里两边学习率序列逐位一致。按 `rl-engine-ports`（R0）"只换底座、不改行为"的约定，R0 期间两条路径都保留原行为；本 change 在两条路径上同时修复。benchmark 的 decoupled 配置之前没有暴露问题，只是因为本地步数恰好等于 `global_rounds`，并非设计上保证。

## What Changes

- **显式声明学习率调度，不再依赖 Miles 默认值。** 两条引擎路径都向 Miles 传入显式的衰减方式、衰减步数、warmup 步数和最小学习率。
- **decoupled 使用常数学习率。** 本地步数没有预知上限，任何有限的线性衰减终点都可能在运行中途到达 0，所以 decoupled 的学习率在整个运行期间保持为配置值。
- **strict-avg 的调度数值不变。** 显式写出原来隐含的线性调度（衰减步数 = `global_rounds × optimizer_steps`），每一步的学习率与修复前逐位相同，已有 strict-avg 证据仍然有效。
- **新增训练进展不变量：非最终轮学习率不得为 0。** 如果某个本地轮在应用优化器步时的有效学习率为 0，而该岛之后还会继续训练，这一轮 SHALL 失败，而不是提交一个零更新。每轮的有效学习率写入 `rl_local_round` 事件。
- **legacy 与 ports 同步修改，并重跑 decoupled 验证**：修复后两条路径的学习率序列仍须一致，decoupled 运行后半程的全局 delta 必须非零。
- **BREAKING（行为变更）**：decoupled 运行的学习率轨迹改变（从线性衰减改为常数），与修复前的 decoupled 运行不可逐位复现。strict-avg 不受影响。

## Capabilities

### New Capabilities
- `rl-learning-rate-schedule`：RL 岛的学习率调度必须与该岛实际执行的本地优化步数相符，在训练结束前不得降为 0；调度由 yeto 显式决定并在两条引擎路径上一致，一轮以 0 学习率执行优化步时必须显式失败。

### Modified Capabilities
<!-- 无。openspec/specs/ 目前只有 head-run-teardown，与学习率无关；
     rl-engine-ports 的 rl-engine-selection 要求两条路径行为等价，本 change 在两条路径上做同一修改，不改变该要求。 -->

## Impact

- **legacy 参数翻译**：`yeto/rl/learner.py` 的 `_legacy_miles_argv`（`--num-rollout` 在第 1000 行附近）；legacy 的 argv 快照 `tests/test_rl_argv_snapshot.py` 需要有意重新生成。
- **ports 参数翻译**：`yeto/rl/engine/run_config.py`（算法配置中新增调度字段）与 `yeto/rl/engine/miles_adapter/config.py`（翻译为 Miles 参数）。已有实现草稿：`/home/michael/work/followups/decoupled-constant-lr-ports.patch`。
- **不变量与事件**：`yeto/rl/engine/driver.py`（ports）与 legacy 的 policy-sync 桥接层（`yeto/rl/miles.py`），读取优化器步的有效学习率并写入 `rl_local_round`。
- **依赖**：ports 部分的代码只存在于 `rl-engine-ports` 分支，本 change 需在该分支合入 agentenv/yeto main 之后实施，或基于该分支实施。
- **不改动** Miles 或 SGLang 的 fork 代码；只通过 Miles 已有的命令行参数控制调度。

## Non-goals

- 不为 decoupled 引入学习率衰减或 warmup 策略。需要衰减时的做法（例如以显式的本地步数预算为终点）留给以后单独设计。
- 不改变 GRPO、奖励、采样或外层优化器（outer lr、momentum）。
- 不在本 change 内修复 head 模式的其他已知问题（island 事件磁带拆除前未回传、确定性失败时无限重试等）。
