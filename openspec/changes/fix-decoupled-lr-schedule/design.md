# Design

## Context

- Miles（legacy fork 的 `miles/backends/megatron_utils/model.py:85-105`，upstream 的同名文件 `:80-106`）在构造调度器时按 `train_iters = num_rollout × rollout_batch_size × n_samples_per_prompt / global_batch_size` 推算总步数，`lr_decay_iters` 为 None 时取 `train_iters`，`lr_decay_style`、`min_lr` 取 Megatron 默认值（linear、0）。Megatron 要求 `lr_decay_steps > 0`。
- yeto 两条路径都只传 `--num-rollout global_rounds`，不传任何调度参数：legacy 在 `yeto/rl/learner.py` 的 `_legacy_miles_argv`（`:1000`），ports 在 `yeto/rl/engine/miles_adapter/config.py:520`。
- decoupled 由 `external_policy_sync_run_until_stop`（`learner.py:1813`）驱动，一直跑到最终 cut；唯一可选的硬上限是 `learner_budget_steps`。strict-avg 每个全局轮一个本地轮。
- 状态应用时的调度器对齐：legacy fork 的 `trainable_state._align_scheduler` 只会把调度器往前推；ports 的 `state_plugin.align_scheduler` 对齐到 `local_step`（strict 为 `policy_version × optimizer_steps`，decoupled 为累计本地步数）。两者都不会让衰减终点后移。
- 证据：head 模式 `yeto-hp929d`（`rl-engine-ports/evidence/2026-09-29-head-ports/attempt4-2cbd45c/`）；decoupled 等价性 `2026-09-29-eq63/`，两边学习率序列逐位一致。

## Goals / Non-Goals

**Goals:**
- 两条路径由同一处运行配置决定学习率调度，传给 Miles 的参数完全相同。
- decoupled 在任意本地步数下学习率非零；strict-avg 学习率逐步不变。
- 学习率降为 0 这种静默空转能被运行时检测并使该轮失败。

**Non-Goals:**
- 不修改 Miles/SGLang fork 代码，不改变调度器的对齐逻辑。
- 不为 decoupled 设计衰减策略。

## Decisions

**D1. 调度在引擎无关的运行配置中决定，两条路径只做翻译。**
在 `yeto/rl/engine/run_config.py` 的算法配置中新增调度字段（衰减方式 linear/constant、衰减步数），由同步方式推导；legacy 的 `_legacy_miles_argv` 与 ports 的 `translate_run_config` 都从这个字段翻译出 `--lr-decay-style`、`--lr-decay-iters`、`--lr-warmup-iters 0`、`--min-lr 0`。
- 备选：两条路径各自写死参数。会让两处定义漂移，违背 rl-engine-ports 的"一处配置、两处翻译"结构。

**D2. decoupled 用 constant，而不是按估计的本地步数上限设 linear。**
run-until-stop 下本地步数取决于 horizon、pipeline、fragment 数和各岛速度，事先不可知；任何有限的线性终点都可能在中途到达 0。`--lr-decay-iters` 仍传一个正数（取 `global_rounds × optimizer_steps`），只为满足 Megatron 的 `lr_decay_steps > 0` 断言，constant 调度不读取它。
- 备选 A：按 `learner_budget_steps` 设 linear 终点。只在设置了预算时有效，且预算只是上限，不是实际步数；留作以后的独立设计。
- 备选 B：把 `--num-rollout` 设得很大。会同时改变 Miles 其他依赖 `num_rollout` 的行为（循环终止、数据源），副作用不可控。

**D3. strict-avg 显式写出原隐式调度，数值不变。**
`linear`，衰减步数 = `global_rounds × optimizer_steps`，warmup 0，min_lr 0。这与 Miles 默认推导在 `rollout_batch_size × n_samples_per_prompt = global_batch_size × optimizer_steps` 时相同；yeto 的 RL 配置总是满足该等式（每轮 `optimizer_steps` 个优化步），实现时须用断言保证，而不是依赖巧合。

**D4. 不变量读取优化器步实际应用的学习率。**
Miles 在优化器步之后推进调度器，并在推进后记录 `train/lr-pg_0`，所以日志里的最后一个值总是下一步的学习率（strict 最后一步之后就是 0）。不变量必须读取优化器步内的学习率：ports 在已有的 `train_one_step` 记录器（`state_plugin.py`，记录 grad_norm 的同一处）取 `optimizer.param_groups[*]["lr"]`；legacy 在 fork 已有的 `custom_megatron_before_train_step_hook_path` 钩子中取（`yeto/rl/grad_audit.py` 已用同一钩子）。"非最终轮"的判定：strict 为 `local_round < global_rounds`，decoupled 为尚未收到最终 cut。
- 备选：只检查 `rl_local_round` 中的 `delta_l2_norm == 0`。delta 为 0 也可能由其他原因造成（例如优势全为 0），不能区分，且无法指出根因。

**D5. legacy 与 ports 在同一个 PR 中修改，并用等价性实验证明仍然一致。**
修复后 decoupled 的学习率轨迹改变，两条路径必须同时改变，才能保持 rl-engine-selection 的等价性要求。验证复用 `scripts/rl_engine_equivalence.py`：decoupled 两条路径的学习率序列逐位一致，且后半程全局 delta 非零。

## Risks / Trade-offs

- [旧 decoupled 运行不可逐位复现] → 在 CHANGELOG 或 `docs/MILES_RL.md` 中说明从哪个版本起 decoupled 改为常数学习率；旧运行的来源记录中本来就有 yeto commit，可据此区分。
- [常数学习率可能让长时间 decoupled 运行后期不稳定] → 不在本 change 内处理；可观测性上，每轮学习率写入事件，便于以后评估是否需要衰减。
- [legacy argv 快照需要重新生成] → 只改变新增的 4 个调度参数；快照测试改为断言除这 4 个参数外与修复前逐项一致。
- [不变量误杀最终轮] → 最终轮的判定按同步方式明确定义，并有专门的测试覆盖 strict 最后一轮和 decoupled 最终 cut 之后的情况。

## Migration Plan

1. 等 `rl-engine-ports` 合入 agentenv/yeto main（或基于该分支开发）。
2. 实施后在 GPU 上跑一次 decoupled（legacy + ports 各一次，配置同 head 模式 `yeto-hp929d`）和一次 strict-avg 回归，确认学习率序列与全局 delta。
3. 回滚：还原该 PR 即恢复原隐式调度；不涉及数据或状态格式变化。
