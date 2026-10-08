# Proposal：RL 算法补充（rl-algo-supplement）

> **S14 对账说明（2026-10-07，基线 integ-decl e8d387ac）**：本 change 自 2026-09-30 规划后从未 apply（见 `infra-drafts/HANDOFF-AUDIT-S14.md` A1/B6）。本次把它与 S4–S14 期间在其它 change 里完成的算法工作对账（结果见 `progress.md`、`tasks.md` 顶部），并确定：**本 change 作为算法工作的唯一基准**，其它 rl-algo-* change（critic-family、loss-variants、grpo-knobs、mismatch-correction、seq-and-adv、fix-decoupled-lr-schedule）的未完成项以引用形式汇入 `tasks.md` §14（不复制、不改写来源 change 的勾选），接下来的阶段计划与 GPU 估算见 `tasks.md` §13。对账口径：只勾有文件:行证据的项；"已实现未验证"与"未做"分开标注；不可凭印象。
>
> 与 09-30 版相比的现状变化（细节见 design.md "S14 现状"）：Miles pin 已从 `e3a11ab3` 移到 `c35702e`（`yeto/rl/__init__.py:43`），`MILES_DECLARED` 24 项（`miles_adapter/entry.py:67-141`）内容未变；trainer v2（GMPO num/den）已合入；critic 族（PPO/VAPO/SAO/CompactionRL）已在 rl-algo-critic-family 中实现并有 G1 证据，本 proposal 原"critic/PPO/VAPO 等 rl-infra 完成后再开"的非目标表述已过时，但其正式声明与 G3 仍未完成，作为汇入项跟踪。阶段二字段全部未实现。

## Why

五个算法 change（P0、1a、1b、2a、2b）已把大部分机制做到“可表达”，但截至 `origin/integ-decl`（Miles pin `e3a11ab3`）仍有一批机制停在“可表达未开放”：CISPO/SAPO/GMPO、dual_clip、KL k1/k2/low_var_kl、kl_unbiased、opsm_rollout、no_rewards_normalization、grpo+whiten。它们只能在单岛无同步下放行，不能进联邦训练。已声明项里也有三处证据偏弱：1b 的 pg_clipfrac 疑似与 loss 不一致，over_sampling 的决定性证据是事后推断，clip_higher 的证据不是在当前 pin 上跑的。另外，rollout 采样参数、MoE routing replay、ref 周期更新、loss mask 策略、RLOO 基线会改变优化目标，但目前或者不在 `AlgorithmSpec` 与哈希里，或者还不能表达，两岛算法身份因此不完整。

用户已拍板范围与顺序：先用便宜卡补齐证据、据证据开放声明（阶段一），再补不受 infra 阻塞的新机制（阶段二）。阶段一的测试计划须经用户确认后才 apply。

## What Changes

**阶段一：补证据，据证据开放声明**（tasks 第 1–6 组）
- 1b `pg_clipfrac` 疑似缺陷：先在 CPU 上离线复现并定性。确认是 fork 缺陷时，在 `michaellchung/miles` 的 `yeto/ports` 上做最小修复，否则如实记录。结论出来之前，clipfrac 不作为任何机制的生效证据。
- fork（只改 `yeto/ports`，默认行为与数值不变）新增两项观测指标：dual-clip 分支生效比例 `dual_clipfrac`，以及超采样下每次提交的批大小。
- 用便宜卡（Modal A10G / L4 / T4，单卡最小配置）做 G1 生效验证。每项判据、最小证据、硬超时、费用上限都事先固定，覆盖：
  - CISPO / SAPO / GMPO：每轮 ≥2 个 optimizer step，避免 ratio=1 的退化；
  - dual_clip：≥2 步，`eps_clip_c` 取 1.01；
  - KL 估计器 k1 / k2 / low_var_kl，以及 kl_unbiased；
  - opsm_rollout（rollout logprob 当作 π_old）；
  - no_rewards_normalization；
  - grpo + whiten（用户已批准开放；变 DP 边仍不认证）。
- over_sampling 用新指标重做强证据；clip_higher 在当前 pin 上独立重跑。
- 通过的项逐条写入 Miles adapter 的声明并附证据路径；未通过的保持“可表达未开放”，如实记录。
- 两岛 G3：1b 的 8.4 组合冒烟；CISPO 两岛 strict-avg 为可选项，列出成本。两者都复用 1a 修复 island-1 磁带截断时用的 launcher 回传磁带 harness。
- 写入规则：用户自带代码的机制（`custom_loss`、`plugins`、`corrections:custom`、任意 `custom_pg_loss_reducer`）**永久**“可表达未开放”，只在单岛无同步下放行，不接受声明。

**阶段二：新机制进入算法身份**（tasks 第 7–12 组，阶段一完成后开始）
- rollout 采样参数 `temperature` / `top_p` / `top_k` 进入 `AlgorithmSpec` 与哈希。取默认值时不输出，哈希不变。
- MoE routing replay（`--use-rollout-routing-replay` / `--use-routing-replay`）从 agent 配置移入算法身份，并用小 MoE 模型做 GPU 验证。**BREAKING（仅身份）**：已经启用 rollout routing replay 的运行，算法哈希会变化；旧 CLI 参数仍可用，会被吸收进 spec。
- `--ref-update-interval`（ref 周期更新）：进入 spec 与哈希，cut 中记录 ref 版本。依赖 rl-infra-spec 4.1/4.2 的接口。
- `rollout.loss_mask_policy`：新声明字段，进入哈希。
- RLOO 留一基线：作为 reward 分派器插件的一个阶段实现，具有独立的算法身份。

**不在本 change（非目标 / 后续）**
- OPD（on-policy distillation）：用户要求单独立 change。
- critic / PPO / VAPO 族：**S14 更新**——已由 `rl-algo-critic-family` 实现（CPU 全通过，G1 PASS：VAPO vapo-20261007e、SAO s14-forkg1-sao-20261007a、GAE 变体 gae-la/gae-cs），其未完项（G3、pin 6.3、正式声明）汇入本 change tasks §14，不在此复制实现。异步 / `max_policy_staleness>0` / partial rollout 契约：仍列 backlog。
- 多轮 agentic 信用分配（GiGPO 等）：依赖 rl-infra 3.3，列为后续。
- 效果 A/B（训练收益）：不做。“已声明”只表示机制在 GPU 上确实生效。
- `execution.max_policy_staleness` 仍固定为 0。

## Capabilities

### New Capabilities
- `rl-algo-declaration-evidence`：机制从“可表达未开放”转为“已声明”时的证据契约（事先固定判据、非退化条件、便宜卡、费用上限、失败即停、按 pin 限定）、需要的 fork 观测指标、既有弱证据的复核，以及用户代码类机制永不声明的规则。
- `rl-algo-identity-extensions`：rollout 采样参数、MoE routing replay、ref 周期更新、loss mask 策略、RLOO 基线进入 `AlgorithmSpec` 规范化与哈希时的行为（默认值哈希不变、吸收与冲突、拒绝规则、cut 中的 ref 版本）。

### Modified Capabilities
（无。`openspec/specs/` 下还没有算法相关的主 spec；前五个 change 的 spec 尚未归档，本 change 以新 capability 的形式追加，不改写它们的需求。）

## Impact

- **ALGO 负责（本 change 直接改）**：`yeto/rl/engine/algorithm.py`、`yeto/rl/algos/*`（新增采样/身份扩展模块、RLOO 阶段）、`yeto/rl/engine/miles_adapter/algorithm_flags.py`（从 `UNMAPPED_OBJECTIVE_FLAGS` 移出 `--rollout-temperature/--rollout-top-p/--rollout-top-k/--use-routing-replay/--ref-update-interval`）、`miles_adapter/config.py` 的算法部分、`miles_adapter/entry.py` 的 `MILES_DECLARED`/`MILES_DECLARED_PINS`、`yeto/rl/engine/fake.py`、`docs/MILES_RL.md` 的算法小节，以及对应测试。
- **INFRA 负责（以补丁或接口请求交付，不直接改）**：`miles_adapter/trainer.py`（GMPO num/den 取数的 v2 补丁是否已合入需先核实；新指标取数）、`driver.py`、`cut.py`/`ledger.py`（cut 中的 ref 版本，对应 rl-infra 4.1/4.2）。
- **IMG 负责**：fork 新提交之后 `yeto/rl/__init__.py` 中的 pin、镜像重建，以及 `loss_variants.FORK_COMMITS`、`MILES_DECLARED_PINS` 的迁移论证。
- **主 agent 指定负责人**：`yeto/cli.py`、`yeto/launcher.py`、`yeto/rl/learner.py` 中 `--use-rollout-routing-replay` 的接线迁移。
- **fork**：`michaellchung/miles` 的 `yeto/ports` 分支，只做小提交（dual_clipfrac、超采样提交批大小、可能的 clipfrac 修复）。每个提交默认不改变数值。
- **云费用**：阶段一 ≤ $35，阶段二 ≤ $25（含 MoE 小模型），均用便宜卡，每项另有上限（见 design）。
