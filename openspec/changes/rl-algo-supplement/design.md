# Design：RL 算法补充

## Context

> **S14 现状（2026-10-07，e8d387ac；本节覆盖下方 09-30 原 Context 中已变化的事实，原文保留以示差异）**
>
> | 项 | 09-30 规划时 | e8d387ac 现状 | 证据 |
> |---|---|---|---|
> | Miles pin | `e3a11ab3` | `c35702e`（= 1023269 + A27-2） | `yeto/rl/__init__.py:43`；`loss_variants.py:78-90` FORK_COMMITS 含之 |
> | `MILES_DECLARED` | 24 项 | 24 项，未增（无 CISPO/SAPO/GMPO/dual_clip/k1/k2/low_var_kl/kl_unbiased/opsm_rollout/no_rewards_normalization/whiten） | `miles_adapter/entry.py:67-141` |
> | `MILES_DECLARED_PINS` | 4 项钉 0af62f4d+ | 同 4 项，迁移集合扩到 7 个 pin（含 c35702e），全部靠 git diff 论证 | `entry.py:143-180` |
> | trainer v2（GMPO num/den） | 未确认合入 | **已合入** | `miles_adapter/trainer.py:101,601-611` |
> | fork 观测指标 dual_clipfrac / 超采样提交批大小 | 待做 | **仍无** | `git -C miles-next grep dual_clipfrac c35702e` 为空 |
> | clipfrac 疑点 | 疑似 | 仍疑似，未定性 | `rl-algo-grpo-knobs/evidence/2026-09-29-clipfrac-offline/report.md` |
> | 磁带回传 harness | 1a g3c/g3d 用过 | 已被 seq-and-adv attempt 6（gspo/rpp/gdpo 5 run PASS）与 critic-family G1（`s1-runs/s13-forkg1-modal.sh`）复用 | `rl-algo-seq-and-adv/progress.md:178`；`rl-algo-critic-family/progress.md:489-547` |
> | 阶段二字段 | 无 | **仍无**：`algorithm.py` 的 `SamplingSpec`（:820）只有 filter/over_sampling/overlong_filter；无 moe/ref_update_interval/loss_mask_policy/RLOO；`--rollout-temperature/top-p/top-k` 仍在 `_UNMAPPED`（`algorithm_flags.py:250-253`）；`--use-rollout-routing-replay` 仍走 `agent.*`（`config.py:430,919`、`run_config.py:405`、`learner.py:1616`） | 同左 |
> | critic 族 | 非目标 | `rl-algo-critic-family` 已实现：`AlgorithmSpec.critic`（`algorithm.py:873-899`）、`execution.needs_critic`、PPO/VAPO/SAO/CompactionRL；G1 PASS 但未声明、G3 未跑 | `rl-algo-critic-family/tasks.md`（33/43 勾，含后续组）、progress.md |
> | GPU 卡型 | A10G 默认（D9） | S14 规则：GPU 默认 Modal，H100 用 `H100!`；需 syncer 的两岛 controller+syncer 放 Nebius 无卡 VM | `infra-drafts/S14-SUBAGENT-RULES.md`；决策点 tasks §13 Q1 |
> | 测试基线 | — | 最近全量 `infra-drafts/s13-full2-pytest.log`：68 failed / 4084 passed / 51 skipped / 26 errors（HEAD 未记）；本机禁止起 Ray 的用例 | 同左 |
>
> **算法清单与当前状态（入口 `yeto/rl/engine/algorithm.py`，注册模块 `yeto/rl/algos/*`，adapter `yeto/rl/engine/miles_adapter/{algorithm_flags,entry,config}.py`）**
>
> | 算法/机制 | spec 字段 → Miles 参数 | 状态 | 来源 change |
> |---|---|---|---|
> | GRPO（R0）、GSPO、REINFORCE++/baseline、MaxRL、MAPO、GDPO | `advantage.estimator`、`advantage.whiten`、`advantage.transform/gdpo`、`advantage.gamma` → `--advantage-estimator`、`--normalize-advantages`、`--gamma` | 已声明（attempt6） | rl-algorithm-capabilities / rl-algo-seq-and-adv |
> | TIS、OPSM(trainer)、IcePop、mis_mask、mismatch_observe | `correction.method/tis_clip/opsm_delta/function/use_rollout_logprobs/mismatch_metrics` | 已声明；`opsm_rollout`（use_rollout_logprobs）未验证 | rl-algo-mismatch-correction |
> | clip_higher、eps_clip、token/constant 聚合、Dr.GRPO reducer、no_grpo_std、KL(loss,k3)+ref、entropy、overlong penalty/filter、over_sampling、custom_reward_postprocess | `loss.eps_clip/eps_clip_high/aggregation/reducer`、`kl.placement/coef/estimator/unbiased`、`entropy_coef`、`sampling.*` | 已声明（clip_higher/over_sampling 证据弱，见 entry.py:106-141） | rl-algo-grpo-knobs |
> | dual_clip（`loss.eps_clip_c`）、KL k1/k2/low_var_kl、kl_unbiased、no_rewards_normalization、grpo+whiten | 可表达 | **可表达未开放**（放行开关仅单岛无同步） | 本 change 4.x |
> | CISPO / SAPO / GMPO | `loss.variant` + `loss.ext`（sapo_tau_pos/neg、gmpo_log_clip_low/high）→ `--policy-loss-variant` 等（fork 路线 B，`yeto/rl/algos/loss_variants.py`） | CPU 通过、T4 parse_args 通过；**G1 未跑** | rl-algo-loss-variants |
> | PPO / VAPO / SAO / CompactionRL（critic 族） | `critic.*`（`algorithm.py:873-899`）、`execution.needs_critic`、`--policy-objective`、`--value-loss-type` 等 | 已实现；G1 PASS；**未声明、G3 未跑** | rl-algo-critic-family |
> | decoupled LR | 非算法身份（调度） | 已修（9bcecdea）；3.2 GPU 复跑未做 | fix-decoupled-lr-schedule |
> | sampling.temperature/top_p/top_k、moe.routing_replay、kl.ref_update_interval、rollout.loss_mask_policy、RLOO | — | **未实现** | 本 change 8–12 |
>
> **D9 修订建议（待拍板）**：若按 S14 规则改用 H100!，阶段一上限由 $35 提至 ≈$45，阶段二 $25 不变；A10G 方案维持原表。pin 更新（D4）与 critic-family 6.3 合并为一次，避免两次重建镜像。

### 09-30 原 Context（保留）

动机与范围见 proposal.md。这里只列影响做法的现状：

- **声明机制**：一个机制算“已声明”，要同时满足两条：出现在 `miles_adapter/entry.py::MILES_DECLARED` 里（目前 24 项，每项附证据路径），并且当前 pin 满足 `MILES_DECLARED_PINS`。未声明的机制只能用 `--rl-allow-unverified-mechanism 维度:名称` 放行，而且只在单岛无外层同步的运行中生效（P0 design D11）。
- **2b 变体**：fork 从 `5c1b49eb` 起包含 CISPO/SAPO/GMPO，当前 pin `e3a11ab3` 在 `FORK_COMMITS` 中，T4 上完整 `parse_args` 已通过。GPU 冒烟还没做，计划已预登记在 `rl-algo-loss-variants/progress.md`（修订版）。GMPO 零梯度判定读 `gmpo_clip_num/den`，取数路径在 `infra-drafts/patches/algo-2b-trainer-v2.patch`（`trainer.py` 归 INFRA），这个补丁有没有合入还未确认。
- **1b 疑点**：`pg_clipfrac` 与 loss 不一致，报告在 `rl-algo-grpo-knobs/evidence/2026-09-29-clipfrac-offline/report.md`。g1e 中 A、B 两组 clipfrac 逐位相同，grad_norm 却不同。有一个怀疑是 `@torch.compile(dynamic=True)` 在 `eps_clip == eps_clip_high` 时做了特化，尚未确认。dual_clip 在 g1 中与基线逐位相同，而且 Miles 没有 dual 分支的指标。over_sampling 的 g1h 判据区分力弱。clip_higher 的证据跑在 `0394715` 上，是靠 git diff 迁移到当前 pin 的。
- **1a G3**：第一次两岛运行因为 island-1 磁带拉取截断而失败。之后改用 launcher 回传磁带，重跑 g3c（TIS）和 g3d（IcePop），判据全部通过。1b 的 8.4 还没跑。
- **不在 spec 中的目标参数**：`--rollout-temperature/--rollout-top-p/--rollout-top-k`、`--use-routing-replay`、`--ref-update-interval` 目前在 `UNMAPPED_OBJECTIVE_FLAGS` 中，写进 extra argv 会被拒绝。`--use-rollout-routing-replay` 走 `agent.use_rollout_routing_replay`（`config.py` 与 `learner.py`），不进哈希。rollout 采样取 Miles 默认值（temperature 1.0 等）。仓库里没有 `loss_mask_policy` 字段。
- **Miles 的 ref 周期更新**：Megatron actor 在 `megatron_utils/actor.py:352` 断言：ref 周期更新只保存在内存里，没有任何 checkpoint 持有它。因此 cut/恢复必须显式处理 ref 版本。
- **与 infra 的关系**：`max_policy_staleness` 固定为 0，与 rl-infra-spec 的严格 profile 一致（alignment A6）。变 DP（E3）边只认证默认 GRPO（§8 第 3 项），`reshard.py` 已经把 `advantage.whiten` 列为不认证。

## Goals / Non-Goals

**Goals**
- 每个声明都有“确实生效”的 GPU 证据，而且证据与判据的提交早于运行。
- 阶段一全部在便宜卡上完成。只有在确有逐位比较必要时才用 H100，需要写明理由；本计划没有这种项。
- 阶段二新增的字段取默认值时，规范化结果与哈希都不变。

**Non-Goals**
- 不改变 infra 的写入边界。INFRA 负责的文件只接受补丁或接口请求。
- 不做效果 A/B，不给出默认开启的建议。
- 不认证任何非默认 GRPO 算法的变 DP 边。

## Decisions

### D1 阶段划分与门禁
阶段一对应 tasks 第 1–6 组，阶段二对应第 7–12 组。第 1 组先提交完整的阶段一 GPU 测试计划：逐项列出判据、最小证据、硬超时、费用上限。用户确认后才进入第 2 组以后的 apply。阶段二的 GPU 计划同样先提交，由用户确认。备选方案是两个阶段并行，没有采用，原因是用户已明确规定顺序，而且阶段二的 cut/ref 接口依赖 infra 的进度。

### D2 生效证据的统一口径
每个 G1 项都满足以下要求：
- 单卡（两岛项是 1+1 卡），模型为 Qwen2.5-0.5B LoRA，单岛 colocated-serial，seed 固定为 17，只跑一次，不挑 seed。
- `num_steps_per_rollout=2`，保证第 2 个 optimizer step 的 ratio ≠ 1。
- 与同 seed 默认 GRPO 对照，用该机制的**专有指标**（`dual_clipfrac`、`kl_loss` 键、`opsm_clipfrac` 等）证明生效；没有专有指标时，比较第 2 步的 pg_loss 或 grad_norm 是否不相等。
- 通用判据：loss 与 grad_norm 有限；零梯度不变量没有误报；事件里的 `algorithm_spec_sha256` 对应的规范化 JSON 包含该机制；`miles_commit` 等于 pin。
对照组复用同一次提交的默认 GRPO 运行，每个 pin 只跑一次，所有项共享。通过后，在 `MILES_DECLARED` 中加一条带证据路径的声明，每项单独提交。未通过就停下，找到并修复原因后才允许重跑一次；修不了就如实记为合法否定结论，保持“可表达未开放”。备选方案是用 clipfrac 作为统一生效指标，没有采用，原因是 1b 的疑点尚未排除。

### D3 clipfrac 疑点先在 CPU 上定性
在 miles-next-venv 中，分别在 compile 与 eager 两种模式下，用相同张量调用 `compute_policy_loss`，覆盖 `eps_clip == eps_clip_high` 和 `≠` 两种情形，与手算 clipfrac 比较。
- 能复现且根因在 fork：做最小修复（例如 clipfrac 在 compile 外计算，或者去掉特化条件），修复后 pg_loss 与修复前逐位相同，只有指标变化。
- 不能复现：记录否定结论和 g1e 现象的其他解释。
无论哪种结果，结论出来之前，GSPO/GMPO 以外依赖 pg_clipfrac 的零梯度放宽都维持现状。GSPO/GMPO 读的是 num/den，不受影响。

### D4 fork 改动只加观测，不改数值
`yeto/ports` 上的小提交包括：
- (a) `dual_clipfrac`：A<0 的 token 中 dual 下界 c·A 生效的比例，按 loss mask 统计；
- (b) 超采样每次提交的批大小，以及丢弃、补采的计数，写进 rollout 指标；
- (c) 可能的 clipfrac 修复。
每个提交都要附 CPU 单测，证明默认参数下 loss 逐位不变。之后按原流程走：独立审查 → 快进 `yeto/ports` → IMG 重建镜像并更新 pin → 更新 `FORK_COMMITS` → 对受 pin 限制的 4 项声明做 git diff 迁移论证。pin 变更集中在一次完成，以免多次重建镜像。备选方案是 dual_clip 通过 pg_loss 差异间接证明，没有采用，因为 g1 中恰好出现了逐位相同的结果，间接证据不可靠。

### D5 单项设计
- **KL**：k1、k2、low_var_kl 与 kl_unbiased 各跑一次，放置位置都是 `loss`，coef 取 0.01，并带显式的 `kl.ref_model`。判据是 `kl_loss` 键存在、有限、非零，且与已声明的 k3 运行数值不同。第 2 步的 ratio ≠ 1 不是 KL 的前提，但 ref ≠ π 是前提，所以要求第 1 轮之后 kl_loss > 0。
- **opsm_rollout**：`--use-rollout-logprobs` 会替换 π_old。判据是 `opsm_clipfrac` > 0（沿用 1a trigger 的小阈值），并且 PPO ratio 的来源确实是 rollout logprob（事件或来源记录）。
- **no_rewards_normalization、grpo+whiten**：在 DP=1 下，whiten 是对整批做白化，效果可以在 advantage 统计上观测。判据是 `advantages` 的均值和方差与对照组不同，且与离线重算的结果一致（容差事先固定为 1e-5）。grpo+whiten 声明后，变 DP 边仍按 `reshard.py` 拒绝。`ESTIMATOR_COMPANIONS` 不变，因为 grpo+whiten 是通过 `features:whiten_advantages` 声明开放的，不是由估计器认领。
- **over_sampling**：用 D4(b) 的指标，事先固定判据：至少有一轮提交批大小大于 rollout batch size，并且被过滤的组数 > 0。实验设计要能触发过滤，沿用 g1h 的动态过滤配置并调高过滤强度。
- **clip_higher**：在当前 pin 上用 g1j 的配置独立重跑。判据是 g1j 预登记的第 3 步 grad_norm 不相等，同时要求 D3 结论确定后的 clipfrac 与离线重算一致。通过后，从 `MILES_DECLARED_PINS` 的“迁移”集合中改记为“在该 pin 上实测”。
- **CISPO/SAPO/GMPO**：直接采用 loss-variants 已预登记的修订计划，判据 (a)–(f) 不改。前置条件是先核实 trainer v2 补丁是否合入；没有合入时，GMPO 的判据 (e) 无法成立，GMPO 暂缓，只跑 CISPO 和 SAPO。

### D6 用户代码类机制永不声明
`corrections:custom`、`plugins`、`losses:custom_loss`、`custom_pg_loss_reducer` 中除 vendored Dr.GRPO reducer 以外的任意 reducer，都属于用户自带代码。GPU 证据只能证明某一份代码，不能推广到任意代码，所以这些机制永久不进入 `MILES_DECLARED`。实现方式是在 adapter 中加一条“不可声明”名单，并用单测断言它与 `MILES_DECLARED` 没有交集。放行开关仍只在单岛无同步下生效。yeto 自带的分派器（`custom_reward_postprocess`，已声明）不在这个名单里。

### D7 阶段二：身份扩展的规范化规则
所有新字段都遵循“默认值不输出”，保证只含 R0 字段时哈希不变。取值被显式设置后，即使等于默认值也会进入规范化结果，与 2b 的变体参数一样。
- `sampling.temperature` / `top_p` / `top_k`：从 `UNMAPPED_OBJECTIVE_FLAGS` 移出，吸收 extra argv，与 spec 冲突时报错。eval 的采样参数不受影响，也不进哈希。温度 ≠ 1 时，rollout logprob 与训练 logprob 的口径会有差异，这一组合与 TIS/OPSM 的关系需要记录；默认先拒绝 temperature≠1 与 `use_rollout_logprobs` 的组合，等 GPU 核实后再开放。
- `moe.rollout_routing_replay` / `moe.routing_replay`：旧的 `--use-rollout-routing-replay` agent 参数被吸收进 spec，行为不变，但哈希变化（proposal 标记为 BREAKING）。非 MoE 模型开启时，在启动检查中拒绝。
- `kl.ref_update_interval`：只允许与 `kl.placement=loss` 或 `reward` 且 coef>0 组合。cut 中必须记录 ref 版本（最近一次 ref 更新的 rollout id 与 ref 权重哈希）。在 rl-infra 4.1/4.2 提供 cut 的 ref 字段之前，只在“不产生 cut 的运行”中接受这个字段；launch check 在需要 cut 的 profile 下拒绝。这一点写成接口依赖，本 change 不改 `cut.py`。
- `rollout.loss_mask_policy`：枚举值先只开放 Miles 当前的实际行为，作为默认值且不输出。其他取值以 `docs/research` 与 fork 核实后的可实现项为准，在第 10 组第一项中固定，没有列入的取值会被拒绝。
- RLOO：作为 yeto reward 分派器的一个阶段，`A_i = r_i − mean_{j≠i} r_j`，要求组大小 ≥2，并与 `std_normalization=false` 配套。它有独立的算法身份。与 Dr.GRPO 只差 G/(G−1) 常数缩放这一点写进文档，不据此合并两者的身份。

### D8 文件归属与跨负责人交付
| 文件 | 负责人 | 本 change 的方式 |
|---|---|---|
| `yeto/rl/engine/algorithm.py`、`yeto/rl/algos/*`、`fake.py` | ALGO | 直接修改 |
| `miles_adapter/algorithm_flags.py`、`config.py` 算法部分、`entry.py` 声明 | ALGO | 直接修改 |
| `miles_adapter/trainer.py`、`driver.py`、`cut.py`、`ledger.py` | INFRA | 补丁（`infra-drafts/patches/algo-supp-*.patch`）或接口请求 |
| `yeto/rl/__init__.py` pin、镜像 | IMG | 请求 |
| `cli.py`、`launcher.py`、`learner.py` 中 routing replay 接线 | 主 agent 指定 | 补丁 |
| fork `yeto/ports` | FORK（经审查） | 小提交 |

### D9 成本纪律与 GPU 选择
- 默认卡型为 Modal A10G（24GB，约 $1.10/h）。L4 与 T4 只用于不训练的解析或冒烟。两岛项用两张同型号卡，canonical LoRA 哈希比较只要求两岛之间一致，不需要跨卡型逐位一致，所以不用 H100。
- 每个运行都满足：Modal 函数 `timeout=` 加本地 `timeout` 加独立 watchdog；有唯一前缀；运行前在计划中写明 task、判据、预计时长、费用；结束后先拉证据再 `stop -y`，并核实没有残留。同一失败修复前不重跑。
- 累计费用达到阶段上限的 80% 时停下，向用户报告。任何单项超出自身上限都立即中止，记为未完成。

| 项 | 卡 | 硬超时 | 上限 |
|---|---|---|---|
| 阶段一对照 GRPO（当前 pin 与新 pin 各一次） | A10G | 40 min/次 | $2.0 |
| CISPO / SAPO / GMPO G1 | A10G ×3 次 | 40 min/次 | $4.5 |
| 变体 + TIS 组合冒烟 | A10G ×3 次 | 30 min/次 | $3.0 |
| CISPO 两岛 strict-avg（可选） | A10G 1+1 | 50 min | $3.5 |
| dual_clip | A10G | 40 min | $1.5 |
| KL k1/k2/low_var_kl + kl_unbiased | A10G ×4 次 | 40 min/次 | $4.0 |
| opsm_rollout | A10G | 40 min | $1.5 |
| no_rewards_normalization、grpo+whiten | A10G ×2 次 | 40 min/次 | $2.5 |
| over_sampling 强证据 | A10G | 50 min | $2.0 |
| clip_higher 重跑 | A10G | 40 min | $1.5 |
| clipfrac 修复后的 GPU 复核（仅当 fork 有修复时） | A10G | 40 min | $1.5 |
| 1b G3 两岛（8.4） | A10G 1+1 | 60 min | $4.0 |
| 新 pin 镜像的 T4 parse_args | T4 | 15 min | $0.5 |
| 预备金（只用于已记录原因后的一次重跑） | — | — | $3.0 |
| **阶段一合计** | | | **≤ $35** |
| 采样参数 G1 | A10G ×2 次 | 40 min/次 | $2.5 |
| loss_mask_policy G1 | A10G | 40 min | $1.5 |
| RLOO G1 | A10G | 40 min | $1.5 |
| ref_update_interval G1 | A10G | 50 min | $2.0 |
| MoE routing replay（小 MoE，见 D10） | L40S 或 A100-40GB，1 卡 | 60 min/次 ×2 次 | $12.0 |
| 预备金 | — | — | $5.5 |
| **阶段二合计** | | | **≤ $25** |

### D10 MoE 小模型
候选按优先级排列：
1. OLMoE-1B-7B：总参数 7B，bf16 约 14GB，在 Modal L40S（48GB，约 $2/h）单卡运行；
2. Qwen1.5-MoE-A2.7B：总参数约 14B，在 A100-80GB（约 $2.5/h）单卡运行。
第 11 组第一项先在 CPU 上确认三件事：Miles/Megatron bridge 支持该架构；SGLang 能返回 `routed_experts`；LoRA 能挂上。只有第一个候选不满足时才换第二个。判据是开启 replay 后，训练侧使用了 rollout 路由（fork 指标或日志中有路由一致率），并且与未开启的运行相比，训推 logprob 差异（`train_rollout_kl`）不增大，比较方向事先固定。两次运行（开/关）合计不超过 $12。

## Risks / Trade-offs

- [clipfrac 修复会改变 fork 的指标口径] → 修复前后 pg_loss 逐位相同，旧证据按新口径重新解读并记录在案，不回溯修改旧结论。
- [pin 变更后，受 pin 限制的声明失效] → 所有 fork 提交集中在一次 pin 更新中完成，并做 diff 迁移论证。clip_higher 在新 pin 上实测。
- [trainer v2 补丁未合入，导致 GMPO 不能验证] → GMPO 暂缓，如实记录，其他两个变体照常推进。
- [temperature≠1 与 rollout logprob 的口径问题] → 默认拒绝这一组合，等验证后再开放。
- [routing replay 进入哈希后，已有 MoE 运行的身份变化] → 在文档中给出迁移说明。旧 CLI 仍能用，只是哈希改变。恢复旧 checkpoint 时，按现有“哈希不一致即拒绝”的规则处理，并在报错中提示原因。
- [ref 周期更新与 cut 的接口依赖] → 在 infra 4.1/4.2 完成之前，需要 cut 的 profile 下拒绝该字段。
- [便宜卡与既有证据（H100）的卡型不同] → 生效证据不要求跨卡型逐位一致，对照组与实验组用同一卡型、同一 pin。

## Migration Plan

阶段二的字段默认值不改变哈希，已有 spec 不需要迁移。唯一的身份变化是 routing replay：在 `docs/MILES_RL.md` 写明变化前后的哈希关系，并在 `rl_algorithm_mismatch` 的报错中提示“routing replay 已进入算法身份”。回退方式是还原对应提交；声明项回退只需删除 `MILES_DECLARED` 中的对应条目。

## Open Questions

- fork 的 `dual_clipfrac` 与超采样指标的具体键名，在 fork 审查时确定。键名不影响判据的含义。
