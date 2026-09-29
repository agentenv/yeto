# Design

## Context

动机见 proposal.md 的 Why，证据见 `docs/research/rl-algorithms/research.md` 的 §3、§4.2、§6。Miles 的行号基于 `9e4260d`。本 design 起草时曾误读一个非 pin 的本地 checkout（`/home/michael/miles`，`3c5e19b9`）；以下事实已在 `9e4260d` 上复核（`git show 9e4260d:miles/backends/training_utils/loss_hub/losses.py`），实现前仍要在当时的 `MILES_NEXT_COMMIT` 上再核对一次（任务 1.1）。

- `--loss-type` 没有 CISPO、SAPO、GMPO。`LH/tinker_losses.py:118` 的 cispo 只在 tinker 路径调用，分母、规约和配置来源都与 ports 不同，只能作为参考实现。
- `--loss-type custom_loss` 配 `--custom-loss-function-path` 会替换**整个** `policy_loss_function`（`LH/losses.py:63` 起）。替换后，下列逻辑都要自己重写：
  - TIS、IcePop、OPSM 修正；
  - GSPO 的 CP 全收集；
  - KL loss；
  - entropy；
  - mismatch 指标；
  - CP 切分与 `sum_of_sample_mean` 聚合。

  另外，batch 的键是固定白名单，不包含 `sample.metadata`。
- `compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c)` 在 `LH/math_utils.py:254`，带 `@torch.compile`。`losses.py` 用 `from ...math_utils import compute_policy_loss` 导入它，调用的是模块内绑定的名字（本地核对）。所以，**不 monkeypatch 就无法只替换 `compute_policy_loss`，同时复用 `policy_loss_function` 的其余部分**。
- 在 `9e4260d` 上，`losses.py` 以 `from ...math_utils import (..., compute_policy_loss, ...)` 按名字导入（第 19 行），调用点在第 210 行，因此不修改 fork 就无法只替换 `compute_policy_loss`；该提交的 `policy_loss_function` 中**没有** `policy_objective` 分支。只有非 pin 的较新 checkout 才有 `policy_objective`（`sao_dis`），它不是本 change 的依据，最多说明这个位置将来可能成为 upstream 的扩展点。
- GMPO 需要序列级聚合，结构上和 GSPO 的 `compute_gspo_kl` 一样（CP 下先全收集，再算序列量），不能只在 `compute_policy_loss` 里逐 token 完成。
- P0 已提供：`loss.variant` 字段、映射表、`losses` 能力维度、`expects_gradient()`、PluginRef 源码哈希。

## Goals / Non-Goals

**Goals:**
- 用户选定路线后，把三个变体从"可表达未开放"推进到"声明支持"，并完成 CPU 数值验证和 GPU 冒烟。
- 变体与 TIS、IcePop、KL loss、entropy、CP 能自然组合，数值可以用参考公式核对。

**Non-Goals:**
- 不做效果 A/B，不声称有收益。
- 不做 decoupled 下的对比实验（依赖 `fix-decoupled-lr-schedule`）。
- 不改 legacy 路径，不改默认 GRPO。
- 不向 radixark/miles、sgl-project/sglang 提 PR。

## Decisions

### D1. 实现路线：决策门（由用户决定）

| 维度 | 路线 A：yeto custom loss 插件 | 路线 B：fork `yeto/ports` 加 variant 分支 |
|---|---|---|
| 代码量 | 大：复制约 250 行 `policy_loss_function`（CP、修正、KL、entropy、指标、聚合），再加三个变体约 80 行 | 小：`math_utils.py` 加约 40–60 行（CISPO、SAPO 逐 token；GMPO 序列级函数，仿照 `compute_gspo_kl`），`losses.py` 加分支约 20 行，`arguments.py` 加 `--policy-loss-variant` 与变体参数约 15 行，另加 fork 测试 |
| 组合性 | 只能靠副本：TIS、IcePop、OPSM、KL、entropy、CP 的组合全部由副本重现，任何遗漏都会悄悄改变语义 | 自然：变体只替换 pg_loss 的计算，后续修正、KL、entropy、聚合仍由原代码执行 |
| 升级漂移 | 高：Miles 升级改动 `policy_loss_function` 时副本不会跟着变，需要 diff 守护测试 | 低到中：每次 rebase `yeto/ports` 时，冲突集中在一个分支点 |
| 审计性 | PluginRef 源码哈希进入算法哈希，身份清晰；但审计者要读一份和上游"几乎一样"的副本 | fork 提交号进入来源记录；diff 很小，易审；需要更新 `MILES_NEXT_COMMIT` pin 与镜像 |
| 与"不重复实现引擎"原则 | **违反**：重复实现了 policy loss 的主体 | 符合：计算留在引擎，yeto 只做描述、翻译、校验 |
| 前置条件 | 无 | 用户同意在 `michaellchung/miles` `yeto/ports` 提交；重建镜像 |

路线 A 的一个子方案是"包装 `policy_loss_function`，只替换 `compute_policy_loss`"。这个子方案不可行：`losses.py` 调用的是模块内绑定的名字，想只换这一个函数，只能在运行时替换模块属性，也就是 monkeypatch。**明确拒绝 monkeypatch**，理由有三：它不进入任何哈希，不可审计；和 `torch.compile` 缓存的交互不确定；Miles 升级后可能静默失效。因此路线 A 只能是"完整副本"形态。

**推荐路线 B。** 理由：组合语义由引擎原代码保证；改动小，易审；符合"不重复实现引擎"原则。代价是要改 fork、更新 pin 和镜像。**最终由用户决定**：选定之前，tasks 第 1 组之后的实现组都不开始。

### D2. spec 字段与翻译

在 P0 的 `loss` 组内增加以下字段：
- `variant ∈ {policy_loss, cispo, sapo, gmpo}`；
- `sapo_tau_pos`（默认 1.0）、`sapo_tau_neg`（默认 1.05）；
- `gmpo_log_clip_low`、`gmpo_log_clip_high`（默认均为 0.4）。

CISPO 复用 `eps_clip`/`eps_clip_high` 作为 ε_l/ε_h。变体参数只在 variant 与之匹配时才进入规范化。variant 不匹配时如果设置了这些参数，按"无效字段"拒绝，以免同一语义出现两种哈希。

翻译规则分路线：
- 路线 B：翻译为 `--policy-loss-variant` 及对应参数（实际参数名以 fork 提交为准），映射表同时登记吸收规则。
- 路线 A：翻译为 `--loss-type custom_loss --custom-loss-function-path yeto.rl.engine.miles_adapter.loss_variants.<fn>`。变体参数通过 yeto 自有的环境变量或参数传入插件；插件以 PluginRef 源码哈希进入身份。

### D3. 数值定义与组合顺序

逐 token 定义（ρ=exp(log π_θ − log π_old)，Â 为优势）：
- CISPO：loss = −sg(clamp(ρ, 1−ε_l, 1+ε_h))·Â·log π_θ；clipfrac 统计越界 token。
- SAPO：loss = −f_τ(ρ)·Â，其中 f_τ(ρ)=σ(τ(ρ−1))·4/τ，τ 按 sign(Â) 选择；Â=0 时损失为 0。
- GMPO：先算 ℓ_t=clamp(sign(Â)·log ρ_t, −δ_l, δ_h)·sign(Â)，序列 ratio 为 exp(mean_t ℓ_t)，loss = −ratio·Â。CP 下 mean 在全序列上计算，做法与 GSPO 相同。

组合顺序与 Miles 现有逻辑一致：先算出变体的 pg_loss，然后依次乘 OPSM mask、TIS/IcePop 权重，再做聚合；KL loss 和 entropy 最后相加。CPU 测试的参考实现独立用 torch 小张量写出，不 import 被测代码。

### D4. 拒绝规则

以下组合启动前拒绝：
- variant≠policy_loss 与 `advantage.estimator=gspo` 组合（两者都在定义序列级 ratio）；
- variant≠policy_loss 与 `eps_clip_c`（dual-clip）组合（dual-clip 是为 PPO clip 目标设计的，与这三个变体的语义不一致）；
- SAPO 的 τ ≤ 0 或为非有限数；
- GMPO 的 δ ≤ 0 或为非有限数。

TIS、IcePop 与变体组合是允许的，但它们要等 `rl-algo-mismatch-correction` 开放之后才能在 GPU 上使用。

### D5. expects_gradient

- CISPO、SAPO：存在有效 token 且 Â 不全为零时，期望非零梯度。组内 reward 方差判定保持 GRPO 语义。
- GMPO：在上面的判定之外再加一条：如果引擎报告的序列级 clip 比例为 1（所有序列都在 clip 外），允许零梯度。读不到 clip 比例时，退回 GRPO 判定（保守做法：可能误报失败，但不会漏报）。
- 任何变体下，grad_norm 非有限都判为失败。

### D6. 外层同步

三个变体只改 loss，与 strict-avg 和 decoupled 正交。ports 串行执行，陈旧度为 0，π_old 就是本轮起点的权重。本地多个 mini-batch 导致的 ratio 偏离，由各变体自身的 clip 或门控处理，`execution` 不需要新的要求。两岛 strict-avg 下，同一 spec 的算法哈希必须一致（P0 已有校验），GPU 上用 CISPO 验证这一点。

## Risks / Trade-offs

- [路线 A 的副本漂移] → 加一个守护测试：比较 pin 提交上 `policy_loss_function` 的源码哈希，Miles 升级时必须人工 diff。
- [路线 B 被拒或延迟] → 变体保持"可表达未开放"，不影响其他 change。
- [`torch.compile` 与新分支交互] → 路线 B 让新函数单独 compile，并在 fork 测试中覆盖动态形状。
- [GMPO 在 CP 下的序列量错误] → CPU 测试模拟切分并比对；GPU 冒烟只在 CP=1 下做，CP>1 在文档中标为"CPU 测试通过，未经 GPU 验证"。
- [论文公式的歧义，例如 CISPO 的 ε_l 默认值、SAPO 是否按 token 归一] → 以论文正文为准，在测试文档中注明出处与取舍；不确定的项在文档中列明。
- [mock 测试被误当作验收] → progress.md 严格区分"已实现""CPU 测试通过""GPU 验证通过"。

## Migration Plan

- 默认 variant 为 `policy_loss`，现有配置的哈希和 argv 不变。
- 路线 B：fork 提交后更新 `MILES_NEXT_COMMIT` pin 与镜像。回滚方式是恢复 pin；未选用变体的运行不受影响。`rl-infra-spec` 的 fork-M1–M6（草稿分支 `yeto-elastic-m1-m6`）也落在同一 `yeto/ports`：两者按 alignment.md A7 串行合回 `yeto/ports`，每次合回只由 pin/镜像负责人（Agent IMG）更新一次 pin 与 digest，任一方 rebase 后须重跑对方的 fork CPU 测试。
- 路线 A：插件是新文件，回滚方式是撤销 `losses` 声明。

## Open Questions

- CISPO 论文中 ε_l 是否实际使用（原文主要讨论 ε_h）：只影响默认值和文档，不影响结构。
- GMPO 默认的 δ 是否需要区分正负优势：字段已经分开，取值可以之后再定。
