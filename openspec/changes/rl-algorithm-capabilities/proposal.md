# Proposal

## Why

R0（`rl-engine-ports`）把 yeto 与 Miles 之间的算法边界收敛成了 `AlgorithmSpec`，但目前 ports 路径只接受 GRPO。已有的算法描述又不完整，有三个问题破坏了"一切可证明"：

- **参数绕过哈希**：通过 extra argv 可以透传 `--eps-clip*`、`--use-tis`、`--loss-type` 等参数改变训练目标，算法哈希却不变。
- **`kl_coef` 在 GRPO 下不起作用**：Miles `9e4260d` 的 grpo/gspo 会丢弃 reward 中的 KL，但仍然加载 ref 模型。
- **算法哈希没有进入导出的来源记录，也不参与外层成员校验**：两个用不同算法的岛可以被平均。

与此同时，Miles 基线已经原生实现了多数无 critic 的 RL 机制，包括 GSPO、clip-higher、dual-clip、token 级聚合、Dr.GRPO、TIS/IcePop/OPSM 和 KL loss。

因此需要先建立一个多算法框架，把这些机制以可声明、可校验、可证明的方式接入。后续的具体算法 change（训推不一致修正、GRPO 家族参数、序列级 loss 与 advantage 变体）都建立在这个框架上。调研依据见 `docs/research/rl-algorithms/research.md`。

## What Changes

- **AlgorithmSpec v2**：把算法描述扩展为结构化字段，按以下几组组织：
  - `advantage`：estimator、组内 std 归一、reward 后处理插件
  - `loss`：clip 下界、clip 上界、dual-clip 系数、聚合方式、loss 变体
  - `kl`：`placement: none|reward|loss`、系数、估计器
  - `correction`：训推不一致修正方式及阈值
  - `sampling`：动态过滤、超采样、overlong 处理
  - `entropy_coef`
  - `plugins`：dotted path 加源码 SHA256
  - `execution`：`needs_critic`、`max_policy_staleness`、`group_required`、`needs_rollout_logprobs`

  本 change 只建立字段、规范化、校验和翻译框架。各机制在 ports 上"声明为支持"由后续算法 change 负责逐项开启。
- **哈希兼容**：只使用 v1 字段时，规范化结果保持 v1 格式，现有算法哈希逐字节不变。用到新字段时才写 v2 schema。
- **参数吸收**：yeto 维护"Miles 算法参数 → spec 字段"的映射表。
  - extra argv 中出现的已知算法参数会被吸收进 spec，进入哈希和能力检查。
  - 同一字段在 spec 与 argv 中取值不同时，启动失败并报告冲突。
  - 影响训练目标、但未纳入映射的 Miles 参数会被拒绝。
  - 所有算法参数改由 adapter 独占，最终 argv 只由 spec 翻译生成。
- **能力声明与执行要求匹配**：`EngineCapabilities` 按机制维度声明支持项，并新增执行能力：是否支持 critic、可提供的最大策略陈旧度、是否提供 rollout logprob。启动前拿算法的 `execution` 要求与执行能力比对，不满足就在创建 GPU 进程前拒绝，并列出缺失项。当前 ports 声明不支持 critic、陈旧度为 0。
- **未验证机制的受控放行**：新增 `--rl-allow-unverified-mechanism NAME`，只在单岛运行中生效，用于各算法 change 在"声明支持"之前完成 GPU 冒烟。放行信息写入事件和来源记录，与多岛组合时拒绝。
- **KL 显式建模**：
  - grpo/gspo 配 `placement=loss`，翻译为 `--use-kl-loss`，此时生效。
  - ppo/rpp 类配 `placement=reward`，翻译为 `--kl-coef`。
  - grpo/gspo 配 `placement=reward` 属于无效组合，拒绝，并提示改用 `placement=loss`。
  - 旧 v1 `kl_coef` 的解析规则见 design。
- **拒绝矩阵**：以下组合在启动前拒绝，报错里给出替代方案：
  - critic 类算法（提示改用 legacy）；
  - 互斥参数，例如 TIS 与 `use_rollout_logprobs`、`kl_coef` 与 `kl_loss_coef`；
  - 缺少必需参数，例如 GSPO 未显式给出 clip；
  - 奖励类型不匹配的机制。
- **按算法判定零梯度不变量**：由算法声明"本轮是否期望非零梯度"。GRPO 的现有判定（组内 reward 方差）保持不变。带 mask 的机制按其声明判定，不会被误判为失败。
- **来源记录**：
  - 算法规范化 JSON 与哈希写入 `yeto_rl_provenance.json`（仅 ports）。
  - 各岛在进入外层同步前校验算法哈希一致，不一致就拒绝。
- **不变的部分**：默认 GRPO 行为及其 argv 不变；legacy 路径（`build_miles_argv` 与旧 fork）不变；不修改 Miles 或 SGLang fork；本 change 不新增任何 GPU 上"声明为支持"的算法。

## 算法能力一览

本 change 只建立框架，**不在 GPU 上开启任何新算法**。表中各列的含义：

- "本 change 提供"：为该机制提供的 spec 字段、翻译规则和拒绝规则；
- "开启于"：哪个后续 change 会把它声明为 ports 支持并完成验证；
- "Miles 实现"：计算由 Miles `9e4260d` 的哪项能力提供，yeto 不重复实现。

状态标记：

- ✅：已支持并在 GPU 上验证（R0）；
- ⚙：可表达，未开放。本 change 完成后，spec 能表达、翻译、校验该机制，但 adapter 不声明支持，选用时会在启动前被拒；直到对应 change 完成 GPU 验证并声明支持，才能使用。
- ⛔：启动前拒绝；
- —：不纳入。

| 机制 / 算法 | Miles 实现 | 本 change 提供 | 开启于 | 状态 |
|---|---|---|---|---|
| GRPO（默认）+ 有界非零方差过滤 | `--advantage-estimator grpo` | 保持 v1 哈希和 argv 逐字节不变 | R0 | ✅ |
| 训推不一致：只观测指标 | `--get-mismatch-metrics` + custom-tis | `correction` 字段、翻译 | `rl-algo-mismatch-correction`（P1） | ⚙ |
| TIS | `--use-tis`、`--tis-clip*` | 同上，外加"与 rollout-logprob 互斥"的拒绝 | 同上（P1） | ⚙ |
| IcePop | 内置 `icepop_function` | 同上 | 同上（P1） | ⚙ |
| OPSM | `--use-opsm`、`--opsm-delta` | 同上 | 同上（P1） | ⚙ |
| MIS / geo-MIS | examples 中的 `mis.py`（能否 import 待确认） | 同上 | 同上（P1） | ⚙ |
| clip-higher（DAPO） | `--eps-clip-high` | `loss` 字段 | `rl-algo-grpo-knobs`（P1） | ⚙ |
| dual-clip | `--eps-clip-c` | 同上 | 同上（P1） | ⚙ |
| token 级聚合（DAPO） | `--calculate-per-token-loss` | 同上 | 同上（P1） | ⚙ |
| Dr.GRPO（≈ RLOO） | `--disable-grpo-std-normalization` + reducer | `advantage`/`loss` 字段、插件哈希 | 同上（P1） | ⚙ |
| KL loss（k1/k2/k3） | `--use-kl-loss`、`--kl-loss-*` | `kl.placement=loss` | 同上（P1） | ⚙ |
| entropy 奖励 | `--entropy-coef` | 字段 | 同上（P1） | ⚙ |
| 超采样 / overlong 塑形与过滤 | 超采样参数；overlong 由 yeto 插件实现 | `sampling` 字段、插件哈希 | 同上（P1） | ⚙ |
| GSPO | `--advantage-estimator gspo` | "必须显式给 clip"的拒绝规则 | `rl-algo-seq-and-adv`（P2） | ⚙ |
| REINFORCE++ / baseline | 对应 estimator、`--normalize-advantages` | `kl.placement=reward`、岛内统计语义 | 同上（P2） | ⚙ |
| MaxRL / MAPO / GDPO | yeto 的 reward 后处理插件 | 插件哈希、"二值奖励"拒绝规则 | 同上（P2） | ⚙ |
| CISPO / SAPO-Qwen / GMPO | 无；需 custom loss 或改 fork | `loss.variant` 字段 | `rl-algo-loss-variants`（P2，另需决策） | ⚙ |
| `kl_coef>0` + grpo/gspo | Miles 会丢弃该 KL | 拒绝，并提示改用 `placement=loss` | 本 change | ⛔ |
| PPO / VAPO / SAO / CompactionRL | Miles 有 critic | `needs_critic` 要求不满足则拒绝，提示改用 legacy | 暂缓：需要 critic 状态和外层归属 | ⛔ |
| 异步目标（staleness>0） | TIS ≈ 截断版解耦 PPO | `max_policy_staleness` 与执行能力的匹配 | 暂缓：需另立独立算法契约 change；rl-infra-spec 2.3 不擅自开放 one-step-off-policy，其 partitioned-overlap 只在已认证契约内重叠（alignment.md A6） | ⛔ |
| OTB / GiGPO / ARPO / SAPO-Gensyn | 无 | — | 暂缓：OTB 要改 fork，GiGPO 依赖 agent，ARPO 要改 SGLang，SAPO-Gensyn 是外层协议 | — |
| Training-free GRPO | — | — | 不做：不更新权重 | — |

本 change 没有 GPU 验收项，表中 ⚙ 项的 GPU 验证都放在对应的后续 change 里。

## 实施索引

本 change 是框架，只负责表达、翻译、校验和证明，不开放任何新机制。表中每项 RL 能力都由一个独立的 change 开放，并在该 change 内完成验证。给 agent 派活时，按"change 目录"找到对应的 `tasks.md` 执行，按"spec"核对验收要求。

路径都相对于 `openspec/changes/`。

| 顺序 | change 目录 | spec（capability） | 开放的 RL 能力 | 依赖 | GPU |
|---|---|---|---|---|---|
| 0 | `rl-algorithm-capabilities/`（本 change） | `specs/rl-algorithm-spec/spec.md` | 无（框架：AlgorithmSpec v2、参数吸收、能力与执行要求匹配、KL 放置规则、拒绝矩阵、按算法判定的梯度不变量、来源记录与岛间一致性） | R0 当前代码 | 不需要 |
| 1a | `rl-algo-mismatch-correction/` | `specs/rl-mismatch-correction/spec.md` | 训推不一致只观测指标、TIS、IcePop、OPSM、MIS/geo-MIS | 0 | 1 卡冒烟与量化；两岛 strict-avg，每岛 1 卡 |
| 1b | `rl-algo-grpo-knobs/` | `specs/rl-grpo-variants/spec.md` | clip-higher、dual-clip、token 级聚合、Dr.GRPO（≈RLOO）、KL loss、entropy、超采样、overlong 塑形与过滤；yeto 统一的 reward 后处理分派器 | 0 | 1 卡冒烟；两岛 strict-avg，每岛 1 卡 |
| 2a | `rl-algo-seq-and-adv/` | `specs/rl-advantage-and-sequence-variants/spec.md` | GSPO、REINFORCE++ / baseline、MaxRL、MAPO、GDPO | 0；1b 的 reward 分派器 | 1 卡冒烟；两岛 strict-avg，每岛 1 卡 |
| 2b | `rl-algo-loss-variants/` | `specs/rl-loss-variants/spec.md` | CISPO、SAPO-Qwen、GMPO | 0；**用户决定**走 custom loss 还是改 fork | 1 卡冒烟；两岛 strict-avg，每岛 1 卡 |
| — | 暂未立项 | — | critic 家族（PPO/VAPO/SAO/CompactionRL）、异步目标（staleness>0）、全参数训练上的算法、OTB、GiGPO、ARPO、SAPO-Gensyn | 见"算法能力一览" | — |

派活规则：
- 1a 与 1b 可以并行；2a 必须在 1b 的分派器完成后开始；2b 必须在用户做出路线决策后开始。
- 各 change 的 GPU 任务组都要先经用户批准卡数和预算。decoupled 下的对比实验要等 `fix-decoupled-lr-schedule` 合入之后才能做。
- 各 change 的 G1（1 卡冒烟）统一使用 P0 提供的 `--rl-allow-unverified-mechanism` 放行，G1 通过后才在 adapter 中正式声明；G3（两岛）只用正式声明。
- 每个机制要满足两个条件才算"可用"：所在 change 的 GPU 冒烟通过，并且 adapter 已声明支持。在那之前，状态都是"可表达，未开放"。

## Capabilities

### New Capabilities

- `rl-algorithm-spec`：yeto 的多算法描述契约，包括：
  - 结构化字段与规范化哈希，以及 v1 哈希兼容；
  - 参数吸收与冲突检测；
  - KL 放置建模；
  - 算法执行要求（critic、陈旧度、分组、rollout logprob）与执行能力的匹配；
  - 拒绝矩阵；
  - 按算法判定的梯度不变量；
  - 算法身份写入来源记录，并在岛间做一致性校验；
  - 未验证机制的受控放行（仅单岛）。

### Modified Capabilities

无。R0 的 `rl-engine-ports`、`rl-engine-selection` 尚未归档，也不在主 specs 中，因此本 change 的要求全部以新 capability `rl-algorithm-spec` 表达。R0 归档后另做一次合并：把"能力声明与握手""算法描述复用引擎算法""不支持的组合明确拒绝"中与本 change 重叠的内容改写为 delta。design D10 记录了需要合并的条目。

## Impact

- **代码（仅 ports 路径）**：
  - `yeto/rl/engine/algorithm.py`：v2 结构、规范化、吸收映射；
  - `yeto/rl/engine/capabilities.py`：新维度与执行能力；
  - `yeto/rl/engine/miles_adapter/config.py`：翻译、`ADAPTER_OWNED_FLAGS`、extra argv 吸收；
  - `yeto/rl/engine/miles_adapter/entry.py`：能力声明；
  - `yeto/rl/engine/driver.py`：梯度不变量；
  - `yeto/rl/engine/fake.py`；
  - `yeto/rl/export.py`：ports 分支的来源记录；
  - `yeto/rl/learner.py` 的 ports 分支：读取 `--rl-algorithm-spec`，并核对预期哈希；
  - `yeto/launcher.py`：计算预期算法哈希并下发给各岛（仅 ports）。
- **不受影响**：
  - `yeto/rl/learner.py::build_miles_argv` 与 legacy 行为；
  - Miles fork（`michaellchung/miles` `yeto/ports`）与 SGLang fork；
  - syncer 的外层平均算法。
- **测试**：新增 `AlgorithmSpec` v2 的 golden 哈希、吸收与冲突、拒绝矩阵、翻译、来源记录和岛间一致性单测。`tests/test_rl_argv_snapshot.py` 保证默认 GRPO 的 argv 逐字节不变。在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv。全部为 CPU 测试；本 change 不需要 GPU。
- **依赖**：以 `rl-engine-ports` 当前提交为基底，R0 合入 main 后 rebase。不依赖 `fix-decoupled-lr-schedule`，但后续算法 change 中 decoupled 下的对比实验依赖它。
- **兼容性**：
  - 现有 ports 配置如果通过 extra argv 传了已知算法参数，将改为被吸收进 spec，因此其算法哈希会变化，而这正是本 change 要修正的问题；
  - 传了未映射的算法参数将被拒绝；
  - 以 `kl_coef>0` 配 GRPO 的 ports 配置将被拒绝。
  - 以上三点对 ports 用户可见。ports 当前不是默认路径。
