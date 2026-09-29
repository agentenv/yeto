# Design

## Context

动机见 proposal.md 的 Why，证据见 `docs/research/rl-algorithms/research.md` §3、§4、§6、§7。本 design 依赖的现状：

- P0 提供 AlgorithmSpec v2 分组字段、映射表 `miles_adapter/algorithm_flags.py`、机制维度的 `EngineCapabilities`、`expects_gradient()`、`TrainStepMetrics.masked_fraction`、PluginRef 源码哈希、KL 放置规则（rpp 类允许 reward-KL）和拒绝矩阵（序列级 ratio 未显式给 clip、要求二值奖励的机制配非二值奖励）。
- P1-b 提供 yeto reward 后处理分派器 `yeto.rl.algos.reward_pipeline`：唯一占用 `--custom-reward-post-process-path`，先 reward 塑形、再 advantage 变换，默认变换等价重现 Miles 内置的按 prompt 分组、按 rollout_key 合并多段、段间奖励一致检查、grpo/gspo std 归一，并预留变换注册点。本 change 只往注册点里加变换，不改分派器的外部行为。
- Miles `9e4260d` [源码]：
  - `gspo` 与 `grpo` 共用 advantage 分支（`LH/advantages.py:53`），区别在 loss：`losses.py` 的 gspo 分支收集完整 log prob，`math_utils.py:224-251` 计算序列级 KL/ratio；clip 仍由 `--eps-clip`/`--eps-clip-high` 给出，默认 0.2。
  - `reinforce_plus_plus`（`advantages.py:79`）：`get_reinforce_plus_plus_returns`，token 级折扣回报，读 `args.gamma` 与 `args.kl_coef`；`--gamma` 定义在 arguments.py 的 PPO/GAE 参数组（默认 1.0，fork 上约 A:1680），与 PPO 共用。
  - `reinforce_plus_plus_baseline`（`advantages.py:92`）：rollout 侧 `_post_process_rewards` 对 grpo/gspo/rpp_baseline 做组内减均值（仅 grpo/gspo 再除 std）；训练侧再加 reward-KL。
  - `--normalize-advantages`：`normalize_advantages` 在 DP 组内 all-reduce 做 token 级 masked 白化。ports 中一个 Miles job 即一个岛，DP 组 ⊆ 岛。
  - `Sample.metadata: dict`（`miles/utils/types.py`），custom reward post-process 能看到完整 `Sample`。
  - 内置归一化的 std 为 `torch.Tensor.std()`（无偏），加 1e-6，std 为 0 时只减均值，组大小为 1 时不除 std。
- R0 零梯度判定在 `driver.py:355-370`：`any(g.reward_std > 0)`。

## Goals / Non-Goals

**Goals:**
- 六个机制（GSPO、rpp、rpp_baseline、MaxRL、MAPO、GDPO）各自有 CPU 数值证明、零梯度判定和 1 卡 G1，然后才声明支持。
- 至少一个 advantage 变换做两岛 strict-avg G3。

**Non-Goals:**
- 不做效果 A/B（G4），不声称收益。
- 不做跨岛统计量聚合。
- 不开放 `gamma≠1.0`、GSPO 与 advantage 变换的组合、PPO/critic。
- 不在 decoupled 下做任何对比实验（等 `fix-decoupled-lr-schedule`）。

## Decisions

### D1. GSPO：只开放估计方式，clip 值不设 yeto 默认

GSPO 翻译为 `--advantage-estimator gspo` 加显式 `--eps-clip`/`--eps-clip-high`。P0 已拒绝未给 clip 的情况；本 change 在报错中补充"引擎默认 0.2、论文 3e-4/4e-4"。yeto 不提供默认值，因为 LoRA 小模型上哪一档合适没有证据，写死默认值等于替用户做了没有依据的选择。

GSPO 与 MaxRL/MAPO/GDPO 组合在本 change 中不开放：分派器对 gspo 走内置 GRPO 归一路径，组合语义可以表达，但没有验证，按"可表达未开放"拒绝。

**一个 optimizer step 时 clip 不起作用**：Miles 用本轮权重重算 π_old，第一个 mini-batch 上 ratio≈1（仅有数值差）。若 `optimizer_steps=1`，GSPO 实质是"序列级 ratio 形式的 GRPO"，clipfrac≈0。G1 用 `optimizer_steps≥2`，并同时报告两种配置下的 clipfrac，才算覆盖了 clip 路径。

备选：给 GSPO 设 yeto 默认 clip（如论文值）。不采用，理由同上。

### D2. GSPO 零梯度判定用 `pg_clipfrac`

Miles 的 `pg_clipfrac` 统计取了 clip 分支（该 token 梯度为 0）的 token 比例。GSPO 序列级 clip 下同一序列所有 token 同时被裁，所以 clipfrac=1 等价于所有序列被裁、梯度为 0，这是合法的。判定：

```
expects_gradient = any(reward_std > 0) and not (masked_fraction is not None and masked_fraction >= 1 - 1e-9)
```

`masked_fraction` 由 adapter 从 Miles 训练指标 `pg_clipfrac` 读取（P0 D6 的字段，本 change 为 gspo 填入具体来源）。多个 mini-batch 时取整轮 token 加权值；只要有一个 mini-batch 未全裁，梯度就应非零，所以只有整轮全裁才不期望梯度。读不到时按期望梯度处理（R0 行为，严格侧）。

备选：看到 GSPO 就关闭零梯度不变量。会漏掉真实的梯度断链（R0 就是为此加的不变量），不采用。

### D3. REINFORCE++：白化只在岛内，复用 Miles `--normalize-advantages`

- `advantage.whiten=true` 翻译为 `--normalize-advantages`。Miles 在 DP 组内 all-reduce，DP 组是本岛的训练进程，**天然是岛内语义**，不需要改 fork。spec 写明"不跨岛"，并禁止任何实现把统计量送进外层协议。理由：跨岛 all-reduce 会让算法依赖外层协议和各后端的同步点，破坏"一个协议多个后端"，也会让 decoupled 下出现跨岛阻塞。
- 本 change 只声明 `whiten=true` 的 rpp/rpp_baseline；`whiten=false` 与论文不符，可表达未开放。
- 分派器对 rpp 走恒等（Miles 内置对 rpp 不做 reward 归一），对 rpp_baseline 走组内减均值、不除 std，两者都用 CPU 测试与 Miles 原函数逐元素比对。
- `kl.placement=reward` 翻译为 `--kl-coef`（P0 已允许 rpp 类）。会加载 ref 模型，ref 身份沿用 P1-b 的做法进入哈希。

### D4. `--gamma` 纳入映射表，只开放 1.0

`--gamma` 直接改变 rpp 回报，属于"影响训练目标"的参数；不纳入映射就只能被 P0 拒绝或绕过哈希。决定：
- 映射到 `advantage.gamma`，默认 1.0，默认值不输出（argv 快照不变）；
- 只允许与 `reinforce_plus_plus` 组合，其他估计方式（含 rpp_baseline，Miles 该分支不读 gamma）给非默认值时拒绝；
- 能力声明只含 1.0，非 1.0 可表达未开放。`--lambd` 只影响 GAE（critic 路径，ports 拒绝），不映射，保持在"影响训练目标但未映射"清单中，出现即拒绝。

备选：不映射、一律拒绝。那样 rpp 永远只能 γ=1，且未来开放要再改映射表；映射成本很低，所以现在纳入。

### D5. advantage 变换在分派器中实现，以 rollout 为单位

`advantage.transform ∈ {builtin, maxrl, mapo, gdpo}`（名称若与 P1-b 的注册点命名不同，以 P1-b 为准，本 change 只加枚举值）。三个变换共享 P1-b 分派器的分组与 rollout_key 合并代码：先得到每组的"rollout 条目"列表（段间奖励一致已检查），在条目上算 advantage，再广播回各段。变换结果作为 per-sample 标量返回给 Miles，由 grpo 估计方式广播到 token（`advantages.py:53`）。只允许 `estimator=grpo`；与其他估计方式组合启动前拒绝。

标准差统一采用内置 GRPO 的定义（无偏 std，除以 std+1e-6，std=0 或 G=1 时只减均值），这样：
- MAPO 在 λ=0（p=0.5）时逐元素等于 Miles 内置 GRPO，可直接用 Miles 原函数作对照；
- GDPO 的分量归一与 Miles 一致。

各变换的边界约定（全部为显式分支，不靠 ε 防除零）：

| 情况 | MaxRL | MAPO | GDPO（分量） |
|---|---|---|---|
| G=1 | 0 | 0 | 0 |
| 全错 μ=0 | 0 | 两项均 0 | 仅减均值 → 0 |
| 全对 μ=1 | r−μ=0 → 0 | λ=1，第二项 0 | 同上 |
| std=0（非二值场景） | 不适用（要求二值） | 第一项 0 | 仅减均值 |

MaxRL 的 r̂ 使用组均值（含自身），与 research §3 的公式一致。论文若使用留一估计则属 [未核实]，列入 Open Questions，不影响接口。

### D6. 奖励向量约定

reward 函数在 `sample.metadata["yeto_reward_components"]` 写入 `{name: float}`。算法描述中 `advantage.gdpo = {components: [{name, weight}], whiten: true}`，规范化时按 name 排序，进入哈希。

- 启动前：分量列表为空、名字重复、权重非有限，拒绝。
- 运行时（分派器内）：缺分量、多分量、值非有限、同一 rollout 各段向量不同，本轮失败（抛错，由 driver 写入事件）；不填默认值，因为静默补 0 会改变目标却不改变哈希。
- 标量 `sample.reward` 仍由 reward 函数给出，只用于指标和日志；GDPO advantage 只看分量。
- 分量是否二值不做要求。

GDPO 的最终白化在分派器里对本轮本岛全部 rollout 条目做（样本级、rollout 为单位），而不是用 `--normalize-advantages`（token 级、按 token 加权）。理由：论文是样本级白化；分派器位置便于 CPU 精确对照。二者都是岛内。分派器看到的是本岛动态过滤之后的整轮 rollout batch。

备选：让 reward 函数返回 tuple。会改变 Miles reward 接口，需改 fork，不采用。

### D7. 二值奖励的声明与运行时检查

P0 已有"奖励类型声明"和启动前拒绝。本 change 在分派器中加运行时检查：MaxRL/MAPO 下出现 {0,1} 以外的奖励（塑形之后的值）时本轮失败。注意：overlong 软惩罚（P1-b）会把奖励变成非二值，所以 MaxRL/MAPO 与 overlong 软惩罚组合在启动前拒绝。

### D8. 零梯度判定

| 机制 | `expects_gradient` |
|---|---|
| GSPO | D2 |
| rpp_baseline、MaxRL、MAPO | `any(reward_std > 0)`，与 R0 相同 |
| GDPO | 分派器输出中存在 |A|>0 的条目 |
| rpp | 本轮 advantage 在白化前不全相等 |

MaxRL 全错组合法地没有梯度：R0 判定用组内 reward_std>0，全错组 std=0，天然不计入，所以不需要为 MaxRL 特判。全对组同理。需要测试证明的是"等价性"：在 MaxRL/MAPO 下，`any(reward_std>0)` 与"变换输出存在非零条目"对所有二值输入等价（std>0 的二值组必有 0<μ<1，MaxRL 输出非零）。

GDPO 需要分派器把"非零条目数"交给 driver。做法：分派器把统计写入 rollout 侧已有的指标通道（P1-b 的分派器指标），driver 的 `batch_summary` 读取；读不到时按期望梯度处理。rpp 的判定同样依赖分派器或 rollout 侧统计，读不到时按期望梯度处理。

### D9. 冒烟阶段如何运行未声明机制

直接复用 P0 的 `--rl-allow-unverified-mechanism`（`rl-algorithm-capabilities` design D11）。它只在单岛运行中生效，写入事件和来源记录，与多岛组合时拒绝。本 change 不另建机制。G1 通过后在 `entry.py` 中声明支持；G3 用正式声明运行，不带放行参数。

### D10. 外层同步兼容性

- **组统计**：组在单岛内生成，组内统计全部由本岛完成；strict-avg 与 decoupled 下假设都成立 [源码 §7.2]。
- **岛内白化**（rpp 的 `--normalize-advantages`、GDPO 的 batch 白化）：各岛 advantage 被各自岛的统计量归一，尺度可能不同；外层对 delta 做等权平均，某岛 advantage 尺度偏大时，其更新方向在平均中占比偏大 [推断，无实验]。白化会把各岛 advantage 拉到单位方差附近，所以这种差异可能反而比不白化更小 [推断]。本 change 不处理，只在文档中写明；是否需要按样本或 token 加权取决于 research §10 问题 4（syncer 加权规则 [未核实]）。
- **GSPO**：ratio 只与本岛 π_old/π_θ 有关，串行下无陈旧度，与同步模式无关。
- **decoupled**：接口和假设同 strict-avg，但 LR 衰减到 0 的问题未修前，decoupled 下的比较没有意义，本 change 不做。

### D11. 验证层级

| 层 | 内容 |
|---|---|
| C2 | gspo/rpp/rpp_baseline/whiten/gamma/kl-reward 翻译；upstream `parse_args` 解析 |
| C3 | 未给 clip、gamma 用错估计方式、变换与估计方式不匹配、二值要求、GDPO 声明非法、放行参数与多岛组合 |
| C4 | 变换数值逐元素对照论文公式；rpp/rpp_baseline/gspo 路径与 Miles 原函数对照；多段语义 |
| G1 | 每机制 1 卡 2–3 轮 |
| G3 | MaxRL 两岛 strict-avg |

## Risks / Trade-offs

- [GSPO 窄 clip 下几乎全裁，训练停滞] → 报告 clipfrac；判定允许全裁而不失败，但 G1 报告中列出每轮 clipfrac，冒烟"通过"只表示链路可用。
- [GSPO 在 LoRA 小模型上无收益证据] → proposal 和文档写明；效果 A/B 不在范围。
- [岛内白化导致各岛尺度差异] → 标注为推断，写入文档；不在本 change 修正。
- [MAPO 证据弱] → proposal 标注；只作为可选机制开放，不做任何推荐。
- [分派器是 P1-b 的产物，接口可能与本 design 假设不同] → 本 change 的实现任务以 P1-b 合入后的接口为准，名称差异只影响枚举值，不影响 spec。
- [奖励向量写在 metadata 中，reward 函数作者可能拼错键] → 运行时严格校验，缺失即失败，报错给出期望的键和分量名。
- [放行参数被误用于正式实验] → 只允许单岛，且在来源记录中标记。

## Migration Plan

1. 只影响 ports 路径，只增加可选机制；未选用的配置哈希与 argv 不变。
2. 依赖顺序：P0 合入 → P1-b 分派器合入 → 本 change。
3. 回滚：从能力声明中移除对应机制即可恢复为"可表达未开放"；来源记录新增字段是附加内容。

## Open Questions

- MaxRL 原文的 r̂ 是否为留一均值 [未核实]：若是，只改变变换内部公式和 CPU 参考值，不改变 spec 要求（spec 以组均值写出，届时同步修改需求文本与测试）。
- MAPO 原文的 σ 是总体 std 还是样本 std [未核实]：本 change 按 D5 采用与 Miles GRPO 一致的定义；若与原文不同，在文档中注明差异。
- syncer 外层平均的加权规则 [未核实]（research §10 问题 4）：影响 D10 的推断是否成立，不影响本 change 的实现。

## Known deviations after implementation (2026-09-29, pending approval; see progress.md "待批准")

- D2 (GSPO fully-clipped round): the relaxing path was not exercised on GPU. In attempt 6 the
  round event's `masked_fraction` was null (before INFRA d9bf29c). While it is null the rule
  stays strict, so a round in which every sequence is clipped and grad_norm is 0 would be
  reported as a zero-gradient failure (a false alarm, never a missed one).
- D8 (REINFORCE++): with identical rewards everywhere and a reward-side KL, the rule gives no
  verdict and falls back to the R0 rule (no gradient expected), because the reward-KL size is not
  reported to the driver (it is exactly 0 on round 0). This is looser than the spec requirement
  "advantages not all equal before whitening -> gradient expected": it can only miss a failure,
  never raise a false one.
- R1 GPU recheck (evidence/r1-gspo, integ-decl 501d71d, gspo_s2): pre-declared checker result
  **not passed** because its round delimiter (Miles "step 0") was wrong -- Miles numbers steps
  cumulatively. Observation only: every round's `clip_fraction` and `masked_fraction` were non-null
  and equal to the mean of that round's two Miles `pg_clipfrac` values (0.09375, 0.25, 0.25). The
  full-clip branch of D2 remains uncovered on GPU.
