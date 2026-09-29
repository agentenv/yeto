## Context

动机见 proposal.md。现状与约束：

- P0（`rl-algorithm-capabilities`）已提供 `CorrectionSpec`、映射表 `miles_adapter/algorithm_flags.py`（含 `--use-tis`、`--tis-clip*`、`--custom-tis-function-path`、`--get-mismatch-metrics`、`--use-opsm`、`--opsm-delta`、`--use-rollout-logprobs`）、`EngineCapabilities.corrections`、`expects_gradient()`、`TrainStepMetrics.masked_fraction`、`PluginRef` 源码哈希，以及"TIS 与 `use_rollout_logprobs` 互斥"的拒绝规则。本 change 不重做这些，只填充各机制的校验、判定与声明。
- Miles 证据（`9e4260d`，research.md §4.1/§4.2）：`--custom-tis-function-path`（A:1759）在 PPO 项之后、聚合之前调用，入参含 pg_loss、train_old 与 rollout logprob、loss_masks；`--get-mismatch-metrics`（A:1705）需要同时给 custom-tis 路径。TIS 实现 `LH/corrections.py:7`，IcePop `LH/corrections.py:35`，OPSM `LH/math_utils.py:183`，MIS 在 `examples/infra_features/train_infer_mismatch_helper/mis.py:311`。
- ports 串行 driver 在生成后用 policy token 校验每组样本来自当前权重（`yeto/rl/engine/driver.py:332-340`）。
- 不改 legacy `build_miles_argv`、不改 Miles/SGLang fork、默认 GRPO argv 与哈希不变。

## Goals / Non-Goals

**Goals：**
- 提供只观测模式，量化 ports LoRA 路径的训推不一致，且证明它不改变训练。
- 把 TIS、IcePop、OPSM、MIS 从"可表达未开放"变为"声明支持"，每项有 CPU 数值对照与 G1 冒烟证据。
- 阈值与 logprob 来源全部显式，进入算法身份。

**Non-Goals：**
- 不做效果 A/B（G4），不声称任何修正带来收益。
- 不决定是否默认开启修正；G2 报告交用户决定。
- 不开放策略陈旧度 > 0；不做 decoupled 下的对比实验（等 `fix-decoupled-lr-schedule`）。
- 不重写 Miles 的修正数学（MIS vendor 副本除外，且只复制不修改）。

## Decisions

### D1. 修正语义与外层同步无关

ports 串行 driver 下，行为策略 π_behav（SGLang 以 W_r 生成）与 π_old（Megatron 在 W_r 上重算）是同一组权重，driver.py:332-340 用 policy token 强制这一点。所以 π_old/π_rollout 比率只反映两个引擎的数值差异（kernel、bf16、LoRA 发布路径），这些修正修的只是训推数值不一致。

- strict-avg 每轮 apply 时重置优化器，decoupled 保留优化器，二者都发生在轮边界，修正在轮内计算，**不影响修正语义**。
- π_θ 相对 π_old 的偏移来自轮内 k 个 mini-batch，由 PPO clip 处理，与修正无关。
- Miles TIS 在数学上等价于截断版的 AReaL 解耦 PPO（research.md §2），是未来 rl-infra-spec 异步模式（staleness>0）的基础。本 change 保持 `execution.max_policy_staleness=0`，spec 中写明不放开。

### D2. 只观测插件：`yeto.rl.algos.mismatch_observe`

一个遵循 Miles custom-tis 函数签名的函数：返回全 1 权重、原样返回 `loss_masks`，并计算与 Miles 内置函数同名的 mismatch 指标（`tis`、`tis_abs`、`train_rollout_kl`、`ess_ratio` 等；具体键名以 `9e4260d` 源码为准，任务 2.1 核对）。翻译为 `--custom-tis-function-path yeto.rl.algos.mismatch_observe.<fn> --get-mismatch-metrics`。

- 指标计算尽量直接调用 Miles 源码中已有的指标辅助函数；若 Miles 未把指标计算单独拆出，则在插件内按 Miles 公式实现，并用 CPU 测试与 Miles 内置 TIS 函数输出的指标逐项比对。
- 若 Miles 仅在 `--use-tis` 为真时才调用 custom-tis 函数（任务 2.1 核对），翻译同时输出 `--use-tis`，由于权重恒为 1，训练不变；这一点由 CPU 梯度对照证明。
- "梯度相同"的证明：在 miles-next-venv 中构造 CPU 张量，分别走"无修正"与"只观测"两条路径调用 Miles 的 `losses`/`corrections` 计算 policy loss 并反传，断言 loss 与梯度逐元素相等（`torch.equal`）。
- 插件哈希按 P0 D7 进入算法身份。

### D3. 阈值显式，yeto 不设默认值

原博客推荐的 TIS C 值、IcePop 论文的 [0.5, 5]、OPSM 原文的 δ 都未在本仓库核实，所以 `CorrectionSpec` 对每个选中机制要求阈值显式给出，缺失即拒绝（spec "阈值必须显式给出"）。文档给出"论文取值"作为参考，并标注来源与"未核实"。Miles 自身的默认值（如 `tis_clip_low=0`）不被静默继承：翻译时总是输出显式值。

### D4. IcePop 与 MIS 通过 custom-tis 路径接入，彼此互斥

IcePop 翻译为 `--custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function --tis-clip-low L --tis-clip H`；MIS 翻译为指向 MIS 函数（原路径或 vendor 副本）并附其阈值参数。custom-tis 路径只有一个，因此只观测、IcePop、MIS 在结构上互斥；TIS（vanilla）与三者也互斥（spec "至多一种修正方式"）。是否开启 `--get-mismatch-metrics` 由描述的 `emit_metrics` 字段决定，修正模式下默认开启以便对比（它只加指标，不改 loss；由 G1 核对）。

### D5. OPSM 的 π_old 来源：显式字段，默认沿用 Miles

Miles 的 OPSM 默认用训练端重算的 π_old，只有 `--use-rollout-logprobs` 时才用推理端 logprob；DeepSeek-V3.2 原文用推理端。决定：

- `CorrectionSpec` 新增 `opsm_old_logprob_source ∈ {trainer, rollout}`，默认 `trainer`（沿用 Miles），进入规范化与哈希。选 OPSM 时该字段总是显式写入规范化 JSON，避免"默认值"随 Miles 变化而静默改变语义。
- `rollout` 翻译为 `--use-rollout-logprobs`。注意这会同时把 PPO ratio 的 π_old 也换成推理端，**不只是 OPSM**，文档与报错中写明；并受 P0 的"与 TIS 互斥"规则约束。
- 语义差异：`trainer` 下 OPSM 衡量的是 π_θ 相对训练端 π_old 的序列级 KL，只屏蔽"轮内更新过猛且 advantage 为负"的序列，不涵盖训推差异；`rollout` 下还包含训推差异，接近原文。
- G1 只对 `trainer` 做冒烟；`rollout` 若也通过 G1 则一并声明，否则只声明 `trainer`，`rollout` 保持 ⚙（能力声明按"机制+来源"粒度）。

### D6. MIS：先核实可 import，否则 vendor

第一项任务：在运行镜像（R0 冒烟所用的官方 radixark/miles 镜像 + yeto fork 层）中执行 `python -c "import examples.infra_features.train_infer_mismatch_helper.mis"` 及等价路径，记录结果。

- 能 import：直接引用，PluginRef 的哈希取 Miles 源文件。
- 不能 import：复制到 `yeto/rl/algos/vendor/miles_mis.py`，文件头记录来源仓库、commit（`9e4260d`）、原路径与许可证（Miles 仓库许可证，任务中核实；预期 Apache-2.0），内容不做语义修改（仅允许修 import）。PluginRef 哈希取副本文件。CPU 测试比对副本与 Miles 原文件在相同输入上的输出。
- geo-MIS 作为 MIS 的变体字段（`mis_mode`），具体可选模式以 `mis.py` 提供的为准，任务中列出。

### D7. 各机制的 `expects_gradient` 判定

| 机制 | 判定 |
|---|---|
| 无修正 / 只观测 / TIS | 默认 GRPO：存在组内 reward 方差 > 0 即期望梯度。vanilla TIS 把权重截断到 [low, high]，不屏蔽 token，按默认处理（`low=0` 时是否可能整轮权重为 0 由任务 3.1 核对源码，若可能则改为屏蔽类判定） |
| IcePop | `masked_fraction == 1.0` 时不期望梯度；否则同默认 |
| OPSM | 同上（OPSM 屏蔽序列） |
| MIS 屏蔽变体 | 同上；截断变体按默认 |

`masked_fraction` 由 adapter 从 Miles 指标读取（键名在任务 3.x 核对，可能需要由修正指标推导，例如 IcePop 的区间外比例）。读不到时为 None，按期望梯度处理（P0 D6 的保守规则）。grad_norm 非有限任何机制都失败。

### D8. 声明门槛与粒度

`EngineCapabilities.corrections` 的元素为机制名（OPSM 附来源），Miles adapter 在本 change 开始时仍为空集。每项 G1 通过后，单独提交一次声明变更，并在 `progress.md` 写入证据目录。fake engine 从一开始就声明全部机制，供 CPU 测试。G3 结果只写入文档与 progress，不作为声明门槛。

### D9. R0/P0 归档后的合并

本 change 的 capability 在 R0、P0 归档后应并入 `rl-algorithm-spec`（修正组要求、零梯度判定）；验证层级要求可保留为独立 capability。合并在归档时做，不在本 change 内。

## GPU 验证方案

复用 R0 冒烟的小模型、镜像与 harness，证据按 `openspec/changes/rl-engine-ports/progress.md` 的习惯放在 `openspec/changes/rl-algo-mismatch-correction/evidence/<日期>-<名称>/`，含 `YETO_SHA`、事件与指标 jsonl、argv、GPU 名断言输出、费用记录。Modal 上用 `H100!:N` 并断言 GPU 名。

- **G1（1 卡）**：只观测、TIS、IcePop、OPSM(trainer)、MIS 各跑 2–3 轮（OPSM(rollout) 可选）。通过标准：指标键存在且有限；不变量无误报；每轮 policy token 与 receipt 正常。
- **G2（1 卡）**：只观测约 20 轮，输出报告：每轮 `train_rollout_kl`、`tis_abs` 分布（分位数）、`ess_ratio`，以及区间 [0.5,5] 外的 token 比例（用于评估 IcePop 会屏蔽多少）。报告只陈述数据，是否推荐默认开启由用户决定。报告注明执行模式与权重传输方式（serial-colocated 为 CUDA IPC；`rl-infra-spec` 的 partitioned 模式下 LoRA 必须走 NCCL broadcast），结论不外推到未测的执行模式（alignment.md A5）。
- **G3（1+1 卡）**：两岛 strict-avg，TIS 与 IcePop 各一次，每次 3 轮左右。通过标准：两岛算法哈希一致、外层同步后权重 hash 一致、不变量无失败。

## Risks / Trade-offs

- [Miles custom-tis 函数签名或指标键名与 research 记录不符] → 任务 2.1 先读源码核对，CPU 测试直接调用 Miles 源码，签名变化会让测试失败而不是静默出错。
- [只观测模式若需要 `--use-tis` 才生效，可能引入额外计算路径] → CPU 梯度对照证明数值不变；G1 对比只观测与无修正的首轮 loss。
- [`--use-rollout-logprobs` 影响范围超出 OPSM] → 文档与报错明确；未通过 G1 则不声明。
- [MIS vendor 副本与上游漂移] → 副本头记录 commit；CPU 测试比对原文件；Miles 升级时在 `docs/MILES_RL.md` 升级步骤中要求重新比对。
- [`masked_fraction` 取不到导致 IcePop 全屏蔽轮误报失败] → 这是保守方向；G1 若出现，记录并补指标推导，不放宽判定。
- [G2 的量化结果受小模型限制] → 报告中注明模型与配置，结论不外推到大模型或 MoE。

## Migration Plan

只影响显式选择修正的 ports 运行。回滚：删除 adapter 中对应的声明即可让机制回到"可表达未开放"；revert 本 change 不涉及数据迁移。

## Open Questions

1. Miles `9e4260d` 中 custom-tis 函数是否只在 `--use-tis` 时调用，以及 `--get-mismatch-metrics` 的指标键全集（任务 2.1 核对，影响 D2 翻译，不影响 spec）。
2. MIS 在运行镜像中能否 import（任务 5.1），以及 Miles 仓库许可证（预期 Apache-2.0，未核实）。
3. `masked_fraction` 在 Miles 日志中的确切来源，IcePop/OPSM/MIS 是否各自输出屏蔽比例。
4. G2 之后是否推荐默认开启某修正：由用户根据报告决定，本 change 不预设。
5. OPSM(rollout) 是否值得验证到声明：取决于 G1 预算，可留为 ⚙。
