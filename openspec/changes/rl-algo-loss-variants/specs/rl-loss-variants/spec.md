# Spec Delta

## Purpose

定义 yeto ports 路径上 token 级 policy loss 变体（CISPO、SAPO-Qwen、GMPO）的契约，包括：每个变体算什么；用哪些参数描述、这些参数怎样进入算法身份；可以和哪些机制组合、哪些组合要拒绝；每轮是否应当产生梯度如何判定；实现来源怎样审计；变体在声明支持之前必须完成哪些验证。

## ADDED Requirements

### Requirement: 变体选择与默认行为
算法描述的 loss 变体 SHALL 可取以下值：默认 PPO 式 clip 目标、`cispo`、`sapo`、`gmpo`。未指定时 MUST 取默认值，此时生成的引擎参数和身份哈希 MUST 与未引入本能力之前逐字节一致。

#### Scenario: 未指定变体
- **WHEN** ports 运行不设置 loss 变体
- **THEN** 引擎参数和算法哈希与未引入本能力之前的 GRPO 完全相同

#### Scenario: 指定变体
- **WHEN** 描述中 loss 变体设为 `cispo`
- **THEN** 规范化结果包含该变体，哈希与默认描述不同

### Requirement: CISPO 数值契约
选用 `cispo` 时，每个有效 token 的损失 SHALL 等于 −sg(clip(ρ, 1−ε_l, 1+ε_h))·Â·log π_θ。其中 ρ 为当前策略与旧策略的概率比，sg 表示停止梯度，ε_l、ε_h 取描述中的 clip 下界和上界。损失 MUST 按 token 归一聚合。所有有效 token 的 log π_θ MUST 保留梯度，包括 ρ 越出 clip 范围的 token。

#### Scenario: ratio 越界仍有梯度
- **WHEN** 某 token 的 ρ 大于 1+ε_h 且优势为正
- **THEN** 该 token 的权重取 1+ε_h，且它对 log π_θ 的梯度不为零

#### Scenario: 负优势
- **WHEN** 某 token 的 ρ 小于 1−ε_l 且优势为负
- **THEN** 该 token 的损失等于 −(1−ε_l)·Â·log π_θ，逐元素误差不超过测试容差

### Requirement: SAPO-Qwen 数值契约
选用 `sapo` 时，每个有效 token 的目标 SHALL 用软门控 f(ρ)=σ(τ(ρ−1))·4/τ 替代硬 clip。优势为正时 τ 取 τ_pos，优势为负时 τ 取 τ_neg。τ_pos、τ_neg MUST 为正的有限数，默认值分别为 1.0 和 1.05。软门控 MUST NOT 对任何 token 产生硬 mask。

#### Scenario: 正负优势使用不同温度
- **WHEN** 同一 batch 中既有正优势 token，也有负优势 token
- **THEN** 正优势 token 按 τ_pos 计算，负优势 token 按 τ_neg 计算，与公式逐元素一致

#### Scenario: 非法温度
- **WHEN** τ_neg 为 0 或负数
- **THEN** 启动前失败，报错指明该字段

### Requirement: GMPO 数值契约
选用 `gmpo` 时，每条序列的目标 SHALL 取有效 token 目标的几何平均。每个 token 的 log ρ 先按优势符号在 log 空间 clip 到 (−δ_l, δ_h)，默认 δ_l=δ_h=0.4。序列 ratio MUST 在该序列的全部有效 token 上计算，即使序列被上下文并行切分也一样。

#### Scenario: 上下文并行下的序列 ratio
- **WHEN** 一条序列被切分到多个上下文并行分片
- **THEN** 各分片使用的序列几何平均 ratio 相同，并与不切分时的计算结果一致

#### Scenario: log 空间越界
- **WHEN** 某 token 的 log ρ 大于 δ_h 且优势为正
- **THEN** 该 token 以 δ_h 参与几何平均，且不再贡献梯度

### Requirement: 变体参数进入身份
变体参数（CISPO 使用的 clip 上下界、τ_pos、τ_neg、GMPO 的 log 空间 clip 范围）SHALL 属于算法描述，参与规范化和身份哈希。额外透传的引擎参数中如果出现这些参数，MUST 按映射吸收进描述；与描述取值冲突时 MUST 在启动前报错。

#### Scenario: 透传温度参数
- **WHEN** 额外参数给出与描述不同的 τ_pos
- **THEN** 启动前失败，报错列出两个取值

### Requirement: 组合与拒绝
变体 SHALL 能与训推不一致修正（TIS、IcePop）、KL loss、entropy 奖励和 token 级聚合组合，组合后的数值 MUST 等于"先按变体计算 token 损失，再按修正权重调整"的结果。以下组合 MUST 在启动前拒绝，并给出替代方案：
- 变体与序列级 ratio 估计器组合；
- 变体与 dual-clip 组合；
- 变体参数非法。

#### Scenario: 与 TIS 组合
- **WHEN** 选用 `cispo` 并开启 TIS
- **THEN** 每个 token 的损失等于 CISPO 损失乘以截断后的 TIS 权重，与参考公式逐元素一致

#### Scenario: 与序列级 ratio 冲突
- **WHEN** 选用 `gmpo`，且 advantage 估计器为 GSPO
- **THEN** 在创建任何 GPU 进程之前失败，报错说明两者都在定义序列级 ratio

### Requirement: 按变体判定梯度
每个变体 SHALL 给出"本轮是否应当有非零梯度"的判定：
- CISPO：只要存在有效 token 且优势不全为零，就应当有梯度；
- SAPO：软门控不产生 mask，判定与 CISPO 相同；
- GMPO：当所有序列都因 log 空间 clip 失去梯度时，允许梯度为零。

梯度范数非有限时 MUST 在任何变体下都判为失败。

#### Scenario: CISPO 全部 ratio 越界
- **WHEN** 选用 `cispo`，所有 token 的 ρ 都越界，且优势不全为零
- **THEN** 判定为应当有梯度，梯度为零时报告失败

#### Scenario: GMPO 全部被 clip
- **WHEN** 选用 `gmpo`，所有序列都在 clip 之外，且引擎报告的 clip 比例为 1
- **THEN** 零梯度不被判为失败

### Requirement: 开放前的验证与实现来源
某个变体 SHALL 满足以下全部条件后，才可被声明为 ports 支持：
- CPU 数值测试（正负优势、ratio 越界、全 mask，以及与 TIS、IcePop 的组合）全部通过；
- 在 GPU 上完成冒烟。

在此之前，选用该变体 MUST 在启动前被拒绝。变体计算的来源 MUST 可审计：如果由 yeto 插件提供，插件源码哈希进入身份；如果由引擎 fork 提供，运行记录 MUST 包含该 fork 的提交号。实现 MUST NOT 在运行时替换引擎内部函数。

#### Scenario: 未完成验证的变体
- **WHEN** 某变体只通过了 CPU 测试，尚未完成 GPU 冒烟
- **THEN** 选用它时启动前失败，报错说明该变体可表达但未开放

#### Scenario: 来源记录
- **WHEN** 以已开放的变体完成一次 ports 运行
- **THEN** 来源记录中包含变体名、变体参数，以及插件源码哈希或引擎提交号中的一项
