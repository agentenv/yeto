# Spec Delta

## Purpose

定义 yeto ports 路径上序列级 ratio（GSPO）、REINFORCE++ 家族以及 MaxRL、MAPO、GDPO 等 advantage 变体的开放契约：各机制的参数约束、数值定义、统计范围、奖励输入要求、零梯度判定，以及在引擎能力中声明支持的条件。

## ADDED Requirements

### Requirement: 验证通过后才声明支持
本 capability 中的每个机制 SHALL 仅在其 1 卡 GPU 冒烟验证通过后，才在引擎适配的能力声明中列为支持。未声明支持的机制 MUST 仍按"可表达未开放"在创建 GPU 进程之前被拒绝。为执行冒烟而临时放行未声明机制时，该运行 MUST 在运行事件和导出的来源记录中标记为未验证机制运行，且 MUST NOT 被用于多岛外层同步。

#### Scenario: 未验证机制被拒
- **WHEN** 某机制尚未通过冒烟验证，用户在描述中选用它并正常启动 ports 运行
- **THEN** 在创建 GPU 进程之前拒绝启动，报错列出该机制为未开放

#### Scenario: 冒烟放行被标记
- **WHEN** 为冒烟显式放行一个未声明机制并完成运行
- **THEN** 运行事件与来源记录都标记本次为未验证机制运行

#### Scenario: 放行不能用于多岛
- **WHEN** 放行未声明机制的同时启动两个岛的外层同步
- **THEN** 启动前失败

### Requirement: 默认行为不变
未选择本 capability 中任何机制的 ports 配置，其算法哈希、生成的引擎参数与训练行为 MUST 与本 change 之前逐字节相同。

#### Scenario: 默认 GRPO 不变
- **WHEN** 以默认 GRPO 描述启动 ports 运行
- **THEN** 生成的引擎参数与算法哈希与本 change 之前相同

### Requirement: GSPO 序列级 ratio
选用 GSPO 时，系统 SHALL 以序列内 token 对数 ratio 的均值取指数作为该序列的 ratio，并对序列 ratio 施加 clip。clip 下界与上界 MUST 在描述中显式给出，缺失时 MUST 在启动前拒绝，报错 MUST 说明引擎默认值与论文取值不同。GSPO 运行 MUST 在每轮指标中报告被裁比例。

#### Scenario: 未给 clip
- **WHEN** 描述选用 GSPO，但未显式给出 clip 范围
- **THEN** 启动前失败，报错提示必须显式给出 clip，并说明论文取值量级

#### Scenario: 报告被裁比例
- **WHEN** GSPO 运行完成一轮训练
- **THEN** 该轮事件包含被裁比例指标

### Requirement: GSPO 零梯度判定
GSPO SHALL 声明：本轮存在非零 advantage，且被裁比例小于 1 时，才期望非零梯度。被裁比例为 1 且 grad_norm 为 0 的轮次 MUST NOT 判为失败，事件 MUST 记录被裁比例。被裁比例无法读取时 MUST 按期望有梯度处理。grad_norm 非有限 MUST 仍判为失败。

#### Scenario: 全部序列被裁
- **WHEN** GSPO 某轮 advantage 非零，被裁比例为 1，grad_norm 为 0
- **THEN** 该轮不因零梯度不变量失败，事件记录被裁比例

#### Scenario: 被裁比例缺失
- **WHEN** GSPO 某轮 advantage 非零，被裁比例读不到，grad_norm 为 0
- **THEN** 该轮失败并写入事件

### Requirement: REINFORCE++ 家族与岛内白化
选用 REINFORCE++ 或 REINFORCE++-baseline 时，advantage 白化 SHALL 只在本岛内的训练数据上计算统计量，MUST NOT 跨岛聚合任何统计量。baseline 版 MUST 先在组内减去均值。reward 中的 KL（`placement=reward`）SHALL 被允许并实际进入 advantage。折扣因子默认 1.0 且不改变生成的引擎参数；非 1.0 的折扣因子 SHALL 可表达，但在未验证前 MUST 按未开放拒绝；折扣因子用于其他估计方式时 MUST 拒绝。

#### Scenario: 白化不跨岛
- **WHEN** 两个岛以 REINFORCE++ 并启用白化参与外层同步
- **THEN** 每个岛的 advantage 只由本岛数据的统计量归一化，岛之间不交换统计量

#### Scenario: reward KL 被允许
- **WHEN** 描述选用 REINFORCE++，KL 放置为 reward 且系数大于 0
- **THEN** 启动不被 KL 放置规则拒绝，生成的引擎参数包含该 KL 系数

#### Scenario: 非默认折扣因子
- **WHEN** 描述选用 REINFORCE++ 并设置折扣因子为 0.99
- **THEN** 启动前以未开放拒绝，报错列出已支持的取值

### Requirement: 组内 advantage 变换保持多段语义
MaxRL、MAPO、GDPO 变换 SHALL 在 reward 塑形之后执行，并以 rollout 为单位计算：同一 rollout 的多段样本 MUST 先合并为一个条目参与组内统计，结果 MUST 广播回每一段；同一 rollout 各段的奖励（或奖励向量）不一致时 MUST 使本轮失败。组的划分 MUST 与内置 GRPO 归一化相同。这些变换 MUST 只选其一，MUST 只与组内估计方式组合，与其他估计方式组合时 MUST 在启动前拒绝。

#### Scenario: 多段样本共享 advantage
- **WHEN** 一个 rollout 被拆成三段样本，所在组启用 MaxRL
- **THEN** 三段样本得到相同的 advantage，组统计中该 rollout 只计一次

#### Scenario: 段间奖励不一致
- **WHEN** 同一 rollout 的两段样本奖励不同
- **THEN** 本轮失败，报错指明该 rollout 与各段奖励

#### Scenario: 与不兼容估计方式组合
- **WHEN** 描述同时选用 MaxRL 与 REINFORCE++ 估计方式
- **THEN** 启动前失败

### Requirement: MaxRL 数值定义
MaxRL SHALL 对每个组计算 A_i=(r_i−r̂)/r̂，r̂ 为组内 rollout 奖励均值。r̂ 为 0（全错组）时 MUST 令该组 advantage 全为 0，不得产生非有限值。组大小为 1 时 advantage MUST 为 0。MaxRL MUST 要求奖励声明为二值；运行时出现 0 与 1 以外的奖励 MUST 使本轮失败。

#### Scenario: 一般组
- **WHEN** 组奖励为 [1, 0, 0, 1]
- **THEN** advantage 为 [1, −1, −1, 1]

#### Scenario: 全错与全对
- **WHEN** 组奖励全为 0，或全为 1
- **THEN** 该组 advantage 全为 0，且均为有限值

#### Scenario: 非二值奖励
- **WHEN** 描述声明的奖励不是二值
- **THEN** 启动前失败

### Requirement: MAPO 数值定义
MAPO SHALL 对每个组计算 A=(1−λ)(r−μ)/σ+λ(r−μ)/μ，其中 p 为组内正确率、μ 为组内均值、λ=1−4p(1−p)，σ MUST 与内置 GRPO 归一化使用相同的标准差定义。σ 为 0 时第一项 MUST 取 0；μ 为 0 时第二项 MUST 取 0；结果 MUST 为有限值。λ 为 0 时结果 MUST 与内置 GRPO 归一化逐元素一致。MAPO MUST 要求奖励声明为二值。

#### Scenario: 半数正确退化为 GRPO
- **WHEN** 组内恰好一半正确
- **THEN** MAPO advantage 与内置 GRPO 归一化结果逐元素一致

#### Scenario: 全错组
- **WHEN** 组奖励全为 0
- **THEN** 该组 advantage 全为 0，且均为有限值

### Requirement: 奖励向量声明与校验
GDPO SHALL 读取 reward 函数随每个样本写入的奖励向量，格式为"分量名 → 数值"的映射。算法描述 MUST 声明分量名集合及每个分量的权重，二者进入算法哈希。运行时任一样本缺少声明的分量、含未声明的分量、或分量值非有限时，MUST 使本轮失败并报告样本与分量名，MUST NOT 以默认值填补。声明为空或权重非有限时 MUST 在启动前拒绝。

#### Scenario: 缺少分量
- **WHEN** 描述声明分量 correctness 与 format，某样本只写入 correctness
- **THEN** 本轮失败，报错指明该样本缺少 format

#### Scenario: 权重进入哈希
- **WHEN** 两个描述只有某分量权重不同
- **THEN** 二者算法哈希不同

### Requirement: GDPO 数值定义
GDPO SHALL 对每个分量在组内独立归一化（减均值，标准差大于 0 时除以标准差），按声明权重求和，再在本岛本轮全部 rollout 条目上做白化（减均值，标准差大于 0 时除以标准差）。白化 MUST NOT 跨岛。组大小为 1 或某分量组内标准差为 0 时，该分量对该组的贡献 MUST 为 0 或仅减均值，结果 MUST 为有限值。

#### Scenario: 与公式逐元素一致
- **WHEN** 对给定的多分量奖励和权重计算 GDPO
- **THEN** 结果与按上述公式独立计算的参考值逐元素一致

#### Scenario: 单分量恒定
- **WHEN** 某分量在一个组内所有 rollout 上取值相同
- **THEN** 该分量对该组 advantage 的贡献为 0

### Requirement: advantage 变体的零梯度判定
每个 advantage 变体 SHALL 声明本轮是否期望非零梯度：
- MaxRL、MAPO、REINFORCE++-baseline：存在组内奖励标准差大于 0 的组，与默认 GRPO 相同；全错组与全对组 MUST 不计为期望梯度；
- GDPO：变换输出中存在非零 advantage；
- REINFORCE++：本岛本轮 advantage 在白化前不全相等。
按上述判定期望梯度而 grad_norm 为 0 时 MUST 判为失败；不期望梯度时 MUST NOT 判为失败。

#### Scenario: MaxRL 全错轮次
- **WHEN** MaxRL 某轮所有组都全错，grad_norm 为 0
- **THEN** 该轮不因零梯度不变量失败

#### Scenario: GDPO 期望梯度但为零
- **WHEN** GDPO 某轮变换输出存在非零 advantage，grad_norm 为 0
- **THEN** 该轮失败并写入事件
