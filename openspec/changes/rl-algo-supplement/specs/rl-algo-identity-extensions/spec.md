## Purpose

规定 rollout 采样参数、MoE routing replay、ref 周期更新、loss mask 策略和 RLOO 基线如何进入算法规范化结果与哈希，保证两岛之间算法身份完整。

## ADDED Requirements

### Requirement: 新身份字段取默认值时哈希不变
本能力新增的每个字段在未设置时 SHALL NOT 出现在规范化结果中；只含既有字段的 spec，哈希与本变更之前逐位相同。字段被显式设置后，SHALL 进入规范化结果与哈希。

#### Scenario: 默认 GRPO 哈希不变
- **WHEN** 使用默认 GRPO spec 构建规范化结果
- **THEN** 哈希与变更前相同

#### Scenario: 显式设置采样温度
- **WHEN** spec 设置 rollout 温度为 0.8
- **THEN** 规范化结果包含该值，哈希与默认值时不同

### Requirement: rollout 采样参数属于算法身份
rollout 的 temperature、top-p、top-k SHALL 由算法 spec 表达并翻译为引擎参数；在 extra argv 中给出的相同参数 SHALL 被吸收进 spec，与 spec 冲突时报错并列出两边的值。评测采样参数 SHALL NOT 影响算法身份。在得到验证之前，温度不等于 1 与“用 rollout logprob 作为 π_old”的组合 SHALL 被拒绝。

#### Scenario: extra argv 与 spec 冲突
- **WHEN** spec 设置 top-p 为 0.9，extra argv 中给出 --rollout-top-p 0.95
- **THEN** 启动前报错，并列出两边的值

#### Scenario: 非默认温度与 rollout logprob 组合
- **WHEN** spec 设置温度为 0.7，并启用 rollout logprob 作为 π_old
- **THEN** 在拒绝矩阵阶段失败，报错写明原因

### Requirement: MoE routing replay 属于算法身份
rollout routing replay 与训练侧 routing replay SHALL 由算法 spec 表达并进入哈希。已有的 agent 级开关 SHALL 继续可用，并被吸收进 spec。模型不是 MoE 时，启用这两个开关 SHALL 在启动前被拒绝。

#### Scenario: 旧开关被吸收
- **WHEN** 运行使用旧的 agent 级 rollout routing replay 开关
- **THEN** 引擎行为不变，规范化 spec 中包含该开关，哈希反映这一开关

#### Scenario: 稠密模型启用 routing replay
- **WHEN** 模型不是 MoE，spec 却启用了 routing replay
- **THEN** 启动前拒绝

### Requirement: ref 周期更新属于算法身份并记录 ref 版本
ref 周期更新间隔 SHALL 由算法 spec 表达并进入哈希，而且只有在 KL 实际生效时才被接受。运行 SHALL 记录当前 ref 版本（最近一次更新时的 rollout 编号与 ref 权重哈希）。在 cut 能够记录 ref 版本之前，需要 cut 的运行配置 SHALL 拒绝这一字段。

#### Scenario: 没有 KL 时设置 ref 更新
- **WHEN** spec 设置了 ref 更新间隔，但 KL 放置为 none
- **THEN** 在拒绝矩阵阶段失败

#### Scenario: 需要 cut 的运行
- **WHEN** 在 cut 尚不支持 ref 版本时，需要 cut 的运行设置了 ref 更新间隔
- **THEN** 启动前拒绝，报错说明依赖

### Requirement: loss mask 策略属于算法身份
rollout 的 loss mask 策略 SHALL 是一个声明式枚举字段；默认值对应引擎当前行为且不输出。未列入允许值的取值 SHALL 被拒绝，报错中列出允许值。

#### Scenario: 未知取值
- **WHEN** spec 给出一个未列入允许值的 loss mask 策略
- **THEN** 构建 spec 时失败，报错列出允许值

### Requirement: RLOO 留一基线
RLOO SHALL 作为 reward 后处理的一个阶段来表达，优势为 r_i 减去组内其余样本 reward 的均值。它 SHALL 要求组大小至少为 2、关闭 std 归一化，并且拥有与 Dr.GRPO 不同的算法身份。

#### Scenario: 组大小为 1
- **WHEN** 使用 RLOO，而每个 prompt 的样本数为 1
- **THEN** 启动前拒绝

#### Scenario: RLOO 数值
- **WHEN** 一组 reward 为 [1, 0, 0, 1]
- **THEN** 优势为 [2/3, −1/3, −1/3, 2/3]
