# Spec Delta

## Purpose

保证 RL 岛在训练结束之前，每一次本地优化步都以非零学习率真正改动适配器；学习率调度由 yeto 按同步方式显式决定，在 legacy 与 ports 两条引擎路径上一致，而不是由引擎依据与实际步数无关的轮数推算。

## ADDED Requirements

### Requirement: 学习率调度必须由 yeto 显式决定

岛启动训练引擎时 SHALL 显式指定学习率调度的衰减方式、衰减步数、warmup 步数和最小学习率，MUST NOT 依赖训练引擎根据轮数推算的默认调度。同一份 RL 运行配置在 legacy 与 ports 两条引擎路径上 SHALL 得到相同的调度。

#### Scenario: 两条引擎路径得到相同调度
- **WHEN** 同一份 RL 运行配置分别以 legacy 和 ports 引擎启动
- **THEN** 两条路径传给训练引擎的衰减方式、衰减步数、warmup 步数和最小学习率完全相同

#### Scenario: 仅评估的运行不设置训练调度
- **WHEN** 运行配置只做评估、不执行任何优化步
- **THEN** 岛不向训练引擎传入学习率调度参数

### Requirement: decoupled 岛在整个运行期间保持配置的学习率

在 decoupled 同步方式下，岛的本地步数直到 syncer 发出最终 cut 才确定，因此学习率 SHALL 在整个运行期间保持为配置值，不随本地步数衰减。

#### Scenario: 本地步数超过全局轮数
- **WHEN** decoupled 岛执行的本地优化步数超过 `global_rounds × optimizer_steps`
- **THEN** 每一步应用的学习率仍等于配置的学习率
- **AND** 只要该步的梯度非零，参数就会改变

#### Scenario: 后半程的全局更新非零
- **WHEN** 两个 decoupled 岛运行到最终 cut，且各轮的优势不全为零
- **THEN** syncer 在最终 cut 之前每个外层步记录的全局 delta 范数都不为 0

### Requirement: strict-avg 岛的学习率轨迹保持不变

在 strict-avg 同步方式下，岛 SHALL 在 `global_rounds × optimizer_steps` 个优化步内把学习率从配置值线性衰减，每一步应用的学习率 SHALL 与本修复之前完全相同。

#### Scenario: strict-avg 学习率逐步一致
- **WHEN** 同一份 strict-avg 配置在修复前后各运行一次
- **THEN** 两次运行每个优化步应用的学习率逐位相同
- **AND** 最后一个优化步应用的学习率大于 0

### Requirement: 以零学习率执行的非最终优化步必须失败

岛 SHALL 在每个本地轮的 `rl_local_round` 事件中记录该轮优化步实际应用的学习率。如果某个本地轮应用的学习率为 0，而该岛在此之后仍会继续训练，该轮 SHALL 失败：岛 MUST NOT 把这一轮的结果作为正常轮提交给 syncer，失败信息 SHALL 指明轮次和观察到的学习率。

#### Scenario: 学习率在运行中途降为 0
- **WHEN** 某个非最终本地轮应用的学习率为 0
- **THEN** 该轮失败，不向 syncer 提交
- **AND** 失败信息包含轮次和学习率 0

#### Scenario: 记录的是实际应用的学习率
- **WHEN** 训练引擎在优化步之后才推进调度器
- **THEN** 事件中记录的是该优化步实际使用的学习率，而不是推进后的下一步学习率
