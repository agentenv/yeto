# Spec Delta

## Purpose

规定 RL learner 如何在旧引擎路径与基于端口的新引擎路径之间选择，并保证两条路径并存期间互不干扰、可以对照验证。同时规定旧路径在验收之后如何退役。

## ADDED Requirements

### Requirement: 引擎路径选择
RL learner 与启动器 SHALL 接受 `--rl-engine`，取值为 `legacy` 或 `ports`，默认为 `legacy`。
- 选择结果 MUST 在启动前确定，在一次运行内保持不变，并写入运行事件与产物来源记录。
- 选择 MUST 由用户显式给出，MUST NOT 从模型、岛数量或其他参数推断。

#### Scenario: 默认行为不变
- **WHEN** 用户不传 `--rl-engine` 启动 RL 运行
- **THEN** 运行走 legacy 路径，参数、源码准备和训练行为与引入该参数之前完全一致

#### Scenario: 选择被记录
- **WHEN** 以 `--rl-engine ports` 启动
- **THEN** 事件记录和导出的来源记录中包含 `rl_engine=ports`

### Requirement: 两条路径各自固定版本
两条路径 SHALL 各自固定并校验引擎源码身份：
- legacy 路径 MUST 继续使用现有的 fork 仓库、commit、bundle 与校验流程；
- ports 路径 MUST 使用 yeto 自有的引擎 fork 与推理引擎 fork（`michaellchung/miles`、`michaellchung/sglang`）上的固定 commit，并同样校验仓库地址、commit、源码 hash、工作区干净，以及导入路径位于所固定的源码目录内。
- ports 路径的固定 commit MUST 能直接从 fork 仓库获取，MUST NOT 依赖随 yeto 分发的 git bundle。

任一路径的校验失败 MUST 在加载模型之前终止启动。修改其中一条路径的固定版本 MUST NOT 影响另一条。

#### Scenario: 各自独立的版本
- **WHEN** 更新 ports 路径的引擎 commit
- **THEN** legacy 路径的 commit、bundle 与校验值不变，legacy 运行不受影响

#### Scenario: 源码不匹配
- **WHEN** ports 路径的远端 checkout 与固定 commit 不一致或工作区不干净
- **THEN** 在加载模型之前启动失败，并报告期望值与实际值

### Requirement: 不支持的组合明确拒绝
在 R0 阶段，`ports` 路径 SHALL 支持以下组合：
- 因果语言模型 + LoRA；
- GRPO；
- 串行共置；
- strict-avg 与 decoupled 同步预设。

对其他组合（全参数、SAO、DeepSeek V4 专用 recipe、critic、分区执行等），`ports` 路径 MUST 在启动前拒绝，并提示改用 legacy 路径。legacy 路径对这些组合的支持 MUST 保持不变。

#### Scenario: 选择了尚未迁移的功能
- **WHEN** 以 `--rl-engine ports` 启动 SAO 运行
- **THEN** 启动失败，错误信息说明该组合仅 legacy 路径支持

### Requirement: 在途 PR 的语义迁移
在 ports 路径开始实现之前，所有会改动 legacy RL 路径的在途 PR SHALL 已合入或已关闭，并各自登记到迁移清单。清单中每一项 MUST 写明：
- PR 编号；
- 引入的行为（不变量、配置项、数据约定或外部格式）；
- 对应的 ports 路径实现位置；
- 用于验证的测试或验收指标。

在 ports 开发期间新合入的 RL PR MUST 同时修改两条路径，或者在迁移清单中登记为待迁移项。任何待迁移项未关闭时，MUST NOT 切换默认值。

#### Scenario: 已合入 PR 的不变量在新路径生效
- **WHEN** 迁移清单中登记了 #64 的梯度流不变量
- **THEN** ports 路径存在对应测试，并且在零梯度注入下与 legacy 路径表现一致

#### Scenario: 开发期间新合入的 PR
- **WHEN** ports 开发期间合入了一个只修改 legacy 路径的 RL PR，并且没有登记
- **THEN** 迁移清单检查失败，默认值切换被阻止

### Requirement: 等价性验收
在切换默认值之前，SHALL 对每个受支持的同步预设完成等价性验收。验收要求：
- 同一模型与 revision、同一数据与 revision、同一 reward、同一 seed 和同一硬件；
- 分别以 legacy 和 ports 运行，比较每轮的 reward 均值、loss、grad_norm、group 与 token 计数，以及同步后的全局策略；
- 数值容差 MUST 在实验前根据 legacy 自身重复运行的噪声确定并写入报告；
- 任何超出容差的指标 MUST 使验收失败。

#### Scenario: strict-avg 等价
- **WHEN** 以相同配置分别在两条路径上运行 strict-avg 若干轮
- **THEN** 各轮指标都在预先声明的容差之内，报告包含两条路径的原始记录与比较结果

#### Scenario: 超出容差
- **WHEN** 某一轮 ports 路径的 grad_norm 超出容差
- **THEN** 验收失败，默认值保持 legacy

### Requirement: legacy 路径的退役
只有在全部受支持的预设通过等价性验收之后，才 SHALL 将默认值切换为 `ports`。
- 切换之后，legacy 路径 MUST 在至少一个完整的真实运行周期内仍可显式选择。
- 删除 legacy 路径时，MUST 同时删除 `--rl-engine` 参数、旧 fork 的固定常量、bundle 与外部策略同步回调集成。
- 此前仅 legacy 支持的功能，MUST 在删除前已迁移，或者在文档中明确标注为不再支持。

#### Scenario: 切换默认值
- **WHEN** 全部预设都通过了等价性验收
- **THEN** 不传 `--rl-engine` 时走 ports 路径，显式传 `legacy` 时仍可运行旧路径

#### Scenario: 删除 legacy
- **WHEN** legacy 路径被删除
- **THEN** `--rl-engine` 不再是有效参数，仓库中不再有旧 fork 的固定常量与 bundle 文件，文档说明了哪些功能已迁移或已不再支持
