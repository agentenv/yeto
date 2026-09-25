# Spec Delta

## Purpose

保证在 SkyPilot 集群上运行的 Miles RL 岛能与 SkyPilot 自身的运行时共存：岛对进程的清理只影响岛自己的 Ray 集群，SkyPilot 对该集群的健康判定不受影响；head 侧的岛监督在健康状态暂时不可读时以 job 的真实状态为准，不会把一个正在跑的岛重新 launch 掉。

## ADDED Requirements

### Requirement: 岛的进程管理不得影响 SkyPilot 运行时

在 SkyPilot 集群上运行的 RL 岛 SHALL 只启动、连接和停止属于它自己的 Ray 集群。岛脚本在启动前的清理和退出时的清理 MUST NOT 终止节点上不属于该岛的 Ray 进程。岛的训练进程 SHALL 连接到岛自己的 Ray 集群，而不是节点上恰好存在的其他 Ray 集群。

#### Scenario: 岛跑起来后 SkyPilot 仍判定集群健康
- **WHEN** RL 岛在 SkyPilot 集群上开始执行 run 脚本并启动 Miles 的 Ray
- **THEN** 该集群随后的状态刷新仍为 `UP`
- **AND** head 侧能够正常读取岛 job 的状态

#### Scenario: 岛退出后 SkyPilot 运行时仍在
- **WHEN** 岛的 learner 进程结束，run 脚本执行退出清理
- **THEN** 岛自己的 Ray 进程被停止
- **AND** SkyPilot 的运行时 Ray 仍在运行，集群状态仍为 `UP`

#### Scenario: 训练进程连接自己的集群
- **WHEN** 节点上同时存在岛的 Ray 集群和 SkyPilot 的运行时 Ray
- **THEN** learner 连接到岛的 Ray 集群
- **AND** 训练 actor 不会被调度到 SkyPilot 的运行时 Ray 上

#### Scenario: 残留的岛 Ray 会被清掉
- **WHEN** 同一节点上残留着上一次岛运行留下的 Ray 进程
- **THEN** 新的岛脚本在启动前把它们清除，并成功启动新的 Ray 集群

#### Scenario: Modal 岛行为不变
- **WHEN** 同一份 run 脚本在 Modal 容器中执行（容器内没有 SkyPilot 运行时）
- **THEN** 岛照常启动 Ray、训练并退出，与改动前的行为一致

### Requirement: 岛监督不得对仍在运行的 job 重新 launch

head 侧的岛监督在无法读取岛 job 状态、而原因是集群健康状态不是 `UP` 时，SHALL 查询该集群的 job 队列。若被监督的 job 仍处于未结束状态（等待、setup 中或运行中），监督 SHALL 将岛视为健康并继续等待；MUST NOT 对它发起恢复性的重新 launch。只有当 job 已不在队列中、已结束为非成功状态，或集群在云端确实不存在时，才进入恢复。

#### Scenario: 集群状态暂时不可读但 job 仍在跑
- **WHEN** 岛 job 状态查询因集群状态不是 `UP` 而失败
- **AND** 该集群的 job 队列显示被监督的 job 仍在运行
- **THEN** 监督继续等待，不重新 launch，不重跑 setup

#### Scenario: job 确实消失时仍会恢复
- **WHEN** 岛 job 状态查询失败
- **AND** job 队列里没有被监督的 job，或它已以非成功状态结束
- **THEN** 监督按现有规则进入恢复（在恢复超时内重新 launch）

#### Scenario: 集群被云端回收时仍会恢复
- **WHEN** 集群在云端已不存在（被抢占或删除）
- **THEN** 监督按现有规则进入恢复，与改动前一致

#### Scenario: 成功结束的 job 照常记为完成
- **WHEN** 岛 job 以成功状态结束
- **THEN** 监督将该岛记为完成，与改动前一致
