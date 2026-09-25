## Purpose

在 Modal 上运行 Yeto 的 SFT 或 RL learner 岛。Modal 没有虚拟机和 SSH，不能走 SkyPilot；本能力定义一个远程运行器，把一个岛作为 Modal 上的一组 GPU 容器启动，并让它像其他岛一样向 syncer 注册。

## ADDED Requirements

### Requirement: `--gpu` 接受 Modal 条目
`yeto launch --gpu` SHALL 接受 `modal:[NODESx]COUNTxGPU[@region]` 条目。启动器 MUST 把此类条目路由到 Modal 运行器而不是 SkyPilot；同一次启动中 Modal 岛与其他云的岛 SHALL 可以混用。NODES 大于 1 时 COUNT MUST 等于该 GPU 在 Modal 的整机数，否则命令报错退出。

#### Scenario: 混合启动
- **WHEN** 用户运行 `yeto launch --gpu aws:8xh100@us-east-1,modal:8xh100 ...`
- **THEN** AWS 岛经 SkyPilot 启动，Modal 岛经 Modal 运行器启动，两者都以各自的 learner id 出现在 syncer 的学习者列表中

#### Scenario: 非整机多节点被拒
- **WHEN** 用户运行 `yeto launch --gpu modal:2x4xh100`
- **THEN** 命令在启动任何资源之前报错，说明 Modal 多容器要求整机 GPU 数

### Requirement: 岛在 Modal 容器内的行为与其他云一致
Modal 岛 SHALL 运行与 SkyPilot 岛相同的 learner 入口（SFT 为普通 learner，RL 为 Miles learner），接收相同的环境变量（syncer 地址、learner id、Hub token、可选的 W&B 设置），并使用相同的数据来源规则。RL 岛 MUST 使用与 SkyPilot 路径相同摘要的容器镜像。多容器岛 MUST 在容器间用 RDMA 级互联并让 rank 0 容器承担 torchrun 主节点角色。

#### Scenario: SFT 岛完成一轮同步
- **WHEN** Modal 上的 SFT 岛启动后
- **THEN** 它连接 syncer、提交片段、接收全局参数，与 AWS 岛在同一轮次内完成同步，syncer 事件带上该岛的 learner id

#### Scenario: RL 岛使用固定镜像
- **WHEN** 用户以 `--training-mode rl` 启动 Modal 岛
- **THEN** 容器镜像的摘要与 `--rl-image` 指定的摘要一致，不一致时启动前报错

### Requirement: syncer 必须可从 Modal 到达
启动器 SHALL 在启动 Modal 岛前确认 syncer 地址是 Modal 容器可连接的公网地址或隧道地址；不可达时 MUST 报错并说明可用的两种办法（head 模式的公网 syncer，或本地模式的隧道）。

#### Scenario: 本地 controller 无隧道
- **WHEN** 用户以 `--controller local` 启动含 Modal 岛的 fleet，且 syncer 地址是私网地址
- **THEN** 命令在启动 Modal 容器前报错，提示改用 head 模式或提供隧道地址

### Requirement: 生命周期与失败处理
Modal 岛 SHALL 在 fleet 结束或 `yeto down` 时被终止；容器被 Modal 抢占时运行器 SHALL 按现有的岛重启策略自动重试，重试次数用尽后向 syncer 报告该岛退出。`yeto status` 和 `yeto logs` SHALL 能列出并读取 Modal 岛。

#### Scenario: 容器被抢占
- **WHEN** Modal 抢占了某岛的容器
- **THEN** 运行器以同一 learner id 重新启动容器，syncer 在配置的宽限期内继续等待该岛

#### Scenario: 停止 fleet
- **WHEN** 用户运行 `yeto down <cluster-prefix>`
- **THEN** 该 fleet 的 Modal 岛全部终止，不留下运行中的 Modal 函数

### Requirement: 手动接入模式
运行器 SHALL 同时可以脱离启动器独立使用：用户先以 `--external-learners N` 启动 fleet，再手动运行 Modal 运行器并传入打印出的 learner id 与 syncer 地址。

#### Scenario: 手动接入
- **WHEN** 用户以 `--external-learners 1` 启动 fleet，随后用启动日志打印的 join 参数运行 Modal 运行器
- **THEN** Modal 岛以该 learner id 加入，与自动路由启动的岛行为一致
