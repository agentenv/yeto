# Spec Delta

## Purpose

规定 RL learner 与 head 侧写入事件磁带的训练指标与运行期事件字段契约，使 ports 引擎路径下训练过程、岛健康、资源与成本可被 dashboard 与 W&B 归档完整观测。

## ADDED Requirements

### Requirement: ports 路径每轮训练指标完整
在 ports 引擎路径上，每个训练轮的 `rl_local_round` SHALL 填写 loss、pg_loss、mean_kl、ess_ratio、lr、train_step 中引擎可提供的值；引擎确实无法提供的字段 MUST 为 null，MUST NOT 以 0 或占位值代替。

#### Scenario: Miles 报告 loss dict
- **WHEN** 一轮训练中 Miles 每个优化器步都返回含 loss 与 pg_loss 的标量字典
- **THEN** 该轮 `rl_local_round` 的 loss 与 pg_loss 为这些步的均值，train_step 为非 null 的优化器步序号

#### Scenario: 引擎不报告 KL
- **WHEN** loss dict 中不存在任何 KL 类键
- **THEN** mean_kl 为 null，事件其余字段正常写出

### Requirement: rl_round_trained 携带轮级训练统计
`rl_round_trained` SHALL 增加以下可选字段：`train_step`；`train_metrics`（该轮全部优化器步 Miles loss dict 各标量键的均值，键名原样保留）；`adv_mean`、`adv_std`；`resp_len_mean`、`resp_len_p95`；`truncated_frac`；`reward_p10`、`reward_p50`、`reward_p90`；`tok_per_s`；`step_seconds`。已有字段语义 MUST 保持不变。

#### Scenario: loss dict 的全部标量进入磁带
- **WHEN** 某轮 loss dict 含 kl_loss、entropy_loss、entropy、ppo_kl
- **THEN** 该轮 `rl_round_trained.train_metrics` 包含这四个键及其轮均值

#### Scenario: 旧读者兼容
- **WHEN** 旧版本磁带读取工具读取含新字段的记录
- **THEN** 读取不报错，新字段被忽略

### Requirement: W&B 训练曲线使用有效步轴
W&B 转发 SHALL 为每条训练记录提供非 null 的 `train/step`；ports 路径下 `train/*` 指标（含 grad_norm）MUST 出现在 W&B 中。无法确定步序号的记录 MUST 以日志说明丢弃原因，不得静默丢弃。

#### Scenario: ports 路径运行
- **WHEN** 以 ports 引擎路径完成 3 轮训练并开启 W&B
- **THEN** W&B 中 `train/grad_norm` 与 `train/loss` 各有 3 个点，横轴为 `train/step`

### Requirement: 岛心跳事件
每个 learner island SHALL 在运行期间约每 30 秒写一条 `rl_heartbeat`，包含当前阶段、rollout_id、policy_version 与进程运行时长；心跳写入 MUST NOT 阻塞训练循环。

#### Scenario: 长时间训练阶段
- **WHEN** 单个 train 阶段持续 5 分钟
- **THEN** 期间磁带出现约 10 条 phase 为 train 的 `rl_heartbeat`

### Requirement: GPU 资源采样事件
每个 learner island SHALL 按可配置周期（30–60 秒）写 `rl_resource_sample`，含每张 GPU 的 index、uuid、利用率、显存已用/总量与功耗。采样源不可用时 MUST 只写一条 `available: false` 的记录并停止采样，训练继续。

#### Scenario: 无 NVML 的环境
- **WHEN** 岛进程无法加载 NVML
- **THEN** 磁带中有且仅有一条 `rl_resource_sample`，其 `available` 为 false

### Requirement: head 侧 fleet 事件与成本心跳
head SHALL 在运行目录写 `fleet.jsonl`，记录每个岛的 launch/ready/lost/stop（含 cloud、region、GPU 型号与数量、价目 key），并每 5 分钟写一条 `cost_tick`（各岛单价、累计墙钟、累计估算金额、预算上限）。金额字段 MUST 标注为估算。

#### Scenario: 岛失联
- **WHEN** FleetController 判定某岛失联
- **THEN** `fleet.jsonl` 出现该岛的 `island_lost` 记录，且其后的 `cost_tick` 仍计入该岛至 stop 为止的墙钟

### Requirement: 重配置与 cell 事件预留
`rl_cell_snapshot`（cell id、角色、GPU 数、状态）与 `rl_reconfig_phase`（txn id、阶段、结果）的字段 SHALL 与 rl-infra-spec 的 cell 声明及 E1 事务定义一致；在该定义验收前，发射端 MUST NOT 写入这两类事件。

#### Scenario: 事务定义未验收
- **WHEN** rl-infra-spec 任务 3.2 尚未验收
- **THEN** 磁带中不存在 `rl_reconfig_phase` 记录
