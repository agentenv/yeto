# rlf-codex-rl Specification

## Purpose
把 rlenvfactory 已验收的 cyber 任务安全地转换为 Codex agentic RL rollout，并产出可用于 Yeto 训练的可信 reward、成功元数据和可审计生命周期证据。

## Requirements

### Requirement: 任务包必须可验证且身份完整
系统 SHALL 只接受带有 rlenvfactory commit、任务包 digest、镜像 digest、grader identity 和验收结果的任务包。对当前 CTF deliverable，系统 SHALL 验证 `public/`、`task.yaml`、`MANIFEST.json` 的存在及 `task.yaml` 的 `task_id`、`domain`、`target_service`、`target_port`、`source_digest`、`grader_mode` 字段；导入 SHALL 在运行前校验 schema、静态约束和允许的 grader mode。缺少或不匹配的身份 SHALL 被拒绝。

#### Scenario: 已验收任务导入
- **WHEN** 任务包身份、digest 和验收记录均匹配
- **THEN** provider 创建 episode manifest 并记录这些身份

#### Scenario: 任务包被篡改
- **WHEN** 任一 digest 或 grader identity 不匹配
- **THEN** episode 在启动前失败并分类为 admission_error

### Requirement: 每个 rollout 使用隔离的 RLF episode
provider SHALL 为每个 trajectory 创建独立的任务实例、随机 secret、网络和资源配额；Codex worker 只能访问 execution/submit 接口，不得访问 trusted grader 或私有 solution。episode SHALL 在正常、超时和异常路径执行幂等清理并记录结果。

#### Scenario: Codex 使用任务环境
- **WHEN** Codex 请求 terminal execution 或提交答案
- **THEN** 请求路由到该 trajectory 的隔离 episode，并返回受限结果

#### Scenario: 清理失败
- **WHEN** episode teardown 不能确认容器、网络或 sandbox 已释放
- **THEN** rollout 分类为 cleanup_error，不得静默作为普通失败样本

### Requirement: Codex rollout 必须保留训练所需轨迹
Codex harness SHALL 保留 prompt、response、tokens、logprobs、loss mask、segments、episode identity、task identity 和 behavior policy version，并输出符合中立 `Trajectory`/`RewardResult` 接口的数据。

#### Scenario: 完整轨迹提交
- **WHEN** Codex 完成或超时
- **THEN** 轨迹包含可重放的 episode/task identity 及策略版本，且能被 Miles 与 generic verl adapter 消费

### Requirement: 可信判分与超时语义必须分离
trusted side SHALL 在 worker 结束后调用 RLF 原生 grader 并写入 reward、passed、success 和 grader evidence。Codex timeout SHALL 不自动跳过仍可判分的环境状态；判分结果、timeout 状态和 cleanup evidence SHALL 分开记录。

#### Scenario: 超时但状态可判分
- **WHEN** Codex 超时且 target/grader 仍可访问
- **THEN** trusted grader 执行并记录真实判分结果，同时 episode status 为 timeout

#### Scenario: 环境故障
- **WHEN** grader 或环境因基础设施故障不可用
- **THEN** reward result 分类为 infrastructure_error，并拒绝进入正常训练数据

### Requirement: success 元数据必须贯穿 reward adapter
成功定义 SHALL 由 RLF grader 结果产生，写入 `metadata["success"]`，并通过 Miles 与 generic verl reward path 传递；声明接口 SHALL 标记该 reward 写入 success。只返回浮点数而丢弃 success 的路径不得作为生产 RLF 训练路径。

#### Scenario: 成功样本进入 learner
- **WHEN** grader 返回 passed=true
- **THEN** reward 与 `metadata.success=true` 同时到达 learner/data acceptance

### Requirement: Modal 运行必须可验证
系统 SHALL 支持 CPU fake-provider 对照、Modal VM Sandbox 的 Docker/RLF environment smoke 和 Modal GPU 上的小规模 Codex RL training gate。Modal environment smoke 必须验证 readiness、secret 传递、隔离和 teardown；GPU gate 必须保存 rollout、grader、reward、policy update 和资源清理证据。

#### Scenario: VM Sandbox 启动 RLF Docker 生命周期
- **WHEN** provider 在 Modal VM Sandbox 启动
- **THEN** dockerd readiness 通过后可创建 RLF target/grader 所需容器，并在结束后确认清理

#### Scenario: GPU gate 失败
- **WHEN** 任一身份、success metadata、reward、policy update 或 cleanup 证据缺失
- **THEN** gate 失败且不得宣称端到端训练成功

### Requirement: Cyber rollout 必须有上下文预算和截断证据
Qwen3-30B-A3B 的 cyber baseline SHALL 使用 thinking mode、32K `max_seq_len`、每轮约 1024 个生成 token、最多 48 turns、单次工具输出上限 8192 bytes 和 20–30 分钟 episode deadline。系统 SHALL 记录 `max_seq_len_hit`、`end_reason`、turns、总模型 token、tool calls、工具输出字节数、提交发生的 turn 和 timeout。工具输出截断必须显式标记，不能伪装成完整环境结果。

#### Scenario: 基线在拿到 flag 前耗尽窗口
- **WHEN** baseline 的 `max_seq_len_hit` 发生且尚未提交 flag
- **THEN** trace 标记 `max_seq_len`，样本不被当作普通 unsolved loss 静默混入，并触发 compaction 评估

### Requirement: Compaction 必须保持逻辑 trajectory 身份
当 baseline 显示单窗口不足时，系统 SHALL 支持 24576 token 的 compaction window、4096–8192 token 的 summary/generation reserve、最多 2–3 次 compaction；所有 execution/summary segments SHALL 共享 trajectory、task、episode、policy version 和 trusted reward 身份。Qwen3 的 context budget 必须显式配置，不能依赖未注册模型的默认值。

#### Scenario: 多窗口完成任务
- **WHEN** Codex 在 compaction 后继续调用工具并提交 flag
- **THEN** segments 被组装为一个逻辑 trajectory，trusted grader 只判分一次，且训练数据保留每个 segment 的身份与 loss mask

### Requirement: RL gate 必须证明 reward 和 GAE 有效
Modal GPU RL gate SHALL 在 learner 更新前后保存 reward 总和、reward 标准差、success 数量、advantage 绝对值总和和 policy update 证据。gate MUST reject a run where all rewards are zero/constant or `advantage_abs_sum == 0`，并记录对应的 trace 和 group reward 分布。

#### Scenario: 有成功与失败样本
- **WHEN** 一个 rollout group 同时包含至少一个成功和一个失败样本
- **THEN** reward 标准差大于零、GAE/advantage 绝对值总和大于零，且 policy update 被记录

#### Scenario: 所有样本失败
- **WHEN** 一个 group 的 reward 全为零
- **THEN** RL gate 失败，不得把该轮宣称为有效训练

### Requirement: 第一阶段范围固定为 Codex
第一阶段 SHALL 只注册 Codex harness；SWE-bench agent sandbox 和 RLF 其他 Solver harness SHALL 不成为启动或验收前置条件，且现有 TB2 production app 默认行为保持不变。

#### Scenario: 非 Codex harness 被请求
- **WHEN** 配置选择 SWE 或其他 Solver
- **THEN** 明确返回 unsupported_in_phase_one，而不是回退到 Codex
