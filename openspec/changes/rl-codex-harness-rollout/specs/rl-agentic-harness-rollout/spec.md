# Spec Delta

## Purpose

定义 ports 引擎路径上 agentic harness rollout 的行为契约。harness 可以是 Codex，也可以是其他能指向 OpenAI Chat、Responses 或 Anthropic Messages 兼容端点的 harness。契约覆盖配置透传、协议网关与 TITO 一致性、reasoning 训练、奖励签名、长轨迹处理，以及对 drain 的计数上报。

## ADDED Requirements

### Requirement: ports 透传 agent 函数配置
ports 引擎 SHALL 接受 `agent.custom_agent_function_path` 与 `agent.agent_max_seq_len`，并分别映射到 Miles 的 `--custom-agent-function-path` 与 `--max-seq-len`。前提是 `agent.custom_generate_function_path` 为上游 `agentic_tool_call.generate`；不满足时 MUST 启动失败。`agent.tito_allowed_append_roles` MUST 继续被拒绝，拒绝理由 MUST 说明追加角色由 `--tito-model` 对应模板决定。启用 session server 时同时请求 partial rollout，MUST 启动失败，并说明两者互斥。

#### Scenario: A 路径配置可启动
- **WHEN** 配置设置 agentic_tool_call 作为 custom generate 函数，并提供 agent 函数路径与 max_seq_len
- **THEN** 生成的 Miles 参数包含这两个参数，不修改 fork Miles

#### Scenario: 缺少 agentic_tool_call 时拒绝 agent 函数
- **WHEN** 设置了 agent 函数路径，但 custom generate 函数不是 agentic_tool_call
- **THEN** 启动失败，错误中指明字段与原因

#### Scenario: session server 与 partial rollout 互斥
- **WHEN** 同时启用 session server 与 partial rollout
- **THEN** 启动失败，不以降级方式运行

### Requirement: harness 启动前完整性校验
使用签名锁定的 Codex agent 时，系统 SHALL 在分配模型资源前完成以下校验，任一项不符 MUST 启动失败：
- Codex 版本、二进制与包 manifest 摘要；
- app-server schema；
- 基础指令摘要；
- 工具面 schema；
- 奖励密钥来源。

#### Scenario: 工具面漂移
- **WHEN** 运行环境中的 Codex 工具面 schema 摘要与签名值不同
- **THEN** 启动在模型分配前失败，并报告不一致的项

#### Scenario: 密钥来源不合规
- **WHEN** 奖励密钥文件权限宽于 0600，或是符号链接，或长度超出 32–4096 字节
- **THEN** 启动失败

### Requirement: 协议网关统一到 TITO
网关 SHALL 提供 Responses、Chat Completions、Anthropic Messages 三种入口，所有生成都 MUST 经过 TITO：
- 模型输出 token 与 behavior logprob 来自推理引擎生成，MUST NOT 由网关重新 tokenize；
- 只有 harness 追加的消息允许由模板增量编码。

网关 MUST NOT 接受 harness 覆盖以下已签名的采样字段：温度、最大长度、logprob 请求。网关遇到不支持的请求形态 MUST 拒绝，MUST NOT 静默忽略。

#### Scenario: Codex Responses 往返
- **WHEN** Codex 以 SSE 调用 Responses 入口，请求包含 reasoning 与 function 工具调用
- **THEN** 网关返回合法的 Responses 事件序列，对应 sample 的 token 与 logprob 来自引擎生成

#### Scenario: 不支持的 Responses 特性
- **WHEN** 请求启用了并行工具调用、托管工具或增量 previous_response_id 引用
- **THEN** 网关拒绝该请求，轨迹按基础设施错误处理

#### Scenario: harness 覆盖签名字段
- **WHEN** 请求携带与配置不同的温度
- **THEN** 网关拒绝该请求

### Requirement: 只追加约束、多 chain 与回滚
每条轨迹的生成 SHALL 组织为一个或多个 chain。网关 MUST 按前缀哈希定位请求所属的 chain：
- 新请求逐 token 延续某个 chain 时，追加到该 chain；
- 在某个 chain 的历史点分叉时，回滚到分叉点，开一个新分支 chain；
- 与所有 chain 都不匹配时，另起新 chain，并记录断链原因。

网关 MUST NOT 修补不匹配的历史后继续沿用旧 chain。断链次数 MUST 按原因计量。

#### Scenario: 正常多轮
- **WHEN** harness 每轮都原样回放上一轮输出，并追加工具结果
- **THEN** 整条轨迹只有一个 chain，断链计数为 0

#### Scenario: harness 重试导致分叉
- **WHEN** harness 丢弃最后一次模型输出，并以相同前缀重新请求
- **THEN** 网关回滚到分叉点，产生新分支 chain；原分支保留为有效 sample

#### Scenario: 历史被改写
- **WHEN** harness 压缩或改写了早期消息
- **THEN** 网关另起 chain，并以对应原因计入断链计数

### Requirement: mask、logprob 与权重版本断言
每个 chain SHALL 满足以下条件：
- token、loss mask 与 rollout logprob 等长对齐；
- mask=1 的位置全部来自引擎生成，且 logprob 非空；
- mask=0 的位置全部来自模板或工具编码。

每次生成 MUST 记录生成时的策略版本。在 policy age 0 配置下，同一轨迹的所有生成 MUST 使用该 rollout 的目标版本。任一断言失败，轨迹 MUST 作废，MUST NOT 产生训练数据，也 MUST NOT 记为 0 奖励；失败 MUST 计入指标。

#### Scenario: logprob 缺失
- **WHEN** 某个生成段缺少 logprob
- **THEN** 该轨迹作废，计入 mismatch 指标

#### Scenario: age 0 下版本漂移
- **WHEN** 轨迹中途发生权重发布，后续生成使用了新版本
- **THEN** 该轨迹作废，计入 policy age 违规指标

### Requirement: reasoning token 训练与模板一致性
每轮生成的 reasoning token SHALL 作为生成段参与训练（mask=1）。每个用于 agentic profile 的 TITO 模型 MUST 声明历史回放时是否保留 reasoning 段，并通过离线模板一致性测试。模板丢弃历史 reasoning 段时，后续轮次 MUST 另起 chain，并以模板原因计量，MUST NOT 被当作 mismatch 失败。

#### Scenario: 保留历史 reasoning 的模板
- **WHEN** 模板保留历史 think 段，harness 原样回放 reasoning
- **THEN** 多轮保持单 chain，reasoning token 的 mask 为 1

#### Scenario: 丢弃历史 reasoning 的模板
- **WHEN** 使用 Qwen3 类会丢弃历史 think 段的模板
- **THEN** 每轮另起 chain，断链原因为模板丢弃 reasoning，比例出现在指标中

### Requirement: 签名奖励与信任分层
agentic 奖励 SHALL 采用签名 outcome 契约：
- 字段集合闭合；
- 奖励严格二值，且 passed 与奖励一致；
- 签名为域分隔的 HMAC。

奖励函数与轨迹证据写入方 MUST 验签，失败即拒收。密钥与 verifier MUST 位于可信层；harness 进程与任务容器 MUST NOT 能读到密钥或 verifier 资产。结果分三类处理：
- 基础设施错误 MUST 不签名，sample 标为中止，不产生训练数据；
- 策略边界（超时、轮数或长度上限）MUST 签名，奖励为 0；
- 答错 MUST 签名，奖励为 0。

#### Scenario: 篡改奖励
- **WHEN** outcome 在回传途中被改为 passed=true
- **THEN** 验签失败，样本被拒收

#### Scenario: 基础设施错误与答错分开
- **WHEN** 沙箱创建失败导致 verifier 未运行
- **THEN** sample 标为中止，不计入奖励 0，也不进入训练

### Requirement: 长轨迹截断与分段预留
超过时长上限或长度上限的轨迹 SHALL 按策略边界截断并结束，本能力 MUST NOT 续跑。每个 sample MUST 带轨迹标识、段标识与段边界原因；在本能力中，段标识恒为 0，边界原因为空。奖励归属范围 MUST 显式声明，本能力只接受整条轨迹归属。段间上下文 MUST 经可替换的上下文提供者拼接，默认为恒等。

#### Scenario: 超时截断
- **WHEN** 轨迹超出 wall-clock 上限
- **THEN** 以超时状态签名 outcome，奖励为 0，不续跑

#### Scenario: 请求按段归属奖励
- **WHEN** 配置声明奖励按段归属
- **THEN** 启动失败，并提示该能力尚未提供

### Requirement: 工具等待与在途计数上报
agentic rollout SHALL 按岛上报以下计数，粒度为每次工具执行：
- 正在执行工具的轨迹数；
- 已建会话、尚未回收 sample 的轨迹数；
- 已获取、尚未确认销毁的环境数。

drain 判定 MUST 要求引擎在途请求、工具等待与存活环境三者都为 0。任一计数未知时 MUST 视为未排空。

#### Scenario: 工具执行中不切换路由
- **WHEN** 引擎在途请求为 0，但有轨迹正在执行工具
- **THEN** drain 未完成，旧路由保留

#### Scenario: 环境销毁未确认
- **WHEN** 轨迹已结束，但其环境尚未确认销毁
- **THEN** 存活环境计数大于 0，drain 未完成

### Requirement: TIS 默认开启
agentic profile SHALL 默认开启截断重要性采样。关闭 MUST 显式配置，并反映在 profile 哈希中。

#### Scenario: 默认配置
- **WHEN** agentic profile 未指定 TIS 设置
- **THEN** 训练使用截断重要性采样，指标中可见其比值统计
