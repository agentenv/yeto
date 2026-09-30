# Design: ports 路径上的 Codex / 多 harness agentic rollout

## Context

动机见 `proposal.md`。现状依据 `/home/michael/work/infra-drafts/codex-harness-ab.html`（2026-09-30 调研；yeto `integ-decl`、fork `yeto/ports`=`e3a11ab3`、上游 `radixark/miles` main=`874b3e275`）。以下标“源码已确认”的内容在调研中核对过源码；标“未核实”的是推断。

- **源码已确认**：上游 Miles 已有以下组件，fork 在 `miles/rollout/session`、`generate_hub`、`generate_utils` 三个目录与上游零差异。
  - `agentic_tool_call.generate`：通过 `load_function(args.custom_agent_function_path)` 调用 agent，注册 `--custom-agent-function-path` 与 `--max-seq-len`；agent 返回的 dict 合并进 sample metadata。
  - Session Server：`POST /sessions`、`/sessions/{id}/v1/chat/completions`、`/sessions/{id}/v1/messages`、`/sessions/{id}/samples`。没有 `/v1/responses`。
  - TITO：`core.py:178` 要求 `output_token_logprobs`；`linear_trajectory.py:187` 校验前缀，只允许 `allowed_append_roles` 中的角色追加。v2 为树形。
- **源码已确认**：上游 `arguments.py:3240` 规定 `--use-session-server does not support --partial-rollout`。
- **源码已确认**：`--tito-allowed-append-roles` 在上游和 fork 中都不是 CLI 参数，由 `--tito-model` 对应 FixedTemplate 的 `allowed_append_roles` 决定。
- **源码已确认（yeto 内）**：ports 侧已有以下组件：
  - `ToolWaitBoard` / `async_tool_wait_scope` / `drain_blockers`（`yeto/rl/engine/tool_wait.py`）；
  - 示例 workload `yeto/rl/tool_wait_workload.py`；
  - `MilesRolloutPool` 的 load sample 与 3.3 drain probe（`miles_adapter/rollout.py:543–611, 845–863`）；
  - `tbench_outcome.py`（闭合字段集、域分隔 HMAC、密钥源校验）与 `trajectory_evidence.py:457–471`（写入证据时再次验签）。
- **未核实**：legacy `yeto_miles_secrlenv`（含 `_ResponsesBridge`、reward、`codex_harness_agent`）的源码不在 `integ-decl`，本机也没找到，行为只能从 `tests/test_secrlenv_codex_harness.py` 反推。`agentenv/miles` 副本中 `--tito-allowed-append-roles` 的来源同样未核实。
- MiMo 网关的调研要点是本 change 的设计输入：
  - 前缀哈希链、多 chain、回滚；
  - mask/logprob 对齐断言；
  - 每次生成记录权重版本；
  - 每条轨迹一个 Pod，删除以 404 为准；
  - 基础设施错误与答错分开；
  - 奖励前防篡改；
  - 默认拒绝出网。

## Goals / Non-Goals

**Goals**
- A 路径：ports + 上游 `agentic_tool_call` 端到端跑通 Codex Terminal-Bench rollout，不改 fork，奖励走签名契约，tool-wait 进入 1.7/3.3 计数。
- B 网关：一个协议网关同时服务 Codex（Responses）、chat 类 harness 和 Anthropic Messages 类 harness，正确性条件统一由 TITO 与网关断言保证。
- 把 reasoning 训练、长轨迹分段、沙箱代理的扩展点定成接口，后续 change 不必重做契约。

**Non-Goals**
- 不实现真实沙箱后端（Daytona / k8s Pod / 云），也不实现租约队列与远程 worker 池。
- 不实现 partial rollout、中断续跑、CompactionRL。
- 不修改 fork Miles/SGLang，不给上游提 PR。
- 不开放 Codex 的 `apply_patch` freeform 工具与并行工具调用（沿用 legacy 固定工具面 `terminal.exec` / `submit`，`parallel_tool_calls:false`）。

## Decisions

### D1. 先 A 后 B；A 的 bridge 就是 B 网关的雏形
A 把 `_ResponsesBridge` 以进程内库的形式入库（`yeto/rl/harness/gateway/`）。B 为同一份库加 HTTP 外壳，独立部署。两条路径共用同一套协议转换与断言代码，A 的测试直接成为 B 的回归集。
- 备选“直接做 B”：租约、取消、HMAC 边界等语义都是新设计，出错时无法定位是协议层还是分布式层的问题，所以不选。

### D2. A 路径的配置
- `agent.custom_agent_function_path` 与 `agent.agent_max_seq_len` 改为透传，分别对应 `--custom-agent-function-path` 与 `--max-seq-len`；只有 `custom_generate_function_path` 等于 `miles.rollout.generate_hub.agentic_tool_call.generate` 时才允许设置，否则 fail closed。
- `agent.tito_allowed_append_roles` 保持拒绝，理由改为“由 `--tito-model` 对应模板决定，不是 CLI 参数”。
- `use_session_server=true` 时，若同时请求 partial rollout，启动即失败；错误信息引用上游互斥规则。
- `config.py` 与 `entry.py` 属于 INFRA 共享文件（alignment.md §1），以接口请求 IR-1 的形式提出，由 INFRA owner 合入或授权本 change 修改。

### D3. agent 包与 preflight 入库
- legacy agent 包搬到 `yeto/rl/harness/codex/`，不进 fork，包括 `codex_harness_agent`、`codex_openenv_agent_function`、`codex_openenv_subprocess_agent_function`、reward 函数。源码若找不到，就按 `tests/test_secrlenv_codex_harness.py` 的行为重写，并标注“重写，非搬运”。
- 以下三项迁到 ports 启动路径，在模型分配前执行：
  - `_preflight_codex_harness`：签名锁定 Codex 0.145.0、二进制 / manifest / schema / 工具面 sha；
  - `_verify_live_codex_app_server_schema`；
  - `tbench_direct_preflight.validate_hmac_key_source`。
- 迁移完成后，legacy 同名函数改为转调新实现。legacy 的删除由 rl-engine-ports 负责。

### D4. tool-wait 与在途计数
- A：agent 函数内，每个“模型已返回、正在等下一次请求”的区间包一层 `async_tool_wait_scope(board_actor(learner_id), trajectory_id)`，粒度是每次工具执行，不是整条轨迹。bridge 在同一进程内，能精确切分这些区间。
- B：网关按会话调用 enter/exit。
- 三个计数一起上报给 1.7：
  - `tool_wait`：正在执行工具的轨迹；
  - `harness_in_flight`：已建会话、尚未收回 sample；
  - `env_live`：已 acquire、尚未确认销毁的沙箱。
- drain 条件扩展为“router 在途请求=0 ∧ tool_wait=0 ∧ env_live=0”。未知计数 fail closed，与现有 `drain_blockers` 一致。对 `drain_blockers` 签名的扩展属于 INFRA 文件，作为接口请求 IR-2。

### D5. 网关：三入口 → TITO，只追加约束与多 chain
- 三个入口：
  - `/v1/responses`（SSE，Codex）
  - `/v1/chat/completions`
  - `/v1/messages`
- 转换后统一调用 Session Server 的 chat 路由，由 TITO 拼接前缀。网关不自行 tokenize 模型输出。
- **只追加约束**：第 k 次请求的消息历史，必须等于第 k−1 次的历史加上上次模型输出（逐 token 一致），再追加 `allowed_append_roles` 中的角色消息。
- **前缀哈希链**：网关对每个 chain 维护 `h_k = H(h_{k-1} ‖ tokens_k)`。新请求按最长匹配前缀定位 chain。有三种结果：
  1. 完全延续 → 追加到原 chain；
  2. 前缀在某个历史点分叉，典型原因是 harness 重试或回滚 → 回滚到分叉点后，开一个新 chain 分支；分叉点之后的原分支仍然是有效样本，是否用于训练由 D5a 决定；
  3. 与任何 chain 都不匹配，典型原因是历史被压缩或改写、模板丢弃历史 think 段 → 另起新 chain，记录 `chain_break_reason`，计入 `tito_chain_breaks{reason}`。
  - 不允许把不匹配的请求“修补”后继续沿用旧 chain；需要修补的情况一律视为 `tito_session_mismatch`。
- **D5a. 多 chain 的训练归属**：一条轨迹可以产出多个 sample（每个 chain 一个）。每个 chain 的 token 与 mask 自洽。奖励归属见 D9。默认所有 chain 都参与训练并共享轨迹奖励。计量按 chain 数与 chain 长度分布进行。
- **对齐断言**（fail closed，轨迹作废，不给 0 奖励）：
  - 每个 chain 的 `len(tokens)==len(loss_mask)==len(rollout_logprobs 对齐位)`；
  - mask=1 的位置必须全部来自 SGLang 生成，logprob 非空；
  - mask=0 的位置必须全部来自模板或工具编码。
- **权重版本**：每次生成都记录 `policy_version`（取 Publisher 的 policy token / hash）。在 age 0 配置下，断言同一 chain 内所有生成的 `policy_version` 等于该 rollout 的目标版本；不一致则标 `policy_age_violation`，轨迹作废。这一条同时服务于 X5 的 drain：drain 期间旧路由的会话必须在旧版本上完成。
- **harness 兼容边界**：接入条件是“指向兼容端点，且满足只追加约束，或能接受分 chain 的计量代价”。harness 覆盖的采样参数（温度、max_tokens、logprobs）以网关配置为准，并拒绝覆盖已签名字段（沿用 legacy）。Responses 侧只支持 legacy 已覆盖的子集，其余一律 fail closed，不静默忽略：
  - function 工具；
  - `store:false` 全量回放；
  - `reasoning.summary:"none"` 且带 `reasoning_content` 往返；
  - `parallel_tool_calls:false`；
  - 过滤 `update_plan`；
  - 禁用自动 compaction，或使用 legacy 的“同会话可训练 compaction”。

### D6. reasoning token 与模板一致性
- 每轮生成的 reasoning token 属于生成段，mask=1，参与训练。
- 回放历史 reasoning 时，由 `--tito-model` 对应模板决定是否保留历史 think 段。每个受支持的 tito_model 必须附带一份模板一致性声明（`keeps_history_reasoning: bool`），并用 fork 中的 `chat_template_verify.py`、`session_verify_runner.py` 做离线验证。
- `keeps_history_reasoning=false`（Qwen3 类）时，第 k+1 轮的前缀不可能逐 token 等于第 k 轮的生成，按 D5 情况 3 另起 chain，`chain_break_reason=template_drops_reasoning`。这是预期行为，但必须计量。每个 tito_model 首次上线时，都要在 GPU 冒烟中报告该比例。
- 备选“改写模板保留 think 段”：会让训练模板与推理模板不一致，偏离模型官方行为，所以不选。

### D7. 奖励契约与信任分层
- 沿用 `yeto/rl/tbench_outcome.py`：
  - 闭合字段集；
  - `yeto-tbench-outcome-v1\0` 域分隔 + 规范 JSON 的 HMAC-SHA256；
  - 二值奖励，要求 `passed == (reward==1.0)`；
  - 密钥来自 `TBENCH_REWARD_HMAC_KEY(_FILE)`，文件为 0400/0600、非符号链接，长度 32–4096 字节。
- **可信层**：持有密钥，运行 verifier（官方 `tests/test.sh` 或 OpenEnv evaluate），签名。A 中是隔离的 OpenEnv 子进程；B 中是 SandboxBroker 的 `verify`，由训练侧或 broker 的可信组件执行，密钥不下发到任务容器。
- **不可信层**：Codex 进程与任务容器。它们不能读取密钥，不能访问推理端点以外的内部服务，也不能改写 verifier 资产（verifier 资产在 verify 前才注入，参考 `TB2_WITHHOLD_TESTS`）。
- 验签点有三个：奖励函数、父进程 `verified_outcome`、`trajectory_evidence`。三处都保持验签，任何一处失败都拒收。
- 失败分类：
  - 基础设施错误（沙箱创建失败、租约超时、verifier 未运行）：不签名，sample 标 `aborted`，不产生训练数据，也不给 0 奖励；
  - 策略边界（turn/seq 上限、超时截断）：签名，reward=0，status 为 `timeout`/`max_turns`/`max_seq_len`；
  - 答错：签名，reward=0，status=`completed`。

### D8. 长轨迹：截断丢弃 + 分段预留
- 本 change 的行为：轨迹超过 wall-clock 上限或 `--max-seq-len` 时按策略边界处理（D7），不续跑。drain 等待在途轨迹自然结束；超过 drain 超时则取消切换，不重放外部副作用（与 3.3 X5 一致）。
- 预留的数据结构：
  - 每个 chain 的 sample metadata 带 `trajectory_id`、`segment_id`（本 change 恒为 0）、`segment_boundary_reason`（本 change 恒为 null）；
  - 网关的上下文拼接经过 `ContextProvider` 接口，默认实现是恒等映射，留给 CompactionRL 在段间替换上下文；
  - 奖励归属字段 `reward_scope ∈ {trajectory, segment}`，本 change 只接受 `trajectory`。
- partial / 续跑受上游互斥约束，并依赖 infra 的异步契约，列为后续 change。

### D9. 沙箱代理只定接口
- `SandboxBroker`（Protocol）：
  - `acquire(env_id, trajectory_id, deadline) -> Lease`
  - `heartbeat(lease)`
  - `exec(lease, argv, timeout)`
  - `copy_in/copy_out(lease, …)`
  - `verify(lease) -> SignedOutcome | InfraError`
  - `destroy(lease)`
  - `describe(lease) -> {live|gone}`
- 语义：
  - 每条轨迹独占一个 lease，对应一个 Pod 或容器；
  - 必须等 `describe` 返回 gone（对 k8s 即 404）才算销毁确认，`env_live` 此时才减一；
  - 租约心跳超时，视为基础设施错误；
  - 网络策略默认拒绝出网，只允许访问网关的会话端点。
- 环境注册表记录 `env_id → {image_digest, cpu/mem/disk, timeout, verifier_ref, network_policy, withhold_assets}`，镜像必须按 digest 锁定。
- 本 change 只交付 Protocol、注册表 schema 校验、一个本地 fake 实现（供 CPU 测试与 drain 计数测试使用），以及 A 路径适配层：把现有 OpenEnv/Daytona 调用包装成该接口，但不改变其行为。

### D10. TIS 默认开启
agentic profile 默认开启截断重要性采样，吸收引擎的数值差异。TIS 无法修复 token 错位，token 错位由 D5 的断言处理。关闭 TIS 需要显式配置，并写入 profile hash。

### D11. 对 INFRA 的接口请求（本 change 不直接修改）
- **IR-1**：`miles_adapter/config.py` / `entry.py`：放开 D2 的两个字段，加入 partial rollout 互斥检查。
- **IR-2**：`tool_wait.drain_blockers` 与 `MilesRolloutPool` 的 drain probe 增加 `harness_in_flight`、`env_live`；1.7 load sample 增加这两个字段，以及 `tito_session_mismatch`、`tito_chain_breaks{reason}`、`policy_age_violation`。
- **IR-3**：driver 在 age 0 配置下，把目标 `policy_version` 传给 rollout 调用（供 D5 断言）。
- **IR-4**：1.7 的指标 schema 登记上述计数的 profile/epoch 标签。

## Risks / Trade-offs

- [legacy `yeto_miles_secrlenv` 源码找不到] → 按测试重写，把 `tests/test_secrlenv_codex_harness.py` 全部用例作为验收；重写部分在 progress 中标明。
- [Qwen3 类模板导致每轮都断 chain，sample 数暴涨、上下文重复计算] → 计量并在冒烟中报告；比例超过阈值时，该 tito_model 不认证用于 agentic profile。
- [harness 静默改写历史（compaction、tool call JSON 重排）] → 网关 fail closed 并计数；每个新 harness 上线前跑一次离线一致性回放。
- [B 网关成为单点、worker 绕过网关直连推理] → B 在本 change 中只做进程外壳与 CPU 测试；网络隔离随沙箱实现 change 落地。
- [age 0 下长尾轨迹拖住整轮] → 本 change 只截断；依赖 rl-infra-spec 的异步契约之后再做续跑。
- [Ray worker 镜像需要带 Codex 二进制与容器权限] → A 冒烟沿用 legacy 镜像构建方式。镜像使用官方 radixark/miles 镜像加 yeto 层，不使用私有 MILES_IMAGE。

## Migration Plan

1. A 路径以 `--rl-engine ports` 加 agentic profile 开启，legacy 保持不变作为对照。
2. A 冒烟通过后，legacy 的 Codex preflight 改为转调新实现。
3. 网关 HTTP 外壳上线前，A 与 B 共用同一库；回滚方式是切回进程内模式。
4. legacy 删除随 rl-engine-ports 的最后阶段进行。

## Open Questions

- 冒烟使用的 Terminal-Bench 任务子集与数量（不影响契约，GPU 任务执行前确定）。
