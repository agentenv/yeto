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

---

## 修订提案（阶段 1 检查点，2026-10-01；主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4；下文各段以 R-* 为准，覆盖对应 Decisions）

依据与查找记录见 `/home/michael/work/infra-drafts/CODEX-PROGRESS.md`。以下每段均标"待批准"。

### R-CTX. 更正 Context 中的"未核实"（主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4）
- legacy `yeto_miles_secrlenv` 源码**已找到**：yeto 仓库提交 `5bfc011`（亦即 merge `e7e3066` 的第一父 `95dd529`）中的 `yeto_miles_secrlenv/{__init__,agent,client,codex_harness_agent,generate,reward}.py`；三份文件的 sha256 与 `yeto/rl/__init__.py` 当前 pin 的 `SECRLENV_AGENT_SHA256` / `SECRLENV_GENERATE_SHA256` / `CODEX_HARNESS_AGENT_SHA256` 逐字节相同。该目录在 merge `e7e3066`（2026-08-27）被静默丢弃，现有 `tests/test_secrlenv_codex_harness.py` 与之配套（仅多一处 importorskip）。D3 与 Risks 中"找不到则重写"的分支作废，改为"搬运并标注出处"。
- `codex_openenv_agent_function.py` / `codex_openenv_subprocess_agent_function.py` / `codex_openenv_agent_worker.py`（legacy 自 `<miles_root>/examples/experimental/openenv/` 加载）在本机所有 git ref、tar/bundle、pip 缓存中**均不存在**，fork pin `e3a11ab3` 的该目录也没有；来源应为 `agentenv/miles` 私有副本。此三模块按 `yeto/rl/learner.py:804–835`、`yeto/rl/tbench_direct_preflight.py:60–75`、`tests/test_rl_codex_schema.py` 约定的接口重写（标"重写"），除非用户提供副本。
- pin 镜像内 site-packages 未能检查（docker socket 无权限）。

### R-D5a. 多 chain 的训练归属（替换 D5a；主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4）

**真实 trace（按上游 `agentic_tool_call` + 上游 session server + legacy bridge 的实际代码路径构造；任务 `fix-git`，3 轮）**

| 步 | 生成事件 / 系统动作 | chain / session | 训练数据 |
|---|---|---|---|
| E0 | `agentic_tool_call.generate`：tracer `POST /sessions` → 会话 S1；调用 `codex_harness_agent.run(…, metadata)`；bridge 起 `/v1/responses`，`_AppServerDriver` 启动 Codex 0.145.0，发送任务 prompt | S1 创建 | — |
| E1 | Codex → `POST /v1/responses`（`store:false`，input=[user prompt]）。bridge 校验初始历史，构造 M1=[system BASE_INSTRUCTIONS, user prompt] → `POST S1/v1/chat/completions`（tools=`terminal.exec`/`submit`，logprobs）。session server：stored 空 → 模板渲染 P1 → SGLang 生成 **G1**（reasoning + `terminal.exec("git status")`），返回 `output_token_logprobs`、`weight_version`；record r1={input_ids=P1, output=G1}；`trajectory_token_ids`=P1‖G1 | S1 | G1：mask=1，logprob 有 |
| E2 | bridge 把 G1 译为 Responses `reasoning` + `function_call`，SSE 返回；`expected_input` := history+output。Codex 在沙箱执行工具（本区间进入 `async_tool_wait_scope`）；Codex → 第 2 次请求 input = 上次 input + output items + `function_call_output`。bridge：`history == expected_input` 校验通过；M2 = M1 + assistant(G1) + tool(result)。session server：stored M1+assistant 是 M2 前缀，追加角色 `tool` ∈ `allowed_append_roles`；**TITO 复用 P1‖G1 的 token，不重新 tokenize**，只渲染 tool 段 T2；生成 **G2**；r2={input_ids=P1‖G1‖T2, output=G2} | S1 | T2：mask=0；G2：mask=1 |
| E3 | 同 E2，G3 = `submit`；r3；bridge 置 terminal（之后任何采样 → 409） | S1 | T3：mask=0；G3：mask=1 |
| E4 | driver 关闭 Codex（进程组回收、isolated HOME 清理）；可信层运行 verifier（`tests/test.sh`）→ `tbench_outcome` 字段集 {task_id, episode_id, status=completed, passed, reward∈{0,1}} + HMAC 写入 metadata | — | — |
| E5 | `agentic_tool_call` 收集：`GET S1/samples` → `compute_samples_from_openai_records([r1,r2,r3], accumulated=P1‖G1‖T2‖G2‖T3‖G3)`：每 record 一个 per-turn Sample（`loss_mask=[1]*len(G_i)`、`rollout_log_probs`、`weight_versions` span），断言各轮 output 与累计序列对齐；`truncate_samples_by_total_tokens(max_seq_len)` 在 turn 边界截断；`merge_samples` → **一条 sample**：tokens=P1‖G1‖T2‖G2‖T3‖G3，mask=[0…0,1…1,0…0,1…1,0…0,1…1]，logprob 与 mask=1 位逐位对齐 | S1 → sample A | A |
| E6 | `reward_func` → `_verified_outcome` 验签 → reward；`check_group` 认证组；`train_data_conversion`：按 `group_index` 分组、按 `rollout_id` 识别同一 rollout 的兄弟段，兄弟奖励必须相等，**基线对每个 rollout 只计一次** | — | A 带 reward r |

**一次执行、上下文重组产生的片段（D5 情况 3）**：以 legacy compaction 为例，窗口 0 第 k 轮后，bridge 在 S1 追加 summary 指令采样 **Gs**（record r_s，mask=1，属于 chain 1），然后把下一请求的消息重建为 M' = [system, user(resume+summary), 保留的原子步]。
- **源码确认的不兼容**：legacy 依赖 `X-Miles-Compaction-*` 请求头让 session server 开新窗口；fork pin `e3a11ab3` 的 session server 不认识这些头（`git grep` 为空）。在 pin 上 M' 的 stored 前缀只匹配到 `system`，匹配前缀内无生成 checkpoint → `linear_trajectory._rollback_to_checkpoint(-1)` **静默丢弃 r1…r_s**。因此：
  - 首批 `YETO_CODEX_COMPACTION_ENABLED` 必须关闭，preflight 发现开启即 fail closed；超出 `max_seq_len` 按 D7/D8 策略边界处理。
  - 网关的多 chain 实现为**另起 session**（`POST /sessions` → S2），绝不对旧 session 回滚；S1 收集为 sample A（G1…Gk, Gs），S2 收集为 sample B（重建上下文为 prompt，mask=0；G_{k+1}… mask=1）。
- **归属规则**：A、B 属于同一 `trajectory_id`，设置相同 `group_index` 与 `rollout_id`（兄弟段），携带同一份签名 outcome → 共享最终奖励；每个生成事件恰好出现在一条 sample 的 mask=1 区间中（重建上下文里的历史文本是 prompt，mask=0），**不重复训练**；`train_data_conversion` 对该 rollout 只计一次基线，**不冒充独立 GRPO 样本**。metadata 增加 `chain_index`、`chains_total`、`chain_break_reason`，`reward_scope=trajectory`。
- **真正独立分支**：定义为"拥有各自环境终态与各自 verifier outcome"的执行（例如 harness 在不同容器各跑一次）。它们是不同 `trajectory_id`、不同 `rollout_id`，各自归因、各算一个 GRPO 组成员。同一容器内的"重试分叉"（D5 情况 2）在 Codex 路径被 legacy bridge 的指纹去重与历史相等校验拒绝（协议违规 → 轨迹作废，不签名、不计 0 奖励）；对其他 harness 的处理留到首批之后，默认选项是"被放弃分支作为同一 trajectory 的兄弟段共享奖励"，但需单独批准。
- **首批断言**：Codex 路径每条轨迹 chain 数恒为 1（compaction 关闭、重试被拒）；GPU 9.2 把 `chains_total==1` 作为硬判据，多 chain 代码路径只在 CPU 用 fake harness 验收。
- 原 D5a 中"奖励归属见 D9"是笔误（D9 是沙箱代理）；归属规则以本段为准。

### R-D9. 沙箱代理接口补充（主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4）
- `acquire(env_id, trajectory_id, deadline)` 的 `deadline` 为硬上限：到期未 `destroy` 由 broker 强制销毁并返回 `InfraError(lease_expired)`，对应 sample `ABORTED`。
- `env_live` 定义：已 `acquire` 且 `describe` 尚未返回 gone 的 lease 数，**包含空闲（已分配但当前不在执行工具）的环境**；不包含未绑定轨迹的预热池（首批无预热池，若后续引入另设 `env_idle_pool`，不进 drain 条件）。
- 因每个 lease 都有 deadline，drain 的最长等待 = min(drain deadline, 最晚 lease deadline)，不会永久等待。

### R-IR. 对 INFRA 的接口请求精确化（由 INFRA-E1 实现，本 change 只做接入验收；主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4）

**IR-1 `miles_adapter/config.py` / `entry.py`**
- 字段：`agent.custom_agent_function_path: str | None`、`agent.agent_max_seq_len: int | None` → 生成 `--custom-agent-function-path <path>`、`--max-seq-len <int>`，仅当 `custom_generate_function_path == "miles.rollout.generate_hub.agentic_tool_call.generate"`；否则抛配置错误，消息须包含 `requires custom_generate_function_path=miles.rollout.generate_hub.agentic_tool_call.generate`。
- `agent.tito_allowed_append_roles` 继续拒绝，消息须包含 `decided by the --tito-model template (allowed_append_roles)`。
- `use_session_server` 与 `partial_rollout` 同时为真 → 配置错误，消息引用 `miles/utils/arguments.py:3240 "--use-session-server does not support --partial-rollout"`。
- `entry.py`：新增钩子 `preflight: Callable[[RunConfig], None] | None`，在 placement/分配之前调用；抛错则不发生任何 allocate 调用。
- 验收（CPU）：透传生成正确 argv；缺 agentic_tool_call 时拒绝且消息匹配；三条拒绝理由文本；互斥；preflight 失败时 fake allocator 调用次数为 0。

**IR-2 `tool_wait.drain_blockers` / `MilesRolloutPool` drain probe / 1.7 load sample**
- 新类型：`HarnessSnapshot(in_flight: int, env_live: int, generation: int)`。
- 新签名：`drain_blockers(router_in_flight: int | None, tool_wait: ToolWaitSnapshot | None, harness: HarnessSnapshot | None) -> list[str]`。`harness is None` → 追加 `"harness counts unknown"`（fail closed）；非 agentic profile 由 pool 传显式 `HarnessSnapshot(0, 0, 0)`，不是 None。`in_flight>0` → `"{n} harness sessions in flight"`；`env_live>0` → `"{n} sandboxes live"`。
- 准入顺序：`ElasticRolloutPool.drain(members, deadline)` 先关闭 `members` 的准入（既有语义），再轮询 blockers 为空。harness 侧接口：`HarnessAdmission.allow_new_session(member: str) -> bool`，drain 中返回 False，agent/网关在 `POST /sessions` 前调用；被拒的轨迹在该 rollout 内改派非 drain 成员或等待 undrain。
- `LoadSample` 新增字段 `harness_in_flight: int = 0`、`env_live: int = 0`；load payload 新增键 `harness_in_flight`、`env_live`、`tito_session_mismatch`、`tito_chain_breaks`（`dict[str,int]`）、`policy_age_violation`。`classify_load` 不变。
- 验收（CPU）：active=0 ∧ tool_wait>0 未排空；active=0 ∧ tool_wait=0 ∧ env_live>0 未排空；harness=None 未排空；显式零快照时与旧行为一致；drain 后 `allow_new_session` 为 False 且不再创建会话。

**IR-3 driver → rollout 的目标 policy version**
- `RolloutPool.generate(rollout_id: int, *, expected_policy_version: str)`，值为既有 `policy_token(rollout_id, policy_hash)`（`driver.py:170`）。`MilesRolloutPool` 把它写入每个 prompt sample 的 metadata `expected_policy_version`，`agentic_tool_call` 原样传给 agent。
- 实际值：session server 记录的每次生成 `weight_version(s)`（`samples/merge.py:83` 的 `WeightVersionsPerCall`），agent/网关汇总为 metadata `policy_versions_actual: list[str]`。
- 校验：age 0 下 `set(policy_versions_actual) == {expected_policy_version}`；否则 metadata `policy_age_violation=1`，sample `ABORTED`，不签名、不计 0 奖励。前提（待 INFRA 确认）：Publisher 更新权重时把 SGLang `weight_version` 设为同一 policy token。
- 验收（CPU）：fake pool/agent 下漂移 → ABORTED 且不进训练；一致 → 正常；缺 `expected_policy_version` → 配置错误。

**IR-4 1.7 指标 schema**
- 登记：`harness_in_flight`（gauge）、`env_live`（gauge）、`tito_session_mismatch_total`、`tito_chain_breaks_total{reason}`、`policy_age_violation_total`，标签 `profile`、`epoch`；`reason ∈ {retry_fork, history_rewrite, template_drops_reasoning, compaction_window}`。
- 验收（CPU）：关闭观测时字段缺省、旧路径快照不变；开启时字段存在且类型正确。

### R-TB. Terminal-Bench 任务子集（主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4）
- 数据集：`harbor-framework/terminal-bench-2`（原 `laude-institute/terminal-bench-2`，GitHub 301）main @ commit `2fd12b88aafdd04a52c298e3940bcb189f9766d6`（2026-04-30）。任务目录在仓库根（89 个）。数据生成沿用 pin 内 `examples/experimental/openenv/make_tbench2_data.py --tasks_dir <checkout>`。
- 子集（6 个，固定）：`fix-git`（legacy 测试已用）、`regex-log`、`sqlite-db-truncate`、`log-summary-date-ranges`、`openssl-selfsigned-cert`、`git-multibranch`。全部为多轮 shell 工具任务，verifier 对修改后的工作目录打分。
- 覆盖矩阵：
  - 多轮工具调用：全部任务；判据 每轨迹 ≥2 次 `terminal.exec`。
  - 修改后 verifier 打分：`fix-git`、`sqlite-db-truncate`、`git-multibranch`（终态依赖修改）。
  - 超时/取消/清理：对 1 条轨迹设 turn 预算=2 → `max_turns` 策略边界（签名，reward 0）；对 1 条轨迹注入 wall-clock 取消 → 进程组/HOME/session/lease 回收证据，`tool_wait`、`env_live` 归零；对 1 条轨迹注入沙箱创建失败 → `INFRASTRUCTURE_STATUS`，未签名，`ABORTED`。
  - 任务失败 vs 基础设施失败：前者 `status ∈ {completed, timeout, max_turns, max_seq_len}` 且带 HMAC；后者 `status=infrastructure`、无 MAC、`ABORTED`、不进训练。
  - 正奖励路径：(a) 用各任务自带 `solution/` 经同一 verifier + HMAC 链路产出 reward=1（基础设施级证明，不作训练数据）；(b) 模型生成至少 1 条 pass。负奖励路径：模型生成的失败轨迹。
  - 一次有效训练更新：≥1 个组通过 `apply_reward_nonzero_std_filter`（组内奖励 std>0），一次优化器 step 的 grad norm>0，LoRA 权重 checksum 变化，发布的 policy version 递增。若 N=4 组内无任何 pass，则记"合法否定结论：qwen35_08b 在该子集无正奖励"，训练更新判据改由主 agent 裁定（备选：换 legacy 的 Qwen3.8 profile，费用需重估）。
- 模型：legacy attested profile `qwen35_08b`（`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`）LoRA。

### R-SCOPE. 首批范围（主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4）
- 接原版 Codex 二进制（黑盒），由 legacy `_AppServerDriver` 经 app-server 协议 v2 驱动，模型端点为进程内 `_ResponsesBridge`。不接白盒 codex-agent。
- 固定版本（沿用 `yeto/rl/__init__.py`）：`codex-cli 0.145.0`；`@openai/codex@0.145.0-linux-x64`，target `x86_64-unknown-linux-musl`；二进制 sha256 `a2a05dafaa1acb002a45eaec0a462de5b13694fcfcd7bc43305f14781ce7be14`（310,730,800 B）；npm tarball sha256 `11239480f8e3efd1430f23bbe91c1a397856b8bbe6185ccbaee2382d25e03df2`；package manifest sha256 `8da5349aa5a4242f5e11c5ca8ff4a16d8f9f912cb8accebea4def94edbf30aee`；app-server schema v2 sha256 `f2415ee36b3c9fa16617c800910cd65b8086ce7c7fecee3dac5f7089eb5973b9`。获取：`npm pack` 该包后校验 tarball sha256，解出二进制校验 sha256 与大小，放到容器路径 `/opt/yeto/codex/codex-x86_64-unknown-linux-musl`；preflight 在线校验 `codex --version` 与 app-server schema。

### R-GPU. 冒烟预算（主 agent 已批 2026-10-01，用户边界见 SESSION6-HANDOFF §7.4；合计 ≤ $50，同一租期）
- 资源：1×H100（Nebius，≈$2.95/GPU·h，SkyPilot `--down` + autostop + 独立 watchdog 按实例 ID 终止；需要 docker 跑 TB2 任务容器，Modal serverless 不满足）；或 1×L40S（Modal ≈$1.95/h）仅当 TB2 容器可在 Modal 沙箱内运行——待确认，默认前者。费用按租期 wall time 计，含镜像拉取、预热、空闲。
- 9.1 A 路径冒烟（≤ $30 ≈ 10 h 上限，计划 6 h）：目标 = 1 个 rollout 步（6 任务 × n=4）+ 1 个训练步。通过条件 = preflight 通过；每条 sample 满足 4.3 断言；HMAC 三处验签通过；`tool_wait`/`env_live` 结束后归零；R-TB 覆盖矩阵全部出现；≥1 个非零 std 组且一次有效更新（或记录合法否定结论）。停机条件 = 费用达 $30、租期达 10 h、preflight 失败、任一对齐断言失败（立即停、拉日志）。
- 9.2 多轮 TITO 一致性（≤ $20 ≈ 6.5 h 上限，计划 3 h，与 9.1 同租期顺序执行）：目标 = `qwen35_08b` ≥20 条多轮轨迹。通过条件 = `chains_total==1` 全部成立；`tito_session_mismatch==0`；`tito_chain_breaks` 全零；trainer 重算 logprob 与 rollout logprob 差异落入 TIS 截断范围的比例 ≥99%。停机条件 = 费用达 $20、租期总计达 16.5 h、断链率>0（停并记录原因）。
- 单价为估算，下单前核对当日价格；两项合计硬上限 $50，超出即 `sky down <cluster>` 并核实释放。
