# Proposal

## Why

`ports` 引擎路径（`rl-engine-ports`）目前不能跑 Codex harness rollout：`miles_adapter/config.py:332–337` 以“upstream Miles has no --custom-agent-function-path”为由拒绝 `agent.custom_agent_function_path` 与 `agent.agent_max_seq_len`。调研（`/home/michael/work/infra-drafts/codex-harness-ab.html`）表明这个理由不成立：上游 `radixark/miles` 主干已有 `agentic_tool_call`、Session Server 与 TITO，fork 在这几个目录与上游零差异，两个参数由 `agentic_tool_call.generate.add_arguments` 注册。legacy 路径的 Codex 链路（agent 包、`_ResponsesBridge`、preflight、`tbench_outcome` HMAC）分散在未入库的包和 `agentenv/miles` 副本里，legacy 删除前必须迁到 ports。

同时，用户确定长期形态为方案 B：Codex 以原版黑盒 CLI 运行在沙箱中，yeto 提供协议网关，把 Responses / Chat / Anthropic Messages 请求统一转换成 TITO。这样也能接入其他 harness。本 change 先落方案 A 作为验证路径，再把网关抽出。沙箱只预留接口，不实现。

## What Changes

- **方案 A 验证路径（先做，不改 fork）**：
  - ports 放开 `agent.custom_agent_function_path` 与 `agent.agent_max_seq_len`，并更正拒绝理由；
  - 以上游 `miles.rollout.generate_hub.agentic_tool_call.generate` 作为 `custom_generate_function_path`；
  - Codex agent 包（legacy `yeto_miles_secrlenv`、`codex_openenv_*_agent_function`）搬入 yeto 仓库；
  - `_preflight_codex_harness`、`tbench_direct_preflight`、app-server schema 比对迁到 ports 启动路径；
  - agent 函数接入 `ToolWaitBoard`（`async_tool_wait_scope`）。
  - `agent.tito_allowed_append_roles` 继续拒绝，理由改为“由 `--tito-model` 对应模板决定”。
- **方案 B 网关（第二步）**：把 `_ResponsesBridge` 抽成 yeto 自有网关服务，提供 `/v1/responses`、`/v1/chat/completions`、`/v1/messages` 三个入口，统一落到 Session Server TITO。借鉴 MiMo 网关的做法：
  - 前缀哈希链与多 chain：只追加失败时另起 chain，并记录原因；
  - 回滚；
  - mask/logprob 对齐断言；
  - 每次生成记录权重版本，用于断言 policy age 0。
  - 凡能指向 OpenAI / Responses / Anthropic 兼容端点的 harness 都可接入，前提是满足“只追加”约束或接受分 chain。
- **奖励契约**：沿用 `tbench_outcome` + HMAC。B 形态下分出两层：可信层持有密钥、运行 verifier 并签名；Codex 进程与任务容器属于不可信层，看不到密钥。
- **思维链**：每轮生成的 reasoning token 进入训练（mask=1）。多轮回放历史推理时，chat template 必须一致，作为明确约束并配测试。模板会丢弃历史 think 段时（如 Qwen3），另起 chain，记录并计量。
- **长轨迹**：本 change 的行为是超时即截断丢弃，同时预留三个接口：轨迹分段 `segment_id`、段间上下文可替换（留给 CompactionRL）、奖励按段或按整条归属。CompactionRL 与 partial/续跑列为后续 change。
- **沙箱只预留入口**：
  - 定义 `SandboxBroker` 接口（acquire/exec/copy/verify/destroy、lease/心跳）；
  - 定义环境注册表数据结构（env_id → 镜像 digest、规格、verifier、网络策略）；
  - drain 条件中预留 `env_live` 计数。
  - 实现留给后续 change。
- **默认值**：TIS 默认开启。`tito_session_mismatch` 与分 chain 计数纳入 rl-infra-spec 1.7 的指标。
- 不修改 fork Miles/SGLang。driver/ports 中属于 INFRA 的文件，以接口请求形式列出，不在本 change 中直接修改。

## Capabilities

### New Capabilities

- `rl-agentic-harness-rollout`：ports 路径上的 agentic harness rollout 契约，覆盖：
  - A 路径配置透传与 preflight；
  - 协议网关（三种入口 → TITO）、只追加 / 多 chain / 回滚；
  - mask 与 logprob 对齐、每次生成的权重版本；
  - reasoning token 训练与模板一致性；
  - 奖励签名与信任分层；
  - 长轨迹截断和分段预留；
  - tool-wait 与在途计数上报。
- `rl-harness-sandbox-broker`：沙箱代理的预留接口契约，覆盖：
  - 租约生命周期与心跳；
  - 环境注册表结构；
  - 以 404 为准的销毁确认；
  - 基础设施错误与答错分开；
  - 奖励前防篡改；
  - 默认拒绝出网；
  - `env_live` drain 计数。
  - 本 change 只交付接口与假实现（fake）。

### Modified Capabilities

（无。rl-infra-spec 的 `island-elastic-reconfiguration` 尚未归档，对其 drain 条件与 1.7 指标的扩展以接口请求形式写入 design，不在此处修改其要求。）

## Impact

- 代码（后续 apply 时）：
  - `yeto/rl/engine/miles_adapter/config.py`：两个字段放开，属于 INFRA 共享文件，走接口请求；
  - 新增 `yeto/rl/harness/`：agent 包、网关、preflight、sandbox broker 接口；
  - `yeto/rl/tbench_outcome.py` 与 `yeto/rl/trajectory_evidence.py` 复用；
  - `tests/test_secrlenv_codex_harness.py` 的用例迁到新包。
- 依赖：
  - 上游 Miles `agentic_tool_call` 与 Session Server，按 fork 现有 pin，不升级；
  - Codex CLI 0.145.0，签名锁定，沿用 legacy。
- 与其它 change 的关系：
  - 阻塞于 rl-infra-spec 的 1.7、3.3/3.3b 的 GPU 验收，以及 X5 drain；
  - 上游 `arguments.py:3240` 规定 session server 与 partial rollout 互斥，所以本 change 不支持 partial rollout。
- GPU：只有 A 路径冒烟与多轮 TITO 一致性两项，使用便宜卡，每项事先定好判据与费用上限（见 tasks）。

> 待批准修订（2026-10-01）：legacy `yeto_miles_secrlenv` 源码已找到（yeto `5bfc011`），"重写"改为"搬运"；`codex_openenv_*` 三模块需重写；legacy 可训练 compaction 与 fork pin 的 session server 不兼容，首批关闭。详见 design.md"待批准修订提案"与 `/home/michael/work/infra-drafts/CODEX-PROGRESS.md`。
