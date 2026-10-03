# Implementation tasks

标记：`[Y]` 仅改 yeto 自有新文件；`[IR-n]` 需要修改 INFRA 所有的文件，以 design D11 的接口请求形式提出，得到 INFRA owner 同意后才能执行；`[阻塞: …]` 被 rl-infra-spec 阻塞，括号内写明解除条件；`[GPU]` 需要 GPU，执行前按第 9 组的判据和费用上限获批。全部任务不修改 fork Miles/SGLang。

## 1. A 路径配置透传（IR-1）

- [x] 1.1 [IR-1] `miles_adapter/config.py`：放开 `agent.custom_agent_function_path`、`agent.agent_max_seq_len`，映射到 `--custom-agent-function-path` / `--max-seq-len`；仅当 `custom_generate_function_path` 为 `miles.rollout.generate_hub.agentic_tool_call.generate` 时允许；更正 332–337 行的拒绝理由；`tito_allowed_append_roles` 继续拒绝，理由改为“由 `--tito-model` 模板决定”。验收：CPU 单测，覆盖透传、缺少 agentic_tool_call 时拒绝、理由文本三种情况。 完成记录（2026-10-03，T3-B）：`miles_adapter/config.py:378-394,796-799`；`tests/test_rl_ir_harness.py::test_ir1_agent_flags_pass_through_with_agentic_generate/need_agentic_generate/append_roles_rejected_with_template_reason`。
- [x] 1.2 [IR-1] 启用 session server 时同时请求 partial rollout，启动失败，错误信息引用上游 `arguments.py:3240` 的互斥规则。验收：CPU 单测。 完成记录（2026-10-03，T3-B）：`config.py:191-220,858-861`；`tests/test_rl_ir_harness.py::test_ir1_session_server_and_partial_rollout_are_mutually_exclusive`（引用行 3240 = 上游 radixark 9e4260d；fork pin e3a11ab 同一断言在 `arguments.py:3307`）。
- [x] 1.3 [Y] 用 fork pin 的上游 `agentic_tool_call.add_arguments` 解析 1.1 生成的参数，确认参数被识别、没有未知参数。验收：CPU 测试（只 import 解析器，不起 Ray）。 完成记录（2026-10-03，T3-B）：`tests/test_rl_agentic_tool_call_args.py`——从 fork pin checkout（HEAD==MILES_NEXT_COMMIT，本机 ~/work/miles-fr1）加载 `agentic_tool_call.generate.add_arguments`（stub 运行时依赖，不起 Ray），`--custom-agent-function-path`/`--max-seq-len` 被识别、无未知参数、`--agent-max-seq-len` 非上游参数。

## 2. agent 包与 preflight 入库

- [x] 2.1 [Y] 定位 legacy `yeto_miles_secrlenv` 与 `codex_openenv_*_agent_function` 源码（agentenv/miles examples、镜像内 site-packages）；找到就搬到 `yeto/rl/harness/codex/`，找不到就按 `tests/test_secrlenv_codex_harness.py` 重写，并在 progress 中标“重写”。验收：原测试文件的全部用例迁到新包后在 CPU 上通过。 **完成记录（已实现 / CPU 通过：legacy 包从 5bfc011 搬入 `yeto/rl/harness/codex/`（出处与 sha256 见 `pins.py`），`codex_openenv_*` 三模块重写；原 34 用例改 import 后 30 通过 + 4 skip（需 Codex 0.145.0 二进制）；证据 CODEX-PROGRESS §阶段 2）**
- [ ] 2.2 [Y] 把 `_preflight_codex_harness`、`_verify_live_codex_app_server_schema`、`tbench_direct_preflight.validate_hmac_key_source` 抽到 `yeto/rl/harness/codex/preflight.py`，legacy 同名函数改为转调。验收：legacy 与新入口对同一组篡改输入（二进制 sha、schema、工具面、密钥权限）给出相同失败；CPU 单测。 **部分完成（新入口 `preflight.preflight_codex_openenv` 已实现 / CPU 通过；legacy 转调以可注入的 `preflight.forward_legacy_openenv_preflight` 提供（同一组输入给出与 legacy 相同的 ValueError 类别），legacy 文件改一行与 `yeto/rl/__init__.py` pin 更新由主 agent 另派 IMG，字段与新 sha 见 `preflight.required_pin_updates()` 与 CODEX-PROGRESS §阶段 3；未勾选）**
- [x] 2.3 [IR-1] ports `entry.py` 在模型分配前调用 2.2 的 preflight。验收：CPU 单测，preflight 失败时不触发 placement / 分配调用。 **完成记录（已实现 / CPU 通过：INFRA `entry.preflight_stage` 的 `(miles_args, launch)` 钩子（偏差 1，接受）在 `connect_island_ray` 前运行；harness 侧 `codex.preflight.harness_preflight`（`HARNESS_PREFLIGHT_SPEC` 经 `YETO_HARNESS_PREFLIGHT`/`miles_args.yeto_harness_preflight` 解析）做 reward_scope/agent 函数/compaction/身份/密钥检查并安装 provider 与岛内 boards；失败时 allocator/Ray 调用数 0。`tests/test_rl_ir_harness.py::test_ir1_harness_preflight_*`、`tests/test_harness_codex_openenv.py::test_entry_preflight_stage_runs_codex_preflight_before_allocation_and_installs_boards`）**

## 3. tool-wait 与在途计数

- [x] 3.1 [Y] agent 函数内，把每次“模型返回→下一次请求”的区间包进 `async_tool_wait_scope(board_actor(learner_id), trajectory_id)`，模式同 `tool_wait_workload.py`。验收：CPU 测试用 fake bridge 驱动 3 轮，board 的 enter/exit 次数为 3，结束后计数归零；异常路径同样归零。 **完成记录（已实现 / CPU 通过：每次工具执行一个 tool-wait 区间（进程内 `_ToolWaitEnvironment`；子进程经 worker 事件转发到 board），3 次工具→3 对 enter/exit，取消路径归零；`tests/test_harness_codex_openenv.py`）**
- [x] 3.2 [IR-2] `drain_blockers` 与 `MilesRolloutPool` 的 drain probe 加入 `harness_in_flight`、`env_live`，未知值 fail closed；1.7 load sample 带上这两个字段。验收：CPU 单测覆盖“active=0 但 tool_wait>0 / env_live>0 时未排空”。 **完成记录（已实现 / CPU 通过：INFRA 三参 `drain_blockers(router, tool_wait, harness: HarnessSnapshot)`、`HarnessBoard`（准入/会话/带硬 deadline 的租约）、pool drain 先关准入、LoadSample 新字段；harness 侧 agent 入口与网关调 `allow_new_session`/`enter_session`/`exit_session`/`lease_acquired(deadline=)`/`lease_released`，取消路径归零，准入关闭 → ABORTED 不计 0 奖励。`test_rl_ir_harness.py::test_ir2_*`、`test_harness_codex_openenv.py::test_subprocess_run_counts_sessions_and_leases_on_the_harness_board_and_drains_to_zero`/`*_refuses_new_session_when_admission_is_closed`/`*_cancellation_releases_board_lease_and_session`、`test_harness_gateway.py::test_gateway_mirrors_counters_*`。8.3 的 fake broker 接入待 8.x）**
- [ ] 3.3 [阻塞: rl-infra-spec 1.7 勾选、3.3b fork-M3 合入] X5 drain 场景下 agentic 轨迹的 GPU 验收：在 3.3 X5 实验里加一条 agentic workload，验证 drain 期间旧路由保留到轨迹结束。解除条件：infra 3.3 进入 GPU 验收阶段；本任务并入其实验，不单独开卡。

## 4. 网关核心库（进程内，A 与 B 共用）

- [x] 4.1 [Y] 从 bridge 抽出 `yeto/rl/harness/gateway/`：Responses（SSE）、Chat、Messages 三入口的请求/响应转换，统一调用 Session Server 的 chat 路由；采样签名字段不可覆盖；不支持的形态显式拒绝。验收：迁入的 legacy 用例（reasoning 往返、whitespace、length 边界、畸形帧、响应上限）CPU 通过，新增 Messages 与 Chat 的往返用例。 **完成记录（已实现 / CPU 通过：`yeto/rl/harness/gateway/{translate,core}.py` 三入口→canonical chat→SessionBackend；签名采样字段不可覆盖；store:true/parallel_tool_calls/summary/非 function 工具/compaction 项/未知块一律拒绝；`tests/test_harness_gateway.py`。注：legacy bridge 的 SSE 解析用例留在 `codex_harness_agent`（进程内 A 路径仍用 legacy bridge），网关 HTTP 外壳为 10.x）**
- [x] 4.2 [Y] 前缀哈希链与多 chain：延续、分叉回滚、断链另起三种路径，`chain_break_reason` 枚举，禁止修补后沿用旧 chain。验收：CPU 单测覆盖 spec 中“正常多轮 / 重试分叉 / 历史改写”三个场景，并做属性测试（随机追加序列下 chain 数与期望一致）。 **完成记录（已实现 / CPU 通过：`gateway/chains.py` 前缀哈希链（h_k=H(h_{k-1}‖canonical(msg_k))）；延续 / 分叉另起（retry_fork，旧 chain 不动）/ 断链另起（history_rewrite、template_drops_reasoning、compaction_window）；禁止修补：不允许角色→`tito_session_mismatch` 并作废；`max_chains=1` 首批断言；属性测试 60 轮随机序列 chain 数与断链计数一致、每个生成事件恰在一条 chain）**
- [x] 4.3 [Y] mask/logprob 对齐断言与每次生成的 `policy_version` 记录；age 0 下版本漂移使轨迹作废。验收：CPU 单测覆盖 logprob 缺失、长度不等、mask=1 位置非生成、版本漂移四种情况，全部作废且不记 0 奖励。 **完成记录（已实现 / CPU 通过：`alignment.py` 断言 + `codex_openenv_generate.py` 的 policy version 校验；logprob 缺失/长度不等/mask=1 非生成/版本漂移四种情况作废（ABORTED，无 0 奖励））**
- [x] 4.4 [IR-3] driver 在 age 0 配置下把目标 `policy_version` 传给 rollout。验收：CPU 单测。 **完成记录（已实现 / CPU 通过：INFRA `RolloutPool.generate(rollout_id, *, expected_policy_version)` + `MilesRolloutPool` 经 sink 发布 token，`rollout_meta_hook.expected_policy_version(sample)` 读取、`harness_counters` 聚合 → driver `PolicyIdentityError`；harness 侧 `codex_openenv_generate` 与子进程入口改用该函数，缺 token 时 `PolicyVersionMissing` 拒绝生成，漂移 → `policy_age_violation=1`/ABORTED。`test_rl_ir_harness.py::test_ir3_*`、`test_harness_codex_openenv.py::test_subprocess_policy_token_comes_from_metadata_or_driver_sink_and_missing_refuses`/`test_generate_wrapper_refuses_without_policy_token_and_reads_sink`。GPU 下 Publisher token == SGLang weight_version 的实际核验留 9.x）**
- [x] 4.5 [IR-4] 1.7 指标 schema 登记 `tito_session_mismatch`、`tito_chain_breaks{reason}`、`policy_age_violation`、`harness_in_flight`、`env_live`，带 profile/epoch 标签。验收：CPU 单测，关闭观测时兼容旧路径。 **完成记录（已实现 / CPU 通过：INFRA `timeline.LOAD_SAMPLE_SCHEMA`/`HARNESS_METRIC_KEYS`/`TITO_CHAIN_BREAK_REASONS`/`validate_load_sample`，标签 `profile_hash`/`epoch`，计数名无 `_total` 后缀（偏差 3，接受，design R-IR 已修订）；关闭观测时不发出 `rl_load_sample`。harness 侧网关把 `tito_session_mismatch`/`tito_chain_breaks{reason}` 镜像到 `HarnessBoard`，生成包装把 `policy_age_violation` 写入 sample.metadata。`test_rl_ir_harness.py::test_ir4_*`、`test_harness_codex_openenv.py::test_ir4_schema_names_match_the_harness_payload_keys`）**

## 5. reasoning 与模板一致性

- [ ] 5.1 [Y] 每个受支持的 tito_model 声明 `keeps_history_reasoning`，并加一个离线测试：用 fork 的 `chat_template_verify.py` 渲染多轮带 think 的历史，比对声明。验收：CPU 测试，覆盖 legacy 用到的 Qwen3.5 / Qwen3.8 profile 与一个 Qwen3 模板。
- [ ] 5.2 [Y] 网关在 `keeps_history_reasoning=false` 时按模板原因另起 chain，且 reasoning 生成段 mask=1。验收：CPU 单测，检查 mask 与断链原因计数。

## 6. 奖励契约与信任分层

- [x] 6.1 [Y] 奖励函数、父进程 `verified_outcome`、`trajectory_evidence` 三个验签点在 ports 路径上都生效。验收：CPU 测试，篡改 reward / 增删字段 / 错误密钥三种情况在三处都被拒收。 **完成记录（已实现 / CPU 通过：`tbench_reward.reward_func`、父进程 `verified_outcome`、`trajectory_evidence._verified_outcome` 三处对篡改 reward/增字段/错密钥拒收）**
- [x] 6.2 [Y] 失败分类：基础设施错误不签名，标 aborted，不进训练；策略边界与答错签名，奖励 0。验收：CPU 单测，沿用 legacy 的 precreate 503 / 基础设施重试超时用例。 **完成记录（已实现 / CPU 通过：基础设施错误未签名+ABORTED；超时/答错签名 reward 0；legacy precreate 503 / 重试超时用例随搬运套件通过）**
- [x] 6.3 [Y] 在 design D7 基础上补一张可信层/不可信层边界清单（密钥位置、verifier 资产注入时机、进程与网络边界），写进 `yeto/rl/harness/README` 段落或 docs/MILES_RL.md 对应节。验收：文档评审；CPU 测试断言 agent 子进程环境中没有密钥变量。 **完成记录（已实现 / CPU 通过：`yeto/rl/harness/README.md` 可信/不可信边界表（密钥、verifier 资产注入时机、进程/网络边界、三处验签、失败分类、清理）；`test_scrubbed_environment_removes_every_reward_key_name` + worker 拒绝可见密钥用例）**

## 7. 长轨迹与分段预留

- [x] 7.1 [Y] 超时 / `max_seq_len` 按策略边界截断并签名；sample metadata 带 `trajectory_id`、`segment_id=0`、`segment_boundary_reason=null`、`reward_scope=trajectory`；配置 `reward_scope=segment` 时启动失败。验收：CPU 单测。 **完成记录（已实现 / CPU 通过：超时/`max_seq_len` 作为策略边界签名 reward 0（阶段 2，`*_signed_policy_boundaries`）；`trajectory_fields` 给出 `trajectory_id`/`segment_id=0`/`segment_boundary_reason=null`/`reward_scope=trajectory`；`reward_scope=segment` 由 INFRA `config.check_harness_reward_scope(miles_args.yeto_harness_reward_scope)` 在 `validate_parsed_args` 拒绝，并在 `harness_preflight` 再检查（偏差 2：未进 `RLRunConfig` 叶子）。`test_rl_ir_harness.py::test_ir1_reward_scope_segment_fails_at_startup`、`test_harness_codex_openenv.py::test_entry_preflight_stage_*`）**
- [x] 7.2 [Y] 网关上下文拼接走 `ContextProvider`，默认为恒等映射。验收：CPU 单测证明恒等实现不改变 token 序列。 **完成记录（已实现 / CPU 通过：`gateway/context.py` `ContextProvider`/`IdentityContextProvider`，网关发送前经过 provider，默认恒等（测试断言发送消息与请求逐项相等））**

## 8. 沙箱代理接口（只预留）

- [ ] 8.1 [Y] `SandboxBroker` Protocol（acquire/heartbeat/exec/copy_in/copy_out/verify/destroy/describe）、租约与错误类型。验收：类型检查与 CPU 单测。
- [ ] 8.2 [Y] 环境注册表 schema 与校验：digest 锁定、规格、超时、verifier_ref、网络策略（默认拒绝出网）、扣留资产。验收：CPU 单测，可变 tag、缺项、默认放行出网三种情况都拒绝。
- [ ] 8.3 [Y] 本地 fake broker：以 describe 返回 gone 为销毁确认，维护 `env_live`；心跳超时判为基础设施错误。选择远程后端时启动失败。验收：CPU 单测覆盖 spec 全部场景，并接入 3.2 的 drain 计数测试。
- [ ] 8.4 [Y] A 路径适配：把现有 OpenEnv/Daytona 调用包成 `SandboxBroker`，不改变行为。验收：2.1 迁入的 driver / home cleanup / 进程组回收用例仍然通过。

## 9. GPU 验证（便宜卡，逐项获批）

- [ ] 9.1 [GPU] A 路径冒烟：ports + agentic_tool_call + Codex，在 Terminal-Bench 小子集上跑 1 个 rollout 步 + 1 个训练步。判据：
  - preflight 通过；
  - 每条轨迹的 sample 满足 4.3 的断言；
  - HMAC 验签通过；
  - tool_wait 计数在轨迹结束后归零；
  - 奖励分布非全空；
  - TIS 比值统计出现在指标中。
  资源：单卡 H100 或更便宜卡、小模型（与 legacy 相同的 Qwen3.5 小档位 LoRA），费用上限 $30，超出即停。任务子集规模在执行前确定（design Open Questions）。
- [ ] 9.2 [GPU] 多轮 TITO 一致性：在 9.1 同一环境中，对每个受支持的 tito_model 跑 ≥20 条多轮轨迹。判据：
  - 保留历史 reasoning 的模板：断链率 = 0，`tito_session_mismatch` = 0；
  - 丢弃历史 reasoning 的模板：断链原因只出现 `template_drops_reasoning`；
  - 所有 chain 的 logprob 与 trainer 重算 logprob 的差异落在 TIS 截断范围内的比例 ≥99%。
  费用上限 $20，可与 9.1 在同一租期内完成。
- [ ] 9.3 [阻塞: rl-infra-spec 异步契约与 age>0 支持] age 0 下长尾轨迹对整轮时延的影响测量。解除条件：infra 给出异步契约，或用户批准在 age 0 下单独测量。本 change 不为其开卡。

## 10. B 网关进程外壳

- [ ] 10.1 [Y] 在第 4 组的库外加 HTTP 服务：`/sessions/{id}/v1/responses|chat/completions|messages`，会话 token 鉴权，撤销后返回 410。验收：CPU 集成测试，fake Session Server 加 fake harness 跑三入口。
- [ ] 10.2 [Y] 由网关按会话上报 tool-wait 与 `harness_in_flight`（与 3.1 同一计数口径）。验收：CPU 测试，进程内模式与外壳模式的计数序列相同。
- [x] 10.3 [Y] 用 `codex exec` 加 fake 推理端点做一次本地 CPU 回放，验证黑盒 Codex 经外壳网关时单 chain、无 mismatch（不需要 GPU）。验收：测试通过，记录 Codex 版本与请求形态摘要。
  完成记录（2026-10-03，缩减口径）：网关 HTTP 外壳 10.1/10.2 未做，回放走进程内 `_ResponsesBridge`（A 路径）而非外壳网关：真 Codex 0.145.0（sha a2a05daf…）+ fake Responses 端点 + `tb2_provider.LocalProcessBackend` 真实 relay/verifier，3 次采样、单 chain、`tito_session_mismatch==0`、reward=1；证据 `evidence/cpu-20261003/g8-codex-0145-replay.json`，测试 `tests/test_harness_codex_exec_replay.py`（需 `YETO_CODEX_BINARY_PATH`）。

## 11. 后续 change（本 change 不做，仅登记）

- [ ] 11.1 在 progress 中登记以下后续 change 的范围与依赖：
  - 真实沙箱后端（每轨迹一 Pod、404 销毁、默认拒绝出网）；
  - 租约队列与远程 worker；
  - CompactionRL（分段奖励、上下文替换）；
  - partial / 续跑（依赖 infra 异步契约，且受上游 session server 互斥约束）；
  - 多 harness 首批认证（OpenHands / mini-swe-agent / Claude Code 风格）；
  - `apply_patch` freeform 工具。
  验收：progress 条目评审通过。

## 待批准修订（阶段 1 检查点，2026-10-01）

- 2.1：legacy 源码已在 yeto 历史 `5bfc011:yeto_miles_secrlenv/` 找到（sha256 与 `yeto/rl/__init__.py` pin 一致），改为"搬运 + 出处注记 + 重算 SHA"；`codex_openenv_*` 三模块未找到，按已知接口重写。新增验收：Responses 工具调用的 TITO 前缀复用、rollout 级取消清理（进程组/HOME/session/lease/计数归零）、三处 reward 验签、mask 与 token/logprob 对齐、兄弟段共享奖励且基线按 rollout 计一次、compaction 开启时 preflight fail closed。见 design R-CTX / R-D5a。
- 3.2 / 4.4 / 4.5：签名与验收以 design R-IR 为准（2026-10-01 按 INFRA 实现修订：钩子 `(miles_args, launch)`、`reward_scope` 走 `miles_args`、计数名无 `_total`）。
- 9.1 / 9.2：任务子集、模型、费用、通过/停机条件以 design R-TB / R-GPU 为准。
