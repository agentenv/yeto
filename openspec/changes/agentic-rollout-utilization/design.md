# Design

## Context
- 调研与实测见 infra-drafts/AGENTIC-GPU-UTIL-RESEARCH.md。现有约束（读码 agentenv/main 2c0b91a0）：
  - `execution_profile.py:205–215` 强制 `max_policy_age == 0`；`driver.py:803–870` 对非当前版本样本抛 `PolicyIdentityError`；
  - `cut.py:314–322` 要求 `buffer_length == 0`、`carried_over == 0`；
  - `adapters/miles/algorithm_flags.py:245` 把 `--partial-rollout` 列为不映射；`adapters/miles/rollout.py:411–415` 注释禁止 ports 路径部分 rollout；
  - `launcher.py:2328–2330` 已设多发数默认值，TB 动态过滤在 `launcher.py:2451–2459` 强制多发数 = rollout_batch_size+1；
  - `harness/codex/codex_openenv_generate.py:117` 逐条检查 SGLang 回报的权重版本，漂移即中止；
  - 多岛 `island_ledger.py:276–310` 已有 `ACCEPT_IS`（按落后版本数打折）判定，传输未实现。
- Miles fork（agentenv/miles，miles-next fb04d6ffa）：`--over-sampling-batch-size`（arguments.py:814）、`--partial-rollout`（856）、`--mask-offpolicy-in-partial-rollout`（866）；`generate_hub/multi_turn.py:32`、`agentic_tool_call.py:46` 断言不支持部分 rollout；会话服务与部分 rollout 互斥（3306）；中止钩子 `call_agent_abort_hook` 只拆轨迹不存状态。
- verl fork：`experimental/fully_async_policy` 有 `partial_rollout` + `staleness_threshold`，yeto verl 适配层未用。

## Goals / Non-Goals
**Goals:** 阶段 0 拿到真实收益与"偏向短轨迹"程度；阶段 1 把落后上限、版本段记账、切点格式、沙箱挂起规则定死并有 CPU 测试；阶段 2/3 在门槛满足后按开关逐步放开。
**Non-Goals:** 完全异步训练；跨岛 spot 弹性扩生成；KV 缓存按时限保留；修改训练数学（除跨版本重要性采样修正外）。

## Decisions
1. **一个开关 `--rl-max-policy-age N`（默认 0）贯穿全链路**，而不是复用 Miles 的 `--partial-rollout` 布尔量。理由：yeto 契约以"落后几版"表达，多岛账本已按落后版本数打折；Miles 开关由适配层从它推导（N>0 ⇒ partial-rollout + mask/TIS 设置）。备选：直接映射 Miles 布尔开关——无法表达上限、不中立，否决。
2. **阶段 0 截止丢弃，不续跑**。理由：零契约改动即可拿到收益上界与偏差数据。代价：偏向短轨迹，以丢弃 token 数和长度分布对照量化。
3. **token 级版本记录**：每个 token 存生成版本（续跑轨迹形成版本段）。训练端重要性采样比值用记录的生成概率计算，截断比例逐轮上报。备选：只按轨迹记录最旧版本——无法按段修正，否决。
4. **TB 动态过滤与多发的关系**：~~保留过滤，但多发数改为显式配置并与过滤所需的 +1 取最大值~~；记录过滤掉与截止丢弃的分别计数，避免两种丢弃混在一起。
   **缩小（10-09 主 agent 代用户拍板）**：`launcher.py:2451` 的 +1 只作用于 SecRLEnv 智能体，它不是多发，而是给同题重试留的一个空位（`legacy/engine.py` `_SecRLEnvRetryDataSource` 首发 target 组、硬性要求 capacity==target+1，`ssh_harness.py` 同样要求）；取最大值会让 SecRLEnv 运行在 rollout 内报错。SecRLEnv（已搁置）维持 ==+1，显式给其他值时报错并说明多发不适用于 SecRLEnv；非 SecRLEnv 路径（含 M1 的 codex OpenEnv）显式 `--over-sampling-batch-size` 直通 Miles，launcher 只校验（≥ rollout batch）与记录，不取最大值。
   **丢弃 token 数（10-09 主 agent 代用户拍板）**：Miles 在部分 rollout 关闭时于 `abort()` 直接丢弃被中止组，yeto 拿不到其 token 数；在 agentenv/miles 分支 `s18-abort-discard-stats`（efbbc63ea，基于 8bc52237a）给 `abort()` 加统计 `args.rollout_abort_discard_stats = {groups, samples, response_tokens, unknown_groups}`，yeto 的 meta hook 读取；镜像未含此提交时报 None（未知），不猜。
5. **切点携带未完成轨迹**：切点新增"在途轨迹"段（token、版本段、生成概率、会话状态引用），只在 N>0 时写；`cut.py` 的空缓冲断言改为"N=0 时必须为空"。
6. **agentic 挂起点在回合之间**：在 Miles fork 的 agentic 生成循环里，截止信号只在模型回合结束、工具结果写回后生效；沙箱用 Modal 沙箱保活（存活上限默认 600 s，取 G6 训练+发布时长量级），超时丢弃。备选：沙箱快照恢复（DeltaBox 式）——Modal 侧能力未验证，列为后续。
7. **多岛**：每岛独立凑够即截止；合并按 outer_version，样本按版本段进入 `island_ledger` 的 `ACCEPT_IS` 判定；落后上限进契约哈希，不一致按 #143 规则只拒该连接。
8. **FN（无前缀缓存的 mamba 混合结构）**：续跑需重算整段前缀，阶段 3 前先测重算代价；若续跑的前缀重算时间超过省下的等待时间，FN 只用阶段 0。

9. **verl 并行线**：yeto verl 适配层现在走同步训练器、vLLM 推理，`entry.py:26` 声明 `max_policy_staleness=0`。verl fork 的 `fully_async_policy` 已有 `partial_rollout`、`staleness_threshold`（`fully_async_rollouter.py:421`）、按副本中止并重试（约 1222–1252 行），生成走 `AgentLoopManager`，可接 `tool_agent_loop`。做法：阶段 0 在同步模式下设多发与截止；阶段 2 起由落后上限推导 `async_training.staleness_threshold` 与 `partial_rollout`，适配层把 verl 的样本版本、丢弃计数翻译成 yeto 统一事件字段；多岛仍由 yeto 同步服务按 outer_version 合并，verl 的内部 message_queue 只在单岛内使用。备选：直接用 verl 原生完全异步、不经 yeto 开关——绕开契约与多岛账本，否决。风险：fully_async 是 experimental 目录，接口可能变动，锁 fork 版本；`tool_agent_loop` 在中止重试时是从头重跑还是续跑要读码确认（列入任务 6.1）。
   **6.1 读码结论（verl fork acad9875，10-09）**：
   - 中止只打断单次模型生成，不打断工具执行：重新分配时对各副本 `abort_all_requests()`（`verl/experimental/fully_async_policy/fully_async_rollouter.py:1249`），权重同步先 `abort_replicas`（`verl/checkpoint_engine/base.py:476-485, 510`）；工具由 `_handle_processing_tools_state` 用 `asyncio.gather` 执行（`verl/experimental/agent_loop/tool_agent_loop.py:317-322`），不经过推理服务的中止，工具执行期间推理侧没有请求，中止对它无效。
   - 是续跑不是重跑：`FullyAsyncLLMServerClient.generate`（`verl/workers/rollout/llm_server.py:171, 243-332`）在 `stop_reason` 为 abort 且 `partial_rollout` 开启时，以 `prompt_ids + 已生成 token` 重新请求、累加 token 与 log_probs、扣减 max_tokens；对 agent loop 不可见（`tool_agent_loop.py:243` 只见一次完整返回）。`partial_rollout` 关闭时被中止的样本丢弃。默认 `partial_rollout: True`（`config/fully_async_ppo_trainer.yaml:23`）。
   - 版本粒度：每轨迹一个区间 `min_global_steps`/`max_global_steps`（`llm_server.py:307-311, 334-338`；多轮时 `tool_agent_loop.py:263-267` 只更新 max），无逐 token 版本，也无逐 token 生成版本与概率的对应——yeto 要逐 token 版本段需在适配层或补丁里补记。
   - `staleness_threshold` 是按样本数限流（`fully_async_rollouter.py:421` 读入，`493-497` 换算 `max_required_samples`，`1155` 比较，`594-605/952` 计数），不比较样本版本号、不丢弃超限样本——与 yeto"超过上限即丢弃"的语义不同，阶段 2 由 yeto 侧按版本段判定。
   - 与 yeto 多岛同步的冲突点：权重版本是 trainer 本地 `current_param_version`（`verl/experimental/fully_async_policy/fully_async_trainer.py:143, 687, 841`，也用于检查点目录与日志步），与外层 outer version 含义错位；rollouter 的 `global_steps` 是样本序号（`fully_async_rollouter.py:448, 711-715, 866-886`），续训时按固定比例推算；权重只经 `CheckpointEngineManager.update_weights` 从 actor 单向推送并先中止全部请求（`checkpoint_engine/base.py:510+`），外部同步器改写权重会绕过版本标记；`MessageQueue`（`message_queue.py:27-105`）不带版本检查。结论：verl 多轮工具续跑"原生可用"（模型生成层面），但版本记录要由 yeto 补成逐段，且外层版本须映射到 `current_param_version`。
   - **10-09 主 agent 代用户拍板**：verl 同步模式的截止不做（原拟 6.2b 构建期补丁，读码后撤回）：需跨 worker 计数并改同步训练器批次假设，代价大且只服务阶段 0 测量，阶段 0 数据用 Miles A/B 已足够；verl 的凑够即截止/续跑并入 6.4，走 fully_async 路径。阶段 2 接入时要在适配层翻译/约束的点（6.4 子要求）：partial_rollout 前缀续写发生在推理服务内、对 agent loop 不可见；版本只有每轨迹 min/max；staleness_threshold 按样本数限流不比版本号；trainer 本地版本号与 outer version 错位。
   - 同步模式：`rollout.over_sample_rate`（`verl/workers/config/rollout.py:178-179`）只有定义、fork 内无使用处，同步模式没有可用的多发截止；逐样本计时 `generate_sequences`/`tool_calls`/`compute_score`（`agent_loop.py:82-83, 1135, 1281-1302`）可直接翻译为统一字段。

## Risks / Trade-offs
- [截止丢弃偏向短轨迹，伤难题学习信号] → 阶段 0 判据含奖励与长度分布对照；偏差过大则阶段 0 只在评测岛或简单任务启用。
- [跨版本 token 使 TIS 截断比例上升、训练不稳] → 上限先取 1，逐轮上报截断比例，设告警阈值；超过阈值自动回退上限 0（运行内只降不升）。
- [沙箱跨轮存活增加 Modal CPU 费用，长时间存活不稳定] → 存活上限 + 超时丢弃；阶段 3 上卡统计存活费用。
- [N15 续训固定偏差尚未定位，叠加在途轨迹状态后更难排查] → 阶段 2 门槛包括"N15 偏差已定位"。
- [Miles fork 放开多轮断言与上游分叉加大] → 改动集中在中止钩子与生成循环，写 fork 侧单测，不向上游提 PR。
- [Modal 沙箱并发与冷启动上限未知] → 阶段 0 补采冷启动；G3 已测 256 并发成功。

## Migration Plan
- 每阶段一个开关档位；默认 0 不变。回退：设回 0 即可；切点里有在途轨迹而以 0 恢复时，丢弃在途轨迹并上报（不报错）。
- 阶段门槛：
  - 进入阶段 2：阶段 0 生成段中位时长 ≤ 基线 0.7 倍、奖励与长度分布偏差 ≤1 标准差、N15 偏差已定位、默认配置标准样本不变。
  - 进入阶段 3：阶段 2 截断比例与奖励曲线在小模型上与基线误差内，且 FN 前缀重算代价已测。

## Open Questions
- 沙箱存活上限 600 s 是否合适：等阶段 0 补采的工具耗时分布再调，不影响规格与任务拆分。
