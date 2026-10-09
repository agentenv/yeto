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
4. **TB 动态过滤与多发的关系**：保留过滤，但多发数改为显式配置并与过滤所需的 +1 取最大值；记录过滤掉与截止丢弃的分别计数，避免两种丢弃混在一起。
5. **切点携带未完成轨迹**：切点新增"在途轨迹"段（token、版本段、生成概率、会话状态引用），只在 N>0 时写；`cut.py` 的空缓冲断言改为"N=0 时必须为空"。
6. **agentic 挂起点在回合之间**：在 Miles fork 的 agentic 生成循环里，截止信号只在模型回合结束、工具结果写回后生效；沙箱用 Modal 沙箱保活（存活上限默认 600 s，取 G6 训练+发布时长量级），超时丢弃。备选：沙箱快照恢复（DeltaBox 式）——Modal 侧能力未验证，列为后续。
7. **多岛**：每岛独立凑够即截止；合并按 outer_version，样本按版本段进入 `island_ledger` 的 `ACCEPT_IS` 判定；落后上限进契约哈希，不一致按 #143 规则只拒该连接。
8. **FN（无前缀缓存的 mamba 混合结构）**：续跑需重算整段前缀，阶段 3 前先测重算代价；若续跑的前缀重算时间超过省下的等待时间，FN 只用阶段 0。

9. **verl 并行线**：yeto verl 适配层现在走同步训练器、vLLM 推理，`entry.py:26` 声明 `max_policy_staleness=0`。verl fork 的 `fully_async_policy` 已有 `partial_rollout`、`staleness_threshold`（`fully_async_rollouter.py:421`）、按副本中止并重试（约 1222–1252 行），生成走 `AgentLoopManager`，可接 `tool_agent_loop`。做法：阶段 0 在同步模式下设多发与截止；阶段 2 起由落后上限推导 `async_training.staleness_threshold` 与 `partial_rollout`，适配层把 verl 的样本版本、丢弃计数翻译成 yeto 统一事件字段；多岛仍由 yeto 同步服务按 outer_version 合并，verl 的内部 message_queue 只在单岛内使用。备选：直接用 verl 原生完全异步、不经 yeto 开关——绕开契约与多岛账本，否决。风险：fully_async 是 experimental 目录，接口可能变动，锁 fork 版本；`tool_agent_loop` 在中止重试时是从头重跑还是续跑要读码确认（列入任务 6.1）。

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
