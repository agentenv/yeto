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
   **实现（S18 ARU-3，10-09）**：
   - 挂起点放在 yeto 的 Codex 桥上（Codex → 桥 → Miles 会话服务）：每条轨迹一个"回合门"文件（`YETO_CODEX_SUSPEND_GATE`）。门关着时，桥在发下一个模型请求之前等待；工具在沙箱里照常跑完，结果已在这次请求的历史里。截止那一刻正在生成的模型回合被引擎中止（会话服务返回 503、不记录），桥在门打开后原样重发这一回合（被作废的只有这一回合已解码的 token）。
   - Miles fork（agentenv/miles `s18-agentic-suspend`，`--agentic-suspend-between-turns --agentic-suspend-max-rounds N`，不用 `--partial-rollout`）：截止时先调 agent 模块的 `suspend()` 关门，再中止引擎请求并确认引擎空闲（`/get_load` 为 0，供训练卸载显存），未完成的组任务不等待、留在常驻事件循环里；下一轮开始调 `resume()` 开门，接回这些组并补发新组到多发数；开始轮超过 N 轮的组在截止时取消（agent 释放沙箱）并计数；凑够后才完成的组也留到下一轮，不丢；最后一轮照旧中止。
   - 沙箱保活：挂起期间沙箱不动；桥的等待上限 `YETO_CODEX_SUSPEND_MAX_SECONDS`（默认 600 s），超时抛 `CodexSuspendExpired` → 基础设施样本、`run` 的 finally 销毁沙箱。Modal 沙箱的 `idle_timeout` 必须大于存活上限加一个模型回合（上卡用 900 s）；Codex 的 `stream_idle_timeout_ms` 在开门控时设为（上限+600 s）。
   - 版本：同一条轨迹的模型调用跨版本（每次调用一个版本段），按"结束时的当前版本 − 最旧版本 ≤ N"判定（5.2，`codex_openenv_generate.window_problem`；Miles 侧按开始轮次先行取消）；工具输出 token 不计跨版本。
   - 切点（3.3 的 Miles 接线）：上限 >0 时切点导出缓冲组（完整 Miles 样本 + 版本段 + 生成概率）与挂起 agentic 轨迹的引用；恢复时缓冲组按规则续跑或丢弃，agentic 引用一律丢弃并上报（进程与沙箱已不在，不从头重跑同一题）。
7. **多岛**：每岛独立凑够即截止；合并按 outer_version，样本按版本段进入 `island_ledger` 的 `ACCEPT_IS` 判定；落后上限进契约哈希，不一致按 #143 规则只拒该连接。
8. **FN（无前缀缓存的 mamba 混合结构）**：续跑需重算整段前缀，阶段 3 前先测重算代价；若续跑的前缀重算时间超过省下的等待时间，FN 只用阶段 0。
   **估算（S18 ARU-3，10-09，未上卡；脚本与结果 s1-runs/s18-aru3-fn-prefix/{analyze.py,estimate.json}，数据 G6 s1-runs/s17-fncodex-r3-20261008a 1 轮 24 条）**：
   - FN 关了前缀缓存：G6 全部 87 行 Prefill 的 `#cached-token` 为 0；生成段内预填充新 token 合计 39.1 万，与各回合上下文合计 48.3 万同量级（比值 0.81，差约两成未查清）——每个模型回合本来就重算整段前缀。所以续跑的额外重算只在"截止那一刻正在生成、被作废重做的那一回合"，挂起后的下一回合本来就要全量预填充。
   - 上界估算（每条在途轨迹多重算一次整段上下文，预填充速率取大批次中位约 2.96 万 token/s，只有 6 行间接值）：长轨迹（≥5 回合）末回合上下文 p50 8462 / p90 约 15554 token → 每条约 0.29 / 0.53 s；24 条全部重做的极端上界约 2.7 s。
   - 省下的等待（代理值，`#running-req` 回落时刻）：去掉最慢 10% 约 32 s，去掉最慢 25% 约 56 s（G6 整轮约 923 s，研究文档的"每轮约省 10%"量级吻合）。
   - 结论：FN 续跑的前缀重算代价（≤约 2.7 s/轮）比省下的等待（约 32–56 s/轮）小一个数量级以上，**FN 不限于阶段 0**。可靠性：只 1 轮 24 条（14 条一回合即结束），预填充速率为间接值，省下的等待为代理值，8×H200 TP2/PP4/EP2 配置专属；结论方向稳健（差一个数量级），绝对数不可靠。不单为此上卡；FN 下次上卡时顺带采集预填充单批耗时、续跑后首回合首 token 延迟、被作废回合的已解码 token 数。

9. **verl 并行线**：yeto verl 适配层现在走同步训练器、vLLM 推理，`entry.py:26` 声明 `max_policy_staleness=0`。verl fork 的 `fully_async_policy` 已有 `partial_rollout`、`staleness_threshold`（`fully_async_rollouter.py:421`）、按副本中止并重试（约 1222–1252 行），生成走 `AgentLoopManager`，可接 `tool_agent_loop`。做法：阶段 0 在同步模式下设多发与截止；阶段 2 起由落后上限推导 `async_training.staleness_threshold` 与 `partial_rollout`，适配层把 verl 的样本版本、丢弃计数翻译成 yeto 统一事件字段；多岛仍由 yeto 同步服务按 outer_version 合并，verl 的内部 message_queue 只在单岛内使用。备选：直接用 verl 原生完全异步、不经 yeto 开关——绕开契约与多岛账本，否决。风险：fully_async 是 experimental 目录，接口可能变动，锁 fork 版本；`tool_agent_loop` 在中止重试时是从头重跑还是续跑要读码确认（列入任务 6.1）。
   **6.1 读码结论（verl fork acad9875，10-09）**：
   - 中止只打断单次模型生成，不打断工具执行：重新分配时对各副本 `abort_all_requests()`（`verl/experimental/fully_async_policy/fully_async_rollouter.py:1249`），权重同步先 `abort_replicas`（`verl/checkpoint_engine/base.py:476-485, 510`）；工具由 `_handle_processing_tools_state` 用 `asyncio.gather` 执行（`verl/experimental/agent_loop/tool_agent_loop.py:317-322`），不经过推理服务的中止，工具执行期间推理侧没有请求，中止对它无效。
   - 是续跑不是重跑：`FullyAsyncLLMServerClient.generate`（`verl/workers/rollout/llm_server.py:171, 243-332`）在 `stop_reason` 为 abort 且 `partial_rollout` 开启时，以 `prompt_ids + 已生成 token` 重新请求、累加 token 与 log_probs、扣减 max_tokens；对 agent loop 不可见（`tool_agent_loop.py:243` 只见一次完整返回）。`partial_rollout` 关闭时被中止的样本丢弃。默认 `partial_rollout: True`（`config/fully_async_ppo_trainer.yaml:23`）。
   - 版本粒度：每轨迹一个区间 `min_global_steps`/`max_global_steps`（`llm_server.py:307-311, 334-338`；多轮时 `tool_agent_loop.py:263-267` 只更新 max），无逐 token 版本，也无逐 token 生成版本与概率的对应——yeto 要逐 token 版本段需在适配层或补丁里补记。
   - `staleness_threshold` 是按样本数限流（`fully_async_rollouter.py:421` 读入，`493-497` 换算 `max_required_samples`，`1155` 比较，`594-605/952` 计数），不比较样本版本号、不丢弃超限样本——与 yeto"超过上限即丢弃"的语义不同，阶段 2 由 yeto 侧按版本段判定。
   - 与 yeto 多岛同步的冲突点：权重版本是 trainer 本地 `current_param_version`（`verl/experimental/fully_async_policy/fully_async_trainer.py:143, 687, 841`，也用于检查点目录与日志步），与外层 outer version 含义错位；rollouter 的 `global_steps` 是样本序号（`fully_async_rollouter.py:448, 711-715, 866-886`），续训时按固定比例推算；权重只经 `CheckpointEngineManager.update_weights` 从 actor 单向推送并先中止全部请求（`checkpoint_engine/base.py:510+`），外部同步器改写权重会绕过版本标记；`MessageQueue`（`message_queue.py:27-105`）不带版本检查。结论：verl 多轮工具续跑"原生可用"（模型生成层面），但版本记录要由 yeto 补成逐段，且外层版本须映射到 `current_param_version`。
   - **10-09 主 agent 代用户拍板**：verl 同步模式的截止不做（原拟 6.2b 构建期补丁，读码后撤回）：需跨 worker 计数并改同步训练器批次假设，代价大且只服务阶段 0 测量，阶段 0 数据用 Miles A/B 已足够；verl 的凑够即截止/续跑并入 6.4，走 fully_async 路径。阶段 2 接入时要在适配层翻译/约束的点（6.4 子要求）：partial_rollout 前缀续写发生在推理服务内、对 agent loop 不可见；版本只有每轨迹 min/max；staleness_threshold 按样本数限流不比版本号；trainer 本地版本号与 outer version 错位。
   - 同步模式：`rollout.over_sample_rate`（`verl/workers/config/rollout.py:178-179`）只有定义、fork 内无使用处，同步模式没有可用的多发截止；逐样本计时 `generate_sequences`/`tool_calls`/`compute_score`（`agent_loop.py:82-83, 1135, 1281-1302`）可直接翻译为统一字段。
   - **10-09 主 agent 代用户拍板：6.4 拆为 6.4a（翻译层，已做）+ 6.4b（fully_async 适配路径，本小节设计，实施另批）**。理由：fully_async 是另一套训练栈（FullyAsyncTrainer 继承旧版 SeparateRayPPOTrainer，不是 yeto 现用的 v1 PPOTrainerSync），接入等同重写 verl 适配层并需 2 卡多轮调试，超出阶段 2 的 $45 与范围；用户要求 verl 同步推进，故翻译层先落地、接入单独设计后再批。阶段 3 门槛只按 Miles 判定；6.5 依赖 6.4b。
   - **6.4a 翻译层**（`yeto/rl/adapters/verl/fully_async_translate.py`）：上限 N→`staleness_threshold = N−1`（fully_async 生成与训练重叠，本身至少落后 1 版；`trigger_parameter_sync_step=1` 时排队样本最多落后 floor(s)+1 版）、`partial_rollout=True`、一轮=一个 verl 参数版本（`require_batches × ppo_mini_batch_size = 每轮样本数`）；`VersionMap` 记录每次发布的 current_param_version↔outer version，检查点/日志步号用外层版本；每轨迹 min/max_global_steps→区间内全部外层版本、按最旧判定；按续写调用记录重建逐 token 版本段；超限由 yeto 丢弃。
   - **10-10 主 agent 代拍板（S19 GPU 跑 s19-verl64b-async8-20261010a 之后）**：映射改为上限 N → `staleness_threshold = N`（原为 N−1）。原因：N=1 时 s=0，rollouter 每轮只生成正好一轮就停（日志 "staleness_samples 32 >= max_required_samples 32"），推权重时没有在途请求，前缀续写 0 次，落后上限 1 退化成同步交替，F5 测不到。改后 verl 节流允许排队样本最多落后 N+1 版（`queue_version_lag`），比上限多 1；上限由 yeto 逐组按最旧版本丢弃保证（`judge_trajectory`/`RoundCollector`，单测 `test_kept_groups_never_exceed_the_limit_when_verl_runs_one_round_ahead`），计入 `rl_rollout_cutoff`。判据 F1–F6 不变；F6 量级可能因跨版本 token 增多而变化，照实记录。
   - **6.4b 设计（草案，待批）**：
     1. 驱动方式：不跑 fully_async 自带的两个 `fit()` 循环。保留 `FullyAsyncRollouter` 作常驻生成 actor（它的 MessageQueue、按样本数限流、partial_rollout 续写原样用），yeto `IslandDriver` 每轮：`generate` = 从 MessageQueue 取够一轮样本（`_get_samples_from_queue` 的取样逻辑）并用 6.4a 翻成 `RolloutBatchHandle`（组版本段、超限丢弃、计数进 `rl_rollout_carry_over`/`rl_rollout_cutoff` 同名字段）；`train_step` = 调 FullyAsyncTrainer 的 `fit_step` 中"算旧 logprob→优势→更新"三段（不含其 `_fit_update_weights`）；`publish` = yeto 发布后再调 `checkpoint_manager.update_weights(global_steps=param_version)`，并 `VersionMap.record(param_version, outer_version)`；rollouter `reset_staleness` 在发布后由 yeto 调。
     2. LoRA：fully_async trainer 在 LoRA 时 `ref_in_actor`（`fully_async_trainer.py:96-100`）；yeto 的 LoRA 导出/应用（`ports_impl.VerlPolicyState`，经 `__ray_call__` 读写 FSDP2 PEFT 权重）要改为对 trainer 侧 actor worker 组执行；发布读回（`vllm_readback`）改为对 rollouter 的 vLLM 副本执行，确认 checkpoint engine 推 LoRA 后副本上能读回同一哈希。外部同步器改写权重必须走 `update_weights`，否则绕过版本标记（6.1 读码）。
     3. 版本：服务端 `extra_fields["global_steps"]` 即推送时的 current_param_version——须真机核实（6.4a 假设）；续写调用记录（6.4a (a)）需在 `FullyAsyncLLMServerClient.generate` 外包一层（yeto 子类或构建期补丁，`patch_verl.py`），每次调用记下版本与新增 logprob。
     4. 多岛：每岛内部 MessageQueue 只在岛内用；跨岛仍按 outer version 由 yeto 同步服务合并；续训切点需保存 rollouter 在途样本（MessageQueue 内容 + 续写状态）或按 3.3 规则丢弃并上报。
     5. 卡数与预算：推理与训练分卡，单岛最少 2×H100（rollouter 1 + trainer 1）；调试估计 3–5 次上卡（起机、发布读回、续写版本、对照），每次 0.6B ≈20–30 min×2 卡 ≈$3–4，合计 ≈$15–25；代码量估计 400–700 行 + 补丁。
     6. 风险：fully_async 在 experimental 目录、接口会变（锁 fork 提交）；trainer 继承旧版训练器，yeto 的 v1 判据钩子（`_compute_old_log_prob` 后接 mismatch 判据）要重挂。
   - **6.4b 实施取舍（2026-10-09，子 agent 代拍板 + 理由）**：
     1. 取样粒度：trainer actor 每次只从 MessageQueue 取 1 个样本（一个提示、n 条回答），由 yeto 逐个判定保留或丢弃，凑够 `groups_per_round` 个保留组才组批。理由：verl 的 `_get_samples_from_queue` 一次取满不判版本，逐个取才能在组批前丢弃超限组；改动只在 yeto 子类。
     2. GRPO 组整体保留或整体丢弃，按组内最旧 token 的版本判定。理由：组内优势要整组算，拆组会改变优势。
     3. 逐 token 版本段只记"每次调用的版本 + 新增 token 数"（构建期补丁写入 `extra_fields["yeto_resume_calls"]`），不另存 logprob。理由：logprob 已在 `rollout_log_probs` 里；版本段只需切点。token 数之和与回答长度不等时记入 `segment_token_mismatches`，不报错（F5 上卡核对）。
     4. 声明上限只到 1（`SUPPORT = stage 2, max_policy_age 1`）。理由：复核文档只测上限 1；上卡通过后再放宽。
     5. 驱动模式记为 `partitioned-serial`（放置 `fixed-partition`）。理由：rollouter 与 trainer 分卡；驱动仍按"生成→训练→发布"顺序调用，rollouter 的后台生成由 verl 限流，不算驱动层重叠。
     6. rollouter 的 `fit()` 在第一次 yeto 发布（v0）之后才启动；verl 初始化时自带的一次权重推送（param 0）保留，yeto 的 v0 发布再推一次，`VersionMap` 记录 (0, 0)。理由：少改 verl 初始化流程。
     7. 多岛续训：本次不保存 rollouter 在途样本；岛重启时 MessageQueue 内容丢失，新进程的 param version 从 0 重新配对外层版本（`VersionMap` 允许）。理由：设计第 4 条允许"丢弃并上报"；复核文档只跑单岛。

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
  - 进入阶段 3：阶段 2 截断比例与奖励曲线在小模型上与基线误差内，且 FN 前缀重算代价已测。（10-09 主 agent 代用户拍板：只按 Miles 判定；verl 阶段 2 接入为 6.4b，另批。）

## Open Questions
- 沙箱存活上限 600 s 是否合适：等阶段 0 补采的工具耗时分布再调，不影响规格与任务拆分。
