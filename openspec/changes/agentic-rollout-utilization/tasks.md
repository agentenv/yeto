# Tasks

## 1. 阶段 0：多发请求、凑够即截止（不续跑）与指标补采
- [x] 1.1 launcher 显式配置多发数~~，与 TB 动态过滤所需 +1 取最大值~~；过滤丢弃与截止丢弃分别计数（验证：launcher 单测覆盖三种组合）
  - 缩小（10-09 主 agent 代用户拍板，理由见 design 决定 4）：SecRLEnv 维持 ==+1 并明确报错；非 SecRLEnv 显式多发直通 Miles，只校验不取最大值。
  - 证据：`tests/test_rl_launcher.py::test_over_sampling_without_filter_is_forwarded_for_cutoff`、`::test_over_sampling_below_the_batch_is_rejected`、`::test_secrlenv_agent_rejects_conflicting_replacement_contract[extra4-not available to SecRLEnv agents]`（三种组合：无过滤显式多发、低于批次拒绝、SecRLEnv 显式多发拒绝；SecRLEnv 默认 +1 由原有 `test_legacy_secrlenv_agent_auto_binds_exact_replacement_contract` 覆盖）；多发提交数改用单调 `sample_group_index`（数据集回绕安全）：`tests/test_rl_miles_adapter_rollout.py::test_submitted_groups_survive_epoch_wrap_via_group_index`；过滤丢弃（`filtered_groups`）与截止丢弃（`discarded_groups`）分列于 `rl_rollout_cutoff`：`tests/test_rl_rollout_cutoff.py::test_cutoff_keeps_first_target_and_counts_filter_and_cutoff_separately`。
- [x] 1.2 截止时丢弃未完成轨迹并发事件（丢弃条数、已生成 token 数）（验证：假引擎单测，目标 24、多发 48，训练样本恰 24 条且全为当前版本）
  - 证据：核心 `yeto/rl/engine/rollout_cutoff.py`（事件 `rl_rollout_cutoff`，字段 `CUTOFF_FIELDS` 两后端共用）；`tests/test_rl_rollout_cutoff.py::test_fake_engine_target_24_over_sampled_48_trains_exactly_24_current`（6 组×4=24 条，多发 12 组=48 条，两轮各训练 24 条且版本令牌全为当前版本，事件记丢弃 6 组 24 条及 token 数）；Miles fork 丢弃统计 agentenv/miles `s18-abort-discard-stats` efbbc63ea，fork 单测 `tests/fast/rollout/test_sglang_rollout.py::TestAbort::test_abort_without_partial_rollout_tallies_discarded_groups`；yeto 读取 `tests/test_rl_miles_adapter_rollout.py::test_fork_abort_discard_stats_reach_the_handle`。
  - 注意：真机要拿到 token 数需重建含 efbbc63ea 的 ports 镜像；未重建时丢弃条数仍有（按 `sample_group_index` 推算，首轮未知），token 数报 None。
- [x] 1.3 补采指标：每轨迹起止时间、每回合模型生成/工具执行/判分三段耗时、沙箱冷启动、KV 占用与排队峰值（验证：假引擎事件流可拆出三段且之和对上生成段；接通 `tool_wait_trajectories`）
  - 证据：核心 `yeto/rl/engine/trajectory_timing.py`（`PhaseClock`、`phase_totals`）；codex OpenEnv 子进程智能体按工具进出边计时、记录沙箱获取耗时与起止墙钟时间；`rl_trajectory_reward` 新增可选字段 `trajectory_started_at/ended_at`、`sandbox_start_seconds`、`worker_seconds`、`turn_generation_seconds`、`turn_tool_seconds`（判分沿用 `evaluate_time`）；`rl_load_sample` 新增 `kv_used_tokens`、`kv_capacity_tokens`（SGLang 报才写），`timeline.load_peaks` 求排队与 KV 峰值；codex OpenEnv 智能体的 `tool_wait_trajectories` 接到岛内具名 ToolWaitBoard（原为未知）。测试：`tests/test_rl_rollout_cutoff.py::test_fake_engine_event_stream_splits_three_phases`、`tests/test_rl_trajectory_timing.py`（4 项）、`tests/test_harness_codex_openenv.py::test_subprocess_run_records_phase_timing`。
- [ ] 1.4 GPU A/B（并入 N17 合并验证运行）：M1 run d 配置单岛 1×H200 各 6 轮，判据见 design Migration Plan；结果与证据路径写回本文件与 AGENTIC-GPU-UTIL-RESEARCH.md

## 2. 阶段 1：落后上限开关与契约（默认 0）
- [x] 2.1 新增 `--rl-max-policy-age`（默认 0）进契约哈希；不一致的岛按只拒该连接处理（验证：严格与 elastic 握手单测；hash-migration.md 记录）
  - 证据：核心 `yeto/rl/engine/policy_age.py`（`bind_policy_age` 0 时身份不变、非 0 绑定进岛身份；`PolicyAgeSupport` 后端声明支持到哪个阶段，适配层角色 `policy_age`）；launcher 起机前按后端检查；`tests/test_rl_policy_age.py::test_strict_handshake_refuses_an_island_with_another_limit`、`::test_elastic_join_carries_the_bound_identity`、`::test_launcher_refuses_an_unsupported_limit_before_launch[miles|verl]`、`::test_limit_zero_keeps_identity_and_nonzero_changes_it`；hash-migration.md "S18 第 2 组"一节（默认 0 无任何哈希变化）。
- [x] 2.2 默认 0 回归：重新生成标准样本，仅契约哈希变化（验证：标准样本比对脚本）
  - 证据：`tests/test_decoupling_golden.py` 12 过且 `tests/golden/` 无改动——默认 0 连契约哈希也不变（上限原本就在 `ExecutionProfile.contract_hash` 内且默认 0；0 时不加参数、岛身份原样），比 spec 预期更严。
- [x] 2.3 execution_profile / driver 版本检查改为"落后不超过上限"，0 时行为不变（验证：单测覆盖 0、1、超限丢弃）
  - 证据：`ExecutionProfile` 新增合约 `bounded-staleness`（上限 >0 必须用它，0 必须 on-policy），上限 >0 时 `train_blockers` 允许窗口内混版本；`AlgorithmSpec` 的 staleness>0 改为要求 TIS/自定义修正；`GroupMetadata.policy_versions`（版本段）；driver 上限 0 时旧规则不变（并拒绝含旧版本段的组），上限 N 时接受窗口内且哈希为已发布版本的组、超限报错（引擎侧用 `policy_age.split_by_age` 丢弃）。测试：`tests/test_rl_policy_age.py::test_profile_limit_needs_the_bounded_staleness_contract`、`::test_split_by_age_zero_one_and_over_limit`、`::test_driver_limit_zero_keeps_refusing_older_tokens`、`::test_driver_limit_one_accepts_version_segments_within_the_window`、`::test_driver_limit_one_refuses_a_group_two_versions_old`。
- [x] 2.4 Miles 适配层由上限推导 partial-rollout 与 mask/TIS 开关，从 algorithm_flags 不映射表移除（验证：Miles 命令行标准样本，上限 0 时无变化）
  - 证据：`yeto/rl/adapters/miles/policy_age.py`（`policy_age_argv`：0 → 无，N>0 → `--partial-rollout --mask-offpolicy-in-partial-rollout`）；`algo_flag_rows` 注册两行（由 `execution.max_policy_staleness` 推导，直接传入被拒并提示用 `--rl-max-policy-age`），从 `_UNMAPPED` 移除，仍属适配层独占参数；`tests/test_rl_policy_age.py::test_miles_switches_are_derived_from_the_limit`；Miles 命令行标准样本不变（`tests/test_decoupling_golden.py`）。TIS 由算法规格的 correction 配置（staleness>0 时必需），fork 未改。

## 3. 阶段 1：版本段记账、切点与多岛账本
- [ ] 3.1 token 级生成版本与生成概率随样本携带（验证：样本序列化往返单测）
- [ ] 3.2 训练端跨版本重要性采样修正与截断比例上报（验证：CPU 数值单测，构造已知比值）
- [ ] 3.3 续训切点"在途轨迹"段，上限 0 时必须为空；以 0 恢复含在途轨迹的切点时丢弃并上报（验证：cut 单测与恢复单测）
- [ ] 3.4 多岛：样本按版本段进入 island_ledger 的 ACCEPT_IS 判定（验证：账本单测，跨 1/2 版本两种情形）
- [ ] 3.5 截断比例告警阈值与运行内自动回退到 0（只降不升）（验证：单测注入高截断比例）

## 4. 阶段 2：单轮任务续跑（门槛：阶段 0 达标 + N15 偏差已定位）
- [ ] 4.1 Miles 单轮路径开启 partial-rollout，接入版本段记账（验证：假引擎单测，跨版本续跑样本版本段正确）
- [ ] 4.2 GPU 对照：小模型单轮任务，上限 0 vs 1，比较每轮时长、截断比例、奖励曲线（上卡前复核与预算报批；验证：判据写入本组并附证据路径）

## 5. 阶段 3：agentic 多轮续跑（门槛：阶段 2 达标 + FN 前缀重算代价已测）
- [ ] 5.1 Miles fork agentic 生成循环：截止只在回合之间生效，工具执行中不中止（验证：fork 侧单测，截止落在工具执行中时等结果写回后挂起）
- [ ] 5.2 codex harness 逐条版本检查改为按段记录，不再因版本漂移中止（验证：harness 单测）
- [ ] 5.3 沙箱挂起保活与存活上限、超时丢弃并释放（验证：Modal CPU 冒烟，统计存活费用）
- [ ] 5.4 会话前缀跨版本保留；FN 无前缀缓存时测重算代价并决定 FN 是否只用阶段 0（验证：测量报告写入 design）
- [ ] 5.5 GPU 对照：M1 配置两岛，上限 0 vs 1，判据含奖励与长度分布、沙箱错误 0、存活费用（上卡前报批；验证：证据路径）

## 6. verl 并行线（与第 1–5 组同阶段门槛）
- [ ] 6.1 读码确认 verl fork fully_async 下 `tool_agent_loop` 被中止后是从头重跑还是续跑、版本记录粒度、与 yeto 多岛同步的冲突点，结论写入 design 第 9 条（验证：design 更新并附文件:行号）
- [ ] 6.2 阶段 0：verl 同步模式多发与截止，丢弃计数与三段耗时翻译成统一事件字段（验证：verl 适配层单测，字段与 Miles 一致）
- [ ] 6.3 阶段 1：verl 适配层按落后上限声明能力，0 时 verl 命令行标准样本不变；不支持的阶段启动前报错（验证：标准样本比对与报错单测）
- [ ] 6.4 阶段 2：由落后上限推导 `staleness_threshold` 与 `partial_rollout`，样本版本段接入 yeto 账本（验证：CPU 单测；GPU 对照与 4.2 合并在同一次上卡，单岛 verl 小模型上限 0 vs 1）
- [ ] 6.5 阶段 3：verl 多轮工具调用续跑（若 6.1 确认原生支持则复用，否则按第 5 组同样规则实现），GPU 对照与 5.5 合并上卡（验证：证据路径）
