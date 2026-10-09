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
- [x] 1.4 GPU A/B（并入 N17 合并验证运行）：M1 run d 配置单岛 1×H200 各 6 轮，判据见 design Migration Plan；结果与证据路径写回本文件与 AGENTIC-GPU-UTIL-RESEARCH.md
  - 结果（2026-10-09 S17 N17，**判据全过**；main a11c313e，镜像 2f7871f-2fa8801，Qwen3.5-4B 单岛 1×H200，TB2 46 题，6 轮×6×4，A=多发 6、B=多发 12；证据 `openspec/changes/rl-infra-spec/evidence/s17-n17/ab-rerun-compare.json`，原始数据 s1-runs/s17-n17-ab-{a,b}-20261009b）：生成段中位 A 200.6 s → B 112.9 s（0.56×，≤0.7 通过；去掉最慢 1 轮 168.1 → 111.4，0.66×）；每轮最慢轨迹 A 109–320 s、B 95–148 s；KV 峰值 A 11.8%、B 20.6%（<60%）；SGLang 排队最大 A 2、B 37（同时运行最大 46，容量 89，未触顶）；沙箱错误 0；奖励均值 A 0.083±0.276、B 0.090；总 token 均值 A 3465±2653、B 3059（短约 12%，p50 2572→2342，在 1 个标准差内）；回合 A 7.94±4.12、B 8.24。代价：B 每轮丢弃 16–24 条、7.9 万–12.8 万回复 token（多于每轮训练用量约 7.3 万），即约 2.5 倍生成量换 44% 生成段时间；沙箱冷启动 p90 A 4.2 s、B 13.0 s。第一组（main 8a599212）暴露的三个问题（沙箱租约强制释放后整轮崩溃、abort 只停工作进程致拖尾 11 分钟、Miles 丢弃统计改在未使用的 sglang_rollout.abort）已由 #158 + Miles 2f7871fb2 修复；重跑中 abort 1 秒内完成，丢弃统计非空。

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
- [x] 3.1 token 级生成版本与生成概率随样本携带（验证：样本序列化往返单测）
  - 证据：`yeto/rl/engine/version_segments.py` `TokenProvenance`（逐 token 版本 + 生成对数概率，序列化为版本段 + 概率）；组级 `GroupMetadata.policy_versions`（第 2 组）；`tests/test_rl_version_segments.py::test_provenance_serialization_roundtrip_and_segments`。Miles 样本上的实际填充属阶段 2（4.1）。
- [x] 3.2 训练端跨版本重要性采样修正与截断比例上报（验证：CPU 数值单测，构造已知比值）
  - 证据：`cross_version_is`（当前版本 token 权重 1，旧版本 token 取 exp(训练−生成) 截断到 [下界, 上界]，统计截断比例）、`batch_truncated_fraction`；`TrainStepMetrics.cross_version_truncated_fraction`，有值时写进 `rl_round_trained`；`tests/test_rl_version_segments.py::test_cross_version_is_with_known_ratios`、`::test_driver_injected_high_truncation_falls_back_to_zero`（事件含截断比例）。Miles 训练内由 TIS（生成时概率）执行，引擎上报截断比例属阶段 2 接线。
- [x] 3.3 续训切点"在途轨迹"段，上限 0 时必须为空；以 0 恢复含在途轨迹的切点时丢弃并上报（验证：cut 单测与恢复单测）
  - 证据：`CutManifest.in_flight`（非空才序列化，默认切点字节不变）；`context_problems`：上限 0 时在途段必须为空、carried_over 仍须 0，上限 >0 时 carried_over 必须等于在途条数（上限记在 ledger.max_policy_age）；`restore_in_flight`：上限 0 全部丢弃并上报，>0 时超限或同题组已完成者丢弃；`tests/test_rl_version_segments.py::test_cut_in_flight_section_empty_at_limit_zero`、`::test_restore_with_in_flight_trajectories`（5 条在途）。
- [x] 3.4 多岛：样本按版本段进入 island_ledger 的 ACCEPT_IS 判定（验证：账本单测，跨 1/2 版本两种情形）
  - 证据：`SampleGroup.version_segments`，`judge` 按最旧段判定、每段哈希必须是已发布版本；`tests/test_rl_version_segments.py::test_ledger_judges_carried_over_groups_by_their_oldest_segment`（跨 1 版本 ACCEPT_IS；上界 1 时跨 2 版本 REJECT、上界 2 时 ACCEPT_IS；伪造段哈希 REJECT）。
- [x] 3.5 截断比例告警阈值与运行内自动回退到 0（只降不升）（验证：单测注入高截断比例）
  - 证据：`PolicyAgeGovernor`（告警 0.2、回退 0.5 默认）；driver `_govern_policy_age` 发 `rl_policy_age_warning` / `rl_policy_age_fallback`，回退后 `_max_policy_age()` 恒 0 并调用引擎可选动词 `set_max_policy_age(0)`；`tests/test_rl_version_segments.py::test_governor_warns_then_falls_back_and_never_goes_up`、`::test_driver_injected_high_truncation_falls_back_to_zero`。阈值默认值为本次拟定，阶段 2 上卡后按实测调整。

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
- [x] 6.1 读码确认 verl fork fully_async 下 `tool_agent_loop` 被中止后是从头重跑还是续跑、版本记录粒度、与 yeto 多岛同步的冲突点，结论写入 design 第 9 条（验证：design 更新并附文件:行号）
  - 证据：design 第 9 条"6.1 读码结论"（verl fork acad9875，附文件:行号）：中止只打断模型生成、不打断工具；partial_rollout 开时在推理服务客户端内保留前缀续写；版本每轨迹 min/max；staleness_threshold 按样本数限流；trainer 本地版本号与 outer version 错位；同步模式 `over_sample_rate` 无使用处。
- [x] 6.2 阶段 0：verl 同步模式多发与截止，丢弃计数与三段耗时翻译成统一事件字段（验证：verl 适配层单测，字段与 Miles 一致）
  - 范围调整（10-09 主 agent 代用户拍板）：verl 的"凑够即截止/续跑"不在同步模式做，并入 6.4（阶段 2，走 fully_async 路径）；本项完成的是统一字段翻译与起机前报错。
  - 证据：`yeto/rl/adapters/verl/rollout_events.py`（agent loop 计时 → `generation_seconds`/`tool_seconds`/`evaluate_time`，min/max_global_steps → `policy_versions`，丢弃统计 → 与 Miles 相同的 `rl_rollout_cutoff` 字段）；Miles `PhaseClock` 给出同名总量字段；verl 多发数 ≠ 批次时起机前报错（`run_config_rules.check_over_sampling`，原为静默忽略）。测试：`tests/test_rl_verl_policy_age.py::test_verl_and_miles_report_the_same_trajectory_fields`、`::test_verl_discard_tally_uses_the_miles_cutoff_fields`、`::test_verl_refuses_over_sampling_before_launch`。
- [ ] 6.2b ~~verl 同步模式截止补丁~~：**不做**（10-09 主 agent 代用户拍板）。原因：批次在 `AgentLoopManager.generate_sequences`（`agent_loop.py:1249-1275`）按 worker 切块分给多个 Ray actor，全局截止需跨 worker 计数，且要改同步训练器的批次假设（uid/重复/ppo_mini_batch 整除），代价大且只服务阶段 0 测量；阶段 0 数据用 Miles A/B 已足够。verl 的截止/续跑并入 6.4 的 fully_async 路径。
- [x] 6.3 阶段 1：verl 适配层按落后上限声明能力，0 时 verl 命令行标准样本不变；不支持的阶段启动前报错（验证：标准样本比对与报错单测）
  - 证据：`yeto/rl/adapters/verl/policy_age.py`（`SUPPORT` 阶段 1、上限 0；`policy_age_overrides(0)` 为空）；`verl/entry.py` 的 `max_policy_staleness` 改为取自 `SUPPORT`；后端注册表新增角色 `policy_age`；测试：`tests/test_rl_verl_policy_age.py::test_verl_declares_stage_one_and_limit_zero_overrides_are_empty`、`::test_verl_command_line_unchanged_at_limit_zero`、`tests/test_rl_policy_age.py::test_launcher_refuses_an_unsupported_limit_before_launch[verl]`。仓库内无 verl 命令行标准样本文件，以 `build_overrides` 逐项相等代替。
- [ ] 6.4 阶段 2：由落后上限推导 `staleness_threshold` 与 `partial_rollout`，样本版本段接入 yeto 账本（验证：CPU 单测；GPU 对照与 4.2 合并在同一次上卡，单岛 verl 小模型上限 0 vs 1）
  - 子要求（6.1 读码所得，10-09 主 agent 拍板列入）：(a) verl 前缀续写发生在推理服务客户端内、对 agent loop 不可见——适配层需自行补记逐段版本与生成概率；(b) verl 版本只有每轨迹 min/max_global_steps——翻译为版本段边界，判定按最旧版本；(c) `staleness_threshold` 只按样本数限流、不比版本号——超限丢弃由 yeto 按版本段判定，不能依赖 verl；(d) trainer 本地 `current_param_version` 与外层 outer version 错位——多岛时适配层要做映射并约束检查点/日志步号。
- [ ] 6.5 阶段 3：verl 多轮工具调用续跑（若 6.1 确认原生支持则复用，否则按第 5 组同样规则实现），GPU 对照与 5.5 合并上卡（验证：证据路径）

## 7. dashboard 可视化（生成阶段利用率；只改看板读取与显示，不改训练代码）
- [x] 7.1 每轮"截止丢弃 / 续跑"表：提交组数、目标组数、丢弃组数、丢弃条数、丢弃 token 数、过滤组数（来自 `rl_rollout_cutoff`）；字段为空显示"未知"而不是 0（验证：reducer 单测含 None 字段；JS 冒烟断言出现"未知"）
  - 证据：`yeto/dashboard/reducer.py`（`CUTOFF_KEYS`、`Reducer.utilization`）；`tests/test_dashboard_utilization.py::test_cutoff_completion_phases_and_load_per_round`（submitted_groups/discarded_tokens 为 None 保持 None）、`::test_page_draws_utilization_and_unknowns`（页面出现"未知"）。N17 A 臂第 0 轮 submitted_groups 为空，页面显示"未知"。
- [x] 7.2 每轮生成段内的轨迹完成曲线：横轴为生成段内秒数，纵轴为已完成轨迹数，标出截止点（生成段结束）；可切换轮次或叠加全部轮次（验证：reducer 单测完成时刻相对生成段起点；JS 冒烟渲染曲线与切换）
  - 证据：`utilization()` 的 `done`（相对生成段起点秒数）与 `gen_s`（截止点）；页面 `uDone`（"全部轮叠加 / 第 N 轮"切换）；单测 `done == [20.0, 40.0]`；JS 冒烟 2 轮×2 运行 = 4 条曲线、4 条截止线；N17 A/B 导出冒烟 12 条曲线。
- [x] 7.3 每轮与每条轨迹的"模型生成 / 工具执行 / 判分 / 沙箱启动"四段耗时条（`rl_trajectory_reward` 的 `generation_seconds`、`tool_seconds`、`evaluate_time`、`sandbox_start_seconds`），显示工具执行占比（验证：reducer 单测四段合计与占比；JS 冒烟）
  - 证据：`TRAJ_PHASES` 四段合计与 `tool_share`；页面 `uPh`（每轮）与 `uTr`（选定轮的每条轨迹）；单测四段合计 {16,2,1,1}、工具占比 0.1。N17 读出 A 臂各轮工具占比 0.9%–56.5%，B 臂 3.5%–9.9%。
- [x] 7.4 KV 占用与排队曲线（`rl_load_sample` 的 `kv_used_tokens/kv_capacity_tokens`、`queued_requests`、`running_requests`），标出生成段（验证：reducer 单测；JS 冒烟）
  - 证据：`load` 序列（KV 比例、排队、在跑、等工具）与每轮生成段内峰值 `peaks`；页面 `uLoad`；单测 kv 峰值 0.4、排队 5。N17 B 臂读出 KV 峰值 20.6%、排队最大 37，与 1.4 记录一致。
- [x] 7.5 为阶段 2 预留：续跑轨迹跨了哪几轮（开始轮 → 训练轮、版本段），读取 `rl_trajectory_reward` 的 `started_rollout_id` 与 `policy_versions`；没有该数据时不显示（验证：reducer 单测构造跨轮轨迹与无该字段两种情形；JS 冒烟区块隐藏）
  - 证据：`_traj` 读 `started_rollout_id`、`policy_versions`（版本段 `[版本, 起, 止)`），`carry` 无数据为 None；页面 `uCarry` 无数据隐藏；`::test_stage_two_carry_over_is_listed`、JS 冒烟 `carry_hidden`。字段名为本看板预留，阶段 2 写事件时按此命名或改这里。
- [x] 7.6 两条运行并排 / 叠加比较：`yeto dashboard serve|export --compare <磁带...> [--label A --compare-label B]`，用 N17 两臂做示例（验证：reducer 单测；N17 A/B 本机起看板并截图存 infra-drafts/dash-aru-screenshots/）
  - 证据：`cli._attach_compare`（`--compare/--label/--compare-label`），`page_view()["compare"]`；`::test_compare_run_is_attached`；N17 A/B 本机 `serve --port 8797` 验证 `/api/view` 带 compare（已关）。离线页 infra-drafts/dash-aru-screenshots/n17-ab-compare.html。截图：本机无无头浏览器（无 chromium/playwright/puppeteer，也无 SVG 光栅化库），按要求未装大依赖，未截图——直接用浏览器打开该离线页查看。
- [x] 7.7 旧磁带兼容：没有新事件/新字段时页面不报错，利用率区块显示"无数据"（验证：既有 JS 冒烟夹具与 test_dashboard_* 全过）
  - 证据：`::test_old_tape_has_no_utilization`、`::test_page_hides_utilization_for_old_tapes`；`tests/test_dashboard_*.py` 共 90 项全过（含原有冒烟夹具）。
