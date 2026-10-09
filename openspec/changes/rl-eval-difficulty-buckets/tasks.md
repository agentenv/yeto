# Tasks：rl-eval-difficulty-buckets

（实现排在 WP7 去耦合阶段 3 之后；判分沙箱由 WP6 实现。）

## 1. 名单与评测数据构建（离线，不上卡）

- [ ] 1.1 `tools/build_eval_holdout.py`：读 tb2-data（钉 commit）各任务 `task.toml` 的 `difficulty`，先排除 S15 冒烟 6 题（`codex-bundle/data/tbench2_smoke6.jsonl`），再固定种子分层抽 30 个（easy 2 / medium 18 / hard 10），名单写 `excluded`；与 WP6 #129 的 `tools/reward_env/holdout.py` 对齐（加 `exclude=`），写 `data/eval/tb2-holdout.json` 与评测 jsonl，记 sha256。**部分**：名单用 #131 的 `tools/reward_env/holdout.py` 生成 `data/eval/tb2-holdout.json`（30 题，排除冒烟 6 题，sha256 `28d6730a…`，与 #131 记录一致）；`tools/build_eval_holdout.py` 未单独写（#131 工具已覆盖），**评测 jsonl 未生成**（S17 N9，分支 s17-eval-island）
- [ ] 1.2 同工具：读 `SWE-bench/SWE-bench_Verified@78f471bf`（组织版，与 #129 一致）`difficulty`，按 design D2 分 3 桶、桶内按仓库分层抽（30/30/全部 45），写 `data/eval/swebench-verified-eval.json` 与评测 jsonl
- [ ] 1.3 生成 TB2 训练数据（其余 59 个任务），行 `metadata` 带 D6.a 字段
- [ ] 1.5 CPU 单测：同种子同哈希、分层题数、名单与 jsonl 一致

## 2. 启动检查与配置

- [ ] 2.1 `EvalConfig` 加名单与评测数据 sha256、每题次数（第 0 轮/常规）；启动校验哈希
- [x] 2.2 训练集与评测集交集检查（D6.c），结果进 `rl_driver_start`；CPU 单测覆盖 `task_id` 交集与 `(repo, base_commit)` 交集。`yeto/rl/eval/guard.py`：task_id、`(repo, base_commit)`、规范化 problem_statement sha256 三种交集，名单 sha256 钉值，评测数据多题/缺题都拒绝；结果以 `eval_guard` 写进 `rl_driver_start`（未配置时不出现，标准样本不变）。接入：环境变量 `YETO_RL_EVAL_HOLDOUT=路径[@sha256]`（+ `YETO_RL_EVAL_DATA`），在 `entry.run_ports_island` 连 Ray 之前检查（`adapters/miles/eval_wiring.py`，不新增启动参数，不改 island_entry/launcher）。单测 `tests/test_rl_eval_island.py`（S17 N9，分支 s17-eval-island）
- [ ] 2.3 第 0 轮次数切换（TB2 4 / SWE 2），常规（2 / 1）；设置固定检查，单测"设置被改即停"

## 3. 评测结果回传与事件

- [ ] 3.1 与 WP6 约定判分回传字段（D6.d），写进 harness 契约
- [x] 3.2 yeto 侧按 `eval_bucket` 算通过率、自助重采样标准误、配对差、截断/回合用尽/`infra_error` 比例，写逐条 jsonl。`yeto/rl/eval/stats.py`：按桶通过率（每题等权，第 0 轮 4 次与常规 2 次同一尺度）、按题自助重采样标准误（固定种子）、相对第 0 轮同题配对差及标准误、截断/回合用尽/超时/infra_error 比例；逐条结果 jsonl 见 5.4（S17 N9，分支 s17-eval-island）
- [ ] 3.3 ports 路径 `evaluate`（`miles_adapter/entry.py`）回传 3.2 结果；`rl_eval` 事件加字段（现有字段不变）
- [x] 3.4 CPU 单测：假轨迹 → 指标、事件字段、逐条文件哈希。`tests/test_rl_eval_island.py`（假判分、假推理，24 项）（S17 N9，分支 s17-eval-island）

## 4. 训练批次分桶

- [x] 4.1 `rollout_meta_hook.build_metadata` 按 `metadata.difficulty` 分组汇总，写 `batch_summary_by_bucket`；CPU 单测。`rollout_meta_hook.batch_summary_by_bucket`（按 `metadata.difficulty`，缺失记 unknown；条数、reward 均值、成功率、截断比例、平均长度），只有样本带难度时才写 `batch_summary_by_bucket`；经 `RolloutBatch.batch_summary_by_bucket` 进 `rl_round_trained`（与 batch_summary 同一事件，D5 原写 `rl_rollout`，以实际事件为准）。单测 `tests/test_rl_eval_batch_buckets.py`。标准样本只变 rollout_meta_hook 源码哈希（见 yeto-framework-decoupling/hash-migration.md）（S17 N9，分支 s17-eval-island）
- [ ] 4.2 真机核对开销（用 `rl_rollout` 时间戳），校正 design D5

## 5. 评测岛（第一版只用 Modal，D11）

- [ ] 5.1 新增"只推理的评测岛"角色：启动器与岛账本区分它与训练岛（不进合并池、不交增量）。**部分**：评测岛是独立进程（`yeto/rl/eval/island.py`），事件带 `role=eval`、`source=eval_island`，不调用 syncer、不进岛账本；**启动器与岛账本里没有加"评测岛"角色字段**（第一版评测岛不经 launcher 启动，由 `yeto/cloud/modal_eval_island.py` 直接起）（S17 N9，分支 s17-eval-island）
- [x] 5.2 训练驱动在评测版本把 adapter + manifest 写到持久存储并登记待评任务，不等评测。`yeto/rl/eval/export.py` `StoreEvalExporter` 挂在 `IslandDriver.eval_export`：第 0 轮、每 N 轮（`YETO_RL_EVAL_STORE_INTERVAL`，默认 10）、最后一轮写 canonical 张量 + 身份 + manifest（每个文件 sha256、policy_tensor_hash、rl/policy_token、采样设置）再登记队列，不等评测；发 `rl_eval_export` 事件（S17 N9，分支 s17-eval-island）
- [ ] 5.3 评测岛加载基座 + adapter，校验 `policy_tensor_hash` 与 `rl/policy_token` 后开评。**部分**：评测岛先核对每个文件 sha256、采样设置与计划一致，加载后要求服务端 token 等于 manifest；`verify_policy_tensor_hash` 从存下的文件回算 policy_tensor_hash（CPU 单测用 torch 验过）。真实"加载基座 + adapter 到推理引擎"未实现（只有 loader 端口）（S17 N9，分支 s17-eval-island）
- [x] 5.4 逐条结果按 (`policy_version`, `task_id`, `trial`) 追加写持久存储；重启后跳过已完成、去重；被回收的进行中轨迹记 `preempted` 不计入。`EvalStore` 逐条日志：每次尝试前写 start、后写 result；只有 start 没有 result 的记为被回收（计入 `eval/preemptions`、不计分、重跑）；同一单位多条 result 保留第一条（S17 N9，分支 s17-eval-island）
- [x] 5.5 版本排队（不丢弃），事件记队列长度、滞后轮数、`eval/preemptions`。评测岛按版本从旧到新消化队列，不丢弃；`rl_eval` 带 `eval/queue_len`、`eval/lag_rounds`、`eval/preemptions`；训练结束标记文件出现后再查一次队列才退出（S17 N9，分支 s17-eval-island）
- [x] 5.6 CPU 单测：模拟回收续跑结果与一次跑完逐位一致。`test_preempted_resume_equals_one_shot`：两次模拟回收后续跑，逐条结果哈希与一次跑完相同，`rl_eval` 除回收次数外逐字段相同（S17 N9，分支 s17-eval-island）
- [ ] 5.7 评测岛起岛（D11.6）：第一版只在 Modal 起 8×H200 评测岛，被抢占后在 Modal 重起续跑；写 `rl_eval_island` 事件；CPU 单测用假 Modal 客户端。多云 spot 选云为后续选项，归 rl-infra-spec 第 8 节。**部分**：`yeto/cloud/modal_eval_island.py` `EvalIslandLauncher`：起岛失败/被回收后在 Modal 重起（默认最多 5 次），每次写 `rl_eval_island`（云、卡型、卡数、单价、等待/运行时长、结果、错误）；`island_main` 在容器内跑评测岛，loader/attempt 由点路径注入。单测用假 Modal 客户端。**真实 GPU 函数定义与推理 agent 未写，未上卡**（S17 N9，分支 s17-eval-island）
- [x] 5.8 持久存储（D11.7，已定 Modal Volume）：训练驱动在评测版本把 adapter + manifest 写入 Modal Volume（训练岛不在 Modal 时经 Modal API 上传），评测岛挂载后校验 sha256 与 policy_token，逐条结果追加写。`store_from_env`：Volume 已挂载时用 `commit/reload`；未挂载时写本地暂存目录、每次提交经 Modal API 上传新文件（manifest 与队列最后传）。CPU 冒烟（2026-10-08 17:33Z）真实上传到 Volume `yeto-eval-store-smoke2` 并评 TB2 留出集 3 题（v0 不做事 0/3，v10 官方解 3/3，模拟回收后续跑 preemptions=1，配对差 +1.0），证据 `s1-runs/s17-n9-evalisland-smoke/`；评测岛容器内挂载 Volume 的路径未实测（S17 N9，分支 s17-eval-island）

## 5b. 尚未实现（C6b 之后）

- [ ] 5.9 评测用推理 agent：在评测岛里起 SGLang（基座 + adapter），用与训练相同的 harness/采样设置对着判分沙箱多回合操作，实现 `tb2_attempt` 的 agent 端口；实现后再做 GPU 冒烟（原计划：小模型、TB2 2–3 题、Modal 最少卡、上限 $10）
- [ ] 5.10 Modal GPU 函数定义（镜像、Volume 挂载、`island_main` 入口）与训练结束标记

## 6. 上卡准备与对接

- [ ] 6.1 训练脚本加评测参数；复核文档写 D7 成本估计并台账预登记
- [ ] 6.2 首次真机后用 `eval/wall_s` 校正 D7
- [ ] 6.3 D10 接口需求交 WP4
