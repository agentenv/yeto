# Tasks：rl-eval-difficulty-buckets

（实现排在 WP7 去耦合阶段 3 之后；判分沙箱由 WP6 实现。）

## 1. 名单与评测数据构建（离线，不上卡）

- [ ] 1.1 `tools/build_eval_holdout.py`：读 tb2-data（钉 commit）各任务 `task.toml` 的 `difficulty`，先排除 S15 冒烟 6 题（`codex-bundle/data/tbench2_smoke6.jsonl`），再固定种子分层抽 30 个（easy 2 / medium 18 / hard 10），名单写 `excluded`；与 WP6 #129 的 `tools/reward_env/holdout.py` 对齐（加 `exclude=`），写 `data/eval/tb2-holdout.json` 与评测 jsonl，记 sha256
- [ ] 1.2 同工具：读 `SWE-bench/SWE-bench_Verified@78f471bf`（组织版，与 #129 一致）`difficulty`，按 design D2 分 3 桶、桶内按仓库分层抽（30/30/全部 45），写 `data/eval/swebench-verified-eval.json` 与评测 jsonl
- [ ] 1.3 生成 TB2 训练数据（其余 59 个任务），行 `metadata` 带 D6.a 字段
- [ ] 1.5 CPU 单测：同种子同哈希、分层题数、名单与 jsonl 一致

## 2. 启动检查与配置

- [ ] 2.1 `EvalConfig` 加名单与评测数据 sha256、每题次数（第 0 轮/常规）；启动校验哈希
- [ ] 2.2 训练集与评测集交集检查（D6.c），结果进 `rl_driver_start`；CPU 单测覆盖 `task_id` 交集与 `(repo, base_commit)` 交集
- [ ] 2.3 第 0 轮次数切换（TB2 4 / SWE 2），常规（2 / 1）；设置固定检查，单测"设置被改即停"

## 3. 评测结果回传与事件

- [ ] 3.1 与 WP6 约定判分回传字段（D6.d），写进 harness 契约
- [ ] 3.2 yeto 侧按 `eval_bucket` 算通过率、自助重采样标准误、配对差、截断/回合用尽/`infra_error` 比例，写逐条 jsonl
- [ ] 3.3 ports 路径 `evaluate`（`miles_adapter/entry.py`）回传 3.2 结果；`rl_eval` 事件加字段（现有字段不变）
- [ ] 3.4 CPU 单测：假轨迹 → 指标、事件字段、逐条文件哈希

## 4. 训练批次分桶

- [ ] 4.1 `rollout_meta_hook.build_metadata` 按 `metadata.difficulty` 分组汇总，写 `batch_summary_by_bucket`；CPU 单测
- [ ] 4.2 真机核对开销（用 `rl_rollout` 时间戳），校正 design D5

## 5. 评测岛（便宜的可中断卡，D11）

- [ ] 5.1 新增"只推理的评测岛"角色：启动器与岛账本区分它与训练岛（不进合并池、不交增量）
- [ ] 5.2 训练驱动在评测版本把 adapter + manifest 写到持久存储并登记待评任务，不等评测
- [ ] 5.3 评测岛加载基座 + adapter，校验 `policy_tensor_hash` 与 `rl/policy_token` 后开评
- [ ] 5.4 逐条结果按 (`policy_version`, `task_id`, `trial`) 追加写持久存储；重启后跳过已完成、去重；被回收的进行中轨迹记 `preempted` 不计入
- [ ] 5.5 版本排队（不丢弃），事件记队列长度、滞后轮数、`eval/preemptions`
- [ ] 5.6 CPU 单测：模拟回收续跑结果与一次跑完逐位一致
- [ ] 5.7 评测岛选云调度（D11.6）：复用 `providers.NebiusSignals`/`VerdaSignals`/`AwsProviders` 探针与 `launch_with_verda_candidates`，按价格+准备成本排序、失败换下一家、兜底 Modal；写 `rl_eval_island` 事件；CPU 单测用假探针
- [ ] 5.8 持久存储（D11.7）：评测岛启动时主动下载 adapter 并校验 sha256，逐条结果追加写；存储方案与 rl-resume-from-checkpoint 对齐后定稿

## 6. 上卡准备与对接

- [ ] 6.1 训练脚本加评测参数；复核文档写 D7 成本估计并台账预登记
- [ ] 6.2 首次真机后用 `eval/wall_s` 校正 D7
- [ ] 6.3 D10 接口需求交 WP4
- [ ] 6.4 （可选，待用户定）数学固定评测集（D9）
