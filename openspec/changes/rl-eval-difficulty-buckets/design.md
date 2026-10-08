# Design：按难度分桶的固定评测集

（第 4 版，2026-10-08，按 S17 用户裁定改：评测集以 Terminal-Bench 2 留出集与 SWE-bench Verified 为主；不用 Qwen3-32B 通过率；数学降为可选；SWE-Gym 暂不采用；SWE-bench Verified 统一用 SWE-bench 组织版；TB2 评测池排除冒烟 6 题；评测放在便宜的可中断卡上（D11）。）

## Context

- 现有评测通路：驱动 `_maybe_eval`（`yeto/rl/engine/driver.py` 约 1098 行）每 `eval_interval` 轮调一次 `evaluate`，发 `rl_eval` 事件；初始发布后会以 `force=start.rollout_id == 0` 调一次（约 1556 行），即第 0 轮基座评测已有。训推分离 + `--yeto-rl-overlap-eval` 时走 `overlap.py`，评测与训练并行且评的一定是刚发布的版本。
- ports 路径的 `evaluate`（`miles_adapter/entry.py` 约 1682 行）调 Miles `EvalDispatcher.dispatch` 后**返回空字典**，数值只进 Miles 日志，`rl_eval` 事件里没有。
- codex 训练数据行现在的形状（`codex-bundle/data/tbench2_smoke6.jsonl`）：`{"prompt": [...], "metadata": {"task_id": "fix-git"}}`。训练批次的逐条奖励已有 `rl_trajectory_reward` 事件（task_id/trajectory_id/reward/success，`rollout_meta_hook.trajectory_reward_records`），训练批次汇总在 `rollout_meta_hook.build_metadata` 里（约 321 行）。
- 判分环境：TB2 在 Modal 常驻沙箱 yeto-tbench2，每任务判分 2.5–4 分钟（FNCODEX-STAGE2-ANALYSIS.md）；SWE-bench Verified 的沙箱**由 WP6 在 Modal 上做**，本 change 只写接口需求（D6）。
- 参考耗时（S15 FN codex 实测，FNCODEX-STAGE2-ANALYSIS.md §5）：8×H200（$46.01/h），6 任务 × 4 样本 = 24 条轨迹，单轮"生成 + 训练"约 35 分钟，上下文 8192。

## Goals / Non-Goals

**Goals：**
- agentic 固定评测集：TB2 留出 30 个任务 + SWE-bench Verified 按官方耗时分档抽题；钉名单与哈希；与训练集有交集即拒绝启动。
- 第 0 轮 + 每 10 轮 + 最后一轮评测；分桶记录通过率、截断（上下文用尽/回合用尽）比例、轨迹长度；逐条原始结果落盘；进 `rl_eval` 事件。
- 训练批次按同一套难度字段分桶统计。
- 字段对 harness 中立。

**Non-Goals：**
- 不实现判分沙箱（WP6）、dashboard（WP4）。
- 不按难度做课程学习或采样加权。
- 本 change 不改代码、不上卡。

## Decisions

### D1 难度从哪来

用**基准自带的官方难度字段**，不用任何外部模型打分（用户裁定：先不用 Qwen3-32B 通过率）：

| 基准 | 字段 | 分布（已核） | 来源 |
|---|---|---|---|
| Terminal-Bench 2.0（89 任务，Apache-2.0） | 每任务 `task.toml` 的 `difficulty` | easy 4 / medium 55 / hard 30 | 本地 tb2-data commit 2fd12b8 |
| SWE-bench Verified（500 题） | HF 数据 **`SWE-bench/SWE-bench_Verified@78f471bf`**（SWE-bench 组织版，与 WP6 #129 一致）的 `difficulty` 列（OpenAI 人工标注修复耗时） | `<15 min fix` 194 / `15 min - 1 hour` 261 / `1-4 hours` 42 / `>4 hours` 3 | HF datasets-server 统计，2026-10-08 |

注：不用 `princeton-nlp/SWE-bench_Verified`。据 WP6（#129 design D7.1）核对，两版 500 个 `instance_id`、`difficulty`、补丁、`base_commit` 相同，但**有 2 题的 FAIL_TO_PASS / PASS_TO_PASS 测试列表不同**；官方判分工具 swebench 5.0.2 只认组织版字段。两版数据卡都没写许可证。

第 0 轮评测得到的"基座通过率"也逐题入档，作第二口径（离线分析用，不改桶）。

### D2 评测集组成、每档题数、每题评几次

**TB2 留出 30 个任务**（只评测、永远不训练；训练用其余 59 个）：
- **先从评测池排除 S15 冒烟 6 题**（`codex-bundle/data/tbench2_smoke6.jsonl`：fix-git〔easy〕、regex-log、sqlite-db-truncate、log-summary-date-ranges、openssl-selfsigned-cert、git-multibranch〔均 medium〕）——它们在 S15 两次上卡里当训练题用过，策略可能已见过。排除后评测池 83 个：easy 3 / medium 50 / hard 30。
- 再按官方难度分层抽 30 个：easy 2 / medium 18 / hard 10（与 #129 已实现的桶数一致，只需加 `exclude=` 冒烟 6 题；按评测池比例严格分配是 1/18/11，差别不大，取前者以免 easy 只剩 1 题）。
- 冒烟 6 题仍可留在训练集。桶：`tb2-easy`、`tb2-medium`、`tb2-hard`；easy 只有 2 题，单独显示但不看趋势，dashboard 默认与 medium 合并显示为"easy+medium"。

**SWE-bench Verified（只做评测，全部不进训练）**，3 桶：
| 桶 | 官方档 | 全集 | 抽取 |
|---|---|---|---|
| `swev-lt15m` | `<15 min fix` | 194 | 30 |
| `swev-15m-1h` | `15 min - 1 hour` | 261 | 30 |
| `swev-ge1h` | `1-4 hours` + `>4 hours` | 42 + 3 | 全部 45 |
（>4 小时只有 3 题，单独成桶没有统计意义，并入 ≥1 小时。各桶内按仓库分层抽，避免 django（全集 231 题）占满；固定种子。）

**每题评几次**：
- 第 0 轮（基座）：TB2 每任务 4 次，SWE 每题 2 次——同时得到基座通过率。
- 之后每次：TB2 每任务 2 次（30×2=60 条），SWE 每题 1 次（105 条）。共 165 条轨迹/次。
- 噪声（只算采样噪声的上界）：TB2 每次 60 条，通过率标准误 ≤0.065；SWE 每桶 30–45 条，≤0.09。跨版本比较用"同题配对差"，噪声比这小；仍偏大，D7 给加量选项。

### D3 评测时机与采样设置

- 第 0 轮、每 10 轮（`eval_interval=10`）、最后一轮。
- 采样参数与训练一致且全程固定（温度、top_p、上下文上限、reasoning effort、最大回合数）。任何一项变化 = 另一把尺子，运行报错停止。
- 训推分离时评测与训练并行（`--yeto-rl-overlap-eval`）；同卡串行。

### D4 分开"题目难度"与"策略变化"

1. 固定题、固定设置：跨版本变化只来自策略。
2. 按官方难度分桶看，不同难度的变化不互相抵消。
3. 相对第 0 轮的同题配对差值及标准误（按题自助重采样）。
4. 训练批次分桶（D5）用来解释训练奖励起伏，不替代评测。

### D5 训练批次按难度分桶统计

- 口径：训练数据行 `metadata` 里的 `difficulty`（TB2 训练用的 59 个任务取其 `task.toml` 的官方难度；以后的 SWE-Gym 等训练来源**没有官方难度**，标 `unknown`，见 D8）。
- 位置：`rollout_meta_hook.build_metadata` 里现有训练批次汇总处，按 `difficulty` 分组，每组给 reward 均值、成功率、截断比例、轨迹长度、条数，写进 `rl_rollout` 事件 `batch_summary_by_bucket`。
- **开销（查代码 + 本机实测）**：该处在 Miles rollout 进程里，时机是一轮生成全部结束之后、样本交给训练之前，只读样本标量（不碰词元、张量）。本机 CPU 微基准（`infra-drafts/wp3-bucket-overhead-bench.py`，假样本、单线程、不起 Ray）：每轮 256 个样本分组 + 汇总约 1.0 ms（含按题干算 sha256 查表；直接读 `metadata.difficulty` 会更少），1024 个约 4.3 ms，送驱动约 1.2 KB/轮。codex 每轮条数远少于 256（S15 为 24 条），每轮数十分钟，开销可忽略。真机**未验证**。

### D6 接口需求（判分环境由 WP6 在 Modal 沙箱实现）

**a. 每题的难度字段**（训练与评测数据行都带，放 `metadata`）：
```json
{"metadata": {"task_id": "fix-git", "benchmark": "tb2", "benchmark_version": "tb2@2fd12b8",
              "difficulty": "medium", "difficulty_source": "tb2-task.toml",
              "eval_bucket": "tb2-medium"}}
```
SWE-bench Verified：`task_id` = 官方 `instance_id`（如 `django__django-11099`），`benchmark`=`swebench-verified`，`benchmark_version` = `SWE-bench/SWE-bench_Verified@78f471bf`，`difficulty` 为官方原文（如 `15 min - 1 hour`），`eval_bucket` 为 D2 的桶名。训练行可没有 `eval_bucket`；没有官方难度的训练来源写 `"difficulty": "unknown"`。

**b. 留出名单文件格式**（构建工具生成，进仓库 `data/eval/`，运行配置钉其 sha256）：
```json
{"schema": "yeto-eval-holdout/1",
 "benchmark": "tb2", "benchmark_version": "tb2@2fd12b8",
 "seed": 20261008, "rule": "exclude smoke6 (S15 trained); stratified by difficulty: easy 2 / medium 18 / hard 10",
 "excluded": [{"task_id": "regex-log", "reason": "S15 smoke training"}],
 "items": [{"task_id": "...", "difficulty": "hard", "eval_bucket": "tb2-hard"}]}
```
每个基准一份（`tb2-holdout.json`、`swebench-verified-eval.json`；WP6 #129 的 `tools/reward_env/holdout.py` 已按此格式实现，需加上 `excluded` 字段与 `exclude=` 参数，生成后的哈希会变）；SWE 的文件 `rule` 写明"全部只评测"。另生成评测数据文件 jsonl（行格式同 a），sha256 也钉进运行配置。

**c. "训练集与评测集有交集就拒绝启动"的检查**（驱动启动时、在任何上卡动作之前，CPU 上做）：
1. 读训练数据所有行的 `task_id`，与所有留出/评测名单的 `task_id` 求交集，非空即拒绝，报出交集个数与前 5 个。
2. 对 SWE 类：再按 `(repo, base_commit)` 与 `problem_statement` 规范化后（去空白、小写）的 sha256 各比一次——不同数据集可能给同一个 issue 不同编号。
3. 留出名单 sha256 与运行配置不符、评测数据中有名单外的题、名单中的题在评测数据里缺失，都拒绝启动。
4. 检查结果（训练行数、各名单题数、交集 0、各文件 sha256）写进 `rl_driver_start` 事件。

**d. 判分结果回传**（WP6 实现，供本 change 读）：每条评测轨迹至少给 `task_id`、`trajectory_id`、`policy_version`、`reward`、`success`、`end_reason`（completed/max_turns/max_seq_len/timed_out/infra_error）、`turns`、`tokens`；判分基础设施出错（沙箱起不来等）记 `infra_error`，**不计入通过率**，单列比例。

### D7 评测成本（全部为**估计，未验证**）

- 锚点：S15 实测 24 条轨迹的"生成 + 训练"约 35 分钟（上下文 8192；WP6 计划 16384 会更长）。假设一台 8×H200 推理节点能同时跑约 24–32 条轨迹（未测），每波约 30 分钟。
- 每次常规评测 165 条 ≈ 6–7 波 ≈ **3–3.5 小时 8×H200**；第 0 轮 330 条约 6–7 小时。
- 按评测岛单价（D11.5）：
  | 卡源 | 8×H200 单价 | 每次常规评测 | 第 0 轮 |
  |---|---|---|---|
  | Modal（GPU 本来就是可抢占价） | 官方价目 $36.3/h；台账实际口径 $46.0/h（含 CPU/内存等） | $110–160 | $220–320 |
  | AWS spot p5en/p5e（us-east-2） | 7 天中位 $28.6–32.0/h | $85–110 | $170–220 |
  | 自有集群 | 未知 | 未知 | 未知 |
- 另加判分沙箱（Modal CPU）：WP6 估 SWE 每次约 $2；TB2 未估。
- 评测岛与训练并行（D11），**不再拖慢训练**；上表即全部额外花费。被回收的重跑损失按 D11.3 只损失进行中的那几条。
- 减量选项：SWE 每桶 20 题（共 85 题）或 TB2 每任务 1 次。第一次真机后用 `eval/wall_s` 校正。

### D8 训练数据来源与 SWE-bench Verified 是否重叠

- **用户裁定：SWE-Gym 暂不采用**，不进 tasks；这里只保留核对方法，以后任何 SWE 类训练来源接入前都按此核对。
- SWE-bench Verified 只评测。曾考虑的训练候选 SWE-Gym（`SWE-Gym/SWE-Gym`，2,438 条，HF 数据卡 MIT）：仓库为 pandas 737、moto 343、MONAI 374、mypy 257、dvc 225、dask 145、modin 107、pydantic 83、conan 75、hydra 66、bokeh 26；SWE-bench Verified 的 12 个仓库是 django、sympy、sphinx、matplotlib、scikit-learn、astropy、xarray、pytest、pylint、requests、seaborn、flask。**按仓库名比对两者无交集**（2026-10-08 用 HF 统计核对）。
- 核对方法（接入任何 SWE 类训练来源时由构建工具执行并入档）：① `instance_id` 求交集；② `repo` 求交集；③ `(repo, base_commit)`；④ `problem_statement` 规范化 sha256；⑤ 补丁涉及的文件路径 + 仓库（防同一仓库的近似题）。任一非空则从训练集剔除并记录；结果写进复核文档。SWE-smith 等其他来源同样处理（SWE-smith 的仓库与 SWE-bench 是否重叠**未核**）。
- SWE-Gym 没有官方难度字段，训练批次分桶对它只能标 `unknown`；以后可用第 0 轮或首次出现时的组内通过率另行分组，但那是策略相关口径，单列不混。

### D9 数学（dapo-math-17k）：降为可选，不分桶

- 理由：(1) 用户定先不用 Qwen3-32B 通过率，而 dapo-math-17k 本身没有难度或来源字段（上一版已核：zhuzilin 版只有 `prompt`/`label`，原版 `data_source`、`ability` 全同值）；(2) 自己用基座离线采样分桶约 $130（估计），给训练批次分桶要覆盖全部训练题约 $3,000（估计），超预算；(3) agentic 是后续主线。
- 但 FN 全量训练的数据目前仍是 dapo-math，**完全不评也不行**。可选做法（零额外采样成本）：随机留出 200 题从训练集删除，作为不分桶的固定数学评测集（每题 4 次，第 0 轮 8 次）；第 0 轮结束后按基座逐题通过率（0 / 部分 / 全对）事后分组，组别在第 0 轮后冻结。不做训练批次分桶。是否启用由用户定；接口与 D6 相同（`benchmark`=`dapo-math-17k`）。

### D10 dashboard 接口需求（WP4 实现）

- 读 `rl_eval`：横轴版本（轮），每桶一条线 + 标准误带；TB2 与 SWE 分两张图；点数 <5 不连线。
- 每桶截断/回合用尽比例、`infra_error` 比例；相对第 0 轮差值。
- 训练批次 `batch_summary_by_bucket`：按难度堆叠的训练成功率。
- `eval/set_sha256`、`eval/sampling` 变化时断线提示。多岛时事件带岛编号，不跨岛平均。

### D11 评测放在便宜的可中断卡上

**D11.1 评测单独起一个推理岛**：只做推理、不参与训练、不进合并池（不向 syncer 交增量，岛账本里不算成员）。按去耦合 design 的可中断性分级属于 **I2（推理/评估优先 spot）**：没有需要保存的训练状态，被回收只损失进行中的轨迹。现在的代码里没有"只推理的岛"这种角色（岛都是训练岛），需新增。

**D11.2 评测岛怎么拿到当前策略**——两条路比较：
- a. 复用训推分离的发布路径（训练节点经 NCCL 直接推给推理引擎）：要求评测引擎与训练节点在同一集群网络里。Modal 上集群内任一容器被抢占会让**整个集群重来**（CLOUD-OPTIONS-S16 §A，引 Modal 文档），可抢占的评测若与训练同集群，被回收会连带训练重启，违背"评测不拖累训练"；跨云（AWS spot、自有集群）也没有这条网络。**不采用。**
- b. **从持久存储加载（推荐）**：训练驱动在每个评测版本（第 0、10、20…轮与最后一轮）把已发布的 LoRA adapter（FN 约 11.2 GB）连同 manifest（`policy_tensor_hash`、`rl/policy_token`、版本号）写到持久存储（Modal Volume / S3 / Nebius 共享盘，按评测岛所在云选）。评测岛加载基座 + 该 adapter，校验哈希与 token 一致后才开评，事件里记录所评的 token。只在评测版本写，每 10 轮一次；写出耗时估计数十秒（**未测**），且与 WP1 发布提速（去掉每轮整份导出）不冲突——只在评测版本额外导出。

**D11.3 被回收时怎么办**：评测是幂等的。
- 结果单位是"策略版本 + 题号 + 第几次"（`policy_version`、`task_id`、`trial`），每完成一条立刻追加写到持久存储的逐条文件。
- 评测岛重启（或换一台卡）后读已有逐条文件，跳过已完成的单位，只续跑剩下的；同一单位出现多条时按此三元组去重，保留第一条完整结果。
- 判分沙箱是 Modal CPU 沙箱，Modal 文档写明不带 GPU 的沙箱不被抢占；推理端被回收时进行中的轨迹作废重跑（标 `end_reason=preempted`，不计入结果）。
- 某版本全部单位完成后才发 `rl_eval` 事件（带 `eval/preemptions` 次数）；不完整的版本不出指标。

**D11.4 与训练并行还是串行**：并行。训练驱动只负责"在评测版本把 adapter 写到持久存储并登记待评任务"，**不等评测**。评测岛按队列逐个评；若上一版本还没评完、新的评测版本又到了，新版本排队（不丢弃，队列长度和滞后轮数写进事件）。训练结束后评测岛把最后一版评完再退出。同卡单岛小规模验证时仍可用现有串行评测。

**D11.5 卡源与价格**（已查，2026-10-08）：
- **Modal**：官方文档（modal.com/docs/guide/preemption）写明"所有 Modal Function 默认可被抢占"，"`nonpreemptible` 不支持 GPU Function"，非抢占只对 CPU/内存加 3 倍价。也就是说 **Modal GPU 本来就是可抢占的价格，没有更便宜的 spot 档**。价目（modal.com/pricing）H200 $0.001261/秒 ≈ $4.54/卡/时，8 卡 ≈ $36.3/h；我们台账的实际口径是 $46.01/h（8×H200，含 CPU/内存等，FNCODEX-STAGE2-ANALYSIS §5）。被抢占时 Modal 发中断信号并在同一输入上重启，宽限期长度文档没写（**未知**）。
- **AWS spot**（CLOUD-OPTIONS-S16 §B，已核）：p5en/p5e（8×H200）us-east-2 七天中位 $28.6–32.0/h（按需 $63.3）；近 30 天回收频率 p5en us-east-2 5–10%；**放置分数全部为 1（很难拿到）**；P 类 spot 配额 us-east-2 384 vCPU，够 2 台。评测岛只要 1 台，配额够；能否拿到是主要风险。
- **自有集群**：约 10-10 到货，型号、价格、是否可抢占**未知**；到货后作为评测岛首选候选（边际成本可能最低）。
- 推荐顺序：先 Modal（已接好、价格与训练相同但不拖慢训练）；AWS spot 能拿到时更便宜约 25–40%；自有集群到货后再评估。

## Risks / Trade-offs

- 评测成本高（D7）；放可中断卡后不拖慢训练，但 AWS spot 可能拿不到。
- 评测岛是新角色（只推理的岛），需要启动器与岛账本区分它与训练岛。
- TB2 留出后训练只剩 59 个任务，训练任务偏少；SWE-Gym 暂不采用，补训练来源以后另议。
- SWE-bench Verified 的 HF 数据卡未写许可证（代码仓库 MIT）。
- 官方难度是人工估计，METR 发现部分 `<15 min` 题实际耗时远超 15 分钟——所以第 0 轮基座通过率作为第二口径一起入档。
- 实现会碰 `driver.py`、`entry.py`、`rollout_meta_hook.py`，排在 WP7 阶段 3 之后。

## Open Questions（需用户拍板）

已定：TB2 + SWE-bench Verified（组织版 78f471bf）为主；TB2 先排除冒烟 6 题再留出 30（分层）；SWE-bench Verified 只评测；SWE-Gym 暂不采用；不用 Qwen3-32B 通过率；每 10 轮评一次；训练批次按基准难度字段分桶；评测放便宜的可中断卡。

1. 评测成本（D7：Modal 每次约 $110–160，第 0 轮约 $220–320；AWS spot 约便宜 25–40%）是否接受，或减量？
2. 评测岛卡源先用 Modal，还是先试 AWS spot（放置分数 1，可能拿不到）？
3. 评测版本额外把 adapter 写到持久存储（D11.2b），用哪个存储（Modal Volume / S3 / Nebius 共享盘）？
4. 数学固定评测集（D9 可选，不分桶）要不要启用？
