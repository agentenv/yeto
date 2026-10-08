# Design：按难度分桶的固定评测集

（第 3 版，2026-10-08，按 S17 用户裁定改：评测集以 Terminal-Bench 2 留出集与 SWE-bench Verified 为主；不用 Qwen3-32B 通过率；数学降为可选。上一版的数学分桶方案与调研保留在 D9，供以后参考。）

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
| SWE-bench Verified（500 题） | HF 数据 `princeton-nlp/SWE-bench_Verified` 的 `difficulty` 列（OpenAI 人工标注修复耗时） | `<15 min fix` 194 / `15 min - 1 hour` 261 / `1-4 hours` 42 / `>4 hours` 3 | HF datasets-server 统计，2026-10-08 |

第 0 轮评测得到的"基座通过率"也逐题入档，作第二口径（离线分析用，不改桶）。

### D2 评测集组成、每档题数、每题评几次

**TB2 留出 30 个任务**（只评测、永远不训练；训练用其余 59 个）：按官方难度分层，easy 2 / medium 18 / hard 10（easy 全集只有 4 个，只能取 2）。桶：`tb2-easy`、`tb2-medium`、`tb2-hard`；easy 只有 2 题，单独显示但不看趋势，dashboard 默认与 medium 合并显示为"easy+medium"。

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
SWE-bench Verified：`task_id` = 官方 `instance_id`（如 `django__django-11099`），`benchmark`=`swebench-verified`，`benchmark_version` = HF 数据集修订号，`difficulty` 为官方原文（如 `15 min - 1 hour`），`eval_bucket` 为 D2 的桶名。训练行可没有 `eval_bucket`；没有官方难度的训练来源写 `"difficulty": "unknown"`。

**b. 留出名单文件格式**（构建工具生成，进仓库 `data/eval/`，运行配置钉其 sha256）：
```json
{"schema": "yeto-eval-holdout/1",
 "benchmark": "tb2", "benchmark_version": "tb2@2fd12b8",
 "seed": 20261008, "rule": "stratified by difficulty: easy 2 / medium 18 / hard 10",
 "items": [{"task_id": "...", "difficulty": "hard", "eval_bucket": "tb2-hard"}]}
```
每个基准一份（`tb2-holdout.json`、`swebench-verified-eval.json`）；SWE 的文件 `rule` 写明"全部只评测"。另生成评测数据文件 jsonl（行格式同 a），sha256 也钉进运行配置。

**c. "训练集与评测集有交集就拒绝启动"的检查**（驱动启动时、在任何上卡动作之前，CPU 上做）：
1. 读训练数据所有行的 `task_id`，与所有留出/评测名单的 `task_id` 求交集，非空即拒绝，报出交集个数与前 5 个。
2. 对 SWE 类：再按 `(repo, base_commit)` 与 `problem_statement` 规范化后（去空白、小写）的 sha256 各比一次——不同数据集可能给同一个 issue 不同编号。
3. 留出名单 sha256 与运行配置不符、评测数据中有名单外的题、名单中的题在评测数据里缺失，都拒绝启动。
4. 检查结果（训练行数、各名单题数、交集 0、各文件 sha256）写进 `rl_driver_start` 事件。

**d. 判分结果回传**（WP6 实现，供本 change 读）：每条评测轨迹至少给 `task_id`、`trajectory_id`、`policy_version`、`reward`、`success`、`end_reason`（completed/max_turns/max_seq_len/timed_out/infra_error）、`turns`、`tokens`；判分基础设施出错（沙箱起不来等）记 `infra_error`，**不计入通过率**，单列比例。

### D7 评测成本（全部为**估计，未验证**）

- 锚点：S15 实测 24 条轨迹的"生成 + 训练"约 35 分钟（上下文 8192；WP6 计划 16384 会更长）。假设一台 8×H200 推理节点能同时跑约 24–32 条轨迹（未测），每波约 30 分钟。
- 每次常规评测 165 条 ≈ 6–7 波 ≈ **3–3.5 小时、约 $140–160**（8×H200 $46/h）；第 0 轮 330 条（TB2 120 + SWE 210）约翻倍，约 **$280–320**。另加 Modal CPU 沙箱费用（判分，未估）。
- 相对训练：codex 每轮约 35 分钟，10 轮约 6 小时；评测串行时多约 50%，**很贵**。缓解：(1) 评测与训练并行（训推分离）；(2) 评测按"可中断性 I2"放 spot/便宜推理卡；(3) 减量：SWE 每桶 20 题（共 85 题）或 TB2 每任务 1 次。是否接受由用户定。
- 第一次真机运行后用 `eval/wall_s` 校正本节。

### D8 训练数据来源与 SWE-bench Verified 是否重叠

- SWE-bench Verified 只评测。训练候选 SWE-Gym（`SWE-Gym/SWE-Gym`，2,438 条，HF 数据卡 MIT）：仓库为 pandas 737、moto 343、MONAI 374、mypy 257、dvc 225、dask 145、modin 107、pydantic 83、conan 75、hydra 66、bokeh 26；SWE-bench Verified 的 12 个仓库是 django、sympy、sphinx、matplotlib、scikit-learn、astropy、xarray、pytest、pylint、requests、seaborn、flask。**按仓库名比对两者无交集**（2026-10-08 用 HF 统计核对）。
- 核对方法（实现时由构建工具执行并入档）：① `instance_id` 求交集；② `repo` 求交集；③ `(repo, base_commit)`；④ `problem_statement` 规范化 sha256；⑤ 补丁涉及的文件路径 + 仓库（防同一仓库的近似题）。任一非空则从训练集剔除并记录；结果写进复核文档。SWE-smith 等其他来源同样处理（SWE-smith 的仓库与 SWE-bench 是否重叠**未核**）。
- SWE-Gym 没有官方难度字段，训练批次分桶对它只能标 `unknown`；以后可用第 0 轮或首次出现时的组内通过率另行分组，但那是策略相关口径，单列不混。

### D9 数学（dapo-math-17k）：降为可选，不分桶

- 理由：(1) 用户定先不用 Qwen3-32B 通过率，而 dapo-math-17k 本身没有难度或来源字段（上一版已核：zhuzilin 版只有 `prompt`/`label`，原版 `data_source`、`ability` 全同值）；(2) 自己用基座离线采样分桶约 $130（估计），给训练批次分桶要覆盖全部训练题约 $3,000（估计），超预算；(3) agentic 是后续主线。
- 但 FN 全量训练的数据目前仍是 dapo-math，**完全不评也不行**。可选做法（零额外采样成本）：随机留出 200 题从训练集删除，作为不分桶的固定数学评测集（每题 4 次，第 0 轮 8 次）；第 0 轮结束后按基座逐题通过率（0 / 部分 / 全对）事后分组，组别在第 0 轮后冻结。不做训练批次分桶。是否启用由用户定；接口与 D6 相同（`benchmark`=`dapo-math-17k`）。

### D10 dashboard 接口需求（WP4 实现）

- 读 `rl_eval`：横轴版本（轮），每桶一条线 + 标准误带；TB2 与 SWE 分两张图；点数 <5 不连线。
- 每桶截断/回合用尽比例、`infra_error` 比例；相对第 0 轮差值。
- 训练批次 `batch_summary_by_bucket`：按难度堆叠的训练成功率。
- `eval/set_sha256`、`eval/sampling` 变化时断线提示。多岛时事件带岛编号，不跨岛平均。

## Risks / Trade-offs

- 评测成本高（D7），可能需要减量或放便宜卡。
- TB2 留出后训练只剩 59 个任务，训练任务偏少；SWE-Gym 等补训练来源需另做接入。
- SWE-bench Verified 的 HF 数据卡未写许可证（代码仓库 MIT）。
- 官方难度是人工估计，METR 发现部分 `<15 min` 题实际耗时远超 15 分钟——所以第 0 轮基座通过率作为第二口径一起入档。
- 实现会碰 `driver.py`、`entry.py`、`rollout_meta_hook.py`，排在 WP7 阶段 3 之后。

## Open Questions（需用户拍板）

已定：TB2 + SWE-bench Verified 为主；TB2 留出 30（分层）；SWE-bench Verified 只评测；不用 Qwen3-32B 通过率；每 10 轮评一次；训练批次按基准难度字段分桶。

1. 评测成本（D7 估计每次约 $140–160，第 0 轮约 $300）是否接受？还是减量（SWE 每桶 20 题 / TB2 每任务 1 次）或放 spot？
2. 数学固定评测集（D9 可选，不分桶）要不要启用？
3. SWE-Gym 作为训练来源是否采用（需另开接入工作）？
