# Proposal：按难度分桶的固定评测集（rl-eval-difficulty-buckets）

## Why

S16 的 FN 2×8 训推分离运行 `s16-rawlora-fn2x8-long-20261008a` 跑了 5 轮，每轮 reward 是 0.547/0.469/0.313/0.234/0.375。每轮训练用的题都不一样，所以 reward 的起伏到底是"这一轮的题更难"还是"策略变差了"，现在分不出来；截断率（0.56–0.95）也一样。全量训练（用户 S17 裁定：每轮 256 样本、DP2、TIS 开）要跑很多轮、花不少钱，必须有一把**固定的尺子**：同一批题、同样的采样设置，每隔固定轮数用当前策略评一次，并且按题目难度分开看。用户已定**每 10 轮评一次**（SESSION17-HANDOFF §9）。

现状（读代码确认）：
- 驱动已有评测钩子：`yeto/rl/engine/driver.py` 的 `_maybe_eval` 按 `eval_interval` 调 `evaluate(rollout_id)` 并发 `rl_eval` 事件；训推分离时 `overlap.py` 支持"评测与训练并行"。
- 但 ports 路径上的 `evaluate`（`miles_adapter/entry.py` 约 1682 行）调用 Miles 的 `EvalDispatcher.dispatch` 后**返回空字典**，评测数值只写进 Miles 日志（`eval/<数据集名>`、`eval/<数据集名>-truncated_ratio`，见 Miles `miles/ray/rollout/metrics.py`），`rl_eval` 事件里没有数值，dashboard 也拿不到。
- 训练数据 `zhuzilin/dapo-math-17k`（钉 2e656129，17,398 条）只有 `prompt`、`label` 两列，**没有任何难度或来源字段**；原版 `BytedTsinghua-SIA/DAPO-Math-17k` 也只有 `data_source=math_dapo`、`ability=MATH`，同样没有难度。

## What Changes

- 新建一份**固定评测集**：从 dapo-math-17k 中按难度分 5 个桶，每桶固定 40 题（共 200 题），评测集文件、分桶定义、题目编号连同 sha256 钉住；这 200 题**从训练数据中剔除**（训练数据换成剔除后的文件，并记录新哈希）。
- 难度来源（调研比较见 design.md D1；B 口径许可证未定，保留方案甲/乙）：分桶用公开的 `qgallouedec/DAPO-Math-17k-Processed-Scored` 里每题的 `Qwen3-32B_solve_rate`（外部、免费、事先固定）；第 0 轮用基座策略在评测集上每题采 8 个回答，记下我们基座模型自己的每题通过率，作为第二个难度口径一并入档，不另开离线采样任务。
- 评测频率：第 0 轮（训练前的基座）+ 之后每 10 轮 + 最后一轮；每题采 4 个回答（第 0 轮 8 个），采样参数与训练一致且整个运行固定。
- 记录：每桶的 reward 均值、标准误、截断率、回答长度中位数/p95、reward 缺失比例；每题每个回答的原始结果落盘（逐题文件），并写进扩展后的 `rl_eval` 事件。
- 把"题目难度"与"策略变化"分开：同一批题跨版本比较（只反映策略），按桶分开看（不同难度的变化不互相抵消），再加上训练批次按同一难度口径做的分桶统计（用户已定要做；开销实测每轮约 1 ms，见 design D4）。
- harness 中立：分桶字段写在评测样本行里，不绑数学；codex/TB2 以后沿用同一接口，难度来源另定（design.md D7）。
- dashboard 只提接口需求，由 WP4（yeto-fleet-dashboard）实现。
- 本 change 只写文档，不改代码、不上卡。

## Capabilities

### New Capabilities
- `rl-eval-difficulty-buckets`：按难度分桶的固定评测集——评测集构建与钉哈希、训练集剔除、定期评测、分桶指标与逐题记录、事件字段、dashboard 接口。

### Modified Capabilities
- 无（只在现有 `rl_eval` 事件上加字段；不改现有要求）。

## Impact

- 代码（实现阶段）：`yeto/rl/engine/miles_adapter/entry.py` 的 `evaluate` 要把 Miles 评测结果带回驱动；`driver.py` 的 `rl_eval` 事件加字段；新增评测集构建工具（`tools/` 下）；FN 训练脚本加 `--eval-prompt-data`（每桶一个数据集名）、`--eval-interval 10`、`--n-samples-per-eval-prompt 4`。
- 与 WP7 去耦合阶段 3（改 driver/bridges/state/publish）有文件交叠：本 change 实现时只碰 `driver.py` 的 `rl_eval` 发事件处和 `entry.py` 的 `evaluate`，需排在 WP7 阶段 3 之后或由主 agent 协调。
- 成本（估计，未验证）：每次评测约 800 个回答，在一台 8×H200 推理节点上约 20–30 分钟；与训练并行时额外成本小，串行时每 10 轮多约 25%（design.md D6）。
- 数据：需下载 `qgallouedec/DAPO-Math-17k-Processed-Scored`（钉 b9e6dd45）；其数据卡没有写许可证和打分方法，需用户知悉。
