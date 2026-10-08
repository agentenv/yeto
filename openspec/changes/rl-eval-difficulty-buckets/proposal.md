# Proposal：按难度分桶的固定评测集（rl-eval-difficulty-buckets）

## Why

S16 的 FN 运行每轮用不同的题，reward 起伏分不清是"题更难"还是"策略变了"。全量 RL 训练、以及接下来的 agentic RL（codex harness）都需要一把固定的尺子：同一批题、同样的设置，每 10 轮（用户已定）用当前策略评一次，并按题目难度分开看。

现状（读代码确认）：驱动已有 `rl_eval` 评测钩子与第 0 轮评测，训推分离时可与训练并行；但 ports 路径的 `evaluate` 返回空字典，评测数值不进事件，dashboard 拿不到。

用户 S17 裁定：agentic 评测以 **Terminal-Bench 2（TB2）** 与 **SWE-bench Verified** 为主；先不用 Qwen3-32B 通过率；SWE-bench Verified 只做评测；训练批次按基准自带难度字段分桶统计。

## What Changes

- 固定评测集：TB2 先排除 S15 用作训练的冒烟 6 题，再留出 30 个任务（按官方难度分层 easy 2 / medium 18 / hard 10，永不训练）+ SWE-bench Verified（`SWE-bench/SWE-bench_Verified@78f471bf`，与 WP6 #129 一致）按官方修复耗时分 3 桶（<15 分钟 30 题、15 分钟–1 小时 30 题、≥1 小时全部 45 题）。
- 评测频率：第 0 轮 + 每 10 轮 + 最后一轮；第 0 轮 TB2 每任务 4 次、SWE 每题 2 次，之后 2 次 / 1 次。
- 记录：每桶通过率、标准误、相对第 0 轮配对差、截断与回合用尽比例、判分基础设施错误比例；逐条结果落盘；扩展 `rl_eval` 事件。
- 训练批次按数据行 `metadata.difficulty` 分桶统计（无官方难度的训练来源标 `unknown`）；本机实测开销约 1 ms/轮。
- 接口需求（判分环境由 WP6 在 Modal 沙箱实现）：每题难度字段、留出名单文件格式、"训练集与评测集有交集即拒绝启动"检查、判分结果回传字段。
- SWE 类训练来源与 SWE-bench Verified 的重叠核对方法（SWE-Gym 暂不采用，只留方法）。
- 评测单独起一个只推理的评测岛，放便宜的可中断卡（可中断性 I2），从持久存储加载评测版本的 adapter，按"策略版本 + 题号 + 第几次"续跑去重，与训练并行、不拖慢训练（design D11）。
- 数学（dapo-math-17k）降为可选、不分桶（理由见 design D9）。
- dashboard 只写接口需求（WP4）。本 change 只写文档，不改代码、不上卡。

## Capabilities

### New Capabilities
- `rl-eval-difficulty-buckets`：固定评测集（留出名单、钉哈希、交集检查）、定期评测、分桶指标与逐条记录、训练批次分桶、事件与 dashboard 接口。

### Modified Capabilities
- 无（只在 `rl_eval`、`rl_rollout`、`rl_driver_start` 事件上加字段）。

## Impact

- 代码（实现阶段）：`miles_adapter/entry.py` 的 `evaluate` 回传结果；`driver.py` 事件字段与启动检查；`rollout_meta_hook.py` 训练批次分桶；新增构建工具与 `data/eval/` 名单文件。与 WP7 阶段 3 交叠，排其后。与 WP6 约定判分回传字段。
- 成本（估计，未验证）：每次评测约 165 条轨迹，约 3–3.5 小时 8×H200；Modal 约 $110–160、AWS spot 约 $85–110；第 0 轮约翻倍。详见 design D7、D11。
- 数据：TB2 训练任务从 89 减到 59。
