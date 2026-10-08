# Spec Delta：rl-eval-difficulty-buckets

## Purpose

为 RL 训练（首先是 agentic / codex）提供按基准官方难度分桶、钉哈希且与训练集不重叠的固定评测集，并在训练中定期评测、分桶记录，以把"题目难度"与"策略变化"分开。

## ADDED Requirements

### Requirement: 留出名单与评测数据钉哈希
每个评测基准 SHALL 有一份留出名单文件（`schema`、`benchmark`、`benchmark_version`、`seed`、`rule`、`items[{task_id, difficulty, eval_bucket}]`）和一份评测数据 jsonl；二者 sha256 SHALL 写入运行配置，启动时校验，不符即拒绝启动。

#### Scenario: 名单哈希不符
- **WHEN** 留出名单文件的 sha256 与运行配置不同
- **THEN** 驱动在任何上卡动作前拒绝启动并报出两个哈希

#### Scenario: 名单与评测数据不一致
- **WHEN** 评测数据中有名单外的题，或名单中的题在评测数据里缺失
- **THEN** 拒绝启动并列出差异

### Requirement: 训练集与评测集有交集即拒绝启动
驱动 SHALL 在启动时比对训练数据与所有评测名单：`task_id` 交集；对 SWE 类另比 `(repo, base_commit)` 与规范化 `problem_statement` 的 sha256。任一交集非空 SHALL 拒绝启动；检查结果 SHALL 写入 `rl_driver_start` 事件。

#### Scenario: 训练数据含留出任务
- **WHEN** 训练数据中有一行 `task_id` 在 TB2 留出名单里
- **THEN** 拒绝启动，报出交集个数与前 5 个 `task_id`

#### Scenario: 不同编号的同一 issue
- **WHEN** 训练行与 SWE-bench Verified 某题 `task_id` 不同但 `(repo, base_commit)` 相同
- **THEN** 拒绝启动

### Requirement: 数据行难度字段
训练与评测数据行的 `metadata` SHALL 带 `task_id`、`benchmark`、`benchmark_version`、`difficulty`、`difficulty_source`；评测行另带 `eval_bucket`。没有官方难度的来源 SHALL 写 `difficulty: "unknown"`，不得用模型打分冒充官方难度。

#### Scenario: 无官方难度的训练来源
- **WHEN** 训练数据来自没有难度字段的来源
- **THEN** 该行 `difficulty` 为 `unknown`，训练批次统计将其归入 `unknown` 桶

### Requirement: 定期评测
驱动 SHALL 在第 0 轮、每 `eval_interval` 轮（全量训练为 10）及最后一轮用已发布策略评测完整评测集；采样与 agent 设置（温度、top_p、上下文上限、reasoning effort、最大回合数、每题次数）SHALL 全程固定，变化即报错停止。

#### Scenario: 每 10 轮评一次
- **WHEN** `eval_interval=10` 且第 10、20 轮已发布
- **THEN** 第 0、10、20 轮各有一条 `rl_eval` 事件

### Requirement: 分桶指标与逐条记录
每次评测 SHALL 在 `rl_eval` 事件中按桶及全体给出通过率、标准误、相对第 0 轮的配对差及标准误、截断与回合用尽比例、`infra_error` 比例、题数与轨迹数，并给出逐条结果文件路径与 sha256；`infra_error` 的轨迹 SHALL 不计入通过率。

#### Scenario: 沙箱故障
- **WHEN** 某条评测轨迹因判分沙箱起不来而无结果
- **THEN** 记 `end_reason=infra_error`，单列比例，不算作失败

### Requirement: 训练批次按难度分桶统计
每轮训练批次 SHALL 按数据行 `metadata.difficulty` 分组，在 rollout 进程现有批次汇总处给出各组 reward 均值、成功率、截断比例、轨迹长度与条数，写入 `rl_rollout` 事件 `batch_summary_by_bucket`；只读样本标量。

#### Scenario: 每轮有分组汇总
- **WHEN** 一轮生成结束、样本交给训练之前
- **THEN** `rl_rollout` 事件带 `batch_summary_by_bucket`，各组条数之和等于本轮训练条数

### Requirement: 评测岛可中断续跑
评测 SHALL 能在单独的只推理评测岛上运行，不参与训练与合并；评测岛 SHALL 从持久存储加载评测版本的 adapter 并在校验 `policy_tensor_hash` 与 `rl/policy_token` 后开评；逐条结果 SHALL 以（`policy_version`、`task_id`、`trial`）为单位持久化，重启后跳过已完成单位并去重；训练驱动 SHALL 不等待评测完成。

#### Scenario: 评测岛被回收后续跑
- **WHEN** 评测岛在评某版本时被回收并重启
- **THEN** 已完成的单位不重算，被中断的轨迹记 `preempted` 不计入，全部完成后才发该版本的 `rl_eval`，指标与一次跑完的结果一致

#### Scenario: 评测慢于训练
- **WHEN** 上一评测版本尚未评完，新的评测版本已发布
- **THEN** 训练继续不等待，新版本排队，事件记录队列长度与滞后轮数

#### Scenario: adapter 哈希不符
- **WHEN** 评测岛加载的 adapter 哈希与 manifest 不一致
- **THEN** 拒绝评测该版本并报错
