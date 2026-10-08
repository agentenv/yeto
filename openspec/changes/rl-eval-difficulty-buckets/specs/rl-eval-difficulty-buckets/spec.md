# Spec Delta：rl-eval-difficulty-buckets

## Purpose

为 RL 训练提供一份按难度分桶、钉哈希且与训练集不重叠的固定评测集，并在训练中定期用当前策略评测，分桶记录 reward 与截断率，以便把"题目难度"与"策略变化"分开。

## ADDED Requirements

### Requirement: 固定评测集的构建与钉哈希
评测集 SHALL 由离线工具一次生成，包含每题的 `eval_item_id`、`bucket`、`difficulty_source`、`difficulty_value`；评测集文件、分桶定义、抽题随机种子 SHALL 记录 sha256 或数值并写入运行配置。训练运行 SHALL 校验评测集哈希，不一致即拒绝启动。

#### Scenario: 评测集哈希不符
- **WHEN** 运行配置里的 `eval/set_sha256` 与实际评测集文件的 sha256 不同
- **THEN** 训练启动失败并报出两个哈希

#### Scenario: 重新生成可复现
- **WHEN** 用同一来源数据版本、同一分桶定义与同一种子重新运行构建工具
- **THEN** 生成的评测集文件与已钉的 sha256 一致

### Requirement: 评测题与训练题不重叠
评测集中的题 SHALL 从训练数据中剔除（按规范化题干完全相同匹配），剔除后的训练数据 SHALL 记录新哈希。训练运行 SHALL 在启动时检查二者交集为空。

#### Scenario: 训练数据含评测题
- **WHEN** 启动检查发现训练数据中有题干与评测集某题规范化后相同
- **THEN** 启动失败并列出重叠题数与前几个 `eval_item_id`

### Requirement: 定期评测
驱动 SHALL 在第 0 轮训练前、之后每 `eval_interval` 轮（FN 全量训练为 10）、以及最后一轮，用当时已发布的策略在完整评测集上评测；整个运行中评测采样参数 SHALL 固定。训推分离且开启评测并行时，评测 SHALL 评的是它被安排时的那个已发布版本（沿用 overlap.py 现有约束）。

#### Scenario: 每 10 轮评一次
- **WHEN** `eval_interval=10` 且第 10、20、30 轮训练完成并发布
- **THEN** 对应版本各有一条 `rl_eval` 事件，且第 0 轮有一条基座评测事件

#### Scenario: 采样参数被改动
- **WHEN** 同一运行中某次评测的采样参数与第一次不同
- **THEN** 运行报错停止，不产生口径不一致的评测数据

### Requirement: 分桶指标与逐题记录
每次评测 SHALL 在 `rl_eval` 事件中按桶及全体给出 reward 均值、标准误（按题自助重采样）、截断率、回答长度 p50/p95、reward 缺失比例、题数与回答数，并给出相对第 0 轮的配对差值及其标准误；SHALL 将逐题逐回答的 reward、是否截断、回答长度写入一个 jsonl 文件，并在事件中记录其路径与 sha256。

#### Scenario: 事件含分桶字段
- **WHEN** 一次评测完成
- **THEN** `rl_eval` 事件包含 `eval/b0/reward_mean` … `eval/b4/reward_mean`、各桶 `truncated_ratio`、`eval/all/*`、`eval/items_path`、`eval/items_sha256`

#### Scenario: 第 0 轮记录基座难度
- **WHEN** 第 0 轮评测完成
- **THEN** 逐题文件包含每题 8 个回答的结果，可据此得到基座每题通过率

### Requirement: harness 中立
分桶字段 SHALL 位于评测样本行与 `rl_eval` 事件中，不依赖具体任务类型；指标计算 SHALL 在 yeto 侧完成，不依赖训练框架专有的日志格式。

#### Scenario: 换 harness 复用
- **WHEN** 以后 codex/TB2 评测集按同样字段提供 `bucket`
- **THEN** 不改驱动与事件结构即可产生分桶指标

### Requirement: dashboard 接口
`rl_eval` 事件 SHALL 带 `eval/set_sha256` 与 `eval/sampling`，使展示端能识别评测口径变化；事件 SHALL 带岛编号（多岛时）。

#### Scenario: 评测口径变化
- **WHEN** 两次评测的 `eval/set_sha256` 或 `eval/sampling` 不同
- **THEN** 展示端可据此断线显示，而不是把两者连成一条曲线
