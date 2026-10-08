# Spec Delta

## Purpose

岛间调度的行为契约：跨岛 policy-version 账本与样本判定、弹性成员（quorum / 心跳租约 / catch-up）、PauseAdvice 合并、IslandStatus 调度字段、journal pool_* 事件。两岛内部通信/合并优化不在本 spec。

## ADDED Requirements

### Requirement: 跨岛 policy-version 账本
系统 SHALL 以 (island_id, outer_version, inner_step, policy_hash) 标识每个样本组与增量条目，增量条目 MUST 带 c_tokens 与 c_steps，样本组 MAY 带 behavior_logprob。账本 SHALL 对样本组给出 ACCEPT / ACCEPT_IS / REJECT 三态判定之一并附原因；ACCEPT_IS MUST 指明 `mismatch_correction.CORRECTION_MECHANISMS` 中的一种修正名称。

#### Scenario: 同版本样本直接接受
- **WHEN** 样本组 outer_version 等于当前版本且 policy_hash 与该版本发布哈希一致
- **THEN** 判定 ACCEPT

#### Scenario: 陈旧但可修正
- **WHEN** outer 版本差为 1 或 2（≤ max_outer_lag，默认 2；inner 步差默认不设上限）、inner 步差不超上限且带 behavior_logprob
- **THEN** 判定 ACCEPT_IS，修正名称为配置的机制（默认 `tis`）

#### Scenario: 缺 behavior_logprob 或过旧被拒
- **WHEN** outer 版本差 > 0 且无 behavior_logprob，或版本差超过 max_outer_lag，或 policy_hash 与发布哈希不符
- **THEN** 判定 REJECT 并记录原因

#### Scenario: GRPO 组不跨岛（G-a）
- **WHEN** 一个 prompt 组
- **THEN** 组内全部样本来自同一岛同一 policy_hash，advantage 由生成端算好随样本传，消费端只做 IS 修正

#### Scenario: critic 家族拒收跨岛陈旧样本
- **WHEN** 算法族声明需要同策略 value 且样本来自其它岛的旧版本
- **THEN** 判定 REJECT

### Requirement: 弹性成员与 P4 算力加权步进
协调器 SHALL 在到齐成员算力和 `Σ cap_i ≥ θ·Σ_{成员} cap_i`（且到齐数 ≥ q_min）或到达软截止 T_soft 时推进外层版本；软截止时到齐数 < q_min MUST 不发布新版本。基于旧 base 的迟到增量在 `lag ≤ max_carry_lag` 时 MUST NOT 被丢弃，而 MUST 以 `γ^lag` 折扣作为 carried_over 并入下一次外层合并；超过 max_carry_lag 才拒收。协调器 MUST 拒收 syncer_epoch 小于当前值的消息（fencing）。心跳租约过期的成员 MUST 被移出成员集并使 membership_epoch 增加，其未提交增量 MUST 被丢弃并记录。新加入（catch-up）成员在首个外层步的合并权重 MUST 为 0。

#### Scenario: 算力加权提前步进
- **WHEN** θ=0.75，算力 3 的岛已提交、算力 1 的岛未提交
- **THEN** 不等软截止，外层版本 +1，慢岛记为 absent

#### Scenario: 迟到增量折扣并入
- **WHEN** 岛 c 基于版本 v 的增量在版本 v+1 发布后到达，γ=0.5
- **THEN** 该增量记 delta_carried_over（lag=1），在 v+1→v+2 的合并中权重为原权重 × 0.5

#### Scenario: 旧 syncer 实例被 fencing
- **WHEN** 消息携带的 syncer_epoch 小于账本当前 syncer_epoch
- **THEN** 拒收

#### Scenario: 租约过期退岛
- **WHEN** 某岛停止心跳超过 lease_s
- **THEN** 该岛移出成员，membership_epoch +1，其未提交增量记为 dropped_uncommitted

#### Scenario: 新岛首轮零权重
- **WHEN** 新岛在 outer_version=v 加入并在本轮提交增量
- **THEN** 该增量合并权重为 0；v+1 轮起按 c_tokens²/c_steps 计权

### Requirement: PauseAdvice 只收紧
系统 SHALL 把本地 `pause_decision` 与所有未过期 PauseAdvice 合并：任一未过期 veto MUST 使结果为拒绝；pause_budget 取最小值；过期 advice MUST 视为不存在；advice MUST NOT 把本地拒绝变为允许。

#### Scenario: 慢岛降级只出建议
- **WHEN** 协调器判定某岛应降级为纯 rollout 岛
- **THEN** 只产出带 `target_resource_intent` 的 PauseAdvice，读取它不触发任何资源操作，执行需人确认

#### Scenario: advice 不能放宽本地拒绝
- **WHEN** 本地 pause_decision 拒绝，advice 给出很大的 pause budget 且无 veto
- **THEN** 合并结果仍为拒绝

### Requirement: IslandStatus 调度字段
`IslandStatus` SHALL 包含 Optional 字段 pool_epoch、round_wall_s、tok_per_s、staleness_outer、pause_budget_s、cloud、region、price_per_hour；`inspect()` MUST 只用已有来源填充，无来源时为 None。

#### Scenario: 无来源字段为 None
- **WHEN** 未提供调度探针且 journal 无 pool 事件
- **THEN** inspect() 的上述字段全部为 None

### Requirement: journal pool_* 事件
journal SHALL 支持 tx_kind 为 pool_join / pool_leave / pool_epoch 的记录写入，并能从 journal 重放出 (pool_epoch, members)；pool_epoch MUST 单调不减，倒退写入 MUST 被拒绝。

#### Scenario: 重启后重放
- **WHEN** 写入 join(a)、join(b)、leave(a)、epoch 后关闭再打开 journal
- **THEN** 重放得到 members={b} 与最后的 pool_epoch

### Requirement: 旧模式与新模式的显式选择
系统 SHALL 提供配置项 `--rl-island-scheduling`，取值 `legacy` 或 `elastic`，默认 `legacy`。legacy 下行为 MUST 与现有 syncer 一致：成员固定、所有岛到齐才合并（到齐数等于岛数、额外等待为 0）、等待超时即失败、迟到增量拒收、不接受跨岛样本、不加载新消息类型、不写 pool_* 记录、IslandStatus 调度字段为空。elastic MUST 显式打开。模式取值 MUST 编入 session_contract_hash，两端模式不同 MUST 拒绝连接。

#### Scenario: 默认是旧模式
- **WHEN** 未给 `--rl-island-scheduling`
- **THEN** 模式为 legacy，三岛中两岛交了增量时不合并，超时报错而不是部分合并

#### Scenario: 新模式需显式打开
- **WHEN** 给 `--rl-island-scheduling elastic`
- **THEN** 按算力比例步进、迟到增量打折扣并入、允许加岛退岛

#### Scenario: 混用被拒
- **WHEN** 一端为 legacy、另一端为 elastic
- **THEN** 连接被拒绝，错误说明模式不一致

#### Scenario: 回退后忽略新记录
- **WHEN** elastic 运行写过 pool_* 记录，之后以 legacy 重启
- **THEN** 这些记录只读、重放时忽略，pool_epoch 与调度字段为空
