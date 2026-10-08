# Proposal: 真正的岛间调度（跨岛样本池 + policy-version 账本、弹性成员、PauseAdvice）

## Why

用户 2026-10-07（S15）裁定：岛间优化只做"真正的岛间调度"——SURVEY §3 方案 4（跨岛样本池 + policy-version 账本）与方案 5（弹性成员：quorum + catch-up + 心跳租约），以及 f-design §1.5 的 `PauseAdvice` 实装；两岛内部通信/合并优化（量化、DyLU、分片并发、LoRA SVD 合并）明确排除，但账本要给它们留接口。用户同时要求"岛间优化开发与验收并行推进"；今晚 GPU 预算被 codex 阶段 2 占满，所以本轮验收只用 CPU 假岛。

代码事实（`infra-drafts/INTER-ISLAND-EXPLORE-S15.md` §A）：现有代码只覆盖"固定成员、单岛事务"——
- syncer 岛身份 = `learner_id` + generation，`--learners` 启动固定（`syncer/src/server.rs:911-913`），14 种消息无 JOIN/LEAVE（`protocol.rs:9-25`）；
- `IslandStatus`（`yeto/rl/engine/controller.py:240`）缺 pool_epoch / 算力 / 陈旧度 / pause 预算 / 云与价格；
- journal（`yeto/rl/engine/journal.py`）无 `tx_kind=pool_*`；
- driver batch 只来自本岛（`driver.py:769,789`），陈旧度语义只有"拒收"（IR-3）；
- `PauseAdvice` 全仓不存在（G3）。

## What Changes

- **跨岛 policy-version 账本**（新模块 `yeto/rl/engine/island_ledger.py`；权威在 syncer，P8a+P9 `syncer_epoch` fencing，用户裁定 2026-10-07）：条目 = (island_id, outer_version, inner_step, policy_hash, behavior_logprob 可选, c_tokens, c_steps)；样本最大陈旧度 ≤2 外层步、inner 不设限；三态判定 ACCEPT / ACCEPT_IS / REJECT，IS 修正口径复用 `mismatch_correction.CORRECTION_MECHANISMS`（默认待 M2PO vs TIS/IcePop 离线比较后定）；GRPO 组规则 G-a；critic 家族拒收跨岛/陈旧样本。
- **外层步进 P4**：算力加权 quorum（Σ到齐 cap_i ≥ θ·Σcap_i）+ 软截止 T_soft；迟到增量按 γ^lag 折扣 carried_over 并入下一轮；增量按 `w=c_tokens²/c_steps`（`merge.rs:26-31`）计权，P5 影子加入首轮零权重，退岛未提交增量丢弃并记录。
- **样本池只做 P3 对象存储**（S3 / Modal Volume / Nebius OS），协调器只存索引。
- **IslandStatus 扩展**：`pool_epoch` 与调度字段（`round_wall_s`、`tok_per_s`、`staleness_outer`、`pause_budget_s`、`cloud`、`region`、`price_per_hour`），全部 Optional，由 `inspect()` 填充（已有值为准，缺省 None）。
- **journal pool_* 事件**：`tx_kind=pool_join / pool_leave / pool_epoch` 的写入与重放（`PoolState`），pool_epoch 单调。
- **PauseAdvice**：数据类与合并规则（只能收紧本地 `pause_decision`，过期即不存在，任一 veto 即拒绝）；心跳租约产生 veto/预算 advice；P6 慢岛降级为纯 rollout 岛只以 advice 形式发出、人确认，不自动执行；P7 云价再分配只留 `target_resource_intent` 与 IslandStatus cloud/region/price 接口。
- **CPU 假岛 harness**：`multiprocessing`（不用 Ray）N 个假岛 + 协调器，演示 P4 步进与 carried_over 折扣、租约过期退岛、新岛 catch-up 零权重，输出 tape/JSON。
- **syncer 协议扩展**只写进 design（D-S），本 change 阶段 0 不改 Rust 代码。

- **模式开关与回退**：显式配置 `--rl-island-scheduling legacy|elastic`，默认 legacy（与现有 syncer 行为逐位一致：固定成员、所有岛到齐才合并、不打折扣、不接受跨岛样本、不写 pool_* 记录）；elastic 必须显式打开；模式编入连接时核对的配置指纹，混用直接拒绝；回退 = 改参数重启。

## Non-Goals

- 两岛内部通信/合并优化：增量量化 + EF、DyLU 按算力配内层步、分片流水/多流传输、LoRA 精确/SVD 合并（用户 2026-10-07 裁定排除）。账本只保留 `wire_dtype`、`c_tokens`、`c_steps`、`extra` 字段作为接口。
- 节点级跨岛资源再分配（G4 launcher 运行中增删节点、G6 fork 运行中 bundle 增删）；本 change 只产出岛级 `target_resource_intent` 建议，不执行云操作。
- 不改 fork Miles / SGLang，不提上游 PR；不新增 provider 能力（沿用 `add-nebius-verda-modal-clouds`）。
- 阶段 0 不改 syncer Rust 代码、不改 driver 的 batch 来源（只给接口与假岛验证）。
- 不做 P1/P2 样本转发、P7 决策器、P6 自动执行、G-b 混组（实验项）。
- 阶段 1/2 GPU 验收不在今晚执行，开卡需用户单独批准，不预登记。

## Capabilities

### New Capabilities

- `rl-inter-island-scheduling`：跨岛 policy-version 账本与样本判定、弹性成员（quorum / 租约 / catch-up）、PauseAdvice 合并、IslandStatus 调度字段、journal pool_* 事件。

### Modified Capabilities

- 无（`rl-infra-spec` 7.1–7.3 / D12 的占位由本 change 承接，见 design §9；不改其既有要求）。

## Impact

- 新增：`yeto/rl/engine/island_ledger.py`、`yeto/rl/engine/pause_advice.py`、`yeto/rl/engine/fake_islands.py`、`tests/test_rl_inter_island_*.py`。
- 修改：`yeto/rl/engine/controller.py`（IslandStatus/inspect）、`yeto/rl/engine/journal.py`（pool_* 事件与重放）。
- 依赖：真机阶段依赖 `rl-infra-spec` 3.8/X6（两岛 strict 暂停）与 syncer 协议扩展（D-S）。
