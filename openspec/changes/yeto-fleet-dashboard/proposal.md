# Proposal

## Why

多岛 DiLoCo RL 运行时，唯一的观测手段是分散的 learner 事件磁带、syncer 事件磁带与 W&B 曲线。真正要回答的问题是“syncer 视角下各 learner 训得好不好、哪一岛出了问题、花了多少钱”，现状答不了：ports 引擎路径下 learner 磁带大部分训练指标为 None 或缺失，`train/step` 为 None 导致 W&B 丢弃全部 `train/*` 曲线（含 grad_norm），岛健康、GPU 利用和成本完全没有数据源。用户已选定 dashboard mock v7（`infra-drafts/dash-mocks/v7.html`，v4 健康取向 × v6 研究取向合并）作为目标界面。

## What Changes

- **训练指标补齐（P0，不受 infra 阻塞）**：ports 路径下 `step_metrics()` 补齐 loss/pg_loss/mean_kl/ess_ratio/lr/train_step；`rl_round_trained` 扩展 `train_metrics{}`（Miles loss dict 全部标量的轮均值，含 kl_loss、entropy_loss、entropy、ppo_kl 等，替代目前每轮被丢弃的 step_losses）、advantage mean/std、response length mean/p95、截断比例、reward p10/p50/p90、tok/s；修复 `train/step` 使 W&B `train/*` 曲线恢复。
- **新增运行期事件**：`rl_heartbeat`（30s）、`rl_resource_sample`（NVML，30–60s）；新增 head 侧 `fleet.jsonl`（FleetController/launcher 写入岛生命周期与每 5 分钟 `cost_tick`）。`rl_cell_snapshot`、`rl_reconfig_phase` 声明 schema，发射依赖 rl-infra-spec（cell 声明、E1 事务 3.2–3.7）。
- **syncer 派生视图**：不改 Rust；以 syncer merge 记录为主键，join 各岛 `rl_fragment_push`/`rl_policy_apply`/`rl_pull_resend`/`rl_member_publication`，反推每 round 的 responded/expected、missed、quorum/sync/merge 时延与重发。
- **单一 reducer + 两种出口**：同一 reducer 把磁带归约为 overview/islands/rounds/fleet/events 视图；实时出口为 head 上只读 HTTP 服务（只绑 127.0.0.1，经 SSH 隧道访问，端点 `/api/overview`、`/api/islands/<id>`、`/api/rounds`、`/api/fleet`、`/api/events`）；离线出口为自包含静态 HTML 导出。
- **v7 页面**：告警卡（按严重度，点击联动定位）、按岛叠加训练曲线（round 边界、missed 带、告警点）、syncer round 表、岛健康卡、成本效率表；岛明细为下钻面板而非独立页面；只读，仅附“复制 CLI 命令”。
- **成本面板**：价目表（配置，示例值标注非账单）× GPU 数 × 墙钟估算，与预算上限联动告警。
- **Ray dashboard 嵌入**：Nebius/本地经 SSH 隧道 iframe；Modal 不可嵌入，退化为事件/资源面板。ssh_harness 路径纳入第一期（实时 tail 远端磁带）。
- W&B 定位为曲线归档，不作为 dashboard 数据源。

## Capabilities

### New Capabilities

- `rl-training-telemetry`: learner/head 侧训练与运行期事件的字段契约（扩展 `rl_round_trained`、`rl_heartbeat`、`rl_resource_sample`、`fleet.jsonl`、预留 `rl_cell_snapshot`/`rl_reconfig_phase`）及 W&B `train/step` 行为。
- `fleet-dashboard`: 从磁带归约的只读 fleet 视图、syncer 派生 round 视图、实时只读服务与静态导出、告警、成本估算、Ray 嵌入与 v7 界面行为。

### Modified Capabilities

无。现有主 spec 仅 `head-run-teardown`，与本 change 无需求级交集。

## Impact

- 本轮仅写规划文档。
- yeto 新增模块（建议 `yeto/dashboard/`：reducer、server、export、静态资源）与 CLI 子命令；改动 `wandb_rl.py`、`wandb_tape.py`、`status_metrics.py`、`launcher.py`（FleetController 写 `fleet.jsonl`）、`rl/ssh_harness.py`。
- INFRA 所有文件（`rl/engine/driver.py`、`rl/engine/miles_adapter/trainer.py`、`state_plugin.py`、`rollout_meta_hook.py`、`rl/engine/controller.py`/`journal.py`）的改动以接口请求形式提交给 rl-infra-spec / rl-engine-ports 负责人，由其合入或授权；本 change 不单方面修改。
- 不改 syncer Rust；syncer 原生实时事件列为 P3 后续项（需 cargo 环境与 rl-infra-spec A5 验收）。
- 无新外部服务依赖；前端为无构建步骤的静态页面。
