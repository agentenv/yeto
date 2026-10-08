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


## 2026-10-08 补充：启动阶段可观测性与成本口径（用户同意按方案 C，等 FN 2×8 早门跑完再做）

**问题**（实测于 `s16-rawlora-fn2x8-long-20261008a`，dashboard 8789 端口）：
- 岛的心跳与资源采样只在 `driver.run()` 内启动（`yeto/rl/engine/driver.py:954-973` `_telemetry_threads`）。容器启动到训练循环开始之间（加载 360 GB 权重、组 Ray 集群、推理引擎初始化，实测 10–15 分钟），磁带上只有一条 `rl_engine_selected`。
- dashboard 心跳规则（`yeto/dashboard/alerts.py:47-63`）不区分"启动中"与"运行中卡住"，启动超过 300 s 必报"严重：心跳超时"，属误报；同时启动阶段完全看不到存活、显存、GPU 利用率，真卡死与误报无法区分。
- 成本：内置示例价目表（`yeto/dashboard/prices.example.json`，`modal:H200` 4.54 $/卡·时）只算 GPU，Modal 2×8 显示 $72.64/h，实际约 $87.93/h（含 CPU 与内存），低估约 17%。
- 只传 `--tapes` 不传 `--run` 时页面显示"未命名运行"。

**改动（分两步）**：
1. 第一步，只改 dashboard（`yeto/dashboard/`）：未见 `rl_driver_start` 前岛状态为"启动中"，按启动子步骤用单独的更宽阈值，不报心跳超时；成本改为 GPU + CPU + 内存（价目表加 CPU/内存单价，缺失时显示"仅 GPU"）；未传 `--run` 时从磁带推断运行名。
2. 第二步，心跳提前（放进 `yeto-framework-decoupling` 的阶段里做，依赖其标准样本保证事件格式不变）：岛一启动即开心跳与资源采样，阶段 `startup`；在加载权重、Ray 集群组好、推理引擎就绪各发一条事件；dashboard 按这些子步骤分别计时与告警，从而能发现真正的启动卡死。


## 2026-10-08 补充：界面改版（用户确认）

**为什么**：现有界面（冷色、所有面板平铺）在单岛训推分离运行（`s16-rawlora-fn2x8-long-20261008a`）上暴露出：标题写"Syncer 总览"但本次无 syncer；cell 表、E1 事务、syncer round 表、Ray 面板全是"无数据"却占据大半屏；最需要的"本轮在哪个阶段、训练与推理各花多久、截断率、每轮耗时拆分"缺失，只能翻日志；两轮数据连成一条"下滑线"、无误差带，易误读；卡片样式一致分不出主次；成本表窄屏需横向滚动。

**改什么**：
- 暖色配色（浅/深两套），训练（T）与推理（R）全页固定一对颜色；数字用等宽字体。
- 两种布局按运行类型自动切换：单岛 → 第一屏为阶段时间线 + 每轮耗时拆分 + 训练/推理两列；有 syncer 的多岛 → 第一屏为岛卡片墙 + syncer 合并时间线。共用组件与配色。
- 借鉴用户给的参考图：左侧导航、顶部关键数字卡片（状态、成本、轮次、时长、吞吐）、可展开的岛健康列表、带时间与级别的事件流、带复制按钮的常用命令。
- 数据真实性：只显示磁带里的真实数据，缺失显示"未采样/本次未使用"，不放演示数据；未使用的多岛与弹性面板默认折叠成一行。
- 曲线：点少于 5 个时只画点不连线；有分位数时画误差带，并注明"每轮题目不同"。
- 悬停交互：鼠标在图上横移时竖线吸附到最近一轮并高亮该点；数值框显示该轮的轮次与策略版本、选中指标精确值（含 p10/p90）、截断率、logprob 差、耗时拆分（推理/训练/同步/发布）；多岛时每岛一行、缺席标"缺席"；曲线、耗时拆分、阶段时间线三处联动高亮同一轮；数值框靠边自动翻转；手机点按等同悬停，键盘左右键逐轮移动；缺失项显示"无"，不插值。
- 先出静态预览页（用该 2×8 运行的真实磁带生成），用户看过定稿后再改 `yeto/dashboard/static/`。等 FN 2×8 早门跑完、与第 8 组一起做。
