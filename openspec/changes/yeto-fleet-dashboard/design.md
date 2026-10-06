# Design: yeto fleet dashboard（以 mock v7 为准）

## Context

动机见 `proposal.md`。界面以 `infra-drafts/dash-mocks/v7.html` 为准（v4 健康 × v6 研究）。以下“源码已确认”基于 `origin/integ-decl`（058b00e）只读核对，其余为设计建议。

**数据面现状（源码已确认）**

- learner 磁带：唯一低层写入者 `yeto.rl.event_echo.append_record`；`learner.install_event_echo()` 在拿不到 `~/yeto-output` 的岛（Modal、no-sync）上把每条记录以 `YETO_RL_EVENT` 回显到 stdout，由 `event_echo.TapeCollector` 在本机重组磁带（`launcher.wait_for_tapes` 等待其完成）。
- syncer 磁带：`launcher.syncer_event_tape(args)`；head 上 `LocalSyncer.start_tape_forwarder`（`launcher.py` 约 3540，经 `cli.py` 约 1761 调用）以 `wandb_tape.TapeForwarder` + `OffsetStore` 跟随转发 W&B；`status_metrics.render_tape_summary` 做 CLI 摘要。
- `driver._train_round` 发射 `rl_round_trained`（rollout_id、trained_groups/samples、masked_fraction、clip_fraction、applied_lrs、nonzero_advantages、filtered_samples、tool_wait、mismatch、data_cursor）；`_stats` 生成 `rl_local_round`（reward_mean/std、loss/pg_loss/grad_norm/lr/mean_kl/ess_ratio/train_step 等）。
- **缺口**：`MilesTrainerGroup.step_metrics()` 在 ports 路径只填 grad_norm、applied_lrs、masked_fraction、clip_fraction；loss/pg_loss/lr/mean_kl/ess_ratio/train_step 恒为 None。`state_plugin._record_step_losses` 每步抓取 Miles loss dict 全部标量，`trainer.mean_step_metrics` 可做轮均值，但结果除 mismatch 与 clipfrac 外被丢弃。`wandb_rl` 把 `train/*` 绑定到 `train/step`，其为 None 时整组被丢弃。advantage 只有非零计数；response length、截断比例、tok/s、GPU 利用、成本、心跳均无数据。
- 已有 syncer 相关岛侧事件：`rl_fragment_push`（`rl/miles.py`）、`rl_policy_apply`（driver/miles）、`rl_pull_resend`（`rl/bridge.py`）、`rl_member_publication`（`rl/engine/controller.py`，INFRA-E1 于 058b00e 新增）。
- `FleetController`（`launcher.py` 约 3745）管理岛生命周期但不落盘事件；`rl/ssh_harness.py` 路径有独立的 plan/事件键定义。

## Goals / Non-Goals

**Goals**：syncer 总览对比各 learner 为主视图、岛明细为下钻；实时与离线共用一个 reducer；训练指标在 ports 路径完整可见；按价目表估算成本并与预算联动告警。

**Non-Goals**：写操作（暂停、重配置、终止等）——仅提供“复制 CLI 命令”；多用户/鉴权/公网暴露；替代 W&B 的曲线归档；修改 syncer Rust（P3 之前）；真实账单对接；自建时序数据库。

## Decisions

### D1. 磁带是唯一真相源，reducer 纯函数化

reducer 输入为若干 JSONL 流（learner 磁带 × N、syncer 磁带、`fleet.jsonl`），输出不可变视图 `{overview, islands[id], rounds[], fleet, events[]}`。实时服务增量喂入（沿用 `wandb_tape.follow_jsonl` 的偏移跟随，独立 `OffsetStore` 文件，不与 W&B 转发共享偏移），静态导出一次性全量喂入后把视图 JSON 内联进 HTML。
- 备选：从 W&B API 取数——拒绝：W&B 丢了 `train/*`、延迟高、离线不可用，且只作为归档。
- 备选：服务端与导出各写一套——拒绝：两套口径必然漂移。
- 未知 event 类型与未知字段 MUST 透传到 events 视图而非报错；缺字段在视图中为 null，界面显示“无数据”而不是 0。

### D2. 只读 HTTP 服务：127.0.0.1 + SSH 隧道

head 上 `yeto dashboard serve --run <run>` 起 stdlib HTTP 服务，只绑 127.0.0.1（拒绝 0.0.0.0 参数），无鉴权（单用户、隧道即边界）。端点：

| 端点 | 内容 |
|---|---|
| `GET /api/overview` | run id、模式、全局状态、告警列表、成本摘要、各岛最新卡片、叠加曲线序列 |
| `GET /api/islands/<id>` | 岛明细：进度/policy、训练指标序列、心跳/资源、cell 表（P2）、E1 事务（P2）、最近事件、Ray 嵌入信息 |
| `GET /api/rounds?only_bad=` | syncer 派生 round 表 |
| `GET /api/fleet` | 岛生命周期与 cost_tick 序列、价目表来源 |
| `GET /api/events?island=&type=&after=` | 事件分页（游标） |
| `GET /` | v7 页面（与导出同一份静态资源） |

所有端点仅 GET；非 GET 返回 405。页面 5s 轮询 `/api/overview`（v7 头部“实时 · SSH 隧道 · 5s 刷新”）。CLI 打印建议隧道命令 `ssh -L <port>:127.0.0.1:<port> <head>`。

### D3. 静态导出

`yeto dashboard export --run <run> -o run.html`：reducer 全量视图 JSON 内联，单文件，无外部请求（不加载 Google Fonts，回退系统字体），离线打开即得与实时同构的页面，头部显示“离线导出 · 生成时间”。Ray iframe 在导出中显示为占位与隧道命令。

### D4. 事件 schema（摘要；字段契约见 specs/rl-training-telemetry）

所有记录保留既有公共字段（`event`、`ts`、`island_id`/learner id、run id）；新增字段均为可选，旧 reducer/W&B 转发忽略未知字段。

- **扩展 `rl_round_trained`**（P0）：`train_step`（优化器步序号，非 None）、`train_metrics{}`（Miles loss dict 各标量轮均值，键原样，如 `loss`、`pg_loss`、`kl_loss`、`entropy_loss`、`entropy`、`ppo_kl`、`pg_clipfrac`）、`adv_mean`、`adv_std`、`resp_len_mean`、`resp_len_p95`、`truncated_frac`、`reward_p10/p50/p90`、`tok_per_s`（本轮 action tokens / (rollout+train 秒)）、`step_seconds`。
- **`rl_local_round`**（P0）：loss/pg_loss/mean_kl/ess_ratio/lr/train_step 由补齐后的 `step_metrics()` 填值，无法得到时保持 None 而非伪造。
- **`rl_heartbeat`**（P1）：`phase`（generate/train/sync/publish/idle）、`rollout_id`、`policy_version`、`uptime_s`；周期 30s，learner 后台线程写入。
- **`rl_resource_sample`**（P1）：每 GPU `{index, uuid, util_pct, mem_used_mb, mem_total_mb, power_w}`，周期 30–60s（可配置）；NVML 不可用时每进程只发一次 `available:false`。
- **`fleet.jsonl`**（P1，head 本地）：`island_launch|island_ready|island_lost|island_stop`（岛 id、cloud、region、gpu 型号与数量、价目 key）与 `cost_tick`（每 5 分钟：各岛 `$/h`、累计墙钟、累计估算 $、预算上限）。
- **`rl_cell_snapshot` / `rl_reconfig_phase`**（P2，预留）：schema 由 rl-infra-spec cell 声明与 E1 事务（3.2–3.7）定义，本 change 只约定 reducer 读取的最小字段（cell id、角色、GPU 数、状态；txn id、阶段、结果）。

### D5. syncer 派生视图（不改 Rust）

以 syncer 磁带的 merge 记录（`wandb_tape._is_merge_record` 判定）为 round 主键，按 fragment/round 与 policy_version 关联各岛 `rl_fragment_push`、`rl_policy_apply`、`rl_pull_resend`、`rl_member_publication`：expected = 当轮 roster 大小，responded = 有 push 的岛数，missed = 差集；quorum_ms = 首个 push 到 merge，sync_ms/merge_ms 取自 merge 记录已有耗时字段，缺失则为 null；重发计数来自 `rl_pull_resend`。视图标注“派生”，与未来 P3 原生事件字段同名，便于替换。

### D6. 告警规则

reducer 内的纯规则表，阈值可配置，严重度 0 严重/1 警告/2 提示，每条带定位目标（岛、round、指标）。首批：岛心跳超时（>60s 警告、>300s 严重）；连续 missed；quorum 时延相对基线（中位数）倍增；grad_norm 相对滑动基线尖峰或 NaN；clipfrac/KL 超阈值；预算达 60%/80%；`rl_reconfig_phase` 进入 RECOVERY_REQUIRED（P2）。

### D7. 成本估算

价目表为配置文件（`gpu 型号 × cloud → $/GPU·h`，本地为 0），仓库内示例值标注“示例，非账单”。估算 = 单价 × GPU 数 × 墙钟（`island_ready` 到 `island_stop`/当前）。效率列：`$/1M tok`、`reward/$ =（末轮−首轮 reward）/累计 $`；本地岛不参与排名。预算上限取运行参数中的 cap（未配置则只显示累计）。

### D8. Ray dashboard 嵌入

岛信息携带 `ray_embed`：`tunnel`（Nebius，给出 `ssh -L <本地端口>:localhost:8265 <host>` 及 iframe URL）、`direct`（本地 127.0.0.1:8265）、`none`（Modal，退化为最近 `rl_resource_sample` 与事件面板）。页面只在用户点击后加载 iframe；隧道需用户自行建立。

### D9. ssh_harness 路径实时 tail

ssh_harness 运行时，head 侧对各远端 learner 磁带起 `ssh <target> tail -F` 跟随写入本地镜像磁带，reducer 与本地岛同一路径处理；断线指数退避重连，并在 events 中记 `dashboard_source_lost`。Modal 继续走 `TapeCollector` 回显重组。

### D10. INFRA 文件改动以接口请求交付

`driver.py`、`miles_adapter/trainer.py`、`state_plugin.py`、`rollout_meta_hook.py`、`controller.py`/`journal.py` 属 rl-infra-spec/rl-engine-ports 负责范围。本 change 为每处给出接口请求（字段、语义、测试），由负责人合入或书面授权后由本 change 提交；dashboard 侧代码只依赖字段契约，字段缺失时降级显示。

## Risks / Trade-offs

- [ports 路径拿不到 Miles 的 lr/train_step] → 从 `applied_lrs` 与驱动 `local_step` 推导，并在字段上标注来源；仍拿不到则保持 None，不伪造。
- [派生 round 视图与 syncer 真实语义偏差] → 标注“派生”；P3 原生事件就绪后替换，并用同一运行对比两者。
- [NVML 采样开销/容器无 NVML] → 周期≥30s；不可用时显式 `available:false`，界面显示“无数据”。
- [成本估算被误读为账单] → 所有成本位置标注“估算，非账单”，价目表需显式配置。
- [127.0.0.1 服务仍可被同机其他用户访问] → 本机单用户前提；文档写明，不做鉴权。
- [磁带体积增长] → heartbeat/resource_sample 频率受控；reducer 对曲线按 round 下采样。

## Migration Plan

全部新增字段可选、新增事件独立，旧运行磁带仍可被 reducer 读取（缺字段显示“无数据”）。W&B `train/step` 修复后旧运行不回填。回滚：关闭 heartbeat/resource_sample 开关、不启动 dashboard 服务即恢复原状。

## Open Questions

- 价目表首批覆盖的 cloud × GPU 型号清单（不影响结构）。
- 告警阈值默认值在首个真实多岛运行后校准。
