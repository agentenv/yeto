# yeto-fleet-dashboard 进度

## S11（2026-10-05/06）
- tasks.md：24 项中 21 项完成（CPU）。代码 `yeto/dashboard/`，文档 `docs/DASHBOARD.md`，样例导出 `infra-drafts/tmp-logs/dashboard-s9-m4x1.html`。
- 未完成：
  - 4.1 浏览器人工核对与截图（本环境无浏览器；仅 node DOM 桩冒烟 `tests/js/dashboard_smoke.js`）。
  - 6.1 syncer Rust 原生 round 事件（阻塞：cargo 构建环境与 rl-infra-spec A5 验收）。
  - 7.1 真实多岛 ports 磁带的 serve/export 与截图（CPU 替代已用单岛 s9-m4x1 磁带做，round 表与成本面板为"无数据"）。

## 生产接线现状
- 已接：`miles_adapter/entry.py` 中仅在 observe 路径开启时 `controller.set_event_sink(driver.emit, journal=True)`，输出 cell-snapshot / reconfig-phase 事件；observe=False 时旧磁带逐字节不变。
- 未接：
  - launcher 的预算/价目表只来自环境变量 `YETO_DASHBOARD_BUDGET_USD` / `YETO_DASHBOARD_PRICES`（`yeto/dashboard/fleet.py`），未接 launcher 配置；
  - ssh_harness 不自动拉起 mirror；
  - adv_* 指标在实机运行中为 None；
  - train_step 在恢复后从 0 计数。

## S14（2026-10-07，子 agent G，纯 CPU）
做了什么：用真实 Modal 运行磁带对看板做 export（4 份）与 serve（1 份，`/`、`/api/{overview,rounds,fleet,events,islands/0}` 全 200，POST 405），产物与覆盖脚本在 `infra-drafts/tmp-logs/`：
- `dashboard-s14-s13-g3-modal-20261007f.html`（2 岛 ports，远端 controller，0 轮）
- `dashboard-s14-s13-g1-modal-20261007c.html`（单岛 critic，2 轮训练 + fleet.jsonl）
- `dashboard-s14-s13-fnsmoke-modal-20261007a.html`（FN 单岛 8xH200，1 轮 + 之后 OOM + fleet.jsonl）
- `dashboard-s14-s14-fnsmoke-modal-20261007a.html`（s14 运行中途的只读快照：events 回显 + fleet.jsonl，未动其目录）
- `dashboard-s14-s11-h200-20261005o-e1.html`（补充：单岛 14 轮 + journal，E1 事务/cell 表）
- 覆盖提取脚本 `infra-drafts/tmp-logs/dashboard-s14-coverage.py`；价目表（Modal H200 $0.001261/s/卡=4.5396/h，H100 $4.39/h，来源 gpu-spend.md，估算非账单）为 /tmp 临时文件。

发现的 bug（已修）：真实 Modal 运行的 `fleet.jsonl` 里 `island_ready` 的 cloud/gpu/gpus 全为 null（`meta_from_task` 只认 sky Task，Modal 岛的是 `ModalIslandConfig`），成本面板因此一律“未定价”。修在 `yeto/dashboard/fleet.py`（识别 gpu/gpus_per_node/num_nodes，cloud=modal，去掉 `!`），加无 Ray 单测 `tests/test_dashboard_cost.py::test_meta_from_modal_island_config_shape`；`tests/test_dashboard_*.py` 55 passed。已有磁带的 fleet.jsonl 不会回填：上面 g1c/fn13/fn14 的成本面板导出时，用的是把 `island_ready` 的 cloud=modal/gpu/gpus 按各运行 args（1xH100 / 8xH200）补进去的 staging 副本（磁带原件未改），成本数值（g1c $3.10、fn13 $36.96，估算）因此依赖该补丁，不是原始 fleet.jsonl 的结果。

面板覆盖（有数据 / 无数据及原因）：
| 面板 | g3f 两岛 | g1c 单岛 | fn13 FN 单岛 | e1 单岛 |
|---|---|---|---|---|
| 训练曲线 | 无：磁带只有 rl_engine_selected 等 2 条，运行 0 轮（真实缺数据） | reward/loss/pg_loss/kl/entropy/grad_norm/clip/tok_s/lr/ess/reward_std/train_step 各 2 点；reward_p10/p90、resp_len、trunc 空：磁带里这些字段就是 null（critic 路径未填） | 同 16 个指标各 1 点（含 p10/p90、resp_len、trunc）；clip_lo/hi 空（磁带无该字段） | 16 个指标各 14 点；clip_lo/hi 空 |
| round 表 | 空：无 syncer 磁带，只有 syncer 文本日志（learner connected x2） | 空：同左 | 空：同左 | 空：同左 |
| 告警 | 无（磁带里无触发条件） | 无 | 无；第 2 轮 OOM 没写进磁带，看板看不到（磁带缺事件，非 reducer bug） | 无 |
| 成本 | 无 fleet.jsonl，无数据 | 补元数据后 $3.10（0.707h x 1 x $4.39）；原件为“未定价”（见 bug） | 补元数据后 $36.96（约 1.02h x 8 x $4.54）；原件未定价 | 无 fleet.jsonl |
| 岛卡/明细 | 两岛卡片，无心跳、无资源 | 无心跳/资源（旧运行无 rl_heartbeat） | 心跳+资源采样+tok/s 有 | 心跳、资源、cell 表（rl_cell_snapshot）、6 个 E1 事务有 |

7.1 结论：未达验收口径（仓库内没有任何带 syncer 磁带/round 事件的真实多岛 ports 磁带；g3 两岛运行 0 轮），故不勾；已做两岛真实磁带 + 多份单岛真实磁带的 export 与 serve 核对。需要：多岛运行带出 syncer 磁带（或 6.1 的 Rust 原生 round 事件）后再验。

4.1：本机无浏览器（chromium/chrome/firefox 均无，playwright 未安装，无 ~/.cache/ms-playwright），未截图、未勾。
