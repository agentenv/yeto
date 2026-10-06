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
