# 暂停审计（task 1.5）

基于 `rl-engine-ports` d355e06 源码只读审计。问题：learner island 在轮次边界暂停本地循环 T 秒、外层连接保持时，哪些超时可能到期。代码实现：`yeto/rl/engine/pause_audit.py`（纯函数，测试 `tests/test_rl_pause_audit.py`）。

## 结论

- RL 路径不发 heartbeat；learner socket `settimeout(None)`（`yeto/protocol.py:489`），接收线程在暂停时继续读；syncer `read_loop` 无超时（`syncer/src/server.rs:1259`）。
- 两个 RL 预设均为 fixed roster：`max_base_lag==0 && quorum==learners && grace_ms==0`（`server.rs:1686-1687`；launcher `yeto/launcher.py:1255-1256`、`:1266-1267`、`:277`；harness `yeto/rl/ssh_harness.py:2755-2775`）。迟到 learner 不被踢出轮次/roster，只重发 PULL。
- strict：`StrictAvgSync.boundary`（`yeto/rl/engine/bridges.py:210-231`）返回前已 `wait_for_round()`，暂停点**已持有下一轮 PULL permit**；其他 learner 在 `wait_for_global_policy`（`yeto/rl/bridge.py:343`，无超时）空等，整个 fleet 停顿 T。
- decoupled：`_boundary`（`bridges.py:456-524`）不持有等待。learner-budget 模式（`yeto_rl_learner_budget_steps`）下，`max_reconnects=None`（无限重连，`yeto/rl/decoupled.py:134`）；在 syncer `collect_budget_reports` 收尾窗口（`server.rs:2563-2660`，lease 长度 = `quorum_timeout_s`，默认 900 s，`server.rs:838-841`）内暂停超过该 lease 会让 run 失败——只有这个窗口 + budget 模式是致命的。非 budget 模式（`max_reconnects=0`）：本审计未找到会致命的外层超时，但**未经证实**（decoupled 未跑 X6），因此仍禁用。
- none：`LocalOnlySync`（`bridges.py:59-90`）无外层超时。

## 超时表

| 项 | 默认 | 暂停可超过? | 后果 |
|---|---|---|---|
| learner permit/全局策略等待 `bridge.py:343,370,382` | 无 | 是 | 同伴空等 |
| syncer quorum 截止 `--quorum-timeout-s`（`syncer/src/main.rs:52-54`，`server.rs:1971`） | 900 s | 是（fixed roster） | `server.rs:2031-2050` 重发同一 PULL 并延长；重复 PULL 被 learner 忽略（`bridge.py:419-435`，`decoupled.py:548-554`） |
| 非 fixed-roster 重启/grace（`server.rs:2052-2092`，`cli.py:588-605`） | 900 s / 1000 ms | RL 预设不适用 | — |
| syncer 写超时 `WRITE_TIMEOUT`（`server.rs:31,1245`） | 180 s | 仅当 learner 停止读 socket | 断连，`max_reconnects=0` 时 learner 退出 |
| 客户端重连（`protocol.py:76-78,398`；strict `bridge.py:268-277` 为 `max_reconnects=0`；decoupled `decoupled.py:128-135` 非 budget 模式为 0、learner-budget 模式为 `None`（无限重连）） | — | 空闲不触发 | **环境风险**：无 TCP keepalive（`protocol.py:487-491`），会丢空闲流的 NAT/LB 可致断连后退出 |
| 连接 generation（`protocol.py:472-477,517-551`，`learner.py:59,1557-1567`） | 0 | 是 | 仅重连时变化 |
| BUDGET_DONE lease（`server.rs:2563-2660,838-841,1555`） | `quorum_timeout_s`（默认 900 s） | **否**（仅 learner-budget 模式、仅 collect_budget_reports 收尾窗口） | syncer 报错结束 run；非 budget 模式不适用 |
| learner 预算合并（`protocol.py:783-835`，`decoupled.py:406-462`） | 900 s | **否** | 已报告的同伴 `TimeoutError` |
| final ACK（`server.rs:3306`，`--final-ack-timeout-s`） | 900 s（harness 3600） | 不可达 | stop 轮直接 finish |
| FleetController（`launcher.py:3019-3311`，`--recover-timeout` 1200 s） | 1200 s | 是 | 只看 job/cluster 状态，不检查进度 |
| Miles 分布式超时 `--rl-distributed-timeout-minutes` | 10 min | 仅全 rank 同点暂停时安全 | 部分 rank 进入 collective 会超时 |

## 可暂停阶段与预算（pause_audit.py 实现）

| profile（mode × outer） | 可暂停阶段 | 预算上界 | 状态 |
|---|---|---|---|
| colocated-serial / partitioned-serial × none | `round-boundary-published` | 无外层上限 | 启用 |
| colocated-serial / partitioned-serial × strict-avg | 同上，且非 stop 轮 | `min(0.5×quorum_timeout_s, 网络空闲流超时)`，默认 450 s。**这是本 change 选择的保守策略，不是代码上限**：fixed roster 下代码没有让暂停致命的超时（quorum 到期只重发 PULL）；取半个 quorum 是为了在 X6 认证跨重发暂停前不触发重发路径。暂停期间同伴停顿（计入成本） | 启用；跨 quorum 重发由 X6 认证后再放宽 |
| × decoupled | — | — | 禁用：budget 模式在收尾窗口有致命 lease；非 budget 模式未证实安全；待 X6-decoupled |
| partitioned-overlap × 任意 | — | — | 禁用（overlap 的 quiescent cut 未认证） |
| 其他/未知 profile | — | — | 禁用 |

一律否决：`boundary()` 内部、stop 轮/`finish()` 待执行、finalization、预算合并、learner-budget 模式、部分 rank 在 collective 中。
