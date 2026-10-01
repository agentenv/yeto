# E1-D §7 (b)：learner 重启后恢复"已提交成员配置"——设计草稿（检查点 1，2026-10-01）

依据：`SESSION6-HANDOFF.md` §7 第 3 条（用户裁定选 (b)）；`plan-3.8-4.4-v2.md` §7（设计限制评估）。
基线：infra-e1 3be4933，worktree `/home/michael/work/infra-recovery`（分支 `infra-e1-recovery`）。
fork：michaellchung/miles `yeto/ports` e3a11ab38（工作副本 `/home/michael/work/miles-fr1`）。
状态：**设计，未实现**。本文件不改任何判据；E1-D ⑤⑥⑦ 原判据（`plan.md` E1-D 行）保持不变。

## 1. 代码追踪：journal → 启动配置 → RayWorkerManager → worker

| 环节 | 位置 | 事实（已核对源码） |
|---|---|---|
| 持久状态 | `yeto/rl/engine/journal.py` | `epochs.json` 是唯一提交点（CAS：`config_epoch, config_id, fork_membership_epoch, members, last_tx_id`）；`journal.jsonl` 是 WAL（`request/phase/fork_op/reconcile/watchdog_*/trainer_recovery_hint/...`）。单写者 `flock(journal.lock)`。 |
| 重启重放 | `controller.py::_replay` (L288) | 从 WAL 得到 `_fork_epoch`（最后 `fork_op done` 的 `result_fork_epoch`，再与 epochs 取 max）、`_pending_retry`（`incomplete`）、`_last_op`、未终态事务 `_open_after_restart`、已记录的 `RECOVERY_REQUIRED`。 |
| 启动配置 | `miles_adapter/entry.py` L540–557 | `ElasticPlacement.restore_committed(configs[config_id].placement["rollout"], epoch)` 已把 **driver 侧** placement 恢复到提交配置；随后 `controller.open(driver.rollout)`。但 fork 侧 cell 集合仍是启动形状。 |
| placement map | `miles_adapter/placement.py::placement_map_arg` (L98) | `rollout_cells=[{name,bundles,start}]`：前 R/g 个 cell `start:true` 绑 rollout 段；接下来绑 standby 段 `start:false`；其余 `start:false` 且 unbound。**每次重启都按这个启动形状重新声明**。 |
| fork 启动 | `miles/utils/workers/ray_worker_manager.py::init` (L73) | 只 `start_cells` 非 deferred cell；绑定/视图只在内存；actor 以 `name=_ACTOR_NAME` 创建、**非 detached**，随 Ray job（= learner 进程）一起结束 → 旧 incarnation 的 worker 不会残留。 |
| fork 成员 API | `miles/ray/rollout/inference_controller.py` | `get_membership_status`(L137) `describe_cells`(L154: state unbound/stopped/running + tracked/serving/awaiting_admission + bundles) `restore_membership_state`(L173, CAS on `expected_current_epoch`) `wait_cells_tracked`(L216) `start_cells/stop_cells(expected_epoch)`(L252/280, 同 op 重复幂等；半失败 → `incomplete`) `admit_cells`(L370) `start_commit_weight_version(expected_epoch)`(L724: 所有 Serving cell 必须报告该 token，持锁)。 |
| yeto 端口实现 | `miles_adapter/rollout.py::MilesRolloutPool` | `members()`=running_members(tracked)；`plan_add` 取 declared−tracked；`add_engines`=`start_cells`+`wait_cells_tracked(track_timeout_s)`；`remove_engines`=`stop_cells`；`membership_status/restore_membership` 直通。 |
| 事务内 fork 调用 | `controller.py::_fork_call` (L895) | 先记 `fork_op issued` → 调用 → 按 `membership_status` 记 `done / incomplete / done(error_after_commit) / failed`；fork epoch 不等于期望+1 → `_enter_recovery`。`_sync_fork_mirror`(L1171) 在 config epoch 不变时把 fork epoch 写回 epochs.json。 |
| 初次发布 | `driver.py::run` (L1153) → `publish` (L591) → `publish.py::_publish` (L397) | 重启后 `sync.start` 给出权威 `rollout_id/state`，`ledger.rebase`，然后**全量** publish：`update_weights` 覆盖所有 tracked cell（PendingWeights → Serving，`end_update_weights` L697），`update_weight_version(token)` 并读回；`driver.publish` 再核 `result.members == rollout.members()`。`publish_members` 需要本进程已有全量参考，所以**重启后新启 cell 只能靠这次全量 publish 拿到权重**。 |
| 消费闸门 | `driver.py::_generate` (L642) / `safe_point` (L1047) | `controller.admission_open=False` 则拒绝 generate；`recovery_required` 在安全点抛 `RecoveryRequired` → `DriverError`。ledger 无新 `prepared`。 |

结论：现有接口足以让 **yeto 在 `open()` 中把 fork 的实际在役集合对齐到 epochs.json 的提交成员**，并借 driver 的初次全量 publish 完成权重/版本一致，再在 publish 之后做放行校验。**本设计不需要改 fork**（见 §5）。

## 2. 目标配置推导规则（不重放历史 up/down）

- **唯一来源** `epochs.json`：`C = (config_id, config_epoch, members, fork_membership_epoch, last_tx_id)`。它只在事务 commit CAS 或 `_sync_fork_mirror` 时变化，因此就是"最后有效提交的目标配置"。
- WAL 只用于：(a) 未终态事务分类（现有 `open()` 逻辑）；(b) 恢复 `_fork_epoch/_pending_retry/_last_op`（现有 `_replay`）；(c) 读取上一次未完成的 `recovery` 记录（新增，§3）。**不**按 `request/phase` 顺序重放 up/down。
- 边界：
  - `members` 为空（首次冷启、从未 `open()`）：无恢复，按现有逻辑把实际成员 CAS 进 epochs（`open()` L486）。
  - `config_epoch == 0` 但 `members` 非空且与实际不一致（例如 ⑦、或重启前 cell 已死）：同样走恢复（目标 = `members`）。
  - 未终态事务（`_open_after_restart`）先分类，再决定目标（§3.2）；目标恒为 **CAS 之后的 epochs.json**，不是事务 target。
- 可恢复性前置（任一不满足 → 直接 (a) 兜底 `RECOVERY_REQUIRED`，附原因）：
  1. `configs[config_id]` 存在，且其 `trainer`、`parallel`（tp/pp/cp/ep）与 `configs[initial_config]` 相同（训练 cell 由 Miles 按启动参数重建，本设计只恢复 rollout cell；role-transfer/trainer-dp 提交的配置不在范围，沿用 E3 `trainer_recovery_hint`）。
  2. `members ⊆ declared_cells`（`--rl-elastic-cells`/`describe_cells`）。
  3. 目标中需要启动的 cell 在 `describe_cells` 里 `state ∈ {stopped, running}`（`unbound` 的 cell 无法由 yeto 重新绑定——绑定来源在 E3 role-transfer，不在本设计范围）。

## 3. 恢复状态机（controller 新增，和事务 phase 正交；journal 记录 kind=`recovery`）

```
open(pool)
 ├─ RECONCILE_EPOCH   现有：fork epoch 0 → restore_membership_state(epoch=journal, incomplete, last_op, expected_current_epoch=0)
 ├─ CLASSIFY_TX       现有分类（§3.2 修改一处）
 ├─ PLAN              actual = pool.members(); target = epochs.members
 │                    无差异 → 无 recovery 记录，照旧返回（零改动路径）
 │                    有差异 → recovery{status=planned, recovery_id, incarnation, config_id, config_epoch,
 │                             target, actual, stop=[...], start=[...], deadline_wall}; admission_open=False
 ├─ RETRY_INCOMPLETE  若 _pending_retry：复用 _retry_incomplete（伪事务 tx_id=recovery_id）
 ├─ STOP_EXTRA        actual−target：_fork_call("stop") → remove_engines（先 drain? 否：这些 cell 尚未被 publish/路由，刚启动即 PendingWeights，不承载请求）
 ├─ START_MISSING     target−actual：_fork_call("start") → add_engines(members=missing)（含 wait_cells_tracked）
 ├─ MIRROR            _sync_fork_mirror()；recovery{status=membership_restored, fork_epoch}
 ├─ AWAIT_PUBLISH     返回 driver；admission_open 仍 False；recovery_pending 非空
 │                    driver.run: sync.start → ledger.rebase → publish(全量) → controller.confirm_recovery(driver)   ← 新钩子
 ├─ VERIFY            §4 校验全部通过 → recovery{status=verified,...}; admission_open=True; health RUNNING
 └─ 任一步失败/超时   _enter_recovery(recovery_id, 原因) → recovery{status=failed} → driver 在 handshake 后/安全点抛 DriverError；
                      ledger 无新 prepared；不做训练更新
```

- 时间预算：`Timeouts.recovery`（900 s）从 `open()` 起算；每步前 `_check_deadline`；`wait_cells_tracked` 用 `track_timeout_s`（600 s）但受总预算约束。超时 → `RECOVERY_REQUIRED`。
- 幂等：恢复是"目标 − 实际"的差分，不依赖历史；重复执行得到同一结果。恢复中的 `fork_op` 记录 `tx_id=recovery_id`，重启后由现有 `_replay/open()` 复用（含 issued-but-lost-ack 分支）。
- 恢复中再次崩溃：新进程重新 `open()`；fork 随 job 一起死 → epoch 0 → `restore_membership_state(epoch=journal 已推进值)` → 重新差分。上一条 `recovery planned` 没有 `verified` 时记 `recovery{status=superseded, by=new_id}`；**连续未验证恢复次数 ≥ `max_recovery_attempts`（默认 3，CLI 可调）→ `RECOVERY_REQUIRED("recovery budget spent")`**，避免无限自愈循环（restart loop 本身也有 `--rl-elastic-restart-attempts` 上限）。
- 资源不足：`start_cells` 被 worker manager 回滚 → `_fork_call` 记 `failed` → `RECOVERY_REQUIRED`（不重试：资源不足不是瞬态；`incomplete` 的半失败按现有 `_retry_incomplete` 在预算内重试）。
- 旧 controller / 旧 epoch 隔离：
  - 旧 learner 进程：journal `flock` 单写者 → 旧进程任何 append/CAS 失败（`JournalLocked/EpochConflict`）；fork actor 非 detached，随旧 job 结束，不残留 worker。
  - 每次 `open()` 记 `incarnation = {pid, uuid, wall_time}`（写入 `reconcile`/`recovery` 记录与 `IslandStatus`）；恢复期间 `request()` 一律 `Rejected("island is recovering")`；`expected_config_epoch` 不等于当前 epoch 的请求照旧 `Rejected`（现有）。
  - 恢复的所有 fork 调用都带 journal 期望 epoch；fork epoch 与期望不符 → 现有 `_fork_call` 判 `RECOVERY_REQUIRED`。

### 3.2 未终态事务分类（restart 后）与恢复目标

| 重启时事务位置 | 现状 (3be4933) | 本设计 | 恢复目标 |
|---|---|---|---|
| CAS 之后（`last_tx_id == tx`）| `SUCCEEDED(recovered_after_restart)` 但成员不一致 → `RECOVERY_REQUIRED` | `SUCCEEDED(recovered_after_restart)` + 恢复到新提交成员 | epochs.members（新） |
| release 之前（无 DESTRUCTIVE phase）| `CANCELLED` | 不变 | epochs.members（旧）|
| release 之后、CAS 之前，rollout-only | `RECOVERY_REQUIRED` | **待裁定 D1**：恢复到旧提交成员，事务终态 `REBUILT_OLD(recovered_after_restart)`（3.7 原文"release 后重建旧 rollout"）| epochs.members（旧）|
| release 之后，trainer 边（trainer-dp/role-transfer）| `RECOVERY_REQUIRED` + `trainer_recovery_hint` | 不变（trainer 不在范围）| — |

## 4. 放行前校验（`confirm_recovery` 全部通过才 `admission_open=True`）

rollout cell（`engine:<cell>`，全部由 fork rollout pool 承载）：
- R1 `pool.members()`（tracked）== `epochs.members`。
- R2 `membership_status()`：`epoch == journal fork epoch` 且 `incomplete is None`。
- R3 `publisher.verify_serving_policy(epoch=fork_epoch, token_rollout_id=driver.published_version, state=driver.published_state)` → fork `start_commit_weight_version(expected_epoch)`：每个 Serving cell 报告 token `policy_token(rollout_id, policy_hash)`；并与 `driver.expected_token` 相同。
- R4 路由准入：`describe_cells()` 中目标 cell 均 `serving=True, awaiting_admission=False`（没有 cordoned/待 admit 的残留）；fork 无 `describe_cells` 时记 `router_check: unavailable`（fake 池）。
- R5 `driver.publish` 自身已核 `result.members == members`（现有）。

training cell（Miles actor pool，按启动参数重建；本设计不迁移/重建训练 cell）：
- T1 静态：`configs[config_id].trainer/parallel == configs[initial_config]`（§2 前置）。
- T2 通信组：`trainer.actual_layout()`（rank coords 读回：world/tp/pp/cp/ep/dp）与 `configs[config_id].dims`、`trainer` 数一致；trainer 没有该方法时记 `layout_check: unavailable`。
- T3 训练状态：`driver.published_version == sync.start.rollout_id`；`ledger.unconsumed() == []`；`ledger.rebase` 已把重启点以上的 `optimizer_applied` 标 superseded（现有）。
- T4 身份：learner_id/trainer generation 由 `handshake()` 校验（现有）；重启后 PID 变化、policy hash 由 R3 覆盖。

校验结果整体写入 `recovery{status=verified|failed, checks={...}}`。

## 5. fork 最小接口需求

**无需新增**。用到的接口与语义（均已存在于 e3a11ab38）：
`describe_cells()`、`get_membership_status()`、`restore_membership_state(epoch, incomplete, last_op, expected_current_epoch)`（CAS）、`start_cells/stop_cells(cell_ids, expected_epoch)`（同 op 重复幂等；半失败 `incomplete`）、`wait_cells_tracked`、`start_commit_weight_version/end_commit_weight_version(expected_epoch)`。
不会在 yeto 侧触碰 fork 私有状态（`_membership_*`、`_awaiting_admission`、cell 绑定）。
若实现中发现 `describe_cells` 在 Ray handle 路径不可用（entry 已有 `_awaitable(describe())` 兼容），退回 `get_cell_statuses` 判 tracked，R4 降级为 `unavailable`——这是 CPU 阶段要确认的点，不是 fork 改动。
因此**不建 fork-recovery worktree**；如阶段 2 出现真实缺口再提。

可选优化（**不在基线**，待裁定 D4）：learner 启动时按 epochs.json 改写 placement map 的 `start` 标志，让 fork 在 init 时就启动提交成员，省去"先起启动形状再 stop 多余 cell"的一次引擎启动（GPU 上约 1–3 min）。代价：启动路径多一个分支、与 `declared_cells` 解析耦合。基线先用差分恢复，行为统一、可测。

## 6. 失败矩阵（恢复路径）

| # | 故障 | 处理 | 终态 / 证据 |
|---|---|---|---|
| F1 | 恢复中（stop/start 之后、verified 之前）再次 kill learner | 新进程重新差分；旧 `recovery` 记 superseded；计数+1 | 第二次 `verified`；journal 两条 recovery |
| F2 | 连续 N 次恢复都未 verified | `RECOVERY_REQUIRED("recovery budget spent")` | admission 关闭，ledger 无 prepared |
| F3 | start_cells 资源不足（worker manager 回滚）| `fork_op failed` → `RECOVERY_REQUIRED` | 原因含 fork 异常 |
| F4 | start 回滚也失败（`start_rollback_failed`，`incomplete=["stop",cells]`）| `_retry_incomplete` 预算内重试 stop；成功后继续 start | `fork_op incomplete` + 重试 `done` |
| F5 | stop 半失败（`stop_failed`）| 同 F4 | 同上 |
| F6 | `wait_cells_tracked` 超时 / 总预算 `T_recovery` 超时 | `RECOVERY_REQUIRED("recovery ran past T_recovery")` | journal `recovery failed, reason=timeout` |
| F7 | `restore_membership_state` CAS 失败（fork epoch ≠ 0）| 现有 → `RECOVERY_REQUIRED` | — |
| F8 | 全量 publish 失败 / 部分 ACK | driver 现有 `PublicationError` → run 失败；restart 后 F1 路径 | — |
| F9 | verify：某 Serving cell token 不符 / 成员不等 / 有 cordoned 残留 | `RECOVERY_REQUIRED`（失败项列在 `checks`）| — |
| F10 | 提交配置 trainer 形状 ≠ 启动形状 / 目标 cell unbound / 不在 declared | 前置失败 → `RECOVERY_REQUIRED`（trainer 情形附 E3 hint）| — |
| F11 | 恢复期间来旧 epoch / 任意请求 | `Rejected`；journal 不记 request | — |
| F12 | 旧 learner 进程仍存活并写 journal | `JournalLocked/EpochConflict`（单写者）| 新进程启动即 `JournalLocked` → 不启动（需人工处理；restart loop 下不会发生）|
| F13 | 恢复失败后 driver 行为 | `handshake` 后立即 `DriverError`（新增前置检查），不 publish、不 generate | tape `rl_reconfiguration result=RECOVERY_REQUIRED` |

## 7. CPU 故障测试清单（新文件 `tests/test_rl_reconfig_recovery.py`，复用 `test_rl_reconfig_e1.py` 的 `ForkMembership/ElasticFakePool/ElasticFakePublisher/_setup`）

1. `test_restart_after_commit_recovers_committed_members`（⑤ 一般情形：up 提交后 kill → 重启 fork 回启动形状 → 恢复到 4 成员，`SUCCEEDED(recovered_after_restart)`，`verified`，之后 generate 成功；ledger 无重复消费）。
2. `test_restart_after_commit_of_down`（(c) 回归：提交配置 = 启动形状 → 无 recovery 记录，行为同 3be4933）。
3. `test_restart_in_quiescing_is_cancelled_no_recovery`（⑥）。
4. `test_restart_before_any_tx_at_epoch0_with_members_mismatch`（⑦：fork epoch 0、cell 死一只 → 恢复）。
5. `test_restart_after_release_rollout_only_rebuilds_old`（D1；若裁定不改则改为断言 RECOVERY_REQUIRED）。
6. `test_recovery_crash_midway_is_idempotent`（F1：在 START_MISSING 后、verified 前再次重启；两条 recovery 记录；最终一致）。
7. `test_recovery_budget_spent`（F2）。
8. `test_recovery_start_rolled_back_enters_recovery_required`（F3）。
9. `test_recovery_retries_incomplete_stop_then_continues`（F4/F5）。
10. `test_recovery_timeout`（F6，fake clock）。
11. `test_recovery_verify_token_mismatch`（F9：fake fork 某 cell 版本错 → RECOVERY_REQUIRED；driver 不 generate）。
12. `test_recovery_refuses_requests_and_old_epoch`（F11）。
13. `test_recovery_trainer_shape_mismatch_falls_back`（F10）。
14. `test_driver_fails_fast_when_recovery_required_at_open`（F13，ledger 无 prepared）。
15. `test_rl_miles_adapter_e1.py` 增：`MilesRolloutPool` 在 Ray-handle 风格 controller 上 `describe_cells` 的 R4 读取（mock）。
另：`test_restart_mid_transaction` 的 `after_release` 期望按 D1 调整或保持。

## 8. tasks.md / alignment.md 需新增或修改（均不勾选）

- tasks.md 3.7 新增"进展"子条目：重启恢复（本设计），已实现+CPU 通过后记录提交 SHA、测试文件、仍未满足的原契约（GPU 验收）。
- tasks.md 3.3a 文本"learner/controller 重启后先读 journal，再以 describe 得到的 fork epoch 对账，不一致则 RECOVERY_REQUIRED" → 追加"（3.7 恢复：对账后按 epochs.json 的提交成员差分启停 cell，校验通过才放行；不可恢复仍 RECOVERY_REQUIRED）"。
- alignment.md G11：记录用户裁定 (b)，"首版重启回启动形状"对 rollout-only 配置不再适用；trainer 形状变化的配置仍回启动形状 + RECOVERY_REQUIRED。
- `evidence/infra-e1/plan-3.8-4.4-v2.md` §7 追加 §7.2：裁定 (b) 实现后 ⑤⑥⑦ 按原判据在任意提交配置上执行；(c) 安排降为回归用例。
- CLI：`--rl-elastic-max-recovery-attempts`（默认 3）走 launcher→learner→build_elastic→controller（与 quorum/idle 参数同一条链）。

## 9. 待裁定（检查点 1）

- **D1** release 之后、CAS 之前 kill（rollout-only）：恢复到旧提交成员并把事务收尾为 `REBUILT_OLD(recovered_after_restart)`，还是保持 `RECOVERY_REQUIRED`？建议前者（符合 3.7 "release 后重建旧 rollout"；trainer 边不变）。
- **D2** 恢复期间收到的请求：拒绝（建议）还是排队到 verified 之后？
- **D3** 连续未验证恢复上限默认 3 次是否合适；是否要 CLI 参数。
- **D4** 是否在基线之外加"启动时按 epochs.json 改写 placement map start 标志"的优化（建议不做，先保证正确性）。
- **D5** 新建测试文件 `tests/test_rl_reconfig_recovery.py` 是否在允许路径内（派发列的是 INFRA 共享文件；测试文件为新增）。

## 10. GPU 验收计划（阶段 2 完成后提出，**未批准、未上卡**；判据 = `plan.md` E1-D ⑤⑥⑦ 原文 + `plan-3.8-4.4-v2.md` §7.2 补充判据，不放宽）

实现状态：已实现 + CPU 通过（`tests/test_rl_reconfig_recovery.py` 23 条；全量与基线 94 同集合）。3.7 仍不勾选。

### 10.1 前提
- 代码：infra-e1-recovery 分支（本提交）合入 integ-decl/gpu-b1 运行分支；镜像 pin 不变（fork e3a11ab38 无改动）。
- 平台：Nebius 8×H100（链 6 同款，$30.8/h = $0.5133/min），**不用 Modal**（函数 retries 在新容器重跑，state dir 不持久，⑤⑥⑦ 需同容器原地重启：restart loop 是 shell 循环，state dir `~/yeto-rl/elastic-state` 在同一容器内保留）。
- 公共配置：T4R2S2 起始（`rollout_cells` 4 个声明、2 个 start），`--rl-elastic-restart-attempts 2`，`--rl-elastic-max-recovery-attempts 3`，up deadline 600 s，每用例 ≥ 6 轮（重启/恢复后至少 3 轮训练以证明消费与 token）。
- CPU 前置（上卡前免费演练）：用链 5 的 journal 时间戳在本地回放 `open()`（已在单测覆盖）；`test_a4_dryrun.py` 加 1 条：`--rl-elastic-max-recovery-attempts` 到达岛上命令。

### 10.2 用例与窗口核算（gpu-evidence-window-lesson：采集窗口 ≥ 采集延迟）
| 顺序 | 用例 | 事件链 | 证据与采集方式 | 采集延迟 vs 窗口 | 预计时长 | 硬超时 |
|---|---|---|---|---|---|---|
| 1（冷启）| ⑥ `r6`：epoch 0 的 up 在 QUIESCING kill（`YETO_RL_TEST_KILL_LEARNER_AT=QUIESCING`）| 冷启 ≤1200 + 启动 300 → up → QUIESCING kill → 原地重启 300 → `CANCELLED`，无 `recovery` 记录 → 继续 ≥3 轮 | journal/epochs/tape 在 state dir，运行结束后 rsync（容器内文件，不依赖探针）| 无终态探针；run 在终态后继续 ≥3 轮（≥45 s）再正常结束，文件落盘即证据 → 延迟 0 | 冷启 1200 + 300 + 60 + 300 + 60 ≈ 1920 s | **2700 s**（1200+300+600+300+收敛 300）|
| 2（热）| ⑦ `r7`：首个事务前容器内 kill learner（restart loop 重启，fork epoch 归 0），重启后发 up | 启动 300 → kill → 重启 300 → `reconcile restore_membership_state(epoch=0, expected_current_epoch=0)`，无 recovery → up 成功 ≥3 轮 | 同上 | 同上 | 300+300+150+60 ≈ 810 s | **1800 s**（300+300+600+收敛 600）|
| 3（热）| ⑤ `r5`：up 提交后 COMMITTED kill（`YETO_RL_TEST_KILL_LEARNER_AT=COMMITTED`，一次性 marker）| 启动 300 → up（引擎启动 ≈150）→ COMMITTED kill → 重启 300 → fork 回 T4R2S2 → `recovery planned → fork_op start c2,c3 done → membership_restored` → 首次 publish → `verified`，tape `RECOVERED` → ≥3 轮，ledger 新 prepared 的 token = 重启发布 token；up `SUCCEEDED(recovered_after_restart)` | 同上 + 容器内 `term_probe`（盯 journal 出现 `recovery verified` 后 2 s 内跑 fork `describe_cells` 快照，兜底核对 R4）| 探针延迟 ≈2 s ≪ 终态后 run 寿命 ≥45 s；主证据仍是文件 | 300+150+300+150(恢复: 引擎启动+publish)+60 ≈ 960 s | **2400 s**（300+600+300+T_recovery 900+收敛 300）|
| 4（热，回归，可选）| ⑤c `r5c`：§7.1 (c) 安排——up 提交，重启后带 kill 发 down 在 COMMITTED 杀 | 期望 `SUCCEEDED(recovered_after_restart)`、**无** `recovery` 记录（提交 = 启动形状）| 同 1 | 同 1 | ≈ 1100 s | **2400 s** |
- 用例均不以 FAILED 退出；若某用例恢复失败（RECOVERY_REQUIRED），learner 以 DriverError 退出 → launcher 拆集群，所以把风险最高的 ⑤ 放在 ⑥⑦ 之后；可选 ⑤c 放链尾。
- 链级 watchdog = Σ硬超时 + 1500 s；n2run 链模式不设 autostop；线程守卫 < 2900（CHAIN6 §0）。

### 10.3 预算（Nebius $0.5133/min）
- 必做 1–3：最坏 (2700+1800+2400)/60 × 0.5133 ≈ **$59.0**；预期 (1920+810+960)/60 × 0.5133 ≈ **$31.6**。
- 加可选 4：最坏 +$20.5 → **$79.6**；预期 +$9.4 → **$41.0**。
- 旧估算 $25（(c) 方案 3 次各 $8）不沿用：本方案多一次原地重启 + 恢复引擎启动，且首用例承担冷启。
- 建议 CAP：本链 **$80**（含可选 4）；与链 6（A4 补齐，最坏 $61.6）合并同集群可省一次冷启 ≈$10，但须先批链 6。
- 失败重跑：同一失败找到原因并修复后才重跑；每次上卡前台账预登记。

### 10.4 通过/失败条件（运行前固定）
- 通过：每用例终态与上表"事件链"一致；⑤ 的 `recovery verified.checks` 中 `policy_token=verified`、`router.not_admitted=[]`、`trainer_layout.world=4`、`unconsumed_batches=[]`；重启前后 learner PID 不同、policy hash 一致（tape `rl_publication` 的 `rl/policy_token`）；ledger 无重复消费（同一 rollout_id 只有一条 `outer_recorded`）。
- 失败：任一恢复以 RECOVERY_REQUIRED 结束、或 `recovery failed`、或重启后成员 ≠ journal 成员、或 ledger 出现重启点以上的重复 `prepared`。失败如实记录，3.7 不勾选。
