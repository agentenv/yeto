# INFRA-E1 执行说明 v2：A5（3.8 X6）、A6b（4.4）与 watchdog 检查（取代 plan-3.8-4.4.md）

状态：**未执行**。

**判据唯一来源**：集成分支上的 `openspec/changes/rl-infra-spec/gpu-plan-v2.md` §2（配置、卡、轮数）与 §3（判据）。A5 看 §3 的"A5（3.8 X6）"第 1–5 条；A6b 看 E2 `evidence/infra-e2/4.2-4.5/plan-v2.md` 的 G-4.4（配置 C3）。本文件**不另立判据**，只写三类内容：执行这些判据需要的 E1 开关与前置条件；gpu-plan-v2 引用原文、但原文没有写到的 E1 观测点（§3 与 §4 的补充，同样在运行前固定）；已知限制。若本文件与 gpu-plan-v2 冲突，以 gpu-plan-v2 为准，本文件视为有误。
v1（`plan-3.8-4.4.md`）的拓扑（2×8 卡、8 轮）、"最终 policy hash 等于 B0""syncer base 等于 B0"两条（rollout engine 数不同且未开确定性推理，不可达），以及 X6-b 的注入点都已废弃。

## 1. 共同前提（缺一项不启动）

1. 代码：集成分支 SHA（运行时记录），须包含 infra-e1 的 3.8（533afdc）、4.4（d397cf3）、watchdog（2b67145）以及本轮审查修复 5946ffd（F2）、e477bcb（F3）、500555a（F6），另含 E2 `SwappableActor` 接线。
2. 镜像与 manifest：按 gpu-plan-v2 §1 与 gpu-plan.md §8。
3. 开关：两岛都带 `--rl-elastic`（不带则没有 data cursor，4.4 会在动 trainer 之前被拒）。岛0 另带 `--rl-elastic-resources/-initial-config/-cells`；resources 按 1.6 格式列出 T1R1S1、T1R2S0 与两条 `rollout-only` 边；fork 启动时声明 2 个 rollout cell，只启动 1 个。
4. **start_cells 前的 150 s 延迟注入**（gpu-plan-v2 A5 第 2 条）：已实现（9a5f181），只供测试使用，默认关闭。开启方式：launcher 加 `--rl-test-inject-start-delay-s 150`（要求同时带 `--rl-elastic`），它在岛的运行命令里 `export YETO_RL_TEST_INJECT_START_DELAY_S=150.0`；`MilesRolloutPool.add_engines` 在本进程**第一次**调用 fork `start_cells` 之前 sleep 这么多秒，同时在 stderr 打印 `TEST INJECTION` 一行。只有 quorum 运行带这个开关。§4 需要的 `update_weights` 阻塞注入点见 §4。
5. quorum 运行的暂停预算（gpu-plan-v2 未写，属于执行 A5 第 2 条的必要条件）：`--quorum-timeout-s 120` 时默认预算为 0.5×120 = 60 s，150 s 的延迟会在 plan 阶段被 `pause_decision` 拒绝，第 2 条就无从执行。因此 quorum 运行中岛0 用 `--rl-elastic-quorum-timeout-s 120 --rl-elastic-pause-margin 2.0`，up 请求的 deadline 设为 230 s（≤ 240 s 预算）。launcher 会把同一个 120 同时传给 syncer 的 `--quorum-timeout-s`。空闲流探测若测得路径会在 150 s 内丢流，按 gpu-plan-v2 A5 第 4 条判为环境阻塞；此时 `--rl-elastic-idle-flow-timeout-s` 取实测值，并会让该请求被拒，正好与"不运行"一致。
6. finalization 用例（A5 第 3 条）：在第 6 轮的 train 期间，经 CommandInbox 提交 `up`。实现上，该请求在 stop 边界被取消（journal：`finalization` 记录与 `phase=CANCELLED`，错误为 "finalization refuses reconfiguration"）；stop 边界之后提交的请求直接被拒（status 文件为 `rejected`）。两种结果都满足第 3 条"拒绝或取消"。

## 2. A5 的 E1 观测点（补充，运行前固定；不替代 gpu-plan-v2 的判据）

- 岛0 journal 中每个执行了的事务都有一条 `pause_decision`，其 `outer_phase=round-boundary-published`、`allowed=true`、`stalls_peers=true`，`budget_s` 等于按第 5 条输入计算的值。这是 3.8 实现本身的正确性检查；不满足时记为 3.8 实现缺陷。
- quorum 运行：该 step 岛0 只有一次 PUSH；bridge 磁带里没有 "conflicting PULL permits" 和 "invalid PULL permit"。

## 3. A6b（4.4）的 E1 操作与补充观测

- 按 E2 plan-v2 的 C3（Qwen3-1.7B，DP=2，DistOpt，GBS=16，`num_steps_per_rollout=1`）和 G-4.4 执行。重建操作：在第 3 轮开始前的安全点提交 `python -m yeto.rl.engine.controller --state-dir <岛 state dir> rebuild-trainer rb1 --expected-epoch 0 --deadline-s 900`。
- 补充观测（运行前固定）：
  1. journal 依次为 `VALIDATING → WAIT_SAFE → REBUILDING_TRAINER → SUCCEEDED`；cut manifest 中 `progress.local_step = 3`（第 3 轮安全点之前已完成 3 次 optimizer step，每轮 1 步），`outer.settled = true`，`ledger.carried_over = 0`，`ledger.ready_unconsumed = 0`。
  2. `rl_driver_start` 只有 1 条；`rl_trainer_rebuilt` 1 条，其 `policy_version = 3`、`sync/publication_members` 为全部成员。
  3. **样本一致（替代数值比较）**：重建后下一轮（rollout 3，即重建后的第 1 轮）的 `trained_sample_ids_sha256` 与不重建基线 B1 的同一轮相等；该轮开始前 rollout 进程的数据游标 `{sample_offset, epoch_id, sample_group_index, sample_index}` 与 B1 相等。这两条都是精确相等。重建前后的 grad_norm/loss 只记录，不作判据。
  4. `SwappableActor.generation` 为 1（**运行前更正，2026-09-30**：原写"REBUILD_OLD 时为 2"不可能成立，因为 `rebuild_same_shape` 只在重建成功时 swap 一次，失败的那次尝试不会 swap）。RESTORED 与 REBUILD_OLD 靠 journal 中 `rebuild.outcome` 和 `rebuild.attempts` 区分：REBUILD_OLD 时 `attempts` 的 stage 依次为 `create_training_models`、`done`。REBUILD_OLD 路径用 `--rl-test-inject-rebuild-fail` 触发，它让 fork 中 `create_training_models` 的第一次调用失败。
- 已知限制（不放宽判据）：`REBUILDING_TRAINER` 阶段**没有** deadline 强制终止。默认 watchdog 只杀 rollout 事务在 INITIALIZING/VERIFYING 阶段新启动的 cell，trainer 重建阻塞时只会写 journal；超时由 E2 plan-v2 G-4.5 的各项和外层硬超时覆盖。

## 4. watchdog 默认动作（与 A4 E1-D 同批执行；判据为 `evidence/infra-e1/plan.md` E1-D 的补充，运行前固定）

注入：launcher 加 `--rl-test-inject-update-weights-block-s 600`（要求同时带 `--rl-elastic`，会 `export YETO_RL_TEST_INJECT_UPDATE_WEIGHTS_BLOCK_S=600.0`）。效果：本进程第一次调用成员 `update_weights` 之前，`MilesPublisher` 最多阻塞 600 s，每 1 s 检查一次目标 cell 的 worker actor 是否存活（经 fork `RayWorkerManager` 按 generation 取句柄并调用 `__ray_ready__`），一旦有 actor 死亡就立即报错。**这是在 yeto 侧模拟的阻塞，不是 SGLang 内部真的挂起**：它验证的是"watchdog 杀掉目标 generation → 阻塞的发布调用失败 → REBUILD_OLD"这一链路，以及被杀进程和 GPU 是否真正释放；真实引擎挂起能否被 kill 打断不在本用例范围内。deadline 120 s。实现提交见本文件所在提交的前一个代码提交。通过条件全部满足：
1. 120 s 到期后，journal 出现 `watchdog`（`target_cells` 为新 cell）和 `watchdog_action`（`killed` 列出这些 cell 的 worker 与 generation）；
2. 阻塞的调用在 60 s 内返回错误，事务终态为 `REBUILT_OLD`（审查 F2 修复后，watchdog 触发的事务不会再提交为 SUCCEEDED）；
3. 旧成员集合与其 SGLang 进程 PID 不变；
4. **被杀 cell 所在 bundle 的 GPU 上 `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 为空**（按 bundle↔UUID 记录定位 GPU），并在 `REBUILT_OLD` 之后 60 s 内满足；
5. fork 的 health monitor 没有把被杀的 cell 重新拉起（`get_cell_statuses` 中不是 Serving）。
任一条不满足即判未通过；第 5 条不满足时，另记为待 fork 处理的问题。

## 5. 与 CPU 测试的关系

`tests/test_rl_reconfig_x6.py`、`tests/test_rl_trainer_rebuild_e1.py`、`tests/test_rl_reconfig_e1.py` 中的 watchdog 测试都使用 fake，只证明 yeto 侧协议，**不作为**上述任何一项的验收证据。

## 6. 第三轮审查补充（2026-09-30，运行前固定）

- §4 探测的口径（审查 M2）：阻塞开始时记录每个目标 worker 的 (name, generation)。之后出现以下任一情况即判定"目标 generation 已死"：worker 列表为空（cell 已停止）、name 或 generation 变化（被 health monitor 以新 generation 拉起）、actor 已死（RayActorError）。其他探测异常一律写入 stderr，并以 "liveness probe failed" 报出，**不算作被 kill**；RESULT 中若出现这类报错，§4 判据 2 判为未通过。
- watchdog 在两类边上的行为不同（审查 L1）：rollout 边在 watchdog 触发后走 REBUILD_OLD，并恢复旧成员。trainer 边（E3，4.7）的 TrainerTransition 执行期间不检查 watchdog，只在提交前复查一次；若已触发，则**不提交**，转 RECOVERY_REQUIRED，并写 `trainer_recovery_hint`（action=restore_old）。这时 trainer 已经换成目标形状，只能人工或按 hint 恢复。
- F-R1（审查 L3）：fork 缺口 F-R1 解决之前，**不得认证任何需要在释放出的 trainer GPU 上启动 engine 的 trainer 边**（role-transfer，trainer→rollout 方向），因为 `bind_members` 在真实 fork 上无 cell 可绑。启动期的能力检查没有实现：yeto 侧没有办法从 fork 读出"是否存在可重绑的停止 cell"，只能靠该边真正执行时 `bind_members`/`rebind_cell` 失败，事务再走 REBUILT_OLD。
