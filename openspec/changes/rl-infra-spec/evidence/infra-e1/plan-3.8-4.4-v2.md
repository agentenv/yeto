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
- gpu-plan-v2 A5 第 2 条"磁带中至少 1 次 PULL 重发"的取证位置（主 agent 裁定，2026-09-30）：以岛0 learner JSONL 磁带中的 `rl_pull_resend` 记录为准（0f0bcf3）。
- A5 岛0、岛1 的 attestation `runtime_fingerprint` 在 CPU 上取：launch 命令与正式运行完全相同，另加 `--rl-print-attestation-fingerprint`。learner 按同一套 Miles argv 构造，调用同一个 `ports_runtime_fingerprint`，打印一行 JSON 后退出，不连接 Ray，也不需要 GPU。

## 3. A6b（4.4）的 E1 操作与补充观测

- 按 E2 plan-v2 的 C3（Qwen3-1.7B，DP=2，DistOpt，GBS=16，`num_steps_per_rollout=1`）和 G-4.4 执行。重建操作：在第 3 轮开始前的安全点提交 `python -m yeto.rl.engine.controller --state-dir <岛 state dir> rebuild-trainer rb1 --expected-epoch 0 --deadline-s 900`。
- 补充观测（运行前固定）：
  1. journal 依次为 `VALIDATING → WAIT_SAFE → REBUILDING_TRAINER → SUCCEEDED`；cut manifest 中 `progress.local_step = 3`（第 3 轮安全点之前已完成 3 次 optimizer step，每轮 1 步），`outer.settled = true`，`ledger.carried_over = 0`，`ledger.ready_unconsumed = 0`。
  2. `rl_driver_start` 只有 1 条；`rl_trainer_rebuilt` 1 条，其 `policy_version = 3`、`sync/publication_members` 为全部成员。
  3. **样本一致（替代数值比较）**：重建后下一轮（rollout 3，即重建后的第 1 轮）的 `trained_sample_ids_sha256` 与不重建基线 B1 的同一轮相等；该轮开始前 rollout 进程的数据游标 `{sample_offset, epoch_id, sample_group_index, sample_index}` 与 B1 相等。这两条都是精确相等。重建前后的 grad_norm/loss 只记录，不作判据。
  4. **运行前更正（主 agent 裁定，2026-09-30）**。原文："`SwappableActor.generation` 在 RESTORED 时为 1，在 REBUILD_OLD 时为 2。"更正后：`generation` 在两种路径下都为 1。原因是原文在逻辑上不可能成立：`rebuild_same_shape` 只在重建成功时 swap 一次，失败的那次尝试不 swap。两条路径改由 journal 的 `rebuild.outcome` 与 `rebuild.attempts` 区分：REBUILD_OLD 时 `attempts` 的 stage 依次为 `create_training_models`、`done`。REBUILD_OLD 路径用 `--rl-test-inject-rebuild-fail` 触发，它让 fork 中 `create_training_models` 的第一次调用失败。更正发生在任何运行之前，替代判据不弱于原意。
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

## 7. F-R1 cell 绑定只在内存：对 E1-D ⑤⑥⑦ 原地重启用例的影响（2026-09-30，运行前评估）

fork `yeto/ports` 2f23a0fc（F-R1）的 cell 绑定（`rebind_cell`/`unbind_cell`、`set_pg_view` 的具名视图）只保存在 `RayWorkerManager` 的内存里。learner 原地重启（`--rl-elastic-restart-attempts`）时，旧 Ray job 连同 fork 控制器一起结束；新 job 按启动参数（placement map 的 `rollout_cells`）重新声明 cell，只启动 `start: true` 的 cell，运行中改过的绑定全部丢失，fork membership epoch 归 0。controller 在 `open()` 时把 fork 实际在役的成员与 journal 提交的成员比对，不一致即判 RECOVERY_REQUIRED，不会自动启停 cell（alignment G11"首版重启回启动形状"仍待用户批准）。据此：

- **⑤（COMMITTED 之后 kill，事务为 up）**：journal 中该请求判 `SUCCEEDED`（`recovered_after_restart`），但重启后 fork 只运行启动时的 cell，与提交的成员（多出 up 加入的 cell）不一致，island 转 **RECOVERY_REQUIRED**。`evidence/infra-e1/plan.md` E1-D ⑤ 原判据"成员 = journal 成员"在现设计下不可达。可选：(a) 按"status=SUCCEEDED 且 island 以 membership mismatch 转 RECOVERY_REQUIRED、不消费数据"判定，属于更改判据，须主 agent 在运行前裁定；(b) 实现"重启后按 journal 重放提交的成员"，这是 G11 的范围，需用户批准；(c) 让 ⑤ 的被杀事务以启动配置为目标（例如先 up，再在 down 的 COMMITTED 后 kill），重启后成员与启动形状一致。
- **⑥（QUIESCING 中 kill）**：未 release，journal 判 CANCELLED；重启后成员等于启动形状，前提是 journal 最近一次提交的配置也等于启动配置。先 up 再在 down 的 QUIESCING 中 kill 时，提交的成员不等于启动形状，同样会 RECOVERY_REQUIRED。**执行前须让被杀事务之前的已提交配置等于启动配置**，即 ⑥ 只在 epoch 0 上做。
- **⑦（fork 重启、epoch 归 0）**：`restore_membership_state` 能把 fork 的 epoch 镜像恢复到 journal 的值，但恢复不了绑定和在役集合；两者与启动形状不一致时，结果与 ⑤ 相同。
- **A9（4.7）**：role-transfer 靠 `bind_members` 做的绑定在重启后丢失；A9 f5（kill learner）因此只能以 RECOVERY_REQUIRED 结束，与 E3 plan 的写法一致。

结论：在 G11 获批并实现"按 journal 重放提交的成员"之前，⑤⑥⑦ 只有在被杀事务之前的已提交配置等于启动配置时，才能得到原判据的结果；否则按原判据应判"未通过（设计限制）"。请主 agent 在运行前从 (a)/(b)/(c) 中选定。

### 7.1 裁定与用例安排（主 agent 裁定，2026-09-30，运行前）

主 agent 选 **(c)**。上文 (a)/(b)/(c) 与评估原文保留不改。`evidence/infra-e1/plan.md` E1-D ⑤⑥⑦ 的原判据（包括"成员 = journal 成员"）**保持不变**，只调整用例的安排，使被杀事务之前的已提交配置等于启动配置：

- **⑤**：先 up 并提交（T1R1S1→T1R2S0，epoch 0→1）；再发 down（T1R2S0→T1R1S1），带 `--rl-test-kill-learner-at COMMITTED`，在 down 的 COMMITTED 处杀掉 learner。杀的是第二个事务，而"只杀一次"按 state dir 计，所以 up 事务写入 COMMITTED 时不能触发。做法：先不带 kill 开关跑 up；up 提交后用 `--rl-elastic-restart-attempts` 与同一 state dir 重启，这次才带 `YETO_RL_TEST_KILL_LEARNER_AT=COMMITTED`，然后发 down。也可以为该开关加"第 N 次命中才杀"的计数；未实现，需要时另提。重启后 journal 中 down 为 SUCCEEDED（`recovered_after_restart`），已提交配置 = T1R1S1 = 启动配置，fork 重启后的在役成员等于 journal 成员。
- **⑥**：在首个事务（epoch 0 的 up）的 QUIESCING 处杀（`--rl-test-kill-learner-at QUIESCING`）。重启后请求为 CANCELLED，成员等于启动形状，也等于 journal 成员。
- **⑦**：同样只在 epoch 0 上做，即首个事务之前、已提交配置等于启动配置时重启 learner（fork 随之重启，epoch 归 0）。journal 的 epoch 为 0，`restore_membership_state` 对账后事务可以继续。
- **已知限制，不作为本轮判据**：已提交配置不等于启动配置时重启，island 转 RECOVERY_REQUIRED（fork 回到启动形状，与 journal 成员不一致）。这是 G11"首版重启回启动形状"的设计语义，记录在此，本轮不测。
- **A9 f5**：以设计语义为预期，即 RECOVERY_REQUIRED 加 `trainer_recovery_hint`（`restore_old`，指向该 cut）。已与 `evidence/infra-e3/plan-v3.md` 第 99 行 f5 的写法核对，一致。

### 7.2 用户裁定 (b) 已实现（2026-10-01，INFRA-E1，CPU 通过；GPU 未跑）

用户（SESSION6 §7.3）改选 **(b)**：实现"已提交成员配置的 learner 重启恢复"，(c) 保留为回归测试，恢复失败走 (a) 兜底。设计与失败矩阵见 `recovery-design.md`；实现在分支 infra-e1-recovery（controller `open()` 差分恢复 + `confirm_recovery` 放行；fork 不改）。对 ⑤⑥⑦ 的影响：

- **⑤**：按 `plan.md` 原判据直接执行——up 提交（T4R2S2→T4R4S0）后在 COMMITTED 处 kill learner，重启后该请求 `SUCCEEDED(recovered_after_restart)`，fork 由恢复启动 c2/c3，首次发布后 `recovery verified`，成员 = journal 成员。§7.1 的"先 up 再在 down 的 COMMITTED kill"安排降为回归用例（期望：无 `recovery` 记录）。
- **⑥**：不变（CANCELLED，无恢复）。
- **⑦**：可在任意 epoch 做；journal epoch > 0 时重启对账 `restore_membership_state(epoch=journal 值, expected_current_epoch=0)` 后由恢复补齐成员。
- 新增判据（运行前固定，补充不替代原判据）：恢复成功的运行，journal 顺序 `recovery planned → fork_op(rec-*) done… → recovery membership_restored → recovery verified`，tape 有 `rl_reconfiguration result=RECOVERED`，其后 ledger 新 `prepared` 的 policy token 等于重启发布的 token；恢复失败的运行必须 `recovery failed` + `RECOVERY_REQUIRED`，且 ledger 无新 `prepared`。
- A9 f5（role-transfer 后 kill learner）：仍以 RECOVERY_REQUIRED + `trainer_recovery_hint` 为预期（trainer 形状不等于启动形状，恢复前置拒绝）。

## 8. `--rl-elastic` 强制启用 Miles router（2026-09-30，运行前）

F-E1 重跑（`evidence/infra-v2-b1/fe1r/`）暴露：fork 的 cordon、drain_cells、admit_cells 以及 cordoned update_weights 都要求 `--use-miles-router`（`server_cell._assert_cordonable`、`inference_controller.start_update_weights`）。自提交 ae42dcf 起，`--rl-elastic` 的 Miles argv 固定带 `--use-miles-router`，默认 argv 不变。岛启动前另做一次检查（`check_elastic_miles_args`），缺少 Miles router、colocate、rollout offload 三种情况直接拒绝。
**对判据的影响（执行要求，不改判据）**：用 Miles router 与用 SGLang router 的运行不是同一配置。凡是拿切换运行和基线逐轮比对 sample id 的用例（A4 E1-A 的固定基线、A5 的基线、A6b 的 B1），基线运行也必须带 `--rl-elastic`（resources 与切换运行相同，不发请求），这样两边都走 Miles router。
