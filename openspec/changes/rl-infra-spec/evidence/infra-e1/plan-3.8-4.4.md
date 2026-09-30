# INFRA-E1 待 GPU 验证计划：3.8（X6，对应 A5）与 4.4（对应 A6b）

> **已被取代（2026-09-30）**：由 `plan-3.8-4.4-v2.md` 取代。判据的唯一来源是集成分支上的 `gpu-plan-v2.md` §2–§3。本文件仅作记录保留，不得据此执行。

状态：**未执行**。本文件在任何运行之前提交。判据、容差、seed 数和比较口径在此固定，执行后不得修改。编号与 `gpu-plan.md` §3 的 A5、A6b 对齐；PLAN-V2（`/home/michael/work/gpu-plan-v2`，由主 agent 另写 `gpu-plan-v2.md`）把这两项排进批次时，以本文件的判据为准。若 PLAN-V2 的机器/时长/费用与本文件不同，以 PLAN-V2 为准，**判据不变**。
执行时在 `evidence/infra-e1/<A5|A6b>/<run>/` 存放：原始日志、两岛事件磁带、syncer 事件磁带、controller journal（`reconfig/journal.jsonl`、`epochs.json`）、ledger、cut manifest、GPU 断言输出和无残留证明。

## 0. 共同前提（缺一项不启动）

1. 代码：`infra-e1` 合入 integ-decl 后的 SHA（运行时记录）。本轮新增的代码都要包含：3.8 outer-phase / finalization / 暂停预算输入（533afdc）、4.4 controller 驱动的 trainer 重建（d397cf3）、默认 watchdog 终止目标 generation（2b67145）、launcher eval 接线（f79e016）；另需 E2/E3 的 `SwappableActor` 接线（integ-s2 已合入）。
2. 镜像：Miles `yeto/ports`=0af62f4d 的钉住镜像（`ghcr.io/michaellchung/yeto-miles-ports@sha256:c6f5455c…d602ed` 或其后继）。运行前生成 1.1 runtime manifest，核对 digest 和 `MILES_NEXT_COMMIT`。必须使用 `--use-miles-router`。
3. profile：1.2 兼容 profile，即 Qwen3-0.6B LoRA r16 all-linear、GRPO 默认 spec、bf16、TP=PP=CP=EP=1。执行模式为 `partitioned-serial`，外层为 `strict-avg`（fixed roster：`quorum==learners`、`grace_ms=0`、`max_base_lag=0`）。运行前记录 `algorithm_spec_sha256`。
4. 每岛资源（单节点 8 卡，同型号）：`T4R2S2` 与 `T4R4S0`，1.6 格式。attestation 中两条 `rollout-only` 边双向列出；fork 启动时声明 c0–c3 四个 cell，初始只启动 c0、c1。
5. 开关：`--rl-elastic`（开启 elastic 元数据，因此 rollout 会报告 data cursor）。A5 另加 `--rl-elastic-quorum-timeout-s` 与 `--rl-elastic-idle-flow-timeout-s`，取值见下。
6. seed：`--seed 1234 --rollout-seed 42`。每项 1 个 seed：判据是逐项相等或出现与否，不依赖数值统计。
7. 故障注入（X6-b 的阻塞、§3 的 `update_weights` 阻塞）：**尚未实现**。执行前要在实验快照里补一个只由环境变量开启的注入点，默认路径不变，并先提交；没有注入点就不跑这两项。
8. 硬超时：每次运行外层 `timeout 5400`，另有独立 watchdog 按 PID 组终止。结束后 `nvidia-smi --query-compute-apps` 必须为空。

## 1. A5 — 3.8 X6：两小岛 strict 下的手动双向 rollout 切换

**拓扑**：两个岛（learner 0 和 1，各 8 卡），一个 syncer/head。head 的放置按 `gpu-plan.md` §6：与两岛同 VPC 或同机房，路径上不经过会丢空闲流的 NAT。开始前先做 30 分钟空闲流探测，得到 `idle_flow_timeout_s`（记录方法与原始结果），并用 `--rl-elastic-idle-flow-timeout-s` 传入两个岛。
**全局轮数** 8。**基线**：同一 seed、同一拓扑，不发任何请求的一次运行（B0）。

| 用例 | 步骤 | 通过判据（全部满足） |
|---|---|---|
| X6-a 预算内暂停 | 默认 quorum 900 s、margin 0.5。岛 0 在 rollout 2 的安全点提交 `up T4R2S2→T4R4S0`（deadline 300 s），在 rollout 5 提交 `down`（deadline 300 s） | (1) 两个事务的 journal 终态都是 `SUCCEEDED`，config_epoch 0→1→2；(2) 两岛每轮的 `trained_sample_ids_sha256` 与 B0 **逐轮相等**，optimizer 步数相等；(3) syncer 磁带：每个 global step 的 PUSH 集合 = {0,1}，数量、`round_attempt`、`base` 与 B0 相等，roster 不变，没有 learner 被移出或重连；(4) 两岛最终 policy hash 相等，且等于 B0；(5) 岛 0 journal 的 `pause_decision` 为 `outer_phase=round-boundary-published`、`allowed=true`、`stalls_peers=true`、`budget_s=min(450, idle_flow_timeout_s)`；(6) 岛 1 暂停期间停在 `wait_global_policy`，事后以同一 global policy 继续 |
| X6-b 跨 quorum 超时的暂停（PULL 重发） | syncer `--quorum-timeout-s 120`（经 `--rl-elastic-quorum-timeout-s 120` 同时传给 syncer 与岛）。岛 0 用 `--rl-elastic-pause-margin 2.0`（仅本用例；预算为 min(240, idle_flow)）。在 rollout 2 提交 `up`（deadline 230 s），并在 publish_members 前人为阻塞 150 s（冻结快照中的测试注入开关），使暂停跨过一次 quorum 超时 | (1)–(4) 同 X6-a；(7) syncer 磁带在暂停期间至少有一次同一 step 的 PULL 重发（`round_attempt=1`）；(8) 岛 0 磁带、bridge 磁带里没有 "conflicting PULL"、"invalid PULL" 或重连；该 step 只 PUSH 一次。若 `idle_flow_timeout_s` < 150 s，本用例只在同 VPC 的 head 上跑；做不到就记为"环境不满足，未运行"，**不得**改用更短的阻塞 |
| X6-c 超预算请求被拒 | 同 X6-b 的 quorum 120、默认 margin 0.5：提交 deadline 61 s 的请求 | `request` 立即被拒（`pause not allowed ... exceeds budget 60`），journal 中没有该 request；训练与 B0 逐轮相等 |
| X6-d finalization 拒绝切换 | 在最后一轮（rollout 7）的 train 期间提交 `up`（经 CommandInbox）；stop 边界之后再提交一次 | 第一个请求的 journal 终态是 `CANCELLED`，错误为 "finalization refuses reconfiguration"，并有一条 `finalization` 记录；岛 0 事件磁带有对应的 `rl_reconfiguration CANCELLED`；第二个请求被拒（status 文件为 `rejected ... finalization`）；没有任何 fork start/stop 调用；两岛最终 policy 与 B0 相等 |
| X6-e 另一岛 finalizing 时的安全点 | 可选：当 syncer 在岛 0 安全点前已进入 finalizing（最后一个 step 已提交）时提交请求 | `pause_decision.outer_phase=finalizing`、`allowed=false`，终态 `CANCELLED` |

报告必须写明：只完成了 rollout 能力（3.8 原文），trainer 边不在此范围。

**时长/资源**：沿用 `gpu-plan.md` A5，2 岛 × 8×H100 加 head，约 5 h。本地自有卡执行时，时长约为 B0 + X6-a + X6-b + X6-c/d 共 4 次 × 约 40 分钟。

## 2. A6b — 4.4：driver 在安全点替换端口背后的 trainer

**拓扑**：单岛 8 卡，strict-avg，本地 syncer（`--num-learners 1`，fixed roster）。配置与 E2 plan-v2 的 C3 一致（Qwen3-1.7B、DP=2、DistOpt 打开，GBS=16）。前提：`args.requested_load is None`。
**全局轮数** 6。**基线** B1：同一 seed，不重建。
**操作**：rollout 3 的安全点，经 CommandInbox 提交 `rebuild-trainer`（`python -m yeto.rl.engine.controller --state-dir … rebuild-trainer rb1 --expected-epoch 0 --deadline-s 900`）。

通过判据（全部满足）：
1. journal：`VALIDATING → WAIT_SAFE → REBUILDING_TRAINER → SUCCEEDED`，`rebuild.outcome ∈ {RESTORED, REBUILD_OLD}`；cut manifest 的 `progress.local_step=3`（即 3×`num_steps_per_rollout`）、`outer.settled=true`、`ledger.carried_over=0`、`ready_unconsumed=0`。
2. 不重复 initialize/after_local_train：learner 日志与 bridge 磁带中 `initialize`（strict runtime 的 `initialize`/`wait_for_initial_policy`）、`after_local_train` 的计数与 B1 相等；事件磁带里 `rl_driver_start` 只有 1 条。
3. 外层进度不重放：syncer 磁带里每个 global step 只有一次 PUSH，`history` 与 B1 相同；ledger 每轮各只有一条 `prepared`、`optimizer_applied`、`outer_recorded`，没有组被训练两次；`rl_local_round` 为 6 条。
4. 重发正确权重：`rl_trainer_rebuilt` 的 `policy_version=3`，`sync/publication_members` 等于全部成员；重建后第一次 publish 的 policy hash 等于 cut 的 `progress.policy_hash`，每个 engine 的 checksum 读回与重建前那次发布相同。
5. 端口未 rebind：`SwappableActor.generation` 从 0 变为 1（或 REBUILD_OLD 时变为 2）；Disposer 退出时只 dispose 当前 handle，旧 handle 不被 dispose 第二次（按 E2 plan-v2 G-4.4 的检查方法）。
6. 数值：重建后第 4 轮 trainer 的 `grad_norm`、loss 与 B1 第 4 轮比较。**本项只记录，不作判据**：strict 每轮都会 reset，而且 rollout 在两次运行之间不是逐位确定的。逐位比较由 E2 的 A6（X3，Modal `H100!:8`）负责。

**失败即**：4.4 不勾选，写明原因；只有提出原因并修复之后才重跑。
**时长**：沿用 `gpu-plan.md` A6b，8×H100 约 3 h（B1 与重建运行各约 1 h）。

## 3. 3.7 watchdog 默认动作（与 A4/E1-D 同批，附在 A5 之后执行）

在 up 事务中人为阻塞新 engine 的 `update_weights`（冻结快照中的测试注入），deadline 120 s。通过判据：120 s 到期后，journal 出现 `watchdog`（`target_cells` = 新 cell）和 `watchdog_action`（killed = 这些 cell 的 worker 与 generation）；阻塞的调用在 60 s 内返回错误；事务终态为 `REBUILT_OLD`，旧成员不变，其 SGLang 进程 PID 不变；fork 的 health monitor 没有把被杀的 cell 重新拉起（`get_cell_statuses` 中不是 Serving）。如果 health monitor 重启了被杀的 cell，就判为未通过，并记录为待 fork 处理的问题。

## 4. 与 CPU 测试的关系

`tests/test_rl_reconfig_x6.py`、`tests/test_rl_trainer_rebuild_e1.py`、`tests/test_rl_reconfig_e1.py::test_default_watchdog_*` 用的是 fake syncer、fork 和 trainer，只证明 yeto 侧协议，**不作为**上述任何一项的验收证据。
