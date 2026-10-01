# INFRA-E1 待本地 GPU 验证计划（3.3 / 3.4 / 3.5 / 3.7，及 3.1/3.2/3.6 的 GPU 旁证）

状态：**未执行**。按用户 2026-09-30 决定暂停一切 GPU/云验证，本计划只供日后在用户自有卡上执行。本文件先于任何运行提交；判据、容差、seed 数与比较口径在此固定，执行后不得修改。执行时另建 `evidence/infra-e1/<task>/<run>/` 存放原始日志、事件磁带、journal、ledger 与 GPU 断言输出。

## 0. 前置条件（缺一项不启动）

1. 代码：`infra-e1` 分支合入 integ-decl 后的 SHA；launcher/entry 接线完成（见 progress.md INFRA-E1 节“接口请求”第 1 条：`entry.py` 构造 `IslandController`、`BatchLedger`、`ElasticPlacement`，并把 fork 启动时声明的 rollout cell 列表传给 `MilesRolloutPool(declared_cells=...)`）。**接线未完成时本计划不可执行。**
2. 镜像：包含 miles `yeto/ports` 0af62f4d（M1–M6 + M2/M4 修复）的钉住镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:c6f5455c…d602ed`（或其后继，运行前 1.1 manifest 核对 `MILES_NEXT_COMMIT`）。必须带 `--use-miles-router`（`admit_cordoned` 依赖 Miles router）。
3. 硬件：单节点 8 卡、同型号（记录 `nvidia-smi -L` 的 UUID 与 `nvidia-smi topo -m`），满足 1.2 选定 profile 的显存；无付费云资源。
4. profile：1.2 的兼容 profile（Qwen3-0.6B LoRA r16 all-linear、GRPO、bf16），执行模式 `partitioned-serial`，外层 `none`（LocalOnlySync，`--rl-single-island-no-sync`）为主实验；strict-avg 两小岛属于 3.8，不在本计划。
5. 资源配置与认证（1.6 格式）：`T4R2S2`（trainer G0–G3，rollout G4–G5，standby G6–G7，TP1 engine）与 `T4R4S0`（rollout G4–G7）；attestation 中两条 `rollout-only` 边双向列出，`runtime_fingerprint` 与运行指纹一致。fork 启动时声明 4 个 rollout cell（c0–c3），初始只启动 c0/c1。
6. 硬超时：每次运行外层 `timeout 3600`；事务 deadline 按下表；独立 watchdog 脚本按 PID 组终止（本地无付费资源，但仍要求无残留进程：结束后 `nvidia-smi --query-compute-apps` 为空）。

## 1. 实验与判据（事先固定）

| ID | task | 步骤 | 通过判据（全部满足） | 失败即 |
|---|---|---|---|---|
| E1-A | 3.4 X2 | 12 轮；第 3 轮前 `request up T4R2S2→T4R4S0`（deadline 600s），第 8 轮前 `request down`（deadline 600s）。同 seed 另跑一次无请求的固定 T4R2S2 基线 | (a) 两次事务 journal 终态 `SUCCEEDED`，config_epoch 0→1→2；(b) 每轮 `trained_sample_ids_sha256` 与基线**逐轮相等**，optimizer 步数相等（12）；(c) 第 3–7 轮 `rl_publication.sync/publication_members` = 4 个成员，其余轮 = 2 个；(d) trainer 进程 PID 与 trainer GPU UUID 集合全程不变；(e) 切换期间 G0–G3 以外无新进程出现在池外 GPU（池=8 张 UUID）；(f) down 事务在 drain 后才 `stop_cells`（journal 顺序）；(g) 备用卡 GPU-hours 计入报告 | 3.4 不勾选，记录原因 |
| E1-B | 3.5 | E1-A 的 up 事务中，对新 cell 在 `admit_cells` 前后各采样 router `/worker_inflight` 与 cordon 列表；另跑一次注入：在 `publish_members` 的 `check_weights` 前把一个新 engine 的权重用 `update_weights_from_disk` 覆盖为 base 模型（测试专用，冻结快照中的故障注入文件开启） | (a) `admit_cells` 前新 cell 在 router cordon 列表中且 in-flight=0、无请求路由到它；(b) 注入运行中事务终态 `REBUILT_OLD`，被注入 cell 从未出现在非 cordon 路由中，`stop_cells` 回收它，旧 2 成员仍报告同一 policy token；(c) 迟到的旧 generation ACK：在 down 后对已停 cell 的旧 URL 发送 `update_weight_version`，不改变任何在役成员版本（`get_cells_weight_versions` 前后相等）；(d) 旧 epoch 的 `start_update_weights(members, expected_epoch=旧值)` 被拒绝（`MembershipEpochMismatchError`） | 3.5 不勾选 |
| E1-C | 3.3 X5 | 带工具等待的多轮 agent 负载（`examples/rl_algorithms` 中的工具环境或 1.7 的 tool-wait 探针负载），在一轮 rollout 进行中提交 down 请求 | (a) 事务在该轮 `generate` 返回后的边界才进入 QUIESCING（journal 中 QUIESCING 的 rollout_id = 请求后的下一边界）；(b) 若 drain 时 active=0 且 tool-wait>0，journal 不出现 `stop` fork_op，路由成员不变；(c) 将 `T_drain` 设为 5s 并让工具等待 30s：终态 `CANCELLED`，被 cordon 的 cell 被 uncordon，外部工具调用计数（工具服务端日志）无重复 | 3.3 不勾选 |
| E1-D | 3.7 | 失败矩阵，每项独立一次运行：①`start_cells` 期间 kill 一个新 SGLang 进程；②`publish_members` 期间 kill 新 engine；③`stop_cells` 半失败（在 provider stop 前注入异常，fork 返回 `incomplete`）后第二次成功；④同③但持续失败超过 `T_recovery`；⑤commit CAS 之后、RESUMING 之前 kill learner，再重启；⑥QUIESCING 中 kill learner 再重启；⑦fork InferenceController 重启（epoch 归 0） | ①② 终态 `REBUILT_OLD`，成员恢复为旧集合且下一轮 token 校验通过；③ `SUCCEEDED`，journal 有一条 `incomplete` 与同一 stop 的重试；④ `RECOVERY_REQUIRED`，driver 以 DriverError 结束且不再消费数据（ledger 无新 `prepared`）；⑤ 重启后该请求 `status` = `SUCCEEDED`（`recovered_after_restart`），成员 = journal 成员；⑥ 重启后 `CANCELLED`；⑦ 重启对账调用 `restore_membership_state(epoch=journal 值, expected_current_epoch=0)`，之后事务可继续。所有项：learner/trainer 身份（PID、policy hash）与 ledger 无重复消费 | 3.7 不勾选 |
| E1-E | 3.1/3.2/3.6 旁证 | 与 E1-A 同一次运行采集 | 梯度累积中（train_step 进行中）提交的请求在该步返回后的边界才执行；重复提交同一 request_id 返回同一 tx_id；ledger 每轮恰好一条 prepared/optimizer_applied/outer_recorded | 对应项记录缺陷 |

容差：以上判据全部为**精确相等/精确出现**，无数值容差。E1-A (b) 若出现不一致即判失败，不以“浮点噪声”解释（样本 ID 与步数与数值无关）。每项只跑 1 个 seed（判据不依赖数值）；失败只有在提出原因并修复后才重跑，不重复启动同一实验。

## 2. 预计时长与费用

本地自有卡，无云费用。E1-A 约 40 分钟（两次 12 轮），E1-B/C 各约 30 分钟，E1-D 7 次各约 15 分钟。总计约 4 小时 GPU 时间。

## 3. 记录

每次运行记录：分支 SHA、镜像 digest、GPU UUID 列表、开始/结束时间、`journal.jsonl`/`epochs.json`/`ledger/journal.jsonl` 原文、事件磁带、router `/worker_inflight` 采样、无残留进程证明。
