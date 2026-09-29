# F 阶段设计：IslandStatus / pool epoch（7.1）与运行中扩池/缩池顺序（7.2）

状态：设计文档，本轮不实现、不做实际云扩缩（出处：用户 2026-09-29 无人值守轮指示；tasks 第 7 组本轮范围说明）。7.3 不在本文件范围，占位见 design D12。

本文只在单岛、单资源池层面展开 design D10（`IslandStatus` 预留）与 D11（两个时间尺度）。它不改变 D4 事务状态机、D2 校验顺序或任何验收原文；与它们冲突时以 design.md 为准。

接口引用约定：`[已实现 file:line]` 指 yeto 集成分支 `origin/integ-decl`（8c64a65）或 miles fork `yeto/ports`（0af62f4d）中已存在的代码；`[缺口 Gx]` 指尚未实现、列在 §6 的接口。fork 路径均相对 `/home/michael/work/miles-elastic`。

---

## 0. 原文逐条覆盖映射

| 原文要点 | 本文位置 |
|---|---|
| 7.1 设计 IslandStatus | §1.1 字段、§1.2 状态机 |
| 7.1 pool epoch | §1.3 三个 epoch 的权威关系、§1.4 journal 字段 |
| 7.1 未来 pause-budget/veto | §1.5 |
| 7.1 包含资源需求 | §1.1 `resources` 组 |
| 7.1 safe point | §1.1 `safe_point` 组；§1.2 |
| 7.1 outer phase | §1.1 `outer` 组；§1.5 |
| 7.1 预计暂停/恢复 | §1.1 `expected_pause_s` / `recovery_upper_bound_s`；§1.5 |
| 7.1 验收：运行/切换/失败状态完整 | §1.2 状态机（含 RUNNING、切换各阶段、失败终态、池操作状态）与 §1.6 完整性检查 |
| 7.1 验收：读取状态不触发云扩缩 | §1.7 只读契约 |
| 7.2 运行中扩池/缩池顺序 | §2.2 扩池、§2.3 缩池 |
| 7.2 云就绪→资源认证→pool 提交/排空→释放 | §2.2 步骤 S1–S8、§2.3 步骤R1–R8 |
| 7.2 验收：区分云供给慢路径与岛内快路径 | §2.1 |
| 7.2 验收：worker 增减不等于 DiLoCo 成员变化 | §2.5 |
| 7.2 验收：记录成本 | §2.6 |
| 7.2 验收：回退边界 | §2.4 失败矩阵、§2.7 回退边界 |
| 7.2 验收：不在本轮实现 | 本文只含设计；§6 列缺口，不实现 |
| （派发附加）不预设全局 barrier、不把容器重启当岛内切换、不重复 provider 能力 | §3 |
| （派发附加）stall/退出码 4/6 | §4 |

---

## 1. 7.1 IslandStatus 与 pool epoch

### 1.1 IslandStatus 字段

`IslandStatus` 是 D4 `inspect() -> ReadinessSnapshot + IslandStatus` 的第二部分。`ReadinessSnapshot`（算法/数据就绪）已存在 `[已实现 yeto/rl/engine/execution_profile.py:374]`，`IslandStatus`（资源与事务）尚无代码 `[缺口 G1]`。两者分开：前者由 driver/bridge 填，后者由 controller 从 journal 与 provider 只读查询组装。

| 组 | 字段 | 来源 | 说明 |
|---|---|---|---|
| 身份 | `run_id`, `island_id`, `learner_id` | launcher 配置 | `learner_id` 是外层 logical learner，池变化不改它（§2.5） |
| 池 | `pool_id`, `pool_epoch`, `pool_gpus`（uuid/model/node/index）, `capacity`, `standby`, `reservation_identity`（云/region/实例或集群 ID 列表） | journal 最近一次 `pool_commit` | 字段名沿用 design D11 与 gpu-plan §8.2；`pool_id`/`pool_epoch` 已是 study manifest 必填 `[已实现 yeto/rl/elastic_benchmark/manifest.py:321]`，GPU 列表校验 `[已实现 yeto/rl/elastic_benchmark/capabilities.py:282]` |
| 配置 | `config_id`, `config_hash`, `config_epoch`, `placement`（role→GPU uuid）, `worker_generation` | journal 最近一次 `COMMITTED` | 与 1.6 的 config schema 同格式 `[已实现 capabilities.py:42 ResourceConfig]` |
| fork 镜像 | `fork_membership_epoch`, `fork_incomplete`, `fork_incomplete_reason` | `get_membership_status` `[已实现 miles/ray/rollout/inference_controller.py:137]` | 只作对账用，不是权威（§1.3） |
| 资源需求 | `resource_request`：目标 GPU 型号/数量/互联/镜像 digest/region 约束；`pending_pool_op`（见 §1.2） | 当前 pool 操作的 journal 记录 | 只描述意图；读取它不下单（§1.7） |
| 事务 | `tx_id`, `tx_phase`（D4 状态名）, `tx_kind`（`reconfigure` / `pool_grow` / `pool_shrink`）, `tx_deadline`, `reconfiguration_allowed`, `reason` | journal | 岛内最多一个事务（D4），池操作占用同一槽位 |
| safe point | `last_safe_point_id`, `last_safe_point_at`, `rollout_id`, `optimizer_step`, `driver_safe_point` | 最近一次 `observe_safe_point`；`ReadinessSnapshot.driver_safe_point` `[已实现 execution_profile.py:393]` | `quiescent_cut_blockers` 判定 `[已实现 execution_profile.py:447]` |
| outer | `outer_protocol`, `outer_phase`, `permit`（是否持有 strict permit、fragment/attempt/version 摘要）, `budget_mode` | bridge；`ReadinessSnapshot.sync_phase` | phase 取值与 `OUTER_PHASES` 一致 `[已实现 yeto/rl/engine/pause_audit.py:38]` |
| 暂停预算 | `expected_pause_s`, `recovery_upper_bound_s`, `pause_budget_s`, `stalls_peers`, `veto`（列表：来源+原因+expires_at） | `pause_decision` `[已实现 pause_audit.py:100]` + 未来 planner（§1.5） | 预计值来自已认证边的实测分布上界，未知则 `None` 且 `reconfiguration_allowed=false` |
| 结果 | `last_success`, `last_failure`（tx_id、phase、error、时间）, `last_recovery`, `health`（见 §1.2） | journal | 每个结果都带证据路径 |

`capacity` 与 1.6 `ResourceConfig.capacity` 同名同义（显存/CPU/pinned/object-store/磁盘峰值）。

### 1.2 状态机

`IslandStatus.health` 是岛级粗状态；切换细节仍由 D4 事务阶段 `tx_phase` 表达，这里不另造一套。

```
            ┌──────────────── pool_grow / pool_shrink（§2）────────────────┐
            v                                                              │
RUNNING ──request──> RECONFIGURING(tx_phase ∈ D4 阶段) ──COMMITTED/RESUMING──> RUNNING
   │                         │
   │                         ├── CANCELLED ──> RUNNING（旧配置，epoch 不变）
   │                         ├── REBUILD_OLD ──ok──> RUNNING（旧配置，新 generation）
   │                         └── RECOVERY_REQUIRED（fence，人工处理）
   │
   ├── POOL_PENDING（慢路径：云侧供给/释放进行中，岛照常训练）
   ├── FINALIZING（外层 finalization，所有切换与池操作被 veto）
   └── FAILED（退出码 4/6 路径，§4；不属于岛内事务）
```

| health | 含义 | 允许的新事务 | 退出条件 |
|---|---|---|---|
| RUNNING | 已提交 config/pool epoch，训练中 | reconfigure、pool_grow 慢路径、pool_shrink | request 被接受 |
| POOL_PENDING | 云侧新资源在供给或认证中（S1–S4），或已从池摘除的资源在释放中（R6–R7）；岛内训练不受影响 | 与池无关的 reconfigure 可进行，但不得引用 pending GPU | pool 提交/放弃；释放核实 |
| RECONFIGURING | D4 事务进行中（含池提交、排空两个快路径步骤） | 无（单事务） | D4 终态 |
| FINALIZING | outer phase 为 finalization/budget gate | 无（veto） | run 结束 |
| RECOVERY_REQUIRED | journal 与 worker/fork 不一致，或提交不确定 | 只接受人工恢复命令 | 人工对账后写 journal |
| FAILED | 岛进程已失败，由 launcher 的 fixed-roster 恢复处理 | 无 | launcher 重启或放弃（§4） |

POOL_PENDING 与 RUNNING 可叠加：它是一个标志 `pending_pool_op != None`，不打断训练。这是慢路径与快路径分离的核心（§2.1）。

### 1.3 三个 epoch 的权威关系

| epoch | 何时加 1 | 权威存放 | 镜像 |
|---|---|---|---|
| `pool_epoch` | 池内 GPU 集合变化并提交（§2.2 S6、§2.3 R5）；同一租用内切配置不变（gpu-plan §8.2） | yeto journal，durable CAS | study manifest `resources.pool_epoch` |
| `config_epoch` | 每次 D4 `COMMITTED`（纯资源切换也加，但不产生新 policy 版本，D3） | yeto journal，durable CAS | `ReadinessSnapshot.config_epoch` `[已实现 execution_profile.py:397]` |
| `fork_membership_epoch` | fork 每次成功的 `start_cells`/`stop_cells`（含 incomplete 的重试）`[已实现 inference_controller.py:261,266-331]` | fork 内存（重启归 0，`:74`） | — |

规则（沿用 3.3a 原文，不放宽）：

1. yeto journal 是唯一权威。fork epoch 只是镜像。journal 每条 membership 操作记录 `(tx_id, op, cell_ids, expected_fork_epoch, result_fork_epoch)`，因此 journal 能推出当前应有的 fork epoch。
2. 每次调用 fork 都传入 journal 推出的 expected epoch：`start_cells/stop_cells(expected_epoch)` `[已实现 :233,:261]`、`start_update_weights(members, expected_epoch, admit_cordoned)` `[已实现 :581]`、`admit_cells(expected_epoch, expected_weight_version)` `[已实现 :351]`、`start_commit_weight_version(expected_epoch)` `[已实现 :705]`。
3. controller 或 fork 重启后：先读 journal，再 `get_membership_status` 对账。
   - fork epoch = 0 且 journal 期望 > 0：fork 是新进程，调用 `restore_membership_state(epoch=journal 值, incomplete=journal 中待重试项, last_op=journal 最后一次成功操作, expected_current_epoch=0)` `[已实现 :154]`，CAS 失败即 RECOVERY_REQUIRED。
   - fork epoch 与 journal 一致：继续。
   - fork epoch 超前 journal 一次且等于 journal 中一条“已发出、结果未写”的操作：说明应答丢失，把结果补写进 journal（幂等重试也会得到同一 epoch，`:269-272`）。
   - 其他不一致：RECOVERY_REQUIRED，不以 fork 状态覆盖 journal。
4. `pool_epoch` 与 fork epoch 之间没有算术关系。扩池后的 cell 启动属于 config 事务，会让 fork epoch 前进，但 pool_epoch 只在 GPU 集合提交时前进。
5. fork `incomplete`（`stop_failed` / `start_rollback_failed` / `restored`）在 journal 记为“待重试”，只允许重试同一操作；超出事务绝对 deadline 转 RECOVERY_REQUIRED（3.3a/3.7 原文）。

### 1.4 journal 字段（在 D4 journal 上追加）

D4 已定义 tx_id、from/to config/hash、expected/current epoch、phase、时间、worker generations、safe-point ID、manifest/hash、恢复依据、error。池操作追加：

| 字段 | 说明 |
|---|---|
| `tx_kind` | `reconfigure` / `pool_grow` / `pool_shrink` |
| `pool_id`, `pool_epoch_from`, `pool_epoch_to` | CAS 用 |
| `pool_gpus_add` / `pool_gpus_remove` | uuid 列表；add 项含认证证据路径 |
| `reservation_add` / `reservation_remove` | 云、region、实例/集群/app ID、owner、创建时间、租期上限、回收机制 |
| `cloud_phase` | `requested / provisioned / attested / registered / committed / drained / released / release_verified / abandoned` |
| `attestation` | GPU 型号与 uuid 断言原文、`nvidia-smi topo -m`、NUMA、镜像 digest、yeto/fork commit、Ray 节点 ID |
| `fork_ops` | `(op, cell_ids, expected_fork_epoch, result_fork_epoch, incomplete)` 列表 |
| `weight_admission` | `(cell_ids, weight_version, check_weights 摘要, admitted_at)` |
| `cost` | 见 §2.6 |
| `rollback_boundary` | 最近一次越过的不可逆点（§2.7） |

写入规则同 D4：manifest 原子写 + fsync 后才能标记可恢复；commit 使用单写者 durable CAS；JSONL 观测事件不能充当 WAL。journal 读写器尚未实现 `[缺口 G2，属于 3.2]`。

### 1.5 pause-budget / veto（未来岛间 planner 接口）

本轮单岛：pause 许可只来自本地 `pause_decision(profile, outer_phase, expected_pause_s, budget_mode, ...)` `[已实现 pause_audit.py:100]`：未认证 profile、非 `round-boundary-published` phase、budget 模式、超过 `quorum_timeout × margin` 均拒绝；decoupled 未认证 `[已实现 pause_audit.py:73-78]`。

未来接口（只定义形状，不实现）`[缺口 G3]`：

```
PauseAdvice {source, issued_at, expires_at, pause_budget_s | None, veto: bool, reason, target_resource_intent | None}
```

- 合并规则：取本地 `pause_decision` 与所有未过期 advice 的最严格值；任一 veto 生效即拒绝；advice 过期视为不存在，不自动续期。
- advice 只能收紧，不能放宽本地拒绝（例如不能让 decoupled 或 finalization 变为可暂停）。
- `target_resource_intent` 只是 planner 对池大小的意图；把它变成云操作仍需人工或 D2 自动策略的独立批准，读取它不触发云操作（§1.7）。
- `expected_pause_s` 对池操作只计快路径（§2.1）；慢路径不暂停岛，因此不计入 pause 预算。

### 1.6 “状态完整”检查

`IslandStatus` 对下列每种情形都能给出唯一的 `(health, tx_phase, pending_pool_op, last_*)` 组合，且每个组合都能从 journal 重建：

- 运行：RUNNING，无事务；
- 切换：RECONFIGURING + D4 每个阶段；池操作在快路径的两个步骤（pool 提交、drain）也以 D4 阶段表示；
- 慢路径：RUNNING + `pending_pool_op.cloud_phase`；
- 失败：CANCELLED / REBUILD_OLD / RECOVERY_REQUIRED（事务级），`fork_incomplete`（fork 半失败），`cloud_phase=abandoned`/释放未核实（云级），FAILED（岛级，§4）。

启动时组装逻辑：journal 最后一条 durable 记录决定 `pool_epoch/config_epoch/tx_phase`；fork 状态、provider 状态只用于对账（§1.3）。

### 1.7 只读契约：读取状态不触发云扩缩

- `inspect()` 只读 journal、`ReadinessSnapshot`、`get_membership_status`（`@lock_exempt`，`inference_controller.py:136-137`）与 provider 的只读查询（如 `sky status <name>`、`modal app list`），不调用任何 launch/down/start/stop。
- `IslandStatus` 中没有“建议扩池”这类会被消费即执行的字段；`resource_request` 与 `target_resource_intent` 都只是记录。
- 任何池操作只能经 `request(tx_kind=pool_*, request_id, expected_pool_epoch, expected_config_epoch, deadline)` 进入，与 D4 同一幂等规则。
- 实现时的测试：对 `inspect()` 注入一个会在 launch/down 被调用时报错的 provider mock，断言无调用（属于实现阶段，不在本轮）。

---

## 2. 7.2 运行中扩池/缩池顺序

### 2.1 慢路径与快路径

| | 慢路径（云供给） | 快路径（岛内切换） |
|---|---|---|
| 内容 | 申请实例、镜像拉取、驱动/拓扑认证、加入 Ray；释放与核实 | pool epoch 提交、cell 启停、权重发布与准入、drain、trainer 重建 |
| 典型耗时 | 分钟到小时，可能失败或无货 | 由已认证边的实测分布给出 |
| 是否暂停训练 | 否（岛在旧池上继续训练，health=RUNNING+POOL_PENDING） | 是（D4 WAIT_SAFE 后的 quiescent cut，受 pause 预算约束） |
| 控制权 | yeto launcher 现有云接入（SkyPilot / Modal / Nebius） | yeto controller + executor（D4） |
| 取消 | 随时取消，释放本次申请的资源 | 按 D4 release 边界 |

规则：慢路径的任何等待都不能计入 pause；快路径只能引用已提交 pool epoch 中的 GPU。

### 2.2 扩池顺序

前置：health=RUNNING、无事务、非 FINALIZING；目标 config 与目标 pool 形状已在 1.6 schema 中声明，且目标边已认证（D2）。

| 步 | 路径 | 动作 | 完成条件 | 依赖接口 |
|---|---|---|---|---|
| S1 云申请 | 慢 | 写 journal `cloud_phase=requested`（含 reservation 意图与硬超时），经现有 launcher 生命周期申请节点；设置云端超时/autostop 与独立 watchdog（BRIEF） | 实例 ID 写入 journal | launcher 云生命周期（sky `--down`/autostop、Modal `timeout=`，gpu-plan §8.1）；运行中追加节点的入口 `[缺口 G4]` |
| S2 云就绪 | 慢 | 等待实例 running、容器按指定 digest 启动 | `provisioned` | 同上 |
| S3 资源认证 | 慢 | 在新节点上断言 GPU 型号与 uuid、topo、NUMA、镜像 digest、yeto/fork commit，与目标 pool 声明一致；不一致即放弃并释放 | `attested`，证据写 journal | `validate_pool` / `placement_rejection` `[已实现 capabilities.py:282,240]`（纯校验部分）；认证采集脚本 `[缺口 G5]` |
| S4 Ray 注册 | 慢 | 新节点加入本岛 Ray 集群，预留 placement bundle；新 cell 在 fork 中“预声明” | `registered`；新 GPU 在 Ray 中可见但未被任何 config 引用 | fork 只接受启动时预声明的 cell（`list_declared_cell_ids`，`miles/utils/workers/worker_provider/ray.py:35`；`inference_controller.py:285-289`）→ 运行中追加 bundle/cell 声明 `[缺口 G6]` |
| S5 等安全点 | 快 | 发起 D4 事务 `pool_grow`：VALIDATING（expected pool/config epoch、pause_decision）→ WAIT_SAFE | 在安全点 | `pause_decision` `[已实现]`；`quiescent_cut_blockers` `[已实现 execution_profile.py:447]`；controller `[缺口 G2]` |
| S6 pool 提交 | 快 | journal CAS `pool_epoch += 1`，GPU 集合加入新 uuid | `committed` | journal CAS `[缺口 G2]` |
| S7 配置使用新卡 | 快 | 同一事务或紧随其后的 reconfigure 事务：rollout 扩容走 `start_cells(expected_epoch)` → `wait_cells_tracked` → `update_weights(members, expected_epoch, admit_cordoned=True)` → `check_weights` 读回 → `admit_cells(expected_epoch, expected_weight_version)` → `set_cells_weight_version`（如需）→ `commit_weight_version(expected_epoch)`；trainer 扩容走 cut → `rebuild_training_models` → restore（4.x 认证后） | config COMMITTED | `[已实现 inference_controller.py:233,197,581,851,351,750,705]`、`[已实现 miles/ray/placement_group.py:499 update_weights, :544 commit_weight_version, :352 rebuild_training_models]`；yeto 端口 `add_engines`/`Publisher.publish(members)`/`Placement.reconfigure` 仍是预留注释 `[缺口 G7：3.4/3.4a/3.5，yeto/rl/engine/ports.py:116,150]` |
| S8 恢复 | 快 | RESUMING → RUNNING | D4 | D4 |

要点：

- S6 与 S7 分开记录。可以只提交 pool（新卡作 `standby`）而不改 config；此时新卡计入成本（§2.6）但不产生暂停以外的变化。S6 本身不需要 quiescent cut，但为了单事务规则和 epoch CAS 简单，首版把 S6 放在 S5 之后；若实测 S6 只有 journal 写，可在实现时改为不暂停，需重新评审。
- S7 的准入顺序固定为 `admit_cordoned → check_weights → admit_cells`（3.5a 方案 (a)）：新 cell 在读回校验前不接收请求；`admit_cells` 同时校验 epoch、incomplete 与每个 cell 的 weight version `[已实现 :351-385]`。
- `commit_weight_version` 在所有 serving engine 报告同一版本后才设 executor 版本，持锁期间不能有成员变化 `[已实现 placement_group.py:544; inference_controller.py:705]`。3.5a 记录的小竞态窗口由 3.5 处理，本设计不假设其已关闭。

### 2.3 缩池顺序

| 步 | 路径 | 动作 | 完成条件 | 依赖接口 |
|---|---|---|---|---|
| R1 选择 | — | 选定要摘除的 GPU；它们上的角色必须能迁出到剩余 GPU（目标 config 在剩余池内合法，D2） | plan 通过 | `placement_rejection` `[已实现]` |
| R2 等安全点 | 快 | D4 事务 `pool_shrink`：VALIDATING → WAIT_SAFE | 安全点 | 同 S5 |
| R3 排空 | 快 | rollout cell：`drain_cells(timeout)`（cordon + 等 in-flight=0，超时返回 False 不 abort）；tool-wait 由 yeto 计（3.3b）；trainer：quiescent cut + `save_cut` | `drained`；超时则 `uncordon_cells` 取消，池不变 | `[已实现 inference_controller.py:388 drain_cells, :337 uncordon_cells]`；`save_cut` `[缺口 G8：4.2]` |
| R4 迁出 | 快 | reconfigure 到只用剩余 GPU 的 config：rollout 走 `stop_cells(expected_epoch)`；trainer 走 `rebuild_training_models` 到剩余 bundle + restore；发布与 `commit_weight_version` 同 S7 | config COMMITTED，被摘除 GPU 不再被任何 config/cell/PG bundle 引用 | 同 S7；fork 释放 bundle 与 Ray 节点摘除 `[缺口 G6]` |
| R5 pool 提交 | 快 | 证明无依赖：无 cell、无 trainer rank、无 cut/staging 文件、无 object-store 对象、无未完成发布位于被摘除节点；然后 journal CAS `pool_epoch += 1` | `committed` | 依赖证明采集 `[缺口 G9]` |
| R6 恢复训练 | 快 | RESUMING → RUNNING；health=RUNNING+POOL_PENDING | D4 | D4 |
| R7 释放 | 慢 | Ray 移除节点；按 journal 记录的实例 ID 经现有云接入释放 | `released` | launcher 生命周期；按 ID 释放 `[缺口 G4]` |
| R8 核实 | 慢 | 云 API 按 ID 核实释放（sky status / modal app list / Nebius API），写 `release_verified`；无法核实记“未确认”并告警 | `release_verified` | gpu-plan §8.4 |

要点：先排空、再提交 pool、最后释放；释放在训练恢复之后进行，不占暂停时间。R5 之前任何失败都能回到旧池。

### 2.4 失败矩阵

| 失败点 | 状态 | 处理 | 越过的回退边界 |
|---|---|---|---|
| S1/S2 申请失败、无货、超时 | RUNNING+POOL_PENDING | 释放本次申请的一切（按 journal ID），`abandoned`；岛不受影响 | 无 |
| S3 认证不一致（型号/uuid/拓扑/digest） | 同上 | 不注册，释放，记录原因 | 无 |
| S4 Ray 注册失败或新节点在注册后掉线 | 同上 | 撤销声明，释放 | 无 |
| S5 等不到安全点 / pause_decision 拒绝 / veto | 事务 CANCELLED | 新卡保持 `registered` 未提交；可稍后重试，超过租期上限释放 | 无 |
| S6 CAS 失败（epoch 被改） | CANCELLED | 重新读 journal；不重试同一 request 体 | 无 |
| S6 提交应答丢失 | — | 读 journal 判定是否已提交（D4 COMMITTED 规则），禁止盲目回滚 | pool 提交 |
| S7 `start_cells` 失败（回滚成功） | fork epoch 不变 | 回到旧 config；新卡留作 standby 或走缩池 | pool 提交 |
| S7 `start_cells` 失败且回滚失败 | fork `incomplete=["stop",ids]`, reason `start_rollback_failed` | 只允许重试 stop；deadline 内成功则继续回旧 config，否则 RECOVERY_REQUIRED | pool 提交 |
| S7 `wait_cells_tracked` 超时（`CellsNotTrackedError`/`TimeoutError`） | fork epoch 已提交 | 在 deadline 内再等；否则 stop 这些 cell 回旧 config | pool 提交 |
| S7 发布失败 / `check_weights` 不符 / `admit_cells` 拒绝 | cell 仍 cordoned 待准入，未接流量 | stop 这些 cell；不 uncordon（fork 拒绝对待准入 cell uncordon，`:337-346`） | pool 提交 |
| S7 `commit_weight_version` 拒绝（epoch 变/版本不齐） | executor 版本不变 | 补发布或 `set_cells_weight_version` 后重试；deadline 超出转 RECOVERY_REQUIRED | pool 提交 |
| S7/R4 `TrainerRebuildError`（`placement_group.py:430`） | 旧 trainer 已 dispose，无 trainer 运行 | fork 不自动回滚：若 `cleanup_error` 非空，先停 `pool_ids` 上的 worker；用 `previous_view`（`view_restored=False` 时显式传入）按旧参数重建并从 cut 恢复 → REBUILD_OLD；失败 RECOVERY_REQUIRED | release（旧 trainer 已释放） |
| R3 drain 超时 | cell cordoned | `uncordon_cells` 取消，池与 config 不变 | 无 |
| R4 `stop_cells` 半失败 | `incomplete=["stop",ids]`, `stop_failed`；cell 已不在路由 | 只重试同一 stop；deadline 超出 RECOVERY_REQUIRED（3.3a/3.7） | config 切换中 |
| R5 依赖证明失败 | config 已提交但 pool 未提交 | 不提交 pool、不释放；保留节点为 standby，人工检查 | config 提交 |
| R7/R8 释放失败或无法核实 | pool 已提交 | 岛继续训练；重试释放；`release_verified` 未达成则告警并记“未确认”，计入成本 | pool 提交 |
| controller 崩溃（任一步） | 读 journal | 按 §1.3 对账；D4 各阶段恢复规则 | 视 journal |
| 岛进程崩溃 / run 停滞 | FAILED | 不属于岛内事务，由 launcher 处理（§4） | — |

### 2.5 worker 增减不等于 DiLoCo 成员变化

- 池与 cell/trainer rank 的增减只改变岛内 `config_epoch`、`fork_membership_epoch` 和 `pool_epoch`。外层 `learner_id`、fixed roster、fragment/attempt/version 与 syncer 所见成员不变（D7 第 9 条、D11 末句）。
- 本设计不调用 syncer 的任何成员接口，也不新增。外层只看到一次在 `round-boundary-published` phase 内的暂停，受 `pause_decision` 预算约束；strict 下其他岛等待该暂停（`stalls_peers=True`，`pause_audit.py:66-72`），这由预算约束，不是成员变化。
- 新增或删除 DiLoCo 岛属于 7.3（D12 占位），不在本文。

### 2.6 成本记录

每次池操作在 journal `cost` 字段与证据目录记录：

- 慢路径：申请到 `provisioned` 的等待、认证耗时、Ray 注册耗时；这段时间新卡已计费但未使用。
- 快路径：D4 各阶段耗时、首个真实 optimizer-ready 时刻、首步与后续 warmup 拖慢（D8）。
- 外部影响：strict 下其他岛在暂停期间的等待（从外层事件带读取）。
- GPU·h：allocated = 池内全部卡（含 standby 与 pending/registered 未提交的卡）× wall time；释放在核实前一律计费；缩池的节省从 `release_verified` 时刻起算（D9、gpu-plan §0 口径）。
- 失败成本：被放弃的申请、REBUILD_OLD、释放重试。

扩池是否值得只由 D9 的 `gain_lower_bound(H) > cost_upper_bound + safety_margin` 判断，其中 cost 必须包含上述慢路径计费；本轮不给出任何收益结论。

### 2.7 回退边界

| 边界 | 之前 | 之后 |
|---|---|---|
| B0 pool 提交（S6/R5） | 放弃即可，资源释放 | 回退需要一次反向池操作（新事务），不能直接撤销 epoch |
| B1 旧资源 release（D4 TRANSFERRING） | 恢复原 handles | 只能 REBUILD_OLD |
| B2 config COMMITTED | 取消可回旧 config | 不能回滚，只能新事务或 RECOVERY_REQUIRED |
| B3 云释放 R7 | 可重新纳入池（仍在 Ray 中时） | 再使用需要完整走 S1 |

pool epoch 与 config epoch 都只前进，不倒退（D8 末句“不可倒退”）。

---

## 3. 与外层、provider、容器重启的边界

- **不预设全局 barrier**：池操作只在本岛安全点暂停，靠 `pause_decision` 与未来 advice 约束对其他岛的影响；不引入跨岛同步点。若 X6 证明暂停影响合法成员或提交，禁用该边，另行设计协调协议（D10）。
- **不把容器重启当岛内切换**：launcher 的 fixed-roster relaunch（`yeto/launcher.py:3256 FIXED_ROSTER_MAX_RELAUNCHES`、`:3656 _enter_recovering`）是失败恢复，不写 journal 的 config/pool 事务，不推进 config/pool epoch，也不能用来“实现”扩缩。重启后岛按 §1.3 从 journal 对账；若 journal 显示事务在途，按 D4 恢复规则处理，而不是当作切换成功。
- **不重复 provider 能力**：云申请、镜像启动、autostop、释放都调用 yeto 现有云接入（launcher/SkyPilot/Modal/Nebius）与在途云 change 的生命周期接口；本设计只定义调用顺序、认证项和 journal 记录。Verda 与 Modal 目前不能承载 head（memory），Modal serverless 节点能否加入运行中的 Ray 集群未知，因此首个实现只考虑 SkyPilot/Nebius 节点池 `[缺口 G4]`。
- **syncer 交互**：池操作期间 syncer 继续运行（D10“暂停不会暂停 syncer”）；bridge 连接保留；不发假心跳。decoupled 与 learner-budget 模式下 `pause_decision` 已拒绝，池操作的快路径同样被拒；慢路径不受影响。

## 4. 与 stall 与退出码 4/6 的关系

launcher 的判定（`yeto/launcher.py`）：

- 退出码 6 `RUN_STALLED_EXIT`（`:3271`）：`--rl-stall-timeout`（默认 900s，`:3272`；`yeto/cli.py:240`）内所有岛都没有新事件且未全部 finalized（`_check_stall`，`:3567`）。
- 退出码 4 `ISLAND_FAILED_EXIT`（`:3279`）：fixed-roster 岛超过重启次数被放弃（`FixedRosterIslandAbandoned`，`:3275`）。

设计约束：

1. 快路径暂停期间，岛必须持续输出事务阶段事件（D9 事件携带 tx/phase），使 `progress_probe`（事件计数，`:4159`）前进；否则单岛运行下暂停超过 stall timeout 会被误判为停滞。多岛运行下其他岛可能因 strict 等待而同样无事件，因此仍需第 2 条。
2. 任何池操作的 `expected_pause_s + recovery_upper_bound_s` 必须小于 stall timeout（留出余量），否则 VALIDATING 拒绝。该检查需要 controller 知道 stall timeout `[缺口 G10：launcher 把 rl_stall_timeout 传给岛内]`。
3. 慢路径不暂停，不影响 stall 判定。
4. 退出码 4/6 表示 run 级失败，不是岛内事务的终态。事务内的失败只能到 CANCELLED / REBUILD_OLD / RECOVERY_REQUIRED；RECOVERY_REQUIRED 若持续无事件，最终会由 launcher 以 6 结束 run，这是预期的 fail-closed 行为，不应为避免退出码 6 而发“假进度”事件。
5. launcher 在 4/6 路径的 teardown 按记录 ID 释放全部资源；池操作在 journal 与 launcher 资源记录中登记的实例（S1 起）必须被这一 teardown 覆盖 `[缺口 G4]`，否则 run 失败后会留下扩池节点。

## 5. 依赖与前置

tasks 原文依赖：7.1 依赖 1.5、3.8；7.2 依赖 7.1。本设计引用的 3.2 journal、3.4/3.5 端口、4.2 cut、4.6a/4.7 trainer 重建都尚未完成；设计中凡依赖它们的步骤，在其完成前不能实现。

## 6. 接口缺口

| ID | 缺口 | 归属建议 |
|---|---|---|
| G1 | `IslandStatus` 数据类与 `inspect()` | yeto `yeto/rl/engine/`（INFRA，随 3.2） |
| G2 | durable journal（原子写/fsync/CAS）、controller `request/plan/cancel`，含 `tx_kind=pool_*` | 3.2 |
| G3 | `PauseAdvice` 输入与合并规则 | 后续（岛间 planner，7.3 之后） |
| G4 | launcher 运行中向已有岛追加/按 ID 释放节点，并纳入 run teardown 记录；Modal 节点能否加入运行中 Ray 集群未验证 | launcher 负责人；需用户批准在线云扩缩 |
| G5 | 新节点认证采集（GPU uuid/型号/topo/NUMA/digest/commit）并与 pool 声明比对 | yeto，复用 1.1 runtime manifest 采集 |
| G6 | fork 运行中追加/移除 placement bundle 与 cell 声明（现只接受启动时预声明 cell，`list_declared_cell_ids`；`rebuild_training_models` 只在已有 bundle 集内重绑） | miles fork（新 M 项，需另批） |
| G7 | yeto 端口 `RolloutPool.add_engines/remove_engines/drain`、`Publisher.publish(members)`、`Placement.reconfigure` | 3.4/3.4a/3.5 |
| G8 | `TrainerGroup.save_cut/restore_cut/rebuild` | 4.2/4.3 |
| G9 | 缩池前“无依赖”证明（cell/rank/cut 文件/object-store/发布位于被摘除节点） | yeto executor |
| G10 | launcher 把 `rl_stall_timeout` 传入岛内，供 VALIDATING 预算检查 | launcher 负责人 |
