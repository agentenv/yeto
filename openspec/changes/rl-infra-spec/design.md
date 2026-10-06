# Design: 基于 miles 的岛内训推弹性重配置

## Context

动机见 `proposal.md`。引擎现状、版本固定与端口定义见 `../rl-engine-ports/design.md`（Context、D1–D11）；本 change 以其完成为前提，全部工作在 `ports` 引擎路径上进行。以下除明确标为“源码已确认”的内容，均为**设计建议**，不是已有 API；白名单能否启用由**待实验验证**门槛决定。

这是跨 yeto learner、miles Ray worker、Megatron 状态和外层同步的改动，必须有设计文档。现状：默认 LoRA 共置但 trainer 可常驻，dense full 分离且 DP=1；R0 后训练循环由 yeto `IslandDriver` 以 serial-colocated 模式按 rollout/train/sync/publish 串行执行；默认 checkpoint 不保存完整 optimizer/RNG。不能承诺仅改卡数参数即可恢复。

## Goals / Non-Goals

**目标**：在 yeto + miles 技术栈中建立面向目标执行模式的训推分区、资源重配置与控制闭环。当前串行共置是兼容基线，不是最终架构限制；允许必要的 driver/worker/通信结构性重构，以正确性、端到端收益和可独立验收决定取舍，不以代码改动量最小为唯一准则。

首轮仍用单岛、单模型、固定 logical learner 成员和一次实验内固定的已分配 GPU 池。资源可由 yeto 租用云服务器，不受当前开发机空闲卡数限制；可在不同实验中租用不同规模或预留备用卡，但每次对比必须如实计入全池成本。岛内变 rank 与新增 DiLoCo 岛是不同操作。

小模型恢复验证保留 LoRA + Megatron + SGLang + strict-avg、TP=PP=CP=EP=1。它用于验证状态/样本正确性，不被包装成可天然并发的目标 profile。dense full、SAO、MoE、critic、indep-DP 等分别认证；现有 DP=1 限制在相应 profile 获得明确实现与实验依据前保持。

交付路线为：A 调查、云实验计划和执行依赖；E0 固定训推分区基线；E1–E3 手动重配置（E1 rollout、E2 同形恢复、E3 trainer DP/角色转移）；C 实测成本优化；D1 shadow 后半自动建议/人工触发；D2 默认关闭的自动控制；F 岛间扩展设计。E0/E1 与 E2 状态研究可并行；E3 必须先有 E2 证据，rollout 的 C/D 不必等待 E3。

**非目标**：引入 veRL 后端、未经认证任意并行维度变换、通过调度隐式改变样本陈旧度/optimizer 更新/策略版本规则、跨云高频迁移状态、自动新增删除 DiLoCo 岛、通用 DAG。首轮不实现在线云扩缩；允许通过现有云生命周期能力为实验预先取得足够资源。

## Decisions

### D0. 目标执行模式先于资源自动化

**现状**：R0 后 yeto `IslandDriver` 串行；yeto 的 decoupled 是外层协议，不证明岛内流水线；现有 LoRA 参数默认共置。**设计建议**：在 yeto `IslandDriver` 上新增显式执行模式，角色资源映射经 `Placement` 端口表达；serial-colocated 保持为兼容模式。

| 模式 | 用途 | 依赖约束 |
|---|---|---|
| `serial-colocated` | 原行为与性能参考 | 相同池按阶段使用；不预设减少 trainer 会增加 rollout 资源 |
| `partitioned-serial` | E0 分区/状态/映射基线 | trainer/rollout 卡组独立，算法有依赖时仍串行；分离本身不算性能成功 |
| `partitioned-overlap` | 目标中有条件的重叠执行 | 只有运行 profile 明确允许、且就绪的任务才重叠；启用前必须完成依赖与版本契约验证 |

`ExecutionProfile` 至少含 policy 对每条轨迹的绑定、batch/组就绪、允许的版本年龄、更新/发布/外层同步顺序、允许重叠的任务对、最大在途 batch/轨迹、缓冲容量和反压、quiescent cut 条件。执行模式及算法契约 hash 在一次运行中固定，不由自动控制临时改写。算法契约 hash 即 `rl-algorithm-capabilities` 的 `AlgorithmSpec` 规范化哈希（`algorithm_spec_sha256`），不另设算法身份；profile 的允许版本年龄 `max_policy_age` 必须不大于 `AlgorithmSpec.execution.max_policy_staleness`，引擎能力 `execution.max_policy_staleness` 由已认证执行模式可产生的最大年龄决定。年龄大于 0 只能来自另立 change 认证的算法契约；本 change 的所有执行模式（含 `partitioned-overlap`）策略年龄均为 0。

E0 默认交付 `partitioned-serial`，同时审计现有算法允许的重叠（例如独立 CPU 工作与不依赖它的 GPU 工作）。如果生成下一 batch 需要本次更新后的权重，明确保留 `update -> publish -> next rollout` 依赖，不启动旧版本 rollout。若只有引入 one-step-off-policy 等新契约才能获得流水线收益，则记录算法变更为独立后续设计，不擅自放宽本 change 的陈旧度约束；本阶段可以得出“该 profile 尚无可启用的并发路径”的有效结论。

目标 runtime 的接口允许重叠，但不承诺所有既有 profile 都支持重叠。算法层决定 ready，调度层把 ready 工作放到 trainer/rollout/备用资源上。E0 必须先测同一 profile 下固定分区的收益面；没有可重复收益的转换边时，只发布手动执行能力，自动模式保持禁用。

DynaResize 第2–3页的核心是 disaggregated async pipeline，第7页示例是 Train:Rollout 6:2→4:4。它不证明当前串行模式的收益，也不证明新的策略版本契约对现有训练等价。论文其余证据按页码在本文各处就地标注。

### D1. 三层职责，控制器随岛内 learner 存活

```
Yeto cloud provisioning / future fleet controller
                         |
                         v
Yeto learner (one per island)
  IslandController: profile + readiness + decision + guard + journal
  IslandDriver:     execution modes + safe points + transaction steps
  existing algorithm/bridge owns readiness and policy progress
                         |
                         v  engine ports only (rl-engine-ports)
Ports: RolloutPool.add/remove_engines/drain  TrainerGroup.save_cut/restore_cut/rebuild
       Placement.reconfigure(plan, epoch)    Publisher.publish(policy, members)
                         |
                         v
MilesAdapter -> michaellchung/miles (cells internal only, no FT)
                         |
                         v
Megatron / SGLang / Ray workers and distinct communication groups
```

- **yeto 岛内控制层**：新增 `yeto/rl/engine/controller.py`（`IslandController`），由 `IslandDriver` 构造并持有，与对应 bridge 协同；接收指标、校验候选、计算收益、请求事务、报告结果。`FleetController` 仍管理岛生命周期，不在远端轮询中操控 rank。
- **执行层**：yeto `IslandDriver` 持有循环，在安全点按事务步骤调用端口动词（quiesce → export cut → release → initialize/restore → verify → commit → resume）；机制实现位于 MilesAdapter，复用 miles 的 rollout、placement、权重发布组件。因为 driver 本身持有端口，重建 worker 只是替换端口背后的实现，不存在旧 handle 残留，无需 `rebind`，也不重新 initialize 外层协议。
- **岛间层**：只读状态 DTO 和 admission veto 接口；不新增 wire message、全局 barrier 或成员修改。若当前协议无法证明暂停安全，就拒绝该 profile 的切换。

替代：执行器放进 miles（原方案 `miles/ray/reconfiguration.py`）会让决策所需的 bridge/permit/finalization 状态回灌引擎、并需要 rebind；全部放进 miles 会耦合云和算法；全部通过 SSH 重启会丢失 bridge/轨迹状态；启用实验性 FT trainer 会引入不同训练聚合和 witness/retry 语义。三者均不作为默认路径。

### D1b. cell 只在适配层内部

upstream miles 的 `TrainerCell`/`ServerCell` 是其默认骨架，只能出现在 MilesAdapter 内部：trainer 强制单 cell，rollout engine 由 `InferenceController` 管理。本 change 不使用 FT、indep-DP、healing、`ft_utils/api_server` 与 `mini_ft_controller`，也不把 FT cell PATCH 当弹性 API；端口签名与 journal 中不出现 cell 概念。upstream 的 cell 状态机（PendingWeights→Serving）可作为 E1 中 engine 准入的实现基础，但准入判定以 `Publisher` 的成员/payload 清单为准。

### D1c. 能力声明与 #66 对齐

`describe_capabilities()` 即 rl-engine-ports 的 `EngineCapabilities`，配置/边/pool 的可运行性判定使用 PR #66（`yeto/rl/elastic_benchmark/capabilities.py`）的认证格式；本 change 不另设 capability schema，#66 的 harness 直接作为 E0–E3 的测量与门控工具。

### D2. 固定资源池、有限配置与有向切换边

首轮继续持有启动时为已分配池建立的 Ray placement group，避免切换时重新竞争。池可由 yeto 云租卡预先获得；现有 LoRA colocate 参数规范化需要改为显式模式分支，不能静默覆盖分区目标。每个配置显式选择 pool 中的 bundle/物理 GPU UUID，重建 rank/engine 时使用对应 slice。Ray fractional GPU 可用于现有共置，但实际资源由 executor 的独占阶段和显存预算保证。进程退出、CUDA context/显存释放确认后，才把冲突卡交给新角色；offload 本身不释放 Ray actor reservation。

**候选配置记录（设计数据，不是随意覆盖启动 args）**：

| 字段 | 内容 |
|---|---|
| identity | config_id、schema、hash、execution_profile/mode、模型/算法契约 hash、代码/image/backend 指纹、已通过实验报告 |
| placement | pool_id、node/GPU UUID/bundle、trainer rank map、rollout engine/rank map、idle GPU、共置/分离/阶段占用表 |
| trainer | backend、world、TP/PP/CP/EP/expert-TP/DP、optimizer 类型/分片方式、precision、microbatch、accumulation/packing 策略 |
| rollout | engine 数、每 engine GPU、TP/DP/EP、router/session affinity、KV 配额、最大请求数、LoRA slot 状态 |
| capacity | 各阶段每卡稳态与峰值显存、CPU RSS/pinned/object-store 预算、checkpoint 空间和传输 scratch、测量余量 |
| safety | 支持 safe_point 类型、最大可接受 pause、允许 sync profile/阶段 |
| transition | source→target 边、rollout-only/rebuild、checkpoint 格式/restore provider、状态清单、恢复配置、成本估计分位数 |

配置合法不代表任意两个配置可互切。边也须通过验证，记录方向成本，双向分别认证。

目标分区示例（仅候选，非已认证配置）：一个通过 yeto 获取的8卡池、TP=PP=CP=EP=1，`P62` trainer使用G0–G5，两个TP1 rollout engines使用G6–G7；`P44` trainer使用G0–G3，四个TP1 engines使用G4–G7。E3切换需要将G4/G5从训练角色转为推理角色，或反向迁回。GBS示例48、microbatch=1，DP6时累积8次，DP4时累积12次；实际还须验证backend分片和样本计划。这里不要求异步算法：分区模式可以先串行验证，只有ExecutionProfile允许的任务才并发。

E1先固定trainer，例如`T4R2S2 -> T4R4S0`（S为池内备用卡），验证rollout生命周期；备用卡也计入GPU-hours。E2保持卡数做完整恢复；E3再认证`P62 <-> P44`等真正转移角色的边。原4卡共置R4/R2实验仍可作为兼容回归，不再代表目标架构。恢复spike可先用DP1↔2，不预设DP6/4合法。

如需要新旧worker短时共存，实验启动前额外申请备用容量并纳入pool与成本；不假设同卡可同时容纳两份状态。缺少备用容量时仍可顺序重建。是否采用更大池由测量决定，不能把免费备用卡当作优化收益。

**校验顺序**：

1. 精确运行指纹和 profile 支持；关闭 feature 时保持原驱动路径。未知配置/未知转换 fail closed。
2. 物理 GPU 存在、同一池且无重复冲突；卡组满足 NVLink/PCIe/跨节点网络约束；TP/EP 卡组不跨未经验证的拓扑。不能把 PACK 排序当拓扑证明。
3. 保持 TP/PP/CP/EP 等固定；普通 dense CP=1 时 `world=TP*PP*DP`；MoE 不能把 EP 再简单相乘，用 backend 实际 group 布局验证。dense full 仍拒绝 DP≠1。engine GPU 数须能按该 engine 的真实并行布局分组。
4. 保持每 rollout 完整 groups×samples、optimizer steps 与 GBS。必须先校验 `groups*samples % optimizer_steps == 0`（现有整除推导不可替代该检查），再校验 GBS 对 DP/microbatch/accumulation 的关系。非打包等长示例：GBS=32、microbatch=1，DP=4 时 accum=8，DP=2 时 accum=16；step 和32条样本不变。动态 batching/序列打包须按实际 token/sample normalization、per-rank batch plan 验证，不强套公式。
5. 按切换路径验证显存峰值和 CPU 峰值：旧状态驻留 + 目标初始化 + staging + 通信 buffer + KV/CUDA graph；共享 storage 按真实分配计，检查 /dev/shm/object store、磁盘容量和恢复时间。任一预算未知则仅允许实验，不进生产白名单。
6. 恢复证据完整、目标/恢复配置都合法，pause 预算覆盖正常和失败恢复；不只验证成功路径。

### D3. 最小就绪状态，算法仍拥有依赖

不建立 DAG 引擎。控制器只读一份 `ReadinessSnapshot`：rollout_id、optimizer_step、batch_id、ready_group_ids/count、reward_pending、unfinished_trajectory_ids、active_request_count、tool_wait_count、policy snapshot/hash、fragment versions、sync phase、publication_complete、outstanding submissions、driver_safe_point、memory 状态。

`generate_rollout`/reward/group 校验与现有 bridge 决定 `ready_for_train`；调度只在就绪任务中分配资源，不能用队列阈值提前截断 GRPO 组或改样本利用。配置 epoch 与权重版本不同：一次纯资源切换增加 epoch，但不制造新策略版本或 optimizer step。

轨迹标识至少 `(run_id, learner_id, rollout_id, group_id, sample_id, attempt_id)`；每个 segment/token 请求绑定 policy token/hash 和 worker/config epoch。数据提交按 group_id 和既有 retry 语义去重；batch 消费清单持久记录 `prepared -> optimizer_applied -> outer_recorded`，与完整 checkpoint 的切点关联。内存里“曾经提交”不能作为 crash 后去重凭据。算法层按描述有意丢弃、数据游标已推进的样本/组记为显式终态 `filtered`，与“丢失”区分；可被后续轮复用的余量组记为非终态 `carried_over`，不属于终态，仍按未消费组进入 cut。

### D4. 安全点与显式协议

**兼容串行模式安全点**：当前 rollout 已完成既有数据选择/abort cleanup，所有训练 optimizer step 已返回，梯度累积边界清零；`after_local_train` 的本轮算法逻辑完成，既有权重发布及必要确认完成，下一轮 `generate` 尚未开始，无 eval/参数导出/外层 apply 同时进行。在 `IslandDriver` 的 KV onload 后/下一 generate 前设安全点，避免插入 optimizer 中间。正常目标为“本轮结束”；不把任意 CUDA idle 当安全点。

**目标分区模式安全点**：controller先停止新任务准入，driver按照当前profile排空已准入batch/轨迹/工具回调；记录所有ready但未消费的组和对应policy，trainer在optimizer-step边界停稳，处理已产生的外层提交与发布，再形成全岛一致cut。首版即使运行期允许重叠，也采用全岛quiescent cut做切换，不在处理中途梯度或未结束工具轨迹上转卡。不同模式安全点不同，但不改变算法就绪规则；无法排空则取消切换。

Strict hook 已可能取得下一 permit，故不能宣称该点无外层义务：记录 permit、当前进度和 timeout 风险；未持有或剩余预算未知就拒绝长暂停。decoupled 还须保护 snapshot、pending fragments 和广播队列；首版未认证前保持 disabled。

建议 API（所有均新增契约）：

- `describe_capabilities() -> supported_profiles/configs/transitions/fingerprint`
- `inspect() -> ReadinessSnapshot + IslandStatus`
- `plan(target_id, expected_epoch) -> validated_plan/rejection`，无副作用。
- `request(request_id, target_id, expected_epoch, deadline) -> transaction_id/status`，同 request/body 幂等；同 ID 不同 body 拒绝；岛内最多一个事务。
- `cancel(transaction_id)`，返回 cancelled / recovery_started / already_committed，而非模糊成功。
- executor 内部 `quiesce/export_cut/release/initialize/restore/verify/commit/resume`；操作带 tx、epoch、worker generation，状态可查询，不对外暴露任意 rank 操作。
- `algorithm_guard.can_pause/observe_safe_point`（重建由 driver 替换端口实现完成，不需要 rebind），只访问当前 bridge 的就绪状态，不创建新算法许可。

手动入口优先 learner 本地控制端点或受限命令文件，由 SSH/Sky 仅转发 request；不使用 miles 的 FT api_server/cell PATCH（见 D1b）。新增无认证公网端口不在设计内。

```
RUNNING -> VALIDATING -> WAIT_SAFE -> QUIESCING -> TRANSFERRING
                                                    |
                                                    v
RUNNING <- RESUMING <- COMMITTED <- VERIFYING <- INITIALIZING
                         |
any failure -> CANCELLED / REBUILD_OLD / RECOVERY_REQUIRED
```

事务 journal 位于岛的持久 state 挂载中；append 事件含 tx_id、from/to config/hash、expected/current epoch、phase、monotonic duration+wall time、worker generations、safe-point ID、状态 manifest/hash、恢复依据和 error。manifest 原子写与 fsync 后才能标记可恢复；commit 使用单写者 durable compare-and-swap epoch；JSONL 观测事件不独自充当 WAL。每次状态迁移写入完成证据，重启通过 journal 和 worker attestation 对账，不能默认从上一条日志继续 collective。

| 阶段 | 前置条件 → 完成条件 | 控制权 | 超时/取消、幂等与失败 |
|---|---|---|---|
| RUNNING | 已提交 epoch，worker 权重就绪 | 算法驱动 | request 不打断训练；重复 request 查已存结果 |
| VALIDATING | expected_epoch 匹配 → 合法边+资源/恢复预算通过 | yeto controller | `T_validate` 有界；取消/失败返回旧运行；无资源变更 |
| WAIT_SAFE | 合法计划 → step+外层 hook+发布完成且 guard 同意 | driver+bridge | `T_safe` 仅等待自然边界，不杀正在 train 的 rank；超时/取消清除请求，保持当前配置 |
| QUIESCING | 在边界 → admission fence、全部轨迹/工具回调/奖励已终态、所有 rank 同一 cut | executor，bridge仍存活 | `T_drain`；未触及破坏性步骤可解除 fence；超时取消切换，不以杀轨迹凑零 |
| TRANSFERRING | 已停稳 → snapshot/ledger cut 完整校验，旧资源按计划释放 | executor+backend restore provider | `T_export/release`；release 前失败可恢复原 handles；release 后取消转 REBUILD_OLD，不能直接 resume。重复 export 使用同 cut 不再 train |
| INITIALIZING | 有可用恢复 cut、旧冲突 context 已退 → 全 rank/engine 初始化并 restore | executor | `T_init/restore` 和子通信 timeout；部分失败 fence 全目标 generation、终止目标进程树，重建旧配置；每次重试新 generation/端口/store 名 |
| VERIFYING | 基线目标全体 restored → 状态计数/样本账本/并行布局/权重 ACK 一致；C认证的延迟路径按D8分级验证 | executor+algorithm guard | `T_verify`；失败按 release 边界处理；同一 cut 可重复只读核验，不做 optimizer step |
| COMMITTED | profile启动条件验证 → durable epoch CAS 完成；基线全ready，延迟路径仍受D8依赖fence约束 | controller授权，executor单写提交 | `T_commit`；超时先读 journal 判定是否已提交，禁止盲目回滚已提交 epoch；旧 worker 永久被 fence |
| RESUMING | committed -> 权重 ACK 保持有效、按D8就绪条件打开任务，基线全ready；延迟路径需首步依赖fence并完成最终验收 | driver | `T_resume`；失败保持 fence，恢复该已提交 epoch 或新事务，不能再消费旧 batch；取消返回 already_committed |
| REBUILD_OLD | 旧进程不可直接恢复，但 cut 可用 → 按旧配置重新构建并验收 | executor | `T_recovery`；新 generation，失败则 RECOVERY_REQUIRED，不自动再训练 |
| RECOVERY_REQUIRED | 状态/外层对账不确定或预算耗尽 | yeto运行管理 | 停止数据消费，记录可用 cut 和失败，按原有运行失败机制报告；不驱逐 learner、不伪造 ACK |

各 `T_*` 是配置值，A/B 从实测 P99 和 profile 协议 deadline 设置，整个事务用一个绝对 deadline，不能每次重试无限延长。停止 collective 不能只 cancel Python future：独立 watchdog 有界等待后 kill 全目标 generation 的 Ray actors/SGLang 子进程，确认 GPU 释放；无法确认时禁止再次占用。原进程发生通信故障也不能承诺原 communicator 可复活。

### D5. 恢复依据与状态完整性

先做可工作的落盘恢复作为正确性与故障基线，再优化 CPU 中转；不是规定最终快速路径必须每次全量落盘。普通 policy export、LoRA adapter checkpoint、TMS offload 只可作为组件，不叫“完整迁移 checkpoint”。

**状态驻留约束**：每一份必需状态在迁移任意时刻都要有可校验的载体，可能是存活源worker、完整host state store、目标已确认分片或持久cut；transport小buffer不是完整backing store。释放源前检查manifest覆盖、所有权与生存期，不能仅因最后一个chunk已发送便删除唯一恢复副本。CPU预算区分backing state、pinned buffers、object store和目标scratch。纯内存快路径必须声明可承受的故障域，并提供与外层cut一致的持久恢复/重放依据；无法恢复所要求的故障就不认证该路径。建议专用 `ReconfigurationCut`，不改变默认 checkpoint 参数，弹性路径显式保存/加载完整所需状态。

| 状态族 | 必需内容与验证 |
|---|---|
| trainer | 模型/adapter/可训练 expert、FP32 master、optimizer moments 与 step、参数组超参、LR scheduler/counters、loss scaler（若有）、backend 所需 precision/量化或 FP8 状态、reference/old-policy 副本或不可变重建引用 |
| stochastic/progress | Python/NumPy/Torch CPU/CUDA/Megatron RNG、rollout seed/数据采样器状态、dataset cursor与分片、packing plan、rollout id、已消费 sample/group/batch、optimizer applied 计数 |
| algorithm/outer | current base/snapshot、fragment versions、permit/attempt/已提交状态、optimizer-reset count、local horizon、budget/token 计数、policy hash、外层 session/contract identity；`algorithm_spec_sha256`、插件 PluginRef 与 `yeto_algo_plugins` 哈希、ref 模型身份（KL 启用时） |
| rollout | completed/aborted/retry 组及既有语义、session 路由版本、engine 权重身份；首版要求无活跃轨迹，KV 可丢弃重建，不能丢未结束工具状态 |
| runtime | source/backend fingerprint、配置、GPU映射、checkpoint shard schema、checksum、tx/cut ID、是否已 commit epoch |

RNG 跨 DP 不是简单复制每个旧 rank 状态到所有新 rank。同形恢复先要求对应 RNG 精确；不同 DP 需可解释的样本/种子映射与数值/学习行为验证。如果当前 backend 不能提供所需 RNG/optimizer 重分片就拒绝该边，不以“浮点允许差异”掩盖遗漏。变 DP 边的认证绑定算法描述哈希：loss 聚合/归一化（token 级、常数分母、DP 组内 advantage 白化）随 DP 变化的算法须单独认证。

**直接恢复边界**：仅旧 worker 都仍存活、未改写权重/通信成员、没有破坏性 release 前允许 unquiesce。若已 offload 但进程/同形状态仍健康，允许既有 onload；失败升级为重建。**release 后**依靠已 fsync 的完整 cut、算法 journal、数据 ledger、不可变 base model 和 fingerprint；旧配置也必须从 cut 重建，不能称无成本回滚。CPU RAM 不能容纳双份：顺序导出分片到磁盘、验证完整 manifest、销毁旧进程、逐块恢复；磁盘/时间也不足就拒绝切换。

E1 不销毁 trainer/learner/bridge：恢复旧 rollout 数量并重发同一 policy 是首选失败路径。E2/E3 保持 learner/bridge 原进程，经 `TrainerGroup.rebuild` 重建 trainer 子进程，driver 替换端口实现后不重新执行 `initialize` 或 `after_local_train`。首版破坏性 trainer 切换要求 cut 对应的外层提交已确定，且重建期间不会产生新的对外提交。controller 本身 crash 后若 accepted PUSH/BCAST 与本地 cut 无法唯一对账，必须停止并从既有一致的全局恢复方案恢复整个受影响运行；不声称任意进度本地回滚兼容外层。该限制须在能力声明和操作结果明确呈现。

### D6. 多轮工具轨迹与新 worker 准入

`active_engine_requests=0` 不蕴含 `unfinished_trajectories=0`。等工具的 session 还可能发下一次推理或具有外部副作用。

首版使用**轮次边界 drain**：收到 request 后仅标记待切换，允许当前 rollout 按既有算法完成；gate 阻止下一批新轨迹，不阻止当前轨迹完成所需请求。沿用当前轮次本来就有的 abort/retry/cleanup 规则，不为切换新增 abort。等待工具轨迹仍绑定旧 policy/worker routing，直到终态；drain 超时取消重配置，恢复 admission。外部工具副作用不能未经幂等契约重放。

后续若要迁移等待轨迹，需独立实验确认 session 可导出、token/history/KV 重建、工具 operation ID 幂等和正确版本路由；不满足就继续 drain。仅普通引擎 retract 不承担此功能。

目标 engines 在 quarantine 集合初始化，router 默认不路由；所有 engine 按确切目标 policy 完成 payload 加载、checksum/版本确认，所有trainer ranks达到该profile一致的启动条件，才commit active集；C的延迟optimizer路径仍受D8就绪fence约束。miles 的 `update_weight_version` 是元数据操作，不足以证明实际权重内容已加载；必须将 ACK 与本次实际发布成员和 payload 绑定。重试同一发布 ID 幂等，晚到旧 engine ACK 或旧 epoch 请求拒绝。

### D7. 正确性约束与验证方式

1. 每轨迹/segment 的 policy token/hash 可追溯；配置 epoch 不替代策略版本，组内混版本必须失败。
2. GRPO 完整组、reward 和现有过滤条件成立才 ready；重配置不得部分消费组。
3. optimizer step 由算法触发，不能因重配置多 step、少 step、重置 moments 或改变 strict/decoupled 原有 reset 规则。
4. GBS、有效 loss normalization、microbatch/accumulation、实际 sample IDs 在变 DP 前后保持语义；不能使用 floor division 静默丢样本。
5. 完整状态按 D5 清单恢复；重建不得通过 import policy 代替 optimizer restore。
6. 第一版不迁移处理中途梯度；若 safe point 检查发现未完成 backward/optimizer 或 grad accum，拒绝。
7. 所有目标 engines 同一 policy payload ACK 后才准入；空闲新 worker 不代表权重正确。
8. `request_id` 只保障控制幂等，另外需要 sample/batch/update ledger 保证训练恰好一次；发生提交不确定不能自动重放。
9. 外层 learner ID/roster、fragment/attempt/version、budget/final ACK 与原协议一致；不以本地成功替代外层确认。
10. 同形 roundtrip 可做字节/checksum 校验与下一步结果对照；异形对相同冻结 batch 检查 logits、loss、grad norm、参数增量、optimizer moments、scheduler/计数。容差由固定配置重复运行噪声和精度定义，实验前写入报告，不事后调宽。后续以匹配数据预算/多 seed reward或heldout 曲线验证，无异常 NaN、发散或系统性退化。

### D8. 通信与成本优化

区分 Ray placement group（资源预留）、Megatron 默认 distributed group、模型并行/DP 子组或 device mesh、indep-DP torchft group、trainer→SGLang 权重更新组。M:actor.sleep/wake_up 的 destroy/reload 是同进程路径，不意味着新进程继承旧 communicator。

每次新 worker generation 建立新 rendezvous、rank mapping 和通信 group；可缓存验证过的 metadata，不能标为通信已就绪。新组建立与 restore 的顺序依 backend；部分 rank 失败整代清理，不能把幸存新 rank 混入旧配置。

阶段 C 的候选按实测瓶颈逐项选择，不要求全部实现：CPU snapshot（需 RAM 预算且有持久恢复 fallback）、分块去重传输（需完整 manifest）、目标初始化与无冲突状态传输重叠（需资源/通信证明）、预建独立 communicator（目标进程真实存在且无冲突）、按 backend 延迟 optimizer load（任何首次依赖该状态的操作前必须ready）。每项保留与落盘基线的 A/B 性能和故障测试；不假设 TMS CPU mirror、torchft 内存 checkpoint 与可迁移 CPU snapshot 相同。

**进程复用候选**：论文第6页§4还描述existing processes中角色切换。保留placement group只保证资源预留；若re-init是瓶颈，再研究保留worker进程并重建内部runtime，分别核查Megatron globals/default PG/device mesh、SGLang子进程与CUDA context。未通过则维持新generation重建，不因论文使用veRL而引入其runtime。权重同步组metadata cache/预热严格按第5页§3.2限定，目标worker存在且上下文可用后才创建实际communicator。

**分级就绪与延迟恢复**：E基线要求全状态ready才提交/运行。C可认证`model_ready`、`forward_backward_ready`和`optimizer_ready`等backend实际依赖状态；不得假设仅optimizer.step读取moments/master。完整cut与恢复依据在源释放前仍须成立，延迟的是目标materialization。配置epoch只有在路由/映射、权重ACK及该profile的启动条件成立后才提交；未就绪的操作被fence。若提交后开放了不依赖optimizer的计算，事务保持RESUMING，直到optimizer完整校验且首步验证完成才记录SUCCESS，不得提前宣称切换成功。延迟失败时停止后续消费，若尚无update则从cut丢弃未提交的局部计算后恢复；若已发生update/外层提交必须先对账，不能盲目回退。epoch提交后恢复遵守原有不可倒退规则。

切换成本同时报告admission阻塞、后台工作、首个真实optimizer-ready时刻、首步及后续warmup拖慢、恢复时间；只把传输移到后台不算成本消失。H窗口收益使用完整轨迹积分，避免把恢复入口提早开放的时间误记为净收益。


### D9. 观测、baseline 与自动选择

现有 `_append_rl_event`、rollout metrics、阶段 timers 为采集入口。事件共同携带 run/island、配置 epoch、policy、rollout/step、tx/phase；采样避免同步阻塞训练。

必须收集：阶段 wall/compute/wait（rollout、reward、工具、trainer、publish、outer sync、eval、checkpoint）；queued/active requests、未结束和 tool-wait 轨迹、长尾分位数；算法 ready 的完整组/有效 token 与 trainer 消费速率；policy age/混版本计数；每 GPU 显存峰值、SM busy、通信等待，CPU/pinned/object-store/disk；重配置分段耗时、失败数、恢复时间、拒绝原因和预算预测误差。

不能只看总 rollout 时间：

| 观测组合 | 第一版动作 |
|---|---|
| 队列持续积压且 engine busy，ready 数据形成慢 | 在已测有收益的白名单中考虑增加 rollout engine |
| 只剩少量长轨迹、队列低 | 默认保持，额外副本通常不能加速单条尾巴 |
| tool_wait 高、engine idle | 默认保持，不把工具延迟当 GPU 缺口 |
| publish/outer sync 占主导 | 不以更多 rollout GPU解决；先报告阻塞归因 |
| trainer 阶段占主导 | 仅在 trainer 合法边已认证时评估，不自动开放 DP |

D1先shadow再半自动：输出带expires_at的建议，包含source/target、expected_epoch、ExecutionProfile hash、收益区间、完整切换/恢复成本、拒绝原因。人工批准仅触发同一request入口；执行前重新检查epoch、负载、有效期、算法guard与容量，过期/状态变化拒绝，不自动续批。D2才允许自动触发，默认关闭。开关区分disabled/manual/recommend/auto，禁用自动不打断已进入恢复的事务。

自动策略启用前须有当前profile下至少一条认证且净收益可重复的转换边；E1成功、论文报告或GPU更忙都不是收益证据。自动策略 D2：有限配置，持续 K 个有效窗口失衡才评估；最短停留时间、cooldown、单位时间最大切换数、失败后自动关闭并保留手动。`gain_lower_bound(H) > cost_upper_bound(source,target) + safety_margin`，H 不超过剩余训练预算；cost 包含 drain、export、release、init、restore、verify、发布、warmup/冷 cache 与对其他岛可测等待的影响。预测不足、无对应 profile 数据、剩余步数不足、tool-heavy 或即将 finalization 则保持。

串行驱动以实际完整周期 `rollout + reward/cleanup未重叠部分 + train + outer sync + publish + offload/onload + amortized eval/checkpoint` 的时间线估算，不能重复累加已经包含在 rollout 的工具时间。目标模式若获准重叠，则用实际依赖关键路径、fill/drain、版本约束和反压模型；异步吞吐按稳定窗口有效更新/样本计，不直接统一使用max(R,T)。profile尚无合法并发时继续用串行模型，不能拿论文并发收益预测串行结果。

baseline 对比默认固定、测试范围内最佳固定、动态三组；相同模型/算法/数据与更新预算、硬件集合、后端版本和工具条件；目标固定分区与动态必须同一ExecutionProfile，兼容串行基线单列，不能把执行模式变化的收益归于resize。稳定负载、阶段性变化、长尾、工具等待四场景，记录 warmup 和重复运行分布。报告端到端时间、有效样本吞吐、GPU-hours、各类等待、切换税、恢复代价、学习曲线；固定池未释放实例时 allocated GPU-hours=整池卡数×wall time，减少 active engines 不能算省租卡。允许动态在稳定负载略逊最佳固定，但必须有预设回归预算、禁用/停留控制，不承诺提升比例。

#### 实现备注（S11，2026-10-06）
以下为 S11 实现后的现状，标注"源码已确认"的条目可在 integ-decl 代码中直接核对；"设计建议"为尚未实现的约束。
- 模式开关（源码已确认）：`IslandController.recommend_mode` ∈ disabled/manual/recommend/auto，默认 disabled；`set_recommend_mode` 写入 journal、replay 恢复；inbox `mode` 动词与 CLI `mode` 子命令；切换不 cancel 进行中事务。
- ElasticHook 接入点（源码已确认）：`miles_adapter/elastic_hook.py`，在 `IslandDriver.safe_point` 中、旧路径 `poll_commands` 之后调用；仅 ports + `--rl-elastic` + `--rl-observe-timeline` 且 mode ∈ {recommend, auto} 时工作，不传 hook 时与旧路径逐事件一致。
- auto 状态持久化（源码已确认）：mode 经 journal 持久化；AutoController 的 dwell/cooldown/切换历史/在途请求也写 journal（kind `auto_state`，elastic.py `auto_state()`/`restore_auto_state()`），learner 重启时由 elastic_hook `_restore` 恢复；无记录时从零（从宽）。
- 候选边（源码已确认）：候选 = 声明 ∩ 认证（`candidate_edges_from_attestation`），当前仅 rollout-only 边；trainer 边未认证不可选。
- 成本表（源码已确认）：`evidence/edge-costs.json`，列表项 `{profile_hash, source, target, cost_lower_s, cost_upper_s, recovery_upper_s, n, provenance}`，由 `edge_costs_from_table` 读取；文件缺失或无对应 profile → 空表 → 永远保持当前配置。
- 单节点 placement（源码已确认，8443a8fd）：岛分配单节点 M 卡且未提供 `--rl-island-gpus-per-node`（无拓扑）时，`ElasticPlacement` 把 `n0:<g>` 解析为逻辑 bundle g；修复前 E1 COMMITTED 后误判 "outside the pool" 进入 RECOVERY_REQUIRED。
- launcher 已知行为（源码已确认，未改）：job FAILED 时 launcher 进入恢复拆除，无视 `--keep` 拆集群；GPU 链因此对后续段重新 provision。
- 实测状态：S11 H100 单节点（n=3）无净收益边（见 `evidence/d1/`），auto 按本节规则不可启用。

### D10. DiLoCo 与岛间边界

暂停不会暂停 syncer。源码 fixed-roster round 超时等待/重发并不等于所有 finalization deadline 可暂停。严格同步会让其他岛等待；decoupled 可能积压广播、错过 attempt/变陈旧；budget gate/finalization 在首版一律 veto 切换。外层通信继续服务，但网络线程存活不能算模型状态已 apply。

首版保留 learner/bridge 连接；controller 只能在 profile 能给出允许暂停状态与预算时执行。无法读取精确 deadline 时使用离线认证的保守上界并拒绝无法证明的情况，不增加“假心跳”延长期限。两个小岛测试确认暂停跨 quorum timeout、PULL重发、最终收敛与退出行为；如果会影响合法成员/提交，禁用该边，后续单独设计协调协议。没有证据要求每次 global barrier，不加新 barrier。

`IslandStatus` 预留：pool/config/epoch、资源请求摘要、reconfiguration_allowed/reason/in_progress、last safe point、outer profile/phase/permit摘要、expected_pause 与 recovery_upper_bound、最后成功/失败/恢复状态。已有yeto云接入负责实验启动前资源申请与结束后释放；将来岛间planner提供pause budget/veto/目标资源意图，运行中改变池大小属于后续change；learner 增删与 DiLoCo 协议演进必须独立设计，不在本阶段承诺。

### D11. 云资源供给与岛内调度的两个时间尺度

实验可通过yeto在已支持的云上租独立GPU岛，记录region、GPU型号/互联、容器与source指纹、租期/清理方式；本设计更新不实际下单。原生云适配能力和在途多云change必须区分，不能假设所有provider都已可用。

岛内切换在已分配池中进行，通常比云申请/镜像启动/模型加载更短；云供给不保证即时或零成本。资源记录增加`pool_id/pool_epoch/capacity/standby/reservation_identity`。首轮pool_epoch不变；后续扩池需先云侧就绪、验收拓扑/镜像、注册Ray资源并提交新的pool epoch，再允许配置引用新卡；缩池先drain并证明确无状态/任务依赖，再释放云资源。worker变化不自动改变logical learner roster。

本机8×H200/NV18/约2TiB RAM仅是一次只读观察，不定义支持范围，不因当时显存占用推迟方案或声称GPU空闲。CPU中转要按租用实例的NUMA、RAM/带宽测量；H20论文数据不可移用。备用容量、初始化重叠所需额外卡和云启动等待都计入实测成本。

### D12. 岛间扩展（后续占位，范围延后，待用户审阅）

出处：用户 2026-09-29 无人值守轮指示：F 只做单岛 7.1/7.2 设计，7.3 跨岛分配、多云、DiLoCo 成员变更留待以后，只写后续占位。本段只占位，不构成 7.3 的交付。

- **延后原因**：首轮只有单岛、固定 DiLoCo 成员；跨岛分配和成员变更要改外层协议；在途云接入 change（`add-nebius-verda-modal-clouds`、fix-verda-provider）尚未合入；Verda 与 Modal 暂不能承载 head。
- **前置条件**：E1 3.8（含 X6 两小岛 strict 暂停）通过；7.1/7.2 设计评审通过；fix-verda-provider 合入；用户批准在线云扩缩范围。
- **后续 7.3 必须遵守的约束**（沿用原 7.3 验收要点）：
  1. 不重复实现 provider 已有能力，跨岛分配与多云只调用现有云接入 change 的生命周期接口；
  2. 不把容器或实例重启当作岛内切换，岛内切换仍走 D4 事务，跨岛或成员变化另走外层协议；
  3. 不预设每次都需要全局 barrier，确需协调时单独设计协议并给出证据。
- **与 7.1/7.2 设计的关系**：单岛 IslandStatus/pool epoch 与扩缩池顺序见 `f-design.md`；其中 `PauseAdvice`（§1.5，缺口 G3）是 7.3 实现的前置接口，在线云节点增减（缺口 G4/G6/G11）也须先解决，§3 已按上述三条约束限定单岛设计。

## Risks / Trade-offs

- [rl-engine-ports 未完成或其等价性未通过] → 本 change 全部阶段以 R0 完成为前提；R0 未通过前只做 A 阶段的调查、计划与观测设计。
- [端口动词需要 miles 新能力] → 优先用 upstream 已有机制（InferenceController、placement group）在 MilesAdapter 实现；确需改 miles 时在 `michaellchung/miles` 的 `yeto/ports` 分支加单独小提交（永不向 radixark/miles 或 sgl-project/sglang 提 PR），避免重现旧 fork 膨胀。
- [禁存 optimizer/RNG，adapter checkpoint 不完整] → 专用完整 cut；无法完成就只交付 E1，并明确 trainer 未完成。
- [兼容共置已满池或目标profile依赖严格，调整未必获益] → 先测 baseline；允许“没有值得自动切换的边”结论。
- [full/SAO 尚未迁移到 ports 路径，DP=1 硬限制] → 分开 profile，首版拒绝而非放松算法约束。
- [工具等待不受 engine flush 控制] → natural boundary drain，超时取消切换，保留旧路由。
- [rank 挂死/显存释放滞后] → watchdog+整代 kill+资源释放确认，禁止在未知 GPU 状态继续。
- [桥接状态与 checkpoint 不同切点] → durable cut/ledger，对外提交不确定时 fail closed，不伪称自动回滚。
- [已有 FT 功能看似可复用但改变聚合语义] → 仅作为 C 阶段技术参考，另行认证。

## Migration Plan

A完成运行指纹、论文适配与云实验计划、ExecutionProfile和观测；E0跑通固定训推分区。E1手动rollout与E2同形状态研究可并行，E3依赖E2认证DP及角色转移。C对已通过边按瓶颈优化；D1先shadow后半自动建议/人工触发，D2才自动。F只设计在线云扩缩和岛间接口。若并发需改变算法契约，单列后续工作而非隐式放宽；必要的工程重构仍在本change目标内。

每阶段可单独合并，yeto 与引擎适配通过 `EngineCapabilities` 握手（rl-engine-ports）。E2 开工时同步修改 `docs/MILES_RL.md` 的“不做 controller”一条为：允许岛内 yeto 侧重配置控制器，仍不做跨岛控制器与通用恢复框架。关闭开关走原路径；遇不支持 fingerprint 默认禁用。运行中禁用仅阻止新事务，已有事务按 journal 完成或恢复，不能直接解开 fence。部署回退需保持 journal/cut schema reader 或显式先完成恢复，不用旧二进制读取未知状态。

## Open Questions / 最小验证实验

这些未知项通过既定 gate 决定哪些可选边启用，不阻塞本设计范围，也不提前承诺它们成立。

| ID | 待实验验证 | 最小实验与否定结果处理 |
|---|---|---|
| X1 | ports 路径的端口动词在固定 miles/sglang 版本上真实可用 | 在 rl-engine-ports 固定镜像中对每个新增动词做 import/feature attestation 与冒烟；失败则该动词不进 capability，相应阶段保持禁用 |
| X2 | rollout active mask、router 与更新成员可一致变化 | 2个TP1 engine→1→2，trainer不动，固定policy，加入旧ACK/晚请求；失败先重建整个 rollout server，仍不动 trainer |
| X3 | 同形完整恢复 | 小LoRA模型训练2步，保存 cut、销毁 trainer、同DP恢复，比较下一冻结batch与连续运行；moments/RNG/主权重缺失即阻断E3 |
| X4 | 普通DP重分片 | 固定TP/PP/CP/EP，DP1→2→1，同GBS，导出完整状态与样本账本，比较下一步与学习短跑；失败保留E1/E2，trainer弹性标为未交付 |
| X5 | 工具轨迹 drain | 两条轨迹，一条等工具、engine请求为0；请求切换，证明不释放旧worker，超时可取消且副作用不重复 |
| X6 | 外层暂停与超时 | strict 两小岛，pause分别小于/跨过quorum timeout；注入PULL重发；检查相同 logical roster/update数；finalization请求必须拒绝。decoupled另测队列/budget，不继承strict结论 |
| X7 | 峰值与恢复预算 | 限CPU内存，禁双份，落盘分块恢复；注入partial-rank失败/通信超时；恢复预算超限必须停止而非继续训练 |
| X8 | 目标profile净收益与决策回本 | 固定分区收益面、四类负载trace replay、半自动建议批准，再小模型自动；批准过期/epoch变化必须拒绝，无可重复净收益维持手动/disabled |
| X9 | 目标执行模式合法重叠 | 在固定分区对严格policy依赖注入延迟，验证无旧版本偷跑；只对现有契约允许的任务对测并发。若需要新陈旧度规则，保留partitioned-serial并提交独立算法设计 |
| X10 | 论文优化在miles适用性 | 比较metadata cache/真实预热、进程重建/复用，注入延迟optimizer失败；验证首次使用fence、完整state backing和首步成本，失败保留baseline |

## 本轮交付与验证记录

本轮只有设计文档，所有 implementation tasks 默认未执行；源码调查已完成不等于 A 阶段 GPU baseline 已完成。验收执行方案见 tasks 与 spec scenarios。
