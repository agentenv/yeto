# Implementation tasks

本轮仅更新规划，未实施以下任务。**前提：change `rl-engine-ports`（R0）已完成并切换到 ports 路径。** Y=yeto（IslandController/IslandDriver/bridge），M=引擎机制：`yeto/rl/engine/` 端口新增动词及其 MilesAdapter 实现，仅在确需时于 `michaellchung/miles` 或 `michaellchung/sglang` 的 `yeto/ports` 分支增加小提交；X=跨层集成。决策、护栏与 journal 一律在 yeto 侧。miles 的 cell 只在适配层内部，不使用 FT/indep-DP/healing/api_server。技术栈保持yeto+miles/Ray/Megatron/SGLang，不引入veRL。引擎版本与端口定义见 ../rl-engine-ports/design.md。

路线：A调查/资源/依赖 → E0固定训推分区基线 → E1–E3手动重配置 → D1半自动 → D2自动；C成本优化对已验证边按瓶颈推进，可与D1建议工具并行。E1 rollout与E2同形恢复研究可并行，E3 trainer DP/角色转移必须有E2证据。实验未通过则不启用对应profile，不宣称trainer弹性完成。固定分区可先串行，获得合法重叠证据后才启用overlap；需要新算法契约时单列后续设计，不在资源参数中偷改。

## 1. 阶段 A：源码、论文、资源与执行契约

- [ ] 1.1 [X；依赖rl-engine-ports完成] 在 rl-engine-ports 固定镜像中生成runtime manifest，核对miles/SGLang fork commit、import路径、Megatron/Torch/CUDA/NCCL/TMS/PEFT，并记录 `EngineCapabilities`；验收：manifest与 `MILES_NEXT_*`/`SGLANG_NEXT_*` 一致，缺接口组合拒绝认证。
- [ ] 1.2 [X；依赖1.1] 选定小模型LoRA/strict-avg兼容profile，在ports路径上执行现有串行固定配置；验收：完成一轮生成、更新、外层同步与权重确认，形成兼容baseline。不以最小代码差异限制后续架构。
- [ ] 1.3 [Y；依赖1.1] 形成通过已有云接入租用独立单岛GPU池的实验计划，记录GPU/NUMA/互联、主存、镜像、备用卡、租期及清理；验收：计划可映射现有launcher/harness和pool身份，不把本机占用当限制、不把在途provider当已支持。本任务交付计划，实际租用按实验执行范围进行。
- [ ] 1.4 [Y+M；依赖1.2] 定义ExecutionProfile和readiness，覆盖策略版本、group/reward、更新/发布依赖、在途batch/队列容量、允许重叠任务和反压；验收X9：依赖表区分serial-colocated/partitioned-serial/partitioned-overlap，严格profile禁止旧版本偷跑，外层decoupled不被误当岛内async。
- [ ] 1.5 [Y；依赖1.4] 逐profile审计permit、quorum/grace、budget/final ACK、连接generation及Fleet恢复超时；验收：给出可暂停阶段和预算，未知profile默认禁用重配置。
- [ ] 1.6 [Y+M；依赖1.3-1.5] 以PR #66 的capability认证格式为准定义配置/有向边schema（不另设schema），含pool epoch、备用资源、显存/CPU/磁盘峰值、执行契约、复杂并行维度和恢复方案；验收：非法GPU映射、未知fingerprint、dense-full DP>1、GBS/DP/microbatch/累积不匹配被纯校验拒绝。
- [ ] 1.7 [M+Y；依赖1.4] 采集带profile/epoch的执行与等待时间线、queued/active/tool-wait、ready组、消费速率、policy age及资源峰值；验收：区分工具等待与GPU饱和，串行和重叠时间不重复计费，关闭观测兼容旧路径。
- [ ] 1.8 [X；依赖1.4,1.7] 将DynaResize证据转成实验假设，区分角色分区、进程复用、权重同步组预热、host backing/传输buffer和延迟optimizer；验收：每项有论文页码、miles适配点和否定结果处理，不采用论文性能常量。

## 2. 阶段 E0：目标执行模式的固定训推分区基线

- [ ] 2.1 [Y+M；依赖1.6] 经 `Placement` 端口表达trainer/rollout/standby物理映射并在启动时显式给出，适配LoRA启动参数避免colocate规范化覆盖分区；验收：小模型固定分区正常启动，旧共置配置保持原行为。
- [ ] 2.2 [Y；依赖2.1,1.4] 在yeto `IslandDriver` 上新增 `partitioned-serial` 执行模式，在目标分区管理就绪任务、权重身份、有限缓冲与反压；验收：partitioned-serial完成固定算法步数，不因分卡改变sample IDs/optimizer时序。
- [ ] 2.3 [X；依赖2.2,1.7] 对同一算法契约允许的独立任务做重叠实验，并实现对应guard；验收X9：延迟发布不能触发旧版本生成，队列有界；若不存在合法训推重叠，记录partitioned-serial结论与独立算法后续项，不擅自开放one-step-off-policy。
- [ ] 2.4 [X；依赖2.2,1.3,1.7；overlap另依赖2.3] 在云实验池扫描少量固定配置，记录默认兼容配置、目标profile最佳固定和收益面；验收：同profile公平比较、全池/备用GPU-hours和原始trace齐全，可得“尚无净收益边”的结论。P62/P44仅候选，不预设合法或更快。

## 3. 阶段 E1：单岛手动 rollout 重配置

- [ ] 3.1 [Y；依赖2.2,1.5] 在 `IslandDriver` 中实现模式对应安全点与全岛quiescent cut：兼容模式轮次边界，重叠模式先停准入并排空；验收：梯度累积中不切换，ready未消费组与策略身份可恢复。
- [ ] 3.2 [Y；依赖3.1,1.6] 实现单事务controller、request ID/expected epoch、journal和plan/status/cancel入口；验收：重复请求幂等、并发/旧epoch拒绝，提交应答丢失可查询，不重复训练。
- [ ] 3.3 [Y+M；依赖3.2,1.7] 实现轨迹级admission fence与工具drain；验收X5：active请求为0但tool-wait>0时保留旧路由，超时取消切换而不重放外部副作用。
- [ ] 3.4 [M；依赖3.3,2.1] 新增端口动词 `RolloutPool.add_engines/remove_engines/drain` 与 `Placement.reconfigure(plan, epoch)` 并在MilesAdapter中基于InferenceController实现，调整active engines、router、health monitor与权重更新成员，先测T4R2S2↔T4R4S0；验收X2：trainer不动、未占用池外资源，不等待故意停用engine，备用卡计入成本。
- [ ] 3.5 [M+Y；依赖3.4] 新增 `Publisher.publish(policy, members)`；新engine隔离加载、payload/版本ACK后原子提交epoch和路由；验收：旧generation ACK、错误payload、迟到请求都不能污染新配置。
- [ ] 3.6 [Y；依赖3.5] 接入group/batch/update账本和既有completed-groups/retry进度；验收：重试/部分组/publish失败无重复消费、无静默丢样本。
- [ ] 3.7 [M+Y；依赖3.2-3.6] 增加绝对deadline/watchdog和release前取消、release后重建旧rollout、commit后恢复；验收：启动/通信/发布失败有界处理，learner/bridge/trainer身份与状态正确。
- [ ] 3.8 [X；依赖3.7,2.4] 验收手动双向rollout切换与两小岛strict暂停兼容；验收X6：样本/step/policy/roster不变，quorum超时/PULL重发正确，finalization拒绝切换。报告明确仅完成rollout能力。

## 4. 阶段 E2/E3：完整恢复、trainer DP 与训推角色转移

- [ ] 4.1 [M；依赖1.2,3.1；可与E1并行] 审计完整ReconfigurationCut所需master/moments/scheduler/RNG/data/ref状态及LoRA保存分支；验收：逐项来源明确，与默认no-save/load-optim/rng路径隔离，缺状态拒绝。
- [ ] 4.1b [Y；依赖4.1，E2开工时] 修改 `docs/MILES_RL.md` 的“不做 controller”一条为：允许岛内yeto侧重配置控制器，仍不做跨岛控制器与通用恢复框架；验收：文档与本change design D1一致，评审通过。
- [ ] 4.2 [M+Y；依赖4.1,3.6] 新增端口动词 `TrainerGroup.save_cut/restore_cut`，实现完整cut导出/加载、manifest/fsync和算法账本对账；验收：坏checksum、截断、step不一致拒绝，源释放前恢复依据完整。
- [ ] 4.3 [M；依赖4.2] 新增 `TrainerGroup.rebuild(plan)`；同形trainer子进程重建、fresh groups和完整restore；验收X3：训练2步后重建，对冻结下一batch比较RNG/计数/moments/参数更新，无额外reset。
- [ ] 4.4 [Y；依赖4.3] `IslandDriver` 替换端口背后的实现（无需rebind）、保留bridge状态并重发正确权重；验收：不重复initialize/after_local_train，外层进度不因重建重放。
- [ ] 4.5 [X；依赖4.4,3.8] 同形恢复故障矩阵；验收：rank失败、collective超时、迁移中断、controller crash提交不确定均有界恢复或RECOVERY_REQUIRED，不继续不确定的消费。
- [ ] 4.6 [M；依赖4.5,1.6] DP1↔2重分片spike，固定TP/PP/CP/EP、GBS及算法；验收X4：master/optimizer/RNG/样本映射和下一步数值比较给出go/no-go，不直接加入白名单。
- [ ] 4.7 [M+Y；依赖4.6通过,2.4] 实现经认证的trainer DP转换及池内角色转移，按资源规模验证P62↔P44或更小等价边；验收：实际GPU从trainer转给rollout及反向，复杂并行维度固定，双向成功/失败恢复、batch语义和epoch都正确。
- [ ] 4.8 [X；依赖4.7] 匹配数据预算、多seed的连续固定/同形恢复/变DP学习验证；验收：预声明数值/学习容差、heldout/reward与NaN/发散检查。未通过不开放trainer边；E1仍可独立交付但不得计为trainer完成。

## 5. 阶段 C：论文机制迁移与实测成本优化

本组先选瓶颈再选做机制，不要求将论文所有机制一并实现；未选机制记录理由，不伪记为已实现。

- [ ] 5.1 [X；依赖3.8；trainer边另依赖4.5或4.8] 分析等待安全点、drain/export/init/restore/publish、首step/后台恢复与资源峰值；验收：形成每个source→target的成本分布，确定首先优化的一个瓶颈。
- [ ] 5.2 [M；依赖5.1选择host路径] 验证bounded host staging/分块的完整状态载体和生命周期，预算分开backing/pinned/object-store/scratch；验收X7：源释放时manifest全覆盖、无唯一状态丢失、内存受限时持久路径可恢复。
- [ ] 5.3 [M；依赖5.1选择通信路径] 对比metadata cache与目标worker-ready后的权重组预热；验收X10：不复用旧generation default PG/device mesh、缓存不虚报ready，测实际关键路径收益。
- [ ] 5.4 [M；依赖5.1选择进程路径] 实验保留worker进程的runtime重建/角色切换，审计Megatron globals、通信、SGLang子进程与CUDA context；验收X10：多次切换无泄漏/旧状态，不通过保留进程重建baseline，不引入veRL。
- [ ] 5.5 [M+Y；依赖5.1选择延迟恢复路径,4.5] 定位backend最早optimizer相关读取，增加分级ready/fence；验收X10：人工拖慢/失败加载阻止依赖操作，epoch提交不等于事务SUCCESS，首步前后故障按cut/update账本恢复。
- [ ] 5.6 [X；依赖已选择的5.2-5.5] 对优化与hard rebuild基线做A/B，覆盖低内存、部分rank与后台失败；验收：阻塞和完整首step成本、峰值、额外备用GPU-hours均报告，持久恢复能力不被隐式削弱。
- [ ] 5.7 [Y；依赖5.6，或5.1测得基线无需优化] 发布按profile/fingerprint/转换方向的成本与恢复上界；验收：无收益优化不默认启用，未知成本不用于自动建议。

## 6. 阶段 D1/D2：半自动、自动与验收

- [ ] 6.1 [Y；依赖1.7,2.4,5.7] 实现shadow负载归因与收益预测，分别对应串行或已认证重叠时间线；验收X8：工具等待/长尾/发布阻塞可区分，不统一套max(R,T)，无净收益边保持当前配置。
- [ ] 6.2 [Y；依赖6.1] 增加半自动建议：source/target、expected_epoch、profile hash、收益/成本区间、有效期与拒绝原因；验收：建议不自行执行，人工触发进入与手动相同事务。
- [ ] 6.3 [Y+M；依赖6.2] 执行前重验建议有效期、epoch、profile、负载与pause guard；验收：批准过期或条件变化的建议明确拒绝，不静默续批或替换目标。
- [ ] 6.4 [Y；依赖6.3及至少一条实测净收益边] 实现默认关闭的auto模式、持续窗口、保守收益门槛、最短停留/cooldown/频率限制、失败停用；验收：振荡负载不抖动，trainer未认证边不可选，论文11步等数值不硬编码。
- [ ] 6.5 [Y+M；依赖6.4] 实现disabled/manual/recommend/auto切换及兼容旧路径；验收：关闭auto不打断事务恢复，手动/半自动无绕过安全校验通道。
- [ ] 6.6 [X；依赖6.5] 稳定/变化/长尾/工具等待四场景，对比兼容默认、同profile最佳固定、动态；验收：相同数据/更新预算，模式变化与resize收益分离，记录端到端、全池GPU-hours、有效吞吐、等待、首step与恢复开销，不预设提升比例。
- [ ] 6.7 [X；依赖6.6] 综合故障、学习行为与部署回退验收；验收：sample/step/policy/roster正确或fail closed，学习符合预设标准，更新capability matrix/fork bundle/手册；未通过收益门槛保持manual/recommend。

## 7. 阶段 F：云资源池与岛间扩展设计

- [ ] 7.1 [Y；依赖1.5,3.8] 设计IslandStatus/pool epoch及未来pause-budget/veto，包含资源需求、safe point、outer phase、预计暂停/恢复；验收：运行/切换/失败状态完整，读取状态不触发云扩缩。
- [ ] 7.2 [X；依赖7.1] 设计运行中扩池/缩池的云就绪→资源认证→pool提交/排空→释放顺序；验收：区分云供给慢路径与岛内快路径，worker增减不等于DiLoCo成员变化，记录成本与回退边界，不在本轮实现。
- [ ] 7.3 [X；依赖7.2] 形成跨岛分配、多云、DiLoCo成员变更的独立后续任务并对照现有云change；验收：不重复provider能力、不把容器重启当岛内切换、不预设每次全局barrier。
