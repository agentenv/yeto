# DynaResize 证据 → 实验假设（task 1.8）

原文：Du, Yan, Bao, Wang,《DynaResize: Runtime GPU Reallocation for Disaggregated LLM Post-Training》，arXiv:2607.22614v2（2026-07-31），<https://arxiv.org/abs/2607.22614>。本地副本 `/home/michael/work/infra-drafts/papers/dynaresize-2607.22614.pdf`（sha256 `97487ca0fe25e868ec676ef8bd75ad3b6798f0646419cef757cb6583020b0650`，共 10 页；因许可证未确认，没有提交进仓库）。页码是 PDF 页码，与原文页眉一致。

## 使用规则

- 论文里的数值（吞吐 +66.5%、总时间 −33%、隐藏 27% 开销、255 s→187 s、约 11 步回本、冷却期 >10 步、6:2→4:4）**都不作为本 change 的门槛、默认值或预期**。它们来自 veRL、8×H20、Qwen3-8B，而且是 one-step-off-policy 异步流水线（见下一节），与本 change 的条件都不同。我们的门槛只来自事先登记的实验计划（gpu-plan.md 与各实验计划）。
- 每条假设都给出：论文依据（页码）、miles/yeto 适配点、验证它的 task、否定结果如何处理。否定结果如实记录，下游按 tasks 原文降级，不重跑到通过。

## 0. 前提差异（先于所有假设）

| 论文前提 | 页码 | 本 change | 结论 |
|---|---|---|---|
| 收益建立在训推解耦且重叠的异步流水线上；Fig. 1(a) 的基线为 one-step-off-policy，实验（§5）使用的 staleness 未明示 | p.2 §2，p.3 Fig. 1，p.6 §5 | 所有执行模式 policy age = 0（design D0、alignment A1/F6）；2.3 不开放 one-step-off-policy | 论文的收益来自在训推重叠的流水线里平衡两阶段耗时；本 change 的 `partitioned-serial` 没有训推重叠，一轮时间是 R+T 而不是 max(R,T)，在串行模式下移动 GPU 只改变 R 与 T 各自长短，收益形态不同，**不能外推论文结果**（design D9 禁止统一套 max(R,T)）。 |
| 基于 veRL，Ray 静态 actor group | p.6 §4 | yeto + miles/Ray/Megatron/SGLang，不引入 veRL | 机制可以借鉴，实现不能照搬。 |
| 在已有进程内迁移角色、保留 placement group | p.6 §4 | E3 默认走进程重建（fork-M6 `rebuild_training_models`），进程复用是 5.4 的可选项 | 见 H2。 |

## 1. 假设表

| # | 机制 | 论文依据 | miles/yeto 适配点 | 验证 task / 验收 | 否定结果处理 |
|---|---|---|---|---|---|
| H1 | **角色分区**：训推用不相交的 GPU 组，运行中在有限个预定义配置之间迁移，而不是在线搜索拓扑 | p.1 摘要；p.3 Fig. 1(e)；p.4 §3.1 与 Fig. 2（逻辑角色与物理 GPU 解耦的抽象层）；p.6 §3.4（只在预定义弹性配置间切换） | E0：`Placement` 端口 + fork-M1 显式 role→bundle 映射（2.1/2.1a，`miles/ray/placement_group.py`）；有限配置与有向边沿用 #66 schema（1.6）；E3 角色转移 4.7（fork-M6） | 2.4：同 profile 的固定配置扫描，得出默认配置、最佳固定配置与收益面；4.7：P62↔P44 角色转移 | 2.4 若得出“尚无净收益边”：按原文交付，D2 不开工（6.4），系统保持 manual/recommend；E1/E3 的正确性工作不受影响。 |
| H2 | **进程复用**：resize 时不销毁 Ray actor，在已有进程内切换角色 | p.6 §4（“retains the underlying placement groups … transitions execution roles within existing processes”）；p.8 Exp#2（基线的阻塞主要来自 actor 销毁、设备状态清空和 `dist.init_process_group`） | Megatron 全局状态、默认 PG、device mesh、SGLang 子进程与 CUDA context 都要审计（5.4）；默认路径是 fork-M6 进程重建 | 5.4 X10：多次切换无泄漏、无旧状态；5.6 与硬重建基线做 A/B | 审计发现旧状态无法清干净，或 A/B 没有收益：5.4 记为 no-go，保留进程重建作为唯一路径；5.7 不默认启用。不引入 veRL。 |
| H3 | **权重同步组预热**：按候选拓扑缓存通信元数据（rank 映射、成员），目标 worker 就绪后、拓扑发布前预建 inter-role 通信组；**明确不复用**默认 PG / device mesh 等 worker 绑定状态 | p.5 §3.2（“conservative by design … We do not claim full reuse of worker-bound distributed runtime states”）；p.4 Fig. 3（Comm Pre-warm 与 Actor/Rollout Init 并行） | LoRA 分区的发布走 NCCL broadcast（trainer rank0 + 全部 engine GPU，`protocol.py:73-89`、`protocols/broadcast.py:27`，见 upstream-mechanisms E0），broadcast 组就是需要预热的对象；成员限定发布依赖 fork-M4（3.5a） | 5.3 X10：不复用旧 generation 的默认 PG/device mesh，缓存不虚报 ready，测实际关键路径收益 | 预热没有缩短关键路径，或者无法证明缓存不虚报 ready：5.3 记为无收益或 no-go，每次切换都新建通信组；5.7 不默认启用。 |
| H4 | **host backing / 传输 buffer**：权重与 optimizer 状态经有界 host 内存分块流式迁移，不走存储 checkpoint，也不把整份状态堆在 host 内存 | p.5 §3.3（分块流水线、有界 buffer，避免 host OOM）；p.2 挑战❷（显存紧张时原地切换会放大瞬时内存） | cut 的导出/加载（4.2 `save_cut/restore_cut`，经 `run_plugin`）；预算分开 backing / pinned / object-store / scratch（1.6 `capacity` 字段）；持久路径必须保留（design D5） | 5.2 X7：源释放时 manifest 全覆盖、没有唯一状态丢失、内存受限时持久路径可恢复 | host 路径无法保证 manifest 全覆盖，或内存受限时会丢唯一状态：只用持久 cut 路径，5.2 no-go；不为了速度削弱持久恢复（5.6）。 |
| H5 | **延迟 optimizer 加载**：切换时只加载权重，optimizer 状态在新拓扑的第一个 step 需要时再加载 | p.6 §3.3 末段（“on-demand deferred transfer … deferred until the first step on the new topology”）；p.4 Fig. 3 (2)；p.7 Fig. 5（Deferred optimizer restore） | 要先定位 Megatron/miles 最早读取 optimizer 状态的位置，并加分级 ready/fence（5.5）；strict-avg 边界的 optimizer reset 规则（design D5）必须保持 | 5.5 X10：人为拖慢或让加载失败时，依赖操作被阻止；epoch 提交不等于事务 SUCCESS；首步前后的故障按 cut/update 账本恢复 | 找不到可靠的 fence 位置，或拖慢/失败注入会让依赖操作越过 fence：5.5 no-go，optimizer 与权重同步恢复。 |
| H6 | **滞回式重配置**：阈值、最短停留、冷却、连续信号，只对持续的单向失衡触发 | p.6 §3.4；p.8 Exp#3（回本步数、冷却期） | D1/D2 控制器（6.1–6.4）；数值从我们自己的 5.7 成本上界与 6.1 预测中得出 | 6.4：振荡负载不抖动，论文“11 步”等数值不硬编码（原文） | 6.6 四场景对比没有净收益：保持 manual/recommend（6.7）。 |

## 2. 对 E0 的直接影响

- **2.3**：论文的收益建立在训推重叠的异步流水线上（p.2–3）。本 change 在 age 0 下，下一轮生成必须等待本轮发布，`generate` 不能与 `train`/`outer_sync` 重叠（`execution_profile.overlap_violation`）。因此论文不构成开放 overlap 的依据；2.3 的结论只能来自 X9 实验，不能来自论文。
- **2.4**：扫描比较的是**同 profile**（partitioned-serial）下的固定配置。论文的 6:2→4:4 只是候选（tasks 2.4 已写明“P62/P44 仅候选，不预设合法或更快”）。

## 3. 没有采用的内容

- 论文的全部性能常量（见“使用规则”）。
- veRL 实现，以及 one-step-off-policy 执行模式（需要另立算法契约 change，alignment A6）。
- “mixed step”（resize 当步执行 2 次 rollout、1 次训练，p.7）：它依赖异步流水线，在 age 0 串行模式下没有对应物，不作为假设。
