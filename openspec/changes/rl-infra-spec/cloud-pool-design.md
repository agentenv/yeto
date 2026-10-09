# 云资源池与云层调度建议设计（tasks 7.1–7.3、7.4）

（S17 夜间，2026-10-08，N6 起草。只写设计，不实现、不开卡、不做实际云扩缩。未经真机验证的结论一律标"未验证"。）

## 0. 范围、来源、与已有文档的关系

- **7.1 / 7.2 的原设计仍有效**：`f-design.md` §1（IslandStatus、三个 epoch、journal、pause-budget/veto、只读契约）与 §2（节点级扩池/缩池的两事务顺序、失败矩阵、成本、回退）写于 2026-09-29，本文件不重写，只做三件事：
  1. 按现在（S17）的代码核对 f-design §6 的接口缺口哪些已经补上（§1）；
  2. 补上 f-design 写时还没有的东西：岛间调度（rl-inter-island-scheduling）已实现的"整岛加入/退出"、可中断性四级 I0–I3、多云岛（§2–§4，即 7.1/7.2 的增补与 7.3）；
  3. 云层自动调度的设计（去耦合阶段 7：spot 分级、自动选云，第一版只给建议、不自动执行；S17 任务表 I9，本文件 §5，新增 tasks 7.4）。
- 参考：`rl-inter-island-scheduling/design.md`（Q4 弹性成员、Q5 跨岛再分配、Q6 多云、D-S 协议）、`yeto-framework-decoupling/design.md` D9/D10 与 spec `yeto-resource-dimensions`、`infra-drafts/YETO-DESIGN-PHILOSOPHY-S16.md` §可中断性分级、`infra-drafts/CLOUD-OPTIONS-S16.md`（Modal 2×8、AWS 只能 spot、新集群接入）、`rl-eval-difficulty-buckets/design.md` D11（评测岛，S17 第八批改为只用 Modal）。

### 0.1 为什么并入 rl-infra-spec，而不是新开 change（I7 与 I9 的归属判断）

结论：**I7 与 I9 合写在本文件，挂在 rl-infra-spec 第 7 节（7.1–7.3 + 新增 7.4）；代码落地仍按各自已有的 change 走，不新开 change。**

理由：
1. rl-infra-spec 第 7 节本来就叫"云资源池与岛间扩展设计"，7.3 的原文就是"形成跨岛分配、多云、成员变更的独立后续任务并对照现有云 change"——I9 的"选云/spot 分级"正是 7.3 要对照的那个云层。拆成两个 change 会出现两份"谁决定岛用什么卡"的设计。
2. 代码已有明确的落点，新 change 只会多一个要追踪的地方：云提供方接口与能力声明 = 去耦合 tasks 8.1–8.5；调度建议 = 去耦合 tasks 8.6；spot 回收接缝 = 去耦合 tasks 8.7；成员变更协议 = rl-inter-island-scheduling（已大部分实现）；节点级扩缩 = 本 change 的缺口 G4/G6/G11。本文件只定"它们之间怎么配合、顺序与规则"，各 change 的 tasks 引用本文件。
3. 用户已定"第一版只出建议、不自动执行"，第一版没有独立的执行器要写，不值得单开 change；等以后要把建议接到真实开卡（去耦合 D10 说的"另开任务并经用户确认"）时，再新开 change（暂名 `yeto-cloud-autoscale`），以本文件为设计输入。

## 1. f-design §6 接口缺口现状（S17 按代码与 tasks 核对）

| ID | 缺口 | 现状 | 依据 |
|---|---|---|---|
| G1 | IslandStatus 与 `inspect()` | **已补**，且已有调度字段（pool_epoch、cloud/region/price_per_hour、capacity、lease_remaining_s、syncer_epoch 等） | `yeto/rl/engine/controller.py:240`、`:1221`；岛间 tasks 0.3、0.18 |
| G2 | durable journal + controller request/plan/cancel，含 `pool_*` | **已补**（pool_* 只在 elastic 写） | rl-infra-spec 3.2 ✔；岛间 0.4、0.8b |
| G3 | PauseAdvice 与合并规则 | **已补**（只在 elastic 生效，读不到即拒绝暂停） | `yeto/rl/engine/pause_advice.py`；岛间 0.5、0.9 |
| G4 | launcher 运行中向已有岛追加 / 按 ID 释放节点 | **未补** | 无代码 |
| G5 | 新节点认证采集 | 未核（本轮未查到专门实现） | — |
| G6 | fork 运行中追加/移除 placement bundle 与 cell | **未补**（Miles fork 仍只认启动时预声明的 cell） | f-design §2.2 S4 |
| G7 | 端口 add_engines/remove_engines/drain、成员限定 publish | **已补** | rl-infra-spec 3.4、3.5 ✔ |
| G8 | save_cut/restore_cut/rebuild | **已补（CPU 证据口径）** | rl-infra-spec 4.2、4.3 ✔（用户 10-04 裁定） |
| G9 | 缩池前"无依赖"证明 | 未核 | — |
| G10 | launcher 把 `rl_stall_timeout` 传入岛内 | 未核 | — |
| G11 | fixed-roster 重启按 journal 的池形状重建 | **未补** | f-design §6 |
| G12 | 轨迹级 fence 与 tool-wait 排空 | **已补** | rl-infra-spec 3.3 ✔ |
| G13 | group/batch 账本与 cut 对账 | **已补（CPU 证据口径）** | rl-infra-spec 3.6 ✔ |

结论：**节点级**（在同一个岛里加减机器）的硬坎仍是 G4、G6、G11，三者都要改 launcher 或 Miles fork；**岛级**（整岛加入/退出）所需的协议与账本已经在 elastic 模式里实现并在真机两岛上跑过（岛间 1.0、1.1，1b 的租约过期与自动重入 0.22）。这决定了 §3 的第一版取舍。

## 2. 7.1 增补：IslandStatus 的云层字段与四个"版本号"

### 2.1 新增字段（只读，来源必须是已有记录，没有就填空，不估）

| 字段 | 含义 | 来源 |
|---|---|---|
| `billing` | `on_demand` / `spot` / `own`（自有集群） | launcher 开卡记录 |
| `interruptibility` | 本岛承担的任务等级 I0–I3（训练岛 I1、纯推理/评测岛 I2） | 启动配置（去耦合 8.4） |
| `preempt_notice_s` | 该云声明的回收提前通知秒数，未知为空 | 云提供方能力声明（去耦合 8.1、8.5） |
| `cut_save_s` | 最近一次切点保存实测耗时 | round_cut 事件 |
| `durable_store` | 本岛切点写到哪（URI），无则空 | 云提供方 `durable_store()`（去耦合 8.5） |
| `instance_ids` | 本岛所有实例/容器 ID（释放与核对用） | launcher / journal |
| `role` | `trainer_island` / `rollout_only` / `eval_only` | 启动配置 |

已有字段 `cloud/region/price_per_hour/capacity/round_wall_ema_s` 保持不变。

### 2.2 四个版本号谁管什么

| 版本号 | 谁写 | 何时加一 | 范围 |
|---|---|---|---|
| `config_epoch` | 岛内 controller | 每次岛内训推切换提交（D4） | 单岛 |
| `pool_epoch` | 岛内 controller（journal CAS） | 节点级扩池/缩池提交（f-design §2） | 单岛 |
| `membership_epoch` | elastic syncer | 有岛加入/退出（JOIN/LEAVE/租约过期） | 全体岛 |
| `syncer_epoch` | elastic syncer | syncer 每次重启 | 全体岛（防旧实例） |

规则：节点级扩缩只动 `pool_epoch`，**不动** `membership_epoch`（f-design §2.5"worker 增减不等于成员变化"仍成立）；整岛加入/退出只动 `membership_epoch`，加入的新岛自己从 `pool_epoch=0` 开始。两类变化互不引用对方的版本号，所以不需要全局屏障。

只读契约不变：读 IslandStatus 不触发任何云操作；本文件 §5 的调度建议也只读 IslandStatus 与云探针。

## 3. 7.2 增补：运行中扩池/缩池——第一版以"整岛"为单位

### 3.1 两种粒度

| | 岛级（整岛加入/退出） | 节点级（同一岛内加减机器，f-design §2） |
|---|---|---|
| 做法 | 扩：在某云新开一个岛 → elastic JOIN（影子成员，首轮权重 0）→ 下一外层步起正常计权；缩：岛 LEAVE（或租约过期被移出）→ 未提交增量丢弃并记账 → 释放该岛全部实例 | pool_grow / pool_shrink 两事务 + 岛内 reconfigure |
| 已有代码 | 有（elastic syncer、`elastic_client.py`、租约与自动重入、退出码 7 处理），真机两岛跑过 | 卡在 G4/G6/G11 |
| 训练暂停 | 其他岛不停；按 θ 算力比例步进 | 本岛 reconfigure 期间暂停 |
| 粒度代价 | 一个岛至少一台完整训练形状（FN 至少 8 卡）；新岛要重新拉镜像与加载底座（Modal/Nebius 15–25 分钟，CLOUD-OPTIONS B.2，小模型实测外推，FN 未测） | 可以只加一台推理机 |
| 适用 | 跨云、spot 回收后补岛、慢岛退出 | 同云同岛内的推理扩容（训推分离时只加推理机） |

**第一版取舍：运行中扩缩一律走岛级**；节点级保留 f-design §2 的设计，等 G4（launcher）、G6（Miles fork，需另批）、G11 补上后再启用，**在那之前 fixed-roster 运行中禁止节点级扩池**（沿用 f-design G11 的要求）。唯一例外是训推分离岛的"只加推理机"，它属于节点级，仍等 G4/G6。

### 3.2 岛级扩池顺序（新开一个岛加入）

| 步 | 动作 | 失败时 |
|---|---|---|
| P1 意图 | 建议器（§5）或人提出 `{action: grow, cloud, region, shape, billing, reason}`；**必须人确认**（第一版） | 不执行 |
| P2 台账预登记 | 按规则预登记本笔金额 | 不登记不开卡 |
| P3 云就绪 | 经现有 launcher 开新岛（与启动时开岛同一路径，不新写开卡代码）；云端超时/autostop + 独立 watchdog 按 ID 回收 | 超时即按 ID 释放，记失败原因，回 P1 换云 |
| P4 资源认证 | GPU 型号、镜像 digest、yeto/fork commit、底座权重哈希与运行配置一致；跨卡型/跨厂商受 §4.3 限制 | 不一致即释放 |
| P5 加入 | 岛发 JOIN，契约哈希（含 `island_scheduling_mode`、θ/γ）一致才收；首轮影子成员，权重 0 | 契约不符直接拒绝，释放 |
| P6 正常计权 | 下一外层步起按 `c_tokens²/c_steps` 计权；IslandStatus 记 `billing/interruptibility/instance_ids` | — |

### 3.3 岛级缩池顺序（让一个岛退出）

| 步 | 动作 |
|---|---|
| Q1 意图 | `{action: leave, island_id, reason}`（慢岛、spot 太贵、预算到顶）；人确认 |
| Q2 等边界 | 等本岛当前内层轮结束（安全点），已交出的增量照常参与合并；**不等**其他岛 |
| Q3 LEAVE | 发 LEAVE；syncer `membership_epoch+1`，未提交增量记 `dropped_uncommitted` |
| Q4 释放 | 按 instance_ids 释放；云 API 按 ID 核对，写 `release_verified`；核对不了记"未确认"并告警 |

被动退出（spot 回收、机器死）走同一条账：租约过期 → 移出成员 → 释放残余实例。

### 3.4 spot 回收按等级怎么处理

| 等级 | 典型任务 | 收到回收通知 / 机器没了 |
|---|---|---|
| I0 | syncer、head | 不允许放 spot（按需或自有）；若被动丢失，按 journal 与 syncer 检查点（`--resume`，`syncer_epoch+1`）恢复 |
| I1 | 训练岛 | 通知秒数 ≥ 切点保存耗时：`save_cut` → LEAVE → 释放；之后按 §3.2 在任一可用云补一个岛，`restore_cut` 后作为影子成员 JOIN。通知不够或未知：不做紧急切点，丢失距上一切点的工作（AWS 路线 CLOUD-OPTIONS B.2 G7 "先接受丢半轮"） |
| I2 | 推理/评测岛、纯 rollout 岛 | 进行中轨迹作废（`end_reason=preempted`），样本组按账本记未消费；评测按"版本+题号+第几次"续跑（eval D11.3） |
| I3 | 数据处理、镜像构建 | 整体重跑 |

接缝（只定义接口，实现归去耦合 8.7）：云提供方报告"回收通知" → 岛内 controller 收到 `preempt_notice(deadline)` → 按上表做；宽限期按 §3.6 的统一口径。通知秒数未知或为 0 的云按"机器没了"处理。

### 3.6 回收宽限期统一口径（10-09 按官方文档核对）

本节取代本文件与 `rl-resume-from-checkpoint/design.md` 里此前的不同说法。以后引用宽限期只引用本节。

| 云 | 有没有提前通知 | 通知后到被停的时间 | 怎么拿到通知 | 状态 |
|---|---|---|---|---|
| Modal | 没有提前通知。抢占发生时向容器发中断信号 | 退出处理函数有 30 秒宽限，超时被强杀。退出处理函数在抢占时也会被调用 | 容器内收中断信号，走 `@modal.exit` 退出处理 | 已核（文档 a、b） |
| AWS EC2 spot | 有，约 2 分钟。AWS 写明是"尽力而为" | 约 2 分钟 | 实例元数据 `spot/instance-action`（建议每 5 秒查一次），或 EventBridge 事件 | 已核（文档 c） |
| Nebius 抢占式 VM | 未知 | 未知 | 未知 | 未核 |
| Verda spot | 未知 | 未知 | 未知 | 未核 |

出处（10-09 用 WebFetch 读取原文）：
- a https://modal.com/docs/guide/preemption ：抢占时向容器发中断信号，触发退出处理函数；GPU 函数不支持 `nonpreemptible`；被抢占后在同一输入上重启。本页没写宽限秒数。
- b https://modal.com/docs/guide/lifecycle-functions ：原文"The exit handler is given a grace period of 30 seconds to finish"，以及"Exit handlers are also called when a container is preempted"。
- c https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html ：通知在停止或终止前两分钟发出；选休眠时没有两分钟提前量；通知是尽力而为。

对设计的影响：
- Modal 的 30 秒是"信号到强杀"的全部时间，不是提前通知。FN 全量切点 0.5–2 分钟写不完（`rl-resume-from-checkpoint/design.md` §2.2），所以 Modal 上回收时只写中断标记，靠定期切点续训。
- AWS 的 2 分钟是尽力而为，不能当保证。只有实测"保存耗时 + 余量 < 120 秒"后，才对 AWS 训练岛开"回收时保存"。
- 推理岛的在途轨迹体积远小于训练切点，可能放得进 30 秒。是否放得进要实测，由 `rl-spot-cost-saving` 处理。

### 3.5 成本与回退边界

- 岛级扩池的成本 = 新岛开机到 JOIN 的全部时间 × 单价（含首轮影子成员）；写进 `pool_grow` 事件，供 §5 的"准备成本"用实测值替换估计。
- 回退：elastic 下任何时候都可以让新岛 LEAVE，不影响其他岛；想完全回到旧行为，按岛间 design"模式选择与回退"把 `--rl-island-scheduling` 改回 legacy 重启（legacy 不支持运行中成员变化）。

## 4. 7.3：跨岛分配、多云成员变更（对照现有云 change）

### 4.1 跨岛分配：谁出主意、谁执行

- **出主意**：岛间协调器的 PauseAdvice `target_resource_intent`（已实现数据形状：`rollout_only` 慢岛降级，岛间 0.11）+ 本文件 §5 的云层建议器。动作集合扩为 `grow | leave | rollout_only | replace`（`replace` = 在另一朵云补一个同形状岛后让旧岛 leave，用于换便宜的云或回收后补岛）。
- **判断规则**（沿用岛间 Q5，补上云价）：只有当"变化后每轮节省的时间 × 剩余轮数 × 单价"大于"新岛准备成本（开机 + 拉镜像 + 加载底座 + 首轮影子）+ 旧岛退出损失（丢弃的未提交增量）+ 安全余量"才出建议；收益与成本都写区间，下界大于上界才出。没有实测值的项用 §5.3 的保守上界，不用乐观估计。
- **执行**：第一版全部由人确认后执行，执行路径就是 launcher 现有的开岛/停岛，加上 elastic JOIN/LEAVE；**不新增任何自动执行器**。
- 不预设全局屏障：成员变化只在外层步边界由 syncer 按 `membership_epoch` 生效，其他岛不停（岛间 Q4 的 P4 步进已实现）。

### 4.2 多云成员变更的前提

新岛在另一朵云加入，开机前必须满足（任一不满足，建议器就不推荐那朵云，或把补齐的成本计入准备成本）：
1. **能连上协调器**：协调器（head + syncer）在有公网入站的按需机器上（现为 Nebius 无卡 VM）；Modal、Verda 岛只出站连接（memory "head placement limits"）。
2. **底座权重在那朵云的存储里**：Modal Volume、Nebius 共享盘已有；AWS、Verda 没有（CLOUD-OPTIONS B.2 G4），要先复制或从 HF 下载（约 360 GB，20–60 分钟，未测）。与评测 D11 同理：**在别家开岛前，先把需要的权重复制到那家的存储并校验 sha256**。
3. **切点存储（只对 I1 训练岛）**：云提供方能给出 `durable_store()`，且回收通知秒数 ≥ 实测切点保存耗时；否则该云的训练岛只能用按需。
4. **镜像**：同一镜像 digest 在那朵云可拉（ghcr 公开镜像 yeto-miles-ports 满足）。
5. **数值口径**：不同云/卡型 bf16 不逐位一致，跨岛不比权重逐位哈希，只核"岛加载的 policy_hash == 协调器发布的 policy_hash"（岛间 Q6）。

### 4.3 跨卡型、跨厂商、跨框架的分期（沿用去耦合 D9a 与 verl 裁定）

| 期 | 允许的成员组合 | 状态 |
|---|---|---|
| 一 | 同厂商同卡型（如全 H200），可跨云 | 第一版 |
| 二 | 同厂商不同卡型（H100/H200 混合），算力权重 cap_i 用 step-time EMA | 后续 |
| 三 | 跨厂商（NVIDIA + 昇腾）、跨框架（Miles 岛 + verl 岛） | 后续；verl 第一版禁止与 Miles 岛混合合并但留接口 |

建议器在第一版只推荐与现有成员同卡型的岛。

### 4.4 与现有云 change 的分工（不重复造 provider 能力）

| 能力 | 归属 | 本设计怎么用 |
|---|---|---|
| 开卡、停卡、按 ID 核对 | launcher（SkyPilot 系云）+ `yeto/modal_runner.py`；以后收进 `yeto/cloud/` 云提供方（去耦合 8.1–8.2） | 只调用，不另写 |
| 容量与价格探针 | `yeto/shape/providers.py`（AwsProviders 放置分数/配额、NebiusSignals Capacity Advisor 与计费计算器、VerdaSignals、Modal 价格）；`tools/nebius_capacity_poll.py` | §5 建议器的输入 |
| 云目录与形状 | `yeto/shape/catalog.py`、`yeto/shape/plan.py`（`build_shape`、`effective_tflops`） | §5 打分 |
| 成员变更协议 | rl-inter-island-scheduling（elastic syncer） | §3 岛级扩缩 |
| Nebius/Verda/Modal 云接入 | add-nebius-verda-modal-clouds | 只读其能力，不改 |
| 自有集群 | rl-local-cluster-deploy（推后，去耦合 8.3 只留接口） | 能力声明里 `billing=own`，第一版不参与建议 |

"不把容器重启当岛内切换"：岛被回收后补岛是**新岛加入**（新 incarnation，影子成员），不是岛内 reconfigure；岛内切换仍只走 D4 事务。

## 5. 云层自动调度：spot 分级与选云（I9；第一版只给建议、不自动执行）

### 5.1 输入

| 输入 | 来源 |
|---|---|
| 任务清单与等级 | 运行配置：每个岛/服务的角色与 I0–I3（去耦合 8.4） |
| 云能力声明 | 云提供方：计费方式、回收提前通知秒数、能否当 head、网络档位、卡数上限、`durable_store()`（去耦合 8.1、8.5；实现前用一张静态表代替，见 §5.5） |
| 实时容量 | §4.4 的探针，全部只读 |
| 价格 | 探针 + `providers.modal_node_price_per_hour`；Nebius 10-08 起可抢占价为动态价，以计费计算器为准 |
| 准备成本 | 权重是否已在该云、镜像拉取、上次同云开机实测（`pool_grow` 事件）；没有实测用保守上界 |
| 预算 | 台账剩余额度（只读 gpu-spend.md 汇总数，人工维护） |

### 5.2 规则（按等级过滤，再按价格排序）

1. **过滤**：
   - I0：只要按需或自有，且能当 head（有公网入站）；Modal、Verda 不能当 head。
   - I1：按需可用；spot 只有在 `preempt_notice_s ≥ cut_save_s`（两者都要有值）且有 `durable_store` 时可用，否则写拒绝原因"回收通知未知/不足"或"无持久存储"。
   - I2：spot 优先；按需作兜底。
   - I3：最便宜者，不论计费方式。
   - 所有等级：卡型/厂商满足 §4.3 当期限制；`multi_node_rejection` 等现有拒绝规则照用。
2. **打分**：每个可行组合算"单位有效算力每小时成本" = 单价 / `effective_tflops`，加上"准备成本 / 预计在该云停留时长"，再加 spot 的"回收期望损失"（回收率 × 每次损失；回收率没有数据时**不估**，只在建议里写"回收率未知"并把该项列为风险，不参与排序）。
3. **输出建议**（不执行）：

```
CloudAdvice {
  advice_id, created_at, valid_until,          # 过期即作废，执行前必须重验（同 rl-infra-spec 6.3）
  task: {role, interruptibility, shape},
  ranked: [ {cloud, region, billing, instance_type, nodes,
             price_per_hour, prep_cost_upper, eff_cost_per_tflop_h,
             risks: [...], evidence: {probe, at}} ... ],
  rejected: [ {cloud, billing, reason} ... ],   # 例如 "I1 spot：回收通知未知"
  requires_human_confirmation: true
}
```
写成事件 `cloud_advice`（进 tape，dashboard 可展示），并提供只读命令行入口（建议挂在现有 `yeto plan` / `shape` 输出上，名字实现时定）。

4. **何时重新出建议**：开跑前一次；某岛被回收或租约过期；评测版本排队超过阈值（评测岛）；价格/容量探针变化超过阈值（只在 head 上低频轮询，≥5 分钟一次，避免打爆云 API）。

### 5.3 保守上界（没有实测时用）

| 项 | 上界 | 依据 |
|---|---|---|
| 新岛开机到就绪（已有权重的云） | 25 分钟 | CLOUD-OPTIONS B.2（小模型实测 15–17 分钟 + 加载） |
| 新岛开机到就绪（需拉 360 GB 权重的云） | 85 分钟 | 上行 + 下载 20–60 分钟（未测） |
| spot 拿不到卡 | 视为不可用 | AWS 放置分数全为 1；Verda 常无库存 |

### 5.4 第一版明确不做

- 不自动开卡、不自动停卡、不自动换云；建议只给人看。把建议接到真实开卡要另开 change 并经用户确认（去耦合 D10）。
- 不估 spot 回收率；没有数据就不进排序。
- 自有集群不参与（约 10-10 到货，型号与是否可抢占未知）。
- 评测岛第一版只用 Modal（用户 S17 第八批裁定，eval D11.6）；建议器可以对评测岛出"别家 spot 更便宜"的建议，但必须附上"需先把评测权重从 Modal Volume 复制到那家存储"的准备成本，且只是参考。

### 5.5 落地顺序（对应已有 tasks，不新建 change）

1. 去耦合 8.1：云提供方能力声明（含回收通知秒数、能否当 head、durable_store）；在此之前用一张静态能力表（每云一行，值标"已核/未核"）。
2. 去耦合 8.4：任务等级声明；syncer 声明 I0。
3. 去耦合 8.6：本节 §5.1–5.3 的建议器（纯函数 + 假探针单测：I0–I3 各一例、训练岛 spot 条件正反例、无回收率不排序、建议过期作废）。
4. 去耦合 8.7：回收通知接缝（§3.4）。
5. 以后（新 change）：建议 → 人确认 → 自动执行。

## 6. 未验证与风险

- 岛级扩池的"开机到 JOIN"在 FN 尺寸上没测过；§5.3 的上界来自小模型外推。
- AWS spot 回收通知 2 分钟与 Modal 抢占后 30 秒退出宽限已按官方文档核对（§3.6）。仓库里还没有代码读 AWS 的通知，也没有代码处理 Modal 的中断信号，两者都未在真机验证。Nebius、Verda 的通知时间未核。
- G5、G9、G10 本轮只按 tasks 文本核对，没有逐行查代码，标"未核"。
- 跨云补岛时 head 必须在 Nebius 按需 VM；Nebius 只有 eu-north1，单点。

## 7. 需要用户拍板

1. 第一版运行中扩缩只做岛级（整岛加入/退出），节点级等 G4/G6/G11 补齐再开（§3.1）——是否同意？
2. 归属：I7+I9 合写在 rl-infra-spec 第 7 节、代码落在去耦合阶段 7 与岛间调度，不新开 change（§0.1）——是否同意？
3. I1 训练岛上 spot 的两个条件里"回收通知秒数"在未知时一律视为不满足（即训练岛不用 spot）——是否同意？
4. spot 回收率无数据时不参与排序、只列风险（§5.2）——是否同意？

### 7.1 建议答案（10-09 规划子 agent 拟，建议，待用户拍板）

1. 同意，建议，待用户拍板。理由：整岛加入/退出已在 elastic 真机两岛验证（M1、V2），重入数据位置也已验证（#149）。节点级仍卡在 G4、G6、G11，其中 G6 要改 Miles fork。
2. 部分改变，建议，待用户拍板。I7（扩缩规则）仍留在本文件。I9 的 spot 部分已扩成完整方案，另开 change `rl-spot-cost-saving`（云能力表、回收通知接口、推理岛用 spot、换区域换云）。原因：spot 方案要新代码与上卡验证，超出"只出建议"。去耦合 8.1、8.5 仍承担能力声明接口，8.7 由新 change 替代。
3. 同意，建议，待用户拍板。并补一条：Modal 宽限已核为 30 秒（§3.6），小于 FN 切点保存时间，所以 Modal 训练岛即使知道秒数也不满足条件。AWS 2 分钟是尽力而为，实测保存时间够之前也不满足。训练岛第一版一律按需。
4. 同意，建议，待用户拍板。补充：`rl-spot-cost-saving` 会把每次回收事件记进 tape，攒够数据后再让回收率进排序。没数据时用保守值做"最坏情况"提示，但不参与排序。
