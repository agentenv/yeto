# Design: rl-inter-island-scheduling

## Context

基线事实见 `infra-drafts/INTER-ISLAND-EXPLORE-S15.md` §A（文件:行号）；文献见 `infra-drafts/INTER-ISLAND-SURVEY.md` §3（方案 4、5）。上游设计：`rl-infra-spec/f-design.md` §1.1 IslandStatus、§1.5 PauseAdvice（G3）、§4 stall/退出码、§6 G1–G13；`design.md` D10（暂停不暂停 syncer）、D11（两个时间尺度）、D12（岛间扩展占位）。

术语：**岛** = 一个 DiLoCo learner（一套 rollout+trainer，岛内可多节点）；**协调器** = 外层权威（目前事实上是 syncer `GlobalState`，`state.rs:116`）；**外层版本** `outer_version` = syncer 已发布 base 的序号；**内层步** `inner_step` = 岛在该 base 上的本地优化步。

标记：**[推荐]** = 本 change 的默认选择；**[用户裁定 2026-10-07]** = 用户已拍板（命名对应 `infra-drafts/INTER-ISLAND-BRAINSTORM-S15.md` §2–§4 的 P1–P9、G-a/G-b）；**[待用户裁定]** = 仍未定；**[待真机校准]** = 推荐默认值，阶段 1 实测后再定。

## Goals / Non-Goals

Goals：方案 4 账本与样本判定、方案 5 弹性成员、PauseAdvice 实装；阶段 0 在 CPU 假岛上证明协议可行。Non-Goals 见 proposal（两岛内部通信/合并优化、节点级再分配、改 fork、阶段 0 改 Rust）。

## 逐题决策（SESSION15-HANDOFF §4.1 的 10 个问题）

### Q1 岛的抽象

| 备选 | 代价 |
|---|---|
| a. 身份 = syncer `learner_id`(u32) | 已有；但启动固定、无法表达"同一槽位换了一个新岛" |
| b. 身份 = 稳定字符串 `island_id` + 化身 `(pool_epoch, generation)`，`learner_id` 只是协调器分配的槽位 | 需在 HELLO/JOIN 中带字符串 id；协调器维护 id→槽位映射 |
| c. 身份 = 硬件指纹（GPU uuid 集合） | 换卡即换岛，扩缩池会误判为退岛 |

选择 **b [推荐]**：`island_id` 取 launcher 岛名（已存在于 tape/manifest），`incarnation=(pool_epoch, generation)` 区分重启；硬件指纹只进 attestation（G5），不作身份；云/region/价格作为 IslandStatus 描述字段。

IslandStatus 是否够表达"可被调度"：不够。本 change 增加 Optional 字段 `pool_epoch`、`round_wall_s`、`tok_per_s`、`staleness_outer`、`pause_budget_s`、`cloud`、`region`、`price_per_hour`（`inspect()` 填充；无来源即 None，不编造）。`round_wall_s` 首版来自岛内 timeline，将来可用 syncer 已有 per-learner step-time EMA（`server.rs:~530`，需协议导出，见 D-S5）。

### Q2 跨岛账本的权威与恢复

| 备选 | 代价 |
|---|---|
| a. syncer 为权威（扩展 `GlobalState` 记录 outer_version×island×policy_hash、成员 epoch） | 需改 Rust（协议+checkpoint）；但外层事实本来就在 syncer，单点一致 |
| b. head controller（Python）为权威，syncer 只做合并 | 两份外层事实（syncer 的 merge 计数 vs controller 账本）需对账，脑裂面大 |
| c. 每岛 journal 复制（Raft/对象存储 CAS） | 实现与验证成本最高，首版不划算 |

选择 **a [用户裁定 2026-10-07：P8a + P9]**：协调器（syncer）是 `outer_version`、成员集合 `membership_epoch`、`policy_hash` 的唯一权威；P9 = `syncer_epoch` fencing（每次 syncer 重启 +1，岛与账本拒收更小 epoch 的消息，`CrossIslandLedger.check_fence`）+ 成员心跳租约。每岛 journal 只记本岛的 `pool_*` 与观察到的 membership_epoch，不跨岛复制。Python `island_ledger.CrossIslandLedger` 是协议的参考实现（阶段 0 假岛的协调器直接用它），Rust 侧照此扩展（D-S1..S6）。
恢复：协调器从 checkpoint（已有 tmp+fsync 原子写，`state.rs:635-647`）恢复 ledger；岛崩溃重启后以 `pool_join(catch_up=True)` 重新加入，从协调器拉 base，本岛 journal 只用于重建本岛 pool_epoch（G11）。对账规则：岛 journal 的 membership_epoch 落后于协调器即视为"已被移出"，必须重新 JOIN，不能续用旧增量。

### Q3 跨岛样本池（方案 4）

- **存储 [用户裁定 2026-10-07：只做 P3]**：对象存储（S3 / Modal Volume / Nebius Object Storage），协调器只存索引（D-S4 SAMPLE_INDEX）。不做 P1（syncer 内存转发）与 P2（岛间直传）。
- **带宽估算**（量级，未实测）：数学题样本 ≈ 数 KB（token ids + logprob，4B×(prompt+resp)≈2k token≈8–16 KB）；TB2 多轮轨迹几十 KB–数百 KB。每轮 256 样本 × 100 KB ≈ 25 MB，相对外层增量（GB 级）可忽略。阶段 1 实测。
- **陈旧度阈值 [用户裁定 2026-10-07]**：跨岛样本最大陈旧度 **≤2 个外层步**（`StalenessPolicy.max_outer_lag=2`），inner 步差**不设上限**（`max_inner_lag=None`，字段保留）。判定：`Δo==0 且 policy_hash 与该版本发布哈希一致` → ACCEPT；`1≤Δo≤2` 且带 behavior_logprob → ACCEPT_IS；否则 REJECT。
- **IS 修正**：复用 `mismatch_correction.CORRECTION_MECHANISMS` 名称（`tis` 默认，`icepop`/`mis` 可选），不新增口径。behavior_logprob **必须**随样本传（没有它就只能 REJECT 跨版本样本）——这是 ACCEPT_IS 的硬前提。 **默认修正暂定 `tis`，待 tasks 0.10 离线比较（用已存 rollout 数据，M2PO vs 现有 TIS/IcePop 在 lag 1–4 的 IS 比分布）后再定 [用户裁定 2026-10-07]。**
- **GRPO 组规则 [用户裁定 2026-10-07：G-a]**：一个 prompt 组不跨岛、组内同一 policy_hash；组内 advantage 在生成端算好随样本传，消费端只做 IS 修正，不重算组内归一化。账本以组为单位判定（组内任一条 REJECT 则整组 REJECT），与已有组账本终态（ledger.py:243-317）一致。G-b（消费端跨岛混组重归一化）列为实验项，不在首版。
- **critic 家族 [用户裁定 2026-10-07]**：首版对 critic 家族（PPO/CompactionRL）跨岛或陈旧样本一律 REJECT（`StalenessPolicy.on_policy_only=True`）。

### Q4 弹性成员（方案 5）

- **加岛 catch-up [用户裁定 2026-10-07：P5 影子加入]**：新岛 JOIN（影子成员）→ 拉当前 base → 本轮贡献权重为 0（只同步不合并）→ 下一外层步起按 `w=c_tokens²/c_steps` 正常计权（已实现，假岛验证）。
- **退岛未提交增量 [用户裁定 2026-10-07]**：丢弃，账本记 `dropped_uncommitted`（含 c_tokens/c_steps/base 版本）便于审计；已排队的 carried_over 增量随退岛一并丢弃。
- **外层步进 [用户裁定 2026-10-07：P4]**：按算力加权 quorum——`Σ_{到齐} cap_i ≥ θ·Σ_{成员} cap_i`（且到齐数 ≥ q_min）即步进，或到软截止 `T_soft`（复用 quorum_timeout_s）即步进；软截止时到齐数 < q_min 则本轮空转不发布。**迟到岛的增量不丢弃**：基于旧 base（lag=当前版本−base）且 `lag ≤ max_carry_lag` 的增量作为 carried_over，按 `γ^lag` 折扣并入下一轮合并；超过 max_carry_lag 才拒收（`carry_lag_exceeded`）。cap_i 阶段 0 由 JOIN 声明（默认 1.0），阶段 1 取 syncer per-learner step-time EMA（server.rs:~530，D-S5）。默认值 **[待真机校准]**：θ=0.75、q_min=1、γ=0.5、max_carry_lag=2（与样本陈旧度上限一致）；理由：θ=0.75 允许两岛中较小一岛（算力 ≤25%）或三岛中最慢一岛缺席而不等待，γ=0.5 每多陈旧一步权重减半（Async Local-SGD 折扣思路）。
- **心跳租约与 PauseAdvice**：租约 = 岛→协调器心跳，`lease_s` 过期 → 协调器把岛移出成员（pool_leave，membership_epoch+1）。租约**不是** veto 本身；但协调器在"某岛即将过期 / 正在 strict 等待"时向其他岛发 veto advice（不允许它们此时暂停，避免全体被拖）。岛主动暂停前需申请 pause budget，预算 ≤ 剩余租约 − margin。
- **strict vs decoupled**：strict 下 quorum 变为"到齐 q 岛或超时"，等价于把 `max_base_lag==Some(0)` 放宽为成员级；decoupled 下没有轮次屏障，弹性只影响合并分母 M 与租约剔除。外层动量/η 随 M 变化需重新缩放（INTELLECT-1 做法）——**放到阶段 1 再定**，阶段 0 只记录 M。
- **单点故障 [用户裁定 2026-10-07：P9]**：syncer 单实例 + `syncer_epoch` fencing token：每次 syncer 重启 epoch+1，岛只接受不小于已见 epoch 的消息，防止旧实例脑裂。不做热备。

### Q5 跨岛资源再分配

决策输入：每岛 `round_wall_s`、生成/训练占比、`price_per_hour`、迁移成本（镜像拉取+模型加载实测）。规则草案：只有当"岛 B 加卡后每轮节省 × 剩余轮数 × 单价"大于"迁移成本 + 岛 A 损失"时才建议。输出只到 `PauseAdvice.target_resource_intent`（岛级：`{island_id, action: grow|shrink|leave, reason}`），**不执行**（G4/G6 是硬坎）。绕开方案：岛级粒度——整个岛退出（pool_leave）+ 在别处新开一个岛 JOIN，不做节点级迁移。本 change 阶段 0 只实现 intent 的数据形状与透传。

**[用户裁定 2026-10-07]**：P6 慢岛降级为纯 rollout 岛——协调器**只发 PauseAdvice（`target_resource_intent={island_id, action: "rollout_only", reason}`），由人确认**，不自动执行。P7 云价/时段再分配**只留接口**：IslandStatus 的 `cloud/region/price_per_hour` 字段 + `PauseAdvice.target_resource_intent` 占位，不做决策器。

### Q6 多云/异构

- 拓扑：协调器（head+syncer）放 Nebius 无卡 VM（有公网入站）；Modal/Verda 岛只出站连接协调器 [推荐，沿用 memory "head 用 Nebius 无卡 VM"]；样本走对象存储。
- 带宽实测：阶段 1 用 iperf3 风格的 TCP 多流探测 + 实际 PUSH 增量计时，结果进 IslandStatus 扩展的 `extra`（不在本 change 字段内）。
- 数值验收口径：不同云/卡型 bf16 不逐位一致，因此跨岛**不比对**权重逐位哈希；`policy_hash` 定义为协调器发布 base 的哈希（协调器侧计算，唯一），岛侧只回报"我加载的是哪个 policy_hash"，验收按"加载哈希 == 发布哈希"而非岛间互比。

### Q7 安全与失败语义

- 认证：syncer 现无 HMAC（A7）。新增消息（JOIN/LEAVE/HEARTBEAT/SAMPLE_INDEX）带 HMAC-SHA256(session_secret, payload)，secret 由 launcher 下发（同 session_contract_hash 渠道）[推荐]；旧 14 种消息首版不强制，避免改动面过大。**[待用户裁定]** 两个备选：(i) 只覆盖新增岛间消息 [推荐，改动面小、可先上]；(ii) 全量 16+ 种消息都上 HMAC（含 PUSH/PULL 大帧，需评估 GB 级帧的 HMAC CPU 开销）。
- 脑裂：`coordinator_epoch` fencing（Q4）。
- fail-closed 边界：成员变化不是失败——退岛走 pool_leave 而不是退出码 4；只有 `M_alive < q_min` 持续超过 `rl_stall_timeout` 时才由 launcher 以 6 结束（f-design §4 约束 4 保持：事务失败不升级为 4/6）。G10（stall_timeout 入岛）仍是前置。

### Q8 验证路径

| 阶段 | 环境 | 能验证 | 不能验证 |
|---|---|---|---|
| 0 | CPU 假岛（multiprocessing，本机） | 账本三态判定、quorum 步进、租约过期退岛、catch-up 零权重、journal pool_* 重放、PauseAdvice 合并 | 真实权重/数值、syncer 实现、网络、云 |
| 1 | 两岛小模型真机（≈$30，**待批**） | syncer 协议扩展（D-S）、中途 kill/拉起一岛、P4 carried_over 折扣、P6 降级 advice、跨岛样本 IS 后 reward 不劣化、带宽实测 | FN 规模显存/时间 |
| 2 | FN 2×8 | 全尺寸跨岛样本收益、异构岛等待时间 | — |

阶段 1 开卡前置：D-S 协议在 Rust 实现 + CPU 单测；`rl-infra-spec` 3.8/X6。

### Q9 与已有 spec 的关系

- `rl-infra-spec` 7.1：IslandStatus 字段由本 change 实装子集（G1 补全 pool_epoch 与调度字段）；7.2 扩缩池顺序不改（本 change 的 pool_join/leave 是 DiLoCo 成员变化，与 f-design §2.5 "worker 增减不等于成员变化"一致，二者是不同 tx_kind）；7.3 / D12 由本 change 承接，D12 三条约束全部继承。
- D10：暂停不暂停 syncer 不变；PauseAdvice 只收紧。
- 依赖：3.8/X6（两岛 strict 暂停）为阶段 1 前置；4.5 不改。
- 边界：`rl-multinode-island`（岛内多节点）与 `add-nebius-verda-modal-clouds`（provider 生命周期）只被调用，不重复实现。

### Q10 非目标与接口

量化/EF、DyLU、分片并发、LoRA SVD 合并不做。接口：账本增量条目保留 `wire_dtype`、`c_tokens`、`c_steps`、`shard`（可选）、`extra`；合并权重函数 `merge_weight()` 单独抽出，DyLU 将来只需改它的输入。

## D-S：syncer 需要的协议扩展（阶段 0 只设计，不改 Rust）

- **D-S1 JOIN/LEAVE**：`JOIN{island_id, incarnation, hmac}` → `JOIN_ACK{learner_slot, membership_epoch, coordinator_epoch, base_version, policy_hash}`；`LEAVE{island_id, reason}`。去掉 `learner_id < --learners` 硬上限，改为 `--max-learners`。
- **D-S2 HEARTBEAT/租约**：`HEARTBEAT{island_id, membership_epoch, inner_step, round_wall_s}`；协调器 `lease_s` 无心跳即移出成员，复用现有 progress lease（`server.rs:1554`）计时。
- **D-S3 P4 步进**：strict 条件（`server.rs:1686-1687`）由"quorum==learners"改为"Σ到齐 cap_i ≥ θ·Σcap_i 或 T_soft"（改 `server.rs:481-498` 的 quorum/grace 语义）；θ、γ、max_carry_lag、T_soft 编入契约哈希；迟到增量以 γ^lag 进入下一轮合并；catch-up 成员当轮权重 0；消息带 `syncer_epoch`。
- **D-S4 SAMPLE_INDEX**：`SAMPLE_INDEX{island_id, outer_version, inner_step, policy_hash, uri, n, has_behavior_logprob}`；协调器只存索引，按账本判定可分发。
- **D-S5 状态导出**：导出 per-learner step-time EMA 与 membership 至 IslandStatus。
- **D-S6 checkpoint**：GlobalState 增加 membership_epoch、coordinator_epoch、ledger 索引，沿用 tmp+fsync 原子写。

## 模式选择与回退（用户要求 2026-10-07 夜）

目的：随时能回到旧版 syncer 的行为。为此提供一个显式配置项 `--rl-island-scheduling legacy|elastic`（launcher 与 driver 参数），**默认 legacy**，要用新行为必须显式写 `elastic`。

| | legacy（旧模式，默认） | elastic（新模式，本 change） |
|---|---|---|
| 成员 | 启动时固定，运行中不能加岛或退岛 | 可运行中加岛（新岛首轮只同步、不参与合并）与退岛（心跳超时即移出） |
| 什么时候做一次外层合并 | 所有岛都交了增量才合并；等待超时就报错退出（与现有 syncer 严格同步一致：要求到齐数等于岛数、额外等待时间为 0） | 已到齐岛的算力之和达到总算力的 θ 比例，或到达软截止时间 |
| 迟到的增量 | 拒收 | 打折扣（每晚一步乘一次 γ）后并入下一轮 |
| 跨岛样本 | 不接受，只用本岛当前版本的样本 | 按陈旧度判定接受、带修正接受或拒收 |
| 新消息类型（加入/退出/心跳/样本索引） | syncer 不加载、不认识 | 加载 |
| journal（每岛本地的持久操作日志）里的 pool_* 记录 | 只读不写；重放时忽略 | 写入并重放 |
| IslandStatus 的调度字段（算力、陈旧度、云与价格等） | 一律为空 | 按已有来源填写 |
| 外层事实以谁为准 | 现有 syncer 的固定成员合并记录 | syncer 维护的跨岛记录（成员变更次数、外层版本、各版本的权重哈希） |

一致性：模式取值编入 session_contract_hash（各岛连接 syncer 时核对的配置指纹），因此 legacy 与 elastic 的岛或 syncer 混用会在连接时被直接拒绝（Python 侧 `check_same_mode`、`contract_fields`）。

回退路径：把参数改回 `legacy` 重启 run 即可，不需要迁移数据。elastic 期间写下的 pool_* 记录保留在 journal 里，legacy 下只读、重放时忽略，不影响旧逻辑。syncer 的 Rust 实现也按契约里的模式字段分支，legacy 分支的代码逐行不动（tasks 0.8a）。

与 syncer（Rust）侧的约定（字段名以此为准）：
- 模式取值字符串：`legacy` / `elastic`。
- 契约哈希（连接时核对的配置指纹）里的模式字段名：`island_scheduling_mode`。
- 仅 elastic 编入的参数：`quorum_theta`（θ，到齐算力比例门槛）、`carry_gamma`（γ，迟到增量每晚一步的折扣）、`soft_deadline_s`（软截止秒数）。
- 消息里的任期字段：`syncer_epoch`（u64，syncer 每次重启加 1，用于拒收旧实例的消息）。
Python 侧 `island_ledger.contract_fields(mode, theta=, gamma=, soft_deadline_s=)` 按这些名字输出。
- 契约哈希字节（与 syncer `elastic.rs::encode_contract` 一致，小端）：legacy 不追加任何字节（原哈希不变）；elastic 在原语义配置编码末尾追加 `"yeto-syncer-island-scheduling-v1\0"` | u8 长度 + `"island_scheduling_mode"` | u8 长度 + `"elastic"` | f64 quorum_theta | f64 carry_gamma | u64 soft_deadline_s | u32 q_min | u32 max_carry_lag。Python 侧在 `yeto/syncer_profile.py` 实现，黄金值见 progress.md。
- syncer 命令行（launcher 生成）：elastic 时追加 `--island-scheduling-mode elastic --quorum-theta --carry-gamma --soft-deadline-s --q-min --max-carry-lag`；岛间消息校验密钥（HMAC，用共享密钥给消息签名以防伪造）通过环境变量 `YETO_ISLAND_HMAC_KEY` 下发：launcher 从启动环境读取，在 syncer 命令前 `export`，不走命令行参数以免进入 shell 历史；elastic 缺少该变量时 launcher 拒绝启动。legacy 下命令行与原来逐字相同。

Python 侧现状：`island_ledger.IslandSchedulingMode`（默认 LEGACY）、`CrossIslandLedger(mode=...)`、`journal.append_pool_event/replay_pool(mode=...)`（默认 legacy）、`IslandController(island_scheduling=...)`、`run_fake_islands(mode=...)` 均已按模式分支；launcher/driver 的命令行参数尚未接线（tasks 0.13）。

## 跨岛样本池：索引格式与 driver 接口草案（tasks 0.12）

谁做什么、以谁为准：
- 产生样本的岛把一个 prompt 组（同一道题的 n 个回答）序列化后上传到对象存储（S3、Modal Volume 或 Nebius 对象存储），然后把一条**索引**交给 syncer。组内优势值（advantage，即每个回答相对组内平均的好坏）由产生方算好、随样本一起存。
- syncer 只存索引，不存样本本身；索引是否可被别的岛使用，以 syncer 维护的跨岛记录的判定为准（接受 / 带修正接受 / 拒收）。
- 消费样本的岛的 driver（每岛的训练主循环）从 syncer 取被接受的索引，自己去对象存储下载，校验大小与 sha256，然后只做重要性采样修正（用产生时的概率与当前策略概率之比给样本加权），不重算组内优势值。

索引字段（`yeto/rl/engine/sample_pool.py: SampleIndexEntry`，格式名 `yeto.rl.sample-index/v1`）：island_id、outer_version、inner_step、policy_hash、group_id、prompt_id、n、uri（只允许 `s3://`、`modal-volume://`、`nebius-os://`）、size_bytes、sha256、has_behavior_logprob（是否带产生时的对数概率）、advantage_included（必须为真）、created_at。
约束：同一个 group_id 只能来自一个岛；同一组重复登记内容必须相同。`SamplePoolIndex.select` 按 syncer 记录的判定筛选，旧版本优先；`prune_below` 清掉过旧的索引。
driver 接口：`CrossIslandSampleSource.fetch(selected)`（只定义形状，未实现下载与接入 batch）。legacy 模式下判定一律拒收跨岛样本，因此 select 结果为空。

## 岛侧 elastic 客户端（tasks 0.14）

谁做什么：岛的桥接层在 elastic 下改用 `ElasticRlBridge`（`yeto/rl/bridge.py`，经 `make_island_bridge(..., island_scheduling="elastic")` 选择；legacy 仍是原 `StrictRlBridge`，代码未改动，legacy 也不会导入新模块）。消息编解码在 `yeto/rl/elastic_client.py`，字节布局以 syncer `elastic.rs` 为准：每帧正文首字段是 `syncer_epoch`（u64），末尾 32 字节是 HMAC-SHA256（覆盖消息号 + 正文，密钥取环境变量 `YETO_ISLAND_HMAC_KEY`）。

流程：连上 syncer 发 JOIN（岛号、算力、化身号）→ 收 JOIN_ACK（是否首轮零权重、当前基版本）；syncer 已有基版本时会紧接着发 ELASTIC_BASE；没有时岛发 ELASTIC_INIT 提交初始参数（先到者生效），再等 ELASTIC_BASE。之后每轮：应用基版本 → 本地训练 → 发 DELTA_TENSOR（增量、c_tokens、c_steps）→ 等更新的 ELASTIC_BASE；训练期间基版本已前进则直接用新的，晚到的增量由 syncer 打折扣并入。后台线程每 lease_s/3 发一次心跳（LEASE_HEARTBEAT），退出时发 LEAVE 并停止心跳。以 syncer 的记录为准；收到 MSG_ERROR（例如被 epoch 拒收）时 JOIN 立即失败。

接入（tasks 0.15）：ports 引擎路径由 `entry.build_sync` 按模式选择，elastic 用 `bridges.ElasticAvgSync`（会话接口与 StrictAvgSync 相同，不走 StrictRlBridge 的分片接口）；驱动里的轮次号仍是本岛本地轮次，syncer 的外层版本另行记录（迟到的岛可能看到版本跳跃）。旧的 Miles 桥接路径（`yeto/rl/miles.py`）只支持 legacy，elastic 下启动即报错并提示改用 ports 引擎。syncer 任期号由 launcher `--rl-syncer-epoch` 显式给出（默认 0），同时传给 syncer 的 `--syncer-epoch` 与各岛；syncer 重启后需由操作者把该值加 1 再启动（尚未自动从 syncer 状态读取）。

## 阶段划分

- **阶段 0（CPU，$0，本次实现）**：账本、IslandStatus 扩展、journal pool_*、PauseAdvice、假岛 harness、quorum/租约协议。
- **阶段 1（两岛小模型真机，≈$30，待批）**：D-S 在 Rust 落地后跑；脚本与判据另写 prelaunch review。
- **阶段 2（FN 2×8，待批）**：依赖 FN 单岛与 codex 阶段 2 结论。

## Risks / Trade-offs

- off-policy 偏差：ACCEPT_IS 只在 `Δo≤1` 且有 behavior_logprob 时放行；阶段 1 比对 reward 曲线。
- 零权重 catch-up 稀释当轮平均（INTELLECT-1 接受）。
- 协调器单点：fencing 防脑裂但不防宕机；宕机即 run 级 stall。
- Python 参考实现与 Rust 实现漂移：阶段 1 前用同一组 JSON 用例（假岛 tape）对两边做黄金比对。

## 用户裁定记录（2026-10-07 夜）

1. 账本权威放 syncer（P8a）+ `syncer_epoch` fencing 与成员心跳租约（P9）。
2. 外层步进 P4：算力加权 quorum（θ）+ 软截止 T_soft；迟到岛增量带 γ^lag 折扣作 carried_over 并入下一轮；退岛未提交增量丢弃并记 dropped_uncommitted。θ=0.75、q_min=1、γ=0.5、max_carry_lag=2 为推荐默认 [待真机校准]。
3. 弹性成员 P5：影子加入 + catch-up，首轮零权重。
4. 样本池只做 P3 对象存储（S3 / Modal Volume / Nebius OS），协调器只存索引；不做 P1/P2。
5. P6 慢岛降级为纯 rollout 岛：只发 PauseAdvice，人确认，不自动执行。
6. GRPO 组规则 G-a（组不跨岛、同 policy_hash、advantage 生成端算好、消费端只做 IS）；G-b 为实验项；critic 家族首版拒收跨岛/陈旧样本。
7. 跨岛样本最大陈旧度 ≤2 外层步，inner lag 不设上限；IS 默认待离线比较 M2PO vs TIS/IcePop（lag 1–4）后定（tasks 0.10）。
8. P7 云价/时段再分配只留接口（IslandStatus cloud/region/price + target_resource_intent 占位）。
9. 阶段 1 两岛真机（≈$30）待批，不预登记。
10. 签名校验（HMAC）只覆盖新增的岛间消息（JOIN/LEAVE/HEARTBEAT/SAMPLE_INDEX），既有消息不加（用户裁定 2026-10-07）。

11. **断开期间用旧基座训出的增量，重新加入后按迟到增量并入**（用户裁定 2026-10-08）：岛因租约过期被移出、随后重新加入时，它在断开前已拿到的基座上训出的增量，不因"重新加入"而丢弃，按迟到增量规则处理——lag ≤ max_carry_lag 时按 γ^lag 折扣并入，超过即拒收；新加入那一轮的 catch-up 条目仍为权重 0。依据：真机 s15-island1b-20261008g 重入后第一步出现一条基于 v2 的迟到增量按 γ=0.5 并入（权重 0.497），judge v2 的 C4b 记为信息；本裁定确认这一行为符合预期，现有代码无需改动。

## 仍待用户裁定

1. elastic 模式的参数传输与合并质量：elastic 下参数是一条扁平 f32 向量（新增消息 21 ELASTIC_INIT 提交初始参数、22 DELTA_TENSOR 提交"本地参数减基版本"的增量、23 ELASTIC_BASE 由 syncer 在每次外层合并后广播新基版本），不使用旧路径的分片/多流传输，不做 RDA、ISO、HeLoCo 等合并修正，也不做 q4 线上压缩。首版接受这一差距（只用 Nesterov 外层更新）；**待用户裁定**是否要在 elastic 下对齐 legacy 的合并质量（属两岛内部通信/合并优化，本 change 原定排除）。

（阶段 1 两岛真机开卡时间另批。）

## 阶段 1 真机观察记录（2026-10-08，仅记录，未改）
- 两岛同一 rollout 的样本批哈希相同：两岛用同一 seed 和同一数据源，一直在训同一批题。legacy 模式同样如此（证据：run s15-island1b-20261008b 的 trained_sample_ids_sha256）。是否按岛错开数据切片（例如按岛号偏移或分片），待用户裁定。
- 1a（legacy/strict）重启后数据游标同样回到 0，已由 tape 证实；按代码推断（未逐条核对 1a 参数）原因与 0.23 同类：没开 `--rl-elastic` 时驱动没有批次账本，`_run` 跳过 `_restore_data_cursor`。0.23 只修了 elastic 路径，legacy 路径未改。
- rl_local_round 比 rl_round_trained 少一条（run s15-island1b-20261008c，已核实）：岛 0 被杀的那一代在 1791399970 写了 rollout 2 的 rl_round_trained 和 export_push 阶段事件，kill.json 的 kill_unix 为 1791399970.25，进程死在导出参数期间。rl_round_trained 由驱动在优化步完成后立即写入，rl_local_round 则在边界里导出参数（以及测试延迟）之后、提交增量之前才写，所以在这段窗口内被杀，就只有 round_trained 没有 local_round。重启那一代从 rollout 2 重训，两条记录又都齐了。不是数据丢失，也不需要改。
