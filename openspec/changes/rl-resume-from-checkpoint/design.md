# 设计：跨启动续训

标注：[读码] 有文件:行号；[文档] 有官方链接；[实测] 有运行记录；[估算] 未验证；"未知" = 查不到。路径相对 yeto 仓库，基于 agentenv/main dcde202e。Miles 适配层现在在 `yeto/rl/engine/miles_adapter/`，去耦合阶段 4 后在 `yeto/rl/adapters/miles/`，下文写"Miles 适配层"。

## 1. 现状

### 1.1 已有机制 [读码]
- **轮切点**（round cut，`round_cut.py` + `cut_plugin.py` + `cut.py`）：每轮训练后由训练进程把以下内容写成分片文件 + manifest + pointer：
  - LoRA adapter 权重；优化器状态（含 FP32 主副本）；学习率调度器状态；Python/NumPy/Torch/CUDA 随机数；Megatron `iteration`、`consumed_train_samples`（cut_plugin.py:9-20、118-135）；
  - pointer 文件：`next_rollout_id`、`local_step`、`policy_version`、`policy_hash`、启动序号（round_cut.py:91-99）。
  - 恢复（`restore_cut_shard`，cut_plugin.py:680）校验：每片 sha256（cut_plugin.py:583）、manifest sha256（cut.py:260）、算法/布局/后端/步数/策略版本（cut.py:395-425）、恢复后重算 `policy_tensor_hash` 与 pointer 比对（round_cut.py:133-135）、账本里上一轮必须已记录（round_cut.py:117-121）；任一不符报错退出。
  - 数据游标：driver 按账本把数据读取位置 seek 到下一轮（driver.py:1448-1490、ledger.py:165-181）。
  - **生效条件**：只在 `LocalOnlySync`（单岛不同步）路径、且 controller 有 `checkpoint_store` 时（round_cut.py:46、153；bridges.py:71-89）；而 `--rl-checkpoint-store` 要求同时开 `--rl-elastic`（launcher.py:1872-1873、1903-1905）。不开时 `round_cuts=None`，从第 0 轮开始（bridges.py:72-73）。
  - `restore_resharded_shard`（cut_plugin.py:854）只用于运行中改数据并行，不用于跨启动。
- **controller 的 store 同步**：把整个状态目录（journal、epochs、cuts、ledger、inbox、round-cut.json）复制到 store，最后写 STORE-MANIFEST.json（controller.py:742-783）；在事务提交、拓扑/卡池变化、每次轮切点后同步（controller.py:849-850、652、700；round_cut.py:103）；失败不致命。恢复只在本地无 journal 且 store 有 MANIFEST 时复制回来，**不校验哈希**（controller.py:714-740）。
- **存储挂载**：`scheme://bucket[/prefix]` 在 sky 上用 `sky.Storage(persistent=True)` 挂到 `~/yeto-checkpoint-store/<prefix>`；绝对路径原样用（launcher.py:1726-1746、4095-4098）。**Modal 岛不读存储挂载并过滤 `s3://`/`gs://`/`r2://`，于是静默写到容器本地**（launcher.py:4801-4804、4846-4850）。
- **syncer**（Rust）：检查点含成员、epoch、外层版本、保留/丢弃记录、外层底座、外层动量与暂存张量；当前轮增量不存（elastic_server.rs:170-217、elastic.rs:907-945）；启动参数 `--checkpoint-path ~/yeto-output/yeto-state.ckpt --checkpoint-every 1`，文件在则 `--resume`（launcher.py:859-862、610-615）——**在 head 本地盘**。岛侧起点取 `max(账本下一轮, 外层底座版本)` 并套用 syncer 底座（bridges.py:421-430），岛内优化器与随机数不恢复。
- **Miles 自己的 `--save/--load`**：LoRA 时只存 adapter 与 `training_state`（iteration、优化器、学习率调度），不存随机数（Miles checkpoint.py:165-187、lora/utils.py:186-214）；数据位置由 `rollout_executor.save` 另存（train.py:91、data_source.py:128-139）。yeto **不传** `--save/--load`，反而传 `--no-load-optim/--no-load-rng/--no-save-optim/--no-save-rng/--finetune`（miles_adapter/config.py:827-831、learner.py:1496-1499），重建训练进程时检测到 `--load` 会拒绝（trainer_rebuild.py:128-129）。**本设计不用 Miles 的 `--save/--load`**：数据位置、策略版本、哈希都由 yeto 管，用 Miles 自带的会出现两套"以谁为准"。
- **tape/dashboard**：事件 tape 追加写（driver.py:274-283），恢复时发 `rl_round_cut_restored`、`elastic_resume`（round_cut.py:142、bridges.py:427）；mismatch tape 按轮号用覆盖模式写（rollout_meta_hook.py 约 615-640）；Modal tape 镜像整文件替换（modal_runner.py:399-422）；dashboard 只从 `rl_round_cut` 取策略版本（dashboard/reducer.py:260-263），无续训处理。

### 1.2 现状表

| 项目 | 单岛 | 多岛 | 缺失 |
|---|---|---|---|
| LoRA adapter 权重 | 有（需 store + `--rl-elastic`） | 只有 syncer 外层底座 | 多岛岛内 adapter；Modal 上全部 |
| 优化器状态 | 有 | 无 | 多岛 |
| 数据读取位置 | 有（账本游标） | 有（账本） | Modal 上账本不持久 |
| 随机数状态 | 有 | 无 | 多岛 |
| 学习率调度位置 | 有 | 无 | 多岛；且无"接上"的显式校验 |
| 策略版本号 | 有（pointer） | 有（外层底座版本） | — |
| 岛间外层版本 | 不适用 | 有，但在 head 本地盘 | 没进持久存储 |
| tape/事件续接 | 追加 + 恢复事件 | 有恢复事件 | mismatch tape 覆盖；Modal 镜像可能覆盖；dashboard 无续训 |
| 恢复时哈希校验 | 切点片/manifest/策略哈希 | 只有 syncer 契约 | store 复制本身无哈希 |
| 任何持久存储 | sky 有 | sky 有（syncer 除外） | **Modal 没有** |

结论：机制大半已有，主要缺口是 (1) Modal 上没有存储、(2) 续训绑死在 `--rl-elastic`、(3) store 层无哈希、(4) 多岛岛内状态与 syncer 检查点不持久、(5) tape/dashboard 不认续训。**FN 全尺寸从切点恢复从未真机验证过**（FN-TRAIN-PLAN §3）。

## 2. 存储选择

### 2.1 比较

| | Modal 卷 | S3（AWS） | Nebius 共享盘 |
|---|---|---|---|
| 写/读速度 | 文档：设计上最高 2.5 GB/s，不保证 [文档¹]；我方实测读 1 GB 文件 2.6 GB/s [实测，FN8-COLO-HEALTHGATE-REVIEW.md:121]；写未实测 | 单前缀每秒 3,500 次写/5,500 次读请求，单机可到网卡上限（最高 100 Gb/s），靠并行分段 [文档²]；从 Modal 写 S3 的实际速度未知 | 每 4 TiB 容量加读 3.70 GiB/s、写 1.89 GiB/s；单客户端最高读 15 GiB/s、写 10 GiB/s [文档³] |
| 跨云访问 | 只能 Modal 容器挂载；外部用 `modal volume` 命令行/SDK 拷贝 [文档¹]，出 Modal 流量 $0.04/GiB [文档⁴] | 任意云可访问；Modal 可用 CloudBucketMount 挂 S3，但不支持改名/追加/随机写 [文档⁵] | 只能同一 Nebius 项目的 VM 挂载 [文档³]；外部访问需经 Nebius 对象存储，是否可行未写明 |
| 费用 | $0.09/GiB/月，每月 1 TiB 免费 [文档⁴] | S3 标准存储价本次官方页面未加载出来，**未知**（以官方计算器为准）；出 AWS 流量收费 | $0.08/GiB/月（2025-07-02 价）[文档³]；但只按容量买，且我们现有 FS 是为模型放的 |
| 回收通知 | Modal：先发中断信号，退出处理有 30 秒，再强杀 [文档⁶]；GPU 函数不能设为不可抢占 [文档⁶] | AWS spot：提前 2 分钟通知，尽力而为 [文档⁷] | Nebius 抢占式 VM 通知时间：未知 |
| 适合 | **Modal 岛（现在默认）** | AWS/Verda 等 sky 岛；跨云共享 | Nebius 岛与 Nebius 上的 syncer head |

¹ https://modal.com/docs/guide/volumes ² https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html ³ https://docs.nebius.com/compute/storage/types ⁴ https://modal.com/pricing ⁵ https://modal.com/docs/guide/cloud-bucket-mounts ⁶ https://modal.com/docs/guide/preemption （30 秒出自 Modal 文档的生命周期/`simulate_preemption` 说明，经搜索摘要转述，上线前需再核一次原文）⁷ https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html

注意：Modal 卷 v2 仍是 Beta，文档写"还不能保证不丢数据"[文档¹]；切点文件数很少（远低于 v1 的 5 万个文件建议），**用 v1 卷**。卷的提交语义是"后台每几秒提交 + 容器退出时最终提交"，其他容器要 `reload()` 才看到[文档¹]——所以"切点算写完"必须以我方 MANIFEST 落盘并显式提交为准，不能只看文件在不在。

### 2.2 FN 切点要存多久 [估算]
- 体积：adapter 11.2 GB（发布载荷，s14 冒烟）。若按 bf16 计约 56 亿参数，优化器 FP32 主副本 + 两个动量 = 12 字节/参数 ≈ 67 GB，合计 ≈ 78 GB；若 11.2 GB 本身是 FP32，则约 28 亿参数，合计 ≈ 45 GB。**取 45–80 GB**，第一次真机保存时实测记账。
- 时间分三段：显存→内存（锁页约 51.7 GiB/s 实测，可忽略，数秒）；sha256（单线程约 0.5–1 GB/s 量级，未在本机型实测；切点按片并行算，4 节点各算自己的片 → 约 15–40 s）；写卷（2.5 GB/s 上限算 18–32 s；4 节点并行能否叠加未知）。**合计约 0.5–2 分钟**，第 6 组真机实测。
- 结论：**Modal 上 30 秒回收窗口写不完 FN 全量切点 → I1 在 Modal 上不成立**。对策：定期切点（每轮或每 N 轮），回收时只写"中断标记"并退出，重启从最近一个完整切点恢复，最多损失 N 轮。按 WP1 后每轮约 4.5 分钟、存一次 0.5–2 分钟，默认每轮都存会多 10–40% 时间 → **FN 默认每 2 轮存一次**（损失上限约 2 轮 ≈ $50–80），0.6B 验证每轮存。可把写卷放后台与下一轮推理重叠（显存→内存后训练进程即可继续），列为第二步优化。
- AWS spot 2 分钟通知：可能够，取决于实测保存时间；只在实测"保存时间 + 余量 < 通知时间"后，才对该云开"回收时保存"。

### 2.3 推荐
- **Modal 岛：Modal 卷**（`modal-volume://yeto-ckpt/<运行名>`），同云、免费额度够、已实测读速度。
- **sky 岛（AWS/Verda/Nebius）：S3 bucket**（已有 `sky.Storage` 路径）；Nebius 岛可用共享盘绝对路径。
- **跨云**：不做实时跨云写；每段结束（或用户需要换云）时，由一个 CPU 任务把最近一个完整切点从卷拷到 S3（FN 约 78 GB × $0.04 ≈ $3/次，[估算]）。恢复端只认 MANIFEST 哈希，不关心来源。
- **syncer 检查点**：从 head 本地盘改为同时同步到 store（Nebius head 用共享盘或 S3）。

## 3. 保存什么、多久、存在哪
- **单岛切点**（沿用格式）：adapter、优化器、学习率调度器、随机数、iteration、consumed_samples、pointer（下一轮号、策略版本、策略哈希、启动序号）+ 新增 `run_fingerprint`（算法、模型、LoRA 形状、并行布局、学习率配置、数据集指纹与洗牌种子）与 `lr_at_next_round`（下一轮应使用的学习率）与 `consumed_prompt_ids_digest`（已用题号集合的摘要，详细列表放账本）。
- **多岛**：每个岛存自己的切点（同上，路径 `<store>/islands/<岛名>/`）；syncer 检查点同步到 `<store>/syncer/`；切点 pointer 多记 `outer_version`。
- **频率**：`--rl-cut-every N`（默认：0.6B 为 1，FN 为 2）；最后一轮、收到"我方停机"时总是存一次。
- **保留**：`--rl-cut-keep K`（默认 2）；新切点 MANIFEST 落盘并提交后才删旧的。
- **目录**：`<store>/<运行名>/cuts/<轮号>/{shards…, manifest.json, MANIFEST.sha256}` + `<store>/<运行名>/LATEST`（最后写，指向最近完整切点）。

## 4. 恢复流程与校验（yeto 核心，按顺序，任一失败即报错退出，不静默从头训）
1. 读 `LATEST` → 读该切点 MANIFEST，逐文件 sha256 比对（新增，补 controller 的缺口）。
2. 比对 `run_fingerprint`；不一致拒绝，除非 `--rl-resume-allow-config-change`（写入 tape 的 `rl_resume` 事件，列出差异项）。
3. 现有切点片校验 + 策略哈希重算（已有）。
4. **版本号接上**：恢复后第一轮的 rollout 号 = pointer.next_rollout_id，策略版本 = pointer.policy_version，发布到推理端的版本号与哈希一致（发布前检查已按 WP1 改为比版本号，进程重建后退回比哈希）。
5. **学习率接上**：恢复后第一轮训练实际学习率 == pointer.lr_at_next_round（容差 0，浮点逐位比）；固定学习率（S17 第七批裁定 5e-6 固定）下等于常数，线性衰减下验证调度器步数接上。
6. **数据不重复**：账本游标接上；恢复后第一轮取到的题号与已用题号摘要无交集（数据跨 epoch 后允许重复，按 epoch 号区分）。
7. 多岛：岛切点 `outer_version` 必须等于 syncer 恢复后的外层版本或差 1（该轮增量未进外层）；否则该岛丢弃岛内切点、按重入路径从外层底座追平，并发 `rl_resume_island_rejoin` 事件。
8. 通过后发 `rl_resume` 事件：启动序号、切点轮号、各项校验结果、恢复耗时、读取字节数。

## 5. 与 tape 和 dashboard 的关系
- 同一运行名续训写入同一运行目录；所有事件带 `incarnation`（启动序号，从 0 起）。
- 事件 tape 追加；Modal tape 镜像改为"按启动序号分文件 + 追加"，不整文件替换。
- mismatch tape 文件名加启动序号；被切点"回滚"掉的轮（中断前已跑但未进切点的轮）保留原记录并在 `rl_resume` 里列出 `discarded_rounds`。
- dashboard：按启动序号画分界竖线；横轴用轮号（连续）；每段启动开销（启动→第一轮开始）单列；成本按段累加；被丢弃的轮灰显。
- 新增事件：`rl_cut_saved`（轮号、字节数、各阶段耗时、哈希）、`rl_resume`、`rl_preempt_notice`（收到回收通知时间、是否写完）。

## 6. 单岛与多岛
- 单岛：续训从 `--rl-elastic` 解绑，`LocalOnlySync` 路径对普通运行也挂轮切点；`--rl-elastic` 下行为不变。
- 多岛：岛切点 + syncer 检查点都进 store；syncer head 重启从 store 拉检查点 `--resume`。现阶段多岛 head 在 Nebius 无卡 VM，岛在 Modal：syncer 存 Nebius 侧（S3 或共享盘），岛存 Modal 卷——两处各自完整即可，不要求一个地方。

## 7. 代码放哪（去耦合原则：数学和流程在 yeto 核心，框架相关的放适配层）
- **yeto 核心**：存储抽象（本地路径 / Modal 卷 / bucket 三种，只管"写目录 + 提交 + 读"）、MANIFEST 与哈希、`LATEST` 原子切换、保留策略、`run_fingerprint`、恢复校验顺序（§4）、事件、launcher 的 Modal 卷挂载、tape/dashboard。
- **Miles 适配层**：把 Megatron 的 adapter/优化器/调度器/随机数/iteration 写成片、读回（现 cut_plugin.py 已有），报告"下一轮学习率"；verl 适配层以后实现同一组接口（FSDP 的优化器/调度器存取）。
- **不用** Miles 的 `--save/--load`（§1.1 末）。

## 8. 风险与未知
- Modal 卷写速度、4 节点并行写能否叠加：未知，第 6 组实测。
- FN 优化器真实体积：45–80 GB 估算。
- SGLang 推理是否能逐位复现：不确定。因此"续训 vs 不中断逐位一致"分两层判：(a) 恢复后状态与中断前保存的状态逐位一致（哈希，必须过）；(b) 续训后 3 轮与不中断对应 3 轮：学习率、题号逐位一致（必须），loss/reward/梯度范数在容差内（容差在 0.6B 先跑两次不中断的同配置，用两次之间的差作为噪声基线；若 SGLang 开确定性推理可用，则要求逐位一致，需先核实镜像里是否支持）。
- Modal 抢占后会在同一输入上重启函数[文档⁶]：重启时 yeto 必须走续训路径而非从头训——这条正好由"有 LATEST 就续训"保证。
