# Spec Delta

## Purpose

在单个 RL learner island 内，把固定的已分配 GPU 池在 trainer、rollout 与备用角色之间安全地重新分配。重分配由 yeto 侧的岛内控制器决策和执行，必须保持样本、optimizer step、策略版本与外层 DiLoCo 协议语义不变，并以实测的端到端净收益作为启用依据。

## ADDED Requirements

### Requirement: 以 ports 引擎路径为前提
岛内重配置 SHALL 只在 `ports` 引擎路径上提供。在 legacy 路径上，或在引擎能力声明中缺少所需端口动词时，任何重配置请求 MUST 在启动或请求阶段被拒绝，训练按原配置继续。

#### Scenario: legacy 路径请求重配置
- **WHEN** 以 `--rl-engine legacy` 运行，同时开启岛内重配置
- **THEN** 启动失败，并提示重配置需要 ports 路径

#### Scenario: 能力缺失
- **WHEN** 引擎能力声明中没有 rollout 增减动词，而控制器收到 rollout 增减请求
- **THEN** 请求被拒绝，拒绝原因中列出缺少的能力，当前配置保持不变

### Requirement: 控制器位于 yeto 侧且每个 island 一份
重配置的决策、安全护栏与事务日志 SHALL 由 yeto learner 进程内的岛内控制器负责，每个 logical learner island 一份。
- 引擎只执行端口动词，MUST NOT 自行决定何时或如何重配置。
- 控制器 MUST NOT 修改其他 island，也 MUST NOT 修改 DiLoCo roster。
- 控制器 MUST NOT 使用引擎的容错语义（独立 DP、自愈、容错控制端点）。

#### Scenario: 不影响其他 island
- **WHEN** island 0 完成一次重配置
- **THEN** 其他 island 的 GPU 映射与配置 epoch 不变，syncer 看到的 learner roster 不变

### Requirement: 显式执行模式
每次运行 SHALL 在启动时固定一种执行模式：`serial-colocated`、`partitioned-serial` 或 `partitioned-overlap`。执行模式与算法契约哈希 MUST 在运行期间保持不变，MUST NOT 由自动控制改写。
- `partitioned-overlap` MUST 只在对应算法契约已经认证允许的任务对之间重叠执行。
- 当下一批生成依赖本轮更新后的权重时，MUST 在发布完成之后才开始生成。
- 算法契约哈希 MUST 等于算法描述的规范化身份哈希；执行模式允许的策略年龄 MUST NOT 超过算法描述声明的最大策略陈旧度，不满足时 MUST 在创建 GPU 进程之前拒绝启动。

#### Scenario: 执行模式年龄超出算法要求
- **WHEN** 算法描述声明最大策略陈旧度为 0，而运行配置的执行 profile 允许策略年龄 1
- **THEN** 在创建 GPU 进程之前拒绝启动

#### Scenario: 分区串行不偷跑旧版本
- **WHEN** 在 `partitioned-serial` 下，rollout 卡组空闲，而新策略尚未发布
- **THEN** 不开始下一批生成，没有任何轨迹带有旧策略 token

#### Scenario: 未认证的重叠
- **WHEN** 配置请求 `partitioned-overlap`，但该算法契约没有经过重叠认证
- **THEN** 启动失败

### Requirement: 配置与有向切换边须经认证
每个候选配置 SHALL 显式给出以下内容：
- 物理 GPU 映射；
- trainer 并行维度；
- rollout engine 布局；
- 显存与 CPU 预算；
- 恢复方案。

每条 source→target 切换边 SHALL 分别认证，并且双向分开认证。配置与边的可运行性 MUST 使用弹性基准工具的 capability 认证格式表达。

纯校验阶段 MUST 拒绝以下情况：非法或重复的 GPU 映射、未知指纹、dense-full 的 DP>1、GBS 与 DP/microbatch/累积步数不一致、`groups*samples` 不能被 optimizer steps 整除、预算未知，以及未认证的边。

#### Scenario: GBS 不一致被拒
- **WHEN** 目标配置的 DP、microbatch 与累积步数的乘积不等于 GBS
- **THEN** 计划阶段就拒绝，不产生任何资源变更

#### Scenario: 反向边未认证
- **WHEN** `P62 -> P44` 已经认证，而请求的是 `P44 -> P62`，该方向尚未认证
- **THEN** 请求被拒绝

### Requirement: 单事务与幂等控制接口
控制器 SHALL 提供以下接口：
- 描述能力；
- 查看状态；
- 无副作用的计划校验；
- 发起请求：必须带 request_id、目标配置、expected_epoch 和绝对 deadline；
- 取消。

同时 MUST 满足：
- 每个 island 同一时间最多只有一个事务。
- 相同 request_id 且内容相同的请求 MUST 返回同一结果；相同 request_id 但内容不同的请求 MUST 被拒绝。
- expected_epoch 与当前 epoch 不一致时 MUST 拒绝。
- 取消 MUST 返回以下三种结果之一：cancelled、recovery_started、already_committed。

#### Scenario: 重复请求
- **WHEN** 同一个 request_id 以相同内容提交了两次
- **THEN** 只执行一次事务，两次返回同一个事务结果

#### Scenario: 过期 epoch
- **WHEN** 请求携带的 expected_epoch 小于当前 epoch
- **THEN** 请求被拒绝，没有任何资源变更

### Requirement: 安全点与全岛一致停止
重配置 SHALL 只在执行模式定义的安全点进行，同时满足以下条件：
- optimizer step 已经返回；
- 梯度累积已清零；
- 本轮外层同步逻辑与权重发布已经完成；
- 下一批生成尚未开始；
- 没有 eval、参数导出或外层 apply 正在进行。

在重叠模式下，MUST 先停止接纳新任务并排空在途任务，再形成全岛一致的停止点。无法在限定时间内排空时，MUST 取消本次切换并恢复接纳新任务。

#### Scenario: 训练中途不切换
- **WHEN** 请求到达时，trainer 正处在梯度累积中途
- **THEN** 控制器等到下一个安全点才开始切换；如果等待超时，就取消切换，训练不受影响

### Requirement: 多轮工具轨迹按轮次边界排空
在活跃推理请求数为 0、但仍有轨迹在等待工具返回时，控制器 MUST NOT 释放这些轨迹所用的 rollout worker，也 MUST NOT 改变它们的路由。
- 排空只阻止新轨迹进入，不阻止已有轨迹完成所需的请求。
- 排空超时时 MUST 取消切换，MUST NOT 重放有外部副作用的工具调用。

#### Scenario: 工具等待中的轨迹
- **WHEN** 活跃请求数为 0、tool-wait 为 1 时收到重配置请求
- **THEN** 旧 worker 与路由都保留，直到这条轨迹结束或切换超时取消

### Requirement: rollout 增减（E1）
在 trainer 保持不动的前提下，控制器 SHALL 能够从池内备用卡增加 rollout engine，或者把 engine 退回备用池。
- 新 engine MUST 先在隔离集合中完成初始化，并对目标策略完成 payload 加载与版本确认，然后才能进入路由。
- 必须拿到按成员发布的完整清单之后，才能提交新的配置 epoch。
- 迟到的旧 generation 确认、错误的 payload、旧 epoch 请求，都 MUST NOT 影响新配置。
- 失败时 MUST 恢复为旧的 engine 集合，并重新发布同一策略。
- 本能力 MUST NOT 需要完整状态快照。

#### Scenario: 双向 rollout 切换
- **WHEN** 手动执行 `T4R2S2 -> T4R4S0`，之后再切回
- **THEN** 两次切换都成功；样本、step、策略版本与 roster 都不变；备用卡计入 GPU-hours

#### Scenario: 新 engine 加载失败
- **WHEN** 新 engine 的权重 checksum 与发布清单不一致
- **THEN** 该 engine 不进入路由，事务回到旧 engine 集合，训练继续

### Requirement: 完整状态快照与同形恢复（E2）
销毁并重建 trainer 之前，SHALL 先形成一份经过校验的完整重配置快照（ReconfigurationCut），内容包括：
- 模型或 adapter 参数、FP32 master、optimizer moments 与 step；
- scheduler 与计数器；
- 各层 RNG；
- 数据游标与已消费的 sample、group、batch；
- 外层同步进度；
- 运行指纹。

具体要求：
- 快照 MUST 原子写入并 fsync，manifest 覆盖全部状态之后，才允许释放源进程。
- 恢复 MUST NOT 用导入策略来代替 optimizer 恢复。
- 同形恢复之后，下一个冻结 batch 的训练结果 MUST 与连续运行一致。
- 快照有缺失、checksum 错误或被截断时，MUST 拒绝恢复。

#### Scenario: 同形重建一致
- **WHEN** 训练 2 步后保存快照，销毁 trainer，再以相同形状重建并恢复
- **THEN** 对冻结的下一 batch，RNG、计数、moments 与参数更新都和连续运行一致，没有额外的 optimizer 重置

#### Scenario: 快照损坏
- **WHEN** 恢复时发现快照的某个分片 checksum 不匹配
- **THEN** 恢复被拒绝，事务进入重建旧配置或 RECOVERY_REQUIRED，不继续训练

### Requirement: 事务日志与故障处理
每个事务 SHALL 把各阶段的迁移追加写入 island 持久存储上的日志，并 fsync。每条记录包括：事务 ID、源与目标配置、epoch、阶段、worker generation、快照 hash 和错误信息。

各阶段的故障处理 MUST 满足：
- **释放源资源之前失败：** 恢复原运行。
- **释放之后失败：** 从快照重建旧配置。
- **提交状态不确定：** 先查日志；MUST NOT 回退已经提交的 epoch。
- **外层提交无法对账：** 进入 RECOVERY_REQUIRED，停止消费数据。

每次失败的重试 MUST 使用新的 worker generation，而且整个事务受一个绝对 deadline 约束。

#### Scenario: 控制器在提交后崩溃
- **WHEN** 控制器在 epoch 提交后、恢复运行前崩溃，然后重启
- **THEN** 根据日志判定为已提交，继续恢复目标配置，不回退、不重复训练

### Requirement: trainer DP 变更与角色转移（E3）
改变 trainer 的 DP，或者在训练和推理角色之间转移 GPU，SHALL 同时满足以下前提：
- E2 已经认证；
- 对应的重分片 spike 通过；
- 学习行为验证通过。

另外：
- TP、PP、CP、EP 等维度 MUST 保持不变；
- GBS 与样本映射 MUST 语义等价；
- 无法重分片 optimizer 或 RNG 的边 MUST 被拒绝。

在认证完成之前，trainer 相关的边 MUST NOT 出现在建议或自动模式中。

#### Scenario: 未认证的 trainer 边
- **WHEN** 自动模式评估到 `P62 -> P44`，但 E3 尚未认证
- **THEN** 该边不可选，控制器保持当前配置

### Requirement: 外层同步兼容
重配置 MUST NOT 改变 DiLoCo 的 learner roster、fragment 与 attempt 版本、budget 或 final ACK 语义，也 MUST NOT 增加任何 wire 消息或全局 barrier。
- 暂停时长 MUST 在该同步预设可以证明安全的预算之内。
- 处于 finalization 或 budget gate 期间，MUST 拒绝切换。
- decoupled 预设在单独认证之前 MUST 保持禁用。

#### Scenario: finalization 期间请求
- **WHEN** syncer 已经下发最终 manifest 之后收到重配置请求
- **THEN** 请求被拒绝，原因是 finalization

### Requirement: 观测与成本记录
每个运行事件 SHALL 带上配置 epoch、执行模式、策略版本和事务阶段。需要记录的内容包括：
- 各阶段的计算与等待时间，其中工具等待单独计；
- 排队、活跃与 tool-wait 的轨迹数；
- ready 组的数量与消费速率；
- 策略年龄；
- 资源峰值；
- 每次切换各阶段的耗时，以及首个恢复 step 的额外开销。

GPU-hours MUST 按整个分配池计算，备用卡也 MUST 计入。

#### Scenario: 区分工具等待
- **WHEN** 某个窗口内 engine 空闲但 tool-wait 很高
- **THEN** 观测把这段时间归为工具等待，而不是 GPU 缺口

### Requirement: 半自动建议（D1）
建议 SHALL 包含以下内容：
- 源与目标配置；
- expected_epoch；
- 执行模式哈希；
- 收益与成本区间；
- 有效期；
- 拒绝原因。

建议本身 MUST NOT 自动执行。人工批准后，请求 MUST 走与手动请求相同的事务入口，并在执行前重新检查有效期、epoch、负载与护栏。条件已经变化时 MUST 拒绝，MUST NOT 替换成其他目标。

#### Scenario: 过期建议
- **WHEN** 人工批准时，建议已经超过有效期
- **THEN** 执行被拒绝，并说明原因

### Requirement: 自动控制（D2）
自动模式 SHALL 默认关闭。只有当前执行模式下至少有一条已认证、并且净收益可重复的边时，才 SHALL 允许开启。自动触发必须同时满足：
- 持续失衡窗口；
- 最短停留时间；
- cooldown；
- 频率上限；
- 保守收益下界大于成本上界加安全余量。

失败后 MUST 自动关闭自动模式。关闭自动模式 MUST NOT 打断正在进行的事务或恢复过程。

#### Scenario: 振荡负载
- **WHEN** 负载在两个状态之间快速交替
- **THEN** 受最短停留时间与 cooldown 限制，不会频繁切换

#### Scenario: 没有净收益边
- **WHEN** 基准测量显示当前模式下没有可重复净收益的边
- **THEN** 自动模式无法开启，只提供手动和建议模式
