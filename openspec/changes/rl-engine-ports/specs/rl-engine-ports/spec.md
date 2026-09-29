# Spec Delta

## Purpose

定义 yeto RL learner 与 RL 引擎之间按角色划分的端口契约。yeto 通过它掌控岛内执行循环，同时复用引擎的 rollout、训练和权重发布能力。引擎的内部结构不会泄露到 yeto。

## ADDED Requirements

### Requirement: 按角色划分的端口集合
yeto RL learner 在 `ports` 引擎下 SHALL 只通过以下角色访问 RL 引擎：
- rollout 池：生成完整的 group，中止在途生成；
- trainer 组：执行一次训练步，onload / offload；
- 策略状态：导出和应用可训练参数；
- 发布器：把当前策略发布到 rollout 池并确认；
- 放置：描述当前 GPU 角色映射；
- 算法描述：选择并配置 loss、advantage 和过滤规则。

yeto MUST NOT 调用这些角色之外的引擎内部对象，MUST NOT 依赖引擎的容错语义（独立 DP、自愈、容错控制端点）。端口名称和语义中 MUST NOT 出现引擎内部的组织单位（例如 cell）。

#### Scenario: 引擎内部结构不外泄
- **WHEN** 引擎升级后改变了内部的 rollout engine 或 trainer 分组结构，但端口行为不变
- **THEN** yeto 的 driver、同步 bridge 和控制逻辑无需修改，全部端口契约测试通过

#### Scenario: 拒绝启用容错语义
- **WHEN** 端口配置要求启用引擎的独立 DP、自愈或容错控制端点
- **THEN** 启动失败并给出明确原因，不会以降级方式运行

### Requirement: 数据面不经过 yeto
rollout 池 SHALL 以不透明句柄返回生成结果，trainer 组 SHALL 直接消费该句柄。
- yeto 可见的内容 MUST 限于元数据：rollout_id、policy token 与 hash、group 与 sample 标识、reward 摘要、token 计数、完成与中止计数。
- 引擎 MUST NOT 为了交给 yeto 而把 token 或张量数据额外复制到 yeto 进程。

#### Scenario: 句柄直接被训练消费
- **WHEN** driver 把一轮 rollout 的句柄交给 trainer 组训练
- **THEN** 训练完成，yeto 进程中没有出现该轮的 token 或 logprob 张量，yeto 只记录了句柄元数据

#### Scenario: 元数据足以做策略身份校验
- **WHEN** 某个 group 的 policy token 与 driver 期望的快照不一致
- **THEN** 在训练前拒绝该轮，错误信息指出不一致的 group 和 token

### Requirement: 统一的可训练状态
策略状态端口 SHALL 用一种统一格式导出和应用可训练参数。该格式包含：
- 参数布局标识；
- 按确定顺序排列的张量；
- 策略版本。

布局 MUST 能表示 LoRA、全参数和分片三种形态，同一接口 MUST NOT 为不同形态分叉。R0 阶段 MUST 实现 LoRA 布局，对未实现的布局 MUST 在启动时拒绝。

应用状态时 MUST 支持两种 optimizer 处理方式：保留 inner optimizer 状态，或重置 LoRA 的 optimizer 状态。无论哪种方式，都 MUST 保持 scheduler 进度与调用方给定的本地进度一致。

#### Scenario: 导出后再应用得到相同 hash
- **WHEN** 导出当前可训练状态，再以保留 optimizer 的方式把它应用回去
- **THEN** 重新导出的完整策略 hash 与第一次导出一致，optimizer moments 未被清零

#### Scenario: 重置 optimizer 应用
- **WHEN** 以重置方式应用一个全局策略
- **THEN** LoRA 的 optimizer moments 被清零，optimizer 对象保留，scheduler 进度等于调用方给定的本地进度

#### Scenario: 未实现的布局
- **WHEN** 请求全参数布局，而当前引擎适配只实现了 LoRA
- **THEN** 启动失败，并指出不支持的布局

### Requirement: 可训练状态的梯度流不变量
策略状态端口在导出或应用状态时 MUST NOT 破坏可训练参数的梯度累加路径。
- 应用或导出之后，下一次训练步 MUST 仍为 adapter 参数产生梯度。
- 如果某一轮的 advantage 不全为零，但训练报告的 grad_norm 恰好为 0，该轮 MUST 失败，MUST NOT 提交本地状态或向外层同步推送，并 MUST 写入事件记录。

#### Scenario: 应用状态后梯度仍然流动
- **WHEN** 应用一次全局策略后执行下一次训练步，且该步 advantage 非全零
- **THEN** 报告的 grad_norm 大于 0

#### Scenario: 零梯度被拦截
- **WHEN** 某一轮 advantage 非全零而 grad_norm 为 0
- **THEN** 该轮以不变量错误失败，没有本地状态提交，事件记录中有对应的失败条目

### Requirement: 发布必须带确认
发布器 SHALL 把指定策略发布给 rollout 池中当前全部成员，并返回一份发布清单。清单包含：
- 目标策略版本与 hash；
- payload 的 hash 与字节数；
- 参与发布的成员集合。

driver MUST 在拿到完整清单之后才开始下一轮生成。发布失败或只有部分成员确认时，MUST NOT 开始下一轮生成。

#### Scenario: 发布后才生成
- **WHEN** 一轮训练与同步结束
- **THEN** 下一轮生成开始前已存在对应策略的完整发布清单，且新生成的轨迹带有该策略的 token

#### Scenario: 部分成员失败
- **WHEN** 发布时有一个 rollout 成员没有确认
- **THEN** driver 不开始下一轮生成，并以明确错误结束或进入重试

### Requirement: 显式放置
放置端口 SHALL 返回当前 trainer 和 rollout 角色到物理 GPU 的映射。R0 阶段 MUST 支持两种放置：
- 共置；
- 启动时固定的分区。

放置 MUST 在启动时根据配置显式给出，引擎 MUST NOT 静默改写所请求的放置。

#### Scenario: 共置放置可描述
- **WHEN** 以共置方式启动一个岛
- **THEN** 放置描述中 trainer 与 rollout 使用同一组 GPU，并标注为共置

#### Scenario: 请求的放置被引擎改写
- **WHEN** 引擎的参数规范化会把请求的分区改成共置
- **THEN** 启动失败并报告冲突，不会以被改写的放置运行

### Requirement: 能力声明与握手
引擎适配 SHALL 在启动时声明自身能力：
- 支持的参数布局；
- 支持的放置方式；
- 支持的算法描述；
- 支持的执行模式；
- 引擎源码指纹。

driver MUST 在创建任何 GPU 进程之前完成比对，任一不支持项 MUST 拒绝启动。能力声明 MUST 能被弹性基准工具读取，并与其 capability 认证格式一致。

#### Scenario: 不支持的算法
- **WHEN** 配置选择了适配未声明支持的 advantage 估计方式
- **THEN** 在创建 GPU 进程之前拒绝启动，并列出支持的选项

#### Scenario: 基准工具读取能力
- **WHEN** 弹性基准工具读取一次运行的能力声明
- **THEN** 它能据此判定哪些配置可运行，不需要额外转换

### Requirement: 算法描述复用引擎算法
算法描述 SHALL 由 yeto 负责配置、校验和身份哈希，具体计算 SHALL 优先使用引擎已有的 loss、advantage 和过滤实现。算法描述的哈希 MUST 写入运行事件与产物来源记录。R0 阶段 MUST 支持现有的 GRPO 配置，以及现有的有界动态采样过滤。

#### Scenario: GRPO 配置等价
- **WHEN** 以 GRPO 和有界非零方差过滤运行 `ports` 引擎
- **THEN** 使用的 advantage 与过滤语义与 legacy 路径相同，事件中记录了算法描述哈希

### Requirement: 串行共置执行循环
yeto driver SHALL 在 R0 阶段提供 `colocated-serial` 执行模式，每轮严格按以下顺序执行：
1. 生成完整 group；
2. 训练一次 optimizer step；
3. 在安全边界执行外层同步 bridge；
4. 完整发布；
5. 开始下一轮。

外层同步 MUST 只在训练步完成之后、发布之前改变 trainer 权重。strict-avg 和 decoupled 两种同步预设 MUST 在该循环中保持现有的 policy 快照、fragment 和 finalization 语义。

#### Scenario: strict-avg 一轮
- **WHEN** 以 strict-avg 运行一轮
- **THEN** 事件顺序为：生成（policy 版本 v）→ 训练 → 导出并推送 → 等待全量 roster 的版本 v+1 → 以重置方式应用 → 发布 v+1

#### Scenario: decoupled 结束
- **WHEN** syncer 下发最终 manifest
- **THEN** driver 在安全边界应用最终 cut、校验 hash、写入进度、发送确认，并在一次完整发布之后停止，不会再开始新的一轮生成

### Requirement: 模型与训练配置映射
`ports` 引擎 SHALL 接受与 legacy 路径相同的 yeto RL 训练配置，并映射到引擎配置。映射范围包括：
- 模型结构；
- LoRA 目标；
- 批大小与采样数；
- 数据列名；
- 已支持的模型 recipe（包括 gated-delta-net hybrid 的层规格）。

对无法映射的配置项 MUST 拒绝启动，MUST NOT 静默忽略。

#### Scenario: 数据列名透传
- **WHEN** 配置指定了 prompt 与 label 的列名
- **THEN** `ports` 引擎读取到相同的列，与 legacy 路径的样本一致

#### Scenario: 未映射的配置项
- **WHEN** 配置中出现 `ports` 引擎尚未映射的选项
- **THEN** 启动失败，并指出该选项名称
