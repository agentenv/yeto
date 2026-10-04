# Spec Delta

## Purpose

定义 yeto ports 路径上多算法描述的契约：用什么字段描述算法，如何规范化和计算身份哈希，如何从引擎参数吸收，如何与引擎能力和执行模式比对，哪些组合启动前拒绝，以及算法身份怎样进入来源记录、怎样在岛之间保持一致。

## ADDED Requirements

### Requirement: 结构化算法描述
ports 路径的算法描述 SHALL 至少覆盖以下几组配置：
- advantage：估计方式、组内 std 归一、reward 后处理插件；
- loss：clip 下界、clip 上界、dual-clip 系数、聚合方式、loss 变体；
- KL：放置方式为 `none`、`reward` 或 `loss`，外加系数和估计器；
- 训推不一致修正：方式与阈值；
- 采样：动态过滤、过滤替换上限、超采样、overlong 处理；
- entropy 系数；
- 插件列表；
- 执行要求：是否需要 critic、可容忍的最大策略陈旧度、是否要求整组就绪、是否需要 rollout logprob。

未显式给出的字段 MUST 取默认值，默认值 MUST 与 R0 的 GRPO 行为一致。字段值非法时（类型错误、非有限数、越界、未知枚举值）MUST 在启动前报错，报错 MUST 指明字段名。

#### Scenario: 默认描述等于 R0 GRPO
- **WHEN** 不指定任何算法字段启动 ports 运行
- **THEN** 算法描述等价于 R0 的 GRPO 描述，生成的引擎参数与 R0 逐字节一致

#### Scenario: 非法字段值
- **WHEN** 描述中 clip 上界为负数，或 KL 放置方式为未知值
- **THEN** 启动前失败，报错指明该字段及允许的取值

### Requirement: 规范化与身份哈希兼容
算法描述 SHALL 有唯一的规范化 JSON 表示，身份哈希为该表示的 SHA256。语义相同的描述 MUST 得到相同的哈希，例如整数与等值浮点数、字段顺序不同的描述。只使用 R0 已有字段的描述，其规范化结果与哈希 MUST 与 R0 逐字节相同。使用了新字段的描述 MUST 以新的 schema 标识规范化，且不得与任何 R0 描述的哈希相同。

#### Scenario: R0 哈希不变
- **WHEN** 以 R0 的 GRPO 加有界过滤配置计算哈希
- **THEN** 哈希与 R0 记录的 golden 值相同

#### Scenario: 新字段改变哈希
- **WHEN** 在同一描述上设置 clip 上界
- **THEN** 规范化结果带新 schema 标识，哈希与未设置时不同

### Requirement: 插件身份
算法描述引用的每个插件 SHALL 以可导入路径和该插件源码的 SHA256 共同标识，二者都进入身份哈希。插件无法导入，或源码哈希与描述不符时，MUST 在启动前拒绝。

#### Scenario: 插件源码被改动
- **WHEN** 描述记录的插件源码哈希与运行环境中该插件的源码不一致
- **THEN** 启动前失败，报错给出两个哈希

### Requirement: 引擎算法参数由描述独占
所有影响训练目标的引擎算法参数 SHALL 只由算法描述翻译生成。额外透传给引擎的参数中如果出现这类参数：
- 已纳入映射的参数 MUST 被吸收进算法描述，参与校验与哈希；
- 同一字段在描述中已有不同取值时，MUST 在启动前报告冲突，不得择一；
- 影响训练目标但未纳入映射的参数 MUST 在启动前拒绝，报错说明该参数尚未纳入算法描述。

#### Scenario: 吸收透传参数
- **WHEN** 额外参数包含 clip 上界 0.28，描述中未设置该字段
- **THEN** 描述中 clip 上界为 0.28，运行事件记录的哈希与直接在描述中设置 0.28 时相同

#### Scenario: 透传参数与描述冲突
- **WHEN** 描述中 clip 上界为 0.28，额外参数给出 0.3
- **THEN** 启动前失败，报错列出两个取值

#### Scenario: 未映射的算法参数
- **WHEN** 额外参数包含一个会改变 loss 计算、但尚未纳入映射的引擎参数
- **THEN** 启动前失败，报错指明该参数

### Requirement: KL 放置方式必须生效
KL 放置方式 SHALL 与 advantage 估计方式匹配，保证所配置的 KL 实际影响训练：
- 若估计方式会丢弃 reward 中的 KL（grpo、gspo），`placement=reward` 且系数大于 0 MUST 在启动前拒绝，报错 MUST 提示改用 `placement=loss`；
- `placement=none` 时 MUST NOT 加载参考模型；
- R0 描述中的 KL 系数 SHALL 解释为 `placement=reward`。

#### Scenario: GRPO 配 reward 中的 KL
- **WHEN** 以 GRPO 和大于 0 的 R0 KL 系数启动 ports 运行
- **THEN** 启动前失败，提示此估计方式忽略 reward 中的 KL，应改用 `placement=loss`

#### Scenario: 无 KL 不加载参考模型
- **WHEN** 描述为默认 GRPO，未配置 KL
- **THEN** 生成的引擎配置不会触发参考模型加载

### Requirement: 能力与执行要求匹配
引擎适配 SHALL 按机制维度声明支持项（advantage 估计方式、loss 变体、聚合方式、KL 放置、修正方式、reward 后处理插件、过滤器），并声明执行能力（是否支持 critic、可提供的最大策略陈旧度、是否提供 rollout logprob）。driver MUST 在创建任何 GPU 进程之前把算法描述与声明比对，任何未声明的机制，或未满足的执行要求，都 MUST 拒绝启动，报错 MUST 列出缺失项和已支持的选项。新增维度 MUST 能被弹性基准工具读取，且不破坏其对已有字段的读取。

#### Scenario: 描述可表达但引擎未开放
- **WHEN** 描述设置了 clip 上界，而引擎适配未声明支持该机制
- **THEN** 在创建 GPU 进程之前拒绝启动，报错列出未支持的机制

#### Scenario: 需要 critic
- **WHEN** 描述声明需要 critic，而引擎适配声明不支持 critic
- **THEN** 在创建 GPU 进程之前拒绝启动，并提示该组合仅 legacy 路径支持

#### Scenario: 陈旧度要求超出执行模式
- **WHEN** 描述要求的最大策略陈旧度为 0，而执行模式声明会产生大于 0 的陈旧度
- **THEN** 在创建 GPU 进程之前拒绝启动

### Requirement: 不兼容组合在启动前拒绝
以下组合 SHALL 在创建任何 GPU 进程之前被拒绝，报错 MUST 给出可行的替代配置：
- 同时启用训推不一致修正，又把 rollout logprob 当作 old logprob；
- 同时设置 reward 中的 KL 系数与 loss 中的 KL 系数；
- 序列级 ratio 的估计方式未显式给出 clip 范围；
- 要求二值奖励的机制，配了未声明为二值的奖励函数。

#### Scenario: 互斥的 logprob 来源
- **WHEN** 描述同时启用 TIS，并把 rollout logprob 当作 old logprob
- **THEN** 启动前失败，报错说明二者互斥

### Requirement: 按算法判定零梯度不变量
ports driver 判定"本轮 advantage 非零但 grad_norm 为 0"时，SHALL 使用算法描述声明的判定条件。默认 GRPO 的判定 MUST 与 R0 相同，即存在组内 reward 方差大于 0。会屏蔽 token 或序列的机制 MUST 声明其判定，使全部 token 被合法屏蔽的轮次不被判为失败。任何算法下 grad_norm 非有限 MUST 仍判为失败。

#### Scenario: GRPO 判定不变
- **WHEN** 默认 GRPO 某轮有组内 reward 方差大于 0，而 grad_norm 为 0
- **THEN** 该轮失败，不提交本地状态，并写入事件

#### Scenario: 合法的全屏蔽
- **WHEN** 某机制声明"全部 token 被屏蔽时不期望梯度"，某轮其屏蔽比例为 100% 且 grad_norm 为 0
- **THEN** 该轮不因零梯度不变量失败，事件记录屏蔽比例

### Requirement: 未验证机制的受控放行
为了让"可表达，未开放"的机制能完成声明前所需的 GPU 冒烟，ports 路径 SHALL 提供按机制名显式放行的开关，可以重复指定。放行 MUST 满足以下全部条件：
- 只豁免被点名机制的能力检查，其他检查（包括拒绝矩阵、执行要求、插件身份）照常执行；
- 只在单岛运行中生效，与多岛或外层同步组合时 MUST 在启动前拒绝；
- 被放行的机制名 MUST 写入运行事件和导出的来源记录，产物 MUST 标记为"含未验证机制"；
- 放行信息 MUST NOT 改变算法描述的身份哈希。

放行开关对 legacy 路径无效，与 legacy 组合时 MUST 在启动前拒绝。

#### Scenario: 单岛冒烟放行
- **WHEN** 以单岛 ports 运行，放行一个尚未声明支持的机制
- **THEN** 该机制通过能力检查，运行事件和来源记录列出被放行的机制名

#### Scenario: 多岛不得放行
- **WHEN** 以两岛 strict-avg 运行并使用放行开关
- **THEN** 在创建 GPU 进程之前拒绝启动

#### Scenario: 放行不绕过其他检查
- **WHEN** 放行某机制，但描述同时触发了拒绝矩阵中的组合
- **THEN** 仍然在启动前拒绝，报错指向该组合

### Requirement: 算法身份进入来源记录并在岛间一致
ports 运行 SHALL 在运行事件和导出的来源记录中写入算法描述的规范化 JSON 与哈希。同一次运行由启动器下发预期的算法哈希，每个岛 MUST 在加入外层同步之前核对本岛算法哈希与预期值一致，不一致 MUST 拒绝加入并写入事件。legacy 路径的来源记录 MUST 保持不变。

#### Scenario: 导出记录包含算法身份
- **WHEN** ports 运行完成并导出产物
- **THEN** 来源记录中包含算法描述的规范化 JSON 与哈希，与运行事件中的一致

#### Scenario: 岛间算法不一致
- **WHEN** 某岛的算法哈希与启动器下发的预期哈希不同
- **THEN** 该岛在加入外层同步之前失败，不推送任何状态

#### Scenario: legacy 不受影响
- **WHEN** 以 legacy 路径运行并导出
- **THEN** 来源记录与本 change 之前逐字节相同
