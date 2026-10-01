# Spec Delta

## Purpose

定义 ports 路径上 GRPO 家族机制（clip-higher、dual-clip、token 级聚合、Dr.GRPO、KL loss、entropy、超采样、overlong 塑形与过滤）从"可表达未开放"到"声明支持"的契约：各机制的参数约束、翻译结果、reward 后处理与内置归一化的等价性、参考模型身份，以及声明支持所需的验证证据。

## ADDED Requirements

### Requirement: 机制声明支持须有验证证据
引擎适配 SHALL 只在某机制的 CPU 验证与单卡 GPU 冒烟都通过后，才在能力声明中加入该机制。未声明的机制 MUST 继续在创建 GPU 进程之前被拒绝。未选择任何新机制的描述，其算法哈希与生成的引擎参数 MUST 与本 change 之前逐字节相同。

#### Scenario: 默认 GRPO 不变
- **WHEN** 以不含任何新机制的描述启动 ports 运行
- **THEN** 算法哈希与引擎参数与本 change 之前逐字节相同

#### Scenario: 冒烟未通过的机制仍被拒
- **WHEN** 某机制尚未完成单卡 GPU 冒烟，描述中选择了它
- **THEN** 在创建 GPU 进程之前拒绝启动，报错列出该机制

### Requirement: clip 与聚合类机制
描述 SHALL 能独立设置 clip 上界、dual-clip 系数和 token 级聚合，并各自翻译为对应的引擎参数。clip 上界 MUST 为有限正数。dual-clip 系数 MUST 大于 1，否则启动前拒绝。未设置时 MUST NOT 输出对应参数。

#### Scenario: clip 上界翻译
- **WHEN** 描述设置 clip 下界 0.2、上界 0.28
- **THEN** 生成的引擎参数包含上界 0.28，且能被引擎的参数解析器接受

#### Scenario: dual-clip 系数不大于 1
- **WHEN** 描述设置 dual-clip 系数为 1.0
- **THEN** 启动前失败，报错指明该字段须大于 1

### Requirement: Dr.GRPO
描述 SHALL 能分别关闭组内 std 归一和启用常数分母聚合。常数分母 MUST 由描述显式给出，为有限正数，并进入算法哈希。常数分母聚合所用的插件 MUST 以源码哈希标识；若插件来自引擎示例的复制，MUST 记录其来源版本与原始源码哈希。常数分母聚合在不支持的并行配置下（例如上下文并行大于 1）MUST 在启动前拒绝。

#### Scenario: 去 std
- **WHEN** 描述关闭组内 std 归一
- **THEN** 一组奖励 [1, 0, 0, 0] 的 advantage 等于奖励减组均值，不除以标准差

#### Scenario: 常数分母未给出
- **WHEN** 描述启用常数分母聚合但未给出分母
- **THEN** 启动前失败，报错指明缺少分母

#### Scenario: 常数分母数值
- **WHEN** 以常数分母 D 对若干样本的逐 token loss 聚合
- **THEN** 结果等于所有有效 token loss 之和除以 D，与样本长度无关

### Requirement: KL loss 与参考模型身份
`placement=loss` SHALL 翻译为启用 KL loss、给出系数和估计器（k1、k2、k3、low_var_kl 之一）的引擎参数，并触发参考模型加载。启用 KL loss 时，描述 MUST 记录参考模型身份（base 模型的来源与修订），该身份 MUST 进入算法哈希，从而进入岛间一致性校验。参考模型身份与运行实际加载的 base 模型不一致时，MUST 在启动前拒绝。

#### Scenario: KL loss 翻译
- **WHEN** 描述为 GRPO 加 `placement=loss`、系数 0.001、估计器 k3
- **THEN** 生成的引擎参数启用 KL loss，系数与估计器与描述一致，且引擎会加载参考模型

#### Scenario: 岛间参考模型不同
- **WHEN** 两个岛的描述只有参考模型修订不同
- **THEN** 两者算法哈希不同，后启动的岛在加入外层同步之前失败

#### Scenario: 未知估计器
- **WHEN** 描述的 KL 估计器为未知值
- **THEN** 启动前失败，报错列出允许的估计器

### Requirement: entropy 与超采样
描述 SHALL 能设置 entropy 系数和超采样批大小，并翻译为对应引擎参数。entropy 系数 MUST 为有限数。超采样批大小 MUST 不小于每轮 rollout 批大小，且只在启用动态过滤时允许设置。超采样导致各岛每轮有效样本数不同时，运行事件 MUST 记录每岛每轮的有效样本数。

#### Scenario: 无过滤时设置超采样
- **WHEN** 描述设置了超采样批大小，但未启用动态过滤
- **THEN** 启动前失败，报错说明超采样须配合动态过滤

#### Scenario: 有效样本数可见
- **WHEN** 两岛以超采样加动态过滤运行一轮
- **THEN** 每个岛的运行事件都记录本轮实际进入训练的样本数

### Requirement: reward 后处理与内置归一化等价
ports 路径 SHALL 只使用 yeto 自有的 reward 后处理入口，该入口按"reward 塑形 → advantage 变换"的顺序执行，且各阶段配置来自算法描述。未配置任何塑形与变换时，入口输出 MUST 与引擎内置的 reward 后处理逐元素相同，包括：按 prompt 分组、同一 rollout 的多段样本合并为一个奖励、同一 rollout 奖励不一致时报错、组大小为 1 时不除以标准差、组内标准差为 0 时不除以标准差。入口 MUST 保留注册新 advantage 变换的扩展点，新变换 MUST 以插件身份进入算法哈希。引擎提供了入口无法获得的分组信息时（例如多 LoRA 的显式分组），MUST 在启动前拒绝。

#### Scenario: 与内置逐元素相同
- **WHEN** 同一批样本（含多段样本、G=1 的组、全部奖励相同的组）分别经 yeto 入口与引擎内置处理
- **THEN** 两者输出的原始奖励与归一化奖励逐元素相同

#### Scenario: 多段样本奖励不一致
- **WHEN** 同一 rollout 的两段样本奖励不同
- **THEN** 入口报错，指明该 rollout 与各段奖励

### Requirement: overlong 软惩罚
描述 SHALL 能以 L_max 与 L_cache 启用 overlong 软惩罚，作为 reward 塑形阶段，在组归一化之前加到原始奖励上。对响应长度 L：L ≤ L_max − L_cache 时惩罚为 0；L_max − L_cache < L ≤ L_max 时惩罚为 (L_max − L_cache − L)/L_cache；L > L_max 时惩罚为 −1。L_cache MUST 为正且不大于 L_max，L_max MUST 不大于生成长度上限，否则启动前拒绝。

#### Scenario: 分段取值
- **WHEN** L_max=100、L_cache=20，响应长度分别为 80、90、100、101
- **THEN** 惩罚分别为 0、−0.5、−1、−1

#### Scenario: 参数越界
- **WHEN** L_cache 大于 L_max
- **THEN** 启动前失败，报错指明两个取值

### Requirement: overlong 过滤
描述 SHALL 能启用 overlong 过滤：被截断的样本 MUST NOT 贡献 loss 梯度，但 MUST 仍参与其所在组的 advantage 统计。该语义与 DAPO 原文的差异 MUST 写入文档。启用过滤后，某轮所有样本都被截断时，零梯度不变量 MUST NOT 误判失败。

#### Scenario: 截断样本不计 loss
- **WHEN** 一组中有一条样本被截断，启用 overlong 过滤
- **THEN** 该样本的 loss mask 全为 0，组内其他样本的 advantage 与未启用过滤时相同

#### Scenario: 全部截断
- **WHEN** 某轮所有样本都被截断且 grad_norm 为 0
- **THEN** 该轮不因零梯度不变量失败，事件记录被过滤的样本数

### Requirement: 两岛组合配置一致
以 clip-higher、token 级聚合与 overlong 处理组合的描述，两岛 strict-avg 运行时 SHALL 在两岛得到相同的算法哈希，外层同步正常完成，零梯度不变量不误报。

#### Scenario: 两岛组合冒烟
- **WHEN** 两个单卡岛以相同的组合描述进行 strict-avg 运行若干轮
- **THEN** 两岛事件中的算法哈希相同，每轮外层同步完成，没有零梯度误报
