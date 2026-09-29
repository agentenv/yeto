# Spec Delta

## Purpose

定义 yeto ports 路径上训推不一致（推理引擎与训练引擎在同一权重下 logprob 数值不同）的观测与修正契约：可选的机制、阈值与 logprob 来源的显式性、对训练结果的影响边界、零梯度判定、引擎声明支持的门槛，以及每个机制的验证层级。

## ADDED Requirements

### Requirement: 修正机制显式选择
ports 路径 SHALL 支持以下训推不一致机制：只观测、TIS、IcePop、OPSM、MIS（含几何平均变体）。未选择任何机制时，运行行为、引擎参数与算法身份哈希 MUST 与未引入本能力时逐字节相同。一次运行 MUST 至多选择一种修正方式；OPSM 可与其他修正方式组合，只观测不得与任何修正方式组合。

#### Scenario: 默认不变
- **WHEN** 描述中未设置任何修正字段
- **THEN** 生成的引擎参数和算法哈希与 R0 GRPO 相同

#### Scenario: 同时选择两种修正
- **WHEN** 描述同时选择 TIS 与 IcePop
- **THEN** 启动前失败，报错列出冲突的两项

### Requirement: 阈值必须显式给出
TIS 的上下截断阈值、IcePop 的区间上下界、OPSM 的 δ、MIS 的截断或屏蔽阈值 SHALL 由描述显式给出，系统 MUST NOT 为它们提供隐式默认值。阈值 MUST 为有限正数（下界允许为 0），且下界 MUST 小于上界。

#### Scenario: 缺少阈值
- **WHEN** 描述选择 TIS 但未给出上截断阈值
- **THEN** 启动前失败，报错指明缺少的字段

#### Scenario: 区间倒置
- **WHEN** IcePop 的下界大于或等于上界
- **THEN** 启动前失败，报错给出两个值

### Requirement: 只观测模式不改变训练
只观测模式 SHALL 输出训推不一致指标，至少包括训推 KL、重要性比率及其绝对偏差、有效样本比例。启用只观测模式时，同一输入上的 loss 与梯度 MUST 与未启用时数值相同（在确定性 CPU 计算上逐元素相等），屏蔽 MUST 原样保留。

#### Scenario: 梯度数值对照
- **WHEN** 在 CPU 上以相同的 logprob、advantage 和屏蔽分别计算启用与未启用只观测模式的 policy loss 及梯度
- **THEN** 两者逐元素相等，且启用时额外输出上述指标

#### Scenario: 指标写入运行记录
- **WHEN** 只观测模式在 ports 上完成一轮训练
- **THEN** 该轮的训练指标事件中包含上述指标

### Requirement: 修正语义与引擎实现一致
TIS、IcePop、OPSM、MIS 的数值行为 SHALL 由引擎的原生实现提供，yeto 只负责描述与翻译。IcePop MUST 在区间内以重要性比率加权、区间外置零。OPSM 的旧策略 logprob 来源 MUST 作为显式字段出现在描述中，取值为训练端重算或推理端；默认取训练端重算，两种取值 MUST 得到不同的算法哈希。选择推理端来源时 MUST 满足与 TIS 互斥的既有规则。

#### Scenario: IcePop 数值
- **WHEN** 在 CPU 上以已知比率调用引擎的 IcePop 实现，区间为 [0.5, 5]
- **THEN** 比率在区间内的 token 权重等于其比率，区间外的 token 权重为 0

#### Scenario: OPSM 来源进入身份
- **WHEN** 两个描述只在 OPSM 旧策略来源上不同
- **THEN** 两者的算法哈希不同

### Requirement: 外部插件可导入并有身份
每个修正机制所用的插件 SHALL 在运行环境中可导入，其源码哈希进入算法身份。若引擎附带的参考实现在运行环境中不可导入，系统 MUST 使用 yeto 命名空间下的副本，并记录副本的来源提交与许可证。

#### Scenario: 参考实现不可导入
- **WHEN** 运行镜像无法导入引擎示例目录下的 MIS 实现
- **THEN** 描述引用 yeto 副本，来源记录中含副本的源码哈希、来源提交与许可证

### Requirement: 屏蔽类机制的零梯度判定
每个会屏蔽 token 或序列的机制（IcePop、OPSM、MIS 屏蔽变体）SHALL 声明零梯度判定：当该轮被屏蔽比例为 100% 时不期望梯度。屏蔽比例不可得时 MUST 按期望有梯度判定。只观测与 TIS MUST 沿用默认 GRPO 判定。任何机制下 grad_norm 非有限 MUST 判为失败。

#### Scenario: IcePop 全屏蔽
- **WHEN** IcePop 某轮所有 token 的比率都落在区间外，grad_norm 为 0
- **THEN** 该轮不因零梯度不变量失败，事件记录屏蔽比例

#### Scenario: 部分屏蔽仍为零梯度
- **WHEN** 某屏蔽类机制某轮屏蔽比例小于 100%、组内 reward 方差大于 0，而 grad_norm 为 0
- **THEN** 该轮失败，不提交本地状态

### Requirement: 声明支持以 GPU 冒烟为门槛
引擎 SHALL 只在某机制于真实推理与训练引擎上通过单卡冒烟（指标存在、不变量无误报）后，才声明支持该机制。未声明的机制 MUST 在启动前被拒。文档 MUST 标注每个机制已达到的验证层级（CPU 测试、单卡冒烟、两岛 strict-avg），且不得声称效果收益。

#### Scenario: 未通过冒烟的机制
- **WHEN** 某机制尚未通过单卡冒烟，用户在 ports 上选择它
- **THEN** 启动前失败，报错列出当前已支持的修正机制

### Requirement: 与外层同步无关
在策略陈旧度为 0 的串行执行下，修正机制 SHALL 只作用于同一权重下的训推数值差异，与外层同步模式无关。两岛 strict-avg 下启用同一修正时，各岛算法哈希 MUST 一致，外层不变量 MUST 正常。本能力 MUST NOT 放开策略陈旧度大于 0 的执行要求。

#### Scenario: 两岛 strict-avg
- **WHEN** 两个岛以相同的 TIS 描述运行 strict-avg
- **THEN** 两岛算法哈希一致，每轮外层同步后权重哈希一致，不变量无失败

#### Scenario: 要求陈旧度大于 0
- **WHEN** 描述要求可容忍策略陈旧度大于 0
- **THEN** 启动前失败
