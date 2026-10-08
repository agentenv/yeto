# Spec Delta

## Purpose

规定 critic 家族算法（PPO、VAPO、SAO、CompactionRL）在 ports 引擎上如何声明、校验、翻译，以及 critic 状态在 layout、receipt、外层同步、checkpoint 与验收中的契约。

## ADDED Requirements

### Requirement: critic 算法参数可声明并进入算法身份
系统 SHALL 接受 critic 相关参数（gamma、lambd、λ 模式、GAE 变体、value_clip、critic_lr、critic 每步更新次数、value loss 类型、critic 初始化方式、warm-up 步数、critic 参数模式），并且只在所选算法需要 critic 时把它们纳入规范化结果与算法哈希。不需要 critic 的算法（如默认 GRPO）的算法哈希与生成的引擎参数 MUST 保持不变。

#### Scenario: 默认 GRPO 不受影响
- **WHEN** 用户不指定任何 critic 参数并使用默认 GRPO
- **THEN** 算法哈希与生成的引擎参数和本 change 之前逐字节一致

#### Scenario: critic 参数改变哈希
- **WHEN** 两次 PPO 运行只有 gamma 不同
- **THEN** 两次运行的算法哈希不同

#### Scenario: 非 critic 算法给出 critic 参数
- **WHEN** 用户在 GRPO 下指定 `--value-clip`
- **THEN** 系统在启动任何 GPU 进程前拒绝，报错指明该字段只适用于 critic 算法

### Requirement: 启动前拒绝不受支持的 critic 组合
系统 SHALL 在启动 GPU 进程之前拒绝以下组合并给出真实原因：critic 与 elastic 重配置或 indep_dp 共存、critic 下 kl_coef 非 0、critic GPU 数与 actor 不同、critic 下仅部署 trainer 组件、critic 与 decoupled 外层同步、critic 参数模式为 lora（首轮）。报错文案 MUST NOT 声称 legacy 引擎支持 critic。

#### Scenario: elastic 与 critic
- **WHEN** 用户同时启用 PPO 与 elastic 重配置
- **THEN** 校验失败，报错说明 shared actor/critic 不支持 indep_dp

#### Scenario: critic LoRA 尚未开放
- **WHEN** 用户指定 critic 参数模式为 lora
- **THEN** 校验失败，报错说明该模式已规划但尚未实现

#### Scenario: 报错文案真实
- **WHEN** 一个不声明 critic 能力的引擎收到 PPO 请求
- **THEN** 报错不包含"只有 legacy 支持"一类表述

### Requirement: ports 单岛运行 PPO
ports 引擎 SHALL 在单岛 colocated 模式下运行 PPO（GAE），critic 与 actor 共用训练 GPU，每轮报告 value_loss 与 explained variance。在 G1 冒烟通过前，该能力 MUST 只能通过未验证机制放行参数在单岛使用。

#### Scenario: 未放行时拒绝
- **WHEN** G1 通过前用户在 ports 上请求 PPO 且未带放行参数
- **THEN** 系统拒绝并提示可用放行参数做单岛冒烟

#### Scenario: 单岛 PPO 指标
- **WHEN** 单岛 PPO 运行完成若干轮
- **THEN** 每轮记录有限的 value_loss、explained variance、policy loss 与 grad_norm

### Requirement: critic 状态契约
系统 SHALL 把 critic 作为独立 role 记录：critic layout 哈希与 actor layout 哈希分别计算并写入 receipt，receipt 同时记录 critic 参数模式与初始化来源哈希；tape/ledger 每轮记录 critic 权重哈希与 value 指标。

#### Scenario: receipt 含 critic
- **WHEN** 一轮 PPO 训练完成
- **THEN** receipt 中包含 actor 与 critic 两个 layout 哈希、critic 参数模式和初始化来源

#### Scenario: layout 不一致拒绝
- **WHEN** 两岛的 critic layout 哈希不同
- **THEN** 外层同步拒绝该轮并指出 critic layout 不一致

### Requirement: actor 与 critic 一起做 strict-avg
两岛 strict-avg 外层同步 SHALL 对 actor 与 critic 分别平均，同一轮内两者都成功才提交该轮；任一 role 失败 MUST 使整轮作废。

#### Scenario: 两岛一致
- **WHEN** 两岛 PPO 完成一轮 strict-avg
- **THEN** 两岛平均后的 actor 权重哈希一致，critic 权重哈希也一致

#### Scenario: critic 平均失败
- **WHEN** actor 平均成功而 critic 平均失败
- **THEN** 该轮不提交，两个 role 都保持上一轮状态

### Requirement: critic checkpoint 与恢复
checkpoint SHALL 包含 critic 权重与 critic 优化器状态；恢复时 actor 与 critic MUST 来自同一轮，否则拒绝恢复。

#### Scenario: kill/resume
- **WHEN** 两岛 PPO 运行在某轮后被终止并从 checkpoint 恢复
- **THEN** 恢复后的 actor 与 critic 权重哈希等于终止前最后提交轮的哈希，训练继续

#### Scenario: 轮次不一致
- **WHEN** checkpoint 中 critic 与 actor 的轮次不同
- **THEN** 恢复失败并报告两个轮次

### Requirement: critic 初始化与 warm-up
当 critic 初始化方式为复制 actor backbone 时，系统 SHALL 用初始 actor 的 backbone 加新的 value head 构造 critic，先执行指定步数的 critic-only warm-up，再进入主训练；warm-up 期间 actor 权重 MUST 不变，warm-up 产物以内容哈希进入 receipt，两岛共用同一产物。

#### Scenario: warm-up 不改 actor
- **WHEN** warm-up 50 步完成
- **THEN** actor 权重哈希与 warm-up 前一致，并产出一个带内容哈希的 critic checkpoint

#### Scenario: 主训练加载 warm-up 产物
- **WHEN** 主训练启动且 warm-up 步数大于 0
- **THEN** 主训练从 warm-up 产物加载 critic，receipt 记录该产物哈希

### Requirement: GAE 变体
系统 SHALL 支持 GAE 变体：vanilla、长度自适应 λ（λ=1−1/(α·l)）、decoupled（critic 目标与优势使用不同 λ）、cross-segment（段内局部 GAE 后乘 (γλ)^{N_{>s}}，不跨段自举）。变体缺省时结果 MUST 与原 vanilla GAE 逐元素一致；无段边界时 cross-segment MUST 退化为 vanilla。

#### Scenario: 缺省不变
- **WHEN** 不指定 GAE 变体运行 PPO
- **THEN** 优势与回报和原实现逐元素一致

#### Scenario: cross-segment 修正
- **WHEN** 一条轨迹有两段，后段 token 数为 n_2
- **THEN** 前段每个 token 的优势等于其段内局部 GAE 乘以 (γλ)^{n_2}

#### Scenario: 长度自适应
- **WHEN** α=1.5、响应长度 l=100
- **THEN** 使用 λ=1−1/150

### Requirement: VAPO、SAO 与 CompactionRL 可声明
系统 SHALL 允许声明 VAPO、SAO、CompactionRL，并把它们翻译为 PPO 基础上的 GAE 变体、value loss、critic 更新次数与 rollout 配置组合。SAO 声明 MUST 保持已验证的双 layout、双 syncer 语义；迁移完成前原 SAO streaming 入口 MUST 继续可用。

#### Scenario: CompactionRL 翻译
- **WHEN** 用户声明 CompactionRL
- **THEN** 生成的配置包含 cross-segment GAE、α=1.5 长度自适应 λ、γ=1、kl_coef=0、critic lr 3e-6、每步 2 次 critic 更新、50 步 warm-up、token 级归一化

#### Scenario: SAO 旧入口保留
- **WHEN** 迁移期间用户使用原 SAO streaming 入口
- **THEN** 运行行为与迁移前一致

### Requirement: CompactionRL rollout 压缩段
CompactionRL rollout SHALL 在剩余上下文小于 T_comp 时由同一策略生成摘要，用"系统提示 + 恢复消息 + 最近 k 步"重建上下文，每条 rollout 最多压缩 3 次，并为每个 token 输出段编号；摘要段与任务共享最终回报。

#### Scenario: 触发压缩
- **WHEN** 剩余上下文降到 T_comp 以下且本条压缩次数小于 3
- **THEN** 生成摘要段并以重建上下文继续，段编号加 1

#### Scenario: 压缩上限
- **WHEN** 本条已压缩 3 次且再次触及阈值
- **THEN** 不再压缩，按上下文耗尽处理

### Requirement: GPU 验收门槛
每个 critic 机制 SHALL 先通过单卡 G1 冒烟（指标有限、记录 GPU 型号）才在 ports 正式声明；两岛 G3 strict-avg 只使用正式声明的机制。所有 GPU 验证 MUST 在用户批准机型与预算后执行。

#### Scenario: 声明前置条件
- **WHEN** 某机制尚无 G1 通过记录
- **THEN** ports 能力表中该机制不被声明，两岛运行拒绝该机制
