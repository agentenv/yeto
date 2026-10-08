## Purpose

让奖励、过滤、优势变换与 codex harness 只依赖 yeto 中立数据与协议，换训练框架时不必重写，只需各写一层薄包装。

## ADDED Requirements

### Requirement: 中立轨迹与奖励结果
系统 SHALL 定义中立轨迹（组号、样本号、提示、回复、标签、词元、推理端 logprob、损失掩码、策略令牌、分段、状态、元数据）、奖励结果（数值、是否中止、原因、附加信息）、过滤决定（保留与否、原因）。奖励函数与组过滤器的中立形式 MUST 只读写这些类型，MUST NOT 读取框架命名空间参数或框架样本对象。

#### Scenario: 中立奖励不依赖框架
- **WHEN** 在未安装 Miles 的环境中对一条中立轨迹调用 math/gsm8k/length/gdpo 奖励与非零方差过滤器
- **THEN** 得到奖励结果与过滤决定（math 奖励运行时仍需 Miles 判分工具，除外）

#### Scenario: 中止语义
- **WHEN** codex 奖励验签失败需要中止样本
- **THEN** 中立奖励返回"中止"结果，由适配层翻译成该框架的中止状态

### Requirement: Miles 包装同输入同输出
Miles 适配层 SHALL 为每个中立奖励与过滤器提供与原签名相同的包装；对同一组输入，包装后的返回值、样本状态、样本元数据、过滤返回 MUST 与改动前逐一相同。

#### Scenario: 对照测试
- **WHEN** 用录制的样本集分别跑改动前实现与新包装
- **THEN** 数值、状态、元数据、保留/丢弃与原因全部相同

### Requirement: 奖励与优势变换为纯函数
yeto 已有的奖励后处理与优势变换（GRPO 归一化复刻、MaxRL、MAPO、GDPO、超长惩罚、超长过滤）SHALL 以"输入奖励列表与分组、输出数值"的纯函数形式存在，不读框架参数与样本对象；Miles 插件改为薄包装，现有等价测试 MUST 继续 `torch.equal` 通过。

#### Scenario: 纯函数与插件等价
- **WHEN** 运行 `test_rl_reward_pipeline_equivalence` 与 `test_rl_seq_adv_miles`
- **THEN** 全部逐位相等

#### Scenario: 插件源码哈希变化有记录
- **WHEN** 拆分导致插件源码哈希变化
- **THEN** 变化前后哈希与 CPU 逐位一致证据写入旧→新哈希对照表，才可沿用旧 GPU 证据

### Requirement: 轮次元数据与策略令牌经核心接口
算法扩展与 harness SHALL 通过核心提供的接口读取策略令牌、写轮次元数据与计数器，MUST NOT 直接 import Miles 适配层的实现模块。

#### Scenario: harness 无反向 import
- **WHEN** 边界检查扫描 `yeto/rl/harness/` 与 `yeto/rl/algos/`
- **THEN** 不再出现对 `rollout_meta_hook` 的 import（白名单对应条目已删）

### Requirement: 会话服务协议
yeto SHALL 以文档与测试替身定义会话服务协议（创建会话、会话内 OpenAI 兼容聊天且强制返回 logprob、取会话样本、删除会话）；Miles 会话服务被视为该协议的一种实现；codex harness MUST 只依赖该协议、轮次元数据接口与奖励结果。

#### Scenario: 用替身跑 codex harness
- **WHEN** 用协议测试替身代替 Miles 会话服务运行 codex harness 的 CPU 测试
- **THEN** 测试通过，行为与接 Miles 时相同

### Requirement: math 判分工具拷贝只作测试用
Miles 的 math 判分工具拷贝 SHALL 只存在于测试目录并标明"许可证待核实"，`yeto/` 运行代码 MUST NOT import 它；运行时 math 奖励在未安装 Miles 时 MUST 明确报错而非静默回退。

#### Scenario: 运行代码不引用测试拷贝
- **WHEN** 边界检查扫描 `yeto/`
- **THEN** 没有对测试目录判分工具拷贝的 import

### Requirement: 用户自定义奖励环境
系统 SHALL 允许用户以中立奖励接口编写自定义奖励，按中立名注册或以 `模块:函数` 加载；加载时 MUST 记录源码哈希进插件身份；各后端适配层 MUST 自动包装已注册的中立奖励，用户代码不需 import 任何训练框架。仓库 SHALL 附一个最小示例及测试。

#### Scenario: 自定义奖励经 Miles 生效
- **WHEN** 用户注册示例奖励并在 Miles 后端启用
- **THEN** Miles 调用到的结果与直接调用该中立函数相同

#### Scenario: 未注册名
- **WHEN** 配置引用未注册的奖励名
- **THEN** 启动前报错并列出可用名
