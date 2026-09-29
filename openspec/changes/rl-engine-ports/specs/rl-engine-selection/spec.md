# Spec Delta

## Purpose

规定 RL learner 如何在旧引擎路径与基于端口的新引擎路径之间选择，并保证两条路径并存期间互不干扰、可以对照验证。同时规定旧路径在验收之后如何退役。

## ADDED Requirements

### Requirement: 引擎路径选择
RL learner 与启动器 SHALL 接受 `--rl-engine`，取值为 `legacy` 或 `ports`。在等价性验收通过之前默认为 `legacy`；通过之后按"legacy 路径的退役"切换为默认 `ports`。
- 选择结果 MUST 在启动前确定，在一次运行内保持不变，并写入运行事件与产物来源记录。
- 选择 MUST 由用户显式给出，MUST NOT 从模型、岛数量或其他参数推断。

#### Scenario: 切换前默认行为不变
- **WHEN** 在默认值切换之前，用户不传 `--rl-engine` 启动 RL 运行
- **THEN** 运行走 legacy 路径，参数、源码准备和训练行为与引入该参数之前完全一致

#### Scenario: 切换后显式 legacy 行为不变
- **WHEN** 在默认值切换之后，用户显式传 `--rl-engine legacy` 启动 RL 运行
- **THEN** Miles 参数、源码准备与训练行为与切换之前的 legacy 路径一致，唯一差异是传给 learner 的命令显式带上 `--rl-engine legacy`

#### Scenario: 选择被记录
- **WHEN** 以 `--rl-engine ports` 启动
- **THEN** 事件记录和导出的来源记录中包含 `rl_engine=ports`

### Requirement: 两条路径各自固定版本
两条路径 SHALL 各自固定并校验引擎源码身份：
- legacy 路径 MUST 继续使用现有的 fork 仓库、commit、bundle 与校验流程；
- ports 路径 MUST 使用 yeto 自有的引擎 fork 与推理引擎 fork（`michaellchung/miles`、`michaellchung/sglang`）上的固定 commit，并同样校验仓库地址、commit、源码 hash、工作区干净，以及导入路径位于所固定的源码目录内。
- ports 路径的固定 commit MUST 能直接从 fork 仓库获取，MUST NOT 依赖随 yeto 分发的 git bundle。

任一路径的校验失败 MUST 在加载模型之前终止启动。修改其中一条路径的固定版本 MUST NOT 影响另一条。

#### Scenario: 各自独立的版本
- **WHEN** 更新 ports 路径的引擎 commit
- **THEN** legacy 路径的 commit、bundle 与校验值不变，legacy 运行不受影响

#### Scenario: 源码不匹配
- **WHEN** ports 路径的远端 checkout 与固定 commit 不一致或工作区不干净
- **THEN** 在加载模型之前启动失败，并报告期望值与实际值

### Requirement: 不支持的组合明确拒绝
在 R0 阶段，`ports` 路径 SHALL 支持以下组合：
- 因果语言模型 + LoRA；
- GRPO；
- 串行共置；
- strict-avg 与 decoupled 同步预设。

对其他组合（全参数、SAO、DeepSeek V4 专用 recipe、critic、分区执行等），`ports` 路径 MUST 在启动前拒绝，并提示改用 legacy 路径。legacy 路径对这些组合的支持 MUST 保持不变。

#### Scenario: 选择了尚未迁移的功能
- **WHEN** 以 `--rl-engine ports` 启动 SAO 运行
- **THEN** 启动失败，错误信息说明该组合仅 legacy 路径支持

### Requirement: 在途 PR 的语义迁移
在 ports 路径开始实现之前，所有会改动 legacy RL 路径的在途 PR SHALL 已合入或已关闭，并各自登记到迁移清单。清单中每一项 MUST 写明：
- PR 编号；
- 引入的行为（不变量、配置项、数据约定或外部格式）；
- 对应的 ports 路径实现位置；
- 用于验证的测试或验收指标。

在 ports 开发期间新合入的 RL PR MUST 同时修改两条路径，或者在迁移清单中登记为待迁移项。任何待迁移项未关闭时，MUST NOT 切换默认值。

#### Scenario: 已合入 PR 的不变量在新路径生效
- **WHEN** 迁移清单中登记了 #64 的梯度流不变量
- **THEN** ports 路径存在对应测试，并且在零梯度注入下与 legacy 路径表现一致

#### Scenario: 开发期间新合入的 PR
- **WHEN** ports 开发期间合入了一个只修改 legacy 路径的 RL PR，并且没有登记
- **THEN** 迁移清单检查失败，默认值切换被阻止

### Requirement: 等价性验收
在切换默认值之前，SHALL 对每个受支持的同步预设完成等价性验收。两条路径使用同一模型与 revision、同一数据与 revision、同一 reward 和同一硬件类型；验收按下列四层判定，阈值 MUST 在 ports 实验之前固定并写入报告头，MUST NOT 根据 ports 的结果调整：
- 第 1 层（第 1 轮严格）：主 seed 的第 1 轮，两条路径从相同权重与相同 prompt 出发，每个岛的 `completed_groups`、`action_tokens`、`reward_mean` MUST 完全相等，grad_norm 相对差 MUST ≤ 3%。
- 第 2 层（teacher forcing）：把 legacy 第 1 轮记录的 rollout（`--save-debug-rollout-data` 产物）分别回放给 legacy 与 ports 的 trainer 各训练一步。回放 MUST 复现记录（reward、token 与 group 计数与原运行相等），两条路径的初始 LoRA MUST 一致；每个岛比较 loss（绝对差 ≤ max(1e-6, 1e-3×|legacy loss|)）、grad_norm（相对差 ≤ 3%）和 optimizer 之前的 LoRA 梯度。两条路径 MUST 在 `optimizer.step()` 入口（DP 规约之后、梯度裁剪之前，去除 loss scale）按 canonical PEFT 名导出全部 LoRA 张量的 f32 梯度及裁剪系数。LoRA 梯度以 fp32 参考为锚判定：在 CPU 上以 float32 计算参考梯度，输入与引擎相同——同一初始 LoRA（取自审计文件 `base.f32`）、同一回放 batch、同一固定 revision 的模型权重（MUST NOT 启用 trust_remote_code），损失与引擎配置一致（GRPO 优势按组归一化、PPO clip、按样本平均后除以样本数；原始裁剪前梯度）。按名字一一对齐拼接后，每个岛 MUST 满足 `relL2(ports, fp32) ≤ relL2(legacy, fp32) + 0.05` 且 `cos(ports, fp32) ≥ cos(legacy, fp32) − 0.005`，其中 `relL2(x, fp32) = ‖g_x − g_fp32‖₂ / ‖g_fp32‖₂`。报告 MUST 记录 fp32 参考所用回放样本、初始 LoRA 与模型权重的来源和 sha256；任一输入缺失或 fp32 参考计算失败时该层为未完成。两条路径之间的梯度相对 L2、余弦与相对 L2 最大的张量，以及 LoRA 更新量（`‖Δ_ports − Δ_legacy‖₂ / ‖Δ_legacy‖₂`）、其余弦相似度、符号翻转比例与范数比，只报告，不判定。
- 第 3 层（第 2 轮起的分布口径）：legacy 与 ports 各以同一组 5 个 seed（17–21）运行；每个 seed 把第 2 轮起全部（岛, 轮）的 reward_mean、loss、grad_norm、action_tokens 分别平均，每个指标每个 seed 得到一个值。对每个指标，以两条路径各 5 个值做双侧双样本置换检验（精确枚举 C(10,5)=252 种划分，统计量为均值差），采用 Bonferroni 校正 α=0.05/4=0.0125：任一指标 p<0.0125 MUST 判定失败，否则通过。报告 MUST 给出各 seed 原始值、p 值、效应量（均值差/合并标准差），并写明 n=5 时的检出力局限（最小可达 p 与 80% 检出力所需的最小效应量）。
- 第 4 层（路径内 hash）：每条路径、每个 seed 内，同一策略版本在各岛应用后的全局策略 hash MUST 一致（decoupled 只比较非部分应用的版本与最终 cut）。跨路径 MUST NOT 比较 hash。
- strict-avg 预设按全部四层判定；decoupled 预设按第 2、3、4 层判定，第 1 层只报告不判定，另外两条路径导出的 PEFT MUST 都能被标准 PEFT 加载。decoupled 的 teacher forcing 未写出梯度审计文件时，LoRA 梯度的判定取自同一 trainer 配置下 strict-avg 的 teacher forcing 结果。
- loss 与 grad_norm 在事件中缺失时，MUST 从 Miles 训练日志补齐，并在报告中注明每个值的来源。
- 任一受检指标超出阈值 MUST 使验收失败；任一层缺数据时验收结果为未完成，同样 MUST NOT 切换默认值。

#### Scenario: strict-avg 等价
- **WHEN** 以相同配置在两条路径上分别运行两岛 strict-avg 3 轮（5 个 seed），并完成 teacher forcing
- **THEN** 四层全部通过，报告包含两条路径的原始记录、每层的逐项比较结果和预先声明的阈值

#### Scenario: 第 1 轮 grad_norm 超出阈值
- **WHEN** 主 seed 第 1 轮某个岛上 ports 的 grad_norm 与 legacy 相对差超过 3%
- **THEN** 验收失败，默认值保持 legacy

#### Scenario: 回放未复现记录
- **WHEN** teacher forcing 回放后的 reward 或 token 计数与 legacy 原运行不相等，或某个 prompt 不在记录中
- **THEN** 回放失败或 teacher forcing 层判定失败，不以重新采样的数据代替

#### Scenario: ports 与 legacy 离 fp32 同等接近
- **WHEN** teacher forcing 中 loss 与 grad_norm 通过，legacy 与 ports 的 LoRA 梯度各自相对 fp32 参考有 bf16 量级的误差（例如 8%～14%），两者之间的跨路径相对 L2 因两份独立误差合成而更大（例如 19%），且 ports 的相对 L2 不超过 legacy 加 0.05、余弦不低于 legacy 减 0.005
- **THEN** 第 2 层通过，报告仍给出跨路径梯度相对 L2/余弦、LoRA 更新量的相对 L2、余弦、符号翻转比例与范数比

#### Scenario: 聚合 bug 被拦下
- **WHEN** ports 的 loss 聚合方式错误（例如按 token 平均代替按样本平均），使其 LoRA 梯度相对 fp32 参考的相对 L2 超过 legacy 加 0.05，或余弦低于 legacy 减 0.005
- **THEN** 第 2 层判定失败，验收失败

#### Scenario: teacher forcing 梯度不一致
- **WHEN** 某个岛 ports 的 LoRA 梯度离 fp32 参考比 legacy 远出上述余量，或梯度张量名字/形状与 fp32 参考不能一一对应，或缺少梯度审计文件
- **THEN** 第 2 层判定失败，验收失败

#### Scenario: fp32 参考缺失
- **WHEN** fp32 参考所需的回放样本、初始 LoRA 或模型权重缺失，或参考梯度计算失败，且第 2 层没有其他失败项
- **THEN** 第 2 层为未完成，验收结果为未完成，MUST NOT 切换默认值

#### Scenario: 第 2 轮起分布显著不同
- **WHEN** 某个指标按 seed 汇总后，legacy 与 ports 的置换检验 p<0.0125
- **THEN** 验收失败，报告给出该指标的各 seed 原始值、p 值与效应量

#### Scenario: 小样本检出力局限
- **WHEN** 两条路径的差异小于 n=5 时 80% 检出力所需的最小效应量
- **THEN** 该层可能判定通过，报告明确写出这一检出力局限

#### Scenario: 跨路径 hash 不同
- **WHEN** 两条路径同一轮的全局策略 hash 不同，但各自路径内各岛一致
- **THEN** hash 层通过，跨路径的差异不作为失败依据

#### Scenario: decoupled 导出
- **WHEN** 两条路径以 decoupled 预设运行到最终导出
- **THEN** 两条路径导出的 PEFT 都能被标准 PEFT 加载，报告给出两者最终 LoRA 的相对 L2 距离以及 legacy 不同 seed 之间的同一距离作为参照，该距离不设硬阈值

### Requirement: legacy 路径的退役
只有在全部受支持的预设通过等价性验收之后，才 SHALL 将默认值切换为 `ports`。
- 切换之后，legacy 路径 MUST 在至少一个完整的真实运行周期内仍可显式选择。
- 删除 legacy 路径时，MUST 同时删除 `--rl-engine` 参数、旧 fork 的固定常量、bundle 与外部策略同步回调集成。
- 此前仅 legacy 支持的功能，MUST 在删除前已迁移，或者在文档中明确标注为不再支持。

#### Scenario: 切换默认值
- **WHEN** 全部预设都通过了等价性验收
- **THEN** 不传 `--rl-engine` 时走 ports 路径，显式传 `legacy` 时仍可运行旧路径

#### Scenario: 删除 legacy
- **WHEN** legacy 路径被删除
- **THEN** `--rl-engine` 不再是有效参数，仓库中不再有旧 fork 的固定常量与 bundle 文件，文档说明了哪些功能已迁移或已不再支持
