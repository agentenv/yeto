## Purpose

让 `yeto shape` 能为 RL 训练（`--training-mode rl`）的岛定价和过滤。RL 岛比 SFT 岛多了 rollout 引擎的卡数和一个固定的容器镜像，规划器必须把这些算进形状里，才能把 RL 岛按岛级分布到任意已接入的云。岛内 rollout 与 trainer 的分配策略不在本能力范围。

## ADDED Requirements

### Requirement: RL 岛形状
当规划目标为 RL 时，系统 SHALL 把每个岛的 GPU 需求算作 actor 卡数与 rollout 卡数之和（共卡模式下 rollout 卡数为 0），并以此数作为向每朵云询问机型和可用量的每机 GPU 数。分卡模式下岛 MUST 为单节点。

#### Scenario: 分卡模式定价
- **WHEN** 用户运行 `yeto shape --training-mode rl --parameter-mode full --rollout-num-gpus 4 --actor-gpus 4 --budget 60`
- **THEN** 规划器只考虑每机至少 8 卡的单节点机型，价格按整机计

#### Scenario: 共卡模式定价
- **WHEN** 用户以默认 LoRA 共卡模式规划 RL，actor 为 8 卡
- **THEN** 规划器按 8 卡岛定价，允许多节点岛（受各云多节点能力限制）

### Requirement: 镜像可用性作为过滤条件
RL 岛使用固定摘要的容器镜像。系统 SHALL 只在已验证支持以容器镜像启动的云上生成 RL 岛候选；未验证的云 MUST 被排除并在告警中说明原因。

#### Scenario: 未验证的云被排除
- **WHEN** 某云尚未通过"容器镜像启动"验证，用户为 RL 规划并启用了该云
- **THEN** 该云不出现在 RL 候选中，告警说明"该云未验证容器镜像启动"

### Requirement: spot 与断点存储绑定
RL 岛启用 spot 时依赖对象存储保存已完成的 rollout 组。系统 SHALL 只在已验证对象存储挂载的云上允许 RL 岛使用 spot 价；其余云的 RL 岛 MUST 按按需价定价。

#### Scenario: 无对象存储的云按按需价
- **WHEN** 用户为 RL 规划并开启 `--spot`，候选云中某云未验证对象存储挂载
- **THEN** 该云的 RL 岛按按需价参与比较，输出注明原因
