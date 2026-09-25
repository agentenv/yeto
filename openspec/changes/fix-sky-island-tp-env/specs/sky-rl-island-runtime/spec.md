# Spec Delta

## Purpose

保证 sky/Modal RL 岛交给 Megatron 的环境与 learner 实际选择的并行度和注意力后端一致：岛不得因缺少 Megatron 要求的环境变量而在参数校验阶段失败，也不得用硬编码的后端开关覆盖 learner 的选择。

## ADDED Requirements

### Requirement: 岛环境满足 Megatron 对并行度的前置要求

RL 岛任务的环境表 SHALL 包含 `CUDA_DEVICE_MAX_CONNECTIONS=1`，且 MUST 在启动 Ray 之前导出，使岛内每个 Ray worker 都继承该变量。

#### Scenario: TP>1 的岛能通过参数校验
- **WHEN** 岛以张量并行度大于 1（或上下文并行度大于 1）启动 learner
- **THEN** Megatron 的参数校验通过，启动继续进入模型加载
- **AND** 不因 `CUDA_DEVICE_MAX_CONNECTIONS` 缺失而在 `parse_args` 阶段退出

#### Scenario: TP1 的岛不受影响
- **WHEN** 岛以张量并行度 1 启动
- **THEN** 该变量存在但不改变岛的行为

### Requirement: 岛不得硬钉注意力后端

RL 岛任务的环境表 MUST NOT 设置 `NVTE_FLASH_ATTN`、`NVTE_FUSED_ATTN`、`NVTE_UNFUSED_ATTN` 中的任何一个。注意力后端 SHALL 由 learner 按 recipe 传入的 `--attention-backend` 决定，并由 Megatron 在各 actor 内据此设置上述变量。

#### Scenario: flash recipe 的岛能构造模型
- **WHEN** 岛以使用 flash 注意力的 recipe（如 gated-delta-net hybrid、DeepSeek V4）启动
- **THEN** 模型构造成功
- **AND** 不出现 `NVTE_FLASH_ATTN set to 0, but expected 1` 之类的断言失败

#### Scenario: generic provider 的岛行为不变
- **WHEN** 岛以使用 unfused 注意力的 generic provider recipe 启动
- **THEN** Megatron 依据 `--attention-backend` 选定 unfused 后端，结果与此前硬钉时一致
