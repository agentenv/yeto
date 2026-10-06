# Spec Delta

## Purpose

规定 RL learner 岛在 Verda 上的运行方式：在 Verda 的虚拟机内以容器运行固定 digest 的镜像，只使用按需实例、不依赖对象存储，并让规划器据此接纳 Verda。

## ADDED Requirements

### Requirement: 岛在虚拟机内以容器运行固定镜像

yeto 在 Verda 上启动需要容器镜像的岛时，SHALL 在 Verda 默认虚拟机镜像上以容器方式运行指定 digest 的镜像，MUST NOT 依赖 SkyPilot 在 Verda 上不支持的容器镜像功能。容器 SHALL 能使用实例的全部 GPU 与主机网络；传入容器的环境变量 MUST NOT 出现在日志中。岛在容器内的准备与运行步骤 SHALL 与其他云上的岛相同。

#### Scenario: 默认参数启动 Verda 岛
- **WHEN** 用户以默认参数在 Verda 上启动一个 RL 岛
- **THEN** 岛在虚拟机内的容器中运行固定 digest 的镜像，完成源码准备与校验并开始训练

#### Scenario: 容器内的准备步骤与其他云一致
- **WHEN** 岛在容器内执行源码准备
- **THEN** 其 checkout 与校验结果与在支持容器镜像的云上完全相同

### Requirement: 本变更只支持按需实例且不挂载对象存储

Verda 上的岛 SHALL 只使用按需实例，MUST NOT 挂载对象存储。请求 Verda spot 实例或需要对象存储的配置 SHALL 在启动前被拒绝，并给出原因。

#### Scenario: 请求 Verda spot
- **WHEN** 用户请求在 Verda 上以 spot 实例运行 RL 岛
- **THEN** 启动前被拒绝，错误信息说明 Verda spot 暂不支持

### Requirement: 规划器接纳 Verda 作为容器岛

规划器 SHALL 把 Verda 视为可运行容器岛的云（通过虚拟机内容器的方式），并按按需价格计分；Verda MUST NOT 被当作支持多节点或支持 spot 存储的云。被拒绝的 Verda 组合 SHALL 在规划输出中写明原因。

#### Scenario: RL 岛规划包含 Verda
- **WHEN** 用户为 RL 岛规划资源，Verda 有可用的单卡按需实例
- **THEN** 规划结果中 Verda 是可选候选，价格按按需计算
