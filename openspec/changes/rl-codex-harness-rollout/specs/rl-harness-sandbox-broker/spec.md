# Spec Delta

## Purpose

预留 agentic rollout 所用沙箱代理的接口契约：租约生命周期、环境注册表、销毁确认、错误分类与网络策略。真实后端在后续 change 中实现，本能力只要求接口、校验与本地假实现满足契约。

## ADDED Requirements

### Requirement: 租约生命周期
沙箱代理 SHALL 为每条轨迹提供独占租约，支持获取、心跳、执行命令、拷入拷出、校验、销毁与状态查询。租约心跳超过期限时，MUST 判为基础设施错误，并触发销毁。

#### Scenario: 心跳超时
- **WHEN** 持有方在期限内未发送心跳
- **THEN** 租约失效，轨迹按基础设施错误中止，环境被销毁

#### Scenario: 一轨迹一环境
- **WHEN** 两条轨迹同时获取同一 env_id
- **THEN** 两者得到互相独立的环境实例

### Requirement: 销毁以确认不存在为准
销毁 SHALL 只在状态查询确认环境已不存在后才算完成（对 Pod 类后端即返回 404）。确认之前，该环境 MUST 计入存活环境数。

#### Scenario: 删除请求已发但未确认
- **WHEN** 已发出删除，但状态查询仍返回存在
- **THEN** 存活环境计数不减少

### Requirement: 环境注册表
每个环境 SHALL 在注册表中声明以下内容，缺项或镜像未锁定时 MUST 拒绝注册：
- 按 digest 锁定的镜像；
- 资源规格；
- 超时；
- verifier 引用；
- 网络策略；
- 需要扣留的 verifier 资产。

#### Scenario: 镜像未按 digest 锁定
- **WHEN** 注册项只给出可变 tag
- **THEN** 注册被拒绝

### Requirement: 错误分类与奖励前防篡改
校验 SHALL 在可信层执行，返回签名 outcome 或基础设施错误两者之一。verifier 资产 MUST 在校验前才注入环境。环境中的 harness 与任务进程 MUST NOT 能读取签名密钥。

#### Scenario: verifier 未能运行
- **WHEN** 校验阶段环境不可达
- **THEN** 返回基础设施错误，不产生签名 outcome

### Requirement: 默认拒绝出网
环境的网络策略 SHALL 默认拒绝出网，只放行注册表中显式列出的目的地（例如网关会话端点）。

#### Scenario: 未声明的出网
- **WHEN** 任务进程访问未在策略中列出的地址
- **THEN** 连接被拒绝

### Requirement: 本 change 仅交付接口与假实现
本能力在本 change 中 SHALL 只提供接口定义、注册表校验与本地假实现。配置选择真实远程后端时，MUST 启动失败，并说明该后端尚未实现。

#### Scenario: 请求未实现的后端
- **WHEN** 配置选择 k8s 沙箱后端
- **THEN** 启动失败，并给出明确原因
