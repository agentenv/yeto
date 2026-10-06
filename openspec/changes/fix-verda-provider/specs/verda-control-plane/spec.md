# Spec Delta

## Purpose

使 yeto 能在 Verda 上运行 head 与 syncer：不依赖 SkyPilot 对 Verda 不支持的端口声明，而是在启动后确认 syncer 端口可以从外部到达，并限制实例对外开放的端口。

## ADDED Requirements

### Requirement: Verda 上的 syncer 不依赖端口声明

在 Verda 上启动 head 或 syncer 时，yeto MUST NOT 依赖 SkyPilot 的端口开放能力。启动后，yeto SHALL 从实例外部验证 syncer 端口可以建立连接；验证失败时，SHALL 报告失败并拆除该 head，MUST NOT 让岛去连接一个不可达的 syncer。

#### Scenario: syncer 端口可达
- **WHEN** head 在 Verda 上启动完成
- **THEN** yeto 从外部成功连接 syncer 端口后才开始启动岛

#### Scenario: syncer 端口不可达
- **WHEN** 从外部无法连接 Verda 上 head 的 syncer 端口
- **THEN** yeto 报告 syncer 不可达并拆除该 head，不启动任何岛

### Requirement: Verda head 只开放必要端口

yeto 在 Verda 上启动的 head SHALL 在实例内配置防火墙，只允许 SSH 与 syncer 端口的入站连接。

#### Scenario: 其他端口不可访问
- **WHEN** Verda head 启动完成后，从外部连接 SSH 与 syncer 以外的端口
- **THEN** 连接被拒绝
