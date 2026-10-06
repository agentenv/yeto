# Spec Delta

## Purpose

保证 yeto 在 Verda 上创建、恢复与拆除实例时，只影响本次运行自己创建的实例，不会因状态查询失败或同名重拉而删除正在运行的实例，并且拆除结果以 Verda 的实际状态为准。

## ADDED Requirements

### Requirement: 不得删除非本次创建的实例

yeto 在 Verda 上的任何创建失败清理、恢复与拆除操作 MUST NOT 终止非本次运行创建的实例，也 MUST NOT 终止本次运行中仍被使用的实例。识别实例 SHALL 依据实例 id 或完整主机名的精确匹配，MUST NOT 依据主机名子串。

#### Scenario: 同名前缀的其他实例不受影响
- **WHEN** Verda 账户中存在主机名以本次集群名为前缀、但属于其他运行的实例
- **THEN** 本次运行的任何清理或拆除都不会终止该实例

#### Scenario: 创建失败只清理本次新建的实例
- **WHEN** yeto 为某个集群创建新实例时因容量不足失败，而该集群名下已有一台正在运行的实例
- **THEN** 清理只作用于本次尝试新建的实例，已在运行的实例保持运行

### Requirement: 状态查询必须如实反映运行中的实例

yeto 查询 Verda 集群状态时，正在运行的实例 SHALL 被报告为运行中。集群名 SHALL 只使用小写字符，使云上主机名与集群名一致。状态查询返回为空而本地记录中有已知实例 id 时，yeto SHALL 按实例 id 逐台复查，MUST NOT 仅凭空结果判定集群已不存在。

#### Scenario: region 名含大写时仍能查到实例
- **WHEN** 岛位于 region `FIN-01`，实例正在运行
- **THEN** 集群名与主机名均为小写，状态查询把该集群报告为运行中

#### Scenario: 空结果触发按 id 复查
- **WHEN** 批量状态查询对一个有已知实例 id 的集群返回空
- **THEN** yeto 按该 id 单独查询；实例仍在运行时，集群记录保持不变

### Requirement: 恢复前必须确认旧实例状态

当岛的作业失败或集群被报告为不存在时，yeto SHALL 先按实例 id 向 Verda 确认旧实例的状态，再决定是否重拉：
- 旧实例仍在运行或正在创建：MUST NOT 重拉新实例；
- 旧实例已删除、已停止或不存在：SHALL 以一个新的集群名重拉，MUST NOT 复用旧集群名。

#### Scenario: 旧实例仍在运行
- **WHEN** 岛的作业失败，但其实例仍处于运行状态
- **THEN** yeto 不创建新实例，并报告作业失败与实例仍在运行

#### Scenario: 旧实例已消失
- **WHEN** 岛的实例被确认已删除
- **THEN** yeto 以新的集群名重拉该岛，旧名字不再使用

### Requirement: 依赖的外部行为未经验证时禁止自动恢复

yeto 依赖的 SkyPilot Verda 适配器修正 SHALL 只在已验证的 SkyPilot 版本上启用，并且对本机与 head（包括其 SkyPilot API 服务进程）都生效。当前运行环境的 SkyPilot 版本不在已验证列表、或修正未能生效时，Verda 岛 SHALL 禁用自动恢复，并在启动时给出告警。

#### Scenario: 未验证的 SkyPilot 版本
- **WHEN** head 上安装的 SkyPilot 版本不在已验证列表中
- **THEN** Verda 岛的自动恢复被禁用，启动日志中有明确告警，岛失败时直接报告而不重拉

### Requirement: 拆除以 Verda 实际状态为准

yeto 报告 Verda 资源已拆除之前，SHALL 按实例 id 向 Verda 确认每台实例已删除或不存在，并确认其系统卷已永久删除（包括回收站）。任何一项未确认时，yeto SHALL 报告拆除未完成并列出剩余资源。

#### Scenario: SkyPilot 报告已删但实例仍在运行
- **WHEN** SkyPilot 的拆除调用返回成功，但按 id 查询到实例仍在运行
- **THEN** yeto 报告拆除未完成，并给出该实例的 id 与主机名

#### Scenario: 卷留在回收站
- **WHEN** 实例已删除，但其系统卷仍在 Verda 回收站中
- **THEN** yeto 永久删除该卷后才报告拆除完成

### Requirement: 拆除前回传诊断信息

拆除 Verda 上的岛或 head 之前，yeto SHALL 尝试回传岛的事件磁带、岛的作业日志以及 head 的 SkyPilot 日志到本机运行目录。回传失败 SHALL 记录告警，但 MUST NOT 阻止拆除。

#### Scenario: 岛失败后日志仍可查看
- **WHEN** 一个 Verda 岛的作业失败，随后运行被拆除
- **THEN** 本机运行目录中保留该岛的作业日志，可以据此定位失败原因
