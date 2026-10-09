# Spec Delta

## Purpose

允许在真机上单独给某一个岛换参数，用来制造身份不符、学习率调度不同、落后上限不同等负例岛，并保证这种运行不会被误当作正式训练。

## ADDED Requirements

### Requirement: 单岛换参数默认关闭
launcher SHALL 默认让所有岛使用同一份参数。只有用户传入单岛换参数开关并指定岛号和参数时，launcher 才 SHALL 给该岛换参数。

#### Scenario: 未传开关
- **WHEN** 用户没有传单岛换参数开关
- **THEN** 所有岛的命令行与环境与今天完全相同（标准样本不变）

### Requirement: 必须同时声明负例运行
单岛换参数开关 SHALL 只在同时传入负例运行开关时生效。只传单岛换参数而没有负例运行开关时，launcher SHALL 起机前报错。生效时 launcher SHALL 在终端打印醒目警告，说明该运行只用于负例验证。

#### Scenario: 缺少负例运行开关
- **WHEN** 用户传了单岛换参数，但没有传负例运行开关
- **THEN** launcher 起机前报错，不创建云资源

### Requirement: 只允许白名单参数
单岛换参数 SHALL 只接受白名单内的参数：学习率调度、落后上限、岛身份测试扰动。补交落后上限是 syncer 参数，SHALL 起机前报错并写明"syncer 参数，不能按岛换"（10-09 主 agent 代用户拍板）。其他参数 SHALL 起机前报错。岛号超出岛数时 SHALL 起机前报错。

#### Scenario: 非白名单参数
- **WHEN** 用户要单独改某岛的模型版本
- **THEN** launcher 起机前报错，并列出允许的参数

### Requirement: 换参数写进 tape 与看板
生效时 launcher SHALL 把每个被换参数的岛号、参数名、原值、新值写进运行清单。该岛 SHALL 在自己的事件 tape 开头写一条换参数事件。看板 SHALL 在该岛旁标出"负例岛"和换了哪些参数。

#### Scenario: 看板显示负例岛
- **WHEN** 岛 1 的学习率调度被换成 constant
- **THEN** 运行清单、岛 1 的 tape 都有这条记录，看板在岛 1 旁显示"负例岛：学习率调度 linear→constant"

### Requirement: 负例运行不能用于正式训练
带负例运行标记的运行 SHALL 不能用续训接上正式运行，导出命令 SHALL 拒绝导出该运行的权重。续训指 `--rl-checkpoint-store` 从仓库恢复，导出指 `yeto merge`（10-09 主 agent 代用户拍板：原文 `--rl-resume` / `yeto export` 在 main 不存在）。

#### Scenario: 导出负例运行
- **WHEN** 用户对负例运行执行导出
- **THEN** 导出命令报错，说明该运行是负例运行
