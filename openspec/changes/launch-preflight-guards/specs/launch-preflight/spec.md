# Spec Delta

## Purpose

起机前自动检查本机线程数与每卡显存峰值，在花钱之前拦下已知会失败的上卡请求，并给出可操作的建议。

## ADDED Requirements

### Requirement: 起机前线程数预检
launcher SHALL 在创建任何云资源之前读取本用户的线程总数（与 `ps -L -u <user> | wc -l` 等价）。线程数低于开机门槛时 launcher SHALL 继续。线程数达到开机门槛但低于硬线时 launcher SHALL 按配置的行为等待或报错。线程数达到硬线时 launcher SHALL 直接报错退出，不进入等待。开机门槛默认 2800，硬线默认 3000，两者都 SHALL 可配置。

#### Scenario: 线程数低于门槛
- **WHEN** 本用户线程数为 2500，门槛 2800
- **THEN** launcher 继续起机，并把读数写进运行清单

#### Scenario: 线程数在门槛与硬线之间且配置为等待
- **WHEN** 线程数为 2900，行为配置为等待，等待上限 30 分钟
- **THEN** launcher 每隔固定间隔重读线程数，降到门槛以下后继续，超过等待上限后报错退出且没有创建任何云资源

#### Scenario: 线程数在门槛与硬线之间且配置为报错
- **WHEN** 线程数为 2900，行为配置为报错
- **THEN** launcher 立即报错退出且没有创建任何云资源

#### Scenario: 线程数达到硬线
- **WHEN** 线程数为 3100，无论行为配置是什么
- **THEN** launcher 立即报错退出且没有创建任何云资源

### Requirement: 线程预检的提示内容
线程预检报错或等待时，输出 SHALL 包含当前线程总数、门槛、硬线、SkyPilot API 服务进程的线程数及占比。输出 SHALL 给出释放线程的办法：没有正在进行的起机时执行 `sky api stop && sky api start`。输出 SHALL 提醒不要在上卡期间跑会拉起 Ray 的本机测试。

#### Scenario: SkyPilot API 服务占大头
- **WHEN** 线程总数 2900，其中 SkyPilot API 服务 1800
- **THEN** 输出写明 SkyPilot API 服务 1800 线程、占 62%，并给出重启命令和"确认没有进行中的起机再重启"的提醒

#### Scenario: 找不到 SkyPilot API 服务
- **WHEN** 本机没有 SkyPilot API 服务进程
- **THEN** 输出写明"未找到 SkyPilot API 服务"，其余内容照常输出

### Requirement: 起机前显存估算预检
launcher SHALL 在起机前为每个岛估算每张卡的显存峰值。估算 SHALL 使用模型参数量、LoRA 或全参、上下文长度、回复长度、词表大小与 logits 块大小、并行布局、训练与推理是否同卡、卡型显存。估算峰值超过阈值（默认卡显存的 90%）时 launcher SHALL 报错退出且不创建云资源。报错 SHALL 列出各分项估算值，并给出至少一条可行建议（降上下文、降回复长度、加卡或换更大显存的卡型），每条建议附上按该建议重算的峰值。

#### Scenario: 词表 logits 块导致超限
- **WHEN** 请求与 S17 M1 run c 相同（Qwen3.5-4B、单卡 H200、上下文 16k）
- **THEN** launcher 起机前报错，分项里 logits 块一项单独列出，并给出降上下文或加卡的建议

#### Scenario: 估算在阈值内
- **WHEN** 请求与 S17 M1 run d 相同（Qwen3.5-4B、单卡 H200、上下文 12288、回复 6144）
- **THEN** launcher 继续起机，并把估算分项写进运行清单

#### Scenario: 模型不在估算表中
- **WHEN** 模型参数量或结构未知
- **THEN** launcher 打印"显存未估算"警告并继续起机，运行清单记录未估算

### Requirement: 预检可关闭且关闭留痕
线程预检和显存预检 SHALL 各有一个显式关闭开关。使用关闭开关时 launcher SHALL 打印警告，并在运行清单中记录哪一项被关闭。

#### Scenario: 关闭显存预检
- **WHEN** 用户传入关闭显存预检的开关
- **THEN** launcher 不做显存估算，打印警告，运行清单记录显存预检已关闭
