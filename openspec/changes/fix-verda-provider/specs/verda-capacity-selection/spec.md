# Spec Delta

## Purpose

保证 yeto 为 Verda 生成的候选资源都是 Verda 实际存在、SkyPilot 能够识别并且当前可能有货的型号与地区，使选择结果与 Verda 控制台看到的可用性一致；同时兼容 Verda 官方的凭据格式与 API 响应。

## ADDED Requirements

### Requirement: 兼容 Verda 官方凭据与 API 响应

yeto SHALL 从 JSON 格式的 `~/.verda/config.json` 或 Verda CLI 写入的 INI 格式 `~/.verda/credentials` 读取 Verda 凭据，也 SHALL 接受环境变量提供的凭据。Verda API 返回纯文本或空响应体时，yeto SHALL 按文本处理而不是报解析错误；HTTP 错误 SHALL 在错误信息中保留 Verda 返回的内容，且 MUST NOT 在任何日志中输出凭据。

#### Scenario: 只有 INI 凭据
- **WHEN** 本机只有 Verda CLI 写入的 INI 格式凭据文件
- **THEN** yeto 能认证并列出 Verda 的实例型号与可用性

#### Scenario: 创建接口返回纯文本 id
- **WHEN** 创建实例的接口返回纯文本的实例 id
- **THEN** yeto 得到该 id，不报解析错误

#### Scenario: 容量不足的错误信息可读
- **WHEN** Verda 以 503 拒绝创建请求
- **THEN** yeto 的错误信息包含 Verda 给出的原因

### Requirement: 候选必须可被 SkyPilot 识别并按实时库存排序

yeto 为 Verda 生成的每个候选 SHALL 同时满足：型号与地区真实存在于 Verda；SkyPilot 能识别该实例型号；下发给 SkyPilot 时带有显式的实例型号。候选 SHALL 按 Verda 实时可用性排序；实时显示无货的候选降低优先级而不直接删除。已下线的地区 MUST NOT 出现在候选中。

#### Scenario: SkyPilot 静态目录缺少的型号
- **WHEN** Verda 实时可用性显示 FIN-02 有 1×L40S，而 SkyPilot 自带的目录中没有 L40S
- **THEN** yeto 仍能以该型号申请实例，SkyPilot 不会因目录缺失而拒绝

#### Scenario: 已下线地区
- **WHEN** Verda 的地区列表中已没有某个地区
- **THEN** yeto 生成的候选不包含该地区

### Requirement: 容量竞争时按候选依次重试

创建因容量不足失败时，yeto SHALL 依次尝试其余候选，并在每轮失败后刷新实时可用性，采用有上限的退避；全部候选都失败时，SHALL 报告已尝试的候选与各自的失败原因。

#### Scenario: 首选候选刚被抢走
- **WHEN** 首选候选在可用性查询后、创建前被他人占用，创建返回容量不足
- **THEN** yeto 转而尝试下一个候选，而不是立即判定运行失败
