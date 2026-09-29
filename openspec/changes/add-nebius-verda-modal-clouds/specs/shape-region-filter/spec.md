## Purpose

`yeto shape --regions` 的多云语法：用户如何限定每朵云的候选区域，旧写法如何继续工作，以及每朵云对"区域"的不同解释。

## ADDED Requirements

### Requirement: 带云前缀的区域条目
`--regions` SHALL 接受逗号分隔的条目，每条形如 `cloud:region`。无前缀的条目 SHALL 按 `aws:` 解释。`all` 和 `cloud:all` SHALL 分别表示"所有云的所有区域"和"该云的所有区域"。对未接入的云名或该云不存在的区域，命令 MUST 报错退出并列出该云的已知区域。

#### Scenario: 混合写法
- **WHEN** 用户运行 `yeto shape --regions us-east-1,nebius:eu-north1,verda:FIN-03`
- **THEN** AWS 候选限于 us-east-1，Nebius 限于 eu-north1，Verda 限于 FIN-03，其余已启用的云不受限制

#### Scenario: 旧写法不变
- **WHEN** 用户运行 `yeto shape --regions us-east-1,us-east-2`
- **THEN** 行为与本次变更前完全一致：只过滤 AWS，非 AWS 云全区域参与

#### Scenario: 未知区域
- **WHEN** 用户运行 `yeto shape --regions nebius:eu-central9`
- **THEN** 命令报错，错误信息列出 Nebius 目录中当前存在的区域

### Requirement: 每朵云的区域含义
对 AWS 和 Nebius，区域 SHALL 指机房，价格、配额或容量按区域区分。对 Verda，区域 SHALL 指 location 代码。对 Modal，区域 SHALL 表示"要求 Modal 把容器固定在该地理范围"，规划器 MUST 对该云的价格乘以对应的区域系数（宽区域和窄区域两档），未指定时 MUST 不固定区域也不加价。

#### Scenario: Modal 区域加价
- **WHEN** 用户运行 `yeto shape --regions modal:us --clouds modal --budget 50`
- **THEN** 候选中 Modal 的每小时价按宽区域系数放大，输出注明系数来源

#### Scenario: Modal 默认不固定区域
- **WHEN** 用户未在 `--regions` 中提及 modal
- **THEN** Modal 候选按无区域基础价计算，生成的启动命令不含区域参数

### Requirement: 规划结果保留云前缀
`yeto shape` 输出的启动命令中每个岛的 `--gpu` 条目 SHALL 使用 `cloud:...@region` 形式，其中 region 为该云的原生区域或位置标识，使 `--apply` 与手写 `yeto launch` 等价。

#### Scenario: 应用规划
- **WHEN** 规划选出一个 Verda FIN-03 的岛并以 `--apply` 启动
- **THEN** 传给启动器的条目为 `verda:8xh100@FIN-03`，启动器把它解释为 Verda 的 FIN-03 位置
