## Purpose

规划器（`yeto shape`）按云取得三类信号的统一契约：凭据是否存在、有哪些机型和价格、某个机型现在能不能开出来。本能力覆盖 Nebius、Verda、Modal 三家的新接入以及 RunPod 向同一契约的迁移；AWS 因为多了配额和用量维度，保留独立处理。

## ADDED Requirements

### Requirement: 每朵非 AWS 云提供同一种信号契约
系统 SHALL 为每朵已接入的非 AWS 云提供三个可独立调用的信号：凭据检测（真/假）、机型目录（每条含云名、机型标识、GPU 名、每机 GPU 数、区域或位置、按需价、spot 价）、可用量伪分（对每个 (GPU 名, 每机 GPU 数) 询问返回 0 到 10 的整数，或 None 表示信号不可得）。伪分 0 MUST 表示"已测得无货"，None MUST 表示"没查到"，二者不得混用。

#### Scenario: 无货和查不到区分开
- **WHEN** 某云的可用量接口明确返回该机型无货
- **THEN** 伪分为 0，规划器把该形状标为不可启动，不产生"信号缺失"告警

#### Scenario: 接口失败退化而不中断
- **WHEN** 某云的可用量接口超时、限流或返回错误
- **THEN** 伪分为 None，规划器按现有的"假定最佳分并告警 / 严格模式拒绝"策略处理，其余云的规划继续

#### Scenario: 缺凭据的云不参与
- **WHEN** 用户未指定 `--clouds`，且某云的凭据检测为假
- **THEN** 该云不出现在候选中，也不发起任何网络请求

### Requirement: `--clouds` 默认值包含所有检测到凭据的云
`--clouds` 未指定时，系统 SHALL 把候选云集合定为 `aws` 加上每一朵凭据检测为真的非 AWS 云。指定了 `--clouds` 时 SHALL 只用指定的云，并对其中凭据缺失的云报错退出。

#### Scenario: 只有 Nebius 凭据
- **WHEN** 本机只有 Nebius 凭据，没有 AWS 凭据，用户运行 `yeto shape --clouds nebius --budget 40`
- **THEN** 规划成功，不要求 AWS 凭据，结果只含 Nebius 岛

#### Scenario: 指定了没有凭据的云
- **WHEN** 用户运行 `yeto shape --clouds aws,verda` 而本机没有 Verda 凭据
- **THEN** 命令报错，错误信息指出 Verda 凭据的期望位置，不发起规划

### Requirement: Nebius 信号
系统 SHALL 从本机 Nebius 凭据文件检测凭据；机型目录 SHALL 来自 SkyPilot 的 Nebius 目录并保留其 spot 价；可用量伪分 SHALL 来自 Nebius 的容量查询接口，把 preemptible 容量换算为伪分（无容量为 0）。当请求的是按需实例时 SHALL 用 on-demand 容量换算。伪分缓存时间 MUST 不长于 15 分钟。

#### Scenario: 区域感知
- **WHEN** 规划器询问 Nebius "8 卡 H100 在 eu-north1 和 us-central1 各多少分"
- **THEN** 每个区域独立返回伪分，目录里没有该机型的区域不产生询问

#### Scenario: 动态 spot 价格覆盖
- **WHEN** Nebius 返回的实时 preemptible 价格与 SkyPilot 目录中的 spot 价不同
- **THEN** 规划器使用实时价格，并在输出中注明该价格来自实时接口

### Requirement: Verda 信号
系统 SHALL 从 `~/.verda/config.json` 或环境变量检测 Verda 凭据；机型目录 SHALL 来自 Verda 的公开机型接口而不是 SkyPilot 目录，包含全部 GPU 机型和 spot 价；可用量伪分 SHALL 来自 Verda 的可用性接口，可用为 9，不可用为 0。Verda 目录中的每条记录 MUST 带 location（例如 FIN-03），location 未知的机型 MUST 对每个已知 location 分别询问可用性后再入候选。

#### Scenario: 8 卡机型进入候选
- **WHEN** 用户运行 `yeto shape --clouds verda --gpus h100 --budget 30`
- **THEN** 候选包含 Verda 的 8 卡 H100 机型，价格与 Verda 公开接口一致

#### Scenario: 多节点被排除
- **WHEN** 模型权重需要的显存超过 Verda 单机最大机型
- **THEN** 规划器不为 Verda 生成多节点岛，并在告警中说明 Verda 本版只支持单节点

### Requirement: Modal 信号
系统 SHALL 从 Modal token 检测凭据；机型目录 SHALL 是一张随代码维护的静态价格表，每条按 GPU 名和每容器 GPU 数给出每小时价，价格 MUST 包含 GPU、CPU 和内存三项；可用量伪分 SHALL 恒为一个固定的"可用但可能排队"值，不发起网络请求。多容器（多节点）形状 MUST 只允许每容器 GPU 数为该 GPU 的整机数。

#### Scenario: 非整机多节点被拒
- **WHEN** 规划器考虑 Modal 上 2 个容器各 4 卡 H100 的岛
- **THEN** 该形状被排除，理由为 Modal 多节点要求整机 GPU 数

#### Scenario: 静态价格表过期提示
- **WHEN** 静态价格表的记录日期距今超过 90 天
- **THEN** 规划输出带一条告警，提示 Modal 价格可能过期

### Requirement: RunPod 迁移到同一契约
RunPod 现有的凭据检测和库存伪分 SHALL 迁移到本契约下，行为与迁移前一致；RunPod 的 H200 机型缺失 SHALL 在规划输出中以告警形式说明，而不是静默不出现。

#### Scenario: 行为不变
- **WHEN** 用户以迁移前相同的参数运行 `yeto shape --clouds aws,runpod`
- **THEN** 候选集合、伪分和最终方案与迁移前一致
