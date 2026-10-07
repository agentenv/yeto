# Spec Delta

## Purpose

规定 `local` 提供者（自有 / 本地 GPU 集群）在 yeto RL 中的声明、节点清单、预检、head/syncer/岛放置规则、镜像与存储约定、失败语义与可观测性。

## ADDED Requirements

### Requirement: local 提供者可声明且不改变其他提供者
系统 SHALL 接受 `--gpu local:[NxM]x<GPU>[@<pool>]`，把 `cloud=local`、`region=<pool>`（缺省 `default`）解析为 `ClusterSpec`，并把岛的生命周期交给 `local` 提供者而不是 SkyPilot 或 Modal。非 `local` 条目生成的 sky Task、Modal 配置与 argv MUST 与本 change 之前逐字节一致。

#### Scenario: 解析 local 条目
- **WHEN** 用户传入 `--gpu local:2x8xh200@rack1`
- **THEN** 得到 cloud=local、region=rack1、num_nodes=2、gpus_per_node=8、gpu=H200

#### Scenario: 其他云不受影响
- **WHEN** 用户传入不含 `local` 的 `--gpu`
- **THEN** 现有 argv 快照测试不改即通过

### Requirement: 节点清单是 local 提供者的唯一事实来源
系统 SHALL 从 `--local-nodes <yaml>`（默认 `~/.yeto/local_nodes.yaml`）读取池与节点（ssh_user、ssh_key、network_interface、nccl_ib_disable、model_store、data_dir、image_cache、price_per_gpu_hour、nodes[host,gpus,gpu,head_reach_ip]），并在 GPU 进程启动前拒绝格式错误、缺少必填字段（`network_interface`、`model_store`、`data_dir`、每节点 `gpu`/`gpus`）或引用了不存在池的 `--gpu`。

#### Scenario: 引用不存在的池
- **WHEN** `--gpu local:1x8xh200@rack9` 且清单无 `rack9`
- **THEN** 预检失败，错误列出清单中存在的池名

#### Scenario: 缺少网卡
- **WHEN** 池未声明 `network_interface`
- **THEN** 预检失败，错误指明该字段用于 `NCCL_SOCKET_IFNAME`，不得使用默认值

### Requirement: 节点分配确定且容量不足即失败
系统 SHALL 按 `--gpu` 声明顺序从池中确定性地分配节点：`num_nodes>1` 的岛只取整机；`gpus_per_node` 小于主机卡数时允许同一主机切分给多个岛并以互不重叠的 GPU 子集隔离。容量不足 MUST 在预检阶段失败，不启动任何容器。

#### Scenario: 两岛整机分配
- **WHEN** 池有两台 8 卡机，`--gpu local:1x8xh200@p,local:1x8xh200@p`
- **THEN** 两岛各占一台，分配结果在两次运行间相同

#### Scenario: 单机切分
- **WHEN** 池有一台 8 卡机，`--gpu local:1x4xh200@p,local:1x4xh200@p`
- **THEN** 两岛分别得到 GPU 0–3 与 4–7，Ray 端口互不相同

#### Scenario: 容量不足
- **WHEN** 池只有一台 8 卡机，`--gpu local:2x8xh200@p`
- **THEN** 预检失败，错误说明需要 2 台整机、池中只有 1 台

### Requirement: 启动前预检覆盖机器、镜像、存储与网络
系统 SHALL 在任何 GPU 进程启动前对每个被分配节点检查：SSH 可达；`nvidia-smi` 报告的 GPU 名与数量与 `--gpu` 及清单一致；docker 与 nvidia-container-toolkit 可用；镜像 digest 可用（本地已有、离线归档可加载并校验、或可登录拉取三者之一）；`model_store` 含所需模型快照且 `data_dir` 可写；节点到 syncer 地址的 29400（critic 算法还含 29401）TCP 可达；多节点岛的节点间 Ray 端口可达；无同名 `yeto.run` 标签的容器占用。任一失败 MUST 使整体失败并按"节点 → 原因"列出全部失败项。

#### Scenario: GPU 型号不符
- **WHEN** 清单声明 H200 而 `nvidia-smi` 报告 H100
- **THEN** 预检失败并给出两者名称

#### Scenario: 镜像三路皆不可用
- **WHEN** 节点无该 digest 镜像、`image_cache` 无归档、`docker login ghcr.io` 失败
- **THEN** 预检失败，错误分别列出三条路径的失败原因

#### Scenario: 多节点同时失败
- **WHEN** 两台节点分别缺 docker 与模型快照
- **THEN** 一次预检同时报告两台节点的原因

### Requirement: 控制器可放在提交机本机
系统 SHALL 提供 `--controller here`，在提交机以与 `cmd_head` 相同的代码路径运行 FleetController 与 LocalSyncer 子进程（含自动重启与 `--resume`），不启动任何云 VM。syncer 对岛可见的地址 MUST 来自 `--syncer-public-addr` 或由节点清单 `head_reach_ip` 所在网段推导；无法确定时 MUST 失败而不是猜测。

#### Scenario: 全 local 岛的本机 head
- **WHEN** 所有岛为 `local` 且未给 `--syncer-public-addr`
- **THEN** 系统从本机与 `head_reach_ip` 同网段的接口推导地址，预检用该地址探测 29400 可达

#### Scenario: 混合 Modal 岛缺公网地址
- **WHEN** `--controller here` 且 `--gpu` 含 `modal:` 条目但未给 `--syncer-public-addr`
- **THEN** 预检失败，错误说明 Modal 岛需从公网到达 syncer 并指向 `--controller head` 方案

### Requirement: head 放置能力表统一
系统 SHALL 维护一张 head 放置能力表：aws、nebius、here、local 池内节点可承载 head；verda、modal 不可。选择了不可承载 head 的放置 MUST 在提交前失败并给出可用选项。

#### Scenario: Verda 作 head
- **WHEN** `--syncer-region verda/<region>`
- **THEN** 提交前失败，错误列出可承载 head 的放置

### Requirement: 容器内路径约定与云上一致
`local` 岛容器 SHALL 以只读方式把 `model_store` 挂到 `/mnt/yeto-models`、以读写方式把 `data_dir` 挂到 `/data/yeto-rl`，tape 写入容器内 `~/yeto-output/`；岛脚本 MUST NOT 因提供者不同而改变这些路径。

#### Scenario: 挂载参数
- **WHEN** 生成 `local` 岛的 docker run 命令
- **THEN** 命令含 `-v <model_store>:/mnt/yeto-models:ro` 与 `-v <data_dir>:/data/yeto-rl`

### Requirement: 岛生命周期经统一 IslandOps 协议
`local` 提供者 SHALL 实现与 sky 岛相同动词的 `IslandOps`（up、job_status、job_alive、cluster_up、rl_strict_failure、down、pull_tapes、stream_logs），FleetController 对 `local` 岛 MUST 使用与 sky 岛相同的决策路径；测试 MUST 能注入假实现。

#### Scenario: 假实现驱动控制器
- **WHEN** 测试注入 `FakeLocalOps` 并让一个岛 `job_alive` 返回 False
- **THEN** FleetController 走与 sky 岛相同的"岛失败"路径

### Requirement: 失败语义
启动阶段任一容器失败时系统 SHALL 关闭同 run 已启动的 `local` 容器（除非 `--keep-abandoned`）；运行中节点 SSH 连续失联达到阈值时 SHALL 把该岛报告为不存活并尽力 `docker rm -f`；`--keep` 时 SHALL 保留容器并打印每台主机的查看命令。系统 MUST NOT 把无法确认的状态报告为成功。

#### Scenario: 第二岛启动失败
- **WHEN** 第一岛容器已启动、第二岛 `docker run` 失败且未给 `--keep-abandoned`
- **THEN** 第一岛容器被关闭，运行以失败结束

#### Scenario: 节点失联
- **WHEN** 某岛节点 SSH 连续超时达到阈值
- **THEN** `job_alive` 为 False，账本记录"失联"而非"完成"

### Requirement: 可观测性与成本记账
系统 SHALL 在岛结束、失败或被 kill 后把 `~/yeto-output/*.jsonl` 拉回 `runs/<run>/islands/<id>/`；账本 SHALL 记录 provider=local、pool、hosts、观测到的 GPU 名、镜像来源（inspect/load/pull）、`price_per_gpu_hour`（默认 0）、按 up→down 时长计算的费用与 `cost_source=local-declared`。`local` 单价 MUST NOT 进入 shape/报价选型。

#### Scenario: 默认零成本
- **WHEN** 清单未声明 `price_per_gpu_hour`
- **THEN** 账本费用为 0 且 `cost_source=local-declared`

#### Scenario: tape 回收
- **WHEN** 岛被 kill
- **THEN** `runs/<run>/islands/<id>/` 含该岛的 tape 文件
