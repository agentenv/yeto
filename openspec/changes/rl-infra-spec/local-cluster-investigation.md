# 本地 / 自有集群部署调查（LOCAL-CLUSTER，2026-09-30）

基线：`local-cluster` 分支，基于 integ-decl ef2d6b0。SkyPilot 版本：`/home/michael/work/gpu-head/venv/bin/sky --version` = **0.13.0**。
本调查只读源码和 `--help`，没有运行 `sky launch`、`sky ssh up`、`sky local up`，也没有部署 k8s，没有启动任何付费资源，没有改 `~/.sky`。

## 0. 结论先行

1. **head（syncer + FleetController）已经能在自有机器上常驻运行，不需要新写"yeto 服务端"。** `cli.cmd_head` 调用 `launcher.run(local_syncer=LocalSyncer(...))`，本身与机器无关。前提是设置 `SYNCER_PUBLIC_IP`，并且岛能直接连到该机器的 29400 端口。旧会话在 7.1 Modal 运行里已用 `infra-a-gpu/infra-a-r3b/run_local_head.py` 在本机跑通过这条路径。缺口在于这不是正式 CLI 模式：`--controller` 只接受 `head` 和 `local`，其中 `local` 仍会另起一台 sky syncer VM。
2. **learner 岛目前不能在"裸"自有机器上由 launcher 拉起**，有两条可行路线：
   - **ssh_harness 路线**：`yeto/rl/ssh_harness.py` 已经是一套完整的"直连 SSH + docker"Miles RL 部署器（prepare/deploy/start/status/kill/restart/stop/collect/verify），syncer 放在第一台主机或指定主机上。但它**硬钉 H200**（`_validate_plan` 拒绝其他型号），网卡默认是 `eno3`，并且强制 `NCCL_IB_DISABLE=1`。它和 launcher 是两条独立代码路径，不走 FleetController 和 sky 岛脚本。
   - **SkyPilot 路线**：SkyPilot 0.13 对自有机器的支持全部基于 Kubernetes。`sky ssh up` 会用 SSH 在你的机器上装 k3s 和 NVIDIA GPU Operator，然后把它当作 `ssh/<pool>` 云；已有的 k8s 集群则用 `kubernetes/<context>`。yeto 的 `--gpu ssh:1x8xH100@pool` 已经能解析，并会原样传成 `infra=ssh/pool`，但在 k8s 上有几个已知的不兼容点（见 §2.3），都没有经过 GPU 验证。
3. **推荐（供用户决定）**：分两步走。
   - **第一步，选方案 A′（约 3–5 人日）**：把"本机 head"做成正式模式；把 ssh_harness 的 H200 钉死和网卡参数改为可配置；只用一台自有 GPU 机器承载岛，把 syncer 和 head 放在同一台或者本机。这最快能进入本地 GPU 验收，也不用在用户机器上装 k3s。
   - **第二步，按需选 B（sky ssh node pool，约 4–6 人日加环境搭建）**：只有当需要"launcher/FleetController 原路径"验收（例如 rl-infra-spec 3.x 池增减、重启岛等必须走 FleetController 的项）或多机弹性时，再上 B。
   - **不建议现在做 D（常驻 yeto 服务端，10–15 人日）。**

## 1. yeto 现状

### 1.1 launcher 的 provider 抽象
- 没有独立的 provider 层。`--gpu` 解析为 `ClusterSpec(cloud, num_nodes, gpus_per_node, gpu, region)`（`yeto/gpu_spec.py`，cloud 可以是任意小写词）。
- 岛（`make_miles_island_task`，launcher.py:2141 附近）统一通过 `sky.Resources(infra=f"{cloud}/{region}", accelerators, image_id=rl_image, use_spot, disk_size, network_tier="best"(多节点))` 交给 sky。**只有 `modal` 分支走自己的 SDK**（`build_modal_island_config`）。所以任何 sky 支持的 infra，包括 `ssh`、`kubernetes`、`slurm`，在 yeto 这一侧"字面上"都能写出来。
- 控制面通过 `SkySDKOps`（job_status/queue/cluster_up/down）操作岛，测试里可以注入替身。这是现有唯一的"provider 接口"；Modal 用的是另一套 modal_ops。
- `yeto/shape/providers.py` 是 **shape/报价信号**（AWS 配额、RunPod 库存、Nebius 可用性），不是部署抽象，和本地集群无关。
- 凭据：`head_cloud_credentials` 只认识 aws/gcp/modal/nebius/runpod/verda。**遇到未知 cloud 会跳过（`continue`）而不是报错**，所以远端 head 不会自动带上 `~/.kube/config` 或 `~/.sky/ssh_node_pools.yaml` 和 SSH 私钥。**head 在本机时这不是问题。**

### 1.2 head 模式与 syncer
- `--controller head`（默认）：`cli.cmd_launch_head` 用 sky 起一台小 VM，`ports=[29400]`，把 syncer 二进制和 workdir 挂上去，然后在 VM 上执行 `yeto _head <args_json>`。在 VM 上，`cmd_head` 构造 `LocalSyncer`（launcher.py:3064，syncer 作为子进程运行，死了会自动重启并 `--resume`），再调用 `launcher.run(local_syncer=...)`，syncer 地址取自 `SYNCER_PUBLIC_IP:29400`。
- `--controller local`：本机跑控制器，但 syncer 仍由 `make_syncer_task` 另起一台 sky VM（带 `ports`）。
- 现状限制（来自记忆，并已与源码核对）：
  - Verda：sky 0.13 的 Verda adapter 不声明 `OPEN_PORTS`，head/syncer 任务一带 `ports` 就失败。
  - Modal：serverless 模式没有入站端口，不能作为 head。
  - 能用的 head：Nebius、AWS，或者**有公网或内网可达 IP 的本机**（`run_local_head.py` 的方式）。
- 结论：head 的代码是"机器无关"的，唯一的硬前提是"岛 → head:29400 可达"。对自有局域网集群，这比云上更容易满足。

### 1.3 ssh_harness（`yeto/rl/ssh_harness.py`，4955 行）
- 用于"直连 SSH 的 Miles RL 验收"。子命令：`prepare`（生成并校验 plan：`--host a,b --host c,d` 表示每组一个岛，`--gpus-per-node`，`--syncer-address`，`--network-interface`，`--ssh-option`）、`deploy`（rsync 源码和 syncer，本地做 attest，拉镜像并校验 digest）、`start`、`status`、`kill-learner`、`restart-learner`、`kill-syncer`、`stop`、`collect`（取证据与磁带）、`verify`（校验并导出 checkpoint）。
- 岛在每台主机上以 `docker run --gpus ... --env NCCL_IB_DISABLE=1 --env NCCL_SOCKET_IFNAME=<iface>` 运行，Ray 在 6379 端口手动组网。
- 限制：
  - 只允许 H200（ssh_harness.py:1069–1079、1591、906）。
  - 所有岛的节点数必须相同。
  - 网卡默认 `eno3`。
  - 关闭 IB，NCCL 走 TCP。
  - 需要 docker、nvidia-container-toolkit，以及能拉取私有 ghcr 镜像的凭据。
  - 它自己编排进程，**不经过 FleetController**，所以 rl-infra-spec 里依赖控制器（重启岛、池增减）的行为在这条路径上是"harness 自己的实现"，不是 launcher 的实现。
- 结论：**岛的部署能力可以直接复用**，是把自有机器接进来最短的路径。

### 1.4 已可在自有机器上跑 / 硬依赖云 一览
| 组件 | 自有机器 | 说明 |
|---|---|---|
| syncer 二进制 | 可以 | Rust 二进制，本机构建（x86 Linux）或在远端构建 |
| head 控制器（FleetController + LocalSyncer） | 可以（非正式） | 需要 `SYNCER_PUBLIC_IP` 和 `run_local_head.py` 这类包装；CLI 没有正式入口 |
| learner 岛（launcher 路径） | 不行（除非走 sky k8s/ssh 池） | 除 modal 外全部通过 `sky.launch` |
| learner 岛（ssh_harness 路径） | 可以（仅 H200） | 需要 docker 和 nvidia runtime |
| 岛的 spot checkpoint 存储 | 不适用 | 只在 `--spot` 时用 `sky.Storage`（S3 等），自有机器不需要 |
| shape/报价 | 不适用 | 自有卡没有价格 |

## 2. SkyPilot 0.13 实际能力（源码实证）

源码路径前缀：`/home/michael/work/gpu-head/venv/lib/python3.12/site-packages/sky/`。

### 2.1 自有机器的三种接入方式
1. **SSH Node Pools**（`sky ssh up [--infra NAME] [-f FILE]`，默认读 `~/.sky/ssh_node_pools.yaml`）：
   - `clouds/ssh.py`：`class SSH(kubernetes.Kubernetes)`，文档注释写明 "SSH Node Pools … use Kubernetes to manage the SSH clusters"，context 名为 `ssh-<pool>`。
   - `ssh_node_pools/deploy/deploy.py`：在第一台机器上用 `curl -sfL https://get.k3s.io | … sudo -E -A sh -` 装 k3s server，其余机器 join；然后用 helm 安装 `nvidia/gpu-operator`（deploy.py:926–929）。GPU 型号通过 `nvidia-smi --query-gpu=gpu_name` 检测（deploy/utils.py:143）。
   - 前提：
     - 每台机器可以 SSH 登录，并且有 sudo（支持用 `password` 字段走 askpass）。
     - 能访问外网（get.k3s.io、helm、NGC）。
     - 检查 `AllowTcpForwarding`（deploy.py:451）。
   - **它不是"直接在裸机上跑 Ray"，而是先把机器变成 k3s 集群。** `sky ssh down` 会卸载 k3s。
2. **Kubernetes**（`infra=kubernetes/<context>`）：直接使用已有的 kubeconfig。
3. **Slurm**（`clouds/slurm.py`）：不支持 OPEN_PORTS；docker 镜像需要 Pyxis；`HOST_CONTROLLERS` 标为"未充分测试"。**不适合承载 head/syncer。**
4. `sky local up/down`：只创建本地 kind 集群，面向开发机，不适合 GPU 验收。
5. 0.13 **没有"接管已存在的裸机或 VM 集群"的 existing-cluster 模式**（旧版的 `sky local` 本地集群已被 ssh node pool 取代）。

### 2.2 能力矩阵（kubernetes 与 ssh 共用，`clouds/kubernetes.py:213–292`）
| 能力 | k8s / ssh 池 | 源码依据 |
|---|---|---|
| OPEN_PORTS | **支持**（不在 unsupported 列表中） | 端口模式 `loadbalancer`（默认）/ `ingress` / `podip`（`utils/kubernetes_enums.py:19`，`provision/kubernetes/config.py:67`）。k3s 默认自带 ServiceLB（deploy.py 没有 `--disable servicelb`），因此 LoadBalancer 会映射到节点的主机端口，在局域网内可达 |
| 常驻（head 长期在线） | 支持 | pod 一直存在，直到 `sky down` |
| STOP / AUTOSTOP | **不支持** | kubernetes.py:261–264。回收只能靠 `sky down` 或 yeto 自己的 watchdog。自有卡不花钱，所以影响不大 |
| SPOT | 仅当集群有 spot 标签 | kubernetes.py:275–282 |
| docker 镜像 | 支持（作为 pod 镜像），私有仓库通过 imagePullSecrets | `provision/kubernetes/utils.py:3880–3910` |
| 多节点 | 支持（多个 pod） | 默认走 pod 网络；只有检测到高性能网络的集群（如 Nebius）才支持 `network_tier=best`，否则算 unsupported feature |
| GPU 检测 | 通过节点标签和 GPU Operator；ssh 池在部署时用 nvidia-smi 获取型号 | |
| /dev/shm | 模板挂载 dshm | `templates/kubernetes-ray.yml.j2:1426` |
| 主机网络 | 有 hostNetwork 探测相关代码（`host_network_probe.py`） | 是否默认开启未核实 |

### 2.3 yeto 在 sky k8s/ssh 池上的已知不兼容点（推断，未经 GPU 验证）
1. 多节点岛会设 `network_tier="best"`，在普通 k3s 上这是 unsupported，会导致优化器找不到资源。需要对 ssh/kubernetes 去掉它。
2. Miles 岛脚本自己执行 `ray start --head` 并使用 6379 端口，和 sky 在 pod 里运行的 Ray 共存。这是 fix-sky-island-ray-stop 相关的问题域，需要在 pod 内核实端口和 `ray stop` 的行为。
3. rl 镜像作为 pod 镜像时，sky 需要在镜像里装自己的运行时，要求有 bash/python/sudo 等。需要确认 `yeto-miles-ports` 镜像可以用。
4. `_docker_login_config` 在 k8s 上是否被映射为 imagePullSecret 未核实。备选做法是提前在集群里建好 secret，或者把镜像放到本地 registry。
5. 远端 head 的凭据表缺少 ssh/kubernetes（见 §1.1）。head 在本机则没有这个问题。
6. 多节点 NCCL 默认走 pod 网络（flannel VXLAN），带宽和延迟都比裸机差。单机多卡的岛不受影响。

## 3. 是否需要"yeto 服务端"：方案对比

| | A′ 本机 head 正式化 + ssh_harness 泛化 | B sky ssh node pool | C 已有 Kubernetes | D 常驻 yeto 服务端 |
|---|---|---|---|---|
| 做法 | 新增 `--controller here`（本机跑 `cmd_head` 并启动 LocalSyncer，`SYNCER_PUBLIC_IP` 由参数或探测得到）；ssh_harness 的 accelerator 改为参数，放开 H200 钉死但记录型号并在运行前断言，网卡改为必填或自动探测 | 用户机器 `sky ssh up`；yeto 对 `ssh`/`kubernetes` 岛去掉 `network_tier`，处理镜像拉取 secret，核实 pod 内 Ray；head 用 A′ 的本机模式，或者放在池内（`--syncer-region ssh/<pool>`） | 同 B，跳过装 k3s | 常驻守护进程托管 FleetController、syncer、run 注册表，对外提供 API（submit/status/stop），岛通过 SSH 或 sky 拉起 |
| 改动文件 | `yeto/cli.py`（controller 选项、入口）、`yeto/launcher.py`（`check_cloud_prerequisites`、`fleet_clouds`）、`yeto/rl/ssh_harness.py`（accelerator/NIC）、`tests/test_head_mode.py`、`tests/test_rl_ssh_harness.py`、docs | `yeto/launcher.py`（`make_miles_island_task` 资源分支、凭据表）、`yeto/cli.py`（head 凭据）、测试、docs/CLOUDS.md | 同 B | 新模块（服务端、API、持久化）、cli 客户端、runs 注册表迁移、鉴权 |
| 工作量 | 3–5 人日（CPU 测试齐全） | yeto 侧 2–3 人日，加上环境搭建和 GPU 调通 2–3 人日 | yeto 侧 2–3 人日 | 10–15 人日 |
| 风险 | 低。ssh_harness 与 launcher 是两条路径，验收结论需要注明"harness 路径"；FleetController 相关行为（3.x 池增减、重启）不经过 launcher 原路径 | 中到高。要在用户机器上装 k3s 和 GPU Operator（需要 sudo、外网，可能影响机器上原有的 docker/driver）；Ray 共存、镜像运行时、pod 网络性能都是未知项 | 中。取决于用户集群策略（特权、GPU operator、LB） | 高。等于重写 head 模式，并且现有 head 模式已覆盖其核心功能 |
| 对本地 GPU 验收的适配度 | **高**：单机多卡当天就能跑，逐位比较不受 k8s 干扰 | 中：能验证 launcher 原路径（最接近生产），但环境成本高 | 视用户是否已有 k8s 而定 | 低（短期） |
| 推荐 | **首选（第一步）** | 第二步，按需 | 用户已有 k8s 时替代 B | 暂不做 |

**为什么不需要 D**：head 模式本身已经是"单进程常驻控制器 + 子进程 syncer + 自动重启"。自有集群缺的只是"让它在本机正式启动"（A′ 的 1–2 人日部分）和"岛如何上机"（ssh_harness 或 sky 池）。D 只有在需要多用户、多 run 排队或 web API 时才值得做，这超出当前 rl-infra-spec 的范围。

## 4. 本地验证的最小前提 & 待用户决定

### 4.1 最小硬件 / 网络
- 岛：至少 1 台 NVIDIA GPU 机器，需要：
  - Linux x86_64，驱动加 nvidia-container-toolkit，docker；
  - 能访问 ghcr 私有镜像（`ghcr.io/michaellchung/yeto-miles-ports@sha256:c6f5…`，需要 PAT），或者可以离线导入镜像；
  - 磁盘够放模型和 `/data/yeto-rl`（JIT 缓存、tms 备份）；
  - GPU 数量按实验要求（多数验收用 1 岛 × 8 卡；Qwen 小模型的冒烟可以少于 8 卡）。
- 需要两个岛的实验（跨岛同步、1.x 以及 3.x 池增减）：2 台机器，或 1 台机器按卡切分成两岛。**ssh_harness 目前"一台主机只属于一个岛"，按卡切分需要额外改动（约 1 人日）。**
- head / syncer：任意一台 Linux x86 机器（可以是 GPU 机或本机），岛到它的 TCP 29400 端口可达；多节点岛之间 Ray 6379 端口和 NCCL 端口互通。
- 逐位比较：自有卡的型号与 Modal `H100!` 不同，**不能和既有 Modal 基线做逐位比较**，只能在同一批自有卡上自比（参见记忆"Modal H100→H200 upgrade"）。gpu-plan 中的逐位判据需要按"同机型自比"重写并事先提交。

### 4.2 需要用户决定
1. 自有卡的型号和数量、机器台数、是否有 IB/RoCE、是否允许装 k3s（B）或已有 k8s（C）。
2. 选 A′ 先行，还是直接选 B/C。
3. 是否同意放开 ssh_harness 的 H200 钉死（改为"声明型号 + 运行前断言"）。
4. 本地验收是否接受"ssh_harness 路径"的结论作为 rl-infra-spec 验收证据，还是要求必须走 launcher/FleetController 原路径（后者需要 B/C，或者给 launcher 做一个 SSH 岛 ops，约再加 4–6 人日）。
5. 逐位类验收（4.3、4.6）改为同机型自比，需要批准并提前提交判据。
6. 私有镜像拉取方式：在用户机器上放 PAT，还是镜像公开（公开需另批）。

## 5. 本次未做 / 未核实
- 没有写原型代码（报告优先）。`parse_gpu_spec("ssh:1x8xH100@mypool")` 已经本地确认能解析为 cloud=ssh、region=mypool、accelerators=H100:8。
- §2.3 各项都是源码推断，没有实际在 k3s 上运行过（按约束不得启动集群）。
- 没有查阅 SkyPilot 在线文档，结论以 0.13.0 安装源码为准。
