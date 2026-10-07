# Design

## Context

动机见 proposal.md。以下只列决定实现方式的现状（已对照 integ-decl e8d387ac 工作树与 `/home/michael/work/gpu-head/venv` 中 SkyPilot 0.13.0 源码核实；行号以该工作树为准）。

**yeto 侧现状**
- `--gpu` 语法（`yeto/gpu_spec.py:41-48`）：`^(?P<cloud>[a-z]+):(?:(?P<nodes>\d+)x)?(?P<count>\d+)x(?P<gpu>…)(?:@(?P<region>…))?`，cloud 任意小写词，`region` 原样透传；`_GPU_CANONICAL` 已含 h100/h200/l40s/rtx-pro-6000 等。`ClusterSpec(cloud, region, num_nodes, gpus_per_node, gpu)`。所以 `local:1x8xh200@rack1` 今天就能解析为 cloud=local、region=rack1。
- 没有独立 provider 层。岛统一由 `make_miles_island_task`（launcher.py:3300）构造 sky Task，资源 `sky.Resources(infra=f"{cloud}/{region}", accelerators, image_id, …)`；多节点时 `multinode_network_tier`（:274）给 `network_tier="best"`（:3746-3749、:4168-4170）。**唯一的非 sky 分支是 `spec.cloud == "modal"`**（:295、:1223、:3605、:4171、:4466、:6311、:6322），走 `build_modal_island_config`（:4472）→ `ModalIslandConfig`（modal_runner.py:177）→ 容器内 `island_main`（:531）。
- 控制面"ops"接口：`SkySDKOps`（:4726；`job_status/job_alive/cluster_up/rl_strict_failure`）与 modal_ops（`pull_tape/stream_logs`，:4586-4660、:5089）两套；测试里可注入替身。
- 预检：`check_cloud_prerequisites(specs, project_ids, args, modal_ok)`（:1192）只有 nebius 与 modal 分支，"在任何云花费前失败"。凭据：`head_cloud_credentials`（:1147）对未知 cloud `continue`；`head_cloud`（:1132）从 `--syncer-region cloud/region` 取 head 所在云，默认 aws；`fleet_clouds`（:1139）只在 `--controller head` 时并入 head 云。
- 控制器放置：`--controller {head,local}`（cli.py:1101）。`head`：`cmd_launch_head`（cli.py:1908）用 sky 起带 `ports=[29400]` 的小 VM，注入 `SYNCER_PUBLIC_IP`（:1965），VM 上 `cmd_head`（:2022）构造 `LocalSyncer` 并 `launcher.run(local_syncer=…)`；syncer 地址 `SYNCER_PUBLIC_IP:29400`（launcher.py:6226）。`local`：本机跑控制器，但 syncer 仍另起 sky VM。Modal 岛连私网 syncer 需 `--syncer-public-addr HOST:PORT`（cli.py:1095）。
- 端口：actor syncer 29400、critic syncer 29401（`CRITIC_SYNCER_PORT`，launcher.py:46、:806）；Ray 6379。
- 存储：`MODEL_STORE_MOUNT = "/mnt/yeto-models"`（launcher.py:3176），Nebius FS 与 Modal model Volume（modal_runner.py:222-223）都挂在此路径；岛数据目录 `/data/yeto-rl`（JIT 缓存、tms 备份）。
- tape：岛侧 `~/yeto-output/yeto-tape.jsonl`、`yeto-critic-tape.jsonl`（launcher.py:524-527）；sky 岛结束后 `rsync cluster:yeto-output/rl-island-*.jsonl`（:4689）；Modal 用 `TapeSync` 镜像到 Volume 再 `pull_modal_tapes`。
- 成本：`yeto/shape/providers.py` 是报价/库存信号（`modal_node_price_per_hour` :1411 等），不是部署抽象；launcher 没有通用的每小时单价表。
- `yeto/rl/ssh_harness.py`（4955 行）：直连 SSH + docker 的 Miles RL 部署器（prepare/deploy/start/status/kill-learner/restart-learner/kill-syncer/stop/collect/verify）。钉死 H200（:910、:1073-1083、:1595），网卡默认 `eno3`（:1097），`NCCL_IB_DISABLE=1`，一台主机只属一个岛，所有岛节点数相同；不经 FleetController。
- 远端 head 先例：`s1-runs/s13-g3-remote.sh`（在 S13 worktree，本基线工作树中不存在，内容按记忆：controller+syncer 放 Nebius 无卡 VM，岛在 Modal）；本机 head 先例：`infra-a-gpu/infra-a-r3b/run_local_head.py`（非正式 CLI）。

**SkyPilot 0.13.0 对自有机器的支持（证据）**
- `sky --version` → `skypilot, version 0.13.0`；`sky ssh --help` 只有 `up`/`down` 两个子命令（"Commands for managing SSH Node Pools"）。
- `sky/clouds/ssh.py:28-34`：`class SSH(kubernetes.Kubernetes)`，"SSH Node Pools … use Kubernetes to manage the SSH clusters"，继承全部 Kubernetes unsupported features（:52）；清单文件 `~/.sky/ssh_node_pools.yaml`（`ssh_node_pools/constants.py`），`sky ssh up` 在目标机装 k3s + NVIDIA GPU Operator（`ssh_node_pools/deploy/deploy.py`）。
- `sky/clouds/kubernetes.py:213-292`：STOP/AUTOSTOP 不支持（:217、:261-264）；`CUSTOM_NETWORK_TIER` 仅在检测到高性能网络的集群支持（:229、:276、:858-860），普通 k3s 上 `network_tier="best"` 会落入 unsupported。OPEN_PORTS 不在 unsupported 列表（支持，经 LoadBalancer/ingress/podip）。
- `sky/clouds/` 目录中没有"existing cluster / bare-metal"适配器；`slurm.py` 不支持 OPEN_PORTS。
- 结论：**SkyPilot 0.13.0 不支持"接管裸机直接跑容器"；支持自有机器的唯一方式是先把机器变成 Kubernetes（`sky ssh up` 装 k3s，或已有 kubeconfig）。** 因此"直接用 sky 的 ssh node pool 作为首版"意味着在用户机器上装 k3s（sudo、外网、与既有 docker/driver 共存风险），并且 yeto 侧需处理 §2.3 的不兼容点（network_tier、pod 内 Ray 共存、镜像运行时要求、imagePullSecret、远端 head 凭据表），这些都未经 GPU 验证。

## Goals / Non-Goals

**Goals**
- 用一个 `local` 提供者把自有 GPU 机器接入 yeto RL：单岛、多岛（多机或单机按卡切分）、head/controller/syncer 可放本机或集群内任一节点。
- 接口、节点清单、预检、放置规则、失败语义在无机器时就能用假提供者（fake ops）和单测钉死；机器到位后只做环境接入与 GPU 验收。
- 一张统一的 head 放置能力表，涵盖 aws/nebius/verda/modal/local。

**Non-Goals**
- 方案 D 常驻服务端；k8s/`sky ssh up` 后端实现；Slurm；shape/报价接入；与 Modal 基线逐位比较；改变非 `local` 分支行为。

## Decisions

### D1. 首版选 A′（SSH+docker 直连后端），不经 SkyPilot；B/C 保留接口

选 A′ 的理由：
1. SkyPilot 0.13.0 没有裸机接管模式（上文证据），B/C 都要求把用户机器 k8s 化；用户"机器未到、先定接口"，不应把 k3s 安装这一环境成本和未知项放进首版。
2. `ssh_harness.py` 已有完整的 SSH+docker 部署/收集/验证能力，缺的只是去钉死与接入 launcher 的 ops 抽象（调查估 3–5 人日）。
3. 自有机器不计费，sky 的 STOP/AUTOSTOP/spot 价值为零；需要 sky 的只剩 provisioning，而 SSH 直连已足够。

备选与取舍：
- **B（`sky ssh up`）**：优点是走 launcher/FleetController "原路径"，最接近生产；代价是 k3s + GPU Operator 安装、pod 网络（flannel VXLAN）对多节点 NCCL 的带宽损失、§2.3 五个不兼容点。作为第二步，前置条件：用户允许在机器上装 k3s；yeto 侧 `make_miles_island_task` 对 cloud∈{ssh,kubernetes} 去掉 `network_tier`；`CLOUD_CREDENTIAL_PATHS` 增加 `~/.kube/config` 与 `~/.sky/ssh_node_pools.yaml`。
- **C（已有 k8s）**：同 B，跳过安装；取决于用户集群策略。
- **D**：head 模式已是"单进程常驻控制器 + 子进程 syncer + 自动重启"，D 只在多用户/多 run 排队时值得做。

A′ 的已知代价（必须在验收结论里注明）："岛生命周期由 `LocalSSHOps` 实现"而不是 sky；rl-infra-spec 依赖 FleetController 的行为（重启岛、池增减）在 local 上以 `LocalSSHOps` 为准。为此 D2 要求 `LocalSSHOps` 实现与 `SkySDKOps` 同一组动词，让 FleetController 不感知后端差异（这比调查报告中"ssh_harness 自己编排、不经 FleetController"更进一步，是本 change 对 A′ 的收紧）。

### D2. 提供者接口：`--gpu local:…@<pool>` + 节点清单 + `IslandOps` 协议

- **语法**：`local:[NxM]x<GPU>[@<pool>]`，不改 `gpu_spec.py` 的正则；`region` 字段即池名，缺省 `default`。`ClusterSpec` 不加字段。
- **节点清单** `--local-nodes <path>`（默认 `~/.yeto/local_nodes.yaml`），YAML：
  ```yaml
  version: 1
  pools:
    rack1:
      ssh_user: ubuntu
      ssh_key: ~/.ssh/id_rack1
      ssh_options: ["-o", "StrictHostKeyChecking=accept-new"]
      network_interface: eno3          # 必填：NCCL_SOCKET_IFNAME
      nccl_ib_disable: true            # 无 IB/RoCE 时 true
      model_store: /srv/models          # bind 到容器 /mnt/yeto-models（只读）
      data_dir: /srv/yeto-rl            # bind 到容器 /data/yeto-rl
      image_cache: /srv/images          # 可选：离线 docker save 归档目录
      price_per_gpu_hour: 0             # 可选，默认 0
      nodes:
        - host: 10.0.0.11
          gpus: 8
          gpu: H200
          head_reach_ip: 10.0.0.11      # 岛访问 syncer 用的地址（可选，默认 host）
        - host: 10.0.0.12
          gpus: 8
          gpu: H200
  ```
  清单由新模块 `yeto/local_cluster.py` 解析为 `LocalPool`/`LocalNode` dataclass，纯函数、可单测。
- **分配**：`--gpu local:2x8xh200@rack1,local:1x4xh200@rack1` 按声明顺序从池中贪心取节点；`num_nodes>1` 的岛取整机；`gpus_per_node` 小于整机卡数时允许**单机切分**（同一主机的不同 GPU 子集属于不同岛，用 `CUDA_VISIBLE_DEVICES`/`--gpus '"device=…"'` 隔离，Ray 端口按岛偏移）。容量不足在预检阶段失败。
- **`IslandOps` 协议**（新，`yeto/launcher.py` 内以 `typing.Protocol` 声明）：`up(spec, task) -> handle`、`job_status(handle)`、`job_alive(handle)`、`cluster_up(handle)`、`rl_strict_failure(handle)`、`down(handle)`、`pull_tapes(handle, run_dir)`、`stream_logs(handle)`。`SkySDKOps` 与 modal_ops 现有方法名保持不变并被视为该协议的实现（首版只做协议声明 + `LocalSSHOps`，不重构前两者）。`LocalSSHOps` 以 `ssh_harness.py` 的 deploy/start/status/kill/collect 为后端；测试通过 `FakeLocalOps` 注入。
- **ssh_harness 泛化**：`accelerator` 改为来自清单（去掉 H200 钉死，保留"声明型号 + 运行前 `nvidia-smi` 断言"）；`network_interface` 必填；`nccl_ib_disable` 可配置；放开"所有岛节点数相同"的限制（launcher 侧已允许异构岛）。

### D3. 镜像分发：digest 拉取优先，离线归档为备

- 岛镜像仍为 `--rl-image` 解析出的 `ghcr.io/michaellchung/yeto-miles-ports@sha256:…`（`image_ref_from_rl_image`）。
- 预检顺序：节点上 `docker image inspect <digest>` 命中 → 通过；否则若 `image_cache/<sha256>.tar` 存在 → 预检阶段 `docker load` 并校验 `RepoDigests` 含该 digest；否则要求节点能 `docker login ghcr.io`（凭据来自 `registry_credentials`，经 SSH 以 `--password-stdin` 注入，不落盘）并 `docker pull`。三者皆失败 → 预检失败，列出每个节点的原因。
- 不公开镜像（记忆 "MILES_IMAGE is private"）。

### D4. head / controller / syncer 放置：统一能力表与 `--controller here`

| 放置 | 能否承载 head（syncer 29400/29401 入站） | 说明 |
|---|---|---|
| aws / nebius（sky VM，`--controller head`） | 能 | 现状 |
| verda | 否 | adapter 无 OPEN_PORTS |
| modal | 否 | serverless 无入站端口 |
| **here（提交机本机）** | 能，若岛 → 本机 29400 可达 | 新增；取代 `run_local_head.py` |
| **local 池内节点**（`--syncer-region local/<pool>[/<host>]`） | 能 | head 作为该节点上的一个 docker 容器或 venv 进程运行，由 `LocalSSHOps` 拉起 |

- `--controller here`：本机以 `cmd_head` 的同一代码路径运行（构造 `LocalSyncer`，自动重启 `--resume`），区别仅在不经 sky VM；`SYNCER_PUBLIC_IP` 取自 `--syncer-public-addr HOST[:PORT]`，缺省时若所有岛都是 `local` 则取清单 `head_reach_ip` 所在网段上本机的地址（探测失败即报错，不猜）。本机必须在整个 run 期间在线，CLI 帮助文本写明。
- **无公网时**：岛为 `local`、head 为 `here` 或池内节点，全部走局域网，不需要任何公网地址；若岛混有 Modal（混合形态），Modal 岛需能从公网到达 syncer，此时 `here` 必须配 `--syncer-public-addr`（隧道或公网 IP），否则预检失败并指向 Nebius 无卡 VM 方案。
- `fleet_clouds` 把 `here` 视为无云；`head_cloud_credentials` 对 `local` 继续 `continue`（本机 head 不需要搬运凭据；池内 head 需要把节点清单与 SSH 私钥带过去——首版只支持 `here`，池内 head 列为任务 4.x）。
- 岛间端口：actor syncer 29400、critic 29401、Ray 6379（+岛偏移）、NCCL 端口范围由 `NCCL_SOCKET_IFNAME` 指定网卡；预检用 SSH 在岛节点上对 head 地址做 TCP connect 探测。

### D5. 存储：路径约定不变，宿主路径由清单声明

- 容器内 `/mnt/yeto-models`（只读 bind `model_store`）、`/data/yeto-rl`（读写 bind `data_dir`）、`~/yeto-output`（容器内，tape 落点）与 Nebius/Modal 完全一致，岛脚本不感知后端。
- 模型预热：首版不做自动下载到 `model_store`；预检检查 `--model` 对应的 HF snapshot 目录存在，不存在则报错并给出 `huggingface-cli download --local-dir` 的命令提示。
- 多机共享：清单可声明 `model_store` 为 NFS 挂载；yeto 不负责挂载。

### D6. tape / 事件回传与可观测性

- 岛 tape 仍写容器内 `~/yeto-output/`；`LocalSSHOps.pull_tapes` 在岛结束、失败或被 kill 后 `rsync` 回 `runs/<run>/islands/<id>/`，与 sky 岛的 rsync 路径同格式。
- `stream_logs` 复用 `ssh_harness status/collect` 的 `docker logs -f`。
- 运行账本新增字段：`provider=local`、`pool`、`hosts`、`gpu_name_observed`（来自 `nvidia-smi`）、`image_digest_source∈{inspect,load,pull}`、`cost_source=local-declared`、`price_per_gpu_hour`。

### D7. 成本记账

- `local` 每 GPU·小时单价来自清单 `price_per_gpu_hour`（默认 0）；按岛实际占用时长（`up`→`down`）计费写入账本。
- 不写入 shape/报价表，不参与自动选型；`infra-drafts/gpu-spend.md` 记账时标注"自有机器"。

### D8. 失败语义

- 预检失败：任何节点不满足条件即整体失败、不启动任何容器；错误列出"节点 → 原因"。
- 启动失败（镜像拉取、docker run 失败）：已启动的同 run 容器全部 `down`，除非 `--keep-abandoned`。
- 运行中节点失联（SSH 超时 > N 次）：`job_alive=False` → FleetController 走现有"岛失败"路径（重启/放弃），由 `LocalSSHOps.down` 尽力 `docker rm -f`；不假装成功。
- `--keep`：容器保留，打印每台主机的 `docker ps` 过滤命令。
- 同一主机被两个 run 占用：预检用 docker 容器标签 `yeto.run=<prefix>` 检测冲突并拒绝（无中心调度器，这是首版唯一的互斥手段）。

### D9. 与 sky 路径（B/C）的关系与接入点

- 保留 `--gpu ssh:…@pool` / `kubernetes:…@ctx` 作为 B/C 入口（已能解析，`infra=ssh/pool`）。
- 若日后启用，yeto 侧必做：`multinode_network_tier` 对 ssh/kubernetes 返回 None；`CLOUD_CREDENTIAL_PATHS` 增加 kubeconfig 与 ssh_node_pools.yaml；核实 pod 内 Ray 6379 与 `ray stop` 行为；镜像需含 bash/python/sudo；imagePullSecret 映射。全部未经 GPU 验证，不在本 change 任务范围内，只作为 tasks 的"后续"占位。

## Risks / Trade-offs

- [A′ 不走 sky，rl-infra-spec 中依赖 FleetController 的验收需注明后端] → D2 让 `LocalSSHOps` 实现同一 `IslandOps` 协议，FleetController 无感；验收报告仍需标注 `provider=local`。
- [`ssh_harness` 与 launcher 双路径并存引入重复] → 首版只复用 harness 的函数，不新增 CLI 子命令；harness 自身 CLI 维持原样以免破坏既有验收脚本。
- [自有卡型号与 Modal `H100!` 不同，不能逐位比较] → 只做同机型自比；判据按记忆 "Modal H100→H200 upgrade" 预先提交。
- [单机切分多岛时 NCCL/Ray 端口与 shm 冲突] → 端口按岛偏移、`--shm-size` 显式；首版验收先用整机岛，切分列为单独任务。
- [无中心调度器，多人共用集群会冲突] → D8 容器标签互斥；更强的互斥属于方案 D，明确不做。
- [机器未到，`--controller here` 路径无法在真实网络验证] → CPU 侧用 fake ops 与端口探测桩覆盖；如需提前验证，可用 Modal 单岛 + 本机 head（需 `--syncer-public-addr` 隧道，另行报批 GPU 预算）。
- [私有镜像凭据经 SSH 注入] → 用 `--password-stdin`，不写入节点磁盘；预检日志脱敏。

## Migration Plan

阶段 1（无机器，CPU）：节点清单模型 + 预检纯函数 + `IslandOps` 协议 + `FakeLocalOps` + `--controller here` + ssh_harness 去钉死，全部单测。阶段 2（机器到位）：单机 1 岛冒烟 → 单机切分两岛 strict-avg → 两机两岛 → head 放池内节点。阶段 3（可选）：B/C。非 `local` 路径行为逐字节不变，无迁移。

## Open Questions

见 tasks.md 顶部"需用户拍板"。
