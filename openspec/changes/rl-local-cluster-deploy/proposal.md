# Proposal

## Why

yeto RL 目前只有两种实际部署形态：GPU 岛在 Modal（单岛 no-sync 直接在 Modal 容器内跑 `yeto/modal_runner.py`；需 syncer 的多岛运行把 controller+syncer 放 Nebius 无卡 VM，走 `yeto/launcher.py` 的 `--controller head` 路径），其余云通过 `sky.launch` 拉岛。用户即将获得自有 / 本地 GPU 集群（机器未到），希望"先把 spec 写了"：在机器到位前把提供者接口、放置规则、预检与失败语义定下来，使 CPU 侧的接口与假提供者可以先行实现并单测，机器到位后只做环境接入与 GPU 验收。

现状与方案调查已在 `openspec/changes/rl-infra-spec/local-cluster-investigation.md`（2026-09-30，基于 SkyPilot 0.13.0 源码）给出：方案 A′（本机 head 正式化 + ssh_harness 泛化，3–5 人日）、B（`sky ssh up` 建 k3s 池）、C（已有 Kubernetes）、D（常驻 yeto 服务端）。本 change 采纳其推荐（A′ 先行，B/C 作为可选第二步），但把它从"调查报告"落成可实现、可验收的 change；本 change 不改动 `rl-infra-spec` 下任何文件。

相关约束（记忆 "head placement limits"）：Verda（sky adapter 不声明 OPEN_PORTS）与 Modal（serverless 无入站端口）都不能托管 head；本地集群是第一个"岛与 head 天然同网"的形态，应顺带把 head 放置规则统一成一张表（见 design D4）。

## What Changes

- **新增提供者 `local`**：`--gpu local:[NxM]x<GPU>[@<pool>]`（复用 `yeto/gpu_spec.py` 现有语法，`cloud` 正则 `[a-z]+` 已可解析 `local`，`@pool` 落到 `region` 字段指向节点清单中的一个池）。首版后端为 **"节点清单 + SSH + docker"**（复用 `yeto/rl/ssh_harness.py` 的 prepare/deploy/start/status/kill/collect 能力，去掉 H200 钉死与 `eno3` 默认），k8s/`sky ssh up` 后端只定接口、不实现。
- **新增控制器放置 `--controller here`**：在提交机本机正式运行 `cmd_head`（FleetController + LocalSyncer 子进程、自动重启 `--resume`），`SYNCER_PUBLIC_IP` 由 `--syncer-public-addr` 显式给出或按节点清单的 `head_reach_ip` 推导；不再需要 `infra-a-gpu/.../run_local_head.py` 这类包装。
- **预检**（`check_cloud_prerequisites` 的 `local` 分支）：节点清单格式、SSH 可达、`nvidia-smi` 型号与数量与 `--gpu` 声明一致、docker + nvidia-container-toolkit、镜像 digest 可拉取或已离线导入、`/mnt/yeto-models` 与 `/data/yeto-rl` 可写、29400/29401 与 Ray 6379 可达——全部在任何 GPU 进程启动前失败。
- **存储与镜像**：模型存储路径约定 `/mnt/yeto-models`（与 Nebius FS / Modal Volume 挂载点一致，`launcher.MODEL_STORE_MOUNT`）不变，由节点清单声明宿主路径并 bind 进容器；私有 ghcr 镜像按 digest 拉取，支持 `docker load` 离线缓存并校验 digest。
- **tape / 事件回传**：岛侧 tape 仍写 `~/yeto-output/*.jsonl`，结束或失败后由控制器用 rsync 拉回 `runs/<run>/`（与现有 sky 岛 `rsync cluster:yeto-output/rl-island-*.jsonl` 一致）。
- **成本记账**：`local` 提供者单价默认 0，可在节点清单给 `price_per_gpu_hour` 自定义；记入运行账本但标记 `cost_source=local-declared`，不进入 shape/报价选择。
- **与 sky 路径的关系**：首版**不**经 SkyPilot 拉岛（见 design D1 证据：0.13.0 的 ssh node pool 本质是装 k3s 的 Kubernetes，不是接管裸机）；但 `--gpu ssh:…@pool` / `kubernetes:…@ctx` 保留为方案 B/C 的接入点，并在 design 列出需去掉 `network_tier="best"` 等已知不兼容点。

## Non-goals

- 不实现常驻 yeto 服务端 / 多用户排队 API（方案 D）。
- 不在本 change 内实现 k8s / `sky ssh up` 后端（方案 B/C 仅定接口与前置条件）。
- 不做 Slurm。
- 不把本地卡接入 shape/报价（`yeto/shape/providers.py`）自动选型。
- 不改变 Modal / Nebius / Verda 现有路径行为；`local` 分支以外的 argv 与 sky Task 构造逐字节不变。
- 不要求本地结果与 Modal `H100!` 基线逐位一致（型号不同，只做同机型自比，见 design 风险）。
- 不启动任何付费资源；机器到位前所有验证均为 CPU 单测与 dry-run。

## Capabilities

### New Capabilities
- `rl-local-cluster-provider`：`local` 提供者的声明、节点清单、预检、head/syncer/岛放置规则、镜像与存储约定、失败语义与可观测性。

### Modified Capabilities
（无：`openspec/specs/` 下现有 `head-run-teardown` 不受影响；本 change 不修改 `rl-infra-spec` 的任何文件。）

## Impact

- yeto：`yeto/gpu_spec.py`（`local` 校验，语法不变）、`yeto/launcher.py`（`check_cloud_prerequisites` local 分支、`fleet_clouds`/`head_cloud_credentials` 对 `local` 的处理、岛 ops 抽象 `LocalSSHOps` 与 `SkySDKOps`/modal_ops 并列、tape 回收、成本记账）、`yeto/cli.py`（`--controller here`、`--local-nodes <yaml>`）、`yeto/rl/ssh_harness.py`（accelerator 与网卡可配置、单主机按卡切分多岛）、新增 `yeto/local_cluster.py`（节点清单模型与预检）、tests、docs/CLOUDS.md。
- 不涉及 fork `michaellchung/miles` 改动，不涉及镜像重建。
- GPU 预算：本地卡为自有，不计云费；若机器到位前需在云上验证 `--controller here` 路径，另行报批（design 风险节给出替代：用 Modal 单岛 + 本机 head，此前 7.1 已有先例）。
