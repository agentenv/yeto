# Tasks

执行约定：
- 测试命令 `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/<file>`；本机禁止运行任何会启动 Ray 的测试。
- 阶段 1 全部在无机器条件下用 CPU 完成；阶段 2 起需要用户的自有 GPU 机器；阶段 3 可选。
- 不改 `openspec/changes/rl-infra-spec/` 下任何文件；不动非 `local` 分支行为（每项都以现有 argv 快照测试不改即通过为回归判据）。
- 每项 ≤2 小时粒度。

**需用户拍板（阻塞阶段 2，不阻塞阶段 1）**
1. 自有卡型号/数量/台数、是否有 IB/RoCE、网卡名；是否允许装 k3s（决定是否启动阶段 3）。
2. 同意放开 ssh_harness 的 H200 钉死（改为"清单声明 + 运行前断言"）。
3. 本地验收是否接受 `provider=local`（LocalSSHOps 后端）的结论作为 rl-infra-spec 相关验收证据。
4. 私有镜像拉取方式：节点上放 ghcr PAT（经 stdin 注入）还是离线 `docker save` 归档。
5. 逐位类判据改为同机型自比，需事先提交判据。

## 1. 节点清单与预检（CPU，无机器）

- [ ] 1.1 新增 `yeto/local_cluster.py`：`LocalPool`/`LocalNode` dataclass、YAML 加载与校验（design D2 清单格式）。验收：`tests/test_local_cluster.py` 覆盖缺字段、未知池、重复 host、非法网卡名；全部在不触网条件下运行。依赖：无。
- [ ] 1.2 节点分配纯函数 `allocate(specs, pool)`：整机优先、单机切分、确定性、容量不足抛错（spec "节点分配"）。验收：三条 Scenario 单测 + 同输入两次结果相同。依赖：1.1。
- [ ] 1.3 `check_cloud_prerequisites` 新增 `local` 分支，预检命令以可注入的 `runner(host, cmd)` 执行（spec "启动前预检"）。验收：用假 runner 模拟 GPU 名不符、docker 缺失、镜像三路失败、端口不可达，断言一次报告全部节点原因；非 local specs 走原逻辑（现有测试不改）。依赖：1.1、1.2。
- [ ] 1.4 镜像三路解析（inspect → image_cache load+digest 校验 → login/pull）为纯函数，返回 `image_digest_source`。验收：单测覆盖三路与全失败；凭据不出现在日志字符串中。依赖：1.3。

## 2. 控制器放置（CPU，无机器）

- [ ] 2.1 `--controller here`：cli.py `choices` 加 `here`，复用 `cmd_head` 构造 `LocalSyncer` 的路径，不起 sky VM（design D4）。验收：`tests/test_head_mode.py` 新增用例：`here` 不调用 sky、syncer 子进程参数与 `head` 相同、`fleet_clouds` 不含 head 云。依赖：无。
- [ ] 2.2 syncer 地址推导：`--syncer-public-addr` 优先；全 local 时按 `head_reach_ip` 网段选本机接口（可注入接口表）；混合 Modal 无公网地址则失败（spec 两条 Scenario）。验收：单测。依赖：1.1、2.1。
- [ ] 2.3 head 放置能力表 `HEAD_PLACEMENT_CAPABLE`（aws、nebius、here、local；verda、modal 不可），`head_cloud` 返回不可承载者时提交前失败。验收：单测含 verda Scenario；现有 nebius/aws head 测试不改即通过。依赖：2.1。

## 3. IslandOps 协议与 LocalSSHOps（CPU，无机器）

- [ ] 3.1 在 launcher.py 声明 `IslandOps` Protocol，确认 `SkySDKOps` 与 modal_ops 现有方法名满足协议（只加类型声明，不改实现）。验收：mypy/静态断言测试通过；现有测试不变。依赖：无。
- [ ] 3.2 ssh_harness 泛化：accelerator/网卡/`nccl_ib_disable` 来自参数，去掉 H200 钉死与 `eno3` 默认，放开"所有岛节点数相同"（design D2）。验收：`tests/test_rl_ssh_harness.py` 原用例在显式传 H200/eno3 时结果不变；新增 H100 + `enp1s0` 用例通过。依赖：用户拍板 2（可先做，合并前确认）。
- [ ] 3.3 `LocalSSHOps`：up（生成 docker run，含 `-v model_store:/mnt/yeto-models:ro`、`-v data_dir:/data/yeto-rl`、`--gpus device=…`、`yeto.run` 标签、NCCL 环境）、job_status/job_alive/cluster_up/rl_strict_failure/down/pull_tapes/stream_logs，底层命令经可注入 runner。验收：docker run 命令快照测试；每个动词对假 runner 的调用序列断言。依赖：1.2、3.1、3.2。
- [ ] 3.4 `FakeLocalOps` + FleetController 集成测试：岛 `job_alive=False` 走与 sky 岛相同的失败路径；第二岛启动失败时第一岛被 down（`--keep-abandoned` 则保留）。验收：`tests/test_local_island_lifecycle.py`。依赖：3.3。
- [ ] 3.5 账本与 tape：provider=local 字段、`cost_source=local-declared`、按 up→down 时长 × `price_per_gpu_hour` 计费、tape rsync 到 `runs/<run>/islands/<id>/`。验收：单测默认 0 成本与自定义单价；rsync 命令快照。依赖：3.3。
- [ ] 3.6 docs/CLOUDS.md 增加 `local` 节（清单示例、`--controller here`、预检清单、已知限制），cli 帮助文本。验收：文档含 design D4 能力表。依赖：2.x、3.x。

## 4. 机器到位后的 GPU 验收（需用户机器与拍板 1–5）

- [ ] 4.1 环境接入：按清单填写真实节点；跑预检（不启动容器）并把输出存入 `openspec/changes/rl-local-cluster-deploy/evidence/preflight-<date>.txt`。验收：全部通过或列出待修环境项。依赖：阶段 1–3。
- [ ] 4.2 单机 1 岛冒烟（小模型，`--controller here`，no-sync 或 strict-avg 单岛）：3 轮，tape 回收，账本字段齐全。验收：运行 ID、`nvidia-smi` GPU 名、指标摘要写入 progress.md。依赖：4.1。
- [ ] 4.3 单机切分两岛 strict-avg：验证端口偏移、GPU 子集隔离、跨岛同步。验收：两岛 tape 中 syncer round 对齐；同机型自比判据按拍板 5。依赖：4.2。
- [ ] 4.4 两机两岛（若有 ≥2 台）：验证 NCCL 走声明网卡、`nccl_ib_disable` 设置生效。验收：岛日志含 `NCCL_SOCKET_IFNAME=<iface>`，吞吐记录入 progress.md。依赖：4.2。
- [ ] 4.5 失败语义实测：kill 一岛容器、拔一台节点 SSH；控制器按 spec 走失败路径，tape 回收，账本记"失联"。验收：evidence 目录含事件 tape 与控制器日志。依赖：4.3 或 4.4。
- [ ] 4.6 head 放池内节点（`--syncer-region local/<pool>/<host>`）。验收：本机断网后 run 继续；此项若不需要可由用户取消。依赖：4.2。

## 5. 后续（可选，方案 B/C；本 change 不实现，仅占位）

- [ ] 5.1 若用户允许 k3s：`multinode_network_tier` 对 ssh/kubernetes 返回 None；`CLOUD_CREDENTIAL_PATHS` 加 kubeconfig 与 `~/.sky/ssh_node_pools.yaml`；核实 pod 内 Ray 与镜像运行时要求。验收：另立 change。依赖：用户拍板 1。
