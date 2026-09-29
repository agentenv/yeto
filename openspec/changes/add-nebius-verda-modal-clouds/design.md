## Context

动机见 proposal.md 的 Why。这里只列影响方案的现状和约束。

**现有结构（2026-09-23 观察）**

- `yeto/shape/plan.py` 的 `build_shape` 显式接收 `providers: AwsProviders` 和 `runpod_providers: RunPodProviders` 两个参数；`clouds` 缺省为 `["aws"]` 加有凭据时的 `runpod`；没有 AWS 凭据直接抛错。
- `yeto/shape/catalog.py` 的 `list_offerings` 只走 `sky.catalog.list_accelerators(clouds=...)`；`_to_rows` 对非 AWS 云不按区域过滤；`efa_capable` 按 AWS 机型前缀判断 RDMA 互联，决定多节点 MFU 用 0.30 还是 0.20。
- `yeto/shape/providers.py`：`AwsProviders`（配额、放置分、用量，三个 AWS API）和 `RunPodProviders`（GraphQL 库存，映射为 1 到 10 伪分）。RunPod 的 GPU 映射表含 H200，但 sky 的 RunPod 目录没有 H200 行。
- `yeto/gpu_spec.py` 的 `--gpu` 语法已允许任意小写云名，`@region` 原样传给 sky 的 `infra=cloud/region`。
- `yeto/launcher.py`：SFT 岛与 RL 岛各自构造 `sky.Task`；RL 岛用 `image_id=docker:<repo>@sha256:...`，`--spot` 时挂 `sky.Storage` 存断点；多节点时 `network_tier="best"`；`GPU_IMAGE_OVERRIDES` 只有 AWS 条目。
- `yeto/cli.py` head 模式只挂载 `~/.aws` 和 `~/.config/gcloud`；`--external-learners N` 让 syncer 为手动加入的岛留座位，启动日志打印 join 命令。
- 岛内 rollout 与 trainer 的分卡由 `yeto/rl/learner.py` 的 `parameter_mode` 分支决定，永远在同一岛内；没有任何远程 rollout 接口。

**外部事实（网页文档 + 目录 CSV + Verda 公开接口，2026-09-23）**

- SkyPilot 原生支持 Nebius（`skypilot[nebius]`，需 Nebius CLI 生成 `~/.nebius/NEBIUS_IAM_TOKEN.txt` 和 `NEBIUS_TENANT_ID.txt`）和 Verda（无额外依赖，`~/.verda/config.json` 或 `VERDA_CLIENT_ID/SECRET`）。不支持 Modal（issue #5505 closed, not planned）。
- SkyPilot v8 目录：Nebius 113 行，含 SpotPrice，6 个区域，H100 8 卡只在 eu-north1，H200 8 卡在 eu-north1/eu-west1/us-central1，B200 在 me-west1/us-central1；Verda 仅 13 行、4 条 GPU、无 8 卡机型；RunPod 14 个国家码、spot 价等于按需价。
- Verda `/v1/instance-types` 公开无鉴权，70 种机型，spot 价固定为按需价一半；`/v1/instance-availability` 需要鉴权。公开机型接口不带 location。
- Nebius 有 Capacity API（resource-advice）按 (region, platform, preset) 报 reserved/on-demand/preemptible 容量；一个 project 绑一个 region；preemptible 从 2026-10-08 起动态 spot 定价。
- Modal：serverless 容器，按秒计费，GPU 单价公开（H100 $3.95/h，H200 $4.54/h，B200 $6.25/h），CPU 和内存另计；区域选择为 1.15 到 1.75 倍价格系数；多容器 `@modal.experimental.clustered(size, rdma=True)`，2026-05-31 起要求整机 GPU 数；容器可主动出站联网。
- SkyPilot 在原生 Windows 上导入失败（依赖 POSIX `resource` 模块）；开发机需用 WSL。

## Goals / Non-Goals

**Goals:**

- 加一朵 SkyPilot 云的成本降到"一个信号类 + 一张 GPU 映射表 + 一个凭据路径"。
- `yeto shape` 在没有 AWS 凭据时也能工作。
- Modal 岛在 syncer 眼里和其他岛无差别。
- RL 岛能被规划器定价，但不改变 RL 岛内部的任何行为。

**Non-Goals:**

- 岛内 rollout / trainer 的分配策略优化（下一个 change）。
- rollout 与 trainer 拆到不同云（需要远程推理服务和 WAN 权重同步，另一个架构）。
- SSH harness（`yeto-rl-ssh`）的任何改动；它服务现有机器，无云概念。
- 让 SkyPilot 在原生 Windows 上跑。
- 把 Modal 做成 SkyPilot 自定义云插件（Modal 无 SSH，插件的第三个接口无法满足）。

## Decisions

### D1. 云信号注册表，AWS 保持独立

新增一个协议 `CloudSignals`（名字可调）：`name`、`available() -> bool`、`offerings(regions) -> list[Offering]`、`scores(asks) -> dict[(gpu, count, region), int | None]`。RunPod、Nebius、Verda、Modal 各实现一个，登记在一个字典里；`build_shape` 遍历字典而不是持有具名参数。

AWS 不进注册表：它的信号是三维的（配额限额、已用量、放置分），且 `--regions` 的旧语义与 AWS 目录耦合。硬塞进同一协议会让协议长出 AWS 专属方法，反而不干净。

替代方案：把 AWS 也塞进协议，配额作为"伪分"的一部分。放弃，因为配额是硬上限（限制岛数）而伪分是概率（影响 goodput），规划器对两者的处理逻辑不同。

### D2. Verda 目录不走 SkyPilot

sky 的 Verda 目录只有当时有货的 4 条，缺全部 8 卡机型。Verda 信号类直接调 `/v1/instance-types`（公开、无鉴权）建目录，再对每个 (机型, location) 调需鉴权的可用性接口打伪分。location 列表来自 `/v1/locations` 或固定为 FIN-01/02/03、ICL-01 并在告警中提示核对。

替代方案：给 sky 提交目录修复。可以并行做，但不能作为依赖。

### D3. Nebius 用 sky 目录做价格，Capacity API 做伪分，实时价覆盖

sky 的 Nebius 目录质量好（含 SpotPrice，区域完整），直接用。伪分来自 Capacity API 的 preemptible 容量：容量 >= 请求岛数 x 每岛节点数为 9，容量 > 0 但不够为 6，为 0 则 0。10 月 8 日后 sky 目录的 SpotPrice 失真，信号类在拿到实时价时覆盖目录价，并在 `Offering` 上标记价格来源。

Nebius 一个 project 一个 region：`~/.sky/config.yaml` 里 `nebius.<region>.project_id` 的形式由 sky 决定；Yeto 在提交前检查 fleet 涉及的每个 region 都有 project id，缺则报错（见 head-cloud-credentials spec）。

### D4. Modal 走远程 learner 运行器，复用 external-learners 座位

新模块（暂名 `yeto/modal_runner.py`）定义一个 Modal app：`@app.function(gpu=f"{GPU}:{count}", image=..., timeout=..., retries=...)`，多容器时加 `@modal.experimental.clustered(size=nodes, rdma=True)`。函数体内起 torchrun（多容器用 `get_cluster_info()` 拿 rank 和 IP）跑 `yeto.learner` 或 `yeto.rl.learner`，环境变量与 sky 岛一致。

launcher 看到 `modal:` 前缀时不构造 `sky.Task`，而是在 syncer 起来后用 Modal SDK `spawn` 该函数，把 learner id、syncer 地址、镜像摘要作为参数传入。syncer 的 `--learners` 计数照旧包含 Modal 岛。`yeto down` 调 Modal 的取消接口。

镜像：SFT 用 `modal.Image.debian_slim().pip_install(requirements)` 加仓库代码（`add_local_dir`）；RL 用 `modal.Image.from_registry("<repo>@sha256:...")` 复用 Miles 镜像，摘要不一致即拒绝。

syncer 可达性：head 模式下 syncer 在有公网 IP 的 head VM 上，开了端口；本地模式要求用户传 `--syncer-public-addr`（新参数）或报错。不做自动隧道。

替代方案：（a）SkyPilot 自定义云插件，放弃（无 SSH）；（b）Modal 上只跑 rollout 服务，放弃（属于 Non-Goal 的拆云）；（c）只做手动接入脚本不改 launcher。(c) 作为第一步交付物保留，launcher 路由在其之上。

### D5. `--regions` 语法向后兼容，Modal 区域是系数

解析规则：条目含 `:` 则按 `cloud:region` 拆分，否则视为 `aws:region`；`all` 为全部。解析结果是 `dict[cloud, set[region] | ALL]`。`_to_rows` 按云查这个字典过滤。Modal 的信号类把 `modal:us` 这类值翻译为价格系数（宽区域 1.5，窄区域 1.75，从 Modal 定价页记录进静态表）并传给运行器的 `region=` 参数。

替代方案：新参数 `--cloud-regions`，保留 `--regions` 纯 AWS。放弃，两个参数容易互相矛盾。

### D6. RL 岛定价：形状进规划器，行为不进

`build_shape` 新增 `island_shape` 输入：`gpus_per_island`（actor + rollout）、`single_node_only`、`image_required`、`spot_needs_storage`。cli 从 RL 参数推出这四个值。规划器用它们过滤候选并把镜像和存储的"已验证云集合"作为过滤条件；这两个集合是代码里的常量，由验证任务填充。不碰 `yeto/rl/learner.py`。

### D7. `efa_capable` 泛化为按云判断的 `rdma_capable(cloud, instance_type)`

AWS 沿用 p4/p5 前缀；Nebius 的 8 卡 SXM 机型视为 InfiniBand 可用（Nebius 文档）；Verda 单节点故不适用；Modal 多容器 `rdma=True` 时视为可用；RunPod 视为不可用。多节点 MFU 由此选 0.30 或 0.20。

### D8. head 凭据按需挂载，提交前校验

head 任务构造时遍历 fleet 的云集合，对每朵云查凭据路径表（AWS `~/.aws`、GCP `~/.config/gcloud`、RunPod `~/.runpod`、Nebius `~/.nebius`、Verda `~/.verda`、Modal `~/.modal.toml`），存在则挂载，缺失则报错退出。当前对 `~/.aws` 缺失只打警告的行为改为：AWS 在 fleet 中才报错，不在则不提。

### D9. 验证用真机，结果写成常量

Nebius / Verda 的三项 RL 云敏感项（容器镜像启动、对象存储挂载、多节点互联）无法在本地验证，每项对应一个真机任务，结论写进 `VERIFIED_DOCKER_IMAGE_CLOUDS`、`VERIFIED_SPOT_STORAGE_CLOUDS`、`RDMA_CLOUDS` 三个常量和 docs/CLOUDS.md。验证未通过的云在对应功能上被规划器排除，而不是让用户在启动时踩坑。

## Risks / Trade-offs

- [Nebius 10 月 8 日动态定价让 sky 目录 spot 价失真] → D3 的实时价覆盖；在此之前先按目录价 + `price_margin` 放大到 0.25 应对。
- [Verda 可用性接口需鉴权] → 实现时核实 `GET /v1/instance-availability?is_spot=true` 是批量接口，一次返回每个 location 有货的机型列表；一次请求加 15 分钟缓存即可覆盖全部候选，原先"只查价格前 24 的形状"的限制不再需要。location 列表来自需鉴权的 `/v1/locations`，读不到时回退到固定表并告警。
- [Modal 容器被抢占后 Ray 状态丢失（RL 岛）] → 复用 sky 岛的重启语义：以同一 learner id 重启，syncer 宽限期等待；RL 岛的断点用 Modal Volume 挂到 `--rl-completed-groups-path`。
- [Modal 出站到 syncer 的带宽和延迟未知] → 手动接入脚本先跑一次 SFT smoke 测 WAN 同步时间，再决定是否把 Modal 岛的 `pipeline` 深度单独调大。
- [SkyPilot 的 `image_id: docker:` 在 Nebius / Verda VM 上可能不支持] → D9 验证任务；不支持则该云的 RL 岛被排除，SFT 岛不受影响。
- [Verda 单节点限制让大模型无法落到 Verda] → 明确告警；Verda 的 cluster API 留作后续。
- [静态 Modal 价格表过期] → 表里记日期，超过 90 天告警（spec 已定）。
- [`--regions` 旧写法用户不知道新语法] → README 和 `--help` 同步更新，旧写法零改动。
- [Windows 开发机无法跑 sky] → docs 写明用 WSL；单元测试全部 mock sky，不依赖导入。

## Migration Plan

1. 先合并信号注册表和 RunPod 迁移（行为不变，靠现有测试守住）。
2. 合并 `--regions` 解析和 `--clouds` 默认值变更（旧写法不变）。
3. 合并 Nebius、Verda 信号类，各自带 mock 测试；真机验证任务在有凭据后执行，结论回填常量。
4. 合并 Modal 手动接入脚本，跑一次 SFT smoke；再合并 launcher 路由。
5. 合并 RL 岛定价钩子和 head 凭据挂载。
6. 回滚：每步独立，`--clouds aws` 和无前缀 `--regions` 始终等价于旧行为。

## Open Questions

- Verda 的 location 列表是否有公开接口（`/v1/locations`），还是只能从鉴权接口拿。不影响方案，只影响信号类里是查接口还是用固定表。
- Modal 上 RL 岛的 Docker-in-Docker（Terminal-Bench 类环境）是否可行。不影响 SFT 岛和 CyberGym 类外部 API 环境；若不可行，Modal RL 岛只支持外部 API 环境并在告警中说明。
- Nebius Capacity API 的调用配额和延迟，决定伪分缓存用 15 分钟还是更短。
