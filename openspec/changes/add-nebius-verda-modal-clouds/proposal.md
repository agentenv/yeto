## Why

Yeto 的跨云能力目前只真正接了 AWS 和 RunPod：`yeto shape` 规划器只会为这两家查配额、库存和价格，`yeto launch` 的 head 模式也只挂载这两家的凭据。实际采购的算力已经扩展到 Nebius、Verda 和 Modal，而这三家在代码里没有任何接入。按今天（2026-09-23）的目录价，8 卡 H100 的 spot 在 Verda 是 AWS 美国区的三分之二，Nebius 也便宜四分之一，不接入就是在多付钱。

## What Changes

- **规划器信号抽象**：把 `yeto/shape/providers.py` 里 RunPod 那种"凭据检测 + 库存伪分"的模式抽成一个云信号注册表，AWS 保持独立（它多了配额、用量、区域三个维度）。RunPod 迁移到注册表。
- **Nebius 接入**：凭据检测（`~/.nebius/` 下的 IAM token 和 tenant id）、sky 目录（含 SpotPrice）、Nebius Capacity API 的 preemptible 容量映射为伪分、多节点按 InfiniBand 计算 MFU。
- **Verda 接入**：凭据检测（`~/.verda/config.json` 或环境变量）、**不用 sky 目录**（它只有 4 条 GPU 记录，缺所有 8 卡机型），改调 Verda 公开的 `/v1/instance-types` 拿全量机型和 spot 价，`is_available()` 作为库存信号。第一版限单节点。
- **Modal 接入**：Modal 不在 SkyPilot 支持列表，走独立的**远程 learner 运行器**：一个 Modal app 在容器里跑 `yeto.learner` 或 `yeto.rl.learner`，借用现有 `--external-learners` 的座位机制向 syncer 注册。`--gpu modal:8xh100` 由 launcher 路由到这个运行器而不是 `sky.Task`。规划器里 Modal 是静态价格表加"永远可用"伪分，并计入 CPU 和内存费。
- **`--regions` 语法**：**BREAKING**（软）：新增 `cloud:region` 前缀写法，例如 `aws:us-east-1,nebius:eu-north1,verda:FIN-03,modal:us`。无前缀的旧写法继续按 AWS 解释。Modal 的区域是一个价格系数而不是机房，规划器默认不指定。
- **解除 shape 对 AWS 凭据的硬依赖**：`--clouds` 不含 aws 时不再要求 AWS 凭据。默认值改为"aws 加上所有检测到凭据的云"。
- **RL 岛定价钩子**：`yeto shape` 能为 RL 岛定价（actor 卡数 + rollout 卡数 + Miles 镜像），使 `--training-mode rl` 也能按岛级分布落到任意已接入的云。岛内 rollout / trainer 的分配策略**不在本次范围**，另起 change。
- **head 模式凭据挂载**：head VM 补挂 `~/.nebius`、`~/.verda`、`~/.runpod` 和 Modal token，让 head 能替用户启动这几家的岛。
- **RL 岛云敏感项验证**：Docker `image_id` 在 Nebius / Verda VM 上的支持、spot 断点 `sky.Storage` 在 Nebius S3 兼容存储上的可用性、`network_tier=best` 在 Nebius 上是否拿到 InfiniBand。这些是验证任务，结果决定对应云是否允许 `--spot` 和多节点。

## Capabilities

### New Capabilities
- `shape-cloud-signals`: 规划器按云取得凭据状态、机型目录、价格和可用量伪分的统一契约，以及 Nebius、Verda、Modal、RunPod 各自的实现。
- `shape-region-filter`: `--regions` 的 `cloud:region` 语法、向后兼容规则、以及各云对区域的解释（AWS/Nebius 机房、Verda location、Modal 价格系数）。
- `shape-rl-islands`: 规划器为 RL 岛定价和过滤时必须知道的形状约束（单节点、rollout 卡数、镜像）。
- `modal-learner-runner`: Modal 上运行 SFT / RL learner 的远程运行器：入口、环境变量、镜像、多机 clustered 规则、与 syncer 的连接方式、launcher 的路由。
- `head-cloud-credentials`: head 模式下 head VM 必须携带的各云凭据文件及其缺失时的行为。

### Modified Capabilities
（无现存 spec。）

## Impact

- 代码：`yeto/shape/providers.py`、`yeto/shape/plan.py`、`yeto/shape/catalog.py`、`yeto/gpu_spec.py`、`yeto/launcher.py`（learner 任务构建、RL 岛任务构建、image override 表、`efa_capable` 泛化）、`yeto/cli.py`（`--regions`、`--clouds`、head 凭据挂载）、新文件 `yeto/modal_runner.py`（名称待定）。
- 依赖：`pyproject.toml` 的 launcher extra 从 `skypilot[aws]` 扩为 `skypilot[aws,runpod,nebius,verda]`；新增可选依赖 `modal`；Nebius 需要本机装 Nebius CLI 生成 token 文件；Verda 的 SDK 可选（REST 足够）。
- 测试：`tests/test_shape_providers.py`、`tests/test_shape_plan.py`、`tests/test_shape_catalog.py`、`tests/test_gpu_spec.py`、`tests/test_head_mode.py`、`tests/test_launch_auto.py` 需要扩展；新增 Modal 运行器的单元测试（不需要 Modal 凭据）。
- 文档：README 的 `--gpu` / `--regions` 说明，新增 docs/CLOUDS.md 记录每家云的凭据、区域、已验证的限制。
- 运行环境：SkyPilot 在原生 Windows 上无法导入（依赖 POSIX `resource` 模块），开发者在 Windows 上需用 WSL 跑 `yeto shape` / `yeto launch`。
- 时间敏感：Nebius preemptible 从 2026-10-08 起改为动态 spot 定价，sky 目录的 SpotPrice 列届时失真，Nebius 信号类需从 Capacity API 或定价接口取实时价。
