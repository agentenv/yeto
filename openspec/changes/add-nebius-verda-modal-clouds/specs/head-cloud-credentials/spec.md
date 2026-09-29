## Purpose

head 模式下，head VM 代替用户机器启动、恢复和回收各云的岛，因此必须携带每朵云的凭据。本能力规定哪些凭据被带上 head、缺失时的行为，以及凭据不泄露到 learner 岛的边界。

## ADDED Requirements

### Requirement: head 携带所有已启用云的凭据
提交 head 任务时，系统 SHALL 把 fleet 中出现的每朵云的本机凭据文件挂载到 head 的相同路径：AWS、GCP、RunPod、Nebius、Verda 和 Modal。某云在 `--gpu` 中出现而本机缺其凭据时 MUST 在提交前报错退出，而不是等 head 启动失败。fleet 中未出现的云的凭据 SHALL 不被挂载。

#### Scenario: 缺 Nebius 凭据
- **WHEN** 用户以 head 模式运行 `yeto launch --gpu nebius:8xh100@eu-north1 ...`，而本机没有 Nebius 凭据文件
- **THEN** 命令在提交 head 之前报错，说明期望的凭据路径

#### Scenario: 只挂载用到的云
- **WHEN** fleet 只含 AWS 和 Verda 的岛，本机同时有 RunPod 凭据
- **THEN** head 上有 AWS 和 Verda 凭据，没有 RunPod 凭据

### Requirement: 凭据不流向 learner 岛
learner 岛 SHALL 只收到运行训练所需的 token（Hub token、可选的 W&B key、环境相关的 API key），MUST NOT 收到任何云的启动凭据。

#### Scenario: 岛上无云凭据
- **WHEN** 任一岛启动完成
- **THEN** 岛内不存在 AWS、Nebius、Verda、RunPod 或 Modal 的凭据文件或环境变量

### Requirement: Nebius 多区域项目
Nebius 的项目与区域一一绑定。当 fleet 含多个 Nebius 区域的岛时，系统 SHALL 要求用户为每个区域提供项目标识（通过配置文件），缺失时 MUST 在提交前报错并列出缺少项目标识的区域。

#### Scenario: 两个 Nebius 区域
- **WHEN** 用户启动 `nebius:8xh100@eu-north1,nebius:8xh200@us-central1`，配置里只有 eu-north1 的项目
- **THEN** 命令在提交前报错，指出 us-central1 缺少项目标识
