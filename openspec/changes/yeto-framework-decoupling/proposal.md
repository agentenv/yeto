## Why

**设计初衷**：yeto 要把各路资源接起来、统一使用——不同的云与资源提供方（Modal、Nebius、Verda、AWS、RunPod、SkyPilot 系各云、自有集群）、不同的开卡与计费方式（spot 抢占式、按需）、不同类型与厂商的加速器（GPU/TPU/NPU；英伟达、昇腾、AMD、摩尔线程）、不同的训练框架（现有 Miles = Megatron + SGLang，即将接 verl）。

**两个目的**：
1. **节约成本**：不怕中断的任务放到 spot 等便宜资源上，只有持有唯一状态的任务用按需。
2. **训练弹性**：按不同卡、不同集群的特性与算力，动态调整每个岛的角色与权重（加入、退出、降为纯 rollout、按算力占比合并）。

**五个维度**：训练框架、云与资源提供方、计费与可中断性、加速器类型与厂商、拓扑。yeto 的价值在于把这五个维度正交组合；任何一处把某个维度写死在核心里，组合数就变成"重写数"。

**现状问题**（详见 `infra-drafts/YETO-DECOUPLING-AUDIT-S16.md`，下称"审计"）：引擎核心有 4 处直接 import Miles（`engine/trainer_transition.py:42`、`engine/driver.py:278`、`engine/bridges.py:26`、`engine/runtime_manifest.py:146`）；算法扩展用"Miles 命令行旗标"注册自己（`algos/*` 经 `miles_adapter.algorithm_flags.register_flag`）；奖励/过滤/harness 的函数签名绑 Miles `Sample`；启动器里 `if spec.cloud == ...` 分支 10+ 处；RL 路径写死 `nccl`/`cuda-ipc`/H200；计费只有一个全局 `--spot`。

**为什么现在做**：verl 适配层（`rl-verl-backend`）即将开工。若不先去耦合，verl 只能复制一份 Miles 式的写死代码，此后每加一朵云、一类卡都要改核心。因此本 change 作为 `rl-verl-backend` 的**前置**。

## What Changes

按阶段推进，**每个阶段可独立合并**，且每阶段都必须保证 Miles 现有行为、tape 事件字段、进度文件格式、契约哈希（`AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash`）、Miles 命令行输出逐字节不变。

- **阶段 0 护栏**：录"标准样本"（典型配置的 Miles 命令行、两种契约哈希、strict/decoupled 进度文件、tape 片段，全部 CPU 生成）；新增静态边界检查（`engine/`、`rewards/`、`harness/`、`algos/` 等禁止 import 训练框架与云库），现有违规写入白名单，**只减不增**。
- **阶段 1 引擎核心切断反向依赖**：`RecoveryRequired`、`CutContext`、`ReshardPlan` 移入核心；重分片检查改走可选端口；事件写入改为中立的事件写入器；进度文件格式抽到核心（磁盘格式字节不变）；运行时清单的探针/版本表/补丁记录由适配层提供；权重传输方式改中立名（审计 E1–E9、E21、A4、A5、V2、V3）。
- **阶段 2 奖励/过滤/harness 中立化**：新增中立轨迹、奖励结果、过滤决定；新 `yeto/rl/rewards/` 目录；Miles 薄包装保持原签名与原行为；轮次元数据/策略令牌改由核心接口提供；codex 中 Miles 胶水收进 Miles 适配层；会话服务协议成文；预留用户自定义奖励环境接口（注册/加载 + 示例）；Miles math 判分工具只拷进测试目录作对照（审计 R1–R15、A2、A6、A7）。
- **阶段 3 配置与算法扩展中立化**：`RunConfig` 拆中立部分与 Miles 部分；算法扩展只注册中立规格字段，"字段→Miles 旗标"表搬进 Miles 适配层；过滤器/插件改中立名，算法配置哈希按新名与新源码重新计算，并维护旧哈希→新哈希对照表（附 CPU 逐位一致证据），并在启动前逐项核对训练端绑定（命令行逐字、插件身份与源码哈希、CPU 实调一次），任一不一致即启动失败；yeto 已有奖励/优势变换拆成纯函数（审计 E10–E15、A1、A3、L5、P3、P4；`RL-ALGO-LOCATION-S16.md` §4 A1）。
- **阶段 4 Miles 代码归位**：`yeto/rl/engine/miles_adapter/` 搬到 `yeto/rl/adapters/miles/`（原位置留转发模块）；旧版引擎 `rl/miles.py`、Miles 专用模型文件、harness 中 Miles 胶水、`rl/__init__.py` 中 Miles 版本常量、`miles_overlay` 一并收进去；新增后端注册表，启动器经注册表取后端（审计 L1–L5、M1–M4、A8、A9）。`engine/` 暂不改名为 `core/`。
- **阶段 5 后端身份进契约**：新增 `BackendIdentity`（后端名、commit、设备族、参数名映射表哈希），哈希与旧哈希**并列**；Miles 旧哈希不变；不同后端身份的岛合并时拒绝（审计 E23）。
- **阶段 6 硬件层**：`yeto/hw/`（设备族表：cuda/npu 起步，预留 AMD/TPU/摩尔线程行；卡型目录合并；通信与确定性环境变量；拓扑中立部分）（审计 H1–H7、V1、V5、V7、E16、M5）。
- **阶段 7 云层与按任务分 spot/按需调度**：`yeto/cloud/` 的云提供方接口（以 `modal_runner.py` 为样板）；启动器去 `if cloud ==`；任务可中断性四级；调度层"能力求交 + 价格"只出建议；自有集群（rl-local-cluster-deploy）推后到 verl 完成且新集群到手后，本 change 只留接口（审计 C1–C9、B1–B7）。
- 跨卡型/跨厂商合并：理想状态都能合并，按"同卡型同厂商 → 同厂商不同卡型 → 跨厂商"三期开放，本 change 只留兼容组字段与接口。
- 中立的逐词元损失参考函数（只用于 CPU 对照，不改 Miles fork）。

**与 verl 的先后关系**：verl 适配层开工前须完成阶段 0–3；阶段 5 须在 verl 首次真跑前完成；阶段 4 建议在 verl 开工前完成（以便 verl 直接落在 `yeto/rl/adapters/verl/`）；阶段 6、7 可与 verl 并行。

无 **BREAKING** 变化：所有搬迁保留旧 import 路径的转发模块，默认后端仍为 Miles。

非目标：`engine/` 改名为 `core/`；Miles fork 改调 yeto 函数（推迟到 verl 第一阶段打通后另议）；critic、GAE 变体、归约器、需拼回整条序列的算法、熵的迁移；k3s；调度层自动执行；跨后端/跨卡型的岛合并本身。

## Capabilities

### New Capabilities
- `rl-framework-neutral-core`: 引擎核心只认中立端口与数据；静态边界检查与白名单；标准样本回归；后端身份进契约且不改旧哈希。
- `rl-neutral-rewards-harness`: 中立轨迹、奖励结果、过滤决定、轮次元数据接口与会话服务协议；奖励/优势纯函数；Miles 包装同输入同输出。
- `rl-backend-adapters`: 适配层目录约定（`yeto/rl/adapters/{miles,verl}/`）、后端注册表、中立配置与算法字段到各后端的映射、中立名兼容与能力声明拒绝。
- `yeto-resource-dimensions`: 硬件设备族与拓扑、云提供方接口与能力声明、任务可中断性分级与按任务选计费方式。

### Modified Capabilities
（无。现有主 spec 不涉及这些行为。）

## Impact

- 代码（实现时）：`yeto/rl/engine/*`（driver、bridges、trainer_transition、runtime_manifest、run_config、algorithm、selection、multinode、cut、ports）、`yeto/rl/algos/*`、`yeto/rl/{math_reward,gsm8k_reward,length_reward,filters}.py`、`yeto/rl/harness/codex/*`、`yeto/rl/miles.py`、`yeto/rl/learner.py`、`yeto/rl/__init__.py`、`yeto/launcher.py`、`yeto/cli.py`、`yeto/modal_runner.py`、`yeto/accel.py`、`yeto/shape/*`；新目录 `yeto/rl/adapters/`、`yeto/rl/rewards/`、`yeto/rl/session/`、`yeto/rl/scheduling/`、`yeto/hw/`、`yeto/cloud/`。
- 测试：新增标准样本回归、静态边界检查、各阶段对照测试；现有测试全过（miles-next-venv 下排除会拉起本机 Ray 的测试）。
- 不需要 GPU；不碰云。若某阶段需要 GPU 证据，另经用户批准。
- 下游：`rl-verl-backend` 依赖本 change 的阶段 0–3、5（见上）。
