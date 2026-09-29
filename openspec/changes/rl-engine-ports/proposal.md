# Proposal

## Why

Yeto 的 RL 路径目前把 Miles 当成一个整体来调用：`yeto/rl/learner.py` 拼接约 600 行 Miles 参数，然后在进程内执行 `agentenv/miles` fork 的 `train.py` 循环。yeto 的逻辑只能通过 `--external-policy-sync-path` 回调插进去。这个 fork 固定在 2026-07-26 分叉出来的 `e2ad83d8`，比 upstream `radixark/miles` 落后 1000 多个提交，其中带着 31 个 yeto 专属提交，还靠 git bundle 补齐。

接下来的岛内弹性 GPU（`rl-infra-spec`）和各类 RL 算法优化，都需要 yeto 掌控执行循环，并按角色操作 rollout 和 trainer。如果继续在旧 fork 上叠加改动，这些新功能会写进上游已经重写过的模块，以后就再也无法 rebase。

因此需要先做一次"只换底座、不改行为"的重构（R0）：
- 把 yeto 与 RL 引擎之间的边界抽象为一组按角色划分的端口；
- 在最新 upstream Miles 的固定版本上实现这组端口；
- 与现有路径逐项做等价性验证。

## What Changes

- 新增 `yeto/rl/engine/` 包，按角色定义 5 个端口和 1 个算法描述：`RolloutPool`、`TrainerGroup`、`PolicyState`、`Publisher`、`Placement`、`AlgorithmSpec`。端口的返回类型复用 `yeto/rl/contracts.py` 已有的契约类型。
- 新增 yeto 自有的 `IslandDriver`，R0 只实现 `colocated-serial` 执行模式。它通过端口完成 rollout、训练、同步和发布的循环，现有的 strict-avg / decoupled bridge 直接在 driver 的安全边界上调用。
- 新增 `MilesAdapter`，把端口实现在 yeto 自有 fork `michaellchung/miles` 的一个固定 commit 上。该 fork 以最新的 upstream `radixark/miles`（`9e4260d`）为基底，只携带少量兼容提交，全部推送到 fork 仓库，不再使用 git bundle。
- SGLang 同样改为 yeto 自有 fork `michaellchung/sglang`：以最新 Miles 使用的 `sgl-project/sglang` `sglang-miles` 分支为基底，移植 `agentenv/sglang` 的全部 5 个 LoRA/TMS 补丁并逐个验证。
- 在途 PR 的处理写入计划：先合入 #64、#65、#59，并确认 #66 的状态，然后才开始 R0；这些 PR 的语义进入迁移清单，由等价性验收逐条覆盖。Miles 的 cell 结构只在适配层内部使用；不启用 FT、indep-DP、healing，也不启用 FT api_server。
- 新增启动参数 `--rl-engine {legacy,ports}`，默认 `legacy`。旧路径（旧 fork、bundle、`MilesPolicySync` 回调）完全不变，作为兜底和对照组。
- 新增等价性验收：同一模型、数据和硬件下，分别跑 `legacy` 和 `ports`，按四层口径判定：第 1 轮严格相等（grad_norm 相对差 ≤ 3%）、legacy rollout 回放的 teacher forcing（loss、grad_norm 与 optimizer 之前的 LoRA 梯度判定，梯度拼接后相对 L2 ≤ 3% 且余弦 ≥ 0.99；LoRA 更新量只报告）、第 2 轮起按 seed 汇总后的置换检验、路径内各岛策略 hash 一致（见 design D12）。验收通过后，按任务清单将默认值切换为 `ports`，再删除 legacy 路径及 `--rl-engine` 参数。**BREAKING**：删除发生在本 change 的最后一个阶段，届时旧 fork pin、bundle 和 `--external-policy-sync-path` 集成都会移除。
- 端口要保留已合并 PR 的语义：
  - #64 的 LoRA 梯度流不变量；
  - #65 的 GDN hybrid recipe；
  - #59 的数据列命名；
  - capability 声明与 #66 弹性基准 harness 的 capability 格式对齐。
- 更新 `docs/MILES_RL.md` 的边界描述：Miles 由 yeto driver 按端口驱动，不再运行 Miles 的 `train.py`。

## Capabilities

### New Capabilities

- `rl-engine-ports`：yeto 与 RL 引擎之间按角色划分的端口契约，包括：
  - 5 个端口和 `AlgorithmSpec` 的行为；
  - 数据面不经过 yeto；
  - 统一的可训练状态格式；
  - 显式放置；
  - 能力声明，不支持时拒绝启动（fail closed）；
  - 端口级不变量。
- `rl-engine-selection`：`--rl-engine` 的选择语义、两条路径的隔离与版本固定、等价性验收门槛，以及 legacy 路径的退役条件。

### Modified Capabilities

无。`openspec/specs/` 目前没有主 spec。

## Impact

- **代码**：
  - 新增 `yeto/rl/engine/`；
  - `yeto/rl/learner.py` 增加一个分发点，并把 Miles 参数映射拆成端口配置；
  - `yeto/rl/__init__.py` 增加新 pin 常量；
  - `yeto/launcher.py` 和 `yeto/rl/ssh_harness.py` 的远端准备脚本需要能按选择的 engine 准备对应的 Miles 源码。
- **依赖**：
  - 新增 `michaellchung/miles` 与 `michaellchung/sglang` 两个 fork 的固定 commit；
  - legacy 路径在退役前保持现有依赖不变（`agentenv/miles` 加 bundle、`agentenv/sglang`）。
- **Miles 侧**：兼容提交控制在最少。已知必需的是 actor 上一个通用插件调用入口（约 20 行）。每个兼容提交都附上"是否提交给 upstream"的判断。
- **在途 PR**：#64（梯度流修复）、#65（GDN recipe）、#59（数据列名）、#66（弹性基准）的处理方式见 design D10。
- **不在本 change 范围内**：
  - 岛内弹性（E1–E3，由 `rl-infra-spec` 承接，以本 change 为前提）；
  - 新算法；
  - 分区执行模式；
  - DSV4 monkey-patch、SAO、dense-full 的迁移。这些继续留在 legacy 路径，但 `PolicyState` 的布局设计从一开始就覆盖 LoRA 和全参。
- **不受影响**：Rust syncer 和 wire protocol、SFT/diffusion/MLX learner、本地 PPO。
