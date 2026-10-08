# yeto/rl/engine：核心与适配层的约定

本目录是 yeto RL 引擎。约定（yeto-framework-decoupling，design D2/D4）：**`engine/` 下除适配层（`miles_adapter/`，以后的 `yeto/rl/adapters/<后端>/`）以外都是核心**。核心只认 `ports.py` 定义的五个角色端口（`RolloutPool`、`TrainerGroup`、`PolicyState`、`Publisher`、`Placement`）与中立数据（批次句柄不透明、`TrainableState`、`AlgorithmSpec`、`ExecutionProfile`），不碰具体训练框架。

## 边界规则（由 `tests/test_import_boundaries.py` 自动检查）

1. 中立核心——`yeto/rl/engine/`（适配层除外）、`yeto/rl/rewards/`、`yeto/rl/harness/`、`yeto/rl/algos/`，以及暂时还放在 `yeto/rl/` 下的奖励/过滤/算法扩展/harness 文件（清单见测试中的 `CORE_FILES`）——不得 import：
   - 训练或推理框架：`miles`、`miles_plugins`、`megatron`、`sglang`、`verl`、`vllm`、`torch_npu`；
   - 后端适配层：`yeto.rl.engine.miles_adapter`、`yeto.rl.adapters.*`；
   - 旧版 Miles 引擎与补丁模块：`yeto.rl.miles`、`yeto.rl.miles_overlay`、`yeto.rl.overlays`、`yeto.rl.learner`、`yeto.rl.miles_full_parameter*` 等；
   - 云库：`sky`、`modal`、`yeto.modal_runner`、`yeto.shape.providers`。
2. 各后端适配层之间不得互相 import。
3. 云层/启动层（`launcher.py`、`modal_runner.py`、`cli.py`、`stage_w_entry.py`、`yeto/shape/`、以后的 `yeto/cloud/`）不得 import 后端适配层。

检查只做语法树分析：函数内的延迟 import、相对 import、字面量 `importlib.import_module(...)` 都算。

## 现有违规与白名单

现有违规列在 `tests/import_boundary_allowlist.txt`（文件 + 被禁模块 + 审计编号），**只减不增**：

- 新增一条违规 → 测试失败，打印文件、行号、被禁模块；
- 修掉一条违规但白名单没删 → 测试失败，提示"请删白名单条目"；
- 白名单条数必须等于测试里的 `ALLOWLIST_CAP`，删条目时同步调小。

需要后端能力时，正确做法是：在端口或能力声明（`capabilities.py`）里加中立的可选方法/字段，由适配层实现；不要从核心直接 import 适配层。

## 标准样本

去耦合各阶段合并前，`tests/test_decoupling_golden.py` 必须通过：典型配置下的 Miles 命令行、`AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash`、插件源码哈希、strict/decoupled 进度内容与假引擎 tape 片段与 `tests/golden/decoupling/` 逐字一致。哈希的有意变更记在 `openspec/changes/yeto-framework-decoupling/hash-migration.md`。
