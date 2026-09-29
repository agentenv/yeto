# Design

## Context

动机见 proposal.md，行为契约见 `specs/rl-engine-ports` 与 `specs/rl-engine-selection`。本节只列出会影响实现方式的现状。

**legacy 路径（源码已确认）**

```
run_miles (yeto/rl/learner.py:1605)
  -> build_miles_argv (learner.py:828)              约 600 行参数映射
  -> MilesPolicySync 通过 --external-policy-sync-path 挂入 (yeto/rl/miles.py)
  -> from train import train                        agentenv/miles@e2ad83d8 fork 的 train.py
  -> asyncio.run(miles_train(...))                  循环归 Miles；yeto 只能在四个回调里插逻辑
```

**upstream `radixark/miles@9e4260d`（2026-09-27）的结构**

`train.py` 只有 175 行，由几个可复用的库函数组装而成：
- `create_rollout_components(args) -> (inference_controller, rollout_executor, ...)`
- `create_training_models(args, rollout_executor) -> (actor_model, critic_model)`
- `update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=...)`
- `rollout_executor.get(rollout_id) -> rollout_data_pack`
- `actor_model.train(rollout_id, rollout_data_pack)`
- `inference_controller.offload / onload_weights / onload_kv`

`TrainGroup` 内部总是由 `TrainerCell` 组成；不开 indep-DP 时只有一个 cell。`TrainerCell.execute(fn_name)` 只能调用 actor 类上已有的方法。

upstream 没有外部策略同步回调，也没有内存形式的可训练状态导出与导入。LoRA 已重构到 `megatron_utils/lora/`，现有的导出方式只有写盘的 `export_slot(path)`。

upstream 提供了 30 多个 `--*-path` 扩展点。与 R0 相关的有：`--rollout-function-path`、`--custom-rm-path`、`--dynamic-sampling-filter-path`、`--rollout-all-samples-process-path`、`--eval-function-path`、`--custom-generate-function-path`、`--custom-loss-function-path`、`--custom-megatron-init-path`。与后续 E1 相关的是 `--custom-inference-engine-provider-path`。

**yeto 已有的可复用部件**
- `yeto/rl/contracts.py`：`LocalStepReceipt`、`TrainerUpdateManifest`、`InferencePublicationManifest`、`TrajectoryEnvelope`，均与算法无关。
- `yeto/rl/core.py`：`CanonicalLoraState`、policy snapshot 与 hash。
- `yeto/rl/bridge.py`：strict bridge 与 `IslandRuntime` Protocol。
- `yeto/rl/decoupled.py`：decoupled 状态机。

**在途 PR（将先合入 main）**
- #64：LoRA 梯度 hook 修复与 grad_norm 不变量。其 Miles 侧修复位于 fork 独有的 `trainable_state.py`，upstream 没有这个函数。
- #65：GDN hybrid recipe 参数。
- #59：数据列名。
- #66：弹性基准 harness，其中包含 capability 认证。

## Goals / Non-Goals

**Goals**
- yeto 持有岛内执行循环。Miles 作为库使用，并且只通过端口访问。
- 端口粒度采用按角色划分（B-2），并且长期保持不变。后续的弹性（E1–E3）与算法优化只在各端口上**增加动词**，不改变端口划分。
- Miles 侧改动为零，或只有一个可以提交给 upstream 的极小补丁。
- 新旧路径可以在同一份 yeto 代码中对照运行。

**Non-Goals**
- 不实现分区执行、重叠、弹性、journal 或 ReconfigurationCut。这些属于 `rl-infra-spec`，本 change 只保证端口能承载它们。
- 不迁移 SAO、dense-full、DeepSeek V4 patch、critic 或本地 PPO。
- 不修改 Rust syncer 或 wire protocol。
- 不引入 Miles 的 FT 能力（indep-DP、healing、api_server、mini_ft_controller）。

## Decisions

### D1. 按角色划分的端口，driver 归 yeto

```
 yeto/rl/engine/
 +---------------------------------------------------------------+
 | driver.py      IslandDriver (colocated-serial)                 |
 |                  bridge 在安全边界调用: strict / decoupled      |
 | ports.py       RolloutPool  TrainerGroup  PolicyState          |
 |                Publisher    Placement     AlgorithmSpec        |
 |                EngineCapabilities                              |
 +-------------------------------+-------------------------------+
                                 |
 +-------------------------------v-------------------------------+
 | miles_adapter/  (依赖最新 upstream pin)                        |
 |   rollout.py   -> inference_controller + rollout_executor     |
 |   trainer.py   -> actor_model (单 cell TrainGroup)            |
 |   state.py     -> actor 内 plugin: export / apply              |
 |   publish.py   -> miles update_weights + 成员/payload 清单     |
 |   placement.py -> create_placement_groups 结果的只读描述        |
 |   config.py    -> RL 配置 -> Miles args                        |
 +---------------------------------------------------------------+
```

端口使用 `typing.Protocol` 定义。返回类型优先复用 `contracts.py`：

| 端口 | 返回类型 |
|---|---|
| 训练步 | `LocalStepReceipt` |
| 发布 | `InferencePublicationManifest`（`publication_mode="full"`），外加成员集合 |
| 可训练状态 | 从 `CanonicalLoraState` 泛化出的 `TrainableState`，带 layout 字段 |

**考虑过的替代方案**
- **A：只加钩子，循环归 Miles。** 改动最小，但重建 actor 后 driver 仍持有旧 handle，而且分区与重叠模式无法表达。弹性阶段必然需要再拆一次。
- **C：声明式 reconcile，Miles 负责收敛。** Miles 会变成黑盒，并且 trainer 侧的收敛逻辑很难实现。
- **B-1：只提供四个动词。** 到 E1 就得拆。

最终选 B-2。driver 以后可以从简单循环演进为调度器，端口不需要改变。

### D2. driver 复用 upstream 的库函数，不复制 train.py

`IslandDriver` 直接调用 D1 列出的 upstream 函数完成初始化与每一轮循环。offload/onload 的时序按 upstream `train.py` 的串行共置分支编写，并通过 `Placement` 描述切换共置和固定分区。

eval 使用 upstream 的 `EvalDispatcher`。save 与 checkpoint 仍然遵守 yeto 现有语义：不保存 optimizer 和 RNG，island 只保存重建进度。

driver 自身只负责三件事：
1. 编排上述调用顺序；
2. 在安全边界调用 bridge；
3. 做 policy 身份校验与事件记录。

**替代方案**：在 yeto 中逐行复刻 upstream 的 `train.py`。这会导致上游修复无法自动获得，因此否决。

### D3. 数据面：`RolloutBatchHandle` 包装 `rollout_data_pack`

- rollout 端口返回的句柄持有 upstream 的 `rollout_data_pack`（Ray ObjectRef 或 object store 引用）。
- yeto 需要的元数据由 `--rollout-all-samples-process-path` 回调在 rollout 进程内提取，只以小对象的形式回传：group、sample 标识，policy token，reward 摘要和计数。
- trainer 端口直接把句柄传给 `actor_model.train`。
- 训练结束后，由 driver 调用 upstream 的 `remove_rollout_data_refs` 释放引用。

policy token 沿用 yeto 现有格式 `yeto:<rollout_id>:<policy_hash>`，经由 SGLang 的 weight version 写入每个样本。

### D4. PolicyState：actor 内插件，外加一个通用调用入口

实现可训练状态的导出与导入，必须在每个 Megatron rank 的 actor 进程内执行代码。upstream 的 `execute(fn_name)` 只能调用 actor 类上的方法。候选方案如下：

| 方案 | 结论 |
|---|---|
| a. 给 Miles 的 train actor 加一个通用方法 `run_plugin(fn_path, **kwargs)`，用 `load_function` 加载函数并以 `(actor, **kwargs)` 调用；`TrainGroup` 加一个对应的透传方法 | **采用**。约 20 行，与 Miles 现有的 `--*-path` 风格一致，可以提交给 upstream |
| b. 用 `--custom-megatron-init-path` 在 actor 内挂接，再另开一条通信通道 | 否决。需要新建 IPC，而且与 Ray 的生命周期脱节 |
| c. 替换 actor 类 | 否决。upstream 不支持配置 actor 类 |

导出和导入的实现本身放在 yeto（`miles_adapter/state_plugin.py`），从 fork 的 `trainable_state.py` 移植到 upstream 新的 `megatron_utils/lora/` 结构上。

**梯度流不变量（来自 #64）：**
- 不能跨 dtype 赋值 `Parameter.data`。需要用 FP32 master 作为模块参数时，应在 `module._parameters[name]` 中临时放入一个共享 master 存储、且 `requires_grad=False` 的新 Parameter，结束后还原。
- #64 的 CPU 测试 `test_rl_grad_accumulator_hook` 原样保留，作为端口契约测试。
- grad_norm 的不变量由 driver 在每轮训练之后检查。

这个 Miles 补丁只作为 `michaellchung/miles` 上 `yeto/ports` 分支的一个提交携带（见 D6），**不向官方 `radixark/miles` 提交 PR**（`sgl-project/sglang` 同理）。签名为 `run_plugin(fn_path, kwargs: dict | None)`：upstream 的 rpc 层不允许公开 worker 方法使用 `*args/**kwargs`，因此参数以显式 dict 传入；在 `--worker-comm-backend rpc` 下 kwargs 与返回值须可 JSON 序列化。以后升级 Miles 基底时，如果 upstream 已有等价能力，就改用它并删掉这个提交。

**从 `agentenv/miles` 复刻兼容思路的原则。** 该 fork 的 31 个提交按类别处理：

| 类别 | 在 `michaellchung/miles` 上的处理 |
|---|---|
| 外部策略同步回调（`external_policy_sync`，train.py 约 450 行差异） | **不复刻**。循环已经归 yeto driver |
| `trainable_state` 导出/导入 | **不复刻进 Miles**。改为 yeto 侧插件，借助通用入口执行 |
| LoRA 搬运与显存修补（TP/PP 状态、colocate reload、IPC 回收、bucket 传输，约 15 个） | 不预先移植。等价性实验里**复现到问题再移植**，每个修复单独成提交 |
| 端口隔离（`--train-master-base-port` 等） | 多个 island 共用一台机器时需要；upstream 若无等价参数则移植 |
| SGLang 版本对齐（约 7 个） | 不移植。由 D11 的 SGLang fork 取代 |
| TITO Qwen3.8 | 检查 upstream TITO 重构后是否已覆盖；未覆盖才移植 |

每个移植提交的消息里都注明来源提交（`agentenv/miles@<sha>`）和移植理由。

### D5. cell 只存在于适配层内部

- 适配层创建 `TrainGroup` 时强制使用单 cell：不设置 indep-DP 相关参数，启动前断言只有一个 cell。
- 适配层不启动 `maybe_start_api_server` 或 `maybe_start_mini_ft_controller`。
- rollout 侧通过 `InferenceController` 管理 engine，因为这是 upstream 的默认实现。端口签名中不出现 cell、`CellStatus` 这类类型。

### D6. 新旧版本并存

**仓库与分支**

```
 radixark/miles@9e4260d (upstream, 2026-09-27)
   |
   +-- michaellchung/miles  main = upstream 镜像，不直接提交
                             yeto/ports  = 9e4260d + 兼容提交 (run_plugin, 按需移植)
                                            ^ MILES_NEXT_COMMIT 固定在这个分支的某个 commit
```

- `main` 只用于同步 upstream，不直接提交。
- 兼容提交全部放在 `yeto/ports` 分支上。
- 升级 upstream 时（这不在本 change 范围内，按需进行），把 `yeto/ports` rebase 到新的 main，重新跑端口契约测试和等价性测试，再更新 `MILES_NEXT_COMMIT`。
- 只要不主动升级，就一直使用同一个固定版本。

`yeto/rl/__init__.py` 增加一组 `MILES_NEXT_*` 常量：repository 为 `https://github.com/michaellchung/miles`，另有 commit 和源码 hash。SGLang 使用另一组 `SGLANG_NEXT_*`（见 D11）。旧的 `MILES_*`、`SGLANG_*` 常量保持不变。

远端准备脚本（`launcher.py` 中的 miles_setup，以及 `ssh_harness.py` 中对应的部分）按 `--rl-engine` 选择 checkout 目标：
- **legacy**：保持现有流程（agentenv fork、`MILES_BASE_COMMIT`、bundle、`MILES_COMMIT`）。
- **ports**：直接 `git fetch` fork 仓库上的固定 commit，不使用 bundle。校验步骤不变：origin、commit、detached、工作区干净、导入路径。

`verify_miles_revision` 改为接收期望常量组作为参数，两条路径共用同一套校验逻辑。

**镜像**：ports 路径的容器镜像以 upstream 的 Dockerfile 为基础构建（base 为 `lmsysorg/sglang:v0.5.20`，Megatron 与 Megatron-Bridge 使用 upstream 固定的 radixark 分支）。SGLang 与 Miles 源码在运行时按上述常量 checkout 并覆盖。镜像 digest 作为 `MILES_NEXT_IMAGE` 固定下来。

### D7. AlgorithmSpec 映射到 Miles 的算法扩展点

`AlgorithmSpec` 是 yeto 侧的冻结数据类，包含 advantage 估计方式、loss 选择与参数、过滤器与上限，以及 KL 等系数。它只在 `config.py` 中被翻译成 Miles 参数（`--advantage-estimator`、`--custom-loss-function-path`、`--dynamic-sampling-filter-path` 等）。

yeto 现有的有界零方差过滤（`yeto.rl.filters.bounded_nonzero_reward_std`）继续通过 Miles 的过滤扩展点挂接。`AlgorithmSpec` 的规范化 JSON 的 SHA256 写入事件和来源记录。

### D8. 配置映射拆分

`build_miles_argv` 拆成两层：
1. **引擎无关的 RL 运行配置**：模型结构、LoRA、批大小、数据、recipe（含 #65 的 GDN 层规格）、#59 的数据列名。
2. **各引擎自己的参数翻译**：legacy 与 ports 各有一份。

两条路径从同一个 parse 结果出发。这样等价性对比时输入确定相同，也避免 #59、#65 这类改动需要在两处重复。

`ports` 翻译层遇到未映射的配置项时拒绝启动（见 spec）。

### D8b. 能力声明的格式与 #66 对齐

`EngineCapabilities` 的字段与序列化格式，以 #66 合入后 `yeto/rl/elastic_benchmark/capabilities.py` 的认证格式为准。如果两者需求有差异，修改 #66 那一侧，保证只存在一套 capability schema。

### D9. 以后的弹性控制器放在 yeto 侧（记录，不在本 change 实现）

`rl-infra-spec` 中岛内重配置的决策、护栏与 journal 都位于 yeto 侧，每个 learner island 一份，由 `IslandDriver` 在安全边界执行。Miles 只通过端口新增的动词提供机制：
- `RolloutPool.add/remove_engines`
- `TrainerGroup.save_cut/restore_cut`
- `Placement.reconfigure`

本 change 的端口签名需要为这些动词预留位置，但不实现。`docs/MILES_RL.md` 中"不做 controller"一条的修改留到 E2 再定。

### D10. 在途 PR 的处理

main 上的 RL 相关 PR（全部由 michaellchung 提交），按与 R0 的关系分类处理：

| PR | 内容 | 与 R0 的关系 | 处理 |
|---|---|---|---|
| #64 | LoRA 梯度 hook 修复与 grad_norm 不变量；bundle 升级到 `ae475060` | 高：修改 `rl/miles.py`、`core.py`、pin 和 bundle | **R0 开始前合入**。其 Miles 侧修复只进 legacy bundle；ports 路径通过 D4 的实现约束和 CPU 测试继承这条不变量 |
| #65 | GDN hybrid 走 Qwen3.5 layer spec | 中：修改 `rl/learner.py` 参数映射 | **R0 开始前合入**。任务 2.4 拆分配置时进入引擎无关层 |
| #59 | 数据列名与 MATH-500 示例 | 中：修改 `rl/learner.py`、`cli.py`、`launcher.py` | **R0 开始前合入**。数据列名进入引擎无关层 |
| #66 | 弹性基准 harness | 概念重叠（capability 格式） | 可以先合也可以后合。D8b 以它的 capability 格式为准 |
| #58 #60 #61 #62 #63 | CLI、云接入、launcher 的 sky/ray 环境 | 低 | 随时合入。#62、#63 改的是 RL 岛的 Ray/env 准备，与 1.5 的准备脚本相邻，合入后需要复核 1.5 的快照测试 |

其他作者的 PR（#43 已失效，建议关闭；#42、#12、#8 与 R0 无重叠）不纳入迁移清单。

**迁移清单**：`openspec/changes/rl-engine-ports/migration-ledger.md`，由任务 0.2 创建，格式见 `rl-engine-selection` spec。R0 开发期间，每合入一个 RL PR 就登记一项。任务 7.1 前检查所有项都已关闭。

**分支策略**：只有一个开发者，因此不建集成分支。#64、#65、#59 合入后从 main 开 `rl-engine-ports` 分支，每组任务单独提交 PR，合回 main。legacy 行为由快照测试保护，所以每个 PR 都可以单独合入。

### D11. SGLang：运行时由 Miles 调用，版本由 yeto 管理

yeto 与 SGLang 的关系：
1. yeto 固定并安装 SGLang 源码（`launcher.py` 中的 `sglang_setup`）。
2. yeto 通过 `--sglang-*` 参数配置，由 Miles 启动 SGLang engine（Ray actor）。
3. 生成请求、权重和 LoRA 更新都由 Miles 发给 SGLang。
4. yeto 直接调用 Miles 返回的 engine handle 写入 policy token（`update_weight_version`）。
5. 仅在 DSV4 场景下，`sitecustomize.py` 在 SGLang 进程内 monkey-patch 其内部模块。

在 ports 路径中，第 4 点改由 `Publisher` 端口负责，yeto 不再直接调用 engine handle。第 5 点仍只属于 legacy 路径。

**fork**：新建 `michaellchung/sglang`，fork 自 `sgl-project/sglang`。以 D6 固定的 Miles 所使用的 `sglang-miles` 分支上的 commit 为基底，开出 `yeto/ports` 分支，移植 `agentenv/sglang` 的全部 5 个补丁：
- `b34df47`：请求 abort 时释放 LoRA 引用
- `95d4d69`：tensor LoRA 请求格式与 Miles 对齐
- `8db3711`：支持多 bucket 的 tensor LoRA payload
- `c2cb40a`：engine 保留多 bucket payload
- `e1b57eb`：weight memory saver 支持落盘

每个补丁单独成一个提交：
- 连同原有测试一起移植；
- 如果上游已有等价实现，就在提交说明中记录，并保留 yeto 侧的测试作为回归测试。

`95d4d69` 的"与 Miles 对齐"部分需要按 upstream Miles 当前的 LoRA 请求格式重新对齐，不能照搬。

### D12. 等价性验收采用分层口径

原口径（"容差 = legacy 重复 3 次的噪声 × 2"）不可用：legacy-baseline-v2 显示 legacy 3 次运行逐位一致，噪声为 0，容差退化为 1e-6 下限，与两条路径真实的数值差无关。同一证据还显示：第 1 轮两条路径的 reward、token、group 计数与 loss 完全一致，grad_norm 相对差 0.6%（岛 0）与 1.0%（岛 1），属于 bf16 与不同 kernel 路径的数值差；第 2 轮起策略已有微小差别，采样分叉并被放大（legacy 岛 0 第 2 轮 reward 全为 0，ports 不是），逐点容差没有意义。因此改为四层（阈值为 `scripts/rl_engine_equivalence.py` 的常量，写入报告头，不随 ports 结果调整）：

| 层 | 比较对象 | 判定 | 阈值依据 |
|---|---|---|---|
| 1 第 1 轮严格 | 主 seed 第 1 轮 | group/token/reward 相等；grad_norm 相对差 ≤ 3% | 相同权重、相同 prompt 下计数已观测到完全相等；3% 为观测最大值 1.0% 的 3 倍 |
| 2 teacher forcing | legacy 第 1 轮 rollout 回放到两条路径各训一步 | loss 绝对差 ≤ max(1e-6, 1e-3×\|legacy\|)；grad_norm 相对差 ≤ 3%；optimizer 之前的 LoRA 梯度拼接后相对 L2 ≤ 3% 且余弦 ≥ 0.99；LoRA 更新量只报告 | 见下（梯度口径为 2026-09-29 实验后经用户批准的修改） |
| 3 分布 | 第 2 轮起，每 seed 每指标取各（岛, 轮）平均；legacy/ports 各 5 个 seed（17–21） | 精确双侧置换检验（252 种划分，均值差），任一指标 p < 0.05/4 = 0.0125 失败 | 用户选定；Bonferroni 控制 4 个指标的全体误判率 ≤ 5% |
| 4 hash | 每条路径每个 seed 内 | 同一版本各岛 hash 一致；不跨路径比较 | 两条路径数值不同，跨路径 hash 必然不同 |

第 2 层阈值的依据与限制：

- loss：on-policy GRPO 第一步的 loss 由组内零均值 advantage 相消，量级约 1e-8（两条路径观测值都是 1.2107e-08），相对差没有信息量。绝对下限 1e-6 高出该量级两个数量级，能拦住 KL/entropy 项或 loss 聚合方式配置错误这类会让 loss 离开零点的错误；loss 不是这一层的主要信号。
- grad_norm：与第 1 层相同的 3%。输入逐 token 相同，差异只来自 trainer 数值，不应大于第 1 轮自由采样时的观测值。
- LoRA 梯度（判定）：两条路径在 `optimizer.step()` 入口导出全部 LoRA 张量的梯度（`yeto.rl.grad_audit`，`YETO_RL_AUDIT_GRADS=1` 启用，写到 `audit/round-00000001.grad.f32` 与索引 `.grad.json`）。语义两边一致：forward/backward 与 Megatron `finalize_model_grads` 之后（DP all-reduce 已完成；两边都开 `--accumulate-allreduce-grads-in-fp32`，取 FP32 `main_grad`，没有才退回 `param.grad`），除以 optimizer 的 loss scale（bf16 无 scaler 时为 1），**裁剪之前**的原始梯度；Megatron 将施加的裁剪系数 `min(1, clip_grad/(‖g‖+1e-6))` 与 `optimizer.step()` 返回的 grad_norm 一并写入索引。张量经 Megatron-Bridge 的 adapter 转换任务变换为 canonical HF 形状，并归一化为 canonical PEFT 名（`base_model.model.*.lora_{A,B}.weight`），按名字排序拼接。判定：按名字一一对齐（名字或形状不一致即失败），拼接后 `‖g_p − g_l‖₂ / ‖g_l‖₂ ≤ 3%` 且余弦 ≥ 0.99，报告列出相对 L2 最大的 5 个张量。3% 与 grad_norm 的阈值相同：输入逐 token 相同，逐元素梯度差只来自 trainer 数值。挂点：ports 由 `state_plugin` 已有的 `train_one_step` 记录器在每步武装一次性的 `optimizer.step` 包装；legacy 由 learner 设置 Miles 的 `--custom-megatron-before-train-step-hook-path yeto.rl.grad_audit.before_train_step`（fork 已有该钩子，不改 fork 代码），两边调用同一个 `grad_audit.arm`。只支持 DP 复制的梯度：TP/PP/EP > 1 或 DP > 1 且使用分布式优化器（`main_grad` 只在本地分片有效）时拒绝运行；teacher forcing 每岛 1 GPU。
- LoRA 更新量（只报告）：`‖Δ_ports − Δ_legacy‖₂ / ‖Δ_legacy‖₂`，Δ 取自两条路径共用的每轮审计文件 `audit/round-00000001.delta.f32`（canonical 布局，同一 `layout_hash`），同时报告余弦、符号翻转比例（`sign(Δ_p) ≠ sign(Δ_l)` 的元素比例）与范数比 `‖Δ_p‖/‖Δ_l‖`。前提 `base.f32` 一致（相对 L2 ≤ 1e-6）仍为判定项。
- 口径修改记录（2026-09-29，实验后依据数据、经用户批准）：原口径以"LoRA 更新量相对 L2 ≤ 5%"判定，并写明"修改该阈值必须先修改 spec 并附数据"。strict-avg teacher forcing（证据 `gpu-eq/evidence/2026-09-29-eq62/report.md`）中 loss 与 grad_norm 通过（grad_norm 相对差 0.56% / 1.06%），初始 LoRA 一致（相对 L2 = 0），但更新量相对 L2 为 0.333 / 0.424（余弦 0.944 / 0.910），范数比 1.000，符号翻转比例 1.5% / 2.5%。原因分析：Adam 第一步 `m̂/√v̂ = g/|g| = sign(g)`，更新量 ≈ `lr·sign(g)`，每个元素的幅度都是 lr、与梯度大小无关，所以范数比恒为 1；两条路径 bf16/kernel 差异只让 |g| 接近 0 的元素改变符号，而每个翻转元素贡献 `(2·lr)²`，于是相对 L2 ≈ 2√(翻转比例)：√0.015×2 ≈ 0.245、√0.025×2 ≈ 0.316，与观测的 0.333 / 0.424 同量级（其余差来自 ε 附近的非饱和元素）。也就是说第一步的更新量把梯度的幅度信息丢掉、只放大符号噪声，不能反映 trainer 误差；optimizer 之前的梯度才是 trainer 数值的直接度量。因此判定对象改为梯度（相对 L2 ≤ 3%、余弦 ≥ 0.99），更新量的四个统计改为只报告。这是一次实验之后的口径修改，不是放宽同一指标的阈值；新口径同样在 ports 梯度数据产生之前固定。
- 回放入口 `yeto.rl.teacher_forcing.replay_generate` 作为 Miles 的 custom generate 函数，在两条路径的正常流程中用记录的 token、logprob 和 reward 替换采样，并按各自路径重打策略 token；记录中没有的 prompt 直接报错，不回退为真实采样。回放运行本身的 reward/token/group 计数必须与原运行相等，作为回放复现的前提检查。

第 3 层的统计局限：每个 seed 先汇总成一个值，避免逐（岛, 轮）检验的多重比较和轮间相关；两条路径同分布时，全体误判率由 Bonferroni 控制在 5% 以内。代价是检出力很低：5 对 5 时共 252 种划分，最小可达 p = 2/252 ≈ 0.0079，只有当两组值完全分离（一方全部大于另一方）时才可能低于 0.0125；按高斯近似，要有 80% 的检出力，均值差需约 3 个合并标准差。因此该层通过只说明没有发现大的分布偏移，不能证明差异小。报告给出各 seed 原始值、p 值、效应量与上述局限，效应量较大但未达显著时应人工复核；若要提高检出力，只能增加 seed，并先改 spec。

decoupled：部分 fragment 应用会把全局 fragment 混入本岛进度，各岛 hash 本来就不同，所以第 4 层只比较非部分应用的版本与最终 cut；decoupled 的 teacher forcing 同样由 `YETO_RL_AUDIT_GRADS=1` 写出梯度审计文件并按同一梯度口径判定；若未写出，第 2 层的梯度判定取自 strict-avg 的 teacher forcing（trainer 与 optimizer 配置相同）；最终 LoRA 的跨路径相对 L2 距离只报告，并附 legacy 跨 seed 的同一距离作为参照，不设硬阈值，因为第 2 轮起采样分叉，该距离主要反映轨迹差异，而非 trainer 误差。

## Risks / Trade-offs

- [upstream 库函数不是稳定 API，版本升级时签名可能变化] → 只在固定版本上开发，所有调用都集中在 `miles_adapter/` 中；升级 pin 时只需要改适配层，并用端口契约测试兜底。
- [D4 的补丁没有被 upstream 接受] → 长期保留在 `michaellchung/miles` 的 `yeto/ports` 分支上。补丁很小，跟随 rebase 的成本低。
- [fork 分支上的兼容提交越积越多，又变成第二个 agentenv/miles] → 按 D4 的表格逐项判断，不预先移植；每个提交注明来源和"是否提交给 upstream"。rebase 时如果冲突提交超过 5 个，就先开 change 评审。
- [SGLang 补丁在 v0.5.20 上无法简单 cherry-pick] → 逐个补丁移植并附带测试；`95d4d69` 按 upstream Miles 的请求格式重写，由任务 1.3 的真实 LoRA 发布测试兜底。
- [upstream 的 LoRA 实现与 fork 不同，导出的张量名或形状与 canonical PEFT 名称不一致] → 以 `canonical_layout_hash` 做对比；导出结果先与 legacy 路径在同一 checkpoint 上逐张量比对（任务 3.4）。
- [SGLang 新版本不再需要旧补丁，或者需要新的补丁] → 任务 1.3 先在真实环境中验证 LoRA 张量发布，再确定 pin。
- [两条路径的 GPU 显存行为不同，导致同一配置在其中一条上 OOM] → 等价性实验记录峰值显存；ports 路径可以单独调整 offload 参数，但必须写入报告。
- [并存期代码量翻倍，维护负担加重] → 退役条件写入 spec；等价性验收通过后，尽快删除 legacy 路径。

## Migration Plan

1. 合入 #64、#65、#59（以及可选的 #66）之后，从 main 开分支。
2. 按 tasks 的顺序实现，保持默认值为 `legacy`。每一步都可以单独合入，legacy 行为不受影响。
3. 分别对 strict-avg 和 decoupled 完成等价性验收，并提交报告。
4. 把默认值切换为 `ports`，legacy 保留至少一个真实运行周期。
5. 删除 legacy 路径（旧 pin、bundle、`MilesPolicySync` 回调集成、`--rl-engine`），同步更新 `docs/MILES_RL.md`。此前仅 legacy 支持的功能，要么已经迁移，要么在文档中标注为暂不支持。

**回滚**：在第 5 步之前，传 `--rl-engine legacy` 即可回到旧路径；第 5 步之后，通过 git revert 回滚。

## Open Questions

- D4 的通用插件入口采用什么名称和签名，以 upstream review 的意见为准，不影响 yeto 侧的端口。
- ports 路径最终固定 upstream 的哪个 commit：以任务 1.1 执行当天的最新 commit 为起点，只要通过冒烟测试就固定下来，之后不再追踪。
