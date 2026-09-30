# 4.1 完整 ReconfigurationCut 状态审计（INFRA-E2，2026-09-30）

范围：rl-infra-spec 4.1（含 alignment A3、F5 的 `carried_over` 核实）与 4.2a 要求的"E2 profile 替代路径"选择。事实均以 Miles fork `yeto/ports` = `0af62f4d` 源码为准（行号引用该提交），yeto 以分支 `infra-e2` 为准。本文只有 CPU 结论，没有 GPU 证据；GPU 验收计划见 `evidence/infra-e2/4.2-4.5/plan-v2.md`。

## 0. 结论摘要

1. **E2 profile 与路径**：沿用 R0 冒烟 profile（bf16 LoRA，DistributedOptimizer 按 Miles 默认），不另选 fp32 profile，也不用 Megatron `torch_dist` 完整 dist-ckpt。cut 由 yeto 自己的 rank 插件（`miles_adapter/cut_plugin.py`，经 `run_plugin`）写入，状态取值复用 fork-M5 的 `dp_invariant_state` 辅助函数（按参数名导出 optimizer 状态，含 FP32 主副本 `param`、moments、step、超参；RNG 含 Python/NumPy/Torch CPU/CUDA/Megatron CUDA tracker）。理由：M5 在 0af62f4d 已支持 bf16 与 DistOpt（alignment §11），因此 4.2a 原文所担心的"bf16 下 M5 不可用"已不成立；而且 yeto 插件不必打开 `--lora-dp-invariant-state`，也不改变 Miles 默认 checkpoint 行为。**风险**：M5 的 bf16/DistOpt 路径在 GPU 上没有验证过，所以本路径属于"已实现、待本地 GPU 验证"。
2. **拒绝的配置**（在写入任何文件之前就拒绝，与 M5 参数检查一致）：`--fp16`（loss scaler 状态不保存）、precision-aware optimizer、`--num-distributed-optimizer-instances>1`、CP>1、EP>1、非 LoRA 的可训练参数、没有 optimizer 或 scheduler 的 trainer。另有两项已知限制，由 E2 边继承：TP/PP 集体导出与 DistOpt 分片主参数不能同时使用（`state_plugin._collective_export` → `master_of` 会报错，恢复后重发权重（4.4）走这条路径），precision-aware optimizer 的分片主参数不支持（`distributed_ranges`）。
3. **`carried_over` 在 Miles ports 路径上恒为 0**（F5 核实，见 §3）。cut 的完整性检查要求 `ledger.carried_over == 0`，如果不是 0 就拒绝。
4. **与默认路径隔离**：ports 引擎一直传 `--no-save-optim --no-load-optim --no-save-rng --no-load-rng --finetune`，不传 `--save/--load`（`miles_adapter/config.py:638-645`）。cut 只能经端口动词 `save_cut/restore_cut` 显式写入或读取，本次没有改动任何默认参数。
5. **缺状态拒绝**：`cut.CutManifest.completeness_problems` 与 `cut_plugin._require/_adapters` 覆盖下表各行，缺任何一项都会拒绝提交 manifest；没有 manifest 的 cut 视为不存在。

## 1. 状态清单（design D5 各行）

| 状态族 | 项 | 来源（Miles/yeto） | cut 中的载体 | 缺失/不一致时 |
|---|---|---|---|---|
| trainer | LoRA adapter 模型副本（bf16/fp32） | `actor.model` 中 `_is_adapter_param_name` 的参数 | 每个 rank 分片的 `adapter` | 名字/shape/dtype 不符即拒绝；存在非 LoRA 可训练参数即拒绝 |
| trainer | FP32 主副本 | Float16Optimizer `fp32_from_float16_groups`，或 DistOpt 的 `_get_main_param_and_optimizer_states(param)["param"]`（M5 `optimizer_slots`） | `optimizer_named.entries[*].tensors.param` | 恢复前先用 `check_named_optimizer_state` 校验，通过后才写入 |
| trainer | Adam moments 与 step | 同上（`exp_avg`、`exp_avg_sq`；标量 `step`） | `optimizer_named` 的 `tensors`/`scalars` | 同上；DistOpt 需要把本 (tp,pp) 的所有 DP 分片合并起来 |
| trainer | 参数组超参 | optimizer param_groups | `optimizer_named.entries[*].hyper` | 同上 |
| trainer | LR scheduler 与计数 | `actor.opt_param_scheduler.state_dict()`（`num_steps` 以样本数计） | `scheduler` | 必须满足 `num_steps == local_step × GBS`，否则拒绝（manifest 与每个 rank 都检查） |
| trainer | loss scaler | 只有 fp16 才有 | — | 拒绝 fp16 |
| trainer | Megatron 全局计数 | `get_args().iteration / consumed_train_samples` | `megatron_counters` | 读不到时写空；恢复时回写 |
| trainer | reference 模型 | `--ref-load`（`config.py:561`），不可变，不训练（`--ref-update-interval` 被拒绝，`algorithm_flags.py:_UNMAPPED`） | 不存权重，只在 manifest `algorithm.ref_model` 记 `{ref_load, base_model_revision}` 身份 | 恢复时与当前运行比对，不一致即拒绝 |
| trainer | old-policy 副本 | `--keep-old-actor` 被拒绝 | — | 不适用 |
| trainer | yeto 进程内记录 | `state_plugin._STEP_*`（每轮都会排空）、`_RECORDER_INSTALLED` | 不存；保存时要求 `_STEP_*` 为空（证明处在 step 边界）；恢复时调用 `install_grad_norm_recorder()` | 不为空即拒绝保存 |
| trainer | colocate CPU 权重备份 | `actor.weights_backuper` | 不存；恢复后执行 `backup("actor")` | — |
| stochastic | RNG | M5 `capture_rng_state`（python、numpy、torch CPU、`torch.cuda.get_rng_state_all`、Megatron tracker） | 每个 rank 分片的 `rng`，策略固定为 `exact` | 只做同形恢复；坐标 (tp,pp,dp,dp_size) 不同即拒绝；恢复后重新采集 RNG，摘要必须等于保存值 |
| stochastic/progress | 数据游标 | Miles `RolloutDataSource.{sample_offset, epoch_id, sample_group_index, sample_index, metadata}`（`data_source.py:128-141`）；shuffle 由 `rollout_seed`+`epoch_id` 决定 | manifest `data` | 四个整数字段缺一即拒绝。**需要 E1 提供读取接口**（§5） |
| stochastic/progress | rollout id、policy version/hash、local_step | yeto driver | manifest `progress` | 恢复时与 driver 当前值比对 |
| stochastic/progress | packing plan | `--balance-data` 由数据决定，无独立状态 | — | — |
| algorithm | `algorithm_spec_sha256` | `AlgorithmSpec.sha256()` | manifest `algorithm` | 不一致即拒绝（4.2 验收） |
| algorithm | 插件 PluginRef 哈希 | spec 内所有 PluginRef 的 `path@sha256` | 同上 | 同上 |
| algorithm | `yeto_algo_plugins` runtime attrs 哈希 | `AlgorithmSpec.to_legacy_runtime_attrs()` 规范化 JSON 的 sha256 | 同上 | 同上 |
| algorithm | 动态过滤、超采样的 rollout 侧状态 | 见 §3：跨轮没有状态 | ledger `carried_over=0`、`ready_unconsumed=0` | 不为 0 即拒绝 |
| algorithm/outer | 外层协议位置 | bridge/sync（进程存活，E2 不重建） | manifest `outer`，必须 `settled: true` | 未确定即拒绝（D5：破坏性切换要求外层提交已确定） |
| rollout | 活跃轨迹、KV | 首版 quiescent cut 要求没有活跃轨迹 | — | 由 3.3/3.8 的 drain 保证 |
| runtime | backend 指纹、布局、精度、DistOpt、RNG 策略、shard schema | args | manifest `runtime` | 布局或指纹与当前不同即拒绝 |
| runtime | 每个分片的 sha256 与字节数 | rank 插件在 fsync 之后计算 | manifest `files` | 截断、校验和不符、缺分片都会拒绝（driver 与 rank 两侧各查一次） |
| runtime | 恢复后状态摘要 | rank 重新导出的 adapter+optimizer+scheduler+counters 摘要，以及 RNG 摘要 | manifest `rank_summaries` | 与保存值不等即拒绝 |

## 2. 与默认 no-save/load-optim/rng 路径的隔离

- 默认参数不变。cut 插件不读取 `args.no_save_optim` 等开关，也不调用 `save_checkpoint`/`load_checkpoint`，只写到调用方给出的 `<root>/<cut_id>/`。
- 同形重建（4.3）走 fork `rebuild_training_models` → `create_training_models`，后者会调用 `rollout_executor.load(start_rollout_id-1)`（`placement_group.py:347`）。**更正（审查 H1）**：原稿写"`args.load is None` 所以 `data_source.load` 直接返回"，这是错的。Miles 解析参数时先设 `requested_load = load`（`arguments.py:3426`）；bridge 模式下，只要 `--load` 目录里没有 checkpoint，就把 `args.load` 改成 `--ref-load`、把 `start_rollout_id` 改成 0（`megatron_config.py:362-367`）。所以在 ports 路径上 `args.load` 永远不是 None。rollout 进程里的 `RolloutExecutor.load`（`rollout_executor.py:304-310`）会依次执行 `data_source.load` 和 `generate_rollout.load`/`eval_generate_rollout.load`。前者读取 rollout 进程自己的 `args.load`（也就是 ref_load 路径），只有该路径下没有 `rollout/global_dataset_state_dict_*.pt` 时才直接返回；后者在 stock rollout 函数中是空操作（`base_types.py:84`），自定义 rollout 函数可能不是。现行做法：
  - `rebuild_preconditions` 改查 `args.requested_load is None`；
  - 重建前后各经 `RolloutPool.data_cursor()`（E1 接口，未就绪时由 `DataCursorSource` 协议占位）读一次数据游标，两次不一致就判 RECOVERY_REQUIRED，不进入 restore。
- L3 结论：yeto ports 引擎（`yeto/rl/engine/**`）不读取 `args.start_rollout_id`，只有旧的非 ports 路径（`yeto/rl/miles.py` 等）读取。因此重建后该值被 Miles 改成 0 不影响 ports 路径，rollout id 由 driver 持有。
- 新 trainer 进程先按 Miles 默认方式初始化（新 adapter、新 RNG），再由 `restore_cut` 覆盖。整个过程不经过 `--lora-adapter-path`/`--lora-dp-invariant-state`。
- **`restore_cut` 只能用于新建的 trainer**（审查 H2）。Megatron `OptimizerParamScheduler.load_state_dict` 最后执行 `step(increment=num_steps)`，是把保存的进度累加上去，所以写入前断言 scheduler `num_steps==0`。scheduler 超参（max/min lr、warmup/decay 步数与方式、weight decay 计划）在写入任何状态之前与 cut 严格比对，等价于 `_check_and_set`，但不受 override 开关影响。写入后再断言 `num_steps` 等于 cut 中的值。
- **任何 restore 异常都判 RECOVERY_REQUIRED**（审查 M1/M2）：新 handle 换入之后，restore、布局读回或游标读取中出现任何异常，该 trainer 都不能继续训练。`shared_filesystem=False` 且 DistOpt、DP>1 时直接拒绝恢复，因为每个 rank 需要读取同一 (tp,pp) 下所有 DP 分片来做合并。
- 保存前（审查 M3）：先检查不依赖分片的 context 项（游标、账本、外层状态、算法身份、runtime、scheduler 与 local_step 一致），再拒绝 TP/PP>1 与 DistOpt 分片 master 的组合，最后才调用 rank 插件写分片。`has_optimizer_state` 的含义是：每个 (tp,pp) 的所有 adapter 名都在该组 DP rank 的 optimizer 状态中出现过（审查 L1）。
- 重建后的布局从 rank 实际读回（`rank_coords` 插件 → `actual_layout()`），不从 args 推算（审查 L1）。
- `SwappableActor.dispose()` 在调用时才解析当前 target（审查 H3）。原因是 Miles `Disposer.add` 在加入时就绑定了 `item.dispose`（`async_utils.py:207`）。`EvalDispatcher` 只保存 `self.actor_model`（即代理对象），每次调用时经代理解析，没有在构造时缓存方法。

## 3. Miles 超采样余量与 buffer 回收（F5）

`generate_rollout_async`（`sglang_rollout.py:444-560`；新路径 `inference_rollout_train.py` 行为相同）：

1. 每次从 data source 取 `over_sampling_batch_size` 组并提交（`:483`），数据游标立即前移。
2. 被动态过滤丢弃的组（`:507`）不会回收 → 账本终态 `filtered`（算法有意丢弃）。
3. 已完成、未被过滤、但超出 `rollout_batch_size` 的组直接丢弃，不放回 buffer（`:511` 注释 "we have not stored all the unused samples back to the data buffer"）。
4. 凑满后仍在飞的组会被 `abort`（`:420-437`）。`partial_rollout` 关闭时直接丢弃；开启时收集回 `data_source.add_samples`（`:698`），带 `start_rollout_id`，前缀来自旧策略。
5. `RolloutDataSourceWithBuffer.save` 不保存 `buffer`（`data_source.py:128-141` 只存游标）。
6. yeto ports 路径拒绝 `--partial-rollout` 与 `--mask-offpolicy-in-partial-rollout`（`algorithm_flags.py` `_UNMAPPED`），并用 `--buffer-filter-path policy_buffer_filter` 只复用当前策略的完整组。

结论：在 ports 路径上 buffer 恒为空，**`carried_over` 的实际范围是空集**。3.6 账本里第 3、4 类（超采样多出的完成组、在飞时被 abort 的组）是**引擎丢弃**：数据游标已经前移，但这些组既没有被算法过滤，也不会被复用。它们不是"丢失"（原因确定、可以计数），也不是 `filtered`（不是算法有意丢弃）。建议 3.6 为它们单列终态（例如 `discarded_surplus`），或在 `filtered` 下记子原因；这属于 E1 的决定，见 §5。复用时的策略版本问题不存在（没有复用）。今后如果开放 partial rollout，就必须把 buffer 写进 cut，并按旧策略前缀处理 age>0，这需要另立 change。

## 4. 验证状态

- CPU：`tests/test_rl_cut.py`、`tests/test_rl_miles_cut_plugin.py`、`tests/test_rl_trainer_cut.py`。其中 CPU 版 X3 模拟：torch Adam + dropout，训练 2 步 → 保存 → 用不同初始化和不同全局 RNG 新建 → 恢复 → 第 3 步参数与 moments 和连续运行 `torch.equal`；另有反例，证明没有 RNG 时结果会不同。这些测试使用纯 torch 后端替身，**不能代替** Megatron/M5 路径，只证明协议与拒绝逻辑正确。
- GPU：未运行（用户暂停）。判据见 `evidence/infra-e2/4.2-4.5/plan-v2.md`。

## 5. 对其他写入者的接口请求

- **E1（ports.py）**：把 `TrainerGroup` 的 E2 预留注释定稿为
  `save_cut(*, epoch: int, context: CutContext) -> str` 与
  `restore_cut(cut_id: str, *, epoch: int, root: str, expect: RestoreExpectation, shared_filesystem: bool = True) -> CutManifest`（类型见 `yeto/rl/engine/cut.py`、`miles_adapter/trainer.py`）。补丁：`/home/michael/work/infra-drafts/patches/infra-e2-ports-v2.patch`。
- **E1（rollout.py / rollout_meta_hook.py）**：在 rollout 进程内读取 data source 游标 `{sample_offset, epoch_id, sample_group_index, sample_index}` 与 `get_buffer_length()`，经元数据汇报；`RolloutPool` 增加 `data_cursor() -> Mapping[str,int]`，供 `CutContext.data` 使用。buffer 长度不为 0 时 cut 应拒绝。
- **E1（3.6 账本）**：按 §3 为超采样余量或在飞被 abort 的组设终态；并提供 `ready_unconsumed`/`carried_over` 计数给 `CutContext.ledger`。
- **E1（driver.py，4.4）**：重建时 driver 不换端口对象，只换 `SwappableActor` 背后的 handle；重建后不重复 `initialize`/`after_local_train`，并用现有 `publish` 重发当前 policy（重发前后的 policy hash 必须等于 cut 中的 `progress.policy_hash`）。
- **entry.py 所有者**：`compose_island` 需用 `SwappableActor(actor)` 包装后再传给 trainer、policy_state、publisher 与 `EvalDispatcher`，并能取到 `RayWorkerManager.get_handle()`。
