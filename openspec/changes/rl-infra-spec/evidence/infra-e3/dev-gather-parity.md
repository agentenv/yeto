# DEV-GATHER harness 与生产路径逐项对照（INFRA-E3，2026-09-30，上卡前）

生产路径：`learner.main` → `run_ports_island`（entry.py）→ `compose_island` → `IslandDriver.run`（driver.py）。harness：`learner_shim`（同一 `learner.main`，只替换 `run_ports_island`）→ `MilesBackend`（gen / 6 个 arm）。代码：infra-e3（本表随代码提交）。

## A. 启动（run_ports_island）

| # | 生产做了什么 | harness gen | harness arm | 不做 / 不同的理由 |
|---|---|---|---|---|
| A1 | learner.main 全部：源码树/奖励 sha、Miles pin、模型与数据下载、run config → Miles argv → parse_args | 是（同一 `learner.main`） | 是 | — |
| A2 | `require_run_plugin()` | 是（MilesBackend.__init__） | 是 | — |
| A3 | runtime 指纹、capabilities、execution profile、`preflight` | 指纹是（cut 用）；profile/preflight 否 | 同 | harness 不开 driver；算法/能力由 dry 阶段 `reshard_problems` 与 argv 检查代替，本地 dry-run 同一检查 |
| A4 | `elastic_wiring_for`（--rl-elastic） | 否 | 否 | 无弹性控制器；trainer 边 argv 用 `trainer_dp_edges=True`（同一 RLRunConfig 字段） |
| A5 | `connect_island_ray()`（RAY_ADDRESS + job runtime_env） | 是 | 是 | — |
| A6 | 选择事件写入事件磁带 | 否 | 否 | 证据写 events.jsonl/progress.log |
| A7 | `init_orchestration_script` + Disposer | 是 | 是 | — |
| A8 | `create_rollout_components` | 是 | 是（debug_train_only：无引擎） | arm 只重放冻结数据 |
| A9 | `create_training_models` | 是（启动时 trainer 大小：共置需每个引擎旁有 trainer rank，run 4） | 是（arm 的 DP） | — |
| A9b | parse 派生字段（`--load-debug-rollout-data` → `debug_train_only`、`rollout_num_gpus=0`、`starts_inference_engines=False`；`world_size = nodes × gpus`） | — | **是**（run 6 缺 `rollout_num_gpus`/`starts_inference_engines`，arm 仍声明引擎 cell，已补；`world_size` 同步） | harness 在 parse 后改 DP，须同时设置 parse 会派生的字段；生产 `resized_args` 同样补 `world_size` |
| A10 | `SwappableActor` 包装、`EvalDispatcher` | 否 | 否 | 不做重建换 handle、不做 eval |
| A11 | `RayMetadataSink()`（命名 actor `yeto_rollout_meta`） | **是**（run 5 缺项，已补） | 否 | arm 不生成 rollout，钩子不运行 |
| A12 | `compose_island`：`MilesPolicyState`、`MilesRolloutPool`、`MilesTrainerGroup`、`MilesPublisher`、placement、sync、事件磁带 | **是**：同一 PolicyState/RolloutPool/TrainerGroup/Publisher 类（run 6 起） | TrainerGroup 是（cut/restore/offload/onload） | 不构造 IslandDriver/sync/placement：不跑外层协议与安全点 |

## B. 每轮（IslandDriver.run / run_round），以 colocated-serial 为例

| # | 生产做了什么 | harness gen | harness arm | 不做 / 不同的理由 |
|---|---|---|---|---|
| B1 | `sync.start` → `export_local()`（state 插件导出，版本 0；装 grad-norm recorder） | 是（`MilesPolicyState.export(policy_version=0)`） | 探针 `install_probe` 装 recorder | — |
| B2 | `publish(state, rollout r)` = `MilesPublisher.publish`：导出核对 → （非首次）`onload_weights` → `update_weights` → `onload_kv` → `start_update_weights` → 每个引擎 `update_weight_version(token)` 并读回 → `end_update_weights` → `check_weights(checksum)` | **是**（同一 MilesPublisher，每个 rollout 发布 token r；run 3/4/5 缺项的根源） | 否 | arm 无引擎 |
| B3 | 首轮 eval（force） | 否 | 否 | 不做 eval |
| B4 | 安全点、准入门控、重配置 | 否 | 否 | 无控制器 |
| B5 | colocated：`trainer.offload()` 后再生成 | **是** | 否（无引擎） | — |
| B6 | `MilesRolloutPool.generate(r)`：sink 设 token → `prepare_rollout` → `executor.get`（钩子：trained groups、元数据、buffer filter；Miles 转换与生产 DP 切分）→ 按 `--offload-rollout` 卸载 → 取元数据 → 构造 handle | **是**（同一 `generate`；另加 `--save-debug-rollout-data` 落盘样本） | `executor.get` 重放冻结样本（`--load-debug-rollout-data`：生产转换与生产 `split_train_data_by_dp`，钩子不运行） | arm 数据固定是比较的前提（plan-v4 §0） |
| B7 | 策略身份核对（组 token = 期望 token） | handle 构造时同一校验 | 否 | arm 无组元数据 |
| B8 | colocated：`trainer.onload()` 后训练 | **是**（gen 不训练，只恢复驻留） | **是**（run 6 起补齐） | — |
| B9 | `MilesTrainerGroup.train_step`：token 校验 → `actor.train` → GRAD_NORM / APPLIED_LRS / STEP_LOSSES 插件 → 释放 object-store 引用 → receipt | 否（不训练） | `actor.train` + 同三个插件 + 同一释放函数 | arm 没有 handle/receipt；DP 变化后的批次守卫只在 `train_step` 内，harness 以 rank 探针直接断言分片（G2） |
| B10 | 梯度不变量、事件、账本 | 否 | 否 | 证据由探针与 compare 给出 |
| B11 | `export` 下一版并 `publish` | gen：每个 rollout 重新发布同一基座状态（token r） | 否 | 冻结数据全部来自基座策略（不影响判据） |
| B12 | 训练后 offload（上游 train.py） | — | 是（`trainer.offload()`） | — |

## C. 本地 dry-run（不起 GPU）已执行的构造步骤

- 与容器相同的 argv 构造（learner.parse_args → resolve_rl_run_config（trainer 边）→ build_ports_launch）与 `argv_check`；
- argv 中每个 `--*-path` / `--custom-rm-path` 钩子：yeto 与本地模块实际 import 并取到可调用对象；`miles.*` 在 fork pin 提交中核对定义存在；
- gen 阶段用生产 PolicyState/Publisher/RolloutPool 的调用顺序在 CPU 替身上跑通并与上表 B1–B8 顺序逐项比对（`tests/test_rl_e3_harness.py::test_gen_phase_runs_the_production_round_components_in_driver_order`）。
- 未能本地执行：Ray 命名 actor 的创建与跨进程查找（本机无 ray），由 gen 阶段首个 rollout 验证。
