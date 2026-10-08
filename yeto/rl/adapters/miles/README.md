# Miles 适配层（`yeto/rl/engine/miles_adapter/`）

把 yeto 引擎核心的五个端口接到 pinned Miles（`yeto.rl.MILES_NEXT_COMMIT`）。计划在去耦合阶段 4 整体搬到 `yeto/rl/adapters/miles/`，原位置留转发模块。边界规则见 `../README.md`；本目录可以 import Miles/Megatron/SGLang/Ray，核心不可以 import 本目录。

## 翻译什么

- 运行配置 → Miles 命令行：`config.py` 把 `RLRunConfig` + `AlgorithmSpec` 翻成上游 Miles argv（`translate_run_config`），并给出 `MilesLaunchArgs`（argv、放置请求、算法哈希、运行时属性）；每个配置叶子有对应的翻译或拒绝。
- 算法字段 ↔ Miles 旗标：`algorithm_flags.py`（`algorithm_argv`、`absorb_extra_argv`，含 `--dry-run` 检查入口）。
- 端口实现：`rollout.py`（RolloutPool，基于上游 InferenceController/RolloutExecutor）、`trainer.py`（TrainerGroup，基于 actor TrainGroup）、`state.py` + `state_plugin.py`（PolicyState，在每个 Megatron rank 内导出/应用可训练状态）、`publish.py`（Publisher，基于 `update_weights`）、`placement.py`（放置描述与改写检测）。
- 组装入口：`entry.py`（`--rl-engine ports` 的组装根：能力声明、执行档案、岛的组装与 Ray 连接）。
- 弹性与切点：`elastic_*`、`cut_plugin.py`、`round_cut.py`、`trainer_rebuild.py`、`trainer_resize.py`、`rebuild_wiring.py`、`reshard.py`、`bundles.py`。
- Miles rollout 进程内的元数据钩子：`rollout_meta_hook.py`。

## 声明什么

- 能力：`entry.miles_capabilities()` 与 `MILES_DECLARED`（每个已声明机制都附 GPU 证据路径）；部分机制按 Miles commit 区分（`MILES_DECLARED_PINS`）。未声明的机制在启动前被拒绝，除非显式 `--rl-allow-unverified-mechanism`。
- 执行档案：`entry.execution_profile_for()`（colocated-serial / partitioned-serial / partitioned-overlap），绑定启动器给出的算法哈希。
- 运行时指纹：`entry.ports_runtime_fingerprint()`。

## 不支持什么

- 没有翻译的配置项抛 `UnmappedConfigError`（如部分仅旧版引擎支持的配置：DeepSeek V4、全参数等）。
- Miles 容错（FT）语义、一个训练组多个 cell、被 Miles 改写的放置：分别以 `FaultToleranceArgsError`、`SingleCellError`、放置改写检测拒绝。
- 只支持 NVIDIA CUDA（确定性环境变量 `cut_plugin.DETERMINISM_ENV` 为 CUDA 专用）。
- 本目录之外的旧版引擎（`yeto/rl/miles.py`）与 Miles 补丁（`miles_overlay.py`、`overlays/`）不属于本适配层，阶段 4 再归位。
