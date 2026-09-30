# 迁移清单（migration ledger）

登记在途 PR 的语义，以及它们在 ports 路径上的落点。等价性验收（任务 6.x）按本清单逐条覆盖。

状态于 2026-09-29 复核：以下 PR 已合入 `agentenv/yeto` main（R0 先基于 main `cb55aca` 整合，随后移到 main `e21a7ff`，工作树 `/home/michael/work/r0-integ`）。

| PR | 分支 | 状态 | 合入提交 |
|---|---|---|---|
| #64 | `fix/rl-lora-grad-hook` | MERGED，ports 侧 CLOSED | `9740259` |
| #65 | `feat/gdn-hybrid-miles-recipe` | MERGED，ports 侧 CLOSED | `741ca70` |
| #59 | `pr11/math-rl-example` | MERGED，ports 侧 CLOSED | `f4645a9` |
| #66 | `feat/rl-elastic-benchmark` | MERGED，ports 侧 CLOSED | `8653152` |
| #62 | `fix/sky-island-ray-stop` | MERGED，ports 侧 CLOSED（共用岛脚本） | `cb55aca` |
| #63 | `fix/sky-island-tp-env` | 经重新提交的 PR #67/#68 进入 main，ports 侧 CLOSED（共用岛 env 表） | `e21a7ff` 之前 |
| #61 | `pr13/head-run-teardown` | 经重新提交的 PR #67/#68 进入 main（与 R0 无关） | `e21a7ff` 之前 |
| #60 | `pr7/head-cloud-credentials`（Nebius/Verda/Modal） | MERGED，ports 侧 CLOSED（7.0 补登记，见下） | `89aee13` |
| #58 | `pr12/head-cli-fixes` | MERGED，与引擎无关（仅 `yeto/cli.py` head 模式） | `e4ee4ac` |
| #67 / #68 | #61 / #63 的重新提交 | MERGED，内容同 #61 / #63 | `e21a7ff` |

---

## #64 梯度流不变量，以及零梯度时拒绝提交

- **PR**：https://github.com/agentenv/yeto/pull/64 ，"rl: restore LoRA gradient flow through the Miles policy-sync layer"。改动 `yeto/rl/miles.py`、`yeto/rl/core.py`、`yeto/rl/__init__.py`（`MILES_COMMIT` 升到 `ae475060`）、vendored bundle，新增 `tests/test_rl_grad_accumulator_hook.py`。
- **行为**：
  1. **梯度流**：`_optimizer_masters_as_model_parameters` 不能再把 fp32 张量赋给 bf16 `Parameter.data`，因为这样会替换 autograd 的 `AccumulateGrad` 节点，hook 不再触发，`main_grad` 恒为 0。修复方式是在 `megatron_module._parameters[<attr>]` 中临时放入一个新的 `Parameter`：它以 view 方式共享 `main_param` 的存储，并设 `requires_grad=False`，结束后还原。
  2. **不变量**：`MilesPolicySync` 提交本地状态之前检查 `_require_training_progress(stats)`。如果 `nonzero_advantage_count > 0` 而 `grad_norm == 0.0`，抛出 `StrictRlInvariantError("zero_grad_norm_with_nonzero_advantages")`，不提交也不推送，并写入事件记录。advantage 全为零的轮次允许零梯度。只有 `grpo`/`gspo` 能统计到 advantage，其他估计器记为未知（`None`），不做判定。
  3. `LocalRoundStats` 新增 `nonzero_advantage_count`。
- **ports 实现位置（计划）**：
  - 梯度流约束：`miles_adapter/state_plugin.py`（任务 3.4，design D4）。导出/应用时遵守“不跨 dtype 赋值 `Parameter.data`”的约束。Miles 侧补丁作为 `michaellchung/miles` `yeto/ports` 上的一个提交携带（D6）。
  - 零梯度拒绝：`yeto/rl/engine/driver.py`（任务 4.1），每轮训练后、安全边界同步之前检查。advantage 计数来自 `miles_adapter/rollout.py` 的元数据提取（任务 3.2）；`LocalStepReceipt` 带 `grad_norm`（任务 3.3）。
- **合入后落点（CLOSED）**：
  - legacy：`yeto/rl/miles.py::_require_training_progress`、`MilesPolicySync._nonzero_advantage_count`；`yeto/rl/core.py::LocalRoundStats.nonzero_advantage_count`；`MILES_COMMIT=ae475060`。
  - ports 梯度流：`yeto/rl/engine/miles_adapter/state_plugin.py::masters_as_module_parameters` / `assert_grad_flow_intact`，与 #64 design D1 的实现一致（`module._parameters[attr]` 临时换成共享 `main_param` 存储、`requires_grad=False` 的新 `Parameter`，退出时还原；engine 代码中没有任何 `Parameter.data =` 赋值）。
  - ports 零梯度拒绝：`yeto/rl/engine/driver.py::IslandDriver._check_gradient`，metric 名已统一为 #64 的 `zero_grad_norm_with_nonzero_advantages`。判据：任一组 `reward_std > 0`（GRPO 下等价于 advantage 不全为零）且 `grad_norm == 0.0`。比 legacy 更严：`grad_norm` 缺失或非有限时也拒绝。ports 不填 `nonzero_advantage_count`（`GroupMetadata` 没有逐样本 advantage，保持 `None`＝未知）。
- **验证（证据）**：
  - `tests/test_rl_grad_accumulator_hook.py`（#64 原文件，3 个测试）通过；原先的替身测试改名为 `tests/test_rl_miles_adapter_state.py::test_export_and_apply_keep_grad_accumulator_hook`，只覆盖 ports 的 export/apply 后 hook 仍触发。
  - `tests/test_rl_core.py::test_zero_grad_norm_with_nonzero_advantages_fails_before_submission` 等 3 个 #64 测试在 legacy 路径通过。
  - `tests/test_rl_engine_driver.py::test_zero_grad_fails_the_round_without_commit`（metric 名已对齐）、`test_zero_grad_with_all_zero_advantages_is_not_a_failure` 通过。GPU 零梯度注入冒烟仍归任务 4.1。
- **原计划验证**：
  - `tests/test_rl_grad_accumulator_hook.py` 原样保留，作为端口契约测试（D4）。
  - #64 在 `tests/test_rl_core.py` 中的不变量测试在 legacy 路径继续通过。
  - ports：driver 单元测试注入“advantage 非零、grad_norm=0”的 receipt，断言该轮失败、不提交、事件记录中有 `zero_grad_norm_with_nonzero_advantages`；任务 4.1 的 GPU 冒烟测试中注入零梯度，该轮失败且不提交（spec rl-engine-selection：“ports 路径存在对应测试，并且在零梯度注入下与 legacy 路径表现一致”）。

## #65 GDN recipe

- **PR**：https://github.com/agentenv/yeto/pull/65 ，"rl: run every gated-delta-net hybrid through Miles' Qwen3.5 layer spec"。改动 `yeto/rl/learner.py`（+9/-1）、`tests/test_rl_launcher.py`。
- **行为**：provider 报告 `experimental_attention_variant == "gated_delta_net"` 时，选择 `qwen3_5` recipe（`--model-name qwen3_5`、`--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec`、`--apply-layernorm-1p`、`--attention-output-gate`、dropout 0、flash attention）。原来按固定 Qwen3.5 模型 id + revision 的判定保留为 fallback。main 现状是：只有固定 checkpoint 走 qwen3_5；GDN provider 只额外加 `--qkv-format bshd`。
- **ports 实现位置**：
  - 引擎无关层（任务 2.4，已建槽位）：`yeto/rl/engine/run_config.py` 的 `GdnRecipe`（`gated_delta_net`、`qkv_format`）由 `select_gdn_recipe(provider)` 生成，挂在 `ModelRecipe.gdn` 上。recipe 选择在 `resolve_rl_run_config` 中完成。**#65 未合入，未搬入其代码**：合入时只需把 `qwen35_recipe = (<固定 id+revision>)` 改为 `gdn.gated_delta_net or (<固定 id+revision>)`（代码中有注释标出这一行）。legacy 翻译（`yeto/rl/learner.py::_legacy_miles_argv`）按 `ModelRecipe.name == "qwen3_5"` 输出参数，不需要再改。
  - ports 翻译：`miles_adapter/config.py`（任务 3.1），把 `ModelRecipe`/`GdnRecipe` 映射到 upstream Miles 的 Qwen3.5 layer spec。
- **合入后落点（CLOSED）**：`yeto/rl/engine/run_config.py::resolve_rl_run_config` 中 `qwen35_recipe = gdn.gated_delta_net or (<固定 id+revision>)`；`yeto/rl/learner.py::build_miles_argv` 的合并冲突按 R0 拆分解决（#65 的判定只保留在 run_config 一处）。
- **验证（证据）**：`tests/test_rl_launcher.py::test_miles_argv_uses_provider_capabilities_without_model_family_branches`（含 #65 断言）通过；`tests/test_rl_argv_snapshot.py` 的 golden 从 main `cb55aca` 未拆分的 `build_miles_argv` 重新捕获，只有 GDN provider 一条（`a37cc3…` → `c7e225…`）变化，其余 14 条与 c40a32c golden 相同，整合树中 3 个快照用例全部通过。
- **原计划验证**：
  - #65 在 `tests/test_rl_launcher.py` 中新增的断言（GDN provider 选择 `qwen3_5` recipe）合入后通过。
  - 合入后按预期重新生成 `tests/test_rl_argv_snapshot.py` 的 golden：只有 GDN provider 对应的条目变化，并在该测试的提交中说明原因。
  - ports：`miles_adapter/config.py` 单元测试断言 GDN provider 选择 Qwen3.5 layer spec；等价性对照（任务 6.x）中两条路径解析出同一个 `ModelRecipe`。

## #59 数据列名

- **PR**：https://github.com/agentenv/yeto/pull/59 ，"rl: name the prompt/label dataset columns; add a MATH-500 smoke test"。改动 `yeto/cli.py`、`yeto/launcher.py`、`yeto/rl/learner.py`（+29/-4）、`yeto/rl/math_reward.py`、`examples/`，以及相关测试。
- **行为**：`yeto launch --training-mode rl` 新增 `--rl-prompt-column` / `--rl-label-column`，透传给 island 的 learner。`prepare_prompt_data` 在规范化时按指定列读取；缺列时报错并指出对应参数，不回退到其他列。两个参数都不给时，行为与现在逐字节一致。规范化后的记录键仍是 `messages`/`label`/`metadata`，所以 Miles argv（`--input-key messages --label-key label --metadata-key metadata`）不变。
- **ports 实现位置**：
  - 引擎无关层（任务 2.4，已建槽位）：`yeto/rl/engine/run_config.py` 的 `DatasetColumns`。`prompt_column`/`label_column` 是源数据列名，现在从 `args.rl_prompt_column`/`args.rl_label_column` 读取，缺省为 `None`；`input_key`/`label_key`/`metadata_key` 是规范化后的键，legacy 翻译据此输出 `--input-key` 等参数。**#59 未合入，未搬入其代码**：parser 参数与 `prepare_prompt_data` 的列选择仍由 #59 带入。合入后，`prepare_prompt_data` 的调用方应改为从 `RLRunConfig.data.columns` 取列名，以免在两条路径各写一遍。
  - ports：ports 路径复用同一份规范化后的 prompt 文件；`miles_adapter/config.py` 从 `DatasetColumns` 的 `*_key` 字段映射参数。
- **合入后落点（CLOSED）**：parser 参数 `--rl-prompt-column/--rl-label-column` 与 `prepare_prompt_data(prompt_column=, label_column=)` 在 `yeto/rl/learner.py`（#59 原样）；`yeto/launcher.py::make_miles_island_task` 透传；`RLRunConfig.data.columns`（`DatasetColumns`）从同一 args 读取。`learner.main` 仍直接从 args 取列名传给 `prepare_prompt_data`（在 run_config 解析之前调用，未改），两条引擎路径共用同一份规范化 prompt 文件，因此不存在两处实现。
- **验证（证据）**：`tests/test_rl_launcher.py::test_miles_island_forwards_dataset_column_flags`、`test_prompt_data_*`、`tests/test_rl_math_reward.py` 通过；argv 快照与 main 一致（列名不进入 argv）。
- **原计划验证**：
  - #59 的测试（`tests/test_rl_launcher.py` 中列名透传/缺列报错，`tests/test_rl_math_reward.py`）合入后通过。
  - spec rl-engine-ports“数据列名透传”场景：配置指定 prompt/label 列名时，两条路径读入的规范化记录一致。
  - `tests/test_rl_argv_snapshot.py`：不给列名参数时 argv 不变。

## #66 capability 格式

- **PR**：https://github.com/agentenv/yeto/pull/66 ，"rl: add the elastic resource benchmark contract, gating, evidence and paired work"。全部为新文件：`yeto/rl/elastic_benchmark/`、`scripts/benchmark_rl_elastic.py`、`tests/test_rl_elastic_benchmark.py`、`docs/RL_ELASTIC_BENCHMARK.md`。
- **行为**（capability 部分）：`yeto/rl/elastic_benchmark/capabilities.py` 定义运行时认证 `Attestation`，由 `attestation_from_dict` / `load_attestation` 从 JSON 读取。字段为 `runtime_fingerprint: str | None`、`execution_modes`（必须属于 `EXECUTION_MODES`）、`certified_edges`（`{source, target, kind}` 对象）、`optimized_paths`、`auto_controller: bool`、`partitioned_driver: bool`。没有认证时视为什么都未认证（`Attestation.none()`）。矩阵中的条目只有在 runner 认证之后才可运行。
- **ports 实现位置**：`yeto/rl/engine/capabilities.py` 中 `EngineCapabilities` 的序列化（任务 2.5，design D8b）。输出能被 #66 的 `attestation_from_dict` 直接解析；两边需求有差异时改 #66 一侧，只保留一套 capability schema。driver 在创建任何 GPU 进程之前比对能力声明，任一项不支持就拒绝启动（spec rl-engine-ports）。
- **合入后落点（CLOSED）**：`yeto/rl/engine/capabilities.py` 直接从 `yeto/rl/elastic_benchmark/manifest.py` 导入 `EXECUTION_MODES`、`EDGE_KINDS`（单一 schema 来源）。执行模式命名统一采用 #66 的 `colocated-serial` / `partitioned-serial` / `partitioned-overlap`：删除了 `serial-colocated` → `colocated-serial` 映射，driver（`EXECUTION_MODE`）、fake、`miles_adapter/entry.py`、测试和本 change 的 proposal/design/tasks/spec/architecture 全部改名（`evidence/` 下的历史事件记录保持原样，仍是 `serial-colocated`）。额外键（`schema`、`engine`、`parameter_layouts`、`placements`、`advantage_estimators`、`dynamic_sampling_filters`、`port_verbs`）保留为顶层扩展字段：#66 的 `attestation_from_dict` 忽略未知键，不需要修改 #66。
- **验证（证据）**：`tests/test_rl_engine_capabilities.py` 删除了 #66 读取函数的副本，改为直接 `from yeto.rl.elastic_benchmark.capabilities import attestation_from_dict`，全部通过；`tests/test_rl_elastic_benchmark.py` 通过。
- **原计划验证**：
  - 任务 2.5：测试中用 #66 的 `attestation_from_dict` 解析 `EngineCapabilities` 的序列化结果并成功。#66 未合入时先按 PR 版本实现，合入时复核。
  - `tests/test_rl_elastic_benchmark.py` 在 #66 合入后继续通过。
  - driver 单元测试：声明中缺少所需能力时，在启动 GPU 进程之前拒绝。

## #62 / #63 远端准备脚本

- **#62**（main `cb55aca`）：`make_miles_island_task` 的岛 run 脚本改为私有 `--temp-dir=$HOME/miles-ray`、按路径 `pkill` 的 `stop_miles_ray`、`RAY_ADDRESS`。ports 不另有 run 脚本，只在同一脚本的 learner 命令上追加 ` --rl-engine ports`，因此自动继承，无需移植。
- **legacy 逐字节一致（证据，已在 `e21a7ff` 上重做）**：用 pytest 插件包住 `make_miles_island_task`，在 main 与整合树上分别跑 `tests/test_rl_launcher.py`，录得的 9 个岛任务 `setup`/`run` 完全相同；`tests/test_rl_engine_pins.py` 的 legacy golden 已按 main 重新捕获（仅 #64 的 `MILES_COMMIT`/bundle sha 变化），`_host_setup_script` 的 sha256 与 main 一致（`1eec0953…`）。
- **#63（经 #67/#68 重新合入，main `e21a7ff`）**：岛 `envs` 表加 `CUDA_DEVICE_MAX_CONNECTIONS=1`、去掉 `NVTE_*_ATTN` 硬钉。ports 与 legacy 共用该表，无需移植。证据：在 `e21a7ff` 干净 worktree 与整合树上分别跑 `tests/test_rl_launcher.py`，录得 9 个岛任务的 `setup`/`run`/`envs` 完全一致（含 `CUDA_DEVICE_MAX_CONNECTIONS`）；pins golden 与 `_host_setup_script` sha256（`1eec0953…`）在 `e21a7ff` 上重新捕获，与 `cb55aca` 相同；argv 快照 3/3 与 `e21a7ff` 一致。
- `yeto/rl/ssh_harness.py` 未被 #62/#63 改动（仍整机 `ray stop --force`，不在 SkyPilot 上，不受影响）。

## #60 Modal 岛的外部 SGLang router（7.0 补登记）

- **PR**：#60 "clouds: add Nebius, Verda and Modal to the planner and launcher"（main `89aee13`，在 R0 基线 `cb55aca` 之内）。RL 相关改动：`yeto/rl/learner.py` 新增 `start_external_sglang_router` 与 `YETO_RL_EXTERNAL_ROUTER`（Modal 岛上 legacy Miles 自带 router 30 s 期限不够，由 yeto 先起 router）；`yeto/launcher.py` 新增 `check_cloud_prerequisites`（Modal + RL 要求镜像按 digest 固定），Modal 岛配置对 RL 设置该 env。
- **ports 落点（CLOSED）**：upstream Miles 以 Ray worker 启动 router（120 s 预算）且已移除外部 router 模式，因此 ports 不复刻，改为显式拒绝/忽略：`yeto/rl/learner.py::require_ports_router_mode`（约 1412 行，预置 `sglang_router_ip` 报错，env=1 打印忽略）；`yeto/launcher.py` 约 2646 行 Modal 岛只对 legacy 设 `YETO_RL_EXTERNAL_ROUTER`；ports 的 `image_ref` 走 `default_rl_image("ports")`。
- **证据**：`tests/test_rl_engine_selection.py::test_modal_ports_island_does_not_request_the_external_router`、`::test_ports_router_mode_ignores_external_router_and_refuses_preset_address`、`::test_run_miles_ports_never_starts_the_legacy_external_router`，`tests/test_launch_auto.py` 117/133 行；2026-09-29 复跑 `test_rl_engine_selection.py`、`test_launch_auto.py`、`test_head_mode.py` 共 97 passed。
- **遗留风险（不阻塞关闭）**：尚无 Modal 上 `--rl-engine ports` 的真实 GPU 运行；upstream router 120 s 预算在 Modal CPU 上是否足够只有文档依据。首次 Modal ports 运行时需确认 router 就绪日志。

## 与引擎无关、无需迁移的 PR（7.0 复核）

- **#58**（`yeto/cli.py`）：head 模式 run name 遮蔽与 external seats，不区分引擎；ports 的 head 模式（`2cbd45c`）在其之上开发，`tests/test_head_mode.py` 通过。
- **#61 / #67**（`yeto/cli.py`、`yeto/launcher.py`、`yeto/modal_runner.py`）：head run 拆除与云端确认，按集群名操作，与引擎无关。
- **#48–#57**（2026-09-25 03:38 合入，早于本 change 创建日 2026-09-28）：均已包含在 R0 基线 `cb55aca` 中（#49、#52 改过 `yeto/rl/learner.py`，#49/#53/#56/#57 改过 `yeto/launcher.py`），ports 直接在其上开发，不属于在途迁移。#54 无代码中 RL 改动。
- 核对方法：`gh pr list -R agentenv/yeto --state merged --limit 100 --json number,title,mergedAt,files`，筛 2026-09-24 之后合入且改动 `yeto/rl/**`、`yeto/launcher.py`、`yeto/cli.py`、`yeto/modal_runner.py`、`scripts/benchmark_rl*.py`、`docs/MILES_RL.md` 的 PR（#48–#68）。R0 期间没有 PR 改动 `scripts/benchmark_rl.py` 或 `docs/MILES_RL.md`。
