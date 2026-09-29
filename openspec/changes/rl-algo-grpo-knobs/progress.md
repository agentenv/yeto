# rl-algo-grpo-knobs 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)（以 `rl-infra-spec` 分支为准）。

## 2026-09-29 第三轮（G1 与声明，Agent ALGO-1b）——以本节为准

### G1 第 2 次尝试逐项结果
证据：`evidence/2026-09-29-algo1b-g1/`，包括 plan.md、`g1_report.json`（原始判定）和 `g1_report_v2.json`（按审查决定把 SGLang freeze_gc 良性链式异常列入白名单后的判定）。所有 run 都用 Modal sandbox，1 张 H100，单岛、no-sync，每个 3 轮。

| 机制 | rc | 轮数 | 秒 | 峰值 MiB | 关键指标 | 是否证明生效 |
|---|---|---|---|---|---|---|
| baseline | 0 | 3 | 418.8 | 38142 | grad_norm 0.63/0.41/0.27 | — |
| clip_higher | 0 | 3 | 330.7 | 38178 | pg_clipfrac 0/0/0；pg_loss 与 baseline 逐位相同 | 否 |
| dual_clip | 0 | 3 | 349.6 | 36784 | pg_clipfrac 0/0/0；pg_loss 与 baseline 逐位相同 | 否 |
| token | 0 | 3 | 363.2 | 38178 | pg_loss 0.035/0.0068/0.0026（与 baseline 不同） | 是 |
| drgrpo | 0 | 3 | 333.2 | 38178 | pg_loss 0.0079/0.0072/0.019（常数分母，与 baseline 不同） | 是 |
| kl_k3 | 0 | 3 | 345.8 | 38184（baseline 38142） | kl_loss 0/0.00079/0.00082 | 是 |
| entropy | 0 | 3 | 327.4 | 38178 | entropy_loss 0.30/0.38/0.45 | 是 |
| over_sampling | 0 | 3 | 316.7 | 36924 | 丢组与补采均为 0 | 否 |
| overlong_penalty | 0 | 3 | 362.8 | 38174 | 3 条 rl_reward_shaping，塑形后奖励均值低于原始 | 是 |

### 8.3 声明（按主 agent 的决定）
- `grpo_knobs.G1_DECLARED` 按机制逐项提交，每项一个提交：token、drgrpo、kl_k3、entropy、overlong_penalty（连同 reward_postprocessors:custom_reward_postprocess）。
- entry.py/fake.py 的接线是共享文件，不在本分支提交，交付物为：
  - `infra-drafts/1b-declare-wiring.patch`：接线本身；
  - `infra-drafts/1b-declare-series/`：git format-patch 序列，第 0001 个是接线，0002–0006 是逐项声明（这些已在本分支），0007 是测试。
- 打上接线后，P0 测试 `test_miles_accepts_each_declared_mechanism_and_rejects_overlong_filter`（KL loss 用例缺 ref_model）和 `test_miles_and_fake_declarations`（断言精确的 R0 集合）失败，2a 的 `test_miles_adapter_declares_none_of_them` 也失败。需要 ALGO-CAP 和 2a 更新各自的测试。
- clip_higher、dual_clip、over_sampling、overlong_filter 不声明。

### 8.1 与 8.2
- 8.1：`yeto launch --dry-run`（P0 319d974）的输出见 `evidence/2026-09-29-algo1b-dryrun/dry-run.txt`：一个岛，modal H100×1，total_gpus 1，no-sync，放行开关与 spec 哈希都出现在输出中。凭据扫描 `cred_scan.txt` 结果为 0 hits。dry-run 里的 gpu 字段写的是 "H100"：这是请求的规格；是否锁定型号（不升级到 H200）由 `--modal-gpu-exact` 控制，它不体现在这个字段里。本 change 不做逐位实验，没有使用该开关。建议 P0 在 dry-run 输出中加一个显示 gpu_exact 的字段，已报告主 agent。
- 8.2：保持未完成：overlong_filter 没有跑，要等 1b-hook.patch 合入。

### 进行中
- `evidence/2026-09-29-algo1b-g1b/`：按审查要求补做"是否生效"的 G1（clip_higher 用更小的 clip 窗口、跑 2 个 optimizer step；over_sampling 用更大的批次和更多轮）。dual_clip 事先判为"未能证明生效"，原因写在 plan.md。

### 其他
- setup.sh 改为 `set -eo pipefail`；attempt 1/2 当时用的是普通 `set -e`，这一点记录在 setup.sh 的注释里。
- teardown_proof.txt 保存了 modal app list 与 sandbox 列表的实际输出：algo1b 的 app 全部 stopped，sandbox 列表为空。
- 对其他分支的观察：algo-cap 声明了 maxrl 和 mapo，而 2a 的 `test_miles_adapter_declares_none_of_them` 断言一个都不声明。这两个分支合在一起就会失败，与本 change 无关。

## 2026-09-29 第二轮（审查修复，Agent ALGO-1b）——以本节为准，覆盖下一节中与之冲突的状态

### 合入与提交
- 已普通 merge：algo-cap 099756c、63a78bd、6149a90（以及其后的 HEAD）和 infra-a（driver 的 `expects_gradient`）。关键提交：
  - `a978b3c`：register_gradient_rule 增加 mechanism=，2a 在等这个修复；
  - `063bc3e`：F1–F5、3.2、3.3、6.5；
  - `2722cad`：register_pipeline_plugin_module；
  - `6dee803`：run4 证据与 G1 计划；
  - `6dcca57`：勾选状态。
- c1a9a5a（及后续）修改了本 change 的 `design.md`。阶段 0 之后允许各 change 负责人维护本 change 的文档，两处改动都没有放宽验收：一处同步 F5 的 carried_over 说法，一处从 rl-infra-spec 取回的 A2 措辞。

### 审查项
- F1：用了分派器的 spec，必须在 `plugins` 里列出每个注册了 shaper/transform 的模块，按模块名排序，由 `grpo_knobs.with_pipeline_plugins` 补齐。这些模块的源码哈希因此进入算法哈希，learner 启动时会重新核验。合并哈希 `pipeline_sha256` 随 `yeto_algo_plugins` 下发；Miles 进程里 `post_process` 先调用 `load_extensions()`，再核对这个值。未注册的阶段在 launch check 阶段就被拒绝。注册函数时会自动调用 `register_pipeline_plugin_module`。
- F2/F3：`sample_filters` 纳入覆盖：spec.plugins 要列出它，另下发 `sample_filters_sha256`。ports 运行中如果 runtime attrs 没送到 Miles 进程（`yeto_algo_plugins` 与 `yeto_rl_dynamic_sampling_max_replacements` 都不存在），`apply_sample_filters` 直接报错，不静默跳过。
- F4：`read_plugins` 比对 payload 中的 `algorithm_spec_sha256` 与 `yeto_rl_expected_algorithm_sha256`。
- F5：Miles 不会把超额完成的组放回 buffer（`sglang_rollout.py:505-510`），只有 `--partial-rollout` 下被 abort 的样本回 buffer，所以超采样不产生 `carried_over` 余量。已更正 sample_filters、design.md、1b-hook.patch 的 docstring，MILES_RL.md 的改法写成 1b-docs-f5.patch（已由 ALGO-CAP 在 6149a90 合入）。
- F6：run1–4 的日志用 `git add -f` 入库，入库前扫描过密钥，无命中；删除了 `__pycache__`；各次运行的 yeto 版本见 `YETO_SHA.txt`；run3、run4 的结论已补进 plan.md。

### 任务状态更新
| task | 状态 | 证据 |
|---|---|---|
| 2.3 / 4.3 / 7.1 | CPU 通过 ✔（按主 agent 的口径决定，alignment §7b 第 7 条） | 钉住镜像中完整 `parse_args` + `validate_parsed_args`：run3 12 项，run4 14 项（含两个示例）。环境差异：用镜像自带的 megatron，miles 解析器为 0394715，而不是原文的 miles-next-venv |
| 3.2 | CPU 通过 ✔ | `test_fake_two_islands_ref_mismatch_fails_before_outer_sync`：fake 组合根两岛走 strict-avg。ref 相同时两岛都加入外层同步并完成一轮；ref revision 不同的岛在 `verify_ports_algorithm` 处失败，client_factory 从未被调用（未加入外层同步），并写出 `rl_algorithm_island_rejected`。revision 比较不区分大小写，与 learner 的 `.lower()` 一致 |
| 3.3 | CPU 通过 ✔ | `kl.ref_model.source` 必须等于 `--model`，`--ref-load` 由它解析得出；KL loss 下拒绝 `--megatron-ref-load`，因为无法把它绑定到 ref 身份。learner 的传参（1b-refload.patch）已由 ALGO-CAP 在 6149a90 合入。测试：`test_ref_source_bound_to_base_model`、`test_learner_binds_ref_source_and_override` |
| 4.1 | CPU 通过 ✔（表述已更正） | run4：镜像 `PYTHONPATH=/pkg/:/root/`，Miles 以 editable 方式装在 /root/miles；在 cwd=/tmp 时 `import examples…` 报 ModuleNotFoundError |
| 5.2 | CPU 通过 ✔（补充） | 在钉住镜像内用镜像自带的 Miles 跑 equivalence，69 passed；源码 sha256 相同（同源） |
| 5.5 / 6.4 / 7.3 | CPU 通过 ✔ | 文档已由 1b-docs.patch 合入 algo-cap 099756c，F5 的更正在 6149a90 |
| 6.5 | CPU 通过 ✔ | driver 调用 `expects_gradient`（来自 infra-a），规则通过 `register_gradient_rule(mechanism="features:overlong_filter")` 注册。fake driver 结果：全部截断且 grad 为 0 时不失败；部分截断且 grad 为 0 时仍失败；grad 为 nan 时仍失败。1b-hook.patch 合入前，`GroupMetadata.filtered_samples` 由测试里的同名子类提供；打上补丁后同一组测试也通过 |
| 6.3 | 已实现（未勾选） | 等 INFRA 合入 1b-hook.patch（已按 F5 更新，在当前 HEAD 上可以 apply，打上后 79 passed） |
| 7.2 | 未完成 | 需要 driver 事件字段（INFRA） |

## 2026-09-29（Agent ALGO-1b）

### 分支与提交

- worktree `/home/michael/work/algo-1b`，分支 `algo-1b`，从 `origin/algo-cap` 3d1b466（P0 接口冻结）拉出，已普通推送到 `origin/algo-1b`（没有强推）。
- 已按普通 merge 合入 algo-cap 的 ebd436b（1b-shared.patch 的四个 hook）和 8a3e041（1b-p0tests.patch）。
- **分派器接口冻结提交：`40ee1a2`**，已通知主 agent。接口见 `yeto/rl/algos/reward_pipeline.py` 的模块 docstring。
- 共享文件只改了 `yeto/rl/algos/__init__.py` 中 EXTENSION_MODULES 的一行（`"yeto.rl.algos.grpo_knobs"`）。其余共享文件的改动都以补丁交出（见"补丁"一节）。

### 新增文件

- `yeto/rl/algos/reward_pipeline.py`：分派器、注册表、runtime attrs 通道、overlong 软惩罚。
- `yeto/rl/algos/grpo_knobs.py`：spec 字段、机制、拒绝规则、runtime attrs、launch/island 检查、梯度规则。
- `yeto/rl/algos/reducers.py`：vendor 的 Dr.GRPO reducer。
- `yeto/rl/algos/sample_filters.py`：overlong 过滤，供共享 hook 调用。
- `examples/rl_algorithms/{dapo-like,dr-grpo}.json`。
- 测试：
  - `tests/test_rl_grpo_knobs.py`（yeto-venv）；
  - `tests/test_rl_reward_pipeline_equivalence.py` 与 `tests/test_rl_grpo_knobs_upstream.py`（miles-next-venv，以及钉住的镜像）。

### 补丁（共享文件，由主 agent 协调合入）

| 补丁 | 目标 | 状态 |
|---|---|---|
| `/home/michael/work/infra-drafts/1b-shared.patch` | algorithm.py 四个 hook；config.py 调用 launch_problems；learner.py 调用 island_problems | 已合入 algo-cap ebd436b |
| `/home/michael/work/infra-drafts/1b-p0tests.patch` | P0 测试夹具（KL loss 带 ref_model、over-sampling 配 filter、reward_postprocess 用分派器） | 已合入 algo-cap 8a3e041（62647fb） |
| `/home/michael/work/infra-drafts/1b-hook.patch` | `rollout_meta_hook.record_trained_groups` 调用 `apply_sample_filters`；元数据加 `filtered_samples`；`GroupMetadata.filtered_samples`；rollout.py 解析该字段 | 已转 INFRA，未合入 |
| `/home/michael/work/infra-drafts/1b-docs.patch` | `docs/MILES_RL.md` 新增"GRPO-family knobs"小节（覆盖 5.5/6.4/7.3 要写的文档内容） | 未合入（算法小节归 WP-CAP） |

账本字段约定（alignment A2/F5）：
- 被过滤的样本：`remove_sample=True`，并写 `metadata["yeto_filtered_by"]="overlong_filter"`；
- rollout 元数据：`filtered_samples={"overlong_filter": n}`，每组另有 `filtered_samples:int`；
- 账本把这些样本记为终态 `filtered`，超采样余量记为非终态 `carried_over`（由 INFRA 3.6 实现）；
- 默认配置下元数据的键集合不变（有测试）。

### 任务状态（五种之一）

| task | 状态 | 证据 / 说明 |
|---|---|---|
| 1.1 | CPU 通过 ✔ | `baseline-failures.txt`：94 条（68 failed + 26 errors），与 P0 基线逐 id 相同 |
| 1.2 | 已实现 ✔ | 接口所在文件见下文"1.2 接口位置"；原先缺的 runtime attrs、launch、island、gradient hook 已由 ebd436b 补上 |
| 2.1 | CPU 通过 ✔ | `test_field_constraints`、`test_over_sampling_needs_filter_and_batch`：报错含字段名与范围，均在 spec 构建、拒绝矩阵或 translate_run_config 阶段失败 |
| 2.2 | CPU 通过 ✔ | `test_translation_fragments` 等测试；`test_rl_argv_snapshot.py` 与 `test_rl_miles_adapter_config.py` 未改动即通过 |
| 2.3 | CPU 通过（口径待批准，未勾选） | miles-next-venv 缺 `megatron.training`，upstream `parse_args` 在该 venv 里跑不起来。实际做法：(a) 在 venv 中用 Miles 自带的参数提供器解析，12 个用例通过；(b) 在钉住的镜像（T4，PYTHONPATH 指向 miles-next 0394715）中跑完整 `parse_args`+`validate_parsed_args`，12 个用例通过（evidence `2026-09-29-algo1b-cpu-img/run3.log`） |
| 2.4 | CPU 通过 ✔ | `test_dual_clip_and_clip_higher`：14 个用例覆盖 A>0、A<0 与 ratio 落在区间内外。Miles 的 clip 边界按 float32 计算，比较精度 1e-6 |
| 3.1 | CPU 通过 ✔ | `test_ref_model_enters_hash`、`test_ref_model_required_for_loss_kl`、`test_default_hash_and_argv_unchanged` |
| 3.2 | CPU 通过（未勾选） | `test_two_islands_ref_model_checked_before_joining`：两个岛经 learner 的 `verify_ports_algorithm`（外层同步之前）校验，rev 不同的岛失败，并写 `rl_algorithm_island_rejected` 事件。用的是 learner 入口的两岛测试，不是原文所说的 fake 组合根两岛测试，待确认是否算满足 |
| 3.3 | CPU 通过（口径同 2.3，未勾选） | k1/k2/k3/low_var_kl 的翻译与解析；未知估计器被拒并列出允许值；`train.py:58` 条件已核对（`test_kl_loss_triggers_ref_load_branch`） |
| 4.1 | CPU 通过 ✔ | 钉住镜像（`radixark/miles@sha256:9094…`）中执行结果为 `IMPORT_FAILED: No module named 'examples'`（run1.log）。按原文仍 vendor |
| 4.2 | CPU 通过 ✔ | `test_vendored_source_pinned`：对 `git show 9e4260d:<path>` 计算 blob `96390ac3` 与 SHA256 `2fe93181…`；分母缺失时报错 |
| 4.3 | CPU 通过（口径同 2.3，未勾选） | 三种拒绝：缺分母、与 token 聚合同时启用（吸收冲突）、CP>1；翻译与 PluginRef 已实现 |
| 4.4 | CPU 通过 ✔ | 分母为 1000 时与原示例 `torch.equal`；D=1/7/4096 时等于有效 token loss 之和除以 D |
| 5.1 | CPU 通过 ✔ | 无 GPU 环境可以 import；`grpo_default` 覆盖非 grpo 估计器与 `rewards_normalization` 关闭的返回值 |
| 5.2 | CPU 通过 ✔ | 69 个用例 `torch.equal`（8 种批次 × 8 种估计器/开关组合，外加报错与 custom 路径用例）；锁定 `train_data_conversion.py` 的 sha256 `ff7448c0…` |
| 5.3 | CPU 通过 ✔ | `yeto_algo_plugins={config,sha256}` 经 `to_legacy_runtime_attrs` 下发（`test_translate_run_config_launch_checks`）；篡改后报错 |
| 5.4 | CPU 通过 ✔ | 默认 argv 不含该参数；多 LoRA 在启动前与运行时都被拒；回退时写警告事件（有内容断言） |
| 5.5 | 已实现（未勾选） | 用恒等变换测试证明扩展点可用；docstring 已写。`docs/MILES_RL.md` 的部分在 1b-docs.patch 中，未合入 |
| 6.1 | CPU 通过 ✔ | 长度 80/90/100/101 对应 0/−0.5/−1/−1；多段样本塑形后奖励一致；越界参数被拒 |
| 6.2 | CPU 通过 ✔ | `metadata["yeto_raw_reward"]`；`rl_reward_shaping` 事件含原始与塑形后奖励 |
| 6.3 | 已实现（未勾选） | `sample_filters.py` 的测试通过；hook 接线在 1b-hook.patch（打补丁后测试通过，本分支中 skip），等 INFRA 合入 |
| 6.4 | CPU 通过（未勾选） | `test_remove_sample_semantics`：loss mask 全为 0；同组 advantage 不变；sample-mean 下被屏蔽样本仍计入 `global_batch_size` 分母。结论写在 1b-docs.patch 中，未合入 |
| 6.5 | 未完成 | 规则 `overlong_gradient_rule` 已注册并有单测。原文要求的 fake driver 测试依赖两项：P0 4.2（driver 调用 `expects_gradient`，尚未落地）与 1b-hook.patch（`GroupMetadata.filtered_samples`） |
| 7.1 | CPU 通过（口径同 2.3，未勾选） | 两个示例在 yeto-venv 中能构建与翻译，在 miles-next-venv 中能被 Miles 参数提供器解析；dry-run 证据见 `evidence/2026-09-29-dry-run/` |
| 7.2 | 未完成 | 每轮事件记录进入训练的样本数与组数，需要在 driver 事件中加字段（INFRA/driver 负责）。接口需求见下文 |
| 7.3 | 已实现（未勾选） | 文档在 1b-docs.patch 中；文档里的 dry-run 命令已执行，输出与描述一致（`evidence/2026-09-29-dry-run/dry-run.txt`） |
| 8.1–8.6 | 未完成 | 见"GPU"一节 |
| 9.1 | CPU 通过 ✔ | 合入 8a3e041 后全量测试失败集合与 1.1 相同（diff 为空）：68 failed / 26 errors，2257 passed |
| 9.2 | CPU 通过 ✔ | `openspec validate rl-algo-grpo-knobs --strict`：valid |
| 9.3 | 已实现 ✔ | 本文件 |

（原"可表达未开放"清单已过时，删除；当前声明状态见本文件第三轮一节。）

### 1.2 接口位置

- `LossSpec`、`AdvantageSpec`、`KlSpec`、`SamplingSpec`、`PluginRef`、`to_legacy_runtime_attrs`、`expects_gradient`，以及 `register_field`/`register_mechanism`/`register_rejection`/`register_runtime_attrs`/`register_launch_check`/`register_island_check`/`register_gradient_rule`：都在 `yeto/rl/engine/algorithm.py`。
- 映射行：`yeto/rl/engine/miles_adapter/algorithm_flags.py`（`--eps-clip(-high/-c)`、`--calculate-per-token-loss`、`--disable-grpo-std-normalization`、KL 行、`--entropy-coef`、`--custom-pg-loss-reducer-function-path`、`--custom-reward-post-process-path`、`--over-sampling-batch-size`）。
- `EngineCapabilities` 的机制维度：`yeto/rl/engine/capabilities.py`。
- 缺口：driver 调用 `expects_gradient()` 属于 P0 4.2，尚未落地。

### 测试命令与结果（原始摘要）

- `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors -p no:cacheprovider` → `68 failed, 2257 passed, 46 skipped, 26 errors`。失败集合与基线按 id 比较，diff 为空。
- `PYTHONPATH=<algo-1b>:/home/michael/work/miles-next /home/michael/work/miles-next-venv/bin/python -m pytest -q tests/test_rl_grpo_knobs_upstream.py tests/test_rl_reward_pipeline_equivalence.py` → `105 passed, 12 skipped`（12 个 skip 都是 full_parse，原因是缺 megatron.training）。
- 钉住镜像，T4，run3：`25 passed`（full_parse 12、Miles 参数提供器 12、ref-load 1），P0 upstream_parse `2 passed`。

### 云资源（Modal，前缀 algo1b-）

| 运行 | app id | 资源 | 时长 | 结果 |
|---|---|---|---|---|
| run1 | ap-NwnNkUKRNgnMPlraQLkxhp | CPU 2 核 / 8GiB | 16:41–16:42 UTC | 4.1 结论；parse 失败于缺 libcuda |
| run2 | ap-JyzgEPFHYfn7MQw5E9lrgP | T4×1 | 16:43–16:46 | 测试 import bug |
| run3 | ap-IFjK3spr0lcOfFUfQIopEL | T4×1 | 16:47–16:49 | 全部通过 |

- 事先写好的计划与逐次结论：`evidence/2026-09-29-algo1b-cpu-img/plan.md`。
- 回收机制：函数 `timeout=1200`，外加本地 `timeout 1800`；`modal run` 是临时 app，入口结束即停止。
- 无残留证明：`teardown_proof.txt`，三个 app 均为 stopped，running 为空。没有创建卷。
- 费用：预估 < $0.30；按 T4 约 $0.6/h × 约 6 分钟加 CPU 分钟数估算，实际约 $0.1。账单未核实，标"未确认"。

### GPU（8.x）：未启动，阻塞原因

1. overlong 过滤的 G1 依赖 1b-hook.patch 合入（INFRA）。零梯度不变量的判定依赖 P0 4.2 的 driver 改动，也还没落地。
2. ports 单岛 G1 需要 head/syncer 放置。memory 记录 Verda/Modal 都不能承载 head，而 BRIEF 禁止用 Verda 跑算法实验。主 agent 提到私有 ports 镜像已在 rl-integ f6194da，但 launcher 在 Modal 上的单岛 ports 路径尚未确认可用。
3. 解除条件：上述补丁与 P0 4.2 合入；主 agent 指定可用的 ports 单岛 G1 harness（或确认 Modal 单岛不需要外部 syncer）。之后按 8.1 写计划（`H100!:1`、运行前断言 GPU 名、前缀 algo1b-、容差与门槛事先写定）再执行。

### 对 INFRA / ALGO-CAP 的接口需求

- INFRA：合入 1b-hook.patch（`filtered_samples` 字段约定如上）；7.2 需要 driver 每轮事件带本岛 `trained_samples`/`trained_groups`，可由 `RolloutBatchHandle.groups` 算出。
- ALGO-CAP：P0 4.2（driver 改为调用 `spec.expects_gradient`）。6.5 依赖这项与 1b-hook.patch。合入 1b-docs.patch（MILES_RL.md 算法小节）。
- 2a（ALGO-SEQ）：接手 `reward_pipeline.py`。修改这个文件会改变分派器的源码哈希，引用它的 spec（包括 `examples/rl_algorithms/dapo-like.json`）需要重新生成。

### 待批准

- 2.3/3.3/4.3/7.1 的"upstream parse_args"口径：venv 缺 megatron.training，改为"venv 中用 Miles 参数提供器解析 + 钉住镜像中跑完整 parse_args"。能否据此勾选？
- 3.2：用 learner 入口的两岛校验代替 fake 组合根的两岛测试，能否算满足原文？

### 下一步（可直接执行）

1. 1b-hook.patch 合入后，去掉 `tests/test_rl_grpo_knobs.py` 中 `needs_hook` 的 skip 并复跑；随后勾选 6.3。
2. 1b-docs.patch 合入后勾选 5.5、6.4、7.3。
3. P0 4.2 与 1b-hook.patch 都合入后，补 6.5 的 fake driver 测试。
4. harness 确定后执行 8.x。
