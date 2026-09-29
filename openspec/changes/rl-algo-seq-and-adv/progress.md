# rl-algo-seq-and-adv 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)。

## 2026-09-29（Agent ALIGN，阶段 0）

- 规划文档从 `/home/michael/work/rl-algos`（分支 `rl-algorithms` HEAD 18695ae）复制到 `rl-infra-spec`（`5753e30`），此后以该分支副本为准。

## 2026-09-29（Agent ALGO-2a，实现轮）

### 分支与状态
- worktree `/home/michael/work/algo-2a`，分支 `algo-2a`（基于 `origin/algo-1b` 40ee1a2），已普通 push 到 `origin/algo-2a`。
- 合并：`origin/algo-cap` ebd436b（756946b）、`origin/algo-cap` 63a78bd（f78dee6），均按主 agent 指示 merge，未 rebase。
- 本轮提交：9dec0f7（实现+测试）、9891c2d（G1 计划，启动前）、68c44af（G1 attempt 1 记录与修复）、ece6744（适配 P0 规则绑定与 `dimension:name` 放行名）及本文件所在提交。
- 未提交改动：`yeto/rl/algos/grpo_knobs.py` 一行（给 `register_gradient_rule` 补 `mechanism="features:overlong_filter"`），**仅本地测试用、不提交**——该文件归 ALGO-1b；合入 algo-cap 63a78bd 后，1b 的旧调用使所有扩展加载失败，等 1b 修复后 merge `origin/algo-1b`。

### 改动文件（本 change 自有）
- `yeto/rl/algos/seq_adv.py`：注册模块。字段 `advantage.gamma`（`--gamma` 映射行，默认 1.0 不输出）、`advantage.gdpo`；变换 `maxrl`/`mapo`/`gdpo`（注册进 1b 分派器）；机制 `features:maxrl|mapo`（二值奖励）、`features:gdpo`；拒绝规则 7 条；梯度规则（绑定 `advantage_estimators:gspo`、`features:gdpo`、`advantage_estimators:reinforce_plus_plus`）；runtime attr `yeto_rl_seq_adv`（GDPO 分量配置，带哈希校验）；`clipfrac_from_losses`/`aggregate_clipfrac`（pg_clipfrac → masked_fraction，token 加权）。
- `yeto/rl/algos/gdpo_reward.py`：两分量示例 reward（correctness + format）。
- `yeto/rl/algos/__init__.py`：`EXTENSION_MODULES` 加一行（唯一允许的共享改动）。
- 测试：`tests/test_rl_adv_transforms.py`、`tests/test_rl_seq_adv.py`、`tests/test_rl_seq_adv_miles.py`（miles-next-venv）。
- 规划目录：`baseline-failures.txt`、`after-failures.txt`、`failure-diff.txt`、`examples/*.json` + `make_examples.py`、`evidence/dry-run/`、`evidence/g1/`。

### 1.2 与 P1-b 分派器实际接口的名称差异（design D5/D8/D9）
| design | 实际（1b 40ee1a2 / P0 63a78bd） | 处理 |
|---|---|---|
| D5 `advantage.transform ∈ {builtin, maxrl, mapo, gdpo}` | 1b 字段 `advantage.transform`，默认值名 `grpo_default`（非 `builtin`）；注册点 `register_advantage_transform(name, fn(args,samples,rewards,groups,params), validate=)` | 只加 `maxrl`/`mapo`/`gdpo` 三个键 |
| D5 分组与 rollout_key 合并 | `reward_groups`、`rollout_segments`、`shared_rollout_rewards`、`rollout_key` | 直接复用；段间奖励不一致沿用 1b/Miles 的 `ValueError` 文本 |
| D6 `advantage.gdpo = {components, whiten}` | 1b 的 `transform_params` 只收 JSON 标量，放不下分量列表 | 新字段 `advantage.gdpo`；分派器只传 `transform_params`，所以 GDPO 配置经 P0 `register_runtime_attrs` 下发为 `args.yeto_rl_seq_adv`（带 SHA256） |
| D8 分派器指标通道 | 1b 只有 `emit_event(args, event)`（日志 + 事件磁带），没有进入 `RolloutBatchHandle`/`GroupMetadata` 的通道 | 变换每轮发 `rl_advantage_transform` 事件（全错/全对组、非零 advantage 数）；进入 `batch_summary` 需 INFRA 挂载点（需求 R2） |
| D9 放行 | P0 `--rl-allow-unverified-mechanism`，63a78bd 起名称为 `dimension:name`，与任何外层同步组合都拒绝 | 第 6 组复用，不另建 |
| 梯度规则 | P0 `register_gradient_rule(name, rule, *, mechanism)`，只能放宽 | GSPO 全裁放宽可用；GDPO/REINFORCE++ 需要"收紧"（D8），见 2a-shared.patch hunk 2 |
| 变换模块身份 | 1b 文档假设变换写在 `reward_pipeline.py` 内（受分派器 PluginRef 覆盖）；本轮派发不允许改该文件 | 拒绝规则要求 `plugins` 含 `yeto.rl.algos.seq_adv.registered_transforms` 的 PluginRef，使 `seq_adv.py` 源码哈希进入算法哈希 |

### 任务状态（五种之一）【已过期：以文末"当前状态总表"为准】
| task | 状态 | 证据 / 说明 |
|---|---|---|
| 1.1 | 完成 | `baseline-failures.txt`：756946b 上 73 failed + 26 errors = 99 条，与 pytest 汇总一致 |
| 1.2 | 完成 | 上表 |
| 2.1 | CPU 通过 | `test_rl_seq_adv.py::test_gspo_without_explicit_clip_*`、`test_gspo_with_advantage_transform_not_opened`。提示以附加拒绝规则 `seq_adv_gspo_clip_hint` 给出（P0 原报错未改） |
| 2.2 | 未完成 | 纯函数与三种情况单测已通过（`test_clipfrac_*`）；"adapter 读取并写入每轮事件"需 INFRA 在 `state_plugin.train_one_step` 包装里记录 `result[0]["pg_clipfrac"]` 与每步 loss token 数，并由 `trainer.step_metrics` 用 `seq_adv.clipfrac_from_losses` 填 `masked_fraction`（需求 R1） |
| 2.3 | CPU 通过 | 合入 algo-cap 63a78bd（驱动已调用 `gradient_expectation`）后四个 fake driver 用例全部运行并通过（`test_fake_driver_gspo_*`、`test_fake_driver_nonfinite_grad_norm_fails_for_gspo`、`test_gspo_expects_gradient`）。阈值 1−1e-9（D2） |
| 2.4 | CPU 通过 | miles-next-venv：`test_rl_seq_adv_miles.py::test_gspo_*`（序列 ratio、序列级 clip、全裁 clipfrac=1 且梯度为 0、部分裁剪有梯度） |
| 2.5 | CPU 通过 | `test_gspo_explicit_clip_parses_upstream` |
| 3.1 | 已实现 | 映射行经 `register_flag` 从 `seq_adv.py` 注册；`--lambd` 仍未映射；upstream 存在性测试与 `test_rl_argv_snapshot.py` 不改即通过。但 P0 的 `test_design_d3_flags_are_all_mapped`（等式）与 `test_unmapped_objective_flag_rejected[--gamma]` 因此失败，修改在 2a-shared.patch hunk 3（待 ALGO-CAP 合入），故不勾选 |
| 3.2 | CPU 通过 | `test_gamma_*`、`test_rpp_family_whiten_rules`（whiten=false 由 P0 `rpp_requires_whiten` 拒绝；rpp 家族未声明＝可表达未开放） |
| 3.3 | CPU 通过 | `test_rpp_dispatcher_equals_builtin`（含多段 rollout，`torch.equal`） |
| 3.4 | CPU 通过 | `test_rpp_returns_include_reward_kl_and_whiten_to_zero_mean`、`test_rpp_baseline_advantages_include_reward_kl`（gloo 单进程 DP=1） |
| 3.5 | CPU 通过 | `test_rpp_whiten_reward_kl_parses_upstream`（两种估计方式） |
| 3.6 | 未完成 | 规则已实现（`rpp_expects_gradient`），单测覆盖期望/不期望/读不到三种；fake driver 覆盖期望与不期望。但 D8 的"组均值不同或 reward KL 而 reward_std 全 0 仍期望梯度"需要 P0 钩子能收紧（2a-shared.patch hunk 2），当前 P0 下这部分比 spec 宽 |
| 4.1 | CPU 通过 | `test_transform_requires_grpo`、`test_binary_transforms_need_binary_reward_and_no_overlong_penalty`、`test_non_binary_reward_fails_round` |
| 4.2 | CPU 通过 | `tests/test_rl_adv_transforms.py`（G=1、全错、全对、[1,0,0,1]、多段三段共享、段间不一致、有限值） |
| 4.3 | CPU 通过 | `test_mapo_half_correct_equals_builtin_grpo`（含多段，`torch.equal` 对 Miles `_post_process_rewards`） |
| 4.4 | CPU 通过 | `test_nonzero_std_iff_nonzero_output`（G=1..8 全部二值组合，MaxRL 与 MAPO）、`test_fake_driver_all_wrong_round_zero_grad_passes` |
| 5.1 | CPU 通过 | `test_gdpo_declaration_*` |
| 5.2 | CPU 通过 | `test_gdpo_reward_vector_failures`、`test_gdpo_segments_must_share_vector`、`test_gdpo_config_hash_checked` |
| 5.3 | CPU 通过 | `test_gdpo_matches_reference`、`test_gdpo_constant_component_contributes_zero`、`test_gdpo_whole_batch_constant_only_subtracts_mean`、`test_gdpo_multi_segment` |
| 5.4 | CPU 通过 | `test_example_reward_writes_accepted_components` |
| 5.5 | 未完成 | 规则已实现、单测通过；fake driver 无法携带分派器的非零条目数（需 INFRA 通道 R2），且"标量 reward_std 全 0 但分量非零"需要收紧钩子（hunk 2） |
| 6.1 | CPU 通过 | `test_allowance_launches_single_island_and_refuses_otherwise`（6 个机制，放行名 `dimension:name`，多岛/外层同步被拒） |
| 6.2 | CPU 通过 | `test_fake_declaration_launches_each_mechanism`（声明经 `fake_capabilities(...)` 覆盖参数给出，`fake.py` 未改——归 ALGO-CAP）；`test_miles_adapter_declares_none_of_them` |
| 6.3 | 未完成 | 文档为补丁 `infra-drafts/2a-docs.patch`（docs 算法小节归 WP-CAP）；dry-run 证据 `evidence/dry-run/`（放行名为 63a78bd 之前的旧格式，1b 修复后按新格式重跑） |
| 7.1 | 完成 | alignment.md §7b-1："用户已授权'执行必要 GPU 实验不再逐轮批准'"；本 change 预算与计划见 `evidence/g1/plan.md`（上限 $20） |
| 7.2–7.4 | 未完成 | attempt 1（H100!，207 s）因 harness 关闭 Ray dashboard 失败（`evidence/g1/plan.md`），已修复；attempt 2 按主 agent 指示暂停：等 P0"无 syncer 单岛入口"与 1b 修复 grpo_knobs |
| 7.5 | 未完成 | 依赖 G1；`entry.py` 能力声明归 ALGO-CAP，届时以补丁提交 |
| 7.6 | 未完成 | 依赖 7.5 正式声明 + 两岛 head 放置（Modal 不能承载 head） |
| 7.7 | 未完成（attempt 1 部分已完成） | attempt 1：`modal app stop -y algo2a-g1`，app `stopped`/0 tasks，watchdog 已杀（`evidence/g1/teardown-attempt1.txt`） |
| 7.8 | 完成（仅记录） | 效果 A/B 不在本 change |
| 8.1 | 未完成 | 失败集合差 2 条（见下），由 2a-shared.patch hunk 3 消除 |
| 8.2 | 完成 | `openspec validate rl-algo-seq-and-adv --strict` → valid |
| 8.3 | 完成 | 本表 |

### 测试命令与结果【历史记录，数字已过期】
- CPU（历史，f78dee6 时）：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rl_seq_adv.py tests/test_rl_adv_transforms.py` → 116 passed（当时本地加 grpo_knobs 一行修复，未提交；已于 f3ee263 还原）。2a-shared.patch 应用后同样 116 passed。
- Miles 对照：`OMP_NUM_THREADS=1 PYTHONPATH=$PWD:/home/michael/work/miles-next /home/michael/work/miles-next-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rl_seq_adv_miles.py` → 20 passed（Miles 0394715，torch 2.13.0）。同环境 `test_rl_algorithm_flags_upstream.py test_rl_reward_pipeline_equivalence.py` → 89 passed。
- 全量（改动前后同口径，`--continue-on-collection-errors -p no:cacheprovider`）：前 756946b 73F+26E=99；后 9dec0f7 75F+26E=101。按 id 差集（`failure-diff.txt`）只多：
  - `tests/test_rl_algorithm_flags.py::test_design_d3_flags_are_all_mapped`
  - `tests/test_rl_algorithm_flags.py::test_unmapped_objective_flag_rejected[--gamma]`
  两者是 3.1 要求的 `--gamma` 映射直接导致的 P0 测试期望变化，修改见 2a-shared.patch hunk 3。与 algo-1b 40ee1a2 的 94 条相比，756946b 多出的 5 条（`test_rl_algorithm_flags` 4 条、`test_rl_algorithm_spec_v2::test_launch_and_island_checks`）来自 algo-cap 与 1b 合并本身，改动前已存在。
- 合入 algo-cap 63a78bd 后（f78dee6）分支上 grpo_knobs 调用旧签名，扩展加载整体失败——待 1b，届时重跑全量。

### 接口需求（给 INFRA / ALGO-CAP / ALGO-1b）
- **R1（INFRA，2.2/7.2）**：`state_plugin.install_grad_norm_recorder` 的 `train_one_step` 包装同时记录 `result[0].get("pg_clipfrac")` 与该步 loss token 数；`MilesTrainerGroup.step_metrics().masked_fraction` 对 gspo 用 `yeto.rl.algos.seq_adv.clipfrac_from_losses(steps, token_counts)`；driver 每轮事件写 `masked_fraction`（clipfrac）。
- **R2（INFRA，5.5/3.6）**：把分派器每轮的 `nonzero_advantages`（`rl_advantage_transform` 事件字段）带进 rollout 元数据 → `RolloutBatchHandle`（或 batch 级属性），供 `batch_summary` 读取；读不到时规则按期望梯度处理。
- **补丁 `/home/michael/work/infra-drafts/2a-shared.patch`**（基于 f78dee6，`git apply --check` 通过）：①`reward_pipeline.post_process` 查表前 `load_extensions()`（否则 Miles rollout 进程里 maxrl/mapo/gdpo 不存在）；②梯度规则可返回 True 收紧（仅对 spec 需要的机制生效，默认 GRPO 不变）；③P0 flags 测试改为子集并换用 `--value-clip`；④`python -m ...algorithm_flags --dry-run` 以 `__main__` 运行时扩展注册的映射行落在另一份模块里（证据 `evidence/dry-run/gamma_extra_grpo.txt`），改为调用包模块的 `main()`。
- **补丁 `/home/michael/work/infra-drafts/2a-docs.patch`**：`docs/MILES_RL.md` 小节（6.3）。
- **ALGO-1b**：`grpo_knobs.py` 需给 `register_gradient_rule` 补 `mechanism=`（合入 63a78bd 后必需）。

### 云资源与费用
- attempt 1：Modal app `algo2a-g1`（ap-q20m8GE1i2WpZcKRBMNj1K），Sandbox `sb-NCEOOOmOOTK45bK8f140xp`，GPU H100! 断言 "NVIDIA H100 80GB HBM3"，16 CPU/128 GiB，2026-09-29T16:50:06Z 起，207 s。预估 ≈$0.35（H100 $3.95/h + CPU/内存；未经账单核实）。回收：手动 `terminate` + `modal app stop -y`，app `stopped`/0 tasks；watchdog 已杀。无卷、无命名 secret（拉取凭据为内存 `Secret.from_dict`）。

## 待批准
- 无新增。GPU 按 §7b-1 已授权；G1 暂停是主 agent 的调度指示。

## 阻塞与下一步（可直接执行）
1. 1b 修复 `grpo_knobs.py` 后：`git -C /home/michael/work/algo-2a fetch -q origin && git merge origin/algo-1b`，还原本地一行修复（`git checkout -- yeto/rl/algos/grpo_knobs.py`），重跑全量与 miles 对照、按新放行名重生成 `evidence/dry-run/`。
2. 主 agent 通知 P0 无 syncer 单岛入口 SHA 后：merge algo-cap，按 `evidence/g1/plan.md` 用新入口重做 attempt 2（`g1_sbx.py`，先 watchdog，后 `timeout 5700`），7 个 run；通过的机制在 `entry.py` 声明（补丁交 ALGO-CAP）。
3. 7.6 G3 需两岛 head 放置（本机或 Nebius head），在 7.5 合入后计划。

## 2026-09-29 主 agent 决定（分派器哈希覆盖）
- 1b 将让分派器 PluginRef 覆盖所有注册 shaper/transform 的模块，并在 post_process 前 `load_extensions()`、未知 transform 启动前拒绝。2a-shared.patch hunk 1 届时作废。
- 本 change 保持注册写在 `seq_adv.py`；已确认它在 `EXTENSION_MODULES` 中，单独导入不引入 torch/miles（`python -c "import yeto.rl.algos.seq_adv"` 通过）。
- 1b 推送后：merge `origin/algo-1b`；删除 `seq_adv_transform_identity` 拒绝规则与 `plugins` 要求（改由分派器哈希覆盖，G1 放行清单去掉 `features:plugins`）；用 `make_examples.py` 重新生成全部示例 spec，重跑 dry-run 证据与全量测试。

## 2026-09-29 主 agent 决定（G1 前提）
- G1 证据必须对应已提交并推送的 SHA：**不再在上传代码上临时打补丁**（evidence/g1/plan.md 中"plus 2a-shared.patch applied"一条作废，attempt 2 前单独提交修订后的计划）。
- 等待合入：load_extensions 与哈希覆盖（1b）；梯度收紧、flags 测试、dry-run `__main__` bug、docs（ALGO-CAP，来自 2a-shared.patch / 2a-docs.patch）；R1/R2（INFRA）。另需 P0 无 syncer 单岛入口 SHA 与 1b 修复 grpo_knobs。
- 1b 推送后撤掉 `plugins` 列 seq_adv 源码哈希与额外放行 `features:plugins` 的做法。
- 预算上限 $20 获准。7.6 G3 放置：参考 R0 7.1（Modal 2 岛 + 本机 syncer，非 head 模式）。
- 8.1：基线中 5 条合并引入的失败待 1b/ALGO-CAP 修复后复核。

## 2026-09-29 merge origin/algo-1b a978b3c（f3ee263）
- 本地 grpo_knobs 临时改动已还原，工作区干净。
- `tests/test_rl_seq_adv.py tests/test_rl_adv_transforms.py` 116 passed；miles-next-venv `test_rl_seq_adv_miles.py` 20 passed。
- 全量：70F + 26E = 96（`after-failures-f3ee263.txt`）。与 756946b 基线相比：合并引入的 5 条（test_rl_algorithm_flags 4 条、test_rl_algorithm_spec_v2::test_launch_and_island_checks）已消失（8.1 复核完成）；仍只多出 2a-shared.patch hunk 3 对应的 2 条（`test_design_d3_flags_are_all_mapped`、`test_unmapped_objective_flag_rejected[--gamma]`）。
- 仍等：1b F1（哈希覆盖/load_extensions）、ALGO-CAP 合入 2a-shared 其余部分与文档、INFRA R1/R2、P0 无 syncer 单岛入口。

## 2026-09-29 merge origin/algo-cap 2f9f02c（570e468）
- ALGO-CAP 已合入 2a-shared.patch 的梯度收紧、flags 子集测试、`__main__` dry-run 修复，以及 2a-docs.patch（`docs/MILES_RL.md` 第 865 行起）。P0 无 syncer 单岛入口 09607d6：`--rl-single-island-no-sync`。
- 新增测试 `test_p0_hook_tightens_for_gdpo_and_rpp`、`test_fake_driver_rpp_reward_kl_constant_round_expects_gradient`：`test_rl_seq_adv.py` 82 passed；miles-next-venv `test_rl_seq_adv_miles.py test_rl_algorithm_flags_upstream.py` 40 passed。
- dry-run 证据按 `dimension:name` 重跑（`evidence/dry-run/`），示例 spec 按新分派器哈希重生成；结果与文档一致（放行后接受、未放行拒绝、gspo 无 clip / rpp gamma=0.99 / grpo+gamma / --lambd 拒绝、gamma=1.0 接受）。
- 全量：68F + 26E = 94（`after-failures-570e468.txt`），是 756946b 基线（99 条）的真子集：没有新增失败，另有 5 条合并引入项已被他人修复。
- 状态更新：3.1、3.6、6.3、8.1 → 完成（CPU 通过）。5.5 仍未完成（等 INFRA R2：非零条目数进入 batch_summary）；2.2 等 R1。
- G1 仍等 1b F1（哈希覆盖/load_extensions）与 INFRA R1/R2；届时撤掉 `plugins` 身份要求、重生成示例、单独提交改用 `--rl-single-island-no-sync` 的修订计划后再跑（这将是该入口首次真实 GPU 使用，会记录）。

## 2026-09-29 merge algo-1b 2722cad / algo-cap 6149a90 / infra-a 39fa0ac；G1
- 适配：撤掉 `seq_adv_transform_identity`，改用 `grpo_knobs.with_pipeline_plugins`；Miles 对照测试补 `pipeline_sha256`；变换写 `args.yeto_rl_round_metadata["nonzero_advantages"]`（R2）；新增 2.2/5.5 测试与示例新鲜度守卫测试。
- 全量（`after-failures-infra-a.txt`）：70F+26E。相对 756946b 基线只多 1 条 `tests/test_rl_grpo_knobs.py::test_examples_build_and_translate[dapo-like]`：1b 的示例 spec 在 1b 分支上生成，不含 seq_adv 模块的 PluginRef（F1 要求所有注册模块都列入）——需 1b 在集成分支重生成该示例；另有 P0 已知的 `test_each_mapped_field_translates[change9]`（ALGO-CAP 修复中，基线中也有）。
- G1（`evidence/g1/plan.md` attempts 2–4，代码均为已推送 SHA，无临时补丁）：
  - attempt 2（8d7f752）：gspo_s2 生成与首轮两步训练成功（step0 clipfrac 0.0，step1 clipfrac 0.4375），随后 `LocalStepReceipt(algorithm='gspo')` 被 `contracts._ALGORITHMS={"grpo","sao"}` 拒绝。所有非 grpo 估计方式受影响 → 补丁 `infra-drafts/2a-receipt.patch`；GSPO/rpp/rpp_baseline 的 G1（7.2/7.3）等其合入。
  - attempt 3：我的示例 spec 过期（seq_adv.py 改后未重生成），启动前被拒；已修并加守卫测试。
  - attempt 4（e54d2f7）：MaxRL、MAPO、GDPO 全部通过预声明标准 → 7.4 完成。
  - 发现 INFRA R2 缺陷：`rl_round_trained.nonzero_advantages` 滞后一轮。
- 7.5：`infra-drafts/2a-entry.patch` 声明 features maxrl/mapo/gdpo（ALGO-CAP 合入；还需 1b 声明 `reward_postprocessors:custom_reward_postprocess`）。
- 费用：attempts 1–4 合计 2056 s 单卡 H100，估 ≈$3.1；无残留（两个 app stopped/0 tasks，watchdog 已杀）。
- 任务状态变化：2.2、5.5、7.4 → 完成；7.2、7.3 未完成（receipt 缺陷）；7.5 未完成（补丁待合入）；7.6 未完成（待 7.5，按 R0 7.1 Modal 2 岛 + 本机 syncer）；7.7 部分（已回收，汇总待 7.2/7.3/7.6 后）。

## 2026-09-29 merge infra-a e9f20cc + 独立审查处理
- R2 新接口：变换每个 rollout 调一次 `rollout_meta_hook.record_round_metadata(args, <round id>, nonzero_advantages=n)`。**轮次 id 取自 driver 发布的 policy token（`yeto:<rollout_id>:<hash>`，`current_policy_token()`）**，不取 `Sample.rollout_id`：后者在 Miles 中是轨迹键（多段合并键，`agentic_tool_call.py:147`），不是训练轮次。给 INFRA 的提示：`record_round_metadata(args, samples)` 与 `build_metadata` 从 `Sample.rollout_id` 推轮次，在多段 agentic rollout 上会拿到轨迹 id。无 sink（CPU）时不记录。测试 `test_transforms_report_round_counter_for_the_metadata_hook`（轨迹 id 100+i，轮次 7 → 记录 (7, 2)）。
- 审查 1：`evidence/g1` 下被 `*.log` 忽略的日志已扫描（仅有值为 None 的 password/api_key 配置项）后 `git add -f`；0 字节 `out.tgz` 已删除。7.4 保持勾选（证据已入库）。
- 审查 2：5.5 取消勾选。需 R2 修复后的逐轮对应证据（端到端或 GPU）再勾。
- 审查 3：8.1 取消勾选。当前 HEAD 全量 69F+26E（`after-failures-review.txt`），相对 756946b 基线只多 `test_rl_grpo_knobs.py::test_examples_build_and_translate[dapo-like]`——依赖 1b 在集成分支重生成该示例（须含 seq_adv 模块 PluginRef）。
- 审查 5：`test_rl_seq_adv_miles.py` 文件头按实际比较方式改写（`torch.equal` 仅用于分派器 vs Miles 与 clip 指示；2.4/3.4 为 `allclose`）。
- 审查 6：REINFORCE++ + reward KL 且全部奖励相同（第 0 轮 LoRA B=0 时 KL=0）不再判"期望梯度"：此情形规则不给结论（退回 R0）。组均值不同或组内有方差仍判期望梯度，其他情形未放宽。测试 `test_fake_driver_rpp_reward_kl_round0_identical_rewards_is_not_a_failure`。D8 的"reward KL 使 advantage 不全相等"分支因 KL 大小未上报而不执行，记为已知限制。
- 审查 7：plan.md 中 GDPO 表述改为如实"D8 收紧分支在 GPU 上未触发"。2a-entry.patch 的 gdpo 声明等 R2 修复后的证据再定（maxrl/mapo 不受影响）。
- 审查 8（A4）：GSPO 且 context parallel size ≠ 1 在启动前拒绝（`register_launch_check("seq_adv")`），测试 `test_gspo_context_parallel_refused_before_launch`。
- 测试：`tests/test_rl_seq_adv.py tests/test_rl_adv_transforms.py tests/test_rl_round_accounting.py` 140 passed；miles-next-venv `test_rl_seq_adv_miles.py test_rl_reward_pipeline_equivalence.py` 89 passed；`openspec validate --strict` valid。

### 当前状态总表（取代上文各表）
| task | 状态 |
|---|---|
| 1.1 1.2 | 完成 |
| 2.1 2.3 2.4 2.5 | CPU 通过 |
| 2.2 | CPU 通过（INFRA R1 + trainer 测试）；GPU 上的 masked_fraction 待 GSPO G1 |
| 3.1 3.2 3.3 3.4 3.5 | CPU 通过 |
| 3.6 | CPU 通过（审查 6 修正后；reward-KL 分支不执行，见上） |
| 4.1–4.4 5.1–5.4 | CPU 通过 |
| 5.5 | 未完成（待 R2 修复后的逐轮对应证据） |
| 6.1 6.2 6.3 | CPU 通过 / 完成 |
| 7.1 | 完成 |
| 7.2 7.3 | 未完成（attempt 2 被 receipt 缺陷挡住；INFRA 已修，待按单独提交计划经 launcher 重跑） |
| 7.4 | GPU 验收通过（attempt 4，e54d2f7） |
| 7.5 | 未完成（2a-entry.patch：maxrl/mapo 可声明；gdpo 待 R2 证据） |
| 7.6 | 未完成（待 7.5；按 R0 7.1 Modal 2 岛 + 本机 syncer） |
| 7.7 | 未完成（attempts 1–4 已回收并核实；汇总待剩余运行） |
| 7.8 8.2 8.3 | 完成 |
| 8.1 | 未完成（dapo-like 依赖 1b） |

## 2026-09-29 G1 attempt 5（launcher 入口）与 merge infra-a 8cf1dec（947fdbc）
- 计划 b06f928（启动前提交）。try 1：未导出私有镜像拉取凭据，Modal 镜像构建失败，未起 GPU；已修 launch_run.sh（进程内解码，不打印）。
- try 2：gspo_s2 在 launcher no-sync 入口上跑完（learner finalized、job SUCCEEDED、launcher rc 2＝预声明的"产物不可取"提示），3 轮×2 步，第二步 clipfrac 0.1875/0.5，grad_norm 全有限。但事件磁带不在流式日志里、Modal 岛的 ~/yeto-output 也取不回，预声明标准 1（rl_round_trained 计数）、5（unverified 事件）以及 R1 masked_fraction、5.5 的逐轮对应**无法评估** → 只记录，不计通过。
- 因 INFRA 8cf1dec 改了 receipt 语义，手动停掉刚启动的 gspo_s1，其余未启动；全部 app stopped/0 tasks，watchdog 已杀。
- 已 merge infra-a 8cf1dec：本 change 测试 141 passed；全量 `after-failures-947fdbc.txt`，相对基线新增仍只有 dapo-like（依赖 1b）。
- **阻塞（需 P0/INFRA）**：Modal no-sync 岛的事件磁带需进入流式日志或被拉回，否则 7.2/7.3 的标准 5 与 5.5 的逐轮证据都无法在 launcher 入口上取得。解除后按 attempt 5 计划重跑 gspo_s2、gspo_s1、rpp、rpp_baseline、gdpo。
- 累计 GPU：attempts 1–5 ≈ 2056 s + ≈ 600 s 单卡 H100，估 ≈ $4（未核账单），上限 $20。

## 2026-09-29 G1 attempt 6 PASS + merge algo-cap 50fe818（2751cda）
- attempt 6（计划 fb588a4 启动前提交；代码 fb588a4；launcher `--rl-single-island-no-sync --controller local`，事件磁带回传）：gspo_s2、gspo_s1、rpp、rpp_baseline、gdpo 五个 run 全部满足预声明标准（逐项见 `evidence/g1/plan.md` Attempt 6 与 `attempt6/<run>/criteria.json`）。
  - GSPO：optimizer_steps=2 时第二步 clipfrac 0.1875/0.5/0.5；=1 时全 0（符合 D1）。
  - 5.5：gdpo 磁带逐轮 nonzero_advantages 32/24/32 与分派器逐轮计数一致 → 勾选。
  - **缺陷（INFRA R1）**：所有 `rl_round_trained` 的 masked_fraction/clip_fraction 为 null（gspo_s2 的 Miles 日志 pg_clipfrac 为 0.5）。GSPO 全裁判定在真实引擎上因此读不到值（按设计退回严格侧，不会误放宽）。2.2 保持"CPU 通过"。
- 能力测试已按主 agent 要求改为断言具体声明集（3d82d0e）。
- 7.5：`infra-drafts/2a-entry.patch` v2（基于 2751cda）追加声明 gspo、reinforce_plus_plus、reinforce_plus_plus_baseline、gdpo；支撑机制（eps_clip/clip_higher、whiten_advantages、custom_reward_postprocess）是否声明由负责人决定。
- 全量（`after-failures-2751cda.txt`）69F+26E，相对基线仅多 dapo-like（依赖 1b）。validate valid。
- 费用：attempt 6 ≈ $3.9；累计 ≈ $8（未核账单），上限 $20；全部 app stopped/0 tasks，watchdog 已杀。

### 当前状态总表（取代上一张）
| task | 状态 |
|---|---|
| 1.1 1.2 7.1 7.8 8.2 8.3 | 完成 |
| 2.1 2.3 2.4 2.5 3.1–3.6 4.1–4.4 5.1–5.4 6.1–6.3 | CPU 通过 |
| 2.2 | CPU 通过；GPU 上 R1 值为空（INFRA 缺陷） |
| 5.5 | GPU 验收通过（逐轮磁带证据） |
| 7.2 7.3 7.4 | GPU 验收通过（Attempt 6 / Attempt 4） |
| 7.5 | 未完成：maxrl/mapo 已由 ALGO-CAP 声明；其余见 2a-entry.patch v2 |
| 7.6 | 未完成（待 7.5 合入；Modal 2 岛 + 本机 syncer） |
| 7.7 | 未完成（已完成运行均已回收并核实；汇总待 7.6） |
| 8.1 | 未完成（dapo-like 依赖 1b） |

## 2026-09-29 merge origin/algo-1a、origin/algo-1b、origin/infra-a（29e0fec）+ 示例重生成
- 逐个 merge，未 rebase。algo-1a 与本分支在 `yeto/rl/algos/__init__.py` 冲突（各加一行），按 rl-integ-2 的顺序解决（grpo_knobs、seq_adv、mismatch_correction）。
- `make_examples.py` 重生成 maxrl/mapo/gdpo 示例（分派器 PluginRef 的合并哈希因 mismatch_correction 与 1b 最新 reward_pipeline 变化）。验证：在 rl-integ-2 ff61352 上 merge 本分支（无冲突），`yeto/rl/algos/*.py` 与本分支逐字节相同，用本分支示例跑 `test_example_specs_are_current` 6 passed，在该树上再生成示例无差异。
- **注意**：示例 spec 的 PluginRef 哈希随任一注册模块（reward_pipeline、grpo_knobs、seq_adv、mismatch_correction、sample_filters 等）源码变化而变。集成分支定稿后须在其最终 SHA 上再运行一次 `PYTHONPATH=. python openspec/changes/rl-algo-seq-and-adv/examples/make_examples.py`（`test_example_specs_are_current` 会在过期时失败）。
- 本 change 测试 133 passed（yeto-venv）；全量见 `after-failures-29e0fec.txt`。

## 2026-09-29 P0 4373cd9（记录）
- 事件磁带回传已扩展到所有 Modal ports 岛（含带 syncer 的多岛），磁带在 `<run dir>/events/<island>.jsonl`；缺 `rl_learner_finalized` → 退出码 3。7.6 MaxRL 两岛 G3 计划将用它读每岛磁带（算法哈希一致、外层应用后状态 hash 一致、不变量无误报），放置按 R0 7.1（Modal 2 岛 + 本机 syncer）。
- 7.6 仍等 7.5 声明在集成分支合入（主 agent 统一处理），届时 merge 后先单独提交计划再跑。

## 2026-09-29 merge origin/algo-1b 176ba25+（fcd58b9）与示例重生成
- 已 merge `origin/algo-1b`（echo 补丁后的 reward_pipeline），`make_examples.py` 重生成 maxrl/mapo/gdpo 示例。
- `origin/integ-decl`（0f13aa7）**未 merge**：它与 algo-1b 在他人文件上有内容冲突（`yeto/rl/algos/grpo_knobs.py` 注释、`tests/test_rl_grpo_knobs.py` 声明集合、`tests/test_rl_algorithm_capabilities.py`、`docs/MILES_RL.md` 的 mismatch_metrics 段），属于 1b/ALGO-CAP/主 agent 的决策，本 change 不替他们裁决；已 `merge --abort`。
- 验证：临时 worktree 取 integ-decl 0f13aa7，放入本分支的示例文件，`test_example_specs_are_current` 6 passed（示例哈希覆盖的模块 reward_pipeline/seq_adv 在两边相同；grpo_knobs、mismatch_correction 的差异不进入这些示例）。本分支本 change 测试 133 passed。
- 再次提醒：集成分支定稿后须在最终 SHA 上重跑 make_examples.py。

## 2026-09-29 7.6 G3（YETO_SHA 4652f73）与 7.7 汇总
- 计划 bae5600 启动前提交；结果 `evidence/g3/results.md`。
- **7.6 未通过（按预声明退出码读法）**：磁带判据 1–4 全部成立（3 次外层同步 2 响应无陈旧；两岛算法哈希一致、无放行；v0..v3 发布 hash 两岛一致；每岛 3 轮有限、finalized、无失败事件），但 head 退出码 143：训练完成后本机 syncer 被终止，岛 1 在 `ray.shutdown` 中 KeyboardInterrupt、作业记为 FAILED，launcher 进入恢复循环挂起（P0 在 c098b5b 修复的缺陷，4652f73 不含），我手动结束 head。检查脚本计数 bug 在运行后修正并注明（不改判据）。
- 观察：第 2 轮两岛 nonzero_advantages=0、grad_norm=0.0 且无不变量失败——"不期望梯度"分支在 GPU 上被触发。
- 重跑需在 ≥ c098b5b 的 SHA 上单独提交计划（退出码 4 = 失败）。
- **7.7 汇总（全部为估算，未核账单）**：G1 attempts 1–4 ≈ $3.1；attempt 5 ≈ $0.9；attempt 6 ≈ $3.9；G3 ≈ $3 → 合计 ≈ $11，上限 $20。所有本 change 的 Modal app（algo2a-g1 ×2、yeto-algo2a-g1-* ×8、yeto-algo2a-g3）均为 stopped / 0 tasks；本 change 的 watchdog 与本机 syncer（:29420）已停；无卷、无命名 secret。7.7 勾选。

## 2026-09-29 7.6 判定口径修正（主 agent）与补充证据
- 7.6 正式结果：未通过——退出码 143，且判据 4 按事先提交的检查脚本判定"不成立"（`run/check-as-declared.json`）；运行后按事件计数的"成立"只作观察（`results.md` 已改）。
- 根因在 launcher（两岛 finalized 后先停 syncer，岛 1 关闭期异常被判 FAILED 并进入恢复循环；c098b5b 后会返回 4，仍属误判），已由主 agent 交 P0 修复。修复推送后允许在新 SHA 上按单独提交的新计划只重跑一次：判据不变；检查脚本运行前改为按 rl_local_round/rl_round_trained 事件计数（已在 harness 中改好，随新计划提交）；退出码 0/2 由磁带判定，3/4/其他非零为失败。
- watchdog 中途消失：原因未查明（疑似他人按模式批量 kill 同形 `sleep 3300` watchdog）；harness 已加固（独立命名脚本、setsid+nohup、忽略 HUP/INT/TERM、60 s 心跳日志、每分钟自检并按原截止时间重启），见 `evidence/g3/results.md`。
- **5.5 补充证据（GPU）**：7.6 运行中 MaxRL 第 2 轮两岛 `rl_round_trained.nonzero_advantages = 0`、`rl_local_round.grad_norm = 0.0`、无不变量失败事件——"非零 advantage 为 0 → 不期望梯度"分支在 GPU 上真实走到（此前仅 fake driver 覆盖）。

## 2026-09-29 7.6 首次失败根因更正 + 重跑通过
- 【已被最终审查修订，见文末】更正：首次 7.6 的 watchdog 消失**已查明**——algo-1a 在约 20:37–20:40Z 执行 `pkill -x -f "sleep 3300"`（见 algo-1a 7ebfc59 progress），杀掉了本运行 watchdog 的 sleep 子进程，watchdog 随即执行 `modal app stop` 与 pkill 本机 syncer，与本运行收尾期重合；岛 1 在 `ray.shutdown` 中 KeyboardInterrupt、作业被判 FAILED 很可能由此触发（launcher 恢复循环缺陷另由 P0 在 a602fa2 修复）。首次运行正式结果仍按预登记为**未通过**，不改判。
- 重跑（唯一一次；计划 1698744 启动前提交；YETO_SHA=a602fa2；加固 watchdog 唯一脚本名）：退出码 0，判据 1–4 全部通过（`evidence/g3/rerun/results.md`、`rerun/check.json`）→ 7.6 勾选（GPU 验收通过）。主 agent 后来提到的 501d71d 在收到通知时本次重跑已按 a602fa2 提交并启动，按指示未更换。
- 7.7 更新：累计 ≈ $11 + $2.9 ≈ $14（估算，未核账单），上限 $20；app ap-aivVXMrZwnfkkB7zj2rQFm stopped/0 tasks，本机 syncer 已停、29420 关闭、watchdog 已结束。

## 2026-09-29 收尾：7.5 与 8.1（集成分支 integ-decl 501d71d）
- 7.5 勾选：声明位于集成分支 integ-decl 501d71d。临时 worktree 核对：六个示例 spec 用 `yeto launch --dry-run`（两岛 strict-avg，不带放行）全部被接受、`unverified_mechanisms: []`；adapter `algorithm_flags --dry-run` 全部 accepted；未声明项 `features:dual_clip` 被拒（"not supported"）；`gspo_noclip`、`rpp_gamma` 仍按拒绝矩阵被拒。证据 `evidence/7.5-501d71d/`。
- 示例 spec 在 501d71d 上是最新的（`test_example_specs_are_current` 通过，make_examples.py 无差异）。集成分支若再变动注册模块源码，需在最终集成 SHA 上再生成。
- 8.1 勾选：501d71d 全量 68F+26E=94，按 id 与 algo-1b 40ee1a2 基线完全相同；相对本 change 1.1 基线（756946b，99 条，含 5 条当时的合并引入项）无新增。

### 最终状态
全部 task 已勾选（1.1–8.3）。GPU 验收通过：5.5、7.2、7.3、7.4、7.6；7.5 声明在集成分支；其余为 CPU 通过/完成。已知限制：INFRA R1 在 attempt 6 时 GPU 上 masked_fraction 为空（infra-a 后续修复，本 change 未复验 GPU）；REINFORCE++ 的 reward-KL 分支因 KL 大小未上报而不执行。费用累计 ≈ $14（估算），无残留。

## 2026-09-29 最终审查修订（不重跑、不改判）
- 8.1：以 1.1 基线（756946b，99 条）为准：无新增失败；差集仅 5 条非本 change 引入、已被他人修复（id 见 tasks.md 8.1）。
- 7.6 首次失败根因：日志时序与 1a 自报的 pkill 时间窗（20:37–20:40Z）不符——syncer 于 20:36:09–14Z 被 Terminated，app stopped_at 20:36:20Z，且顺序与 watchdog 命令顺序相反。改为"很可能由外部进程被结束触发；归因于 1a 的 pkill 未证实"（evidence/g3/results.md）。上文"已查明"一条作废。
- 2.2：adapter 接线引用 INFRA d9bf29c（R1 真正修复）；GPU 未复验（attempt 6 时 masked_fraction 为 null）。
- 7.7：费用统一为 ≈$14（估算，未核账单）。
- harness/check_g3.py：YETO_SHA 树路径改为参数（第二参数或环境变量 G3_TREE），不影响已有结果。

## 待批准（已知偏离，需用户确认）
1. **GSPO 全裁放宽路径（design D2）在 GPU 上未跑到**：attempt 6 时 `masked_fraction` 为 null（早于 INFRA d9bf29c）。若真实引擎上 masked_fraction 仍为 null，则一轮全部序列被裁、grad_norm=0 时会**误报**零梯度失败（严格侧）。需在 d9bf29c 之后的 SHA 上做 GPU 复验。
2. **REINFORCE++ 在奖励全相同但有 reward KL 时退回 R0 规则**：比 spec.md 第 135–136 行（"本岛本轮 advantage 在白化前不全相等 → 期望梯度"）宽松——只会漏报、不会误报；原因是 reward-KL 大小未上报给 driver（第 0 轮恰为 0）。若要严格满足 spec，需要上报每轮 reward-KL 统计。

## 2026-09-29 R1 GPU 复验（integ-decl 501d71d，计划 f6e25b4 启动前提交）
- 预登记检查脚本结果：**未通过**——脚本按 Miles "step 0" 切分轮次，但 Miles 的 step 编号在整次运行中累计（0..5），导致判定失效；按规则不改判。
- 观察（不计通过）：三轮 `rl_round_trained` 的 clip_fraction/masked_fraction 均非 null，且等于该轮两步 pg_clipfrac 的均值（0.09375、0.25、0.25），与 R1 通道在 GPU 上工作一致。未出现全裁剪轮次，D2 全裁放宽路径仍记为 GPU 未覆盖。已写入 design.md Known deviations。
- 退出码 2（按磁带判定）；app ap-xbkY9rLSJ3vjXCBmcuSrVi stopped/0 tasks；watchdog 已结束。费用 ≈ $0.9；本 change 合计 ≈ $15（估算，未核账单），上限 $20（后续更新见下）。
- 偏离 2（rpp reward-KL）保持待用户决定。

## 2026-09-29 R1 复验重跑（批准的唯一一次，501d71d）
- 计划与修正后的检查脚本 5b0ae97 启动前提交（按 optimizer_steps=2 每两步一组划分轮次）。第一次启动在任何云资源创建前被 launcher 同名磁带保护拒绝（9415033 记录，改用前缀 algo2a-r1-gspo2），随后正式运行。
- 结果：**通过**——三轮 clip_fraction = masked_fraction = 该轮两步 pg_clipfrac 均值（0.09375、0.25、0.25），均非 null；退出码 2 按磁带判定。偏离 1 收窄为：全裁剪放宽路径本身未在 GPU 上发生。
- 费用：本次 ≈ $0.9；本 change 累计 ≈ $16（估算，未核账单），上限 $20。app ap-sEPPygSJmxCvTy4kz7Y1bk stopped/0 tasks，watchdog 已结束。
