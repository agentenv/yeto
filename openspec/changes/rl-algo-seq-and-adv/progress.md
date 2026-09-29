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

### 任务状态（五种之一）
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

### 测试命令与结果
- CPU：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rl_seq_adv.py tests/test_rl_adv_transforms.py` → 116 passed（本地加 grpo_knobs 一行修复；该修复不提交）。2a-shared.patch 应用后同样 116 passed。
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
