# Tasks

## 1. 显式学习率调度

- [x] 1.1 在 `yeto/rl/engine/run_config.py` 的算法配置中新增学习率调度字段（衰减方式 linear/constant、衰减步数），由同步方式推导：strict-avg 为 linear、步数 `global_rounds × optimizer_steps`；decoupled 为 constant；仅评估的运行为空。对 strict-avg 断言 `rollout_batch_size × n_samples_per_prompt == global_batch_size × optimizer_steps`，不满足时启动前拒绝。可参考草稿 `/home/michael/work/followups/decoupled-constant-lr-ports.patch`。验证：单测覆盖三种同步方式的推导结果、非法值拒绝和断言失败。
- [x] 1.2 ports 翻译（`yeto/rl/engine/miles_adapter/config.py`）输出 `--lr-decay-style`、`--lr-decay-iters`、`--lr-warmup-iters 0`、`--min-lr 0`，并加入未映射项检查表。验证：单测断言 argv 中的四个参数；upstream Miles `parse_args` 接受翻译结果（沿用现有 parse_args 测试）。
  - 完成记录（2026-09-29）：已实现 + CPU 单测通过；upstream 验收在公开镜像 `MILES_NEXT_IMAGE`（radixark/miles@sha256:9094…，miles-next 0394715）内运行 `tests/test_rl_miles_adapter_config.py`，`test_upstream_parse_args_accepts_translation[True/False]` 均通过且未跳过（56 passed, 1 skipped——跳过的是仅在无 Miles 时运行的 cell 计数测试）。证据：`evidence/2026-09-29-gpu/parse-args-p.log`。
- [x] 1.3 legacy 翻译（`yeto/rl/learner.py` 的 `_legacy_miles_argv`）从同一字段输出相同的四个参数。验证：有意重新生成 `tests/test_rl_argv_snapshot.py` 的 golden，并新增断言：除这四个参数外，每个快照配置的 legacy argv 与修复前逐项一致；同一配置下 legacy 与 ports 的四个参数完全相同。
- [x] 1.4 用 Megatron 调度公式在 CPU 上复算学习率：decoupled 在超过 `global_rounds × optimizer_steps` 的前 1000 步都等于配置值；strict-avg 在 optimizer_steps 为 1 和 2 时逐步等于修复前的隐式调度，且最后一步大于 0。验证：对应单测通过，并保留"修复前 argv 在第 `global_rounds × optimizer_steps` 步降为 0"的回归见证测试。

## 2. 零学习率不变量

- [x] 2.1 ports：在 `state_plugin.py` 已有的 `train_one_step` 记录器中取优化器步实际应用的学习率，经 trainer 端口的步指标返回，由 `yeto/rl/engine/driver.py` 写入 `rl_local_round`（字段名与 legacy 一致），并在非最终轮学习率为 0 时让该轮失败、不提交。验证：用 fake engine 的单测覆盖非最终轮失败、strict 最后一轮不失败、decoupled 最终 cut 之后不失败，失败信息包含轮次与学习率。
- [x] 2.2 legacy：通过 fork 已有的 `custom_megatron_before_train_step_hook_path` 钩子（与 `yeto/rl/grad_audit.py` 共存，不得互相覆盖）取应用的学习率，在 legacy 的 policy-sync 桥接层写入 `rl_local_round` 并执行同一不变量。验证：CPU 单测覆盖与 2.1 相同的三种情况，以及与 grad audit 钩子同时启用时两者都生效。
- [x] 2.3 文档：`docs/MILES_RL.md` 说明两种同步方式的学习率调度、decoupled 从本版本起为常数学习率，以及零学习率不变量。验证：文档中的命令示例在 `--dry-run` 下可执行。

## 3. 集成验证

- [x] 3.1 CPU 全量回归：`python -m pytest -q tests/test_rl_*.py tests/test_provenance.py` 与修复前相比没有新增失败。验证：记录前后失败清单并逐条比对。
- [ ] 3.2 GPU：两岛 decoupled（配置同 head 模式 `yeto-hp929d`：`--total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2`）在 legacy 与 ports 上各跑一次。验证：两条路径每轮应用的学习率逐位一致且都等于配置值；最终 cut 之前 syncer 每个外层步的全局 delta 范数都不为 0；两岛最终 hash 一致。证据存入本 change 目录。
  - 进展记录（2026-09-29，未勾选）：已在 Modal 2×L40S 上用 `scripts/benchmark_rl.py` 以相同参数（`--global-rounds 4 --fragments 4 --pipeline 2 --local-horizon 2 --inner-lr 1e-5`）跑 legacy 与 ports 各一次，三条验证都成立：两条路径每轮 `applied_lrs` 逐位一致且均为 1e-05；syncer 8 个外层步 `global_delta_norm` 全非 0；两岛最终 hash 一致（legacy feebb2…，ports 150ea1…）。见 `evidence/2026-09-29-gpu/compare-dec.txt`。未勾选原因：benchmark 把 learner 预算固定为 `global_rounds`，每岛只跑 4 个本地轮，没有复现 head 模式 `yeto-hp929d` 的 run-until-stop（12 个本地轮、超过 `global_rounds × optimizer_steps`）这一真正的回归场景；需要 head 模式运行（Modal/Verda 目前不能当 head）才满足"配置同 head 模式"。
  - 补充（2026-09-29，仍未勾选）：`scripts/benchmark_rl.py` 新增 `--learner-budget-steps`（仅 decoupled），可让非 head 模式的本地轮数超过 `global_rounds`（学习率 horizon 仍为 `global_rounds × optimizer_steps`）。用 `--learner-budget-steps 8` 在 Modal 2×L40S 上补跑 legacy+ports 的尝试失败：ports sandbox 30 分钟内没能执行第一条命令（`nvidia-smi` 没有输出），legacy 运行未完成，两者都被 1900 s 费用上限 watchdog 终止，没有产出可用数据。见 `evidence/2026-09-29-gpu-budget8-aborted/`。这类补充运行即使成功也不满足"配置同 head 模式"。
- [x] 3.3 GPU：strict-avg 两岛 3 轮回归（配置同 `rl-engine-ports` 的 `2026-09-29-rerun-strict2`）。验证：每步应用的学习率与修复前证据逐位相同，最后一步大于 0，没有触发零学习率不变量。
  - 完成记录（2026-09-29）：GPU 验收通过（Modal 2×L40S，ports，yeto 20aba48，命令与 `rerun-strict2` 相同）。两岛 `applied_lrs` = [1e-4, 6.666666666666668e-05, 3.333333333333334e-05]，与修复前证据逐位相同；修复前证据没有 applied_lr 字段，按 Miles 在 scheduler.step 之后记录 `train/lr-pg_0` 推导（第 k 步应用值 = 第 k-1 步记录值，首步为配置值），且修复前后的 `train/lr-pg_0` 序列本身逐位相同。最后一步 3.33e-5 > 0，rc=0，未触发不变量。证据：`evidence/2026-09-29-gpu/compare-strict.txt`（脚本 `compare-strict.py`）、`evidence/2026-09-29-gpu/strict2-ports/`。

