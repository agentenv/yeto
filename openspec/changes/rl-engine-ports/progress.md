# R0 实施进度与证据（2026-09-29，未提交）

测试环境：`/tmp/yeto-venv`（CPU torch）。全量 `pytest tests/ --continue-on-collection-errors` 与 HEAD `c40a32c` 干净 worktree 对比：失败/错误集合完全相同（94 项，均为缺 cargo/syncer 二进制、miles 模块等环境原因），无新增失败。

## 已勾选（代码实现 + 任务自身验证已通过）
- 0.2 `migration-ledger.md`（#64/#65/#59/#66/#62/#63 已按 main `e21a7ff` 复核并关闭；#61/#63 经 #67/#68 进入 main）
- 1.1/1.2/1.5 `yeto/rl/__init__.py`（`MILES_NEXT_PINS` 等，commit/镜像仍为占位，见阻塞）、`verify_miles_revision(expected=)`、launcher/ssh_harness 远端脚本；`tests/test_rl_engine_pins.py`，legacy 脚本逐字节一致
- 2.1–2.5 `yeto/rl/engine/{ports,trainable_state,algorithm,capabilities,run_config}.py`；argv 快照 `tests/test_rl_argv_snapshot.py`；#66 合入后已删除测试内副本，直接使用 `yeto.rl.elastic_benchmark.capabilities.attestation_from_dict`
- 3.1/3.6 `miles_adapter/{config,placement}.py`；upstream `parse_args` 在 `/home/michael/work/miles-next-venv` 下通过
- 5.1/5.2 `yeto/rl/engine/selection.py`、CLI/launcher/learner/harness/export/benchmark 接线；`docs/MILES_RL.md` Engine selection，dry-run 命令已执行
- 6.1 `scripts/rl_engine_equivalence.py`（`--dry-run`、`--fake`；fake 的 legacy 侧为合成数据，仅证明脚本可用）

## 代码已实现、验收未通过（需 GPU / 授权）
- 3.2–3.5、4.1–4.4：CPU 单测与 fake/stub 组合根测试通过；spec 要求的 GPU 冒烟未跑。
- 1.3 SGLang：补丁文件 `/home/michael/work/sglang-patches/`，详见 `sglang-patch-port.md`。**偏差**：`c2cb40a` 只涉及 upstream 已删除的 API，未移植；`b34df47`/`95d4d69` 上游已有等价实现，仅移植测试。需用户确认。
- 1.4/1.4b Miles：`/home/michael/work/miles-next-run_plugin.patch`、`miles-next-5494a6ce.patch`；TITO 两提交 upstream 已等价（用 `qwen38small`）。**偏差**：`run_plugin(fn_path, kwargs)` 用显式 dict（upstream rpc 层禁止 *args/**kwargs），design D4 需同步。
- 审查修复：publisher hash 统一为 `policy_tensor_hash`、trainer `step_metrics`、ports 在 pin 缺 `run_plugin` 时启动前拒绝、strict local_step、成员集合同源、非有限 reward 拒绝、token 单一 helper；`state_plugin.py` 改用 upstream `_get_hf_bridge`（yeto 源码不得出现 trust_remote_code=True）。

## 阻塞（需用户动作）
1. git commit（yeto、miles fork、sglang fork）、创建 `michaellchung/sglang`、push、向 radixark/miles 提 run_plugin PR。之后把 `MILES_NEXT_COMMIT`/`SGLANG_NEXT_COMMIT` 固定到 fork 提交，并构建 `MILES_NEXT_IMAGE`。
2. #64/#65/#59/#66/#62 以及 #61/#63（经 #67/#68）已在 main `e21a7ff`，R0 已在 `/home/michael/work/r0-integ` 整合；仍需关闭 #43（0.1）。全量 pytest：main 68 failed/26 errors，整合后失败集合相同，无新增。
3. GPU 资源（付费）：3.2–3.5、4.x、1.4b 同机冒烟、6.2/6.3 等价性实验。
4. 7.x 依赖 6.x 通过；rl-infra-spec E0–E3 以 R0 验收为前置，未开工。

## 已知风险
- R0 等价结论只覆盖单轮任务：upstream Miles 以 `rollout_mask_sums`/`num_rollouts` 归一化 loss，单轮时与 legacy 等价，多轮/共享 `rollout_id` 时变为按 token 平均，未被实验覆盖（design Risks，后续项）。
- legacy fork 的 dashboard_columns 写入在两岛同机时偶发 rename 竞态（`evidence/2026-09-29-tf-diagnosis/tfl-a-failed-race`），R0 不修。
- 第 2 层梯度判定改为锚定 CPU fp32 参考（design D12 第二次口径修改，方案 A）；v3 报告：`evidence/2026-09-29-eq62-v3/`、`evidence/2026-09-29-eq63-v3/`。
- 驱动在 trainer 常驻时发布（upstream train.py 先 offload），共置可能 OOM。
- ports 仅上报 grad_norm，loss/lr/KL 为 None，等价性 loss 对比受限。
- upstream LoRA 导出名非 canonical PEFT（无 `base_model.model.` 前缀），state_plugin 哈希/PEFT 导出前需归一化——GPU 上核实。
- launcher/harness 未接 `MILES_NEXT_IMAGE`。

## GPU 验收（在已提交代码上复跑，2026-09-29）

代码 `923b3049a6dea345186adaa16da11d76693614b3`（`git archive` 上传，证据目录内有 `YETO_SHA`），Miles `0394715`、SGLang `9f29303`，Modal H100。证据：`evidence/2026-09-29-rerun-*`。

| 任务 | 结果 | 证据 |
|---|---|---|
| 1.4b | 通过：两岛同机无端口冲突（端口由 router 端口派生），TITO Qwen3.8 由 upstream `qwen38small` 覆盖（fork 测试 30 项通过） | `rerun-strict2`、`miles-port-notes.md` |
| 3.2 | 通过：yeto 侧只持有元数据；过期 token 在训练前被 `PolicyIdentityError` 拒绝 | `rerun-smoke-normal`、`rerun-smoke-badtoken` |
| 3.3 | 通过：receipt 字段齐全 | `rerun-smoke-normal` |
| 3.4 | 部分：导出→应用→再导出 hash 一致、reset 清零与 scheduler 对齐、PEFT 名归一化通过；`test_rl_grad_accumulator_hook`（#64）CPU 通过；**与 legacy 逐张量比对未做**（由 6.2 teacher forcing 覆盖后再判） | `rerun-smoke-normal/probe.jsonl` |
| 3.5 | 通过：checksum 与清单一致；单 engine 失败返回错误、无清单 | `rerun-smoke-normal`、`rerun-smoke-pubfail` |
| 4.1 | 通过：事件顺序符合 spec；零梯度注入该轮失败、不提交 | `rerun-smoke-normal`、`rerun-smoke-zerograd` |
| 4.2 | 通过：两岛 strict-avg 3 轮 hash 一致 | `rerun-strict2` |
| 4.3 | 通过：两岛 decoupled 到最终 cut，最终 hash 一致，PEFT 可被标准 PEFT 加载（Nebius 首跑） | `rerun-decoupled`、`decoupled2/verify_peft.log` |
| 4.4 | 部分：第 2 轮 kill 后以 reset 应用权威 cut 并跑完；**与 legacy 的对照未做** | `rerun-kill44` |

## 已知问题（R0 不修，后续单独 PR）：decoupled 下学习率衰减到 0

- 现象：head 模式两岛 decoupled（`2cbd45c`，global_rounds=4、local horizon 2）中 island 的 `train/lr-pg_0` 为 7.5e-6→5e-6→2.5e-6→0，此后 grad_norm 非零但 delta 为 0，syncer 从第 7 步起 global_delta_norm=0。证据：`/home/michael/work/gpu-head/evidence/2026-09-29-head-ports/attempt4-2cbd45c/`。
- 根因：Miles 以 `train_iters = num_rollout × rollout_batch × n_samples / global_batch` 推出 `lr_decay_iters`，默认 linear 衰减到 0；yeto 传 `--num-rollout = global_rounds`，而 decoupled 为 run-until-stop，本地步数没有预知上限。legacy（agentenv fork `model.py:85-105`，`learner.py:1000`）与 ports 完全相同；eq63 中两边 lr 序列逐位一致。benchmark 未暴露是因为本地步数恰好等于 global_rounds。
- 决定（用户，2026-09-29）：R0 只换底座不改行为，ports 与 legacy 保持相同行为，6.3 等价结论有效。之后单独提 agentenv/yeto PR，在 legacy 与 ports 两条路径上同时把 decoupled 改为显式常数 lr（strict 显式写出原线性调度，数值不变），并重跑 decoupled。ports 侧改动草稿：`/home/michael/work/followups/decoupled-constant-lr-ports.patch`。
- head 模式验收（基础设施层面）：默认公开镜像、fork checkout、`rl_engine_selected=ports`、两岛到达最终 cut 且 hash 一致均通过；训练后半程无更新属于上述已知问题，与 legacy 行为一致。
- 任务 0.3 完成（2026-09-29）：三个分支经 `git ls-remote` 与 compare API 核对，基底 commit（yeto `e21a7ff`、Miles `9e4260d`、SGLang sglang-miles `571212b`）与当前 HEAD 记入 design D6/D11。
