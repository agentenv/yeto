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
- 驱动在 trainer 常驻时发布（upstream train.py 先 offload），共置可能 OOM。
- ports 仅上报 grad_norm，loss/lr/KL 为 None，等价性 loss 对比受限。
- upstream LoRA 导出名非 canonical PEFT（无 `base_model.model.` 前缀），state_plugin 哈希/PEFT 导出前需归一化——GPU 上核实。
- launcher/harness 未接 `MILES_NEXT_IMAGE`。
