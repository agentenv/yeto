# Proposal

## Why

main 上有一批长期失败的测试，"本机全绿"无法作为"完成"的依据。同一批问题也让 CI 的 python 检查从 10-04 起一直失败（run 37887731542：33 个失败、13 个错误），gpu 作业因自托管机器离线一直排队。另外，有几组测试会在本机拉起 Ray，S17/S18 期间被误跑 3 次，每次都可能触发上卡链的线程守卫。

范围扩大说明：本 change 起初只修本机老失败测试与隔离 Ray 测试。10-09 主 agent 代用户拍板，把 CI 转绿并入本 change，理由是 CI 红和本机老失败是同一件事，分开做会重复。

## What Changes

- 逐个查清下列本机老失败测试的根因并修复，或确认是环境依赖后加带原因的跳过标记：
  - test_rl_codex_schema.py::test_codex_openenv_preflight_attests_pinned_adapter_and_environment
  - test_rl_ir_harness.py::test_ir1_append_roles_accepted_for_signed_codex_agent_when_equal_to_profile
  - test_rl_dense_full_parameter_sweep.py 的 test_two_real_clients_match_central_grpo_for_identical_frozen_batch 与 test_two_real_clients_resume_mid_sweep_without_double_accounting
  - test_rl_m1_dense_full_direct_launch.py::test_terminal_reconciler_requires_complete_cross_island_policy_and_accounting
  - test_rl_benchmark.py::test_resume_identity_survives_json_round_trip
  - test_verda_provider.py::test_down_uses_recorded_names_verbatim
  - 缺依赖的 test_rl_algorithm_provenance.py::test_export_records_algorithm_like_the_event（缺 accelerate）与 test_ports_megatron_pythonpath.py（缺 megatron.post_training）
- 修 CI：安装步骤补 pylatexenc 与 pytest-asyncio；处理浅克隆导致的 `git show fd37129e` 失败；处理 test_delta_protocol 在 CI 超时；查清 test_rl_fn_provider_view 与 test_rl_fn_boot_only 的 8 例断言失败；修 tool_wait_workload 缺 generate 等代码与测试不一致；gpu 作业改为可选或允许失败；复核 test_rl_integration 的 `--resume` 检查点问题。
- 新增 pytest 标记 `ray_local`，标出会在本机拉起 Ray 的测试，默认不收集，显式开启才跑。
- 给出"本机安全测试集"的一条命令，写进文档。
- 目标：CI 的 python 与 rust 检查转绿，本机安全测试集全绿，作为以后"完成"判断的依据。

## Capabilities

### New Capabilities
- `local-test-suite`: 本机安全测试集与 CI 检查的约定，包括 Ray 测试隔离、跳过标记要求、一条命令和全绿目标。

### Modified Capabilities

（无。）

## Impact

- 测试：tests/conftest.py、上面列出的测试文件、会拉起 Ray 的测试文件（test_rl_harness_mismatch_tape.py、test_delta_protocol.py、test_rl_algorithm_flags_upstream.py、test_rl_miles_adapter_config.py 等，最终名单在审计后确定）。
- CI：.github/workflows/ci.yml（安装步骤 32–33 行、checkout、gpu 作业 37 行起）。
- 代码：只在根因确实是代码缺陷时修改，例如 tool_wait_workload 与 cli `down` 的调用约定。
- 文档：新增或扩充测试说明。
- 不上 GPU，无云花费。
