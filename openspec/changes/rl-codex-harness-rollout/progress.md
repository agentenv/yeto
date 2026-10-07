# rl-codex-harness-rollout 进度

- 2026-10-07 S14：A17（fork Ray 测试 test_cell.py + test_group.py）在 Modal CPU 运行 `s1-runs/s14-a17-cpu-20261007a/a17-pytest.txt`：120 passed, 1 failed（`test_group.py::TestUpdateWeightsExternalFailure::test_an_engine_side_failure_keeps_the_cell_and_is_not_retried`，1493 s）；GPU 容器内的 A17 因 grpo 对照臂崩溃被截断（无结论）；用户已决定不追查。
