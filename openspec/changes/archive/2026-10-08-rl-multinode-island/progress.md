# rl-multinode-island progress

- 分支 `infra-multinode`（worktree /home/michael/work/infra-multinode，基 integ-decl 8347351），未推送。
- 状态：§0 0.1 完成、0.2 未完成（需镜像内 Miles 的 Flash-Next recipe）；§1 1.1–1.9 CPU 通过；§2 2.1–2.6 CPU 通过（本地双 Ray 节点演练，2026-10-03）；§3 GPU 未开始（需另批，预算 ≤$60，判据见 tasks §3）；§4 4.1 完成、4.2 待 §3 后收尾。
- 测试命令：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/test_rl_multinode_*.py tests/test_rl_launcher_multinode.py`（43 passed）；全量 rl 子集失败 id 27 个 ⊂ 基线 94（`infra-drafts/s1-pytest.log`）。
- 演练命令：`PY=/tmp/review-miles-venv/bin/python tests/multinode_sim/run_sim.sh`（证据 `infra-drafts/s1-sim/summary.txt`：4/4 PASS；线程回落；`ray stop --force`）。
- 云资源/费用：无（全部 CPU）。
- 待批准：Q1–Q6 用户复核；§3 GPU 验证。
- 下一步：用户复核 Q1–Q6 → 0.2 → §3（主 agent 批准后）。

## 收尾（2026-10-04，S7 T4）
- 分支 `infra-multinode`，HEAD = 本提交（前序 14a7ac01 → a43a5a74 G3 修复 → 本提交 G4 修复 + 判读）；未推送；合入 integ-decl（merge --no-ff）由主 agent 完成。
- §3 GPU 结果：3.0 G0 PASS、3.1 G1 PASS（s1-mn-20261004b）、3.2 G2 PARTIAL（E1 边 2×1 不可验，合法边界）、3.3 G3 PASS（s1-mn-20261004d，node_lost 15 s，重启被预检拒绝）、3.4 G4 FAIL→修复待复验（失败路径 teardown 逐节点确认行缺失；云端已确认 0 实例、cleanup clean twice）、3.5 未触发。
- 测试命令：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/test_rl_multinode_*.py tests/test_rl_launcher_multinode.py tests/test_rl_engine_driver.py tests/test_controller.py`；全量 68 failed + 26 errors = 94 = 基线（/tmp/base.sorted）。
- 证据：/home/michael/work/s1-runs/s1-mn-20261004{b-g12,c-g3,d-g3}/{judgment-*.json,pulled/,launch.ts.log}；记录 infra-drafts/T4-S7-PROGRESS.md §7–§9、s1-gpu.md。
- 费用：T4 累计 ≈$20.8（G0 0.75 + a–g 11.85 + 复验 a 1.6 + b 1.7 + c 3.2 + d 1.7）；台账 infra-drafts/gpu-spend.md。
- 待批准：Q1–Q6 用户复核；G2 E1 边 PARTIAL 是否接受为合法边界；G4 是否再上卡复验；多节点 PP/EP（Q1/Q3）放开。
