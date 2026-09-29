# rl-infra-spec 进展

状态分三级：**已实现**（代码存在）/ **CPU 测试通过** / **GPU 验收通过**。只有验收原文满足才勾选 tasks.md。

## 2026-09-29（Agent I，分支 `rl-infra-spec`，基于 rl-engine-ports d355e06）

| 任务 | 状态 | 证据 |
|---|---|---|
| 1.1 runtime manifest | 未开始：需要镜像环境（见 GPU 计划 F0） | — |
| 1.2 兼容 baseline | 未开始：需要 GPU（A1） | — |
| 1.3 云实验池计划 | 草案已交付，待用户确认，未勾选 | `/home/michael/work/infra-drafts/gpu-plan.md`（2026-09-29 修订：默认 Nebius/Modal，合计≈$3,052） |
| 1.4 ExecutionProfile/readiness | 已实现 + CPU 通过，依赖未满足未勾选（依赖1.2；X9为GPU实验） | `yeto/rl/engine/execution_profile.py`，`tests/test_rl_execution_profile.py` |
| 1.5 暂停审计 | 已实现 + CPU 通过，依赖未满足未勾选（1.4） | `pause-audit.md`，`yeto/rl/engine/pause_audit.py`，`tests/test_rl_pause_audit.py` |
| 1.6 配置/边 schema | 已实现 + CPU 通过，依赖未满足未勾选（1.3–1.5） | `yeto/rl/elastic_benchmark/capabilities.py`、`plan.py`，`tests/test_rl_elastic_config_schema.py` |
| 1.7 时间线观测 | 纯计量已实现 + CPU 通过；事件发射**阻塞**（driver.py / miles_adapter 冻结） | `yeto/rl/engine/timeline.py`，`tests/test_rl_timeline.py` |
| 1.8 DynaResize 假设 | **阻塞**：本地没有原文 | — |

### 阻塞项需要的改动（等 lr-fix 合入后才能做）

- `yeto/rl/engine/driver.py`
  - 每个阶段（generate/reward/train/outer_sync/publish/offload/onload）发射 `timeline.Span`，带 profile hash 和 epoch；
  - 在安全点构造 `ReadinessSnapshot`；
  - 由 `ExecutionProfile.execution_mode` 选择 serial/partitioned 分支（2.2）；
  - 暂停前调用 `pause_audit.pause_decision`。
- `yeto/rl/engine/bridges.py`：暴露 outer phase，取值为 `round-boundary-published` / `in-boundary` / `stop-round` / `finalizing` / `budget-consolidation`，以及是否处于 learner-budget 模式。
- `yeto/rl/engine/miles_adapter/rollout.py`：提供 queued/active/tool-wait 计数，数据来自 fork-M3 的 in-flight 计数。
- `yeto/rl/engine/ports.py` 的预留注释更新见 tasks 补充草案 3.4a（待确认）。

### Miles fork（本地，未推送）

分支 `yeto-elastic-m1-m6`（worktree `/home/michael/work/miles-elastic`，基于 03947150）。提交：M1 3ae99fd1、M2 4c96cf45、M3 b14392b3、M4 f91020e8、M5 ed9282e6、M6 b2837089；审查修复 738136c2（M3）、9ba38f6d（M2/M4）、10b52a9e（M5）、965ef314（M6）。仅 CPU 单测；已知缺口：M4 无 payload checksum ACK（需 yeto Publisher 读回校验），M5 DistOpt DP gather 未实现、CUDA/Megatron RNG 未在 GPU 验证；real_ray 测试在本机 ray.init 卡住（基线同样），未运行。tasks.md 的 M1–M6 补充草案在 `/home/michael/work/infra-drafts/tasks-m1-m6.patch`，**待用户确认，未写入**。

### 测试

`/tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors`：基线为 68 failed / 26 errors，均为已知环境性失败。改动后按测试 id 去重对比，见提交说明。

## 2026-09-29（Agent ALIGN，阶段 0 对齐）

- 对齐文档：[`alignment.md`](alignment.md)，包含矩阵、修订清单（每条一个提交）、D1/D2 划分、M1–M6 映射、F 延后说明、工作包与待批准事项。
- 本轮写入 tasks.md 的内容：M1–M6 fork 任务（2.1a/3.3a/3.3b/3.4a/3.5a/4.2a/4.6a）、A1–A5 的接口与验收补充、A9 依赖、A10 阶段边界。没有勾选任何任务，没有改代码，没有使用 GPU 或云资源。
- lr-fix 实际 9/10（3.2 未完成），fix-verda-provider 实际 11/19（4.x 等 PR #69，6.x 未做）；如实记录，未重复实现。
