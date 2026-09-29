# rl-infra-spec 进展

状态分三级：**已实现**（代码存在）/ **CPU 测试通过** / **GPU 验收通过**。只有验收原文满足才勾选 tasks.md。

## 2026-09-29（Agent I，分支 `rl-infra-spec`，基于 rl-engine-ports d355e06）

| 任务 | 状态 | 证据 |
|---|---|---|
| 1.1 runtime manifest | 未开始：需要镜像环境（见 GPU 计划 F0） | — |
| 1.2 兼容 baseline | 未开始：需要 GPU（A1） | — |
| 1.3 云实验池计划 | 草案已交付，待用户确认，未勾选 | `gpu-plan.md`（本目录，源自 infra-drafts）（2026-09-29 修订：默认 Nebius/Modal，合计≈$3,052） |
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
- `yeto/rl/engine/ports.py` 的预留注释更新见 tasks 3.4a（已写入 tasks，代码未改）。

### Miles fork（本地，未推送）

分支 `yeto-elastic-m1-m6`（worktree `/home/michael/work/miles-elastic`，基于 03947150）。提交：M1 3ae99fd1、M2 4c96cf45、M3 b14392b3、M4 f91020e8、M5 ed9282e6、M6 b2837089；审查修复 738136c2（M3）、9ba38f6d（M2/M4）、10b52a9e（M5）、965ef314（M6）。仅 CPU 单测；已知缺口：M4 无 payload checksum ACK（需 yeto Publisher 读回校验），M5 DistOpt DP gather 未实现、CUDA/Megatron RNG 未在 GPU 验证；real_ray 测试在本机 ray.init 卡住（基线同样），未运行。tasks.md 的 M1–M6 任务（2.1a/3.3a/3.3b/3.4a/3.5a/4.2a/4.6a）已于 `15e864d` 写入（底稿 `/home/michael/work/infra-drafts/tasks-m1-m6.patch`），并按审查在 F1–F3 修订为 fork 的实际语义。

### 测试

`/tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors`：基线为 68 failed / 26 errors，均为已知环境性失败。改动后按测试 id 去重对比，见提交说明。

## 2026-09-29（Agent ALIGN，阶段 0 对齐）

- 对齐文档：[`alignment.md`](alignment.md)，包含矩阵、修订清单（每条一个提交）、D1/D2 划分、M1–M6 映射、F 延后说明、工作包与待批准事项。
- 本轮写入 tasks.md 的内容：M1–M6 fork 任务（2.1a/3.3a/3.3b/3.4a/3.5a/4.2a/4.6a）、A1–A5 的接口与验收补充、A9 依赖、A10 阶段边界。没有勾选任何任务，没有改代码，没有使用 GPU 或云资源。
- lr-fix 实际 9/10（3.2 未完成），fix-verda-provider 实际 11/19（4.x 等 PR #69，6.x 未做）；如实记录，未重复实现。
- `openspec validate --strict`：rl-infra-spec、rl-algorithm-capabilities、rl-algo-grpo-knobs、rl-algo-loss-variants、rl-algo-mismatch-correction、rl-algo-seq-and-adv 全部 valid。
- 待批准：见 alignment.md §8。下一步：主 agent 按 alignment.md §7 派发工作包。

## 2026-09-29（Agent ALIGN，审查修订 F1–F9）

- 按独立审查逐条修订，每条一个提交，SHA 见 alignment.md §3（F1–F9、GP、DEC）；只改了 openspec/，没有改代码。
- GPU 计划纳入本目录的 `gpu-plan.md`，其中 E3 的 gather 阶段改名为 DEV-GATHER。
- 主 agent 的决定见 alignment.md §7b。infra 代码工作的基底为 `rl-integ`（merge c5e05f4）。
- 六个 change 的 `openspec validate --strict` 在推送前重新运行，全部 valid。

## 2026-09-29（Agent INFRA，阶段 A + E0 的 CPU 部分，分支 `infra-a`）

- 分支与 worktree：`infra-a`，位于 `/home/michael/work/infra-a`，基于 `origin/rl-integ` a50e9d2。所有提交均已普通 push 到 `origin/infra-a`，没有未提交改动（本条目所在提交之后）。
- 提交：bf47641（1.4/1.5/1.6 A1/A4）、115ee9a（driver profiles/观测，1.4/1.7/2.2）、e4d227a（2.1/2.1a 分区与 M1 map、入口 preflight）、47e7d7e（1.1 manifest 工具、1.3 计划定稿、1.8 假设）、本条目所在的文档提交。
- 没有使用任何 GPU 或云资源，费用为 $0，没有残留。

### 状态（五选一）

| task | 状态 | 证据 |
|---|---|---|
| 1.1 | 未完成（manifest 工具 CPU 通过，待镜像） | `yeto/rl/engine/runtime_manifest.py`，`tests/test_rl_runtime_manifest.py` |
| 1.2 | 未完成（待镜像/GPU A1） | — |
| 1.3 | 已实现（计划定稿；依赖 1.1，未勾选） | `gpu-plan.md` §8 |
| 1.4 | CPU 通过（依赖 1.2，未勾选；AlgorithmSpec v2 接口待对齐） | `tests/test_rl_execution_profile.py`、`tests/test_rl_driver_profiles.py` |
| 1.5 | CPU 通过（未勾选） | `tests/test_rl_pause_audit.py` |
| 1.6 | CPU 通过（未勾选） | `tests/test_rl_elastic_config_schema.py` |
| 1.7 | 未完成（driver 事件与"关闭观测兼容"CPU 通过；rollout 计数待 M3 镜像） | `tests/test_rl_driver_profiles.py` |
| 1.8 | 已实现（依赖 1.4/1.7，未勾选） | `dynaresize-hypotheses.md` |
| 2.1 | CPU 通过（GPU 未做） | `tests/test_rl_miles_adapter_config.py`、`tests/test_rl_miles_adapter_placement.py` |
| 2.1a | CPU 通过（fork M1 单测与 yeto 侧校验；GPU 未做） | miles-elastic `tests/fast/ray/test_placement_map.py` |
| 2.2 | CPU 通过（GPU A2 未做） | `tests/test_rl_driver_profiles.py` |
| 2.3 | 未完成（CPU 重叠分析已做；X9 待 GPU） | tasks.md 2.3 进展 |
| 2.4 | 未完成 | — |

### 测试

- 命令：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors`。结果：68 failed / 2081 passed / 41 skipped / 26 errors（基线 `/tmp/integ-full.txt`：68 failed / 2062 passed / 26 errors）。
- 失败集合按测试 id 去重后与基线**完全相同**（均为 94 个 id，`diff` 为空）。新增的 19 个测试全部通过。输出见 `/tmp/infra-a-full.txt`。
- fork M1：`/tmp/review-miles-venv/bin/python -m pytest -q tests/fast/ray/test_placement_map.py tests/fast/ray/test_placement_group.py`，61 passed。

### 阻塞与解除条件

- 1.1/1.2/2.1–2.4 的 GPU 部分：需要 Agent IMG 提供含 fork M1（`yeto-elastic-m1-m6` 3ae99fd1 及其后的审查修复）的 `MILES_NEXT_COMMIT` 与镜像 digest。拿到 digest 后按 gpu-plan §8 顺序执行：F0（1.1 manifest）→ A1（1.2）→ F2 → A2（2.1/2.1a/2.2/2.3 X9）→ A3（2.4）。
- 1.7 的 queued/active/tool-wait 计数：依赖 M3 进入镜像。
- CLI：`--rl-placement`/`--rl-standby-gpus` 需要加到 `yeto/rl/learner.py`（以及 launcher、harness 的透传），不在 INFRA 的写入范围内。
- launcher 的 Modal `H100` 映射没有 `!`（`modal_runner.py:59-60`）：逐位实验在修复前改用独立的 Modal 脚本。

### 待批准

无新增。1.8 原先的阻塞"缺原文"已解除（见 alignment.md 追加条目）。

### 下一步（可直接执行）

1. 主 agent 合入 ALGO-CAP 的 `_check_gradient` 补丁后，在其上继续开发；ALGO-CAP 的 v2 `ExecutionSpec` 冻结后，复核 `execution_profile.algorithm_max_policy_staleness`。
2. 镜像 digest 到位后，在镜像内运行 `python -m yeto.rl.engine.runtime_manifest --image <digest> --out manifest.json --capability ports-partitioned-serial --capability fixed-partition-standby`。
3. 先写 A1/A2 的实验计划（容差、seed、硬超时、前缀 `infra-a-`）并提交，然后再租用。

### 审查修订 F1–F7（2026-09-29 INFRA）

- F1：1.7 的兼容性证据改为对照 a50e9d2 录制磁带（`tests/data/r0_driver_tape_a50e9d2.json`，在 a50e9d2 的临时 worktree 中以 `PYTHONPATH` 指向该 worktree 录制）。生产路径 profile 存在、observe=False 时 `rl_driver_start` 不再多出字段，事件逐条相同。
- F2：A1 比对改为外部哈希（launcher 下发的 `--rl-expected-algorithm-sha256`/环境变量）对运行时 AlgorithmSpec，在 `connect_island_ray` 前 fail-closed；partitioned 缺外部哈希即拒绝。colocated（R0）缺失时退回运行时哈希并记录来源，这是有意保留的 R0 兼容，不算独立验证。learner 需把 `--rl-expected-algorithm-sha256` 写入 `miles_args.yeto_rl_expected_algorithm_sha256`（ALGO-CAP/主 agent）。
- F3：colocated-serial 不执行 readiness 门，行为与 R0 相同；partitioned 模式的训练门不再按 rollout_batch_size 计组数（部分 rollout/过滤后组数不足时照常训练）。测试：`test_fewer_groups_than_rollout_batch_size_still_trains`。
- F4：readiness 放行与 R0 发布清单、每组 token 检查基本重复，不作为独立证据；它是 overlap 模式将来需要的挂点。
- F5：`rl/filtered_groups` 取 hook 元数据 `filtered`，`rl/aborted_groups` 单列，`rl/carried_over_groups` 未跟踪记 None（3.6/4.1）。
- F6：拒绝 offload_train 注明为本方保守决定，非上游约束（上游 train.py:139-151 非 colocate 也会 offload）。
- F7：full 模式或未请求 fixed-partition 时给 `--rl-standby-gpus` 显式报错，有测试。
- selection：`ports_rejections(placement=...)` 在显式 `fixed-partition` 且 `rollout_num_gpus≥1` 时放行；learner/launcher/harness 的调用需传 `placement=args.rl_placement`（不在 INFRA 写入范围）。
- 1.8 措辞按审查修正；2.3 将来若交否定结论须补流式 reward/多 minibatch 流水的细粒度论证。
