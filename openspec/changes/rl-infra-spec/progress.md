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

### 合并与 GPU（2026-09-29 INFRA，续）

- 已合入：origin/algo-cap ebd436b（entry.py 冲突时保留 `with_partitioned_serial` 与 `unverified_mechanisms`；preflight/driver 调用 `caps.check(..., max_policy_age=profile.max_policy_age)`）；`p0-driver.patch` 干净应用；origin/rl-integ（镜像 pin 5da40a07）。全量测试：68 failed / 2246 passed / 26 errors，失败 id 集合与基线相同（94）。
- **1.1 GPU 验收通过（已勾选）**：`evidence/infra-a/1.1/`。云资源：Modal app `infra-a-manifest`，sandbox sb-8avihvjXJ5HpcSMbpNcqxQ（attempt1，91 s）与 attempt2（同 app 名，新 app ap-lEMebdd69SIlDhCPtW8YB2），各 1×L40S，合计约 3 分钟，费用约 $0.10。两个 app 均为 stopped，见 `modal_app_list_after_stop.txt`，无残留卷。
- 1b hook 补丁（`1b-hook.patch`）：审阅结论是与 A2/F5 一致。样本级 `filtered_samples` 属于终态 filtered；组级 `filtered` 计数不变；carried_over 未涉及，仍由 3.6/4.1 负责。**选择等 algo-1b 并入 rl-integ 后再合入**：该补丁在 hook 中硬导入 `yeto.rl.algos.sample_filters`，本分支没有这个模块，现在合入会让所有 ports 运行的 hook 失败；如果改成惰性导入并在缺失时静默 no-op，又会掩盖"配置了过滤器却没有生效"的错误。
- 1.2：尚未启动。需要 strict-avg 外层同步的 head（按 memory，Modal 不能承载 head），计划为 Nebius 4×H100 island，head 放在本机或 Nebius CPU VM。

### 1.2 与后续（2026-09-29 INFRA，续 2）

- **1.2 GPU 验收通过（已勾选）**：计划单独提交于 65b56ca，之后才启动。Modal 2 个 island，各 1×`H100!`（`--modal-gpu-exact` 首次真 H100 验证：两个 island 断言均得到 NVIDIA H100 80GB HBM3），本机作为 head。app `yeto-infra-a-b12`（ap-b43AWL3WDVThSe6RHSUEEp），16:48–17:03 UTC，约 $2，已 stopped、0 tasks；watchdog 与 puller 已结束。证据：`evidence/infra-a/1.2/`，提交 810b03b。
- 已合入：p0-driver-2.patch（7d5457b，基于 algo-cap 09607d6）；1b/2a 所需通道（d235e0b）；origin/algo-cap 2e40652（a66219c）。每次合入后全量失败集合按 id 与基线相同。
- **2.2/2.3/2.4 GPU 阻塞**：`yeto launch` 在 fixed-partition 下仍把整节点 GPU 作为 trainer（`--actor-num-gpus-per-node {spec.gpus_per_node}`），且不透传 `--rollout-num-gpus`。补丁 `/home/michael/work/infra-drafts/infra-launcher-partition.patch`（launcher.py 与新增测试，对 infra-a HEAD `git apply --check` 通过）需要 launcher 负责人合入，或由主 agent 授权 INFRA 合入。合入后按顺序执行：先单独提交 2.2/2.3 计划（T2R2 或 T4R4，Modal `H100!`，X9 用发布延迟注入），再提交 2.4 计划。

### 依赖复核（2026-09-29 INFRA，续 3）
- 1.3：依赖 1.1 已满足，计划满足验收原文 → **勾选**。
- 1.4：依赖 1.2 已满足，但验收是 X9（design 表：在固定分区对严格 policy 依赖注入延迟，是 GPU 实验），要等 2.3 的 GPU 运行 → 仍不勾选。
- 1.5 依赖 1.4，1.6 依赖 1.3–1.5，1.8 依赖 1.4/1.7 → 随 1.4 一起不勾选。
- launcher partition 补丁已按授权合入 infra-a（launcher.py 中 fixed-partition 的 GPU 划分部分归 INFRA）。

### E0 GPU（2026-09-29 INFRA，续 4）
- 2.2：第四轮满足预登记条件 1–4；第三轮按预登记判为不通过（A 第 3 轮 applied_lrs 缺证据），两轮都如实记录在 tasks 2.2。因依赖未满足，不勾选。
- 2.3：X9 guard（第三轮 C）通过；2.3 整体未完成。
- 云资源（全部为 Modal，均 stopped、0 tasks，列表见 `evidence/infra-a/2.2-2.3/modal_apps_round*.txt`）：
  - A-attempt2 约 15 分钟 × 1 卡；
  - 第二轮 A 约 15 分钟 × 1 卡；
  - 第三轮 A 12.5 分钟 × 1 卡、B 12.5 分钟 × 2 卡、C 14.5 分钟 × 2 卡；
  - 第四轮 A 13.5 分钟 × 1 卡、B 18.5 分钟 × 2 卡；
  - 合计约 2.5 H100·h，约 $10。
- 所有 watchdog 均已结束（`sleep 3900` 中没有 infra-a 残留）。
- 2.4 计划已单独提交（3b26fde），尚未启动。

### 2.4 与剩余事项（2026-09-29 INFRA，续 5）
- 2.4 阻塞：trainer DP>1 在 ports 路径上失败（DistOpt 分片主参数，详见 tasks 2.4）。sweep 已停止；app ap-4UQFe8lS3YdXdC9sISEu32 已 stopped、0 tasks；本机 syncer 与 watchdog 已清理。
- echo 统一补丁（echo-writers-infra.patch）依赖 `event_echo.append_record`，该函数目前只在 origin/integ-decl 与 algo-1a 上，origin/algo-cap 4373cd9 尚没有。已先 merge algo-cap（af83975），补丁等 append_record 进入 algo-cap 后再合入。
- 未开始（依赖 elastic 镜像 9f0977）：2.1a GPU 验收、1.7 M3 in-flight 计数（router `/worker_inflight`）、第二批接线（restore_membership_state、admit_cordoned→check_weights→admit_cells、commit_weight_version）。

### F 阶段（2026-09-29 INFRA）
- 已合入 origin/infra-f c6e233d（f-design.md，design D12 追加引用）。7.1/7.2 写完成记录但**不勾选**（依赖未满足）。
- alignment §12 追加待批准：G4、G6、G11。

### 2.4 与 2.1a（2026-09-30 INFRA）
- 2.4：扫描完成，结论为合法否定结论"尚无净收益边"，同 profile 下最佳固定配置为 T1R3，见 `evidence/infra-a/2.4/RESULT.md`；因依赖 2.2、1.7 未勾选，本项不勾选。本轮扫描 6 次有效运行加 1 次基础设施失败，约 10.5 H100·h，约 $41。
- 2.1a：attempt1 以显式映射启动并记录了 UUID，但"standby 上无进程"缺少直接证据，未通过预登记条件。已补充采样手段，重跑 attempt2 已排队。

## 交接（2026-09-30，Agent INFRA 会话结束；GPU 验证按用户决定暂停）

### 分支与状态
- 分支 `infra-a`，worktree `/home/michael/work/infra-a`，已推送；integ-decl 已快进到同一 SHA（见本条目所在提交）。本条目提交后没有未提交改动。
- 运行中的云资源：无。`modal app list` 中 infra-a 前缀的 app 全部为 stopped、0 tasks（`evidence/infra-a/modal_app_list_final_20260930.txt`）。本机没有残留的 arm/run 脚本、watchdog 或 syncer 进程。
- 实验脚本在 `/home/michael/work/infra-a-gpu/`（arm*.sh、stop_arm.sh、run*.sh），不在仓库中。以后在自有集群上验证时可参考这些脚本：计划先单独提交；每次运行配独立 watchdog；launcher 退出码 4/6 判失败；需加 `--rl-stall-timeout`。

### task 状态（五选一）
- 已勾选：1.1、1.2、1.3（GPU/计划验收），2.1a（GPU 验收）。
- GPU 证据齐备、依赖未满足未勾选：2.1（依赖 1.6）、2.2（依赖 2.1、1.4）。
- 合法否定结论（依赖 2.2、1.7 未勾选，未勾选）：2.4"尚无净收益边"（`evidence/infra-a/2.4/RESULT.md`，4 卡池最佳固定配置为 T1R3）。
- CPU 通过、未勾选：1.4（验收 X9 需 2.3 GPU）、1.5、1.6（依赖链）。
- 已实现、未勾选：1.8（依赖 1.4/1.7），7.1/7.2（设计文档，依赖 1.5、3.8）。
- 未完成：1.7（CPU 已完成 tool_wait、router in-flight 探针与观测采样线程；GPU 验证暂停），2.3（X9 guard GPU 已通过；age 0 下合法重叠 train‖eval 等未实现）。
- E1–E3（3.x、4.x）尚未开工；按用户决定拆分给 INFRA-E1（3.x）、INFRA-E2（4.1–4.5）；INFRA-A 继续 2.3/1.4/1.7/5.1；DEV-GATHER 延后。

### 关键代码（infra-a）
- `yeto/rl/engine/execution_profile.py`：ExecutionProfile 绑定 AlgorithmSpec 哈希，`check_algorithm_contract`。
- `yeto/rl/engine/driver.py`：profile/partitioned-serial；readiness（只在分区模式启用）；观测事件；`rl_round_trained`（trained groups/samples、sample-id 哈希、applied_lrs、masked/clip fraction、mismatch 与 A5 标签、nonzero_advantages、filtered_samples、tool_wait、submitted/aborted_in_flight、dynamic_filter_source）；测试用的发布延迟注入。
- `miles_adapter/entry.py`：preflight 用外部期望哈希；`with_partitioned_serial`；`receipt_role_family`。
- `miles_adapter/state_plugin.py`：DistOpt 分片主参数的 `full_masters`/`write_masters`，每步 loss 记录。
- `miles_adapter/trainer.py`：按单 cell 的 worker 数校验输出数；gspo/corrections 时拉取 step losses。
- `miles_adapter/rollout_meta_hook.py`/`rollout.py`：round metadata 走独立 sink 记录；轮次号取自 policy token；submitted_groups；router 探针。
- `yeto/protocol.py`：TCP keepalive。
- `yeto/launcher.py`：fixed-partition 的 GPU 划分与 `--rl-rollout-gpus`（这一部分归 INFRA）。
- `yeto/rl/engine/runtime_manifest.py`：1.1 使用的 manifest 工具。

### 下一步（可直接执行，均为 CPU）
1. INFRA-A：2.3 实现 age 0 下的合法重叠（eval 与 train/outer_sync 重叠），加 guard 与 CPU 测试；GPU 的 X9 overlap 实验等自有集群。
2. INFRA-A：1.7 剩余 GPU 验证（M3 in-flight 计数，带工具等待的负载）等自有集群；CPU 侧已完成。
3. INFRA-E1：3.4a 更新 ports.py 预留动词；3.1/3.2 控制器与 journal；第二批 fork 接线（restore_membership_state 重连回写、admit_cordoned→check_weights→admit_cells(expected_weight_version)、commit_weight_version）。fork 为 miles d002615f（yeto/ports），镜像 9f0977 已包含 M1–M6。
4. INFRA-E2：4.1 cut 状态审计（含 A3 算法状态与 Miles 超采样余量回收行为；已知：partial_rollout 关闭时在飞中被中止的组被丢弃，不回收）。
5. 已知限制：TP/PP 集体导出与 DistOpt 分片主参数的组合仍拒绝；precision-aware optimizer 拒绝。
6. 测试基线：`/tmp/integ-full.txt` 中的 94 个失败 id（环境性）；每次合入按 id 对比。

## 2026-09-30（Agent INFRA-E1，3.x CPU 部分，分支 `infra-e1`）

### 分支与状态
- 分支 `infra-e1`，worktree `/home/michael/work/infra-e1`，基于 integ-decl ef2d6b0；提交 3a8b8f5、6145c06、04e0b93、16ea216、f4ec49d 及本条目所在文档提交；已普通推送 `origin infra-e1`。无云资源、无费用；未启动任何 GPU。
- 本机 fork 参照：/home/michael/work/miles-elastic `origin/yeto/ports` = 0af62f4d（只读）。

### task 状态（五选一）
- 3.4a：CPU 通过，已勾选（纯 Y 任务，验收不需要 GPU）。
- 3.1、3.2、3.6：已实现 + CPU 通过，未勾选（依赖 2.2/1.5/1.6/3.5 未勾选）。
- 3.3、3.4、3.5、3.7：已实现 + CPU 通过，未勾选（验收需 GPU；计划 `evidence/infra-e1/plan.md` 已事先固定判据）。
- 3.3a/3.3b/3.5a：fork 任务，yeto 侧接线已完成，勾选归 fork/GPU 验收。
- 3.8：未完成（依赖 3.7、2.4；两小岛 strict 属后续）。

### 关键代码
- `yeto/rl/engine/journal.py`：fsync WAL + `epochs.json` CAS（单写者 flock）。
- `yeto/rl/engine/controller.py`：`IslandController`（D4 状态机、plan/request/status/cancel/inspect、fork epoch 对账、fence+drain、REBUILD_OLD、watchdog、CommandInbox/CLI）。只接受 attestation 认证的 `rollout-only` 边与 partitioned profile，pause 许可取自 `pause_audit.pause_decision`。
- `yeto/rl/engine/ledger.py`：group/batch/update 账本。
- `yeto/rl/engine/driver.py`：`controller=`/`ledger=` 可选参数；不传时事件磁带不变（R0 磁带测试通过）。
- `yeto/rl/engine/ports.py`：E1 可选协议；`capabilities.py` `RESERVED_PORT_VERBS` 加两项。
- `miles_adapter/rollout.py`（成员动词）、`publish.py`（`publish_members`、`verify_serving_policy`）、新文件 `elastic_placement.py`。

### 测试
- 全量 `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider`：ef2d6b0 基线 68 failed / 2668 passed / 49 skipped / 26 errors；infra-e1 68 failed / 2700 passed / 49 skipped / 26 errors。失败+错误 id 去重后两边均为 94 个且集合完全相同（`/tmp/infra-e1-base.ids`、`/tmp/infra-e1-new.ids`）。新增 `tests/test_rl_reconfig_e1.py`（26）与 `tests/test_rl_miles_adapter_e1.py`（6）。
- `openspec validate rl-infra-spec --strict`：valid。
- 说明：这些 CPU 测试的 fake 只模拟 fork 协议（epoch CAS、incomplete、cordon/admission），不作为 3.3/3.4/3.5/3.7 的验收证据。

### 追加（2026-09-30，响应 INFRA-E2 接口请求；entry.py INFRA 部分本轮归 E1）
- 吸收 `infra-e2-ports.patch`：`TrainerGroup` 的 E2 注释定稿为 `layout()`、`save_cut(*, epoch, context: CutContext) -> str`、`restore_cut(cut_id, *, epoch, root, expect: RestoreExpectation, shared_filesystem=True) -> CutManifest`，保持可选（不进 R0 协议）。同意，无异议。
- `RolloutPool.data_cursor()`（可选）：`rollout_meta_hook.data_cursor()` 在 rollout 进程内读 `{sample_offset, epoch_id, sample_group_index, sample_index}` 与 `get_buffer_length()`，经元数据进入 `RolloutBatchHandle.data_cursor/buffer_length`；`MilesRolloutPool.data_cursor()` 返回最近一批的游标（未知为 None）。buffer 长度非 0 时是否拒绝 cut 由 E2 在 `save_cut` 判定。
- 3.6 终态命名：超采样多出的**已完成**组按 3.6 原文记 `filtered`（机制名写明“超采样余量，不回 buffer”）；partial 关闭时**在飞被 abort** 的组新增终态 `engine_discarded`（原因、机制名、数量；非 lost、非 filtered、非 consumed）。报告 `buffer_length==0` 时 `carried_over=0`，否则 None（未知不猜）。`BatchLedger.cut_summary()` 提供 `ready_unconsumed`/`carried_over`（含 group id 列表、引擎报告的 carried_over 与 buffer 长度）给 `CutContext.ledger`。
- 4.4 钩子：`IslandDriver.rebuild_trainer(rebuild, *, cut_policy_hash)`：只在安全点调用；先要求 cut 的 policy hash = 当前发布的 policy，调用 `rebuild()`（E2 的 `rebuild_same_shape` 闭包，换 `SwappableActor` 背后的 handle 并 restore），再核对 trainer 导出的 hash，用同一 publisher 重发同一 policy 并做成员 ACK 校验；不调用 sync session（无外层副作用、不重复 initialize/after_local_train），发 `rl_trainer_rebuilt` 事件。
- entry 接线：`compose_island(..., elastic=ElasticWiring)`；`miles_adapter/elastic_wiring.build_elastic(state_dir, resources, attestation, profile, initial_config, runtime_fingerprint, declared_cells, pool_gpus)` 构造 controller（含 CommandInbox）、ledger；compose 时包 `ElasticPlacement` 并 `controller.open(pool)` 对账。**launcher/CLI 开关尚未加**（`config.py`/launcher 不在本 agent 范围），run_ports_island 目前不构造 elastic，默认行为不变。
- 与待合补丁的冲突预检：`infra-e2-entry-swappable-actor.patch` 三方合并干净；`infra-a-driver-2.3-eval-overlap-v2.patch` 有 4 处纯并列冲突（driver `__init__` 字段、`boundary` 后 `_join_eval()` 与 `ledger.outer_recorded`、entry 参数 `elastic`/`evaluate_start` 与 driver kwargs），两边都保留即可；建议 `_join_eval()` 放在 `ledger.outer_recorded` 之前。
- 全量测试复跑：68 failed / 2704 passed / 49 skipped / 26 errors，失败 id 集合与 ef2d6b0 基线相同（94）。

### 接口请求（需主 agent 协调，不在本 agent 写入范围）
1. **launcher/config 开关**（原“entry.py 接线”已完成 compose 部分）：给 run_ports_island 增加启用 E1 的参数（resources manifest、attestation、state dir、declared cells）后调用 `build_elastic`。原接线说明：构造 `MilesRolloutPool(declared_cells=<fork 启动时声明的 rollout cell id>)`、`ElasticPlacement(MilesPlacement, pool_gpus=...)`、`IslandController(state_dir=<岛持久 state>/..., configs, attestation, profile, initial_config, runtime_fingerprint, inbox=CommandInbox(...))` 与 `BatchLedger(state_dir)`，传给 `IslandDriver(controller=, ledger=)`；需 `--use-miles-router`。fork 的 InferenceController 没有列出已声明 cell 的公开方法（`list_declared_cell_ids` 只在 provider 上），接线时从 M1 映射/启动参数得到，或请 fork 负责人加一个 `@lock_exempt` 只读方法（新 M 项，需批准）。
2. **capabilities.py**（原 ALGO-CAP 文件）：本 agent 只在 `RESERVED_PORT_VERBS` 增加两项（3.4a 原文要求），单独提交 3a8b8f5，可单独审阅/回退。
3. **publish.py** 不在派发列出的文件中，但 3.5 必须改它，且无其他写入者；已改并在此声明。
4. **INFRA-E2**：cut 需要读取 `BatchLedger.unconsumed()/open_carried_over()` 与 journal `epochs.json`（config_epoch、成员）做 A3/4.2 对账；carried_over 实际范围待 4.1 审计后，在 `rollout_meta_hook`/`RolloutBatchHandle.carried_over` 报告后接入 ledger。
5. **INFRA-A**：`rollout.trajectory_load()`（active 请求 + tool-wait 计数）尚无 Miles 实现；1.7 的 tool-wait 仅有秒数。E1-C 需要一个在途 tool-wait 计数探针（可复用 1.7 采样线程）。

### 已知限制
- member publish 会让 trainer 内部整数 weight version 前进，而 executor 的整数版本只在下一次全量发布时更新（仅 fully-async 过滤使用，本 change 不用）。
- `check_weights` 返回值不带 cell id，payload 读回按“所有可寻址 engine 的 checksum 与上次同 token 全量发布的参照逐一相同”判定；异构 engine 形状 fail closed。
- watchdog 不能中断阻塞中的 Python 调用：到期只记 journal、回调 `on_watchdog`，调用返回后按失败处理（REBUILD_OLD 或 RECOVERY_REQUIRED）；进程级终止由 `on_watchdog` 接线提供（待 entry 接线）。

### 待批准
- 无新增。fork 增加只读 `list_declared_cell_ids` 属于新 M 项，如需要按 G6 同类另批（可用启动参数替代，不阻塞）。

### 下一步（可直接执行）
1. 主 agent 指派 entry.py 接线（接口请求 1），之后执行 `evidence/infra-e1/plan.md`（需自有 GPU）。
2. 独立审查本分支 5 个代码提交。

## 2026-09-30（Agent INFRA-A，分支 `infra-a`，基于 ef2d6b0；GPU/云暂停，未起任何付费资源）

### task 状态（五选一）
- 2.3：**CPU 通过**（age 0 下 eval‖train/outer_sync 重叠 + guard）；X9 overlap 实验待本地 GPU（`local-gpu-plan.md` L-2.3）。未勾选。
- 1.4：CPU 通过（新增 age 0 在途 batch=1、eval_in_flight 阻塞 cut）；X9 待 L-2.3。未勾选。
- 1.5、1.6、2.1、2.2、2.4：状态不变（CPU 通过 / GPU 证据齐备 / 合法否定结论），依赖链卡在 1.4 与 2.3 的 GPU 验收，未勾选。
- 1.7：未完成（CPU 部分完成；GPU 验证计划 L-1.7）。
- 5.1：未完成（测量与选择规则已实现 + CPU 通过；依赖 3.8；计划 L-5.1）。

### 提交与补丁
- 134a438：`yeto/rl/engine/overlap.py`、`execution_profile.py`（age 0 在途 batch=1、`eval_in_flight`）、`tests/test_rl_overlap.py`、`tests/test_rl_execution_profile.py`。
- d3629d3：`timeline.transition_cost_distribution`/`select_bottleneck`（5.1）。
- 本节提交：`local-gpu-plan.md`（INFRA-A 部分）、tasks.md 进展。
- 补丁（driver.py 归 INFRA-E1，未直接提交）：`/home/michael/work/infra-drafts/patches/infra-a-driver-2.3-eval-overlap.patch`，内容为 driver.py（DRIVER_MODES 加 partitioned-overlap、`evaluate_start` 参数、generate 前后与 publish 前的三个钩子、`_maybe_eval(defer=)`）、entry.py（`yeto_rl_overlap_eval` → partitioned-overlap profile、`loop_eval_starter` 接线、能力声明加 partitioned-overlap）与 `tests/test_rl_driver_overlap.py`。基于 ef2d6b0，需要 134a438 的 overlap.py；在临时 worktree 上 apply 后相关测试 52 passed。

### 待批准 / 待协调
- 主 agent 协调合入上述补丁（INFRA-E1 拥有 driver.py）。
- learner/launcher CLI 暴露 `--rl-overlap-eval`（→ `miles_args.yeto_rl_overlap_eval`），不属于 INFRA-A 写入范围。

### 下一步
- 本地有卡后按 `local-gpu-plan.md` L-2.3 → L-1.7 顺序执行；L-5.1 等 E1 本地通过。

### 追加（2026-09-30 INFRA-A）：审查修复与工具等待计数
- 审查 H1/M1/M2/L1–L4 已修复（215e1b8）；driver/entry 补丁换为 `patches/infra-a-driver-2.3-eval-overlap-v2.patch`（取代 v1）。
- 应 INFRA-E1 请求新增 `yeto/rl/engine/tool_wait.py`（瞬时在途工具等待计数与 drain 判定），不需要改 driver/rollout，因此没有补丁；E1 对接时用的接口：`board_actor(learner_id)`、`read_tool_wait(handle)`、`drain_blockers(router_in_flight, snapshot)`；生产侧用 `async_tool_wait_scope(handle, trajectory_id)`。

### 审查修复（2026-09-30 INFRA-E1，响应独立审查"需修复"）
- H1：`publish._commit_version` 只在 `start_commit_weight_version` 成功后才调用 `end_commit_weight_version`（fork `@acquires_lock` 失败时自行释放锁）；测试 fake 模拟 acquires/releases 语义并覆盖失败用例。
- H2：`build_elastic(on_watchdog=...)` 已接到控制器；**未实现** kill 目标 generation 的默认动作。结论如实：3.7 的“有界处理”未证明，阻塞中的引擎调用不会被截止时间打断（3.7 进展已注明）。
- M1：data_cursor/buffer_length 只在 `args.yeto_rl_elastic_metadata` 或 `YETO_RL_ELASTIC_METADATA=1` 时上报；默认元数据逐键不变、`carried_over` 仍为 None（回归测试 `test_default_metadata_is_unchanged_without_elastic`）。launcher 开关落地时需在 rollout 进程启动前设置该属性/环境变量。
- M2：合并 origin/infra-a（cf3e713，含 215e1b8），应用 `infra-a-driver-2.3-eval-overlap-v2.patch`，冲突两边保留，顺序为 `ledger.outer_recorded` → `_join_eval()` → `publish`；`safe_point_snapshot` 带 `eval_in_flight`，控制器 WAIT_SAFE 因此在延迟 eval 未完成时不 drain/remove（测试）。`MilesRolloutPool(tool_wait_board=...)` 的 `trajectory_load()` 用 `tool_wait.drain_blockers`（未知计数 fail closed），控制器按 blockers 判定排空；`ElasticWiring/build_elastic` 增加 `tool_wait_board`。
- M3：commit 后重启时 `compose_island` 按 `configs[config_id].placement["rollout"]` 调 `ElasticPlacement.restore_committed` 重建描述。恢复范围：只恢复 yeto 侧描述与 epoch；实际 engine 成员由控制器 `open()` 与 journal 成员比对，不一致转 RECOVERY_REQUIRED，不自动重启/停止 cell。
- M4：`rebase(start)` 把 rid<start 且停在 `optimizer_applied` 的批次提升为 `outer_recorded`（`recovered: true`）。
- L1：`engine_discarded` 进入 `_replay`（`batch()`、`cut_summary()` 可见）。L2：缺 `group_index` 时 `group_record` 明确报错（fail closed，所有路径）。L5：3.4a 完成记录注明 `publish_members` 命名。L6：E2/4.4 接口保留。
- 全量：68 failed / 2732 passed / 49 skipped / 26 errors，失败 id 集合与 ef2d6b0 基线相同（94）。

## INFRA-E2（2026-09-30，4.1–4.5；分支 `infra-e2`，worktree `/home/michael/work/infra-e2`，基于 integ-decl ef2d6b0）

### task 状态（五选一，均未勾选）
- 4.1：已实现（缺状态拒绝的部分 CPU 通过）。依赖 3.1 未勾选。审计见 `cut-audit.md`。
- 4.1b：已实现（`docs/MILES_RL.md`），待评审。
- 4.2：CPU 通过。依赖 3.6 未完成；GPU 待本地验证。
- 4.3：已实现（同形 rebuild 与 restore 编排）；X3 GPU 待本地验证。
- 4.4：未完成（driver.py/entry.py 不归 E2；补丁与接口请求已交）。
- 4.5：未完成（rebuild/restore 失败分支已有 CPU 测试；其余依赖 4.4、3.8 与 GPU）。

### 关键结论
- E2 profile：bf16 LoRA，走 yeto 自己的 cut 插件，复用 fork-M5 的命名 optimizer 状态（含 FP32 master、moments）与 RNG 采集；不开 `--lora-dp-invariant-state`，也不改默认 checkpoint 参数。拒绝的配置：fp16、precision-aware、DistOpt 多实例、CP>1、EP>1、非 LoRA 可训练参数。沿用已知限制：TP/PP 集体导出与 DistOpt 分片主参数不能同时使用。
- F5：Miles ports 路径上 `carried_over` 恒为 0。超采样多出的完成组和在飞被 abort 的组都被引擎丢弃（游标已前移，不复用），建议 3.6 为它们单列终态（由 E1 决定）。
- 同形重建的前提是 `args.load is None`（否则 `create_training_models` 会调用 `rollout_executor.load` 回卷游标），且 `start_rollout_id` 已设置。

### 测试
- 全量 `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider`：68 failed / 26 errors / 2721 passed / 49 skipped。失败 id 集合（94 个）与 ef2d6b0 基线（68 failed / 26 errors / 2668 passed）按 id 完全相同。新增 53 个测试全部通过。
- 首轮全量曾新增 1 个失败：`test_provenance::test_production_tree_has_no_unsafe_torch_load...`，原因是 cut 分片用了 `weights_only=False`。已修复：NumPy RNG 编码为张量，加载改为 `weights_only=True`。

### 证据与计划
- `cut-audit.md`；`evidence/infra-e2/4.2-4.5/plan.md`（待本地 GPU 验证，判据已预先固定）。
- 云资源：无；费用 $0；没有启动任何 GPU 或云资源。

### 交给其他写入者
- `/home/michael/work/infra-drafts/patches/infra-e2-ports.patch`（E1：ports.py `TrainerGroup` 的 E2 签名注释定稿）。
- `/home/michael/work/infra-drafts/patches/infra-e2-entry-swappable-actor.patch`（entry.py：用 `SwappableActor` 包装 actor）。
- 接口请求（`cut-audit.md` §5）：`RolloutPool.data_cursor()`、buffer 长度、3.6 终态与 `ready_unconsumed/carried_over` 计数、4.4 driver 重建后重发且不重复 initialize。
- 待合入：ALGO-2b 的 `infra-drafts/patches/algo-2b-trainer.patch` 可以在 infra-e2 上干净应用（`git apply --check` 通过）；它依赖 ALGO-2b 的 spec 字段，E2 未应用，由主 agent 在集成时合入。

### 下一步
1. E1 定稿 ports 签名，提供 data_cursor 与账本计数后，driver 实现 4.4（重建 → restore → 重发 → 校验）。
2. 本地 GPU 到位后按 plan 依次跑 G-4.2 → G-4.3（DP1、DP2+DistOpt）→ G-4.4 → G-4.5。

### INFRA-E2 审查修复（2026-09-30，"需修复"结论）
- H1：重建前提改为检查 `args.requested_load is None`。原因是 bridge 模式下 `args.load` 等于 ref_load，`start_rollout_id` 等于 0。重建前后各读一次 `data_cursor()`（E1 接口未就绪时用 `DataCursorSource` 协议占位），不一致即判 RECOVERY_REQUIRED；该检查覆盖 `generate_rollout.load`。`cut-audit.md` §2 已更正。
- H2：写入前断言 scheduler `num_steps==0`（`restore_cut` 只用于新建 trainer），并预检超参；测试替身改为 Megatron 的累加语义；新增"在线 trainer 就地恢复被拒绝"的测试。
- H3：`SwappableActor.dispose()` 在调用时解析当前 target。已确认 EvalDispatcher 只保存代理对象，没有缓存方法。
- M1：swap 之后出现任何异常都判 RECOVERY_REQUIRED。M2：`shared_filesystem=False` 且 DistOpt、DP>1 时拒绝。M3：写分片前先做 context 检查，并拒绝 TP/PP+DistOpt。M4：新版计划 `evidence/infra-e2/4.2-4.5/plan-v2.md`（v1 保留，并标注已被取代）。
- L1：`has_optimizer_state` 要求每个 (tp,pp) 覆盖全部 adapter 名；重建后的布局从 rank 读回（`actual_layout`）。L3：yeto ports 引擎不读 `args.start_rollout_id`。L2：已写入 plan-v2 §3，待 GPU 确认。
- 补丁：`infra-e2-ports-v2.patch`、`infra-e2-entry-swappable-actor-v2.patch`；v1 已改名为 `*.v1-OBSOLETE.patch`。
- task 状态不变：4.1 已实现；4.1b 已实现；4.2 CPU 通过；4.3 已实现；4.4、4.5 未完成。均未勾选。

## LOCAL-CLUSTER 调查（2026-09-30）
- 交付：`openspec/changes/rl-infra-spec/local-cluster-investigation.md`（只读调查，无代码改动，无云资源，无集群启动，未改 ~/.sky）。
- 结论：head（FleetController+LocalSyncer）已可在自有机器常驻（`run_local_head.py` 已证），缺正式 CLI 入口；岛可经 ssh_harness（钉 H200）或 SkyPilot 0.13 的 ssh node pool（k3s+GPU Operator）/kubernetes。推荐 A′（本机 head 正式化 + ssh_harness 泛化，3–5 人日）先行，B（sky ssh 池）按需，暂不做常驻服务端 D。
- 状态：报告已实现；无 task 勾选；GPU 相关全部"待本地 GPU 验证"。
- 待用户决定：见报告 §4.2（卡型号/台数、A′ vs B/C、放开 H200 钉死、harness 路径能否作验收证据、逐位验收改同机型自比、私有镜像拉取方式）。
- 分支 local-cluster（worktree /home/michael/work/local-cluster）。测试：未运行（仅文档）。

## 集成 integ-s2（2026-09-30，主 agent）
- 分支 integ-s2（worktree /home/michael/work/integ-s2），基于 origin/integ-decl 0727a31；修复由子 agent FIX-S2 完成，推送 integ-s2，等主 agent 核对后快进 integ-decl。
- 合并内容与 SHA：infra-e1 d33d541；infra-e2 df28780；algo-2b e9a8d19；local-cluster 0d4c9e8；infra-e2-ports-v2 注释补齐 0ae6f30；infra-e2-entry-swappable-actor-v2 补丁 f7f0ce6；algo-2b-trainer 补丁（GMPO 收集 pg_clipfrac）de31115；开关接线 0727a31。
- 新增 CLI 开关（都是可选，默认关闭，默认命令不变）：launcher/learner `--rl-overlap-eval`（2.3）；`--rl-elastic`，配套 `--rl-elastic-resources`、`--rl-elastic-attestation`、`--rl-elastic-initial-config`、`--rl-elastic-cells`（learner 还有 `--rl-elastic-state-dir`；launcher 固定用 `~/yeto-rl/elastic-state`）。
- 本次修复（集成审查结论）：
  1. 中：默认 GRPO 不取走 `_STEP_LOSSES`，列表无界增长，save_cut 报 "per-step records not drained"。改为每轮 train 成功后无条件取走，只在 GSPO/corrections/GMPO 时用于指标（e368f4e，8d03c65 更新组合根测试替身）。新增测试：GRPO 跑多轮后列表为空，且 save_cut 的 drain 检查通过；GSPO/GMPO 的 clip_fraction 不变。
  2. 低：`YETO_RL_ELASTIC_METADATA` 只在 driver 进程可见。现在 `connect_island_ray` 在开关打开时把它放进 job 级 runtime_env.env_vars（0791db8）。更正：0727a31 提交说明写的是"run_ports_island 之前设置环境变量"，实际只覆盖 driver，Ray worker 拿不到，本提交已修正。
  3. 低：launcher 本地 `_check_ports_infra_switches` 在开资源前检查：overlap 需要 fixed-partition（`execution_profile.check_overlap_eval`，与 entry 共用）；elastic 拒绝 colocated（`check_elastic_placement`）；initial-config 必须在 manifest `parse_configs` 的结果中；attestation 能被 `load_attestation` 解析（7139080）。7139080 引入了回归：launcher 读取的 `args.eval_interval` 恒为 None，导致 overlap 一律被拒。后续提交已修正：追查了来源，`--rl-placement` 原样转发给 learner，也就是岛上的 `launch.placement.kind`，因此在本地检查；launcher 完全没有 eval 配置来源（没有 `--eval-interval`/`--eval-uses-snapshots`，没有运行配置文件，也没有 Miles argv 透传），因此这两项以 `UNKNOWN` 传入，留给岛上用 miles_args 检查。已知缺口：经 launcher 启动的岛，learner 拿不到 `--eval-interval`，所以 overlap 仍会在岛上被拒。要打通，需要 launcher 增加 eval 配置接线（范围外，待定）。
  4. 低：elastic 与 overlap 同时开启时，`rl_reconfiguration` 事件带 `eval_due`，表示已排期但未启动的 eval；有组合测试（36f4a9a）。
  5. build_elastic、journal、ledger、inbox 的路径统一做 expanduser，有测试（本节之前的最后一个代码提交）。
- 测试：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider -rfE` 结果为 68 failed, 2904 passed, 49 skipped, 26 errors。（回归修复后复跑：68 failed, 2905 passed, 49 skipped, 26 errors）失败和错误的 id 共 94 个，按 id 前缀规范化后与 /tmp/integ-s2-base.ids（修复前基线）完全一致，没有新增失败，都是已知环境性失败。`openspec validate rl-infra-spec --strict` 通过。
- 仍存限制：3.7 watchdog 默认没有接 kill（`on_watchdog` 默认未接线，阻塞的引擎调用不受截止时间约束）；H2 限制见 E1 记录。GPU 验收均未进行，task 勾选状态不变。
- 云资源：无；费用 $0。

## GPU 验收计划 v2（2026-09-30，PLAN-V2；分支 gpu-plan-v2，基于 integ-decl 65ca03b）
- 用户决定：GPU 验收先做 A1–A9，A10（4.8）暂缓；A1–A9 硬上限 $300（含重跑）。
- 交付：`gpu-plan-v2.md`（逐项 task/原文、复用证据、卡×数、时长、GPU·h、估价、事先固定判据、硬超时、回收、前缀 `infra-v2-*`、批次与预算、砍项顺序）；`gpu-plan.md` 顶部注明被 v2 取代（A1–A9 部分）；alignment §7b 追加用户决定。
- 期望费用：4 卡档 ≈$154（4.6 no-go ≈$131），8 卡原文档 ≈$213；批次硬预算 4 卡档 B1 $115 / B2 $95 / B3 $65 / 未分配 $25，8 卡档 B1 $170 / B2 $80 / B3 $50。
- 复用不重跑：A1（1.2 已勾）、A3（2.4 合法否定结论）、2.1/2.2/2.3 X9 guard 已有 GPU 证据。
- 待用户确认：A4 4 卡代替原文 8 卡（实质改变验收）；A2+（L-1.7）；可选 Nebius；未分配 $25 的动用授权。
- 状态：仅规划；未启动任何资源，费用 $0；无 task 勾选变化。4.8：未完成（用户决定暂缓）。

## 2026-09-30（Agent INFRA-E1 第二轮：launcher eval 接线、3.8、4.4、H2；分支 `infra-e1`）

### 分支与状态
- worktree `/home/michael/work/infra-e1`，基于 integ-decl 65ca03b。提交：f79e016（launcher eval）、533afdc（3.8）、d397cf3（4.4）、2b67145（H2 watchdog）、4d8ec81（计划与 tasks 进展），以及本条目所在的提交。已普通推送 `origin infra-e1`。未启动任何 GPU 或云资源，费用 $0。

### task 状态（五选一）
- launcher eval 接线（非 task，A2/L-2.3 前置）：**已实现 + CPU 通过**。
- 3.8：**已实现 + CPU 通过，未勾选**（X6 需两岛 GPU；依赖 3.7、2.4 未勾选）。
- 4.4：**已实现 + CPU 通过，未勾选**（依赖 4.3；需 GPU A6b）。
- 3.7（H2）：**已实现 + CPU 通过，未勾选**。默认 watchdog 会杀掉目标 generation；限制仍在，见 tasks 3.7。
- 4.5：未完成（新增两条 CPU 覆盖）。

### 关键改动
- launcher 新增 `--rl-eval-interval/-data/-dataset-name/-samples-per-prompt`，转发为 learner 的 `--eval-*`。heldout 文件内联进运行命令（≤1 MiB），岛上按 SHA256 校验。learner 的 `_verify_eval_dataset_identity` 与 `run_config._resolve_eval` 共用 `ports_training_eval()`：ports LoRA 允许训练期 heldout eval（必须是与 `--data` 不同的文件），legacy LoRA 仍然拒绝。launcher 本地的 overlap 检查改用它实际转发的同一个 interval。**此前的真实缺口**：不止 launcher，直接启动的 learner 在 ports LoRA 上带 `--eval-interval` 也会被 `_resolve_eval` 拒绝（"restricted to dense full mode"），所以 2.3 的 overlap eval 以前在任何入口都起不来。
- 3.8：`SyncSession.outer_phase`（strict、decoupled）；`IslandController.run_at_safe_point(..., outer_phase=)` 重新做 pause 决定并写入 journal；`enter_finalization`；暂停预算的输入沿 launcher → learner → build_elastic 传递，同时传给 syncer 的 `--quorum-timeout-s`。
- 4.4：`IslandController.request_trainer_rebuild`、`REBUILDING_TRAINER`、`RebuildRefused`；`miles_adapter/rebuild_wiring.py`；driver 新增 `local_step`；`entry._wire_trainer_rebuild`。
- H2：`elastic_wiring.kill_target_generation`（默认启用，`on_watchdog=None` 可关闭）；controller 的 journal 追加加了线程锁。

### 写入范围说明
- 本轮改了 `yeto/cli.py`（launch 新参数）、`yeto/rl/learner.py`（eval 身份校验与 elastic 暂停参数）、`yeto/rl/engine/run_config.py`（`_resolve_eval` 的 ports 分支）、`bridges.py`（`outer_phase`）。这些都属于 launcher/learner 的 INFRA 接线，没有其他写入者，在此声明。trainer.py、state_plugin.py 未改，没有交给 E3 的补丁。

### 测试
- 全量 `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider -rfE`：68 failed / 2937 passed / 49 skipped / 26 errors。失败与错误的 id 按第二列去重共 94 个，与 `/tmp/integ-s2-base.ids` 完全相同，没有新增（`/tmp/infra-e1-r2.ids`）。
- 新增和修改的测试：`tests/test_rl_infra_switches.py`（eval 与暂停参数，+9）、`tests/test_rl_reconfig_x6.py`（10）、`tests/test_rl_trainer_rebuild_e1.py`（10）、`tests/test_rl_reconfig_e1.py`（watchdog，+3）。
- `openspec validate rl-infra-spec --strict`：valid。
- fake 只证明协议，不作为 3.8、4.4、3.7 的验收证据。

### GPU 计划
- `evidence/infra-e1/plan-3.8-4.4.md`：A5（X6-a..e）、A6b（4.4）、§3 watchdog。判据在运行前固定。X6-b 与 §3 需要的故障注入点**尚未实现**，执行前须先补上并提交。

### 已知限制与阻塞
- 4.4：`MilesRolloutPool.data_cursor()` 返回缓存值，不在 rollout 进程内实时读取；重建前后的游标比对因此检查不到 `rollout_executor.load` 的回卷。
- 3.7：阻塞在旧成员集上的调用仍不受截止时间约束；fork health monitor 对被杀 cell 的行为未知。
- head 两跳（fleet head 模式）下，`--rl-eval-data` 与 `--rl-elastic-resources` 一样，只内联到岛的运行命令，没有另行处理 head 上的暂存。
- 待批准：无新增。

### INFRA-E1 第二轮审查修复（2026-09-30，结论"需修复"）
- F1：新执行说明 `evidence/infra-e1/plan-3.8-4.4-v2.md`，以集成分支 `gpu-plan-v2.md` §2–§3 为唯一判据来源，只补充开关、前置条件、E1 观测点与已知限制。v1 `plan-3.8-4.4.md` 保留，并标注"已被取代"。删去了两条不可达的判据："最终 policy hash 等于 B0"与"syncer base 等于 B0"。A6b 的 `progress.local_step` 写成确定值 3。需要主 agent 知悉、gpu-plan-v2 可能需要补充的两点：(a) A5 quorum 用例要让 150 s 延迟通过 pause 审计，必须带 `--rl-elastic-pause-margin 2.0`，deadline 设 230 s，否则请求会在 plan 阶段被拒；(b) "start_cells 前注入 150 s"的注入点尚未实现，执行前须先补上并提交。
- F2（5946ffd）：watchdog 的判定与终止、`tx.phase` 变更、commit CAS 前的复查共用一把锁。watchdog 已触发时，事务走 REBUILD_OLD；提交开始后，watchdog 不再终止任何 cell。新增 2 个测试。
- F3（e477bcb）：driver 新增 `RebuildNotStarted`（无发布、不在安全点、cut hash 不符），rebuilder 在写 cut 之前检查 `rebuild_preconditions(miles_args)`，这两类拒绝都判 CANCELLED。前提不满足时 entry 不接线 rebuilder，请求在 plan 阶段即被拒。新增 3 个测试。
- F4：已写入 v2 §4 第 4 条（被杀 cell 所在 bundle 的 GPU 上 `nvidia-smi --query-compute-apps` 为空）。tasks 4.4、4.5 已写明 `REBUILDING_TRAINER` 阶段不受 deadline 强制终止的限制。
- F5：v2 §3 第 3 条：重建后下一轮的 `trained_sample_ids_sha256` 与数据游标须与 B1 相等。
- F6（500555a）：内联 eval 数据上限改为 96 KiB，并加测试。
- 全量：68 failed / 2943 passed / 49 skipped / 26 errors，失败 id 共 94 个，与 `/tmp/integ-s2-base.ids` 相同（`/tmp/infra-e1-r2b.ids`）。validate strict 通过。
- E3 吸收工作暂停在本地分支 `infra-e1-e3wip`（430f49f，未推送，未完成）。

## INFRA-E3（2026-09-30，4.2a/4.6/4.6a/4.7 的 CPU 部分；分支 `infra-e3`，worktree `/home/michael/work/infra-e3`，基于 integ-decl 65ca03b）

### task 状态（五选一，均未勾选）
- 4.2a：未完成。fork 接口已满足（按名 optimizer 状态 + DistOpt 文件式收集/切片），缺 GPU 验证（DEV-GATHER/A8）。
- 4.6：CPU 通过。GPU 验收 A8 未运行；依赖 4.5、1.6 未满足。
- 4.6a：未完成。trainer 侧 M6 满足；角色转移缺 fork 需求 F-R1（启动时声明延迟绑定的 rollout cell，需另批）。
- 4.7：已实现（yeto 侧，CPU 通过）；controller 接线以补丁交 E1；A9 依赖 4.6 go、F-R1。

### 改动
- `yeto/rl/engine/miles_adapter/reshard.py`（新）；`cut_plugin.restore_resharded_shard`；`MilesTrainerGroup.restore_cut_resharded/rebind_args`；`trainer_rebuild.rebuild_resharded/resized_args/trainer_view`；`miles_adapter/trainer_resize.py`（新，`MilesTrainerOps`）；`engine/trainer_transition.py`（新）。
- 测试：`tests/rl_reshard_fakes.py`、`tests/test_rl_trainer_reshard.py`（13）、`tests/test_rl_trainer_transition.py`（12）。纯 torch 替身，只证协议，不作验收。
- 计划：`evidence/infra-e3/plan.md`（DEV-GATHER、A8、A9，判据与上限预先固定；E3 上限合计约 $46）。

### 测试
- 全量 `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider -rfE`：68 failed, 2930 passed, 49 skipped, 26 errors；失败/错误 id 94 个，与 `/tmp/integ-s2-base.ids`（第二列）完全相同，无新增失败。
- 补丁验证：在本分支临时应用两个补丁后 `tests/test_rl_controller_trainer_edge.py` + `tests/test_rl_reconfig_e1.py` 37 passed，随后撤回。
- `openspec validate rl-infra-spec --strict` 通过。云资源：无；费用 $0。

### 交给其他写入者
- `infra-drafts/patches/infra-e3-controller.patch`、`infra-e3-elastic-wiring.patch`（基于 65ca03b；在 infra-e1 533afdc 上 controller.py `__init__` 附近冲突，需手工合入）。接口见 plan.md §5。
- E1 需实现 `MilesRolloutPool.bind_members(members, gpus)` 与 `compose_island` 中 `MilesTrainerOps` 的构造。

### 待批准 / 阻塞
- F-R1（fork 新需求，与 G6 相邻）：阻塞 A9/4.7 GPU。
- PLAN-V2 需采纳本计划的缩小规模（A8 2×H100!、A9 4×L40S T2R2↔T1R3、DEV-GATHER 用 A10G）。
- dropout>0 的变 DP 边不在首轮认证范围（plan.md §0）。

### INFRA-E3 审查修复（2026-09-30，"需修复"结论）
- H1：`reshard.py` 改按 fork scheduled 路径建模（`scheduled_partitions`/`sample_mapping`/`step_problems`，GBS 按 rollout 计、`num_rollouts` 归一）；删除 round-robin 假设；拒绝 `--balance-data`、`--balance-by-flops`、动态 batch、部分步、vpp>1。A8 arm 改走 `split_train_data_by_dp` 真实分派（plan-v2）。
- H2：rollout→trainer 要摘除的 engine 由 pool 的成员→GPU 映射按 `moved_gpus` 选出（`members_on_gpus`），选不出、跨界或数量不符在 plan 阶段拒绝，执行前再核一次（变化则 CANCELLED）；新增"成员名顺序与 GPU 顺序不一致"测试。
- M1：DP 变化时 `lora_dropout`/`hidden_dropout`/`attention_dropout` 任一非 0 或未知即拒绝。M2：loss 权重按 fork `loss_function` 缩放参数化复算；CPU 只验证算术，归一化证据交 A8 G2（tasks 4.6 已改述）。
- M3：`infra-e3-controller-v2.patch`（v1 改名 `.v1-OBSOLETE`）：提交 CAS 失败 → `_enter_recovery` + `trainer_recovery_hint`（restore_old，含 cut_epoch），有测试。
- M4：`resize/restore_source` 显式接收 cut epoch；`trainer_cut` 记录 `cut_epoch`，`recovery_decision` 的提示带 epoch 并写明边界；新增"重启后无 save_cut 也能 restore_source"测试。
- L1：yeto 侧替身（`tests/rl_reshard_fakes.py`）自带简化的合并逻辑，**不证明** fork-M5 的区间重叠/覆盖检测；那部分依赖 fork 自己的 CPU 单测与 GPU 验证。L2：reshard 文档措辞已改。L3：`trainer_view` 调用前检查 `_slice_pg_info` 存在且签名为 `(info, indices)`（miles 在 CPU 环境不可 import，故在调用时而非模块 import 时检查），有签名测试。L4：plan-v2 写明 G3 第二次仍不可判定即 no-go。
- F-R1：已从源码核实（`RayWorkerManager.init` 对全部已声明 cell 执行 `start_cells`；cell 只能绑本 pool 视图），需求写入 plan-v2 §4。
- 测试：全量 68 failed, 2937 passed, 49 skipped, 26 errors；失败/错误 id 94 个与 /tmp/integ-s2-base.ids（第二列）完全相同。v2 补丁临时应用后 controller 级 + E1 reconfig 测试 38 passed，随后撤回。openspec validate --strict 通过。
- 状态不变：4.2a 未完成；4.6 CPU 通过；4.6a 未完成；4.7 已实现（yeto 侧，CPU 通过）；均未勾选。

### INFRA-E3 复审修复（2026-09-30；复审结论"A8 可按 plan-v2 执行"，以下为 A9/生产路径问题）
- M1：`infra-e3-controller-v3.patch`（v2 改名 `.v2-OBSOLETE`）：提交 CAS 抛错后读回 epochs，`last_tx_id` 为本事务（写入后 fsync 才抛错）→ hint `restore_target`；否则 `restore_old`；读不回 → `recovery_required`；均进入 RECOVERY_REQUIRED。新增"写入后才抛错"测试。
- M2：`batch_problems` 写入前拒绝 `--indep-dp`、`--multimodal-keys`；DP 变化后 `MilesTrainerGroup.train_step` 训练前经 `batch_guard_problems` 守卫：各 rank 须公布完整 `train_parallel_config` 且 dp 等于计划，批次 rollout 数须为 steps×GBS 且被两侧 scheduled 路径接受，否则拒绝（payload 照常释放）。残余：yeto 看不到分片内容，无法直接读 `micro_batch_indices`；守卫覆盖的是 fork 退回 raw 的全部条件（`can_schedule_on_rollout_side`，`rollout_ids` 由 fork 恒设），A8 在 rank 内直接断言分片带 `micro_batch_indices/num_rollouts`。
- L1：静态整除检查的"每 rollout 1 条样本"假设已写入文档，生产每批由守卫调用 `step_problems`。
- L2：`plan-v3.md`（v2 保留并标注已取代）A9 判据新增"新 engine 所在 GPU 等于 moved GPU"。
- 测试：全量 68 failed, 2940 passed, 49 skipped, 26 errors；失败/错误 id 94 个与 /tmp/integ-s2-base.ids（第二列）完全相同。v3 补丁临时应用后 controller 级 + E1 reconfig 测试 39 passed，随后撤回。
- 状态不变：4.2a 未完成；4.6 CPU 通过；4.6a 未完成；4.7 已实现（yeto 侧，CPU 通过）；均未勾选。无云资源，$0。
### INFRA-E1 追加（2026-09-30）：测试注入点与 E3 接口吸收
- A5 用的 `start_cells` 前延迟注入：9a5f181。开关为 launcher `--rl-test-inject-start-delay-s`，也可直接设环境变量 `YETO_RL_TEST_INJECT_START_DELAY_S`。只在本进程第一次 start_cells 时生效；默认关闭。
- §4 watchdog 用的 `update_weights` 阻塞注入：d0970ba。开关为 launcher `--rl-test-inject-update-weights-block-s`，也可直接设 `YETO_RL_TEST_INJECT_UPDATE_WEIGHTS_BLOCK_S`。阻塞在 yeto 侧模拟：每秒探测一次目标 worker actor 是否存活，actor 死亡后立即报错。默认关闭。执行说明见 `evidence/infra-e1/plan-3.8-4.4-v2.md` §1、§4。
- E3 接口（7e381d7）：按 controller-v3 语义手工合入 `infra-e3-controller` 补丁与 `infra-e3-elastic-wiring` 补丁。
  - 提交点只有一个，即 `_commit`，rollout 边与 trainer 边共用。CAS 失败一律转 RECOVERY_REQUIRED；trainer 边会额外按 durable epochs 写 `trainer_recovery_hint`：`last_tx_id == tx` 时为 `restore_target`，否则为 `restore_old`。F2 的 watchdog 复查也覆盖 trainer 边。
  - 对 `trainer_transition` 的 import 加了保护：infra-e3 集成之前，trainer 边一律拒绝。E3 的 `tests/test_rl_controller_trainer_edge.py` 加了 importorskip。
  - 新增：`MilesRolloutPool.bind_members/member_gpus/members_on_gpus`、`miles_adapter/bundles.StartupBundles`、`rebuild_wiring.CutSource`、`entry._wire_trainer_edges`（构造 `MilesTrainerOps`；需要 `ElasticWiring.pool_gpus`）、`IslandController.set_trainer_edges`。
  - **依赖**：在真实 fork 上，`bind_members` 依赖 fork 缺口 F-R1（启动时无法声明位于 rollout 视图之外的停止 cell）。`pool_gpus` 目前没有由 learner 或 launcher 传入（`build_elastic` 默认 None），因此生产路径上 trainer 边不会被接线，需要另行接线或由主 agent 决定。
- 测试：与 origin/infra-e3（71207b7）临时合并后，相关测试 133 通过；合并后全量 68F/2994P/49S/26E，失败 id 94 个，与基线相同。infra-e1 本身全量 68F/2960P/51S/26E，失败 id 94 个（`/tmp/infra-e1-r2d.ids`），与 `/tmp/integ-s2-base.ids` 相同。validate strict 通过。

### INFRA-E1 第三轮审查修复（2026-09-30，结论"小修后可合入"）
- M1：`_commit` 恢复为 v3 的三态语义，trainer_recovery_hint 增加 `committed` 字段：True 对应 restore_target，False 对应 restore_old，epochs 不可读时为 None 并对应 recovery_required。v3 补丁中 `tests/test_rl_controller_trainer_edge.py` 的两条端到端 CAS 失败用例已原样补回（infra-e3 集成之前由 importorskip 跳过），另加 unreadable 用例。
- M2：注入阻塞的存活探测改为 `RayTargetLiveness`：阻塞开始时记录 (name, generation)，之后 worker 列表为空、generation 变化或 actor 死亡都判为死亡；其他异常单独以 `InjectedBlockProbeError` 报出，不算作被 kill。三种情形都有测试。
- L1：`_trainer_record` 修改 tx.phase 时持 `_watchdog_lock`；watchdog 在 trainer 边与 rollout 边上的差异写入 plan v2 §6。
- L2：`_wire_trainer_edges` 在 `rebuild_preconditions(miles_args)` 不满足时不接线 trainer 边，请求在 plan 阶段即被拒；有测试。
- L3：**F-R1 未解决前不得认证 role-transfer 的 trainer→rollout 边**（plan v2 §6）。启动期的 bind 能力检查不可行（fork 没有可读接口），也未实现。
- pool_gpus 入口：按主 agent 安排，等 F-R1 在 fork 实现后再做。

### INFRA-E2 GPU harness 与注入（2026-09-30，仅 dry-run；未启动任何 GPU/云资源）
- 容器内 harness：`miles_adapter/e2_harness.py`。snapshot 根目录有 `yeto-rl-e2-harness.json` 时代替 `driver.run()`，依次执行：逐 rank 确定性读回、G-4.2 (a)–(g)、G-4.3 冻结 batch 的双 arm 逐位比较、L2。CPU 伪岛测试覆盖通过、RNG 丢失、确定性缺失、dropout 不符四种情形。
- 主机侧：`tools/probes/e2_cut_harness.py`，按 plan-v3 顺序生成 11 个 run 目录。`run.sh` 带 `YETO_E2_GPU_APPROVED` 守卫、watchdog、puller 与 rebuild 触发器；工具本身拒绝 `--execute`。本地两级检查为 launcher `--dry-run` 与 learner preflight（与容器内 learner 在 GPU 之前的检查相同，并翻译出 Miles argv）。
- 本地 dry-run 结果（`evidence/infra-e2/4.2-4.5/dry-run-20260930.md`）：11 个 run 的 launcher dry-run 全部 rc=0。C3 九个 run 的 preflight rc=0；C1/C2 的 preflight rc=1，原因是 ports 路径 `--lora-dropout` 固定为 0，而计划要求 0.05。这条阻塞已在本地暴露，判据未改。
- A7 注入开关（`miles_adapter/cut_injection.py`，默认关闭）：
  - save 途中 kill rank；
  - restore 途中 kill rank；
  - restore 前 sleep；
  - rebuild 在 fork 的 `create_training_models` 处失败；
  - rebuild 前写入偏移的数据集游标文件；
  - 第 6 项（CAS 前后 kill controller）使用 E1 的 `--rl-test-kill-learner-at` 加 restart loop，在同一容器内重启。
- plan-v3：镜像 17d428a2…（5c1b49e-9f29303）、Miles 5c1b49eb、Qwen3-0.6B c1899de…、Qwen3-1.7B 70d244cc…；判据沿用 plan-v2 原文。
- 阻塞：① LoRA dropout 0.05 无法表达（需要 config 翻译开关，或由主 agent 裁定）；② G-4.5 第 5 行需要 E1 提供实时 `data_cursor`；③ E1 需合入补丁 `infra-e2-e1-harness-injections-v1.patch`。
- 测试：全量 68 failed / 26 errors / 3059 passed。失败 id（94 个）与 integ-decl b90c18d 基线（3037 passed）按 id 相同。

### INFRA-E1：第 1 批 GPU 冒烟发现的集成缺口（2026-09-30；已合并 integ-decl f6938d9）
每项都有端到端测试：真实 `yeto launch` CLI 生成岛运行命令 → 临时 HOME 下执行岛上的 prelude → 真实 `learner.parse_args` 解析。测试辅助在 `tests/rl_e2e_launch.py`，用例在 `tests/test_rl_launch_e2e_b1.py`、`tests/test_rl_e1_injections.py`。
1. **7c6cd90，经本条后续提交改正（F-R1 的 yeto 侧准备）**：按主 agent 裁定，以 FR1 的 `rollout_cells` 为唯一接口，`--rl-elastic-deferred-cells`/`deferred_rollout_cells` **已作废删除**。
   - 新开关 `--rl-elastic-declare-cells`：把 `--rl-elastic-cells` 的 yeto 名字写进 `--yeto-placement-map` 的 `"rollout_cells": [{name, bundles, start}]`。
   - 布局按名字顺序：先在 rollout 角色的连续 bundle 上声明启动的 cell，再在 standby 上声明停止但已绑定的 cell，其余为停止且未绑定（`bundles=[]`）。
   - fork 保留自己的 cell id，并在 `describe_cells` 中报告 `alias`。组装时 `resolve_declared_cells` 按 alias 把 yeto 名字映射成 fork cell id，此后 E1 动词与 journal 使用 fork cell id；未知名字一律拒绝。
   - 不带该开关时行为不变（旧 fork 可用，名字即 fork cell id）。
   - pool_gpus 入口等 F-R1 合入后再接。
2. **601d0f1**：新增 `--rl-eval-temperature/-top-p/-max-prompt-len/-max-response-len/-max-context-len`，转发为 learner 的 `--eval-*`。A2 贪心 eval 用 `--rl-eval-temperature 0`。
3. **6838fe9**：新增 `--rl-observe-timeline`，经 `miles_args.yeto_rl_observe_timeline` 传到 entry 的 observe。新增 `--rl-elastic-tool-wait-board`，弹性接线用岛上具名的 ToolWaitBoard actor，在 Ray 连上之后才创建。
4. **b90c18d**：测试负载 `yeto.rl.tool_wait_workload.generate`，经 Miles 支持的 `--custom-generate-function-path` 接入，配合 `--rl-test-tool-delay-s S`。每条训练轨迹先在 board 上登记一次假工具等待，时长计入 `non_generation_time`；环境变量由 `connect_island_ray` 转发到所有 Ray worker。A2+ 用这组参数；A4b 另加 `--rl-elastic-tool-wait-board`。
5. **fea44cc**：注入开关，均仅供测试、需 `--rl-elastic`、默认关闭。
   - `--rl-test-inject-weight-override ISLAND_PATH`（E1-B）：经 SGLang `/update_weights_from_disk` 把一个新 engine 换成另一个同架构 checkpoint。**运行前须确认** `check_weights` 的 checksum 覆盖被替换的张量，否则该注入无效，按环境阻塞处理。
   - `--rl-test-inject-stop-failures N`（E1-D ③④）：让 fork 自身的 engine provider 在 stop 时抛错，走 fork 真实的 incomplete 路径。
   - `--rl-test-kill-learner-at PHASE`（E1-D ⑤ 用 COMMITTED，⑥ 用 QUIESCING）：在事务写入该 phase 后立即 `os._exit`，每个 state dir 只杀一次。
   - E1-D ⑦ 的 fork 重启：learner 原地重启时旧 Ray job 随之结束，fork 的 InferenceController 以 epoch 0 重建，与 ⑤⑥ 用同一个入口。
6. **fea44cc**：`--rl-elastic-state-dir ISLAND_PATH` 可指向持久卷（Modal volume 挂载点）。`--rl-elastic-restart-attempts N` 让岛运行命令用 bash 循环，以相同参数和 state dir 原地重启 learner。**Modal 注意**：这需要 Modal island 执行的是同一段 run 脚本；若 Modal runner 自己拼 learner 命令，要在 runner 里套同样的循环。未在 Modal 上验证。
7. sky 0.13 私有镜像登录报 `asdict() should be called on dataclass instances`，**已定位，未修；主 agent 决定暂缓（不再用 Nebius）**。原因：launcher 在 Resources 中传入 `DockerLoginConfig` 对象；sky 0.13 客户端/服务端之间把 Task 序列化成 YAML 再读回时，`Resources.from_yaml_config` 直接 `config.pop('_docker_login_config')`，得到的是 dict，没有转回 dataclass；下一次 `to_yaml_config` 调用 `dataclasses.asdict(dict)` 就报错（`sky/resources.py:2654` 与 `:2755`）。可选方案：(a) 在 yeto 侧给 sky 打补丁，读回时把 dict 包成 DockerLoginConfig；(b) 改用 `SKYPILOT_DOCKER_*` 环境变量（`task.py:198`），但 0.13 会把它导出到所有 setup/run 进程，launcher 原本正是为此回避它；(c) 升级 sky 或向上游报告。待主 agent/用户决定。Nebius 不使用 spot，已知悉。
- 全量：68F/3046P/49S/26E，失败 id 94 个，与基线相同（`/tmp/infra-e1-b1.ids`）。

### INFRA-E1：就绪审计 7–13 项（2026-09-30）
- cell 接口对齐 FR1（032878d）：见上一节第 1 条的更正。
- 7（0f0bcf3）：A5 按两岛各 3 卡 T1R1S1 做端到端检查，elastic/declare-cells/quorum/margin/start-delay 各开关都能到达岛上的 learner。launcher 不支持按岛分别配置。
- 8（0f0bcf3）：`StrictRlBridge` 收到同一 step 的第二个 PULL（fixed roster 下 quorum 超时后的重发）时，向 learner 的 JSONL 磁带追加一条 `rl_pull_resend`（global_step、round_attempt、fragment_id、pulls_received）。兼容性：没有重发时磁带不变；仓库内没有任何磁带消费者会拒绝未知事件。Rust syncer 磁带未改：本机没有 cargo，无法编译测试；A5 判据取证位置请以 learner 磁带为准（主 agent 裁定）。
- 9（888d177）：`scripts/idle_flow_probe.py`，本机 listener 加 Modal CPU 客户端，在 60/180/350/600/900/1200/1800 s 各空闲点检查连接存活，输出 `idle_flow_timeout_s`。本地测试通过，未在 Modal 上运行。
- 10（9e73c9b）：`--rl-deterministic-trainer` 给 Miles argv 加 `--deterministic-mode`，并在 learner 与所有 Ray worker 上设 `NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`。SGLang 确定性推理沿用 `--sglang-deterministic-inference`（默认开）。默认不变。
- 11（8f316a0）：`rl_round_trained` 事件在 batch 带数据游标时（elastic 元数据开启）附带 `data_cursor`。
- 12（79b1e18）：`--rl-test-inject-rebuild-fail` 让 fork 的 `rebuild_training_models` 第一次在 `create_training_models` 阶段失败，走 fork 真实的 TrainerRebuildError 路径，结果为 REBUILD_OLD。**运行前更正** plan v2 §3 第 4 条：REBUILD_OLD 时 `generation` 仍为 1，因为只有重建成功才会 swap。
- 13：**未完成**。F-R1 尚未提交（miles-fr1 HEAD 仍为 5c1b49eb，工作区有 12 个未提交文件），pool_gpus 入口接线等它合入。
- 全量：68F/3055P/49S/26E，失败 id 94 个，与基线相同（`/tmp/infra-e1-b2.ids`）；validate strict 通过。

### INFRA-E1 第 14 项与两项裁定（2026-09-30）
- 14：新增 `--rl-print-attestation-fingerprint`（launcher → learner）。learner 在 `build_ports_launch` 与 `verify_ports_algorithm` 之后调用 `print_attestation_fingerprint`，用 `ports_runtime_fingerprint(launch)` 打印一行 JSON（runtime_fingerprint、learner_id、miles_argv）后返回。`run_ports_island` 用的是同一个函数和同一个 launch 对象；该开关只属于 learner，不进入被哈希的 Miles argv。CPU 上运行仍需要 Miles 镜像（parse_miles_args/run_plugin 检查）和模型快照下载，不需要 Ray 和 GPU。测试 `tests/test_rl_attestation_fingerprint.py`。
- 裁定记录：第 8 项以 learner 磁带中的 `rl_pull_resend` 为准；第 12 项的"运行前更正"已在 plan v2 §3 第 4 条原位标注，保留了原文。
- 全量：68F/3058P/49S/26E，失败 id 94 个，与基线相同（`/tmp/infra-e1-b3.ids`）；validate strict 通过。
### INFRA-E2 GPU harness 与注入（2026-09-30，仅 dry-run；未启动任何 GPU/云资源）
- 容器内 harness：`miles_adapter/e2_harness.py`。snapshot 根目录有 `yeto-rl-e2-harness.json` 时代替 `driver.run()`，依次执行：逐 rank 确定性读回、G-4.2 (a)–(g)、G-4.3 冻结 batch 的双 arm 逐位比较、L2。CPU 伪岛测试覆盖通过、RNG 丢失、确定性缺失、dropout 不符四种情形。
- 主机侧：`tools/probes/e2_cut_harness.py`，按 plan-v3 顺序生成 11 个 run 目录。`run.sh` 带 `YETO_E2_GPU_APPROVED` 守卫、watchdog、puller 与 rebuild 触发器；工具本身拒绝 `--execute`。本地两级检查为 launcher `--dry-run` 与 learner preflight（与容器内 learner 在 GPU 之前的检查相同，并翻译出 Miles argv）。
- 本地 dry-run 结果（`evidence/infra-e2/4.2-4.5/dry-run-20260930.md`）：11 个 run 的 launcher dry-run 全部 rc=0。C3 九个 run 的 preflight rc=0；C1/C2 的 preflight rc=1，原因是 ports 路径 `--lora-dropout` 固定为 0，而计划要求 0.05。这条阻塞已在本地暴露，判据未改。
- A7 注入开关（`miles_adapter/cut_injection.py`，默认关闭）：
  - save 途中 kill rank；
  - restore 途中 kill rank；
  - restore 前 sleep；
  - rebuild 在 fork 的 `create_training_models` 处失败；
  - rebuild 前写入偏移的数据集游标文件；
  - 第 6 项（CAS 前后 kill controller）使用 E1 的 `--rl-test-kill-learner-at` 加 restart loop，在同一容器内重启。
- plan-v3：镜像 17d428a2…（5c1b49e-9f29303）、Miles 5c1b49eb、Qwen3-0.6B c1899de…、Qwen3-1.7B 70d244cc…；判据沿用 plan-v2 原文。
- 阻塞：① LoRA dropout 0.05 无法表达（需要 config 翻译开关，或由主 agent 裁定）；② G-4.5 第 5 行需要 E1 提供实时 `data_cursor`；③ E1 需合入补丁 `infra-e2-e1-harness-injections-v1.patch`。
- 测试：全量 68 failed / 26 errors / 3059 passed。失败 id（94 个）与 integ-decl b90c18d 基线（3037 passed）按 id 相同。

### INFRA-E1：主 agent 队列 ①–⑤（2026-09-30）
- ① c51735e：`--rl-elastic-trainer-edges`。只有开启它时才去掉 `--balance-data`，其余 argv 逐字节不变（有 e2e 测试）。
- ② 0860a14：应用 INFRA-E2 harness/注入补丁 v1（先合并 origin/infra-e2 ad26f2d）；rebuild-fail 只保留 E2 的一份实现，rank 0 不再被当作未设置。
- ③ b2a8e3c：`--rl-lora-dropout`，默认值 0 时 argv 不变；canonical/导出配置仍是 dropout 0；E3 的 DP 边照旧拒绝 dropout>0。
- ④ d8245c5：`live_data_cursor()` 实时读 executor 的 data_source；`data_cursor()` 优先用实时值，读不到时退回上一 batch 的缓存。
- ⑤（本节提交）F-R1 接线：
  - `--rl-elastic-trainer-edges` 时，从 manifest 的 `resources.gpus`（按逻辑 bundle 顺序）得到 `pool_gpus`，并接线 trainer 边；其他 elastic 路径不传 pool_gpus。
  - `member_gpus` 改用 fork `describe_cells` 的 bundles。
  - 新增 `unbind_members`。
  - `trainer_view` 优先用公开的 `slice_pg_info`，私有名留作兜底。
  - 按 FR1 的调用顺序核对：正向 `bind_members`→`add_engines`（start_cells/wait_cells_tracked）→`publish_members`（cordoned 发布→check_weights→admit_cells）一致。反向缺 `unbind_cell`，给 E3 出了补丁 `infra-drafts/patches/infra-e1-e3-unbind-after-stop.patch`：stop 之后调用 `unbind_members`，REBUILD_OLD 时先把原 GPU 绑回再 start。已在本地套用验证，E3 测试 20 个通过，**未提交**（trainer_transition.py 归 E3）。
  - F-R1 的绑定只在内存，对 E1-D ⑤⑥⑦ 与 A9 f5 的影响写入 plan v2 §7，需主 agent 在运行前从 (a)/(b)/(c) 中选定。
- 全量：68F/3091P/49S/26E，失败 id 94 个，与基线相同（`/tmp/infra-e1-b4.ids`）；validate strict 通过。

### INFRA-E2 plan-v4 与镜像内 CPU preflight（2026-09-30）
- 合并 origin/integ-decl 9d2029d 与 origin/infra-e1（含 d8245c5 `live_data_cursor`、`--rl-lora-dropout`，以及已合入的 E2 注入/harness 补丁）。
- plan-v4：镜像 db815884（2f23a0f-9f29303），Miles 2f23a0fc；判据文字不变。工具的 pin 校验改为对应 plan-v4；C1/C2 使用 `--rl-lora-dropout 0.05`；G-4.5 第 5 行已解除阻塞，但要求实时游标可读，否则拒绝。
- **pin 缺陷（上报）**：integ-decl 9d2029d 的 `MILES_NEXT_IMAGE` digest 仍是 17d428a2，只改了 commit 和注释。工具在真实检出上 rc=3。在模拟修正后的检出上，11 个 run 共 22 项本地检查全部 rc=0（`dry-run-v4-20260930.md`）。
- 镜像内 CPU preflight（B2 批准，app ap-AOEKeEOxdqbPDpGJNSPQyD，≤$0.02，已 stopped）：learner 在 import transformer_engine 时因缺 libcuda 失败，没有得到 Bridge/Miles parse 的结论；runtime manifest 的 commits 与当时的 pin 一致。需要 GPU 容器（例如 T4），待批准。证据：`preflight-cpu-20260930/`。
- 测试：全量 68 failed / 26 errors / 3092 passed，失败 id（94 个）与 integ-decl 9d2029d 基线（3058 passed）一致。
- 裁定（2026-09-30）：F-R1 绑定只在内存对 E1-D ⑤⑥⑦ 的影响按 (c) 处理，调整用例安排、原判据不变，写入 plan v2 §7.1；已提交配置≠启动配置时重启 → RECOVERY_REQUIRED 记为已知限制，不作为本轮判据；A9 f5 与 E3 plan-v3 一致。
## GPU-B1 第 1 批 GPU 验收执行（2026-09-30）
- 分支/worktree：`gpu-b1` @ /home/michael/work/gpu-b1（基于 a303cbb，运行中先后合并 integ-decl 15d88bd、a5123ca → 724fc7b）；已普通推送。计划：`gpu-plan-v2.md` §9（判据运行前提交，未修改）。证据：`evidence/infra-v2-b1/`（RESULT.md 逐项）。台账：`infra-drafts/gpu-spend.md`。
- 结果：Nebius 路径冒烟**不通**（launcher→sky 0.13 客户端 `asdict()` 报错，未开通 VM，退回 Modal）；F0 **通过**（门）；F-E1 暴露**代码缺陷**：`--rl-elastic-cells` 未传给 fork，fork 只声明已启动的 `inference-engine-all-0-0-00000`，up 事务 `start_cells(['c0'])` KeyError → REBUILT_OLD。据此停止本批其余 GPU 运行。
- task 状态：3.3、3.4、3.5、3.7 未完成（等待代码；3.7 ⑤⑥ 环境阻塞）；2.3（A2）未完成（等待代码：launcher 不转发 eval temperature，贪心 eval 无法设置）；1.7（A2+）未完成（等待代码：observe 无 launcher 入口、无工具负载）。本批未勾选任何 task。
- 费用：≤$2.91（f0 $0.42、fe1fp $1.56、fe1 $0.93、nsmoke $0），全部 Modal L40S。本批预算 $127（v2 8 卡档 $170 按 2.95/3.95 折算）。资源：Modal app ap-4WHOoo6jjpVNP3CkJ8DT1p / ap-M16C4QJ9WiiU1KPdPyWua2 / ap-F4XpsACPevWqjbkXVjcuyC 均 stopped/0 tasks；sky 无集群、nebius 无实例；本地进程已清理。
- 其他发现：Nebius H100 SkyPilot 目录价 $3.85/GPU·h（非 v2 所写 $2.95），与 Modal H100! $3.95 基本持平，且 Nebius 只有 1/8 卡规格；attestation 指纹无离线计算入口，需先跑同参数运行取 `rl_driver_start.runtime_fingerprint`。
- 需要的代码修复（交代码负责人）：(1) F-R1 fork 声明停止 cell + yeto 把声明 cell id 传给 fork 并与在役成员名统一；(2) launcher 转发 `--eval-temperature`（A2）；(3) observe 开关与工具负载/tool_wait_board 接线（A2+、A4b）；(4) E1-B 权重覆盖注入、E1-D ③④ stop_cells 半失败注入、⑦ fork 重启入口；(5) 可选：sky 0.13 私有镜像登录序列化、Nebius 不用 spot。
- 下一步：以上 (1) 合入并重建镜像后，先重跑 F-E1（≈$1–3），通过再按 §9.3/§9.6 跑 E1-A 基线、E1-A、watchdog 用例（Modal H100!:8，最坏合计 ≈$95）。
