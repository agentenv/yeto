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
