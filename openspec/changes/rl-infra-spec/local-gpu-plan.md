# rl-infra-spec 本地 GPU 验证计划（待自有卡；判据事先固定）

按用户 2026-09-30 决定，云上 GPU 验证暂停。本文件汇总"待本地 GPU 验证"项的计划，**判据在运行前已提交，运行后不得修改**；事后说明只能追加在各节末尾"追加"小节，并注明"判定条件未改"。各 agent 只写自己负责的节，主 agent 合并。

## 通用口径（所有节适用）

- 镜像：运行当日集成分支 `MILES_NEXT_IMAGE` 的 digest（记录到证据目录）；先生成 1.1 runtime manifest 并与 pin 比对，不一致不跑。
- GPU：同一次实验内各 arm 用同型号卡；启动时记录 `nvidia-smi --query-gpu=name,uuid`。需要逐位比较的项在同一台机器同一次运行内跑两个 arm。
- 模型/算法：Qwen3-0.6B LoRA（TP=PP=CP=EP=1），strict-avg 外层（单岛可用 none），默认 GRPO AlgorithmSpec；`--rl-expected-algorithm-sha256` 由启动方给出。
- 磁带：启用 `yeto-rl-echo-events`（stdout 全量采集）并 `--rl-stall-timeout`；launcher 退出码 4/6 判失败。
- 同一失败只在查明原因并修复后重跑；不挑 seed；每次运行（含失败）都保留证据目录 `evidence/local/<task>/<attempt>/`。
- 本地卡无按时计费，但仍记录 wall time 与 GPU·h（整池×时长，含备用卡），以便与云上数据同口径比较。

## INFRA-A 负责部分

### L-2.3 age 0 合法训推重叠：eval‖train/outer_sync（验收 X9；同时是 1.4 的 X9）

前提：`infra-drafts/patches/infra-a-driver-2.3-eval-overlap.patch` 已合入集成分支；learner CLI 暴露 `yeto_rl_overlap_eval`（映射到 `miles_args.yeto_rl_overlap_eval`，归主 agent/launcher 负责人）。

配置：2 卡，T1R1（fixed-partition），`eval_interval=1`，一个小 eval 集（≥16 prompt，确保 eval 时长与 train 同量级），3 轮，seed 固定为 17，`eval_uses_snapshots=False`。
- arm S：partitioned-serial（不开 overlap）。
- arm O：partitioned-overlap（`yeto_rl_overlap_eval=1`）。
- arm OD：同 arm O，并注入 `publish_delay_s=30`（`YETO_RL_FAULT_INJECTION`）。
三个 arm 在同一台机器上依次运行；预计总时长 < 1 h。

判据（全部满足才算通过；任一不满足即判未通过并记录原因）：
1. 三个 arm 均正常结束（退出码 0），无 `rl_strict_failure`、无 `OverlapGuardError`。
2. S 与 O 的逐轮 `trained_sample_ids_sha256`、trained_groups/samples 相同；每轮 `applied_lrs` 长度 = 1。
3. O 与 OD：每个 `rl_eval(overlapped=true)` 的 `rl/policy_token` 等于该 eval 开始时最近一次 `rl_publication` 的 token；每个 `rl_eval_overlap_start` 与其 `rl_eval` 之间磁带上没有 `rl_publication` 和 `phase=generate`。
4. OD：4 次发布前都有 `rl_fault_injected`；所有 `phase=generate` 的 `policy_version` 等于其前最近一次发布版本（延迟发布不触发旧版本生成）；overlapped eval 同时在飞 ≤ 1（`rl_eval_overlap_start` 与 `rl_eval` 严格交替）。
5. eval 固定用贪心解码（temperature=0，eval 集 N≥16 条）。硬条件：S 与 O 的每个 eval 点的 `policy_version` 与 `rl/policy_token` 逐项相同。分数条件：同一 policy_version 上 S 与 O 的 eval 平均分之差 ≤ 2/N（最多 2 条 prompt 结果不同；用于容纳 SGLang 在不同批组成下的贪心数值不确定性），超出即不通过。
6. 观测（`observe=True`）：O 中至少 2 个 eval 点满足"真实 eval 区间 ∩ 同轮 train span 的长度 > 0"。真实 eval 区间取 `rl_timeline_span(task=eval)`，它由 `LoopEvalHandle` 在 eval 协程第一条语句与返回前（finally）用 driver 时钟记录，**不是** generate 结束到 join 的调度窗口。**若判据 1–5 通过而判据 6 不满足**（例如 trainer 调用在事件循环外阻塞，eval 未与训练交错推进），则 2.3 交付为"guard 正确，但当前运行时下无实际重叠收益"，按合法否定结论处理（记录 partitioned-serial 结论与后续项），不宣称重叠已生效。

通过后：2.3 勾选；1.4 的 X9 由本实验与已完成的 partitioned-serial X9 guard（第三轮 C，95203615）共同满足，1.4 依赖 1.2 已勾，可勾选。

### L-1.7 观测 GPU 验证（区分工具等待与 GPU 饱和；资源峰值；关闭观测兼容）

配置：2 卡 T1R1，partitioned-serial，`observe=True`，3 轮，两种负载各一次：
- W-tool：带工具调用的 rollout 函数，工具固定 sleep 5 s（确定性），保证存在 active=0、tool-wait>0 的时段；
- W-gen：无工具、长生成（max_new_tokens 调大），保证 engine 饱和。
另跑一次 W-gen `observe=False` 作为兼容对照。

判据：
1. W-tool 的采样中存在 `classify_load == "tool-wait"` 的样本，且这些样本的 router in-flight（M3 计数）= 0；W-gen 存在 `rollout-saturated` 或 `rollout-busy` 样本，且没有任何 `tool-wait` 样本。
2. 每轮 `rl_timeline_span` 的 `summarize` 满足 `wall_s ≤ Σspan`，partitioned-serial 下 `overlap_s == 0`（±1 ms）。
3. `rl_round_labels` 每轮都有 profile_hash、config_epoch、weight_transport 标签，与 `rl_driver_start` 一致。
4. 资源峰值：每轮记录 trainer/rollout GPU 的 `memory.used` 峰值、主存 RSS 峰值（采样周期 ≤ 1 s），写入证据目录。
5. `observe=False` 的磁带去掉时间戳/耗时字段后，与 CPU 录制磁带的事件种类与顺序一致（无新增事件）。

通过后：1.7 勾选（依赖 1.4 须先勾选）。

### L-5.1 重配置成本分布与瓶颈选择（依赖 3.8；trainer 边另依赖 4.5/4.8）

只在 E1（3.4–3.8）实现并在本地通过之后执行；否则本节不启动。
配置：4 卡池，rollout 边 T2R1S1↔T2R2S0（及 3.8 实际验收的边），每个方向 ≥ 3 次切换（`MIN_SAMPLES_PER_EDGE=3`），每次切换记录 `timeline.TRANSITION_PHASES` 各阶段耗时（wait_safe_point/drain/export/init/restore/publish/first_step）与 `background_restore`，以及资源峰值。
判据与选择规则（已实现为 `timeline.select_bottleneck`，提交 d3629d3，运行后不改）：
1. 每条边样本数 ≥ 3，否则结论为 insufficient，补测而不是放宽。
2. 对每条边取各阶段 p50 占该边 p50 阻塞总时长的比例，边间等权平均；比例最大的阶段即"首先优化的瓶颈"。平局时不自动选择，由人记录选择与理由。
3. 输出每个 source→target 的 p50/p90/max 分布（`transition_cost_distribution`），原始样本入证据目录。
4. 若所有边的 p50 阻塞总时长 < 该边一轮训练中位时长的 10%，结论为"基线无需优化"（5.7 的另一依赖路径），5.2–5.5 记"未选中"。

### L-2.3 追加：运行前修正（2026-09-30，尚未运行任何 GPU；判定条件在运行前修改，提交记录可查）

- 判据 6 原稿用 driver 发射的 eval span（generate 结束→join），该区间恒包住 train，判据无法证伪，也会让 1.7 计费虚高（独立审查 H1）。已改为 eval 协程内部记录的真实区间与 train span 的交集，并同步修改代码（`overlap.LoopEvalHandle`）。
- 判据 5 原稿允许运行前在两种口径中二选一；现已定死为上文口径（审查 L4）。
- 已知差异（审查 L3），不作为不通过理由，但须在结果中如实记录：(a) O 中 eval(v_r) 挪到 generate(r) 之后执行，SGLang 引擎内的采样 RNG 消耗顺序与 S 不同；eval 为贪心不受影响，但**训练 rollout** 若使用引擎内 RNG，第 r+1 轮起 generate 前的 RNG 状态可能与 S 不同，判据 2（sample-id 哈希）只比较样本身份，不比较生成文本；若判据 2 失败而原因为 RNG 顺序，按"未通过"记录并另行分析，不改判据。(b) 若某轮在 train/outer_sync 中失败，O 会取消在飞的 eval(v_r)（`rl_eval_overlap_aborted`），该点的 eval 结果丢失；S 中 eval(v_r) 在 generate(r) 之前已完成，不会丢失。
