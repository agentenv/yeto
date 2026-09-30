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

### L-2.3 追加：A2 判据 5 未通过后的代码修复（2026-09-30 INFRA-A；判定条件未改）

- A2 结果（gpu-b1 `evidence/infra-v2-b1/a2/RESULT.md`）：判据 1/2/3/4/6 通过，判据 5 硬条件不通过（v1 起 S 与 O 的 policy token 不同）。判据不改；按"查明原因并修复后才重跑"处理。
- 原因分析：eval 为 temperature 0，SGLang 将其规范化为 top_k=1，采样走 argmax（`sampling_params.py:217-220`、`sampler.py:184`），**不消耗**引擎采样 RNG。因此 L3(a) 中登记的"RNG 消耗顺序"不是主要机制。更符合证据的解释是引擎状态：发布时 Miles 在暂停引擎后 `flush_cache`（`weight_update/session.py pause_engines`，`--pause-generation-mode` 不是 in_place 时），S 在发布之后、generate 之前跑 eval，eval prompt 留在 radix cache 与 KV 分配器中，训练 generate 的前缀命中与 KV 布局随之改变，数值不同，采样分叉；O 与无 eval 的 B 在 generate 前都是刚 flush 的状态，所以 token 相同（与旁证一致）。该解释在 CPU 上无法证实，只能由重跑检验。
- 修复（代码，已提交）：配置了 eval 时，每次训练 generate 之前对所有 rollout engine 调 SGLang `/flush_cache`（URL 取自 fork-M3 router `/worker_inflight`；忙时重试，最终失败即中止运行，不在未隔离的 cache 上生成）。S 与 O 的训练 generate 因此都从同一个刚 flush 的引擎状态开始，与 eval 的有无和时序无关。stdout 打印 `YETO_RL_EVAL_CACHE_FLUSH {"rollout_id", "engines"}` 作为证据。overlap 另外要求 eval 为贪心（temperature 0，缺省则取 rollout temperature 并拒绝），因为带采样的 eval 会消耗引擎 RNG，而本修复不隔离 RNG。
- 对默认路径的影响：不配置 eval 的运行完全不变（不 flush）。配置了 eval 的 serial 运行每轮 generate 前多一次 flush：prefix cache 只影响性能，不改变采样分布；但生成文本的具体数值会与修复前的运行不同，所以修复前运行的 token 不能与修复后的运行逐位比较。既有判据都是同一批次内各 arm 之间的比较，不受影响。
- 重跑 A2 的前提：同一计划、同一判据、三个 arm 用修复后的同一 SHA。已知残余风险（事先登记）：多 engine 时 router 的负载均衡状态也会受 eval 请求影响，A2 为单 engine（T1R1），不涉及。

### L-2.3 追加：更正上一节的原因分析；运行前补充（2026-09-30 INFRA-A；判定条件未改）

- **更正**：上一节（"A2 判据 5 未通过后的代码修复"）把原因归于 radix cache / KV 分配器残留，这是**错误**的；该节原文保留。更早登记的 L3(a)（"eval 改变 SGLang 采样 RNG 消耗顺序"）同样不成立。依据（独立审查 C1，证据见 gpu-b1 `evidence/infra-v2-b1/a2/{S,O}/launch.log.gz` 与 `evidence/infra-a/2.2-2.3/round3-r3b/`、`round4-r4b/`）：
  - S、O、r3b、r4b 四次运行第 0 轮的 `rollout/rollout_log_probs`、`rollout/log_probs`、`response_len` 逐位相同。S 在 generate(0) 之前跑过 eval，生成结果却与 O 一样，说明 eval 没有改变训练生成。这与 ports 路径默认开启 `--sglang-enable-deterministic-inference`（每请求固定采样种子）一致。
  - 分叉只出现在训练步：ppo_kl、train_rollout_logprob_abs_diff、grad_norm（S 0.4499274790，O 0.4496529400）不同，trainer 前向本身就已不同。
  - 按 v1 token，这些运行分成两组，且与有无 eval 无关：3660f9f 组为 S、r3a、r3b、r4a；e5362df 组为 O、OD、r4b。r3b 与 r4b 都是无 eval 的 partitioned 运行，却落在不同组。
  - 结论：trainer 侧逐次运行不确定。A2 所用的 11911b8 没有 `--rl-deterministic-trainer`。
- **处理**：flush 机制（ffbbc79）已撤销。它还会让默认 router（没有 `/worker_inflight`）下配置了 eval 的运行全部中止（审查 H1）。overlap 的"贪心 eval"限制一并撤回。
- **运行前补充（配置，不改判据）**：重跑 A2 时三个 arm 都加 `--rl-deterministic-trainer`，即 Megatron `--deterministic-mode` 加上 `NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`，并新增 `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`。
  - 新增最后一项的原因：Megatron 0.19 的 `apply_determinism_to_args` 只在校验参数的那个进程里 setdefault 这个变量，而 Transformer Engine 在各 trainer rank（Ray worker）里读取它，缺省为 1。现在由 `entry.DETERMINISM_ENV` 经 `connect_island_ray` 下发到所有 Ray worker。
  - 注意：镜像内的 Megatron 版本未在本地核对；本地参照的是 `/tmp/review-miles-venv` 中的 megatron-core 0.19.2。
- **排查项**：
  - LoRA dropout：A2 没有传 `--rl-lora-dropout`，缺省 0.0（`run_config._lora_dropout`），不产生 dropout RNG。
  - TE / flash-attn：Megatron 注释称 TE 的 FlashAttention 在受支持配置下是确定性的；TE DotProductAttention 在 deterministic_mode 下若 `NVTE_ALLOW_NONDETERMINISTIC_ALGO` 不为 0 会直接报错。本次补上该变量后，不会出现"开关打开但 TE 静默走非确定算法"的情况。
  - `torch.use_deterministic_algorithms(True)` 同样只在校验参数的进程里调用；各 rank 是否生效未验证，由下面的确认实验检验。
  - 其他未排除的来源：cuBLAS 算法选择随机器而异（不同物理 H100 或驱动）。

### L-D0 trainer 确定性确认实验（不计入任何 task；待主 agent 批准，未上卡）

- 目的：确认 `--rl-deterministic-trainer` 能让相同配置的两次运行逐位一致，这是重跑 A2 的前提。
- 代码：本节所在提交（infra-a），镜像为运行当日 integ-decl 的 `MILES_NEXT_IMAGE` digest（记录到证据目录）。
- 配置：Modal `H100!:2`（`--modal-gpu-exact`，运行前断言 `nvidia-smi` 为 H100 80GB HBM3），T1R1 fixed-partition，Qwen3-0.6B LoRA r16，gsm8k（与 A2 相同的 model/data revision），`--total-steps 3 --seed 17`，strict-avg 单岛加本机 head（同 A2），不配置 eval，`--rl-deterministic-trainer`，`--rl-observe-timeline`。两次运行 D1、D2 参数完全相同，只有前缀不同（`infra-a-d0-{1,2}-<UTC 日期>`），依次运行。
- 记录：每次运行的 GPU UUID、驱动版本、镜像 digest；launch.log 中确定性变量在各 rank 的实际值（`NVTE_ALLOW_NONDETERMINISTIC_ALGO` 等），缺失即记为"环境未确认"。
- 判据（事先固定）：
  1. D1、D2 均 rc=0，无 `rl_strict_failure`。
  2. D1 与 D2 的 `rl_publication` 中 v1/v2/v3 的 `rl/policy_token` 逐位相同。
  3. 辅助记录（不作判定）：逐轮 grad_norm、ppo_kl、train_rollout_logprob_abs_diff 是否逐位相同。
- 结论规则：判据 1、2 都满足，则确定性开关可用，按原判据重跑 A2（三个 arm 都加该开关）。判据 2 不满足，则记为"当前确定性开关不足以让 trainer 逐次运行一致"；不重跑 A2，下一步按辅助记录定位（前向或反向）后另行计划。不对同一失败重复启动。
- 费用：预计每次约 15 min（含镜像拉取与预热）× 2×H100! × $3.95/GPU·h ≈ $1.98，两次约 $3.95。硬超时为外层 30 min，另加独立 watchdog 35 min 执行 `modal app stop`；最坏 2×2×35/60×3.95 ≈ **$9.2**。
- 回收：结束后拉取磁带与 launch.log，`modal app stop -y` 并用 `modal app list` 核实 0 tasks。

### L-1.7 追加：运行前补充（2026-09-30 INFRA-A；判定条件未改）

- A2+ 各次运行必须带 `--rl-elastic`（及其必需参数 `--rl-elastic-resources`/`--rl-elastic-initial-config`；run_config 只在该开关下设置 `use_miles_router`）。原因：`rl_load_sample` 的全部字段都依赖 fork-M3 Miles router 的 `/worker_inflight`，默认的 Rust sglang_router 没有这个端点，此时 `load_sample()` 返回 None，不会发任何样本事件。W-tool 还需 `--custom-generate-function-path yeto.rl.tool_wait_workload.generate --rl-test-tool-delay-s 5`，工具等待才会计入 ToolWaitBoard。
- 字段来源：
  - `running_requests`、`queued_requests`：SGLang `/get_load`；
  - `engine_capacity`：SGLang `/server_info` 的 `effective_max_running_requests_per_dp`；
  - `tool_wait_trajectories`：ToolWaitBoard，stock generate 记 0，其他自定义 generate 记 None；
  - `load_class`：`classify_load` 的结果，任一输入未知则为 `"unknown"`；
  - `ready_groups`：rollout 中途不可观测，记 None。
- 已知缺口（审查 L1 同类问题，本节不修）：`/get_load` 每个 DP rank 一项，代码已对所有 rank 求和；`engine_capacity` 在 `internal_states` 缺失时只能退回单个 `max_running_requests`。
