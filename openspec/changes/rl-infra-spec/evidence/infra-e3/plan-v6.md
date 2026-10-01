# E3 待验证计划 v6：A8（4.6 X4）与 A9（4.7）（INFRA-E3，2026-09-30；取代 plan-v5.md；运行前提交，判据与容差不变）

与 v5 的差别（独立审查"小修后可合入"的 M-A、L-1）：
- **pin 更新**：Miles `e3a11ab38cbb7fd911b23fdd62a4eb6dfbb1c841`（含 fork-M5 lazy-state 修复），镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:2cc5cc52de2444e59ddefba4f9546d1aaa13f9807ab441f56e2c70a7e7936eff`（tag e3a11ab-9f29303）；同时合入 yeto 侧规避 `cut_plugin.side_effect_free_state`（读取优化器状态不创建空条目，否则 exp_avg/exp_avg_sq 会静默不恢复，E2 已在 H100 上实证）。harness 与测试一律从 `yeto/rl/__init__.py` 读取 pin，不再写死；容器内仍断言 `/root/miles` 的提交等于该 pin。旧 pin（5c1b49eb、2f23a0fc）的镜像**不得**用于 A8。
- **重分片恢复自检加强**：恢复后重新导出的每个参数的状态键（param、exp_avg、exp_avg_sq、step）必须与 cut 完全一致，缺任何一个即拒绝（此前只比较导出里存在的键）。
- **批次守卫复位**：`restore_cut`（同形）与 `rebind_args` 回到非目标布局时清除 DP 变化批次守卫（回滚后不再误拒训练）。
- 其余（数据兜底、G1 口径、G1–G6 判据、容差、go/no-go、费用上限 $15.8）同 v5。

以下保留 v5 全文。

---

# E3 待验证计划 v5：A8（4.6 X4）与 A9（4.7）（INFRA-E3，2026-09-30；取代 plan-v4.md；运行前提交，判据与容差不放宽）

与 v4 的差别：

1. **A8 数据兜底**（DEV-GATHER 第 7 次 compare 出错且状态文件未取回的教训）：
   - 容器退出（成功或失败）前运行 `pack_states.py`：把每个 (arm, tag) 的各 rank 状态用 fork-M5 合并成一个汇总文件 `packed/<arm>_<tag>.pt`（adapter、FP32 主参数、exp_avg、exp_avg_sq、step、hyper、scheduler、Megatron 计数与 weight_version、各 rank RNG 摘要；`s8` 只留 FP32 主参数以控制大小，按 LoRA r16 约 1000 万参数估计总计约 1.8 GB），`packed/index.json` 记录每个文件的字节数、sha256 与逐字段摘要；
   - 容器输出 `=== PACKED READY ===` 后最多等待 20 分钟，本地 `modal_run` 经 Sandbox 文件接口逐块拷出并校验 sha256，写回释放标记后容器才打包事件证据并退出（等待期间 H100 最多多花约 $2.6，仍在 $15.8 上限内）；
   - 本地离线重跑入口：`python compare.py <取回目录> --offline`，只用事件与 packed 文件即可算出 G1–G6；缺状态文件时只报告能算的部分，判定为 `incomplete`，不给 go/no-go。已用 DEV-GATHER 第 7 次取回的事件演练（`dev-gather-run7/offline_drill_RESULT.json`：G2、G3（事件部分）、G4（loss/grad_norm 部分）、G5、G6（loss 部分）可算，G1 与状态比较部分如实为 unavailable）。
2. **G1 比较口径写明（不放宽）**：G1 只比较"恢复后的 trainer 汇总状态"与"同一个 cut 的源状态"——B1、B1p 对 A1 的 `s2`（C1），B2 对 A2 的 `s2`（C2），RT 对 A1 的 `s2`（C1→C1′→DP1 往返）；逐位比较的字段为 adapter、FP32 主参数、exp_avg、exp_avg_sq、每个参数的 step、scheduler、Megatron 计数与 weight_version（v3 已列出计数，compare 之前漏比，本版补上）；hyper 与 shape 同样记录并报告差异。**不同 cut 之间（C1 与 C2）从不比较。**
3. **"A1/A2 步 3 grad_norm 相同但 C1/C2 摘要不同"的静态结论**：C1 与 C2 来自两条不同的训练（A1 为 DP1、A2 为 DP2），DP 不同导致梯度归约顺序不同，FP32 主参数与动量可以在末位不同；前向用的是 bf16 模型副本，末位差异在转 bf16 后通常消失，因此逐样本 loss 与 grad_norm 可以逐位相同而 FP32 状态摘要不同。这与 G1 的口径不冲突（G1 不跨 cut 比较），G4 也只比较同一 cut 出发的两条 arm（A1 对 B1、A2 对 B2），不会因此误判。A8 的 `index.json` 逐字段摘要会给出 C1 与 C2 究竟在哪些字段不同（仅作记录，不是判据）。
4. **生产改动单列**：`trainer_rebuild.resized_args` 同步设置 `world_size`（已有单独测试）；E2 f898516（cut 携带 `weight_version`）合入集成分支后，E3 补"重建后重发版本连续"测试。

以下保留 v4 全文。

---

# E3 待验证计划 v4：A8（4.6 X4）与 A9（4.7）（INFRA-E3，2026-09-30；取代 plan-v3.md；判据与容差不变，运行前提交）

与 v3 的差别（全部是执行配置与前提，**G1–G6、A9 判据、容差、go/no-go 规则均不变**）：
- **镜像/pin**：Miles `2f23a0fc`（含 F-R1，训练路径未变），镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:db815884…0cbf`（tag 2f23a0f-9f29303）；harness 从 `yeto/rl/__init__.py` 读取 pin，不再写死。
- **A8 配置 = 生产 trainer 边配置**：harness 的 Miles argv 走生产翻译的 trainer 边分支（`RLRunConfig.trainer_dp_edges=True`，即 `--rl-elastic-trainer-edges` 所设字段），因此不含 `--balance-data`，不再手动覆盖 `balance_data`；A8 的确定性来自 learner 开关 `--rl-deterministic-trainer`（Megatron `--deterministic-mode` 与 NCCL/cuBLAS/TF32 环境），本地 dry-run 检查 argv 含 `--deterministic-mode`；LoRA dropout 用 `--rl-lora-dropout`（默认 0）；Megatron hidden/attention dropout 没有 yeto 开关，仍在 parse 后置 0 并记录。
- **harness 进度看门狗**（DEV-GATHER 第 3 次在生成阶段卡住约 58 分钟的教训）：每个阶段在容器内单独监控，`progress.log` 20 分钟无新进展或该阶段日志中 5xx/`request failed with server error` 超过 200 行即杀掉该阶段并失败；任何退出路径都打包证据；本地实时镜像容器输出到 `container.log`，25 分钟无输出即终止 Sandbox；`modal_run` 在任何退出路径 stop app。
- **生成冻结数据**按上游 `train.py` 的启动顺序：建 rollout 组件 → 建 DP=1 trainer → `update_weights` → 需要时 `onload_kv` → 每个 rollout `prepare_rollout` + `get`（不训练，基座策略）。第 3 次失败原因是旧实现 parse 后置 `debug_rollout_only` 且没有推权重/载入 KV，共置的引擎对 /generate 返回 400/503。
- **A9 拓扑核对（F-R1 限制：只有启动时声明为停止的 cell 可以解绑）**：A9 以 T2R2 起步，启动参数加 `--rl-elastic --rl-elastic-trainer-edges --rl-elastic-declare-cells --rl-elastic-cells <含 g1 上的备用 cell>`；T2R2→T1R3 时新 engine 用这个声明为停止的 cell 绑到 g1（`bind_members`），T1R3→T2R2 时摘除的正是它（按 GPU 选择，停后 `unbind_members` 释放 bundle 给 trainer，回退时先绑回 g1 再启动），因此让出的总是可解绑 cell。所有故障注入都从 T2R2 或经 T2R2→T1R3 到达的 T1R3 开始，**不从启动即为 T1R3 的配置开始**（那样 g1 上的 cell 启动即运行，不能解绑）。
- **A9 f5**（trainer_cut 之后、COMMITTED 之前 kill learner）：F-R1 的绑定只在内存中，重启后按设计判 RECOVERY_REQUIRED 并写 `trainer_recovery_hint`（restore_old，含 cut_epoch）；与 v3 §3.1 的期望一致，不自动恢复训练。

以下保留 v3 全文。

---

# E3 待验证计划 v3：A8（4.6 X4）与 A9（4.7）（INFRA-E3，2026-09-30；取代 plan-v2.md；判据运行前固定，事后不改）

与 v2 的差别（复审，A8 部分判据不变，复审结论"A8 可按 plan-v2 执行"仍成立）：
- A9 判据新增：trainer→rollout 方向新 engine 所在 GPU 必须等于 moved GPU（§3.2）。
- 生产守卫：DP 变化后每批在训练前经 `reshard.batch_guard_problems` 检查（各 rank 公布完整 `train_parallel_config` 且 dp 等于计划、批次被两侧 scheduled 路径接受），否则拒绝；`--indep-dp`、`--multimodal-keys` 写入前拒绝。A8 的 arm 若触发该守卫即判环境阻塞（说明未走 scheduled 分支）。
- controller 补丁改为 v3（提交 CAS 失败时读回 epochs 判断是否已提交）。

以下保留 v2 全文，仅在上述位置修改。

---



与 v1 的差别（独立审查"需修复"）：
- H1：生产路径是 rollout 侧调度 `split_train_data_by_dp_scheduled_raw`/`dp_schedule.build_dp_schedule`（GBS 按 rollout 计、先打包再按 micro batch k→rank k%dp 分派、loss 以 `num_rollouts` 归一），不是 round-robin。A8 的每个 arm 改走 `split_train_data_by_dp` 的真实分派；G2、G4 按此重写；拒绝 `--balance-data`/`--balance-by-flops`/部分步/vpp>1。
- M1：profile 固定 `lora_dropout=hidden_dropout=attention_dropout=0`（Megatron 后两者默认 0.1），代码在 DP 变化时拒绝任一非 0 或未知。
- M2：CPU 只复算 fork `loss_function` 缩放的算术，归一化证据由 A8 G2 提供。
- L4：G3 第二次仍不可判定 → 判 no-go。
- F-R1 的理由已从源码核实（§4）；补丁改为 v2。

状态：**未运行**。本文件只是计划；没有启动任何 GPU 或云资源。先把本文件单独提交，之后才允许起卡。
预算约束（主 agent 转达的用户要求，2026-09-30）：A1–A9 的 GPU 总成本 ≤ $300；本文件内 A8 目标 ≤ ~$20，A9 ≤ ~$35，DEV-GATHER 尽量不用 H100。
对齐：`gpu-plan.md`（A8/A9/DEV-GATHER 行）与 PLAN-V2（`/home/michael/work/gpu-plan-v2`，本文件不改它的内容）。与 gpu-plan.md 相比本计划**缩小了规模**：A8 从 4×H100·7 h 缩到 2×H100!·≤2 h；A9 从 8×H100 Nebius P62↔P44 改为 4×L40S Modal T2R2↔T1R3（tasks 4.7 允许"更小等价边"）。这一缩减需要主 agent 在 PLAN-V2 里采纳。

## 0. 通用前提（固定）

- 代码：yeto `infra-e3`（运行时 SHA 写入 RESULT），加上 `infra-drafts/patches/infra-e3-controller-v3.patch`、`infra-e3-elastic-wiring.patch`（A9 需要；A8 不需要）。Miles fork `yeto/ports`=`5c1b49eb`；镜像须含该提交（IMG 负责重建，运行前生成 1.1 runtime manifest 并核对 digest）。不满足则不运行。
- 模型与 profile P-E3：Qwen3-0.6B，LoRA r=16、alpha=32、**lora_dropout=0、`--hidden-dropout 0`、`--attention-dropout 0`**，bf16，DistributedOptimizer（Miles 对 Adam 的默认），TP=PP=CP=EP=1，GBS=16，micro batch=1，`num_steps_per_rollout=1`，默认 GRPO spec（运行前记录 `algorithm_spec_sha256`），`--seed 1234 --rollout-seed 42`，不开 `--data-parallel-random-init`、`--balance-data`、`--balance-by-flops`、动态 batch、`--allow-partial-train-step`、vpp。GRPO 下每个 rollout 1 条样本，GBS=16 个 rollout。
  - 为什么全部 dropout=0：DP 变化时新 rank 使用 fresh RNG（`keep_on_dp_change`），dropout>0 会让下一步比较混入 RNG 差异，容差就没有意义。**因此 go 结论只覆盖 dropout=0 的 profile**；dropout>0 的变 DP 边需单独认证，在此之前拒绝（写入 attestation 说明）。
- 确定性（A8 必须全部生效；任一项不可用即判环境阻塞、不运行，也不降级）：Megatron `--deterministic-mode`；`NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`。
- 冻结数据：训练样本在 A8 开头一次性生成（colocated SGLang，8 个 rollout 批 × 16 个 rollout），落盘为**未切分的** train data（含 `rollout_ids`、tokens、loss mask、reward、advantage 输入）；之后每个 arm 用 fork 的 `split_train_data_by_dp(args, data, train_parallel_config)` 分派，其中 `train_parallel_config` 取自本 arm 新建 trainer 的 `get_train_parallel_config()`（含本 arm 的 dp_size），因此走的是生产上的 scheduled 分支（运行时断言 `can_schedule_on_rollout_side` 为真、分片带 `micro_batch_indices`/`num_rollouts`，否则判环境阻塞）。每个批次运行前用 `reshard.step_problems` 核对两侧布局都接受。
- 硬件断言：启动时 `nvidia-smi --query-gpu=name` 全部等于预期型号，否则退出；记录驱动/CUDA 版本。
- 回收：Modal 函数 `timeout=` 等于硬超时；外加本机独立 watchdog（按 app ID `modal app stop`）；结束后 `modal app list` 核实 stopped、0 tasks，截图/文本存证。
- 调试一律在便宜卡（DEV-GATHER）；H100 只用于 A8 的逐位/数值对比本身。同一失败未定位原因不重跑。

## 1. DEV-GATHER（A8 前置调试；不上 H100）

目的：在真实 Megatron/DistOpt 上跑通 A8 的 harness（导出命名状态、跨 DP 合并切片、`restore_cut_resharded`、摘要比较），发现 GPU 才有的问题（TE FusedAdam 状态键、真实梯度 buffer、多 bucket、`to_safe` + `weights_only=True`）。**不做数值判定**，不作为验收证据。

- 复用：若 E2 的 A6（plan-v2 G-4.2/G-4.3 的 C2 配置：DP2+DistOpt）先跑并通过 plan-v2 §3 的 L2 三项，则"DistOpt 命名导出完整、`weights_only` 可加载、master 与 bf16 副本一致"已被证明，DEV-GATHER 只剩变 DP 的冒烟。
- 已在 CPU 证明、不需要上卡的部分：fork-M5 `merge_named_optimizer_states`/`load_named_optimizer_state` 的合并切片（fork CPU 单测 DP 2→1、2→3、1→4、4→2 下一步 AdamW 逐位一致）；yeto 侧协议（`tests/test_rl_trainer_reshard.py`，纯 torch 替身）。
- 资源：Modal 2×A10G（24 GB，bf16 可用），硬超时 1.5 h。GPU·h ≤ 3.0；按 $1.10/GPU·h 计 **≤ $3.3**（若 A10G 不可用改 2×L40S，≤ $5.9）。
- 通过条件（功能性）：DP1→2、DP2→1 各一次 `restore_cut_resharded` 不报错，`full_state_digests` 在每个 (tp,pp) 的 DP rank 间一致，恢复后能训练 1 步且 loss 有限。不通过 → 修 harness/代码，再在便宜卡上重试；A8 不启动。

## 2. A8 = 4.6 验收 X4（DP1↔2 重分片 spike）

- 资源：Modal `H100!:2`，一个容器内顺序跑所有 arm；硬超时 **2.0 h**。预计 ~1.3 h：样本生成 ~10 min；6 次 trainer 启动 × ~5 min（0.6B）；训练步合计 < 10 min；比较脚本 < 5 min。GPU·h：预计 2.6，上限 4.0；按 $3.95/GPU·h（gpu-plan 同口径）计预计 ~$10，**上限 $15.8**。
- 前置：DEV-GATHER 功能通过；1.1 manifest 与镜像一致；GPU 名断言通过。

### 2.1 arm（同一容器，按顺序）

| arm | GPU | 步骤 | 记录 |
|---|---|---|---|
| A1 | 1 | DP1 从头：步 1、2 → `save_cut` C1 → 步 3 → 步 4–8 | 每步后逐 rank：gathered FP32 master、exp_avg、exp_avg_sq、step、scheduler `num_steps`、grad_norm、逐样本 loss、消费的 sample id |
| A2 | 2 | DP2 从头：步 1、2 → `save_cut` C2 → 步 3 → 步 4–8 | 同上 |
| B1 | 2 | 新建 DP2 → `restore_cut_resharded(C1)` → 立即 `save_cut` C1′（不训练）→ 步 3 → 步 4–8 | 同上 + 恢复摘要、RNG 映射 |
| B1′ | 2 | 与 B1 完全相同（确定性复核），只跑到步 3 | 同上 |
| B2 | 1 | 新建 DP1 → `restore_cut_resharded(C2)` → 步 3 → 步 4–8 | 同上 |
| RT | 1 | 新建 DP1 → `restore_cut_resharded(C1′)`（1→2→1 往返），不训练 | 恢复摘要 |

步 3–8 全部使用冻结样本的第 3–8 个 rollout。

### 2.2 判据（全部预先固定）

- **G1 状态逐位（数据搬运无损）**：B1 恢复后 gathered 状态与 C1 逐位相等（`torch.equal`）：adapter、FP32 master、exp_avg、exp_avg_sq、step、scheduler、Megatron 计数；B2 对 C2 同样；RT 恢复后 gathered 状态与 C1 逐位相等，且 `full_state_digests` 相同。容差 0。
- **G2 批语义与 loss 归一化（A4、D7.4）**：每一步 A1 与 B1、A2 与 B2 的分片记录中：每步消费的 sample 集合相同、micro batch 组成（每个 micro batch 含哪些样本）相同、`num_rollouts` 相同（=16），只有 micro batch→rank 的分派不同，且实际分派等于 `reshard.scheduled_partitions` 的预测（k→k%dp）；`num_microbatches` DP1=16、DP2=8/rank；在 rank 内读回 `loss_function` 的 `loss_normalizer`（=num_rollouts）与 `loss_parallel_size`（=dp），与 `reshard.miles_microbatch_loss_scale` 一致。任一不符即失败。
- **G3 确定性**：B1 与 B1′ 的步 3 结果（gathered master、moments、grad_norm、逐样本 loss）逐位相等，恢复后 RNG 摘要相等。不等则判**不可判定**（环境不确定），不继续解读 G4，不重跑，写原因。
- **G4 下一步数值（跨 DP，只允许归约顺序差异）**：比较 B1 对 A1、B2 对 A2 的步 3：
  - 逐样本 loss：**逐位相等**。论证：scheduled 静态路径下 micro batch 是按步内样本顺序切的 mbs=1 块，与 dp 无关（G2 核对）；同一 micro batch 在两侧的输入张量、形状和（恢复后逐位相等的，G1）权重都相同，确定性模式下前向逐位相同；DP 只改变该 micro batch 在哪个 rank 上算、梯度如何归约，不影响前向。若 G2 发现组成不同则 G4 不适用、直接失败；
  - step 与 scheduler：逐位相等；
  - grad_norm：相对差 ≤ 1e-3；
  - 更新量 Δ=master(步3后)−master(步3前)：‖ΔB−ΔA‖₂/‖ΔA‖₂ ≤ 1e-3；|ΔA|>1e-8 的元素中 sign(ΔB)=sign(ΔA) 的比例 ≥ 99.9%；
  - exp_avg、exp_avg_sq：相对 L2 ≤ 1e-3。
- **G5 RNG 映射可解释（4.2a keep_on_dp_change）**：每个新 rank 的 RNG 来源（fresh）与种子推导记录在 RESULT（`reshard.rng_mapping`），并在 rank 内读回 `torch.initial_seed()`/`torch.cuda.initial_seed()` 与 Megatron tracker 种子，与推导一致；B1 与 B1′ 的 RNG 摘要相等。缺记录或不一致即失败。
- **G6 学习短跑（冻结样本，步 4–8）**：B1 对 A1、B2 对 A2：步 8 的 master 相对 L2 差 ≤ 1e-2；每步 loss 相对差 ≤ 1e-2；无 NaN/Inf。（统计学习验证属 4.8/A10，不在此处。）

### 2.3 go / no-go

- **go**：G1、G2、G3、G4、G5、G6 全部通过。go 只对本次 `algorithm_spec_sha256`、profile P-E3（含 dropout=0、DistOpt、bf16、GBS=16、mbs=1）与 DP 1↔2 成立；attestation 的 trainer-dp / role-transfer 边只列这一个哈希（A4）。不直接加入白名单（4.6 原文），由主 agent 更新 attestation。
- **no-go**：G1、G2、G4、G5、G6 任一失败。按 tasks 4.6 属合法否定结论：记录原因，不跑 A9/A10，trainer 弹性标为未交付，E1/E2 照常交付。
- **不可判定**：G3 失败或环境阻塞 → 视同非 go（不跑 A9），写明原因与需要的修复；修复后只允许在提出原因后重跑一次；**第二次仍不可判定即判 no-go**。

## 3. A9 = 4.7（仅 A8 为 go 时运行）

- 边：4 卡 T2R2 ↔ T1R3（trainer DP2 在 g0,g1；rollout 两个 TP1 engine 在 g2,g3；转移 g1）。复杂并行维度固定（TP=PP=CP=EP=1），GBS=16 两侧整除。
- 资源：Modal 4×L40S（功能验收，不做逐位比较，所以不用 H100），硬超时 **3.5 h**，预计 ~2.5 h。GPU·h：预计 10，上限 14；按 $1.95/GPU·h 计预计 ~$19.5，**上限 $27.3**。学习/数值统计属 4.8（A10），不在此处。
- 前置（任一不满足则不运行）：A8 go；controller/elastic-wiring 补丁已合入；E1 提供 §5 的 `bind_members` 与 `MilesTrainerOps` 接线；fork 需求 F-R1 已解决（§4），否则 T2R2→T1R3 的新 engine 无处声明。

### 3.1 序列（同一容器）

1. T2R2 下 2 个 outer round（基线）。
2. 请求 T2R2→T1R3（trainer→rollout）→ 须 SUCCEEDED；2 个 round。
3. 请求 T1R3→T2R2（rollout→trainer）→ 须 SUCCEEDED；2 个 round。
4. 故障注入（每项注入后回到起点配置再做下一项）：
   - f1：新 engine 启动失败（注入 `add_engines` 抛错）→ REBUILT_OLD：trainer 回 DP2（来自同一 cut），engine 集合恢复；随后 1 个 round 正常。
   - f2：目标 trainer 重建失败（注入 `create_training_models` 抛错）→ `rebuild_resharded` 回退旧形状 → REBUILT_OLD。
   - f3：新 engine ACK 的 policy hash 不符 → VERIFYING 失败 → REBUILT_OLD。
   - f4：rollout→trainer 方向 drain 超时 → CANCELLED，engine undrain，trainer 未动。
   - f5：在 `trainer_cut` 记录之后、COMMITTED 之前 kill learner → 重启后 RECOVERY_REQUIRED，journal 中有 `trainer_recovery_hint`（`restore_old`，指向该 cut）；不自动恢复训练（D5 允许的有界结果）。
   - f6：旧形状回退也失败（f1 + 回退时注入重建错误）→ RECOVERY_REQUIRED，不再消费数据。

### 3.2 判据（全部预先固定）

- 每次 SUCCEEDED：config epoch 恰好 +1；`epochs.json` 的成员等于目标 engine 集；trainer `actual_layout()` 的 DP 等于目标；重建后首次发布前 trainer 导出的 policy hash 等于 cut 中的 hash，新 engine ACK 同一 hash；数据游标在事务前后相同；账本无重复消费、`ready_unconsumed=0`；optimizer 步数不多不少（事务期间 `optimizer_applied` 不变，之后每 round +1）；scheduler `num_steps` 等于 local_step×GBS。
- trainer→rollout 方向：新 engine 所在 GPU 集合必须**等于** plan 的 `moved_gpus`（运行时从 fork `get_cell_bundles`/`get_worker_infos` 读回 GPU，经 M1 bundle 映射换成 GPU id；E1 的 `bind_members` 若提供绑定结果则再核一次）；rollout→trainer 方向：被摘除的 engine 正是 `member_gpus` 中服务于 `moved_gpus` 的那些，trainer 新 GPU 集合等于目标 placement。
- 每次 REBUILT_OLD / CANCELLED：config epoch 不变；成员与 trainer 布局回到事务前；policy hash 不变；之后 1 个 round 正常。
- RECOVERY_REQUIRED：不再调用 `train_step`/`generate`；journal 记录完整。
- batch 语义：每个 round 消费的组数等于 GBS/组大小，DP1 与 DP2 下 sample id 集合的切分符合 `sample_mapping`。
- 全部满足 → 4.7 验收通过；任一不满足 → 4.7 未完成，写原因。

## 4. fork 缺口与需求清单（不改 fork；需要时另批）

核对对象：Miles fork `yeto/ports`=`5c1b49eb`（只读）。

- **4.2a（M5）**：接口已满足。`export_named_optimizer_state` 按参数名导出（含 FP32 master `param`、moments、step、超参，DistOpt 记录各 rank 的区间）；`merge_named_optimizer_states` 在文件层面收集所有 DP 分片并校验覆盖；`load_named_optimizer_state` 按当前区间切片写回（新建 DistOpt 先 `_init_optimizer_states_with_dummy_values`）。所以"DistOpt 分片状态的 gather/reshard"已实现（文件式收集，不是 collective）。缺口只在验证：GPU 未验证（TE FusedAdam 状态键、多 bucket、VPP 前缀、真实梯度 buffer），由 DEV-GATHER/A8 确认。RNG：M5 的 `keep_on_dp_change` 只在 `load_lora_checkpoint` 路径里；yeto 的 cut 路径自行实现同一策略并记录种子映射，**不需要改 fork**。
- **4.6a（M6）**：`rebuild_training_models(args, …, trainer_pg_view=…)` 能以新的 `actor_num_gpus_per_node` 在新的 bundle 视图上重建并 dispose 旧 handle，失败抛 `TrainerRebuildError`（无自动回滚），`rebind_cell`、`set_pg_view`（可以新建视图名）已有。新 trainer 由 `create_training_models` 重新把 `train_parallel_config`（含新 dp_size）设给 rollout executor，所以下一批按新 DP 切分。缺口：
  - **F-R1（阻塞 A9，需要 fork 改动，需另批；已从源码核实）**：角色转移需要在 trainer 释放出的 GPU 上启动一个新的 rollout engine，现有 fork 做不到：
    1. `RayWorkerManager.init` 对**全部已声明 cell** 执行 `start_cells`（`ray_worker_manager.py:72-87`），不存在"声明但不启动"的 cell；
    2. M1 的 placement map 禁止角色重叠（`validate_placement_map`），rollout 视图启动时只含 rollout bundle，`_validate_binding` 拒绝超出视图的 slot；standby 角色虽存在，但 cell 只能绑定到本 pool 的视图；
    3. 因此 T2R2 启动时无法预先声明第三个 rollout cell。`set_pg_view` 可以新建视图名、`rebind_cell` 可以重绑停止的 cell，但需要先有这个 cell。
    需求：允许声明时不启动（初始停止）的 cell，且其绑定可延迟到 `rebind_cell` 之后才允许 `start_cells`；或允许 cell 声明在启动时为空的具名视图上。与 alignment §12 G6 相邻，需要用户批准后由 fork 负责人实现。
  - F-R2（非阻塞）：`_slice_pg_info` 是私有函数；yeto 的 `trainer_view()` 直接引用它。建议 fork 提供公开的"按 bundle 位置切视图"函数，否则 fork 改名会静默打断 E3。
  - F-R3（非阻塞，GPU 时核实）：新 rank 的 fresh 种子推导（Megatron `_set_random_seed` 以及 Miles 自己的种子设置）需要在 A8 G5 中实测读回；若 Miles 另有偏移，按实测更新 `reshard.fresh_seed` 并在 RESULT 说明（这是记录方式，不改判据）。

## 5. 接口请求（给 INFRA-E1；补丁基于 65ca03b）

- `infra-drafts/patches/infra-e3-controller-v3.patch`（v1、v2 已改名 `*.v1-OBSOLETE.patch`/`*.v2-OBSOLETE.patch`；controller.py、elastic_placement.py，新测试 `tests/test_rl_controller_trainer_edge.py`）：
  - `IslandController(..., trainer_edges=None)`：为 None 时 trainer 边照旧拒绝（原 E1 测试的 "not E1" 文案保留）；否则 `plan()` 把只含 `trainer-dp`/`role-transfer` 的边交给 `trainer_transition.plan_trainer_edge`（无副作用），`Plan` 新增可选字段 `trainer`；`_execute` 对 trainer 边调用 `_execute_trainer`：`TrainerTransition.run()` 返回 READY_TO_COMMIT 后由 controller 做唯一的 epoch CAS（COMMITTED→RESUMING→SUCCEEDED），CANCELLED/REBUILT_OLD/RECOVERY_REQUIRED 按原语义收尾；**v3**：提交 CAS 失败时按 E1 同语义 `_enter_recovery`；先读回 epochs，`last_tx_id` 为本事务（写入后才抛错）则 hint 为 restore_target，否则 restore_old，读不回则 recovery_required；`plan` 把 pool 的 `member_gpus()` 交给 `plan_trainer_edge`，按 moved GPU 选要摘除的 engine；transition 的 phase 记录经 `_trainer_record` 同步 `tx.phase`。
  - 重启时对未提交的破坏性 trainer 事务仍判 RECOVERY_REQUIRED（与 E1 相同），额外写 `trainer_recovery_hint`（`trainer_transition.recovery_decision`）。
  - `ElasticPlacement.reconfigure_trainer(plan, epoch)`：记录已提交的 trainer GPU 变化（嵌套集合）。
  - 在 65ca03b 上 `git apply --check` 通过；**在 infra-e1 当前 HEAD 533afdc 上 controller.py 第 180 行附近冲突**，需要 E1 按上下文手工合入（逻辑独立，冲突只在 `__init__` 参数区）。
- `infra-drafts/patches/infra-e3-elastic-wiring.patch`：`build_elastic(..., trainer_edges=None)` 透传（依赖上一补丁）。
- 还需 E1 实现（无补丁）：
  - `MilesRolloutPool.member_gpus() -> {member: [gpu ids]}`：摘除 engine 必须按 GPU 选（审查 H2），缺此接口时 rollout→trainer 边在 plan 阶段被拒绝。
  - `MilesRolloutPool.bind_members(members, gpus)`：对停止的 rollout cell 调用 fork `set_pg_view(<新视图名>, 释放出的 bundle 切片)` + `rebind_cell(cell, pg_name, 0)`；依赖 F-R1。
  - `compose_island` 在 `--rl-elastic` 下构造 `miles_adapter.trainer_resize.MilesTrainerOps`：`context_for(cut_id)`（driver 的 progress/游标/账本/外层状态，即 E2 的 CutContext）、`expect_for(layout)`、`view_for(gpu_ids)`（GPU id → 启动 PG 视图中的 bundle 位置 → `trainer_view`）、`policy_hash_fn`、`certified_for(plan)`（attestation 中该边的算法哈希），并把 `trainer_edges` 传给 `build_elastic`。
  - driver 的 4.4 钩子已在 E1 侧（`rebuild_trainer`）；E3 的 trainer 边不经过 driver 的 rebuild 钩子，而是由 controller 在安全点直接执行，重发同一 policy 仍经 `publish_members`。

## 6. 费用汇总（上限，按 gpu-plan 同口径单价）

| 项 | 资源 | GPU·h 预计/上限 | 费用预计/上限 |
|---|---|---|---|
| DEV-GATHER | Modal 2×A10G | 2.0 / 3.0 | ~$2.2 / $3.3 |
| A8 | Modal 2×H100! | 2.6 / 4.0 | ~$10.3 / $15.8 |
| A9（仅 go） | Modal 4×L40S | 10 / 14 | ~$19.5 / $27.3 |
| 合计 | | 14.6 / 21 | ~$32 / $46.4 |

依据：0.6B LoRA 的 trainer 启动约 5 min、单步训练 < 1 min（1.2 基线实测 2×H100 15 min 完成 ≥2 round）；A9 每次 trainer 重建约 5 min，共 2 次成功切换 + 6 项故障（约 9 次重建）+ 若干 round。
