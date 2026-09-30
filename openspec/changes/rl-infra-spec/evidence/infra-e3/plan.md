# E3 待验证计划：A8（4.6 X4）与 A9（4.7）（INFRA-E3，2026-09-30；判据运行前固定，事后不改）

状态：**未运行**。本文件只是计划；没有启动任何 GPU 或云资源。先把本文件单独提交，之后才允许起卡。
预算约束（主 agent 转达的用户要求，2026-09-30）：A1–A9 的 GPU 总成本 ≤ $300；本文件内 A8 目标 ≤ ~$20，A9 ≤ ~$35，DEV-GATHER 尽量不用 H100。
对齐：`gpu-plan.md`（A8/A9/DEV-GATHER 行）与 PLAN-V2（`/home/michael/work/gpu-plan-v2`，本文件不改它的内容）。与 gpu-plan.md 相比本计划**缩小了规模**：A8 从 4×H100·7 h 缩到 2×H100!·≤2 h；A9 从 8×H100 Nebius P62↔P44 改为 4×L40S Modal T2R2↔T1R3（tasks 4.7 允许"更小等价边"）。这一缩减需要主 agent 在 PLAN-V2 里采纳。

## 0. 通用前提（固定）

- 代码：yeto `infra-e3`（运行时 SHA 写入 RESULT），加上 `infra-drafts/patches/infra-e3-controller.patch`、`infra-e3-elastic-wiring.patch`（A9 需要；A8 不需要）。Miles fork `yeto/ports`=`5c1b49eb`；镜像须含该提交（IMG 负责重建，运行前生成 1.1 runtime manifest 并核对 digest）。不满足则不运行。
- 模型与 profile P-E3：Qwen3-0.6B，LoRA r=16、alpha=32、**lora_dropout=0**，bf16，DistributedOptimizer（Miles 对 Adam 的默认），TP=PP=CP=EP=1，GBS=16，micro batch=1，`num_steps_per_rollout=1`，默认 GRPO spec（运行前记录 `algorithm_spec_sha256`），`--seed 1234 --rollout-seed 42`，不开 `--data-parallel-random-init`、`--balance-data`、动态 batch。
  - 为什么 dropout=0：DP 变化时新 rank 使用 fresh RNG（`keep_on_dp_change`），dropout>0 会让下一步比较混入 RNG 差异，容差就没有意义。**因此 go 结论只覆盖 dropout=0 的 profile**；dropout>0 的变 DP 边需单独认证，在此之前拒绝（写入 attestation 说明）。
- 确定性（A8 必须全部生效；任一项不可用即判环境阻塞、不运行，也不降级）：Megatron `--deterministic-mode`；`NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`。
- 冻结数据：训练样本在 A8 开头一次性生成（colocated SGLang，8 个 rollout × 16 条），落盘为**未切分的**样本列表（含 sample id、tokens、loss mask、reward、advantage 输入）；之后每个 arm 用 Miles `split_train_data_by_dp_raw(args, data, dp_size=本 arm DP)` 重新切分后喂给 `actor.train`。rollout 侧的不确定性因此不进入比较，而 DP 切分走的是真实的 Miles 代码。
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
- **G2 批语义与 loss 归一化（A4、D7.4）**：步 3 在 A1 与 B1、A2 与 B2 消费的 sample id 集合相同（与 DP 无关），各 rank 分片按 Miles 轮转切分的实际记录与 `reshard.sample_mapping` 一致；日志中每个 rank 的 normalizer 与 `num_microbatches` 符合 `sample_weight=1/GBS`（DP1：16 个 micro batch；DP2：每 rank 8 个）。任一不符即失败。
- **G3 确定性**：B1 与 B1′ 的步 3 结果（gathered master、moments、grad_norm、逐样本 loss）逐位相等，恢复后 RNG 摘要相等。不等则判**不可判定**（环境不确定），不继续解读 G4，不重跑，写原因。
- **G4 下一步数值（跨 DP，只允许归约顺序差异）**：比较 B1 对 A1、B2 对 A2 的步 3：
  - 逐样本 loss（前向，mbs=1、形状相同）：**逐位相等**；
  - step 与 scheduler：逐位相等；
  - grad_norm：相对差 ≤ 1e-3；
  - 更新量 Δ=master(步3后)−master(步3前)：‖ΔB−ΔA‖₂/‖ΔA‖₂ ≤ 1e-3；|ΔA|>1e-8 的元素中 sign(ΔB)=sign(ΔA) 的比例 ≥ 99.9%；
  - exp_avg、exp_avg_sq：相对 L2 ≤ 1e-3。
- **G5 RNG 映射可解释（4.2a keep_on_dp_change）**：每个新 rank 的 RNG 来源（fresh）与种子推导记录在 RESULT（`reshard.rng_mapping`），并在 rank 内读回 `torch.initial_seed()`/`torch.cuda.initial_seed()` 与 Megatron tracker 种子，与推导一致；B1 与 B1′ 的 RNG 摘要相等。缺记录或不一致即失败。
- **G6 学习短跑（冻结样本，步 4–8）**：B1 对 A1、B2 对 A2：步 8 的 master 相对 L2 差 ≤ 1e-2；每步 loss 相对差 ≤ 1e-2；无 NaN/Inf。（统计学习验证属 4.8/A10，不在此处。）

### 2.3 go / no-go

- **go**：G1、G2、G3、G4、G5、G6 全部通过。go 只对本次 `algorithm_spec_sha256`、profile P-E3（含 dropout=0、DistOpt、bf16、GBS=16、mbs=1）与 DP 1↔2 成立；attestation 的 trainer-dp / role-transfer 边只列这一个哈希（A4）。不直接加入白名单（4.6 原文），由主 agent 更新 attestation。
- **no-go**：G1、G2、G4、G5、G6 任一失败。按 tasks 4.6 属合法否定结论：记录原因，不跑 A9/A10，trainer 弹性标为未交付，E1/E2 照常交付。
- **不可判定**：G3 失败或环境阻塞 → 视同非 go（不跑 A9），写明原因与需要的修复；修复后只允许在提出原因后重跑一次。

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
- 每次 REBUILT_OLD / CANCELLED：config epoch 不变；成员与 trainer 布局回到事务前；policy hash 不变；之后 1 个 round 正常。
- RECOVERY_REQUIRED：不再调用 `train_step`/`generate`；journal 记录完整。
- batch 语义：每个 round 消费的组数等于 GBS/组大小，DP1 与 DP2 下 sample id 集合的切分符合 `sample_mapping`。
- 全部满足 → 4.7 验收通过；任一不满足 → 4.7 未完成，写原因。

## 4. fork 缺口与需求清单（不改 fork；需要时另批）

核对对象：Miles fork `yeto/ports`=`5c1b49eb`（只读）。

- **4.2a（M5）**：接口已满足。`export_named_optimizer_state` 按参数名导出（含 FP32 master `param`、moments、step、超参，DistOpt 记录各 rank 的区间）；`merge_named_optimizer_states` 在文件层面收集所有 DP 分片并校验覆盖；`load_named_optimizer_state` 按当前区间切片写回（新建 DistOpt 先 `_init_optimizer_states_with_dummy_values`）。所以"DistOpt 分片状态的 gather/reshard"已实现（文件式收集，不是 collective）。缺口只在验证：GPU 未验证（TE FusedAdam 状态键、多 bucket、VPP 前缀、真实梯度 buffer），由 DEV-GATHER/A8 确认。RNG：M5 的 `keep_on_dp_change` 只在 `load_lora_checkpoint` 路径里；yeto 的 cut 路径自行实现同一策略并记录种子映射，**不需要改 fork**。
- **4.6a（M6）**：`rebuild_training_models(args, …, trainer_pg_view=…)` 能以新的 `actor_num_gpus_per_node` 在新的 bundle 视图上重建并 dispose 旧 handle，失败抛 `TrainerRebuildError`（无自动回滚），`rebind_cell`、`set_pg_view`（可以新建视图名）已有。新 trainer 由 `create_training_models` 重新把 `train_parallel_config`（含新 dp_size）设给 rollout executor，所以下一批按新 DP 切分。缺口：
  - **F-R1（阻塞 A9，需要 fork 改动，需另批）**：角色转移需要在释放出的 trainer GPU 上启动一个新的 rollout engine。M1 的 placement map 禁止角色重叠，而 rollout 视图在启动时只有 rollout bundle，因此启动时无法声明第三个（停止状态的）rollout cell——其默认绑定会超出 rollout 视图，被 `_validate_binding` 拒绝。需求：允许在启动时声明"未绑定/延迟绑定"的停止 cell（只能通过 `rebind_cell` 获得绑定后才能 `start_cells`），或允许把 cell 声明在一个启动时为空的具名视图上。这与 alignment §12 的 G6 相邻，需要用户批准后由 fork 负责人实现。**待核实**：可用 fork CPU 测试确认现状（本 agent 未运行 fork 测试）。
  - F-R2（非阻塞）：`_slice_pg_info` 是私有函数；yeto 的 `trainer_view()` 直接引用它。建议 fork 提供公开的"按 bundle 位置切视图"函数，否则 fork 改名会静默打断 E3。
  - F-R3（非阻塞，GPU 时核实）：新 rank 的 fresh 种子推导（Megatron `_set_random_seed` 以及 Miles 自己的种子设置）需要在 A8 G5 中实测读回；若 Miles 另有偏移，按实测更新 `reshard.fresh_seed` 并在 RESULT 说明（这是记录方式，不改判据）。

## 5. 接口请求（给 INFRA-E1；补丁基于 65ca03b）

- `infra-drafts/patches/infra-e3-controller.patch`（controller.py、elastic_placement.py，新测试 `tests/test_rl_controller_trainer_edge.py`）：
  - `IslandController(..., trainer_edges=None)`：为 None 时 trainer 边照旧拒绝（原 E1 测试的 "not E1" 文案保留）；否则 `plan()` 把只含 `trainer-dp`/`role-transfer` 的边交给 `trainer_transition.plan_trainer_edge`（无副作用），`Plan` 新增可选字段 `trainer`；`_execute` 对 trainer 边调用 `_execute_trainer`：`TrainerTransition.run()` 返回 READY_TO_COMMIT 后由 controller 做唯一的 epoch CAS（COMMITTED→RESUMING→SUCCEEDED），CANCELLED/REBUILT_OLD/RECOVERY_REQUIRED 按原语义收尾；transition 的 phase 记录经 `_trainer_record` 同步 `tx.phase`。
  - 重启时对未提交的破坏性 trainer 事务仍判 RECOVERY_REQUIRED（与 E1 相同），额外写 `trainer_recovery_hint`（`trainer_transition.recovery_decision`）。
  - `ElasticPlacement.reconfigure_trainer(plan, epoch)`：记录已提交的 trainer GPU 变化（嵌套集合）。
  - 在 65ca03b 上 `git apply --check` 通过；**在 infra-e1 当前 HEAD 533afdc 上 controller.py 第 180 行附近冲突**，需要 E1 按上下文手工合入（逻辑独立，冲突只在 `__init__` 参数区）。
- `infra-drafts/patches/infra-e3-elastic-wiring.patch`：`build_elastic(..., trainer_edges=None)` 透传（依赖上一补丁）。
- 还需 E1 实现（无补丁）：
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
