# 4.2–4.5 待本地 GPU 验证计划 v2（INFRA-E2，2026-09-30；取代 plan.md，判据在运行前固定，事后不改）

与 v1 的差别（审查 M4、L2）：
- 确定性设置改为固定写死，不再写"如镜像支持"。任何一项在镜像中不可用，就判为**环境阻塞，不运行**，也不降级运行。
- 每个判据都固定了模型与 DP 配置。
- G-4.2 补两种情形：DistOpt 对端分片损坏；在线（已训练）trainer 上就地恢复。
- 新增 L2 需 GPU 确认的项（§3）。
- 反映审查修复后的语义：`restore_cut` 只用于新建的 trainer；任何恢复异常都判 RECOVERY_REQUIRED；重建前后比对数据游标。

状态：**未运行**。用户已暂停 GPU/云验证，等自有卡到位后在本地执行。

## 0. 通用前提（固定）

- 代码：yeto `infra-e2`（运行时的 SHA 记入 RESULT）；Miles fork `yeto/ports`=`0af62f4d`；镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:c6f5455c8a88d131c780cf99d07c3fdd913d2beda25b6d94739db06916d602ed`。运行前生成 1.1 runtime manifest 并核对 digest。
- 前置接线（未完成则不运行）：ports-v2 补丁、entry-v2 补丁（`SwappableActor`）；E1 的 `RolloutPool.data_cursor()` 与账本计数；driver 4.4。
- 固定配置 C1：Qwen3-0.6B，LoRA r=16、alpha=32、dropout=0.05，bf16，TP=PP=CP=EP=1，**DP=1（1 卡）**，不用 DistOpt，GBS=16，`num_steps_per_rollout=1`。
- 固定配置 C2：与 C1 相同，但 **DP=2（2 卡），DistOpt 打开**（Miles 对 Adam 的默认）。
- 固定配置 C3（仅用于 G-4.4/G-4.5）：Qwen3-1.7B，其余同 C2。
- 共同设置：GRPO 默认 spec（运行前记录 `algorithm_spec_sha256`），`--seed 1234 --rollout-seed 42`。
- 确定性（固定，全部必须生效）：
  - Megatron `--deterministic-mode`；
  - `NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`；
  - `--sglang-enable-deterministic-inference`；
  - 两个 arm 的第 3 步使用**同一份冻结的 `rollout_data_pack`**：先由 arm A 落盘，arm B 回放，因此 rollout 侧的不确定性不进入比较。
- 硬件：同一台机器、同型号 GPU，两个 arm 在同一次运行内先后执行；运行前断言 `nvidia-smi --query-gpu=name` 输出，并记录驱动与 CUDA 版本。
- 硬超时：每个实验 90 分钟，外层 `timeout` 加独立 watchdog（按 PID 终止）。

## 1. 判据

### G-4.2 cut 保存、校验、拒绝（配置 C1 与 C2 各跑一遍）
训练 2 个 outer round → quiescent cut → `save_cut`。再逐项构造下列情形，每项都在新建的 trainer 上调用 `restore_cut`，只有 (g) 例外：
- (a) 本 rank 分片翻转最后一个字节；
- (b) 本 rank 分片截断 1 KiB；
- (c) driver local_step+1；
- (d) 改一个 spec 字段（算法哈希不同）；
- (e) 插件哈希不同；
- (f) **仅 C2：DP 对端分片（dp1）翻转一个字节，在 dp0 上恢复**；
- (g) **在线 trainer 就地恢复**：不重建，直接在刚保存 cut 的存活 trainer 上调用 `restore_cut`。
- 通过：
  - (a)–(f) 全部抛出 `CutError`/`CutPluginError`；
  - (g) 以 "freshly built" 拒绝（scheduler `num_steps≠0`）；
  - 每次拒绝后，被拒绝 trainer 的 adapter、optimizer、scheduler 摘要与拒绝前逐位相同；
  - 正常 cut 的 manifest 存在，所有分片的 sha256 与 manifest 一致。
- 失败：任何一项被接受，或拒绝过程中改动了状态。

### G-4.3 = X3 同形完整恢复（配置 C1 与 C2 各 1 次；确定性对比，不按 seed 统计）
- arm A（连续）：训练 2 步 → 落盘第 3 步的冻结 batch → 训练第 3 步。逐 rank 记录 adapter、FP32 master、exp_avg、exp_avg_sq、step、scheduler `num_steps`、RNG 摘要、grad_norm。
- arm B（cut）：同一 seed 训练 2 步 → `save_cut` → `rebuild_same_shape`（销毁并重建 trainer 子进程）→ `restore_cut` → 在同一冻结 batch 上训练第 3 步，记录相同的项。
- 通过（全部满足）：
  1. 恢复后、第 3 步之前：每个 rank 的 `state_digest`/`rng_digest` 等于保存值；scheduler `num_steps` 等于 2×GBS（没有被累加两次）；
  2. 第 3 步之后：adapter、FP32 master、exp_avg、exp_avg_sq 与 arm A `torch.equal`；step、scheduler、grad_norm 也逐位相等；
  3. optimizer.step 共 3 次，没有 moments 归零，没有额外的 scheduler 步；
  4. 重建前后 `data_cursor()` 相同，`actual_layout()` 相同。
- 只有 bf16 模型副本不等也判失败（不放宽）。同一失败先定位原因再重跑。失败即在 E3 之前阻断。

### G-4.4 driver 端口替换（配置 C3）
- 通过：
  - `initialize`/`after_local_train` 调用计数不变；
  - 重建后第一次 publish 的 policy hash 等于 cut 中的 `progress.policy_hash`，所有 engine 读回校验通过；
  - outer round 不回退、不重放；
  - `optimizer_applied` 只 +1；
  - Disposer 退出时释放的是**当前** trainer（旧 handle 不会被 dispose 第二次，新 handle 没有泄漏）。

### G-4.5 同形恢复故障矩阵（配置 C3）
| 注入 | 通过判据 |
|---|---|
| 重建后 `restore_cut` 期间 kill 一个 rank | `T_restore`=600 s 内判 RECOVERY_REQUIRED，不训练，不自动重试恢复 |
| DP=2 时一个 rank 在 `restore_cut` 前 sleep 超过 `--distributed-timeout-minutes` | 同上 |
| `save_cut` 写分片途中 kill 一个 rank | 没有 manifest，cut 不存在；release 前回到旧进程继续，或判 RECOVERY_REQUIRED |
| `create_training_models` 抛错 | `TrainerRebuildError` 后重建一次，结果为 REBUILD_OLD；有 `cleanup_error` 时判 RECOVERY_REQUIRED |
| 重建期间人为改动数据游标（自定义 rollout load） | 游标比对失败，判 RECOVERY_REQUIRED，不调用 restore |
| cut 已提交后，在 epoch CAS 之前或之后 kill controller | 按 journal：未提交则按旧配置从 cut 重建；已提交则不回滚；无法对账则 RECOVERY_REQUIRED |

所有行共同要求：`optimizer_applied` 不出现第二次；没有组被消费两次；游标不回卷。

## 2. 产物
`evidence/infra-e2/4.2-4.5/<run-id>/`：命令、1.1 manifest、GPU 名断言、确定性环境变量回显、事件磁带、各 rank 摘要 JSON、比较脚本输出、RESULT.md（逐条对照本文件判据）。

## 3. 审查 L2：需 GPU 确认的项（运行 G-4.2 C2 时一并记录）
- fork-M5 在真实 Float16Optimizer 与 DistOpt 下导出的命名状态（含 `hyper`）以及 Megatron scheduler `state_dict`，经 `to_safe` 编码后能否用 `torch.load(weights_only=True)` 加载；不能加载即判失败，不得改回 `weights_only=False`。
- DistOpt DP=2 下，每个 (tp,pp) 的 `optimizer_names` 并集覆盖全部 adapter 名（`has_optimizer_state`）。
- 恢复后 bf16 模型副本与 FP32 master 的一致性：恢复后立即 `bf16(master) == model copy`。
