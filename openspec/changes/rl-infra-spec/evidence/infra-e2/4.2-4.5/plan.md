# 4.2–4.5 待本地 GPU 验证计划（INFRA-E2，2026-09-30；判据在运行前固定，事后不改）

状态：**未运行**。用户已暂停一切 GPU/云验证（2026-09-30），本计划等用户的自有卡到位后在本地执行。本文件提交后，其中的判据、容差、seed 与比较口径不得修改；如需修改，另起新版本文件并说明原因，旧版本保留。

## 0. 通用前提

- 代码：yeto `infra-e2`（或其合入 integ-decl 后的 SHA，运行前在 RESULT 中记录）；Miles fork `yeto/ports`=`0af62f4d`，镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:c6f5455c8a88d131c780cf99d07c3fdd913d2beda25b6d94739db06916d602ed`。运行前先生成 1.1 runtime manifest 并核对。
- 前置接线（未完成前不能运行）：ports.py 的 `save_cut/restore_cut` 定稿；entry.py 用 `SwappableActor` 包装 actor；driver 实现 4.4 的端口替换与重发；E1 的 `data_cursor()` 与账本计数（见 `cut-audit.md` §5）。
- 模型与配置：Qwen3-0.6B LoRA（r=16）做功能验证，Qwen3-1.7B LoRA 做验收；bf16；TP=PP=CP=EP=1；GRPO 默认 spec（`algorithm_spec_sha256` 在运行前记录）；GBS=16，`num_steps_per_rollout=1`；`--seed 1234 --rollout-seed 42`；LoRA dropout 设为 0.05，使 RNG 影响结果。DP=2 验收再加一个 DistOpt 配置（Miles 默认对 Adam 打开）。
- 硬件：同一台机器、同型号 GPU，两个 arm 在同一次运行内先后执行；运行前断言 `nvidia-smi --query-gpu=name` 与计划一致，并记录驱动/CUDA 版本。
- 确定性：`--sglang-enable-deterministic-inference`；Megatron 确定性设置（`NCCL_ALGO=Ring`、`--deterministic-mode`，如镜像支持）；同一 rollout 数据：对比 arm 使用冻结的同一份 `rollout_data_pack`（先由连续 arm 落盘，再由 cut arm 回放）。
- 硬超时：每个实验 90 分钟，由外层 `timeout` 包裹，另有独立 watchdog 按 PID 终止；本地卡不涉及费用，只需记录时长。

## 1. 实验与预先固定的判据

### G-4.2（F4 对应）cut 保存/校验/拒绝
- 步骤：训练 2 个 outer round → quiescent cut → `save_cut` → 分别构造坏 checksum（翻转分片最后一个字节）、截断（去掉末尾 1 KiB）、step 不一致（driver local_step+1）、算法哈希不一致（改一个 spec 字段）、插件哈希不一致，然后逐一 `restore_cut`。
- 通过：5 种情况全部抛出 `CutError`/`CutPluginError`，且拒绝后 trainer 的参数与 optimizer 摘要与拒绝前相同；正常 cut 的 manifest 在 `fsync` 后存在，所有分片 sha256 与 manifest 一致。
- 失败：任何一种坏 cut 被接受，或拒绝过程中写入了状态。

### G-4.3 = X3 同形完整恢复（原文："训练2步后重建，对冻结下一batch比较RNG/计数/moments/参数更新，无额外reset"）
- arm A（连续）：训练 2 步，落盘第 3 步的冻结 batch，再训练第 3 步，记录每个 rank 的 adapter、FP32 master、exp_avg、exp_avg_sq、step、scheduler `num_steps`、RNG 摘要和 grad_norm。
- arm B（cut）：同一 seed 训练 2 步 → `save_cut` → `rebuild_same_shape`（销毁并重建 trainer 子进程，同形）→ `restore_cut` → 用同一冻结 batch 训练第 3 步，记录同样的项目。
- 配置：DP=1（1 卡）与 DP=2+DistOpt（2 卡），各跑 1 次（确定性对比，不按 seed 统计）。
- 通过（全部满足）：
  1. 恢复后、第 3 步之前：每个 rank 的 `state_digest` 与 `rng_digest` 等于保存值（插件自检）；
  2. 第 3 步之后：adapter、FP32 master、exp_avg、exp_avg_sq 与 arm A **逐位相等**（`torch.equal`）；`step` 与 scheduler `num_steps` 相等；grad_norm 逐位相等；
  3. `optimizer.step` 次数 = 3，没有 moments 归零（exp_avg 非零），也没有额外 scheduler 步。
- 如果第 2 条只在 bf16 模型副本上出现不相等，而 FP32 master 相等，判为**失败**（不放宽容差）；需要先定位原因再重跑，同一失败不重复运行。
- 失败：任何一项不等 → 在 E3 之前阻断，按 design"moments/RNG/主权重缺失即阻断E3"处理。

### G-4.4 driver 端口替换
- 步骤：在 G-4.3 arm B 的流程中通过 driver 执行（不是直接调用插件）。
- 通过：`initialize`/`after_local_train` 调用计数不变（事件磁带核对）；重建后第一次 publish 的 policy hash 等于 cut 的 `progress.policy_hash`，所有 engine 读回校验通过；outer round 编号不回退、不重放；账本 `optimizer_applied` 只 +1。

### G-4.5 同形恢复故障矩阵（原文："rank失败、collective超时、迁移中断、controller crash提交不确定均有界恢复或RECOVERY_REQUIRED，不继续不确定的消费"）
| 注入 | 注入点 | 通过判据 |
|---|---|---|
| rank 失败 | 重建后 `restore_cut` 期间 kill 一个 rank | 在 `T_restore`（固定 600 s）内得到 `REBUILD_OLD` 成功，或 `RECOVERY_REQUIRED`；不训练 |
| collective 超时 | DP=2 时让一个 rank 在 `restore_cut` 前 sleep 超过 `--distributed-timeout-minutes` | 同上 |
| 迁移中断 | `save_cut` 写分片途中 kill 一个 rank | 没有 manifest（cut 不存在），恢复路径拒绝该 cut，并回到旧进程（release 前）继续，或 `RECOVERY_REQUIRED` |
| rebuild 失败 | 让 `create_training_models` 抛错（错误的 `trainer_pg_view`） | fork `TrainerRebuildError` → 重建一次（`REBUILD_OLD`）；`cleanup_error` 时 `RECOVERY_REQUIRED` |
| controller crash 提交不确定 | cut 已提交，在 epoch CAS 之前或之后 kill controller | 重启后读 journal：未提交则从 cut 按旧配置重建；已提交则不回滚；无法对账则 `RECOVERY_REQUIRED` |
- 所有行都要求：不出现第二次 `optimizer_applied`，没有组被消费两次，数据游标不回卷。

## 2. 产物

`evidence/infra-e2/4.2-4.5/<run-id>/`：运行命令、1.1 manifest、GPU 名称断言输出、事件磁带、每个 rank 的摘要 JSON、比较脚本输出、RESULT.md（逐条对照上面的判据）。
