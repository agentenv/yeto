# A8 G4 未通过的静态排查（INFRA-E3，2026-09-30；不上卡）

对象：A8 第 2 次（Miles e3a11ab3，megatron.core 0.19.0+a84b10547，本地对照源码 megatron-core 0.19.2），argv 见 `miles_args.arm.*.json`。数据：取回的 15 个汇总状态（本地 `/home/michael/work/infra-e3-gpu/b3a8r/out/work/packed/`），分析输出 `g4_grad_analysis.txt`。

## 1. 梯度归约精度与路径
- argv 含 `--accumulate-allreduce-grads-in-fp32`；fork `lora/bridge.py:250` 以 `DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32)` 建 DDP → **梯度缓冲（main_grad）与 DistOpt 的 reduce-scatter 都是 fp32**。
- `average_in_collective` 默认 False → 先按 `gradient_scaling_factor = 1/dp` 缩放缓冲再求和（`distributed_data_parallel.py:212`，`param_and_grad_buffer.py:660`）；DP=2 时乘 0.5，是 2 的幂，精确。
- `overlap_grad_reduce` 未开（argv 无；同步归约）。`--no-gradient-accumulation-fusion`：每个 micro batch 的权重梯度先得 bf16 `param.grad`，再加到 fp32 `main_grad`。
- 结论：归约与累加在 fp32，DP1 与 DP2 的求和次序不同只会带来约 1e-7 级相对误差，**不能解释 0.83%**。

## 2. loss 归一化
- Miles `loss_function` 缩放 `num_microbatches / num_rollouts × dp`：DP1 为 16/16×1=1，DP2 为 8/16×2=1；Megatron schedule 再除以 `num_microbatches`（16 或 8，`schedules.py:335/341`），DDP 再乘 1/dp → 每个样本的权重两边都是 1/16，且所有因子都是 2 的幂。`calculate_per_token_loss` 未开。数学等价、数值上也应逐位等价（2 的幂缩放在 bf16/fp32 中精确，除非下溢）。

## 3. bf16 与 fp32 梯度
- 每个 micro batch 的激活梯度与权重梯度在 bf16 计算（`--bf16`），累加与归约在 fp32（见 1）。

## 4. 确定性设置
- `--deterministic-mode`（`torch.use_deterministic_algorithms(True)`、cuDNN deterministic）与 `NCCL_ALGO=Ring`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` 使**同一配置**可复现（G3：DP2 的 B1 与 B1p 逐位相同），但不保证 DP1 与 DP2 两种进程配置之间逐位相同。

## 5. 离线数据
- 从同一 cut 出发，步 3 的有效梯度（由 exp_avg 反推）DP1 与 DP2 相对 L2 差 0.83%（A1/B1）与 0.83%（A2/B2，方向相反的一对，结果一致）。
- **按层分布**：最后一层（27）差异为 0，往前逐层增大，第 0 层约 1.5%（中位数由 0.0006 → 0.0148 单调上升）；各类 adapter（q/k/v/proj/fc1/fc2，linear_in/out）量级相同，与参数种类、大小或 bucket 边界无关。
- 逐元素相对差中位数 0.7%，90 分位 4.9%——与 bf16 精度（2^-8≈0.39%）同量级，远大于 fp32（1.2e-7）。
- 逐样本 loss（前向）两边逐位相同。

## 判断
- 差异不是来自梯度归约、缩放或累加（fp32、2 的幂，且与层深无关），而是**逐样本反向传播本身**在 DP1 与 DP2 两种进程配置下不逐位相同：最后一层输入的梯度完全相同，误差从输出端向输入端逐层累积、放大，是 bf16 激活梯度在反传链中舍入不同的典型形态。DP=1 与 DP=2 在同一 micro batch 上执行同样形状的计算，但反向 kernel 的数值结果不同（推测为算法/kernel 选择或实现随进程配置变化；现有数据无法再细分）。
- 分类：**c) 属当前 bf16 profile 下不可避免的跨 DP 数值差异**，不是 E3 重分片实现缺陷（G1 逐位证明状态搬运无损；G2/G3/G5/G6 通过）。
- a) 配置能否消除：唯一有依据的方向是整条前反向用 fp32（去掉 `--bf16`），但那会改变被认证的 profile（生产为 bf16），go 只对 fp32 profile 成立，且不能证明 bf16 profile 的边；判据不放宽的前提下，这不构成对生产 profile 的修复，不建议为此重跑。若主 agent 希望进一步确认"DP1 自身可复现、差异只在 DP1 与 DP2 之间"，可加跑一个 DP1 从 C1 同形恢复的 arm（约 $4），但不会改变 G4 结论。

## 4.6 完成记录草稿（未勾选，待主 agent 确认）
- [x] 4.6 ……
  - 完成记录（2026-09-30 INFRA-E3，**合法否定结论：no-go**）：A8（plan-v6 预注册判据，Modal H100!:2，Miles e3a11ab3，代码 a016f7f）。G1 通过：DP1↔2 重分片后 adapter、FP32 主参数、两个动量、step、超参、scheduler、计数与源 cut 逐位相等，1→2→1 往返无损；G2 通过（样本集合、micro batch 组成、num_rollouts 归一化一致）；G3 通过（同配置逐位可复现）；G5 通过（新 rank fresh RNG，种子可复现）；G6 通过；**G4 未通过**：从同一 cut 出发的下一步，更新量相对 L2 0.91%（容差 0.1%）、符号一致率 99.894%（≥99.9%）、exp_avg 0.30%（0.1%）。原因分析（`evidence/infra-e3/a8-run2/g4-analysis.md`）：bf16 profile 下 DP1 与 DP2 的逐样本反向传播不逐位相同，误差随反传深度累积，非重分片缺陷。按 tasks 4.6 与 BRIEF，no-go 为合法否定结论：trainer 变 DP 边不加入白名单，4.7/4.8 按原文降级（不运行 A9/A10），E1/E2 照常交付。程序说明：容器内 compare 因读取 step 字段位置错误未产出结论，结论来自取回状态的离线 compare（修正字段位置，判据与容差未改）。
