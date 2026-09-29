# 2.4 前置：DistOpt 分片主参数 GPU 冒烟（INFRA，事前计划；单独提交后才启动）

- 目的：验证 state_plugin 的分片主参数导出/应用（66afb1e）在真实 Megatron DistributedOptimizer、trainer DP=2 下可用。这是 2.4 T2R2 第一次运行失败的修复。本冒烟不属于 2.4 的判据运行。
- 镜像：`MILES_NEXT_IMAGE` = …@sha256:9f0977da…（elastic，miles d002615f）。代码为本计划提交之后的 infra-a HEAD。
- 拓扑（最小可行）：Modal 1 个 island，3×`H100!`（`--modal-gpu-exact`），`--rl-placement fixed-partition --rl-rollout-gpus 1`，即 T2R1，trainer DP=2。profile 同 2.2（Qwen3-0.6B LoRA r16、GRPO、strict-avg、1 个 learner、本机 head、seed 17），total-steps 2。前缀 `infra-a-dos1`。
- 成功条件（全部满足）：
  1. learner SUCCEEDED，没有 `StatePluginError` 与 `rl_strict_failure`；
  2. launch 日志中 `data_parallel_size == 2` 且 `use_distributed_optimizer == True`；
  3. 两轮都有 `rl_round_trained`；每轮 `rl_policy_apply` 的 `sync/global_policy_hash` 与紧随其后的 `rl_publication` token 哈希一致（导出 → 应用 → 发布闭环）。
  失败则记录原因，修复后才重跑。
- 费用：约 15 分钟 × 3 H100 ≈ $3；硬超时 60 分钟，上限约 $12。回收方式同 2.2（`timeout 3600`、独立 watchdog、用 `modal app list` 核实）。
