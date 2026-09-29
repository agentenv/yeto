# 2.4 T2R2 seed 17：运行失败（运行时限制，不是偶发）

- 启动 19:31 UTC，4×H100!（Modal，断言通过）。首次训练后 state plugin 导出失败：`StatePluginError: low-precision adapter parameter has no FP32 optimizer master`（见 launch.log）。
- 原因：miles 在 adam 下默认启用 DistributedOptimizer（`miles/backends/megatron_utils/arguments.py:22`）。trainer DP=2 时 fp32 主参数按 DP 分片，bf16 LoRA 参数上没有完整的 `main_param`，而 ports 的 `state_plugin.master_of` 只支持完整主参数（DP=1 时成立，所以 1.2 和 2.2 都在 DP=1 下通过）。
- 影响：trainer DP>1 的配置在当前 ports 运行时都跑不起来。这包括计划中的 T2R2，以及 C4（共置 4 卡，DP=4）。
- 处理：按计划"失败的配置记为失败并附原因，修复原因后才重跑"，同时"不重复启动同一实验"。19:52 UTC 终止了 run 脚本和该次 launch，剩余配置（T1R3、C4、seed 29）都没有启动。app 已 stopped、0 tasks；本机 syncer 与 watchdog 均已结束。
- 费用：约 21 分钟 × 4 H100 ≈ 1.4 GPU·h ≈ $5.5。
