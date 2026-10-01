# 2.6 第 2 次运行计划（写于运行前）

- **第 1 次结果**（sandbox sb-19VMbFwFBWDkDbQKMCyrsA，只用 CPU，已 terminate）：在导入 `megatron.training` 时失败。原因是 `transformer_engine` 需要 `libcuda.so.1`，只有 CPU 的容器里没有 GPU 驱动库（日志见 `attempt1/run.log`）。解析一例都没有执行，结果不计入。
- **修复**：给容器分配最便宜的 GPU `T4`（1 张），只为提供驱动库。其余内容与 `../2.6-plan.md` 完全一致：同一镜像、同样 20 个用例、同样的成功条件与容差、`timeout=1800`、看门狗、app 名称 `algocap-parse`。
- **费用估算**：T4 约 $0.59/h，加 CPU，预计 30 分钟内 < $0.5；上限 $1。
