# E2 合租 C1 第 1 次（2026-09-30，未完成：代码缺陷）

- 代码 a6ee31d（= integ-decl e052dc8 + d4aa108 + 工具提交），镜像 db815884（tag 2f23a0f-9f29303）。容器内 guard 通过：`NVIDIA H100 80GB HBM3` ×1，/root/miles HEAD 2f23a0fc，image-manifest tag 2f23a0f-9f29303（`gpu.txt`、`image.txt`）。
- app `ap-pgHWfRGVl6bs0y373amTJm`，06:33:29–06:52:20Z，已 stopped/0（`teardown_proof.txt`）。费用 ≤ $1.40。
- 过程：确定性读回通过；2 轮 warmup（磁带中有 2 条 `rl_round_trained`，data_cursor 可读）；`save_cut` 完成；G-4.2(g) 的在线恢复被拒，原因是 "LR scheduler already at 32: restore_cut only restores into a freshly built trainer"，与判据预期相符。
- 失败原因（代码）：rank 插件以**抛异常**的方式拒绝，而 Miles `TrainerController` 在 `run_plugin` 抛出任何异常后都会把 cell 标为 `StateAllocatedErrored`。之后 harness 读状态摘要时报 "run_plugin requires all cells alive"，harness 中止，learner 退出 1。Modal 随后自动重试 island，直到手动 stop。
- 修复（3b8a63e）：
  - 写入前的拒绝改为返回 `{"refused": ...}`，`MilesTrainerGroup.save_cut/restore_cut` 收到后抛 `CutError`；已开始写入后的失败仍然抛异常，trainer 按不可用处理。
  - puller 在 learner 退出后先拉取证据再 stop app，并且不拉取 cut 分片（本次 state 包因分片过大被截断，results.json 未拉到）。
- 判据：本次没有得出任何判据结论。G-4.2(g) 的"state unchanged"未能读取，不计。
