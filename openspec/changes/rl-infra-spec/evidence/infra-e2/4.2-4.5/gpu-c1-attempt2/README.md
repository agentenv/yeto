# E2 合租 C1 第 2 次（2026-09-30，未完成：harness 代码缺陷）

- 代码 1c803a8（= integ-decl 3f88c1d 合并后，含 3b8a63e 修复与 `--modal-retries 0 --modal-timeout-s 5700`）。镜像 db815884。guard 通过：H100 80GB HBM3 ×1，miles 2f23a0fc，tag 2f23a0f-9f29303。
- app `ap-mL9pAYSP2F9OUtl3bOFAeD`，07:05:53–07:13:25Z，stopped/0。费用 ≤ $0.55。**Modal 没有重试**（`rank 0/1 starting` 只出现 1 次）。
- 在 harness 结果中已通过的判据：
  - 确定性设置逐 rank 读回通过；
  - 2 轮 warmup 与 `save_cut` 完成（manifest 见 `cut-manifest.json`）；
  - **G-4.2(g) 在线恢复被拒且状态不变：通过**（3b8a63e 修复生效）。
- 失败原因（harness 代码，我的文件）：arm A 为了与重建后的重新发布对称，在共置岛上把**已经发布并加载**的同一 policy 又发布了一次。SGLang `resume_memory_occupation` 对没有 offload 的 weights 抛 `KeyError: 'weights'`，调度器退出，publish 失败，harness 中止。
- 修复（见下一提交）：两个 arm 都不重新发布（冻结 batch 不需要 rollout）；arm B 直接调用 `rebuild_same_shape`，不经 `driver.rebuild_trainer`（后者会重新发布）；恢复后另核对 policy hash 等于 cut；共置岛训练前按 driver 的顺序先 onload。判据未改。
- 另外发现：
  1. `driver.rebuild_trainer`（E1 4.4）在**共置**岛上会做同样的重复发布，E1 的 G-4.4 在 fixed-partition 上运行，不受影响，但共置路径有同样风险。
  2. launcher 在 island 退出后有自己的 relaunch 循环（"relaunch attempt ... will retry"）。app 已 stop 时 lookup 失败，不会产生费用；本次在 rc 前手动结束了本地 launcher。
