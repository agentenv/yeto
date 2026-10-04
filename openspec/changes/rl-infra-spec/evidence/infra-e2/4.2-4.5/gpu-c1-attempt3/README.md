# E2 合租 C1 第 3 次（2026-09-30，未完成：共置布局下同形重建不可行）

- 代码 d233128（integ-decl a382490 合并后）。镜像 db815884。guard 通过（H100 80GB HBM3 ×1，miles 2f23a0fc，tag 2f23a0f-9f29303）。app `ap-rOhq1LQl6A5ZjYM5Gxa0Ha`，07:37:10–07:45:34Z，stopped/0。费用 ≤ $0.60。Modal 未重试，launcher 未 relaunch。
- 通过的检查：determinism_settings；warmup 2 轮与 save_cut；**G-4.2(g)**；**G-4.3 arm A 训练前状态 == cut**（state_digest 与 rng_digest 都相等）。arm A 的冻结 batch 训练完成。
- 失败：`rebuild_same_shape` → fork `rebuild_training_models` 连续两次 `TrainerRebuildError`，在旧 trainer 被 stop 之后约 20 ms 内失败 → RecoveryRequired "trainer rebuild failed 2 times"。本次没有记录 stage 与 cause，harness 现已补记（见下一提交）。
- 原因（代码阅读，未在卡上确认 stage）：fork `RayWorkerManager.start_cells` 在启动前执行 `_assert_bundles_free(cell.bundles())`，与任何存活 cell 共用 bundle 就抛 `BundleInUseError`（`ray_worker_manager.py` start_cells，2f23a0fc；5c1b49eb 中也有）。C1/C2 按 plan-v2 使用共置布局，trainer 与 SGLang rollout cell 在同一个 bundle 上，rollout cell 没有停，所以新 trainer cell 起不来。同形重建（M6）只适用于 trainer 独占 bundle 的 fixed-partition 布局（E1 的 G-4.4 走的就是这种布局）。
- 不属于可在卡上修复的问题。需要主 agent 裁定：(a) C1/C2 改为 fixed-partition（C1 1 卡 trainer + 1 卡 rollout = H100!:2；C2 2+1 = H100!:3），这是配置变更，判据文字不变；或 (b) fork 支持共置重建（例如重建期间停 rollout cell 再恢复），属于新的 fork 工作，需要批准。
