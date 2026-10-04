# E2 合租 C1 第 4 次（plan-v5，fixed-partition T1R1，2026-09-30）

- 代码 706016a，镜像 db815884，guard 通过（H100 80GB HBM3 ×2，miles 2f23a0fc）。app `ap-7z0qR79EpNnzMIL0ZpKcSm`，07:54:07–08:05:20Z，stopped/0。费用 ≤ $1.50。
- 通过：
  - determinism_settings；
  - **G-4.2 (a) 坏 checksum、(b) 截断、(c) step 不一致、(d) 算法哈希不一致、(e) 插件哈希不一致、(g) 在线恢复**：六项都被拒，且拒绝后状态不变；
  - G-4.3 arm A 训练前状态与 RNG 摘要都等于 cut。
- 重建本身成功：fixed-partition 下新 trainer cell 正常启动。
- **发现（未通过）**：合法 restore 之后，rank 0 重新导出的 `state_digest` 与 cut 不等，`rng_digest` 相等。`restore_cut` 因此判 RECOVERY_REQUIRED，G-4.3 arm B 与比较没有执行。
  - 这说明同形恢复后的 trainer 状态与保存时**不逐位一致**。
  - 可能的来源（未确认）：optimizer 参数组级字段（例如 TE FusedAdam 的 group `step`，M5 按参数恢复时可能不覆盖）、Megatron 计数器或 scheduler 字段。CPU 替身上完全一致，说明问题出在 Megatron/M5 的真实状态上。
- 已补诊断：保存与恢复两侧都记录分项摘要；不一致时 `CutError` 会列出不同的组件名（如 `state/optimizer_named/entries/<param>/hyper`）。判据不变。
