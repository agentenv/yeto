# plan-v6 正式 C1 第 1 次（2026-09-30）：未完成（cut 状态缺项，已修）

- 代码 7c041d4；镜像 2cc5cc52（miles e3a11ab3，tag e3a11ab-9f29303）；guard 通过（H100 80GB HBM3 ×2）。app `ap-kxFnsxAErPwoGG4bV2LUua`，09:20:16–09:31:23Z，stopped/0。费用 ≤ $1.50。
- 通过：
  - determinism_settings；
  - **G-4.2 (a)(b)(c)(d)(e)(g)**：六项都被拒，且拒绝后状态不变；
  - G-4.3 arm A 训练前状态 == cut；
  - 合法 restore 通过插件自检（重新导出的摘要 == cut）。说明 fork M5 修复在真实 DistOpt 上生效，没有再出现 exp_avg 缺失。
- 失败：`driver.rebuild_trainer` 在恢复后重新发布时，fork 的 rollout executor 断言失败："Engine weight version went backwards: 4 -> 1"。
  - 原因：Miles trainer 进程内的 `weight_updater.weight_version` 在重建后从 0 重新计数，cut 没有携带这个计数。4.1 状态审计漏了这一项。
- 修复（下一提交）：cut 分片新增 `miles_counters.weight_version`；同形恢复和重分片恢复（E3）都在写入前检查 trainer 有无 `weight_updater`，写入时恢复该计数；它也计入状态摘要。已补 CPU 测试。
- 影响：同样的缺陷会让 E1 的 G-4.4（`driver.rebuild_trainer`，fixed-partition）在重新发布时失败，也会影响 E3 在 DP 变化后的重新发布。本次运行不作判据结论；G-4.2 在 C1 上的通过需要在修复后的正式运行中复现后才计入。
