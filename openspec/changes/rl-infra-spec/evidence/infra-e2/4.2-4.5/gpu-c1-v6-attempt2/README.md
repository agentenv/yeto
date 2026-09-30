# plan-v6 正式 C1 第 2 次（2026-09-30）：未完成（harness 比较口径缺陷）

- 代码 f898516；镜像 2cc5cc52；guard 通过（H100 80GB HBM3 ×2，miles e3a11ab3）。app `ap-FoedYXCYFuoS6Wx9BhbpLL`，09:39:08–09:49:42Z，stopped/0。费用 ≤ $1.40。
- harness 跑完全程。重建后重新发布成功：磁带中有 `rl_trainer_rebuilt`，说明 weight_version 修复生效。
- `HarnessFailed` 只列出两项失败：
  - "G-4.3 arm A pre-step state == cut"
  - "G-4.3(1) restored state/RNG == cut before step 3"
- 原因（harness 比较口径，我的代码）：这两项比较的 `state_digest` 自 f898516 起包含 `miles_counters.weight_version`。fixed-partition 下两个 arm 都会重新发布一次，计数 +1，于是与 cut 必然不等。训练状态本身是否一致，从这两项看不出来。
- 其余判据不在失败列表中，按 harness 逻辑应为通过：G-4.2 (a)–(e)(g)、G-4.3(2) adapter/master/moments/step/RNG 逐位相等、grad_norm 逐位相等、G-4.3(3)、moments 未重置、L2、恢复后 policy hash、数据游标、布局。但 `results.json` 没有拉回（容器停止前最后一次 tar 被截断），**不作判据结论**。
- 修复（下一提交）：
  - 训练前比较改用 `train_state_digest`（不含发布计数）；
  - 计数另设判据："weight_version == cut + 重新发布次数"；
  - puller 每轮单独拉取 `results.json` 与 `steps.jsonl`。
