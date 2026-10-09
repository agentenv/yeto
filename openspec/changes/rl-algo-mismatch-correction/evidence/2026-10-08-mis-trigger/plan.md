# mis 截断变体触发验证：预登记草稿（S17，2026-10-08，未上卡）

目的：补 rl-algo-mismatch-correction 7.3 的最后一项。mis（`mis_mode=truncate`）在 G1 时能跑通（evidence/2026-09-29-g1b/runs/mis/check.json），但 `train/mis_tis_truncate_fraction` 三步都是 0，没有证明截断确实生效，所以还没写进声明。本次只做"截断确实生效"的触发验证，阈值是测试用的，不是推荐值。

## 为什么改小上界就一定能触发（依据已存数据，不是推测）
- G1 时上界是 2.0，三步的逐 token 重要性比最大值分别是 1.1907 / 1.1908 / 1.2473（同一 check.json 的 `train/mis_is_ratio_max_final`），都低于 2.0，所以截断比例为 0。
- 截断变体只有上界：权重 = min(比值, 上界)。把上界设成 **1.01**，上面三步的最大值都超过它，每步至少有一个 token 被截断，截断比例必然大于 0。
- 参考做法：2026-09-29 的触发运行（evidence/2026-09-29-trigger/plan.md）把 tis 和 mis-mask 的区间设成 [0.99, 1.01]，三步的截断/遮挡比例约 0.19–0.27。本次只有上界，预计比例约为它的一半（估计，未验证）。

## 设定（沿用 2026-09-29 触发运行，只改算法说明）
- 入口：`yeto launch --training-mode rl --rl-engine ports --gpu modal:1xh100 --modal-gpu-exact --controller local --rl-single-island-no-sync --rl-allow-unverified-mechanism corrections:custom --rl-allow-unverified-mechanism corrections:mis`（放行项名称上卡前按当前代码核对），其余同 g1c：Qwen3-0.6B LoRA、4×8、3 轮、seed 17。
- 算法说明：g1b 的 mis spec（evidence/2026-09-29-g1b/runs/mis/spec.json），只把 `mis_upper_bound` 从 2.0 改为 1.01；`mis_level=token`、`mis_mode=truncate`，函数 `yeto.rl.algos.vendor.miles_mis.compute_mis_weights_with_cp`（sha256 上卡前按当前代码重新核对）。
- 镜像：当前 main 的默认镜像（Miles 8bc52237a，与 09-29 当时不同）。上卡前确认 MIS 计算路径在两个 Miles 版本之间没有变化。

## 判据（上卡前固定）
- 有效性：同 g1c 的 1–6 条（launcher rc 0；train step 0..2 都在日志里；磁带有 rl_local_round 1..3、轨迹 32、grad_norm 有限；发布 v0..3 带 policy token；无失败事件；所选 sha 与放行项一致）。
- 生效：至少一个已记录的训练步中 `train/mis_tis_truncate_fraction > 0`，并且同一步的 `mis_tis_weight_after_bound` ≤ `mis_tis_weight_before_bound`。
- 零梯度等不变量报错记为失败，不放宽。运行有效但生效为 0 就照实报告。

## 卡数、卡型、费用
- **1 张 Modal H100（`H100!:1`，启动时断言 GPU 名称）**。理由：单岛、不同步，和 09-29 的触发运行一致；0.6B 模型不需要更多卡，H100 是当时取证用的卡型，换卡型会引入新的数值差异。
- 时长：09-29 的 4 次触发运行合计约 33 H100 分钟，平均每次约 8 分钟；按 12–15 分钟估（含拉镜像）。
- 单价：台账里 Modal H100! 约 $3.9/卡·小时（例：2026-09-30 05:37–05:52，16 分钟×2 卡 ≤$2.09）。
- **估计 ≈$1（上限按 $2 设：超时 2700 s、watchdog 3000 s、结束时 app stop 并保存 app list）。只跑一次；在 app 创建之前出现 harness 故障可以修一次，修之前先提交说明。**
- 台账 gpu-spend.md 要等批准后再预登记，本草稿不算预登记。

## 结果怎么用
- 通过：在 Miles 适配层的声明里加 `corrections:mis`，附证据目录；更新 `tests/test_rl_mismatch_correction.py` 的已声明集合；勾选 7.3，然后 rl-algo-mismatch-correction 只剩可选的 7.7，可以归档。
- 不通过：照实写进 progress.md，mis 保持不声明。
