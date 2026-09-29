# algo1b-g1h 计划：over_sampling 生效（submitted_groups 配对对照）——新实验，实验前提交，事后不改

与 g1d 的关系：这是新实验。g1d 的结论（未能证明生效）保持不变。INFRA 的结论是：Miles 每次提交 over_sampling_batch_size 个组，接受到 rollout_batch_size 个组后就中止还在飞的请求，被中止的组不会进入 all_samples，所以 g1d 用的 generated 实际只是"完成的组数"。

## 版本与入口
- 代码：分支 `algo-1b-os`，基于 integ-decl 28724bc（含 cbb5d98 的 `submitted_groups` / `aborted_in_flight_groups`，以及 1b-hook）。`submitted_groups` 是 data_source 的 sample_offset 在本轮的增量：首轮、数据回绕、buffer 非空时为 None（`rollout_meta_hook.py:357-376`）。事件 `rl_round_trained.submitted_groups` 由 driver 写出（`driver.py:833`）。
- sandbox harness（同 g1d），Miles 检出 0af62f4d（= 该 SHA 的 MILES_NEXT_COMMIT），SGLang 为 9f29303。yeto 使用 `git archive HEAD`，SHA 写入 harness/YETO_SHA。
- 两个臂，除 over_sampling 外完全相同：同一 bounded nonzero-std filter、max_replacements 2、rollout_batch_size（每轮组数）**4**、每组 8 条、seed 17、lr 1e-5、**6 轮**。prompt 池 240 条，保证 6 轮内不会回绕（最坏情况 6×16=96）。
  - os_off：over_sampling_batch_size 未设，Miles 取 rollout_batch_size=4；
  - over_sampling：over_sampling_batch_size=**8**。

## 事先固定的判据
- 前提：两个臂都 rc=0、6 轮完成、无 invariant 错误（freeze_gc 良性链不算）。
- "有值的轮次"指 `submitted_groups` 不为 None 的轮次，预期是第 2–6 轮。**每个臂都至少要有 2 个有值轮次**，否则判"证据不足"。
- **over_sampling 生效**需要同时满足：
  - (a) over_sampling 臂的每个有值轮次，submitted_groups 都是 8 的正整数倍（Miles 按 8 组一批提交，过滤补采可能再提交一批），因此都 ≥ 8，大于 rollout_batch_size=4；
  - (b) os_off 臂的每个有值轮次，submitted_groups 都是 4 的正整数倍，且**至少有一个**有值轮次正好等于 4。
  (b) 保证同样的过滤条件下，不开超采样时存在只提交 4 组的轮次；(a) 保证开了超采样时每轮至少提交 8 组。两者对照，可以把多出来的提交归因于超采样本身，而不是过滤补采。
- 若 (a) 不成立，判"未能证明生效"；若 (b) 不成立（os_off 每个有值轮次都 ≥8），判"对照无法区分"。两种情况都不声明。
- 结果按预登记如实交付；只有 harness 或环境问题可以在修复后重跑。

## 资源与回收
Modal Sandbox `H100!`×1（运行前断言型号），app `algo1b-g1h`；sandbox timeout 10800 秒，独立 watchdog 11100 秒，每个 exec 1800 秒，本地 `timeout 11400`，EXIT trap 按 id 终止；结束后执行 `modal app stop algo1b-g1h`。预计约 40 分钟，≤ $4。在 g1g 结束后再运行。

## 结论（sandbox 已终止，app 已 stopped；YETO_SHA 见 harness/YETO_SHA）
- 两个臂都 rc=0、6 轮完成（rollout 0–5），无 invariant 错误。
- 各轮 submitted_groups（第 1 轮为 None，与预期一致）：
  - os_off：8/4/8/8/8，aborted 为 2/0/2/2/1；
  - over_sampling：8/8/8/8/8，aborted 为 4/3/1/2/3。
- 判据 (a)：over_sampling 的 5 个有值轮次都是 8 的倍数（均为 8），满足。
- 判据 (b)：os_off 的有值轮次都是 4 的倍数，并且第 3 轮（rollout 2）正好为 4，满足。
- 两个臂的有值轮次都有 5 个，不少于 2。
- **结论：over_sampling 预登记判据满足（区分力弱，另有补充证据，见 results.md），可以声明 `features:over_sampling`。**（审查更正：区分力与决定性证据见 `results.md`。）
