# algo1b-g1i 计划：overlong_filter 的 G1 与配对对照（实验前提交，事后不改）

## 版本与入口
- 分支 `algo-1b-os`，基于 integ-decl 28724bc：已包含 1b-hook（record_trained_groups 中运行 sample filters，`rl_round_trained.filtered_samples` 汇总）。Miles 为 0af62f4d。harness 与 g1h 相同，YETO_SHA 写入 harness/YETO_SHA。
- 两个臂：of_off（默认 GRPO）与 of_on（`sampling.overlong_filter=true`，由 with_pipeline_plugins 列出 sample_filters 的 PluginRef）。其余完全相同：seed 17，每轮 4 组 × 8 条，lr 1e-5，3 轮。**rollout_max_response_len=160**，用来制造截断（本模型在 gsm8k 上的响应长度约 74–384，中位数约 150–330）。

## 事先固定的判据
- 前提：两个臂都 rc=0、3 轮完成，没有 invariant 错误，尤其是没有 `zero_grad_norm_with_nonzero_advantages`，即零梯度不变量不能误报（freeze_gc 良性链不算）。
- 配对有效：第 1 步的 `rollout/raw_reward` 相同，且第 1 步 `rollout/truncated_ratio` 大于 0（存在截断）。
- **overlong_filter 生效**需要同时满足：
  - (a) of_on 的事件中，`rl_round_trained.filtered_samples` 至少有一轮大于 0，且不超过该轮的样本数；
  - (b) of_off 各轮的 `filtered_samples` 为 None 或 0；
  - (c) 第 1 步 `train/grad_norm` 在 of_on 与 of_off 之间不相等（截断样本的 loss mask 被置 0，梯度应当改变）。
- 任一条不满足，判"未能证明生效"，不声明；配对无效时判"对照无效"。
- 结果按预登记如实交付；只有 harness 或环境问题可以在修复后重跑。

## 资源与回收
Modal Sandbox `H100!`×1（运行前断言型号），app `algo1b-g1i`；sandbox timeout 10800 秒，独立 watchdog 11100 秒，每个 exec 1800 秒，本地 `timeout 11400`，EXIT trap；结束后执行 `modal app stop algo1b-g1i`。预计约 25 分钟，≤ $3。在 g1h 结束后再运行。
