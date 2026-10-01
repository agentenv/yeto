# C1 诊断运行 1（仅诊断，不作判据结论）

- 代码 b15a04b，app ap-jJOWvaeXNswcv9MAAeRwsN，08:11:28–08:22:21Z，stopped/0，费用 ≤ $1.45。
- 分项摘要显示：合法恢复后与 cut 不同的组件是**全部 392 个优化器条目的 `tensors`**（即 DistOpt 的 main param、exp_avg、exp_avg_sq 这一组）。adapter 模型副本、每个条目的 hyper 与 scalars、scheduler、Megatron 计数器、RNG 全部相同。原始列表见 `diff_components.txt`。
- 分项深度只到 `tensors`，分不出具体是哪个键、差多少。已加逐叶差异报告（每个张量的 dtype/shape 或最大绝对差、不等元素数），见下一提交。
