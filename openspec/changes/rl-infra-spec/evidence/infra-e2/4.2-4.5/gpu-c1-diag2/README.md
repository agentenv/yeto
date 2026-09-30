# C1 诊断 2（仅诊断，不作判据结论）

- 第 1 次（app ap-ld4Hr0ztI1W3BhWyEa8FT2）：容器刚启动时 guard 的 exec 返回空（image.txt 为空），guard 误判失败并 stop，费用 ≤ $0.30。工具已修（c62a24c：空结果重试）。
- 第 2 次（app ap-rt2YdWGlUERxJ3OaKjc0kZ，08:35:03–08:45:02Z，2×H100!，≤ $1.35）：诊断输出见 `restore_diff.txt`。
  - 第一段，cut → 加载后立即读取：`param` 在 392 个条目上全部逐位相等，dtype 均为 fp32，DistOpt 区间与形状一致（ranges=0）；`exp_avg`、`exp_avg_sq` 在 392 个条目上都**缺失**（加载后不存在）。
  - 第二段，加载后读取 → 重新导出：无差异。
  - 结论：不是比较口径问题（假设 1 排除），不是加载后被覆盖（假设 2 排除），也不是区间写错（假设 3 排除）。是加载时 Adam 状态根本没写进去。
- 根因（代码阅读，与现象一致）：harness 在合法恢复之前对新 trainer 做了"状态不变"读取；fork-M5 的读取经 Megatron 的 defaultdict 留下空 state 条目；M5 的 load 只在 state 全空时才初始化 Adam 状态，其 setter 又只拷贝已有键，于是 exp_avg、exp_avg_sq 被静默丢弃。需求已写给 fork 负责人（`infra-drafts/fork-req-m5-lazy-state.md`）；yeto 侧的规避和 CPU 复现替身见后续提交。
