# algo1b-g1j 结果（clip_higher 的最终实验；按预登记判据交付；sandbox 已终止，app 已 stopped）

- 版本：YETO_SHA 为 92a88c40（harness/YETO_SHA），Miles 0394715（沿用 g1e 的 sbx.py 配置）。worker.json 核实为 groups_per_round=6、optimizer_steps=3、seed=17。
- 两个 run 均 rc=0、3 轮完成、无 invariant 错误。
- 配对有效：第 1 轮第 1 步 grad_norm 两边都是 1.1293506622314453，逐位相等；第 1 步 raw_reward 都是 0.5833。
- 第 1 轮第 3 步 grad_norm：A（对称）为 0.5642381310462952，B（上界 10）为 0.6571707725524902，**不相等，判据满足**。
- **结论：clip_higher 在 GPU 上证明生效，可以声明 `features:clip_higher`。**
- 如实记录（不作判据）：
  - 第 2 步 grad_norm 也不同（0.5746 对 0.4456），与 g1e 的数值相同；
  - 第 2 步 pg_clipfrac 两边仍然相同（0.23509），第 3 步不同（0.2220 对 0.2648）。第 2 步 clipfrac 与 loss 不一致的疑点仍然存在，见 `2026-09-29-clipfrac-offline/report.md`，列为待办。
