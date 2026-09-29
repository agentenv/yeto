# algo1b-g1d 计划：over_sampling 生效的配对对照（实验前提交，事后不改）

## 背景
g1b run C 判为"未能证明生效"，原因是当时 ports 路径没有接上 `dynamic_filter_*`。INFRA 已在 infra-a da9d993 接好：`generated_groups = 训练组数 + filtered`，其中 filtered 来自 all-samples hook，数的是本轮生成了但没有训练的组。`replacement_attempts` 是代理值（等于 filtered），**不使用**。主 agent 要求判据必须能把"超采样多生成的组"与"动态过滤丢掉的组"区分开。

## 设计（配对，同一 sandbox，同一 seed，同一 filter）
- os_off：bounded nonzero-std filter，max_replacements 2，不设 over_sampling（Miles 的 over_sampling_batch_size 等于 rollout 批大小 4）。
- over_sampling：同一 filter 与 max_replacements，over_sampling_batch_size 8。
- 其余与 algo1b-g1 相同（Qwen3-0.6B@c1899de2，每轮 4 组 × 8 条，response 384，lr 1e-5，seed 17），**5 轮**。
- 入口：sandbox 内的 benchmark worker（`harness/g1.py`），事件磁带（island-0/events.jsonl）从 sandbox 直接取回。yeto 代码包含 infra-a 的接线（本分支已 merge）。

## 事先固定的判据
- 前提：两个 run 都 rc=0、5 轮完成、无 invariant 错误。freeze_gc 良性链不算错误。
- **生效**需要同时满足：
  - (a) over_sampling 每一轮的 `dynamic_filter_generated_groups` ≥ 8（Miles 每次至少提交 over_sampling_batch_size 个组）；
  - (b) os_off 至少有一轮 `dynamic_filter_generated_groups` < 8。
  (b) 用来说明 (a) 中多出来的组不是过滤补采造成的：同样的过滤条件下，不开超采样时生成数会低于 8。
- 若 os_off 每一轮都 ≥ 8，说明过滤补采本身就能生成那么多组，判为"对照无法区分"，不声明。
- 若 over_sampling 有任意一轮 < 8，判为"未能证明生效"，不声明。
- 若 generated_groups 缺失或为 None，判为"证据不足"，不声明。
- 结果按预登记如实交付；只有 harness 或环境问题查明原因并修复后才允许重跑。

## 资源与回收
- Modal Sandbox，`H100!`×1（运行前断言型号），app `algo1b-g1d`。
- 硬超时：sandbox timeout 10800 秒，独立 watchdog 11100 秒，每个 exec 1800 秒，本地 `timeout 11400`，EXIT trap 按 id 终止 sandbox；结束后执行 `modal app stop algo1b-g1d`。
- 预计约 35 分钟，费用 ≤ $4。

## 结论（sandbox 已终止，app 已 stopped）
- 两个 run 均 rc=0、5 轮完成、无 invariant 错误。
- 各轮 `dynamic_filter_generated_groups`：
  - os_off：6/6/6/6/6（dropped 2/2/2/2/2）；
  - over_sampling：6/4/6/7/7（dropped 2/0/2/3/3）。
- **判据 (a) 不满足**：over_sampling 每一轮的 generated_groups 都小于 8。按预登记，结论为**未能证明 over_sampling 生效，不声明**。
- 如实补充（未预登记，不作判定依据）：两个 run 的生成组数处在同一量级。可能原因有两种，一是 Miles 在 over_sampling_batch_size=8 时并没有一次多提交 8 组，二是 all-samples hook 看到的组只包含已完成的组。两者无法区分；需要 INFRA 在 Miles 侧确认"每轮实际提交的组数"，然后另立计划。
