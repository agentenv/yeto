# algo1b-g1c 计划：token 聚合与 no_grpo_std_normalization 的隔离对照（实验前提交，事后不改）

## 背景（审查结论）
- 第一次 G1 中，token 运行 3 个 step 的 grad_norm（0.6349/0.4057/0.2744）与 baseline **逐位相同**，第 2、3 轮的 rollout 也相同，说明权重更新完全一样；pg_loss 从 1e-8 变成 0.035，只是记录口径变了。token 的声明因此撤回。
- no_grpo_std_normalization 当时与 constant 聚合放在同一个 run 里，没有隔离对照，声明也已撤回。

## 原因排查（离线，已完成的部分）
- `--calculate-per-token-loss` 确实送到了 Miles：miles.log 的参数表中为 True。
- Miles `loss.py:203-220`：per-token 模式下返回 (token 求和的 loss, num_tokens)，不乘 num_microbatches/global_batch；Megatron 按 `config.calculate_per_token_loss` 决定是否除以 num_tokens（`model_provider.py:49` 会把它写进 provider）。按数学推导，只要各样本长度不同（本 run 的 response 长度确实不同），两种聚合的梯度就应该不同，不应当逐位相等。
- 所以"逐位相等"**不能**用"在此配置下数学等价"来解释。更可能的原因是 per-token 的归一化在这条 LoRA/bridge 路径上没有真正作用到反向（例如训练 actor 中 Megatron 的模型 config 没有带上这个值，或 Miles 的 scaling 被另一处抵消）。原因目前**未确认**，本计划的 run 用来确认它是否生效，不负责找到根因。

## 设计：用第 1 步做配对对照
- 同一 sandbox、同一镜像、同一 seed。第一次 G1 的 baseline、token 和 drgrpo 第 1 步的 rollout 一致（token 与 baseline 第 1 步 grad_norm 逐位相同也印证了这一点），可见 rollout 是确定性的，因此**各 run 第 1 个训练 step 的输入样本相同**，比较第 1 步的 grad_norm 就是隔离对照。
- 三个 run：baseline（重跑，作为同一 sandbox 内的对照）、token（aggregation=token）、no_std（只设 std_normalization=false）。其余配置与 algo1b-g1 完全相同：Qwen3-0.6B@c1899de2，每轮 4 组 × 8 条，response 384，lr 1e-5，seed 17，3 轮。

## 事先固定的判据
- 前提：3 个 run 都 rc=0（这里指 sandbox 内 worker 的 rc，不经过 launcher）、3 轮完成、没有 invariant 错误；freeze_gc 良性链按审查决定不算错误。
- **配对有效性**：token/no_std 与 baseline 第 1 步的 `rollout/raw_reward` 必须相同（同样的样本）。若不同，该对比判为"对照无效"，不做结论。
- **token 生效**：配对有效，并且 token 第 1 步的 train/grad_norm 与 baseline 第 1 步**不相等**。若相等，结论为"未能证明生效（梯度与 baseline 相同）"，不声明，并作为缺陷报告给 ALGO-CAP/INFRA。
- **no_std 生效**：配对有效，并且至少有一组奖励 std 既不为 0 也不为 1（从 rollout 日志核实），同时 no_std 第 1 步的 grad_norm 与 baseline 第 1 步不相等。若所有组的 std 都是 0 或 1，判为"本批数据无法区分"，不声明。
- 结果按预登记如实交付。只有查明原因并修复（仅限 harness 或环境问题）才允许重跑，不因为结果不理想而重跑。

## 资源与回收
- Modal Sandbox，`H100!`×1（运行前断言型号），app `algo1b-g1c`。
- 硬超时：sandbox timeout 10800 秒；独立 watchdog 在 11100 秒时 kill；每个 exec 1800 秒；本地 `timeout 11400`；EXIT trap 按 id 终止 sandbox。
- 预计约 40 分钟，费用 ≤ $4。结束后核实 sandbox 列表为空、app 为 stopped（`sbx.py list` 会建出空 app，结束后需 `modal app stop algo1b-g1c`）。
- 在 algo1b-g1b 的 run A-r1 结束之后才开始（GPU 串行使用）。
