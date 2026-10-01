# algo1b-g1e 计划：隔离 clip 上界（clip_higher）的配对对照（实验前提交，事后不改）

## 为什么能区分
- A `clip_sym`：eps_clip=0.001，不设 eps_clip_high。Miles 的 `arguments.py:3488-3489` 会把 eps_clip_high 置为 eps_clip，于是裁剪窗口为 [0.999, 1.001]。
- B `clip_hi`：eps_clip=0.001，eps_clip_high=10，窗口为 [0.999, 11]。
- 其余完全相同：同 seed、同 rollout（rollout 是确定性的，g1c 已验证）；每轮 2 个 optimizer step，lr 1e-4。
- 第 1 轮第 1 步是 on-policy，ratio≡1，两者的 loss 与更新相同（用第 1 步 grad_norm 相等来核验），所以第 1 轮第 2 步的 ratio 在 A 和 B 中也完全相同。在同一组 ratio 上，A 会同时裁掉 ratio>1.001 和 ratio<0.999 的 token，B 只裁掉 ratio<0.999 的 token。只要上界真的生效，**B 的 pg_clipfrac 必然严格小于 A**（前提是有 ratio>1.001 的 token）；如果 eps_clip_high 被忽略，B 退化成 A，两者的 clipfrac 就会逐位相等。

## 事先固定的判据
- 前提：两个 run 都 rc=0（sandbox 内 worker）、3 轮完成、无 invariant 错误（freeze_gc 良性链不算）。
- 配对有效：两者第 1 轮第 1 步的 train/grad_norm 逐位相等，且第 1 步的 rollout/raw_reward 相同。否则判"对照无效"，不声明。
- **clip_higher 生效**：配对有效，并且 B 在第 1 轮第 2 步的 pg_clipfrac **严格小于** A 在同一步的 pg_clipfrac。
- 若 A 的这一步 pg_clipfrac 为 0（没有触发裁剪），判"无法区分"；若 B ≥ A，判"未能证明生效"。这两种情况都不声明。
- 只用第 1 轮第 2 步作判定，因为这一步的配对最干净；之后几轮的数据只记录。结果如实交付，只有 harness 或环境问题可以在修复后重跑。

## 资源与回收
- 与 algo1b-g1c 相同：Modal Sandbox `H100!`×1（运行前断言型号），app `algo1b-g1e`；sandbox timeout 10800 秒，watchdog 11100 秒，每个 exec 1800 秒，本地 `timeout 11400`，EXIT trap 按 id 终止；结束后执行 `modal app stop algo1b-g1e`。
- 预计约 25 分钟，费用 ≤ $3。在 algo1b-g1d 结束之后再运行。

## 结论（sandbox 已终止，app 已 stopped）
- 两个 run 均 rc=0、3 轮完成、无 invariant 错误。
- 配对有效：第 1 步 grad_norm 都是 1.1293506622314453，逐位相等；第 1 步 raw_reward 都是 0.8125。
- 第 1 轮第 2 步的 pg_clipfrac：A 与 B 都是 0.23509125411510468。**B 没有严格小于 A，按预登记判为未能证明 clip_higher 生效，不声明。**
- 如实补充（未预登记，不作判定依据）：同一步的 grad_norm 却不同，A 为 0.5746，B 为 0.4456，说明 eps_clip_high 确实改变了 loss 的梯度，而 pg_clipfrac 这个指标没有反映出差异。按 `math_utils.compute_policy_loss` 的定义（clipfrac = pg_losses2 > pg_losses1），只要存在 A>0 且 ratio>1.001 的 token，两者的 clipfrac 就应该不同。这个矛盾目前没有解释，可能是 clipfrac 的统计口径或记录位置与我的理解不一致。要声明 clip_higher，需要改用"第 2 步 grad_norm 不同"作为判据，另立计划再验证，不能对本次结果事后改换判据。
