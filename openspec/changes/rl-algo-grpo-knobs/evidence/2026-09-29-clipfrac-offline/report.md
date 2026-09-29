# pg_clipfrac 与上界：离线排查（ALGO-1b，未使用 GPU）

- Miles `losses.py:210-211` 调用 `compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high, ...)`，`math_utils.py:263-265` 定义 `pg_losses2 = -ratio.clamp(1-eps_clip, 1+eps_clip_high) * A`，`clipfrac = gt(pg_losses2, pg_losses1)`。按这个定义，clipfrac **包含**上界的贡献（A>0 且 ratio>1+eps_high 的 token），上界并没有被忽略。
- 离线复现（`check_clipfrac.py`，输出见 `output.txt`）：对同一组 ratio（`exp(N(0, 0.01))`）与 A，A 组（对称 0.001）的 clipfrac 为 0.4600，B 组（eps_high=10）为 0.2354；两者之差 0.2246 正好等于"A>0 且 ratio>1.001"的 token 比例。
- 与 g1e 的矛盾：g1e 中 A 组第 2 步的 pg_clipfrac 为 0.23509，与 B 组逐位相同，而两者的 grad_norm 不同。按上面的定义，只要 grad_norm 不同，就说明存在 A>0 且 ratio>1.001 的 token，A 组的 clipfrac 就应当更大。因此 GPU 上 A 组**记录下来的 clipfrac 缺少上界那一部分，而 loss 和梯度中包含它**。
- 可能原因（未确认）：`compute_policy_loss` 带有 `@torch.compile(dynamic=True)`。A 组中 eps_clip 与 eps_clip_high 是同一个浮点值 0.001，编译后的计算图在 clipfrac 这个输出上可能出现特化或别名问题。另一种可能是 clipfrac 走了别的统计路径。定位需要在训练 actor 中探针式记录两种 eps 下 clipfrac 与 loss 各自的逐 token 分量，或者在 GPU 上用 compiled 与 eager 两种方式对比同一输入。
- 结论：pg_clipfrac 不能作为 clip_higher 的生效证据（至少在 eps_clip == eps_clip_high 的配置下，它与 loss 不一致）。这一点记录为疑似缺陷（Miles / torch.compile），交由主 agent 判断是否转 FORK。
