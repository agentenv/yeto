# rl-algo-critic-family progress

## 6.1 参考实现（2026-10-06）
- `tests/rl_gae_reference.py`：独立 torch 参考（不 import Miles）——vanilla、length_adaptive（λ=1−1/(αl)，α=1.5）、decoupled（A 用 λ_policy，R=GAE(λ_critic)+V）、cross_segment（段内局部 GAE、段尾 bootstrap 0、终局回报在末段段尾、乘 (γλ)^{N_{>s}}）。
- `tests/test_rl_gae_reference.py`：8 passed（手算 3 元 vanilla、γ=λ=1 等于 MC、λ(100,1.5)=1−1/150、α→∞→λ=1、decoupled 手算、单段 cross_segment==vanilla、两段手算 ×(γλ)^{n_2}、三段因子）。

## 6.2 fork 扩展点（本地提交，未 push）
- 分支 `yeto-gae-variant`（worktree /home/michael/work/miles-gae，基于 `yeto/ports` 039471508），提交 `ce96fc060`。
- 改动：loss_hub/math_utils.py（`segmented_gae`、`length_adaptive_lambda`、`get_advantages_and_returns_batch` 新 kwargs）、loss_hub/advantages.py（缺省不传任何新 kwarg）、loss.py（传 `rollout_data["segment_ids"]`）、arguments.py（`--gae-variant/--gae-lambd-mode/--gae-length-alpha/--gae-critic-lambd`）、ray/rollout/train_data_conversion.py（`sample.metadata["segment_ids"]` → train data → 分片）。
- 语义：段与 N_{>s} 在可训练 token 子序列上计（与掩码 token 非 MDP 转移一致）；length_adaptive 的 l 用响应长度 R_i；returns = A + V（decoupled 时用 critic λ）。
- 测试（/home/michael/work/miles-next-venv）：`test_ppo_gae_variants.py` 22 passed（对拍拷贝的参考实现；缺省/显式 vanilla 与冻结的原实现 `torch.equal` 逐元素一致，fp32/fp64、chunked/非 chunked、带掩码）；`test_segment_ids_conversion.py` 2 passed。
- 定向回归（GAE/loss 相关文件 + tests/test_chunked_gae.py）：基线 039471508 13 failed/99 passed，新 13 failed/121 passed，失败集合完全相同（megatron.core 缺失等环境原因）。未跑 tests/fast/ray 全量（会启动 Ray，线程上限）。
- yeto 全量回归：log /home/michael/work/infra-drafts/critic-gae-pytest.log，68 failed/26 errors，与 /tmp/base2.sorted 按用例 id 比对集合相同，新增失败 0。
