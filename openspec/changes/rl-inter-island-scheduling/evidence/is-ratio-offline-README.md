# tasks 0.10 离线 IS 比比较（S17 N6）

脚本：`tools/offline_is_ratio_compare.py`；单测：`tests/test_offline_is_ratio_compare.py`（3 passed）。

## 已有数据（只有 lag 0）
- `s1-runs/s16-verl-mismatch-20261008/.../E_train10_merge0/dump/step001-010.pt` 与 `C_merge1_3step/dump/step001-003.pt`：
  每步一批新采样（128×1024），含 `rollout_log_probs`（推理端）与 `old_log_probs`（同版本训练端重算），即仅训推不一致，**lag=0**。
  各步 `responses` 不同，同一批样本没有在后续版本下的 logprob，推不出 lag 1–4。
- 其它检查过的：`s1-runs/*/tape-direct/*.jsonl`、`*/pulled/rl-island-0.jsonl`、`*/home/yeto-output/*.jsonl`（只有步级指标，无逐 token logprob）；`*/yeto/tests` 下是代码副本的测试夹具。

## lag 0 结果（`is-ratio-lag0-verl-s16.json`，121.7 万 token）
raw var=0.00156, ESS/N=0.9984；TIS[0,2] 改动比例 4.9e-6；IcePop[0.5,2] 1.3e-5；M2PO(τ=0.04) 0。
三者在 lag 0 无可区分差异 → **不能据此定默认值**，暂维持 `tis`。

## 缺的数据与最便宜采集方案（下次上卡顺带）
需要：同一批样本的行为 logprob（推理端，版本 v）+ 训练端在版本 v+1..v+4 上对同一批 token 的 logprob，带 response_mask。
做法：训练 ≥5 步（lr>0，如 E_train10 配置），保留第 t 步 rollout 批（tokens+mask+rollout_log_probs），
在第 t+1..t+4 步更新后用当前 actor 只做一次前向重算 logprob，写 `lag{k}_step{t}.pt`（字段同现 dump）。
每次多一遍 128×1024 前向，秒级，几步即可；取 t=1..3 共 3 批 × 4 个 lag。然后 `--lag k` 跑本脚本。
