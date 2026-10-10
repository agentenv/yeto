# 阶段一 GPU 结果（S19 第三批 #13，2026-10-09/10）

- 判据：`phase1-plan.md` §1–2，读法见 `infra-drafts/S19-ALGOSUP-PRELAUNCH-REVIEW.md` §4、§7（开卡前写定，未放宽）。
- 镜像：yeto-miles-ports 64b591a-2fa8801@sha256:fa2413be（Miles 64b591a4b，无 overlay）。旧 pin 复核用 ddce209-2fa8801。
- 代码：分支 s19-algosup-run（agentenv/main a7917d64 + A10G 卡名修正）。
- 卡：Modal 1×A10G（容器报 "NVIDIA A10"），6.1 为 2×A10G + Nebius CPU head。
- 模型与数据：Qwen3-0.6B@c1899de2 LoRA r16，gsm8k@0cbd9f31，seed 17，4 组×8 条，3 轮×2 步（5.2 为 6 组、3 步、lr 1e-4）。
- 判读脚本：`s19-p1-judge.py`（G1）、`s17-g1-knobs-judge.py`（6.1）。逐跑判读 json 在 `judgments/`。原始数据在本机 `/home/michael/work/s1-runs/<run>/`，归档见 ARCHIVE-MANIFEST.tsv。

| 项 | run | 结论 | 关键数据 |
|---|---|---|---|
| 4.1 | s19-p1-base-20261009e | 通过 | G 全满足；miles_commit 64b591a4b；规格 sha 27df1133… |
| 4.1 旧 pin 复核 | s19-p1-baseold-20261010b | 通过（只判 G） | miles_commit ddce2099 |
| 4.2 CISPO | s19-p1-cispo-20261010a | 通过 | (a)–(f) 满足；第 1 轮第 2 步 pg_loss −0.0324 vs 对照 0.000166 |
| 4.2 SAPO | s19-p1-sapo-20261010a | 通过 | (d) pg_loss −0.0148 vs 0.000166 |
| 4.2 GMPO | s19-p1-gmpo-20261010a | 通过 | (e) num/den 0/71.4、0/127.7、0.0625/171.7，clip_fraction 一致 |
| 4.3 | 见下文 | 待补 | — |
| 4.4 dual_clip | s19-p1-dual-20261010a | 通过 | dual_clipfrac 0.0030 / 0.0341 / 0.0371 |
| 4.5 k3 对照 | s19-p1-kl-k3-20261010a | 对照 | kl_loss 0.000268 / 0.000882 / 0.001039 |
| 4.5 k1 | s19-p1-kl-k1-20261010a | **失败** | kl_loss 0.000165 / 0.000532 / **−0.000642**；判据要求 >0。k1 估计器（log ratio）本来可以为负，这是判据与估计器不匹配，不是代码故障；按 D2 不放宽，记为合法否定结论 |
| 4.5 k2 | s19-p1-kl-k2-20261010a | 通过 | 0.000278 / 0.000888 / 0.001007（≠ k3） |
| 4.5 low_var_kl | s19-p1-kl-lowvarkl-20261010b | 通过 | 0.000281 / 0.000862 / 0.001007 |
| 4.5 kl_unbiased | s19-p1-kl-unbiased-20261010b | 通过 | 0.000245 / 0.000718 / 0.000954 |
| 4.6 opsm_rollout | s19-p1-opsm-rollout-20261010a | 通过 | opsm_clipfrac 0.0625 / 0.219 / 0.25；spec opsm_old_logprob_source=rollout |
| 4.7 no_rewards_normalization | s19-p1-nonorm-20261010a | **失败（证据不全）** | rollout/advantages 均值 0.9375/0.4375/0.0625（对照 ≈0），等于该轮 reward_mean；方差 Miles 与磁带都不输出（adv_std 为 null），无法做 1e-5 离线核对 |
| 4.7 grpo+whiten | s19-p1-whiten-20261010a | **失败（证据不全）** | 均值 0.061/0.128/0.069（对照 ≈0）；方差缺失，同上 |
| 5.1 over_sampling | s19-p1-os-20261010b | 通过 | 第 1 轮 submitted_groups 8 > 4，filtered_groups 4 |
| 5.2 clip_higher | clip-sym-20261010b / clip-hi-20261010c | grad_norm 部分通过；clipfrac 核对**未验证** | 第 1 步两臂 grad_norm 都是 0.8681032（配对有效）；第 3 步 0.40779 vs 0.42270（生效）；取不到逐 token ratio，clipfrac 离线重算未做 |
| 5.3 | — | 不适用 | 2.2 未改 fork |
| 6.1 G3 | s19-p1-g3-20261010a | 通过 | 两岛 sha bf55a45f… 相同、无放行项；v1–v3 两岛全局策略哈希一致；无失败事件；filtered_samples 0/0、3/3、6/7 |
| 6.2 | — | 未做 | 4.x 结果不需要 |

## 未解决
- 4.5 k1：若要声明 k1，需要改判据（例如 |kl_loss|>0 或与 k3 不同），这需要用户重新确认，本次不改。
- 4.5 "每项单独声明"需要按估计器区分的机制名（代码改动，未做）。
- 4.7：要判通过，需要 Miles 或 yeto 输出逐样本 advantage 或其方差（代码改动，未做）。
- 5.2 clipfrac 离线核对：需要逐 token ratio 输出。
