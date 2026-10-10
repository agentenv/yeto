# 阶段一 GPU 结果（S19 第三批 #13，2026-10-09/10）

判据按 `phase1-plan.md` 与 `infra-drafts/S19-ALGOSUP-PRELAUNCH-REVIEW.md` §4、§7（开卡前写定），没有放宽。
用词：通过 = 上卡按预登记判据验过；失败 = 判据不满足；失败（证据不全）= 缺少判据要求的数据；未验证 = 没有取得能判的数据。

## 共同设定
- 代码：分支 `s19-algosup-run`（agentenv/main a7917d64 + A10G 卡名修正）。镜像 `64b591a-2fa8801@sha256:fa2413be…`，事件 `miles_commit` = 64b591a4bec1…（每跑都检查）。
- 模型 Qwen3-0.6B@c1899de2，gsm8k@0cbd9f31，LoRA r16 all-linear，4 组×8 条，seed 17，3 轮×2 步（5.2 为 6 组、3 步、lr 1e-4）。卡：Modal 1×A10G（容器报 `NVIDIA A10`），6.1 为 2×A10G + Nebius CPU head。
- 判读脚本 `scripts/s19-p1-judge.py`（G1）、`scripts/s17-g1-knobs-judge.py`（6.1）；判读 json 在 `judgments/`。原始数据在 Modal 卷 `yeto-evidence-archive:s19/algosup/<run>.tgz`（登记于 infra-drafts/ARCHIVE-MANIFEST.tsv）。

## 逐项结论
| 项 | run | 结论 | 关键数据 |
|---|---|---|---|
| 4.1 对照（新 pin） | s19-p1-base-20261009e | **通过**（G） | 3 轮，loss/grad_norm 有限；spec 27df1133… |
| 4.1 旧 pin 复核 | s19-p1-baseold-20261010b | **通过**（G） | miles_commit ddce2099…，spec 27df1133… |
| 4.2 CISPO | s19-p1-cispo-20261010a | **通过** (a)(b)(c)(d)(f) | 第 1 轮第 2 步 pg_loss −0.0324 vs 对照 0.000166；pg_clipfrac 0.00136/0.00293/0.00150 |
| 4.2 SAPO | s19-p1-sapo-20261010a | **通过** (a)(b)(c)(d) | 第 1 轮第 2 步 pg_loss −0.0148 vs 0.000166；grad_norm 0.34926 vs 0.34914 |
| 4.2 GMPO | s19-p1-gmpo-20261010a | **通过** (a)(b)(c)(d)(e) | num/den 0/71.4、0/127.7、0.0625/171.7；clip_fraction = num/den |
| 4.3 CISPO+TIS | s19-p1-cispo-tis-20261010b | **通过** (a)(b)(c) | TIS clip [0.99, 1.01] |
| 4.3 SAPO+TIS | s19-p1-sapo-tis-20261010b | **通过** (a)(b)(c) | |
| 4.3 GMPO+TIS | s19-p1-gmpo-tis-20261010b | **通过** (a)(b)(c)(e) | num/den 0/71.4、0/79.7、0.031/205.3 |
| 4.4 dual_clip | s19-p1-dual-20261010a | **通过** | dual_clipfrac 0.00299/0.0341/0.0371 |
| 4.5 k3（对照） | s19-p1-kl-k3-20261010a | 对照，满足 G | kl_loss 0.000268/0.000882/0.001039 |
| 4.5 k1 | s19-p1-kl-k1-20261010a | **失败**（判据缺陷，非实现缺陷） | 第 3 轮 kl_loss = −0.000642，不满足预登记的"> 0"。k1 估计器（log ratio）本身可以为负，所以这条判据与 k1 不匹配；运行本身无故障。主 agent 代拍板：结论按预登记记失败，本次不重跑；后续见 phase1-plan §6。 |
| 4.5 k2 | s19-p1-kl-k2-20261010a | **通过** | 第 2 轮 0.000888 ≠ k3 0.000882 |
| 4.5 low_var_kl | s19-p1-kl-lowvarkl-20261010b | **通过** | 第 2 轮 0.000862 ≠ k3 |
| 4.5 kl_unbiased | s19-p1-kl-unbiased-20261010b | **通过** | 第 2 轮 0.000718 ≠ k3 |
| 4.6 opsm_rollout | s19-p1-opsm-rollout-20261010a | **通过** | opsm_clipfrac 0.0625/0.219/0.25；事件 spec `opsm_old_logprob_source=rollout`、`use_rollout_logprobs=true` |
| 4.7 no_rewards_normalization | s19-p1-nonorm-20261010a | **失败（证据不全）** | 均值与对照不同（rollout/advantages 0.9375/0.4375/0.0625 vs 对照 ≈0），且等于该轮 reward_mean（均值离线一致）。方差：Miles 与 tape 都不输出（adv_std 为 null），无法离线重算比对。 |
| 4.7 grpo+whiten | s19-p1-whiten-20261010a | **失败（证据不全）** | 均值与对照不同（0.0614/0.128/0.0692）；方差同上缺失。 |
| 5.1 over_sampling | s19-p1-os-20261010b | **通过** | 3 轮 submitted_groups 8 > 4；filtered_groups 4/3/2 |
| 5.2 clip_higher | s19-p1-clip-sym-20261010b / s19-p1-clip-hi-20261010c | grad_norm 部分**通过**；clipfrac 与离线重算一致**未验证** | 第 1 轮第 1 步 grad_norm 两臂相等 0.8681032（配对有效）；第 3 步 0.40779 vs 0.42270（生效）。取不到逐 token ratio，clipfrac 离线重算未做。 |
| 5.3 | — | 不适用 | 2.2 未改 fork（见 phase1-plan） |
| 6.1 G3 两岛 strict-avg | s19-p1-g3-20261010a | **通过** | 两岛 spec sha bf55a45f… 相同、无放行项；v1–v3 两岛权重哈希一致；3 轮/岛；无失败事件；过滤样本 0/3/6–7 已记录 |
| 6.2 | — | 未做（按计划，4.x 不需要） | |

## 未开卡记录（$0）
base a–d、clip-sym a、clip-hi a/b、kl-lowvarkl a、kl-unbiased a、baseold a：本机线程超过门限，yeto preflight 或脚本线程检查拒绝，未建云资源。os a：脚本参数拆分错误，未起跑。cispo-tis a：排队等线程时被另一会话停下，未开卡。

## 发现
- k1/k2/low_var_kl 在 dry-run 中不需要放行（估计器不是独立机制名）。"每项单独声明"需要新增机制名，属代码改动，未做。
- 4.7 的方差数据当前拿不到：要判 4.7，需要 Miles 输出 advantage 方差，或 tape 填 adv_mean/adv_std。
