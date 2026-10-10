# 阶段一 GPU 测试计划（tasks 1.1；起草 2026-10-09，未上卡）

状态：**计划已定稿，主 agent 10-09 代用户确认**（确认记录见 `progress.md` "S19 阶段一门禁"）。本文件只是预登记，判据写定后不放宽（design D2、D9）。所有费用都是估算上限，未实测。

## 0. 确认记录
- 确认人：主 agent，2026-10-09，代用户确认。
- 理由：沿用用户在 S14 已给出的裁定（progress.md "S14 用户裁定"）：Q1 先用 A10G；Q2 阶段一 4.x/5.x 全做；Q3 pin 更新与 critic-family 6.3 合并为一次。
- 6.2（CISPO 两岛，可选）：**不做，除非 4.x 的结果需要**。需要时另行登记并确认。
- 2.5 fork：**不真推送** `yeto/ports`，继续用 overlay 补丁（主 agent 代拍板；理由：推送要额外审批，overlay 方式已在 critic-family 验证可用）。
- 本计划不是上卡授权。上卡前仍要：第三批报批（`infra-drafts/S19-BATCH3-PLAN.md` #13）、每批复核文档、`gpu-spend.md` 预登记、本机线程 < 3000。

## 1. 共同设定（design D2、D9）
- 模型：Qwen2.5-0.5B LoRA；单岛 colocated-serial；seed 17；只跑一次，不挑 seed。
- `num_steps_per_rollout=2`，3 轮（除非单项另写）；`max_policy_staleness=0`。
- 卡型：Modal A10G 单卡（两岛项为 A10G 1+1）；T4 只用于不训练的解析检查。
- 每次运行：Modal 函数 `timeout=` + 本地 `timeout` + 独立 watchdog；唯一前缀；结束先拉证据再 `stop -y`，并给出无残留证明。
- 通用判据（G）：loss 与 grad_norm 有限；零梯度不变量无误报；事件 `algorithm_spec_sha256` 的规范化 JSON 含该机制；`miles_commit` 等于新 pin（含 overlay 时记录 overlay sha256）。
- 对照组：4.1 的默认 GRPO 运行，同一 pin，所有 G1 项共用。
- 未通过：停下，修复原因后最多重跑一次（用预备金）；修不了就记为合法否定结论。
- 累计达阶段上限 80%（$28）时停下报告；单项超出自身上限立即中止，记为未完成。

## 2. 逐项计划
前缀格式：`s19-p1-<项>-<YYYYMMDD><字母>`。

| 项 | 内容 | 判据（预登记） | 最小证据 | 对照 | 卡 | 硬超时 | 上限 |
|---|---|---|---|---|---|---|---|
| 2.5 | 新 pin（ddce209 + overlay：critic-family + 本 change 2.3/2.4）parse_args | 完整 `parse_args` + `validate_parsed_args` 通过；`--eps-clip-c`、`--over-sampling-batch-size` 解析无误 | 容器日志、overlay 记录 | 无 | T4 | 15 min | $0.5 |
| 4.1 | 默认 GRPO 对照（新 pin；另在旧 pin 复核一次） | G；指标 jsonl 完整 | `evidence/g1-baseline/` | — | A10G | 40 min/次 | $2.0 |
| 4.2 | CISPO、SAPO、GMPO 各一次 | loss-variants progress 修订版 (a)–(f)，不改 | `evidence/g1-variants/results.md` | 4.1 | A10G ×3 | 40 min/次 | $4.5 |
| 4.3 | 4.2 通过的变体 + `corrections:tis` | (a)(b)(c)；GMPO 另加 (e) | `evidence/g1-variants-tis/` | 4.1 | A10G ×≤3 | 30 min/次 | $3.0 |
| 4.4 | dual_clip，`eps_clip_c=1.01` | 每轮 `dual_clipfrac` > 0 且有限；G | `evidence/g1-dual-clip/` | 4.1 | A10G | 40 min | $1.5 |
| 4.5 | KL k1、k2、low_var_kl、kl_unbiased（placement=loss，coef 0.01，显式 ref） | 第 1 轮之后 `kl_loss` 存在、有限、> 0，且与已声明 k3 运行数值不同；G | `evidence/g1-kl/` | 4.1 + k3 证据 | A10G ×4 | 40 min/次 | $4.0 |
| 4.6 | opsm_rollout（1a trigger 小阈值） | `opsm_clipfrac` > 0，且来源记录显示 π_old 取自 rollout logprob；G | `evidence/g1-opsm-rollout/` | 4.1 | A10G | 40 min | $1.5 |
| 4.7 | no_rewards_normalization、grpo+whiten 各一次 | advantage 均值与方差和对照不同，并与离线重算一致（容差 1e-5）；G | `evidence/g1-adv-norm/` | 4.1 | A10G ×2 | 40 min/次 | $2.5 |
| 5.1 | over_sampling 强证据（g1h 动态过滤配置，调高过滤强度） | 至少一轮 `rl/over_sampling/submitted_groups` > rollout batch size（组数），并且 `filtered_groups` > 0；G | `evidence/g1-over-sampling/results.md` | 4.1 | A10G | 50 min | $2.0 |
| 5.2 | clip_higher 在新 pin 独立重跑（g1j 配置） | g1j 预登记的第 3 步 grad_norm 不相等；clipfrac 与离线重算一致（2.1 的 GPU 残余疑点在此检查） | `evidence/g1-clip-higher/` | 同次 A/B 两臂 | A10G | 40 min | $1.5 |
| 5.3 | clipfrac 修复后复核 | **不适用**：2.1 在 CPU 上不能复现，2.2 未改 fork | — | — | — | — | $0 |
| 6.1 | 1b 8.4 G3：两岛 strict-avg（clip-higher + token 聚合 + overlong 软惩罚与过滤），3 轮，不带放行参数 | 两岛算法哈希一致；每轮外层同步后权重哈希一致；不变量无失败；两岛有效样本数已记录 | `evidence/g3-grpo-knobs/` | 无 | A10G 1+1 | 60 min | $4.0 |
| 6.2 | CISPO 两岛（可选） | **不做，除非 4.x 结果需要** | — | — | — | — | $0（如启用另登记，≤ $3.5） |
| — | 预备金（只用于已记录原因后的一次重跑） | — | — | — | — | — | $3.0 |

## 3. 汇总
| 部分 | 上限 |
|---|---|
| 2.5 T4 | $0.5 |
| 4.1–4.7 | $19.0 |
| 5.1–5.3 | $3.5 |
| 6.1 | $4.0 |
| 预备金 | $3.0 |
| **合计** | **$30.0（≤ $35）** |

6.2 若启用，合计 $33.5，仍 ≤ $35。

## 4. 顺序
1. 2.5（与 critic-family 6.3 合并为一次 pin/overlay 检查）。
2. 4.1 对照。
3. 4.2、4.4、5.1、5.2（解除 HANDOFF-AUDIT A1 的四个遗留）。
4. 4.5、4.6、4.7，然后 4.3。
5. 6.1。

## 5. 上卡结果（S19 #13 执行子 agent，2026-10-10 回填）
逐项结论、run id、关键数据见 `evidence/phase1-gpu/results.md`；判读 json 在 `evidence/phase1-gpu/judgments/`。摘要：
- 通过：4.3 三项、4.1（新 pin + 旧 pin 复核）、4.2 CISPO/SAPO/GMPO、4.4、4.5 k2/low_var_kl/kl_unbiased、4.6、5.1、6.1。
- 失败：4.5 k1（第 3 轮 kl_loss < 0）。判据缺陷，非实现缺陷：k1 估计器本身可为负。主 agent 代拍板，按预登记记失败，本次不重跑。
- 失败（证据不全）：4.7 两项（advantage 方差无数据）。
- 5.2：grad_norm 判据通过；clipfrac 离线重算未验证。
- 4.3：CISPO/SAPO/GMPO + TIS 三项通过（*-tis-20261010b）。
- 声明（MILES_DECLARED_PINS 等）未改：属代码改动，留给后续 PR。

## 6. 后续（不在本次上卡范围）
- 4.5 k1：如需重测，先预登记适合 k1 的判据再上卡，例如"3 轮 kl_loss 均值 > 0 且有限"，或"与同 seed k3 运行逐轮同号、量级一致"。新判据要经确认后再开卡。
- 4.7：需要先输出 advantage 方差（Miles 或 tape 的 adv_std），再重测。
- 5.2：需要逐 token ratio 输出，才能做 clipfrac 离线核对。
