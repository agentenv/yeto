# progress — rl-algo-supplement

## S14 对账（2026-10-07，子 agent S1；基线 integ-decl e8d387ac，无 GPU）

来源：change 原文取自分支 `algo-supplement` dbced65d（唯一提交，仅 openspec 文件，无代码改动）。对账只认 e8d387ac 中的文件:行、其它 change 的 tasks/progress、infra-drafts 记录；状态五档：已实现+已验证 / 已实现未验证 / 部分 / 未做 / 已取代。

### 汇总
| 状态 | 数 | 项 |
|---|---|---|
| 已实现+已验证 | 1 | 1.2 |
| 已实现未验证 | 0 | — |
| 部分 | 0 | — |
| 未做 | 38 | 其余全部（其中需 GPU 24 项：4.1–4.7、5.1–5.3、6.1–6.2、8.2、9.2、10.3、11.3、12.2 及 2.5 的 T4；纯 CPU 14 项） |
| 已取代 | 0 | 无本 change 内的项被取代；design D9 的卡型假设（A10G）与 S14 规则（Modal H100!）冲突，待拍板 |

### §1 1.2 三项核实结论
1. trainer v2（GMPO `gmpo_clip_num/den` 取数）**已合入**：`yeto/rl/engine/miles_adapter/trainer.py:101`（注释引用 D5）、`:601-611`（`gmpo_clip_fraction(step_losses)`）。GMPO 不暂缓。
2. launcher 回传磁带 harness 可复用：`rl-algo-seq-and-adv/progress.md:178`（attempt 6，`--rl-single-island-no-sync --controller local`，5 个 run PASS）；critic-family G1 的 `s1-runs/s13-forkg1-modal.sh`（`rl-algo-critic-family/progress.md:489-547`）同一形态。两岛形态：`s13-g3-modal-20261007e/f` 远端 controller（Nebius VM）已跑通但 0 轮（`infra-drafts/gpu-spend.md` 08:54/09:39 行）。
3. 当前 pin `MILES_NEXT_COMMIT = c35702e`（`yeto/rl/__init__.py:43`）；`loss_variants.FORK_COMMITS` 含 5c1b49eb…c35702e（`yeto/rl/algos/loss_variants.py:78-90`）；`_PINS_0AF62F4D_PLUS` 7 个 pin（`entry.py:143-180`）。

### 逐项对账
| task | 状态 | 证据 / 说明 |
|---|---|---|
| 1.1 | 未做 | `evidence/phase1-plan.md` 不存在 |
| 1.2 | 已实现+已验证 | 见 §1 |
| 1.3 | 未做 | 无 `baseline-failures.txt`；最近全量 `infra-drafts/s13-full2-pytest.log` 末行 68 failed/4084 passed/51 skipped/26 errors，HEAD 未记录 |
| 2.1 | 未做 | `rl-algo-grpo-knobs/evidence/2026-09-29-clipfrac-offline/report.md`：只有手算对照与"可能原因（未确认）"，无 compile/eager 对照；grpo-knobs progress.md:75-76 记 g1 中 clip_higher/dual_clip pg_loss 与 baseline 逐位相同 |
| 2.2 | 未做 | 依赖 2.1 |
| 2.3 | 未做 | `git -C /home/michael/work/miles-next grep dual_clipfrac c35702e -- miles` 为空 |
| 2.4 | 未做 | fork c35702e 只读 `over_sampling_batch_size`（`miles/rollout/inference_rollout/inference_rollout_train.py:120`、`sglang_rollout.py:483`），不记录提交批大小 |
| 2.5 | 未做 | 无 2.2–2.4 提交可合并；pin 迁移论证模板已存在（entry.py:143-180 注释，IMG 五次 pin 变更）；建议与 critic-family 6.3 合并 |
| 2.6 | 未做 | `grep -rn dual_clipfrac yeto tests` 为空 |
| 3.1 | 未做 | 无"不可声明"名单；多岛拒绝放行的单测已有 `tests/test_rl_algorithm_capabilities.py:413,428` |
| 3.2 | 未做 | `docs/MILES_RL.md:857-905` Declaration policy 无"用户代码类永不声明"规则；A12 审计（HANDOFF-AUDIT-S14.md:21）文档多处过时 |
| 4.1–4.7 | 未做（GPU） | `MILES_DECLARED`（entry.py:67-141）无 losses:cispo/sapo/gmpo、features:dual_clip、kl 估计器 k1/k2/low_var_kl、kl_unbiased、corrections:opsm_rollout、features:no_rewards_normalization、whiten；loss-variants 6.1–6.4 未勾；mismatch progress.md:48 "opsm_rollout 未跑" |
| 5.1 | 未做（GPU） | entry.py:106-114 over_sampling 证据仍为"inferred afterwards…strong evidence awaits a rerun" |
| 5.2 | 未做（GPU） | entry.py:123-141 clip_higher 仍"RAN ON Miles 0394715…code-diff argument" |
| 5.3 | 未做 | 依赖 2.2 |
| 6.1 | 未做（GPU） | = grpo-knobs 8.4（tasks.md:62 未勾） |
| 6.2 | 未做（GPU，可选） | loss-variants 6.3 未勾 |
| 6.3 | 未做 | — |
| 7.1–7.2 | 未做 | 无 `evidence/phase2-plan.md`；`rl-infra-spec/alignment.md` 无 ref 版本字段请求（`grep -n ref_update` 为空） |
| 8.1–8.2 | 未做 | `algorithm_flags.py:250-253` `--rollout-temperature/top-p/top-k` 仍在 `_UNMAPPED`；`SamplingSpec`（algorithm.py:820-826）无 temperature/top_p/top_k |
| 9.1–9.2 | 未做 | `KlSpec`（algorithm.py:735-741）无 ref_update_interval；`grep -rn ref_update yeto` 为空 |
| 10.1–10.3 | 未做 | `grep -rn loss_mask_policy yeto` 为空 |
| 11.1–11.3 | 未做 | routing replay 仍是 agent 字段：`run_config.py:405`、`miles_adapter/config.py:430,919`、`learner.py:1616`；`--use-routing-replay` 未在 MAPPINGS |
| 12.1–12.3 | 未做 | `grep -rni rloo yeto` 为空 |

### 其它 change 未完项的汇入
见 `tasks.md` §14（critic-family 4.5/6.3/7.3/8.4/9.4/9.5/10.x + 正式声明；loss-variants 6.1–6.4，3.1–3.4 已取代；grpo-knobs 8.4/8.5/8.6；mismatch 5.2/7.3/7.7；decoupled-lr 3.2；seq-and-adv 3.6 已知限制）。来源 change 的勾选未改。

### 需用户拍板
- Q1 卡型：阶段一按 design D9 用 A10G（≈$26，上限 $35）还是按 S14 规则用 Modal `H100!`（≈$43，上限需提至 ≈$45）。A10G 24 GB 跑 Qwen2.5-0.5B LoRA 单岛应够，但 S13/S14 所有 G1 先例都在 H100!，两者证据不能逐位互比（D2 不要求）。
- Q2 是否仍要阶段一全部 4.x（10 项 G1）；最小集建议 4.1/4.2/4.4/5.1/5.2（直接解除 HANDOFF-AUDIT A1 的四个遗留），≈$8–13。
- Q3 pin 更新是否与 critic-family 6.3 合并为一次（含 fork 推送授权）。
- Q4 汇入的 critic-family G3（4.5/7.3/8.4，≈$24）是否在本阶段额度（$400，S14 末已用 ≈$300，剩 ≈$100；本 session 另有 ≈$33 复验在跑）内排期，排在阶段一之前还是之后。
- Q5 mismatch 5.2：接受 vendor 副本，还是放开 `examples.` 插件命名空间。
- Q6 loss-variants 3.1–3.4 已被路线 B 取代，是否允许在来源 change 标注关闭（本次未改）。

### 验证
- `openspec validate rl-algo-supplement`（含 `--strict`，openspec 1.13.2）：valid。
- 未跑测试（本次只改 openspec 文件）。

## S14 用户裁定（2026-10-07）
- Q1 卡型：能用便宜卡就用便宜卡（A10G 优先），万不得已再开贵卡（H100!）。
- Q2 范围：阶段一 4.x/5.x **全做**。
- Q3 pin 更新与 critic-family 6.3 合并做（fork push 授权仍沿用"overlay 补丁代替 push"，除非用户另说）。
- Q4 critic G3 排在算法阶段一之后。
- Q5/Q6：待用户看完解释后裁定。
