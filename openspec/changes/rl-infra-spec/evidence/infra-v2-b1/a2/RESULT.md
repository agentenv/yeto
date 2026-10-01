# A2（L-2.3，task 2.3）GPU 结果，2026-09-30

计划：`gpu-plan-v2.md` §9.7；判据：`local-gpu-plan.md` L-2.3 判据 1–6（含"追加"），运行前提交，未修改。
代码 11911b8，镜像 `sha256:17d428a2…`（miles 5c1b49eb）；Modal `H100!:2`（`--modal-gpu-exact`，gpu.txt 两张均为 NVIDIA H100 80GB HBM3）；T1R1 fixed-partition，strict-avg 单岛 + 本机 head，3 轮，seed 17，eval 每轮一次，gsm8k test 前 32 条（N=32），temperature 0。
分析脚本：`analyze_a2.py`，输出 `RESULT-analysis.json`。

| 判据 | 结果 | 说明 |
|---|---|---|
| 1 三 arm 正常结束，无 rl_strict_failure / OverlapGuardError | 通过 | 三个 rc=0 |
| 2 S 与 O 逐轮 sample-id 哈希、组数/样本数相同；applied_lrs 长度 1 | 通过 | 3 轮完全相同 |
| 3 O/OD 的 overlapped eval token = eval 开始时最近发布；窗口内无 publication/generate | 通过 | O、OD 各 3 个 overlapped eval，无违例 |
| 4 OD：4 次发布前都有 rl_fault_injected；generate 版本 = 最近发布；overlap_start 与 rl_eval 严格交替 | 通过 | |
| 5 硬条件：S 与 O 每个 eval 点的 policy_version 与 rl/policy_token 相同 | **未通过** | v0 相同；v1/v2/v3 的 token 不同（S `3660f9f…`/`fdbd6e1…`/`aa83be6…`，O `e5362df…`/`6364c8c…`/`6c1a1bb…`）。分数条件本身满足（差 ≤ 1/32 ≤ 2/N），但硬条件不满足即判未通过 |
| 6 O 中 ≥2 个 eval 点真实 eval 区间与同轮 train span 有交集 | 通过 | 3 个点（rollout 0/1/2）交集 2.0–2.8 s |

**结论：2.3 GPU 验收未通过（未完成）**，按已提交判据 5，不勾选。1.4 的 X9 也因此不可勾选。

原因分析（不改判据，供后续）：S 与 O 采样身份和奖励逐轮相同（`rl_local_round.reward_mean` 三轮都相等），但第 0 轮训练后的策略哈希不同。这与 L-2.3"追加"中事先登记的已知差异 L3(a) 一致：O 把 eval(v_r) 挪到 generate(r) 之后，SGLang 引擎内采样 RNG 的消耗顺序与 S 不同，生成文本（及 logprob）不同，而判据 2 只比较样本身份。旁证：infra-a 2.2 第四轮的 arm B（T1R1 partitioned、无 eval，generate 前不消耗 eval RNG）在 v1 的 token 恰为 `e5362df…`，与本次 O 相同；在 generate 之前先跑 eval 的 S 则得到 `3660f9f…`（与 2.2 的 colocated arm A 相同）。要让判据 5 可满足，需要代码侧让 eval 不消耗训练 rollout 的采样 RNG（例如 eval 使用独立 seed/采样器），或由主 agent/用户另行裁定判据——本 agent 不修改判据。

费用：S ≤$2.09、O ≤$2.55、OD ≤$2.34，合计 ≤$6.98。
资源：ap-6LOJzBHdWX0MFSvASIHepv、ap-BekUE6S4mODusFocxFMQXD、ap-VCWa0odqdRqAYIU0Wj5tMX 均 stopped/0（各 arm `modal_app.txt`）；本地 head/syncer/watchdog 已终止。
