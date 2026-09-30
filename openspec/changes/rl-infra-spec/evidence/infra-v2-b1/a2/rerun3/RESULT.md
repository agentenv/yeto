# A2（L-2.3）第三次：三 arm 同一 SHA 37155d8，全部判据通过（2026-09-30）

配置：gpu-plan-v2 §9.7/§9.13（Modal H100!:2，T1R1 fixed-partition，strict-avg 单岛 + 本机 head，3 轮 seed 17，eval 每轮、N=32、temperature 0，`--rl-observe-timeline --rl-deterministic-trainer --modal-retries 0`）；镜像 2cc5cc52…（Miles e3a11ab3）。判据 L-2.3 1–6 原文，未改。分析：`analyze_a2.py` → `RESULT-analysis.json`。
S = `S/`（09:10–09:30），O = `O/`（09:30–09:45），OD = `OD/`（11:44–12:02，`infra-v2-b1-a2od-20260930-4`；用户暂停时中止的那次保留在 `OD-paused-rc143/`，不参与判定）。

| 判据 | 结果 |
|---|---|
| 1 三 arm rc=0，无 rl_strict_failure / OverlapGuardError | 通过 |
| 2 S=O 逐轮 sample-id 哈希、组/样本数；applied_lrs 长度 1 | 通过 |
| 3 O、OD overlapped eval token = eval 开始时最近发布；窗口内无 publication/generate | 通过（各 3 个 overlapped eval） |
| 4 OD：4 次发布前均有 rl_fault_injected；generate 版本 = 最近发布；overlap_start/rl_eval 严格交替 | 通过 |
| 5 S 与 O 每个 eval 点 policy_version 与 token 相同；分数差 ≤ 2/N | 通过（v0–v3 token 相同；分数 0.5625/0.5625/0.53125/0.53125 两边相同） |
| 6 O 中 ≥2 个 eval 点真实 eval 区间 ∩ 同轮 train span > 0 | 通过（rollout 0/1/2：2.55/2.03/2.01 s） |

结论：2.3 X9 GPU 验收**通过**。按 L-2.3 "通过后"条款，1.4 的 X9 由本实验与 partitioned-serial guard（infra-a 第三轮 C）共同满足。
费用：S3 ≤$2.58、O3 ≤$2.00、OD（补跑）≤$2.34；中止的 OD3 ≤$1.59。app 均 stopped/0。
