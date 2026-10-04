# A2 重跑（--rl-deterministic-trainer），2026-09-30：O arm rc=3，链条停止，OD 未跑

代码 aeaf6e2，镜像 db815884…；§9.7 配置 + `--rl-deterministic-trainer --modal-retries 0 --modal-timeout-s 3000`；判据 L-2.3 1–6 不变。

- S2：rc=0（08:16:32–08:38:26）。O2：**rc=3**（08:38:43–08:55:55）。按"失败即停"，OD2 未启动。
- O2 的 rc=3 原因：launcher `event tape incomplete … exit 3`——`rl_learner_finalized` 事件确实由岛输出（launch.log 第 6102 行 `YETO_RL_EVENT {"event":"rl_learner_finalized",…}`），但 launcher 在 6107 行检查本地磁带时该记录尚未写入（同时报"1 malformed prefixed line discarded"），把磁带标为 `.incomplete` 并返回 3。learner 作业 SUCCEEDED，3 轮训练、4 次发布、4 个 eval 都完整。这是 launcher 日志流收集磁带的竞态/截断（代码侧），不是运行语义失败；但判据 1 写明"退出码 0"，故 O2 判据 1 **不满足**，不放宽。
- 其余判据（S2 对 O2，用 analyze_a2.py；OD 缺，判据 4 未评估）：判据 2 通过；判据 3（O）通过；**判据 5 通过**（v0–v3 token 完全相同：`a34cd30…`/`3660f9f…`/`fdbd6e1…`/`aa83be6…`；eval 分数逐点相同 0.5625/0.5625/0.53125/0.53125）；判据 6 通过。与第一次 A2（无开关，O 的 token 在 v1 起不同）对比，说明 `--rl-deterministic-trainer` 消除了第一次的判据 5 差异。
- 结论：2.3 仍**未完成**（O2 退出码 3，OD2 未跑）。需要代码修复：launcher 在判定磁带完整前应等待/冲刷日志流中的最后事件（或以 learner 作业 SUCCEEDED + 日志中出现 rl_learner_finalized 为准）；修复后重跑 O 与 OD（S2 可复用，同一 SHA 时）。
- 费用：S2 ≤$2.89，O2 ≤$2.26。app ap-Ng0LEhJJZhGyoZgFAsU6lc、ap-r8FStYbeGiGyDk8fTmU4nS stopped/0。
