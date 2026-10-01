# L-D0 trainer 确定性确认实验结果（不计入 task），2026-09-30

计划：local-gpu-plan.md L-D0（判据未改）；执行安排 gpu-plan-v2 §9.11。代码 aeaf6e2，镜像 db815884…（miles 2f23a0fc）。Modal H100!:2，strict-avg 单岛 + 本机 head，T1R1，无 eval，`--rl-deterministic-trainer --rl-observe-timeline`，3 轮 seed 17。

| 判据 | 结果 |
|---|---|
| 1 D1、D2 均 rc=0，无 rl_strict_failure | 通过 |
| 2 v1/v2/v3 policy token 逐位相同 | **通过**（v1 `3660f9f…`、v2 `fdbd6e1…`、v3 `aa83be6…`，两次相同；逐轮 sample-id 哈希也相同） |

- GPU：D1 `GPU-0590b34b…`、`GPU-0ad203a6…`（驱动未记录，puller 当时未取）；D2 `GPU-673e59c4…`、`GPU-effac368…`，驱动 580.95.05。Miles 参数 `deterministic_mode = True`（launch.log）；确定性环境变量在 driver 进程设置、经 runtime_env 转发（dry-run 核对），各 rank 的实际值未在 launch.log 中回显 → 按计划记为"rank 侧环境未确认"。
- 结论（按计划结论规则）：确定性开关可用 → 按原判据重跑 A2 三个 arm（均加开关）。
- 观察（供分析，不改结论）：D1/D2 的 v1–v3 token 与第一次 A2 的 S arm（无确定性开关，代码 11911b8）完全相同，也与 infra-a 2.2 的 colocated arm A 相同；而第一次 A2 的 O arm 与 infra-a 2.2 第四轮 arm B（partitioned、无 eval）得到另一组 token（v1 `e5362df…`）。即同一配置族出现了两组可复现的结果，是否由确定性开关消除尚需 A2 重跑确认。

费用：D1 ≤$2.21（07:41:29–07:58:14），D2 ≤$2.00（07:59:19–08:14:31），合计 ≤$4.21。app ap-GUlSdTbuZFxW7C7wE5Irfi、ap-Cw1dBpxjvhql79BMd4A5gX stopped/0。
