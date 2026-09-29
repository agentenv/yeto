# 2.4 固定配置扫描（INFRA，事前计划；本文件单独提交，得到 SHA 后才启动）

## 验收原文
"在云实验池扫描少量固定配置，记录默认兼容配置、目标 profile 最佳固定和收益面；验收：同 profile 公平比较、全池/备用 GPU-hours 和原始 trace 齐全，可得'尚无净收益边'的结论。P62/P44 仅候选，不预设合法或更快。"

## 池与 profile
- 池：Modal 单 island，4×`H100!`（`--modal-gpu-exact`，断言型号），每个配置单独 launch，前缀 `infra-a-s4-<cfg>-s<seed>`。池规模按"最小可行拓扑"缩小为 4 卡（P62/P44 需要 8 卡；本轮不扩到 8 卡，只做 4 卡等价候选）。
- 目标 profile：partitioned-serial，Qwen3-0.6B LoRA r16 all-linear、GRPO、strict-avg（1 个 learner，本机 head），bf16，rollout-batch-size 4 × 8，response 384，seq 1024，lr 1e-5，total-steps 6。
- 配置（按 DP 整除规则预先筛选：T3R1 的 DP=3 不整除 32 个样本，属非法，记录为 rejected，不跑）：
  - `T2R2`：`--rl-placement fixed-partition --rl-rollout-gpus 2`（trainer DP=2，2 个 rollout engine）
  - `T1R3`：`--rl-placement fixed-partition --rl-rollout-gpus 3`
  - 默认兼容配置 `C4`：colocated-serial，4 卡共置（不同 profile，只作参考，不参与同 profile 比较）
- seed：17、29，每个配置各跑 2 个（共 6 次运行）。

## 指标与比较口径（事先登记）
- 主指标：每轮 wall 时间，取第 2–6 轮（第 1 轮包含预热，排除）的中位数；来源为 echo 磁带中相邻 `rl_round_trained` 的时间差，以及 `rl_driver_phase` 的 generate/train/sync/publish 分段。
- 成本：全池 GPU-hours = 4 × 该次运行的 wall 时间（launch 开始到结束，含预热；没有备用卡，standby=0）。
- 同 profile 比较：只在 T2R2 与 T1R3 之间比较。"更快"的判据：两个 seed 下，一个配置的中位数轮时间都比另一个低 ≥10%。否则记为"无显著差异"。
- 收益面：记录每个配置的 generate 与 train 分段占比，不外推。
- 净收益边：本 change 尚未实现任何运行中切换（E1/E3 未完成），因此不存在可以测量"切换收益减切换成本"的边。本实验只能给出固定配置之间的差异；除非出现上面定义的显著差异，并且 E1/E3 之后能测到切换成本，否则按验收原文记"尚无净收益边"。
- 不设学习效果容差（每个配置 6 轮只做性能对比）；reward 等数值只记录。
- 失败：运行失败的配置记为失败并附原因，只有修复原因后才重跑；不补跑到出现差异为止，也不挑 seed。

## 资源、费用、回收
- 每次运行预计 20 分钟：6 次 × 4 GPU × 0.33 h × $3.95 ≈ $32；硬超时每次 60 分钟，上限 24 GPU·h ≈ $95。
- 回收：同 2.2/2.3 的 arm.sh（`timeout 3600`、独立 watchdog 3900 s、结束后用 `modal app list --json` 核实），串行执行，本机用户线程数 < 3200 才启动。
- 原始 trace：每次运行的 launch.log（已脱敏）、echo 磁带、`modal app list` 记录全部入库。

## 追加（事后，配置、seed、判据均未改）：恢复扫描
- 第一次运行（T2R2 s17）失败的原因是运行时能力缺失（DistOpt 分片主参数、trainer 按单输出校验），并非实验结果本身。两处修复（66afb1e、8f2c801）已由 DistOpt 冒烟验证（`distopt-smoke/`）。按主 agent 决定 (a)，用修复后的同一 SHA 重跑全部 6 次运行，第一次运行的证据保留在 `t2r2-s17-attempt1/`。
- 已合入 launcher 修复（algo-cap）：island 最终失败时 launcher 退出码为 4，**判为失败**；每次运行仍保留独立 watchdog 与 `stop_arm.sh` 兜底。

## 追加（在第一次 s5 运行出结果之前写入）：launcher 关闭阶段误判
本扫描每次运行都带 syncer（strict-avg，1 个 learner），会受到 launcher 已知缺陷影响：learner 已 finalized、停 syncer 时关闭 Ray，被判为 FAILED。约定如下：若退出码为 4，且该岛已发出 `rl_learner_finalized`，失败只发生在关闭阶段，则该次运行记为"launcher 缺陷导致的无效运行"，既不判通过，也不判机制失败，其数据不进入比较；P0 修复后按同一计划重跑该次。其他退出码 4 仍判为失败。
