# DLR 3.2 上卡前复核（fix-decoupled-lr-schedule 3.2 两岛 head 模式复跑）

2026-10-07，S14 子 agent G2。代码 integ-decl e8d387ac（含 9bcecdea 修复；worktree /home/michael/work/s14-dlr，分支 s14-dlr）。

## 1. 要复现的回归场景与判据（按 tasks.md 3.2）
- 配置同 head 模式 `yeto-hp929d`：`--rl-sync-preset decoupled --total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2 --inner-lr 1e-5`，Qwen3-0.6B + LoRA r16 all-linear，gsm8k，rollout 4×8，resp 384，seq 1024，seed 17。
- 修复前证据（rl-engine-ports attempt4-2cbd45c）：每岛 12 个本地轮（run-until-stop，远超 global_rounds×optimizer_steps=4），syncer 16 个外层步中第 7–16 步 `global_delta_norm=0`（LR 衰减到 0 的直接表现）。本判定脚本在该旧证据上输出 INCOMPLETE/第 7–16 步为 0，证明能识别该回归。
- 判据（s1-runs/s14-dlr-judge.py，量化）：
  1. 每岛本地轮数 > 4（预期 12）、轮号连续；每个 `rl_local_round.applied_lrs` 存在且每个值 == 1e-05（精确相等）；
  2. syncer 第 1..16 外层步 `sync/global_delta_norm` 全部非 0；
  3. 两岛最后一次 `rl_policy_apply` 的 policy_version 与 `sync/global_policy_hash` 一致；
  4. 两岛 `rl_learner_finalized`、日志无零 LR 不变量失败文本、head job SUCCEEDED（rc=0）、两容器报 H100。
  5. legacy 与 ports 各跑一次后用 `--compare` 检查两岛 applied_lrs 序列逐位（repr）一致。
- 不含 kill/resume（3.2 未要求）。

## 2. 形态（s1-runs/s14-dlr-remote.sh，由 s13-g3-remote.sh 改）
- Nebius eu-north1 无卡 VM（cpu-d3_8vcpu-32gb，$0.20/h）托管 controller + actor syncer :29400；Modal 两岛 `H100!`（gpu_exact，retries 0），`--modal-timeout-s 2700` 从容器启动计时（排队不计）。
- 相比 s13-g3：去 critic（无 spec、无 :29401）、去 killer、去 `--keep`；`--rl-stall-timeout 2400`：S13 §7 中 0.6B fp32 全量 critic 初始参数跨 WAN 上传 14 min 触发 900 s；本任务只有 LoRA 增量，预计远小于 900 s，但仍按主 agent 要求设 2400 并记录。stall 2400 < HARD 2700，HARD 之后本机 watchdog（+600 s）兜底 teardown。
- 镜像：两引擎都用公开 MILES_NEXT_IMAGE（legacy 默认 MILES_IMAGE 为私有 ghcr，Modal 不能拉；legacy setup 在镜像内 clone agentenv/miles + vendored bundle，与 09-29 L40S sandbox 做法相同——该组合在 launcher 路径上**未验证**，legacy 若在 setup 阶段失败不重试）。
- PLAN_ONLY（ports/legacy 各一次）：controller=head、preset=decoupled、stall=2400、keep=False、无 critic、syncer `--total-steps 16 --pipeline 2`、Modal token 以 sky secret 传递、sky dryrun 选中 Nebius cpu-d3_8vcpu-32gb。

## 3. 费用
- 单价：Modal H100 $4.39/h × 2 = $8.78/h；VM $0.20/h。
- 预计每次：hp929d 训练段 12 轮约 7 min + Modal 容器启动/连接约 6–8 min（s13-g3 f：app 起 → syncer 连接 6 min）≈ 15–20 min → ≈$2.2–3.0 + VM ≈$0.1。两次（ports 先，legacy 后）≈ $5–7。
- 最坏：HARD 2700 s/次 → $6.6/次 + VM，两次 ≈ $13.7 < cap $15。ports 跑完后按实际费用决定是否起 legacy（剩余不足 $6.6 则不起）。
- 审计估 $9–12（按 Nebius 2×H100 时价）；本形态更便宜。

## 4. 上卡前检查（10-07）
- 线程 2433 < 3000；raylet/gcs_server 为空；Modal app list 无 running GPU；sky 无集群；磁盘 431G 可用。
