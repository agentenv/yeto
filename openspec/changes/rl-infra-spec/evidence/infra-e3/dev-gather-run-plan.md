# DEV-GATHER 运行计划（B3，INFRA-E3，2026-09-30；运行前提交）

- **目的**：调试 A8 harness（`tools/probes/e3_reshard/`）在真实 Megatron/DistOpt/Miles 上能否跑通。**不计入任何 task，不作为验收证据，不做数值判定**（plan-v3 §1）。
- **资源**：Modal `A10G:2`，单个 Sandbox，`timeout=5400`（90 min 硬超时）；app 名前缀 `infra-v2-b3-devgather-`。最坏费用 2×1.5 h×$1.10 ≈ **$3.3**。
- **代码/镜像**：yeto `infra-e3` 当前推送提交（运行时 SHA 写入 `resources.txt` 与证据目录）；上传的是 `git archive` 冻结快照加上 1.2 用过的 `gsm8k_reward.py`（sha256 743d433c…）。镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:17d428a2…0bcef`（tag 5c1b49e-9f29303），Miles pin 5c1b49eb；容器内先断言 GPU 型号为 `NVIDIA A10G`×2、Miles pin 一致，再生成 runtime manifest。
- **profile**：Qwen3-0.6B（rev c1899de2），LoRA r16 all-linear，bf16，gsm8k，`--rollout-batch-size 2 --n-samples-per-prompt 8`（GBS=16），`--rl-single-island-no-sync --controller local`；harness 在 parse 之后把 lora/hidden/attention dropout 设为 0（记录在 `miles_args.*.json`）。
- **冻结 rollout 由基座策略生成**（Miles `debug_rollout_only`），**不影响判据与容差**：各 arm 在同一份冻结数据上比较 trainer。
- **要暴露的风险**：
  1. `--save/--load-debug-rollout-data` 与 `debug_rollout_only/debug_train_only` 在 parse 之后设置是否足够（包括 dropout 覆盖是否真的进入模型）；
  2. 源码树哈希：上传目录与本地 dry-run 时的树是否一致；
  3. 奖励函数文件在容器内的位置；
  4. 同一 Ray 集群上多个 driver 进程依次创建/释放 placement group。
- **阶段**：dry → gen（8 个冻结 rollout）→ arm A1、A2、B1、B1p、B2、RT → compare（结果只作参考，不判 go/no-go）。
- **停止条件**：任一阶段失败即停（`set -e`），不在卡上反复排错；先拉 stdout/stderr 与证据包，再释放。
- **回收**：Modal `timeout=`；本地独立 watchdog（`timeout` 包裹 + 按 `resources.txt` 中的 app id 在 95 min 时 `modal app stop`）；结束后 `modal app list` 只核实 `infra-v2-b3-devgather-*` 为 stopped、0 tasks。不动他人资源，不做全局清理。
- **花费台账**：`infra-drafts/gpu-spend.md` 批次 B3。
