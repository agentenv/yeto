> **已被 [plan-v4.md](plan-v4.md) 取代**（2026-09-30，镜像与 Miles 提交更新）；本文件保留作记录。

# 4.2–4.5 待本地/批准后 GPU 验证计划 v3（INFRA-E2，2026-09-30；取代 plan-v2.md，旧版保留）

## v3 相对 v2 的变更：只改环境，判据不变

v3 只更新环境记录与执行手段。plan-v2 §1 的判据文字、容差、配置（模型、DP、GBS、seed、LoRA dropout 0.05）、确定性要求与运行顺序原样沿用，未放宽任何一条。具体变更：

| 项 | v2 | v3 | 性质 |
|---|---|---|---|
| 镜像 | `sha256:c6f5455c…`（tag 0af62f4-9f29303） | `ghcr.io/michaellchung/yeto-miles-ports@sha256:17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef`（tag 5c1b49e-9f29303） | 环境更新 |
| Miles fork | `0af62f4d` | `5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba` | 环境更新。已核对：v3 用到的 M5 函数（`export/merge/check/load_named_optimizer_state`、`capture/restore_rng_state`）、`rebuild_training_models`、`_is_adapter_param_name`、`get_parallel_state` 在 5c1b49eb 中签名不变 |
| Qwen3-0.6B | 未钉死 revision | `c1899de289a04d12100db370d81485cdf75e47ca` | 钉死 |
| Qwen3-1.7B | 未钉死 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`（HF API 2026-09-30 查询） | 钉死 |
| 数据 | 未写明 | `zhuzilin/gsm8k@0cbd9f31d91ac21a7613dcbc7fef992adac459ae`，reward `gsm8k_reward:score` | 钉死 |
| 确定性 | 列出环境变量与 Megatron 参数 | 由 launcher `--rl-deterministic-trainer` 设置（INFRA-E1：Megatron `--deterministic-mode`，以及 `NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`，经 Ray runtime_env 下发到每个 worker）。harness 在第 0 步经 `rank_determinism` 插件逐 rank 读回，任一 rank 不符即 `environment_blocked`，不运行 | 执行手段 |
| 执行手段 | 无 harness | 见 §2 | 新增 |

## 1. 判据
与 plan-v2 §1 完全相同（G-4.2 (a)–(g)、G-4.3 (1)–(4)、G-4.4、G-4.5 六行、§3 L2），这里不重抄，以 `plan-v2.md` 原文为准。

## 2. 执行手段（harness，本地只到 dry-run）

- 主机侧：`tools/probes/e2_cut_harness.py --root <dir> --yeto-sha <sha> [--launcher-dry-run]`。
  - 按下表顺序为每个 run 生成 `args.txt`、`harness.json`、`spec.json`、`run.sh`。
  - `run.sh` 的内容：launch、独立 watchdog（硬超时 90 min + 5 min 后 `modal app stop`）、puller（事件磁带、GPU 名、harness 结果），C3 另有 `rebuild-trainer` 触发器。
  - `run.sh` 开头有守卫：未设 `YETO_E2_GPU_APPROVED=1` 就退出；工具本身拒绝 `--execute`。
- 容器内：`miles_adapter/e2_harness.py`。snapshot 根目录有 `yeto-rl-e2-harness.json` 时，由它代替 `driver.run()`，按 plan-v2 §1 G-4.2/G-4.3 的顺序执行，结果写到 `~/yeto-rl/e2-harness/<C>/results.json` 与 `steps.jsonl`。入口接线见补丁 `infra-e2-e1-harness-injections-v1.patch`（归 INFRA-E1）。

| 顺序 | run | 配置 | GPU（`--modal-gpu-exact`） | 覆盖判据 |
|---|---|---|---|---|
| 1 | c1 | C1：0.6B、DP1、共置、harness | H100 ×1 | G-4.2 (a)–(e)(g)，G-4.3 (1)–(4)，L2 |
| 2 | c2 | C2：0.6B、DP2 DistOpt、共置、harness | H100 ×2 | 同上，另加 (f) |
| 3 | c3-b1 | C3：1.7B、T2R1S0（DP2 DistOpt），不重建 | H100 ×3 | G-4.4 基线 |
| 4 | c3-rb | C3，第 2 轮后执行 controller `rebuild-trainer` | H100 ×3 | G-4.4 |
| 5 | c3-rbold | C3 + `--rl-test-inject-rebuild-fail` | H100 ×3 | G-4.4 REBUILD_OLD、G-4.5 第 4 行 |
| 6 | c3-f1 | C3 + `--rl-test-inject-cut-restore-kill-rank 1` | H100 ×3 | G-4.5 第 1 行 |
| 7 | c3-f2 | C3 + `--rl-distributed-timeout-minutes 5 --rl-test-inject-cut-restore-sleep 1:420` | H100 ×3 | G-4.5 第 2 行 |
| 8 | c3-f3 | C3 + `--rl-test-inject-cut-save-kill-rank 1` | H100 ×3 | G-4.5 第 3 行 |
| 9 | c3-f5 | C3 + `--rl-test-inject-rebuild-cursor-shift 1` | H100 ×3 | G-4.5 第 5 行（**阻塞**，见 §3） |
| 10 | c3-f6a / c3-f6b | C3 + `--rl-test-kill-learner-at REBUILDING_TRAINER`（CAS 前）或 `COMMITTED`（CAS 后），均加 `--rl-elastic-restart-attempts 1` | H100 ×3 | G-4.5 第 6 行 |

说明：
- C1/C2 在各自的一次 run 内先后执行两个 arm，满足 v2 §0"两个 arm 在同一次运行内先后执行"。C1、C2 之间是两次 Modal 租用，GPU 型号由 `--modal-gpu-exact` 和 puller 的 `nvidia-smi` 输出核对。
- C3 在第 6 行 kill learner 后，由 E1 的 restart loop 在**同一容器内**重启；state dir 位于容器内的 `~/yeto-rl/elastic-state`，容器存活期间保持不变，所以不需要 Volume。如果重启后发现 state dir 丢失（容器被回收），则判为**环境阻塞**，不判失败，也不判通过。

## 3. 运行前必须解决（未解决就不运行，也不改判据）

1. **LoRA dropout 0.05 无法表达**：ports 路径把 `--lora-dropout` 固定为 0（`miles_adapter/config.py` 的翻译部分）。v2 C1/C2 要求 0.05，用来让 RNG 影响第 3 步。harness 逐 rank 读回 `lora_dropout`，与计划不符即 `environment_blocked`。需要 config 翻译部分的写入者加一个显式开关，或由用户/主 agent 另行裁定判据（本计划不自行改为 0）。
2. **G-4.5 第 5 行（游标篡改）无法检测**：`MilesRolloutPool.data_cursor()` 返回的是上一个 batch 的游标，不是实时读取；重建期间 `rollout_executor.load` 造成的回卷要到下一次 generate 才能看到。需要 E1 提供实时读取（在 rollout 进程内读 data source），在那之前该行标为阻塞。
3. E1 补丁 `infra-e2-e1-harness-injections-v1.patch`（entry 中的 harness 钩子、Ray runtime_env 转发 cut 注入变量、launcher/cli 的四个注入开关）合入；`--rl-deterministic-trainer`、`--rl-test-inject-rebuild-fail`、`--rl-test-kill-learner-at`、`--rl-elastic-restart-attempts` 已在 infra-e1 分支。
4. 本地 dry-run（§4）全部 rc=0。容器内才能执行的检查（Bridge provider、Miles `parse_args`/`validate_args`、runtime manifest）在本地做不了。建议上 GPU 之前先在 CPU 容器里跑一次 learner preflight（需另批）。

## 4. 本地 dry-run 结果（CPU，不计入验收）
见 `dry-run-20260930.md`。
