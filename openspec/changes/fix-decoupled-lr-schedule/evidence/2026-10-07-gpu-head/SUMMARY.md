# 3.2 两岛 decoupled head 模式复跑（2026-10-07，S14 G2）

代码 integ-decl e8d387ac（含 9bcecdea 修复）。形态：Nebius eu-north1 无卡 VM 托管 controller + syncer，Modal 2×（1×H100!）岛；配置同 `yeto-hp929d`（`--rl-sync-preset decoupled --total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2 --inner-lr 1e-5`，Qwen3-0.6B LoRA r16，gsm8k 4×8，seed 17）。判定脚本 `s14-dlr-judge.py`，复核 `DLR-32-PRELAUNCH-REVIEW.md`。

## ports：`s14-dlr-ports-20261007a` — PASS（`ports/judgment.json`）
- head job rc=0；两容器均 `requested H100!:1, got ['NVIDIA H100 80GB HBM3']`（`ports/gpu-confirm.txt`）。
- 两岛各 13 个本地轮（round_ids 1..13 连续，> global_rounds×optimizer_steps=4，复现 run-until-stop 场景）；每轮 `applied_lrs == [1e-05]`（13/13 × 两岛，精确相等于配置值）；两岛 `rl_learner_finalized`。
- syncer 外层步 1..16 的 `sync/global_delta_norm` 全部非 0：0.01690, 0.01953, 0.02254, 0.02528, 0.01321, 0.01310, 0.01191, 0.01459, 0.01155, 0.01102, 0.01094, 0.00953, 0.00890, 0.00887, 0.00925, 0.00899（`ports/syncer-key-lines.txt`，来自 `ports/head-yeto-tape.jsonl`）。修复前证据为第 7–16 步全 0。
- 两岛最后一次 `rl_policy_apply` 均为 v13，policy hash `9401304a…402b8f` 一致。
- launch.log 的 8 个 Traceback 均为 urllib3 连接重试（SGLang 就绪轮询），非错误。
- 备注：岛 1 本地 `delta_l2_norm` 在第 1/4/7 轮为 0（岛 0 为第 1/4 轮）；这是本地轮级别的量，不在 3.2 判据内，且 syncer 全局 delta 全非 0；此处仅记录，未分析。
- 费用（估算）：Modal app 11:45:15–11:57:30 ≈12.25 min × $8.78/h ≈ $1.79；VM ≈19 min ≈ $0.06。

## legacy：`s14-dlr-legacy-20261007a` — INCOMPLETE（未产生任何训练事件，`legacy/judgment.json`、`legacy/judgment-compare.json`）
- head job rc=1：Modal 构建 legacy 岛镜像时 skopeo 拉取 `ghcr.io/michaellchung/yeto-miles-ports@sha256:37ac689e…` 报 `unable to retrieve auth token: invalid username/password: unauthorized`（`legacy/image-build-error.txt`）。
- 原因：`yeto/rl/__init__.py` 的 `MILES_NEXT_IMAGE` 指向私有 ghcr 镜像（与复核文档 §2 "公开镜像"的假设不符）。ports 能跑是因为 Modal 侧已有该镜像的构建缓存（同日 lossvar / 之前的 ports 运行）；legacy 在其上追加 setup 层需要重新拉源镜像，Modal 没有 ghcr 凭据。
- 按规则 setup 阶段失败不重试。Modal app 存活约 2 s（≈$0），VM 12:01–12:06 ≈ $0.02。
- `--compare`：legacy 无 applied_lrs 序列，无法与 ports 做逐位比较。

## 3.2 结论
ports 一侧判据 1–4 全部成立；legacy 一侧缺证据（镜像拉取失败）→ **3.2 未勾选**。补齐方式：给 Modal 配 ghcr 拉取凭据（Modal secret）或把 legacy 的 `--rl-image` 指向公开可拉镜像（radixark/miles 官方镜像 + 镜像内 setup），再以 `ENGINE=legacy bash s1-runs/s14-dlr-remote.sh` 跑一次并 `--compare` ports 目录。
