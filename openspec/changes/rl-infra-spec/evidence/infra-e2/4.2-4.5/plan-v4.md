> **已被 [plan-v5.md](plan-v5.md) 取代**（2026-09-30，C1/C2 改为 fixed-partition）；本文件保留作记录。

# 4.2–4.5 GPU 验证计划 v4（INFRA-E2，2026-09-30；取代 plan-v3.md，旧版保留）

## v4 相对 v3 的变更：只改环境，判据文字不变
判据以 `plan-v2.md` §1 原文为准，v3 的执行手段、运行顺序和配置（包括 LoRA dropout 0.05）不变。本版只做下列更新：

| 项 | v3 | v4 | 性质 |
|---|---|---|---|
| 镜像 | `sha256:17d428a2…`（5c1b49e-9f29303） | `ghcr.io/michaellchung/yeto-miles-ports@sha256:db81588406e157baa6a579f6378484b890371065abcc51eacd5a9650b5820cbf`（tag 2f23a0f-9f29303） | 环境更新 |
| Miles fork | `5c1b49eb` | `2f23a0fca9b80f6a7300da401703c343014b03c0` | 环境更新。这是 5c1b49eb 加 F-R1，只改 ray/workers/placement，不涉及 backends 或训练路径（见 `evidence/2026-09-30-img-2f23a0f/`）；harness 用到的 `slice_pg_info` 改为公开名，yeto 已兼容 |
| LoRA dropout 0.05 的表达 | 阻塞（ports 路径固定为 0） | 由 INFRA-E1 的 `--rl-lora-dropout 0.05` 表达（仅 C1/C2；默认不变） | 执行手段，判据不变 |
| G-4.5 第 5 行 | 阻塞（游标只有缓存值） | 用 INFRA-E1 的 `MilesRolloutPool.live_data_cursor()` 实时读取。若启用了游标偏移注入而实时读取不可用，`trainer_rebuild` 直接拒绝，避免这一行空转通过 | 执行手段，判据不变 |

## 运行前还须满足
1. 检出的代码中 `MILES_NEXT_IMAGE` 必须是上表的 db815884 digest。工具 `tools/probes/e2_cut_harness.py` 的 pin 校验对应 plan-v4，不一致时 rc=3，不生成任何 run。
2. 镜像内 learner preflight（Bridge provider、Miles `parse_args`/`validate`、runtime manifest）不能在纯 CPU 容器中执行：2026-09-30 B2 的 CPU 试跑在 import `transformer_engine` 时因 `libcuda.so.1` 缺失而失败（见 `preflight-cpu-20260930/`）。需要一个能 dlopen libcuda 的容器（例如 T4），待批准。
3. 其余同 plan-v3 §3 第 3、4 条。

## 本地 dry-run
见 `dry-run-v4-20260930.md`。
