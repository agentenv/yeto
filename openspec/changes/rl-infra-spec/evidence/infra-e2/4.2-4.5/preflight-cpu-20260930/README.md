# 镜像内 CPU learner preflight（B2，主 agent 批准；2026-09-30）

- 驱动：`tools/probes/e2_modal_preflight.py`（Modal Sandbox，CPU 4 核/16 GiB，sandbox timeout 1080 s，本地 timeout 1200 s，独立 watchdog 1320 s）；容器内脚本 `tools/probes/e2_container_preflight.py`。
- 镜像：`sha256:17d428a2…`（当时的 pin；本次运行在 plan-v4 更新之前）。代码快照：origin/infra-e1 c51735e + `infra-e2-e1-harness-injections-v1.patch` + infra-e2 的 cut/harness 文件。
- app `ap-AOEKeEOxdqbPDpGJNSPQyD`，sandbox `sb-utonN4lnDv42rvkO2XjzkF`；05:58:48–06:01:17Z；app 已 stopped/0（`teardown_proof.txt`）；watchdog 已终止。
- 费用：≤ $0.02（约 2 分钟 × 4 核/16 GiB，未取账单）。
- 结果：**环境不满足，未得到 preflight 结论**。
  - c1、c2、c3-rb 的 `learner.main` 都在 import `megatron.core` → `transformer_engine` 时报 `OSError: libcuda.so.1: cannot open shared object file`，没有进入 Bridge provider 和 Miles parse_args。
  - `runtime_manifest` rc=2，原因是读不到 megatron.core 版本（同一问题）；manifest 中的 commits（miles 5c1b49eb、sglang 9f29303）与当时的 pin 一致。
- 结论：这些检查需要能 dlopen libcuda 的容器。ALGO-2b 4.4 在 T4 上跑过同类检查，原因相同。
