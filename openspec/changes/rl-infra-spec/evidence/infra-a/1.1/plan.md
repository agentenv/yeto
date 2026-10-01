# 1.1 runtime manifest（INFRA，事前计划，运行前提交）

- task / 验收：1.1 "manifest 与 MILES_NEXT_*/SGLANG_NEXT_* 一致，缺接口组合拒绝认证"。
- 镜像：`MILES_NEXT_IMAGE` = ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07…aa540（不含 M1–M6）。
- 资源：Modal Sandbox，1×L40S，8 CPU / 64 GiB；app 名 `infra-a-manifest`。不需要 H100（只做导入与版本探测，不做数值比较）。
- 预计时长 15 分钟（主要是拉取镜像）；费用约 $0.5。硬超时：sandbox `timeout=1500` s，外层再用 `timeout 1800` 包裹；结束后执行 `modal app stop -y infra-a-manifest`，并用 `modal app list` 核实。
- 步骤：把 yeto 工作树（git archive HEAD）上传到容器，运行 `python -m yeto.rl.engine.runtime_manifest --image <ref> --out manifest.json`，依次请求以下能力：
  1. `ports-colocated-serial`、`ports-partitioned-serial`：预期 **通过**（镜像含 run_plugin，pin 一致）。
  2. `fixed-partition-standby`：预期 **拒绝**，原因是缺 `fork_m1_placement_map`（该镜像不含 M1），即"缺接口组合拒绝认证"。
  3. `RolloutPool.add_engines`：预期 **拒绝**，缺 M2/M3。
- 成功条件：1 通过，2、3 按预期拒绝，manifest 记录的 torch/megatron/cuda/nccl/TMS/peft 版本齐全。失败条件：1 被拒绝，或 2/3 被通过（任一出现即判 1.1 未通过，记录原因，不重跑到通过）。
- 凭据：进程内从 ~/.docker/config.json 解码 ghcr 凭据，只放进 Modal Secret，不打印、不落盘。

## attempt1 结果与重跑理由（事后追加，计划条件未改）
attempt1（sb-8avihvjXJ5HpcSMbpNcqxQ，91 s，L40S）：基础能力认证通过，standby 与 add_engines 按预期被拒绝。但 `torch_memory_saver` 模块没有 `__version__`，被记成 "unknown"，而 `check_manifest` 把 "unknown" 当作已记录。原因已修复：采集时回退到 `importlib.metadata`，检查时拒绝 "unknown"。修复后按同一计划重跑一次（attempt2）。
