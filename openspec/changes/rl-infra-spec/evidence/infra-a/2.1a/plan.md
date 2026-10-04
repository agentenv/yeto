# 2.1a（fork-M1）GPU 验收计划（INFRA，事前计划；单独提交后才启动）

- 验收原文："CPU 单测覆盖重复/越界/重叠拒绝与缺省等价，打入镜像后 2.1 以显式映射启动并记录 bundle↔GPU UUID"。CPU 部分已完成（fork 单测 61 passed，yeto 侧校验）。
- 镜像：elastic `MILES_NEXT_IMAGE` …@sha256:9f0977da…（miles d002615f，含 M1）。代码为本计划提交之后的 infra-a HEAD。
- 拓扑（最小）：Modal 1 个 island，3×`H100!`；`--rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 1`，即 T1R1S1，yeto 生成 `--yeto-placement-map {"trainer":[0],"rollout":[1],"standby":[2]}`。profile 同 2.2，total-steps 2，前缀 `infra-a-m1a`。
- 成功条件（全部满足）：
  1. learner SUCCEEDED（launcher 退出码 0；退出码 4 判失败，关闭阶段误判按 2.4 计划的约定处理），无 `rl_strict_failure`；
  2. 日志中出现 M1 的 "Creating placement group with 3 GPUs from explicit map"，以及 3 条 bundle→(node, gpu) 行；解析后的参数中 placement map 与请求一致（启动时 `check_placement_not_rewritten` 未拒绝）；
  3. 从容器 `nvidia-smi --query-gpu=index,uuid` 与 M1 的 bundle 行得到 trainer/rollout/standby 各自的 GPU UUID，写入 `bundle_gpu_uuid.json`；standby GPU 上没有 trainer 或 rollout 进程（日志中 trainer 与 rollout engine 使用的 bundle 都不是 standby bundle）。
- 费用：约 15 分钟 × 3 H100 ≈ $3，硬超时 60 分钟。回收同前（watchdog、stop_arm.sh、`modal app list`）。

## 追加（事后，条件未改）：attempt1 结果与重跑
- attempt1（1cf23fb）：条件 1、2 满足，条件 3 的 UUID 记录已完成；但"standby GPU 上无 trainer/rollout 进程"缺少直接证据，按预登记不判通过。
- 修复（证据采集，不改代码路径）：运行期间每 10 s 在容器内采集 `nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu` 与 `--query-compute-apps`，追加到 `compute-apps.txt`。判据："standby GPU 在所有采样中 memory.used 不超过空闲基线（≤ 1 GiB）且无 compute app；trainer/rollout GPU 有占用"。按同一计划重跑（attempt2），代码 SHA 不变（1cf23fb）。
