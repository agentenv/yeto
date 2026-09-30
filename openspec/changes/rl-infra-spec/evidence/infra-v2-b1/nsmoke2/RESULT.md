# Nebius 冒烟（第二次，on-demand）：通过，2026-09-30

代码 47efd25（合并 integ-decl e7e16eb：私有镜像登录走 task secrets），`--on-demand`，Nebius eu-north1 `gpu-h100-sxm_1gpu-16vcpu-200gb`（$3.85/h），strict-avg 单岛 + 本机 head（185.189.44.160:29400），1 轮。
- VM 开通、Docker 容器（私有 ghcr 镜像）启动：launch.log "Instance is up / Docker container is up"。
- 岛连回本机 head：syncer `learner connected … learner_id=0`（13:04:38），`outer step step=1 … responders=[learner 0]`（13:08:16）。
- 1 轮训练完成（rl_round_trained），learner job SUCCEEDED，rc=0，launcher teardown。
- 释放核实：`sky status` 无集群；`nebius compute instance list` 为空。
- 费用 ≤$1.35。空闲流探测（scripts/idle_flow_probe.py）未做。
