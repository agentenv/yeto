# 1.2 兼容 baseline（INFRA，事前计划；本文件单独提交，得到 SHA 后才启动）

## 验收（tasks 1.2 原文）
"选定小模型 LoRA/strict-avg 兼容 profile，在 ports 路径上执行现有串行固定配置；验收：完成一轮生成、更新、外层同步与权重确认，形成兼容 baseline。"

## 选定 profile
- 模型 Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca；数据 zhuzilin/gsm8k @ 0cbd9f31d91ac21a7613dcbc7fef992adac459ae；reward gsm8k_reward:score。
- LoRA r16 all-linear，GRPO（默认 AlgorithmSpec），strict-avg，colocated-serial，`--rl-engine ports`（默认）。bf16（R0 冒烟配置）。按 tasks 4.2a，该 profile 下 M5 路径不可用，E2 需另选路径，在 4.1 处理。
- 批次：rollout-batch-size 4，n-samples-per-prompt 8，response 384，seq 1024，total-steps 3，fragments 1，pipeline 1，inner-lr 1e-5，seed 17。
- 两个 island（strict-avg 外层同步至少需要两个 learner 才有意义），每个 1×H100，不涉及 fixed partition 与 M-fork。

## 资源
- 云：Modal（与 rl-engine-ports 7.1 已跑通的路径相同：Modal island + 本机 head/syncer，公网 IP 185.189.44.160）。偏离 gpu-plan A1（Nebius 4×H100）的理由：Modal+本机 head 是 R0 已验证的路径；Nebius 公网 IPv4 配额只有 3，head 放置不确定；本实验只做功能 baseline，不做跨 provider 的统计比较（2.4 将与本 baseline 同 provider、同型号）。
- GPU：`--gpu modal:1xh100,modal:1xh100 --modal-gpu-exact`（请求 `H100!`，容器启动时断言型号，这是该选项首次在真 H100 上使用，结果记为证据）。
- 镜像：`MILES_NEXT_IMAGE` = ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07…aa540。
- head：本机，隔离 HOME=/home/michael/work/infra-a-gpu/b12/home，YETO_RUNS_DIR 独立；syncer 二进制复制自 /home/michael/work/gpu-default-modal/home/yeto-syncer（2026-09-29 12:06 构建；syncer/ 源码自 55076ff 起未变），记录 sha256。
- 前缀：`--cluster-prefix infra-a-b12`，Modal app 名为 `yeto-infra-a-b12`。

## 时长、费用与硬超时
- 预计 30–45 分钟 wall；2×H100×0.75 h×$3.95 ≈ $6。上限：硬超时 90 分钟，对应 2×1.5×3.95 ≈ $12。
- 回收机制：(1) launcher 结束时自行拆除 Modal island；(2) 外层 `timeout 5400` 包裹 launch；(3) 独立 watchdog（`setsid nohup`，与终端和 agent 无关），5700 s 后执行 `modal app stop -y yeto-infra-a-b12`；(4) 结束后执行 `modal app list` / `modal container list`，核实状态并存档。

## 事先登记的成功/失败条件
成功（全部满足）：
1. 两个 learner job 均为 SUCCEEDED，任何 island 事件磁带中都没有 `rl_strict_failure`。
2. 每个 island 至少完成 1 轮：有 `rl_driver_phase` 的 generate→train→sync→publish 序列和 `rl_local_round`。
3. 外层同步与权重确认：每个完成轮次都有 `rl_policy_apply`，两个 island 在同一 policy_version 上的 `sync/global_policy_hash` 相同；紧随其后的 `rl_publication` 的 policy_token 中的哈希等于该 apply 的哈希；`sync/publication_members` 非空。
4. `rl_driver_start.execution_mode == colocated-serial`，两个 island 的 `rl/algorithm_spec_sha256` 相同，并等于默认 AlgorithmSpec 的哈希。
5. `--modal-gpu-exact` 的断言输出显示 GPU 型号为 H100。
失败：任一条件不满足，即判 1.2 未通过并记录原因。只有找到原因并修复后才重跑；不改上述条件。
数值（reward、loss、grad_norm、每轮耗时）只作为 baseline 记录，不设容差（后续比较的容差由 2.2/2.4 的计划另行登记）。
