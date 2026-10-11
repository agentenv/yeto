# Tasks

## 1. RLF package admission

- [x] 1.1 固定当前 CTF deliverable (`public/`, `task.yaml`, `MANIFEST.json`) 与旁路 admission manifest 的字段映射、rlenvfactory commit、task package/image digest 和 admission error；拒绝把 AgentFlow node output 当作身份或验收证据。
- [x] 1.2 实现已验收 cyber task catalog/import，记录 grader mode、image digest、acceptance evidence；支持消费 AgentFlow/RLF 发布包，但不依赖 AgentFlow node trace。
- [x] 1.3 增加 native `flag`/`hacked` grader contract 的 trusted-side 测试，禁止 worker 读取私有 solution/secret。

## 2. Codex episode provider

- [x] 2.1 在 neutral reward environment 接口上实现 RLF lease、trajectory identity、隔离 network/secret/quota 和幂等 teardown。
- [x] 2.2 复用 Codex subprocess harness，补齐 task/episode/policy metadata、token/logprob/loss-mask/segment 输出。
- [x] 2.3 实现 timeout 后仍可判分的 trusted finish 状态机，并区分 admission/infrastructure/grader/cleanup 错误。

## 3. Reward and training adapters

- [x] 3.1 将 RLF grader 转换为 `RewardResult`，写入 `metadata.success`、grader evidence 和 rejected-data reason。
- [x] 3.2 注册 success declaration，打通 Miles reward path。
- [x] 3.3 打通 generic verl reward path，保留 metadata，不允许仅 float 的路径通过 RLF gate。
- [x] 3.4 加入 rollout/data acceptance preflight，拒绝身份、success 或 cleanup evidence 缺失的样本。

## 4. Modal runtime

- [x] 4.1 实现 Modal VM Sandbox 的 dockerd readiness、RLF container lifecycle、短期 secrets 和 episode-labeled cleanup。
- [x] 4.2 保持 GPU training path 与 VM environment path 的资源边界，增加 CPU/local fake fallback。
- [x] 4.3 增加 Modal VM environment smoke，并保存 lifecycle/teardown evidence。

## 5. End-to-end gates

- [x] 5.1 增加 CPU fake-provider 到 learner 的端到端测试。
- [x] 5.2 在 Modal GPU 上运行小规模 Codex-only RL gate，核对 rollout、trusted reward、success metadata 和 policy update。
- [x] 5.3 记录资源清理、失败分类和可复现 manifest；确认 SWE/其他 harness 未成为前置依赖。
- [x] 5.4 更新运行文档与 handoff，说明旧 `rl-agentic-reward-env` change 保持搁置及后续 apply 顺序。
- [x] 5.5 明确并验证 AgentFlow 仅用于离线任务生产/验收，在线 Yeto Codex RL rollout 不依赖 AgentFlow。
- [x] 5.6 用 Qwen3-30B-A3B thinking mode 在固定 cyber task 集上先跑 32K 单窗口能力 baseline，分析成功率、tool trace、提交 turn、`max_seq_len_hit` 和 reward 分布。
- [x] 5.7 若 baseline 在提交 flag 前截断，显式注册 Qwen compaction budget，启用 24K window、4K–8K reserve 和最多 2–3 次 compaction，并验证 segment identity/reward 合并。
- [x] 5.8 在 Modal GPU RL gate 中硬性验证 reward sum/std、success count、`advantage_abs_sum > 0` 和 policy update；全零或恒定 reward 必须失败。


<!-- 5.6 evidence: s1-runs/s21-rlf-baseline-20261011a/run10-out (ezshell reward=1.0, ret2win 0.0); turn cap widened 1024->8192 user-approved 2026-10-11 -->
<!-- 5.7 evidence: run14 (24K profile gating) + run18 (12K forced firing at 10320 tokens, segment continuation); acceptance-status.json compaction_real_firing=pass -->
