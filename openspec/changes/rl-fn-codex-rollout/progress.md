# rl-fn-codex-rollout progress

## 阶段 1 首跑（2026-10-07，S14 子 agent L1，基线 cdf86720）
- run `s15-fncodex-4layer-modal-20261007a`（Modal 1×H100!，profile `qwen38_next_4layer`，tbench2_smoke6，6×4，1 步，SGL_MEM_FRAC 0.6）：**INCOMPLETE**，0 轮，rc=4，费用 ≈$0.31（估算）。
- 根因：learner 容器内 `_preflight_codex_harness → forward_legacy_openenv_preflight` 抛 `the Codex OpenEnv adapter requires backend profile qwen35_08b`（OpenEnv 子进程 agent 路径 profile pin 硬编码）。模型未加载；H100、Volume（HF d19a6b60 命中，无 Hub 下载）、tape sink 正常。
- 结论：阶段 1 判据全部未到达（样本 0/24，失配四类计数/逐条 tape/奖励/logprob/grad/hash 均无数据，不是 0）；新增任务 1.0 放开 pin 后再开卡。1.1 不勾。
- 已知可得分样例：smoke-11/12 全零奖励，历史无任何 smoke6 任务得分；judge 以 `fix-git` 为候选（`--known-scorable fix-git`，标注非"已知"），且 tbench_reward 无逐轨迹日志，`known_scorable_passed` 预计恒 null，需另加逐轨迹奖励日志/tape 才能验证。
- 脚本修正：`CONTAINER_MARK` 需匹配 `requested H100!:1, got`（已改）。
- 证据：`s1-runs/s15-fncodex-4layer-modal-20261007a/`、`infra-drafts/FNCODEX-PRELAUNCH-REVIEW.md` §8–9、`infra-drafts/gpu-spend.md`。
