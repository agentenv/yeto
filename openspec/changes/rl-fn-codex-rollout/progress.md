# rl-fn-codex-rollout progress

## 阶段 1 首跑（2026-10-07，S14 子 agent L1，基线 cdf86720）
- run `s15-fncodex-4layer-modal-20261007a`（Modal 1×H100!，profile `qwen38_next_4layer`，tbench2_smoke6，6×4，1 步，SGL_MEM_FRAC 0.6）：**INCOMPLETE**，0 轮，rc=4，费用 ≈$0.31（估算）。
- 根因：learner 容器内 `_preflight_codex_harness → forward_legacy_openenv_preflight` 抛 `the Codex OpenEnv adapter requires backend profile qwen35_08b`（OpenEnv 子进程 agent 路径 profile pin 硬编码）。模型未加载；H100、Volume（HF d19a6b60 命中，无 Hub 下载）、tape sink 正常。
- 结论：阶段 1 判据全部未到达（样本 0/24，失配四类计数/逐条 tape/奖励/logprob/grad/hash 均无数据，不是 0）；新增任务 1.0 放开 pin 后再开卡。1.1 不勾。
- 已知可得分样例：smoke-11/12 全零奖励，历史无任何 smoke6 任务得分；judge 以 `fix-git` 为候选（`--known-scorable fix-git`，标注非"已知"），且 tbench_reward 无逐轨迹日志，`known_scorable_passed` 预计恒 null，需另加逐轨迹奖励日志/tape 才能验证。
- 脚本修正：`CONTAINER_MARK` 需匹配 `requested H100!:1, got`（已改）。
- 证据：`s1-runs/s15-fncodex-4layer-modal-20261007a/`、`infra-drafts/FNCODEX-PRELAUNCH-REVIEW.md` §8–9、`infra-drafts/gpu-spend.md`。

## 阶段 1 第 2 跑（2026-10-07，S14 子 agent L2，分支 s14-fncodex-l2，基线 9ac476f9）
- 1.0 已实现并提交（8bacb90d）：OpenEnv adapter 运行时 profile 取 `YETO_CODEX_CHAT_TEMPLATE`，允许集合 {qwen35_08b, qwen38_next, qwen38_next_4layer}，镜像 identity env 视为构建记录不重建；逐轨迹奖励 `rl_trajectory_reward` tape 事件 + judge 读取；design D3 SGL_MEM_FRAC 0.6。
- run `s15-fncodex-4layer-modal-20261007b`：**INCOMPLETE**，0 轮，rc=4，≈$0.40。preflight 通过，但 `check_config_mapped` 拒绝 `agent.tito_allowed_append_roles`（tito 家族 qwen4exp 不是 profile 名）。已修 da62e264（按 tito 家族解析），PLAN_ONLY 加 CPU 等价检查。详见 REVIEW §10。
- 第 3 跑 `s15-fncodex-4layer-modal-20261007c` 预登记后启动。

## 阶段 1 第 3 跑（2026-10-07，S14 子 agent L2，代码 da62e264）
- run `s15-fncodex-4layer-modal-20261007c`：judge **FAIL**，但 1 轮完整跑通（preflight → torch_dist → SGLang 0.6 → 24/24 TB2 轨迹 → 签名奖励 → train → publication → teardown），rc=2 仅为 Modal 岛无 `~/yeto-output` 可取。≈$1.76；阶段 1 累计 ≈$2.47。
- 未过判据及归因：失配 100%（全部可由 `rl_harness_mismatch` 逐条分类：23 条 4096 截断缺尾 `<|im_end|>`，1 条 `<|endoftext|>` 收尾）；logprob −12.0≈ln(1/词表)（4 层变体输出近似随机）；奖励全 0 → grad 0（GRPO 必然）；counters_zero_at_end 为采样窗口伪影。
- 新增可观测性已落地：`rl_trajectory_reward` 24 条（fix-git 0/4 得分，首次得到确定值）。judge 的 unclassified 分母修正（per-record），见 judgment-v2.json。
- 待裁定：D7 阶段 1 的 grad_norm>0 与 Q4 非零奖励不要求相冲突；4 层变体是否适合作为阶段 1 载体（输出随机导致失配/截断/零奖励三者同源）。


## 阶段 2 全尺寸（2026-10-07，S15，run `s15-fncodex-full-modal-20261007a`，Modal 8×H200，≈$79）
- 基础设施链路通过：2 轮 × 24/24 轨迹、权重发布 v0→v2、rollout/trainer logprob |Δ| 0.076、tis_clipfrac 0.58%。judge FAIL 项为 kl_le_max（0.022 > 自设 0.01）。rc=5 = 拆除未确认（Modal app 仍 stopping）。
- 48 条奖励全 0 的根因：任务说明从未送达模型。smoke6 数据只有通用 system 消息、无 metadata.prompt，subprocess agent 用 `str(prompt)` 作首条用户消息（6 个任务首条用户消息同为 101 词元，tape `rl_harness_mismatch` 395–405、704–708 行）。阶段 1 同样受影响。
- KL 0.022 在 v0（未训练）即存在、两轮不变、与失配数无关 → 引擎与训练端数值差异，非训练漂移；建议重新标定门限。
- 已修（分支 s15-fncodex-l3）：7e22c37f 任务说明取 `instruction.md`，无法解析时 fail closed；dad2fd57 `rl_trajectory_reward` 增加 exit_status/turns/testsh_rc 等可选字段。单测通过，未上卡。
- 详见 `infra-drafts/FNCODEX-STAGE2-ANALYSIS.md`。下一步：TB2 专用系统提示（待裁定）、16384 上下文、1 轮全尺寸复跑（估 $55–65，需追加预算）。
- 2026-10-07 为 TB2 更换系统提示（d449ef37 起）：TB2 专用签名系统提示、任务说明 preflight、judge kl_max 0.03；r2 脚本 `s1-runs/s15-fncodex-full-modal-r2.sh`（1 轮、最坏 $54.4），PLAN_ONLY 通过，6 个任务首条用户消息均为 instruction.md 原文。见 REVIEW §13。

## 阶段 2 r2（2026-10-07，run `s15-fncodex-full-modal-20261007c`，代码 a978c15b，1 轮，≈$39.4）
- judge PASS（KL 0.0209），但 24/24 奖励仍为 0。新采集字段生效：第 1 次回复就违反协议 10 条、第 1 次回复被 4096 截断 4 条、上下文 8192 用尽 8 条、多回合后违规 2 条，submit 0 次，没有一条用满 12 轮。
- 8895dbc9：TB2 提示写明严格格式和预算；TB2 工具描述去掉 CTF 措辞（签名表面哈希 707e144f）；tape 增加 end_reason 和最后一次回复的形态。详见 STAGE2-ANALYSIS §6。下一步先做判分链路阳性对照（CPU 沙箱），上卡等用户裁定。
