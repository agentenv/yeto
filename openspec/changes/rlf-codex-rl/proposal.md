# Proposal

## Why

Yeto 已有 Codex agentic rollout、可信 reward 签名和 Modal 沙箱，但它们目前围绕 TB2 运行器组织，尚未把 rlenvfactory 的 cyber 环境生成、任务验收、环境生命周期和原生判分接成一条可训练链路。结果是 RLF 任务可以独立生成或判分，却不能稳定地产生带完整轨迹、策略版本和可信 reward 的 Yeto RL 样本。

现在需要把这条链路作为一个完整能力落地，先在 Modal GPU 上验证训练闭环。第一阶段只使用 Codex 作为解题 harness；SWE-bench 及其他 harness 不属于本 change 的训练范围。

## What Changes

- 接入 rlenvfactory 的已验收 cyber 任务包，并保留来源、任务包、镜像和 grader 身份。
- 将 AgentFlow 作为 RLF 离线任务生产/验收阶段的可选 DAG 编排器；只消费其发布的已验收任务包，不把 AgentFlow 的普通 node output 当作 RL rollout 数据。
- 增加面向 agentic RL 的环境 provider：按 trajectory 创建隔离 episode，向 Codex 提供工具执行接口，结束后由可信侧调用 RLF grader 并清理环境。
- 复用现有 Codex subprocess harness，保留 prompt、response、tokens、logprobs、loss mask、segments、episode 身份和 behavior policy version。
- 将 RLF 的原生判分结果转换为中立 reward；success 元数据必须写入轨迹，环境故障、未准入和清理失败不得静默变成普通负样本。
- 保留 RLF 的超时语义：Codex 超时后仍对可判分的环境状态执行可信判分；判分结果与清理证据分开记录。
- 将 provider 接入 Modal 运行路径，支持今晚 GPU RL 测试使用 Modal；环境侧使用 Modal VM Sandbox 运行需要 Docker 的 RLF 生命周期。
- 增加从任务生成、静态和运行验收、Codex 解题、可信判分到 Yeto RL 更新的 CPU 对照、Modal 沙箱验证和小规模 GPU 训练门禁。
- 为今晚的 Modal GPU 验证固定 Qwen3-30B-A3B thinking baseline：先以 32K 单窗口完成能力基线，再按 `max_seq_len_hit` 和 trace 膨胀情况启用 24K compaction；baseline 与 RL gate 都必须记录 reward 方差和非零 GAE 证据。
- 明确第一阶段不接 SWE agent 沙箱、不接 RLF 的其他 Solver harness、不改变现有 TB2 生产 app 的默认行为。
- 明确在线 RL rollout 不依赖 AgentFlow，继续使用 Yeto Codex harness 负责 token/logprob/loss mask、segments、policy version 和 trusted finish。

## Capabilities

### New Capabilities

- `rlf-codex-rl`: 提供 rlenvfactory cyber 任务（可由 AgentFlow 离线编排生成并验收）到 Yeto Codex agentic RL 训练的完整端到端流程，包括任务身份、episode provider、可信判分、轨迹元数据、故障分类、清理证明和 Modal 验证门禁。

### Modified Capabilities

- None. Existing capability specs do not define this RLF-to-Codex training behavior; the existing `rl-agentic-reward-env` change remains the benchmark-neutral TB2/SWE adapter work and is not broadened here.

## Impact

- Yeto 的 `yeto/rl/harness/codex/`、`yeto/rl/harness/reward_env/`、`yeto/cloud/`、中立 reward 类型与 Miles/verl reward adapter。
- RLF 任务包、任务身份和 grader 结果的导入边界；可能需要 pin rlenvfactory commit 和任务包 digest。
- AgentFlow 仅作为离线 Creator/QA/reference-solve DAG 的可选依赖；在线训练路径不得依赖其调度器或普通 trace。
- Modal provider、VM Sandbox/Docker 运行依赖、Secret 传递和清理证据。
- Codex rollout metadata、training data acceptance 和 RL launch preflight。
- 新增 CPU 单测、fake RLF provider、Docker/VM smoke、Modal CPU environment smoke 和 Modal GPU RL evidence。
