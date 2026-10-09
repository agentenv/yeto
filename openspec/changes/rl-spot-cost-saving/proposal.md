# Proposal：rl-spot-cost-saving

## Why

生成（推理）占 RL 墙钟的大头，且可以被打断后续跑，适合放到便宜的 spot 卡上。业界 RLBoost（arXiv 2510.19225）把训练放按需、生成放可抢占卡，自报成本效率提升 28%–49%。yeto 已经在真机验证了续训切点（G2、N15）、整岛离开与重入（M1、V2）、重入数据位置（#149）、agentic 续跑与带在途轨迹的切点（agentic-rollout-utilization 阶段 3，#166）。缺的是：各云回收通知没人读、回收后只能在原云原区域重开、没有成本打分、没有"推理岛"这一种岛。

## What Changes

- 云能力静态表：每朵云一行，写回收通知秒数、通知获取方式、持久存储、是否支持 spot、价格来源，每格标"已核/未核"。
- 回收通知接口：收到通知 → 时间够就存切点（含在途轨迹）→ 整岛离开 → 记事件。先接 Modal（中断信号与退出处理），再接 AWS（实例元数据轮询）。
- 推理岛用 spot、训练岛保持按需。分期：先把评测岛放 spot；再在 elastic 下允许"可丢弃岛"；最后在跨岛样本传输做完后引入独立推理岛。
- 重开岛支持换区域、换云：候选按"单价÷有效算力＋准备成本＋回收期望损失"重新排序。自动开卡有用户确认开关，默认关。
- 回收与续跑结合：通知期内存在途轨迹，换岛后接着生成。与 agentic 回合间挂起、沙箱保活衔接。
- 替代 yeto-framework-decoupling 任务 8.7（回收接缝）。去耦合 8.1、8.5 的能力声明接口保留，本 change 的静态表是它的数据来源。

## Capabilities

### New Capabilities
- `cloud-preemption-notice`：云能力静态表与回收通知处理（存切点、离开、记事件）。
- `spot-rollout-islands`：推理型岛使用 spot、回收后换区域换云重开、成本打分、自动开卡确认开关。
- `preemption-inflight-resume`：回收时保存在途轨迹并在新岛续跑。

### Modified Capabilities
（无。现有 `openspec/specs/` 下的能力不改要求。）

## Impact

- 代码：`yeto/cloud/`（能力表、通知监听）、`yeto/launcher.py` 的岛恢复路径（`FleetController`，现只在原云原区域重开）、`yeto/shape/plan.py` 打分、elastic 客户端离开路径、Modal 岛入口的退出处理、agentic 续跑切点。
- 其他 change：yeto-framework-decoupling（8.7 被替代，8.1/8.5 引用本表；7.7 兼容组限制换云候选）、rl-infra-spec `cloud-pool-design.md`（§3.6 宽限期口径、§7.1 建议答案）、rl-inter-island-scheduling（离开/重入）、agentic-rollout-utilization（在途轨迹切点）、rl-eval-difficulty-buckets（评测岛）。
- 预算：上卡项都需报批，见 tasks。
