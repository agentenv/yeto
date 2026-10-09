# agentic-rollout-utilization：agentic RL 生成阶段的 GPU 充分利用

## 为什么
- 前期主打先跑通、要确定性：yeto 规定同一批样本必须全部由刚发布的那一版权重生成（`max_policy_age` 恒为 0，`yeto/rl/engine/execution_profile.py:205`），续训切点要求剩余缓冲为 0（`yeto/rl/engine/cut.py:314`）。S17 五个目标真机通过、N15 证明不中断时逐位可复现，确定性基线已经立住，可以开始在不破坏它的前提下提升利用率。
- 实测（infra-drafts/AGENTIC-GPU-UTIL-RESEARCH.md）：M1（4B 两岛）每轮生成段 85–200 s，一半轨迹 45–56 s 就跑完，后面基本在等最慢的约 10%；SGLang KV 缓存峰值只用 12%；在途轨迹约一半不在模型生成（在等工具、沙箱或判分）。评测岛 GPU 平均利用率 8.8%，G6 生成段 GPU 利用率 27–42%。
- 用户最初的想法："凑够训练要的条数就开始训练，没跑完的下一轮接着跑"，即多发请求（over-sampling）+ 部分 rollout（partial rollout）。Miles fork 已有这两个开关，但多轮和 agentic 路径直接禁用；APRIL、slime 都没有多轮工具调用的续跑方案。

## 改什么（分四个阶段，后一阶段以前一阶段的实测为门槛）
0. **只多发、凑够即截止（不续跑）**：放开 yeto 侧多发请求的配置，截止时没完成的轨迹丢弃并计数；补采每条轨迹起止时间、每回合"模型生成 / 工具执行 / 判分"三段耗时、沙箱冷启动、被丢弃轨迹已生成 token 数。A/B 已并入 N17 合并验证运行。
1. **设计与开关（本 change 主体）**：引入"权重版本落后上限"开关，默认 0 = 现在的确定性模式，且写进契约哈希；定义样本按版本段记账、续训切点携带未完成轨迹的格式、多岛账本按版本段合并、TIS 对跨版本 token 的修正与上报。
2. **单轮任务续跑**：上限设 1，在小模型单轮任务上对照收益、TIS 截断比例、奖励曲线。
3. **agentic 多轮续跑**：沙箱跨轮存活、只在两个模型回合之间中止、对话前缀跨版本保留（FN 无前缀缓存时的重算代价单独评估）。

**verl 并行线**：落后上限是后端中立的开关，Miles 与 verl 两条适配线同步推进、共用阶段门槛。verl 侧以 fork 里的 `experimental/fully_async_policy`（`partial_rollout` + `staleness_threshold`，基于 `AgentLoopManager`，含 `tool_agent_loop` 多轮工具调用）为接入点：阶段 0 用 verl 同步模式的多发与截止，阶段 2 起把 `staleness_threshold` 由落后上限推导。verl 侧多轮工具调用的续跑是否已可用要实测，若可用可能先于 Miles 达到阶段 3。

## 不改什么
- 默认行为不变：落后上限为 0 时，标准样本、契约哈希以外的所有输出逐字节不变；确定性模式继续作为对照基线。
- 不做完全异步训练（AReaL/verl fully_async 式），不做跨岛 spot 弹性扩生成（RLBoost 式），不做 KV 缓存按时限保留（Continuum 式）；这些记入 design 的备选，等本 change 数据出来再议。

## Capabilities

### New Capabilities
- `agentic-rollout-utilization`：多发请求与截止、权重版本落后上限开关、版本段记账、未完成轨迹续跑（单轮与 agentic）、利用率指标采集与上报。

### Modified Capabilities
（无。续训切点、多岛账本的改动作为本能力的要求写入；`rl-resume-from-checkpoint`、`rl-inter-island-scheduling` 仍是在途 change，合并归档时再对齐。）

## 影响
- 代码：yeto 核心（execution_profile、driver 的版本检查、cut 切点、island_ledger/sample_pool、launcher 多发配置、tape/看板指标）；Miles 适配层（algorithm_flags 映射 partial-rollout/over-sampling、rollout 中止钩子、multi_turn/agentic_tool_call 的断言需在我们的 fork 上放开）；codex harness（逐条版本检查改为按段记录）；verl 适配层（`entry.py:26` 声明 `max_policy_staleness=0` 改为按开关声明；`config.py` 生成 fully_async 配置；verl 用 vLLM 推理，指标采集需单独接）。
- 契约：新增开关进契约哈希；默认值下哈希迁移记录进 hash-migration.md。
- 费用：阶段 0 约 $20–25（并入 N17）；阶段 2、3 各自单独预算，上卡前报批。
- 约束：Miles fork 改动只在 agentenv/miles，不向上游提 PR。
