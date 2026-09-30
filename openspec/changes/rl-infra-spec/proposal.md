# Proposal

## Why

yeto 需要在 miles 后端上建立可安全重配置的岛内训推资源系统，使资源分配适应负载变化并提升端到端效率。现有串行共置仅作为兼容与性能基线；允许重构训练驱动、worker 生命周期和资源管理，不以维持现有结构为目标架构的限制。

## What Changes

- **前提**：以 change `rl-engine-ports`（R0）为基础，在其 `ports` 引擎路径上实现；引擎版本固定于 `michaellchung/miles` 与 `michaellchung/sglang` 的 `yeto/ports` 分支（见 rl-engine-ports design D6/D11），不再基于 `agentenv/miles` 与 bundle。
- 技术栈保持 **yeto + miles + Ray + Megatron/SGLang**；迁移 DynaResize 的机制思想，不引入 veRL 运行后端。
- 建立目标执行模式的固定训推分区基线，显式建模策略版本、样本依赖、可重叠任务与反压；物理分区不自动意味着算法允许并发。
- 岛内控制器（决策、护栏、journal）与执行（安全点、事务编排）都在 **yeto 侧**，每个 learner island 一份，由 yeto `IslandDriver` 在安全边界执行；miles 只通过 rl-engine-ports 的端口新增动词提供机制（rollout 增减/排空、trainer cut 保存/恢复/重建、放置重配置、按成员发布）。miles 的 cell 只在适配层内部使用，不使用 FT/indep-DP/healing/api_server。
- 阶段命名为 E0（固定分区基线）/ E1（rollout 增减）/ E2（同形恢复）/ E3（trainer DP 与角色转移），避免与 rl-engine-ports 的 B-2 接口粒度混淆。按“固定分区基线 → 手动重配置 → 半自动建议与人工触发 → 自动控制”交付。rollout-only、同形恢复是独立能力里程碑；trainer DP 与训推角色转移须另行认证，不以 rollout 成果代替。
- profiling 后选择 host staging、分块、通信准备、进程复用或延迟 optimizer restore；建立完整恢复依据、分级就绪与首个恢复 step 成本模型。
- 云租卡是实验和部署资源获取方式；首轮在一个已分配 GPU 池、固定 DiLoCo 成员中验证。预留资源池扩张、跨岛与多云调度接口，不在本轮实现在线云扩缩或成员变更。

## Capabilities

### New Capabilities

- `island-elastic-reconfiguration`: 面向目标执行模式的单岛有限配置、训推资源重分配、恢复、观测、半自动与自动决策。

### Modified Capabilities

无。miles 当前没有主 specs；yeto 在途云接入 change 保持独立。

## Impact

- 本轮仅更新规划文档。代码改动集中在 yeto 的 controller/driver 与 `yeto/rl/engine/` 端口及 MilesAdapter；miles fork 只在端口动词确需时增加小提交；优先可分批合并与独立验收，不把“最小代码差异”作为硬约束。
- yeto 后续负责执行模式契约、任务就绪、配置选择、云资源租用与外层同步边界；保留训练算法、样本使用及 DiLoCo 成员约束。新陈旧度/更新规则不属于普通资源开关，需单独明确算法契约。
- 引擎版本、源码校验与固定方式完全沿用 rl-engine-ports；本 change 不单独维护 pin。能力声明与 PR #66 弹性基准 harness 的 capability 认证格式保持一致。
- E2 开工时同步修改 `docs/MILES_RL.md` 中“不做 controller”一条：允许岛内 yeto 侧重配置控制器，仍不做跨岛控制器与通用恢复框架。
- 已阅读本地 DynaResize 原文，页码证据在 design 中就地标注；论文的 veRL 实现、H20 性能和异步语义不作为 miles 已有能力。DynaRL 仍仅作为用户提出的依赖建模方向，不引用未核实实现。
