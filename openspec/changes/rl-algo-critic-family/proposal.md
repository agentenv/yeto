# Proposal

## Why

P0（`rl-algorithm-capabilities`）把 critic 家族（PPO、VAPO、SAO、CompactionRL）列为"暂未立项"，原因是"需要 critic 状态和外层归属"。目前 ports 引擎在启动前拒绝所有需要 critic 的算法。报错文案说 legacy 可以驱动 critic，但核查结果显示 legacy 也没有结构化的 critic 支持：`learner.py:1443` 把估计器写死为 grpo，critic 只能靠 extra argv 透传进去。唯一真正跑过 critic 的是 SAO streaming。它走一条独立的全参数路径，不经过 AlgorithmSpec，在 Qwen3.5-0.8B、8×H200 上做过端到端验证（docs/TBENCH21_SAO_QWEN35_08B_VALIDATION_20260826.md）。本 change 让 critic 家族在 ports 上可声明、可校验、可验证，先补齐 P0 欠下的 critic 状态契约和外层归属。

## What Changes

- **已确认的决策（用户，2026-10-06）**：
  1. 两岛外层同步：critic 与 actor 一起做 strict-avg，先跑通；独立 DiLoCo 等其它方式留作后续探索。
  2. critic 参数模式：先全参数跑通（与 Miles 现状一致）；本 change 的 design 与 tasks 必须包含 critic LoRA 的开发计划（预留接口与契约，不在首轮实现）。
  3. critic 初始化：复制初始 actor 的 backbone，加新的 value head，再做 critic-only warm-up（与 CompactionRL 的"同一 checkpoint 初始化 + 50 步 value pretraining"一致）。
  4. 接受 Miles shared PPO 的限制（不支持 indep_dp 因而不与 elastic 共存、`kl_coef` 为 0、critic 与 actor 共用训练 GPU）；更好的弹性分配方式后续另行探索。
  5. CompactionRL 指 arXiv:2607.05378（Li 等，v2 2026-10-01）：单策略同时优化任务执行与上下文压缩摘要；PPO 风格 clip 目标，token 级归一化，KL=0，单 rollout/提示；critic 与策略同一 checkpoint 初始化加标量 value head，50 步 value pretraining，critic lr 3e-6，每批 2 次 critic 更新对 1 次策略更新；cross-segment GAE（段内局部 GAE，再乘 (γλ)^{N_{>s}} 修正，不跨压缩边界自举），γ=1，长度自适应 λ=1−1/(αl)、α=1.5；rollout 侧在剩余上下文 < T_comp（10,240）时触发压缩，摘要由同一策略生成、与任务共享回报，每条 rollout 最多 3 次压缩。论文用 slime 训练，未见代码发布。
- **AlgorithmSpec 扩展**：新增 critic 相关字段（gamma、lambd、value_clip、critic_lr、critic-only 步数、critic 初始化方式与 warm-up 步数），从 `_UNMAPPED` 中移出并翻译成 Miles 参数，进入算法哈希。修正 legacy 报错文案。
- **PPO（GAE）on ports**：复用 Miles 原生实现（value head、vanilla/chunked GAE、value loss、critic load/save/offload）。放开 entry、trainer_rebuild 与 receipt 中对 critic 的阻断，ports 声明 `execution.critic=True`，先只支持单岛 colocated。
- **critic 状态契约**：新增 critic 的 layout family 与 receipt；外层同步、checkpoint/恢复、tape/ledger 纳入 critic。
- **VAPO**：在 `michaellchung/miles` 的 `yeto/ports` 分支上实现 Miles 没有的部分（length-adaptive GAE、decoupled GAE、value pretrain），yeto 侧声明与翻译。
- **SAO 迁移**：把现有 SAO streaming 接入 AlgorithmSpec 与 ports，保留已验证的双 layout、双 syncer 语义。
- **CompactionRL**：rollout 侧加入压缩与摘要段（复用 Terminal-Bench/agent 路径与 Miles `examples/experimental/terminus-compaction` 可复用部分）；训练侧在 fork 上实现 cross-segment GAE、长度自适应 λ 与 token 级归一化（与 VAPO/SAO 共用 GAE 扩展点）。
- **验证**：每步先做 CPU 单测与 dry-run argv 快照；GPU 验证用小模型，对齐 Miles `test_qwen3_4B_ppo` 的指标（value_loss、explained variance），两岛做 bit 级一致性检查，并做 kill/resume。GPU 计划与预算另行报批。

## Non-goals

- 不向 radixark/miles 或 sgl-project/sglang 提交任何改动。
- 不在本 change 中支持 critic 与 elastic 重配置共存（受 indep_dp 限制，见决策 4）。
- 不做异步目标（staleness>0）和 Flash-Next 上的 critic。

## Capabilities

### New Capabilities
- `rl-critic-algorithms`：critic 家族算法在 ports 上的声明、校验、翻译、状态契约（layout、receipt、外层同步、checkpoint）与验收要求。

### Modified Capabilities
（无：`openspec/specs/` 下目前只有 `head-run-teardown`，与本 change 无关。）

## Impact

- yeto：`yeto/rl/engine/algorithm.py`、`algorithm_flags.py`、`capabilities.py`、`selection.py`、`run_config.py`、`miles_adapter/entry.py`、`trainer_rebuild.py`、`local_learner.py`、外层同步（strict-avg、decoupled）、elastic checkpoint store、SAO streaming 模块。
- fork：`michaellchung/miles` `yeto/ports` 分支（VAPO 部分），需要更新 ports 镜像与 pin。
- GPU 预算：PPO 单岛冒烟、两岛一致性、kill/resume、VAPO、SAO 迁移验证，金额待 design 估算后报批。
