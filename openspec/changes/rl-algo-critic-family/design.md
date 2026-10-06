# Design

## Context

动机见 proposal.md（Why）。以下只列决定实现方式的现状与约束（已核实，行号以当前工作树与 Miles c35702e 为准）。

**yeto 侧现状**
- `yeto/rl/engine/algorithm.py`：`ADVANTAGE_ESTIMATORS` 已含 ppo（:58-64），`CRITIC_ESTIMATORS={"ppo"}`（:65），但 v1 只接受 grpo（:51）；`ExecutionSpec.needs_critic`（:818）；`_reject_critic`（:1363，注册于 :1413）；`AdvantageSpec` 无 gamma/lambd（:653-670）。
- `algorithm_flags.py` `_UNMAPPED`（:200-209）中 `--gamma/--lambd/--value-clip/--num-critic-only-steps/--critic-load/--critic-lr` 出现即拒绝。
- `capabilities.py` `execution.critic`（:69），不匹配时报 "only --rl-engine legacy"（:321-324），文案不实：legacy `learner.py:1443` 写死 `--advantage-estimator grpo`，critic 只能经 :2245 extra_argv 透传。
- ports：`entry.py:236` 与 `fake.py:70` 声明 critic=False；阻断点 `entry.py:245-262`（`receipt_role_family` 遇 ppo raise）、`entry.py:1590-1592`、`trainer_rebuild.py:253,358`；`selection.py:45,68` 把 use_critic/非 grpo 路由到 legacy；`run_config.py` 无 critic 字段。
- 外层：strict-avg、decoupled（`yeto/rl/decoupled.py`）、trainable_state、run_config、elastic checkpoint store、tape/ledger 均未建模 critic；LayoutHash/receipt 只有 grpo 与 sao 两个 family。
- SAO（现有独立路径）：`local_learner.py:20-21` `_ROLES_BY_ALGORITHM={grpo:{actor}, sao:{actor,critic}}`；`miles_sao_streaming.py:3-7,148-173` actor/critic 独立 syncer 端口与 layout、lockstep 成对 fragment；`sao_streaming_runtime.py:508-509,562-598`（use_critic、`num_critic_only_steps==0`、`--sao-online-recipe`、critic 步数=actor×num_critic_epochs）；提交 60b61b3c、5bfc011a；验证见 docs/TBENCH21_SAO_QWEN35_08B_VALIDATION_20260826.md（EV≈+0.136，pass@k 无提升；actor→critic 经 Gloo/CPU）。SAO 算法本体在 legacy fork agentenv/miles ae475060，本地未查到源码。

**Miles c35702e（ports 镜像）**
- 参数：`--critic-num-nodes/gpus`（arguments.py:270-273）、`--num-critic-only-steps`（:1597）、`--critic-load/save/lr/lr-warmup`（:1603-1613）、`--value-clip=0.2`（:1648）、`--gamma/--lambd` 默认 1.0（:1729-1730）；估计器有 ppo 无 vapo（:1682-1690）；`use_critic` 由 estimator=="ppo" 推导（:3591）。
- Shared Actor/Critic 约束：不支持 indep_dp、只 megatron、kl_coef==0（:3592-3604）；critic GPU 数被赋值为 actor 的值（:3605-3606，静默覆盖而非断言，1.2 复核）；critic_load/lr 默认继承 actor（:3607-3610），强制 offload_train（:3708-3716）；rebuild 模式要求 `num_critic_only_steps==0`（:3212）；`--deploy-component trainer` 禁 critic（:3058）。
- 实现：placement_group.py:320,338 共卡；model_provider.py:340-341 1 维 value head；actor.py:227,604,635 value_loss；loss_hub/math_utils.py:647,705 GAE 入口、:875 vanilla_gae、:899 chunked_gae；losses.py:452 value loss；LoRA 下 critic 仍全参数（test_lora_model_branches.py:122-130）。示例 examples/ppo/、tests/e2e/megatron/test_qwen3_4B_ppo.py:76-86、test_shared_ppo_lifecycle.py。
- 没有：vapo、sao、hl_gauss、length_adaptive、decoupled_gae、value_pretrain、cross-segment GAE。CompactionRL 只有 rollout 侧 examples/experimental/terminus-compaction。

**约束**：不重复实现引擎（数学放 Miles/fork，yeto 只声明、校验、翻译、编排）；fork 改动只进 `michaellchung/miles` `yeto/ports`（先例 rl-algo-loss-variants 路线 B）；新机制先经 `--rl-allow-unverified-mechanism` 做 G1（1 卡），通过后才在 adapter 正式声明，G3 两岛 strict-avg 只用正式声明。

## Goals / Non-Goals

**Goals:**
- 用一套 critic 状态契约（layout family、receipt、外层同步、checkpoint、tape/ledger）覆盖 PPO、VAPO、SAO、CompactionRL，四个算法只在 GAE 与 value loss 变体上不同。
- GAE 变体在 fork 上只有一个扩展点，VAPO、SAO、CompactionRL 共用。
- critic LoRA 的接口与契约在首轮就预留，后续实现时不改契约结构。

**Non-Goals:**
- 不做 critic 与 actor 分卡/弹性分配、不做独立 critic DiLoCo（决策 1、4，后续探索）。
- 首轮不实现 critic LoRA（只出开发计划，见 D9）。
- 不在 yeto 中实现任何 GAE/value loss 数学。

## Decisions

### D1 critic 字段放在 AlgorithmSpec 的新 `critic` 组与 `advantage` 组
- `advantage` 组增加 `gamma`、`lambd`、`lambd_mode`（`fixed`|`length_adaptive`，带 `alpha`）、`gae_variant`（`vanilla`|`decoupled`|`cross_segment`）。
- 新 `critic` 组：`value_clip`、`critic_lr`、`critic_lr_warmup`、`critic_updates_per_step`、`value_loss`（`mse`|`hl_gauss`，后者带 bins）、`init`（`copy_actor_backbone`|`load`）、`warmup_steps`、`param_mode`（首轮只接受 `full`，`lora` 保留，见 D9）。
- 只有 `needs_critic` 为真时这些字段才进入规范化与算法哈希；grpo 默认哈希不变（P0 golden 用例不改即可通过）。`--gamma` 等从 `_UNMAPPED` 移入映射表，并做吸收与冲突检测。
- 备选：继续 extra argv 透传。否决：不进入哈希，两岛无法证明同一算法，也无法校验。

### D2 PPO 直接复用 Miles shared PPO
翻译为 `--advantage-estimator ppo` 及对应 `--gamma/--lambd/--value-clip/--critic-lr...`，`kl_coef=0`、colocated 共卡、offload_train 按 Miles 强制项生成。yeto 在启动前复刻 Miles 的 shared 约束（indep_dp、kl_coef、critic GPU 数、`--deploy-component trainer`），在 fake 组合根中先失败，不等 Miles 在 GPU 进程里报错。修正 capabilities.py:321-324 的文案为实际原因。
- 备选：分离式 critic（独立 GPU）。否决：Miles c35702e 未提供，违背决策 4。

### D3 ports 放开顺序：单岛 colocated 优先
先放开 `entry.py:245-262` 的 receipt family、`entry.py:1590-1592`、`trainer_rebuild.py:253,358` 与 `selection.py` 路由，并在单岛下用 `--rl-allow-unverified-mechanism execution:critic` 跑 G1；G1 通过后 `entry.py:236` 正式声明 `critic=True`，fake.py 同步。两岛只在 D4 的状态契约完成后开放。

### D4 critic 状态契约：critic 作为与 actor 并列的第二个 role
- LayoutHash：新增 `ppo_family`（actor+critic），沿用 SAO 的双 layout 思路：critic layout 由 backbone layout + value head 形状决定，与 actor layout 分开哈希，receipt 中同时记录两者与 `critic.param_mode`。
- 外层同步（决策 1）：strict-avg 对 actor 与 critic 分别做平均，同一轮内先 actor 后 critic，两者都完成才提交该轮；任一 role 失败整轮作废。decoupled 外层遇 critic 直接拒绝（后续探索）。
- checkpoint/恢复：elastic checkpoint store 增加 critic 子目录与 critic 优化器状态；恢复时 actor/critic 必须来自同一轮，否则拒绝。
- tape/ledger：每轮记录 critic 权重哈希、value_loss、explained variance。
- 备选：把 critic 当 actor 的附属张量一起哈希。否决：critic LoRA（D9）与 SAO 双 syncer 都需要独立 role。

### D5 warm-up 初始化与 rebuild 模式冲突的解法（决策 3）
Miles 的 `--num-critic-only-steps` 在 rebuild 模式下必须为 0（arguments.py:3212），而 ports 走 rebuild。解法：把 warm-up 拆成 yeto 编排的独立阶段。
1. 阶段 W（critic-only）：用非 rebuild 的单次 Miles 启动（与 Miles 原生 PPO 示例相同模式），`--critic-load` 指向初始 actor checkpoint（复制 backbone，value head 由 model_provider 新建），`--num-critic-only-steps=warmup_steps`，跑完 warm-up 后只保存 critic checkpoint，actor 权重不变（校验 actor 哈希前后一致）。
2. 主阶段：ports rebuild 模式，`--num-critic-only-steps=0`，`--critic-load` 指向阶段 W 的输出。
3. 阶段 W 产物以内容哈希进入 receipt（`critic.init=load` + 来源哈希）；两岛共用同一 W 产物，只做一次。
- 备选 a：放宽 fork 上 :3212 约束。否决：rebuild 下 critic-only 步与 actor 同步节奏未定义，改动面大。备选 b：主阶段前若干轮把 actor lr 置 0。否决：仍会走 rollout 权重同步与优化器状态，哈希与成本都不干净。

### D6 fork 上唯一的 GAE 扩展点
在 `yeto/ports` 分支 math_utils.py 现有 vanilla/chunked GAE 旁加一个按 `--gae-variant` 分派的入口：
- `length_adaptive`：λ=1−1/(α·l)，l 为序列响应长度（VAPO、SAO、CompactionRL 共用，α 默认 1.5）。
- `decoupled`：critic 目标与 actor 优势用不同 λ（VAPO）。
- `cross_segment`：段内局部 GAE `A^loc_{s,i}=∑_{ℓ=0}^{n_s−i}(γλ)^ℓ δ_{s,i+ℓ}`，再乘 `(γλ)^{N_{>s}}`，`N_{>s}=∑_{j>s} n_j`；终局回报放在最后一段段尾，不跨压缩边界自举（CompactionRL）。
- 段边界作为样本元数据（每 token 的 segment id）随 batch 传入；无边界时退化为 vanilla。缺省参数下逐元素等于原实现。
- 先用 yeto 仓库内独立 torch 参考实现（不 import 被测代码）对拍，仿照 rl-algo-loss-variants D3。

### D7 VAPO 与 SAO
- VAPO = PPO + length_adaptive + decoupled GAE + value pretrain（复用 D5 阶段 W）+ 论文中的其它组件；具体变体参数为开放问题，不影响结构。
- SAO 迁移：把 `sao_streaming_runtime.py` 的 recipe 翻译为 AlgorithmSpec（sao_dis、α=1.5、γ/λ=1/1、HL-Gauss 51-bin、critic 步数=actor×num_critic_epochs），保留双 layout、双 syncer、lockstep 成对 fragment 语义。SAO 特有数学（HL-Gauss value loss、sao_dis）需从 agentenv/miles ae475060 移植到 `yeto/ports`；该源码本地未查到，第一步是取得并核对。迁移完成前现有 SAO streaming 入口保持可用。

### D8 CompactionRL（决策 5）
- rollout 侧（yeto/agent 路径）：剩余上下文 `C−|h_t| < T_comp`（10,240）时触发；同一策略按 `<analysis>/<summary>` 9 节模板生成摘要；重建 `h̄_t = s ⊕ u_resume(S_t) ⊕ 最近 k=2 步`；每条最多 3 次压缩；摘要段与任务共享回报；每段输出 segment 元数据。尽量复用 Miles `examples/experimental/terminus-compaction` 的 rollout 代码。
- 训练侧：PPO clip、token 级归一化（批内全部被优化 assistant token 平均）、KL=0、每提示 1 条 rollout、critic lr 3e-6、每批 2 次 critic 更新对 1 次策略更新（`critic_updates_per_step=2`）、50 步 warm-up（D5）、cross_segment GAE + length_adaptive λ（D6）。
- 消融验收：关掉 cross_segment（退回 vanilla）作为对照臂，只在用户另批预算时跑。

### D9 critic LoRA 开发计划（决策 2，首轮不实现）
- 接口预留（首轮实现）：`critic.param_mode ∈ {full, lora}`，`lora` 下 `critic.lora_rank/alpha/target_modules`；首轮校验阶段对 `lora` 明确拒绝并提示"计划中"；receipt 与 critic layout 中写入 param_mode 与 LoRA 形状，使后续不改契约结构。
- fork 需要改的（后续阶段）：Miles 当前在 LoRA 下跳过 critic 的 LoRA 设置（test_lora_model_branches.py:122-130）。需在 model_provider/LoRA 包装处让 critic 也挂 adapter，并决定 value head 是否全参数（开放问题，默认全参数可训练）；critic checkpoint 只存 adapter+value head；critic-only warm-up 与 offload 路径兼容 adapter。
- yeto 需要改的：strict-avg 对 critic 只平均 adapter+value head；layout hash 区分 full/lora；checkpoint 存储与恢复按 adapter 粒度；`init=copy_actor_backbone` 下 backbone 冻结共享，可考虑与 actor 共用一份 backbone 权重（省显存，需单独评估）。
- 验证：CPU 上 fork 单测（critic LoRA 参数数、冻结掩码、value head 可训练）；dry-run argv 快照；G1 1 卡对比 full critic 的 EV 曲线；G3 两岛 strict-avg 只平均 adapter 后哈希一致。
- 备选：首轮直接做 LoRA critic。否决：Miles 无现成实现，先用全参数确立基线（决策 2）。

## Risks / Trade-offs

- [共卡 + offload_train 导致显存与时长翻倍] → G1 用 0.5B 级小模型；记录每轮时长作为后续分卡决策依据。
- [strict-avg 平均 critic 与 value head 可能使 value 估计偏移] → G3 记录平均前后两岛 EV；明显下降时作为外层探索课题，不阻塞首轮。
- [阶段 W 使用非 rebuild 启动，代码路径与主阶段不同] → 阶段 W 结束时校验 actor 哈希不变、critic 产物可被 rebuild 模式 `--critic-load` 加载（CPU dry-run + G1）。
- [SAO 源码在 ae475060，本地未查到] → 迁移任务第一步取得源码，取不到则 SAO 迁移暂停，不影响其它组。
- [CompactionRL 无官方代码，复现偏差] → 公式级参考实现对拍；超参数严格按论文；结果标注"复现，未与论文数值对齐"直到实测。
- [fork pin 更新引入回归] → 缺省参数逐元素不变测试 + 原有 fork 测试全过才更新 pin。
- [elastic 与 critic 不能共存] → 校验阶段明确拒绝并给出原因（决策 4）。

## Migration Plan

1. 字段与翻译先落地，默认 grpo 哈希与 argv 快照不变；critic 算法在 adapter 正式声明前只能经 unverified 放行在单岛使用。
2. 每个 fork 改动经用户同意后 push 到 `yeto/ports`，更新 `MILES_NEXT_COMMIT` 与镜像；回滚 = 回退 pin 到上一提交。
3. SAO 迁移期间旧 streaming 入口保留，新路径 G3 通过后再标记旧入口弃用（不删除，删除另议）。

## Open Questions

- VAPO 的具体变体参数（decoupled λ 取值、是否含 positive-example LM loss 等）在 VAPO 组开始前按论文核定，不影响结构。
- critic LoRA 下 value head 是否全参数（默认全参数），在 D9 后续阶段 CPU 原型时定。
- CompactionRL 的数据集与 agent 环境（Terminal-Bench 子集或其它）在该组 GPU 报批时选定。
- critic LoRA 是否与 actor 共享冻结 backbone 以节省显存，待后续评估。
