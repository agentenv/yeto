# yeto RL 算法能力：研究与工程决策材料

状态：explore 阶段产物，未 propose、未实现、未提交。

## 0. 基线、证据等级与约定

| 项 | 值 |
|---|---|
| yeto | 分支 `rl-algorithms` 于 `18695ae`（与 `origin/rl-engine-ports` 相同） |
| Miles 基线 | radixark/miles `9e4260d`。下文 Miles 行号均取自该提交（`git show 9e4260d:<path>`） |
| Miles fork 增量 | michaellchung/miles `yeto/ports` 在 `9e4260d` 之上有 `e8b3b65`（run_plugin）和 `0394715`（worker 动态端口）。改动 7 个文件：`miles/ray/train/group.py`、`miles/ray/train_actor.py`、`miles/utils/arguments.py`（+9 行）、`addr_allocator.py`、`ray_worker_manager.py` 以及两个测试。**不涉及 loss、advantage、rollout 转换代码**，所以本文所有算法能力在基线和 fork 上相同；只有 `arguments.py` 在 fork 上行号后移约 9 行 |
| 缩写 | `A` = `miles/utils/arguments.py`；`LH` = `miles/backends/training_utils/loss_hub/` |

证据标签，每条结论都标注其一：

- **[论文]**：论文原文报告，未复现。
- **[源码]**：本次在上述提交里读过的代码，给出文件和行号。
- **[推断]**：由论文或源码推出，没有实验或原文直接支持。
- **[未核实]**：来源拿不到或不确定。

**本次研究没有运行任何算法的 CPU 测试或 GPU 实验。** 下文"可以做 CPU 验证"指的是验证方案，不代表已经通过。

---

## 1. 关键发现（决策前必读）

1. **[历史基线 yeto `18695ae`，已由 rl-algorithm-capabilities 改变，现状见 §8]** **ports 当时只允许 GRPO**，另外可选有界非零方差过滤和 `kl_coef`。见 [源码] `yeto/rl/engine/algorithm.py:22-24`；能力声明在 `miles_adapter/entry.py:49-50`，只有 `{"grpo"}`。

2. **Miles 基线已原生实现大部分无 critic 机制**，都是 [源码]：
   - GSPO：`--advantage-estimator gspo`
   - clip-higher：`--eps-clip-high`
   - dual-clip：`--eps-clip-c`
   - Dr.GRPO 去 std：`--disable-grpo-std-normalization`
   - token 级 loss：`--calculate-per-token-loss`
   - TIS：`--use-tis`
   - IcePop：通过 `--custom-tis-function-path` 指向内置函数
   - OPSM：`--use-opsm`
   - REINFORCE++ 及其 baseline 版：对应的 estimator
   - KL loss：`--use-kl-loss`
   - entropy：`--entropy-coef`

   第一阶段的主要工作是在 yeto 这一侧完成规范化、声明和证明，**不需要改 fork**。

3. **provenance 漏洞** [源码]：
   - `check_extra_argv`（`miles_adapter/config.py:405-409`）只拦截 `ADAPTER_OWNED_FLAGS`（`config.py:71-96`）。
   - `--eps-clip*`、`--use-tis`、`--loss-type`、`--custom-loss-function-path`、`--custom-reward-post-process-path`、`--use-kl-loss`、`--entropy-coef`、`--calculate-per-token-loss`、`--disable-grpo-std-normalization` 等都**可以通过 extra argv 透传**。这样改变了训练目标，`algorithm_sha256` 却不变。

4. **`AlgorithmSpec.kl_coef` 在 GRPO 下不影响梯度** [源码]：
   - `--kl-coef ≠ 0` 会让 Miles 加载 ref 模型（`miles/ray/specs/train.py:58`）并逐 token 计算 KL（`miles/backends/training_utils/loss.py:81-97`）。
   - 但 `grpo` 和 `gspo` 的 advantage 只把 reward 广播到每个 token，KL 被丢弃（`LH/advantages.py:53-57`；`LH/math_utils.py:462-469` 的 `get_grpo_returns`）。
   - KL 只有在 `--use-kl-loss` 时才进入 loss（`LH/losses.py` 的 `use_kl_loss` 分支）。
   - 因此 `kl_coef>0` 加 GRPO 的效果是多占一份 ref 模型的显存和前向算力，梯度没有任何变化。
   - legacy 的 `expert_full` 分支固定传 `--kl-coef 0.001`（`yeto/rl/learner.py:1068`）。legacy 用的是 agentenv fork，那边的 GRPO 是否同样忽略 KL **[未核实]**。

5. **在 ports 的串行 driver 下，外层同步不会让 rollout 数据变旧** [源码]：
   - 每轮的顺序是：用已发布策略生成，训练一次，在安全点 apply cut，再发布（`driver.py:1-10, 418-443`）。
   - 生成的组必须带有当前已发布策略的 token，否则拒绝（`driver.py:332-340`）。
   - 因此行为策略等于训练起点的权重，strict-avg 和 decoupled 都一样。
   - 实际存在的"离策略"来源只有两个：
     - (a) SGLang 和 Megatron 在同一权重下的数值差异，即训推不一致；
     - (b) 同一轮内多个 mini-batch 带来的 PPO 常规偏移。
   - 我在第一轮 explore 说过"decoupled 下 TIS 是正确性问题"，**这个判断不成立，已撤回**。当前所有执行模式的策略陈旧度都是 0；rl-infra-spec 不会通过资源调度开放陈旧度 >0，陈旧度 >0 的算法需要另立独立的算法契约 change（alignment.md A6）。

6. **零梯度不变量是按 GRPO 写死的** [源码]：
   - `driver.py:355-370` 用 `any(g.reward_std > 0)` 判定"advantage 非零"，一旦 grad_norm 为 0 就让本轮失败。
   - IcePop、OPSM、MIS、GSPO 的窄 clip 都可能合法地把整批 token 掩掉。
   - 所以这个不变量必须改成随算法变化的判定，否则新算法会被误判失败。

7. **算法哈希没有进入导出的 provenance，也没有进入外层成员校验** [源码]：
   - 事件里有 `rl/algorithm_spec_sha256`（`driver.py:240`、`entry.py:178`）。
   - 但 `yeto_rl_provenance.json` 只写了 `rl_engine`（`yeto/rl/export.py:386-391`）。
   - 外层身份 `CanonicalLoraState` 只校验 base revision、LoRA 配置和 layout（`yeto/rl/core.py:57-76`）。
   - 这意味着两个岛用不同算法也能被平均。spec 要求"写入产物来源记录"，这一条目前未满足。

8. **critic 在 ports 上被拒绝的原因有多层** [源码]：
   - `IMPLEMENTED_LAYOUTS={lora}`（`trainable_state.py:21`），状态里没有 critic。
   - ports 要求 actor TrainGroup 恰好 1 个 cell（`config.py:729` 的 `SingleCellError`），而 Miles 的 critic 是独立的 trainer 组（`A:3510-3525`，`use_critic = advantage_estimator == "ppo"`）。
   - spec `rl-engine-selection` 明确要求拒绝 critic 和 SAO（`selection.py:56` 会把 SAO 路由到 legacy）。
   - 需要 critic 的算法包括 PPO、VAPO、SAO、CompactionRL，**都不能进入近期计划**。

---

## 2. 把算法拆到训练环节

```
 prompt --> [采样/分组] --> [奖励] --> [奖励后处理/advantage] --> [ratio] --> [clip/截断/mask] --> [loss 聚合] --> 优化器
              |              |              |                      |              |                    |
            G, 超采样,     rm, 长度塑形,  组归一, 留一, 全局白化,   old vs rollout  PPO clip, dual,      per-sample mean,
            动态过滤,      多奖励向量      /mean, 能量基线,         vs 当前,        seq clip, 软门控,     per-token, 常数分母
            树状分支                       GAE(critic)              seq 几何平均    TIS/IcePop/MIS/OPSM
                                                                                   + KL(reward 或 loss) + entropy
```

四种性质：

| 性质 | 成员 | 说明 |
|---|---|---|
| **完整算法**（一整套配方） | PPO、GRPO、DAPO、VAPO、SAO、CompactionRL | 由下面的机制组合而成。实现时应拆成机制，不按名字单独做 |
| **可组合机制** | clip-higher、dual-clip、token 级聚合、Dr.GRPO 去 std / 去长度归一、GSPO 序列 ratio、GMPO、CISPO、SAPO-Qwen 软门控、RLOO、REINFORCE++、MaxRL、MAPO、GDPO、OTB、GiGPO、动态采样、overlong 塑形、TIS、IcePop、MIS、OPSM、KL 放置位置 | 每个机制只作用于一两个环节 |
| **训练系统方案** | ECHO、INTELLECT-2、PipelineRL、AReaL、PULSELoCo、SAPO-Gensyn（共享经验）、ARPO（rollout 树） | 影响执行、同步和数据流，不是 loss |
| **不更新权重** | Training-free GRPO | 与 yeto 的权重同步无关，不纳入 |

名字不同但数学上接近的，**不需要各自实现**：

- **RLOO 与 Dr.GRPO**：留一基线 r_i − mean_{−i} = G/(G−1)·(r_i − mean)，所以关闭 std 后两者只差一个常数缩放 [论文：Dr.GRPO 原文也这么说]。Adam 对梯度的整体缩放近似不变，所以在 yeto 里 RLOO 约等于 Dr.GRPO 的 advantage，差别只在 KL 的放置位置 [推断]。不必单独实现。
- **REINFORCE++-baseline**：等于组内减均值，再做全局 batch 白化。Miles 的 `reinforce_plus_plus_baseline` 加 `--normalize-advantages` 已覆盖 [源码]。
- **DAPO**：等于 clip-higher、token 级聚合、动态采样、overlong 塑形、去 KL 这几个机制的组合，不是一个新的 loss。
- **AReaL 的解耦 PPO 与 Miles 的 TIS**：AReaL 的目标是 (π_prox/π_behav)·clip(π_θ/π_prox)。Miles `vanilla_tis_function` 做的是 clamp(exp(train_old − rollout)) 乘以 PPO 项（`LH/corrections.py:7-32`）：train_old 相当于 π_prox，rollout 相当于 π_behav，clip 以 π_prox 为锚。所以 **Miles 的 TIS 就是截断版的解耦 PPO** [推断，公式对照]。

同名但不是同一篇的，已按原文核实：

- **SAPO**：Qwen 的 Soft Adaptive PO（2511.20347），与 Gensyn 的 Swarm sAmpling PO（2509.08721）无关。
- **GDPO**：NVIDIA 的多奖励解耦归一化（2601.05242）；另有一篇扩散语言模型的 GDPO（2510.08554），与本任务无关。
- **ARPO**：Agentic RPO（2507.19849）；另有一篇 GUI 经验回放的 ARPO，编号凭记忆，[未核实]。
- **MAPO**：Mixed Advantage PO（2509.18849，ICLR 投稿已撤回），与 2018 年的 Memory Augmented PO 无关。
- **SAO**：Single-Rollout Async Optimization（2607.07508）。legacy 已有的 SAO（`selection.py:56`、`learner.py:1845` 的 `sao_compaction`）是否就是这篇的实现 **[未核实]**。作者和 CompactionRL 相同，推测是同一条技术线。
- **OTB**：Optimal Token Baseline（2602.07078）。
- **ECHO**：2508.05387。
- **MaxRL**：2602.02710。
- **CompactionRL**：2607.05378。

---

## 3. 算法对照

记号：
- 组内 G 条样本，奖励为 R_i；μ、σ 为组内均值和标准差。
- ρ_t = π_θ/π_old。
- 以 GRPO 为参照：Â_i = (R_i − μ)/σ，loss 为 (1/G)Σ_i (1/|o_i|)Σ_t min(ρÂ, clip(ρ, 1±ε)Â) − β·KL_k3。

| 算法 | 核心公式或机制 | 相对 GRPO 的变化 | 所需数据和状态 | 实证（论文报告） | 链接 |
|---|---|---|---|---|---|
| PPO | min(ρA, clip(ρ,1±ε)A)，A 由 GAE 给出；InstructGPT 把逐 token KL 放进 reward | 用 critic 加 GAE 替代组基线 | critic、value 状态、ref；每个 prompt 1 个样本即可 | MuJoCo 上 ε=0.2 最好（0.82 对不裁剪的 −0.39）；1.3B InstructGPT 比 175B GPT-3 更受偏好 [论文] | 1707.06347，2203.02155 |
| GRPO | 如上；k3 估计的 KL 直接加进 loss | 参照 | ref（β>0 时）；token 级 π_old | DeepSeekMath-Instruct 7B 经 RL 后 GSM8K 82.9→88.2，MATH 46.8→51.7 [论文] | 2402.03300 |
| Dr.GRPO | Â = R − μ，除以常数 MAX_TOKENS | 去掉 σ 和 1/\|o\| | 无 | Oat-Zero-7B AIME24 43.3% [论文] | 2503.20783 |
| DAPO | 按 token 归一的 clip(1−0.2, 1+0.28)；0<正确数<G；overlong 软惩罚；不加 KL | 四个机制的组合 | 超采样，不需要 ref | Qwen2.5-32B AIME24 从 30 到 50，逐项消融 [论文] | 2503.14476 |
| RLOO | A_i = r_i − mean_{j≠i} r_j；KL 放进 reward | 留一基线，不除 σ，原文不用 clip | ref，k≥2 | TL;DR 胜率 77.9（PPO 为 67.6）[论文] | 2402.14740 |
| REINFORCE++ | 全局 batch 白化；k1 KL 放进 reward；baseline 版先减组内均值，再加 k2 KL loss | σ 从组内改为 batch 级 | ref；k≥1 | 与 GRPO 大致持平，少数任务略好 [论文] | 2501.03262 |
| GSPO | s_i = exp(mean_t log ρ_t)；对序列做 clip，ε 为 3e-4/4e-4 | ratio 与 clip 改为序列级 | 无 | MoE 上不需要 routing replay；被裁掉的 token 比 GRPO 多约 100 倍但效率更高 [论文] | 2507.18071 |
| GMPO | 对 token 目标取几何平均，clip 为 (e^−0.4, e^0.4) | token 目标的聚合方式 | 无 | 7B 平均提升 4.1 [论文] | 2507.20673 |
| CISPO | sg(clip(ρ, ·, 1+ε_h))·Â·log π_θ，按 token 归一，不加 KL | 裁剪的是 IS 权重而不是目标，所有 token 都保留梯度 | 无 | 达到 DAPO 同等效果只用约一半步数（Qwen2.5-32B）；ScaleRL 采用 [论文] | 2506.13585，2510.13786 |
| Dual-clip | A<0 时取 max(…, c·A)，c=3 | 给负优势加下界 | 无 | 原文没有单独消融 [论文] | 1912.09729 |
| SAPO-Qwen | 用 f(ρ)=σ(τ(ρ−1))·4/τ 替代 clip，τ_pos=1.0、τ_neg=1.05，不加 KL | 硬 clip 换成软门控 | 无 | 只有曲线：比 GSPO/GRPO 更稳定 [论文] | 2511.20347 |
| MaxRL | A_i = (r_i − r̂)/r̂，全错的组没有梯度 | 除以 σ 改为除以 μ | 奖励必须为二值 | pass@k 帕累托优于 GRPO；GSM8K 约 30 epoch 后才反超 [论文] | 2602.02710 |
| MAPO | A = (1−λ)(r−μ)/σ + λ(r−μ)/μ，λ=1−4p(1−p) | 按确定性混合两种 advantage | 奖励为 0/1 | 7B 多模态提升小，证据弱 [论文] | 2509.18849 |
| GDPO | 每个奖励分量先组内归一，再加权求和，最后做 batch 白化 | 支持多奖励 | 需要奖励向量 | BFCL 平均 30.2→32.8，AIME（1.5B）23.1→29.4 [论文] | 2601.05242 |
| OTB | A_t = G_t − Σ_i G^(i)Ŵ_t^(i)/Σ_i Ŵ_t^(i)，其中 ŵ = 1 − 2π(y_t) + Σπ² | 逐 token 的最优基线 | 每个 token 的 Σ_v π(v)²，按组聚合 | Qwen3-8B AIME25 30.3（GRPO 25.3）；N=4 约等于 N=32 [论文] | 2602.07078 |
| GiGPO | A = A^E + ω·A^S；A^S 在相同环境状态的 step 组内计算 | 在轨迹级之外加 step 级信用分配 | 多轮轨迹、可哈希的状态、逐步奖励 | ALFWorld 7B 77.6→90.2 [论文] | 2505.10978 |
| ARPO | 工具调用后按熵变化决定是否分叉，形成树状 rollout | 改变 rollout 结构 | 生成时需要熵、前缀 fork、冷启动 SFT | 10 个基准平均 +1.8 到 +4.2 [论文] | 2507.19849 |
| CompactionRL | 多段轨迹；A = (γλ)^{N_{>s}}·A_loc；PPO 加 critic | 多段轨迹加 critic | critic、段元数据 | GLM-4.5-Air SWE-V 59.8→66.8 [论文] | 2607.05378 |
| SAO | 每个 prompt 1 条样本；ρ 对 π_rollout 计算，越界直接 mask（区间 0.7/6.0）；critic 加 GAE | 去掉组，改用 critic | critic、rollout logprob | Qwen3-30B-A3B AIME25 93.5→97.3（对照为 GRPO+DIS）；约 1000 步不崩 [论文] | 2607.07508 |
| VAPO | PPO 加 value 预训练、解耦 GAE、长度自适应 λ、clip-higher、正例 NLL | 需要 critic | critic | AIME24 60.4，5000 步 [论文] | 2504.05118 |
| TIS | pg·min(π_old^train/π_rollout, C) | 训推不一致修正 | rollout logprob | DAPO-32B 训练恢复稳定；C 的推荐值 [未核实] | fengyao.notion.site（博客） |
| IcePop | ratio 落在 [α,β]=[0.5,5] 内时乘以该 ratio，区间外置零 | 双侧 mask | rollout logprob | Ring-mini AIME25 比基线提升超过 14% [论文] | 2510.18855 |
| MIS / Seq-TIS | 序列级或几何平均权重做截断或 mask，并有 veto | 序列级修正 | rollout logprob | Miles 在 Qwen30B-A3B 上 token TIS 加 geo-MIS 稳定 [博客或教程，二手] | yingru.notion.site |
| OPSM | Â<0 且 seq_KL(π_old‖π_θ)>δ 时 mask，π_old 取推理端返回值 | 负优势的序列级 mask | rollout logprob（按原文） | δ 值未给出 [论文] | 2512.02556 §3.1 |
| Training-free GRPO | 用 LLM 总结经验库，放进上下文 | 不更新权重 | API | AIME25 67.9→73.3，花费 $18 [论文] | 2510.08191 |

系统和联邦类工作（完整摘要在调研记录中）：

- **ECHO**（2508.05387）：版本号加上 Δ_max 同步阈值；异步模式没有做实验。
- **INTELLECT-2**（2505.07291）：异步度为 2；A<0 时 ratio 上界 δ=4，本质上就是 dual-clip。
- **AReaL**（2505.24298）：解耦 PPO；陈旧度 η≤8 基本无损，普通 PPO 在 η=4 时 AIME 从 42.2 掉到 23.3。
- **PipelineRL**（2509.19128）。
- **ScaleRL**（2510.13786）：CISPO 和 GSPO 优于 DAPO；lm head 用 FP32 后渐近值从 0.52 升到 0.61。
- **PULSELoCo**（2602.03839）：第一个在 LLM RL 上验证 DiLoCo 的工作，内层步数 H 越大越离策略，所以用了很小的 H。
- **Asynchronous RLHF**（2410.18252）。
- **SAPO-Gensyn**（2509.08721）：只共享 rollout 文本，0.5B 模型上累计奖励提升 94%，证据弱。

这些工作对**当前串行 ports** 的直接适用性都很低，因为当前没有陈旧度。它们对以后另立的陈旧度 >0 算法契约 change 有参考价值（rl-infra-spec 本身不开放陈旧度，alignment.md A6）。

---

## 4. Miles 源码核查（`9e4260d`）

### 4.1 原生参数（`A` 行号）

| 机制 | 参数 | 定义 | 实现位置 |
|---|---|---|---|
| advantage estimator | `--advantage-estimator {grpo,gspo,ppo,reinforce_plus_plus,reinforce_plus_plus_baseline}` | A:1624 | `LH/advantages.py:53`（grpo/gspo），`:59`（ppo），`:79`（rpp），`:92`（rpp_baseline） |
| 组归一化 | 默认开；`--disable-rewards-normalization`、`--disable-grpo-std-normalization` | A:1681，A:1675 | `ray/rollout/train_data_conversion.py` 的 `_normalize_rewards_by_rollout`，约 240-266 行（只对 grpo/gspo 除以 σ；按 rollout_key 合并多段样本并要求同一 rollout 的奖励一致） |
| 全局白化 | `--normalize-advantages` | A:1673 | `LH/advantages.py` 的 `normalize_advantages`（在 DP 组内做 all-reduce） |
| PPO clip / clip-higher / dual-clip | `--eps-clip`（0.2）、`--eps-clip-high`、`--eps-clip-c` | A:1582/1583/1585 | `LH/math_utils.py:254-277` 的 `compute_policy_loss`，dual 分支在 267-273 行 |
| GSPO 序列 ratio | estimator 取 `gspo` | 同上 | `LH/losses.py` 中 `need_full_log_probs` 和 `compute_gspo_kl`（`math_utils.py:224-251`）。**默认 clip 仍是 0.2，论文的窄 clip 需要显式设置** |
| token 级聚合 | `--calculate-per-token-loss` | A:1556 | `cp_utils.py:92` 的 `get_sum_of_sample_mean` |
| 自定义聚合 | `--custom-pg-loss-reducer-function-path` | A:1765 | `losses.py` 的 reducer 分支，示例在 `examples/experimental/DrGRPO/` |
| KL 放进 reward | `--kl-coef` | A:1592 | **只有 ppo/rpp/rpp_baseline 使用**，grpo/gspo 忽略（见 §1.4） |
| KL 作为 loss | `--use-kl-loss`、`--kl-loss-coef`、`--kl-loss-type {k1,k2,k3,low_var_kl}`、`--use-unbiased-kl` | A:1650/1653/1617/1659 | `losses.py` 的 `use_kl_loss` 分支；`kl_coef` 与 `kl_loss_coef` 互斥（A:3436） |
| ref 模型 | `--ref-load` | A:1501 | 只在 `kl_coef≠0 or use_kl_loss` 时加载（`ray/specs/train.py:58`） |
| entropy | `--entropy-coef` | A:1670 | `losses.py` |
| TIS | `--use-tis`、`--tis-clip`、`--tis-clip-low`（默认 0） | A:1741/1747/1753 | `LH/corrections.py:7` 的 `vanilla_tis_function`；与 `--use-rollout-logprobs` 互斥（A:3445） |
| IcePop | `--custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function` | A:1759 | `corrections.py:35`。区间内**乘以 ratio**，区间外置零；区间用 `tis_clip_low`/`tis_clip`，论文取 0.5/5 |
| MIS | 同上，指向 `examples/infra_features/train_infer_mismatch_helper/mis.py:311` 的 `compute_mis_weights_with_cp` | — | 在 examples 目录下，**运行镜像能否 import 这个路径 [未核实]** |
| mismatch 指标 | `--get-mismatch-metrics` | A:1705 | 需要同时给 custom-tis 路径 |
| OPSM | `--use-opsm`、`--opsm-delta` | A:1798/1804 | `math_utils.py:183`。π_old 默认取**训练端重算**的值，只有加 `--use-rollout-logprobs` 才换成推理端的值；原文用的是推理端的值 |
| 用 rollout logprob 当 π_old | `--use-rollout-logprobs` | A:1720 | `losses.py` 的 old_log_probs 选择 |
| critic / PPO | estimator 为 `ppo` 时 `use_critic=True` | A:3510 | 独立的 critic TrainGroup、value loss、GAE（`math_utils.py:624`） |
| 超采样 / 动态过滤 | `--over-sampling-batch-size`、`--dynamic-sampling-filter-path` | A:789，A:811 附近 | yeto 已通过 bounded 过滤接入 |
| rollout logprob 采集 | SGLang 请求默认 `return_logprob=True` | — | `rollout/sglang_rollout.py:187`；`train_data_conversion.py:115-116` 会写入 `rollout_log_probs` |

### 4.2 扩展点：能做什么、不能做什么

| 扩展点 | 输入 | 执行时机和位置 | 限制 |
|---|---|---|---|
| `--custom-reward-post-process-path`（A:2393） | `fn(args, samples) -> (raw, normalized)`，可以看到 `Sample`（含 metadata） | rollout 侧转换训练数据时（`train_data_conversion.py` 的 `_post_process_rewards`） | **会整体替换内置归一化**，包括多段 rollout_key 合并；只能输出**每个样本一个标量**，之后由 grpo estimator 广播到各 token。可用于 MaxRL、MAPO、GDPO、RLOO、Dr.GRPO 这类组内标量变换；**不能**表达 token 级 advantage（OTB）或同一样本内随步变化的 advantage（GiGPO，除非每步拆成独立样本） |
| `--custom-loss-function-path` 加 `--loss-type custom_loss`（A:1598/1608） | `fn(args, batch, logits, sum_of_sample_mean)`；batch 的键是固定白名单（`train_data_conversion.py` 约 375-395 行：tokens、rewards、loss_masks、rollout_log_probs、advantages、log_probs、ref_log_probs 等），**不含 sample.metadata** | 训练前向之后，替换**整个** `policy_loss_function` | 替换后 TIS、OPSM、GSPO 的 CP 全收集、KL loss、entropy、mismatch 指标都要自己重写。micro-batch 可能把组拆开（`--balance-data` 加 DP 切分），**做不了组级统计** |
| `--custom-tis-function-path`（A:1759） | pg_loss、train_old、rollout logprob、masks 等关键字参数 | 在 PPO 项之后、聚合之前 | 适合所有训推修正（TIS、IcePop、MIS、只观测不修正） |
| `--custom-pg-loss-reducer-function-path`（A:1765） | lengths、masks | 聚合时 | 适合 Dr.GRPO 的常数分母 |
| `--rollout-data-postprocess-path`（A:992 附近） | `fn(args)` | log prob 算完之后、计算 advantage 之前（`megatron_utils/actor.py:757-761`） | 只能拿到 args，接口很窄 |
| `--rollout-sample-filter-path`、`--buffer-filter-path`、`--rollout-all-samples-process-path`、`--dynamic-sampling-filter-path`、`--rollout-function-path` | — | — | **已被 yeto adapter 占用**（`config.py:78-96`、`530-532`）。新逻辑只能组合进 yeto 自己的 hook，不能直接指向别的函数 |
| tinker loss（`LH/tinker_losses.py:118` 的 cispo） | 从 batch 读 `loss_fn_config` | 只在 tinker 路径通过 `get_loss_function(loss_fn=...)` 调用 | 规约方式是按样本求和，不是 sample-mean，**不能直接当 `--loss-type` 用**，只能作参考实现 |

---

## 5. yeto ports 现状与缺口

本节“现状”一列是**历史基线**（yeto `18695ae`，rl-algorithm-capabilities 之前）；“缺口”一列的处理结果以 §8 的已确认方案为准，其中 extra argv 与 provenance 两行已由该 change 实现。

| 项 | 历史基线 [源码] | 缺口（处理见 §8） |
|---|---|---|
| `AlgorithmSpec` | 5 个字段、schema v1、规范化 JSON 的 SHA256（`algorithm.py`） | 缺 clip、聚合、KL 放置位置、修正方式、reward 后处理、loss 变体 |
| 翻译 | 只输出 `--advantage-estimator`、`--kl-coef`、`--dynamic-sampling-filter-path`（`config.py:535, 550-553`） | 新字段需要映射，并把对应参数加入 `ADAPTER_OWNED_FLAGS` |
| 能力 | `advantage_estimators`、`dynamic_sampling_filters`（`capabilities.py`） | 缺 loss、修正、reward 后处理、KL 模式等维度；`check()` 只比较 estimator 和 filter |
| extra argv | （历史基线）算法参数可以透传（§1.3） | 已映射的吸收进 spec，冲突报错，未映射的影响目标参数拒绝（§8 第 3 条） |
| 不变量 | 按 GRPO 判定（§1.6） | 需要由算法声明"期望有梯度"的判定条件 |
| provenance | （历史基线）事件里有哈希，导出文件里没有；外层身份不含算法（§1.7） | 导出时写入；岛之间做一致性校验 |
| legacy | 不改（约束） | 新算法只走 ports |
| runtime attrs | 已有机制把 yeto 参数挂到 Miles namespace 上（`to_legacy_runtime_attrs`，`config.py:691`） | 插件的配置可以复用这个机制下发，不必新开参数 |

---

## 6. 工程能力矩阵

"外层兼容"一列分三级：
- **接口**：ports 接口能接入；
- **假设**：算法本身的假设在该同步模式下成立；
- **实验**：已有实验依据。

目前**所有条目的"实验"一级都是"无"**，yeto 上没有跑过任何非 GRPO 算法。

| 机制 | Miles 原生 | 扩展点实现 | yeto 改动 | fork 改动 | strict-avg | decoupled |
|---|---|---|---|---|---|---|
| GRPO（默认） | ✅ | — | 无 | 无 | 接口✅ 假设✅ 实验✅（R0 GPU 验收） | 接口✅ 假设✅ 实验：部分（受 LR 衰减到 0 问题影响） |
| clip-higher | ✅ `--eps-clip-high` | — | spec 字段、owned、翻译 | 无 | 接口✅ 假设✅ | 同左 |
| dual-clip | ✅ `--eps-clip-c` | — | 同上 | 无 | ✅✅ | ✅✅ |
| token 级聚合 | ✅ `--calculate-per-token-loss` | — | 同上 | 无 | ✅✅ | ✅✅ |
| Dr.GRPO | ✅ 去 σ；去长度归一靠 reducer | reducer 插件（示例可复用） | 字段加 yeto 插件 | 无 | ✅✅ | ✅✅ |
| RLOO | 约等于 Dr.GRPO（§2） | 可选 reward post-process | 不单独实现 | 无 | ✅✅ | ✅✅ |
| KL loss（k1/k2/k3） | ✅ `--use-kl-loss` | — | **改正 KL 的建模方式**（§1.4） | 无 | 接口✅；假设：ref 是 base，各岛相同，但**岛之间一致未被校验** | 同左 |
| entropy 奖励 | ✅ | — | 字段 | 无 | ✅✅ | ✅✅ |
| DAPO 动态采样 | ✅（yeto 已有 bounded 过滤） | — | 把 `--over-sampling-batch-size` 纳入 spec | 无 | 接口✅；假设：各岛有效 batch 不同，而外层是等权平均 [推断] | 同左 |
| overlong 塑形或过滤 | ❌ | 塑形：组合进 yeto 的 reward 链；过滤：组合进 yeto 已占用的 sample-filter hook，设置 `remove_sample` | yeto 插件 | 无 | ✅✅ | ✅✅ |
| GSPO | ✅ | — | 字段；必须显式设窄 clip | 无 | 接口✅；假设✅ | 接口✅；假设✅（没有陈旧度）；LoRA 加窄 clip 的效果**没有证据** |
| REINFORCE++ / baseline | ✅ | — | 字段；global 统计语义写明为"岛内（DP 组内）" | 无 | 接口✅；假设：论文的"全局 batch"在 yeto 里只是岛内 batch [推断] | 同左 |
| MaxRL、MAPO | ❌ | reward post-process（组内标量变换），必须保留 rollout_key 多段语义 | yeto 插件 | 无 | ✅✅（纯组内计算） | ✅✅ |
| GDPO | ❌ | reward post-process，读取 `sample` 上的奖励向量；reward 函数需要把各分量写进 metadata | 插件；reward 接口约定 | 无 | ✅✅ | ✅✅ |
| TIS | ✅ | — | 字段、owned；默认不开 | 无 | 接口✅；假设：修正的是训推不一致，与外层无关 | 同左 |
| IcePop | ✅（内置函数加 custom 路径） | — | 同上 | 无 | 同上 | 同上 |
| MIS / geo-MIS | examples 目录 | custom-tis 路径 | 可 vendor 进 yeto 插件；镜像能否 import 需确认 | 无 | 同上 | 同上 |
| OPSM | ✅ | — | 字段；决定 π_old 取哪一个（见 §4.1） | 无 | 同上 | 同上 |
| 只观测不修正的 mismatch 指标 | ✅ `--get-mismatch-metrics` | custom-tis 用恒等函数 | yeto 插件（权重恒为 1，只出指标） | 无 | ✅ | ✅ |
| CISPO | ❌（tinker 路径的规约方式不同） | custom loss 能做，但要重写 TIS、OPSM、CP、KL、entropy | 插件，工作量中到大 | **可选**：在 `compute_policy_loss` 里加一个 variant 分支，改动小，但要改 fork | 接口✅；假设✅ | 同左 |
| SAPO-Qwen 软门控 | ❌ | 同 CISPO | 同 CISPO | 同 CISPO | 同 CISPO | 同 CISPO |
| GMPO | ❌ | 同 CISPO，且需要序列级 token 乘积 | 同 CISPO | 同 CISPO | 同 CISPO | 同 CISPO |
| OTB | ❌ | **不可行**：需要每个 token 的 Σπ² 和按组聚合；前向只返回 log_prob 和 entropy，组又可能被 DP 切开 | — | 需要在前向输出 Σπ²，并在 rollout 侧或 DP 全局做组级 advantage | — | — |
| GiGPO | ❌ | 按步拆成独立样本后，reward post-process 可以表达；依赖 agent rollout 和状态哈希 | 需要 agent 环境 | 可能不需要 | 接口：取决于 agent rollout 能否在 ports 上跑 | 同左 |
| PPO / VAPO / SAO / CompactionRL | ✅（critic 由 Miles 提供） | — | **critic 状态、外层归属、多 cell** | 可能要处理 critic 的导出和应用 | **ports 拒绝** | **ports 拒绝** |
| ARPO | ❌ | — | — | SGLang 需要支持树状 fork 和生成时熵 | 延后 | 延后 |
| SAPO-Gensyn | — | — | 新的外层协议（共享 rollout） | — | 另立 change | 另立 change |
| Training-free GRPO | 不适用 | — | — | — | 不适用 | 不适用 |

---

## 7. 外层同步兼容性

### 7.1 三种 logprob 和策略版本（ports 串行 driver）

```
 round r:  publish(pi_r) --> generate (SGLang, weights W_r) --> train_step (Megatron starts at W_r, k minibatches)
                                   |                                  |
                           rollout_log_probs              log_probs = pi_old (trainer recompute at W_r)
                           = pi_behav (numerics_S)         current = pi_theta (drifts over k minibatches)
                                                           ref_log_probs = base (--ref-load), fixed
           --> sync.boundary: strict: push delta, wait v+1, apply avg (optimizer RESET)
                               decoupled: drain broadcasts, apply fragments (optimizer PRESERVE), submit ready
           --> publish(pi_{r+1} = post-apply weights) --> next round
```

- **行为策略和 π_old 是同一组权重**，差别只在数值实现：SGLang 与 Megatron 的 kernel 不同、bf16，以及 LoRA 发布路径 [源码]。
- **因此修正 π_old 与 π_rollout 之间差异的方法（TIS、IcePop、MIS），在两种同步模式下的语义完全相同**，修的都是训推不一致，与外层无关。
- **π_θ 与 π_old 的偏移**来自同一轮内的 k 个 mini-batch（`--num-steps-per-rollout = batch.optimizer_steps`，`config.py:524`），由 PPO 的 clip 处理，也与外层无关。
- **ref 策略**是 `--ref-load` 指向的 base 模型，各岛相同的前提是配置相同。但外层身份不校验它 [源码 §1.7]，建议纳入算法和 provenance 的一致性校验。

### 7.2 按同步模式看

| 维度 | strict-avg | decoupled |
|---|---|---|
| 优化器状态 | 每轮 apply 时 **reset**（`bridges.py:185`），Adam 的一阶和二阶矩每轮重新开始 [源码] | 应用 fragment 时 **preserve**（`bridges.py:421, 498`），本地 Adam 矩继续作用在被外层改写过的权重上 [源码] |
| 对算法的影响 | 依赖长时间优化器记忆的算法不适用，所列算法都不依赖。clip 系列的有效步长受每轮 warmup 或 bias correction 影响 [推断] | Adam 矩与权重错位，对所有算法的影响相同，不是某个算法特有的问题 [推断] |
| 学习率 | 调度按 `policy_version × local_optimizer_steps` 计算，正常 | **已知问题：LR 衰减到 0**。修复在 `fix-decoupled-lr-schedule`，本分支不重复修。**在修复合入前，任何 decoupled 下的算法 A/B 实验都没有意义**，因为后半程没有更新 |
| rollout 陈旧度 | 0，串行 | 0，串行。`execution.max_policy_staleness` 固定为 0；rl-infra-spec 的重叠只在陈旧度 0 的契约内进行（alignment.md A1/A6） |
| 组统计 | 组在单岛内生成（`--n-samples-per-prompt`），组内统计由本岛完成 | 同左 |
| "全局"统计（REINFORCE++、GDPO 的 batch 白化、`--normalize-advantages`） | 只在岛内 DP 组；跨岛 all-reduce 会破坏"一个协议多个后端"，**建议 spec 明确规定为岛内语义** | 同左 |
| 各岛有效样本数（动态采样、mask 类修正） | 外层对 delta 做等权平均，不按 token 数加权；样本少的岛权重偏大 [推断，需确认 syncer 的加权规则] | 同左 |
| critic | 状态不在 `TrainableState` 里，外层不认识；ports 拒绝 | 同左；另外 critic 是岛内局部还是参与平均需要单独设计 |
| 异构算法 | 两个岛用不同 AlgorithmSpec 也能被平均（§1.7），**必须禁止** | 同左 |

### 7.3 对异步和离策略修正的价值判断

- **现在（串行 ports）**：TIS、IcePop 的价值取决于训推不一致有多大，**yeto 还没有这个数据**。应该先用只观测的方式量化（mismatch 指标：`train_rollout_kl`、`tis_abs`、`ess_ratio`），再决定是否默认开启修正。LoRA 路径与全参 MoE 的不一致程度可能差别很大 [推断]。
- **以后（另立的陈旧度 >0 算法契约 change，不由 rl-infra-spec 开放）**：Miles TIS 等价于截断版解耦 PPO（§2），AReaL 证明在陈旧度 η≤8 时基本无损 [论文]。所以现在把修正机制接进 spec，就是在为以后提前准备算法契约。

---

## 8. 建议的 change 拆分

```
 rl-algorithm-capabilities (P0, framework, CPU)
   |-- rl-algo-mismatch-correction (P1)   observe -> TIS / IcePop / (MIS) / OPSM
   |-- rl-algo-grpo-knobs (P1)            clip-higher, dual-clip, token-agg, Dr.GRPO, KL-loss, entropy, overlong, over-sampling
   |-- rl-algo-seq-and-adv (P2)           GSPO, REINFORCE++(-baseline), MaxRL/MAPO/GDPO via yeto reward pipeline
   |-- rl-algo-loss-variants (P2, needs fork decision)  CISPO / SAPO-Qwen / GMPO
   '-- deferred: critic family (PPO/VAPO/SAO/CompactionRL), OTB, GiGPO, ARPO, SAPO-Gensyn, async objective (own algorithm-contract change; not opened by rl-infra-spec)
```

框架 change（`rl-algorithm-capabilities`）的**已确认方案**（实现见 `yeto/rl/engine/algorithm.py`、`miles_adapter/algorithm_flags.py`，用法见 `docs/MILES_RL.md` 的 "Algorithm specs"）：

1. **AlgorithmSpec v2 的结构**：按 advantage、loss、kl、correction、sampling、execution 分组的冻结数据类，外加 `entropy_coef` 与 `plugins`。每个插件记录 dotted path 和模块源码 SHA256，只允许 `yeto.`/`miles.` 命名空间。默认值精确还原当前 GRPO 的 argv（逐字节）。
2. **哈希兼容（§10 问题 1 已定）**：v1 可表达时输出与 R0 逐字节相同的 v1 规范化 JSON，哈希不变；只有用到新字段才写 `yeto-rl-algorithm-spec-v2`。
3. **参数归属：吸收取代一律拒绝**。映射表内的 Miles 算法参数出现在 extra argv 时被**吸收**进 spec（进入校验、能力检查和哈希，事件记录 `rl/algorithm_absorbed_flags`）；与 spec 取值不同时报冲突；影响训练目标但未映射的参数（清单由 upstream parser 测试锁定）拒绝。所有映射参数归 adapter 所有。
4. **能力声明**：`EngineCapabilities` 按机制维度扩展（新增顶层字段，#66 的读取方忽略），另有 `execution`（critic、`max_policy_staleness`、rollout logprob）。**execution 与陈旧度**：算法的 `execution.max_policy_staleness` 固定为 0，>0 启动前拒绝；执行能力中的陈旧度由已认证的执行模式给出（serial-colocated、partitioned-serial 都是 0），`ExecutionProfile.max_policy_age ≤ spec.execution.max_policy_staleness`，算法契约身份即 spec 的规范化哈希（alignment.md A1/A6）。
5. **KL 放置（§10 问题 2 已定为方案 a）**：`kl.placement ∈ {none, reward, loss}`。grpo/gspo 配 `reward` 且系数 >0 启动前拒绝并提示改用 `placement=loss`（`--use-kl-loss --kl-loss-coef --kl-loss-type`）；v1 `kl_coef` 解释为 `reward`（`0.0` 保持 R0 行为与哈希）；`none` 不输出 KL 参数、不加载 ref 模型；reward 与 loss 两种 KL 在结构上互斥。
6. **拒绝矩阵**（启动前，报错给出替代配置）：critic 类（提示 legacy）；grpo/gspo 的 reward KL；TIS 与 `use_rollout_logprobs` 同时开；reward KL 与 loss KL 同时给出；GSPO 未显式给 clip；要求二值奖励的机制配非二值奖励（`advantage.reward_binary` 声明）；REINFORCE++ 系未开 `whiten`（upstream 断言）；陈旧度 >0。
7. **不变量**：`AlgorithmSpec.expects_gradient()`，默认 GRPO 与 R0 相同；声明会屏蔽 token 的机制在 `masked_fraction == 1.0` 时不期望梯度，读不到屏蔽比例时沿用 R0。
8. **provenance 与岛间一致**：launcher 下发预期哈希，每个岛在连接 bridge 之前核对，不一致写 `rl_algorithm_mismatch` 并退出；ports 导出记录 `algorithm_spec` 与 `algorithm_spec_sha256`。syncer 协议层校验留作后续加固。未验证机制放行（`--rl-allow-unverified-mechanism`）只限单岛，写入事件与来源记录，不进哈希。
9. **插件的运行位置**：yeto 插件跑在 Miles 进程里，要求镜像里能 import yeto（R0 的 run_plugin 机制已经这么做）。reward 后处理由 yeto 统一持有一个 `--custom-reward-post-process-path` 分派器，组合 overlong 塑形和 advantage 变换，并保持内置的多段 rollout_key 语义。

---

## 9. 验证方案

| 层级 | 证明什么 | 方法 | 需要 GPU |
|---|---|---|---|
| C1 规范化和哈希 | 默认 GRPO 的哈希和 argv 不变；新字段规范化稳定 | golden 哈希；`tests/test_rl_argv_snapshot.py` 逐字节比对 | 否 |
| C2 翻译和归属 | 每个字段都映射到明确的 Miles 参数；extra argv 里的已映射算法参数被吸收、冲突报错、未映射参数被拒绝 | 单测；在 `/home/michael/work/miles-next-venv` 里用 upstream `parse_args` 解析生成的 argv（R0 用过这个做法） | 否 |
| C3 拒绝矩阵 | 不支持的组合在启动前被拒绝，并列出可选项 | 参数化单测，走 fake engine | 否 |
| C4 插件数值 | reward 后处理和修正函数在 CPU 张量上的数值与论文公式一致；保持多段语义 | 纯 torch 的 CPU 单测，直接调用 Miles 的 `corrections.py` 和 `math_utils.py`（这两个不依赖 GPU，在 miles venv 里运行） | 否 |
| C5 provenance 和成员一致性 | 导出文件含算法哈希；算法不同的岛被拒绝 | fake bridge 测试 | 否 |
| G1 冒烟 | 每个新增机制在真实 Miles 和 SGLang 上能跑通；指标存在；不变量没有误报 | 1 卡、小模型（与 R0 冒烟相同），每个机制跑 2 到 3 轮 | 是，1 卡 |
| G2 训推不一致量化 | ports 的 LoRA 路径上，`train_rollout_kl`、`tis_abs`、`ess_ratio` 有多大，决定是否需要默认修正 | 只观测模式，1 个岛，约 20 轮 | 是，1 卡 |
| G3 外层兼容 | 新机制下两岛 strict-avg 的 hash 一致、不变量正常；decoupled 要等 LR 修复合入后再跑 | 2 个岛，每岛 1 卡 | 是，2 卡 |
| G4 效果 | 某机制相对 GRPO 有收益（奖励或验证集曲线） | 固定种子、多次重复的 A/B；预算单独申请。**没有 G4 不能声称有收益** | 是 |

mock 或 fake 测试只能证明 C1 到 C5，不能替代 G1 到 G4。

---

## 10. 未解决的问题及其影响

1. **v1 哈希是否保持**：**已定**（rl-algorithm-capabilities D2）：v1 可表达时序列化为 v1 的规范化 JSON，哈希不变；只有使用了新字段才写 v2。
2. **`kl_coef` 加 GRPO 怎么处理**：
   - (a) 拒绝，并提示改用 KL loss；
   - (b) 保持现状，只输出警告；
   - (c) 自动映射到 KL loss。

   (c) 会改变已有配置的行为，不建议。**已定为 (a)**（rl-algorithm-capabilities D5）：ports 上拒绝并提示改用 `kl.placement=loss`；legacy 不变。
3. **CISPO、SAPO-Qwen、GMPO 走 custom loss 还是改 fork**：
   - custom loss 要在 yeto 里重写 TIS、OPSM、CP、KL、entropy，维护成本高；
   - 改 fork 是在 `compute_policy_loss` 里加一个 variant 分支，大约几十行，但需要你同意，而且只能在 `yeto/ports` 分支上改。
4. **syncer 的外层平均是否按样本或 token 加权**：[未核实]。影响动态采样和 mask 类机制的跨岛公平性。
5. **legacy SAO 是否就是 2607.07508**：[未核实]。影响 critic 路线的迁移成本估计。
6. **MIS 示例在镜像里能否 import**：[未核实]。影响 MIS 能否零代码接入；不能的话要 vendor 进 yeto。
7. **训推不一致的实际大小**：未测。直接决定 P1 修正项的价值，建议 G2 作为 P1 的第一个 GPU 实验。
8. **TIS 的推荐 C 值和 MIS 阈值**：原博客正文没拿到 [未核实]。实现时参数必须显式给出，不写死默认值。
9. **PPO、GRPO、RLOO 三篇**第一轮子 agent 是凭记忆写的，第二轮已联网核对原文（本文数字取自核对后的结果）。

---

## 11. 参考链接

PPO 1707.06347 · InstructGPT 2203.02155 · GRPO 2402.03300 · Dr.GRPO 2503.20783 · DAPO 2503.14476 · RLOO 2402.14740 · REINFORCE++ 2501.03262 · GSPO 2507.18071 · GMPO 2507.20673 · CISPO 2506.13585 · ScaleRL 2510.13786 · Dual-clip 1912.09729 · SAPO-Qwen 2511.20347 · SAPO-Gensyn 2509.08721 · MaxRL 2602.02710 · MAPO 2509.18849 · GDPO 2601.05242 · OTB 2602.07078 · GiGPO 2505.10978 · ARPO 2507.19849 · CompactionRL 2607.05378 · SAO 2607.07508 · VAPO 2504.05118 · Training-free GRPO 2510.08191 · ECHO 2508.05387 · IcePop 2510.18855 · OPSM（DeepSeek-V3.2）2512.02556 · R3 2510.11370 · INTELLECT-2 2505.07291 · AReaL 2505.24298 · PipelineRL 2509.19128 · PULSELoCo 2602.03839 · Async RLHF 2410.18252 · DiLoCo 2311.08105 · TIS 博客 fengyao.notion.site · MIS 博客 yingru.notion.site

（arXiv 编号均对应 https://arxiv.org/abs/<id>）
