# Design

## Context

动机见 proposal.md 的 Why。本 design 的现状来自源码，Miles 基线为 `9e4260d`（`A` = `miles/utils/arguments.py`，`LH` = `miles/backends/training_utils/loss_hub/`），并以 P0 `rl-algorithm-capabilities` 已完成为前提：

- P0 已提供 AlgorithmSpec v2 分组字段、`algorithm_flags.py` 映射表、`EngineCapabilities` 机制维度、`expects_gradient()`、PluginRef 源码哈希、KL placement 规则。本 change 的机制在 P0 之后都是"可表达未开放"。
- clip 与 dual-clip 在 `LH/math_utils.py:254-277` 的 `compute_policy_loss`，dual 分支在 267-273 行；token 级聚合在 `cp_utils.py:92`。
- 自定义 pg_loss reducer 签名为 `(total_lengths, response_lengths, loss_masks, calculate_per_token_loss) -> Callable`。Miles 示例 `examples/experimental/DrGRPO/custom_reducer.py`（blob `96390ac3`）的分母写死为 `DIVISOR = 1000.0`，import 了 `megatron.core.mpu`，并断言 CP=1。
- 内置 reward 后处理 `_post_process_rewards`（`ray/rollout/train_data_conversion.py:274-288`）：有 custom 函数时直接 `return f(args, samples)`；否则对 grpo/gspo/reinforce_plus_plus_baseline 且 `rewards_normalization` 为真时，调用 `_normalize_rewards_by_rollout`（224-271 行）。后者用 `_reward_group_segments`（189-221 行）分组：优先 `prompt_group_sizes`（多 LoRA），其次 `sample.group_index`，再次按固定 fan-out 连续切分，最后整批一组；组内按 `rollout_id`→`index`→行号得到 rollout_key 合并多段；同一 rollout 奖励必须一致；减均值，grpo/gspo 且开启 std 归一且组大小大于 1 且 std>0 时除以 `std + 1e-6`。**custom 函数拿不到 `prompt_group_sizes`。**
- `sample.remove_sample=True` 会在转换训练数据时把该样本 loss mask 置 0（`train_data_conversion.py:99-100`），此时奖励和 advantage 已经算完。
- `--rollout-sample-filter-path` 已被 yeto 占用为 `TRAINED_GROUPS_HOOK_PATH`（`miles_adapter/config.py:41, 530`），调用点在 `sglang_rollout.py:536` 等。
- ref 模型只在 `kl_coef≠0 or use_kl_loss` 时加载（`ray/specs/train.py:58`），来自 `--ref-load`。
- runtime attrs 机制 `to_legacy_runtime_attrs`（`config.py:691`）可以把 yeto 参数挂到 Miles args 上。
- 约束：legacy `build_miles_argv` 不变；默认 GRPO 行为、哈希、argv 不变；不改 fork；不向上游提 PR。

## Goals / Non-Goals

**Goals:**
- 把 proposal 表中的机制逐项变成"声明支持"，每项都有 CPU 证据和 G1 证据。
- 建立 yeto 自有的 reward 后处理分派器，与 Miles 内置逻辑逐元素等价，并为 P2 留出扩展点。

**Non-Goals:**
- 效果 A/B（G4）。任何"收益"结论都不在本 change 中给出。
- decoupled 模式下的对比实验（依赖 `fix-decoupled-lr-schedule`）。
- 改变 syncer 的外层加权规则（research §10 问题 4 仍未核实，见 D8）。
- RLOO 的独立实现。

## Decisions

### D1. 原生机制只做"确认翻译 + 约束 + 声明"

clip-higher、dual-clip、token 级聚合、去 std、entropy、超采样的映射行 P0 已写好。本 change 对每项只做三件事：
1. 在 spec 校验中补机制特有约束：dual-clip `c > 1`（Miles 断言也要求，yeto 提前报错）；clip 上界有限正数；超采样须配动态过滤且不小于 rollout 批大小；entropy 有限。
2. 在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv，并核对解析后 namespace 上的字段值。
3. G1 通过后在 `entry.py` 的能力声明中加入该机制，fake engine 同步。

不采用"一次性全部声明"，因为 R0 的教训是只有 GPU 冒烟能发现指标键缺失、不变量误报这类问题。

### D2. Dr.GRPO：去 std 走原生参数，常数分母 vendor reducer

- 去 std：`advantage.std_normalization=false` → `--disable-grpo-std-normalization`。
- 常数分母：先在运行镜像里核实 `examples.experimental.DrGRPO.custom_reducer` 能否 import（`examples/__init__.py` 存在，但 Miles 是否以包含 `examples` 的方式安装未核实）。**无论能否 import，都计划 vendor 进 `yeto/rl/algos/reducers.py`**，原因是分母写死为 1000，而论文的分母是 MAX_TOKENS（生成长度上限），必须能配置并进入哈希。vendor 时文件头记录来源（radixark/miles `9e4260d`，`examples/experimental/DrGRPO/custom_reducer.py`，原始 blob `96390ac3` 与源码 SHA256），测试锁定该来源哈希。分母经 runtime attrs 下发（D5），不是模块常量。
- `loss.aggregation=constant` 且 CP>1 时启动前拒绝，与原示例的断言一致，但更早报错。
- 常数分母 reducer 与 `calculate_per_token_loss=True` 组合时，示例退化为逐 token 求和。yeto 拒绝这种组合，二者只能选一，避免语义含糊。
- 只作用于 pg_loss；clipfrac、KL、entropy 等指标仍用默认聚合（Miles reducer 的既有语义）。

**RLOO 不单独实现**：留一基线 r_i − mean_{j≠i} r_j = r_i − (Gμ − r_i)/(G−1) = G/(G−1)·(r_i − μ)。关闭 std 后，它与 Dr.GRPO 的 advantage 只差常数 G/(G−1)。Adam 对梯度的整体常数缩放近似不变（ε 项除外），因此在 yeto 里 RLOO ≈ Dr.GRPO 的 advantage。差别在于 RLOO 原文把 KL 放进 reward，而 grpo estimator 会丢弃 reward KL（P0 已拒绝），需要 KL 时用 `placement=loss`。若以后需要精确 RLOO，可在分派器注册一个变换（D4），不影响本 change 的 spec。

### D3. KL loss 与参考模型身份

`kl.placement=loss` 的翻译沿用 P0：`--use-kl-loss --kl-loss-coef c --kl-loss-type t`，t ∈ {k1, k2, k3, low_var_kl}；`--use-unbiased-kl` 保持 P0 映射，本 change 不单独开放。

岛间 ref 一致性：P0 的岛间校验只比算法哈希，而 ref 来自 `--ref-load`，由运行配置决定，不在描述里。两个岛可能算法哈希相同，ref 却不同。评估了三种做法：

| 方案 | 优点 | 缺点 |
|---|---|---|
| A. 不处理，只写文档 | 零改动 | 错配时静默得到不同的目标 |
| B. KL loss 描述增加 `ref_model: {source, revision}`，进入哈希；启动时与本岛实际 `--ref-load` 对应的 base 身份比对 | 复用 P0 的岛间哈希校验，无需新协议 | 描述里多一个与模型相关的字段 |
| C. 把 ref 身份放进外层 `CanonicalLoraState` 身份 | 与 base revision 校验同处 | 要改外层身份，legacy 共用，影响面大 |

**采用 B**。`ref_model` 只在 `placement=loss`（以及以后的 reward KL）时必填，默认 GRPO 不出现该字段，哈希不变。取值与外层身份用的 `base_model_revision` 同源；ports 当前 ref 就是 base，因此 learner 在构建 spec 后核对 `ref_model.revision == base_model_revision`，不一致则启动前拒绝。以后若支持"ref 不等于 base"，再放宽这条规则。

### D4. yeto reward 后处理分派器

入口 `yeto.rl.algos.reward_pipeline.post_process(args, samples) -> (raw, normalized)`，是 ports 上唯一的 `--custom-reward-post-process-path`。只有描述中存在塑形或非默认变换时才输出该参数；默认 GRPO 不输出，继续用 Miles 内置路径，argv 不变。

结构：

```
post_process(args, samples)
  cfg   = read_pipeline_config(args)            # runtime attrs, from AlgorithmSpec (D5)
  raw   = [s.get_reward_value(args) for s]      # 同 Miles
  shaped= raw
  for stage in cfg.reward_shapers:              # 本 change: overlong_penalty
      shaped = stage(args, samples, shaped)
  groups= reward_groups(args, samples)          # 复刻 _reward_group_segments（无 prompt_group_sizes 分支）
  adv   = ADV_TRANSFORMS[cfg.advantage](args, samples, shaped, groups)
                                                # 本 change: "grpo_default"；P2 注册 maxrl/mapo/gdpo
  return raw_or_shaped, adv
```

要点：
- **等价重现**：`grpo_default` 逐行复刻 `_normalize_rewards_by_rollout` 的 rollout_key 合并、一致性检查、组大小 1、std=0、`+1e-6`、仅 grpo/gspo 除 std、`rewards_normalization` 关闭时直接返回原值、非 grpo/gspo/rpp_baseline 估计器返回原值。CPU 测试在 miles-next-venv 中直接 import Miles 的 `_post_process_rewards` 做逐元素比对（`torch.equal`，不是近似）。
- **返回的 raw**：Miles 用第一个返回值记录 raw reward 指标。塑形后返回 `shaped` 作为 raw，还是返回未塑形的原始值，影响指标含义。决定：返回塑形后的值作为训练用 raw，原始奖励另外写入 `sample.metadata["yeto_raw_reward"]` 并由 yeto 事件汇总，这样 reward 曲线不被惩罚项混淆。
- **多 LoRA**：custom 函数拿不到 `prompt_group_sizes`。ports 目前只有单 LoRA；若检测到多 LoRA 配置，启动前拒绝使用分派器。
- **分组回退**：若样本没有 `group_index`，且不满足固定 fan-out，Miles 会把整批当一组。yeto 保留同样行为以保证等价，但在该情况下写一条警告事件，因为 ports 的 bounded 过滤总会带 group_index，出现回退说明数据有问题。
- **扩展点**：`ADV_TRANSFORMS` 与 `REWARD_SHAPERS` 是模块内注册表，键名进入描述；每个已注册阶段对应一个 PluginRef（源码哈希取分派器模块文件，P0 D7 的规则）。P2 只需注册新变换并加映射，不需要改入口。

备选：让每个机制各自占用 `--custom-reward-post-process-path`。二者不能共存，也无法复用内置等价逻辑，因此不采用。

### D5. 插件配置经 runtime attrs 下发

分派器和 reducer 的配置（塑形阶段列表及参数、advantage 变换名、reducer 分母、overlong 过滤开关）由 `AlgorithmSpec` 派生，经 `to_legacy_runtime_attrs`（`config.py:691`）挂到 Miles args 上，统一放在一个 `yeto_algo_plugins` 命名空间属性中（JSON 可序列化字典），并附带该字典的规范化哈希。插件在 Miles 进程内读取后，核对哈希与描述一致，不一致即抛错。这样不新增 Miles CLI 参数，也不需要改 fork。

### D6. overlong 软惩罚

公式（DAPO）：设 L 为响应长度，

- L ≤ L_max − L_cache：0；
- L_max − L_cache < L ≤ L_max：(L_max − L_cache − L)/L_cache；
- L > L_max：−1。

作为 `REWARD_SHAPERS["overlong_penalty"]`，在组归一化之前加到原始奖励上。L 取 `sample.response_length`（多段样本按 rollout 合并后，取同一 rollout 各段的总响应长度，保证同一 rollout 的塑形后奖励仍然一致，不触发一致性检查报错）。约束：`0 < L_cache ≤ L_max ≤ rollout 最大生成长度`，否则启动前拒绝。惩罚系数固定为 1（原文），不另设系数字段。

### D7. overlong 过滤

`sampling.overlong_filter=true` 时，在 yeto 已占用的 `record_trained_groups` hook 中，对 `status == truncated` 的样本设置 `sample.remove_sample=True`，再执行原有记录逻辑。

语义与 DAPO 的差异（写入文档）：
- yeto：截断样本 loss mask 置 0，但它的奖励仍参与组均值和 std 计算，影响同组其他样本的 advantage；
- DAPO 原文的 overlong filtering 是在 loss 中屏蔽截断样本，原文未明确其奖励是否参与组统计 [未核实]；实现上常见的是同样保留参与统计。
- 另外在默认 sample-mean 聚合下，被屏蔽样本是否仍计入"样本数"分母，会影响其他样本的有效权重 [推断，需在 CPU 测试中用 Miles `get_sum_of_sample_mean` 确认并写入文档]。

零梯度不变量：该机制通过 P0 的 `expects_gradient()` 声明"若本轮所有非零方差组的样本全部被过滤，则不期望梯度"；判定所需的被过滤样本数由 hook 写入 rollout 元数据，driver 从 batch summary 读取。

overlong 过滤与软惩罚可以同时启用，二者不冲突。

与 `rl-infra-spec` 3.6 账本的对齐（alignment.md A2）：`record_trained_groups` 是两边共用的唯一 hook。本 change 先设置 `remove_sample` 并写被过滤样本数，infra 账本在同一 hook 之后把这些样本记为终态 `filtered`（超采样不产生可复用余量：Miles `sglang_rollout.py:505-510` 不把多出的组放回 buffer，只有 `--partial-rollout` 下被 abort 的样本回 buffer；审查 F5）；两边都不得另开 `--rollout-sample-filter-path`。

### D8. 超采样与外层等权平均

`sampling.over_sampling_batch_size` → `--over-sampling-batch-size`，要求同时启用 bounded 动态过滤。各岛每轮最终训练样本数可能不同（过滤掉的组数不同、替换次数上限不同）。strict-avg 与 decoupled 的外层对 delta 做等权平均，不按样本数加权 [推断，research §10 问题 4 未核实]。影响：样本少的岛在外层权重偏大。本 change 的处理：
- 不改外层加权；
- 每轮事件记录本岛进入训练的样本数与组数，供以后判断是否需要加权；
- 在 `docs/MILES_RL.md` 写明这一行为。

### D9. 命名示例

在 `examples/rl_algorithms/` 下提供 `dapo-like.json`（clip 0.2/0.28、token 级聚合、bounded 过滤 + 超采样、overlong 软惩罚、无 KL）与 `dr-grpo.json`（去 std、常数分母）。示例只通过 `--rl-algorithm-spec` 显式使用，不改默认值。CPU 测试加载每个示例，确认能构建、能通过能力检查（全部机制声明后）、argv 能被 upstream 解析。

### D10. P0 归档后的合并

P0 归档后，把 `rl-grpo-variants` 中"KL loss 与参考模型身份"对描述结构的要求，改写为 `rl-algorithm-spec` 的 delta；其余要求保留在本 capability。

## Risks / Trade-offs

- [分派器与 Miles 内置逻辑漂移：Miles 升级后内置 `_post_process_rewards` 改动，yeto 复刻版不再等价] → 等价测试直接 import Miles 原函数比对；把 `train_data_conversion.py` 的源码哈希锁在测试里，Miles 升级时测试失败并提示重新审查。
- [分派器替换了内置路径，任何 bug 都直接改变训练目标] → 默认 GRPO 不启用分派器；启用时 CPU 逐元素等价测试覆盖多段样本、G=1、std=0、rewards_normalization 关闭、非 grpo 估计器。
- [KL loss 显存：多加载一份 ref 模型（LoRA 下 ref 即 base 权重，但 Miles 是否复用 actor 的 base 权重未核实），并增加一次 ref 前向] → G1 冒烟记录峰值显存与每轮耗时，与默认 GRPO 同配置对比写入报告；OOM 时记录为"该卡型不支持"，不静默降配。
- [ref 身份字段依赖 base_model_revision 的准确性] → 与外层身份同源，不引入新的真相来源。
- [overlong 过滤在全部截断的轮次无梯度，若判定写错会误报或漏报] → CPU fake 测试覆盖全截断、部分截断两种情况；G1 中人为把生成长度调小触发截断。
- [vendor 的 reducer 与上游示例分叉] → 文件头记录来源 commit 与原始哈希，测试锁定；只改分母来源，逻辑保持一致。
- [超采样下外层等权平均的偏差] → 只记录，不修正；需要修正时另立 change。
- [dual-clip 与 clip-higher 的组合、token 聚合与常数分母的组合语义] → 前者允许；后者拒绝（D2）。

## Migration Plan

1. 只影响 ports 路径，所有新机制都需显式选择；合入后现有 ports 配置行为与哈希不变。
2. 能力声明按机制逐项加入，每项对应一次 G1 通过的记录；某项 G1 失败时保持未声明，不阻塞其他机制合入。
3. 回滚：从能力声明中去掉该机制即可让它恢复为启动前拒绝；分派器在默认路径上不启用，回滚不影响默认 GRPO。

## Open Questions

- Miles 在 LoRA 下加载 ref 时是否复用 actor 的 base 权重，决定 KL loss 的实际显存开销。由 G1 测量回答，不影响方案。
- syncer 外层平均是否按样本加权（research §10 问题 4）。本 change 只记录数据，不依赖其答案。
