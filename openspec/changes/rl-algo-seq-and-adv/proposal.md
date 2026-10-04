# Proposal

## Why

P0（`rl-algorithm-capabilities`）完成后，GSPO、REINFORCE++（及 baseline 版）已经能在 `AlgorithmSpec` 中表达，MaxRL、MAPO、GDPO 这类组内 advantage 变换也能以插件形式表达，但 Miles adapter 都没有声明支持，启动时会被拒绝。GSPO 和 REINFORCE++ 由 Miles `9e4260d` 原生实现，缺的是 yeto 侧的约束、零梯度判定、岛内统计语义和验证；MaxRL/MAPO/GDPO 在 Miles 中不存在，需要挂在 P1-b（`rl-algo-grpo-knobs`）引入的 yeto reward 后处理分派器上实现。调研依据见 `docs/research/rl-algorithms/research.md` §3、§4、§6、§7。

## What Changes

- **GSPO**：开放 `advantage.estimator=gspo`（序列级几何平均 ratio 与序列级 clip）。clip 必须显式给出（P0 已拒绝未给的情况）；Miles 默认 0.2，论文为 3e-4/4e-4，本 change 不替用户选值。GSPO 声明"全部序列被裁掉时不期望梯度"的零梯度判定，依据 `pg_clipfrac`。
- **REINFORCE++ / REINFORCE++-baseline**：开放 `reinforce_plus_plus` 与 `reinforce_plus_plus_baseline`，配合 advantage 白化（`--normalize-advantages`），允许 `kl.placement=reward`。**"全局 batch 白化"在 yeto 中只在岛内（本岛 DP 组）进行，不做跨岛 all-reduce**，以保持"一个协议多个后端"。`--gamma` 纳入映射表，默认 1.0 不输出；非 1.0 可表达但本 change 不开放。
- **MaxRL**：A=(r−r̂)/r̂，r̂ 为组均值，全错组 advantage 为 0。作为分派器的 advantage 变换实现，要求奖励声明为二值。
- **MAPO**：A=(1−λ)(r−μ)/σ+λ(r−μ)/μ，λ=1−4p(1−p)。同样走分派器，要求二值奖励。**证据弱**：原文只有 7B 多模态实验，小幅提升，ICLR 投稿已撤回。
- **GDPO**：每个奖励分量先组内归一化，再按权重求和，最后在岛内 batch 白化。约定 reward 函数写入样本元数据的**奖励向量**格式；分量名与权重在算法描述中声明并进入哈希，运行时缺分量、多分量、非有限值都会使本轮失败。
- **多段语义**：所有新变换都保持 Miles 内置的 rollout_key 多段合并语义（同一 rollout 的多段样本共享一个 advantage，奖励不一致则报错）。
- **零梯度判定**：每个机制声明 `expects_gradient` 判定。MaxRL/MAPO 的全错组、全对组 std 为 0，与 R0 的 `reward_std>0` 判定一致，天然不会误报；GSPO 按 clipfrac 判定；GDPO 与 REINFORCE++ 按变换输出判定。
- **声明支持的时机**：每个机制在 GPU G1 冒烟通过后才加入 adapter 能力声明。
- **文档**：更新 `docs/MILES_RL.md`。
- **不变的部分**：默认 GRPO 的行为、哈希和 argv 逐字节不变，新机制必须显式选择；legacy `build_miles_argv` 不变；不修改 Miles/SGLang fork。

## 机制与状态变化

验证层级沿用 research §9：C1–C5 为 CPU 层，G1 为 1 卡冒烟，G3 为两岛外层兼容。G4（效果 A/B）不在本 change 范围，**本 change 不声称任何效果收益**。

| 机制 | Miles 实现位置（`9e4260d`） | 状态变化 | 验证层级 |
|---|---|---|---|
| GSPO | `--advantage-estimator gspo`（A:1624）；`LH/losses.py` gspo 分支；`math_utils.py:224-251` `compute_gspo_kl` | 可表达未开放 → 声明支持 | C2、C3、C4、G1（报告 clipfrac） |
| REINFORCE++ | `LH/advantages.py:79`；`--normalize-advantages`（A:1673）；`--kl-coef`（A:1592）；`--gamma`（A:1680 附近） | 同上 | C2、C3、C4、G1 |
| REINFORCE++-baseline | `LH/advantages.py:92`；组内减均值在 `train_data_conversion.py` `_post_process_rewards` | 同上 | C2、C3、C4、G1 |
| MaxRL | 无；yeto 分派器（P1-b）经 `--custom-reward-post-process-path`（A:2393） | 同上（yeto 变换） | C4（逐元素对照公式）、G1、G3 |
| MAPO | 同上 | 同上（yeto 变换，证据弱） | C4、G1 |
| GDPO | 同上，另需奖励向量约定 | 同上（yeto 变换 + reward 接口） | C3、C4、G1 |
| `--gamma` ≠ 1.0 | `get_reinforce_plus_plus_returns` | 不可表达 → 可表达未开放 | C2、C3 |

"声明支持"只表示 G1（以及适用时的 G3）通过，不表示有效果收益。

### 风险提示

- **GSPO**：论文收益主要来自 MoE 全参训练（不需要 routing replay）；在 ports 的 LoRA 加小模型上**没有证据**。窄 clip 下被裁比例可能很高（论文报告比 GRPO 多约 100 倍），必须监控 `pg_clipfrac`。另外，每轮只有一个 optimizer step 时 ratio 恒约为 1，clip 实际不起作用，GSPO 退化为"序列级 advantage 的 GRPO"，G1 必须用多于一个 optimizer step 的配置才算覆盖了 clip。
- **REINFORCE++**：论文的"全局 batch"在 yeto 里只是岛内 batch [推断]；各岛 advantage 尺度可能不同，对外层等权平均的影响未知 [推断]。
- **MAPO**：证据弱，见上。

## Capabilities

### New Capabilities

- `rl-advantage-and-sequence-variants`：ports 路径上序列级 ratio 与 advantage 变体的开放契约，包括：
  - GSPO 的 clip 要求与零梯度判定；
  - REINFORCE++ 家族的白化范围（岛内）、KL 放置与 gamma；
  - MaxRL、MAPO、GDPO 的数值定义、二值奖励要求和边界行为；
  - 奖励向量的声明与校验；
  - 多段 rollout 语义保持；
  - 各机制的零梯度判定；
  - "验证通过后才声明支持"的规则。

### Modified Capabilities

无。R0、P0、P1-b 的 capability 均未归档，本 change 的要求全部以新 capability 表达。

## Impact

- **代码（仅 ports 路径）**：
  - `yeto/rl/engine/algorithm.py`：advantage 变换种类、GDPO 分量声明、gamma、白化字段校验，各机制的 `expects_gradient`；
  - `yeto/rl/engine/miles_adapter/algorithm_flags.py`：新增 `--gamma` 映射行，确认 gspo/rpp 相关翻译；
  - `yeto/rl/engine/miles_adapter/entry.py`：按 G1 结果逐项加入能力声明；
  - `yeto/rl/algos/reward_pipeline.py`（P1-b 创建）：注册 MaxRL、MAPO、GDPO 变换和 rpp/rpp_baseline 内置等价路径；
  - `yeto/rl/engine/driver.py` / adapter 指标读取：`pg_clipfrac` 进入 `masked_fraction`；
  - `yeto/rl/engine/fake.py`：同步能力声明。
- **不受影响**：legacy `build_miles_argv`；Miles/SGLang fork；syncer。
- **测试**：CPU 单测覆盖翻译、拒绝、变换数值（逐元素对照论文公式，覆盖 G=1、std=0、全对、全错、多段）、与 Miles 原函数等价（rpp/rpp_baseline/gspo）、奖励向量校验、零梯度判定；在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv。
- **GPU**：需要用户批准卡数与预算。每个机制 1 卡冒烟 2–3 轮；至少一个 advantage 变换做两岛 strict-avg（1+1 卡）。decoupled 下的任何对比实验等 `fix-decoupled-lr-schedule` 合入后再做，本 change 不包含。
- **依赖**：P0 `rl-algorithm-capabilities`；P1-b `rl-algo-grpo-knobs` 的 reward 后处理分派器（本 change 只扩展变换，不重新定义分派器）。
- **兼容性**：只增加可选机制。未选择新机制的 ports 配置，哈希和 argv 不变。
