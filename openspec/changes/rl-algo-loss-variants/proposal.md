# Proposal

## Why

P0（`rl-algorithm-capabilities`）之后，CISPO、SAPO-Qwen、GMPO 已能通过 `loss.variant` 表达，但处于"可表达，未开放"状态：选用会在启动前被拒。原因是 Miles `9e4260d` 的 `--loss-type` 没有这三种变体（research.md §4.2、§6）。`LH/tinker_losses.py:118` 中的 cispo 只在 tinker 路径调用：分母用 rollout logprob，规约是按样本求和，配置从 batch 的 `loss_fn_config` 读取，因此不能直接当 `--loss-type` 用。这三个变体只改 token 级目标（ScaleRL 采用 CISPO），要开放它们，必须先决定计算放在哪里：yeto 插件，还是 fork。

## What Changes

- **决策门（第一组任务）**：由用户在两条实现路线中选一条，决定之前不写实现代码。
  - 路线 A：在 yeto 插件中通过 `--loss-type custom_loss` 实现。不改 fork，但要维护一份 policy loss 副本。
  - 路线 B：在 `michaellchung/miles` 的 `yeto/ports` 分支上给 policy loss 加 variant 分支和一个新参数。改动约几十行，需要用户事先同意。

  对比和推荐见 design D1，最终**由用户决定**。
- **三个变体的数值契约**：
  - CISPO：sg(clip(ρ, 1−ε_l, 1+ε_h))·Â·log π_θ，按 token 归一，所有 token 都保留梯度；
  - SAPO-Qwen：软门控 f(ρ)=σ(τ(ρ−1))·4/τ，正负优势分别用 τ_pos 和 τ_neg；
  - GMPO：token 目标取几何平均，在 log 空间逐 token clip，默认范围 (e^−0.4, e^0.4)。

  每个变体都用小张量逐元素对照论文公式做 CPU 测试，并测与 TIS、IcePop 组合时的数值。
- **spec 字段与翻译**：定义变体参数（τ_pos、τ_neg、GMPO 的 log 空间 clip 范围；CISPO 复用 clip 上下界），写好映射表的翻译和吸收规则。变体参数进入算法哈希。
- **能力声明**：路线实现通过 CPU 测试、完成 GPU 冒烟之后，adapter 的 `losses` 维度才声明这三个变体。
- **梯度判定**：三个变体各有一条 `expects_gradient` 判定规则。
  - CISPO：所有有效 token 都保留梯度；
  - SAPO：软门控不产生硬 mask；
  - GMPO：log 空间 clip 可能让整条序列都没有梯度，需按 clip 结果判断。
- **拒绝规则**：以下组合在启动前拒绝：
  - 变体与 GSPO 序列级 ratio 组合；
  - 变体与 dual-clip 组合；
  - SAPO 配非正的 τ；
  - GMPO 的 clip 范围非法。
- **外层同步**：三个变体只改 loss，与 strict-avg 和 decoupled 正交。ports 串行执行，陈旧度为 0。本地多个 mini-batch 造成的 ratio 偏离，由各自的 clip 或门控处理。
- **不变的部分**：默认 GRPO 的哈希和 argv 不变；legacy `build_miles_argv` 不变；不向 radixark/miles、sgl-project/sglang 提 PR。

## 机制一览

| 机制 | 实现路线 | 状态变化 | 验证层级 |
|---|---|---|---|
| CISPO | 路线 A（yeto custom loss）或 B（fork variant 分支），由用户决定 | ⚙ 可表达未开放 → ✅ 声明支持 | CPU 逐元素数值测试；与 TIS/IcePop 组合的数值测试；1 卡冒烟 2–3 轮；两岛 strict-avg（1+1 卡）哈希一致 |
| SAPO-Qwen | 同上 | ⚙ → ✅ | CPU 数值测试；组合测试；1 卡冒烟 2–3 轮 |
| GMPO | 同上；需要序列级聚合，CP 下要做全收集 | ⚙ → ✅ | CPU 数值测试；组合测试；1 卡冒烟 2–3 轮 |

"✅"要等 GPU 冒烟通过后才成立。CPU 测试通过只说明实现正确，不代表 GPU 验证通过。本 change 不做效果 A/B，也不声称有效果收益。decoupled 下的对比实验要等 `fix-decoupled-lr-schedule` 合入之后再做，不在本 change 范围内。

## Capabilities

### New Capabilities

- `rl-loss-variants`：ports 路径上的 token 级 policy loss 变体（CISPO、SAPO-Qwen、GMPO），包括：
  - 数值契约；
  - 变体参数与身份哈希；
  - 与其他机制的组合与拒绝规则；
  - 按变体判定梯度；
  - 实现来源可审计；
  - 开放前需要的验证。

### Modified Capabilities

无。`rl-algorithm-spec` 尚未归档，本 change 的要求全部用新 capability 表达。

## Impact

- **yeto 代码（仅 ports）**：
  - `yeto/rl/engine/algorithm.py`：变体参数字段、校验、`expects_gradient`；
  - `yeto/rl/engine/miles_adapter/algorithm_flags.py`、`config.py`：映射与翻译；
  - `miles_adapter/entry.py`：`losses` 声明；
  - `engine/fake.py`。
  - 路线 A 另加 `yeto/rl/engine/miles_adapter/loss_variants.py`（插件，受 PluginRef 源码哈希约束）。
- **fork（仅路线 B，需用户同意）**：
  - 在 `michaellchung/miles` 的 `yeto/ports` 上加提交：修改 `loss_hub/math_utils.py` 与 `losses.py`、`arguments.py`，并补测试；
  - 更新 `MILES_NEXT_COMMIT` pin 并重建镜像。
- **测试**：CPU 数值测试、组合测试、翻译与拒绝测试，以及在 miles-next-venv 中用 upstream `parse_args` 解析。
- **GPU**：需用户批准卡数和预算。三个变体各 1 卡冒烟；CISPO 做 1+1 卡两岛 strict-avg。
- **文档**：`docs/MILES_RL.md`。
- **依赖**：依赖 P0 `rl-algorithm-capabilities`。与 TIS/IcePop 组合的 GPU 验证依赖 `rl-algo-mismatch-correction` 已开放这两个机制；在那之前，组合只做 CPU 验证。
