# Proposal

## Why

P0（`rl-algorithm-capabilities`）完成后，GRPO 家族的常用机制（clip-higher、dual-clip、token 级聚合、Dr.GRPO、KL loss、entropy、超采样、overlong 处理）在 `AlgorithmSpec` 中都能表达，但 Miles adapter 没有声明支持，启动时会被拒绝。这些机制大多由 Miles `9e4260d` 原生实现，缺的是 yeto 这一侧的翻译核对、插件、CPU 证明和 GPU 冒烟。overlong 塑形则需要 yeto 自己持有一个 reward 后处理分派器；这个分派器也是 P2（`rl-algo-seq-and-adv`）实现 MaxRL/MAPO/GDPO 的前提。调研依据见 `docs/research/rl-algorithms/research.md` §3、§4.1、§4.2、§6。

## What Changes

- **逐项开放 Miles 原生机制**：clip-higher（`--eps-clip-high`）、dual-clip（`--eps-clip-c`，c 必须大于 1）、token 级聚合（`--calculate-per-token-loss`）、Dr.GRPO 去 std（`--disable-grpo-std-normalization`）、KL loss（`kl.placement=loss` → `--use-kl-loss --kl-loss-coef --kl-loss-type`）、entropy（`--entropy-coef`）、超采样（`--over-sampling-batch-size`）。每项在 GPU G1 冒烟通过后，才在 adapter 的能力声明中加入。
- **Dr.GRPO 常数分母 reducer**：经 `--custom-pg-loss-reducer-function-path` 接入一个常数分母的 pg_loss reducer。先核实 Miles 自带的 `examples/experimental/DrGRPO/custom_reducer.py` 在运行镜像中能否 import；该示例的分母写死为 1000，且依赖 `examples` 包路径，因此计划 vendor 进 yeto，分母改为 spec 字段，并记录来源 commit 和源文件哈希。
- **RLOO 不单独实现**：design 给出与 Dr.GRPO 只差常数缩放的推导。
- **yeto reward 后处理分派器**：新增唯一的 `--custom-reward-post-process-path` 入口。它依次执行 reward 塑形和 advantage 变换，本 change 的 advantage 变换是默认 GRPO 组归一化。Miles 的 custom post-process 会整体替换内置 `_post_process_rewards`，所以分派器必须等价重现内置逻辑（按 prompt 分组、按 rollout_key 合并多段样本、同一 rollout 奖励一致性检查、grpo/gspo 的 std 归一），并用 CPU 测试与 Miles 原函数逐元素比对。分派器预留变换注册点，供 P2 复用。
- **overlong 软惩罚（DAPO）**：作为分派器的第一个 reward 塑形阶段，参数为 L_max 与 L_cache。
- **overlong 过滤**：组合进 yeto 已占用的 sample-filter hook，对截断样本设置 `remove_sample=True`，使其不计入 loss。这些样本仍参与 advantage 计算，与 DAPO 原文的差异在 design 中写明。
- **ref 模型身份**：KL loss 会加载 `--ref-load` 指向的参考模型。P0 的岛间一致只校验算法哈希，不覆盖 ref。本 change 把 ref 身份纳入 KL loss 描述，使其进入算法哈希（见 design D6）。
- **命名示例（可选）**：提供 `dapo-like`、`dr-grpo` 等 spec JSON 示例。示例只是写好的配置，不改变默认行为。
- **文档**：更新 `docs/MILES_RL.md`。
- **不变的部分**：默认 GRPO 的行为、哈希和 argv 逐字节不变，新机制必须显式选择；legacy `build_miles_argv` 不变；不修改 Miles/SGLang fork。

## 机制与状态变化

验证层级沿用 research §9：C1–C5 为 CPU 层，G1 为 1 卡冒烟，G3 为两岛外层兼容。G4（效果 A/B）不在本 change 范围。

| 机制 | Miles 实现位置（`9e4260d`） | 状态变化 | 验证层级 |
|---|---|---|---|
| clip-higher | `--eps-clip-high`，A:1583；`math_utils.py:254-277` | 可表达未开放 → 声明支持 | C2、C3、G1、G3（组合） |
| dual-clip | `--eps-clip-c`，A:1585；`math_utils.py:267-273` | 同上 | C2、C3、C4、G1 |
| token 级聚合 | `--calculate-per-token-loss`，A:1556；`cp_utils.py:92` | 同上 | C2、G1、G3（组合） |
| Dr.GRPO 去 std | `--disable-grpo-std-normalization`，A:1675 | 同上 | C2、C4、G1 |
| Dr.GRPO 常数分母 | `--custom-pg-loss-reducer-function-path`，A:1765；示例 `examples/experimental/DrGRPO/custom_reducer.py` | 同上（yeto vendor 插件） | C4、G1 |
| RLOO | 不单独实现（≈ Dr.GRPO） | 不变（不纳入） | design 推导 |
| KL loss（k1/k2/k3/low_var_kl） | `--use-kl-loss` 等，A:1650/1653/1617；ref 加载 `ray/specs/train.py:58` | 同上 | C2、C3、G1（含显存记录） |
| entropy | `--entropy-coef`，A:1670 | 同上 | C2、G1 |
| 超采样 | `--over-sampling-batch-size`，A:789 | 同上 | C2、G1 |
| overlong 软惩罚 | 无；yeto 分派器经 `--custom-reward-post-process-path`（A:2393） | 同上（yeto 插件） | C4（与 Miles 原函数逐元素比对）、G1、G3（组合） |
| overlong 过滤 | 无；yeto sample-filter hook 设置 `remove_sample`（`train_data_conversion.py:99`） | 同上（yeto 插件） | C4、G1 |

"声明支持"只表示 G1（以及适用时的 G3）通过，不表示有效果收益。

## Capabilities

### New Capabilities

- `rl-grpo-variants`：ports 路径上 GRPO 家族机制的开放契约，包括：
  - clip 与聚合类机制的开放条件和参数约束；
  - Dr.GRPO 与常数分母插件；
  - KL loss 与参考模型身份；
  - entropy 与超采样；
  - yeto reward 后处理分派器及其与内置归一化的等价性；
  - overlong 塑形与过滤；
  - 每个机制"验证通过后才声明支持"的规则。

### Modified Capabilities

无。R0 与 P0 的 capability 尚未归档，本 change 的要求全部以新 capability 表达。P0 归档后，与 `rl-algorithm-spec` 重叠的部分（KL loss 描述中的参考模型身份）再改写为 delta，见 design D10。

## Impact

- **代码（仅 ports 路径）**：
  - `yeto/rl/engine/algorithm.py`：dual-clip 约束、reducer 分母、KL 参考模型身份、overlong 参数；
  - `yeto/rl/engine/miles_adapter/algorithm_flags.py`：确认翻译行；
  - `yeto/rl/engine/miles_adapter/entry.py`：按 G1 结果逐项加入能力声明；
  - `yeto/rl/engine/miles_adapter/config.py`：分派器路径、runtime attrs 下发插件配置；
  - `yeto/rl/engine/miles_adapter/rollout_meta_hook.py`：组合 overlong 过滤；
  - 新增 `yeto/rl/algos/reward_pipeline.py`（分派器）与 `yeto/rl/algos/reducers.py`（vendor 的常数分母 reducer）；
  - `yeto/rl/engine/fake.py`：同步能力声明。
- **不受影响**：legacy `build_miles_argv`；Miles/SGLang fork；syncer。
- **测试**：CPU 单测覆盖翻译、拒绝、分派器与 Miles 原函数的逐元素等价、overlong 数值、reducer 数值；在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv。
- **GPU**：需要用户批准卡数与预算。每个机制 1 卡冒烟 2–3 轮；组合配置两岛 strict-avg（1+1 卡）。decoupled 下的对比实验等 `fix-decoupled-lr-schedule` 合入后再做，本 change 不包含。
- **依赖**：P0 `rl-algorithm-capabilities`。分派器被 P2 `rl-algo-seq-and-adv` 复用。
- **兼容性**：只增加可选机制。未选择新机制的 ports 配置，哈希和 argv 不变。
