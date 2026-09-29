## Why

框架 change `rl-algorithm-capabilities`（P0）完成后，训推不一致修正（TIS、IcePop、OPSM、MIS）已经能在 AlgorithmSpec 的 `correction` 组里表达和翻译，但 Miles adapter 不声明支持，选用时启动前被拒。同时 yeto 还没有 ports LoRA 路径上 SGLang 与 Megatron 训推不一致的任何数据，无法判断这些修正是否值得开启（research.md §7.3）。本 change 先提供"只观测"手段量化不一致，再把各修正机制逐一开放，并在真实 GPU 上验证。

## What Changes

- **只观测模式**：新增 yeto 插件，作为 Miles 的自定义修正函数挂入，修正权重恒为 1、屏蔽原样返回，只输出 Miles 的 mismatch 指标（`tis`、`tis_abs`、`train_rollout_kl`、`ess_ratio` 等）。它不改变训练，梯度与未启用时数值相同。
- **TIS**：开放 `--use-tis --tis-clip --tis-clip-low`。两个阈值必须在描述中显式给出，yeto 不设默认值。与 `use_rollout_logprobs` 的互斥沿用 P0 的拒绝规则。
- **IcePop**：开放 Miles 内置的 `icepop_function`（区间内乘以 ratio，区间外置零）。上下界必须显式给出。
- **OPSM**：开放 `--use-opsm --opsm-delta`。δ 必须显式给出；新增显式字段选择 π_old 的来源（训练端重算或推理端），默认沿用 Miles 的训练端重算。
- **MIS / geo-MIS**：先核实运行镜像能否 import Miles examples 中的 `mis.py`；不能时 vendor 进 yeto 插件，记录来源 commit 与许可证，插件哈希进入算法身份。
- **零梯度判定**：每个带屏蔽的机制声明自己的 `expects_gradient` 判定，合法的全屏蔽轮次不判失败。
- **能力声明**：Miles adapter 只在 G1 冒烟通过后，才在 `EngineCapabilities.corrections` 中逐项加入对应机制；fake engine 同步声明供 CPU 测试。
- **文档**：`docs/MILES_RL.md` 说明各修正的用法、阈值含义、logprob 来源和验证状态。
- 默认 GRPO 的行为、argv 和哈希不变；新机制只在显式选择时生效。不改 legacy `build_miles_argv`，不改 Miles/SGLang fork。

## 机制覆盖

状态标记沿用 P0：⚙ 可表达，未开放；✅ 声明支持。验证层级：C = CPU 测试，G1/G2/G3 见 design。

| 机制 | Miles 实现（`9e4260d`） | 状态变化 | 声明所需验证 |
|---|---|---|---|
| 只观测 mismatch 指标 | `--get-mismatch-metrics` + `--custom-tis-function-path`（yeto 插件），A:1705/1759 | ⚙ → ✅ | C（梯度不变对照）+ G1 + G2 报告 |
| TIS | `--use-tis --tis-clip --tis-clip-low`，`LH/corrections.py:7` | ⚙ → ✅ | C + G1 + G3 |
| IcePop | `--custom-tis-function-path ...corrections.icepop_function`，`LH/corrections.py:35` | ⚙ → ✅ | C + G1 + G3 |
| OPSM | `--use-opsm --opsm-delta`，`LH/math_utils.py:183` | ⚙ → ✅（仅声明的 π_old 来源） | C + G1 |
| MIS / geo-MIS | `examples/infra_features/train_infer_mismatch_helper/mis.py:311`（或 vendor 副本） | ⚙ → ✅ | C + G1 |

任何一项 G1 未通过，该项保持 ⚙，不影响其他项声明。G3 未通过时 TIS/IcePop 仍可声明，但文档标注"两岛未验证"。

## Capabilities

### New Capabilities
- `rl-mismatch-correction`：ports 路径上训推不一致的观测与修正：只观测模式、TIS、IcePop、OPSM、MIS 的描述要求、阈值显式性、logprob 来源、零梯度判定、能力声明门槛与验证层级。

### Modified Capabilities
（无。P0 与 R0 的 capability 尚未归档，本 change 只以新 capability 表达；归档后的合并见 design D9。）

## Impact

- 代码：`yeto/rl/algos/`（新增只观测插件，可能新增 MIS vendor 插件）、`yeto/rl/engine/algorithm.py`（correction 组校验、OPSM logprob 来源字段、`expects_gradient` 各机制判定）、`miles_adapter/algorithm_flags.py`（翻译）、`miles_adapter/entry.py` 与 `engine/fake.py`（能力声明）、`miles_adapter` 的指标读取（mismatch 指标、`masked_fraction`）。
- 测试：新增 CPU 数值对照测试（依赖 Miles 源码，在 `/home/michael/work/miles-next-venv` 运行，miles 不可用时 skip）；upstream parser 解析测试。
- 文档：`docs/MILES_RL.md`。
- GPU：G1（1 卡）、G2（1 卡）、G3（1+1 卡），需用户批准卡数与预算。decoupled 下的对比实验不在本 change 内，须等 `fix-decoupled-lr-schedule` 合入。
- 不在本 change 范围：效果 A/B（G4，另行申请预算）；staleness>0 的异步模式（需另立独立算法契约 change，alignment A6）；是否默认开启某修正的决定（由 G2 报告交用户决定）。
