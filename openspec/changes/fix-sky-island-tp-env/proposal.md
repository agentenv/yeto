# Proposal

## Why

sky/Modal RL 岛的环境变量表里有两处与 Megatron 的实际要求不符，只要岛用到张量并行或 flash 注意力就必现：

- **缺 `CUDA_DEVICE_MAX_CONNECTIONS=1`**。Megatron 的参数校验在 TP>1 或 CP>1 时直接拒绝启动。SSH harness 早就按容器设了这个变量，岛任务没有，于是第一个 TP4 岛（Qwen3.6-27B on 8xH100）在 `parse_args` 阶段就失败，模型都没开始加载。
- **硬钉 `NVTE_FLASH_ATTN=0` / `NVTE_FUSED_ATTN=0` / `NVTE_UNFUSED_ATTN=1`**。learner 已经按 recipe 传 `--attention-backend`，Megatron 会在每个 actor 里依据该 flag 自行设置这三个 NVTE 变量。这个钉子对 generic provider 是冗余的，对 flash recipe 是致命的：Qwen3.6-27B 岛在模型构造时断言 `NVTE_FLASH_ATTN set to 0, but expected 1`。

两处都只影响 sky/Modal 岛任务的环境表，此前 RL 岛跑在裸机 SSH 路径上，所以没暴露。

## What Changes

- 岛任务的环境表加 `CUDA_DEVICE_MAX_CONNECTIONS=1`，在 `ray start` 之前导出，使每个 Ray worker 都继承；TP1 时无副作用。
- 移除三个 `NVTE_*_ATTN` 的硬钉，注意力后端完全交给 learner 的 `--attention-backend` 与 Megatron 自身。`deepseek-v4-flash` recipe 里原本用来撤销这个钉子的 `envs.pop` 也一并删掉，不再需要。
- 补回归测试：断言岛脚本的环境表含 `CUDA_DEVICE_MAX_CONNECTIONS=1`、不含任何 `NVTE_*_ATTN`。

## Capabilities

### Modified Capabilities
- `sky-rl-island-runtime`：该 capability 由在途的 fix-sky-island-ray-stop 引入，描述 sky 集群上 RL 岛的运行时约束。本 change 在同一 capability 下增加一条对岛环境表的要求。

## Impact

- `yeto/launcher.py`：`make_miles_island_task` 的 `envs` 表，以及 `deepseek-v4-flash` 分支里对 NVTE 变量的撤销逻辑。
- `tests/test_rl_launcher.py`：岛环境表的既有断言。
- Modal 岛执行同一份任务定义，改动同样生效。
- 不改 learner 的 `--attention-backend` 选择逻辑，不改 Miles，不改 SSH harness（它已自行设置该变量）。

## Non-goals

- 不调整各 recipe 该用哪个注意力后端——那是 learner 的既有决定。
- 不处理岛与 SkyPilot 运行时 Ray 共存的问题（见 fix-sky-island-ray-stop）。
