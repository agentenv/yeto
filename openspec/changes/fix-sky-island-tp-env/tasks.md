# Tasks

## 1. 用测试钉住岛环境表

- [x] 1.1 在 `tests/test_rl_launcher.py` 的 `test_miles_task_checks_out_exact_commit_and_builds_multinode_ray` 里改写环境表断言：`NVTE_FLASH_ATTN`/`NVTE_FUSED_ATTN`/`NVTE_UNFUSED_ATTN` 三者都不在 `task.envs` 中；`task.envs["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"`。验证：修复前两条断言均失败，修复后通过

## 2. 环境表

- [x] 2.1 在 `make_miles_island_task` 的 `envs` 中加入 `CUDA_DEVICE_MAX_CONNECTIONS=1`，并确认它随 `task.envs` 在 run 脚本执行（含 `ray start`）之前导出。验证：任务 1.1 通过
- [x] 2.2 删除 `envs` 中三个 `NVTE_*_ATTN` 的硬钉，以及 `deepseek-v4-flash` 分支里用于撤销该钉子的 `envs.pop` 循环（钉子没了，撤销也就多余）。验证：任务 1.1 通过；`git grep NVTE_.*_ATTN yeto/launcher.py` 仅剩说明性注释（`NVTE_GROUPED_LINEAR_SINGLE_PARAM` 属 deepseek 分支的另一变量，不在本 change 范围）

## 3. 回归

- [x] 3.1 跑 `tests/test_rl_launcher.py` 全量，确认岛脚本与环境表的既有断言不回归

## 4. 待补

- [ ] 4.1 `deepseek-v4-flash` recipe 目前只有 `_prepare_rl_args` 层的测试，没有针对该 recipe 岛环境表的断言。本 change 未新增——移除 `envs.pop` 后该分支的环境表已由 2.2 的通用断言覆盖，但一条显式的 recipe 级断言更稳妥
