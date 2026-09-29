# algo1b-cpu-img 计划（实验前提交，事后不改）

- task：4.1（运行镜像中 `examples.experimental.DrGRPO.custom_reducer` 能否 import，结果只作文档，不论成败都 vendor）；
  以及 2.3/3.3/4.3 的"完整 upstream `parse_args`"补充（miles-next-venv 缺 `megatron.training`，只能用 Miles 参数提供器解析）。
- 镜像：R0 冒烟所用 ports 镜像 `docker.io/radixark/miles@sha256:90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d`（yeto `MILES_NEXT_IMAGE`，公开，无需凭据）。
  另把 miles-next 检出（`0394715`，= `MILES_NEXT_COMMIT`）与 yeto 工作树作为本地目录挂入，parse 测试以 `PYTHONPATH=/yeto:/miles-next` 运行（解析器与 pin 一致），4.1 的 import 检查只用镜像自带的 Miles（不加 /miles-next）。
- 资源：Modal CPU 函数（无 GPU），cpu=2、memory=8192MiB；app 名 `algo1b-cpu-img`（前缀 algo1b-）；`modal run` 临时 app，入口退出即结束。
- 预计时长：拉镜像 + 运行 ≤ 15 分钟；费用估算 < $0.20（CPU 计费；无 GPU）。
- 硬超时：函数 `timeout=1200` 秒；本地 `timeout 1800` 包裹 `modal run`；结束后 `modal app list` 核实无 running 的 `algo1b-*` app，残留则 `modal app stop` 按 app id 停止。
- 成功条件：4.1 记录 import 结果（成功或失败均为有效结论）；`tests/test_rl_grpo_knobs_upstream.py::test_full_parse_args` 全部参数化用例通过（不 skip）。
- 失败条件：full parse 任一用例失败或被 skip（megatron.training 不可 import）→ 2.3/3.3/4.3 仍按 Miles 参数提供器结论记录，不勾选"upstream parse_args"部分；不重跑同一实验，除非给出原因并修复。
- 容差：不涉及数值比较（解析值精确相等）。seed：不涉及。

## 第 1 次运行结论（2026-09-29，app ap-NwnNkUKRNgnMPlraQLkxhp，CPU，exit=0，已自动停止）

- 4.1：`IMPORT_FAILED`（`ModuleNotFoundError: No module named 'examples'`；镜像内 Miles 以包安装，`examples` 不在 sys.path）。按原文，结论只用于文档，照样 vendor。
- full parse_args：失败。原因是 `OSError: libcuda.so.1: cannot open shared object file`：CPU 容器里没有 CUDA 驱动库，而 Miles/Megatron 导入链会 dlopen 它。P0 自己的 `test_upstream_parse_args_accepts_translation` 在同一容器里同样失败（2 failed），所以这与本 change 的代码无关。

## 第 2 次运行计划（修复：换有驱动的容器，其余不变）

- 原因与修复：在带 GPU 驱动的容器里运行，libcuda 可以 dlopen；测试本身不在 GPU 上计算。
- 资源：Modal `gpu="T4"`（最便宜的卡），1 张；app `algo1b-cpu-img`（同名，临时 app）；函数 `timeout=1200`；本地 `timeout 1800`。
- 预计 ≤ 15 分钟，费用 < $0.30。
- 只运行 full parse 部分（4.1 已有结论，不重复）；pytest 加 `-rfEs` 输出失败与跳过的原因。
- 成功条件：`test_full_parse_args` 12 个参数化用例全部 passed（不 skip），P0 的 `upstream_parse` 2 个用例 passed。
- 失败条件：任一用例 failed 或 skipped，按实际结论记录；除非查明原因并修复，否则不再重跑。

## 第 2 次运行结论（app ap-JyzgEPFHYfn7MQw5E9lrgP，T4，exit=0，入口结束后已自动停止）

- `nvidia-smi`：Tesla T4。P0 的 upstream_parse 2 个用例 passed，说明换到有驱动的容器后完整 parse_args 可用。
- 本 change 的 13 个用例失败，原因是测试代码的 bug，与被测代码无关：
  - 12 个 full_parse 失败于 `ModuleNotFoundError: tests.test_rl_miles_adapter_config`：镜像里另有一个 `tests` 包遮蔽了本仓库的 tests 目录；
  - `test_kl_loss_triggers_ref_load_branch` 失败，因为测试里写死了本机的 miles-next 路径。
- 修复：改用 P0 的导入写法 `from test_rl_miles_adapter_config import ...`；MILES_REPO 改为从 `miles.utils.arguments.__file__` 推出。

## 第 3 次运行计划（只修上述测试 bug，其余与第 2 次相同：T4、timeout 1200/1800、成功与失败条件不变）

## 第 3 次运行结论（app ap-IFjK3spr0lcOfFUfQIopEL，T4，exit=0，已停止）

- 本 change 25 项通过：full_parse_args 12 个、Miles 参数提供器 12 个、ref-load 条件 1 个。P0 的 upstream_parse 2 个通过。满足本计划的成功条件。
- 各次运行挂载的 yeto 代码版本见 `YETO_SHA.txt`；日志 run1/2/3.log 已用 `git add -f` 入库，入库前扫描过密钥，无命中。

## 第 4 次运行计划（审查 F6/4.1/7.1/5.2 补充；实验前提交）

- 目的：
  - (a) 4.1 补证据：打印镜像内 miles 的实际位置（`miles.__path__`、`pip show -f miles`）、`examples/` 目录是否存在，以及在 run1 相同条件下（cwd=/tmp、不设 PYTHONPATH）和在 Miles 仓库根目录作为 cwd 时分别 import 的结果；
  - (b) 7.1：两个示例 `dapo-like`、`dr-grpo` 跑完整 `parse_args` + `validate_parsed_args`（miles-next 0394715 解析器，镜像内的 megatron）；
  - (c) 5.2 补充：只用镜像自带的 Miles（PYTHONPATH 只含 /yeto），跑 `test_rl_reward_pipeline_equivalence.py`，并打印镜像内 `train_data_conversion.py` 的 sha256。
- 资源、超时、回收：与第 2、3 次相同（T4×1，函数 timeout 1200，本地 timeout 1800，临时 app `algo1b-cpu-img`）。预计 ≤ 15 分钟，费用 < $0.3。
- 成功条件：
  - (b) 的 `test_full_parse_args` 共 14 项（12 项机制 + 2 个示例）全部 passed，无 skip；
  - (c) 的 equivalence 全部 passed。若只有源码哈希锁定一项失败，说明镜像内的 Miles 与 0394715 不同源；该结论照实记录，不改锁定值；
  - (a) 只记录事实。
- 失败条件：(b) 或 (c) 中任一项 failed，照实记录，除非查明原因并修复，否则不重跑。
