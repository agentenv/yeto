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
