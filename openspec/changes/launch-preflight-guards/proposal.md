# Proposal

## Why

上卡前缺少三道自动检查，已经造成真金白银的浪费。本机线程数超限曾两次触发线程守卫，拆掉了已开机的 Flash-Next 运行（SkyPilot API 服务约占 1–2k 线程）。S17 M1 两次显存不足约花 $8，其中一次是词表 logits 块 7.6 GiB（infra-drafts/S17-MORNING-REPORT.md）。N17 的 N12/N14 负例（身份不符、学习率调度不同被拒）只能在本机测，因为 launcher 不能单独给某一个岛换参数（infra-drafts/S17-NIGHT-REPORT.md §九）。

## What Changes

- launcher 在起任何云资源之前读本用户线程数。低于开机门槛（默认 2800）才继续。达到门槛时按配置等待或报错。达到硬线（默认 3000）时直接报错。报错与等待信息里写出 SkyPilot API 服务占多少线程，并给出重启办法。
- launcher 在起机前估算每张卡的显存峰值。估算按模型参数量、LoRA 或全参、上下文与回复长度、词表 logits 块、并行布局、卡型显存计算。峰值超过阈值时起机前报错，并给出建议（降上下文、降回复长度、加卡或换卡型）。现有 `warn_if_model_wont_fit` 只比权重总量，保留为第一道粗检。
- launcher 新增"单独给某一个岛换参数"的开关，只用于真机造负例岛。默认不开。开了必须同时打开负例运行开关，launcher 打印警告，岛参数差异写进 tape 和运行清单，看板标出该岛。带这个标记的运行不能导出为正式权重。
- 所有新检查都可以用显式开关关闭，关闭动作本身写进运行清单。

## Capabilities

### New Capabilities
- `launch-preflight`: 起机前的线程数预检与显存估算预检，包括门槛、行为配置、报错内容与关闭方式。
- `island-param-override`: 单独给某一个岛换参数的负例开关，包括允许的参数、安全开关、tape 与看板记录、正式训练禁用。

### Modified Capabilities

（无。现有 spec 的要求不变。）

## Impact

- 代码：`yeto/launcher.py`（`launch` 6915 起、`warn_if_model_wont_fit` 5162、岛任务循环 7086–7095、dry-run 7616–7636、`_write_run_manifest` 7542）、`yeto/cli.py`（rl 参数组 476–520、head 路径 1733 与 2032 调 `prepare_launch_args`）、`yeto/models.py`（`MODEL_WEIGHT_GB` 151）、看板 reducer。
- 新增一个显存估算模块和一个线程计数模块（纯 Python，读 /proc）。
- 运维：上卡链脚本里手写的 `ps -L -u michael | wc -l` 等待循环（如 rl-infra-spec/evidence/s17-i3-45/chain.sh:6）可以改用 launcher 内置检查。
- 预算：一次小卡上卡验证 N12/N14 负例，估计 $10–15，上限 $20，需报批。
