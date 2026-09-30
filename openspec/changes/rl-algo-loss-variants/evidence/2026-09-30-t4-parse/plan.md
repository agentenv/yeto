# algo2b-t4-parse 计划（运行前提交，事后不改）

- task：rl-algo-loss-variants 4.4 的"在钉住环境中用 upstream `parse_args` 解析生成的 argv"。按 rl-infra-spec alignment §7b 第 7 条口径：钉住镜像内运行完整 `parse_args`（Miles + Megatron 两半，含 Miles `validate_args`）+ yeto `validate_parsed_args`（即 `mc.parse_miles_args`）。
- 用户批准：2026-09-30，Modal T4 补跑，单独记账（不占 rl-infra-spec $300），上限 $2。
- 镜像：`ghcr.io/michaellchung/yeto-miles-ports@sha256:17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef`（tag 5c1b49e-9f29303，miles 5c1b49eb）。私有：拉取凭据从 ~/.docker/config.json 只读取出，只作为 `from_registry` 的 pull secret，不打印、不进任务环境。yeto 工作树 algo-2b（提交见 run 记录）挂载到 /yeto，Miles 与 Megatron 只用镜像自带的。
- 输入：yeto `translate_run_config` 为 CISPO（eps 0.2/0.28，token 聚合）、SAPO（τ+ 0.9, τ− 1.2）、GMPO（δl 0.3, δh 0.5）生成的完整 launch argv，原文保存在 result.json。不训练、不加载权重（tiny Qwen3 config.json，无权重文件）。
- 资源：Modal `gpu="T4"` ×1（只为 libcuda 可 dlopen；ALGO-1b 已证明 CPU 容器会因 libcuda 失败），cpu=2、8 GiB；app `algo2b-t4-parse`（前缀 algo2b-t4-）；`modal run` 临时 app。
- 硬超时：函数 `timeout=840` 秒（14 分钟）；本地 `timeout 900` 包裹 `modal run`；另起独立后台 watchdog（`nohup`，与 agent 终端无关）在 1080 秒后对所有 description 以 `algo2b-t4-` 开头且未 stopped 的 app 执行 `modal app stop -y <id>`。
- 预计时长：≤ 12 分钟（拉镜像为主）；费用估算：T4 约 $0.59/h + CPU/内存，< $0.30；上限 $2。
- 成功判据（全部满足才勾 4.4）：
  1. 三个变体：`parse_miles_args` 不抛异常，解析值与期望相等（policy_loss_variant、变体参数、eps、calculate_per_token_loss），且 yeto `rejections()` 为空；
  2. 应拒绝组合全部被拒：yeto 侧 gmpo+token、cispo+default 聚合、cispo 缺 eps（`rejections()` 非空）；Miles 侧在有效 argv 上追加 `--calculate-per-token-loss`（GMPO）、`--advantage-estimator gspo`（SAPO）、`--eps-clip-c 3.0`（CISPO）、`--sapo-tau-pos 0`、`--gmpo-log-clip-low=-0.1`，完整 parse_args 必须抛异常；
  3. 镜像内 /root/miles `git rev-parse HEAD` 以 5c1b49eb 开头；`nvidia-smi` 显示 T4。
- 失败处理：任一项不满足照实记录、不勾 4.4；只有查明原因并修复后才重跑，不重复启动同一实验。
- 容差：精确相等；无 seed。
- 结束：拉日志 → `modal app stop -y` 本前缀 app → `modal app list` 只核实 algo2b-t4-* → 记录实际费用。

## 第 1 次运行结论（app ap-z00PQUsU90bkkq0UVMqKq4，未产出结果）
- 失败原因：`modal run` 在容器内也会 import 驱动脚本，模块级读取 ~/.docker/config.json 在容器里 FileNotFoundError，容器反复重启（13 次 Runner failed）直到本地 `timeout 900` 结束（exit=124）；之后 app 仍为 ephemeral、0 task，已手动 `modal app stop -y`，16:17 后 stopped。未执行任何解析。凭据未出现在日志中（日志里只有源码行）。
- 修复：改为 Modal Sandbox 驱动（与 rl-algo-seq-and-adv g1_sbx.py 同法，脚本只在本地执行），`Sandbox.create(..., gpu="T4", timeout=840)`；判据、镜像、资源、上限不变；本地 `timeout 900` 与独立 watchdog（1080 秒）照旧。

## 第 2 次运行计划：同上，只换驱动方式。
