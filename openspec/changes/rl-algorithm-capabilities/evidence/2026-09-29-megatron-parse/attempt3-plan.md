# 2.6 第 3 次运行计划（写于运行前，提交后才运行，之后不改本文件）

**起因**：复审 E2。第 2 次运行没有执行原计划中的 argv 比较，事后改了比较口径；原计划的比较对象 `test_rl_argv_snapshot` 本身也选错了，那是 legacy argv。本计划修正比较口径，并把比较脚本化。

**argv 的来源**：走 ports learner 自己的代码路径。
- `yeto.rl.learner.parse_args(... --rl-algorithm-spec <case>.json)` 得到 args。
- `yeto.rl.learner.build_ports_launch(args, run_config)` 生成 launch。这个函数就是 `run_miles` 实际调用的那一步（本次从 run_miles 中提取出来，逻辑不变）。
- run_config 用测试夹具 `make_config`（tiny Qwen3，HF 模块名，路径固定在 `/tmp/algocap26`）。真实的 run_config 需要 Megatron model bridge 生成 provider，本次不覆盖这一段；本次覆盖的是“spec → argv → 带 Megatron 校验的 upstream parse”。

**用例**：`cases3.py` 固定 28 例。
- 默认 GRPO；v1 格式的 kl_coef=0 加 bounded filter。
- 所有单字段非默认映射，其中包括：constant 聚合加 reducer、reinforce_plus_plus_baseline、KL loss 的 k1/k2/k3/low_var_kl 四种估计器及 unbiased。
- 两个多字段组合：DAPO 风格，以及 KL+TIS+dual-clip。
- 不含 ppo：ports 在启动前会以 critic 为由拒绝它，与 Megatron 解析无关。

**步骤与判定**：
1. 本地（/tmp/yeto-venv）：`parse3.py local <repo> local3.json`，得到每例 argv。
2. 远端（私有 ports 镜像 digest 5da40a07…，Modal T4 ×1，`timeout=1800`）：`parse3.py remote`，得到每例 argv，并对每例跑 upstream `parse_miles_args`（megatron 后端 `parse_args`，含 Megatron validate、miles/sglang validate、yeto validate_parsed_args），再核对 expected 字段。
3. `compare3.py local3.json remote3.json compare3.json`。

**成功条件（全部满足才重新勾选 2.6）**：
- 28 例远端解析全部通过；
- expected 字段全部相等（浮点相对误差 1e-12）；
- 每例本地与远端 argv 的 JSON 逐字节相同（`compare3.py` 退出码 0）。

**记录**：运行时的 yeto commit SHA 由 `git rev-parse HEAD` 取得（运行前工作区必须干净），打印在 log 首行；同时记录 `/root/miles` 的 HEAD。

**资源与回收**：app `algocap-parse`，sandbox 在 finally 中 terminate；运行后执行 `modal app stop -y`，并核实 `Sandbox.list`。另有独立看门狗（sleep 2400 后 stop app）。

**费用**：估计 < $0.2，上限 $1。失败时如实记录；同一失败只在查明原因并修复后才重跑。
