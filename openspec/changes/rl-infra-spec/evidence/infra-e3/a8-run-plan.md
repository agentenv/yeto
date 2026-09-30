# A8 运行记录计划（B3，INFRA-E3，2026-09-30；运行前提交）

- 判据、容差、go/no-go：`plan-v6.md`（与 v3 相同，不放宽）。本页只记执行参数。
- 资源：Modal Sandbox `H100!:2`，`timeout=7200`（2 h 硬超时）；app 名 `infra-v2-b3-a8-20260930-1`；上限 $15.8（含取回 packed 状态最坏约 $2.6）。Sandbox 无自动重试（等同 `--modal-retries 0`）。
- 代码：infra-e3 当前推送提交（运行时 SHA 写入证据）；上传 `git archive` 快照 + gsm8k_reward.py（sha256 743d433c…）。pin 从 `yeto/rl/__init__.py` 读取：Miles e3a11ab3、镜像 `@sha256:2cc5cc52…`（容器内断言 `/root/miles` 提交；GPU 名必须全部为 `NVIDIA H100 80GB HBM3`）。
- learner 参数：`yeto launch ... --gpu modal:2xh100 --rl-single-island-no-sync --controller local --rollout-batch-size 2 --n-samples-per-prompt 8 --seed 1234 --rl-deterministic-trainer --dry-run` 生成；本地 dry-run 通过（argv 含 `--deterministic-mode`，不含 `--balance-data`，钩子全部可导入）。确定性环境变量（与 `entry.DETERMINISM_ENV` 相同）在 `ray start` 之前导出。
- 顺序：dry → gen（8 个冻结 rollout）→ A1、A2、B1、B1p、B2、RT → 容器内 compare → pack_states → 本地拉取 packed 并校验 → 事件证据打包 → 本地离线 compare，与容器内结论比对。
- 停止：任一阶段失败即停（阶段 20 分钟无进展或 5xx 超 200 行由看门狗杀掉）；本地 25 分钟无输出终止 Sandbox；任何退出路径 `modal app stop` 并用 `modal app list` 核实；另有按 app id 的独立 watchdog。线程 <3000 才启动。
- 不可判定（G3 失败）时按 plan 规则最多重跑一次（先写明原因与修复并另批）。
