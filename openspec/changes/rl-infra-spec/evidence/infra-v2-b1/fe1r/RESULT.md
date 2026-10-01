# F-E1 重跑（F-R1 镜像）结果，2026-09-30（冒烟，不计入 task）

计划 §9.9。代码 1a5ccd5，镜像 `sha256:db815884…`（manifest：镜像内 miles 2f23a0fc，sglang 9f29303b；launch.log 中 launcher 拉取 digest db815884… 已核对）。

- 指纹运行 fe1rfp：取得 `sha256:88ef2727…`，≤$1.17。
- F-E1 fe1r（journal `fe1r/elastic-state/reconfig/journal.jsonl`）：
  - up1（T1R1S1→T1R2S0）：`add_intent`/`fork_op start` 使用 fork cell id `engine:inference-engine-all-0-0-00001`，**start done**（F-R1 生效，B1 缺陷 1 已解除）；进入 VERIFYING 后发布失败：`ValueError: admit_cordoned needs the Miles router (--use-miles-router)`（fork inference_controller.py:621）→ REBUILD_OLD → 新 cell `stop done` → **REBUILT_OLD**。epochs：config_epoch 0，members 仍为 `engine:inference-engine-all-0-0-00000`（旧成员不变）。
  - down1 未执行（up 未成功，外层 17.5 min 预算超时 rc=124 后停止）。
- 判据（up、down 都 SUCCEEDED）：**不满足**。"成员用 fork cell id""旧成员不变"满足。
- 缺口（代码，交 INFRA-E1/launcher）：launcher/learner 没有 `--use-miles-router` 入口（grep 为空），而 E1 的 cordon 接纳发布必须用 Miles router（infra-e1/plan.md §0 第 2 条已写明前置）。需在 `--rl-elastic` 时强制或提供开关传递 `--use-miles-router`。
- 费用 ≤$1.75（单容器，无重试），F-E1 两次合计 ≤$2.92（上限 $3）。app ap-Ie74LuNjw5bYL92viIg6hO、ap-dD2OccuxhP4YHMir6q4lvB stopped/0。
