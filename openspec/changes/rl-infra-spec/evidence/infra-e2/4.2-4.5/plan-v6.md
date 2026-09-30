# 4.2–4.5 GPU 验证计划 v6（INFRA-E2，2026-09-30；取代 plan-v5.md，旧版保留）

## 变更：只改环境，判据文字不变
- **原因**：C1 诊断 2 发现 fork M5 的恢复在"state 非空但缺 Adam 状态"时会静默丢弃 `exp_avg`/`exp_avg_sq`（`gpu-c1-diag2/`、`infra-drafts/fork-req-m5-lazy-state.md`）。主 agent 裁定：等 fork 修复（分支 yeto-m5-lazy-state）→ 审查 → 快进 yeto/ports → 重建镜像与 pin，然后在最终镜像上一次性正式跑 C1 → C2 → C3。
- **pin**（fork 合入后填写，运行前固定；工具 `e2_cut_harness.py` 的 pin 校验同步改到这两项，否则 rc=3）：
  - Miles fork：`<待填：yeto/ports 快进后的提交>`
  - 镜像：`ghcr.io/michaellchung/yeto-miles-ports@sha256:<待填>`（tag `<待填>-9f29303`）
- **其余不变**：plan-v5 的配置（C1/C2 fixed-partition，C1 H100!×2、C2 H100!×3，C3 H100!×3）、运行顺序、确定性要求、dropout 0.05、费用表与执行规则（`--modal-retries 0` 即不 relaunch；进度看门狗；B2 累计接近 $40 时停下）。
- **yeto 侧防御**：`MilesCutBackend.export_optimizer` 包在 `side_effect_free_state` 中，只删除本次读取新建的空 state 条目，从不吞异常。fork 修复后，如果出现键不匹配之类的报错，照常向上抛出并判 RECOVERY_REQUIRED（CPU 测试 `test_side_effect_free_state_never_swallows_errors`、`test_a_fixed_fork_setter_error_reaches_restore_cut`）。
- **运行前检查**：用新 pin 重跑本地 dry-run（22 项）和镜像内 T4 learner preflight（c1/c2/c3-rb），全部通过后才启动 C1。
