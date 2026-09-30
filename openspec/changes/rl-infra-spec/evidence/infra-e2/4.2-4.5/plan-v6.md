# 4.2–4.5 GPU 验证计划 v6（INFRA-E2，2026-09-30；取代 plan-v5.md，旧版保留）

## 变更：只改环境，判据文字不变
- **原因**：C1 诊断 2 发现 fork M5 的恢复在"state 非空但缺 Adam 状态"时会静默丢弃 `exp_avg`/`exp_avg_sq`（`gpu-c1-diag2/`、`infra-drafts/fork-req-m5-lazy-state.md`）。主 agent 裁定：等 fork 修复（分支 yeto-m5-lazy-state）→ 审查 → 快进 yeto/ports → 重建镜像与 pin，然后在最终镜像上一次性正式跑 C1 → C2 → C3。
- **pin**（运行前固定，2026-09-30 填写；与 integ-decl b2fe5dd 的 `yeto/rl/__init__.py` 一致，主 agent 已用 crane 对 registry 核对）：
  - Miles fork：`e3a11ab38cbb7fd911b23fdd62a4eb6dfbb1c841`（= fb04d6ff M5 修复，外加报错文本、docstring 与两个 bucket 的测试）
  - 镜像：`ghcr.io/michaellchung/yeto-miles-ports@sha256:2cc5cc52de2444e59ddefba4f9546d1aaa13f9807ab441f56e2c70a7e7936eff`（tag `e3a11ab-9f29303`）
  - 工具 `e2_cut_harness.py` 的 pin 校验与容器内 guard 已改为这两项（tag 同时校验）。
- **诊断子运行 c1-unsafe（不计判据）**：紧接 C1，配置与 C1 相同（H100!×2，预估 $1.5 < $3 上限），关闭 yeto 的读取保护（harness 计划 `unsafe_state_reads: true`，经 rank 插件切换），在真实 DistOpt 上确认 fork 修复生效：重建后的新 trainer 在恢复前被读取过，恢复后仍须与 cut 逐位一致。结果只作 fork 修复的证据。
- **其余不变**：plan-v5 的配置（C1/C2 fixed-partition，C1 H100!×2、C2 H100!×3，C3 H100!×3）、运行顺序、确定性要求、dropout 0.05、费用表与执行规则（`--modal-retries 0` 即不 relaunch；进度看门狗；B2 累计接近 $40 时停下）。
- **yeto 侧防御**：`MilesCutBackend.export_optimizer` 包在 `side_effect_free_state` 中，只删除本次读取新建的空 state 条目，从不吞异常。fork 修复后，如果出现键不匹配之类的报错，照常向上抛出并判 RECOVERY_REQUIRED（CPU 测试 `test_side_effect_free_state_never_swallows_errors`、`test_a_fixed_fork_setter_error_reaches_restore_cut`）。
- **运行前检查**：用新 pin 重跑本地 dry-run（22 项）和镜像内 T4 learner preflight（c1/c2/c3-rb），全部通过后才启动 C1。
