# Progress: fix-decoupled-lr-schedule

更新：2026-09-29（Agent L）

## 状态
- 分支 `fix-decoupled-lr-schedule` 已 rebase 到 `rl-engine-ports` d355e06（实现提交 20aba48，无冲突），force-with-lease 推送到 michaellchung/yeto。
- CPU 全量测试：rebase 前后在 d355e06 基线与本分支上失败集合按测试 id 去重后完全相同（94 个，均为已知环境性失败：缺 syncer 二进制、缺 miles 等），本分支多 34 个通过。
- 1.1 / 1.3 / 1.4 / 2.1 / 2.2 / 2.3 / 3.1：此前已完成。
- 1.2：完成——upstream parse_args 测试在公开 `MILES_NEXT_IMAGE` 中通过（`evidence/2026-09-29-gpu/parse-args-p.log`）。
- 3.3：GPU 验收通过（`evidence/2026-09-29-gpu/compare-strict.txt`）。
- 3.2：GPU 数据全部符合三条验证（`evidence/2026-09-29-gpu/compare-dec.txt`），但 benchmark 只能跑 4 个本地轮（预算 = global_rounds），没有复现 head 模式 run-until-stop 超过 `global_rounds × optimizer_steps` 的场景，因此未勾选。

## GPU 运行
- Modal app `lrfix-gpu`，2 个 sandbox，各 2×L40S（开跑前断言 GPU 型号），ports 1817 s、legacy 941 s，估算约 $4.5。结束后 sandbox 已 terminate、app 已 stop（`evidence/2026-09-29-gpu/modal_app_list_after_stop.txt`）。
- 驱动脚本：`evidence/2026-09-29-gpu/drive.sh`、`sbx.py`（镜像定义与 rl-engine-ports eq63 相同）。

## 下一步
1. 3.2：用 head 模式按 `yeto-hp929d` 配置（`--total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2`，`--inner-lr 1e-5`）分别以 `--rl-engine legacy` 与 `ports` 各跑一次（需要可做 head 的云，如 Nebius 2×H100），检查每岛 12 个本地轮的 `applied_lrs` 全为 1e-5 且两路径逐位一致、最终 cut 前 `global_delta_norm` 全非 0、两岛最终 hash 一致。
2. 3.2 通过后跑 `openspec validate --strict` 并准备归档。
