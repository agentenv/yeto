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

## 审查修复（2026-09-29，fe2f50f 起）
- F1：`evidence/2026-09-29-gpu/` 下被 `.gitignore`（`*.log`）忽略的 16 个日志已用 `git add -f` 提交（先做了密钥扫描：只有 sglang/router 的 `api_key`/`password` 配置项，值都是 None 或 []）。镜像身份记录在 `evidence/2026-09-29-gpu/image-identity.txt`：sandbox 用 `radixark/miles@sha256:90940828…`，与 yeto `MILES_NEXT_IMAGE` 相同；sandbox 内 miles/sglang HEAD 与 `MILES_NEXT_COMMIT`/`SGLANG_NEXT_COMMIT` 相同。
- F2：legacy 的 extra argv 现在也拒绝 `LR_SCHEDULE_FLAGS`（`--lr-decay-style/--lr-decay-iters/--lr-warmup-iters/--min-lr`，支持 `--flag=value` 写法）；ports 的 `ADAPTER_OWNED_FLAGS` 改为引用同一常量。只共享 LR 调度这一组，没有把 ports 的放置/容错拒绝项整体搬到 legacy，因为 legacy 的 argv 由另一套代码构建，那些项可能是它的合法覆盖。测试见 `tests/test_rl_applied_lr.py`。
- F3：`--inner-lr <= 0` 在 launcher（`_prepare_rl_args`）和 learner `parse_args`（仅评估的运行除外）启动前就被拒绝。测试见 `tests/test_rl_applied_lr.py`。
- F4：coordinator 的消息里没有给出 F4 的具体内容，这里未处理，需要补充。
- benchmark：新增 `--learner-budget-steps`（仅限 `--arms decoupled`，且必须 >= `--global-rounds`），测试见 `tests/test_rl_benchmark.py`。budget=8 的补充 GPU 运行被费用上限终止，没有产出数据（`evidence/2026-09-29-gpu-budget8-aborted/`）。
- 累计 Modal 费用估算：第一批约 $4.4；第二批两个 sandbox 各存活约 1900 s，legacy 实际占用了 GPU（约 $2.3），ports 没能执行任何命令，是否计费不确定（最多约 $2.3）。合计约 $6.7–9.0。

## 下一步
1. 3.2：用 head 模式按 `yeto-hp929d` 配置（`--total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2`，`--inner-lr 1e-5`）分别以 `--rl-engine legacy` 与 `ports` 各跑一次（需要可做 head 的云，如 Nebius 2×H100），检查每岛 12 个本地轮的 `applied_lrs` 全为 1e-5 且两路径逐位一致、最终 cut 前 `global_delta_norm` 全非 0、两岛最终 hash 一致。
2. 如需非 head 的补充证据：`--learner-budget-steps 8` 在 legacy/ports 上各约需 25 分钟 2×L40S（约 $2/次），并且应在 GPU 空闲时段运行，避免排队。
3. 3.2 通过后跑 `openspec validate --strict` 并准备归档。
