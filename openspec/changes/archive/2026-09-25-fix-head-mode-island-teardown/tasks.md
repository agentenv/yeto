# Tasks

## 1. 把 prB 落到当前栈上

- [x] 1.1 在 pr8 之上合并 `prB/head-side-teardown`，按 design D1 解决 `yeto/cli.py` 的冲突：先把 `clusters` 三分类（Modal / 经 head 的 learner / 本机直接 down），再各自分派。验证：`tests/test_head_mode.py` 中 prB 带来的两个测试与 pr8 的 Modal 停止测试同时通过
- [x] 1.2 新增测试：head 模式 run 含一个 Modal 岛和一个 sky 岛时，Modal 岛走 app 停止、sky 岛经 head 拆除，且两者都不经过本机 `sky down`。验证：测试通过，且断言本机 `sky.down` 从未被以 learner 名调用

## 2. 本机失败不再静默

- [x] 2.1 改 `_down_one`：本机 `sky down` 异常记为"未确认"，仅当该 cluster 已在 head 侧确认时忽略。验证：新增测试——head 模式下 learner 未在 head 确认、本机又报 "does not exist" 时，命令非零退出、head 未被删除、错误信息列出该 learner
- [x] 2.2 确认 `--controller local` 行为不变。验证：既有 local 模式的 `yeto down` 测试全部通过，无需改动（一处例外按 spec 改了：`test_down_survives_sky_errors` 原本把任意 down 异常当成功；现在 "does not exist" 仍算已消失、其他异常为未确认并非零退出，见 `test_down_no_longer_claims_success_on_other_sky_errors`）

## 3. 云端核对

- [x] 3.1 head cluster 的拆除改用 `terminate_and_verify`（含云端探针）替换裸 `sky.down`。验证：测试用假探针模拟"仍有实例存活"→ 重试后仍存活 → 非零退出并打印实例 id；"探针为 None"→ 打印未核验、零退出
- [x] 3.2 把 prB 的 `HEAD_DOWN_SCRIPT` 改为在 head 上调用 `terminate_and_verify`，确认行反映其返回值。验证：`_unconfirmed_head_downs` 的解析测试覆盖"down 但云端仍存活"输出为未确认
- [x] 3.3 Modal learner 在 app 停止后核对 app 状态为 stopped 且 tasks 为 0，否则非零退出。验证：测试用假的 app 状态覆盖两种结果
- [x] 3.4 只有全部确认后才打印 `run '<name>' is down` 并返回 0；部分失败时 `runs.update_run` 记录为 teardown 未完成，`yeto status` 能显示。验证：测试断言部分失败时不出现成功信息且状态可见

## 4. 文档与真机确认

- [x] 4.1 更新 `docs/CLOUDS.md`：把"`yeto down` 后手工看 `nebius compute instance list`"改为说明新行为与非零退出的含义；在 live-run-failures 第 36 条下记录处理。验证：文档改动与实现一致
- [x] 4.2 （2026-09-25 `yeto-td3` 完成：Nebius 岛经 head 确认 down、head 云端核验重试至空、退出 0、无残留。此前 2026-09-24 部分完成，见 CLOUDS.md：`yeto-td2` 用 Modal 岛做了运行中 `yeto down`，Modal app 确认 + head 云端核验通过；`yeto-td1` 的 Nebius 岛因租户公网 IPv4 配额（上限 3，被另一 run 占用）没开出来，只验证了经 head 的“learner 不存在”确认路径。Nebius learner 的经 head 拆除待配额空出后补做）真机：在 Nebius 开一个 head + Nebius 1 卡 SFT 岛（8.1 的配置，避开第 37 条的 docker 岛问题），岛运行中执行 `yeto down`，确认输出含每个 learner 的确认行、head 最后删除，`nebius compute instance list` 与 `sky status` 无该前缀残留。验证：结果记入 `docs/CLOUDS.md`；**不要动不属于本次运行的实例**
- [x] 4.3 （2026-09-25 `yeto-td4` 完成：手工删 head 后 `yeto down` 非零退出、列出 `yeto-td4-l0-eu-north1`、run 记 `TEARDOWN_INCOMPLETE`；岛随后手工删除。）真机反例：先手工删掉 head 再执行 `yeto down`，确认命令非零退出并列出未确认的 learner，随后手工删除该岛。验证：记录输出；同样只动本次运行的实例
