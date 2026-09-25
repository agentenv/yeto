# Proposal

## Why

head 模式下的 `yeto down <prefix>` 会把 head 删掉、却把 head 开出来的 learner 岛留在云上继续计费：岛只记在 head 的 sky 里，本机 `sky down` 得到 "does not exist" 后被当成"已经没了"吞掉，退出码 0，之后再没有任何 sky 认识这台 VM。2026-09-24 回收 `yeto-gh1`、`yeto-gh2` 时各泄漏一台 Nebius H100（live-run-failures 第 36 条），2026-09-23 的 8.4 运行也发生过一次；每次都靠人去 `nebius compute instance list` 里找。

## What Changes

- **head 模式下先经 head 拆岛，再拆 head。** `yeto down` 对 `controller == "head"` 的 run，先取消 head 上的控制器 job（防止它在拆的过程中重新拉起岛），再在 head 上用 head 的 sky 逐个 down learner，逐个确认；有任何一个未确认就**保留 head**、非零退出并说明原因，而不是把 head 删掉。这部分沿用已有的 `prB/head-side-teardown` 分支，本 change 负责把它落到当前 PR 栈上（它与 `pr8/rl-island-fixes` 在同一段代码冲突）。
- **本机拆岛失败不再静默。** 本机 `sky down` 对一个本机不认识的 learner 返回 "does not exist" 时，只有在该 learner 已经从 head 上确认拆除的情况下才算成功；否则记为未确认，走上一条的保留 head 路径。
- **拆完之后向云查一遍。** `yeto down` 结束前，用 sky 的 provision 查询按 cluster 名到云上核对没有存活实例（`terminate_and_verify` 已经在 head 收尾时做这件事，`yeto down` 没做）；查到残留就重试 down，仍在则非零退出并打印实例 id，让人能立刻删。Modal 岛按 app 状态核对（app stopped 且 tasks 为 0）。
- **不再给出"run is down"的假成功。** 只有所有 cluster 都确认拆除、云端核对为空时，才打印 `run '<name>' is down` 并返回 0。

## Capabilities

### New Capabilities
- `head-run-teardown`：head 模式 run 的回收顺序、确认与云端核验——learner 必须从能看见它的那台 sky 拆除并逐个确认，head 只有在所有 learner 确认后才可删除，回收结束前必须向云核对无残留，任何未确认都以显式失败告终。

### Modified Capabilities
<!-- 无。openspec/specs/ 当前为空，没有已归档的既有能力可改。 -->

## Impact

- `yeto/cli.py` 的 `down` 命令（当前第 1696-1783 行附近）：head 模式分支、Modal 分支与新的云端核验收尾。
- `yeto/launcher.py`：`_cloud_live_instances_probe` / `terminate_and_verify` 需要能在本机对 head cluster 使用，并（经 head）对 learner 使用。
- `prB/head-side-teardown` 分支：合并并解决与 pr8 在 `yeto/cli.py` 的冲突；其测试 `tests/test_head_mode.py` 随之进入栈。
- `docs/CLOUDS.md`：把"`yeto down` 后必须看 `nebius compute instance list`"的手工步骤改成对新行为的说明。
- 不影响 `--controller local` 的 run（岛由本机 sky 开，本机 `sky down` 本来就有效），也不影响 head 自己在运行结束时的收尾（那条路径已有 `terminate_and_verify`）。

## Non-goals

- 不解决 Nebius docker RL 岛停在 `INIT` 的问题（live-run-failures 第 37 条），那是开岛侧的 bug。
- 不引入各云的专用 API 客户端；云端核验只用 sky 已经提供的 provision 查询，覆盖不到的云按现状信任 `sky down` 并明确打印"未核验"。
