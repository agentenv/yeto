# Design

## Context

动机见 `proposal.md`。当前机制：

- head 模式下 `yeto launch` 用本机 sky 开 head，把 `head_cluster`、`head_job_id`、`clusters`（head + 按 `learner_cluster_names` 预先算好的 learner 名）写进 run 元数据；head 上的控制器 job 再用 **head 自己的 sky** 开 learner。两套 sky 状态库互不相通。
- `yeto down`（`yeto/cli.py` 的 `_down_one`）按元数据里的名字在本机并行 `sky down`，异常一律 `except` 后打印 `teardown failed` 继续，随后 `runs.update_run` 并打印 `run is down`、返回 0。对 head 模式的 learner，本机 sky 必然回 "does not exist"。
- head 自己在 run 结束时通过 `launcher.terminate_and_verify` 拆岛，并用 `_cloud_live_instances_probe` 向云核对（sky 的 `provision.query_instances`，云无关）。这条路径只在 head job 正常走到结尾时执行；`yeto down` 不用它。
- `prB/head-side-teardown` 已实现：取消控制器 job → ssh 到 head 用其 sky 逐个 down → 按输出行确认 → 未确认则保留 head 并返回 1，重试 3 次、间隔 20 秒。它与 `pr8/rl-island-fixes` 在 `_down_one` 前面那段（Modal app 停止分支）冲突。

## Goals / Non-Goals

**Goals:**
- head 模式 `yeto down` 的顺序固定为：取消控制器 job → 经 head 拆 learner 并确认 → 删 head → 云端核对 → 才报成功。
- 把 prB 落到当前栈上，而不是再写一份。
- 云端核对复用 `terminate_and_verify` 的探针，不新增云 SDK 依赖。

**Non-Goals:**
- 不改 head 自身收尾路径（已有核验）。
- 不处理"元数据里根本没记录的 cluster"（例如 head 因 bug 用了别的名字）；那属于开岛侧的契约。
- 不做跨 run 的"清扫所有 yeto-* 实例"命令。

## Decisions

### D1：合并 prB，并把 Modal 分支与 head 分支按"先分类、再分派"重排

冲突的两段其实互不相干：pr8 那段把 Modal 岛从 sky cluster 列表里挑出来交给 `_modal_stop_app`，prB 那段把 head 模式的 learner 交给 head。解法是在 `clusters` 上先做三分类——`modal_names`（按名字后缀）、`on_head`（`controller == "head"` 且不是 head 本身、也不是 Modal）、其余本机 sky 直接 down——再各走各的。Modal 岛不需要经 head：它是 head 上 Modal app 的函数调用，本机 `modal app stop` 就能全部结束（prB 写在 pr8 之前，当时没有 Modal 岛）。

*替代方案——放弃 prB、在 head 上加一个"收到信号就自拆"的 job*：需要 head 在 `yeto down` 时还活着并能调度新 job，而 prB 的注释已记录过"队列里的 sky job 没有输出也没有保证"；ssh 同步执行并解析确认行更可靠。否决。

### D2：本机 "does not exist" 不再算成功

`_down_one` 的 `except` 保留，但语义改为"记录为未确认"；只有 learner 已在 head 侧确认（D1 的结果集合）时，本机的失败才可以忽略。这样 `--controller local` 的 run 行为不变（本机 sky 认识自己的 cluster），head 模式则不再产生假成功。

### D3：云端核对复用 `terminate_and_verify`，从"head 收尾"扩展到"本机 `yeto down`"

`_cloud_live_instances_probe(cluster)` 依赖本机 sky 状态库里的 cluster 记录来构造查询，所以：
- head cluster：本机有记录，直接 `terminate_and_verify(sky, head_cluster)` 替换现在的裸 `sky.down`。
- head 模式的 learner：本机没有记录，探针只能在 head 上构造。让 prB 的 `HEAD_DOWN_SCRIPT` 在 head 上调用 `terminate_and_verify` 而不是裸 `sky.down`，确认行由它的返回值决定；这样 learner 的云端核对发生在 head 被删之前，正好是唯一还能做的时机。
- Modal learner：`modal app list`（或 SDK）核对 app 为 stopped 且 tasks 为 0，已有 `_modal_stop_app` 附近的调用可复用。
- 探针返回 `None`（云不支持或构造失败）时打印"未核验、信任 sky"，不算失败——与 spec 的"云端无法核验"场景一致。

*替代方案——直接调各云 CLI（`nebius compute instance list` 等）按名字前缀找残留*：能覆盖"元数据没记录的 cluster"，但要为每个云写一份、依赖本机装了各家 CLI；sky 的 provision 查询已经按云分派。作为后续可选项记录，本 change 不做。

### D4：退出码即结论

新增的每一条失败路径都返回非零，并且 `run is down` 只在最后打印。`runs.update_run` 在部分失败时把 run 标成"teardown incomplete"而不是"down"，以便 `yeto status` 能看出来、重跑 `yeto down` 能续做。

## Risks / Trade-offs

- **head 已经不在了（被人手工删、或之前版本的 `yeto down` 删掉的）** → 经 head 的路径必然失败，命令非零退出并列出 learner 名；这正是希望暴露的情况，文档给出 `nebius compute instance list` 的手工兜底。
- **ssh 到 head 依赖 sky 生成的 ssh 配置** → prB 已用 `BatchMode`/`StrictHostKeyChecking=no`；head 若刚被 `sky down` 到一半，重试逻辑覆盖。
- **云端核对增加 `yeto down` 时长** → 每个 cluster 最多几次查询，秒级；相比一台 H100 每小时几美元可以接受。
- **`terminate_and_verify` 在探针出错时信任 `sky down`** → 与现状相同，但现在会打印出来。

## Open Questions

- 是否把"清扫所有 `<prefix>-*` 实例"作为 `yeto down --sweep` 的兜底加进来（D3 的替代方案）？不影响本 change 的 spec 与任务拆分，可以另立 change。
