# Design

## Context

动机见 `proposal.md` 的 Why。决定方案的事实：

- 岛脚本（`yeto/launcher.py` `make_miles_island_task` 的 `run`）当前是：
  `ray stop --force` → rank 0 `ray start --head --port=6379` 并 `trap 'ray stop --force' EXIT` →
  `python3 -m yeto.rl.learner`；其他 rank `ray start --address=$MASTER_ADDR:6379` 后轮询 `ray status`。
  `ray stop` 没有"只停某个集群"的选项，它按进程名扫整机；`--force` 只是改成 SIGKILL。
- SkyPilot 在它开的每个节点上跑一份自己的 Ray（`~/skypilot-runtime`，端口 6380，temp 目录
  `/tmp/ray`），skylet 和健康检查都靠它。`backend_utils._update_cluster_status` 的规则是
  "云端实例 RUNNING 且 `ray status` 成功"才标 `UP`，否则标 `INIT`；`sky.job_status` 对非 `UP`
  集群直接抛错（`Getting job status: skipped for cluster ... (status: INIT)`）。
- head 上 `FleetController._probe` 的顺序是 `ops.job_status` → 异常即返回失败原因 →
  `_enter_recovering` → `_drive_recovery` 重新 `sky.launch` 同名集群。sky 对已存在集群的
  `launch` 会重新 provision（重启 sky Ray、标 UP）并重跑 setup、提交新 job。
- Miles 用 `ray.init(address="auto")` 连集群（`miles/utils/misc.py`）。Ray 2.57 的解析顺序是
  `RAY_ADDRESS` 环境变量优先；否则读 `<temp_dir>/ray_current_cluster`，默认 temp 目录是
  `/tmp/ray`——也就是 sky 的 Ray 写地址的地方。
- Modal 岛通过 `modal_runner` 在容器里执行同一份 run 脚本，容器内没有 sky Ray。
- `sky.queue(cluster_name)` 返回该集群的 job 列表，每项含 `job_id` 和 `status`
  （`PENDING`/`SETTING_UP`/`RUNNING`/`SUCCEEDED`/`FAILED*`/`CANCELLED`），它不经过
  `check_cluster_available` 的 `UP` 门槛。

## Goals / Non-Goals

**Goals:**

- sky 云上的 RL 岛：setup 只跑一次，job 号保持为 1，集群全程 `UP`，训练正常进行。
- Modal 岛和 SSH harness 行为不变。
- 控制器对"状态暂时不可读但 job 仍在跑"不再误判，且不削弱对真实失联的恢复。

**Non-Goals:**

- 不改 SkyPilot 的健康检查，不给 sky 的 Ray 换端口或 temp 目录。
- 不做通用的"任意进程清理"框架；只解决 Ray 这一处。

## Decisions

### D1：给 Miles 的 Ray 独立 temp 目录，按目录清理，不再整机 `ray stop`

`ray start --head --port=6379 --temp-dir=$HOME/miles-ray`（worker 同样带 `--temp-dir`）。
Ray 用 `--temp-dir` 启动后，gcs、raylet、dashboard、worker 的命令行里都带着该目录
（session 路径、log 目录），因此 `pkill -f "$HOME/miles-ray/"` 只命中岛自己的进程；sky 的
Ray 用 `/tmp/ray`，不会匹配。脚本开头和 EXIT trap 都改成这个按目录的清理，保留"清掉上一次
残留"的原有目的。

*替代方案——直接删掉 `ray stop --force`*：sky 每次 relaunch 都是新 job，看起来不需要清场；
但 recovery 会在同一台 VM 上重新 `sky exec`，上一次的 Miles Ray 若没退干净，
`ray start --head` 会因 GCS 已在而失败。否决。

*替代方案——`ray stop` 不加 `--force`*：同样按进程名扫整机，只是换成 SIGTERM，一样杀掉
sky 的 Ray。否决。

*替代方案——按端口 `--port=6379` 匹配*：只有 gcs 进程的命令行带端口，raylet/worker 不带，
清不干净。否决。

### D2：learner 显式 `RAY_ADDRESS=$MASTER_ADDR:6379`

Miles 的 `ray.init(address="auto")` 在没有 `RAY_ADDRESS` 时读 `/tmp/ray/ray_current_cluster`。
现在 sky 的 Ray 不再被杀，那个文件里可能是 sky 的 6380 地址，learner 会连错集群、把训练 actor
调度到 sky 的 Ray 上。给 learner 进程导出 `RAY_ADDRESS` 后，`auto` 优先用它。这也让 worker
rank 的 `ray start --address` 与 head 的地址来源一致。

*替代方案——改 Miles 传显式 address*：要动 Miles、走 bundle，改动面更大且本条 change 不改
Miles。否决。

### D3：`_probe` 在 `job_status` 抛错时用 `sky.queue` 二次确认

新增 `ops.job_alive(cluster, job_id)`：`sky.get(sky.queue(cluster))` 找到该 `job_id`，状态在
`{PENDING, SETTING_UP, RUNNING}` 内即返回 True。`_probe` 里 `job_status` 抛错时先调它：
True → 返回 `(None, None)`（健康，继续等）；False 或再次抛错 → 保留现有的
`"job status unavailable (...)"` 失败路径。`cluster_up` 那条分支（job 状态读到了但集群不
`UP`）同样先看 `job_alive`，避免 D1 之外的偶发 `INIT`（例如 sky 刷新时的网络抖动）触发
relaunch。

*替代方案——把 `INIT` 一律当健康*：会漏掉集群真的坏了但 job 记录未更新的情况。否决。

*替代方案——延长 `--recover-timeout`*：不解决问题，只是让循环跑得更久。否决。

### D4：SSH harness 只核对，不顺手重构

`yeto/rl/ssh_harness.py` 生成的脚本也有 `ray start --head`/`ray stop`，但它跑在用户自己的
裸机上，没有 sky Ray。若它与 `make_miles_island_task` 共用生成代码，则一起获得 D1/D2；若是
独立拼接的字符串，则保持原样并在 tasks 里记录，不在本 change 内统一。

## Risks / Trade-offs

- **`pkill -f` 按路径匹配可能漏掉不带 temp 目录参数的 Ray 子进程** → 用 `ray stop` 的进程
  名集合（gcs_server、raylet、dashboard、default_worker、log_monitor 等）核对：这些进程都由
  raylet 带 session 目录参数拉起。任务里要求在 sky 岛上实测退出后 `pgrep -f miles-ray` 为空
  且 sky 的 Ray 仍在。
- **`sky.queue` 本身也可能因集群不可达而抛错** → 这时按现有失败路径处理，行为不比现在更差。
- **`RAY_ADDRESS` 影响 learner 之外的子进程**（SGLang engine、reward 函数子进程）→ 它们本就
  应连岛的集群；但要在 Modal 上回归确认没有进程依赖 `auto` 去连别的东西。
- **改了控制器判定后，真实失联的恢复会慢一拍**（多一次 `sky.queue` 调用）→ 可接受，`_probe`
  本就按 poll 间隔跑。

## Migration Plan

1. 改岛脚本（D1、D2）与控制器（D3），补测试。
2. Modal 单卡回归（沿用 `yeto-gh4` 的命令）确认无回归。
3. Nebius 1×H100 岛跑通（沿用 `yeto-gh2` 的命令），记录到 `docs/CLOUDS.md`：这是第一条 sky
   云 RL 岛的成功记录。

回滚：revert 即可，没有数据或协议变化。

## Open Questions

- 无。
