# Tasks

## 1. 用测试钉住两处行为

- [x] 1.1 在 `tests/test_rl_launcher.py` 为 `make_miles_island_task` 的 run 脚本加断言：不含整机 `ray stop`（无论带不带 `--force`）；`ray start --head` 和 worker 的 `ray start --address` 都带 `--temp-dir`；learner 启动行带 `RAY_ADDRESS="$MASTER_ADDR:6379"`；启动前和 EXIT trap 的清理都只按 temp 目录匹配。验证：修复前这些断言失败，修复后通过
- [x] 1.2 在 `tests/test_controller.py` 的 fake ops 上加 `job_alive`，新增测试：`job_status` 抛错但 `job_alive` 为 True 时控制器不进入恢复、不调用 relaunch；`job_alive` 为 False 时按原逻辑恢复；`cluster_up` 为 False 但 `job_alive` 为 True 时同样不恢复。验证：修复前"不恢复"的两条失败，修复后全部通过，既有恢复/超时测试不变

## 2. 岛脚本

- [x] 2.1 改 `yeto/launcher.py` `make_miles_island_task` 的 run 脚本：定义 `MILES_RAY_DIR=$HOME/miles-ray` 与只清该目录进程的 `stop_miles_ray`；开头和 EXIT trap 调用它替代 `ray stop --force`；head/worker 的 `ray start` 都加 `--temp-dir="$MILES_RAY_DIR"`；learner 启动行前置 `RAY_ADDRESS="$MASTER_ADDR:6379"`。验证：任务 1.1 通过；`git grep "ray stop" yeto/launcher.py` 只剩注释或为空
- [x] 2.2 核对 `yeto/rl/ssh_harness.py` 是否与上述脚本共用生成代码。验证：共用则一起改并让 `tests/test_rl_ssh_harness.py` 通过；独立则在此记录"未改，裸机无 sky Ray"（核对结果：`ssh_harness.py:3512` 是独立拼接的字符串，与 `make_miles_island_task` 不共用；它跑在用户裸机上，没有 sky Ray，**未改**）

## 3. 控制器恢复判定

- [x] 3.1 给 sky 的 ops 实现加 `job_alive(cluster, job_id)`：`sky.get(sky.queue(cluster))` 中找到该 job 且状态为 PENDING/SETTING_UP/RUNNING 之一返回 True；找不到、已结束或查询抛错返回 False。Modal 的 ops 给等价实现（函数仍在运行即 True）。验证：单元测试用 fake queue 记录覆盖三种返回（`SkySDKOps.job_alive`；Modal 的 ops 不在 main 上（pr6 分支），等价实现放在 live 分支合并时补，`FleetController._job_alive` 对没有该方法的 ops 回退为 False 即旧行为）
- [x] 3.2 改 `FleetController._probe`：`job_status` 抛错或 `cluster_up` 为 False 时先问 `job_alive`，True 则返回健康。验证：任务 1.2 通过；`_enter_recovering` 的既有日志与超时语义不变

## 4. 回归与真机

- [x] 4.1 全量 `pytest tests/`，与改动前基线对比无新增失败（基线已知有 48 个既有失败：36 个缺 `cargo`、2 个缺 `megatron`）。验证：失败集合一致（2026-09-24：改动前后同为 48 个失败/错误，集合一致，无新增）
- [x] 4.2 Modal 单卡回归：用 `docs/CLOUDS.md` 里 `yeto-gh4` 的命令再跑一次（可把 `--total-steps` 降到 2）。验证：job SUCCEEDED、`train/grad_norm` 非零、syncer 收到 2 步；`yeto down` 后 Modal app stopped（2026-09-24 `yeto-gh6`，Modal 1×H100，2 步：job SUCCEEDED，`train/grad_norm` 0.02795/0.02176 与 gh4 逐位相同，syncer 收到 2 步 gnorm 0.0117/0.0045；`yeto down` 后 Modal 无 running app）
- [x] 4.3 Nebius 1×H100 岛：用 live-run-failures 第 37 条里 `yeto-gh2` 的命令重跑（`--gpu nebius:1xh100@eu-north1`）。验证：head 日志里岛的 setup 只出现一次、job 号始终为 1、无 `recovered: relaunched`；在岛上 `pgrep -f miles-ray` 非空且 sky 的 Ray（`/tmp/ray`）仍在；训练 `train/grad_norm` 非零；结束后 `pgrep -f miles-ray` 为空。结果记入 `docs/CLOUDS.md`（第一条 sky 云 RL 岛记录），并在 live-run-failures 第 37 条下补记已修（2026-09-24 两次 Nebius 1×H100 岛：`yeto-gh5`（TERM 版清理，3 步）和 `yeto-gh7`（最终 TERM→10 s→KILL 版，2 步）。两次都是 setup 只跑 1 次、job 号始终为 1、`recovered: relaunched` 0 次、`status: INIT` 0 次、集群全程 UP；岛上同时活着 sky 的 Ray（`/tmp/ray_skypilot`，8 个进程）和 Miles 的 Ray（`/root/miles-ray`），learner 正常训练：gh5 `train/grad_norm` 0.02795/0.02176/0.02048、tape gnorm 0.0117/0.0059/0.0025；gh7 gnorm 0.0117/0.0045。**退出后的 `pgrep -f miles-ray` 为空这一项在真机上没能直接观测**：head 在 job SUCCEEDED 后几秒内就拆岛，ssh 窗口来不及；gh5 上 learner 刚退出时看到 TERM 版清理还剩 4 个 Ray 进程（gcs_server/raylet/monitor 对 SIGTERM 不响应 30 s+），据此把清理改成 TERM→KILL，并在 Miles 镜像里复现验证：旁边 8 进程的模拟 sky Ray 不受影响且 `ray status` 健康，Miles Ray 11 s 内清零。sky 只有在 run 脚本（含 EXIT trap）跑完后才报 SUCCEEDED，而 SIGKILL 不可忽略，因此 gh7 的 SUCCEEDED 本身即说明清理已执行完毕。结果记入 `docs/CLOUDS.md` 与 live-run-failures 第 37 条）
- [x] 4.4 收尾核验：`yeto down <prefix>` 后再看 `nebius compute instance list`，head 开出的岛 VM 若残留则手工删除（第 36 条的已知缺口）。验证：无本次前缀的实例；`modal app list` 无 running app（gh5/gh6/gh7 各自 `yeto down` 后：`nebius compute instance list` 无 `yeto-gh*` 实例——这三次 head 都先于本机 `yeto down` 自行拆掉了岛，没再出现第 36 条的孤儿 VM；`modal app list` 无 running app）
