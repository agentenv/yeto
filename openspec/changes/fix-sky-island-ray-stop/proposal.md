# Proposal

## Why

在 SkyPilot 云（Nebius、AWS、RunPod）上开 Miles RL 岛时，岛的 run 脚本第一句 `ray stop --force` 会把 SkyPilot 装在同一节点上的运行时 Ray 一起杀掉；此后 sky 的每次状态刷新都把集群标成 `INIT`，head 上的 launcher 把 `INIT` 当作"岛失联"立即重新 `sky launch`，setup 从头再跑、learner 被杀、脚本再次 `ray stop --force`，约 90 秒一圈无限循环，训练永远开始不了（2026-09-24 `yeto-gh2`，live-run-failures 第 37 条）。这行脚本自 `0f0f2c7`（2026-07-30）接入 Miles RL 岛起就在主分支上，但此前所有 RL 岛都跑在 Modal 容器或 SSH 裸机上，节点上没有别的 Ray，所以从未暴露；只要走 sky 云，它就必现。

## What Changes

- **岛脚本只管理自己的 Ray**。Miles 的 Ray 集群用独立的 temp 目录启动，清场和退出清理只针对该目录下的进程，不再执行会杀掉整机 Ray 的 `ray stop --force`；learner 通过显式 `RAY_ADDRESS` 连接自己的集群，而不是靠 `/tmp/ray/ray_current_cluster` 自动发现（否则会误连 sky 的 Ray）。
- **launcher 的恢复判定不再把 `INIT` 一律当失联**。`sky.job_status` 因集群状态不是 `UP` 而拒答时，先查该集群的 job 队列：job 仍在 `RUNNING`/`SETTING_UP`/`PENDING` 就视为健康，继续等待；只有 job 真的不在了或集群确实消失才进入恢复。
- **补回归测试**：断言生成的岛脚本不含整机 `ray stop`、含 temp-dir 与 `RAY_ADDRESS`；断言控制器在"job 状态拒答但 job 仍在运行"时不触发 relaunch。
- 在 `docs/CLOUDS.md` 记录一次 sky 云 RL 岛的成功运行（此前从未有过）。

## Capabilities

### New Capabilities
- `sky-rl-island-runtime`：在 SkyPilot 集群上运行的 RL 岛必须与 SkyPilot 自身的运行时共存——岛的进程管理不得破坏 SkyPilot 对该集群的健康判定；head 侧的岛监督在集群健康状态暂时不可读时，必须以 job 的真实状态为准，不得对仍在运行的 job 重新发起 launch。

### Modified Capabilities
<!-- 无。`openspec/specs/` 为空；在途的 add-nebius-verda-modal-clouds 和 fix-rl-lora-grad-hook
     尚未归档，它们的 spec 里没有对岛脚本进程管理或控制器恢复判定的既有要求。 -->

## Impact

- `yeto/launcher.py`：`make_miles_island_task` 的 run 脚本（`ray stop --force` 两处、`ray start` 两处、learner 启动环境）；`FleetController._probe` 与 `SkyOps.job_status`/`cluster_up`（或其等价实现）的失败分类。
- `yeto/rl/ssh_harness.py`：SSH 验收 harness 生成的脚本用同一套 `ray start --head`/`ray stop` 习惯，需要核对是否共用逻辑；裸机上没有 sky Ray，行为上不受影响，但若共用代码要一起改。
- `tests/test_rl_launcher.py`、`tests/test_controller.py`、`tests/test_rl_ssh_harness.py`：断言岛脚本内容与控制器恢复行为的既有测试。
- Modal 岛（`modal_runner` 执行同一份 run 脚本）：容器内无 sky Ray，改动对它是无害的等价替换，但必须回归验证。
- 不改 Miles、不改 syncer、不改 SFT 岛。

## Non-goals

- 不改 Miles 自身 `scripts/run-*.sh` 里的 `ray stop --force`（那是裸机脚本，不经过 Yeto）。
- 不改 SkyPilot 的健康检查语义。
- 不处理 head 侧 `yeto down` 看不见 head 开出的岛这一问题（live-run-failures 第 36 条），另立 change。
