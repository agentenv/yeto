# A4 续跑（会话 3）结果，代码 9a06c4b（= origin/integ-decl），Nebius 8×H100 on-demand，2026-09-30

计划：`gpu-plan-v2.md` §9.20（运行前提交）。工具：`tools/`（脚本本机运行、此处留档）。本批已花 ≤$85.44（台账 `infra-drafts/gpu-spend.md`：运行 $44.58 + 孤儿实例事故 ≤$40.86）。

## 步骤 1 指纹
- 本机用与真机同一 CLI→learner→`build_ports_launch` 路径重建 Miles argv，先在 47efd25 上复现真机值 `sha256:2d0a00f4…`（argv 逐项相同），再在 9a06c4b 上取值：12 轮 = `2d0a00f4…`（argv 未变）；`--total-steps` 进入 argv（`--num-rollout`/`--lr-decay-iters`），3/4/5 轮分别为 `1c4ceeb1…`/`210f27dc…`/`c72f80da…`（`tools/attestation-*.json`）。真机 `rl_driver_start.runtime_fingerprint` 在 E1-B（12 轮）与 watchdog（4 轮）两次运行中与本机值一致（selfcheck 比对）。

## 步骤 2 E1-A：未上卡（待主 agent 裁定）
§9.18 口径下第 8 轮只能取 v7 的 `rl_publication`（4 成员），down 事务没有任何成员发布事件 → (c) 必然未通过（代码路径 `controller._emit_member_publication` 只在 up/REBUILD_OLD 调用）。需裁定：给 down 补成员限定事件，或改口径。裁定前重跑只会重复失败（约 $25）。

## 步骤 3 E1-B：**未通过——yeto 代码缺陷（停）**
- 三次启动：前两次在 selfcheck 因探针工具问题停止（解释器无 ray；`ray.init("auto")` 连到 SkyPilot 自带 Ray 2.9.3 而非 Miles Ray），已修（`tools/probe_remote.sh`、`fork_probe.py`）并在 watchdog 运行中验证探针可用；第三次（`a4e1b-…-5`）跑到底。
- 注入已执行（launch.log：`TEST INJECTION YETO_RL_TEST_INJECT_LORA_PERTURB: member update of [...00002, ...00003] ships a LoRA adapter perturbed by 0.01`）；事务：VERIFYING → REBUILD_OLD → REBUILT_OLD，**错误原因 = `PublicationError: member update_weights failed: This event loop is already running`**，不是 `check_weights` 拒绝。随后 driver 以 `Fatal async misuse, aborting: coroutine 'RayWorkerHandle.__getattr__.<locals>.call' was never awaited` 中止（17:50:15，rollout 2 的 sync 阶段）。
- 缺陷定位（静态）：`publish.py::_publish_members`（async）在事件循环里同步调用 `perturb_trainer`（`entry.py::lora_perturber` → `driver.policy_state.apply`），后者经 RayWorkerHandle 需要自己的事件循环/await；`finally` 里的 `perturb_trainer(None)` 同样。属于测试注入钩子的同步/异步接线错误（应放到线程/executor 或改为 await）。**此运行不能证明 check_weights 拒绝路径**。判据 (b) 终态 REBUILT_OLD 表面满足但原因不对，不计通过；(c)(d) 探针在 driver 崩溃后集群仍在但 learner 已重启，返回的是重启后的空状态，不作为证据。
- (a) 旁证：router 采样 1170 条（463 条为 driver 崩溃后的连接错误），up 窗口内新 cell 从未出现在 router 的 inflight/cordoned 里（只有 2 个旧 worker，cordoned 始终空）——即新 cell 在 admit 前未注册到 router（比"在 cordon 列表中"更强，字面与判据 (a) 不同，需裁定口径）。
- 费用 ≤$12.96（17:29:37–17:54:52）。

## 步骤 4 watchdog：**判据 1 未通过（待裁定）**
- selfcheck 通过（router 发现、指纹、探针）。up 于 17:22:11 提交、deadline 120 s；watchdog 在 +120 s 触发时 `start_cells` 仍在进行（Nebius 上 ≈143 s），`watchdog_action.killed = []`（新 cell 进程尚不存在）；事务在 start 完成后 REBUILD_OLD → REBUILT_OLD（watchdog→终态 27.8 s），成员与 config epoch 回到旧值，4 轮在 17:27:48 自然结束。
- 按 plan-3.8-4.4-v2 §4：(1) 要求 `killed` 列出新 cell 的 worker → **未满足**；(2) 终态 REBUILT_OLD 且 ≤60 s 满足，但不是由被阻塞的 update_weights 被打断所致；(3)(4)(5) 的事后采样（gpu_samples、fork 探针）因岛在事务后约 3 分钟自然退出、集群随之销毁而未取到。
- 原因：deadline 120 s 短于该平台的 engine 启动时间，阻塞注入（在 update_weights 前）从未被执行到。重跑需把 deadline 设为 ≥ 启动时间 + margin（如 240 s），属于改实验参数，需主 agent 裁定；后续 after-hook 已改为在事务终态后立即探针。
- 费用 ≤$12.21（17:04:01–17:27:48）。

## 步骤 5、6：未运行
E1-D ①–⑦ 与 E1-C/A4b 未跑（停在代码缺陷/待裁定处；预算未到 $90 上限）。D 的运行安排、注入器（`dkill.py`、`dctl.py`）已写好留档，计划见 §9.20。

## 事故：孤儿实例（≤$40.86）与核验口径
首次 E1-B 启动（`a4e1b-…-3`）在 selfcheck 失败后由 nstop 执行 `sky down`；launcher 的 `yeto.cli _worker` 子进程随后重新开通了同名集群（实例创建 16:40:34），我只杀了 `_worker` 而没有复核，实例一直 RUNNING 到 18:00 收尾核对才被发现并 `sky down`（约 79.6 min，≤$40.86）。此前 nstop/台账里的 “nebius instance list = []” 用的是 CLI 默认 project（project-e07…），而 sky 的 eu-north1 project 是 `project-e00eqrj3pr00622zrgdeyc`，所以核验一直为空，口径无效（此前各轮同样口径的“无残留”证明需重新核实）。已修：nstop 先杀 `_worker`，并按正确 project 列实例。
