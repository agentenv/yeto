# A4 / A4b 4 卡 L40S 批 结果（2026-10-01，代码 b19b781，计划 gpu-plan-v2 §9.22）

## 总结
本批在步骤 1（冒烟）因 **Nebius eu-north1 4×L40S 开通失败** 停止。依指示未换平台、未原样重跑。
**T_start 未测得，UP_DEADLINE_S 未填，watchdog 及其后各项均未运行。**

| 项 | 结论 |
|---|---|
| 冒烟（测量） | 未运行成功：开通失败，无结论（不产生验收结论） |
| E1-A（含基线） | 未运行（前置：冒烟/开通） |
| E1-B | 未运行（前置） |
| watchdog | 未运行（前置：T_start 未测得，deadline 未能按 §9.22 步骤 2 填写） |
| E1-D（d123/d4/d5/d6/d7） | 未运行（前置） |
| A4b（E1-C） | 未运行（前置） |

tasks.md 勾选状态未改动。

## 冒烟运行记录（infra-v2-b1-a4sm-20261001-1，runs/smoke-1/）
- 代码 b19b781（`git archive`，yeto_sha.txt）；配置：`a4go4.sh smoke`，3 轮，T1R1S2↔T1R3S0，硬超时 2400 s，watchdog 2520 s，`--no-island-relaunch --modal-retries 0`。
- 时间线（UTC）：01:43:42 启动；01:43:51 sky 创建实例 `infra-v2-b1-a4sm-20261001-1-l0-eu-north1-1057ec71-309a-head`（computeinstance-e00jyc0xhzc1h7bach，gpu-l40s-d_4gpu-128vcpu-768gb，$9.14/h）；创建被接受（说明配额本身未拒绝）；之后到 01:49:37 约 5.5 min 内实例状态始终为 `STOPPED, Reconciling: True`，从未到 RUNNING；sky 等待 3 轮（共 120 次内的重试）后判 `Failed to wait for instances ... to be ready`，终止实例（01:49:37），launcher 报 `1 learner cluster(s) failed to provision`，exit 1；01:50:11 selfcheck 记 `startup_failed: launch_ended_before_first_generate (rc=1)`。
- 判断：属于 L40S 4 卡分配（容量）问题——实例对象创建成功但 Nebius 未能把它调度到 GPU 节点。Nebius 配额接口只给名字不给数值，**无法区分配额与容量，未确认**；provision.log 无明确的 quota/ResourceExhausted 字样。属于"新的基础设施阻塞"，按规则停下。
- 费用 ≤ $0.92（实例存在 ~6 min，从未 RUNNING，上界）。本批已花 ≤ $0.92 / $45。
- 注入/判读：无（未起岛）。

## 清理证据
- 实例在开通失败时由 sky 自行终止；sky 的 SG 删除第 1、2 次因实例尚在而失败，之后该 SG 已不在 `vpc security-group list`（核对为 0 条匹配）。
- **脚本缺陷（已修）**：自动清理未闭环。selfcheck 在 startup_failed 时调用 `nstop.sh`→`cleanup_run.sh`，同时最终守卫（`a4go4.sh` 末尾）也在 20 s 后调用 `nstop.sh`→`cleanup_run.sh`；两个 cleanup 的进程扫描会匹配到对方（命令行含 `/<prefix>/`），互相 SIGTERM，结果 `cleanup_rc.txt` 缺失、cleanup.out 只有 phase 1 一行（selfcheck.txt 末尾 `Terminated`）。这是 §9.22 里 "cleanup_run.sh 闭环" 冒烟要查的点，冒烟因此查出该缺陷。
  修复（脚本小问题，仅一次）：`cleanup_run.sh` 入口加 `flock`（并发调用者串行化）；`selfcheck.sh` 的三处 startup_failed 路径在 nstop 后自己写 `cleanup_rc.txt`。CPU 桩测试 test_selfcheck.sh / test_cleanup.sh 仍 ALL PASS。该修复**未在真机验证**。
- 手动 `cleanup_run.sh infra-v2-b1-a4sm-20261001-1`：退出码 0，两次复查（间隔 60 s）干净（runs/smoke-1/cleanup_manual.out）。
- 收尾复查（Nebius `--parent-id project-e00eqrj3pr00622zrgdeyc` 实例列表 + `sky status`）：无本前缀实例；列表中仅有他人的 rlf-h200-pilot-03、rlf-h200-qualification-02、yeto-rl84f-head-2aebdd34-8456-head、cyberrl-verl-smoke-20260923（均 STOPPED，未触碰）；sky 无集群。

## 剩余阻塞与解除条件
1. Nebius eu-north1 4×L40S 当前无法开到 RUNNING。解除条件需用户/主 agent 决定：稍后重试同一平台（再花约 $1 的探测）、请求/确认 L40S 配额与容量、或批准其他平台（Modal L40S 路径需要新代码，§9.22 已说明）。
2. 恢复后的第一步仍是冒烟，测 T_start，填 §9.22 步骤 2，单独提交，再跑后续项。预算 $45 基本未动。
