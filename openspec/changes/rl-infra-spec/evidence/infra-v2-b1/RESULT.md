# GPU 验收第 1 批（GPU-B1）结果，2026-09-30

计划与判据：`gpu-plan-v2.md` §9（e15af0d 提交，§9.5/§9.6 为运行前就绪复核，判据未改）。台账：`infra-drafts/gpu-spend.md`。
代码 SHA：Nebius 冒烟与 F0 用 a303cbb；F-E1 指纹运行用 05822ff；F-E1 用 724fc7b（= 合并 integ-decl a5123ca）。镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:17d428a2…0bcef`（miles 5c1b49eb，sglang 9f29303b）。

## 逐项结果

| 项 | 结果 | 证据 |
|---|---|---|
| Nebius 路径冒烟 | **不通**（未开通任何 VM，$0）。launcher 经 sky 0.13 提交岛任务时客户端报 `asdict() should be called on dataclass instances`（推测为 `resources["_docker_login_config"] = DockerLoginConfig` 私有镜像登录路径与 sky 0.13 客户端序列化不兼容，launcher.py `_sky_docker_login_config`）；另 launcher 以 spot 方式提交 Nebius。按规则不在 Nebius 上排错，退回 Modal。 | `nsmoke/` |
| F0（L40S sm_89） | **通过**（门，不计入 task）：learner 作业 SUCCEEDED，1 轮完成（trained_groups 4 / samples 32）；manifest certify rc=0，commits 与 pin 一致。yeto CLI rc=2 是 no-sync Modal 无可取回产物的既定返回（launcher.py `--rl-single-island-no-sync island ran on Modal … return 2`），不是运行失败。 | `f0/` |
| F-E1 指纹 | 取得 T1R1S1 3 卡 elastic 运行的 runtime_fingerprint `sha256:b3efcc64…`，按计划在 `rl_driver_start` 后手动停止 | `fe1fp/` |
| F-E1（冒烟，不计入 task） | **up 事务失败 → REBUILT_OLD**，发现代码缺陷，停止本批其余 GPU 运行（见下）。rebuild-old 路径本身工作：旧成员不变、config_epoch 保持 0、训练继续到第 4 轮无异常。 | `fe1/`（journal、epochs、inbox status、事件磁带） |
| A4 E1-A / E1-E（3.4、3.1/3.2/3.6 旁证） | **未完成（等待代码）**：同一缺陷使 8 卡 up 无法执行，未启动（$0） | — |
| A4 E1-D watchdog 用例（§8.7(2)） | **未完成（等待代码）**：需先能启动新 cell，同一缺陷阻塞，未启动 | — |
| A4 E1-B（3.5） | **未完成（等待代码）**：无权重覆盖注入（§9.5） | — |
| A4 E1-D ①–④⑦ | **未完成（等待代码）**；⑤⑥ **环境阻塞**（state dir 在容器内非持久路径，无原地重启 learner 入口） | — |
| A4b（3.3） | **未完成（等待代码）**：`elastic_wiring_for` 不传 tool_wait_board，无工具负载 | — |
| A2（L-2.3，2.3） | **未完成（等待代码）**：launcher 不转发 `--eval-temperature`，Miles 回落到 rollout_temperature=1.0，判据 5 的贪心 eval 无法设置 | — |
| A2+（L-1.7，1.7） | **未完成（等待代码）**：`yeto_rl_observe_timeline` 无 launcher 入口，无工具负载 | — |

没有任何 task 满足原文验收，本批不勾选任何 task。

## 发现的代码缺陷（F-E1，需代码负责人处理，本 agent 未改代码）

F-E1 journal（`fe1/elastic-state/reconfig/journal.jsonl` seq 7–11）：
```
add_intent members=["engine:c0"]
fork_op start cells=["engine:c0"] -> failed:
KeyError: "cells ['c0'] were not declared at startup; declared cells are ['inference-engine-all-0-0-00000']"
phase REBUILD_OLD -> REBUILT_OLD
```
1. `--rl-elastic-cells c0,c1` 只进入 yeto 侧 `MilesRolloutPool(declared_cells=…)`，没有传给 fork；fork 启动时只声明了它实际启动的 cell `inference-engine-all-0-0-00000`，standby GPU 上没有"已声明、未启动"的 cell，因此任何 up 事务都无 cell 可启动。这正是 fork 需求 F-R1（启动时声明不启动的停止 cell）+ yeto 把声明的 cell id 传给 fork 的接线。
2. 命名不一致：yeto 把声明列表中的 `c0` 当作"新增"成员，而初始在役成员以 fork 名 `engine:inference-engine-all-0-0-00000` 出现（epochs.json `members`），两套 cell id 没有映射。
3. 次要：launcher 经 sky 0.13 提交私有镜像岛失败（Nebius 冒烟），以及对 Nebius 使用 spot。

解除条件：F-R1 fork 实现 + 镜像重建 + yeto 传递/映射声明 cell，合入集成分支后先重跑 F-E1（3×L40S，≈$1–3）确认 up/down 到达 SUCCEEDED，再跑 E1-A 基线、E1-A、watchdog 用例。

## 费用（上界按墙钟×卡数×单价）

| 运行 | Modal app | 墙钟 | 费用上界 |
|---|---|---|---|
| nsmoke | —（未开通） | — | $0 |
| f0 | ap-4WHOoo6jjpVNP3CkJ8DT1p | 03:43–03:56，1×L40S | $0.42 |
| fe1fp | ap-M16C4QJ9WiiU1KPdPyWua2 | 03:59–04:15，3×L40S | $1.56 |
| fe1 | ap-F4XpsACPevWqjbkXVjcuyC | 04:16–04:26，3×L40S | $0.93 |
| 合计 | | | **≤ $2.91**（本批预算 $127，全局 $300） |

## 无残留
- Modal：三个 app 均 `stopped`、tasks 0（`fe1/app_final.txt`、`modal_app_list_final.txt`）。
- Nebius/sky：`sky status` 无集群；`nebius compute instance list` 为空（冒烟时核实）。
- 本地：launcher、`_worker`、puller、watchdog、submit 进程均已终止；无付费卷创建。
