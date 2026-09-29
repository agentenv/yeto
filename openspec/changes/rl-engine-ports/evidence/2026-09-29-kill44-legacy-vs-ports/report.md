# 4.4 kill-restart：legacy vs ports 对照（2026-09-29）

## 结论

**PASS（行为一致）**，另有一处 ports 的小缺陷（只影响记录，不影响训练），不阻塞 4.4。

- 恢复轮次、权威 cut 的应用方式（reset）、group 复用或丢弃的结果、此后每轮的 group 数、syncer 最终状态，两边全部一致。
- 发现一处差异：重启后，ports 不会重写第 2 轮的进度记录，`rollout_metrics` 保留了被 kill 那次生成的值；legacy 会用重新生成的结果覆盖。这是 ports 的小 bug，只影响记录保真度，没有下游读取方（见"差异分析"）。
- 两边共有一个预先存在的行为：重启后 prompt 流从偏移 0 重新开始。这不是 ports 引入的差异。

## 运行环境

| | legacy | ports |
|---|---|---|
| yeto 代码 | `rl-engine-ports` 分支 **2bb651dcffd8871e278622303488b973a2e57b50**（未改动，两边同一份 tar） | 同左 |
| GPU | Modal `H100!:1`，`NVIDIA H100 80GB HBM3`（开跑前已断言，见 `env-*/gpu.txt`） | 同左 |
| Modal app | `yeto-rl-kill`，sandbox `sb-HefqX4ZrqZ5o1MZqASHv56` | 同 app，sandbox `sb-OKv3k0NJhqj9kj0u6CGFIA` |
| 底座镜像 | `radixark/miles@sha256:cd40db92…`（v0.1.0） | `radixark/miles@sha256:90940828…` |
| Miles | agentenv/miles `ae475060`（基底 6062afe 加上 `miles-qwen38.bundle`，sha256 da3464d3…），`peft==0.20.0` | michaellchung/miles `0394715083c9…` |
| SGLang | agentenv/sglang `e1b57eb8` | michaellchung/sglang `9f29303bef1e…` |
| 其他 | router 等待 shim `yeto_legacy_router_timeout.py`（通过 .pth 注入，与 legacy-baseline-v2 相同） | 无 |
| `--rl-engine` / miles_root | legacy / `/opt/miles-legacy` | ports / `/opt/miles-next` |

`env-*/sandbox-env.txt` 记录了 nvidia-smi、提交 SHA 和 pip freeze。

## 配置与 kill 点（两边相同）

`harness/kill44x.py <engine> <out>` 由 `2026-09-29-harness/kill44.py` 泛化而来，只把引擎和 miles_root 做成参数。配置如下：

- 单岛 strict，接 syncer（learners=1）
- Qwen3-0.6B@c1899de
- 4 轮，每轮 4 个 group，每 group 8 条样本
- over_sampling = 4
- LoRA r16 all-linear，lr 1e-5，seed 17，gsm8k 前 16 题

**统一的 kill 点**：`island-0/completed-groups.pt` 出现 `policy_version==2` 且 `local_round_stats is None`，说明第 2 轮 rollout 已生成、这一轮还未提交。此时等 3 秒，再对 learner 进程组发 SIGKILL，20 秒后重启。

之所以不用旧 harness 的 `rl_driver_phase` 事件做触发，是因为 legacy 不产生这个事件。

- ports 在 kill 时最后一条事件是 `rl_driver_phase:train:2`。
- legacy 的 `miles-first.log` 最后一行是 `Timer train start / compute_log_prob`。
- 所以两边都是在第 2 轮训练中被 kill。

harness 还有一个 watcher 线程，把进度文件的每一次变化都写入 `cg-states.jsonl`。原因是 `*.pt` 不会被拷回，要靠它看到文件内容。

## 对照表

| 项目 | legacy | ports | 一致? |
|---|---|---|---|
| 第一次 learner 退出 | rc -9（SIGKILL） | rc -9 | ✓ |
| kill 时进度文件 | schema 3，policy_version 2，local_round_id 3，stats None，completed_groups 0 | 同左 | ✓ |
| 重启后从哪个 rollout 继续 | rollout 2（syncer 以 `rebound outstanding pull … step=3` 把原 pull 交给新 generation） | rollout 2（日志同左） | ✓ |
| 应用的权威 cut | v2，hash `998b18db…`，与 kill 前应用的 v2 相同 | v2，hash `a7f6ab7b…`，与 kill 前相同，也与 923b304 rerun 相同 | ✓（两个运行时的 hash 本就不同，v0 分别为 62f3…/a34c…，这里比较的是结构） |
| optimizer 应用方式 | reset（`rl_policy_apply` 中 `optimizer_reset_count` 从 1 重新计数，`reset_parameter_count=392`） | `optimizer="reset"`，`local_step=2` | ✓ |
| 未训练 group 的复用 | 重启时恢复队列（`_restore_completed_groups`），恢复出 0 个 group；被 kill 那轮的 32 条轨迹丢弃，rollout 2 重新生成 | 没有恢复队列的路径，队列本来为空；rollout 2 重新生成 | ✓（结果相同） |
| completed-groups 各次状态 | 始终为 0 个 group；重启后先重写 pv2 记录（sha dab6d3bc…），提交后 lrid 3 stats=4 组，再 pv3 → lrid 4 | 始终为 0 个 group；重启后**没有**重写 pv2，直接提交 lrid 3 stats=4 组，再 pv3 → lrid 4 | 队列一致，记录不同（见下） |
| 重启后每轮 group 数 | rollout 2：4 组 / 32 条；rollout 3：4 / 32 | rollout 2：4 / 32；rollout 3：4 / 32 | ✓ |
| 重启后使用的 prompt | rollout 2 用 prompt 0–3，rollout 3 用 4–7（kill 前 rollout 2 用的是 8–11） | 同左 | ✓（两边都从偏移 0 开始，属于共有行为） |
| 最终 syncer 状态 | `training complete after 4 outer steps`，`all learners acknowledged final cut learners=1`，step 4，rejected_stale 0 | 同左 | ✓ |
| 最终 cut | v4 `913e1ce6…` | v4 `dd7b6da0…`（与 923b304 rerun 逐版本 hash 完全一致） | ✓（结构一致） |
| harness / 重启 rc | 0 / 0 / syncer 0 | 0 / 0 / syncer 0 | ✓ |

## 差异分析

### 1. 重启后第 2 轮的 `rollout_metrics` 是旧值（ports 小 bug）

- **legacy**：rollout 函数 `yeto.rl.miles.generate_rollout` 每轮都会写进度。开头调用 `_restore_completed_groups`（`yeto/rl/miles.py:1299`，定义在 `:1021`），结尾调用 `_save_completed_groups`（`:1464`，定义在 `:1074`），用本次生成的 metrics 和剩余队列覆盖记录。因此重启后 pv2 的记录被重写，watcher 看到了 sha `dab6…` 这次写入。
- **ports**：rollout 走 upstream，只有 `StrictIslandProgress.after_generate` 写记录（`yeto/rl/engine/bridges.py:112-133`，调用点在 `yeto/rl/engine/driver.py:422`）。它的判断是：同一 `policy_version` 已有记录且 `rollout_metrics` 不为空，就直接跳过（`bridges.py:113-121`）。这条分支的本意是"rollout 进程已经写过这一轮"，但 ports 的 rollout 进程从不写这个文件。于是重启后命中跳过分支的，是被 kill 那次尝试留下的记录。之后 `commit_round` 只更新 `local_round_id` 和 `local_round_stats`。

结果：ports 最终的 round-2 记录里，`rollout_metrics` 来自被丢弃的那批，`local_round_stats` 来自重新生成的那批，两者不一致。

判断：这是 ports 的缺陷，因为它偏离了 legacy 的写入语义。影响仅限于记录：ports 中没有任何代码读取 `rollout_metrics`（grep 只命中 `bridges.py`），训练、同步、恢复都不受影响。规格（design D2："island 只保存重建进度"）没有要求这个字段。建议的修复方向（本次未改代码）：`after_generate` 在同一进程内只跳过本进程刚写过的记录，或者一律覆盖。

### 2. 队列恢复路径缺失（潜在差异，本配置下未触发）

ports 没有与 `_restore_completed_groups` 对应的实现。在 strict 且 `over_sampling_batch_size == rollout_batch_size` 时，队列始终为空，本次两边的每个状态都是 0 个 group，所以观察不到差异。若 over_sampling 大于 batch，legacy 会在重启时复用同一 policy_version 下剩余的完整 group，ports 则会丢掉它们。

判断：规格允许。legacy 代码自己也把这个队列标为可丢弃（`miles.py:1051` 的注释："local queue is disposable, global state is not"），而且 strict 下这些 group 用的是同一 cut，丢弃只会损失算力，不影响正确性。如需严格对齐，可以另开任务。

### 3. prompt 流从偏移 0 开始（两边共有）

两边重启后，第 2 轮都重新使用 prompt 0–3。原因是 Miles 的 `RolloutDataSource.load` 在 `args.load is None` 时直接返回（legacy fork 的 `miles/rollout/data_source.py:142`），而 yeto 以 `--finetune`/`--no-load-*` 启动，不恢复数据偏移。ports 在重启时的日志也写着 `--start-rollout-id 0 was asked for, so it starts at 0`。这不是 ports 与 legacy 之间的差异，但值得单独记录为已知问题。

## 成本

- sandbox 生命周期：legacy 766 s，ports 1163 s，合计 0.536 GPU·h（H100×1），`timing.txt` 记录了 unix 时间戳。
- 估算费用：约 $3.1（H100 约 $2.1，按 $3.95/h；CPU 16 核与 128 GiB 内存约 $1.0）。预算为 $10。
- 两个 sandbox 均在 trap 中终止，app `yeto-rl-kill` 已 stop。`modal_app_list.txt` 与 `modal_container_list.txt` 显示活跃容器为 0。

## 文件

- `kill44-{legacy,ports}/`：`kill44.jsonl`、`cg-states.jsonl`、`island-0/events.jsonl`、`syncer.jsonl`、`worker.json`、`prompts.jsonl`、`nvidia-smi.txt`、`YETO_SHA`、`rc`、`cmd.txt`，以及下面列出的 `*.log`。已排除 `*.pt`、`*.f32`、`*.safetensors`、`state.ckpt*`。
- `env-{legacy,ports}/`：`sandbox-env.txt`、`gpu.txt`
- `harness/`：`sbx.py`、`drive.sh`、`kill44x.py`、`worker2.py`、`gsm8k_reward.py`、`yeto_legacy_router_timeout.py`
- `drive-{legacy,ports}.log`、`timing.txt`、`modal_app_list.txt`、`modal_container_list.txt`

`*.log` 被 gitignore，下列文件需要 `git add -f`：

```
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/drive-legacy.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/drive-ports.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-legacy/miles-first.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-legacy/miles-restart.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-legacy/run.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-legacy/syncer.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-ports/miles-first.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-ports/miles-restart.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-ports/run.log
openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports/kill44-ports/syncer.log
```
