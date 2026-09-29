# rl-algo-mismatch-correction 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)。

## 2026-09-29（Agent ALIGN，阶段 0）

- 规划文档从 `/home/michael/work/rl-algos`（分支 `rl-algorithms`，HEAD 18695ae，含未跟踪文件）原样复制到分支 `rl-infra-spec`（提交 `5753e30`），此后以该分支上的副本为准。
- 本 change 编号 1a，依赖 P0。与 infra 的接口：A5（G2 报告注明执行模式与权重传输方式，不外推到分区模式）、A6（staleness>0 另立 change）。

## 2026-09-29（Agent ALGO-1a，实现轮）

### 分支与状态

- 分支 `algo-1a`，worktree `/home/michael/work/algo-1a`；基底 algo-cap `3d1b466`，之后 merge 了 `origin/algo-cap` ebd436b（P0 扩展点）和 `origin/rl-integ` f6194da（私有 ports 镜像 pins）。已普通推送到 `origin/algo-1a`。HEAD 与未提交改动见文末“状态快照”。
- 共享文件只改了 `yeto/rl/algos/__init__.py` 中的一行（EXTENSION_MODULES 注册）。其余共享接口的需求写成补丁 `/home/michael/work/infra-drafts/1a-shared.patch`，本分支没有应用它。

### 新增文件

- `yeto/rl/algos/mismatch_correction.py`：注册模块。命名机制包括 mismatch_observe、icepop、opsm_trainer、opsm_rollout、mis、mis_mask；扩展字段包括 `opsm_old_logprob_source`、`mis_*`；另有拒绝规则、MIS 的 runtime attrs，以及 `masked_fraction_from_metrics`。
- `yeto/rl/algos/mismatch_observe.py`：只观测插件。
- `yeto/rl/algos/vendor/miles_mis.py`：Miles `mis.py` 的逐字副本，文件头记录来源、commit 与 Apache-2.0 许可证。
- `tests/test_rl_mismatch_correction.py`：yeto venv 测试。
- `tests/test_rl_mismatch_observe.py`：在 miles-next-venv 中针对 Miles 源码做数值测试。
- `docs/MILES_RL.md` 新增一节 “Train/inference mismatch corrections”。


### 任务状态

状态只用五种之一。勾选与完成记录以 tasks.md 为准。

| task | 状态 | 说明 |
| --- | --- | --- |
| 1.1 | CPU 通过 | baseline-failures.txt，94 条 |
| 1.2 | 已实现 | 见下文“1.2 接口核对” |
| 2.1 | 已实现 | 见下文“2.1 源码核对” |
| 2.2 / 2.3 / 2.4 | CPU 通过 | miles-next-venv 数值测试，loss 与梯度 `torch.equal` |
| 3.1 / 3.2 | CPU 通过 | |
| 3.3 | 未完成 | OPSM 与其他修正组合需要 1a-shared.patch |
| 3.4 | 未完成 | fake 已通过；adapter 需调用 `masked_fraction_from_metrics`（INFRA） |
| 4.1 / 4.2 | 未完成 | “总是写入 JSON”与“与 TIS 同开时的 π_old 报错”需要 1a-shared.patch |
| 4.3 | CPU 通过 | |
| 5.1 | 已实现 | 镜像可 import 原文件；Apache-2.0 |
| 5.2 | CPU 通过 | vendor 仍保留，原因是插件命名空间（偏离 D6，待批准） |
| 5.3 | CPU 通过 | |
| 6.1 | 未完成 | fake.py 归 ALGO-CAP；补丁见 1a-declare.patch |
| 6.2 / 6.3 | CPU 通过 | |
| 7.1 | 已实现（原撤销后已按原文补做 dry-run） | evidence/2026-09-29-dryrun |
| 7.2 | GPU 验收通过（G1） | tis、icepop、opsm_trainer、mismatch_observe、mis、mis_mask；opsm_rollout 未跑 |
| 7.3 | 未完成 | 已声明 none/tis/opsm/opsm_trainer（opsm 为 OPSM 维度，不放行来源）；其余机制待集成分支处理或补触发验证 |
| 7.4 | GPU 验收通过（G2） | evidence/2026-09-29-g1b/runs/g2-observe/report.md |
| 7.5 | 未完成 | G3（TIS）跑过一次，未通过（判据 3、4 不成立：磁带截断），待重跑并补跑 IcePop |
| 7.6 | 已实现 | 无残留，费用见下 |
| 7.7 | 未完成（可选） | |
| 8.1 / 8.2 / 8.3 | CPU 通过 / 已实现 | |

（更正，以下一行为准）早先记录的“仍为 ⚙（Miles adapter 未声明）的机制：全部”已被取代：tis、opsm、opsm_trainer 已声明；仍为 ⚙ 的是 mismatch_observe、icepop、opsm_rollout、mis、mis_mask。原句：包括 mismatch_observe、tis、icepop、opsm_trainer、opsm_rollout、mis、mis_mask。其中 tis 和 opsm_trainer 的 G1 已通过，等 ALGO-CAP 合入 1a-declare.patch 后即变为 ✅。

### 1.2 接口核对

- `CorrectionSpec`、`PluginRef`、`register_*` 与 `expects_gradient`：`yeto/rl/engine/algorithm.py`。
- 修正参数映射与 `dry_run`：`miles_adapter/algorithm_flags.py`。
- `EngineCapabilities.corrections`/`features`/`with_unverified`：`engine/capabilities.py`。
- `TrainStepMetrics.masked_fraction` 与 `_check_gradient` 调用 `expects_gradient`：`engine/driver.py`（合入 algo-cap 2f9f02c 后可用）。
- trainer 读取 `masked_fraction` 键：`miles_adapter/trainer.py`。
- 放行开关与 `--rl-single-island-no-sync`：`learner.py`/`launcher.py`。

### 2.1 源码核对（Miles 0394715 = 9e4260d + yeto 补丁，loss_hub 与上游相同）

- custom-tis 函数的签名为 `f(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, total_lengths, response_lengths, parallel_state, max_seq_lens)`，返回 `(pg_loss, masks, metrics)`（`loss_hub/losses.py:252-264`）。
- 调用条件是 `args.get_mismatch_metrics or args.use_tis`（`losses.py:233`），也就是说**不要求 `--use-tis`**。但 P0 的翻译与吸收规定 custom 必须带 `--use-tis`；CPU 对照证明两种写法梯度相同。
- `--get-mismatch-metrics` 要求同时给出 custom 路径（`arguments.py:3456-3459`）；`use_rollout_logprobs` 与 `use_tis` 互斥（`:3453`）。
- 指标：custom 函数返回的键（内置 TIS/IcePop 为 `tis`、`tis_clipfrac`、`tis_abs`）在触发时额外输出 `ois`（`losses.py:386-393`）；`train_rollout_kl` 与 `train_rollout_logprob_abs_diff` 只要有 rollout logprob 就会输出（`:347-366`）；`ess_ratio` 总是输出，它是 π_θ/π_old（`:295`）；OPSM 输出 `opsm_clipfrac`（`:395`）。这些指标都用修正前的 mask 聚合。
- vanilla TIS（`corrections.py:7`）用 `clamp(ratio, low, high)`，不屏蔽 token；IcePop（`:35`）区间内取 ratio、区间外为 0；OPSM（`math_utils.py:183`）在 adv<0 且序列 KL>δ 时屏蔽整条序列，它的 `opsm_clipfrac` 是按 1/len 累加得到的，不是比例。
- MIS 的参数（`tis_mode` 等）没有 CLI flag，靠 `--custom-config-path` 写入 namespace（`arguments.py:3195`）。yeto 改为通过 `register_runtime_attrs` 写入。

### 测试命令与结果

- `OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors -p no:cacheprovider -rfE`：68 failed、2377 passed、51 skipped、26 errors。失败集合按 id 与基线 3d1b466 相同（diff 为空）。
- `tests/test_rl_mismatch_correction.py`：84 passed、7 skipped，skip 的都需要 1a-shared.patch。在打了补丁的 scratch 树（`1a-shared.patch` + `p0-driver.patch`）上全部通过，全量失败集合也与基线相同。
- `PYTHONPATH=/home/michael/work/miles-next:$PWD /home/michael/work/miles-next-venv/bin/python -m pytest -q tests/test_rl_mismatch_observe.py`：44 passed、3 skipped（其中 geometric 与 batch normalize 组合被 Miles 拒绝，spec 也拒绝）。
- `openspec validate rl-algo-mismatch-correction --strict`：valid。
- 证据：`evidence/2026-09-29-cpu/`。

### GPU 与云资源

| 轮次 | 资源 | 创建 → 释放 (UTC) | 用途 | 估算费用 |
| --- | --- | --- | --- | --- |
| g1 | Modal app `algo1a-g1` (ap-nYorxc1jQ0b3TJOcEoZOVN)，sandbox sb-rHvfMERF2KVJArErjzeMps，H100!×1 | 16:43:43 → ~17:05 | 5.1、G1 tis/icepop/opsm-trainer（observe 第一次尝试因 HF 网络失败） | ≈ $1.8 |
| g1b | Modal app `algo1a-g1b` (ap-kx0C1e9SPC9xdLahz1pyDx)，sandbox sb-1kos5UHRiLNRFIevunVJT1，H100!×1 | 17:09:01 → 17:36:54 | G1 observe/mis/mis-mask，G2 | ≈ $2.2 |

合计约 0.83 H100·h，约 $4（按 Modal 标价估算，未经账单确认）。两个 app 在 `modal app list` 中均为 stopped、0 tasks（证据在 `*/modal_app_list_after_stop.txt`）；watchdog 进程已结束；没有创建卷或命名 secret。
暂停说明：g1 进行中收到主 agent 通知，要求在 P0 新入口就绪前不要启动需要放行开关的 G1。当时在跑的 opsm-trainer 是通知前已启动的，跑完后没有再启动新的运行。P0 入口就绪、合入并另写计划后，才启动 g1b。

### 对 INFRA / ALGO-CAP 的接口需求

1. **ALGO-CAP，1a-shared.patch**：（a）`FieldDef.always_emit`：选 OPSM 时来源总是写入规范化 JSON；（b）允许 `opsm_delta` 与 tis/custom 同用，`opsm` 机制按 `opsm_delta` 检测；（c）`register_named_correction_function`：已命名的 custom 函数不再要求通用的 `corrections:custom`，否则 adapter 一旦声明 custom 就会放开任意函数。补丁已在 scratch 树上验证：全量失败集合与基线相同。
2. **ALGO-CAP，1a-declare.patch**：`entry.py` 声明 `tis`/`opsm`/`opsm_trainer`（G1 已过）；`fake.py` 声明全部修正机制。P0 自己的 3 个测试（`test_rl_algorithm_capabilities.py` 中写死 `supported: ['none']` 或 fake 默认值的断言）需要同步修改。
3. **ALGO-CAP**：`--custom-config-path` 可以任意改写 Miles namespace（包括 use_tis 等目标参数），目前 P0 的 objective flag 表没有它，建议加入 unmapped 表拒绝。另请决定 `examples.` 是否可以作为插件命名空间（见 5.2）。
4. **INFRA，trainer.py**：屏蔽比例目前只读 `masked_fraction` 键。需要用 `yeto.rl.algos.mismatch_correction.masked_fraction_from_metrics(spec, metrics)`，从 Miles 的 `train/tis_clipfrac` 与 `train/mis_tis_mask_fraction_*` 推导，并确认 `actor.train` 的输出里带有 reported-loss 指标。
5. **INFRA，事件**：spec 场景“该轮训练指标事件中包含 mismatch 指标”目前只在 Miles 的 `train/*` 日志中满足，`rl_local_round` 事件没有这些键（`mean_kl`/`ess_ratio` 为 None）。需要把 Miles 的 reported-loss（至少 tis、tis_abs、train_rollout_kl、ois、ess_ratio、tis_clipfrac、opsm_clipfrac、mis_*）接入 `TrainStepMetrics`/事件，并带上 A5 标签。
6. **INFRA/ALGO-CAP**：benchmark worker 路径（`run_training_worker`）不经过 CLI 解析，D11 放行检查不会执行，G1 harness 只好手动调用。launcher 入口 `--rl-single-island-no-sync` 还没有真实 GPU 运行。

### 待批准

- 5.2 偏离 D6：原文件在镜像里可以 import，但 P0 插件命名空间只允许 `yeto.`/`miles.`，因此仍用 vendor 副本。二选一：接受 vendor，或由 ALGO-CAP 放开 `examples.` 命名空间。
- 1a-shared.patch 与 1a-declare.patch 由主 agent 转给 ALGO-CAP，审阅后合入。

### 阻塞与解除条件

- 3.3、4.1、4.2：1a-shared.patch 合入后，去掉相应 skip 即完成（测试已写好）。
- 3.4：INFRA 按需求 4 完成接线。
- 6.1、7.3：ALGO-CAP 合入 1a-declare.patch；observe、icepop、mis、mis_mask 的声明需先合入 1a-shared.patch 的（c）。
- 7.5：两岛 G3 需要 head 与 syncer 的放置方案（Modal/Verda 不能承载 head，见 memory），并且只能使用已声明的机制，所以要在 7.3 之后。

### 下一步

1. 等 ALGO-CAP 合入两个补丁后 `git merge origin/algo-cap`，重跑 `tests/test_rl_mismatch_correction.py`，确认 skip 项转为通过，然后勾选 3.3、4.1、4.2、6.1、7.3。
2. G3：写计划，在 Nebius（或 AWS head）上跑两岛 strict-avg，TIS 与 IcePop 各约 3 轮。
3. 可选：OPSM(rollout) 的 G1。

## 2026-09-29 复审处理（ALGO-1a）

- 已合入 origin/algo-cap fefcbe0，其中包含 1a-shared/1a-declare，并经 P0 收紧：已命名的 custom 函数只能用 `yeto.` 路径。因此 IcePop（`miles.` 路径）仍需要 `corrections:custom`，我这边的注册已相应调整。`--custom-config-path` 现在由 P0 的拒绝表拒绝。
- 测试：恢复了 `test_custom_config_path_refused_on_ports`。adapter 测试改为对具体集合断言：恰好声明 {none, tis, opsm, opsm_trainer}，只有 tis 和 opsm_trainer 能通过，其余项被拒并列出支持集合。文档 dry-run 测试在“已声明”时断言 accepted 且 argv 一致，未声明时断言被拒。
- tasks 状态：3.3、4.1、4.2、6.1 在补丁合入后转为 CPU 通过；5.2、7.1 撤销勾选。7.1 撤销的原因是 `yeto launch` 没有 `--dry-run`，这一点列为接口需求。
- 4.1 合入说明：**所有已有 OPSM 规格的哈希都因 `opsm_old_logprob_source` 字段而改变**，这是预期结果，为 4.1 所要求。
- 7.2 完成记录补充了 receipt 的说明（没有显式判据，由 driver 失败即报事件间接覆盖）、GPU 证据的局限（截断/屏蔽分支都没有触发；每轮一步时 OPSM 结构上不可能触发；`ess_ratio`/`ois` 恒为 1，不反映训推差异），以及上传时刻与 YETO_SHA 的对应（精确上传时刻没有记录）。
- 7.4：design 把 `ess_ratio` 当作训推差异指标，这一意图**未达成**。
- proposal.md 按 A6 改写为“需另立独立算法契约 change”。
- **重新验证（G-2）受阻**：计划为 `evidence/2026-09-29-g1c/plan.md`（先提交后运行）。前两次尝试是 harness 问题：第一次 venv 里没有 sky；之后两次（第 2、3 次）都在 Modal 部署上传工作树时报 `RuntimeError: can't start new thread`。原因是本机接近每用户线程上限（系统约 1.08 万线程，`ulimit -u` 4096），不是机制本身的结论。三次都没有创建 Modal app，没有花 GPU。按计划停止，icepop 和 opsm-trainer 未启动。所以 tis/opsm/opsm_trainer 的声明**尚未经新入口复验**；是撤回还是保留等复验，请主 agent 决定并转告 ALGO-CAP。
- G3：计划草案见 `evidence/2026-09-29-g3-plan.md`，未启动。它同样受上面的线程上限阻塞；IcePop 的声明也还没做。
- 待批准（新增）：5.2 偏离原文（能 import 却仍使用副本），需用户批准；主 agent 倾向保留副本。
- 全量测试：68 failed + 26 errors，按 id 与基线 3d1b466 相同。
- 云资源：本轮没有新建任何资源（g1c 的 3 次尝试都在 app 创建前失败）。之前的 algo1a-g1 和 algo1a-g1b 仍为 stopped、0 tasks。g1c 留下 3 个孤儿 `sleep 3000` 进程，它们原本是 watchdog 的子进程，父进程已结束，醒来后不会执行任何操作。

## 2026-09-29 复验 / G3 / 生效验证（ALGO-1a）

- 7.1 补做完成：`yeto launch --rl-single-island-no-sync --controller local --dry-run` 的输出为 total_gpus 1、syncer null、outer_sync false（`evidence/2026-09-29-dryrun/`）。
- g1c 入口复验：tis、icepop、opsm-trainer 全部 PASS（判据 1-6）。前 4 次尝试都是环境或入口问题：venv 缺 sky、线程上限、FleetController 缺陷（已由 P0 修复）。opsm-trainer 有一处流程偏离：启动时线程数高于我自定的前置阈值；中止时误杀了 wrapper，事件改用 launcher 回传的磁带。两处都已在 plan 中如实记录。
- 生效验证（`evidence/2026-09-29-trigger/`）：tis、icepop、mis_mask 的截断/屏蔽比例 >0；OPSM 在 `--rl-optimizer-steps 2` 下 opsm_clipfrac >0。因此“声明需 GPU 上确实生效”的条件对 tis 和 opsm_trainer 已满足。
- G3（TIS，两个 Modal 岛 + 本机 syncer 在 29410 端口）**未通过**：判据 3、4 无法成立，原因是 island-1 的磁带拉取截断；syncer 显示 3 步 strict 同步、两个 responder 都在，v0/v1 的权重 hash 一致。重跑需先提交 harness 修复（改用 launcher 回传磁带，前提是两岛路径也支持回传）。
- 本节云资源：g1c 3 次（约 32 min）+ trigger 4 次（约 33 min）+ G3 2×H100 约 17 min，合计约 1.65 H100·h、约 $6.5（估算，未经账单确认）；连同此前两轮，总计约 $10.5。所有 algo1a 应用均无容器残留（`modal container list` 为 0），29410 端口已关闭，watchdog 已结束。
