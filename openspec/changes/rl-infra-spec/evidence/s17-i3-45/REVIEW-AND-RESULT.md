# S17 I3：rl-infra-spec 4.5（同形恢复故障矩阵）上卡前复核

负责：夜间子 agent N7。写于 2026-10-08 约 21:30Z，上卡前。范围按主 agent 决定：今晚只做 4.5，上限 $40；4.6–4.8 不跑（4.6 的 no-go 留给用户），I5/I6 不跑。判据唯一来源：`evidence/infra-e2/4.2-4.5/plan-v2.md` G-4.5（配置 C3），本文件不另立判据。

## 1. 4.5 现状（读 tasks.md 与台账得出）
G-4.5 共 6 行（第 6 行分 a/b），09-30 已在 GPU 上跑过（代码 a016f7f / bbc830c，Miles e3a11ab，旧镜像 2cc5cc52）：
| 行 | 注入 | 09-30 结果 |
|---|---|---|
| 1 | restore 期间 kill rank1 | 通过（74 s 后 RECOVERY_REQUIRED） |
| 2 | restore 前 rank1 sleep 超过 distributed timeout | **不通过 / 该故障在现设计中不存在**：恢复插件各 rank 独立执行，没有集合通信，sleep 只拖慢（495 s 后 SUCCEEDED） |
| 3 | save_cut 写分片途中 kill rank1 | 通过（无 manifest，RECOVERY_REQUIRED） |
| 4 | create_training_models 抛错 | 通过（rbold：REBUILD_OLD） |
| 5 | 重建期间改游标 | 通过（e2r，bbc830c） |
| 6a | REBUILDING_TRAINER 处 kill learner（CAS 前） | 通过（原地重启后对账 RECOVERY_REQUIRED，无重复消费） |
| 6b | COMMITTED 之后 kill | 不适用：同形重建的状态机没有 COMMITTED 阶段 |
另有已知限制：REBUILDING_TRAINER 阶段没有 deadline 强制。

**4.5 无法靠今晚上卡勾选**：第 2 行和第 6b 行是设计层面的问题（要么接受"该故障模式在同形重建里不存在 / 不适用"，要么新增一个在集合通信内部卡住的注入点，例如在重建后第一个训练步的数据并行梯度归约前让一个 rank 睡过超时），需要用户定；第 2 行若要新注入点，得改适配层代码（应叠在 s17-decouple-p4 上，并避开 N8 正在改的 round_cut.py）。

## 2. 今晚做什么：在当前代码上回归已通过的 5 行 + 正常重建对照
自 09-30 以来代码变化很大（main 80e944b6：Miles 钉点 e3a11ab → 8bc52237a，含"引擎加入发布失败不算训练失败"等改动；E1/E2 合并后的 controller；镜像 4aeafd77），09-30 的通过结论不能直接沿用到现在的代码。本批在现代码上复验：
| run | 行 | 注入 | 期望（G-4.5 原文） |
|---|---|---|---|
| s17-i3-c3-rb | 对照（G-4.4） | 无 | 重建 SUCCEEDED，之后照常训练 |
| s17-i3-c3-rbold | 第 4 行 | `--rl-test-inject-rebuild-fail 1` | TrainerRebuildError 后结果 REBUILD_OLD；有 cleanup_error 才 RECOVERY_REQUIRED |
| s17-i3-c3-f1 | 第 1 行 | `--rl-test-inject-cut-restore-kill-rank 1` | 600 s 内 RECOVERY_REQUIRED，不训练、不自动重试 |
| s17-i3-c3-f3 | 第 3 行 | `--rl-test-inject-cut-save-kill-rank 1` | 无 manifest；回旧进程继续或 RECOVERY_REQUIRED |
| s17-i3-c3-f5 | 第 5 行 | `--rl-test-inject-rebuild-cursor-shift 1` | 游标比对失败 → RECOVERY_REQUIRED，不调用 restore |
| s17-i3-c3-f6a | 第 6a 行 | `--rl-test-kill-learner-at REBUILDING_TRAINER --rl-elastic-restart-attempts 1` | 按 journal 对账：RECOVERY_REQUIRED 或从 cut 重建，不重复消费 |
所有行共同要求：`optimizer_applied` 不出现第二次；没有组被消费两次；游标不回卷。第 2、6b 行不跑（见 §1）。

运行目录由 main 上的 `tools/probes/e2_cut_harness.py` 生成（`/home/michael/work/s17-i3-runs/`，`plan.json`），改动三处（原文件保留为 `run.sh.orig`）：
1. 触发方式：不再用容器内常驻的 `e2_inwatch.py`（I2 补测 c 里同类常驻进程无故消失），改用本机 `s1-runs/s17-a5-deliver.py`：按岛磁带进度，在第 3 轮 train（rollout_id 2）时用 `modal container exec` 写 `rb1.rebuild.json`（body `{"kind":"trainer-rebuild","expected_config_epoch":0,"deadline_s":900}`，与 `controller rebuild-trainer` 相同），读回核对，再确认岛已取走；任何一步失败或 420 s 未取走即停 app。
2. 开机前跑请求预检 `s1-runs/reconfig_request_preflight.py`（已扩展支持 rebuild：epoch 链、截止、body 种类；同形重建不是配置边，不需要 attestation——读码确认 `rebuild_plan` 不查 attestation），不过即退出，结果存 `preflight.json`。
3. 超时缩短：launcher 1680 s、Modal 1620 s、独立看门狗 1800 s（09-30 同类运行实际 11–15 min）。

## 3. 初始化断言对照（C3：Qwen3-1.7B LoRA r16/α32，单岛 3×H100!，T2R1S0 = 训练 2 卡 DP2 + 推理 1 卡）
| # | 条件 | 本次 | 结论 |
|---|---|---|---|
| 1 | global batch = rollout_batch × n_samples / steps | 2×8/1 = 16 | 通过 |
| 2 | over_sampling ≥ rollout_batch | 2 ≥ 2 | 通过 |
| 3 | prompt ≤ context−1 | seq 1024、回答 384 | 通过 |
| 4 | LoRA 不走 p2p/disk-delta | broadcast | 不触发 |
| 5 | Megatron world % (TP·PP) | 2 % 1 → DP2 | 通过 |
| 6 | global % (mbs × DP) | 16 % 2 | 通过 |
| 7 | 层数 % PP、max_position | 28 % 1；1024 ≤ 40960 | 通过 |
| 8 | 变 DP 认证要求 dropout = 0 | 本批同形（DP 不变），`--rl-lora-dropout 0.05` 允许 | 不触发 |
| 9 | `--rl-elastic` 要求 Miles router、非 colocate、不 offload | fixed-partition | 通过 |
| 10 | 确定性训练 `--rl-deterministic-trainer` | 同 09-30 | 09-30 已过 |
| 11 | 重建要求 partitioned profile、trainer 独占 bundle | T2R1S0 | 通过（A6a 曾因共置失败，本形状非共置） |
| 12 | 同样参数在 09-30 跑通 | 唯一差别是代码/镜像版本 | 以 c3-rb 对照为准 |

## 4. 费用
6 次 × 3 张 H100!（$11.85/h）。09-30 同类每次 11–15 min，期望每次 ≈$2.5–3，合计 ≈$17；每次最坏 30 min ≈$5.9，合计最坏 ≈$35.6 ≤ $40。按顺序一次一个，每次开机前确认我方线程 < 2800 并写进台账。c3-rb 对照若失败（重建本身不成功），后面 5 行暂停，先报告。

## 5. 这次顺带采集什么
- 岛事件磁带、elastic-state（journal、epochs、cuts/*/manifest、ledger、inbox status）：harness 的 puller 每次 exec 拉取，另由投递脚本每 60 s 打包一份；launch.log；`deliver.jsonl`（每次 exec 写入/读回耗时、被取走的时间）。
- 重建分段时长：save_cut / 释放 / create_training_models / restore_cut / 首步，用 journal 与磁带时间戳算；故障路径从注入到终态的时长（"有界"的实测值）。
- 训练卡与推理卡显存、利用率（磁带 rl_resource_sample）；冷启动各段（容器分配、镜像、模型加载、引擎就绪）。
- 与 09-30 同一行结果并排对比（同判据、不同代码），差异逐条记录。
- 原始数据先落盘到各 run 目录，判读只读这些文件。

## 6. 运行记录
- 21:00Z `s17-i3-c3-rb`（对照）**浪费 ≈$1.9**，脚本问题两处，链在对照处停下，其余 5 次未启动：
  1. harness 生成的 puller 守卫（`puller.sh` 第 30 行）要求 `/opt/yeto/image-manifest.json` 里出现标签字符串 `8bc5223@4aeafd77`（`e2_cut_harness.py` 的 `IMAGE_TAG`，只是标签）。现镜像清单里有 Miles 提交 8bc52237a（守卫这一项通过）、GPU 名也对，但没有这个标签（镜像也不可能在清单里写自己的摘要），守卫判失败并 `app stop`。09-30 的旧镜像清单里大概带有该标签，所以当时没触发。修法：守卫改为核对 Miles 提交 + SGLang 提交（清单里有），镜像摘要由启动参数 `--rl-image ...@sha256:4aeafd77…` 钉死，不在容器里查。
  2. 本机投递脚本 `s17-a5-deliver.py` 在找岛容器时 `modal container exec` 超过 60 s，`subprocess.TimeoutExpired` 未捕获，脚本直接退出（若守卫没停 app，后面就会出现"请求没投递却空跑"）。修法：所有 exec 调用捕获超时按失败计数处理；投递进程意外退出时由链脚本发现并停 app。
  - 教训：用旧工具生成的运行目录，上卡前要把守卫/判定条件和当前镜像实际内容对一遍（这次可以先对一个已知的容器 exec 一次清单再上卡）；投递脚本要有"自身退出 = 停机"的兜底。
- 修复与预演（21:3xZ）：守卫改为核对清单里的 Miles 提交 8bc52237a（`git rev-parse` 与清单）和 SGLang 提交 4e4148f1。**防护没有变弱**：镜像本身由启动参数 `--rl-image ghcr.io/michaellchung/yeto-miles-ports@sha256:4aeafd77…` 按摘要钉住，Modal 只会拉这个摘要；旧守卫里的 `8bc5223@4aeafd77` 只是 harness 自己拼的标签字符串，并不是对摘要的校验。投递脚本：所有 exec 捕获超时按失败处理，任何未预期的退出都会先停 app。预演 `yeto-s17-i3-rehearsal`（Modal CPU，正式镜像，模拟磁带和 inbox 消费）：守卫通过；预检通过；投递写入 1.24 s、读回 1.26 s，24 s 后确认被取走；rc=0。旧对照目录移到 `s17-i3-runs/attempt1/`（其中残留的 guard.fail 会让新守卫跳过检查，所以不复用）。主 agent：对照 rb 再失败就停，I3 今晚收工。
- 21:32Z：对照 `s17-i3-c3-rb` **通过**（主 agent 认可）：依据是磁带事件 `rl_trainer_rebuilt`（policy_version 3，向全部成员发布）和 learner 作业终态 SUCCEEDED；重建后第 4–6 轮照常训练，游标 2→12 单调，账本无重复 optimizer_applied。REBUILDING_TRAINER→SUCCEEDED 94 s。链脚本先前判"未到 SUCCEEDED"是误判（拉回的 journal 滞后，快照因带 cut 大文件被截断），已去掉该判断、快照排除 cut 分片。**判读口径**：其余各行以磁带事件和作业终态为准，journal 快照只作辅助。launcher 末行 "exit code 2" 是本分支（main 80e944b6）单岛 Modal 路径的常态，N16 已在 PR #147 修（未合）。

## 7. 结论（22:55Z；判读以磁带事件 `rl_reconfiguration`/`rl_trainer_rebuilt`/`rl_round_trained`、作业终态和 journal 为准；每次的摘要 `s17-i3-runs/<run>/g45-extract.json`）
| 行 | run | 结果 | 依据 |
|---|---|---|---|
| 对照 | c3-rb | **通过** | `rl_reconfiguration` SUCCEEDED（rollout 3），`rl_trainer_rebuilt` v3 向全部成员发布；REBUILDING_TRAINER→SUCCEEDED 94 s；之后 3 轮照常训练；游标 2→12 单调；账本 optimizer_applied 无重复 |
| 1 restore 中 kill rank1 | c3-f1 | **通过** | 注入可见（`restore_cut after the adapter write: os._exit(137)`）；REBUILDING_TRAINER 起 85.5 s 后 RECOVERY_REQUIRED（< 600 s）；之后无训练轮、无重试恢复 |
| 3 save_cut 中 kill rank1 | c3-f3 | **通过** | 注入可见（`save_cut shard trainer_tp0_pp0_dp1.pt half written`）；1.8 s 后 RECOVERY_REQUIRED（岛级和请求级两条记录）；没有任何 cut 文件/manifest；之后无训练轮 |
| 4 create_training_models 抛错 | c3-rbold2 | **通过** | journal `rebuild.outcome = REBUILD_OLD`，attempts：`create_training_models`（注入的 RuntimeError，`cleanup_error: null`）→ `done`，generation 1；事务 SUCCEEDED（同形重建下 REBUILD_OLD 的事务终态就是 SUCCEEDED，09-30 相同） |
| 5 重建中改游标 | c3-f5 | **通过** | "data cursor changed across the trainer rebuild: sample_offset 6 → 7" → RECOVERY_REQUIRED；日志里没有 yeto `restore_cut` 调用；之后无训练轮 |
| 6a REBUILDING_TRAINER 处 kill learner | c3-f6a | **通过** | 第二条 `rl_driver_start`（原地重启）；对账判 RECOVERY_REQUIRED（"learner restarted after release, before commit"）；重启后无训练轮、无重复消费 |
| 2 restore 前睡过超时 | — | 未跑 | 设计层问题（恢复插件没有集合通信），留用户定 |
| 6b COMMITTED 后 kill | — | 未跑 | 同形重建没有 COMMITTED 阶段 |
共同要求（optimizer_applied 不重复、组不重复消费、游标不回卷）在各次可见的账本与磁带里都满足（rb/rbold2 的账本完整；故障行在 RECOVERY_REQUIRED 后不再训练，账本条目止于故障前）。
- 与 09-30 对比：同一行结论全部一致；本次 f1 85.5 s（09-30 74 s），f3 1.8 s（09-30 1.9 s）；f5 在当前代码上第一次就能读到实时游标（09-30 要到 bbc830c 才行）。
- 4.5 **仍不勾选**：第 2 行、第 6b 行和"REBUILDING_TRAINER 阶段没有 deadline 强制"三项需要用户定（见 §1）。
- 顺带记录：首轮训练步约 25–30 s，重建（REBUILDING_TRAINER→SUCCEEDED）约 94 s；launcher 末行 exit 2（正常路径）/4（RECOVERY_REQUIRED 路径）与 09-30 一致，PR #147 未合。
- 费用：I3 合计 ≈ $19.0（上限 $40），其中 rb 第一次（守卫误判）$1.9 浪费、rbold 第一次（快照截断）$2.5 证据不全。
