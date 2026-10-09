# S17 I2：rl-infra-spec 3.8（A5，手动双向切换 + 两岛 strict 暂停兼容）上卡前复核

负责：夜间子 agent N7。写于 2026-10-08，上卡前。判据不放宽：以 `openspec/changes/rl-infra-spec/gpu-plan-v2.md` §3 A5 第 1–5 条、§8.2、§8.7(1)、§9.15 与 `evidence/infra-e1/plan-3.8-4.4-v2.md` §1、§2、§8 为准；本文件只写执行安排和改动的理由。

## 1. 代码与镜像
- yeto：agentenv/main 80e944b6（本人分支 s17-infra-i2，尚无新提交）。3.8 的实现（533afdc）、pause 审计、finalization、PULL 重发记录、start 延迟注入（9a5f181）都已在 main 上。
- Miles：钉点 8bc52237a（`MILES_NEXT_COMMIT`），fork-M2/M3/M4 都在其中（见 tasks 3.3a/3.3b/3.5a 的勾选记录），分支 agentenv/miles `s17-infra-fork-m` 指向同一提交。
- 镜像：`MILES_NEXT_IMAGE` = ghcr.io/michaellchung/yeto-miles-ports@sha256:4aeafd77…（公开，不需要新镜像）。
- 本地 dry-run（`evidence/infra-v2-b1/a5/test_a5_dryrun.py`，CPU，不起 Ray）在 80e944b6 上 4 passed：两岛 rollout_cells c0 启动 / c1 声明不启动、`--use-miles-router`、quorum 用例的 120/2.0/150 都到达 learner，syncer 命令带 `--quorum-timeout-s 120`。（该测试文件 `parents[5]` 少算一层，需 `PYTHONPATH=tests` 才能收集，顺手记一笔，不影响结论。）
- 本次实际参数（`s1-runs/s17-a5-prep/dump_flags.py` 用真实 launcher 生成两岛 learner 参数）：每岛 97 个参数（quorum 101 个），只有 quorum 用例导出 `YETO_RL_TEST_INJECT_START_DELAY_S=150.0`。

## 2. 与 gpu-plan-v2 §9.15 的差异（主 agent 代拍板范围内，理由如下）
| 项 | §9.15 | 本次 | 理由 |
|---|---|---|---|
| 卡型 | 两岛各 `H100!:3` | 两岛各 Modal `L40S:3`（gpu-exact） | 预算上限 $25：H100 三次期望 ≈$35；L40S 约 $1.94/卡·时（F-E1 实测），6 卡 ≈$11.7/h。判据都是"与同卡型基线比样本 id / 精确出现某事件"，不比数值，卡型不影响判据。F-E1 已在 Modal 3×L40S 上跑通同一 T1R1S1 elastic 形状。 |
| head | 本机 head | Nebius eu-north1 无卡 VM（cpu-d3 8 核 32 GB，$0.20/h） | memory：Modal 给不了固定地址，多岛 head 现在一律放 Nebius 无卡 VM；s15 两岛（含 elastic 调度、重入）都走这条路。 |
| 空闲流探测（§3 A5 第 4 条） | Modal CPU 函数对本机端口做 30 min 探测 | 不单独做 | 该条是为"本机 head + NAT"写的。现在 head 在 Nebius 公网 IP，s15 两岛 pause 用例已实测 120 s 静默后连接仍在（ISLAND-STAGE1-PRELAUNCH-REVIEW 第七跑，岛 1 静默 120 s 后正常重入）。若 quorum 用例中真的出现丢流，按原条款判"环境阻塞"，不换路径冒充。 |
| quorum 用例轮数 | 6 | 4 | 判据第 2 条只看 up 所在那一轮（第 2–3 轮），4 轮足够，省约 1/3。 |
| 指纹（0a） | 每组参数在 Modal CPU 容器取一次 | 已尝试（L4 探针 `yeto-s17-a5-argprobe`），learner 在打印指纹前要和同步服务握手，探针卡住 420 s 超时，三次共约 $0.15；改为从正式运行的磁带读取 attestation 指纹 | 指纹是记录项，不是判据。 |

## 3. Miles / Megatron / yeto 初始化断言逐条对照（本形状：每岛 trainer 1 卡 DP1、rollout 1 引擎 1 卡 + 1 张备用卡）
| # | 位置 | 条件 | 本次 | 结论 |
|---|---|---|---|---|
| 1 | Miles arguments.py global batch | rollout_batch×n_samples/steps = global batch | 4×8/1 = 32（`--groups-per-round 4 --samples-per-group 8 --optimizer-steps 1`） | 通过 |
| 2 | over_sampling ≥ rollout_batch | 4 ≥ 4 | 通过 |
| 3 | prompt_len ≤ context−1 | seq 1024，回答 384，gsm8k 提示约 100–250 词元 | 通过 |
| 4 | p2p / disk-delta 不支持 LoRA | 走 broadcast（默认） | 不触发 |
| 5 | use_rollout_logprobs 与 use_tis 互斥；get_mismatch_metrics 需自定义 TIS | 都未开 | 不触发 |
| 6 | kl_coef 与 kl_loss_coef 不能同时非 0 | 都未设 | 不触发 |
| 7 | offload / colocate | fixed-partition、不 offload | 不触发；yeto `check_elastic_miles_args` 另外拒绝 colocate/offload/缺 Miles router 三种情况 |
| 8 | `--use-miles-router` | `--rl-elastic` 固定带上（ae42dcf） | 通过（dry-run 断言） |
| 9 | Megatron world % (TP·PP·CP) | trainer 1 卡，TP1 PP1 → DP1 | 通过 |
| 10 | Megatron global % (mbs×DP) | 32 % 1 | 通过 |
| 11 | Megatron 层数 % PP、seq % TP、max_position | 28 % 1；1024 % 1；1024 ≤ 40960 | 通过 |
| 12 | yeto run_config：seq ≤ 模型最大位置；world 整除；strict 下 rollout_batch×n == global×steps | 32 == 32×1 | 通过（dry-run 走真实 `resolve_rl_run_config`） |
| 13 | yeto fixed-partition 布局 | trainer bundle 0，c0 bundle 1（启动），c1 bundle 2（声明不启动） | 通过（dry-run 断言） |
| 14 | sglang mem-fraction ∈ (0,1) | 0.4（L40S 48 GB，0.6B 模型） | 通过 |
| 15 | fork `start_cells` 只认启动时声明的 cell、epoch 不符拒绝 | up 的 expected_config_epoch 0、down 1、finalization 用例 2 | 与 journal 推进一致 |
| 16 | pause 审计（pause_audit.py:163）：strict 预算 = quorum × margin（默认 margin 0.5），再与空闲流超时取小 | 基线/切换不设 quorum（无外层上限）；quorum 用例 120×2.0 = 240 s ≥ deadline 230 s | 通过（读码） |
| 17 | 同模型同数据同批次在本栈上跑过 | s15 两岛（Qwen3-0.6B LoRA r16、gsm8k、4×8、384/1024、seed 17）；F-E1（3×L40S T1R1S1 elastic） | 已有实测 |

## 4. 已知风险（不放宽判据）
- quorum 用例：plan-3.8-4.4-v2 §1 第 5 条写"up 请求的 deadline 设为 230 s（≤ 240 s 预算）"。up 事务含 150 s 注入延迟 + 新引擎启动（L40S 上约 91 s，memory 记录）+ 发布与校验，**很可能超过 230 s**，事务会走 watchdog → REBUILT_OLD。判据第 2 条只要求"该轮至少 1 次 PULL 重发、岛 1 不退出、roster 不变、该轮完成、无 `rl_strict_failure`"，不要求 up 成功，所以 up 的终态只记录、不作判据。若预算计算与上面的假设不同导致 plan 阶段被拒（第 2 条无从执行），如实记为"未执行"，不改参数重跑超过一次。
- 切换用例：up 在第 2 轮 train 期间提交，等到第 2 轮发布后的安全点执行；新引擎在 L40S 上约 91 s 启动。deadline 600 s。
- Modal 容器内 sidecar 用 `modal container exec` 从本机注入（与 A4 的 inwatch 同一做法）。若注入失败，切换用例无请求，按"未执行"处理并立刻停机。

## 5. 运行安排、费用与上限
| 序 | run id | 内容 | 期望 | 硬上限 |
|---|---|---|---|---|
| 1 | s17-a5-base-20261008a | 两岛 6 轮，无请求 | ≈25 min × $11.9/h ≈ $5 | HARD 45 min → $9.1 |
| 2 | s17-a5-switch-20261008a | up（r1 train）→ down（r3 train）→ 最后一轮 up（r5 train，finalization 应拒绝/取消） | ≈$5.5 | $9.1 |
| 3 | s17-a5-quorum-20261008a | quorum 120 / margin 2.0 / 延迟 150 s，up（r1 train，deadline 230 s），4 轮 | ≈$5 | $9.1 |
| | 合计 | | ≈$16 | 最坏 ≈$27.5（含 head 和探针） |

顺序执行（先基线，基线启动正常再开另两个；另两个可以并行，因为它们不共享任何资源，并行时线程先核 <2500）。脚本：`s1-runs/s17-a5-remote.sh`（由 s15-island1b-remote.sh 派生，去掉杀岛/冻结，加 sidecar）。每次运行结束：Modal app 停止、Nebius head 删除，按名字核对（`final-applist.json`、`final-skystatus.txt`）。

## 6. 这次顺带采集什么（按 gpu-metrics-checklist / gpu-run-value-max）
- 两岛完整事件磁带（rl-island-0/1.jsonl，含 rl_timeline_span、rl_driver_phase、rl_reconfiguration、rl_member_publication、rl_pull_resend、rl_resource_sample、心跳）；syncer 磁带 yeto-tape.jsonl 与 syncer 日志；head 作业日志。
- 岛 0 的 elastic-state（journal、epochs.json、inbox、status）每 15 s 快照一次，结束后解包。
- 每张卡每 2 s：利用率、显存、功率、计算进程（用于"新引擎只在 c1 的卡上出现、down 后该卡释放"的观察和显存峰值）。
- Miles router `/worker_inflight` 每 0.5 s（cordon/drain 过程中在途请求数的真实曲线；3.3b 的计数第一次在真机留下时间序列）。
- 切换成本分段：wait_safe / drain / init（新引擎启动）/ verify / publish / resume，按 journal 与 timeline 计算；L40S 上 SGLang 引擎启动时间（memory 里 H100 145 s vs L40S 91 s 的差异原因待查，这里再记一组）。
- 每轮生成/训练时长、回答长度分布、截断率、reward、组数/样本数、`trained_sample_ids_sha256`、`sync/global_policy_hash`、publication token。
- 冷启动：容器分配、镜像拉取、模型加载、Ray/引擎就绪各段（从 launch.ts.log 和磁带时间戳）。
- 容量与价格实况：卡型、分配用时、失败原因。
- 原始数据全部先落盘到 `s1-runs/<run>/`，判读脚本只读这些文件。

## 7. 运行中修订（17:05Z）
- `s17-a5-base-20261008a`（3×L40S）在 Modal 排队约 17 分钟始终 0 个容器（L40S 无货），只花了 head VM 约 $0.1；已停 app、删 head（final-applist：stopped/0；sky：not found）。
- 改回 §9.15 原定的 `H100!:3`（每岛），其余不变。6×H100! = $23.7/h；每次期望约 25 min ≈ $10，三次期望 ≈ $30，最坏（HARD 45 min）≈ $54。超出 $25 的部分按用户"稍超可以"处理；如果基线实际时长明显超过 30 min，quorum 用例与切换用例改为二选一，先报告主 agent。
- 17:51Z：基线 b 实测开机约 17 min（17:23 提交 → 17:40 第一轮），每轮约 3 min，整次约 40 min ≈ $16。三次会到约 $48，按上一条预案改为**两次**：基线 + **合并运行**（切换 + finalization + quorum 放在同一次）。
  - 合并运行参数：`--rl-elastic-quorum-timeout-s 120 --rl-test-inject-start-delay-s 150`（判据第 2 条的两个数字不变），`--rl-elastic-pause-margin 4.0`（预算 120×4.0 = 480 s），up 的 deadline 450 s（≤ 预算），down 与 finalization 的 deadline 600 s。触发点同切换用例：up 在第 2 轮 train、down 在第 4 轮 train、最后一轮 train 中再提交 up。
  - 为什么 margin 改成 4.0：plan-3.8-4.4-v2 §1 第 5 条选 2.0/230 s 是为了让第 2 条"能执行"，当时 up 的成败不在第 2 条里；合并后同一个 up 还要满足第 1 条"终态 SUCCEEDED"，150 s 延迟 + H100 上新引擎约 145 s 启动 + 发布校验会超过 230 s，所以把预算放大到能容下整个 up。两个判据数字（quorum 120 s、延迟 150 s）都没改，第 2 条要观察的现象（同一轮至少 1 次 PULL 重发、岛 1 不退出、roster 不变、该轮完成、无 `rl_strict_failure`）不受 margin 影响。
  - 判读：第 1、3、5 条与基线比对；第 2 条在 up 所在轮看 syncer 磁带与岛 0 learner 磁带 `rl_pull_resend`。基线没有 quorum 超时参数，这一项只影响 syncer 等待策略，不影响样本 id，第 1 条的比对仍然成立。
  - HARD 55 min（最坏 ≈ $21.7）。
- 19:1xZ：合并运行 a（`s17-a5-merged-20261008a`，≈$7.8）**浪费**：启动脚本漏传 `--rl-elastic-attestation`，岛 0 的 up 在 plan 阶段被拒（"no capability attestation: no transition is certified"），dn1/fin1 随之因 epoch 不符被拒，全程无事务。属于脚本缺陷，不是被测功能的问题。plan-3.8-4.4-v2 §2 写明 attestation 指纹是前置条件，本复核 §2 只把"取指纹"当记录项处理，漏了"运行时必须带 attestation 才能做任何切换"。
  - **教训**：上卡前把启动脚本最终参数逐项和复核文档、执行说明（plan-3.8-4.4-v2 §1 前提清单）对一遍，前置条件（attestation、resources、cells、router 等）每一项都要在脚本里找到对应参数，并在 PLAN_ONLY 输出里打印出来核对。
  - 重跑 b：attestation = 岛 0 指纹 sha256:0d0f7779…（基线与合并 a 的 `rl_driver_start` 逐字相同；elastic 附属参数不进 Miles argv，所以加 attestation 不改变指纹），PLAN_ONLY 已核对两岛运行脚本都写入 `~/yeto-rl/elastic_attestation.json` 并带 `--rl-elastic-attestation`。主 agent：I2 上限放宽到 $40；b 若再因脚本/参数失败，先停下报告，不再直接重跑。
- 19:40Z：合并运行 b（`s17-a5-merged-20261008b`，≈$10.3）结果，逐项分开写（判读 `s1-runs/s17-a5-merged-20261008b/judgment.json`，脚本 `s1-runs/s17-a5-judge.py`）：
  - **第 2 条（quorum / PULL 重发）：通过。** up1 在第 3 轮前的安全点暂停（pause_decision rollout_id 2，outer_phase round-boundary-published，allowed，stalls_peers，budget 480 s），注入 150 s 后 start_cells，事务共 367 s，终态 SUCCEEDED。其间岛 0 磁带有 3 次 `rl_pull_resend`（global_step 3，pulls_received 2/3/4）；岛 1 没有退出（有 `rl_learner_finalized`）；syncer 每步 roster 都是 {0,1}，与基线相同；该轮两岛都完成；没有 `rl_strict_failure`；没有 "conflicting/invalid PULL permit"；每步每岛只有一次 PUSH。另：step 1 两岛各有 1 次 PULL 重发（开机快慢不一，quorum 120 s 下出现，基线不设 quorum 所以没有），不计入本条。
  - **第 1 条（样本/步数/策略/roster 不变）：部分通过。** (b) 两岛 6 轮 `trained_sample_ids_sha256`、组数、样本数与基线逐轮全等；(c) 每岛每轮恰好一次训练步（train_step 1..6）；(d) 每个 policy_version 两岛 `sync/global_policy_hash` 相同，紧随其后的 publication token = `yeto:<v>:<hash>`；(e) roster 全程 {0,1} 与基线相同。(a) up1 SUCCEEDED，**dn1 没有执行**：被 pause 审计拒绝，原因 "expected pause 600s exceeds budget 480s"——我给 dn1 的截止 600 s 超过了 margin 4.0 下的预算 480 s（参数错误，见下）。所以"up 和 down 两个事务都 SUCCEEDED"这一项未测到。
  - **第 3 条（finalization 拒绝）：未测到（受第 1 条 dn1 连带）。** fin1 在最后一轮 train 中提交，但因 dn1 未执行、配置 epoch 停在 1，fin1 先在 epoch 检查被拒（"expected config epoch 2, current is 1"），没有走到 finalization 判断。基线和合并 a 两次都看到 stop 边界写入 `finalization` 记录（rollout_id 6），但没有在那时有待处理的请求，不能作为本条证据。
  - 第 4 条：未单独做空闲流探测（见 §2）；quorum 用例中连接未丢，岛 1 正常完成。第 5 条：报告注明仅完成 rollout 能力。
  - **第二个参数错误与教训**：把 margin 从 2.0 改到 4.0 时，只核对了 up 的截止（450 ≤ 480），没有核对 down 和 finalization 请求的截止（600）也必须 ≤ 预算（pause_audit 以请求 deadline 作为预计暂停时长）。教训同上一条：改任何影响前置检查的参数后，要把**每一个**请求体逐个过一遍检查（epoch 链、deadline ≤ 预算、attestation 覆盖的边）。
  - 补测方案（待主 agent 决定）：同一脚本，只把 dn1、fin1 的截止改为 450 s，其余不变，再跑一次合并运行（期望约 26 min ≈ $10.5）。I2 累计会到约 $44，超过 $40 上限。
- 20:10Z：补测 c（`s17-a5-merged-20261008c`，≈$9.2）**浪费**：请求预检全过（`s1-runs/s17-a5-merged-20261008c/preflight.json`），6 轮跑完，但岛 0 容器里的注入 sidecar 在开始后约 29 s 消失（GPU 采样只有 14 条，停在 19:46:57Z；日志为空，无异常输出），三个请求一个都没投递，所以 down 与 finalization 仍未测到。原因未查明：前三次同样方式注入的 sidecar 都活到运行结束；岛脚本里的 `stop_miles_ray` 只杀 `~/miles-ray/` 路径的进程，不匹配。
  - 教训：容器内注入工具不能只"装一次"，要有存活检查。修法（未实施，待主 agent 定）：本机 arm 循环每 60 s 用 `modal container exec` 查 sidecar 是否存活，死了就重启；sidecar 启动时读回 `s17-inwatch-<岛>.jsonl` 里已投递的请求，避免重复投递；第一轮 train 之前若 sidecar 不在，直接停机，不白跑。
  - 顺带所得：这是第三次无事务的同参数运行（与基线、合并 a 一起），两岛逐轮样本 id、步数、全局哈希与基线全等，说明本形状下样本与策略链在多次运行间完全可重复。
- 20:35Z：主 agent 决定改用"本机投递"：不再在容器里放常驻 sidecar，由本机脚本 `s1-runs/s17-a5-deliver.py` 按岛 0 磁带进度用 `modal container exec` 直接写 inbox 文件（临时文件再改名）、读回核对、再轮询直到岛把请求取走（request 文件消失或出现 status 文件），任何一步失败或 420 s 内没取走就立刻停机；进度改为用 exec 读岛 0 自己的磁带（见下条线索，head 汇总日志对岛 0 不可靠）；每次确认后与每 60 s 用 exec 打包一次 elastic-state 存到本机。
  - 上卡前小检查 `yeto-s17-a5-execcheck`（Modal CPU 1 核，≈$0.01）：exec 写 inbox → 读回 5 次，全部一致；写入每次 1.24–2.29 s，读回 1.23–1.34 s。结果 `s1-runs/s17-a5-execcheck/`。
  - **sidecar 消失原因：未查明**，已知线索留待以后查：(1) 补测 c 中 sidecar 在 19:46:28Z 开始写日志，19:46:57Z 后再无输出，日志文件为空，没有 Python 异常；(2) 同一时刻之后，岛 0 的标准输出也不再出现在 head 汇总日志里（c 的 head 日志里岛 0 只有 1 条事件，岛 1 有 370 条；基线与合并 b 两岛都是约 400 条），但岛 0 自己的磁带完整（6 轮全有）；(3) 岛脚本的 `stop_miles_ray` 只杀 `~/miles-ray/` 下的进程，不匹配 sidecar；(4) 前三次（基线 b、合并 a、b）同样方式注入的 sidecar 都活到结束。猜测（未验证）：该容器在 19:46:57Z 前后 exec 会话或日志转发通道被重置，连带杀掉了 exec 启动的进程组。
  - 这次只判 down 与 finalization；第 1 条其余四项与第 2 条已在合并 b 通过，不重复判定（判读脚本仍会输出这些项，只作记录）。

## 8. 结论（20:55Z，补测 d `s17-a5-merged-20261008d`，判读 `s1-runs/s17-a5-merged-20261008d/judgment.json`）
本机投递：三个请求都在触发后约 7 s 写入并读回一致，约 95 s 后（下一安全点）被岛取走（`deliver.jsonl`）。逐项：
- **第 1 条：通过。** up1（第 3 轮前，T1R1S1→T1R2S0）SUCCEEDED，dn1（第 5 轮前，T1R2S0→T1R1S1）SUCCEEDED（TRANSFERRING→COMMITTED 5 s）；两岛 6 轮样本 id/组数/样本数与基线逐轮全等；每岛每轮一步；每个版本两岛全局哈希一致、token 链正确；syncer roster 全程 {0,1} 与基线相同。（合并 b 已独立通过除 down 外的各项，本次再次全部满足。）
- **第 2 条：通过**（合并 b 与 d 两次都满足）：up 所在轮岛 0 有 3 次 PULL 重发（global_step 3），岛 1 不退出，roster 不变，该轮完成，无 strict 失败。
- **第 3 条：通过。** fin1 在最后一轮 train 中投递，被岛取走后在 stop 边界写入 `finalization`（rollout_id 6）并转 CANCELLED，错误 "finalization refuses reconfiguration"；config_epoch 保持 2（epochs.json），成员仍为 1 个引擎。
- 第 4 条：未单独做 30 分钟空闲流探测（理由见 §2：head 在 Nebius 公网，非本机 NAT）；三次带 quorum 的运行中连接都没有丢。若评审要求按原文补探测，3.8 的勾选应撤回。
- 第 5 条：仅完成 rollout 能力（trainer 边不在本项）。
- plan-3.8-4.4-v2 §2 观测：两个执行的事务 pause_decision 都是 round-boundary-published / allowed / stalls_peers / budget 480 s（= 120×4.0）；每步每岛一次 PUSH；没有 conflicting/invalid PULL permit。
- 顺带数据：切换成本——up 共 328 s（注入 150 s + 新引擎启动与发布约 176 s，VERIFYING→SUCCEEDED 2 s）、down 5 s；每轮训练步 13–17 s，首轮约 33 s；回答长度均值约 130 词元；GPU 显存峰值约 35.6 GB/卡（0.6B，mem-fraction 0.4）；四次运行（基线、合并 a/c 无事务、d 有事务）逐轮 reward 与截断率完全相同。原始数据：各 run 目录的 tape-direct、head、es-snapshots、deliver.jsonl。
- 费用：I2 合计 ≈ $53.5（上限 $56），其中浪费 ≈ $17（合并 a 漏 attestation、c 注入工具故障），合并 b 一半有效。
