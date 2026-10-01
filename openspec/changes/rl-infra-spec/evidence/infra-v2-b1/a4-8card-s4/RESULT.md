# A4 / A4b 8 卡 H100 批 结果（2026-10-01，代码 b19b781，计划 gpu-plan-v2 §9.23）——中间版本（等新 SHA/新镜像后继续）

平台 Nebius eu-north1 8×H100 on-demand（$30.8/h）。每项只给四态：通过 / 不通过 / 测试无效 / 未运行。tasks.md 勾选状态未改动。历史失败记录（`../a4-4card/RESULT.md`、`../a4s3/RESULT.md` 等）原样保留，不覆盖。

## 总结
| 项 | 状态 | 依据 |
|---|---|---|
| 冒烟（测量，无验收结论） | 完成 | up SUCCEEDED 144.3 s、`start_cells` 142.4 s（≤160 s 停止线）、down 5.3 s；查出 cleanup 闭环缺陷（已修，base/e1a/e1b 真机 cleanup 退出码 0） |
| E1-A 基线 base | 完成（e1a 的比较基线） | 12 轮 rc=0 |
| **E1-A（e1a）** | **通过** | (a)(b)(c)(d)(e)(f)(g)+E1-E 全部满足；(c) 首次判读 FAIL 为 judge 单位缺陷，勘误后重判（原始输出保留） |
| **E1-B（e1b）** | **不通过** | 注入真实发生（applied=true），但终态 SUCCEEDED，无 payload_mismatch；疑为读回校验不覆盖 LoRA adapter（待代码确认），不重跑 |
| watchdog | 未运行 | 见"阻塞" |
| A4b | 未运行 | 同上 |
| E1-D d123/d4/d5/d6/d7 | 未运行 | 同上 |

## 阻塞与剩余
- 主 agent 通知：integ-decl 已到 06a754bf，LoRA 成员读回校验改为 fail-closed，需要重建镜像（sglang a1240c530，用户已批准，进行中）并改 pin；之后统一在新 SHA + 新镜像上重取 attestation 指纹、重核各开关 argv，按 wd → A4b → E1-D → E1-B（验证修复，预期 PAYLOAD_MISMATCH → REBUILT_OLD）运行，用复用链，本批剩余预算按 $40。在此之前不上卡。
- 费用：本批累计 ≤$60.0（含主 agent 误叫停的 $5.7，见"费用"），上限 $100。

## 冒烟 infra-v2-b1-a8sm-20261001-1（runs/smoke/）
- 平台断言 8×H100 通过，指纹 `sha256:1c4ceeb1…` 与 rl_driver_start 一致，router 采样器与 fork 探针自检通过。
- **阶段耗时（launch 起算，s）**：sky 实例创建（Launching→Instance is up）≈ 302；拉镜像/起容器（→Docker container is up）≈ 521（至 02:34:38）；Cluster launched 863；job submitted 870；岛首条日志 897；首个 generate 1187；up 事务 144.3（其中 start_cells 142.4，fork_op start issued→VERIFYING）；down 5.3；Job finished 1395；launcher 拆除 1412→1534（122 s）。整次 1534 s = 25.6 min，≤$13.1。→ 每次运行底价 ≈ $12–13.5，与轮数关系不大。
- **cleanup 闭环缺陷（冒烟查出，已修）**：自动 cleanup 的退出码 141（cleanup.out 只有 phase 1 一行）。根因：`nstop.sh` 用 `cleanup_run.sh | tee <run dir>/cleanup.out`，tee 的命令行含 `/<prefix>/`，被 cleanup 阶段 1 的前缀进程扫描 SIGTERM，cleanup_run 随后写管道 SIGPIPE。修复：改为 shell 重定向；`tests/test_nstop.sh` 回归测试（旧版复现 rc=141，新版 rc=0）。真机验证：base、e1a 的 cleanup_rc=0，cleanup.out 四阶段齐全，两次复查（间隔 60 s）干净。冒烟本身的 cleanup 用手动 `cleanup_run.sh` 补做：退出码 0（runs/smoke/cleanup_manual.out 在运行目录，已复制）。
- **采集缺口（已修）**：岛随 job 结束被 launcher 拆除，router/gpu 采样、inwatch 日志没拉到；n2run puller 加了周期性小文件包（`samplers.tgz.b64`），base 起生效。

## base infra-v2-b1-a8base-20261001-1（runs/base/）
12 轮，rc=0，无请求。1405 s，≤$12.0。作为 e1a 的 sample-id 基线。cleanup_rc=0。

## E1-A e1a infra-v2-b1-a8e1a-20261001-1（runs/e1a/）——通过
注入/请求真实发生：触发器在 train rid1 提交 up1、rid4 重复提交同一 id、rid6 提交 dn1（`samplers/inwatch.out`）；两事务 SUCCEEDED（up 总 146.3 s，start_cells 143.9 s；down 5.3 s）。1578 s，≤$13.5。
| 判据 | 结果 | 依据 |
|---|---|---|
| (a) 两事务 SUCCEEDED，config_epoch 0→1→2 | 通过 | analysis.json a |
| (b) 12 轮 sample-id 哈希与基线相等，每轮 1 个 optimizer 步 | 通过 | analysis.json b，differ=[] |
| (c) 成员数序列（rl_membership）+ 缩容后首次发布成员数 | **通过**（见勘误） | judgment.json：`[2,2,4,4,4,4,4,2,2,2,2,2]` 精确相等；缩容后首次 rl_publication（pv8）2 成员 |
| (d) trainer PID / G0–G3 UUID 全程不变 | 通过 | 16 次采样唯一 |
| (e) 池外 GPU 无新进程 | 通过 | 36 次采样，池外 0 |
| (f) down：QUIESCING→TRANSFERRING→stop_cells | 通过 | |
| (g) 备用卡 GPU-hours | 记录 | 8 × 26.3 min |
| E1-E | 通过 | 重复提交同一 request_id 同一 tx（status 文件 tx-0-…-up1，journal 仅 1 条 up1 request）；WAIT_SAFE 边界 up1=rollout 2、dn1=rollout 7；ledger 12 轮每类恰 1 条 |
**(c) 原始 FAIL 与勘误**：第一次判读（`judgment.v1-FAIL.json`，`judge.v1.out` 原样保留）输出 FAIL，`members_per_round=[null,4,4,4,4,4,2,2,2,2,2,2]`，且 `publication_after_down_is_1_member=false`。原因是工具缺陷而非产品：(i) 磁带的 `rl_membership.round` 是 0-based rollout_id（up=2、down=7），judge 当成 1-based 轮次（off-by-one）且第一个事件之前没有初始成员；(ii) "缩容后发布成员数"写死成 4 卡的 1。§9.22/§9.23 里"`round ≤ r`"是笔误，应与 §9.18 的 rollout_id = r−1 一致；勘误在判读前写入 §9.23 并连同修复提交（a4960bb），回归测试用真机 e1a 磁带片段，断言新读法 PASS、旧读法 FAIL。预期序列与判据文字未改。

## E1-B e1b infra-v2-b1-a8e1b-20261001-3（runs/e1b/）——不通过
- **第一次启动 a8e1b-20261001-2（runs/e1b-attempt2-provision-failed/）= 平台开通失败**：06:20:25–06:26:45 实例 STOPPED/Reconciling 6.3 min，超出 sky 的等待上限（"Failed to wait for instances ... after max retries 3"），sky 终止实例；未起岛。≤$3.3。cleanup 退出码 0。无脚本/判据缺陷，仅同配置重试一次（-3），成功（实例 up 用时 3.6 min）。同一天 8×H100 其余开通：5.0 / 约 5 / 约 5 min。
- 配置：`--rl-test-inject-lora-perturb 0.01 --rl-test-hold-before-check-s 10`，4 轮，train rid1 提交 up→T4R4S0。指纹 `sha256:210f27dc…` 与岛一致。
- **注入真实发生**：tape 与 journal 均有 `test_injection(kind=lora_perturb, applied=true, scale=0.01, target_members=[c2,c3], phase=VERIFYING)`；launcher 日志有 "TEST INJECTION YETO_RL_TEST_INJECT_LORA_PERTURB … ships a LoRA adapter perturbed by 0.01"；随后 update_weights 走 LoRA adapter 注册/加载（`register_lora_adapter`、"LoRA adapter loading from tensors completes"）。
- **(a)** hold 窗口（`test_hold` stage=end：10.05 s）内 19 个 router 样本；新 worker URL 在窗口内**不在 router 工作列表中**（`inflight` 只有原 2 个，`cordoned=[]`），admit 之后（+~2 s）才出现 4 个；因此没有请求能路由到新 cell（满足"新 cell in-flight=0"，但 router 里并没有"新 cell 在 cordon 列表中"这一字面现象——新 cell 在 admit 之前根本未被注册进 router；judge 的谓词"无忙碌的新 URL"在 new_urls 为空时是空真，如实记录）。
- **(b) 不通过**：终态 `SUCCEEDED`（up 160.4 s，其中 start_cells 145.8 s + hold 10 s），**没有** REBUILD_OLD/REBUILT_OLD，`cause` 为空。`check_weights(action=checksum)` 照常运行（WeightChecker "checksum computed for 226 tensors"），放行了被扰动的 adapter。
- **根因（后经 integ-decl 06a754b 的代码说明证实："The stock checksum covers only base weights"，LoRA 模式下 adapter 键需要打了 yeto/lora-checksum 补丁的 sglang a1240c530 才会输出）**：226 = 28 层×8 个融合张量 + embed + final norm，即 Qwen3-0.6B 的**基础权重**；被扰动的是 LoRA adapter（引擎经 `load_lora_adapter` 以流式张量加载，不在这 226 个被检张量里）。因此读回校验对 adapter 载荷不敏感：要么是"产品的载荷读回校验不覆盖 LoRA adapter"（与主 agent 通知的 06a754bf 改 fail-closed 一致），要么是"该注入在 LoRA 下对被检张量无效"。judge 的规则只区分"注入未发生/未施加=测试无效"与"施加了但终态不对=不通过"，故判**不通过**（产品缺陷：b19b781 的成员准入读回校验在 LoRA 模式下对 adapter 无感）；不原样重跑，06a754b + 新镜像到位后再验证（预期 PAYLOAD_MISMATCH → REBUILT_OLD）。注意：06a754b 改了成员准入路径，E1-A 的 up 事务也要走它——E1-A 在 b19b781 上的通过是针对 b19b781 的，是否需要在新 SHA 上重跑由主 agent 裁定（预算 $40 有限）。
- 费用 ≤$12.4（06:29:36–06:53:40）。cleanup_rc=0。

## 阶段耗时（launch 起算；证据：各 run 的 `scan.json`/`scan.md`、`launch.ts.log.gz`）
| 阶段 | smoke | base | e1a | e1b(-3) |
|---|---|---|---|---|
| sky 实例创建（Launching→Instance is up） | 302 s | — | — | 215 s |
| 拉镜像/起容器（→Docker container is up） | 521 s | — | — | — |
| Cluster launched | 863 s | 828 s | 858 s | 740 s |
| 岛首条日志 | 897 s | 862 s | 890 s | 773 s |
| 首个 generate | 1187 s | 1155 s | 1183 s | 1062 s |
| Job finished | 1395 s | 1274 s | 1456 s | 1287 s |
| launcher 拆除（teardown→run finished） | 122 s | 127 s | 101 s | 129 s |
| 整次 launch 墙钟 | 1534 s | 1405 s | 1578 s | 1444 s |
| 每轮（首轮后） | ≈7 s（首轮 ≈21 s） | 同 | 同 | 同 |
| up 事务总耗时 / start_cells | 144.3 / 142.4 s | — | 146.3 / 143.9 s | 160.4 / 145.8 s（含 hold 10 s） |
| down 事务 | 5.3 s | — | 5.3 s | — |
结论：每次运行从 launch 到首个 generate ≈ 17.5–20 min（开通 + 拉镜像 ≈ 14 min + 岛启动 ≈ 5–6 min），整次 ≈ 23–26 min，**与轮数几乎无关**，所以每次运行底价 ≈ $12–13.5；这是决定上"复用链"的依据。

## 集群复用链（chain8）——写好并通过 CPU 桩测，真机未验证
- 目的：省掉每项重复的实例创建（5.0 min）+ 拉镜像（8.7 min）+ 容器/setup（0.8 min）≈ 14.5 min ≈ $7.4/项。设计与登记见 gpu-plan-v2 §9.23"集群复用链"。工具：`scripts/chain8.sh`、`reset_island.sh`、`nstop_item.sh`，`n2run.sh`/`a8go.sh` 增加 `RUN_ROOT/CLUSTER_PREFIX/KEEP/SHARED/NSTOP`；`tests/test_chain8.sh`（27 项检查）、`tests/test_reset_island.sh`、既有 `test_cleanup/test_nstop/test_selfcheck/test_judge_inject` 全部通过。
- 07:03:53 启动链（wd 冷启动，b19b781），07:14 主 agent 通知"先不要上卡"（该通知针对 06a754bf，而链用的是 b19b781——主 agent 事后确认是其误叫停）。我按指示立即 SIGTERM 链：trap 触发 `cleanup_run.sh`，sky down 完成；cleanup 第一次退出码 2（check 2 看到 2 个瞬时进程残留 842634/842636，随后消失），立即第二次 `cleanup_run.sh` 退出码 0（两次复查干净，`runs/chain-aborted/`）。实例 07:08:50 up、岛未起，**无实验结果**，复用方案**未在真机验证**。费用 ≤$5.7（07:03:53–07:15:02），台账原因记"主 agent 误叫停"。

## 观察到的异常（smoke/base/e1a/e1b 全部日志的扫描汇总；每条：来源 · 频率 · 是否影响判读 · 初步判断）
扫描脚本 `scripts/scan_run.py`（模板归并；原始表见各 run 的 `scan.md/scan.json`，类别为启发式、此处已人工复核）。频率按"每次运行"的发生次数（smoke / base / e1a / e1b）。

**A. 产品或产品相邻（需要关注）**
| 现象 | 来源 | 频率 | 影响判读 | 初步判断 |
|---|---|---|---|---|
| E1-B：LoRA adapter 被扰动后读回校验仍放行（SUCCEEDED，无 payload_mismatch）；check_weights 只含 226 个基础权重张量 | e1b launch.log + journal | e1b 1 次 | **是（E1-B 不通过）** | 产品（读回校验不覆盖 adapter）或注入对被检张量无效；与主 agent 通知的 06a754bf fail-closed 修复吻合；待新镜像验证 |
| 引擎被 stop（down 事务/收尾）时 Miles 的 process 监控把子进程 SIGTERM 退出当作崩溃："Subprocess detokenizer/scheduler crashed with exit code -15 → Triggering SIGQUIT for cleanup"、"SIGQUIT received. It usually means one child failed"、"did not exit within 5.0s after SIGTERM; escalating to SIGKILL"、"Killed actor …" | launch.log（island） | 3 / 2 / 4 / 4 次（e1a/e1b 比 base 多 2 次 = 缩容/扩容事务里被停的 cell） | 否（down SUCCEEDED、ledger 完整、无重复消费） | 上游 Miles 对"有意停掉的 cell"的反应过激（日志噪声）；产品侧 stop_cells 是 SIGTERM 5 s 后升级 SIGKILL，需确认这对 cell 内 sglang 子进程是否总是干净；不影响本批判据 |
| launcher 拆除："N instance(s) still live after down; retrying teardown (1–3/4)" + "sky.down … Cluster does not exist" + "WARNING: could not confirm termination of [cluster]; leaving the head up so they stay reachable" | launch.log（launcher） | 4+1 / 1 / 4+1 / 3 次 | 否（实例确实随后消失，cleanup 两次复查干净，台账费用按墙钟上界计） | 平台（Nebius 异步删除延迟）+ launcher 的重试窗口偏短；warning 会让 launcher 声称"保留 head"而实际已被 sky 删除，措辞误导；建议 launcher 拆除确认窗口放宽 |

**B. 上游/环境噪声（不影响判读）**
| 现象 | 来源 | 频率 | 判断 |
|---|---|---|---|
| post-warmup freeze_gc 对已停引擎 `ConnectionRefused`（Traceback：urllib3 MaxRetryError → requests ConnectionError → "post-warmup freeze_gc failed"） | launch.log | 4 / 2 / 4 / 4 次 | 上游 SGLang/Miles：收尾时对正在被停的引擎调用 `/freeze_gc`；非产品逻辑 |
| InferenceController 启动等待重试 `wait_init_expected_num_cells` attempt=…、`retry_until_deadline gate_url=…` | launch.log | 28+10 / 28+4 / 28+10 / 28+10 行 | 平台/预期：新引擎起来之前的轮询；up 事务多出的 retry_until_deadline = start_cells 等待 |
| Miles FT check "decision=no_retry reason=survivors normal" | launch.log | 3 / 12 / 12 / 4 | 上游 Miles：每轮一次的健康检查记录，良性 |
| 导入期告警：megatron.bridge muse_glimmer / miles_plugins.qwen3_vl CP-local vision / trust_remote_code / argparse 类型推断 / transformer_engine Jax RuntimeWarning / sky nebius `user_agent_prefix` FutureWarning | launch.log、selfcheck_probe.txt、autostop.out | 每次 35–40 行 | 上游/环境 |
| HF Hub 未认证请求告警 | launch.log | 每次 2 行 | 平台/环境（无 HF_TOKEN；下载未受限） |
| sky autostop "Cluster(s) failed …" | autostop.out | 每次 1 行 | 平台：集群已被 launcher 拆除后 puller 仍调用 autostop，无害 |

**C. 测试工具与基础设施事件（均已处理或记录）**
| 现象 | 来源 | 频率 | 影响判读 | 判断 |
|---|---|---|---|---|
| cleanup 闭环缺陷：`| tee <run>/cleanup.out` 被 cleanup 阶段 1 的前缀进程扫描杀死 → SIGPIPE，cleanup_run 退出 141 | smoke（最终守卫） | 1 | 否（冒烟本身无验收结论） | 测试工具；已修（改重定向）+ `tests/test_nstop.sh` 回归（旧版复现 141、新版 0）；base/e1a/e1b/链 真机 cleanup 均 0 |
| router/gpu 采样、inwatch/dkill/dctl 日志没有被拉到（岛随 job 结束被 launcher 拆除） | smoke pulled/ 空文件 | 1 | 否 | 测试工具；已修（puller 周期性拉小文件包 `samplers.tgz.b64`，nstop 回退取用） |
| judge_inject e1a_c 单位缺陷（rl_membership.round 为 0-based rollout_id；缩容后发布成员数写死 4 卡的 1） | e1a judge.out（`judgment.v1-FAIL.json`） | 1 | 是（首次判读 FAIL，重判 PASS） | 测试工具；勘误 + 回归测试（真机磁带片段，旧读法 FAIL）先提交再重判 |
| 用户线程防护 `abort: 3008 user threads`（≥3000） | e1b 第一次启动 | 1 | 否（未起云资源） | 环境：共享 sky API server 的 executor 池 + VS Code server 线程长期 ≈3000；ulimit -u=4096；按用户裁定重启 sky API server（线程 2999→2374）后继续，门槛不变 |
| Nebius 分配延迟：实例 STOPPED/Reconciling 6.3 min 超过 sky 等待上限 → 开通失败 | e1b 第一次（sky provision.log） | 1（本批另有 4×L40S 的 §9.22 开通失败，`a4-4card/RESULT.md`） | 否 | 平台容量/排队；重试一次成功；今天其余 8×H100 实例创建 3.6–5 min |
| ssh "Connection timed out during banner exchange" / "Connection to UNKNOWN port 65535 timed out" | diag.err | 2 / 8 / 4 / 6 次 | 否（diag 采样丢一两个周期） | 平台：head 节点 ssh 偶发超时；工具均有超时与重试，未影响 puller/selfcheck |
| dmesg 在容器内不可读（"read kernel buffer failed: Operation not permitted"），OOM/Xid 无法从容器内采集 | diag/dmesg.txt | 每次 | 否 | 平台/容器权限；nvidia-smi 全量+进程表快照（每 ~45 s）已保存；本批所有运行未见 OOM 迹象（无进程异常退出、无 CUDA OOM 日志） |
| inwatch.log/dkill.log/dctl.log 周期拉取为空（inwatch 的输出在 inwatch.out） | pulled/ | 每次 | 否 | 测试工具文件名差异；`samplers/inwatch.out` 有触发记录，已用 |
| 链级 cleanup 第一次退出码 2：check 2 看到 2 个瞬时进程残留（pid 842634/842636），随后消失 | runs/chain-aborted/final_cleanup.out | 1 | 否（无云资源残留，第二次 0） | 测试工具：cleanup 的进程扫描可能匹配到 `sky down` 的子进程/executor 的瞬时进程；需要时给扫描加"存活 >N 秒才算残留"的去抖 |
| 本批 py-spy dump 未触发（所有运行都没有 tape 240 s 无新事件的卡住） | — | 0 | — | 采集脚本已备（`diag_pull.sh`），未验证 py-spy 在岛上是否可装 |

## 费用（本批上限 $100；$0.5133/min；上界按墙钟）
| 运行 | 起止（UTC） | 费用 | 备注 |
|---|---|---|---|
| a8sm 冒烟 | 02:20:47–02:46:21 | ≤$13.1 | 25.6 min |
| a8base | 02:52:30–03:15:55 | ≤$12.0 | 23.4 min |
| a8e1a | 03:18:37–03:44:55 | ≤$13.5 | 26.3 min |
| a8e1b-2（开通失败） | 06:20:16–06:27:00 | ≤$3.3 | 实例从未 RUNNING |
| a8e1b-3 | 06:29:36–06:53:40 | ≤$12.4 | 24.1 min |
| a8ch（链，主 agent 误叫停） | 07:03:53–07:15:02 | ≤$5.7 | 仅 wd 冷启动阶段 |
| **合计** | | **≤$60.0** | 剩余 $40 |

## cleanup 证据
- smoke：自动 cleanup 退出码 141（缺陷，见上）；手动 `cleanup_run.sh` 退出码 0（`runs/smoke/cleanup_manual.out`）。base/e1a/e1b-3/e1b-2：`cleanup_rc.txt`=0，`cleanup.out` 四阶段齐全、两次复查（间隔 60 s）干净。链：见上。
- 每次收尾后用 `nebius compute instance list --parent-id project-e00eqrj3pr00622zrgdeyc` 与 `sky status` 复查：只有他人实例（rlf-h200-pilot-03、rlf-h200-qualification-02、yeto-rl84f-head-2aebdd34-8456-head、cyberrl-verl-smoke-20260923，均 STOPPED，未触碰），sky 无集群。
