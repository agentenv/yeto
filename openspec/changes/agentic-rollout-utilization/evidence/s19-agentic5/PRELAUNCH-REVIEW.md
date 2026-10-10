# S19 第三批 #7 + #8 上卡前复核（agentic-rollout-utilization 5.5 判据 5 + rl-spot-cost-saving 4.4）

日期 2026-10-10。子 agent AGENTIC5。代码：agentenv/main c9ca5321（含 #176 打分路由键、#196 沙箱 Secret 旗标、#199 在途轨迹限时保存），运行检出 /home/michael/work/s19-agentic5-run（detached，干净）。镜像：main pin `yeto-miles-ports@sha256:fa2413be…`（Miles 64b591a4b，已核 ddce20992 挂起改动是其祖先，arguments.py 含 `--agentic-suspend-between-turns`）。脚本 /home/michael/work/s1-runs/s19-agentic5/run.sh（由 s18-aru3-ab.sh 改）。

## 1. 上卡内容（主 agent 10-10 拍板）
| 岛 | 运行目录 | 设置 | 卡 |
|---|---|---|---|
| A | s19-agentic5-a-20261010a | M1：多发 12、上限 1，seed 17，8 轮，不回收 | Modal H200!:1，CPU 16，内存 128 GiB |
| B | s19-agentic5-b-20261010a | 同 A，seed 18；**第 6 轮（rollout_id=6）挂起日志出现后 8 s 执行一次 `modal container stop`**（真实回收信号） | 同上 |

两岛互相独立（各自 Modal app，单岛不同步），同时开。共同形状与 S18 ARU-3 M1 相同：Qwen3.5-4B（851bf6e8）LoRA r16 attention，上下文 12288 / 回复 6144，每轮 6 题×4 条，lr 1e-5 固定，TB2 46 题，codex_openenv + 会话服务 v1，判分 modal_provider，SECRLENV_MAX_TURNS 12，沙箱 TTL 1800 s、空闲超时 900 s，规格 aru3-m1-age1.spec.json（TIS[0,2] + entropy 0.01 + staleness 1）。只跑 M1 臂：MB/M0 上限 0，没有跨版本 token，对判据 5 没有数据（主 agent 同意）。
新增：`--modal-sandbox-secret yeto-sandbox-modal`；`--modal-env YETO_SPOT_INFLIGHT_SAVE_DIR=/yeto-tape/inflight-<run>`、`YETO_SPOT_INFLIGHT_SAVE_VOLUME=yeto-event-tapes`。PLAN_ONLY 已过：learner 命令行含 `--rl-max-policy-age 1`、多发 12、lr constant；Modal 配置 H200:1 gpu_exact、两个 INFLIGHT 环境变量、named_secrets=('yeto-sandbox-modal',)；46 题提示解析通过。

## 2. 代码复核
1. #7 查因（CPU，已完成）：S18 M1 的 48 条未打分样本 = 路由器拒收不带 `X-SMG-Routing-Key` 的打分请求。证据：镜像路由器 radixark/sgl-router-for-miles `header_utils.rs::missing_routing_key_response`（manual/consistent_hashing 且未开 `--allow-requests-without-routing-key` 时返回 4xx）；M1 launch.log 48 行 `smg::response request failed with client error`（每轮 16/8/8/8/8，与未打分数一致）；同日志引擎访问记录 0 条 `/generate`。修复 #176（70d4ca79）：打分请求带样本自己的 key，否则带固定 key `yeto-cross-version-score`；manual 策略对新 key 走 Vacant 分配，单引擎时必到同一引擎。
2. 4.1 保存（#199）：每轮 `rl_round_trained` 后调用 rollout 执行器内的 `export_in_flight`（Ray `__ray_call__`，带超时），写 `rehearsal-r<k>.json`（tmp+fsync+rename+目录 fsync）再 commit 卷，打印 `[yeto] rl_inflight_save {...}`（export_s/write_s/commit_s/total_s/entries）。回收：Modal 函数进程把 SIGINT/SIGTERM 只转发给学习器 pid（pid 文件），`ModalExitHandler`（25 s 上限，last_save_s = 上一轮演练耗时）调用同一保存写 `reclaim-<ts>.json`，打印 `[yeto] spot_reclaim {...}`。超时只留 .tmp。
3. 已知风险：①信号处理器只能装在学习器主线程（与 spot 5.2 的可丢弃岛同一位置，那次 handler 生效）；②回收后学习器继续训练直到平台强杀，B 的 rc 预计非 0（--modal-retries 0、--no-island-relaunch），不影响判读；③agentic 挂起轨迹只写引用（设计：不能在新容器续跑），文件很小，耗时主要是卷 commit；④新镜像 fa2413be 没跑过 agentic 路径（只过了 T4 镜像检查），起机失败记失败、修一次脚本类故障。
4. Miles 初始化断言：与 S18 ARU-3 复核 §3 逐条相同（同一参数形状；64b591a4b 相对 ddce20992 只增加算法补充/critic 与 over-sampling 统计），不重复列出。显存：与 ARU-3 M1 同形状同卡型（峰值约 126–130 GB / 141 GB）。

## 3. 判据（上卡前固定，不放宽）
有效性：A rc 0 且 8 轮完成；A、B 各至少 3 轮 `resumed_groups>0`。
#7（判据 5）：
- 5a：A 全部轮、B 回收前全部轮中，凡 `cross_version_tokens>0` 的轮 `cross_version_unscored_samples == 0`。不为 0 → #7 失败，保全证据（含 `cross_version_unscored_reasons`）。
- 5b：比值分位数 `cross_version_ratio_p50/p90/p99` 在 A 至少 3 轮有值。
- 5c：没有任何一轮 `cross_version_truncated_fraction ≥ 0.5`（不触发回退）。
- 标定建议规则（预登记）：令 F = 两岛各轮截断比例的最大值。F < 0.1 → 维持告警 0.2 / 回退 0.5（余量 ≥2 倍）；0.1 ≤ F < 0.2 → 维持，并注明余量不足 2 倍；F ≥ 0.2 → 阈值不改，记"真实运行会触发告警"，交主 agent 定。另报每轮 p50/p90/p99/min/max 与训练端 tis_clipfrac。
#8（4.4）：
- 8a：A、B 每轮演练副本都有 `rl_inflight_save`（kind=rehearsal，无 error），报 total_s 的 p50/p90/max 与 commit_s；全部 ≤ 25 s。
- 8b：B 在第 6 轮停容器后，日志有信号转发行与 `spot_reclaim`（outcome=saved，handler_s ≤ 25 s），且有 kind=reclaim 的 `rl_inflight_save`，total_s ≤ 25 s。
- 8c：卷 yeto-event-tapes 上 `inflight-s19-agentic5-b-20261010a/reclaim-*.json` 存在、可解析，条目数等于打印的 entries。
- "回收后续跑节省的重做 token 比例"：**未验证**（agentic 挂起轨迹按设计丢弃，4.2/4.3 本次不做）。
数值健康另报：grad_norm 有限、奖励无 NaN、无 rl_invariant_failed、沙箱获取失败数。

## 4. 费用与安全
$6.32/h（H200 + CPU16 + 128 GiB）。每岛 HARD 3000 s + 看门狗余量 240 s，最坏 $5.69；两岛最坏 $11.4 ≤ 上限 #7 $15 + #8 $5。预计 A ≈$4.2、B ≈$3.5。超上限立刻停。
开卡：持 GPU-LAUNCH.lock 到容器出现，后台子进程全部关 fd 9；锁内再查线程 <2800。看门狗 setsid（不含令牌）；结束后 modal app stop，拉卷上 tape 与 inflight 目录。原始数据上传 yeto-evidence-archive 并登记 ARCHIVE-MANIFEST.tsv。

## 5. 结果（运行后追加）
- 第 1 次（`s19-agentic5-a/b-20261010a`，代码 c9ca5321）**失败**（rc=4，0 轮）：两岛都在 Codex preflight 的 `validate_hmac_key_source` 报 `UntrustedTBenchOutcome`。根因是 #201 修的缺陷（sky.Task 把 secrets 存成 SecretStr，岛上拿到 "**********"），c9ca5321 不含 #201。容器时长 a 02:42:48–02:47:09Z、b 02:45:29–02:50:04Z，约 $0.94 [估算]。两个 app 已 stopped。证据 s1-runs/s19-agentic5-{a,b}-20261010a/launch.log。
- 第 2 次：代码改为 agentenv/main 655992f9（含 #201），判据、轮数、回收轮次不变，run id 后缀 20261010b。
- 第 2 次（`s19-agentic5-a/b-20261010b`，代码 655992f9，02:58–03:40Z）。判读 s1-runs/s19-agentic5/judgment.json（analyze.py 同目录）。
  - 有效性**通过**：A rc=0、8 轮，7 轮续跑；B 5 轮续跑（B 第 6 轮训练中被停，rc=4，按预期）。
  - 5a **通过**：A 第 1–7 轮、B 第 1–5 轮 `cross_version_unscored_samples` 全为 0（S18 是每轮 8–16）。跨版本 token 全部打分（A 每轮 2.9 万–6.8 万，B 2.2 万–7.0 万）。
  - 5b **通过**：A 7 轮有分位数。p50 全是 1.0；p90 1.002–1.015；p99 1.093–1.117；max 1.28–1.78；**min 每轮都是 0.0**。
  - 5c **通过**：各轮截断比例都是 0.0（TIS[0,2]：没有比值 >2，下界 0 不截断）。F=0 → 按预登记规则建议维持告警 0.2 / 回退 0.5。
  - 待查（主 agent 10-10 指示，不上卡）：比值最小值 0.0 = 至少一个 token 的 exp(当前 − 生成) 下溢，即当前 logprob 比生成时低 745 以上或为 −inf。tape 只存分位数，没有 token 级数据，CPU 上无法定位是哪些 token、是否在续跑段首。已在 PR #212 加诊断字段（比值 <1e-6 的计数与最多 8 个例子：位置、是否版本段首、token、loss_mask、两边 logprob），随 8b 补跑一起拿数据。
  - 8a **通过**：14 次演练（A 8、B 6）全部成功，total_s 中位约 1.4–1.7 s、最大 1.91 s；export <0.01 s、写文件 <0.002 s，其余是卷 commit（1.25–1.90 s）。每次 5–11 条挂起轨迹，只写引用，约 2 KB。
  - 8b、8c **失败（证据不全）**：03:31:42Z container stop，日志有 `signal 2: forwarded to the learner 1130`，之后没有 `spot_reclaim`，卷上没有 reclaim 文件；旧容器日志 03:32:03Z 停止（信号后约 21 s）。疑似原因（未查实）：Python 信号处理只在主线程执行，主线程在训练调用里。修复见 PR #212（标记文件 + 守护线程 + 每步打印）。
  - 新发现：container stop 后 Modal 在新容器（03:32:36Z 起）从头重跑 B。原因已查实：`modal container stop` 按设计把在跑输入改派到其他容器（CLI 帮助原文），与 --modal-retries 无关。函数调用日志流没有送来新容器的行，ContainerIdGuard 没触发。子 agent 先拉日志（app-logs-after-stop.txt）再 03:39:30Z 手动 app stop。PR #212 加了不依赖日志流的容器守卫。
  - "回收后续跑省下的重做 token 比例"：**未验证**（agentic 挂起轨迹按设计丢弃，4.2/4.3 本次不做）。
  - 花费 [估算]：第 1 次 ≈$0.94；第 2 次 A 37.9 min ≈$3.99、B 39.1 min ≈$4.12（含重跑容器约 7 min）；合计 ≈$9.05 / 上限 $20。
  - 原始数据：Modal Volume yeto-evidence-archive `/s1-runs/s19-agentic5/*.tar.gz`（4 个运行目录 + 脚本与 inflight 文件），本地 s1-runs/s19-agentic5-*。
