# rl-inter-island-scheduling progress

## 2026-10-07 S15 子 agent（分支 s15-interisland，基线 fd37129e）
- 步骤 A：change 建立（proposal/design/spec/tasks）。
- 步骤 B（fef9a314 + 本提交的 harness 修正）：阶段 0 实现。
  - `yeto/rl/engine/island_ledger.py`：`StalenessPolicy`:44（修正名须属 `CORRECTION_MECHANISMS`，去掉 observe）、`SampleGroup`:59、`DeltaEntry`:70（留 wire_dtype/shard/extra 接口）、`merge_weight`:91（=merge.rs c_tokens²/c_steps）、`CrossIslandLedger` join/leave/heartbeat/expire_leases/judge/quorum_size/submit/weight_of/try_advance（:140–:233）。
  - `yeto/rl/engine/controller.py`：`IslandStatus` 新增 pool_epoch 与 7 个调度字段（:259 起，全 Optional）、`SCHEDULING_FIELDS`:269、构造参数 `scheduling_probe`:300、`inspect()`:1202 用 journal `replay_pool` 填 pool_epoch，探针只取非 None 已知键，探针异常不影响 inspect（`_scheduling_fields`:1225）。
  - `yeto/rl/engine/journal.py`：`POOL_TX_KINDS`:237、`replay_pool`:248、`append_pool_event`:270（kind="pool"，tx_kind=pool_join/pool_leave/pool_epoch；pool_epoch 与 membership_epoch 倒退抛 EpochConflict）。
  - `yeto/rl/engine/pause_advice.py`：`PauseAdvice`:19、`merge_pause`:32（只收紧）、`lease_veto`:49。
  - `yeto/rl/engine/fake_islands.py`：spawn 多进程假岛 + 协调器（`run_fake_islands`:74），写 `tape.json` 与 journal。
- 步骤 C：CPU 验收（本机，无 Ray：PYTHONPATH 前置 `/tmp/s15-noray` 让 `import ray`/`import miles` 直接 ImportError，作为防误起 Ray 的护栏）。
  - 命令：`PYTHONPATH=/tmp/s15-noray /home/michael/work/miles-next-venv/bin/python -m pytest -p no:cacheprovider -q tests/test_rl_inter_island_{ledger,status,harness}.py tests/test_rl_controller_events.py tests/test_rl_controller_trainer_edge.py tests/test_rl_reconfig_{d4,recovery,e1,x6}.py tests/test_rl_pause_audit.py tests/test_rl_e1_injections.py tests/test_rl_trainer_rebuild_e1.py`
  - 结果：**146 passed, 0 failed**（新测试 17 条；harness 用例连跑 3 次均通过）。日志 `infra-drafts/s15-interisland-evidence/pytest-stage0.log`。
  - 假岛演示（fast / crash(1 轮后静默退出) / slow(work 1.2s > quorum_timeout 0.8s) / late(v≥2 加入)，6 轮，ratio=1.0，lease 0.5s）：最终 outer_version=6、membership_epoch=5、members={fast,late,slow}；crash 租约过期退岛，`dropped_uncommitted={c_tokens:1000,c_steps:4,outer_version:0}`；slow 6 轮全部 timed_out 缺席、3 次 delta `stale_base` 被拒；late 首轮（base_version=2）raw weight 0，之后 250000；样本判定 ACCEPT 20 / ACCEPT_IS 1 / REJECT(outer_lag_exceeded) 4；journal 重放 members 与 membership_epoch 与账本一致。证据 `infra-drafts/s15-interisland-evidence/fake-run/{tape.json,journal/}`、`fake-run-summary.txt`。
  - 观察（非失败）：ratio=1.0 时一个常驻慢岛使每轮都等满 quorum_timeout——这正是 design Q4 推荐 "比例 r<1 + q_min" 的理由；默认值待用户裁定。
- 未完成：0.8 syncer Rust 协议扩展（D-S1..S6）；阶段 1/2 GPU（待批）；driver batch 来源未接跨岛样本（只有判定，无存储/传输）；PauseAdvice 尚未接入 controller 的 pause 判定调用点（只有纯函数与测试）。

## 2026-10-07 夜：折入用户裁定（P8a+P9、P4、P5、P3、P6、G-a、lag≤2、P7 接口、HMAC 仍待定）
- 代码：`island_ledger.py` 默认 `max_outer_lag` 1→2；固定比例 quorum 改为 P4 算力加权（`theta`，`capacity_arrived`）+ 软截止（`timed_out`=T_soft）；迟到增量由 `stale_base` 拒收改为 `delta_carried_over`（γ^lag 折扣并入下一轮，`carry_lag_exceeded` 超限才拒）；退岛同时丢弃 carried 增量并记 dropped_uncommitted；新增 `syncer_epoch` 与 `check_fence`（P9）。默认 θ=0.75、q_min=1、γ=0.5、max_carry_lag=2 [待真机校准]。`fake_islands.py` 参数 quorum_ratio→theta/gamma。
- 测试：同一命令（`PYTHONPATH=/tmp/s15-noray`，不拉 Ray）**149 passed, 0 failed**，连跑 2 次；新增用例：默认策略、算力加权提前步进、carried_over 折扣与超限拒收、syncer_epoch fencing；harness 断言改为 slow 岛增量被 carried_over 并入（不再出现 stale_base）。日志 `infra-drafts/s15-interisland-evidence/pytest-stage0-r2.log`。
- 新任务：0.10 M2PO vs TIS/IcePop 离线比较；0.11 P6 降级 advice；0.12 P3 索引格式草案。阶段 1 待批不预登记。

## 2026-10-07 夜：模式开关（legacy 默认 / elastic 显式）
- Python 侧 `IslandSchedulingMode` 落地（账本、journal、IslandController、假岛）；legacy 下 pool_* 只读、重放忽略。
- 测试：同一命令 154 passed, 0 failed（2 次），日志 infra-drafts/s15-interisland-evidence/pytest-stage0-r3.log。

### 0.10 离线比较 M2PO 与 TIS/IcePop（lag 1–4）：数据不足，未比较（2026-10-07 夜）
- 查了什么：`/home/michael/work/s1-runs/`（845 项，36G）下的 tape 与事件文件。找到的与概率有关的数据只有三类：
  1. 每轮汇总的标量：`train_rollout_logprob_abs_diff`、`train/train_rollout_kl`（例：`s11-h200-20261005o-c17/pulled/rl-island-0.jsonl`，每轮 1 个数；`s1-mn-20261003c-g12/.../rl-algo-mismatch-correction/evidence/2026-09-29-g1/runs/{tis,icepop,opsm-trainer,observe}` 的 `mismatch_metrics`）。它们比的是**同一版本权重**下推理引擎与训练端的差异（陈旧度 0），不是隔 1–4 个外层步的策略差异。
  2. `rl_harness_mismatch`（75 条）：模板 / token 段数不一致的诊断，不含概率。
  3. `rl_load_sample`（1229 条）：引擎负载，不含概率。
- 没有找到逐 token 的产生时对数概率与"若干外层步之后的策略"在同一批 token 上的对数概率，所以无法算出 lag 1–4 的重要性采样比（两者之差的指数）分布，也就无法比较 M2PO 与 TIS/IcePop 的截断比例和方差。**不编造结果；`StalenessPolicy.correction` 默认暂留 `tis`，0.10 不勾。**
- 需要采集的字段（下次两岛或单岛真机顺带存，不额外开卡）：每条样本的 `island_id`、`outer_version`、`inner_step`、`policy_hash`、`group_id`、逐 token `behavior_logprob`（产生时）与 `loss_mask`；以及在外层版本 v+1..v+4 的权重上对**同一批 token** 重新前向得到的逐 token 对数概率（训练端算，可在每次外层合并后对固定的 64–256 条留存样本做一次只前向的计算）。有了这些即可离线算：比值分布分位数、TIS 截断比例（按现有 `tis_clip`/`tis_clip_low`）、IcePop 掩掉比例、M2PO 的二阶矩约束下被掩比例与有效样本数。

## 2026-10-07 夜：0.9 / 0.11 / 0.12 完成，契约字段名与 Rust 侧对齐
- 0.9 fb0d2a4b、0.11 1ca9c4ca、0.12 53e196b0；0.10 360da6ae 只记录"数据不足"（未勾，已改派他人）；0.8 改派他人，本分支未改 syncer/。
- 契约字段名：island_scheduling_mode / quorum_theta / carry_gamma / soft_deadline_s / syncer_epoch。
- 测试：同一命令加 tests/test_rl_inter_island_sample_pool.py，**162 passed, 0 failed**（2 次），日志 infra-drafts/s15-interisland-evidence/pytest-stage0-r4.log。

## tasks 0.8 Rust 侧进展（分支 s15-interisland-rs，2026-10-07）

- 已实现并通过单测（`cd syncer && cargo test`：129 通过、0 失败）：
  - `syncer/src/main.rs`：命令行参数 `--island-scheduling-mode legacy|elastic`（默认 legacy）、`--quorum-theta` 0.75、`--carry-gamma` 0.5、`--soft-deadline-s`（缺省取 `--quorum-timeout-s`）、`--q-min` 1、`--max-carry-lag` 2、`--island-hmac-key`（缺省读环境变量 YETO_ISLAND_HMAC_KEY，elastic 下必填）。
  - `syncer/src/server.rs`：`Config.island_scheduling`、`Config.island_hmac_key`；`semantic_profile_hash` 末尾追加模式段。legacy 追加零字节，原有 Python 黄金哈希向量测试不变即证明 legacy 哈希逐字节不变；elastic 追加 `yeto-syncer-island-scheduling-v1\0`、字段名 `island_scheduling_mode`、模式串与五个参数，因此不同模式的 HELLO 由现有的哈希比对直接拒绝。elastic 目前在 `run()` 开头拒绝启动（状态机未接入连接循环）。
  - `syncer/src/protocol.rs`：消息号 15 JOIN、16 JOIN_ACK、17 LEAVE、18 LEASE_HEARTBEAT、19 SAMPLE_INDEX。legacy 学习者循环原样对未知类型报错，即"不加载新消息"。
  - `syncer/src/elastic.rs`（新文件）：帧编解码（首字段 syncer_epoch，末尾 HMAC-SHA256，覆盖消息号与正文；HMAC 手写，用 RFC 4231 用例 2 校验）、`ElasticCoordinator`（照 Python `CrossIslandLedger` elastic 分支：防旧实例、加入 / 退出 / 租约过期、算力比例步进与 q_min、软截止空转、迟到增量按 γ^lag 并入下一轮、超过 max_carry_lag 拒收、退岛丢弃未提交增量并记 dropped_uncommitted、新岛首轮权重 0）。
- legacy 等价证明：`git diff` 对 server.rs、protocol.rs、main.rs 只有新增行、无删除行；原有 117 项测试全部照旧通过（含哈希黄金向量）。
- 未完成：状态机接入服务器（真正处理新消息与外层步进）、SAMPLE_INDEX 处理（只定义了帧）、D-S5 状态导出、D-S6 检查点字段、与 Python 假岛 tape 的黄金比对、Python `syncer_profile.py` 需按上述字节布局追加 elastic 段才能与 Rust 哈希一致（未做，属 Python 侧）。

### 0.8 续：elastic 接入服务器（同分支，2026-10-07）

- 新文件 `syncer/src/elastic_server.rs`：elastic 模式下 `server::run` 不再拒绝，而是改走独立的控制面服务器（legacy 永远不进这个分支）。每个连接循环收 JOIN / LEAVE / LEASE_HEARTBEAT / DELTA_READY，先验 HMAC、再按 syncer_epoch 防旧实例，失败回 MSG_ERROR 并保留连接；JOIN 回 JOIN_ACK。50 毫秒定时器负责租约过期、软截止计时（超时用 timed_out 步进或记 round_idle）、到 total_steps 退出。tape 事件按 Python 账本的 kind 名逐行写入 `--event-tape`（JSONL）。
- 新增消息 20 DELTA_READY（只带基版本与 c_tokens / c_steps，不带张量）；递交增量同时续租。
- 新命令行参数：`--island-lease-s`（默认 30）、`--syncer-epoch`（默认 0，D-S6 应改为从检查点递增）。两者未编入契约哈希。
- 验证：`cargo test` 130 通过、0 失败；集成测试 `elastic_server::tests::three_islands_late_expiry_and_midrun_join` 连跑 5 次均过（3 岛，θ=0.6；旧 epoch 被拒；第 0 轮缺 3 号岛照样步进；3 号岛迟到增量按 0.5 折扣并入第 1 轮；3 号岛租约过期，tape 记 dropped_uncommitted；4 号岛中途加入 catch_up=true，首轮权重不为满额）。真实二进制 `--island-scheduling-mode elastic` 能启动监听；缺密钥时拒绝启动。
- legacy：相对改动前基线 360da6ae，server.rs / protocol.rs / main.rs / state.rs / merge.rs 删除行为 0。
- 仍未做：elastic 轮的张量合并与广播（elastic 服务器目前只做成员与步进决策）、SAMPLE_INDEX 处理、D-S5、D-S6、Python tape 黄金比对、Python `syncer_profile.py` 的 elastic 哈希段。

### 0.8 续：elastic 张量合并（同分支，2026-10-07）

- 新消息：21 ELASTIC_INIT（岛提交初始扁平 f32 参数，先到者生效）、22 DELTA_TENSOR（θ − 基版本 + c_tokens / c_steps，与 PUSH_FRAGMENT 同号约定，服务器取负成外层梯度）、23 ELASTIC_BASE（每次步进后广播新基版本给全部成员；加入时也会收到当前基版本）。都带 syncer_epoch 与 HMAC。
- `elastic_server.rs` 的 `Shared::advance`：按 StepPlan 归一化权重（迟到增量含 γ^lag、新岛首轮 0、退岛 / 租约过期的增量在 purge_departed 时丢弃且不进计划）调用 `merge::merge_avg`，再 `merge::nesterov_step`（`--outer-lr` / `--outer-momentum`）。merge.rs 本身未改。有基版本后再发只有元数据的 DELTA_READY 会被拒；计划里缺张量视为致命错误，服务器退出。
- 与旧版的差别（有意为之、未接）：不用旧版的 layout / 分片 / 多流帧，不做按张量的 RDA / ISO 和 HeLoCo 修正，参数是一条扁平 f32 向量；现有 Python 学习者还不会说这套 elastic 消息。
- 验证：`cargo test` 131 通过、0 失败；`elastic_server` 两个集成测试各连跑 5 次均过。新测试 `tensor_rounds_match_hand_computed_weights`：3 岛 4 维张量跑 3 轮，outer_lr=1、momentum=0，基版本依次为 1.5、3.9、4.9，与手算一致（第 1 轮迟到岛按 0.5 折扣并入；第 2 轮过期岛的 100 未合并、新岛的 1000 权重 0）。
- legacy：相对基线 360da6ae，server.rs / protocol.rs / main.rs / state.rs / merge.rs 删除行仍为 0。
- 下一位：学习者端（Python）接入 elastic 消息、D-S5、D-S6（syncer_epoch 与成员 / 账本持久化）、SAMPLE_INDEX、Python tape 黄金比对、按分片合并是否需要（与 legacy 合并质量对齐）。
## 0.13 launcher 接线
- 测试：同一套 + tests/test_rl_inter_island_{sample_pool,launcher}.py + tests/test_rl_launcher.py + tests/test_rl_infra_switches.py：**307 passed, 0 failed**；日志 infra-drafts/s15-interisland-evidence/pytest-0.13.log。

## 契约哈希与 Rust 对齐、HMAC 密钥下发
- `yeto/syncer_profile.py`：可选字段 island_scheduling_mode / quorum_theta / carry_gamma / soft_deadline_s / q_min / max_carry_lag；legacy（或缺省）不追加字节，原黄金哈希 b904a25c417a24deef77b8526c4e982a13a35c65d331a4cafb74e579b584d4b7 不变；elastic 追加 island-scheduling 段。
- **Rust 黄金比对值**：在 tests/test_rl_sao_streaming_runtime.py::_syncer_profile 的配置上加 elastic 默认参数（quorum_theta=0.75, carry_gamma=0.5, soft_deadline_s=900, q_min=1, max_carry_lag=2）→ sha256 = **4b61bb37c3dfafce169058a26f4e76dd1fcaad257c115ea468133916a24560b8**（单测 tests/test_rl_inter_island_contract.py 固定）。
- 契约字段名：`contract_fields` 只产出 island_scheduling_mode 等，单测确认无 `island_scheduling` 键；learner/entry 内部配置键 `island_scheduling` 是 build_elastic 的参数名，不进入契约。
- launcher：elastic 时从启动环境读 YETO_ISLAND_HMAC_KEY，在 syncer 命令前 export（缺失即拒绝）；legacy 无变化。
- 测试：tests2 套件（原套件 + sample_pool/launcher/contract + test_rl_launcher/infra_switches/sao_streaming 三个契约相关文件）**356 passed, 0 failed**（2 次），日志 infra-drafts/s15-interisland-evidence/pytest-contract.log。test_delta_protocol.py 需在本工作树编译 syncer（cargo 不在 PATH，且本分支不应写 syncer/），未纳入。

## 0.14 岛侧 elastic 客户端
- 实现：yeto/rl/elastic_client.py、yeto/rl/bridge.py（追加 ElasticRlBridge / make_island_bridge / flatten_state / unflatten_state；StrictRlBridge 未改）。
- 黄金帧来源：把 s15-interisland-rs 101a172c 的 syncer/ 源码复制到 /tmp/s15-noray/rs（只读原工作树），加一个只打印 ElasticMsg::encode(b"k1") 结果的测试后 cargo test，得到 15–23 九条帧的十六进制；tests/test_rl_inter_island_elastic_client.py 逐字节比对并互相解码，另含 RFC 4231 case 2。
- 端到端（真实 Rust 二进制，CPU，本机 127.0.0.1）：同一副本 cargo build 后以 --island-scheduling-mode elastic --quorum-theta 0.75 --carry-gamma 0.5 --soft-deadline-s 30 --q-min 1 --max-carry-lag 2 --island-lease-s 6 --total-steps 3 启动；两个 Python 岛（ElasticRlBridge + 假 runtime，每轮参数 +1）各跑 3 轮：syncer 记录 2 次 pool_join、6 次 delta_accepted、3 次 outer_step（每次两岛到齐、未超时）、2 次 pool_leave（requested，dropped_uncommitted 为空），syncer 退出码 0，两岛错误为空、应用的基版本均为 [0,1,2,3]、最终参数一致。证据：infra-drafts/s15-interisland-evidence/e2e-rust/{result.json,syncer-tape.jsonl,syncer.log,island-0.jsonl,island-1.jsonl}，脚本 infra-drafts/s15-interisland-evidence/e2e-rust/e2e.py。
- 测试：tests3 套件 **373 passed, 0 failed**（2 次），日志 infra-drafts/s15-interisland-evidence/pytest-0.14.log。
- 未做：0.15（Miles learner / ports 引擎路径接 ElasticRlBridge）。

## 0.15 ports 引擎接 ElasticAvgSync、syncer 任期号接线
- 代码：yeto/rl/engine/bridges.py（新增 ElasticAvgSync）、miles_adapter/entry.py（build_sync 按 yeto_rl_island_scheduling 选择）、yeto/rl/miles.py（elastic 下报错，legacy 分支不变）、yeto/rl/learner.py（--rl-syncer-epoch；elastic 时设置 yeto_rl_island_scheduling / yeto_rl_syncer_epoch）、yeto/cli.py + launcher.py（--rl-syncer-epoch，只限 elastic，传 syncer --syncer-epoch 与 learner --rl-syncer-epoch）。
- 测试：tests/test_rl_inter_island_ports.py（假 syncer 单岛 2 轮；build_sync 模式选择；真实 Rust 二进制两岛 2 轮，需 YETO_TEST_ELASTIC_SYNCER，否则跳过）。命令：`PYTHONPATH=/tmp/s15-noray YETO_TEST_ELASTIC_SYNCER=/tmp/s15-noray/rs/target/debug/yeto-syncer YETO_TEST_ELASTIC_OUT=infra-drafts/s15-interisland-evidence/e2e-ports python -m pytest -p no:cacheprovider -q $(cat /tmp/s15-noray/tests4.txt)`（tests3 + ports + test_rl_engine_driver + test_rl_critic_dual_syncer）：**423 passed, 0 failed**（2 次），日志 infra-drafts/s15-interisland-evidence/pytest-0.15.log。
- 端到端证据（syncer_epoch=5，两岛各 2 轮）：infra-drafts/s15-interisland-evidence/e2e-ports/syncer-tape.jsonl——2 次加入、4 次增量接受、2 次外层合并（两岛到齐、未超时）、2 次主动退出；两岛最终 LoRA 张量相同，syncer 退出码 0。二进制来自 /tmp/s15-noray/rs（s15-interisland-rs 101a172c 源码副本，临时目录，非仓库产物）。
- 未做：任期号自动从 syncer 状态读取（现为显式参数）；旧 Miles 桥接路径不支持 elastic；critic 家族在 elastic 下直接拒绝。

## 交接（2026-10-07 夜，给下一位 Python agent；本 agent 停止实现）
- 分支 s15-interisland，worktree /home/michael/work/s15-interisland，交接时 HEAD 见本条 commit（上一实现 commit 3040bd10）。工作区干净。
- 剩余任务（tasks.md）：0.16 旧 Miles 桥接路径支持 elastic（待批）；0.17 launcher 从 syncer status.json 自动读 syncer_epoch；0.18 IslandStatus 从 syncer status.json 读调度字段（经 IslandController 的 scheduling_probe）；0.19 SAMPLE_INDEX 线路对齐（policy_hash 发 32 字节原始 sha256、island_id 为 u32，SampleIndexEntry ↔ elastic_client.SampleIndex 转换）；0.10 离线比较已改派；0.8/0.8a Rust 在 s15-interisland-rs（勿改本分支 syncer/）；阶段 1/2 GPU 待批。
- 护栏（本机禁止拉起 Ray）：
  `mkdir -p /tmp/s15-noray/ray /tmp/s15-noray/miles && echo 'raise ImportError("ray blocked on this host (S15 rule)")' | tee /tmp/s15-noray/ray/__init__.py > /tmp/s15-noray/miles/__init__.py`，所有 pytest 前置 `PYTHONPATH=/tmp/s15-noray`。不要跑 tests/test_rl_harness_mismatch_tape.py；tests/test_delta_protocol.py 会在本工作树编译 syncer，也不要跑。
- 测试命令：`cd /home/michael/work/s15-interisland && PYTHONPATH=/tmp/s15-noray YETO_TEST_ELASTIC_SYNCER=/tmp/s15-noray/rs/target/debug/yeto-syncer YETO_TEST_ELASTIC_OUT=/home/michael/work/infra-drafts/s15-interisland-evidence/e2e-ports /home/michael/work/miles-next-venv/bin/python -m pytest -p no:cacheprovider -q $(cat /tmp/s15-noray/tests4.txt)`；上次结果 423 passed。tests4.txt 内容（一行，空格分隔）：
```
tests/test_rl_inter_island_ledger.py tests/test_rl_inter_island_status.py tests/test_rl_inter_island_harness.py tests/test_rl_controller_events.py tests/test_rl_controller_trainer_edge.py tests/test_rl_reconfig_d4.py tests/test_rl_reconfig_recovery.py tests/test_rl_reconfig_e1.py tests/test_rl_reconfig_x6.py tests/test_rl_pause_audit.py tests/test_rl_e1_injections.py tests/test_rl_trainer_rebuild_e1.py tests/test_rl_inter_island_sample_pool.py tests/test_rl_inter_island_launcher.py tests/test_rl_inter_island_contract.py tests/test_rl_launcher.py tests/test_rl_infra_switches.py tests/test_rl_sao_streaming_runtime.py tests/test_rl_miles_sao_streaming.py tests/test_tbench21_sao_streaming_contracts.py tests/test_rl_inter_island_elastic_client.py tests/test_rl_inter_island_ports.py tests/test_rl_engine_driver.py tests/test_rl_critic_dual_syncer.py
```
- 端到端 syncer 二进制（只读复制 Rust 工作树，不在其中编译）：
  `rm -rf /tmp/s15-noray/rs && mkdir -p /tmp/s15-noray/rs && cp -r /home/michael/work/s15-interisland-rs/syncer/{Cargo.toml,Cargo.lock,src} /tmp/s15-noray/rs/ && cd /tmp/s15-noray/rs && ~/.cargo/bin/cargo build`（cargo 已用户态安装在 ~/.cargo/bin，1.99.0）。未设置 YETO_TEST_ELASTIC_SYNCER 时真实二进制用例自动跳过。
- 黄金帧更新方法：在 /tmp 副本的 src/elastic.rs 末尾加一个打印 `ElasticMsg::encode(b"k1")` 十六进制的测试（消息同 Rust 测试 all_frames_roundtrip...），`cargo test golden_dump -- --nocapture`，把输出替换 tests/test_rl_inter_island_elastic_client.py 的 RUST_GOLDEN。契约哈希 elastic 默认黄金值 4b61bb37…（tests/test_rl_inter_island_contract.py）。
- git 身份：GIT_AUTHOR/COMMITTER_NAME=MichaelChung，EMAIL=michaelchung668@gmail.com，提交末尾 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`；不 push。

## 2026-10-07 夜（接续的 Python agent）：0.17、0.18、0.19 完成
- 0.18（commit 01e85a52）：新增 `yeto/rl/engine/island_status.py`，读 syncer 写在 `--event-tape` 同目录的 status.json（schema `yeto.syncer.elastic-status/v1`），按数字岛号取 capacity、round_wall_ema_s、lease_remaining_s、arrival_history、pending、carried_over_lag，外加全局 syncer_epoch、outer_version、policy_hash，经 `scheduling_probe` 进 `IslandController.inspect()`。缺文件、坏 JSON、schema 不符都返回空，字段保持 None；legacy 下控制器不调用探针。`build_elastic` 新增 `syncer_status`、`island_number` 两个可选参数（learner 侧尚未传入，见未解决项）。
- 0.19（commit d20908f9）：`elastic_client.SampleIndex` 改为与 Rust `SampleIndexEntry` 全字段一致的线路布局；新增 SAMPLE_VERDICT（类型 24）解码、`sample_index_from_entry` / `sample_index_to_entry` 转换（policy_hash 十六进制 ↔ 32 字节原始 sha256；岛名经映射表或十进制字符串 ↔ u32）、`ElasticIslandClient.sample_index()`（发送一条并等待裁决）。黄金帧 19、24 用 Rust 1be303e3 的 encode 重新导出。对真实 syncer 二进制的端到端：接受（ACCEPT，同版本）、需重要性采样修正（ACCEPT_IS，落后一版）、拒收（REJECT，policy_hash 不符）三态全部符合；同一用例里 syncer 真实写出的 status.json 被 0.18 的读取函数解析，sample_index.jsonl 的条目能被 Python `SampleIndexEntry.from_json` 还原。
- 0.17（本条之前一个 commit）：`launcher.resolve_island_syncer_epoch` + `syncer_status_reader`。elastic 且没有显式 `--rl-syncer-epoch` 时，syncer 起来后、构建岛任务前轮询 status.json（头节点模式读本机，独立 syncer 集群用 ssh cat），读到就写回 `args.rl_syncer_epoch`；120 秒内读不到则沿用 0 并打 WARN。legacy 命令行逐字不变的单测仍通过。
- 验证：交接中的完整测试命令（含 YETO_TEST_ELASTIC_SYNCER，二进制由 rs 工作树 1be303e3 的只读副本在 /tmp/s15-noray/rs 编译）结果 434 passed（上次 423，新增 11）。
- 未解决：(1) 独立 syncer 集群的 ssh 读取 status.json 只有单测级覆盖，未在真实集群上跑过；(2) learner 侧还没把 syncer 状态路径和数字岛号传给 `build_elastic`（岛和 syncer 通常不在同一台机器，status.json 需要另行同步或转发，属于后续设计）；(3) 0.16 旧 Miles 桥接路径仍待批。
### 0.8 续：黄金哈希、D-S6 检查点、张量规模估算（同分支，2026-10-07）

- 黄金哈希：合并 Python 侧 b6dccad8 后，Rust 新单测 `server::tests::elastic_profile_hash_matches_the_python_canonical_vector` 用与 Python `_syncer_profile` 相同的输入加 elastic 默认参数（θ 0.75、γ 0.5、软截止 900、q_min 1、max_carry_lag 2），得到 4b61bb37c3dfafce169058a26f4e76dd1fcaad257c115ea468133916a24560b8，与 Python 一致。HMAC 密钥：未给 `--island-hmac-key` 时读环境变量 YETO_ISLAND_HMAC_KEY（`main.rs`），空串视为未提供，elastic 下缺密钥拒绝启动。
- D-S6 检查点（elastic 独立格式，不改旧版检查点格式与 state.rs）：
  - `elastic.rs` `ElasticCoordinator::encode_state / decode_state`：syncer_epoch、membership_epoch、outer_version、成员表（加入版本、是否首轮补齐、算力）、carried_over（增量元数据与 lag）、dropped_uncommitted 账本（新字段 `dropped`）；内嵌 elastic 契约编码，恢复时参数不同即拒绝。恢复后成员租约从重启时刻重新计时；本轮未完成的增量不保存（岛需重发）。
  - `elastic_server.rs`：文件头 `YELSRV1\0` + 协调器状态 + 基版本参数与动量缓冲 + 待合并的张量。沿用 `--checkpoint-path` / `--checkpoint-every` / `--resume`，每 checkpoint_every 次步进写一次，写法为临时文件 + fsync + rename + 目录 fsync。`--resume` 时 syncer_epoch = max(保存值 + 1, `--syncer-epoch`)，并立刻写回，保证再崩再起继续自增。
  - 注意：步进那一刻待合并和迟到集合都已清空，所以按步进写的检查点里 carried_over 通常为空；若要保留"步进之间到达的迟到增量"，需要在收到迟到增量时也写检查点（未做，待裁定是否值得）。
- 测试：`cargo test` 134 通过、0 失败。新增 `coordinator_state_roundtrips_membership_carry_and_dropped_ledger`、`resume_restores_state_and_bumps_syncer_epoch`（一轮后停机，--resume 重启：旧 epoch 被拒，epoch 0→1，基版本 3.0 恢复后再一轮得 5.0，检查点里 epoch=1、outer_version=2）。
- legacy：相对基线 360da6ae，server.rs / protocol.rs / main.rs / state.rs / merge.rs 删除行仍为 0。

#### elastic 扁平 f32 张量帧的规模（供裁定是否分片 / 压缩）

- 帧上限：`elastic_server.rs` `MAX_ELASTIC_FRAME` = 4 GiB（单帧，含 HMAC）。整帧读入内存后才验 HMAC。
- Flash-Next LoRA r16 / 专家 r_e 8 的可训练参数：4 层版本 117,969,408 个（`yeto/rl/profiles/qwen3_8_next.py::expected_trainable_params_4layer(16, 8)`，已在 GPU 上核对过的口径）。48 层完整版按每层平均线性外推 ×12 ≈ 14.2 亿个 **[估算，未核对完整版层型比例]**。
- 每个岛每轮的增量（f32）：4 层约 0.44 GiB；完整版约 5.3 GiB，**超过 4 GiB 帧上限，当前实现装不下**。bf16 约 2.6 GiB（elastic 目前只支持 f32）。
- 每轮带宽（N 个岛，f32，完整版）：上行 N × 5.3 GiB 增量 + 下行 N × 5.3 GiB 基版本广播；两岛约 21 GiB / 轮。syncer 内存峰值约（成员数 + 迟到数 + 2）× 5.3 GiB。
- 可选方向（待用户裁定）：沿用旧版的分片帧与 bf16 / q4 编码（旧版已有，需把 elastic 合并改成按分片）；或只广播增量而非完整基版本；或分片流式验 HMAC。

### 0.8 续：D-S5 状态导出、SAMPLE_INDEX 处理（同分支，2026-10-07）

- D-S5（`elastic.rs::status_json`，`elastic_server.rs` 定时写出）：elastic 下若给了 `--event-tape`，在它同目录写 `status.json`（临时文件 + fsync + rename，每 250 毫秒最多一次，结束时再写一次）。schema `yeto.syncer.elastic-status/v1`，字段：syncer_epoch、membership_epoch、outer_version、policy_hash（当前基版本 f32 小端字节的 sha256 十六进制，无张量时为 null）、cap_arrived、cap_total、soft_deadline_remaining_s、pending_count、carried_over_count、dropped_uncommitted_count、sample_index_count；`islands` 以岛编号字符串为键，每岛 capacity（cap_i）、joined_at、catch_up（本轮是否首轮权重 0）、round_wall_ema_s（LEASE_HEARTBEAT 上报的 round_wall_s 的指数平均，α=0.2，无上报为 null）、lease_remaining_s、arrival_history（最近 8 次步进是否按时到齐，新的在后）、pending、carried_over_lag。
  - Python 读法：`json.loads(Path(event_tape).with_name("status.json").read_text())`，按 `islands[str(island_id)]` 取值填 IslandStatus 的调度字段；文件不存在或 schema 不符时字段保持 None（不编造）。只读，不需要新消息。legacy 不写此文件。
  - 说明：step-time EMA 取的是岛自报的整轮墙钟时间，不是旧版 server.rs 里按学习者推送间隔算的 EMA（旧版代码未动）。
- SAMPLE_INDEX（P3，只索引不转发）：
  - 正文按 Python `SampleIndexEntry`（schema yeto.rl.sample-index/v1）定义，字节顺序见 `elastic.rs` 中 `SampleIndexEntry` 的注释（字符串为 u32 长度 + UTF-8；policy_hash 为 32 字节原始 sha256，Python 侧十六进制解码后发送；island_id 在线路上是 u32，JSON 里写成十进制字符串）。
  - 校验与 Python `__post_init__` 一致（schema、URI 前缀 s3:// / modal-volume:// / nebius-os://、n ≥ 1、sha256 为 64 位小写十六进制、advantage_included 必须为真），另要求 group_id 非空。格式错误回 MSG_ERROR。
  - 去重键（岛, group_id）：内容完全相同的重复提交幂等返回原判定；同键不同内容报错。
  - 三态判定（相对当前外层版本，因为索引时不知道消费岛）：不是成员 / 版本未知 / policy_hash 不符 / 陈旧超过 2 步 / 缺行为 logprob → 拒收；同版本 → 接受；陈旧 1–2 步且有行为 logprob → 需 IS 修正。陈旧上限 2 为常量 `MAX_SAMPLE_OUTER_LAG`（与 Python StalenessPolicy 默认一致），尚无命令行参数。消费岛相关的规则（同岛、inner_lag、on_policy_only）仍由消费端账本判定。
  - 回复新消息 24 SAMPLE_VERDICT（判定、原因、outer_lag，带 HMAC）；被接受的条目追加到 `--event-tape` 同目录的 `sample_index.jsonl`；tape 记 `sample_indexed`（含 duplicate 标志）。
  - 未做：样本索引不进检查点（重启后索引为空，岛可重发，重复提交是幂等的）；索引查询 / 分发消息。
- 测试：`cargo test` 137 通过、0 失败。新增 `sample_index_validates_dedups_and_judges_three_ways`、`status_snapshot_exports_scheduling_fields`、集成测试 `sample_index_over_the_wire`（真实张量轮次得到的基版本哈希，接受 / 需 IS / 拒收各一）；原 3 岛集成测试另检查了 status.json。连续 85 次完整运行中有 1 次出现 1 项失败，未能复现，也没抓到是哪一项（之后 80 次全过），**原因未查明**，可能是某个依赖计时的测试偶发不稳定。
- legacy：相对基线 360da6ae，server.rs / protocol.rs / main.rs / state.rs / merge.rs 删除行仍为 0。

### 1.0 / 1.1 阶段 1a 旧版基线（两岛小模型真机，2026-10-07）

- 脚本与评判（工作区外，s1-runs）：`s15-island1a-remote.sh`、`s15-island1a-judge.py`；1b 预备 `s15-island1b-remote.sh`、`s15-island1b-judge.py`。复核与结果：infra-drafts/ISLAND-STAGE1-PRELAUNCH-REVIEW.md。代码 s15-interisland@58144906（git archive），默认 legacy，同步服务命令行与现状相同。
- 形状：Nebius eu-north1 无卡 VM 跑总控+同步服务，两岛各 Modal 1×H100；Qwen3-0.6B LoRA，gsm8k，rollout 4×8，6 外层步，fragments 1，quorum 2，max-base-lag 0，`--rl-observe-timeline`。
- 结果 [实测]：s15-island1a-20261007c **PASS**——岛 1 在应用 v2 后被杀，启动器 366 s 后重启，从 v2 续跑（哈希与杀前一致），v1..v6 两岛哈希一致，rc=0；legacy 处理成员变化的方式是同步服务原地等待：第 3 步合并 718 s（正常 85–93 s），无退出码 4/6、stall 看门狗未触发、rejected_stale 0。a/b 两跑因杀岛脚本缺陷未杀（训练 6 步完成，reward 逐轮相同），保留为无杀岛对照组。
- 费用 [估算]：a ≈$2.86，b ≈$2.70，c ≈$4.50，合计 ≈$10.06（cap $18，阶段 1 总 cap $40）。
- θ/γ 校准：本跑是 legacy，未校准（elastic 默认 θ 0.75、γ 0.5 原样进入 1b）。
- 观察（未验证）：岛 1 重启后首轮 reward 与第 1 批相同，可能重启后数据游标从头开始。
- 重要性采样离线比较：本跑只拿到轮级数据（每轮 delta_l2_norm、current_vs_rollout_kl≈0、ess_ratio 1.0、同步服务每步 global_delta_norm）；逐样本 behavior_logprob / 版本 / 优势符号需加采集点，清单见复核文档。
- 1.2（1b elastic）未执行：Rust elastic 已合入、PLAN_ONLY 通过，但岛侧与 head 拿不到 `YETO_ISLAND_HMAC_KEY`（launcher 只把它明文拼进同步服务命令），需先修密钥传递再开卡。

## 0.20 密钥改为 secret 下发 + 阶段 1a 数据游标疑点
- 0.20 实现见 tasks.md。验证：交接完整测试命令再加 tests/test_launch_auto.py、tests/test_rl_launcher_multinode.py 共 483 passed（日志 /tmp/s15-noray/full2.log）。head 控制作业的 secrets 合入只有源码级断言（cli.py 的 head 启动流程没有现成单测替身），真机未验证。
- 疑点结论（只读，未改）：elastic 模式下岛重启后数据游标从头开始。原因：`yeto/rl/engine/bridges.py` 中 `ElasticAvgSync.start` 固定返回 `SyncStart(state, 0, ...)`（约 362 行），驱动 `yeto/rl/engine/driver.py` 的 `_run`（约 1511 行）随后 `ledger.rebase(0)`、`_restore_data_cursor(0)`，而后者对 rollout_id <= 0 直接返回（约 1444 行），不做 seek。对比：strict 模式从 syncer 拿到版本 v>0 时按账本记录的游标 seek（同函数）；无 syncer 的单岛模式只有配置了检查点存储时经 round cut（`miles_adapter/round_cut.py`）恢复 next_rollout_id 后才 seek。影响：elastic 岛重启会重新抽取已训练过的组（若账本保留了旧记录，`ledger.prepare` 会拒绝重复组号而失败；若账本是新的则静默重复训练）。需要给 ElasticAvgSync 补一个“重启时从本岛账本/检查点取 next_rollout_id”的逻辑，属于新任务，待定。

## 0.21 elastic 重启恢复轮次与数据游标
- 实现与限制见 tasks.md 0.21。验证：交接完整测试命令 + test_launch_auto、test_rl_launcher_multinode、test_rl_restart_data_cursor、新增 test_rl_elastic_restart_cursor 共 494 passed（/tmp/s15-noray/full3.log）。建议把 tests/test_rl_elastic_restart_cursor.py 和 tests/test_rl_restart_data_cursor.py 加入 tests4.txt。真机未验证。
- 待核实：1a 是 legacy（strict）模式，却观察到“重启后首轮 reward 与第 1 批相同”，与“strict 会按账本跳游标”的代码结论不一致（driver.py `_restore_data_cursor`，tests/test_rl_restart_data_cursor.py 覆盖）。可能原因包括：账本没有记录游标（游标元数据只在开启 in-island elastic 时上报，learner.py 约 481 行）、重启落在 rollout 0、或 reward 相同另有原因。需用 1a 的 tape 里重启前后各批次的样本 id / group_id 核实。

### 1.2 阶段 1b elastic 真机（2026-10-07）——FAIL，未勾

- s15-island1b-20261008b（代码 d65fe066，elastic，soft deadline 180 s，岛 1 于 v2 后被杀并重启），≈$4.69；中止的 20261008a ≈$0.81；阶段 1 累计 ≈$15.56。详情与证据见 infra-drafts/ISLAND-STAGE1-PRELAUNCH-REVIEW.md "1b 结果"。
- 通过 [实测]：真机联通、密钥 secret 生效、杀岛无退出码 4/6、catch-up 首轮权重 0 可见（raw_weights [[1,3,0.0]]）。
- 需修（按严重度）：① 岛被 lease_expired（lease 30 s < 本地轮时长）移出后不 rejoin，卡在 wait_global 且心跳使 stall 不触发，run 不会结束；② 0.21 游标恢复未生效（elastic_resume ledger_next=0，重启岛重训 rollout 0/1 的同一批）；③ 软截止 / carried_over 未出现，θ/γ 无校准数据；④ elastic 路径无 rl_local_round。
- 1a 疑问核实：legacy 重启同样把数据游标回到 0（1a-c 岛 1 rollout 2/3 = rollout 0/1 的样本批）；两岛同 rollout 的样本批相同（同 seed）。

### 1.2 第三跑 s15-island1b-20261008c（421dd484，2026-10-07）——机制判据可见，收尾与 reward 未过，未勾

- 可见 [实测]：无 lease_expired；catch-up 首轮权重 0；cursor_restored 且无重复批；软截止 5/6 步触发（到齐比例 0.5）；carried_over 2 次（γ^lag 折扣 0.25/0.5）；rl_local_round 补发生效；无退出码 4/6/7。
- 未过：同步服务结束时迟到岛 BrokenPipe 退出 1 → 启动器重启循环，run 不能自然结束（手动 cancel）；reward 0.500 < 1a 0.544；rejoin 未在真机触发。≈$5.78，阶段 1 累计 ≈$21.34。详见 infra-drafts/ISLAND-STAGE1-PRELAUNCH-REVIEW.md。
### 0.8 续：收尾宽限期与 FINISHED（同分支，2026-10-08，起因：真机 1b 第三跑岛 1 BrokenPipe）

- 到达 total_steps 后不再立即退出：不再步进，向所有在线成员广播一次新消息 25 FINISHED（syncer_epoch、最终 outer_version、基版本 policy_hash；带 HMAC），status.json 写 `"state":"finished"`（运行中为 `"running"`）。
- 宽限期内继续接受连接：LEAVE 照常处理；其他任何 elastic 消息（DELTA_TENSOR / DELTA_READY / JOIN / LEASE_HEARTBEAT / SAMPLE_INDEX / ELASTIC_INIT）在通过 HMAC 与 epoch 检查后都回 FINISHED，迟到增量不合并。
- 退出条件：宽限期到，或所有成员都已 LEAVE（或租约过期）。新参数 `--final-grace-s`，缺省等于 soft_deadline_s（缺省 900 s）。
- 测试：新增 `finished_grace_window_answers_late_islands_and_exits_on_leave`（慢岛收到主动广播；迟到增量、心跳、新 JOIN 均回 FINISHED；最终参数不含迟到增量；两岛 LEAVE 后提前退出）。`cargo test` 138 通过，完整运行 10 次无失败。legacy 相对 360da6ae 删除行仍为 0。

### 1.2 第四跑 s15-island1b-20261008d（fe8ec072，2026-10-07）——INCOMPLETE，未勾

- 收尾修复真机生效 [实测]：两岛收到 FINISHED 后 LEAVE、status.json state=finished、head 作业自行 SUCCEEDED、无重启循环；其余机制判据（catch-up 未计权、cursor_restored 无重复批、软截止 6/6、carried_over 4 次 γ^lag、rl_local_round 对齐、无退出码 4/6/7）全部通过；reward 0.514 vs 1a 0.544 只作信息。
- 唯一缺项：租约过期→重入未在真机触发（冻结脚本 `modal container exec` 缺 `--` 参数分隔，已修）。≈$4.70，阶段 1 累计 ≈$26.04。

### 1.2 第五跑 s15-island1b-20261008e（2026-10-07，今晚阶段 1 最后一次上卡）——FAIL，未勾

- 租约过期真机触发 [实测]：冻结岛 1 持有两条 :29400 连接的 python 进程 150 s，同步服务记 lease_expired（丢弃未提交增量）。
- 重入未验证：冻结期间岛 1 容器被判失败，再没上线；同步服务由岛 0 跑完并正常结束（status finished），launcher 不重启岛 1，但因岛 1 tape 未 finalized 以 exit 3 收尾。其余判据同第四跑通过。≈$3.87，阶段 1 累计 ≈$29.91（cap $40）。
- 1.2 剩余缺口：仅"租约过期→重入→继续训练"真机证据；下次只冻结 learner 进程、冻结 ≈lease+30 s、时机提前。

### 1.2 第七跑 s15-island1b-20261008g（2026-10-08，调试开关只断链路）——重入真机验证通过（judge v2 PASS，v1 FAIL），1.2 仍未勾

- 新增测试开关 `--rl-elastic-debug-pause ISLAND:AFTER_V:S`（8c978c33）：岛进程照常运行，只让它与同步服务的链路静默 S 秒（不续租、发送挂起、收到的帧挂起）。默认关闭时岛任务/Modal 配置逐字不变；不进契约哈希（只改时序）。tests/test_rl_elastic_debug_pause.py 含真 Rust 同步服务端到端。
- 真机 [实测]：岛 1 在 v1 后静默 120.3 s（lease 90）→ 04:00:00 lease_expired → 静默结束后同步服务拒收 "1 is not a member (rejoin required)" → 13.4 s 后 elastic_rejoin（incarnation 1，catch_up，base 3）→ 同步服务 pool_join catch_up → 之后 v5–v10 共 6 步岛 1 权重≈0.5 → 两岛正常结束，rc=0，status finished，两岛 H100。
- 发现：① 客户端事件在岛 tape 里各写两次（自建客户端被重复挂钩，0.22 起就有），已修 eb6ff049；② 重入后第一步里岛 1 除 catch-up 零权重条目外，还有一条用静默前已收到（被挂起）的 v2 训练出的迟到增量，按 γ=0.5 折扣并入（权重 0.497）。judge v1 的 C4 "首步岛 1 全部为 0" 因此不过；v2（看数据后改的口径，已注明）拆成 C4a catch-up 条目为 0（过）+ C4b 迟到增量记为信息。是否允许"过期前拿到的基座训出的增量在重入后按迟到并入"待裁定。
- ≈$3.0 [估算]。证据 s1-runs/s15-island1b-20261008g/{judgment-pause.json(v1),judgment-pause-v2.json,launch.ts.log,head/yeto-output/yeto-tape.jsonl,head/yeto-syncer.log,tape-direct/}。
