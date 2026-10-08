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
