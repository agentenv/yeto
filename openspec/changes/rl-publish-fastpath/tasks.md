# Tasks

## 1. 设计与剖析

- [x] 1.1 按 s16-rawlora-fn2x8-long-20261008a 的阶段耗时与读码定位驱动侧慢因，写入 design.md（背景表、D1–D6、预估表，预估标"未验证"）

## 2. 训练进程内算哈希

- [x] 2.1 新增 `yeto/rl/engine/policy_digest.py`：`digest_canonical_tensors`（两个哈希逐字节复刻旧定义，两线程并行）、`PolicyDigest`（线上格式）、`layout_hash_of`、`TrainerResidentState`（`policy_tensor_hash/payload_digest/with_version/recheck/materialize`）。验证：`test_digest_is_byte_identical_to_driver_side_hashes`（并行与串行都与 `core.policy_tensor_hash`、`publish.payload_digest` 相等）
- [x] 2.2 `state_plugin.export_digest` 插件（复用 `export_state`，只回传哈希、清单与耗时）。验证：`test_export_digest_matches_full_export_and_ships_no_tensors`（只调用了 `export_digest`，哈希与整份导出相同）
- [x] 2.3 `MilesPolicyState.export_digest/_digest_result/_current_digest`（主 rank 唯一、版本一致、布局哈希比对与首次学习；打印耗时日志）

## 3. 单岛同步与发布改用句柄

- [x] 3.1 `bridges._local_state`，`LocalOnlySync.start/boundary` 使用；`IslandDriver.export_local_resident`（无 `export_digest` 或 `YETO_RL_PUBLISH_FASTPATH=0` 时回老路）。验证：`test_local_only_sync_uses_resident_state`、`test_fastpath_switch_off_falls_back`
- [x] 3.2 `MilesPublisher._check_trainer_holds` 与 `_payload_of`；`publish`、`publish_members` 改用
- [x] 3.2a（用户 S17 裁定）发布前检查改比训练进程权重版本号（design D3）：`state_plugin` 的 `_WEIGHTS_PROCESS_ID/_WEIGHTS_VERSION`、`bump_weights_version`、`current_weights_version`、插件 `weights_version`；`train_one_step` 包装、`apply_state`、`cut_plugin.restore_cut_shard/restore_resharded_shard` 加一；`export_digest` 带回版本号；`MilesPolicyState.weights_version/check_holds`（进程被替换时退回比内容）；`policy_digest.WeightsChanged`。验证：`test_publish_check_compares_weights_version_without_export`（只调用 `weights_version`、无导出；训练一步后报错）、`test_apply_and_restore_paths_bump_the_version`、`test_wrapped_train_one_step_bumps_even_when_it_raises`、`test_replaced_trainer_process_falls_back_to_content_check`、`test_missing_mark_is_refused`
- [x] 3.3 兜底 `materialize` 核对哈希并缓存。验证：`test_resident_versions_and_lazy_materialize`
- [x] 3.4 既有测试适配：`test_rl_engine_selection` 假训练组加 `export_digest`、最终策略在关闭循环前取张量（端到端 token、发布成员断言不变）；`test_rl_fn_layout` 源码断言改认 `_local_state`

## 4. CPU 回归（本机，不起 Ray）

- [x] 4.1 `PYTHONPATH=/tmp/s15-noray:.` 下跑：test_rl_publish_fastpath（13 项）、miles_adapter_state、miles_adapter_trainer_publish、engine_driver、a27_target_watch、trainer_transition、trainer_rebuild_e1、trainer_cut、e1_injections、e2_harness、fn_layout、driver_profiles、fake_profiles、engine_selection、state_plugin_distopt*、decoupling_golden、rl_decoupled、miles_cut_plugin、trainer_reshard。结果：386 通过，1 跳过
- [x] 4.2 引用这些模块的全部测试文件（除去 Ray/cargo 相关的 test_rl_integration 与禁止的三类）：1193 通过；3 项失败与本 change 无关（`test_rl_ir_harness` 1 项在基线 1d1a0f60 上同样失败；`test_upstream_parse_args_*` 2 项因本机禁 Ray）

## 5. 下一次真机（需主 agent 批准上卡、进统一 GPU 表；用户裁定对照实验等统一批预算后再跑）

- [x] 5.1 复用 FN 2×8 训推分离脚本（s16-rawlora-fn2x8-long.sh 或 WP2 的新截断配置），只换 yeto 源码到本分支；上卡前按惯例复核（Miles 初始化断言逐条对照、台账预登记、线程 <3000）
- [x] 5.2 采集：每轮 `outer_sync`、`publish` span；日志 `[rl] publish-fastpath digest export ... total/trainer_export/hash`；Miles update_weights 耗时；驱动进程内存峰值
- [x] 5.3 判定：v0→vN 每次发布 `end_weight_update` 200、WeightChecker 校验和行数照旧、无 `[LORA-CHECK]`、适配器哈希每版变化、无 "trainer weights differ"；`rl_publication` 字段齐全；稳态每轮耗时回填 design.md 预估表
- [ ] 5.4（本轮不跑，主 agent 代拍板：S16 已有老路径同形状数据）对照（`YETO_RL_PUBLISH_FASTPATH=0`，需单独一次启动）：确认老路径仍可用，并给出同机型下的老路径每轮耗时
- [ ] 5.5（待定：真机 trainer_export 约 39 s/次，是剩余同步时间的大头，建议另开 change）若 trainer_export 占大头，另开 change 改 PP 汇总方式

### 5.6 卡数、时长、费用（**估计**，供统一 GPU 表）

依据：s16-rawlora-fn2x8-long-20261008a 实测——Modal 2×8 H200 按 $87.93/h（台账口径，按容器时间）；容器起到初始发布 v0 约 23.5 min；稳态每轮约 11 min（老路径）；第 N 轮后停机等待 300 s。新路径每轮按 design.md 预估 4.8–6.6 min（未验证）。

| 实验 | 卡数 | 轮数 | 时长（估计） | 费用（估计） |
|---|---|---|---|---|
| A 快路径（主测） | 2 节点 × 8 H200 = 16 卡 | 5 | 23.5 + 5×(4.8–6.6) + 首轮多出约 8 + 停机 5 ≈ 61–70 min | ≈ $89–103 |
| B 老路径对照（`=0`） | 16 卡 | 2 | 23.5 + 2×11 + 首轮多出约 8 + 停机 7 ≈ 61 min | ≈ $89 |
| A+B 合计 | 16 卡 | — | ≈ 2.0–2.2 h | ≈ $178–192 |

建议：A 与 WP2 截断配置的上卡合并（同一次启动、同样只换 yeto 源码）；B 可省——S16 那次已有老路径同机型稳态数据（每轮约 660 s），只有在 A 的结果需要同批次对照时再跑。若合并到 WP2，A 的边际费用只是"每轮变短"带来的节省，不另计启动费用。上限建议按 A 单跑 $110 设硬停（HARD ≈ 4500 s）。

### 5.x 真机结果（S17 G4，`s17-fn2x8-fastpath-20261008a`，2026-10-08 16:21–17:21Z，Modal 2×8 H200，yeto main 80e944b6）
复核 infra-drafts/S17-G4-PRELAUNCH-REVIEW.md；原始数据 /home/michael/work/s1-runs/s17-fn2x8-fastpath-20261008a/（launch.log、tape-direct/、metrics.json、metrics-table.md、judgment*.json、gate-wait.json、hostprobe-*、dashboard.html）。配置相对 S16 早门：回答 12288 / seq 16384、mem 0.6、TIS 2.0/0、lr 5e-6（线性视界 20000，近似固定）。
- 5.1 [实测] 参数探针 rc=0（s1-runs/s17-g4-prep/probe2.out）；PLAN_ONLY rc=0；5 轮跑满。
- 5.2 [实测] 快路径日志 v0–v5 六次：total 46.7–48.2 s = trainer_export 38.7–40.1 s + hash 7.8–7.9 s。tape `outer_sync` span 46.7–48.2 s/轮（S16 215.7 s）；`publish` span 25.9–31.4 s（S16 242.6 s，含 Miles update_weights_implementation 6.3–7.2 s，等效 1.45–1.66 GiB/s）。训练结束→该版发布事件约 78–80 s（S16 448 s）。驱动进程发布期间只在等插件（py-spy，hostprobe-pub0-*）。
- 5.3 [实测] v0→v5 每次 end_weight_update 200（6 次）、WeightChecker 校验和 48 行（6×8）、无 [LORA-CHECK]、无 "trainer weights differ"；适配器哈希每版都变（bfb3ea07→0c2adbd0→7a5ad1ac→f1ec445c→e895ff64→b2bcc0e2）；单发送方 miles-pp_0；NCCL IB。s16-rawlora-4l-judge 判 PASS。
- 稳态每轮（相邻两次训练完成事件间隔）：462 / 404 / 440 / 425 s，均值约 433 s，S16 约 660 s（−34%）。注意本次回答从 8192 加到 12288，生成每轮 222–243 s（S16 151–171 s），所以**同长度下省得更多**；驱动侧同步+发布从约 458 s 降到约 75 s（−84%）。
- 未验证：5.4 老路径对照（不跑）；驱动进程内存峰值只有发布期一次快照，没有连续采样。
