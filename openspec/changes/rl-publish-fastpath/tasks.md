# Tasks

## 1. 设计与剖析

- [x] 1.1 按 s16-rawlora-fn2x8-long-20261008a 的阶段耗时与读码定位驱动侧慢因，写入 design.md（背景表、D1–D6、预估表，预估标"未验证"）

## 2. 训练进程内算哈希

- [x] 2.1 新增 `yeto/rl/engine/policy_digest.py`：`digest_canonical_tensors`（两个哈希逐字节复刻旧定义，两线程并行）、`PolicyDigest`（线上格式）、`layout_hash_of`、`TrainerResidentState`（`policy_tensor_hash/payload_digest/with_version/recheck/materialize`）。验证：`test_digest_is_byte_identical_to_driver_side_hashes`（并行与串行都与 `core.policy_tensor_hash`、`publish.payload_digest` 相等）
- [x] 2.2 `state_plugin.export_digest` 插件（复用 `export_state`，只回传哈希、清单与耗时）。验证：`test_export_digest_matches_full_export_and_ships_no_tensors`（只调用了 `export_digest`，哈希与整份导出相同）
- [x] 2.3 `MilesPolicyState.export_digest/_digest_result/_current_digest`（主 rank 唯一、版本一致、布局哈希比对与首次学习；打印耗时日志）

## 3. 单岛同步与发布改用句柄

- [x] 3.1 `bridges._local_state`，`LocalOnlySync.start/boundary` 使用；`IslandDriver.export_local_resident`（无 `export_digest` 或 `YETO_RL_PUBLISH_FASTPATH=0` 时回老路）。验证：`test_local_only_sync_uses_resident_state`、`test_fastpath_switch_off_falls_back`
- [x] 3.2 `MilesPublisher._check_trainer_holds`（句柄走训练进程就地重算，其他走原导出）与 `_payload_of`；`publish`、`publish_members` 改用。验证：`test_publish_check_still_refuses_changed_trainer_weights`
- [x] 3.3 兜底 `materialize` 核对哈希并缓存。验证：`test_resident_versions_and_lazy_materialize`
- [x] 3.4 既有测试适配：`test_rl_engine_selection` 假训练组加 `export_digest`、最终策略在关闭循环前取张量（端到端 token、发布成员断言不变）；`test_rl_fn_layout` 源码断言改认 `_local_state`

## 4. CPU 回归（本机，不起 Ray）

- [x] 4.1 `PYTHONPATH=/tmp/s15-noray:.` 下跑：test_rl_publish_fastpath、miles_adapter_state、miles_adapter_trainer_publish、engine_driver、a27_target_watch、trainer_transition、trainer_rebuild_e1、trainer_cut、e1_injections、e2_harness、fn_layout、driver_profiles、fake_profiles、engine_selection、state_plugin_distopt*、decoupling_golden、rl_decoupled。结果：全部通过（286 + 46 项，另 1 项跳过）
- [x] 4.2 另跑引用这些模块的其余 36 个测试文件：876 通过；失败/错误均与本 change 无关（12 项缺 `cargo`、2 项 Ray 被本机禁止、1 项 `test_rl_ir_harness` 在基线 1d1a0f60 上同样失败）

## 5. 下一次真机（需主 agent 批准上卡，不在本 change 内开卡）

- [ ] 5.1 复用 FN 2×8 训推分离脚本（s16-rawlora-fn2x8-long.sh 或 WP2 的新截断配置），只换 yeto 源码到本分支；上卡前按惯例复核（Miles 初始化断言逐条对照、台账预登记、线程 <3000）
- [ ] 5.2 采集：每轮 `outer_sync`、`publish` span；日志 `[rl] publish-fastpath digest export ... total/trainer_export/hash` 与 `recheck ...`；Miles update_weights 耗时；驱动进程内存峰值
- [ ] 5.3 判定：v0→vN 每次发布 `end_weight_update` 200、WeightChecker 校验和行数照旧、无 `[LORA-CHECK]`、适配器哈希每版变化；`rl_publication` 字段齐全；稳态每轮耗时与 design.md 预估表对照并回填实测
- [ ] 5.4 对照组（可选，同一次上卡末尾）：`YETO_RL_PUBLISH_FASTPATH=0` 跑 1 轮，确认两条路径的 `rl/policy_token` 与载荷哈希在相同权重下相同
- [ ] 5.5 若 trainer_export 占大头，按 design.md D6 另开 change
