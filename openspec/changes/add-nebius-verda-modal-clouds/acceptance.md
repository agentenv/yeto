# 验收对照表

5 个 spec 的 34 个场景，每个对应至少一个测试函数（无凭据即可跑的单元测试）或任务组 8 的真机记录条目。真机条目在 docs/CLOUDS.md 的"Verified on real machines"表填写后才算完成。

## shape-cloud-signals

| 场景 | 测试 / 真机记录 |
|---|---|
| 无货和查不到区分开 | `test_sold_out_is_a_measurement_not_a_missing_signal` |
| 接口失败退化而不中断 | `test_one_cloud_signal_failure_does_not_disturb_the_others`, `test_nebius_capacity_api_failure_degrades_to_none` |
| 缺凭据的云不参与 | `test_cloud_without_credentials_makes_no_requests` |
| 只有 Nebius 凭据 | `test_only_nebius_credentials_plans_without_aws` |
| 指定了没有凭据的云 | `test_listed_cloud_without_credentials_is_a_clear_error` |
| 区域感知（Nebius） | `test_nebius_scores_from_capacity_advice`, `test_cloud_prefixed_region_keeps_only_that_region` |
| 动态 spot 价格覆盖 | `test_live_price_overrides_catalog_for_budget_and_is_marked`, `test_nebius_live_prices_need_a_project_per_region`, `test_nebius_estimate_parses_hourly_total` |
| 8 卡机型进入候选（Verda） | `test_verda_offerings_from_live_fixture` |
| 多节点被排除（Verda） | `test_verda_never_plans_multi_node_and_says_so` |
| 非整机多节点被拒（Modal） | `test_multi_node_rejection_reasons`, `test_modal_multi_container_requires_whole_nodes` |
| 静态价格表过期提示 | `test_modal_price_table_staleness_warning` |
| 行为不变（RunPod） | `test_aws_runpod_plan_matches_pre_registry_snapshot`, `test_catalog_gap_warning_names_unlisted_known_gpus` |

## shape-region-filter

| 场景 | 测试 / 真机记录 |
|---|---|
| 混合写法 | `test_mixed_entries_restrict_each_named_cloud_only`, `test_cloud_prefixed_region_keeps_only_that_region` |
| 旧写法不变 | `test_legacy_bare_regions_mean_aws_and_nothing_else`, `test_unnamed_cloud_is_unrestricted_and_legacy_spelling_unchanged` |
| 未知区域 | `test_unknown_region_error_lists_the_regions_that_exist`, `test_unknown_cloud_and_empty_region_are_errors` |
| Modal 区域加价 | `test_modal_price_composes_gpu_cpu_memory_and_region`, `test_modal_unpinned_has_no_region_and_pinned_carries_surcharge` |
| Modal 默认不固定区域 | `test_modal_unpinned_has_no_region_and_pinned_carries_surcharge`, `test_define_builds_the_function_from_the_config` |
| 应用规划 | `test_launch_key_keeps_native_region_and_parses`, `test_runpod_launch_key_parses_in_gpu_grammar` |

## shape-rl-islands

| 场景 | 测试 / 真机记录 |
|---|---|
| 分卡模式定价 | `test_rl_split_mode_prices_fixed_single_node_islands`, `test_rl_island_shape_from_shape_flags` |
| 共卡模式定价 | `test_rl_colocated_mode_allows_multi_node_islands` |
| 未验证的云被排除 | `test_rl_split_mode_prices_fixed_single_node_islands` |
| 无对象存储的云按按需价 | `test_rl_split_mode_prices_fixed_single_node_islands` |

## modal-learner-runner

| 场景 | 测试 / 真机记录 |
|---|---|
| 混合启动 | `test_modal_entries_get_modal_names_and_mix_with_sky_entries`, `test_island_ops_relaunch_keeps_the_learner_id_and_routing`; 真机：任务 8.7 |
| 非整机多节点被拒 | `test_modal_prerequisites_fail_before_launch`, `test_whole_node_rule_for_multi_container_islands` |
| SFT 岛完成一轮同步 | `test_modal_island_config_reuses_the_sky_task_script`, `test_container_sees_skypilot_variables_and_runs_the_sky_script`; 真机：任务 5.3 |
| RL 岛使用固定镜像 | `test_rl_config_requires_a_digest_pinned_image`, `test_rl_island_uses_the_digest_image`, `test_modal_rl_island_needs_a_digest_pinned_image` |
| 本地 controller 无隧道 | `test_syncer_must_be_reachable_from_modal` |
| 容器被抢占 | `test_preempted_island_is_relaunched_with_the_same_learner_id_then_abandoned` |
| 停止 fleet | `test_down_stops_the_modal_app_and_skips_sky_for_modal_islands`, `test_spawn_status_cancel_and_logs` |
| 手动接入 | `test_manual_cli_builds_a_config`; 真机：任务 5.3 |

## head-cloud-credentials

| 场景 | 测试 / 真机记录 |
|---|---|
| 缺 Nebius 凭据 | `test_launch_head_fails_before_submit_when_nebius_credentials_missing`, `test_head_cloud_credentials_files_env_and_errors` |
| 只挂载用到的云 | `test_launch_head_mounts_only_the_clouds_the_fleet_uses`, `test_fleet_clouds_include_the_head_cloud` |
| 岛上无云凭据 | `test_learner_islands_never_receive_cloud_credentials` |
| 两个 Nebius 区域 | `test_nebius_fleet_needs_a_project_per_region`, `test_missing_regions_are_all_listed` |

## 真机验收（需凭据，任务组 8）

| 任务 | 状态 |
|---|---|
| 8.1 Nebius SFT 岛 | 待做 |
| 8.2 Nebius RL 云敏感项 | 待做 |
| 8.3 Verda SFT / RL | 待做 |
| 8.4 Modal RL 岛 | 待做 |
| 8.5 三家 `yeto shape` 对照 | 待做 |
| 8.7 混合 fleet | 待做 |
| 5.3 Modal 手动接入 | 待做 |
| 9.2 旧写法回归（需 AWS 凭据；RunPod 已按超凡哥决定移出范围） | 已完成，见 tasks.md 9.2 与 `run-logs/task-9.2-*`：与 e7e3066 基线逐字一致，仅多出 `island_shape` 和 `price_source` 两个新增键 |
