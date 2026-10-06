# 弹性分配操作手册（草稿，6.7"手册"的一部分）

状态：d2-wire 分支。CPU 已验证；**未在 GPU 上验证，也没有任何一条实测净收益边**。默认行为不变：不传 `elastic_hook` 时 driver 与旧路径逐事件一致；`recommend_mode` 默认 `disabled`。

## 1. 组件与数据流

```
IslandDriver.safe_point(rollout_id)
  ├─ sync.at_safe_point / controller.poll_commands()   （旧路径）
  ├─ elastic_hook.at_safe_point(driver, rollout_id)      （新增，可选）
  │    observe=True 且 mode ∈ {recommend, auto} 时才工作：
  │    events ─ timeline.load_windows(window_s) → LoadSummary（带 profile_hash/epoch）
  │    candidates = attestation.certified_edges ∩ declared_edges
  │    costs = edge_costs_from_table(edge_costs_path)
  │    recommend → 事件 rl_elastic_recommendation + journal "recommendation"（仅可执行的）
  │    auto      → AutoController.step → controller.request（与人工同一入口）
  └─ controller.run_at_safe_point(...)                    （旧路径：执行待决事务）
```

* 代码：`yeto/rl/engine/elastic.py`（`ElasticHook`、`auto_state`/`restore_auto_state`）；`IslandDriver(elastic_hook=...)`。
* 剩余预算：`(total_rounds - rollout_id) × 最近 ≤5 轮的平均轮时`；评估开始后至少要看到两个 safe point 才算得出，算不出时 auto 保持。
* tool_heavy：当前 (profile, epoch) 的最后 K 个窗口里只要有一个以 tool_wait 为主，就算 tool-heavy。
* 收益模型只按 rollout 计算量缩放（`LoadSummary.rollout_busy_fraction`）；trainer 计算量不计入 resize 收益。

## 2. 打开 observe

构造 driver 时传 `observe=True`（会多出 1.7 的 `rl_timeline_span`、`rl_load_sample`、`rl_round_labels` 事件）。observe=False 时 hook 什么也不做。

## 2a. 启动参数（launcher → learner）

```
yeto launch ... --rl-engine ports --rl-elastic ... --rl-observe-timeline \
    --rl-recommend-mode recommend \          # disabled(默认)/manual/recommend/auto
    --rl-edge-costs-path /path/edge_costs.json \  # 5.7 成本表；缺失 → hold
    --rl-elastic-window-s 300                  # 负载窗口秒数（默认 300）
    # 以下调参均可选，缺省 = 代码默认值；必须与 --rl-recommend-mode 同用（非 disabled）
    --rl-auto-k-windows 6 --rl-auto-safety-margin-s 120 --rl-auto-horizon-s 3600 \
    --rl-auto-min-dwell-s 1800 --rl-auto-cooldown-s 1800 --rl-auto-max-switches 2 \
    --rl-auto-switch-window-s 3600 \        # → AutoPolicy
    --rl-recommend-ttl-s 300 --rl-recommend-min-windows 3 \
    --rl-recommend-efficiency-lower 0.7      # → Recommender
```

* 透传路径与 `--rl-observe-timeline` 相同：`yeto/cli.py` → `launcher` 校验并拼进 learner 命令行 → `yeto.rl.learner.apply_ports_infra_switches` 写到 `miles_args.yeto_rl_recommend_mode / yeto_rl_edge_costs_path / yeto_rl_elastic_window_s` → `compose_island` 里由 `miles_adapter/elastic_hook.elastic_hook_for` 构造 `ElasticHook`。
* 三个参数任一非默认时，都要求同时给 `--rl-elastic` 和 `--rl-observe-timeline`，否则 launcher（开机前）和 learner 都会拒绝。`disabled` 或不传参数时不构造 hook，事件序列与不传参数完全一致。
* 构造 hook 时：configs 取 controller 的资源配置；Flash-Next full profile 的候选边再与 `flash_next_elastic_declaration()` 的声明边取交集，其它 profile 只用 attestation 认证过的边；`total_rounds` = `num_rollout`。
* 启动时按 mode 调 `controller.set_recommend_mode`。`auto` 如果被拒（attestation 未声明 `auto_controller`），会记 warning 并退回 `recommend`（journal 里 `recommend_mode` 记录的 reason 为 `auto refused at startup: ...`）。重启时先恢复 journal 里最后一条 `auto_state`。
* 调参参数：`--rl-auto-*` 构造 `AutoPolicy`（k_windows、safety_margin_s、horizon_s、min_dwell_s、cooldown_s、max_switches、switch_window_s；deadline_s/fallback_mode 仍为默认），`--rl-recommend-*` 构造两处 `Recommender`（ttl_s、min_windows、efficiency_lower）。校验：时长/窗口为正数，k_windows、max_switches、min_windows ≥ 1，efficiency_lower ∈ (0,1]；不带 `--rl-recommend-mode`（或为 disabled）时给这些参数会被拒。不给则完全不拼进命令行。

## 3. 切换模式

模式：`disabled`（默认）/ `manual` / `recommend` / `auto`。切换写入 journal，重启后会 replay；切换不会打断进行中的事务或恢复。

```
python -m yeto.rl.engine.controller --state-dir <STATE> mode m1 recommend
python -m yeto.rl.engine.controller --state-dir <STATE> status m1   # 查看 inbox/m1.status.json
```

在下一个 safe point 生效。切到 `auto` 需要 attestation 声明了 `auto_controller=True`，否则会被拒（`rejected: auto mode refused ... auto_controller`），当前模式保持不变。另外，如果 driver 的 `EngineCapabilities.auto_controller` 为 False，hook 每次都会输出 hold 并说明原因。

人工批准建议：`Recommender.approve(rec, controller, ...)`，或者把建议里的 `target` 和 `expected_epoch` 提交成一条普通的 `request` 命令。两种方式都要重新校验 epoch、有效期、负载、`controller.plan`。

## 4. 成本表格式（5.7 产出；`RLRunConfig.edge_costs_path`）

JSON 列表，每条边、每个 profile 一行：

```json
[{"profile_hash": "sha256:<ExecutionProfile.contract_hash>", "source": "FN-T16R8S8",
  "target": "FN-T16R16S0", "cost_lower_s": 40, "cost_upper_s": 90,
  "recovery_upper_s": 60, "n": 3, "provenance": "<5.7 run id / evidence path>"}]
```

* `profile_hash` 必须等于运行时 `ExecutionProfile.contract_hash`；不相等就查不到，结果是 hold。
* 以下情况会被拒（ValueError）：`n<1`、没有 provenance、`lower>upper`、行重复。
* 文件缺失或为 None 时成本表为空，所有评估都 hold。
* 成本必须覆盖 drain/export/release/init/restore/verify/publish/warmup，以及对其他岛的等待（design D2）。

## 5. Flash-Next 配置声明（仅数据，未认证）

`yeto.rl.profiles.qwen3_8_next.flash_next_elastic_declaration(nodes=4, gpus_per_node=8, trainer_gpus=16, gpu="H200")`：

| 配置 | trainer | rollout（引擎数×8卡） | standby |
|---|---|---|---|
| FN-T16R8S8 | 16（TP2 PP8 EP2，LoRA） | 8（1×SGLang TP8/EP8） | 8 |
| FN-T16R16S0 | 16 | 16（2×TP8/EP8） | 0 |

声明边 `FN-T16R8S8 ⇄ FN-T16R16S0`，类型 rollout-only：只增删整个引擎副本，trainer、引擎形状和池大小都不变。

profile 由 `flash_next_execution_profile(AlgorithmSpec)` 给出（partitioned-serial，on-policy）。它的 `contract_hash` 贯穿窗口、建议和成本表三处。

声明的边**不代表已认证**：只有同时出现在 attestation.certified_edges 里的边才能成为候选，`controller.plan` 也会再拒一次。

## 6. 启用 auto 的前置条件（全部满足才可开）

1. D1 验收通过：recommend 模式在真实负载上输出的建议经人工复核是合理的。
2. 当前 profile_hash 下至少有一条已认证的边，并且 5.7 实测成本表里该边的净收益可以重复出现（E1 成功、论文数字、GPU 更忙都不算证据）。
3. attestation 里 `auto_controller=True`，引擎 capabilities 里也声明了 `auto_controller`。
4. 已设置 `AutoPolicy`：K、safety_margin、dwell、cooldown、max_switches；`total_rounds` 已知。
5. 回退路径：auto 一旦切换失败会自动降到 `manual`；也可以随时手动 `mode ... disabled`。

AutoController 的 dwell、cooldown、切换历史和在途请求都写在 journal 里（kind `auto_state`），重启后恢复，不会从零开始。

## 7. 已知限制

* 事件来源：driver 每次 `emit` 都会把 observe 事件同步喂给 `hook.feed`（内存镜像），因此不依赖事件带文件路径（Miles `_append_rl_event` 路径同样适用）。没有被喂过事件时（例如单独使用 hook），才回退为增量读取 JSONL 文件或使用 `events_source`。只有经过 driver `emit` 的事件可见；Ray worker 直接写进事件带的记录不会进入 hook。
* `eval` 阶段（span `task="eval"`）的 rollout 计算不计入 `rollout_busy_fraction`，但仍计入 GPU busy。
* AutoPolicy 不可通过 CLI 调整。
