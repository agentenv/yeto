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

* hook 默认按增量方式读事件带的 JSONL 文件。如果事件带走 Miles 的 `_append_rl_event`（`EventTape(args=...)`），需要传 `events_source`。
* learner/CLI 还没有 `--rl-edge-costs-path` / `--rl-elastic-window-s` 参数，hook 也还没有在 learner 里构造。RLRunConfig 字段已经预留（从 `args.rl_edge_costs_path` / `args.rl_elastic_window_s` 读取）。
* `eval` 阶段的 rollout 计算量会计入 `rollout_busy_fraction`。
