# 6.6 CPU 四场景 trace replay —— 以 5.1-BASELINE n=1 数据回放（**n=1，非验收**）

生成：`yeto/rl/engine/replay.py`（commit 见分支 d2-replay），脚本 `infra-drafts/tmp-logs/d2_replay_run.py`，原始 JSON `infra-drafts/tmp-logs/d2-replay-result.json`。纯 CPU 模拟，非 GPU 实测。

## 输入（全部来自 d1/5.1-BASELINE.md，n=1）
- 参考配置 T2R1S1；稳态轮时 10.87 s，拆为 rollout 可扩展生成 0.3 s + 训练 10.57 s（使 T2R1S1→T2R2S0 Δ=0.15 s/轮，与实测 10.87 vs 10.72 一致）。
- 切换成本：up（rollout engine 增加）阻塞 97.0 s + 首 step 恢复 2.5 s；down 5.2 s + 0.1 s。T2R2S0↔T1R3 无实测，**假设**同 up/down 值。
- 池 4 卡固定（allocated GPU-hours = 4 × wall）；更新预算 1000 轮；执行模式 serial（兼容基线与目标 profile 相同，故 mode_gain=0；overlap 模式尚无认证 profile）。
- 动态组：真实 `Recommender`+`AutoController`，默认 `AutoPolicy`（K=6, margin 120 s, H=min(3600 s, 剩余预算), dwell/cooldown 1800 s, ≤2 次/h），efficiency_lower=0.7。
- 场景（seed=11，抖动 ±2%）：stable；changing = 后半程生成 ×3 阶跃；long_tail = 5% 轮次 +30 s 不可扩展尾；tool_wait = 每轮 +5 s 工具等待。

## 结果（wall 单位 s）

| 场景 | 兼容默认 | 默认固定(目标模式) | 最佳固定(事后) | 动态 | 最佳固定配置 | 动态切换次数 | mode 收益 | resize 收益(最佳固定) | resize 收益(动态) | 动态 GPU-h | 动态 样本/s(1样本/轮) | 动态 等待 s | R1→R2 回本轮数 | R1→T1R3 回本 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stable | 10872 | 10872 | 10722 | 10872 | T2R2S0 | 0 | 0 | 150 | 0 | 12.08 | 0.0920 | 0 | 663 | 无收益 |
| changing | 11172 | 11172 | 10872 | 11172 | T2R2S0 | 0 | 0 | 300 | 0 | 12.41 | 0.0895 | 0 | 332 | 无收益 |
| long_tail | 12516 | 12516 | 12366 | 12516 | T2R2S0 | 0 | 0 | 150 | 0 | 13.91 | 0.0799 | 1651 | 663 | 无收益 |
| tool_wait | 15872 | 15872 | 15722 | 15872 | T2R2S0 | 0 | 0 | 150 | 0 | 17.64 | 0.0630 | 5000 | 663 | 无收益 |

回本轮数 = (cost_upper + recovery_upper) / 平均每轮节省 = 99.5 s / 0.15 s ≈ 663 轮（≈2 h 训练）；changing 场景按全程平均节省算 ≈332 轮。T1R3 训练卡减半，轮时变长，任何场景都无收益。

## 结论（n=1，非验收）
1. 在该成本/收益量级下，**按保守门槛不存在可自动触发的净收益边**：动态组在四场景全部 0 次切换，结果等于默认固定（无回归）。原因：H=3600 s 内收益下界 ≈ 0.0079×3600 ≈ 28 s（理想 ≈ 50 s），远低于 97+2.5+120 = 219.5 s 门槛。
2. "最佳固定"比默认快 150–300 s / 1000 轮（≈1.4%–2.7%），这是**事后选配置**的收益，不是动态能拿到的；要靠切换拿到，单次 up 边需 ≈663 轮稳态才回本。
3. 门槛反推：在 10.9 s 轮时、默认 policy 下，需每轮 Δ 下界 ≳0.66 s（即收益下界占比 >6.1%），或 up 边成本降到 ≲ 收益×H − 122.5 s，动态才会触发。改善方向是降低 up 边 init（占 94%），而不是调策略门槛。
4. tool_wait 场景动态因 tool-heavy 保持；long_tail 的尾部不计入 resize 收益；无成本表时动态 ≡ 默认固定（见测试）。
5. mode 收益此处为 0（同 serial）；获得认证 overlap profile 后，用同一 `compare(..., TimeModel(mode="certified-overlap"))` 单独报告 mode_gain_s，与 resize 收益分开。

## 模型局限
- 轮时模型：serial=gen+tool+tail+train+publish；overlap=max(R,T)+publish，无 fill/drain/版本约束；生成按 engine 数线性扩展（可设 efficiency）。
- 首 step/恢复开销取 5.1 的 +2.5/+0.1 s；未建模后台 restore 与对其他岛的影响。
- replay 窗口里 `gpu_busy_fraction` 只填 rollout 可扩展部分；而 `timeline.LoadSummary.gpu_busy_fraction` 含 trainer compute，直接送入 `predict_gain` 会高估 resize 收益（见回报中的配合需求）。
- 单 seed、单次；GPU 验收需重复运行分布。
