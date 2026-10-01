# A4 E1-A（3.4 X2）+ E1-E：Nebius 8×H100 on-demand，代码 47efd25，2026-09-30

基线 `base/`（13:14–13:38，无请求，12 轮）；切换 `x2/`（13:39–14:04；第 2 轮 train 时提交 up→T4R4S0，第 5 轮 train 时重复提交同一 up，第 7 轮 train 时提交 down→T4R2S2；容器内触发器）。两者都带 `--rl-elastic --rl-elastic-declare-cells`（Miles router），指纹 2d0a00f4…，镜像 2cc5cc52…。分析：`analyze_e1a.py` → `x2/analysis.json`。判据：`evidence/infra-e1/plan.md` E1-A (a)–(g)、E1-E 原文。

| 判据 | 结果 | 说明 |
|---|---|---|
| (a) 两事务终态 SUCCEEDED，config_epoch 0→1→2 | 通过 | a4up、a4down 均 SUCCEEDED；epochs.config_epoch=2 |
| (b) 逐轮 sample-id 哈希与基线相等，optimizer 步数 12 | 通过 | 12/12 轮相等，每轮 applied_lrs 长度 1 |
| (c) 第 3–7 轮 `rl_publication.sync/publication_members` = 4，其余 = 2 | **按字面未满足（需裁定）** | up 事务在 v2 发布之后、generate(2) 之前执行，weight_admission 把 v2 权重接纳到新 cell（`rl_reconfiguration` rollout 2 members=4），generate(2)–generate(6) 实际在 4 成员上进行；down 在 v7 发布之后、generate(7) 之前提交。但 `rl_publication` 事件本身：v2=2、v3–v7=4、其余=2。即按"第 r 轮的 rl_publication"字面计为第 4–8 轮=4，与判据差一轮；按"第 r 轮生成所用的成员"计则第 3–7 轮=4 完全吻合。判据在运行前固定为 rl_publication 字段，本 agent 不自行改口径 |
| (d) trainer PID 与 trainer GPU UUID 全程不变 | 通过 | G0–G3 上 (UUID,PID) 集合在 16 次采样中唯一 |
| (e) 池外 GPU 无新进程 | 通过 | 39 次采样的全部进程都在 8 张池内 UUID 上 |
| (f) down 在 drain 后才 stop_cells | 通过 | a4down：QUIESCING → TRANSFERRING → fork_op stop issued/done |
| (g) 备用卡 GPU-hours 计入 | 记录 | 池 8 卡 × 墙钟（13:39:33–14:04:43）全部计费；G6/G7 在第 1–2、8–12 轮空闲 |
| E1-E：train 中提交的请求在该步返回后的边界执行 | 通过 | a4up 于 train(1) 开始时写入 inbox，request 记录与 WAIT_SAFE（safe_point_rollout_id=2）在 publish(2) 之后 |
| E1-E：重复提交同一 request_id 返回同一 tx_id | 通过 | 第二次提交后 `a4up.status.json` = tx-0-d13c83e0da4b-a4up / SUCCEEDED，journal 仅 1 条 a4up request |
| E1-E：ledger 每轮恰好一条 prepared/optimizer_applied/outer_recorded | 通过 | 12 轮 × 5 类记录，无重复 |

## (c) 按主 agent 裁定复判（2026-09-30）
- 口径：事件自身轮次字段。`rl_publication` 没有单独的 round 字段，自身轮次字段为 `policy_version`；driver 的轮次定义（driver.py：publish(v) 之后 generate(rollout_id=v)，轮次 r（从 1 计）= rollout_id r−1）与运行前计划 infra-e1/plan.md（"第 3 轮前 request up，第 8 轮前 request down"，轮次从 1 计）一致 → 第 3–7 轮对应 policy_version 2–6，其余轮对应 0、1、7–11。
- 事件（原始行见 `x2/c-events.txt`）：policy_version 0、1、**2** 为 2 成员；3、4、5、6、**7** 为 4 成员；8–11 为 2 成员。
- 判定：v2 = 2（应为 4）、v7 = 4（应为 2）→ **(c) 未通过**。成员变化在 v2 发布后生效（up 的 weight_admission 以 v2 权重接纳新 cell，`rl_reconfiguration` rollout_id 2 members=4，generate(2) 在 4 成员上执行），发布事件晚一轮体现；down 同理（在 v7 发布之后生效）。

结论：**3.4 不勾选**（(c) 未通过）；(a)(b)(d)(e)(f)(g) 与 E1-E 通过。成员全程使用 fork cell id（engine:inference-engine-all-0-0-0000{0..3}）。
费用：基线 ≤$12.11，切换 ≤$12.92（25.2 min×$30.8/h）。释放：nebius instance list 为空，sky 无集群。
