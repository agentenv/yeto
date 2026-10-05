# Tasks

标记：[D] 本 change 直接实现；[INFRA-REQ] 改动落在 rl-infra-spec / rl-engine-ports 负责文件，以接口请求提交，负责人合入或书面授权后方可实施。全部验证只用 CPU 单测与已有磁带，不起 GPU/云/Ray。

## 1. P0：训练指标补齐与 W&B train/step 修复（不受 infra 阻塞，可先行）

- [ ] 1.1 [INFRA-REQ；`miles_adapter/trainer.py`、`state_plugin.py`] 提交接口请求：`MilesTrainerGroup.step_metrics()` 由 `last_step_losses` 经 `mean_step_metrics` 填 loss/pg_loss/mean_kl（KL 类键）/ess_ratio，lr 取 `applied_lrs` 末值，train_step 取优化器累计步；新增 `round_metrics()` 返回完整轮均值字典。验收：CPU 单测用伪造 step_losses 断言各字段非 None 且等于均值，缺 KL 键时 mean_kl 为 None。
- [ ] 1.2 [INFRA-REQ；`engine/driver.py`，依赖1.1] 提交接口请求：`_train_round` 的 `rl_round_trained` 增加 `train_step`、`train_metrics{}`、`tok_per_s`、`step_seconds`，`rl_local_round` 使用补齐后的指标；验收：`fake.py` 引擎的 driver 单测断言新字段存在、旧字段不变。
- [ ] 1.3 [INFRA-REQ；`rollout_meta_hook.py`/`rollout.py`，依赖1.2] 提交接口请求：rollout 批次汇总 adv_mean/adv_std、resp_len_mean/p95、truncated_frac、reward_p10/p50/p90 并挂到 batch，由 driver 写入 `rl_round_trained`；验收：CPU 单测用构造样本断言分位数与截断比例。
- [ ] 1.4 [D；`wandb_rl.py`、`wandb_tape.py`，依赖1.2] `train/step` 取自记录 `train_step`，缺失时以 learner 本地步推导并记日志说明；`train_metrics{}` 键展开为 `train/<key>` 并纳入白名单；验收：单测对 ports 样例磁带断言 `train/grad_norm`、`train/loss` 不再被丢弃。
- [ ] 1.5 [D；`status_metrics.py`，依赖1.2] CLI 摘要显示新增训练字段，缺失显示“无数据”；验收：对旧磁带与新磁带各跑一次 `render_tape_summary` 快照测试。

## 2. P1a：运行期事件（heartbeat、resource_sample、fleet.jsonl）

- [ ] 2.1 [INFRA-REQ；`rl/learner.py` 或 driver，依赖无] 提交接口请求：learner 后台线程每 30s 写 `rl_heartbeat`（phase、rollout_id、policy_version、uptime_s），经 `append_record`（Modal 回显路径自动覆盖）；验收：单测以假时钟断言频率与不阻塞主循环。
- [ ] 2.2 [INFRA-REQ；同上，依赖2.1] 提交接口请求：`rl_resource_sample`（NVML，周期 30–60s 可配置，不可用时单条 `available:false`）；验收：单测 mock NVML 可用/不可用两种情况。
- [ ] 2.3 [D；`launcher.py` FleetController] 写 head 侧 `fleet.jsonl`：岛 launch/ready/lost/stop 与每 5 分钟 `cost_tick`；价目表配置文件（示例值注明“示例，非账单”）与预算上限来自运行参数；验收：FleetController 单测以假时钟断言事件序列与金额计算，未定价岛金额为 null。

## 3. P1b：reducer、只读服务、静态导出

- [ ] 3.1 [D] 新建 `yeto/dashboard/reducer.py`：多流 JSONL 增量归约为 overview/islands/rounds/fleet/events 视图，未知事件透传、缺失为 null；验收：以仓库内已有 ports 磁带与构造的 4 岛磁带做快照测试，重复喂入同一偏移不重复计数。
- [ ] 3.2 [D；依赖3.1] syncer 派生 round 视图：merge 记录 join `rl_fragment_push`/`rl_policy_apply`/`rl_pull_resend`/`rl_member_publication`，得 responded/expected、missed、quorum/sync/merge、重发，标注派生；验收：构造“1 岛未推送”“重发”两例断言行内容。
- [ ] 3.3 [D；依赖3.1] 告警规则表（心跳超时、连续 missed、quorum 异常、grad_norm 尖峰/NaN、clipfrac/KL、预算 60/80%），阈值可配置，每条带定位目标；验收：逐规则单测触发与不触发。
- [ ] 3.4 [D；依赖3.1,2.3] 成本视图：价目表×GPU×墙钟、速率、预计触达、`$/1M tok`、`reward/$`，本地岛不排名；验收：单测与 v7 示例口径一致。
- [ ] 3.5 [D；依赖3.1] `yeto dashboard serve`：stdlib HTTP，仅 127.0.0.1、仅 GET，端点见 design D2，独立 OffsetStore 跟随磁带，打印 SSH 隧道命令；验收：单测非回环地址拒绝启动、POST 返回 405、各端点返回 JSON schema。
- [ ] 3.6 [D；依赖3.1] `yeto dashboard export`：单文件自包含 HTML，内联视图 JSON，无外部请求；验收：测试断言输出无 `http(s)://` 资源引用，且内联 overview 与 `/api/overview` 对同一磁带相等。

## 4. P1c：v7 页面主体

- [ ] 4.1 [D；依赖3.5,3.6] 按 v7 实现静态页（无构建步骤）：头部（run、模式 chip、全局状态、成本条、更新时间）、告警卡、按岛叠加曲线（12 个指标 tab、round 边界、missed 带、告警点）、round 表（仅异常筛选）、岛健康卡、成本效率表，浅/深色；验收：以示例视图数据在浏览器人工核对与 v7 一致，并记录截图到 change 的 evidence。
- [ ] 4.2 [D；依赖4.1] 联动与下钻：告警点击定位岛/round/指标，健康卡点击高亮并展开同页岛明细（进度/policy、最近事件、Ray 区；cell 表与 E1 事务在 P2 前显示“待 infra 就绪”），“复制 CLI 命令”按钮；验收：前端单元脚本或手动检查清单全部通过。
- [ ] 4.3 [D；依赖4.2] Ray 嵌入：`tunnel`/`direct` 岛点击后加载 iframe 并显示隧道命令，Modal 退化为资源/事件面板；验收：三类岛各一例视图快照。
- [ ] 4.4 [D；`rl/ssh_harness.py`，依赖3.5] ssh_harness 运行的远端磁带 `tail -F` 跟随写本地镜像，断线退避重连并记 `dashboard_source_lost`，恢复后按偏移去重；验收：以本地 `sh -c` 模拟 SSH 断开/恢复的单测。
- [ ] 4.5 [D；依赖4.1-4.4] 文档：`docs/` 增加 dashboard 使用说明（隧道、导出、价目表示例非账单、W&B 仅归档）；验收：按文档对已有磁带完成一次 serve 与 export。

## 5. P2：cell 表与 E1 事务面板（被 infra 阻塞）

- [ ] 5.1 [INFRA-REQ；`engine/controller.py`/`journal.py`；阻塞：rl-infra-spec cell 声明（1.6）与 3.2 验收] 提交接口请求：`rl_cell_snapshot`（cell id、角色、GPU 数、状态）在配置提交与周期性写出；解除条件：rl-infra-spec 1.6 与 3.2 勾选完成；验收：controller 单测断言事件字段。
- [ ] 5.2 [INFRA-REQ；同上；阻塞：rl-infra-spec 3.2–3.7 验收] 提交接口请求：`rl_reconfig_phase`（txn id、阶段、结果含 COMMITTED/RECOVERY_REQUIRED）由 journal 状态转换写出；解除条件：3.7 勾选完成；验收：失败矩阵单测每个转换各有一条事件。
- [ ] 5.3 [D；依赖5.1,5.2] reducer 与岛明细接入 cell 表与 E1 事务面板，RECOVERY_REQUIRED 产生严重告警；验收：构造磁带快照测试。

## 6. P3：syncer 原生实时事件（后续）

- [ ] 6.1 [D；阻塞：cargo 构建环境与 rl-infra-spec A5 验收] syncer Rust 写出每 round 的 quorum/missed/resend 原生事件，字段与 3.2 派生视图同名；验收：同一运行派生与原生视图对比一致后切换数据源。

## 7. 集成检查

- [ ] 7.1 [D；依赖1-4] 以一份真实 ports 多岛运行磁带（由其他已批准实验产生，本 change 不另起 GPU）执行 serve 与 export，核对训练曲线非空、round 表、告警与成本面板；验收：记录结果与截图。
