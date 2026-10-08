# Tasks

标记：[D] 本 change 直接实现；[INFRA-REQ] 改动落在 rl-infra-spec / rl-engine-ports 负责文件，以接口请求提交，负责人合入或书面授权后方可实施。全部验证只用 CPU 单测与已有磁带，不起 GPU/云/Ray。

## 1. P0：训练指标补齐与 W&B train/step 修复（不受 infra 阻塞，可先行）

- [x] 1.1 [INFRA-REQ；`miles_adapter/trainer.py`、`state_plugin.py`] 提交接口请求：`MilesTrainerGroup.step_metrics()` 由 `last_step_losses` 经 `mean_step_metrics` 填 loss/pg_loss/mean_kl（KL 类键）/ess_ratio，lr 取 `applied_lrs` 末值，train_step 取优化器累计步；新增 `round_metrics()` 返回完整轮均值字典。验收：CPU 单测用伪造 step_losses 断言各字段非 None 且等于均值，缺 KL 键时 mean_kl 为 None。 **接口请求（未实施，`miles_adapter/` 在 yeto/rl/engine 下，由 d2-wire 合并后另派）**：`MilesTrainerGroup.step_metrics()` 填 loss/pg_loss/mean_kl/ess_ratio/lr/train_step；新增 `round_metrics()`。 （已实现 dash-events c7bf03ca：step_metrics/round_metrics，tests/test_rl_dash_events.py）
- [x] 1.2 [INFRA-REQ；`engine/driver.py`，依赖1.1] 提交接口请求：`_train_round` 的 `rl_round_trained` 增加 `train_step`、`train_metrics{}`、`tok_per_s`、`step_seconds`，`rl_local_round` 使用补齐后的指标；验收：`fake.py` 引擎的 driver 单测断言新字段存在、旧字段不变。 **接口请求（未实施，driver.py 属 d2-wire）**：`rl_round_trained` 增 `train_step`、`train_metrics{}`、`tok_per_s`、`step_seconds`；看板/W&B/status 侧已按字段契约读取，发射即生效。 （已实现 fa8cbf33：train_step/train_metrics/step_seconds/tok_per_s；ports 路径 wandb tee 开关）
- [x] 1.3 [INFRA-REQ；`rollout_meta_hook.py`/`rollout.py`，依赖1.2] 提交接口请求：rollout 批次汇总 adv_mean/adv_std、resp_len_mean/p95、truncated_frac、reward_p10/p50/p90 并挂到 batch，由 driver 写入 `rl_round_trained`；验收：CPU 单测用构造样本断言分位数与截断比例。 **接口请求（未实施）**：rollout 汇总 `adv_mean/adv_std/resp_len_mean/resp_len_p95/truncated_frac/reward_p10/p50/p90` 挂 batch，由 driver 写入 `rl_round_trained`。 （已实现 3959167e：batch_summary，仅 observe 或 YETO_RL_BATCH_SUMMARY=1 时生成；adv_* 实机多为 None，待 actor 侧取值）
- [x] 1.4 [D；`wandb_rl.py`、`wandb_tape.py`，依赖1.2] `train/step` 取自记录 `train_step`，缺失时以 learner 本地步推导并记日志说明；`train_metrics{}` 键展开为 `train/<key>` 并纳入白名单；验收：单测对 ports 样例磁带断言 `train/grad_norm`、`train/loss` 不再被丢弃。 证据：`yeto/rl/wandb_rl.py` 读 `rl_local_round` 顶层裸字段（loss/pg_loss/grad_norm/mean_kl/ess_ratio/clip_fraction/lr/reward_mean）与 `train_metrics{}`→`train/<key>`（键名白名单、仅有限标量），`train/step` 取 `train/step|train_step`，缺失时以 `local_round_id` 推导并记一次 info 日志；`rl_round_trained` 只投影 train/*。`tests/test_wandb_rl.py` 28 passed（含 s9 ports 样例 `train/grad_norm` 不再丢弃；legacy 用例期望随本条更新）。注：ports 路径当前未调用 `wandb_rl.tee`（仅 `rl/miles.py`），接线随 1.2 一并做。
- [x] 1.5 [D；`status_metrics.py`，依赖1.2] CLI 摘要显示新增训练字段，缺失显示“无数据”；验收：对旧磁带与新磁带各跑一次 `render_tape_summary` 快照测试。 证据：`status_metrics.render_tape_summary` 追加每岛 `TRAIN island=N ...` 行，缺失显示“无数据”；syncer 旧磁带输出不变。`tests/test_status_metrics_training.py` 3 passed（旧 syncer 磁带 / s9 ports learner 磁带 / 新字段磁带）。

## 2. P1a：运行期事件（heartbeat、resource_sample、fleet.jsonl）

- [x] 2.1 [INFRA-REQ；`rl/learner.py` 或 driver，依赖无] 提交接口请求：learner 后台线程每 30s 写 `rl_heartbeat`（phase、rollout_id、policy_version、uptime_s），经 `append_record`（Modal 回显路径自动覆盖）；验收：单测以假时钟断言频率与不阻塞主循环。 **接口请求（未实施，learner/driver 归 d2-wire）**：`rl_heartbeat{phase,rollout_id,policy_version,uptime_s}` 每 30 s 经 `append_record`。看板已消费（心跳卡、超时告警；未发射时按最后事件计并标注）。 （已实现 fa8cbf33：rl_heartbeat，--rl-heartbeat-interval，observe 时默认 30 s）
- [x] 2.2 [INFRA-REQ；同上，依赖2.1] 提交接口请求：`rl_resource_sample`（NVML，周期 30–60s 可配置，不可用时单条 `available:false`）；验收：单测 mock NVML 可用/不可用两种情况。 **接口请求（未实施）**：`rl_resource_sample{gpus:[{index,uuid,util_pct,mem_used_mb,mem_total_mb,power_w}]}` 或 `available:false`。看板已消费（GPU util/显存、Modal 退化面板）。 （已实现 fa8cbf33：rl_resource_sample NVML，--rl-resource-sample-interval，observe 时默认 60 s，不可用写 available:false）
- [x] 2.3 [D；`launcher.py` FleetController] 写 head 侧 `fleet.jsonl`：岛 launch/ready/lost/stop 与每 5 分钟 `cost_tick`；价目表配置文件（示例值注明“示例，非账单”）与预算上限来自运行参数；验收：FleetController 单测以假时钟断言事件序列与金额计算，未定价岛金额为 null。 证据：`yeto/dashboard/fleet.py` `FleetLog`（ready/lost/stop、5 min `cost_tick`、未定价金额 null）；`launcher.FleetController(fleet_log=...)` 在首次/重启/丢失/结束/放弃时写入，`_dashboard_fleet_log` 接入 `runs/<run>/fleet.jsonl`，异常只打印不影响监督；价目表/预算经 `YETO_DASHBOARD_PRICES`/`YETO_DASHBOARD_BUDGET_USD`。`tests/test_dashboard_cost.py`（假时钟事件序列与金额、FleetController 生命周期、写失败不影响）+ `tests/test_controller.py` 通过。

## 3. P1b：reducer、只读服务、静态导出

- [x] 3.1 [D] 新建 `yeto/dashboard/reducer.py`：多流 JSONL 增量归约为 overview/islands/rounds/fleet/events 视图，未知事件透传、缺失为 null；验收：以仓库内已有 ports 磁带与构造的 4 岛磁带做快照测试，重复喂入同一偏移不重复计数。 证据：`yeto/dashboard/reducer.py` + `sources.py`；`tests/test_dashboard_reducer.py`（s9-m4x1 真实 ports 磁带+journal 片段 `tests/fixtures/dashboard/`、构造 4 岛磁带、同偏移重复喂入不重复计数、残行、未知事件透传、缺字段为 None）。
- [x] 3.2 [D；依赖3.1] syncer 派生 round 视图：merge 记录 join `rl_fragment_push`/`rl_policy_apply`/`rl_pull_resend`/`rl_member_publication`，得 responded/expected、missed、quorum/sync/merge、重发，标注派生；验收：构造“1 岛未推送”“重发”两例断言行内容。 证据：`Reducer.rounds()` 以 merge 记录为主键 join push/apply/resend/member_publication，标 `derived`；“1 岛未推送 3/4”“重发 1”两例与无 expected 的旧记录回退均有单测。
- [x] 3.3 [D；依赖3.1] 告警规则表（心跳超时、连续 missed、quorum 异常、grad_norm 尖峰/NaN、clipfrac/KL、预算 60/80%），阈值可配置，每条带定位目标；验收：逐规则单测触发与不触发。 证据：`yeto/dashboard/alerts.py` 规则表（心跳、连续 missed、quorum、grad_norm 尖峰/NaN、clipfrac/KL、预算 60/80、RECOVERY_REQUIRED、来源丢失），`--thresholds` 可配；`tests/test_dashboard_alerts.py` 逐规则触发/不触发。
- [x] 3.4 [D；依赖3.1,2.3] 成本视图：价目表×GPU×墙钟、速率、预计触达、`$/1M tok`、`reward/$`，本地岛不排名；验收：单测与 v7 示例口径一致。 证据：`yeto/dashboard/cost.py` + `prices.example.json`（示例非账单）；v7 口径 8×H200@3.6×2.6 h = $74.88、$/1M tok、reward/$、预计触达、本地不排名、未定价为 null，见 `tests/test_dashboard_cost.py`。
- [x] 3.5 [D；依赖3.1] `yeto dashboard serve`：stdlib HTTP，仅 127.0.0.1、仅 GET，端点见 design D2，独立 OffsetStore 跟随磁带，打印 SSH 隧道命令；验收：单测非回环地址拒绝启动、POST 返回 405、各端点返回 JSON schema。 证据：`yeto/dashboard/serve.py`，`yeto dashboard serve`；`tests/test_dashboard_serve.py` 起线程服务 GET 各端点、POST/PUT/DELETE/PATCH→405 且状态不变、0.0.0.0 等非回环拒绝（含 CLI 返回 2）。偏移为服务内存独立跟随（不与 W&B `OffsetStore` 共享；重启从头重放以重建视图）。
- [x] 3.6 [D；依赖3.1] `yeto dashboard export`：单文件自包含 HTML，内联视图 JSON，无外部请求；验收：测试断言输出无 `http(s)://` 资源引用，且内联 overview 与 `/api/overview` 对同一磁带相等。 证据：`yeto/dashboard/export.py`，`yeto dashboard export`；测试断言无 `https?://`、无 `<link>`、内联 overview == 离线时钟 `/api/overview`，`</script>`/`<!--`/`://` 已转义。

## 4. P1c：v7 页面主体

- [ ] 4.1 [D；依赖3.5,3.6] 按 v7 实现静态页（无构建步骤）：头部（run、模式 chip、全局状态、成本条、更新时间）、告警卡、按岛叠加曲线（12 个指标 tab、round 边界、missed 带、告警点）、round 表（仅异常筛选）、岛健康卡、成本效率表，浅/深色；验收：以示例视图数据在浏览器人工核对与 v7 一致，并记录截图到 change 的 evidence。 已实现 `yeto/dashboard/static/{index.html,app.js}`（v7 结构，无构建、系统字体、浅/深色）；以 node DOM 桩 `tests/js/dashboard_smoke.js` 对导出页做渲染冒烟。**未做：浏览器人工核对与截图**（本环境无浏览器）。
- [x] 4.2 [D；依赖4.1] 联动与下钻：告警点击定位岛/round/指标，健康卡点击高亮并展开同页岛明细（进度/policy、最近事件、Ray 区；cell 表与 E1 事务在 P2 前显示“待 infra 就绪”），“复制 CLI 命令”按钮；验收：前端单元脚本或手动检查清单全部通过。 证据：告警点击（切指标/定位 round/高亮 round 行/展开岛明细）、健康卡点击（高亮+下钻）、指标 tab、复制 CLI 按钮；node 冒烟 `test_old_tape_renders_no_data_in_the_page` 走告警点击、卡片点击、grad_norm tab。
- [x] 4.3 [D；依赖4.2] Ray 嵌入：`tunnel`/`direct` 岛点击后加载 iframe 并显示隧道命令，Modal 退化为资源/事件面板；验收：三类岛各一例视图快照。 证据：`ray_embed` tunnel/direct/none/unknown，用户点击后才建 iframe（协议相对 `//127.0.0.1:port`），离线导出只显示占位；Modal 退化为 GPU util/显存/tok/s；`test_ray_embed_per_island_kind`。
- [x] 4.4 [D；`rl/ssh_harness.py`，依赖3.5] ssh_harness 运行的远端磁带 `tail -F` 跟随写本地镜像，断线退避重连并记 `dashboard_source_lost`，恢复后按偏移去重；验收：以本地 `sh -c` 模拟 SSH 断开/恢复的单测。 证据：`yeto/dashboard/mirror.py` `SshTapeMirror` + `yeto dashboard mirror`（`tail -c +offset -F` 续传、指数退避、`<mirror>.source.jsonl` 记 lost/restored）；`tests/test_dashboard_mirror.py` 以 `sh -c` 模拟断线/恢复（不丢不重、残行重取）。注：未改 `rl/ssh_harness.py` 自动拉起镜像，按文档手动启动。
- [x] 4.5 [D；依赖4.1-4.4] 文档：`docs/` 增加 dashboard 使用说明（隧道、导出、价目表示例非账单、W&B 仅归档）；验收：按文档对已有磁带完成一次 serve 与 export。 证据：`docs/DASHBOARD.md`；按文档对 s9-m4x1 真实磁带跑通 serve（curl `/`、`/api/overview`、`/api/events`，POST→405）与 export（`infra-drafts/tmp-logs/dashboard-s9-m4x1.html`）。

## 5. P2：cell 表与 E1 事务面板（被 infra 阻塞）

- [x] 5.1 [INFRA-REQ；`engine/controller.py`/`journal.py`；阻塞：rl-infra-spec cell 声明（1.6）与 3.2 验收] 提交接口请求：`rl_cell_snapshot`（cell id、角色、GPU 数、状态）在配置提交与周期性写出；解除条件：rl-infra-spec 1.6 与 3.2 勾选完成；验收：controller 单测断言事件字段。 **接口请求（未实施）**：`rl_cell_snapshot{cells:[{cell_id,role,gpus,state}]}`。看板已支持；未发射时 cell 表由 journal `gpu_pool` 派生并标注来源。 **已实施（dash-ctrl）**：`IslandController.set_event_sink(sink, journal=)` 开启后，分配/rebind/拒绝（`record_gpu_pool`）、`node_lost`、每个事务终态各发 `rl_cell_snapshot{tx_id,txn_id,cause,t,cells:[{cell_id,role,node,gpu_uuid,gpus,state,config,epoch}]}`；证据 tests/test_rl_controller_events.py。
- [x] 5.2 [INFRA-REQ；同上；阻塞：rl-infra-spec 3.2–3.7 验收] 提交接口请求：`rl_reconfig_phase`（txn id、阶段、结果含 COMMITTED/RECOVERY_REQUIRED）由 journal 状态转换写出；解除条件：3.7 勾选完成；验收：失败矩阵单测每个转换各有一条事件。 **接口请求（未实施）**：`rl_reconfig_phase{txn_id,phase,result}`。看板已支持；当前直接读 controller journal 的 `request`/`phase` 记录（`tx_id`、COMMITTED/SUCCEEDED/CANCELLED/REBUILT_OLD/RECOVERY_REQUIRED）。 **已实施（dash-ctrl）**：`_phase` 每次转换发 `rl_reconfig_phase{tx_id,txn_id,request_id,source,target,phase,result,t,expected_epoch,config_epoch,reason}`，与 journal phase 一一对应、replay 不重发；reducer 优先消费（tests/test_dashboard_ctrl_events.py）。
- [x] 5.3 [D；依赖5.1,5.2] reducer 与岛明细接入 cell 表与 E1 事务面板，RECOVERY_REQUIRED 产生严重告警；验收：构造磁带快照测试。 证据：reducer 读 journal `request/phase/gpu_pool/node_lost` 与磁带 `rl_reconfiguration`/`rl_cell_snapshot`/`rl_reconfig_phase`；RECOVERY_REQUIRED→严重告警；真实 s9 journal（island-scope RECOVERY_REQUIRED）与构造事务生命周期单测。

## 6. P3：syncer 原生实时事件（后续）

- [ ] 6.1 [D；阻塞：cargo 构建环境与 rl-infra-spec A5 验收] syncer Rust 写出每 round 的 quorum/missed/resend 原生事件，字段与 3.2 派生视图同名；验收：同一运行派生与原生视图对比一致后切换数据源。

## 7. 集成检查

- [ ] 7.1 [D；依赖1-4] 以一份真实 ports 多岛运行磁带（由其他已批准实验产生，本 change 不另起 GPU）执行 serve 与 export，核对训练曲线非空、round 表、告警与成本面板；验收：记录结果与截图。 CPU 替代已做：s9-m4x1-20261005aa（单岛 ports、node_lost→RECOVERY_REQUIRED）导出 `infra-drafts/tmp-logs/dashboard-s9-m4x1.html`，reward/grad_norm/tok/s 曲线非空、RECOVERY_REQUIRED 严重告警、E1 面板与派生 cell 表；无 syncer/fleet.jsonl 故 round 表与成本为“无数据”。**未做：多岛 ports 运行与截图。** S14 补做：2 岛真实磁带（s13-g3-modal-20261007f，0 轮）+ 3 份真实单岛 Modal 磁带 export/serve 核对，见 progress.md S14；仍无带 syncer/round 事件的真实多岛磁带，故未勾。


## 8. 启动阶段可观测性与成本口径（2026-10-08 补充；等 FN 2×8 早门跑完再做）

- [x] 8.1 [D；`yeto/dashboard/reducer.py`、`alerts.py`] 未见 `rl_driver_start` 的岛标为"启动中"，心跳告警改用启动阈值（默认 30 min，可配）。验收：用 `s16-rawlora-fn2x8-long-20261008a` 启动段磁带重放，不再出现"心跳超时"严重告警；人为截断到启动后 40 min 无事件时报警。
- [x] 8.2 [D；`yeto/dashboard/cost.py`、`prices.example.json`] 价目表增加 CPU 核·时与内存 GiB·时单价，按岛申请的核数/内存计入；缺失时页面标"仅 GPU"。验收：Modal 2×8（32 核、768 GiB ×2）估算与启动脚本 $87.93/h 相差 <5%。
- [x] 8.3 [D；`yeto/dashboard/cli.py`/`sources.py`] 未传 `--run` 时从磁带记录推断运行名并显示。验收：只传 `--tapes` 时页面标题显示运行名。
- [ ] 8.4 [INFRA-REQ；依赖 yeto-framework-decoupling 标准样本；S17 未做：要改 driver.py，与去耦合阶段 3 同文件，等其完成后再做] 岛启动即开心跳与资源采样（phase=`startup`），并在加载权重完成、Ray 集群组好、推理引擎就绪各发一条事件；dashboard 按子步骤分别计时与告警。验收：标准样本中既有事件字段不变；新事件有单测；下一次真机运行启动段可见显存曲线与子步骤时间。

- [x] 8.5 [D+INFRA；`yeto/modal_runner.py`、`reducer.py`] 训推分离时推理节点 GPU 未被采样（`rl_resource_sample` 只在驱动所在节点）：Modal 主机探针加记 GPU 利用率与 `node_rank`，多节点岛默认开启（30 s，可用 `YETO_MODAL_HOSTMEM_SAMPLE_S` 改或设 0 关闭），容器一启动就采样（顺带覆盖启动段显存）；reducer 按节点分别保留显存峰值与利用率。验收：单测；用 2×8 旧磁带重放可见节点 0/1 两条显存（旧磁带无利用率，显示"未采样"）；下一次真机看到推理节点利用率。

## 9. 界面改版（2026-10-08 用户确认；等 FN 2×8 早门跑完，与第 8 组一起做）

- [ ] 9.1 [D；`tools/dashboard_preview.py`/`.html`，S17 已生成，等用户定稿] 静态预览页：用 `s16-rawlora-fn2x8-long-20261008a` 真实磁带生成，含暖色两主题、单岛布局、悬停竖线与数值框。验收：用户确认定稿。
- [ ] 9.2 [D；`static/index.html`] 暖色令牌（D-UI1），深浅主题与 `prefers-color-scheme` 均正确；训练/推理固定配色。验收：两主题截图对比度检查通过。
- [ ] 9.3 [D；`reducer.py`、`static/app.js`] 布局按 `run_kind` 自动切换（D-UI2），未使用面板折叠。验收：单岛 2×8 磁带与一份多岛磁带各渲染一次，第一屏内容符合设计。
- [ ] 9.4 [D；`reducer.py`] 逐轮记录补齐：阶段起止（generate/train/sync/publish）、截断率、回答长度分位、训练/推理 tok/s、耗时拆分。验收：对 2×8 磁带重放，第 0、1 轮数值与 FN2X8-MODAL-PRELAUNCH-REVIEW §9 记录一致。
- [ ] 9.5 [D；`static/app.js`] 阶段时间线 + 每轮耗时拆分堆叠条（训练侧/推理侧占比）。验收：2×8 第 0 轮显示 R 171 s、T 615 s、S 216 s、P 243 s。
- [ ] 9.6 [D；`static/app.js`] 曲线：点少于 5 不连线；分位误差带；注明"每轮题目不同"。验收：2 轮数据只画两个点。
- [ ] 9.7 [D；`static/app.js`] 悬停竖线与数值框（D-UI3）：吸附最近轮、内容齐全、多岛逐岛一行、三图联动、靠边翻转、点按与键盘。验收：浏览器手动检查 + 对数值框内容的 reducer 单测；缺失项显示"无"。
- [ ] 9.8 [D] 借鉴参考图：左侧导航、顶部关键数字卡片、可展开岛健康列表、带时间与级别的事件流、命令复制按钮；成本表窄屏不横向滚动。验收：400 px 宽度下无横向滚动。
