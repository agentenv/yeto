# Tasks

## 1. 线程数预检
- [ ] 1.1 新增线程计数模块：遍历 /proc 累加本 uid 的 `Threads:`，单独统计 SkyPilot API 服务线程（验证：单测用假 /proc 目录，结果与构造值一致）
- [ ] 1.2 cli 增加 `--preflight-threads {wait,error,off}`、`--preflight-thread-start`、`--preflight-thread-hard`、`--preflight-thread-wait-s`，默认 wait/2800/3000/1800（验证：参数解析单测）
- [ ] 1.3 在 `launch`（launcher.py:6915）`sky_patches.install()` 之前与 head 入口（cli.py:1733、2032）调用预检；门槛内继续、门槛到硬线之间按配置等待或报错、硬线直接报错，读数写进运行清单（验证：单测注入线程读数 2500/2900/3100 三种情形，断言没有调用任何 sky 或 Modal 接口）
- [ ] 1.4 报错与等待输出包含总数、门槛、硬线、SkyPilot API 服务线程数与占比、`sky api stop && sky api start` 提示与"确认无进行中起机"提醒、不跑 Ray 测试提醒（验证：单测比对输出文本，含"未找到 SkyPilot API 服务"情形）
- [ ] 1.5 文档：docs/CLOUDS.md 增加上卡前线程预检一节（验证：文档写明默认值与关闭方式）

## 2. 显存估算预检
- [ ] 2.1 新增显存估算模块，按 design 决定 2 计算六项分项与总峰值；模型结构从 models.py 或 HF config 读取，读不到返回"未估算"（验证：单测构造已知结构，分项数值与手算一致）
- [ ] 2.2 回测：从 s1-runs 取 M1 run b/c/d、N17、ARU-2 的配置与实测峰值（或 OOM 记录），用 b/c/d 校准 k_act、k_logit、R，用 N17、ARU-2 检验；结果写 evidence/memory-backtest.json 与 design 附表，取不到的数据标"无数据"（验证：b、c 估算超限；c 的 logits 项与 7.6 GiB 相差 ≤10%；d 估算通过且与 125–126 GB 相差 ≤10%）
- [ ] 2.3 在 `launch` 的 6960 处接入预检：超阈值时报错退出，列出分项与可通过的建议及重算峰值；保留 `warn_if_model_wont_fit` 作为粗检（验证：单测用 run c 配置断言报错且无云调用，用 run d 配置断言通过）
- [ ] 2.4 cli 增加 `--preflight-memory {error,warn,off}` 与 `--preflight-memory-margin`（默认 error、0.9）；关闭时打印警告并写清单（验证：单测）
- [ ] 2.5 dry-run 输出每岛估算分项（验证：dry-run 单测含 `memory_estimate` 字段）
- [ ] 2.6 文档：docs/MILES_RL.md 增加显存估算一节，写公式、校准常数与回测误差（验证：文档与 evidence 数值一致）

## 3. 单岛换参数
- [ ] 3.1 cli 增加 `--rl-island-override ISLAND:KEY=VALUE`（可重复）与 `--rl-negative-test-run`；缺负例开关、非白名单参数、岛号越界时起机前报错（验证：参数解析与校验单测）
- [ ] 3.2 在岛循环（launcher.py:7086–7088）与 dry-run 循环（7616–7630）为被点名岛生成 args 副本，并对副本重跑相关校验（验证：dry-run 单测，岛 1 命令行含新学习率调度、岛 0 不变；不传开关时 tests/test_decoupling_golden.py 不变）
- [ ] 3.3 新增仅测试参数 `identity_test_salt`，岛侧混入 `island_contract_sha256`（backend_identity.py:108）（验证：单测，加盐后身份哈希变化，不加盐时不变）
- [ ] 3.4 运行清单写 `negative_test` 与 `island_overrides`；岛环境带 `YETO_ISLAND_OVERRIDE`，岛入口启动时在 tape 写 `rl_island_override` 事件；终端打印醒目警告（验证：单测检查清单字段与假岛 tape 首条事件）
- [ ] 3.5 看板 reducer 识别 `rl_island_override`，在岛旁标"负例岛"与参数差异（验证：reducer 单测与看板快照）
- [ ] 3.6 `--rl-resume` 指向负例运行时报错，`yeto export` 遇负例运行报错（验证：两条单测）
- [ ] 3.7 文档：docs/RL_ELASTIC_BENCHMARK.md 增加负例岛用法与禁用说明（验证：文档含示例命令与警告）

## 4. 上卡验证 N12/N14 负例（需报批）
- [ ] 4.1 上卡前复核：按 memory gpu-review-before-launch 捋清代码路径，把预登记判据与复核结论写进 evidence/n12-n14-plan.md（验证：文档存在且判据写明）
- [ ] 4.2 报主 agent 批预算：Modal 两岛各 1×H100 + Nebius 无卡 head，Qwen3-0.6B gsm8k，三个短运行各约 15 分钟（严格模式学习率调度不同、elastic 身份不符、落后上限不同），估计 $10–15，上限 $20（验证：主 agent 批复记录在 evidence）
- [ ] 4.3 上卡运行，判据：被换参数的岛被拒，拒绝原因与参数对应，只拒该连接，另一岛继续训练并完成全部轮次，看板标出负例岛；本 change 的线程与显存预检在同一次运行中真机执行（验证：tape 与看板截图存 evidence，结论按"通过/失败"填写）

## 5. 收尾
- [ ] 5.1 本机安全测试集全绿（命令见 fix-known-red-tests），`openspec validate launch-preflight-guards` 通过（验证：命令输出存 evidence）
