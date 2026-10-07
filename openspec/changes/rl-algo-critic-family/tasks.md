# Tasks

执行约定：
- 测试命令 `/tmp/yeto-venv/bin/python -m pytest -q`；upstream 参数解析在 `/home/michael/work/miles-next-venv` 中运行。
- fork 改动只提交到 `michaellchung/miles` `yeto/ports`，且需用户同意；绝不向 radixark/miles 或 sgl-project/sglang 提 PR。commit 与 push 需用户确认。
- 所有 GPU 任务**需用户批准预算与机型**后执行；估算按 1×H100 ≈ $4/卡·时（8 卡 H100 ≈ $30.8/h）。上卡前先复核代码逻辑并写入 progress.md；先断言 `nvidia-smi` GPU 名；运行结束拆除实例并报告实际费用。
- 每项 ≤2 小时粒度。

## 1. 决策确认与基线

- [x] 1.1 在 `progress.md` 记录 proposal 中 5 项用户决策原文与日期（2026-10-06）。验证：文件存在且 5 项齐全。
- [x] 1.2 在当前 pin 的 Miles 提交上复核 design Context 的行号（arguments.py shared 约束、:3212、GAE、LoRA critic 跳过）。验证：核对结果与行号写入 progress.md，差异处标注。
- [x] 1.3 记录 pytest 失败基线到 `baseline-failures.txt`。验证：条目数与 pytest 汇总一致。

## 2. AlgorithmSpec critic 字段与翻译（CPU）

- [x] 2.1 在 `algorithm.py` 增加 advantage 组的 gamma/lambd/lambd_mode/alpha/gae_variant 与新 critic 组（含 `param_mode` 与预留的 lora 字段），仅在 needs_critic 时进入规范化（design D1、D9）。验证：新增 `tests/test_rl_critic_spec.py`：grpo golden 哈希不变、critic 字段改变哈希、非 critic 算法给 critic 字段被拒。
- [x] 2.2 v1 接受 ppo；`algorithm_flags.py` 把 `--gamma/--lambd/--value-clip/--num-critic-only-steps/--critic-load/--critic-lr` 移出 `_UNMAPPED`，做吸收与冲突检测。验证：翻译单测；miles-next-venv 中 upstream `parse_args` 解析生成 argv 通过。
- [x] 2.3 实现启动前拒绝：elastic/indep_dp、kl_coef≠0、critic GPU 数≠actor、`--deploy-component trainer`、decoupled 外层、`param_mode=lora`（design D2）。验证：参数化单测在 fake 组合根中、GPU 进程前失败。
- [x] 2.4 修正 `capabilities.py:321-324` 报错文案，`run_config.py` 增加 critic 字段。验证：单测断言文案不含"legacy"误导表述；`tests/test_rl_argv_snapshot.py` 不改即通过。

## 3. ports 单岛 colocated PPO

- [x] 3.1 放开 `entry.py:245-262` receipt family（新增 ppo family）、`entry.py:1590-1592`、`trainer_rebuild.py:253,358`，`selection.py:45,68` 不再把 ppo 路由到 legacy（design D3）。验证：fake engine 单测：未放行拒绝、带 `--rl-allow-unverified-mechanism` 单岛可启动；dry-run argv 快照含 `--advantage-estimator ppo` 与 critic 参数。
- [x] 3.2 `local_learner.py` 角色表增加 ppo→{actor,critic}；指标管线透传 value_loss 与 explained variance。验证：fake 运行的指标中出现这两项。
- [ ] 3.3 GPU G1：PPO 1×H100 单岛 3 轮（0.5B 级小模型），对齐 Miles `test_qwen3_4B_ppo` 的指标项（value_loss、EV 有限；policy loss、grad_norm 有限）。**需用户批准预算与机型**，估 ~1 h，≈$4（上限 $8）。验证：运行 ID、指标摘要、GPU 名写入 progress.md；通过后 `entry.py:236`、`fake.py:70` 正式声明 critic=True。

## 4. critic 状态契约与两岛 G3

- [x] 4.1 LayoutHash/receipt 增加 critic layout 哈希、param_mode、初始化来源哈希（design D4）。验证：单测：receipt 含两个 layout 哈希；critic layout 不一致时拒绝。
- [ ] 4.2 strict-avg 对 actor、critic 分别平均，两者成功才提交；decoupled 外层遇 critic 拒绝。验证：CPU 单测（fake 两岛）：哈希一致、critic 失败时整轮回滚。
- [ ] 4.3 elastic checkpoint store 增加 critic 权重与优化器状态，恢复时轮次一致性校验。验证：CPU 单测：保存/恢复哈希一致、轮次不一致拒绝。
- [x] 4.4 tape/ledger 记录 critic 权重哈希、value_loss、EV。验证：单测读取 ledger 条目。
- [ ] 4.5 GPU G3：PPO 两岛 strict-avg 1+1×H100 3 轮 + 一次 kill/resume。**需用户批准预算与机型**，估 ~1.5 h×2 卡，≈$12（上限 $20）。验证：两岛 actor/critic 平均后哈希一致；resume 后哈希等于最后提交轮；结果写入 progress.md。

## 5. critic warm-up 初始化

- [ ] 5.1 实现阶段 W 编排：非 rebuild 单次启动，`--critic-load` 指向初始 actor、`--num-critic-only-steps=warmup_steps`，只产出 critic checkpoint 并计算内容哈希；主阶段 `--num-critic-only-steps=0`、`--critic-load` 指向产物（design D5）。验证：dry-run argv 快照两阶段；单测确认主阶段在 rebuild 下 critic-only 步数为 0。
- [ ] 5.2 校验与复用：warm-up 前后 actor 哈希一致校验；两岛共用同一产物（按哈希复用，不重复跑）。验证：fake 单测。
- [ ] 5.3 GPU G1：warm-up 50 步 + 主阶段 2 轮，1×H100。**需用户批准预算与机型**，估 ~1 h，≈$4（上限 $8）；可与 3.3 合并在同一容器以省冷启动。验证：actor 哈希不变、主阶段 receipt 含产物哈希、warm-up 后 EV 高于随机初始化对照的首轮（记录数值，不设硬阈值）。

## 6. fork 共享 GAE 扩展点（需用户同意 fork 提交）

- [x] 6.1 新建 `tests/rl_gae_reference.py`：独立 torch 参考实现 vanilla、length_adaptive、decoupled、cross_segment GAE（注明论文公式），不 import 被测代码。验证：手算 3–5 个元素自检。
- [x] 6.2 fork `yeto/ports` 在 math_utils.py 加 `--gae-variant` 分派与 segment id 输入，arguments.py 加参数，缺省逐元素不变（design D6）。验证：fork CPU 测试对拍参考实现，覆盖无段边界退化、两段修正 (γλ)^{n_2}、α=1.5 λ 值；原有 fork 测试全过；结果写 progress.md。
- [ ] 6.3 经用户确认 push、更新 `MILES_NEXT_COMMIT` pin 与镜像；yeto 映射表登记新参数。验证：pin 指向新提交、镜像 digest 记录；parse_args 解析通过。
- [ ] 6.4 GPU G1：length_adaptive 与 cross_segment（人造两段数据）各 1×H100 2 轮，可同容器串行。**需用户批准预算与机型**，估 ~1 h，≈$4（上限 $8）。验证：指标有限，通过后正式声明。

## 7. VAPO

- [ ] 7.1 按论文核定 VAPO 组件与参数（Open Question），写入 design 补充。验证：progress.md 记录论文出处与参数表，用户确认。
- [ ] 7.2 fork 上补 VAPO 缺失部分（若有 GAE 外组件），yeto 声明 vapo 并翻译为 PPO+length_adaptive+decoupled+warm-up。验证：fork CPU 对拍测试；yeto 翻译单测与 dry-run 快照。
- [ ] 7.3 GPU G1 1×H100 3 轮 + G3 1+1×H100 3 轮。**需用户批准预算与机型**，估 G1 ~1 h + G3 ~1 h×2 卡，≈$12（上限 $20）。验证：G1 指标有限后正式声明；G3 哈希一致。

## 8. SAO 迁移到 ports

- [ ] 8.1 取得 agentenv/miles ae475060 中 SAO 源码（sao_dis、HL-Gauss value loss），核对与 docs/TBENCH21_SAO_QWEN35_08B_VALIDATION_20260826.md 一致；取不到则暂停本组并报告。验证：progress.md 记录来源提交与文件清单。
- [ ] 8.2 移植到 fork `yeto/ports`（value_loss=hl_gauss 51-bin、sao_dis），缺省不变。验证：fork CPU 对拍测试。
- [ ] 8.3 yeto 声明 sao 并把 `sao_streaming_runtime.py` recipe 翻译为 AlgorithmSpec，保留双 layout、双 syncer、lockstep 成对 fragment；旧入口保留。验证：单测：新旧路径生成的 critic/actor 配置等价；旧入口回归测试不变。
- [ ] 8.4 GPU：SAO on ports G1 1×H100 + G3 1+1×H100 各 3 轮（Qwen3.5-0.8B）。**需用户批准预算与机型**，估 ~2.5 h 卡时，≈$10（上限 $20）。验证：EV 与旧路径同量级（记录数值），两岛哈希一致。

## 9. CompactionRL

- [ ] 9.1 核对 Miles `examples/experimental/terminus-compaction` 可复用部分与 yeto agent/Terminal-Bench rollout 路径的接入点。验证：progress.md 记录复用清单与缺口。
- [ ] 9.2 rollout 侧实现压缩触发（`C−|h_t|<T_comp`，10,240）、`<analysis>/<summary>` 9 节摘要、重建上下文（k=2）、最多 3 次压缩、segment id 输出、共享回报（design D8）。验证：CPU 单测用假模型：触发、上限、段编号、重建内容。
- [ ] 9.3 yeto 声明 compactionrl 并翻译（cross_segment、α=1.5、γ=1、kl=0、critic lr 3e-6、critic_updates_per_step=2、warm-up 50、token 级归一化、每提示 1 条 rollout）。验证：翻译单测与 dry-run 快照；若 critic 每步 2 次更新需 fork 改动，并入 6.2 流程。
- [ ] 9.4 GPU G1：1×H100 小模型 agent 环境 3 轮（数据集/环境待选定）。**需用户批准预算与机型**，估 ~2 h，≈$8（上限 $16）。验证：出现至少一次压缩、段编号与优势修正可在样本中核对、指标有限。
- [ ] 9.5 GPU G3：1+1×H100 3 轮。**需用户批准预算与机型**，估 ~2 h×2 卡，≈$16（上限 $30）。验证：两岛哈希一致。消融臂（去掉 cross-segment）只在用户另批预算时执行。

## 10. critic LoRA 开发计划阶段（后续阶段，首轮不实现）

- [ ] 10.1 （后续）fork：model_provider/LoRA 包装让 critic 挂 adapter，value head 默认全参数可训练，critic checkpoint 只存 adapter+value head，warm-up 与 offload 兼容（design D9）。验证：fork CPU 单测：可训练参数数、冻结掩码、value head 可训练，替代 `test_lora_critic_skips_lora_setup` 的期望。
- [ ] 10.2 （后续）yeto：放开 `param_mode=lora`，layout 哈希区分 full/lora，strict-avg 只平均 adapter+value head，checkpoint 按 adapter 粒度。验证：CPU 单测与 dry-run 快照。
- [ ] 10.3 （后续）评估 critic 与 actor 共享冻结 backbone 的显存收益。验证：设计备忘写入 design 补充。
- [ ] 10.4 （后续）GPU G1 1×H100 对比 full critic EV 曲线 + G3 1+1×H100 哈希一致。**需用户批准预算与机型**，估 ~3 h 卡时，≈$12（上限 $20）。验证：progress.md 记录。

## 11. 文档与能力页

- [ ] 11.1 更新 `docs/MILES_RL.md`：critic 家族参数、默认值、拒绝规则、warm-up 两阶段、GAE 变体、各机制验证状态（未验证的标"未验证"）。验证：文档示例命令 `--dry-run` 执行结果与描述一致。
- [ ] 11.2 更新能力页/P0 change 中 critic 家族状态。验证：能力表与 adapter 声明一致的单测通过。

> GPU 预算估算总表（未经实测，首轮不含第 10 组）：3.3 $4、4.5 $12、5.3 $4（可与 3.3 合并）、6.4 $4、7.3 $12、8.4 $10、9.4 $8、9.5 $16 → 合计 ≈ $70，建议上限 $130（含重试）；第 10 组后续另批 ≈ $12（上限 $20）。
