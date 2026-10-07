# rl-algo-critic-family 进展

## 2026-10-06（Agent CRITIC-APPLY，第 1–5 组 CPU 任务）

worktree `/home/michael/work/critic`，分支 `critic`，基于 integ-decl `de27c301`。不改 fork（第 6 组另由他人负责），不使用 GPU 或云资源。

### 1.1 用户决策原文（用户，2026-10-06；摘自 proposal.md "已确认的决策"）

1. 两岛外层同步：critic 与 actor 一起做 strict-avg，先跑通；独立 DiLoCo 等其它方式留作后续探索。
2. critic 参数模式：先全参数跑通（与 Miles 现状一致）；本 change 的 design 与 tasks 必须包含 critic LoRA 的开发计划（预留接口与契约，不在首轮实现）。
3. critic 初始化：复制初始 actor 的 backbone，加新的 value head，再做 critic-only warm-up（与 CompactionRL 的"同一 checkpoint 初始化 + 50 步 value pretraining"一致）。
4. 接受 Miles shared PPO 的限制（不支持 indep_dp 因而不与 elastic 共存、`kl_coef` 为 0、critic 与 actor 共用训练 GPU）；更好的弹性分配方式后续另行探索。
5. CompactionRL 指 arXiv:2607.05378（Li 等，v2 2026-10-01）：单策略同时优化任务执行与上下文压缩摘要；PPO 风格 clip 目标，token 级归一化，KL=0，单 rollout/提示；critic 与策略同一 checkpoint 初始化加标量 value head，50 步 value pretraining，critic lr 3e-6，每批 2 次 critic 更新对 1 次策略更新；cross-segment GAE（段内局部 GAE，再乘 (γλ)^{N_{>s}} 修正，不跨压缩边界自举），γ=1，长度自适应 λ=1−1/(αl)、α=1.5；rollout 侧在剩余上下文 < T_comp（10,240）时触发压缩，摘要由同一策略生成、与任务共享回报，每条 rollout 最多 3 次压缩。论文用 slime 训练，未见代码发布。

### 1.2 Miles 行号复核

pin：`yeto/rl/__init__.py:43` `MILES_NEXT_COMMIT = c35702eefcf2862cee155e46870e6ad30568d2c6`，`entry.py:178` 的 `_PINS_0AF62F4D_PLUS` 也含该提交（与 ports 镜像一致）。方法：`git -C /home/michael/work/miles-2b show c35702e:<path>`。

| design 表述 | 实测 | 结论 |
|---|---|---|
| `--critic-num-nodes/gpus` arguments.py:270-273 | :270 `--critic-num-nodes`、:273 `--critic-num-gpus-per-node` | 一致 |
| `--num-critic-only-steps` :1597 | :1597 | 一致 |
| `--critic-load/save/lr/lr-warmup` :1603-1613 | `--critic-load` :1603、`--critic-lr` :1611、`--critic-lr-warmup-iters` :1613 | 一致 |
| `--value-clip=0.2` :1648 | :1648 default 0.2 | 一致 |
| `--gamma/--lambd` 默认 1.0 :1729-1730 | :1729/:1730 | 一致 |
| 估计器 choices :1682-1690 | :1682-1690，含 ppo，无 vapo | 一致 |
| `use_critic` 由 estimator 推导 :3591 | :3591 `args.use_critic = args.advantage_estimator == "ppo"` | 一致 |
| shared 约束（indep_dp、megatron、kl_coef==0）:3592-3604 | :3592-3604 | 一致 |
| critic GPU 数=actor :3605-3606 | :3605-3606 **是赋值**（`critic_num_* = actor_num_*`），不是断言 | **差异**：Miles 静默覆盖用户给的 critic GPU 数；yeto 侧 2.3 显式拒绝不一致的值，避免被静默改写 |
| critic_load/lr 默认继承、强制 offload_train :3708-3716 | 继承在 **:3607-3610**（`critic_load=load`、`critic_lr=lr`）；offload_train 强制在 :3708-3716 | **差异**：继承的行号是 :3607-3610 |
| rebuild 要求 `num_critic_only_steps==0` :3212 | :3210-3212 断言 | 一致 |
| `--deploy-component trainer` 禁 critic :3058 | :3058 `assert not args.use_critic` | 一致 |
| placement_group.py:320,338 共卡 | :320-321、:338-339 `ans["critic"] = ans["actor"]` | 一致 |
| model_provider.py:340-341 1 维 value head | :340-341（GPTModel 分支），另有 custom provider 分支 :160-164 | 一致（补充 :160-164） |
| actor.py:227,604,635 value_loss | :227 role==critic、:604-606 `_train_critic`、:635 `loss_type="value_loss"` | 一致 |
| math_utils.py:647,705,875 vanilla_gae、:899 chunked_gae | 文件为 `miles/backends/training_utils/loss_hub/math_utils.py`；:647 `get_advantages_and_returns`、:705 `get_advantages_and_returns_batch`、:875 `vanilla_gae`、:899 `chunked_gae` | 一致（:647/:705 是 GAE 入口而非 vanilla_gae 本体，表述补充） |
| losses.py:452 value loss | `loss_hub/losses.py:452 value_loss_function` | 一致 |
| LoRA 下 critic 全参数 test_lora_model_branches.py:122-130 | :122 `test_lora_critic_skips_lora_setup`、:129-130 | 一致 |
| test_qwen3_4B_ppo.py:76-86 | :76-86 ppo_args（`--num-critic-only-steps 1`、`--critic-lr 1e-5`、`--kl-coef 0.00`） | 一致 |

Miles `train.py`（c35702e）:118-126 的 critic 训练顺序：`values = critic.train(rollout_id, data)` → `critic.offload()`（offload_train 时）→ `rollout_id >= num_critic_only_steps` 时 `actor.train(..., external_data=values)` → `actor.offload()` → `remove_train_output_refs(values)`。ports 训练器按此顺序接入（3.1/3.2）。

### 1.3 pytest 失败基线

`PYTHONPATH=. /tmp/yeto-venv/bin/python -m pytest tests -q --continue-on-collection-errors -p no:cacheprovider`（worktree 未改动时）：`68 failed, 3962 passed, 49 skipped, 26 errors`，日志 `/home/michael/work/infra-drafts/critic-pytest-base.log`。`baseline-failures.txt` 共 94 行 = 68 + 26，与汇总一致。按"去 ' - ' 截 80 字符 sort -u"规约后 85 条，与 `/tmp/base2.sorted` 去重后完全相同。

### 第 2 组（CPU，已实现、已验证）

实现：
- `yeto/rl/engine/algorithm.py`：新 `CriticSpec` 组（value_clip、critic_lr、critic_lr_warmup、critic_updates_per_step、value_loss、hl_gauss_bins、init、load、warmup_steps、param_mode、预留 lora_rank/lora_alpha/lora_target_modules），全部缺省 None；`execution.needs_critic` 为真时填入具体缺省值（value_clip 0.2、init copy_actor_backbone、warmup 0、param_mode full 等）并整组进入规范化 JSON，否则 `critic` 键不出现。advantage 组注册扩展字段 `lambd/lambd_mode/alpha/gae_variant`（缺省 None，不输出）；critic 规格填入 lambd 1.0、fixed、vanilla（length_adaptive 时 alpha 1.5）。v1 `advantage_estimator="ppo"` 被接受并隐含 `needs_critic=True`。拒绝矩阵新增 `critic_fields_without_critic`、`critic_reward_kl`、`critic_param_mode`（lora："planned … not implemented"）、`critic_not_at_pin`（非 vanilla GAE / length_adaptive / hl_gauss / 每步多次 critic 更新需第 6 组 fork 扩展点，当前 pin 没有）、`critic_init`。`_reject_critic` 文案去掉 legacy。`EXECUTION_ALLOWANCES={"execution:critic"}`、`allowance_names()`；`effective_default_at()` 让吸收时"填入的缺省"不算用户值。
- **与 design D1 的偏离**：`advantage.gamma` 已由 `rl-algo-seq-and-adv`（`yeto/rl/algos/seq_adv.py`，缺省 1.0、不输出）注册并映射 `--gamma`，因此 critic 复用该字段而不新建；ppo 的 gamma=1.0 不显式进入哈希（缺省值确定，argv 中 `critic_argv` 总是显式输出 `--gamma`）。为不改变 seq_adv 源码哈希（它是 GDPO/MaxRL/MAPO 规格中 PluginRef 的一部分），在新模块中包装了 seq_adv 的 `seq_adv_gamma` 规则（critic 算法放行）与 `--gamma` 翻译（critic 下由 `critic_argv` 输出，避免重复）。
- 新模块 `yeto/rl/algos/critic.py`（加入 `EXTENSION_MODULES`）：flag 行 `--lambd/--value-clip/--critic-lr/--critic-lr-warmup-iters/--num-critic-only-steps/--critic-load`（吸收 + 冲突检测；`--critic-load` → `critic.init=load`；extra argv 中的 `--num-critic-only-steps N` → `critic.warmup_steps=N`，主阶段一律输出 0）；`critic_argv`；运行级拒绝 `critic_run_problems`（elastic、`--indep-dp`、`--deploy-component trainer`、critic GPU 数≠actor、decoupled 外层），注册为 launch check 与 island check。`--critic-lr-warmup-iters` 补入 `_UNMAPPED`（上游存在、改变目标，之前漏列）。`--advantage-estimator ppo` 吸收时同时设 `execution.needs_critic=True`。
- 调用点：`launcher.py`（sync_preset、elastic）、`learner.verify_ports_algorithm`（同）、`miles_adapter/config.translate_run_config`（extra_argv、actor GPU 数、elastic=use_miles_router）；`driver.py` 握手新增 `_critic_run_problems`（elastic_hook 存在、sync 为 decoupled 时在任何引擎动作前失败）；`bridges.DecoupledSync.OUTER_SYNC_KIND="decoupled"`。
- `capabilities.py`：critic 不匹配文案改为真实原因并提示 `--rl-allow-unverified-mechanism execution:critic`；`execution:critic` 放行时豁免该项。
- `run_config.py`：`AlgorithmConfig.critic: CriticRunConfig | None`（critic_load、init_sha256，成对出现、hash 格式校验）与 `resolve_critic_run_config`；learner 新增 `--rl-critic-load/--rl-critic-init-sha256`；`config.critic_load_argv`：warm-up>0 时主阶段必须给出 W 产物，否则拒绝。

验证：
- `tests/test_rl_critic_spec.py` 35 项通过（golden 哈希、哈希变化、非 critic 拒绝、翻译、吸收/冲突、fake 组合根中 kl/lora/elastic/decoupled 在任何引擎动作前失败、运行级拒绝、文案、run config）。
- 哈希不变证据：`evidence/hash_compare.py` 在 integ-decl de27c301（`git archive` 到 /tmp/critic-base）与本分支各跑一次，17 个既有规格（默认 GRPO、v1 bounded、kl_loss、clip_higher、rpp、cispo、tis、`examples/rl_algorithms/*.json`、`rl-algo-seq-and-adv/examples/*.json`）的 sha256 与 `algorithm_argv` 输出 `evidence/hash-critic-base.txt` 与 `evidence/hash-critic.txt` 逐字节相同。`tests/test_rl_argv_snapshot.py` 未改动并通过。
- upstream 解析：`evidence/upstream_parse_ppo.py`，`PYTHONPATH=/tmp/miles-c35702e`（`git archive c35702e`）+ `/home/michael/work/miles-next-venv`。该 venv 无 megatron-core，故用 FSDP 解析器解析、`miles_validate_args` 前把 train_backend 置为 megatron（shared PPO 块要求）。3 例（缺省、调参、critic-load）全部通过：`use_critic=True`、`num_critic_only_steps=0`、gamma/lambd/value_clip 与规格一致、`offload_train=True`、`kl_coef=0.0`。日志 `evidence/upstream_parse_ppo.log`。**未验证**：Megatron 解析器路径本身（venv 缺 megatron-core）。
- 为反映有意的行为变化而修改的既有测试：`test_rl_algorithm_capabilities.py::test_critic_rejected_with_the_real_reason`（原断言 legacy 文案）、`test_rl_seq_adv.py`（`--lambd` 现已映射）、`test_rl_algorithm_spec_v2.py` / `test_rl_engine_algorithm.py`（v1 拒绝示例由 ppo 改为 gspo）、`test_rl_algorithm_flags.py`（允许的组加 critic）、`test_rl_miles_adapter_config.py`（叶子表加 critic run config）。
- 全量回归（OMP/OPENBLAS/MKL 线程数=1，因本机用户线程数接近 4096 上限）：`68 failed, 3995 passed, 51 skipped, 26 errors`，失败集合规约后与基线完全相同（新增 0）。

### 第 3 组 CPU 部分（3.1、3.2 已实现、CPU 已验证；3.3 GPU 未执行）

实现：
- `selection.py`：`--advantage-estimator ppo`（及 `use_critic` + ppo）不再路由到 legacy；`--use-critic`（legacy fork 专有）与其它非 GRPO 估计器经 extra argv 仍拒绝。
- `entry.py`：`receipt_role_family` 对 ppo+needs_critic 返回 `"ppo"`（未声明 needs_critic 仍拒绝）；组合根保留 Miles 创建的 critic（与 spec 的 needs_critic 必须一致，否则启动失败），包成 `SwappableActor` 交给 disposer 与 `compose_island(critic_model=...)`。`entry.py:236` 与 `fake.py:70` 仍声明 `critic=False`（design D3：正式声明在 G1 之后）。
- `trainer.py`：按 Miles `train.py:118-126` 顺序：安装 critic 记录器插件 → `critic.train` → 读 critic grad norm 与 step losses → `offload_train` 时 `critic.offload()` → `actor.train(..., external_data=critic_outputs)` → 释放 critic 输出（`remove_train_output_refs`）与 rollout 数据；critic 失败抛 `TrainStepError`，两类引用都释放。
- `trainer_rebuild.py`：`rebuild_same_shape/rebuild_resharded` 增加可选 `critic` 句柄（旧句柄一并传给 Miles，重建后 `swap_critic`）；重建结果与句柄不一致时报错（elastic+critic 已在 2.3 拒绝，本轮不会走到）。
- `contracts.py` `_ALGORITHMS` 增加 ppo；`local_learner.py` `_ROLES_BY_ALGORITHM["ppo"]={actor,critic}`，ppo 可用 role lane（`stream_role`），与 SAO 相同的双 layout 方式。
- 指标（3.2）：`state_plugin.install_value_metrics_recorder` 在 critic 进程中包装 Miles `value_loss_function`，把 `explained_variance = 1 − Var(returns − values_old)/Var(returns)`（掩码后 token；returns 与 values 均由 Miles 计算，yeto 不做 GAE）加入 loss dict；`trainer.round_metrics()` 增加 `critic/value_loss`、`critic/value_clipfrac`、`critic/explained_variance`、`critic/grad_norm`，进入 `rl_round_trained.train_metrics`。fake 引擎新增 `critic=True` 模式（先 critic 后 actor，receipt 家族 ppo，报告两项指标）。

验证（CPU）：`tests/test_rl_critic_ports.py` 14 项：未放行拒绝且提示 `execution:critic`、Miles 适配器未声明 critic；带 `advantage_estimators:ppo` + `execution:critic` 放行时 fake 单岛跑 2 轮，事件中每轮 `train_metrics` 含有限的 `critic/value_loss` 与 `critic/explained_variance`；两岛或有外层同步时放行被拒；dry-run argv 快照；receipt 家族；critic 句柄替换；Miles 训练器调用顺序（critic 先训练并 offload，actor 收到 external_data）、失败时释放；GRPO 训练器调用不变；EV 数值；ppo 角色表与 role lane 哈希。修改的既有测试：`test_rl_engine_selection.py`（ppo 不再路由 legacy，原 ppo 例改为 gspo，并新增 ppo 放行用例）、`test_rl_round_accounting.py`（ppo+needs_critic → "ppo"）；`local_learner` 报错文案保留原前缀 "require an SAO role"。

**未验证（需 GPU G1，3.3）**：真实 Miles 下 critic 训练顺序与 offload 是否与 shared PPO 生命周期兼容；`external_data` 与 ports 单 cell 训练的配合；EV 在 Miles loss dict 中跨 micro-batch 的归约方式（按 micro-batch 计算，归约语义未核实）；critic 进程中 `train_one_step` 记录器是否生效。

### 第 4 组 CPU 部分（4.1、4.4 已实现并验证；4.2、4.3 只完成协议层，**未勾选**，见"设计问题"；4.5 GPU 未执行）

实现：
- 新模块 `yeto/rl/critic_state.py`：`critic_layout_hash`（critic 参数 specs + value head 形状 + param_mode + 预留 LoRA 形状，域分离 `yeto-rl-critic-layout-v1`，与 actor layout 分开）；`critic_weights_sha256`；`CriticRoundReceipt`（rollout_id、actor/critic 两个 layout 哈希、critic_param_mode、critic_init、critic_init_sha256、critic_weights_sha256、value_loss、explained_variance）；`check_critic_layouts`（任一岛 actor/critic layout 或 param_mode 不同即拒绝）；`TwoRoleStrictAvg`（同一轮先 actor 后 critic，二者都成功才提交，任一失败抛 `RoleAverageFailed` 且两 role 保持上一轮）；`CriticCheckpointStore`（`critic/round-N/` 权重 + 优化器状态 + 最后提交的 manifest；恢复时校验哈希，actor 轮次≠critic 轮次拒绝并报告两个轮次）。
- 生产接线（4.1/4.4）：`state_plugin.critic_state_summary`（critic 进程内每个 rank 的可训练参数 specs 与权重哈希）；`MilesTrainerGroup.critic_round_receipt`（汇总各 rank，value head = Miles critic `output_layer.weight`，init 来源哈希取自 run config 经 `runtime_attrs` 设置的 `yeto_rl_critic_init_sha256`）；driver 每轮在 `rl_round_trained` 后写 `rl_critic_round` 事件（仅 critic 算法；GRPO 的 tape 不变）。fake 引擎 critic 模式提供 critic 张量与同样的 receipt。

验证（CPU）：`tests/test_rl_critic_state.py` 11 项 + `test_rl_critic_ports.py::test_critic_round_receipt_from_the_critic_processes`：layout 哈希区分形状/LoRA；receipt 字段；layout 不一致拒绝；fake 两岛两 role 平均后哈希一致；critic 平均失败整轮回滚；layout 不一致在平均前拒绝；checkpoint 保存/恢复哈希一致、轮次不一致拒绝、篡改拒绝；fake 单岛 2 轮 tape 中 `rl_critic_round` 含每轮不同的 critic 权重哈希与有限 value_loss/EV；GRPO tape 无该事件。

#### 设计问题（暂停点，需主 agent/用户决定）

1. **4.2 生产 strict-avg 未接线**：ports 的 `StrictAvgSync` 经 `StrictRlBridge` 与 syncer 只交换 LoRA `CanonicalLoraState`（actor）。把全参数 critic 纳入 strict-avg 需要：(a) 第二条 syncer 通道（design D4"沿用 SAO 双 layout/双 syncer"，launcher 需为 critic 起第二个 syncer 与端口），或扩展单个 syncer 的 layout 同时容纳 LoRA actor 与全参数 critic；(b) Miles critic 进程内的 critic 张量导出/写回插件（全参数、可能经 distributed optimizer 分片）；(c) 两条通道之间的"同轮两者都成功才提交"——现有 syncer 每条通道各自提交，跨通道原子提交需要协议层（`TwoRoleStrictAvg` 只是该语义的 CPU 参照实现）。这超出 tasks 4.2 的 CPU 粒度，且选择 (a)/(b) 影响 launcher 与 syncer，故暂停。
2. **4.3 生产 checkpoint 未接线**：`RoundCutCheckpoint`/`MilesTrainerGroup.save_cut` 只存 actor LoRA+优化器分片。critic 需同样的 Miles 进程内保存/恢复（全参数 + 优化器状态 + 调度器），并在 pointer 中记录 critic 轮次；`CriticCheckpointStore` 提供了存储格式与轮次一致性校验，但 critic 张量的取得依赖第 1 点 (b)。另：design 写"elastic checkpoint store"，而 critic 与 elastic 已在 2.3 互斥；实际可用的是 `--rl-elastic-checkpoint-store` 驱动的 round-cut（单岛无 sync 也可用），建议在 design 中改述。
3. 4.1 的"critic layout 不一致时拒绝"目前在 `check_critic_layouts`/`TwoRoleStrictAvg` 中实现并测试；生产外层同步中的强制检查随第 1 点接线。

### 第 5 组 CPU 部分（5.1、5.2 已实现、CPU 已验证；5.3 GPU 未执行）

实现：新模块 `yeto/rl/critic_warmup.py`。
- `warmup_stage_argv(main_argv, spec, actor_checkpoint, critic_save)`：从主阶段（ports）argv 派生阶段 W argv：去掉 ports driver 专属钩子（rollout sample filter、all-samples hook、buffer filter）与 eval 参数、原有 critic 调度/检查点参数，追加 `--num-rollout N --num-critic-only-steps N --critic-load <初始 actor> --critic-save <产物目录> --save-interval N`（N=warmup_steps；Miles train.py 中 rollout_id < N 时不训练、不保存 actor）。只接受 `init=copy_actor_backbone` 且 warmup_steps>0 的 critic 规格。
- 主阶段：`critic_argv` 一律 `--num-critic-only-steps 0`，`config.critic_load_argv` 用 run config 的产物路径给出 `--critic-load`；产物哈希经 `runtime_attrs.yeto_rl_critic_init_sha256` 进入 `rl_critic_round` receipt。
- `checkpoint_sha256`（目录内容哈希，排除 manifest）；`finish_warmup`（阶段 W 后校验初始 actor 检查点哈希未变，计算 critic 哈希，原子写 `critic-warmup.json`）；`load_product`（校验 schema、算法哈希、warmup_steps、初始 actor 哈希、critic 内容哈希）；`ensure_warmup`（以 sha256(算法哈希:初始 actor 哈希) 为键，产物存在且有效则复用，否则只跑一次阶段 W）；`python -m yeto.rl.critic_warmup --dry-run` 输出两阶段 argv。

验证：`tests/test_rl_critic_warmup.py` 9 项：主阶段 critic-only 步数为 0 且加载产物；从完整主阶段 argv 派生的阶段 W argv（钩子/eval 去除、模型/批次/算法参数与主阶段一致）；非 warm 规格拒绝；dry-run 两阶段快照；目录哈希按内容；两岛复用同一产物（阶段 W 只跑 1 次）；warm-up 中 actor 被改动时拒绝；篡改/异算法/异 actor 产物拒绝；主阶段 runtime attr 携带产物哈希。

**设计备注**：1.2 复核时发现 Miles :3210-3212 的"rebuild 模式要求 num_critic_only_steps==0"位于 `--rematerialize-param-from-master-weight` 的校验函数内（该模式不支持 LoRA），ports（LoRA）并不传该参数。design D5 所述"ports 走 rebuild 所以不能 critic-only"的理由在 c35702e 上不成立；真正的约束是 yeto 驱动每轮都训练并发布 actor（门槛、梯度不变量、发布），在 ports 循环内跳过 actor 需要另设计。独立阶段 W 的方案仍成立，建议更正 D5 的理由表述（未改 design，留给主 agent/用户）。
**未验证（GPU 5.3）**：阶段 W argv 在真实 Miles train.py 上能否启动（去掉的钩子集合是否足够/过多）、critic 保存目录结构、主阶段 `--critic-load` 能否加载阶段 W 产物。
- 回归：全量 `69 failed` 中唯一新增项 `test_provenance::test_production_tree_has_no_unsafe_torch_load` 由组 4 的 `critic_state.py` 引入（torch.load 未加 weights_only），本提交已修复并复测通过；其余失败集合与 /tmp/base2.sorted 相同。
