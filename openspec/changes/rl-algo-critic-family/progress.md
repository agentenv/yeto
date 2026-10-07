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

## 第 6 组（来自分支 critic-gae）
# rl-algo-critic-family progress

## 6.1 参考实现（2026-10-06）
- `tests/rl_gae_reference.py`：独立 torch 参考（不 import Miles）——vanilla、length_adaptive（λ=1−1/(αl)，α=1.5）、decoupled（A 用 λ_policy，R=GAE(λ_critic)+V）、cross_segment（段内局部 GAE、段尾 bootstrap 0、终局回报在末段段尾、乘 (γλ)^{N_{>s}}）。
- `tests/test_rl_gae_reference.py`：8 passed（手算 3 元 vanilla、γ=λ=1 等于 MC、λ(100,1.5)=1−1/150、α→∞→λ=1、decoupled 手算、单段 cross_segment==vanilla、两段手算 ×(γλ)^{n_2}、三段因子）。

## 6.2 fork 扩展点（本地提交，未 push）
- 分支 `yeto-gae-variant`（worktree /home/michael/work/miles-gae，基于 `yeto/ports` 039471508），提交 `ce96fc060`。
- 改动：loss_hub/math_utils.py（`segmented_gae`、`length_adaptive_lambda`、`get_advantages_and_returns_batch` 新 kwargs）、loss_hub/advantages.py（缺省不传任何新 kwarg）、loss.py（传 `rollout_data["segment_ids"]`）、arguments.py（`--gae-variant/--gae-lambd-mode/--gae-length-alpha/--gae-critic-lambd`）、ray/rollout/train_data_conversion.py（`sample.metadata["segment_ids"]` → train data → 分片）。
- 语义：段与 N_{>s} 在可训练 token 子序列上计（与掩码 token 非 MDP 转移一致）；length_adaptive 的 l 用响应长度 R_i；returns = A + V（decoupled 时用 critic λ）。
- 测试（/home/michael/work/miles-next-venv）：`test_ppo_gae_variants.py` 22 passed（对拍拷贝的参考实现；缺省/显式 vanilla 与冻结的原实现 `torch.equal` 逐元素一致，fp32/fp64、chunked/非 chunked、带掩码）；`test_segment_ids_conversion.py` 2 passed。
- 定向回归（GAE/loss 相关文件 + tests/test_chunked_gae.py）：基线 039471508 13 failed/99 passed，新 13 failed/121 passed，失败集合完全相同（megatron.core 缺失等环境原因）。未跑 tests/fast/ray 全量（会启动 Ray，线程上限）。
- yeto 全量回归：log /home/michael/work/infra-drafts/critic-gae-pytest.log，68 failed/26 errors，与 /tmp/base2.sorted 按用例 id 比对集合相同，新增失败 0。

## S13 7.1 VAPO 参数（2026-10-07，用户已确认，见"S13 VAPO/CompactionRL 用户决定落实"）

出处：VAPO, arXiv 2504.05118v3（11 Apr 2025），取自 arXiv HTML 版（https://arxiv.org/html/2504.05118v3）。

| 组件 | 论文值 | 出处 | yeto 规格字段 / Miles 参数 |
|---|---|---|---|
| 基座 | Qwen2.5-32B base | Abstract、图 1、§5.1 | 运行配置 |
| γ | 1.0 | §5.1 basic PPO | `advantage.gamma` / `--gamma` |
| Value-Pretraining | 固定策略采样、MC 回报训练 value 至 value loss/EV 足够低；50 步 | §4.1 步骤 1–3；§5.1 第 1 条 | `critic.warmup_steps=50`（阶段 W，D5） |
| value 初始化 | 由奖励模型初始化 | §4.1、§5.1 | **不一致**：yeto `copy_actor_backbone`（无 RM） |
| Decoupled GAE | critic 目标 λ=1.0，policy 用另一 λ | §4.1；§5.1 第 2 条 | `advantage.gae_variant=decoupled`、`critic_lambd=1.0` / `--gae-variant decoupled --gae-critic-lambd 1.0` |
| Length-adaptive GAE | λ_policy=1−1/(α·l)，**α=0.05** | §4.2 式 (4)(5)；§5.1 第 3 条 | `lambd_mode=length_adaptive, alpha=0.05` / `--gae-lambd-mode length_adaptive --gae-length-alpha 0.05` |
| Clip-Higher | ε_low=0.2、ε_high=0.28 | §4.3 式 (8)；§5.1 第 4 条 | `loss.eps_clip/eps_clip_high` |
| Token-level loss | 批内全部 token 等权 | §4.2 式 (7)；§5.1 第 5 条 | `loss.aggregation=token` / `--calculate-per-token-loss` |
| Positive-example LM loss | L=L_PPO+μ·L_NLL，L_NLL 对正确样本 token 平均，μ=0.1 | §4.3 式 (9)(10)；§5.1 第 6 条 | `loss.positive_lm_coef=0.1` / `--positive-example-lm-loss-coef 0.1`（fork 新增） |
| Group-Sampling | 每 prompt 16 次，512 prompts/采样，mini-batch 512 | §4.3；§5.1 第 7 条 | 运行配置（n_samples_per_prompt 等，非算法哈希） |
| 学习率 | actor 1e-6、critic 2e-6，AdamW，warmup-constant | §5.1 basic PPO | `critic.critic_lr=2e-6`；actor lr 为运行配置 |
| 评测 | AIME24 avg@32，top_p=0.7，temperature=1.0 | §5.1 | — |
| 消融（AIME24） | Vanilla PPO 5；w/o Value-Pretraining 11；w/o Decoupled-GAE 33；w/o Length-adaptive 45；w/o Clip-Higher 46；w/o Token-level 53；w/o Pos-LM 54；w/o Group-Sampling 55；VAPO 60 | 表 1 | — |

论文未给出：value_clip（yeto 用 Miles 缺省 0.2）、KL（VAPO 未述；Miles shared PPO 要求 kl_coef=0）、lr warmup 步数、正确样本的奖励阈值（yeto 取 reward>0.0）、mini-batch 512 的单位（prompt 还是样本）、critic/actor 每批更新次数、max response length。basic PPO 基线的 λ=0.95、sample-level loss、ε=0.2、8192 prompts×1 为对照设置，非 VAPO 值。
**注意**：design D6 / fork `--gae-length-alpha` 缺省 α=1.5 来自 SAO；VAPO 论文值为 0.05（l=100 时 λ=0.8），`vapo_spec` 显式写 0.05。

## S13 7.2 VAPO 实现（CPU；GPU 7.3 未执行）

缺失判断（fork yeto-vapo 基于 ce96fc060）：decoupled / length-adaptive GAE 已有（6.2）；clip-higher（`--eps-clip-high`）、token-level loss（`--calculate-per-token-loss`）、group sampling（`--n-samples-per-prompt`）为既有参数；value-pretraining 由 D5 阶段 W 覆盖。**唯一缺失：positive-example LM loss**。

fork（/home/michael/work/miles-vapo，分支 yeto-vapo，本地提交 `cbf8c4737`，未 push）：
- `arguments.py`：`--positive-example-lm-loss-coef`（缺省 0 关闭）、`--positive-example-reward-threshold`（缺省 0.0，reward 严格大于阈值为正例）。
- `loss.py`：coef≠0 时在 `compute_advantages_and_returns` 由 rewards 生成每样本 `positive_example_flags`；`megatron_utils/model.py`、`fsdp_utils/actor.py` 的 get_batch 键表加入该键（缺失时为 None）。
- `loss_hub/losses.py`：`_positive_example_lm_loss` 用正例掩码构造与 PG loss 同一归约（`get_sum_of_sample_mean`，per-token 或 per-sample），loss += μ·NLL，指标 `positive_lm_loss`。
- 测试（miles-next-venv，OMP=1）：`tests/fast/backends/training_utils/loss/test_positive_example_lm_loss.py` 8 passed（独立参考：logits 上 log_softmax 重算 token logprob，per-sample/per-token 两种归约对拍 NLL 与总 loss；无正例时 loss/grad 不变；coef=0 时 loss/grad 逐元素相等且指标键不变；缺 flags 报错；flags 生成；rollout_data 只在启用时加键）。定向回归（loss/ + gae variants/masks + true_on_policy_loss_metrics）：基线 ce96fc060 13 failed/96 passed，新 13 failed/104 passed，失败集合相同（环境原因）。未启动 Ray。
- **与论文的偏差（需确认）**：论文式 (9) 按正例 token 总数归一；fork 实现沿用 PG loss 的归一（per-token 模式下除以批内全部 token 数，per-sample 模式下按样本均值再除 batch size），即 NLL 实际权重 ≈ μ×正例 token 占比。按正例 token 全局归一需跨 micro-batch/DP 的额外 all-reduce，未做。

yeto（/home/michael/work/s13-vapo，分支 s13-vapo）：
- `algorithm.py`：新扩展字段 `advantage.critic_lambd`（gae_variant=decoupled 时自动填 1.0 进入哈希）；`critic_not_at_pin` 放行 `gae_variant=decoupled` 与 `lambd_mode=length_adaptive`（cross_segment、hl_gauss、critic_updates_per_step≠1 仍拒绝）；critic_lambd 只能配 decoupled。
- `algos/critic.py`：`gae_variant_argv`（接在 `critic_argv` 尾部，vanilla+fixed 时为空，普通 PPO argv 不变）、flag 行 `--gae-variant/--gae-lambd-mode/--gae-length-alpha/--gae-critic-lambd`、字段 `loss.positive_lm_coef/positive_lm_reward_threshold`（二者必须同时给出）与 `--positive-example-*` 行；机制 `features:gae_decoupled`、`features:gae_length_adaptive`、`features:positive_example_lm_loss`（Miles 适配器不声明，需 `--rl-allow-unverified-mechanism`，正式声明待 G1）；`FORK_FLAGS`。
- `algorithm_flags.py` `_UNMAPPED` 加入 6 个 fork flag（使其为 adapter-owned，不能裸透传）；`tests/test_rl_algorithm_flags_upstream.py` 的 pending 并入 `critic.FORK_FLAGS`（上游 c35702e 无这些参数）。
- `algos/vapo.py`：`vapo_spec()`、`PAPER_PARAMETERS`（带出处）、`PAPER_RUN_SETTINGS`、`NOT_IN_PAPER`。vapo 规格 sha256 `7ee1dde4…386f`。
- 测试：`tests/test_rl_vapo.py`（论文值、翻译快照、dry-run 快照与 extra argv 吸收等价、未放行拒绝、阶段 W 50 步且 GAE 参数与主阶段一致、哈希区分、decoupled 自动填 λ、拒绝矩阵、普通 PPO/GRPO argv 不变）；两批定向运行（critic/algorithm/flags/capabilities/seq_adv/loss_variants/selection/adapter config 等）合计 760 passed / 14 skipped（含未改动的 `test_rl_argv_snapshot.py`）。
- 哈希不变：`evidence/hash_compare.py` 在 d398d443（`git archive` 到 /tmp/vapo-base）与本分支输出 `evidence/hash-vapo-base.txt` / `evidence/hash-vapo.txt` 17 行逐字节相同，且与第 2 组 `hash-critic.txt` 相同。

**未验证（需 GPU 7.3）**：fork 正例 LM loss 在真实 Megatron/FSDP 训练中的 flags 传递（get_batch 键）、CP>1 下的归约；PPO 下 `rollout_data["rewards"]` 是否为原始标量奖励（未核实 reward 后处理对阈值语义的影响）；GAE 变体在真实 critic 下的数值；yeto pin 尚未指向包含 ce96fc060+cbf8c4737 的 fork 提交（loss_variants 的 FORK_COMMITS 式 pin 门控未为 VAPO 实现，目前仅靠未声明机制拦截）。

## S13 8.x SAO（2026-10-07，分支 s13-sao / fork yeto-sao；8.1–8.3 CPU 已实现并 CPU 验证，8.4 GPU 未执行）

### 8.1 源码来源（核对结论：任务里写的 ae475060 不是 SAO 源码）
- `ae475060fa670145aef75d678809039ae999cb97` 只存在于 yeto 的 vendored bundle `yeto/rl/vendor/miles-qwen38.bundle`（GitHub API 422、`upload-pack: not our ref`）。从 bundle 取出后：提交信息 "fix(trainable_state): expose fp32 masters without resetting grad accumulators"，仅改 `trainable_state.py` 与其测试；树内无 `sao_dis`/`hl_gauss`/`sao.py`。它是 legacy 流水的 `MILES_COMMIT`，不是 SAO 数学的来源。
- 实际来源：`agentenv/miles` 分支 `feat/sao-tbench21-e2e-validation` @ `16a9bea409de61549e233dda8a684e8cdd1f7448`（SAO 数学引入于 `e25048edd` "feat(rl): add SAO value and online training"，后续 `d5eb5e82a`、`eb33e9e2e`、`ca1390beb`、`79bc3ad89`）。只读 fetch 到 /home/michael/work/miles-sao 的 `refs/remotes/agentenv/*`（未 push）。
- 文档核对：该分支的 `docs/TBENCH21_SAO_QWEN35_08B_VALIDATION_20260826.md` 与 yeto 同名文档只差测试计数一段（yeto 版补了"Yeto 65 Python + 83 Rust"），其余一致；文档声明的 Feature branch 即此分支。
- 文件清单（相对 merge-base 7da1079a）：`miles/backends/training_utils/sao.py`（recipe、DIS 数据校验、attention 冻结）、`loss_hub/math_utils.py`（`compute_sao_dis_policy_loss`）、`loss_hub/losses.py`（sao_dis 分支、`_hl_gauss_target_distribution`、`_two_hot_target_distribution`、classification value loss）、`loss_hub/logit_processors.py`（`_value_support`、`predict_values_from_logits`、`apply_temperature`）、`loss_hub/gae_adaptive.py`、`loss_hub/advantages.py`、`megatron_utils/model_provider.py`（`_value_head_output_size`）、`utils/arguments.py`、`megatron_utils/actor.py`。

### 8.2 fork 移植（miles-sao，分支 yeto-sao，基于 ce96fc060，提交 6b5bd88c，未 push）
- 逐字移植：`compute_sao_dis_policy_loss`；`--policy-objective {ppo,sao_dis}`、`--sao-dis-eps-low/high`（sao_dis 时强制 use_rollout_logprobs、与 TIS/OPSM/mismatch 互斥）；`--value-loss-type {mse,classification}`、`--value-num-bins 51`、`--value-reward-range 0 1`、`--value-target-type {hl_gauss,two_hot}`、`--hl-gauss-sigma-ratio 0.75`；critic 头输出维度 = bins；value 预测走 softmax·support 且不乘温度。缺省（ppo + MSE）路径代码未改。
- 未移植：`gae_adaptive.py`（fork 已有 ce96fc06 的 `--gae-variant decoupled` + `--gae-lambd-mode length_adaptive`，二者语义对应，**但未与上游 gae_adaptive 数值对拍**）、`--num-critic-epochs`（fork 无，SAO 需 2）、`critic_freeze_attention`、离线 value pretrain、EV 统计、compaction 校验。
- 测试（miles-next-venv，`tests/fast/backends/training_utils/test_sao_ports.py`，参考实现用 python math，不 import 被测代码）：DIS 两档对拍 + 严格拒绝 + 梯度含 ratio；HL-Gauss 7 个目标值对拍；51-bin CE 损失对拍；分类 value 期望且跳过温度；MSE 缺省路径不变；CLI 缺省不变。13 passed；连同 `test_ppo_gae_variants.py` 35 passed。`tests/fast/backends/training_utils/loss_hub/*` 因 conftest 需 megatron.core 在本环境无法收集（环境原因，改前同样）。

### 8.3 yeto 声明（s13-sao）
- 新模块 `yeto/rl/algos/sao.py`（加入 EXTENSION_MODULES）：loss 扩展字段 `policy_objective`/`sao_dis_eps_low`/`sao_dis_eps_high`（None 时不入规范 JSON）；`sao_algorithm_spec(domain)`（ppo+needs_critic、γ=1、λ_critic=1、length_adaptive α=1.5、decoupled、hl_gauss 51、critic_lr 5e-6、warmup 10 iters、critic_updates_per_step=num_critic_epochs=2、warmup_steps 0、full）；`recipe_settings`；`sao_role_contract`（actor/critic 两 role、分开 layout、每 role 一个 syncer、lockstep 成对 fragment、critic 步数=actor×epochs）；`sao_fork_argv`；拒绝 `sao_dis_fields`、`sao_not_at_pin`（FORK_COMMITS 为空，ports 运行 SAO 仍被拒，旧 streaming 入口照常）。`algorithm_flags._UNMAPPED` 加三个 fork flag（与 loss_variants 同法）。旧入口 `sao_streaming_runtime.py` 未改。
- 测试 `tests/test_rl_sao_spec.py` 8 passed：两档 recipe 与冻结的上游 `apply_sao_online_recipe`（`tests/sao_recipe_reference.py`，逐字拷贝）逐键相等（spec 不持有的 lr/critic_freeze_attention/kl_loss_coef 显式列出）；规格哈希稳定/往返；role 合同的步数被旧入口 `_validate_miles_runtime` 接受、错误步数被拒；fork argv 快照；拒绝；缺省规格不含新字段。
- 回归：`evidence/hash_compare.py` 在 integ-decl(d398d443) 与本分支输出逐行相同，且与 `evidence/hash-critic.txt` 哈希相同；定向集（argv_snapshot、critic_spec、algorithm_flags、sao_streaming_runtime、miles_sao_streaming、tbench21 合同、critic_warmup、critic_ports、miles_config、loss_variants）本分支 7 failed/140 passed，基线失败集合相同（均为 test_rl_miles_sao_streaming 缺 pytest-asyncio）。`tests/test_rl_argv_snapshot.py` 未改且通过。

### 未验证 / 遗留
- fork decoupled+length_adaptive 与上游 `gae_adaptive`（含 terminal reward 落在最后 action token、跨 observation 桥接）未数值对拍；fork 缺 `--num-critic-epochs`、critic attention 冻结；value_reward_range/sigma 为翻译常量，不在 spec 哈希内。
- 第二 syncer（4.2）不在本组；`sao_role_contract` 只是声明，未接线到 launcher。
## S13 9.x CompactionRL（2026-10-07，CPU；GPU 9.4/9.5 未执行）

出处：CompactionRL, arXiv 2607.05378v1（https://arxiv.org/html/2607.05378v1，经 WebFetch 摘取，未逐字核对 PDF）。

| 项 | 论文值 | 出处 | yeto |
|---|---|---|---|
| 触发 | `C−|h_t|<T_comp`，T_comp=10,240 | §4.1 式 (7) | `compaction.should_compact` |
| C | GLM-4.7-Flash 64k、GLM-4.5-Air-SFT 80k | §5.1 | `CompactionConfig.context_budget`（运行配置） |
| 摘要 | 同一策略 `S_t~π(·|h_t⊕q_sum)`；需保留原目标、已完成动作、重要观察、未解决错误、当前状态、下一步 | §4.1 式 (8) | 9 节模板为 **yeto 草稿**（论文未给出节名、`<analysis>` 文本、u_resume 文本） |
| 重建 | `h̄_t=s⊕u_resume(S_t)⊕(z_{t−k+1..t})`，k=2，必要时减小；z=(a,o) 原子 | 式 (6)(9) | `rebuild_context`：k 减到重建后不再触发为止（yeto 对"必要时"的解读） |
| 压缩上限 | 每条 3 次 | §5.1 | `max_compactions=3`；用尽后跑到上下文满即标 truncated（yeto 选择，论文未给出） |
| 回报 | 终局回报给每个可训练段（段尾），无摘要质量奖励 | §4.2 | 每段 sample `reward` 相同 |
| GAE | 段内局部 GAE ×(γλ)^{N_{>s}}，N_{>s}=后续段优化 token 数 | 式 (13)–(15) | `advantage.gae_variant=cross_segment` |
| 归一化 | token 级（全部被优化 token） | 式 (12) | `loss.aggregation=token` |
| α | 1.5（λ=1−1/(αl)） | §5.1 | `advantage.alpha=1.5` |
| critic | lr 3e-6；每批 2 次 value 更新对 1 次策略更新；由策略 ckpt 初始化，50 步预训练 | §5.1 | `critic_lr/critic_updates_per_step/warmup_steps`，init=copy_actor_backbone |
| 运行设置 | actor lr 2e-6（Adam）、global batch 128、group size 1、单次回复上限 10,240 | §5.1 | `PAPER_RUN_SETTINGS`（不入哈希） |

论文未给出：γ（design D8 取 1.0）、KL（design 取 0，即 kl.placement=none）、clip ε 数值、value_clip、length-adaptive 的 l 按段还是按整条、训练时 turn 上限（评测 250）、batch 128 的单位。

### 9.1 复用清单与缺口
- Miles `examples/experimental/terminus-compaction`（fork worktree miles-gae）只有 `run.py`（启动参数）与 README，**不含压缩/摘要代码**：压缩由外部 Harbor `harbor-miles-v0.20.0` 的 Terminus 2（`HARBOR_TERMINUS_2_ENABLE_SUMMARIZE/LINEAR_HISTORY`）完成，Miles session server v2 按轨迹树每叶返回一个 Sample，后处理给兄弟样本相同终局回报与 rollout id、屏蔽共享前缀。
- 可复用：(1) "每段一个 Sample + 共享 rollout id + 共享回报" 的数据形态与 session server v2 后处理思路；(2) README 的核查指标（`rollout/num_training_samples`、episode 级 reward/长度）；(3) `--use-session-server v2`、`agentic_tool_call.generate` 生成入口。
- 不可复用/缺口：Terminus 的摘要提示、触发阈值、k、上限均不是论文配置，且该例是 GRPO（无 critic、无 segment GAE）；Harbor 不在本机。
- yeto 接入点：Terminal-Bench 路径是 `yeto/rl/harness/codex/codex_openenv_generate.generate`（包 Miles `agentic_tool_call`）→ Codex CLI agent；回合循环在 Codex 内部，yeto 无法直接插入压缩。可选接入：(a) 在 session server / TITO 层按 `should_compact` 拦截并注入 q_sum、重建上下文；(b) 换成 yeto 自管回合循环的 agent（`yeto.rl.compaction.run_episode` 的 policy/env 接口）。**均未实现，需定方案**。
- **关键缺口（fork）**：fork ce96fc060 的 cross_segment 把整条 rollout 当一个 sample、按 `metadata.segment_ids` 分段，奖励只在末段段尾。但压缩后各段的条件上下文不同（重建上下文），不能拼成一个 sample 做前向；论文也是每段独立优化、每段段尾都放终局回报。正确形态是每段一个 sample，GAE 用 Miles 原生"样本末 token 放 reward、末尾 bootstrap 0"的局部 GAE，再乘 `(γλ)^{tokens_after}`。数学上与 fork 单 sample 形态在 R 项上不同（fork 前段不含 R）。

### 9.2 rollout（`yeto/rl/compaction.py`）
- `CompactionConfig`（C、T_comp=10,240、k=2、上限 3）、`should_compact`、`SUMMARY_PROMPT`（`<analysis>`/`<summary>` 9 节）、`extract_summary`（丢弃 analysis；无 summary 标签时记 ok=False）、`rebuild_context`、`run_episode(policy, env, prompt, cfg)`。
- 输出：`CompactionEpisode.samples()` 每段一个 sample，metadata `rollout_id/segment_index/num_segments/segment_tokens/tokens_after(N_{>s})/compactions/truncated`，reward 共享；摘要 token 计入所在段的可训练 token；`segment_ids()` 给出 fork 单 sample 形态的逐 token 段号。
- 测试 `tests/test_rl_compaction.py` 10 passed（假模型/假环境：触发边界、无压缩、段编号与 N_{>s}、共享回报、重建=系统提示+u_resume(S)+最近 2 步且不含 analysis、k 自动减小、最多 3 次后 truncated、上限 0、缺 summary 标签、模板 9 节）。

### 9.3 声明与翻译
- `yeto/rl/algos/compactionrl.py::compactionrl_spec()`，sha256 `506b4ba4…c932`。argv：`--calculate-per-token-loss --gamma 1.0 --lambd 1.0 --value-clip 0.2 --critic-lr 3e-06 --critic-updates-per-step 2 --num-critic-only-steps 0 --gae-variant cross_segment --gae-lambd-mode length_adaptive --gae-length-alpha 1.5`；阶段 W `--num-critic-only-steps 50`。
- `algorithm.py` `critic_not_at_pin` 放行 cross_segment 与 critic_updates_per_step≠1（hl_gauss 仍拒）；`critic.py` 加 cross_segment 翻译、`--critic-updates-per-step` 行（≠1 才发出，普通 PPO/VAPO argv 不变）、机制 `features:gae_cross_segment`、`features:critic_multi_update`（未声明，需 `--rl-allow-unverified-mechanism`）；`--critic-updates-per-step` 入 `FORK_FLAGS` 与 `_UNMAPPED`。
- **`--critic-updates-per-step` 不存在于任何 fork 提交**：是向 fork 提的需求（并入 6.2 流程）；在 fork 实现前即使放行未声明机制，Miles 解析也会报错退出。
- `tests/test_rl_vapo.py` 的拒绝用例 cross_segment 改为 hl_gauss（cross_segment 不再被拒）。
- 测试 `tests/test_rl_compactionrl.py` 7 passed（参数、翻译快照、dry-run 拒绝/放行快照与吸收哈希相等、阶段 W、哈希区分、消融臂只少 cross_segment、普通 PPO/VAPO 不变）。定向批次 16 文件 526 passed/6 skipped（含未改动的 `test_rl_argv_snapshot.py`），另 critic_ports/seq_adv_miles 12 passed/1 skipped；log /tmp/s13-compact-pytest.log。未跑会启动 Ray 的测试。
- 哈希不变：`evidence/hash-compactionrl-base.txt`（HEAD 5f940576 git archive）与 `hash-compactionrl.txt` 17 行逐字节相同，且与 `hash-vapo.txt` 相同。

**fork 需求**：(1) `--critic-updates-per-step N`（每批 N 次 critic 更新、1 次 actor 更新）；(2) cross_segment 的每段一 sample 形态：读 `sample.metadata["tokens_after"]`，局部 GAE 后乘 `(γλ)^{tokens_after}`（λ 用该样本 length_adaptive λ 还是整条的，待定）；(3) `train_data_conversion` 透传 `tokens_after`。
**未验证**：真实 tokenizer 下的触发与重建；与 Codex/TB 路径的接入；fork 侧上述需求；GPU 9.4/9.5。
### S13 4.2/4.3（worktree `/home/michael/work/s13-dual`，分支 `s13-dual`，基于 d398d443；仅 CPU）

提交：`f90eab72`（4.2.1）、`b7589faa`（4.2.2 + 4.3 接线）、`b34bbb93`（4.3 测试）、`28ed832a`（4.2.3）。tasks.md 勾选由主 agent 负责。

实现：
- **4.2.1 launcher**（`yeto/launcher.py`）：`CRITIC_SYNCER_PORT=29401`；`rl_needs_critic(args)` 读 `rl_algorithm_spec_json.execution.needs_critic`；`critic_syncer_command` 与 actor RL syncer 参数相同，仅端口/检查点 `~/yeto-output/yeto-critic-state.ckpt`/事件 tape `yeto-critic-tape.jsonl` 不同；`syncer_command` 在 critic 算法下把 critic syncer 放后台、actor syncer 仍为前台进程（task 与 head 子进程共用）；`syncer_ports` 加开 29401；岛命令加 `--critic-syncer $CRITIC_SYNCER_ADDR`，env `CRITIC_SYNCER_ADDR`=actor syncer 主机:29401；`dry_run_plan` 仅 critic 算法时多出 `critic_syncer` 段（port/address/layout=critic_layout_hash/command/checkpoint/tape）。`yeto/rl/learner.py` 新增 `--critic-syncer` → `miles_args.yeto_rl_critic_syncer_addr`。无 critic 时所有字符串/计划不变。
- **4.2.2 插件**（`state_plugin.py`）：`export_critic_tensors`（每 rank 可训练 critic 参数 fp32 CPU 拷贝 + 哈希）、`import_critic_tensors`（按 rank 写回；名称/形状不符写前拒绝；写后按 fp32 重哈希，不等则拒绝——例如 bf16 参数放不下 fp32 平均值）；均包在 `trainer_resident` 内。`critic_state.py` 新增 `critic_fragments`/`assemble_critic_fragments`（按名排序、按字节预算切 fragment，张量不跨 fragment；拼回时校验 layout、fragment 完整性、内容哈希与 specs）。`MilesTrainerGroup.critic_layout/export_critic_state/import_critic_state`（键 `r<rank>:<chunk>:<name>`）；`critic_state_summary` 的 rank 改用同一 `_rank()`（行为不变）。fake 引擎 trainer 加同名方法。
- **4.2.3 原子提交**（`bridges.py`）：`StrictAvgSync.boundary` 拆为 `_submit/_await/_commit`（组合后行为与原实现逐步相同）；新 `DualStrictAvgSync`（`OUTER_SYNC_KIND="strict"`）：actor 通道=原 StrictAvgSync，critic 通道=第二个 StrictAvgSync 经 `_CriticDriverView` 连 critic syncer（张量名前缀 `critic.`；`core.py` 名称白名单加 `^critic\.`；通道 layout_hash=critic fp32 张量的 canonical 哈希，config 哈希位放 `critic_layout_hash`）。每轮先推两通道再等两通道，均得 v+1 才先 critic 后 actor 应用；任一失败抛 `CrossChannelCommitError`、两者都不应用，`keep_committed=True` 时把 actor/critic 恢复为上一提交轮内容，生产默认依赖 round-cut 恢复。critic 优化器状态不随 strict 轮重置（只替换权重）。`entry.build_sync`：有 `yeto_rl_critic_syncer_addr` 用 Dual；`use_critic` 但无 critic syncer 时拒绝。decoupled+critic 拒绝早已在 `critic_run_problems` 中（2.3），未改。
- **4.3 round-cut**（`trainer.py`）：`save_cut` 在 actor 分片后、manifest 提交前经 critic 句柄 `SAVE_CRITIC_CUT` 写 `<cut>/critic/rank-<r>/critic/round-<v>/`（CriticCheckpointStore：权重 + {optimizer, scheduler} state_dict，manifest 最后写），manifest `runtime["critic"]={round, directory, ranks:[rank, weights_sha256, optimizer_sha256]}`（无 critic 时 runtime 不变）；`restore_cut` 在 actor 校验后：pointer 轮次≠actor policy_version 拒绝、有 critic 无 pointer / 无 critic 有 pointer 拒绝，各 rank 经 store 再次校验轮次与 manifest，恢复后权重哈希须等于 pointer。`restore_cut_resharded` 遇 critic 直接拒绝。

验证（CPU，`PYTHONPATH=.:tests /tmp/yeto-venv/bin/python -m pytest`，OMP/OPENBLAS/MKL=1，未启动 Ray）：
- `tests/test_rl_critic_dual_syncer.py` 12 项全过：dry-run 含两个 syncer/独立端口/检查点/tape、critic-free 计划与 syncer 命令不变、岛 env；fake 双 rank 张量经 fragment 往返哈希一致、写回后再导出哈希一致；fragment 篡改/外来 layout/重复/specs 不符拒绝；写回名称/形状/bf16 写后哈希拒绝；critic cut 保存→新 trainer 恢复后权重/优化器/调度器一致，轮次不一致、无 pointer、无 critic、权重篡改均拒绝；fake 两岛 2 轮 actor 与 critic 平均后哈希一致；critic 通道失败时两岛都抛 `CrossChannelCommitError`、actor 的 v+1 未应用、两 role 回到第 0 轮内容；build_sync 选择 Dual / 缺 critic syncer 拒绝。
- 回归定向：driver/grpo_knobs/restart_data_cursor/selection/critic_state/critic_ports/core/ledger_faults/argv_snapshot/algorithm_provenance/trainer_cut 等 335 passed 5 skipped；trainer_cut/reshard/cut_plugin/distopt/transition 127 passed；reconfig_x6/rebuild_e1 34 passed。
- 规格哈希：`evidence/hash_compare.py` 输出 `evidence/hash-s13.txt` 与 `hash-critic.txt` 17 项逐字节相同；`tests/test_rl_argv_snapshot.py` 未改动并通过。

未验证（需 GPU 4.5）：真实 Rust syncer 双进程（同 VM 后台 critic syncer、29401 端口开放、head 模式 SYNCER_PUBLIC_IP 防火墙）；真实 Miles critic 进程内导出/写回（Megatron 参数、TP/PP 分片、offload 唤醒）；Megatron/DistributedOptimizer 的 `optimizer.state_dict()/load_state_dict` 往返（DP>1 分片优化器很可能需按 cut_plugin 方式处理）；全参数 critic 经 StrictRlBridge 的带宽/内存（7B fp32 ≈28 GB/轮）；critic 通道 bf16 参数写回 fp32 平均值必然哈希不等（见下）。

设计问题：
1. bf16 critic 参数无法精确承载 fp32 平均值：现实现写后按 fp32 重哈希会拒绝。需决定：critic 平均结果先量化到参数 dtype 再比较（哈希以参数 dtype 为准），或保持 fp32 主权重（DistOpt main params）写回。GPU 前需定。
2. `check_unverified_allowance` 仍禁止 critic 两岛/外层同步（G3 前门控）；4.5 需要用户批准后放开或用专门 allowance。
3. 跨通道原子性只在 trainer 应用层：actor syncer 已提交 v+1 而 critic 失败时，两个 syncer 的持久检查点可能分叉；恢复以 round-cut（critic 轮=actor 轮）为准，syncer 侧需同轮恢复（与 SAO 模块注释所述相同限制）。
4. critic 优化器跨 strict 轮保留（actor 每轮 reset）；是否也 reset 待定。

## S13 fork 合流（2026-10-07，CPU；未上 GPU、未启动 Ray）

### fork（/home/michael/work/miles-critic，分支 yeto-critic-family，本地提交，未 push）
- `85a571cde` 合入 yeto-vapo `cbf8c4737`（无冲突）；`feb485461` 合入 yeto-sao `6b5bd88c`（`loss_hub/losses.py` 冲突：`_positive_example_lm_loss` 与 HL-Gauss/two-hot/classification value loss 两组新函数，两边全保留；arguments.py 自动合并）。
- `e07e51c07`：按上游 SAO（agentenv/miles feat/sao-tbench21-e2e-validation@16a9bea4，miles-sao 只读查看）语义移植：
  - `--num-critic-epochs N`（别名 `--critic-updates-per-step`，同一 dest `num_critic_epochs`，缺省 1，<1 报错）：critic 每个 rollout 批（=每个 actor 步）在固定的 value 目标上做 N 次优化器更新；第 2 次起重建已消耗的 data iterator（`megatron_utils/actor.py::_train_critic`）；critic 的调度器 `train_iters` 乘 N（`model.py::get_optimizer_param_scheduler(..., role)`）。两拼写语义相同，只保留一份实现；argparse 下后出现者生效，yeto 侧两拼写给不同值时拒绝。
  - `--critic-freeze-attention`（缺省关）：critic 中名字含 attention/attn/self_attention 等组件的模块参数 requires_grad=False（`training_utils/critic_freeze.py`，在三种 model provider 的 critic 分支调用）。
  - 顺带修复：原生 megatron provider 的 critic 头此前固定 output_size=1，改为 `_value_head_output_size(args)`（mse 时仍为 1；classification 时为 bins，SAO 端口此处遗漏）。
  - 未移植（缺口）：上游 sao_dis 在 critic 更新后再前向一次、把新 value 送给 actor（`actor.py` 740 行后）；`full_parameter_state.py` 中按 epochs 计数的 critic 状态；离线 value pretrain。
- 测试（miles-next-venv，OMP/OPENBLAS/MKL=1）：新增 `tests/fast/backends/training_utils/test_critic_updates_per_step.py` 6 passed（缺省 1、无新 dest；两拼写同 dest；冻结只触及 attention；无匹配报错）。定向集（`loss/`、`test_sao_ports.py`、`test_ppo_gae_variants.py`、`test_ppo_gae_masks.py`、`test_ppo_cp_advantages.py`、新测试）：合流后 13 failed/119 passed；基线 ce96fc060（git archive 到 /tmp/critbase，无 sao 测试）13 failed/92 passed，失败用例集合完全相同（cp2 一致性、loss snapshot、logprob reuse，环境原因）。
- **未验证**：actor.py/model.py/model_provider.py 需 megatron，CPU venv 无法 import，仅 py_compile；N 次更新循环、调度器步数、冻结在真实模型上的效果均未运行验证（需 GPU）。

### yeto（/home/michael/work/s13-fork，分支 s13-fork，基于 81645702）
- 新 `yeto/rl/algos/critic_fork.py`：`CRITIC_FORK_PIN = e07e51c07…`、`FORK_COMMITS`（loss_variants 先例）。这是声明层 pin：ports 镜像仍跑 `MILES_NEXT_COMMIT` c35702e（不含这些 flag，且 yeto-critic-family 基于其祖先 039471508，缺 c35702e 的 A27/M3 改动），未加 launch 门控。
- `sao.py`：`FORK_COMMITS` 指向 critic_fork；`sao_not_at_pin` 改查 `CRITIC_FORK_PIN`；新增 flag 行 `--value-loss-type/--value-num-bins/--value-target-type/--hl-gauss-sigma-ratio`（只接受 spec 可表达的值），`--value-reward-range` 入 `_UNMAPPED`（双值、翻译常量）；`--policy-objective sao_dis` 吸收时同时置 `execution.needs_rollout_logprobs`；新未声明机制 `features:sao_dis`、`features:value_hl_gauss`。
- `algorithm.py` `critic_not_at_pin`：hl_gauss 仅在 pin 携带且 `policy_objective=sao_dis` 时放行（非 SAO 的 hl_gauss 无翻译，仍拒）。
- `critic.py`（改动很小）：注释更新；`--num-critic-epochs` 行映射到 `critic.critic_updates_per_step`（critic_argv 仍只发 `--critic-updates-per-step`，≠1 时）；FORK_FLAGS 加 `--num-critic-epochs`；`algorithm_flags._UNMAPPED` 同步。
- 测试：新 `tests/test_rl_critic_fork_pin.py` 14 passed（VAPO/CompactionRL/SAO 两档无 pin 拒绝；SAO dry-run 无放行时因未声明机制拒绝、放行后 accepted 且哈希与声明相同、无剩余 argv；非 fork pin 仍拒；非 SAO hl_gauss 仍拒；别名同哈希、冲突值拒绝；越界 value flag 与裸 `--value-reward-range` 拒绝）；`test_rl_sao_spec.py::test_rejections` 改为期望无拒绝。定向 32 文件（含未改动的 `test_rl_argv_snapshot.py`）796 passed/9 skipped，log /tmp/s13-fork-pytest.log。`hash_compare.py` 输出与 `evidence/hash-critic.txt` 17 行相同。
- 已知遗留：SAO argv 中 GAE flag 由 critic_argv 与 sao_fork_argv 各发一次（值相同，既有行为，未改）；SAO 的 `critic_freeze_attention` 仍不在 spec/argv 中（fork 已支持，待决定是否进 spec）；cross_segment 每段一 sample 未做（等用户决定）。

## S13 cross_segment 每段 sample（2026-10-07，CPU；未上 GPU、未启动 Ray、未 push）

用户决定：CompactionRL 的 cross_segment GAE 按论文"每段单独优化"实现（每段独立 sample，各段段尾放终局回报，局部 GAE ×(γλ)^{N_{>s}}），目的是把奖励传到正确的压缩 action。

### 论文核对（arXiv 2607.05378v1 HTML，WebFetch 摘录，未对 PDF 逐字核对）
- §4.2 式 (13)：`A^loc_{s,i}=∑_{ℓ=0}^{n_s−i}(γλ)^ℓ δ_{s,i+ℓ}`，"For a segment σ_s with n_s optimized tokens"。
- N_{>s}："Let N_{>s}=∑_{j>s} n_j be the number of optimized tokens generated after segment σ_s in the same rollout."（只计被优化 token）
- 式 (14)：`Â_{s,i}=(γλ)^{N_{>s}} A^loc_{s,i}`（"trajectory-position correction"）——确认乘 (γλ)。
- 式 (15)：终局回报在"each independently optimized segment"末 token 时，奖励项折扣为 `(γλ)^{N_{>s}+n_s−i}`，与拼接轨迹中到终局的距离一致；§4.1 "We assign this rollout-level reward to all trainable segments"；"Since each segment is optimized independently"。
- 段尾 bootstrap：论文只在 §3 式 (3) 给出终态 `V(x_{T+1})=0`，式 (13) 的求和止于段尾，**未明说**段边界 V 取 0 还是下一段的值。实现取 0（与"每段独立 sample"一致）——**待确认**。
- λ：§5.1 "λ=1−1/(αl) and α=1.5, where l denotes the response length"——**未说明** l 按段还是整条。
- critic 目标用校正后还是局部优势：论文只说 "standard PPO value regression loss"——**未说明**。γ、per-token r_{s,i} 也未给出。

### fork（/home/michael/work/miles-critic，分支 yeto-critic-family，`ffe769c1e`，基于 e07e51c07，本地未 push）
- 方案：新增 `--gae-variant cross_segment_per_sample`（新取值而非改写 `cross_segment`，旧取值语义逐元素不变、可审计）。每个 sample 普通局部 GAE（终局回报在本 sample 末个可训练 token、其后 bootstrap 0，掩码 token 不是转移），优势 ×`(γλ_i)^{tokens_after_i}`；returns = **局部**优势 + V（未校正）；length_adaptive 时 l = `metadata.gae_length`（若有）否则本 sample 响应长度。缺 tokens_after 或负值报错。
- 改动：math_utils.py（`get_advantages_and_returns_batch` 新 kwargs `tokens_after_list/gae_length_list`）、advantages.py（仅该取值时传）、loss.py（传 `rollout_data["tokens_after"/"gae_length"]`）、train_data_conversion.py（metadata→train data，int64 ndarray ValueSpec，入分片；gae_length 缺省填本 sample response_length）、arguments.py（choices/help，标注旧 cross_segment 已废弃）。
- 测试（miles-next-venv，OMP/OPENBLAS/MKL=1）：`test_ppo_gae_variants.py` 新 10 例（对拍拷贝的独立参考 `gae_cross_segment_per_sample`，掩码/固定与自适应 λ/有无 gae_length；把一条 rollout 切成段 sample 后 V=0 时优势 = R·(γλ)^{T−1−t}（式 15）；tokens_after=0 等于 vanilla；缺失/负值报错；kwargs 只在该取值时转发），`test_segment_ids_conversion.py` 新 2 例。合计 37 passed。定向集（loss/、sao_ports、gae_variants、gae_masks、cp_advantages、critic_updates_per_step）：新 13 failed/130 passed；基线 e07e51c07（git archive /tmp/perseg-base）13 failed/119 passed，失败集合 diff 为空。log /tmp/s13-perseg-fork.log、/tmp/s13-perseg-base.log。

### 旧模式关系
- 旧 `cross_segment`（整条 rollout 一个 sample + segment_ids）：前段不含终局回报，只靠末段的 δ 经 ×(γλ)^{N_{>s}} 不会把 R 传到前段（前段局部 GAE 里没有 R）；与论文式 (15) 不符。fork 保留不动（不影响缺省），**建议废弃**；yeto 吸收 `--gae-variant cross_segment` 时拒绝，`CompactionEpisode.segment_ids()` 降为诊断用途。且压缩后各段条件上下文不同，本也无法拼成一个 sample 前向。

### yeto（/home/michael/work/s13-perseg，分支 s13-perseg，基于 cd760f3d）
- 规格值 `advantage.gae_variant=cross_segment` 不变（哈希 506b4ba4… 不变），`critic.gae_variant_argv` 翻译为 `--gae-variant cross_segment_per_sample`；`--gae-variant` 吸收：`cross_segment_per_sample`→`cross_segment`，`cross_segment` 拒绝（提示改用新值）。
- `compaction.py`：每段 metadata 加 `gae_length = 整条 rollout 被优化 token 数`（同一 rollout 各段共用 λ，保持式 15；yeto 选择，待确认）。
- pin：`critic_fork.py` `CRITIC_FORK_PIN/FORK_COMMITS` → `ffe769c1eb8ad65e42954ebb31285120bc2d9040`。
- 参考与测试：`tests/rl_gae_reference.py` 加 `gae_cross_segment_per_sample`、`split_rollout_into_segment_samples`；`test_rl_gae_reference.py` +3（手算、式 15 距离且旧模式前段为 0、tokens_after=0 等于 vanilla）；`test_rl_compactionrl.py` 快照改新值 + 旧值拒绝用例；`test_rl_compaction.py` 检查 gae_length；`test_rl_critic_fork_pin.py` PIN 更新。
- 结果：定向 26 文件 600 passed/5 skipped/7 failed，7 个失败全在 `test_rl_miles_sao_streaming.py`，在未改动的 /home/michael/work/integ-decl（同 HEAD cd760f3d）同样 7 failed，非新增（注：git archive 到 /tmp 时该文件通过，疑与工作目录/环境相关，未深究）。`test_rl_argv_snapshot.py` 未改且通过。`hash_compare.py` 输出与 `evidence/hash-critic.txt` 17 行相同。log /tmp/s13-perseg-pytest.log。

### 未验证 / 待确认
- 未验证：真实 Miles 训练路径（loss.py 需 megatron，仅 CPU 函数级测试）、Ray 序列化下新键传输（只测了 split_train_data_by_dp_raw）、GPU 9.4/9.5。
- 待用户确认：(1) 段尾 bootstrap 取 0；(2) l 取整条 rollout 被优化 token 数（备选：段长）；(3) critic 目标用局部优势（备选：校正后优势，会使前段目标趋近 V）；(4) 旧 fork `cross_segment` 是否从 fork 删除。

## S13 critic fp32 主权重同步（worktree `/home/michael/work/s13-fp32`，分支 `s13-fp32`，基于 03d0197c；仅 CPU，未启动 Ray）

实现：
- `state_plugin.py`：新 `_critic_masters`（复用 `full_masters`：DistributedOptimizer 分片 main shard 经 DP 组 all-reduce 拼全 / 完整 `main_param` / fp32 参数自身）、`_write_critic_masters`（复用 `write_masters` 写主权重，再把低精度模型参数设为主权重的 cast）。`_export_critic_tensors` 导出 fp32 主权重；`_import_critic_tensors` 写主权重→生成模型参数→按主权重重哈希，并核对每个低精度参数等于其主权重 cast，否则拒绝；`_save_critic_cut` 存主权重；`_restore_critic_cut` 先 load 优化器/调度器、再写主权重并生成模型参数（主权重优先于优化器 load），哈希按主权重。低精度参数无主权重时拒绝（不再有损写 bf16）。注释写明 critic 优化器状态每轮保留为第一版选择。
- 门控：`algorithm.py` 新 `CRITIC_STRICT_AVG_ALLOWANCES`（advantage_estimators:ppo、execution:critic、critic_multi_update、gae_*、positive_example_lm_loss、value_hl_gauss、sao_dis）；`check_unverified_allowance(..., sync_preset=None)`：strict-avg 且全部属该集合并含 execution:critic 时多岛/外层同步放行；仍须显式放行参数。`launcher.py`（传 `rl_sync_preset`）、`learner.py`（传 `sync_preset`）各加一行。decoupled+critic 仍由 `critic_run_problems` 拒绝。
- design.md D4 补充上述三点。

验证（`PYTHONPATH=.:tests /tmp/yeto-venv/bin/python -m pytest`，OMP/OPENBLAS/MKL=1）：
- 新 `tests/test_rl_critic_fp32_masters.py` 7 passed：fake DistOpt（bf16 参数 + fp32 shard、step=shard Adam + main→model cast）两岛平均写回后主权重哈希逐位一致、bf16 参数=主权重 cast、无拒绝；写回后一次 step 等于"主权重即平均值"的参考、不等于旧主权重路径；DP=2 线程模拟 all-reduce 拼全/各 rank 写回本段；无主权重 bf16 拒绝、fp32 自身读写；round-cut 存/恢复哈希=通道哈希、优化器矩与调度器恢复、轮次不符拒绝；门控 strict-avg 放行、decoupled/None/dense-full/非 critic 机制/无 execution:critic/未知名拒绝；launcher 两岛 PPO strict-avg+放行通过、无放行拒绝、decoupled 拒绝。
- `tests/test_rl_critic_dual_syncer.py` 原"bf16 写后哈希拒绝"用例改为"无 fp32 主权重拒绝"，12 passed。
- 回归：critic*/algorithm*/loss_variants_spec/seq_adv/state_plugin*/trainer_cut*/cut*/argv_snapshot/sao_spec/vapo*/compaction* 629 passed 5 skipped（/tmp/s13-fp32-pytest.log）；learner*/launcher*/launch*/driver*/selection* 209 passed 1 skipped（/tmp/s13-fp32-pytest2.log）。`hash_compare.py` 与 `hash-critic.txt` 17 项一致；`test_rl_argv_snapshot.py` 未改并通过。

未验证（需 GPU 4.5）：真实 Megatron DistributedOptimizer 下 critic 参数的 gbuf_ranges/main shard 读写与 DP 组 all-reduce；直接 cast 写 bf16 param buffer 与 Megatron `_copy_main_params_to_model_params` 逐位一致；Megatron `optimizer.load_state_dict` 是否改写主权重（已用"load 后再写主权重"规避，但未实测）；TP/PP 下 critic 参数键与分片；precision-aware optimizer（拒绝）。`critic_state_summary`（4.1/4.4 tape）仍按模型参数哈希，与通道口径不同，未改。
## S13 文档（2026-10-07，分支 s13-docs，基于 integ-decl 03d0197c；仅 CPU，未上 GPU、未启动 Ray、未 push）

### 11.1 docs/MILES_RL.md
- 新小节 "Critic family (`rl-algo-critic-family`)"（位于 Policy-loss variants 之后）：状态声明（可表达未开放、GPU 全未验证）、fork pin（CRITIC_FORK_PIN ffe769c1e 仅声明层，镜像仍 c35702e）、字段/默认值/argv 表、拒绝规则（spec 规则 + `critic_shared_ppo` 启动检查 + 两岛被放行开关拒绝）、warm-up 两阶段、GAE 变体表（vanilla/length_adaptive/decoupled/cross_segment_per_sample/旧 cross_segment）、双 syncer 与 round-cut、四个算法（含用户 10-07 决定与 VAPO 论文差异）、验证状态表。
- Ports boundary 表 critic 行与 "Capabilities and execution" 处加注指向新小节。
- 用户决定中**本分支尚未实现**的项一律标 "in progress (not merged)"：warm-up 启动接线与回报拟合/校准/EV 成功判定；VAPO 显式成功判定与正例 token 数归一（本分支仍为 PG loss 归一 + reward>0.0）；CompactionRL 旧实现作显式对照（本分支吸收时拒绝）与 Codex 桥压缩接入；critic fp32 主权重写回与放开两岛门控（本分支 bf16 写回哈希校验会拒、`check_unverified_allowance` 拒绝有外层同步的 critic 运行）。
- dry-run 核对（/tmp/yeto-venv，OMP/OPENBLAS/MKL=1；脚本 /tmp/s13-docs-dry/doc_cmds.sh 从文档代码块抽取执行，输出 /tmp/s13-docs-dry/doc_cmds.out）：warm-up accepted（stage W `--num-rollout 50 --num-critic-only-steps 50 --critic-load /ckpt/actor --critic-save /ckpt/critic-w --save-interval 50`，主阶段 `--num-critic-only-steps 0 --critic-load <stage-W product>`）；PPO 无放行 rejected、放行 accepted（argv 与文档一致）、`--kl-coef 0.1` rejected `[critic_reward_kl]`；VAPO accepted sha 7ee1dde4…（=vapo_spec()）；CompactionRL accepted sha 506b4ba4…，argv 与文档逐项一致；`--gae-variant cross_segment` rejected（另跑，/tmp/s13-docs-dry/crl_old.json）；`critic.param_mode=lora` 规格文件 rejected `[critic_param_mode]`。全部与文档描述一致。

### 11.2 能力页 / P0 change
- `openspec/changes/rl-algorithm-capabilities/proposal.md`：能力表 critic 行改为 ⚙（rl-algo-critic-family，可表达未开放，仅 CPU）；实施索引新增第 3 行；第 42 行加注。
- `/home/michael/work/infra-drafts/rl-algo-capabilities.html`（不在 git）：矩阵 "PPO（critic）" 行替换为 "critic 家族" 11 行（9 行可表达未开放、2+1 行拒绝：旧 cross_segment、critic LoRA、两岛）；拒绝规则表 critic 规则 5 行；新增卡片 3（tasks 29/43）；`node --check` 通过。备份 /tmp/s13-docs-dry/rl-algo-capabilities.html.bak。
- 单测：`tests/test_rl_algorithm_capabilities.py`（含 `test_miles_and_fake_declarations`、`test_critic_rejected_with_the_real_reason`）、critic_spec、vapo、compactionrl、critic_fork_pin、critic_warmup、sao_spec、argv_snapshot、critic_ports：136 passed（均不启动 Ray）。

### 合入其它 S13 分支后需刷新
docs/MILES_RL.md Critic family 中所有 "in progress (not merged)" 处（Warm-up 小节末段、GAE 表旧 cross_segment 行、Outer sync 的 fp32 条、VAPO 用户决定段与 dry-run 输出中的 `--positive-example-reward-threshold`、CompactionRL 段、验证状态表）；能力页对应矩阵行与卡片 3 的 dec/open；若 VAPO/CompactionRL 规格哈希或 argv 变化，需重跑文档 dry-run 并更新 sha 与 argv 注释。
## S13 VAPO/CompactionRL 用户决定落实（2026-10-07，CPU；未上 GPU、未启动 Ray、未 push）

### fork（/home/michael/work/miles-critic，分支 yeto-critic-family，`70e3d7761`，基于 ffe769c1e，本地）
- C 正例判定：`--positive-example-source {success,reward}`，缺省 success = 奖励函数给的布尔 `sample.metadata["success"]`/`["is_correct"]`（`train_data_conversion.sample_success_flag`：非布尔/0-1 值或两键矛盾报错；缺失记 -1，键 `positive_example_success` 入分片）；启用 LM loss 时任一样本缺字段即报错，不回退 reward。`--positive-example-reward-threshold` 缺省改为 None，仅 source=reward 时允许（`arguments.validate_positive_example_args`）。
- D 归一化（式 9）：`loss.positive_example_lm_weights` 按 optimizer step 分组（与 `get_data_iterator` 同切分；有 `micro_batch_indices` 时按 `num_microbatches` 分组）统计本地 (正例 token 数 P, Miles step 归一量 D)，`_dp_all_reduce_sum` 在 effective-DP 组求和，样本权重 w_i=flag_i·D/P；`losses._positive_example_lm_loss` 用 w_i 做加权 token 求和（D 由 Miles 原有的 per-token 全局 token 数 / per-sample rollout 数归一抵消）。batch 键 `positive_example_flags` → `positive_example_lm_weights`（megatron/fsdp get_batch 键表同步）。coef=0 路径不变。
- F 旧模式：更正后的决定为改名保留——`cross_segment` → `cross_segment_whole_rollout`（显式对照，help 注明与式 15 不同），`cross_segment` 被 argparse 与 math_utils 拒绝（提示两个明确取值）；`cross_segment_per_sample` 不变。
- 测试（miles-next-venv，`-p no:cacheprovider`，单进程，OMP/OPENBLAS/MKL=1，均不用 ray_local_mode）：`test_positive_example_lm_loss.py` 重写（式 9 独立参考对拍 per-token/per-sample；2 rank×2 micro-batch 模拟：先收集各 rank 本地统计再求和作为 all_reduce，所有 micro-batch 项之和/D = 全 8 样本式 9；按 step 分组权重手算；无正例/coef=0 不变；success 缺省与缺失报错、reward 源阈值、参数校验、success 字段解析）、`test_ppo_gae_variants.py`（改名 + 含糊取值拒绝）、`test_segment_ids_conversion.py`（+2 success 字段透传）。定向集（loss/、sao_ports、gae_variants、gae_masks、cp_advantages、critic_updates_per_step、segment_ids_conversion）：新 13 failed/143 passed；基线 ffe769c1e（git archive /tmp/vapo2-base）13 failed/134 passed，失败集合 diff 为空。log /tmp/s13-vapo2-fork.log、/tmp/s13-vapo2-base.log。

### yeto（/home/michael/work/s13-vapo2，分支 s13-vapo2，基于 03d0197c）
- A：`vapo_spec` α=0.05 不变。
- B：`critic_warmup.value_quality`（mse、relative_error、explained_variance、10 箱等量分箱校准误差/最大偏差/箱表）、`quality_gate_problems`；`WarmupProduct.value_quality` 写入 manifest；spec 扩展字段 `critic.warmup_max_value_mse/warmup_max_value_rel_error/warmup_max_calibration_error/warmup_min_explained_variance`（`algos/critic.py` 注册，None 不入 JSON；只在 critic+warmup_steps>0 时允许）；缺省只记录，设阈值则 `finish_warmup` 不达标报错且不写 manifest、`load_product` 复用重查。**接线点未实现**：阶段 W 的 value/returns 导出（fork 侧）与 `run_stage` 返回值，需 GPU 路径。
- C：spec 新字段 `loss.positive_lm_source ∈ {success, reward}`；coef 必须配 source；reward 必须配阈值；success 不得带阈值；argv `--positive-example-lm-loss-coef 0.1 --positive-example-source success`（reward 时再加阈值）；`--positive-example-source` 入 FORK_FLAGS/`_UNMAPPED`。VAPO 哈希 7ee1dde4…386f → `5e9b38edb3eec5e160d221c7f62275e6ebd4a9e163d743ddca94a9b406922113`。
- E：`vapo.py::DEVIATIONS`（项/值/来源/与论文关系，`NOT_IN_PAPER` 由其派生），design D7 同表。
- F：spec `GAE_VARIANTS=(vanilla, decoupled, cross_segment_per_sample, cross_segment_whole_rollout)`，`cross_segment` 在字段解析与 `--gae-variant` 吸收时报错；翻译直译同名 fork 取值；机制 `features:gae_cross_segment`（per_sample）、新 `features:gae_cross_segment_whole_rollout`（均未声明）；`compactionrl_whole_rollout_control_spec()` 为 9.5 对照臂。CompactionRL 哈希 506b4ba4…c932 → `16fb68d50b365b0bf182ee1989b650da6601efe38ac0ab5c3ec7ba4526bf3c4f`。design D8 写入已确认项。
- pin：`critic_fork.py` → `70e3d77618841235330fee2f4634b1ddd6b924da`。
- 测试：新 `tests/test_rl_critic_warmup_quality.py` 8 passed（手算指标、偏置 critic 的校准、退化输入、哈希/JSON 不含阈值字段、记录不拦截、门控拦截不写 manifest、有阈值无指标拒绝）；`test_rl_vapo.py`（快照/哈希/拒绝矩阵更新 + reward 源声明 + DEVIATIONS 覆盖）、`test_rl_compactionrl.py`（新哈希、含糊值拒绝、对照臂快照与 dry-run）、`test_rl_critic_fork_pin.py`（PIN）。定向 23 文件（critic*/vapo/compaction*/sao_spec/algorithm*/gae_reference/argv_snapshot/loss_variants_spec/seq_adv*/miles_adapter_config）600 passed/7 skipped，log /tmp/s13-vapo2-yeto.log；未改动的 `test_rl_argv_snapshot.py` 通过；`hash_compare.py` 输出与 `evidence/hash-critic.txt` 17 行相同。

### 未验证 / 待确认
- 未验证（需 GPU）：fork 中 `_dp_all_reduce_sum` 的真实分布式行为（只在 CPU 上以注入函数模拟）、Megatron 是否确实按 Σ max(mask,1) 做 per-token 全局归一（据 `loss_function` 返回的 num_tokens 推断，未在 Megatron 内核对）、CP>1、动态 batch 下 step 分组与 `get_data_iterator` 一致性（只按代码阅读对齐）、FSDP 后端的归一语义；`positive_example_success` 经 Ray 序列化的传输（只测 `split_train_data_by_dp_raw`）；阶段 W 价值质量的真实数据导出。
- 新待确认：(1) 定向 grep 未找到 yeto 奖励实现向 `sample.metadata` 写 `success`/`is_correct`——VAPO 上卡前需由具体任务的奖励函数提供，否则 fork 报错；(2) 式 9 的"批"取一个 optimizer step（若一个 rollout 分多 step，各 step 单独归一）；(3) 价值质量阈值的具体数值未定（缺省只记录）；(4) 对照臂 `cross_segment_whole_rollout` 的 rollout 侧拼单 sample 未接线。
## S13 Codex 桥压缩拦截（2026-10-07，CPU；未上 GPU、未启动 Ray、未 push）

worktree `/home/michael/work/s13-tbcompact`，分支 `s13-tbcompact`（基于 integ-decl 03d0197c）。用户决定"先拦截"：在 Codex 本地 Responses 桥层实现 D8 压缩。

### 1. Codex 0.145.0 自带压缩（结论）
- **有**。证据（pinned 二进制 `/home/michael/work/codex-bundle/codex/codex-x86_64-unknown-linux-musl`，`--version`=codex-cli 0.145.0，sha256 a2a05daf…）：`strings` 含 `model_auto_compact_token_limit`、`model_auto_compact_token_limit_scope`、`model_context_window`、`compact_prompt`、`experimental_compact_prompt_file`、`responses/compact`（远端压缩）、`remote_compaction`、`context_compaction`/`compaction_trigger` 历史项、`run_auto_compact`；app-server schema（`codex_app_server_protocol.v2.schemas.json`，sha a88d865c）含 `thread/compact/start`、`thread/compacted`、`ContextCompaction` 项、config 字段 `model_auto_compact_token_limit`/`model_context_window`/`compact_prompt`。
- yeto 当前给 Codex 的配置（`codex_harness_agent._codex_argv`）**未设置**上述任何键；用同类配置（model=deepseek-v4、自定义 provider、`--strict-config`）在本机起 app-server 调 `config/read`：`model_auto_compact_token_limit=None`、`model_context_window=None`、`compact_prompt=None`。**运行时是否会按 models-manager 的内置 model_info 推出默认阈值：未确认**（未逆向）。Codex 看到的 token 用量就是桥回给它的 session server usage（每段 ≤C）。
- 现有桥对 Codex 侧压缩一律 fail closed：历史项 `compaction`/`compaction_trigger`/`context_compaction` 拒绝（`_canonical_history_item`）、非 `/v1/responses` 路由（含 `responses/compact`）拒绝、`thread/compacted` 事件视为违规、历史被改写即 "truncated or mutated" 拒绝。
- **选择：关闭 Codex 自带压缩、由桥触发**。理由：Codex 本地压缩的摘要提示是 Codex 自己的 `compact_prompt`（非 D8 9 节模板），触发基于 Codex 自己的计数，压缩后改写的是 Codex 的历史（需要桥接受被改写历史并重新映射，且远端压缩走另一个端点）；桥触发时摘要由同一策略经同一 Miles session 采样（logprob 由 session server 记录、mask=1），可直接成为可训练段，Codex 历史保持 append-only。CompactionRL 模式下 Codex argv 追加 `-c model_auto_compact_token_limit=9223372036854775807`（已对 pinned 二进制验证 `--strict-config` 接受此键与 i64 值，字符串值报 "expected i64"、拼错键报 unknown field）。

### 2. 可行性与边界（桥改写上下文）
- **可行**：桥从不转发 Codex 的 Responses `input`。它只校验 `input` 恰为预期的 append-only 历史（`_expected_input`），自行维护 Chat 消息表 `_messages` 发给 Miles；Codex 请求被强制 `store:false`、无 `previous_response_id`、`truncation` 禁用（`_validate_codex_request`）。所以 Codex 自身仍持有完整历史，桥只把重建后的 `_messages` 发给 session server；Codex 不感知压缩。
- **不可行的部分与绕法**：fork pin 的 v1 session server 是 append-only 线性轨迹，重写历史会被 `_rollback_to_checkpoint(-1)` 静默丢弃前面的 record（R-D5a），旧 legacy 压缩依赖的 `X-Miles-Compaction-*` 头 pin 不认识。故**每段一个 session**：摘要在当前 session 里采样（append `user` 摘要指令，模板允许 `user` 追加），重建上下文从下一个预建 session 开始。
- 边界：(a) Codex 侧历史无上限增长，桥 HTTP body 上限 2 MiB（stock 常量，未改）——4 段×C token 级别通常在内，超过则请求被拒、轨迹作废；(b) 触发用"上次 usage.total_tokens（session server 精确计数）+ 未观测工具输出字节上界(+256)"，偏保守（早触发）；k 的选择用字节上界估计重建后长度；(c) 摘要请求沿用 stock `_sample_miles(summary=True)`：`tools=[]`、`tool_choice="none"`，与执行回合的 tools 不同，TITO 前缀复用是否受影响**未验证**；(d) 剩余空间不足以放摘要指令时直接按满截断（CodexSequenceLimit→status max_seq_len）；(e) 预建 session 的 `evaluation` 固定 False（agent function 拿不到 eval 标志）。

### 3. 实现（commit 见下）
- 新 `yeto/rl/harness/codex/compaction_bridge.py`：`CompactionRLBridge(_ResponsesBridge)` 覆写 `_should_compact`（`C−|h|<T_comp`、需已有完整原子步、≤3 次）、`_compact_context`（`compaction.SUMMARY_PROMPT` 同策略摘要→`extract_summary`→切到下一个 session→`RESUME_TEMPLATE` 重建 `system ⊕ u_resume(S) ⊕ 最近 k 步`，k 从 2 减到重建后不再触发）、`_rebuild_compacted_messages`（413 回退沿用 stock 逻辑减 k）、`_compaction_headers`→{}；记录 `metrics.codex_compaction`（compactions、每段 trigger tokens、summary_ok、finish_reason、kept_steps）。摘要 max_tokens = 与执行回合相同的单次回复上限；`finish_reason=length` 的摘要保留并记 ok=False。`drive_codex_compactionrl` + `_CompactionAppServerDriver`（仅 argv 加关闭自带压缩的键）。
- **`codex_harness_agent.py` 未改**（它被 `CODEX_HARNESS_AGENT_SHA256` pin 且签进 run bundle）。`compaction.py` 未改。
- `codex_openenv_agent_function.py`：`drive_untrusted` 仅在 `YETO_CODEX_COMPACTIONRL` 开启时走新驱动（缺预建 session 则拒绝；关闭却带 session 也拒绝），否则调用与原来完全相同的 `harness._drive_codex`；`_metrics_dict` 只在桥运行过时多出 `codex_compaction`；新 `prepare_segment_sessions/create_segment_sessions`（**可信层**预建 3 个 session：`POST {router}/sessions`，body 为 CreateSessionRequest 的 evaluation/temperature/top_p/top_k；不可信 worker 只能用这些 session），in-process `run` 同步接入。
- `codex_openenv_subprocess_agent_function.py`：lease 后、起 worker 前预建 session，元数据 `codex_compaction_sessions` 随所有返回（成功/基础设施失败）带回。
- `codex_openenv_generate.py`：若 sample 0 带 `codex_compaction_sessions`，逐个按上游 tracer 方式 collect（collect 即删除，失败也继续），校验"前 compactions 个 session 各 1 个 sample、其余为空"，否则整条 ABORTED；每段 sample 复制 sample 0 的元数据（签名 outcome→共享回报），写 `segment_index/num_segments/segment_tokens/tokens_after(N_{>s})/gae_length(整条被优化 token)/compactions/truncated/chain_index/chains_total/chain_break_reason=compaction_window`；`group_index`/`rollout_id` 由既有 `apply_trajectory_bookkeeping` 统一。>1 段时用 `dataclasses.replace(GenerateFnOutput, samples=[...])` 返回列表。
- 开关：`YETO_CODEX_COMPACTIONRL=1`（显式参数），`YETO_CODEX_COMPACTIONRL_T_COMP`（缺省 10240）；k=2、上限 3 取自 `compaction.py`。C = `--max-seq-len`。默认关闭。**未做**：由 compactionrl 规格自动设置该 env（属 launcher/learner，另一 worktree 的文件）。

### 4. 验证（CPU，`PYTHONPATH=.:tests /tmp/yeto-venv/bin/python -m pytest`，OMP/OPENBLAS/MKL=1，单进程，未启动 Ray）
- 新 `tests/test_codex_bridge_compaction.py` 18 passed（假 Codex 客户端直接按 app-server 方式 POST 完整历史 + aiohttp 假 session server 按 session 路由）：触发与边界（`C−|h|==T_comp` 不触发、+1 触发、上限后不触发）；摘要请求在旧 session、末条为 SUMMARY_PROMPT、tool_choice none、无 `X-Miles-Compaction-*` 头；新 session 首请求 = system + u_resume(去掉 analysis 的摘要) + 最近 2 步，Codex 侧历史仍完整；k 自动减小；3 次压缩后跑到满 → CodexSequenceLimit、`max_seq_len_hit=1`，S0..S3 各 2 个请求；无 summary 标签/length 截断记 ok=False；空间不足直接截断；argv 只多关闭键；**关闭时**：stock 桥与"开启但永不触发"的新桥对同一脚本的 Miles 请求体、回给 Codex 的 SSE 字节、metrics（除墙钟 total_generation_time）逐字节相同；`drive_untrusted` 关闭时调用参数与原来相同、job 不被改、metrics 键集合不变；可信层预建 session 的 URL/body；wrapper 分段 metadata（tokens_after=[25,5,0]、gae_length=55）、共享 outcome/rollout_id/group_index、无压缩时单 sample、四类不一致→ABORTED 且三个 session 都被 collect、基础设施失败只排空、无压缩元数据时与原行为一致。
- 回归：codex/harness/compaction 相关 12 文件 128 passed/7 skipped（log `/tmp/s13-tbcompact-pytest.log`）；所有引用 codex_openenv/tbench_reward/harness.codex 的 13 个测试文件 223 passed/8 skipped（`/tmp/s13-tbcompact-pytest2.log`）。`tests/test_rl_argv_snapshot.py` 未改并通过。`hash_compare.py` 输出与 `evidence/hash-critic.txt` 17 行相同。

### 5. 未验证 / 待确认
- 未验证：真实 Miles v1 session server 上的摘要回合（user 追加、`tools=[]` 对 TITO 的影响）、`POST /sessions` 预建与 `collect_samples` 对空 session 的返回（按 pin 源码 `OpenAIEndpointTracer` 编写，未实跑）；v1 路径下 custom generate 返回多 sample 列表能否被 rollout/`train_data_conversion` 正确接收（R-D5a 设计上允许兄弟段，未在真实 Miles 跑）；真实 tokenizer 下的触发时机与 k；Codex 自带压缩在运行时的有效默认阈值；GPU。
- 已知遗留：预建 session 在 worker 崩溃且 agent function 未返回元数据、或预建中途失败、或 `finish_trusted` 抛异常时会泄漏（无人 collect）；`CODEX_OPENENV_AGENT_MODULES`（`yeto/rl/__init__.py`）未列入新模块 `compaction_bridge.py`；env 透传到 rollout worker/容器未接线；R-D5a 的 GPU 硬判据 `chains_total==1` 在开启压缩时不再成立，需改判据；preflight/launcher 仍只拒 legacy `YETO_CODEX_COMPACTION_ENABLED`，对新开关无门控。
- 待用户确认：(1) 每段一 session 方案（替代 v2 session server 树形）；(2) 摘要采用 `tools=[]`（stock 行为）还是保留同一工具表；(3) 触发计数用"精确 usage + 工具输出字节上界"是否可接受；(4) 是否需要把新开关接入 compactionrl 规格/launcher 与 preflight 门控；(5) 预建 3 个 session/rollout 的开销与泄漏处理是否可接受。

## S13 Codex 压缩接线（2026-10-07，CPU；未上 GPU、未启动 Ray、未 push）

worktree `/home/michael/work/s13-tbwire`，分支 `s13-tbwire`（基于 integ-decl e401d5fe）。用户已确认：每段一个独立 session；触发用"精确 token 用量 + 工具输出字节上界"（可偏早）；现在接开关。

### 改动
- **规格→开关**（`yeto/rl/algos/compactionrl.py` 新增 `check_rollout_compaction`/`compaction_switch`/常量）：`advantage.gae_variant=cross_segment_per_sample` ⇔ 必须开 `YETO_CODEX_COMPACTIONRL`，且 harness 必须是 Codex OpenEnv agent（唯一能压缩的）；其它 gae_variant 开开关拒绝；`YETO_CODEX_COMPACTIONRL_T_COMP` 无开关时拒绝、非正整数拒绝。**对照臂 `cross_segment_whole_rollout` 在 Codex harness 上一律拒绝**（不论开关）：桥的每段是独立 session、以重建上下文为条件，无法拼成该模式要求的单个 sample；不开压缩则没有段可做对照。非 Codex harness 上该对照臂行为不变（未接任何压缩 rollout）。
- **launcher**（`yeto/launcher.py`：`rl_gae_variant`、`compactionrl_launch_env`，在 `codex_harness_launch` 开头调用）：CompactionRL 规格时容器 env 加 `YETO_CODEX_COMPACTIONRL=1`（启动环境若有 T_COMP 一并透传）；岛 preflight `worker_runtime_env` 已按前缀 `YETO_CODEX_` 转发给 Ray rollout worker（未改）。启动环境显式 `=0` 与 CompactionRL 规格矛盾时拒绝；CompactionRL 规格但 agent 不是 `CODEX_OPENENV_AGENT`（含 legacy 签名 Codex agent 与非 Codex 运行）拒绝。非 CompactionRL 规格且未设开关时返回 {}，env 不变。
- **岛 preflight**（`harness/codex/preflight.py::assert_compactionrl_consistent`，`harness_preflight` 中调用）：按 Miles `--gae-variant`（`miles_args.gae_variant`）核对开关，同上规则，失败 `PreflightError`。
- **模块表**：`yeto/rl/__init__.py::CODEX_OPENENV_AGENT_MODULES` 与 `preflight.required_pin_updates()` 同步加入 `compaction_bridge.py`。该表只有文件名，无逐模块哈希 pin（`CODEX_HARNESS_AGENT_SHA256` 只 pin `codex_harness_agent.py`，未改动），故无需更新哈希。
- **预建 session 泄漏**：`create_segment_sessions` 中途失败时删除已建的；新 `delete_segment_sessions`（尽力 `DELETE {router}/sessions/{id}`，pin `miles/rollout/session/sessions.py` 返回 204，404 视为已删；不抛异常以免掩盖原错误）、`release_unreturned_segments`。subprocess 与 in-process `run`：元数据未随返回值交给 `codex_openenv_generate`（worker 抛非 harness 异常、取消、`finish_trusted` 抛异常）时在 finally 删除；`CodexHarnessError` 路径与成功路径照旧把 session 列表交给 wrapper 收集。关闭压缩时 `segments={}`，路径不变。
- **R-D5a 判据**：仓库内没有现成的 `chains_total==1` 判据脚本（只写在 `openspec/changes/rl-codex-harness-rollout/design.md` R-D5a 与 9.2 文字中）。新增 `yeto/rl/harness/codex/chain_judge.py::judge_chain_counts(metas, compaction_enabled)`：关闭时每条轨迹恰 1 个 sample 且 `chains_total==1`（不变）；开启时每条轨迹 `chains_total==num_segments==compactions+1==sample 数`、`chain_index/segment_index`=0..n−1、非首段 `chain_break_reason=compaction_window`，中止轨迹（单 sample 无段键）按 `chains_total==1`。design 文字同步更新。
- **摘要工具表**：行为不变（`tools=[]`、`tool_choice="none"`），做成显式常量 `compaction_bridge.SUMMARY_REQUEST_TOOLS=()`/`SUMMARY_REQUEST_TOOL_CHOICE="none"`，测试核对线上请求体与常量一致；design D8 注明理由与"待 GPU 核实 TITO 前缀复用"风险。

### 验证（`PYTHONPATH=.:tests /tmp/yeto-venv/bin/python -m pytest -p no:cacheprovider`，OMP/OPENBLAS/MKL=1，单进程，未启动 Ray）
- 新 `tests/test_codex_compaction_wiring.py` 25 passed（launcher 注入/T_COMP/矛盾/默认不变/对照臂/非压缩 harness；preflight 10 组合；半途预建失败删除；best-effort 删除；in-process 与 subprocess 的 worker 崩溃、finish_trusted 异常删除、成功与 CodexHarnessError 交接；judge 开/关；常量与模块表）。`test_codex_bridge_compaction.py` 加工具表常量断言。
- 定向 14 文件（含未改动的 `test_rl_argv_snapshot.py`、launcher/codex bundle/codex openenv/compactionrl/vapo/gateway 等）298 passed/2 skipped，log `/tmp/s13-tbwire-pytest.log`。
- 更大批次（引用 harness.codex/codex_openenv/launcher 的其余测试文件，排除 5 个可能起 Ray 的文件）1054 passed/23 skipped/38 failed；38 个失败全在 `test_diffusion.py`/`test_island_backend.py`/`test_wandb_plumbing.py`，在 e401d5fe 的 git archive（/tmp/s13-tbwire-base）上失败集合相同，非新增。log `/tmp/s13-tbwire-pytest2.log`。
- `hash_compare.py` 输出与 `evidence/hash-critic.txt` 无差异（17 个既有规格哈希不变）。

### 未验证 / 待确认
- 未验证：真实 Miles session server 上 `DELETE /sessions/{id}`（按 pin 源码编写，CPU 用 aiohttp 假服务器）；Ray rollout worker 实际收到开关（只验证了 `worker_runtime_env` 转发函数）；Miles 实际解析出的 `miles_args.gae_variant` 属性名（按 `--gae-variant` 推断）；摘要 `tools=[]` 对 TITO 前缀复用的影响；`chain_judge` 尚未接入任何 GPU 采集脚本（GPU 9.2/9.4 时需对采到的 sample metadata 调用）；GPU 全部。
- 仍可能泄漏：agent function 已返回元数据但 Miles 上游在调用 wrapper 收集前失败（不在 yeto 控制内）；进程被 SIGKILL 时 finally 不执行。
- 待确认：(1) 对照臂在 Codex 上一律拒绝（含不开压缩）是否符合预期；(2) 启动环境显式 `YETO_CODEX_COMPACTIONRL=1` 而规格为 CompactionRL 时允许（冗余但一致）。

## S13 success 字段与 SAO 缺口（2026-10-07，CPU；未上 GPU、未启动 Ray、未 push）

### A. 奖励函数写 `sample.metadata["success"]`（yeto worktree `/home/michael/work/s13-succ`，分支 `s13-succ`，基于 d26111d4）
- 成功定义（完整成功，分数不变，只加字段）：
  - `yeto.rl.math_reward:reward_func`（`examples/math_reward.py` 再导出同一函数）：答案经 mathd/sympy 判定正确 ⇔ success=True（新 helper `set_success`，metadata 为 None 时建 dict）。
  - gsm8k：`gsm8k_reward:score` 不在 yeto 包内——GPU 运行从工作目录加载仓外文件（`/home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py`，evidence 下为冻结副本，均未改）。新增包内 `yeto.rl.gsm8k_reward:score`：数值逻辑逐字相同（测试与 evidence 副本对拍），success = 抽出数字与 `####` 后金标相等。**旧名 `gsm8k_reward:score` 仍不写 success，VAPO 需改用新名**。
  - `yeto.rl.algos.gdpo_reward`（`reward_func`/`correctness_reward`）：success = correctness 分量；format 分不算成功。
  - Terminal-Bench（`harness/codex/tbench_reward.reward_func`）：success = HMAC 签名 outcome 中的 `passed`（= reward==1.0，全部测试通过）；不可信的预置 success 被覆盖；基础设施失败样本（ABORTED）删除 success 键。
  - 未加：`length_reward`（无正确性概念）、`harness/codex/reward.py`（secrlenv，reward∈[0,1] 有部分分，"完整成功"定义待确认）。
- 传递路径：Miles `rm_hub.async_rm` 直接以同一 `Sample` 调自定义奖励函数，原地写 metadata；fork `train_data_conversion.py:133` `sample_success_flag(s.metadata)` 读 `success`/`is_correct`（布尔/0-1，矛盾报错）→ `positive_example_success`。无需 rollout 侧转换。
- 测试：新 `tests/test_rl_reward_success.py` 14 passed；引用这些模块的测试 + `test_rl_argv_snapshot.py`（未改）共 159 passed/1 skipped（`/tmp/s13-succ-pytest.log`）；`hash_compare.py` 与 `hash-critic.txt` 无差异。

### B. SAO 缺口（fork `/home/michael/work/miles-critic`，分支 yeto-critic-family，`5182e37f05b06dddd01a5454cbb37f6ab8fe47f2`，基于 70e3d7761，本地）
- (1) 已移植：`policy_objective=sao_dis` 且 `rollout_id >= num_critic_only_steps` 时，`_train_critic` 在 N 次 critic 更新后重建 iterator 再 `forward_only(get_values)`，新 value 交给 actor；value 目标/returns 仍用更新前的预测（与上游一致）。判定在新文件 `miles/backends/training_utils/sao_critic.py`，`actor.py` 只加一个 if 块和 import。ppo 缺省不变。上游另一条件 `offline_value_pretraining` 在 fork 不存在，略去。
- (2) 不适用：上游按 epoch 计数的是 `megatron_utils/full_parameter_state.py`（full-parameter 外部状态追踪，scheduler 增量 ×num_critic_epochs），fork 没有这个文件/功能；fork 中对应语义（critic 调度器 train_iters ×N）已在 e07e51c07 的 `model.py` 中。
- 测试（miles-next-venv）：新 `test_sao_value_refresh.py` 3 passed（独立参考对拍判定网格；AST 检查顺序 train 循环→refresh→发送 values）；`tests/fast/backends/training_utils` + `tests/fast/utils/test_arguments.py` 改前/改后均 13 failed/724 passed/1 error，失败集合相同。

### 未验证 / 待确认
- 未验证：actor.py 需 megatron，仅 py_compile + AST；真实运行中再前向的开销与正确性（GPU）；奖励字段在真实 Miles rollout 中经 Ray 传到 `train_data_conversion`。
- 待确认：(1) VAPO gsm8k 运行改用 `yeto.rl.gsm8k_reward:score`；(2) secrlenv `reward.py` 的完整成功定义；(3) TB timeout 样本记 success=False（有签名 verdict，非基础设施失败）。
