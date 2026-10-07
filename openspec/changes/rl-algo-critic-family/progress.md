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

## S13 7.1 VAPO 参数（2026-10-07，待用户确认）

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
