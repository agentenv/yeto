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
