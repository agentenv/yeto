约定：
- 规划文档以 `rl-infra-spec` 分支上的副本为准；实现 worktree、分支与可修改路径按 `../rl-infra-spec/alignment.md` 的工作包派发（原 `/home/michael/work/rl-algos` 分支 `rl-algorithms` 不再作为规划来源）。测试命令为 `/tmp/yeto-venv/bin/python -m pytest -q`；全量测试以"改动前后失败集合相同"为准。
- 依赖 Miles 源码的 CPU 测试与 upstream 参数解析在 `/home/michael/work/miles-next-venv` 中运行，miles 不可用时 skip；任务说明中记录命令与结果。
- 不修改 legacy `build_miles_argv`，不修改 Miles/SGLang fork，默认 GRPO 的 argv 与哈希不变；不向 radixark/miles 或 sgl-project/sglang 提 PR。
- 状态严格区分"已实现""CPU 测试通过""GPU 验证通过"；mock/fake 测试不能替代 GPU 验收。能力声明只在 G1 通过后加入（第 8 组）。
- commit 和 push 需要用户确认。

## 1. 基线与前置接口

- [x] 1.1 改动前运行 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败与错误集合保存到 `openspec/changes/rl-algo-grpo-knobs/baseline-failures.txt`。验证：文件存在，条目数与 pytest 汇总一致。
- [x] 1.2 确认 P0 接口可用：`LossSpec`/`AdvantageSpec`/`KlSpec`/`SamplingSpec`、`algorithm_flags.py` 中本 change 涉及的映射行、`EngineCapabilities` 机制维度、`expects_gradient()`、`PluginRef`、`to_legacy_runtime_attrs`。验证：在 `progress.md` 记录各接口所在文件；缺失项列出并暂停，先向用户报告。

## 2. 原生机制的约束与翻译核对（design D1）

- [x] 2.1 在 spec 校验中补充约束：dual-clip 系数须大于 1；clip 上界为有限正数；entropy 系数有限；超采样须配动态过滤且不小于 rollout 批大小。验证：参数化单测逐项报出字段名与允许范围，均在构建 GPU 进程前失败。
- [x] 2.2 为 clip-higher、dual-clip、token 级聚合、去 std、entropy、超采样逐项写翻译单测（非默认值 → 精确 argv 片段），并确认默认 GRPO argv 不变。验证：`tests/test_rl_argv_snapshot.py` 与 `tests/test_rl_miles_adapter_config.py` 不改即通过；新增用例全部通过。
- [x] 2.3 在 miles-next-venv 中用 upstream `parse_args` 解析上述每项生成的 argv，并核对 namespace 上的字段值（如 `eps_clip_high`、`eps_clip_c`、`calculate_per_token_loss`、`grpo_std_normalization=False`、`entropy_coef`、`over_sampling_batch_size`）。验证：测试在 miles-next-venv 实际运行通过，命令与输出写入 `progress.md`。
- [x] 2.4 CPU 数值核对：在 miles-next-venv 中直接调用 Miles `compute_policy_loss`，对 dual-clip（A<0 时下界为 c·A）和 clip-higher（上界 1+ε_high）构造张量，对照手算结果。验证：单测通过，覆盖 A>0、A<0、ratio 落在区间内外。

## 3. KL loss 与参考模型身份（design D3）

- [x] 3.1 在 KL 描述中加入 `ref_model: {source, revision}`，`placement=loss` 时必填、默认 GRPO 不出现；进入规范化与哈希。验证：单测确认默认 GRPO 哈希不变、只改 ref revision 时哈希改变、缺失时启动前报错。
- [x] 3.2 learner 构建 spec 后核对 `ref_model.revision` 与本岛 `base_model_revision` 一致，不一致在启动前拒绝并写事件。验证：单测覆盖一致、不一致两种情况；fake 两岛测试中 ref 不同的岛在加入外层同步前失败。
- [x] 3.3 核对 `placement=loss` 翻译（k1/k2/k3/low_var_kl）及 `--ref-load` 的来源，未知估计器拒绝；在 miles-next-venv 用 upstream `parse_args` 解析，并确认 `use_kl_loss=True` 时 Miles 会走 ref 加载分支（`ray/specs/train.py:58` 的条件）。验证：单测与解析测试通过，结果写入 `progress.md`。

## 4. Dr.GRPO 常数分母 reducer（design D2）

- [x] 4.1 在运行镜像（R0 冒烟所用镜像，CPU 容器即可）中核实 `examples.experimental.DrGRPO.custom_reducer` 能否 import。验证：命令与输出写入 `progress.md`；无论结果如何都按 4.2 vendor，结果只用于文档说明。
- [x] 4.2 新建 `yeto/rl/algos/reducers.py`，vendor 该 reducer，分母改为从 runtime attrs 读取；文件头记录来源 radixark/miles `9e4260d`、路径、原始 blob 与源码 SHA256。验证：单测锁定来源哈希（对 miles-next 仓库 `git show 9e4260d:<path>` 计算）；分母缺失时报错。
- [x] 4.3 spec 增加 `loss.aggregation=constant` 与分母字段；拒绝与 `calculate_per_token_loss` 同时启用、拒绝 CP>1；翻译为 `--custom-pg-loss-reducer-function-path` 并登记 PluginRef。验证：参数化单测覆盖三种拒绝与正常翻译；upstream `parse_args` 解析通过。
- [x] 4.4 CPU 数值：用 CPU 张量（mpu 以桩替代 CP 查询）比较 vendor reducer 与原示例在分母 1000 时逐元素相同，以及任意分母 D 下等于有效 token loss 之和除以 D。验证：单测通过。

## 5. reward 后处理分派器（design D4/D5）

- [x] 5.1 新建 `yeto/rl/algos/reward_pipeline.py`：入口 `post_process(args, samples)`、`REWARD_SHAPERS` 与 `ADV_TRANSFORMS` 注册表、`grpo_default` 变换（复刻 `_reward_group_segments` 与 `_normalize_rewards_by_rollout`，含非 grpo 估计器和 `rewards_normalization` 关闭时的返回值）。验证：模块可在无 GPU 环境 import。
- [x] 5.2 等价测试：在 miles-next-venv 中 import Miles `_post_process_rewards`，对以下批次与 yeto 入口逐元素比对（`torch.equal`）：普通分组、多段样本（同一 rollout_id 多行）、G=1、组内奖励全相同（std=0）、无 group_index 的固定 fan-out、无 group_index 且不满足 fan-out、`rewards_normalization` 关闭、`grpo_std_normalization` 关闭、estimator 为 gspo/rpp_baseline/ppo。另测同一 rollout 奖励不一致时两者都报错。验证：测试通过；测试同时锁定 `train_data_conversion.py` 源码哈希，Miles 升级时失败并提示重审。
- [x] 5.3 插件配置经 `to_legacy_runtime_attrs` 下发到 `yeto_algo_plugins` 属性，附规范化哈希；分派器与 reducer 读取后核对哈希。验证：单测覆盖哈希一致、被篡改两种情况。
- [x] 5.4 翻译规则：只有描述中存在塑形或非默认变换时才输出 `--custom-reward-post-process-path`；检测到多 LoRA 时拒绝；无 group_index 回退为整批一组时写警告事件。验证：单测确认默认 GRPO argv 不含该参数、多 LoRA 被拒、警告事件内容正确。
- [x] 5.5 扩展点文档化：在模块 docstring 和 `docs/MILES_RL.md` 中说明注册新变换的步骤（注册表键、spec 字段、PluginRef、等价测试要求），供 P2 使用。验证：用一个测试内注册的恒等变换跑通入口，证明扩展点可用。

## 6. overlong 塑形与过滤（design D6/D7）

- [x] 6.1 实现 `REWARD_SHAPERS["overlong_penalty"]`，按 rollout 合并后的响应长度计算惩罚；spec 增加 L_max、L_cache 字段与约束（0 < L_cache ≤ L_max ≤ 生成长度上限）。验证：单测覆盖 L_max=100、L_cache=20 时长度 80/90/100/101 对应 0/−0.5/−1/−1，多段样本塑形后同 rollout 奖励仍一致，越界参数启动前拒绝。
- [x] 6.2 原始奖励写入 `sample.metadata["yeto_raw_reward"]`，事件汇总原始与塑形后奖励。验证：单测确认两者都出现在事件中。
- [x] 6.3 在 `record_trained_groups` hook 中组合 overlong 过滤：`sampling.overlong_filter=true` 时对 `status == truncated` 的样本设置 `remove_sample=True`，并把被过滤样本数写入 rollout 元数据。验证：单测确认截断样本被标记、未截断样本不变、默认配置下 hook 行为与之前完全一致。
- [x] 6.4 CPU 核对语义：在 miles-next-venv 中对含 `remove_sample` 样本的批次调用 Miles 训练数据转换，确认 loss mask 全 0、同组其他样本的 advantage 不变；用 `get_sum_of_sample_mean` 确认被屏蔽样本是否计入分母。验证：测试通过，结论写入 `docs/MILES_RL.md`。
- [x] 6.5 通过 `expects_gradient()` 声明 overlong 过滤的判定：非零方差组的样本全部被过滤时不期望梯度。验证：fake driver 测试中全截断且 grad_norm=0 的轮次不失败；部分截断且 grad_norm=0 仍失败；grad_norm 非有限仍失败。

## 7. 示例与文档（design D8/D9）

- [x] 7.1 （可选）新增 `examples/rl_algorithms/dapo-like.json` 与 `dr-grpo.json`。验证：CPU 测试加载每个示例，能构建 spec、argv 能被 upstream `parse_args` 解析；默认配置的 argv 快照不变。
- [x] 7.2 超采样：每轮事件记录本岛进入训练的样本数与组数。验证：fake 两岛测试中两岛各自记录，数值与过滤结果一致。
- [x] 7.3 更新 `docs/MILES_RL.md`：各机制的 spec 字段与翻译、dual-clip 约束、Dr.GRPO 与 RLOO 的关系、KL loss 的 ref 身份规则与显存影响、分派器与内置归一化的等价性及多 LoRA 限制、overlong 塑形公式、overlong 过滤与 DAPO 的差异、超采样下外层等权平均的影响、"声明支持 ≠ 效果收益"。验证：文档中的示例命令用 `--dry-run` 执行，输出与描述一致。

## 8. GPU 验证（需用户批准卡数与预算后执行）

> 本组的 G1（1 卡冒烟）统一使用 P0 提供的 `--rl-allow-unverified-mechanism <机制名>` 放行（见 `rl-algorithm-capabilities` design D11），只在单岛运行中生效；G1 通过后再在 adapter 中正式声明支持；两岛 G3 只用正式声明，不带放行参数。

- [x] 8.1 准备：复用 R0 冒烟小模型与 harness，证据目录 `openspec/changes/rl-algo-grpo-knobs/evidence/<日期>-<名称>/`，含 `YETO_SHA`、argv、事件与指标 jsonl。只开所需卡数（1 卡或 1+1 卡），不开整机；Modal 使用 `H100!:N` 并在启动时断言 GPU 名；日志与证据中不打印凭据。验证：dry-run 输出的资源请求与预期卡数一致；凭据扫描（grep token/key 模式）无命中。
- [x] 8.2 G1（1 卡）：clip-higher、dual-clip、token 级聚合、Dr.GRPO（去 std + 常数分母）、KL loss（k3）、entropy、超采样、overlong 软惩罚、overlong 过滤（调小生成长度以触发截断）各 2–3 轮。验证：每项相关指标键存在且有限（如 clipfrac、kl_loss、entropy_loss、被过滤样本数）、零梯度不变量无误报、policy token 与 receipt 正常；KL loss 另记录峰值显存与每轮耗时，并与同配置默认 GRPO 对比；结果逐项写入 `progress.md`。
- [ ] 8.3 对 G1 通过的每项，在 Miles adapter 能力声明中加入该机制（每项单独变更），fake engine 同步。验证：`check()` 单测接受已声明项、仍拒绝未通过项；`progress.md` 引用对应证据目录。
- [ ] 8.4 G3（1+1 卡）：两岛 strict-avg，组合配置 clip-higher + token 级聚合 + overlong（软惩罚与过滤），约 3 轮。验证：两岛算法哈希一致、外层同步后权重 hash 一致、不变量无失败、两岛有效样本数已记录。不做 decoupled 对比（须等 `fix-decoupled-lr-schedule` 合入）。
- [ ] 8.5 拆除与费用：每次运行后拆除全部资源。验证：provider 侧列出 app/实例/卷为空的输出存入证据目录（无残留证明）；按运行汇总卡时与费用写入 `progress.md`。
- [ ] 8.6 （可选，需另行申请预算）G4 效果 A/B 与 decoupled 对比：不在本 change 验收范围内，仅在用户批准且 `fix-decoupled-lr-schedule` 合入后记录方案。

## 9. 集成检查

- [x] 9.1 全量 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败集合与 1.1 相同。验证：集合 diff 为空，结果写入 `progress.md`。
- [x] 9.2 `openspec validate rl-algo-grpo-knobs --strict` 通过。验证：命令输出无错误。
- [x] 9.3 `progress.md` 逐项列出任务状态，严格区分"已实现""CPU 测试通过""GPU 验证通过（G1/G3）"，并列出仍为"可表达未开放"的机制。验证：文件存在且与任务逐项对应。
