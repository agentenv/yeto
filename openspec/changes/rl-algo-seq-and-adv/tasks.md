# Tasks

执行约定：
- 规划文档以 `rl-infra-spec` 分支上的副本为准；实现 worktree、分支与可修改路径按 `../rl-infra-spec/alignment.md` 的工作包派发（原 `/home/michael/work/rl-algos` 分支 `rl-algorithms` 不再作为规划来源）；前置：P0 `rl-algorithm-capabilities` 与 P1-b `rl-algo-grpo-knobs` 的分派器已合入。
- 测试命令为 `/tmp/yeto-venv/bin/python -m pytest -q`；全量以"失败集合前后相同"为准。涉及 upstream 参数解析或 Miles 原函数对照的测试在 `/home/michael/work/miles-next-venv` 中运行。
- 不修改 legacy `build_miles_argv`，不修改 Miles/SGLang fork，不向 radixark/miles 或 sgl-project/sglang 提 PR。
- 状态严格区分"已实现""CPU 测试通过""GPU 验证通过"；mock/fake 测试不能替代 GPU 验收。
- commit 和 push 需要用户确认。

## 1. 基线

- [x] 1.1 改动前运行 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，把失败与错误集合保存到 `openspec/changes/rl-algo-seq-and-adv/baseline-failures.txt`。验证：文件存在，条目数与 pytest 汇总一致。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 1.2 核对 P1-b 分派器实际接口（变换注册点、分组与 rollout_key 合并函数、指标通道、），在 `progress.md` 记录与 design D5/D8/D9 的名称差异。验证：记录存在；若 P1-b 已有放行机制，第 6 组改为复用。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。

## 2. GSPO（design D1/D2）

- [x] 2.1 在 clip 缺失的拒绝报错中补充"引擎默认 0.2、论文 3e-4/4e-4"；GSPO 与 advantage 变换组合按未开放拒绝。验证：参数化单测检查报错文本与拒绝场景。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 2.2 adapter 从 Miles 训练指标读取 `pg_clipfrac`（整轮 token 加权）填入 `masked_fraction`，并写入每轮事件。验证：单测覆盖有值、多 mini-batch 聚合、缺失三种情况。 完成记录：INFRA R1（infra-a 39fa0ac）+ test_trainer_reads_gspo_clipfrac_through_seq_adv；progress.md。
- [x] 2.3 实现 GSPO 的 `expects_gradient`（D2）。验证：fake driver 测试覆盖：clipfrac=1 且 grad_norm=0 不失败；clipfrac 缺失且 grad_norm=0 失败；clipfrac<1 且 grad_norm=0 失败；grad_norm 非有限失败。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 2.4 在 miles-next-venv 中用 Miles `compute_gspo_kl`/`compute_policy_loss` 对小张量计算序列 ratio 与 clip，与手写公式（exp(mean log ρ)、序列级 clip）逐元素比对，并确认全裁时 clipfrac=1、梯度为 0。验证：测试在 miles 不可用时 skip，在 miles-next-venv 实际通过，命令与结果写入 `progress.md`。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 2.5 翻译用例：gspo + 显式 clip 生成的 argv 经 upstream `parse_args` 解析通过。验证：miles-next-venv 中测试通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。

## 3. REINFORCE++ 家族与 gamma（design D3/D4）

- [x] 3.1 在 `algorithm_flags.py` 新增 `--gamma` → `advantage.gamma` 映射行（默认 1.0 不输出），从"未映射清单"移出；`--lambd` 保留在未映射清单。验证：映射表子集测试、upstream 存在性测试通过；`tests/test_rl_argv_snapshot.py` 不改即通过。 完成记录：progress.md（2026-09-29 merge algo-cap 2f9f02c）。
- [x] 3.2 校验规则：gamma≠1.0 只允许 `reinforce_plus_plus`，其他估计方式拒绝；rpp/rpp_baseline 与 `whiten=false` 为可表达未开放。验证：参数化单测。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 3.3 分派器对 rpp（恒等）与 rpp_baseline（组内减均值、不除 std）的路径，与 Miles 内置 `_post_process_rewards` 逐元素比对，含多段 rollout。验证：miles-next-venv 中对照测试通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 3.4 在 miles-next-venv 中对 `get_reinforce_plus_plus_returns`、`get_reinforce_plus_plus_baseline_advantages` 与 `normalize_advantages`（单进程 DP=1）构造小输入，确认 reward-KL 进入 advantage、白化后均值约 0。验证：测试通过，记录于 `progress.md`。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 3.5 翻译用例：rpp/rpp_baseline + whiten + `kl.placement=reward` 生成的 argv 经 upstream `parse_args` 解析通过。验证：miles-next-venv 中测试通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 3.6 rpp 与 rpp_baseline 的 `expects_gradient`（D8）。验证：fake driver 单测覆盖期望/不期望梯度与读不到统计三种情况。 完成记录：progress.md（2026-09-29 merge algo-cap 2f9f02c）。

## 4. MaxRL 与 MAPO（design D5/D7/D8）

- [x] 4.1 在分派器注册 `maxrl` 与 `mapo` 变换，复用 P1-b 的分组与 rollout_key 合并；只允许 `estimator=grpo`；运行时非 {0,1} 奖励使本轮失败；与 overlong 软惩罚组合启动前拒绝。验证：单测覆盖拒绝与运行时失败。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 4.2 数值测试：与独立写的论文公式参考实现逐元素比对，边界覆盖 G=1、全错、全对、std=0、[1,0,0,1]、多段 rollout（三段共享一个 advantage）、段间奖励不一致报错；所有输出有限。验证：`tests/test_rl_adv_transforms.py` 通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 4.3 MAPO 在 p=0.5 时与 Miles 内置 GRPO 归一逐元素一致。验证：miles-next-venv 中对照测试通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 4.4 零梯度等价性：对 G≤8 的全部二值组合穷举，证明 `any(reward_std>0)` 与"变换输出存在非零条目"在 MaxRL/MAPO 下等价；fake driver 中全错轮次 grad_norm=0 不失败。验证：单测通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。

## 5. GDPO 与奖励向量（design D6/D8）

- [x] 5.1 AlgorithmSpec 增加 GDPO 分量声明（name、weight），规范化按 name 排序并进入哈希；空列表、重名、非有限权重启动前拒绝。验证：单测覆盖哈希随权重变化、各拒绝场景。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 5.2 分派器实现 GDPO：读取 `sample.metadata["yeto_reward_components"]`，缺分量、多分量、非有限、段间不一致时本轮失败并报告样本与分量；组内分量归一、加权、岛内样本级白化。验证：单测覆盖各失败场景。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 5.3 数值测试：与论文公式参考实现逐元素比对，覆盖 G=1、某分量组内恒定、全批恒定（白化仅减均值）、多段 rollout。验证：`tests/test_rl_adv_transforms.py` 通过。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 5.4 提供一个两分量示例 reward 函数（如 correctness + format），供 G1 使用，并写单测。验证：单测确认写入的 metadata 格式可被分派器接受。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 5.5 GDPO 的 `expects_gradient` 读取分派器上报的非零条目数，读不到时按期望梯度处理。验证：fake driver 单测。 完成记录：INFRA R2 + fake driver 测试 test_fake_driver_gdpo_*；progress.md（注意 R2 在 GPU 上滞后一轮，已报缺陷）。

## 6. 能力声明机制与文档

- [x] 6.1 确认 P0 的 `--rl-allow-unverified-mechanism` 能放行本 change 的每个机制名（GSPO、REINFORCE++、REINFORCE++-baseline、MaxRL、MAPO、GDPO）。验证：fake 组合根单测覆盖每个机制在单岛放行时可以启动、不放行时被拒。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 6.2 在 fake engine 中声明全部六个机制，供 CPU 组合根测试；Miles adapter 暂不声明。验证：fake 组合根测试中各机制可启动，Miles adapter 声明未变。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 6.3 更新 `docs/MILES_RL.md`：六个机制的配置示例、GSPO clip 必须显式及 clipfrac 监控、单步 optimizer 下 clip 不起作用、岛内白化语义与外层尺度推断、奖励向量格式、二值奖励要求、MAPO 证据弱、"声明支持不等于有收益"。验证：示例命令 `--dry-run` 执行结果与文档一致。 完成记录：progress.md（2026-09-29 merge algo-cap 2f9f02c）。

## 7. GPU 验证（需用户批准卡数与预算后执行）

> 本组的 G1（1 卡冒烟）统一使用 P0 提供的 `--rl-allow-unverified-mechanism <机制名>` 放行（见 `rl-algorithm-capabilities` design D11），只在单岛运行中生效；G1 通过后再在 adapter 中正式声明支持；两岛 G3 只用正式声明，不带放行参数。

约定：Modal 用 `H100!:N` 并断言 GPU 名为 H100；只开所需卡数；日志不打印凭据；每项结束后拆除并证明无残留；报告费用。decoupled 下的任何对比实验等 `fix-decoupled-lr-schedule` 合入后再做，不在本组。

- [x] 7.1 向用户提交预算申请：G1 共 6 次 1 卡冒烟（可在同一 1 卡会话中顺序执行）、G3 一次 1+1 卡，估算卡时与费用。验证：用户书面批准，记录于 `progress.md`。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [ ] 7.2 G1 GSPO：1 卡，使用放行参数，2–3 轮，`optimizer_steps≥2`，显式 clip；另跑一组 `optimizer_steps=1` 作对照。报告每轮 clipfrac、grad_norm、零梯度判定结果。验证：运行完成、无不变量误报，日志与指标存入 `openspec/changes/rl-algo-seq-and-adv/evidence/`。
- [ ] 7.3 G1 REINFORCE++ 与 REINFORCE++-baseline：各 1 卡 2–3 轮，whiten 开，`kl.placement=reward`。报告 ref 加载、advantage 统计、grad_norm。验证：同 7.2。
- [x] 7.4 G1 MaxRL、MAPO、GDPO：各 1 卡 2–3 轮（GDPO 用 5.4 的示例 reward）。报告全错/全对组比例、零梯度判定。验证：同 7.2。 完成记录：evidence/g1/plan.md Attempt 4（e54d2f7，H100）。
- [ ] 7.5 对 G1 通过的机制在 `miles_adapter/entry.py` 中声明支持，未通过的保持未开放并记录原因。验证：能力声明单测更新；Miles adapter 组合根测试中已声明机制可启动。
- [ ] 7.6 G3：MaxRL 两岛 strict-avg（每岛 1 卡），用正式声明（不带放行参数），2–3 轮。验证：两岛算法哈希一致、每轮外层应用后状态 hash 一致、不变量无误报；证据存入 `evidence/`。
- [ ] 7.7 拆除全部 GPU 资源，列出云端资源证明无残留，汇总实际费用。验证：无残留截图或命令输出与费用写入 `progress.md`。
- [x] 7.8（可选，另需预算）效果 A/B（G4）不在本 change 范围；如需，另立 change 申请。验证：无（仅记录）。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。

## 8. 集成检查

- [x] 8.1 运行 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败集合与 1.1 基线相同。验证：两集合 diff 为空，结果写入 `progress.md`。 完成记录：progress.md（2026-09-29 merge algo-cap 2f9f02c）。
- [x] 8.2 `openspec validate rl-algo-seq-and-adv --strict` 通过。验证：命令输出无错误。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
- [x] 8.3 在 `openspec/changes/rl-algo-seq-and-adv/progress.md` 中逐项列出任务状态，区分"已实现""CPU 测试通过""GPU 验证通过（G1/G3）"，并列出未通过或未声明的机制。验证：文件存在且与本任务列表逐项对应。 完成记录：progress.md（2026-09-29 ALGO-2a 任务状态表）。
