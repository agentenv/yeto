# Tasks

> **S14 对账（2026-10-07，基线 integ-decl e8d387ac）**：本 change 是算法工作的唯一基准。39 项中 1 项已完成（1.2，CPU 核实），38 项未做（其中 24 项需 GPU）；逐项证据见 `progress.md`。阶段二全部字段（sampling.temperature/top_p/top_k、moe.*、kl.ref_update_interval、rollout.loss_mask_policy、RLOO）在 `yeto/rl/engine/algorithm.py` 中均不存在，`--rollout-temperature/--rollout-top-p/--rollout-top-k` 仍在 `algorithm_flags.py` 的 `_UNMAPPED`。接下来的计划见 §13，其它 rl-algo-* change 的未完项见 §14。

> 阶段一对应第 1–6 组，阶段二对应第 7–12 组。GPU 运行一律遵守 design D2、D9 与 BRIEF：判据事先提交，不放宽判据；失败时先修复再重跑；运行前写明卡型、硬超时和上限；结束后提供无残留证明。每项的费用上限见 design D9，括号内为该项上限。`max_policy_staleness` 固定为 0。INFRA、IMG 负责的文件只以补丁或请求方式交付（design D8）。

## 1. 阶段一门禁与准备（无 GPU）

- [x] 1.1 编写阶段一 GPU 测试计划 `evidence/phase1-plan.md`：对 4.x、5.x、6.x 的每一项逐条列出判据、最小证据、对照组、卡型、硬超时、费用上限和唯一前缀，并汇总阶段一上限 ≤ $35。验证：计划已提交，并得到**用户确认**（确认记录写入 `progress.md`）。确认之前不得开始第 2 组以后的 apply。（S14：未做，`evidence/phase1-plan.md` 不存在；需先定卡型，见 §13 Q1） **S19 10-09：计划 `evidence/phase1-plan.md` 已定稿；主 agent 10-09 代用户确认（沿用 S14 Q1–Q3；6.2 不做，除非 4.x 需要），记录见 progress.md "S19 阶段一门禁"。**
- [x] 1.2 核实前置状态：trainer v2 补丁（GMPO num/den）是否已合入 integ-decl；1a G3（g3c/g3d）所用的 launcher 回传磁带 harness 是否能直接复用于 1b；当前 pin 与 `FORK_COMMITS`。验证：结论与提交号或行号写入 `progress.md`。补丁未合入时，按 design D5 把 GMPO 标为暂缓。（S14 对账：三项结论见 progress.md §1，trainer v2 已合入 `yeto/rl/engine/miles_adapter/trainer.py:101,601-611`；磁带回传 harness 已被 seq-and-adv attempt 6 与 critic-family G1 复用；pin `c35702e` 在 `loss_variants.FORK_COMMITS`。GMPO 不暂缓。）
- [x] 1.3 记录全量测试基线（`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider -rfE`），失败 id 存入 `baseline-failures.txt`。验证：文件存在，命令与计数写入 `progress.md`。（S14：未做；最近全量基线只有 `infra-drafts/s13-full2-pytest.log`（68 failed/4084 passed/26 errors，HEAD 未记录），不能当 e8d387ac 基线） **S19 10-09：改用本机安全测试集（docs/TESTING.md），main 512bb773：9 failed / 5061 passed / 47 skipped / 23 deselected，失败 id 见 `baseline-failures.txt`，命令见 progress.md。**

## 2. clipfrac 定性与 fork 观测指标（CPU，fork 小提交）

- [x] 2.1 在 miles-next-venv 中离线复现 pg_clipfrac 疑点：compile 与 eager 两种模式、`eps_clip==eps_clip_high` 与 `≠` 两种情形，同时对照手算结果（design D3）。验证：`evidence/clipfrac-cpu/report.md` 给出“复现并定位根因”或“不能复现”的结论和可重跑脚本。（S14：未做；`rl-algo-grpo-knobs/evidence/2026-09-29-clipfrac-offline/report.md` 只有疑点与手算，compile/eager 对照未做） **S19 10-09：不能复现（CPU，compile 与 eager clipfrac 逐位相同且等于手算），见 `evidence/clipfrac-cpu/report.md`。**
- [x] 2.2 如果 2.1 确认是 fork 缺陷：在 `yeto/ports` 上做最小修复，并补 CPU 单测，证明修复前后 pg_loss 逐位相同、clipfrac 与手算一致。验证：fork 单测通过，经独立审查。如果不是缺陷，本项按合法否定结论记录并勾选。（S14：未做，依赖 2.1） **S19 10-09：合法否定结论——2.1 未确认 fork 缺陷，不改 fork；GPU 残余疑点由 5.2 检查。**
- [x] 2.3 在 fork 中新增 `dual_clipfrac`（A<0 且 dual 下界生效的 token 比例），附 CPU 单测：默认参数下 loss 逐位不变，构造张量上的比例与手算一致。验证：fork 单测通过，经独立审查。（S14：未做；fork pin c35702e 无 `dual_clipfrac`，`git grep` 为空） **S19 10-09：fork 本地提交 ed75bd2a0（分支 s19-algosup-metrics，基于 ddce209，未推送），5 个 CPU 单测通过，独立审查通过（子 agent 审查）。**
- [x] 2.4 在 fork 中记录超采样每次提交的批大小以及丢弃、补采计数，附 CPU 单测。验证：fork 单测通过，经独立审查；指标键名写入 `progress.md`。（S14：未做；fork 只在 `inference_rollout_train.py:120`/`sglang_rollout.py:483` 取 `over_sampling_batch_size`，未记录提交批大小） **S19 10-09：fork 本地提交 57d93872e（同分支），键名见 progress.md；独立审查后按意见补了 resumed/failed 计数，单测通过。**
- [ ] 2.5 把 2.2–2.4 的提交合并为一次 pin 更新请求交给 IMG：快进 `yeto/ports`、重建镜像、更新 pin；在 ALGO 侧更新 `FORK_COMMITS`，并按 git diff 为 `MILES_DECLARED_PINS` 中的 4 项做迁移论证。验证：新 pin 在 Modal T4 上通过完整 parse_args + validate_parsed_args（$0.5），迁移论证写入 `progress.md`，`tests/test_rl_algorithm_flags_upstream.py` 通过。（S14：未做；但 pin 迁移论证流程已在 IMG 的 5 次 pin 变更中跑通，模板见 `entry.py:143-180` 注释与 `loss_variants.py:78-90`；与 critic-family 6.3 的 pin 更新合并为一次）
- [x] 2.6 ALGO 侧接入新指标：在 fake 与 adapter 的指标映射中加入 `dual_clipfrac` 与超采样计数；需要改 trainer 取数时，以补丁 `infra-drafts/patches/algo-supp-metrics.patch` 交给 INFRA。验证：单测通过，补丁在临时副本上 apply 后相关测试通过。（S14：未做） **S19 10-09：ALGO 侧 ports/rollout/rollout_meta_hook/fake 已接入；driver 输出以补丁 `infra-drafts/patches/algo-supp-metrics.patch`（副本在 `patches/`）交 INFRA，在临时副本上 apply 后测试通过。**

## 3. 用户代码类机制永不声明（CPU）

- [x] 3.1 在 adapter 中加入“不可声明”名单（`corrections:custom`、`plugins`、`losses:custom_loss`、除 vendored Dr.GRPO reducer 以外的任意 `custom_pg_loss_reducer`），并加单测断言它与 `MILES_DECLARED` 不相交，同时断言放行开关在多岛下仍被拒绝。验证：新单测通过，全量测试失败集合与 1.3 基线一致。（S14：未做；多岛拒绝放行的单测已有 `tests/test_rl_algorithm_capabilities.py:413-440`，缺"不可声明名单"及其与 `MILES_DECLARED` 不相交断言） **S19 10-09：`entry.NEVER_DECLARABLE` + `tests/test_rl_algo_never_declarable.py`（8 个用例通过）；全量失败集合与 1.3 基线一致。**
- [x] 3.2 在 `docs/MILES_RL.md` 的 “Declaration policy” 中写入这条规则，并同步修正 rl-algo-capabilities 页第 6 节列出的四处文档与代码不一致。验证：文档中的 dry-run 示例按所写命令执行，输出一致。（S14：未做；`docs/MILES_RL.md:857` Declaration policy 仍写 dual_clip/opsm_rollout/custom 为"pending evidence or approval"，A12 审计称文档多处过时） **S19 10-09：Declaration policy 写入"永不声明"规则，修正第 6 节四处不一致及 dry-run 命令路径；dry-run 复跑输出见 `evidence/docs-dry-run/output.txt`。**

## 4. 单卡 G1 生效验证（A10G；前置：1.1 已确认，2.5 新 pin 已就绪）

- [ ] 4.1 在新 pin 上跑同 seed 的默认 GRPO 对照组（$2.0，含对旧 pin 的一次复核），`num_steps_per_rollout=2`，3 轮。验证：通用判据满足，指标 jsonl 存入 `evidence/g1-baseline/`，供 4.2–4.7 与 5.x 共用。（S14：未做，GPU；无对应声明）
- [ ] 4.2 CISPO、SAPO、GMPO 各跑一次 G1（$4.5），判据采用 loss-variants progress 修订版中的 (a)–(f)，不做修改；如果 1.2 判定 GMPO 暂缓，只跑 CISPO 和 SAPO。验证：每个变体的判据逐项写入 `evidence/g1-variants/results.md`；通过的变体在 `MILES_DECLARED` 中各自单独声明并附证据路径；adapter `check()` 单测接受已声明项，仍拒绝未通过项。（S14：未做，GPU；无对应声明）
- [ ] 4.3 对 4.2 通过的每个变体，与已声明的 `corrections:tis` 组合做冒烟（$3.0），判据为 (a)(b)(c)，GMPO 另加 (e)。验证：结果写入 `evidence/g1-variants-tis/`；组合不另作声明，结果记为“组合 GPU 冒烟通过”。（S14：未做，GPU；无对应声明）
- [ ] 4.4 dual_clip：`eps_clip_c=1.01`，每轮 2 步，3 轮（$1.5）。判据：`dual_clipfrac` > 0 且有限，加上通用判据。验证：通过后声明 `features:dual_clip`；如果比例恒为 0，记为合法否定结论，不声明。（S14：未做，GPU；无对应声明）
- [ ] 4.5 KL：k1、k2、low_var_kl、kl_unbiased 各跑一次（$4.0），放置位置为 `loss`，coef 0.01，带显式 `kl.ref_model`。判据：第 1 轮之后 `kl_loss` 存在、有限、> 0，并与已声明的 k3 运行数值不同。验证：每项单独判定、单独声明，结果写入 `evidence/g1-kl/`。（S14：未做，GPU；无对应声明）
- [ ] 4.6 opsm_rollout（$1.5）：沿用 1a trigger 的小阈值，每轮 2 步。判据：`opsm_clipfrac` > 0，并且来源记录显示 π_old 取自 rollout logprob。验证：通过后声明 `corrections:opsm_rollout`。（S14：未做，GPU；无对应声明）
- [ ] 4.7 no_rewards_normalization 与 grpo+whiten 各跑一次（$2.5）。判据：advantage 的均值和方差与对照组不同，并与离线重算结果一致（容差 1e-5）。验证：通过后分别声明；单测断言 grpo+whiten 的变 DP 边仍被 `reshard` 拒绝；`docs/MILES_RL.md` 记录这一限制。（S14：未做，GPU；无对应声明）

## 5. 既有弱证据复核（A10G）

- [ ] 5.1 over_sampling 强证据（$2.0）：使用 2.4 的指标，配置上能触发动态过滤。判据：至少一轮的提交批大小大于 rollout batch size，并且被过滤组数 > 0。验证：`evidence/g1-over-sampling/results.md`；通过后把声明证据替换为新路径；不通过时撤回声明，并在 `progress.md` 说明。（S14：未做，GPU；无对应声明）
- [ ] 5.2 clip_higher 在新 pin 上独立重跑（$1.5），沿用 g1j 配置与 g1j 预登记的判据。验证：通过后 `MILES_DECLARED_PINS` 中这一项改记为“在该 pin 上实测”；不通过时撤回声明。（S14：未做，GPU；无对应声明）
- [ ] 5.3 只有当 2.2 做了修复时才执行（$1.5）：在 GPU 上核对修复后 pg_clipfrac 与离线重算一致，并且 pg_loss 与修复前逐位相同。验证：结果写入 `evidence/g1-clipfrac/`；2.2 为否定结论时，本项记为不适用并勾选。（S14：未做，GPU；无对应声明）

## 6. 两岛 G3 与阶段一收尾（A10G 1+1）

- [ ] 6.1 1b 的 8.4 G3（$4.0）：两岛 strict-avg，组合配置为 clip-higher + token 级聚合 + overlong（软惩罚与过滤），3 轮，使用 launcher 回传磁带 harness，不带放行参数。验证：两岛算法哈希一致，每轮外层同步后的权重哈希一致，不变量无失败，两岛有效样本数已记录；结果同步回 rl-algo-grpo-knobs 8.4 的完成记录。（S14：未做；= rl-algo-grpo-knobs 8.4，仍未勾）
- [ ] 6.2 （可选，需用户在 1.1 中确认）CISPO 两岛 strict-avg（$3.5），前提是 4.2 已经声明 CISPO，判据采用 loss-variants 6.3 的判据。验证：结果写入 `evidence/g3-cispo/`；未获确认时记为未执行。（S14：未做，GPU；无对应声明）
- [ ] 6.3 阶段一收尾：更新 `docs/MILES_RL.md` 的声明清单与证据路径；逐项汇总费用（≤ $35）并提供无残留证明；`progress.md` 按五种状态列出每项结果，并列出仍为“可表达未开放”的机制。验证：全量测试失败集合与基线一致；`openspec validate rl-algo-supplement --strict` 通过。（S14：未做）

## 7. 阶段二门禁（无 GPU）

- [ ] 7.1 在阶段一完成之后，编写阶段二 GPU 测试计划 `evidence/phase2-plan.md`（格式同 1.1，上限 ≤ $25，含 MoE）。验证：计划已提交，并得到**用户确认**。（S14：未做）
- [ ] 7.2 向 INFRA 提出接口请求：cut manifest 中的 ref 版本字段（rollout 编号 + ref 权重哈希），与 rl-infra-spec 4.1/4.2 对齐。验证：请求写入 `rl-infra-spec/alignment.md` 的追加待批准项，并记录 INFRA 的回复或排期。（S14：未做）

## 8. rollout 采样参数进入身份（CPU + A10G）

- [ ] 8.1 在 spec 中加入 `sampling.temperature/top_p/top_k`：默认不输出；把对应参数从 `UNMAPPED_OBJECTIVE_FLAGS` 移出；实现吸收与冲突检测；拒绝 temperature≠1 与 rollout logprob 作为 π_old 的组合。验证：单测覆盖默认哈希不变（R0 与默认 GRPO 快照不改即通过）、吸收、冲突报错和拒绝；upstream parse_args 解析通过。（S14：未做）
- [ ] 8.2 G1（$2.5）：temperature 0.7 与 top_p 0.9 各跑一次。判据：SGLang 请求参数中出现所设的值（来源于日志或事件），规范化 spec 包含该值，通用判据满足。验证：通过后声明；在 `docs/MILES_RL.md` 记录字段。（S14：未做）

## 9. ref 周期更新（CPU + A10G；依赖 7.2）

- [ ] 9.1 实现 `kl.ref_update_interval` 的字段、翻译与拒绝规则（KL 不生效时拒绝；在需要 cut 的 profile 下，只要 INFRA 尚未提供 ref 版本字段就在启动检查中拒绝），并在事件中记录 ref 版本。需要 driver 或 trainer 配合时以补丁交付。验证：单测通过；补丁在临时副本上 apply 后测试通过。（S14：未做）
- [ ] 9.2 G1（$2.0）：间隔为 1，3 轮。判据：每轮的 ref 版本递增，第 2 轮之后的 `kl_loss` 小于同 seed 下间隔为 None 的运行。比较方向事先固定。验证：通过后声明，证据写入 `evidence/g1-ref-update/`。（S14：未做）

## 10. loss mask 策略（CPU + A10G）

- [ ] 10.1 核实 Miles 在当前 pin 下的实际 loss mask 行为与可实现的变体，固定 `rollout.loss_mask_policy` 的枚举（默认值等于当前行为）。验证：结论与行号写入 `progress.md`；枚举集合在本项中固定之后不再扩展。（S14：未做）
- [ ] 10.2 实现字段、翻译与拒绝规则（未知取值时列出允许值）。验证：单测确认默认哈希不变，非默认值进入哈希。（S14：未做）
- [ ] 10.3 G1（$1.5）：一个非默认取值。判据：训练侧的 masked token 数与对照组不同，并与离线按策略重算的结果一致。验证：通过后声明。（S14：未做）

## 11. MoE routing replay（CPU + L40S 或 A100）

- [ ] 11.1 CPU 核实 MoE 候选（design D10）：OLMoE-1B-7B 是否满足 Megatron bridge 支持、SGLang 返回 `routed_experts`、可挂 LoRA；不满足时改用 Qwen1.5-MoE-A2.7B。验证：结论写入 `progress.md`，并确定卡型与费用估算。（S14：未做）
- [ ] 11.2 把 `moe.rollout_routing_replay/routing_replay` 移入 spec 与哈希：吸收旧的 agent 开关，非 MoE 模型时拒绝，从 `UNMAPPED_OBJECTIVE_FLAGS` 移出 `--use-routing-replay`；`cli/launcher/learner` 的接线改动以补丁交主 agent 指定的负责人。验证：单测覆盖默认哈希不变、旧开关被吸收、稠密模型被拒；在 `docs/MILES_RL.md` 写入迁移说明。（S14：未做）
- [ ] 11.3 GPU（两次运行合计 $12）：分别在开启与关闭 routing replay 下运行，其余配置相同，3 轮。判据：开启时有训练侧使用 rollout 路由的证据，且 `train_rollout_kl` 不大于关闭时的值，比较方向事先固定。验证：通过后声明；证据写入 `evidence/g1-moe-replay/`。（S14：未做）

## 12. RLOO 与阶段二收尾

- [ ] 12.1 在 yeto reward 分派器中实现 RLOO 阶段：组大小 ≥2，配套 `std_normalization=false`，身份独立；在文档中说明它与 Dr.GRPO 只差常数缩放。验证：单测覆盖 [1,0,0,1]→[2/3,−1/3,−1/3,2/3]、组大小为 1 时被拒、哈希与 Dr.GRPO 不同。（S14：未做）
- [ ] 12.2 G1（$1.5）。判据：分派器输出的 advantage 与离线重算一致（容差 1e-6），通用判据满足。验证：通过后声明。（S14：未做）
- [ ] 12.3 阶段二收尾：逐项汇总费用（≤ $25）并提供无残留证明；`progress.md` 按五种状态列出结果；在 `progress.md` 的后续清单中列出 OPD、critic/PPO/VAPO、异步/staleness>0/partial rollout、GiGPO 等后续 change。验证：全量测试失败集合与基线一致；`openspec validate rl-algo-supplement --strict` 通过。（S14：未做）

## 13. 接下来的计划（S14 制定；待用户拍板 Q1–Q4 后生效）

> 原则：每项 GPU 运行前仍按 design D2/D9 与 S14-SUBAGENT-RULES（预登记 gpu-spend.md、线程守卫、H100 用 `H100!`、单岛 no-sync 的 controller 可本地，需 syncer 的两岛把 controller+syncer 放 Nebius 无卡 VM）。费用按 Modal 牌价估算：H100 $4.39/h、A10G ≈$1.10/h、L40S ≈$2/h；单岛 G1 实测一次 ≈20–35 min（critic G1 先例 $1.0–1.6/次 @H100）。

| 阶段 | 内容 | 依赖 | CPU 估算 | GPU 估算（A10G 方案 / H100! 方案） |
|---|---|---|---|---|
| P0 门禁 | 1.1 阶段一计划、1.3 基线（在 e8d387ac 上重跑全量，排除会起 Ray 的用例或在 GPU 容器内跑）、3.1 不可声明名单、3.2 文档 | 无 | 1 个子 agent 轮次 | $0 |
| P1-a CPU | 2.1 clipfrac compile/eager 定性（miles-next-venv，fork c35702e `loss_hub/math_utils.py`）、2.3 dual_clipfrac、2.4 超采样提交批大小（fork 小提交 + CPU 单测）、2.2 视 2.1 结论 | P0 | 1–2 轮次 | $0 |
| P1-b pin | 2.5 一次 pin 更新（与 critic-family 6.3 合并：fork `yeto/ports` 快进含 SAO/critic 参数 + 2.x 指标）、IMG 重建镜像、`FORK_COMMITS`/`MILES_DECLARED_PINS` 迁移论证、2.6 指标接入 | P1-a | 1 轮次（IMG） | T4 parse_args $0.5 |
| P1-c G1 | 4.1 对照；4.2 CISPO/SAPO/GMPO；4.4 dual_clip；4.5 KL×4；4.6 opsm_rollout；4.7 norm/whiten；5.1 over_sampling；5.2 clip_higher；5.3（条件）；4.3 变体+TIS ×3 | P1-b | 每批 1 轮次 | 18 次单岛：≈$18 / ≈$29 |
| P1-d G3 | 6.1 1b 8.4 两岛（grpo-knobs 8.4）；6.2 CISPO 两岛（可选） | P1-c | 1 轮次 | 2 次 1+1：≈$7.5 / ≈$13（含 Nebius VM ≈$0.2/h） |
| P1 收尾 | 6.3 | P1-d | 1 轮次 | $0 |
| **阶段一合计** | | | | **≈$26 / ≈$43**（design D9 的 $35 上限按 A10G 定；若用 H100! 需把上限提到 ≈$45） |
| P2-a CPU | 7.1/7.2；8.1 采样参数入 spec；10.1/10.2 loss_mask_policy；11.1/11.2 MoE replay 入 spec（接线补丁交主 agent）；12.1 RLOO 阶段 | 阶段一收尾 | 2–3 轮次 | $0 |
| P2-b G1 | 8.2 ×2、10.3、12.2、9.2（依赖 rl-infra 4.1/4.2 的 ref 版本字段，可能顺延） | P2-a | 1 轮次 | 5 次单岛：≈$5 / ≈$8 |
| P2-c MoE | 11.3 开/关各一次（OLMoE-1B-7B 优先，L40S；或 H100!） | 11.1 | 1 轮次 | ≈$8 / ≈$9 |
| **阶段二合计** | | | | **≈$13 / ≈$17**（design 上限 $25 维持） |

- 本阶段额度：记忆 `gpu-budget-700` 记 10-07 额度 $400、累计 ≈$195；阶段一+二按 H100! 方案 ≈$60，仍在额度内，但与 §14 汇入的 critic-family G3（≈$12–30/项）合计会接近上限，需用户排序。
- 排序建议：P0 → P1-a → P1-b（与 critic 6.3 合并 pin）→ P1-c（先跑 4.1/4.2/4.4/5.1/5.2，这 5 项直接解除 HANDOFF-AUDIT A1 列的四个遗留：clipfrac、over_sampling、dual_clip、GSPO/变体）→ 其余。
- 上卡前每批都要写复核文档（记忆 `gpu-review-before-launch`）并在 `evidence/phase1-plan.md` 预登记判据，不得事后放宽。

## 14. 汇入待办（其它 rl-algo-* change 的未完成项，只引用不复制；状态以来源 change 的 tasks.md 为准）

| 来源 change | task | 内容（摘要） | 类型 | 估算 | 本 change 承接位置 |
|---|---|---|---|---|---|
| rl-algo-critic-family | 4.5 | PPO 两岛 strict-avg 1+1×H100 3 轮 + kill/resume（S13 远端 controller 形态已跑通但 0 轮 INCOMPLETE，见其 progress.md "S13 critic G3"） | GPU G3 | ≈$12（上限 $20） | P1-d 之后，独立批次 |
| rl-algo-critic-family | 6.3 | push fork、更新 `MILES_NEXT_COMMIT` pin 与镜像、映射表登记 | pin/IMG | T4 $0.5 | **并入 P1-b 一次 pin 更新** |
| rl-algo-critic-family | 7.3 | VAPO G1 已 PASS（vapo-20261007e），G3 1+1×H100 未跑；正式声明待 G3 | GPU G3 | ≈$6 | 与 4.5 同批 |
| rl-algo-critic-family | 8.4 | SAO G1 已 PASS（s14-forkg1-sao-20261007a，EV ≤0 已记录），G3 未跑 | GPU G3 | ≈$6 | 与 4.5 同批 |
| rl-algo-critic-family | 9.4 / 9.5 | CompactionRL G1（agent 环境待选）/ G3 | GPU | ≈$8 / ≈$16 | 阶段二之后，需用户另批 |
| rl-algo-critic-family | 10.1–10.4 | critic LoRA（后续） | CPU+fork+GPU | ≈$12 | backlog，不排期 |
| rl-algo-critic-family | — | PPO/VAPO/SAO 的 `MILES_DECLARED`/`fake.py critic=True` 正式声明（progress.md "S13 GPU G1 结果汇总"第 4 条，待 G3） | CPU | $0 | G3 通过后与 6.3 收尾一起做 |
| rl-algo-loss-variants | 3.1–3.4 | 路线 A（adapter 插件）；用户 09-30 已选路线 B，tasks.md 顶部复核已说明不执行 | **已取代** | — | 建议来源 change 标注关闭 |
| rl-algo-loss-variants | 6.1–6.4 | CISPO/SAPO/GMPO G1 + CISPO 两岛 + 拆除证明 | GPU | 见 4.2/4.3/6.2 | **= 本 change 4.2、4.3、6.2** |
| rl-algo-grpo-knobs | 8.4 | 两岛 G3 组合（clip-higher+token+overlong） | GPU G3 | ≈$4–6.5 | **= 本 change 6.1** |
| rl-algo-grpo-knobs | 8.5 | 拆除与费用汇总 | CPU | $0 | 随 6.3 |
| rl-algo-grpo-knobs | 8.6 | G4 效果 A/B（可选，需另批） | GPU | 未估 | 不排期（proposal 非目标） |
| rl-algo-mismatch-correction | 5.2 | vendor 副本 vs `examples.` 命名空间（偏离 D6，待批准） | 决策 | $0 | Q5 |
| rl-algo-mismatch-correction | 7.3 | 其余机制声明；其中 opsm_rollout 未跑 G1 | GPU | 见 4.6 | **= 本 change 4.6**；icepop/mis_mask/observe 已在 `MILES_DECLARED`（entry.py:94-96），来源 change 可勾 |
| rl-algo-mismatch-correction | 7.7 | G4 效果 A/B（可选） | GPU | 未估 | 不排期 |
| fix-decoupled-lr-schedule | 3.2 | 两岛 decoupled legacy vs ports 各一次，学习率逐位一致 | GPU G3 | ≈$6（1+1 短跑） | 与 6.1 同批（同为 1+1 两岛） |
| rl-algo-seq-and-adv | 3.6 | rpp 梯度规则"收紧"分支依赖 KL 大小上报，记为已知限制（progress.md:50,144） | 已知限制 | $0 | 阶段二不处理；若要补证据并入 4.5 KL 批次 |
| rl-algo-seq-and-adv / rl-algorithm-capabilities | — | tasks 全勾，无汇入 | — | — | — |
| rl-fn-codex-rollout（新，S14 10-07） | 0.1–3.3 | FN 以 codex harness 为 rollout/奖励源：阶段 0 CPU（FN profile、5.1/5.2、A16、9.2 失配根因）→ 阶段 1 四层 1×H100! → 阶段 2 全尺寸 8×H200 两轮 → 阶段 3 并入 FN-TRAIN-PLAN | CPU+GPU | ≈$2–3 + $46–70（上限 $100） | 独立 change，需用户拍板 Q1–Q5（其 design Open Questions） |

## 15. S19 阶段一门禁盘点（2026-10-09，只读盘点，未改勾选）

> 范围：阶段一 = 第 1–6 组。当前 39 项中只有 1.2 完成。用户已在 S14 裁定 Q1–Q6（见 progress.md "S14 用户裁定"）。用户 10-09 另定：SAO 与 CompactionRL 最先（在 rl-algo-critic-family，见其 tasks.md 顶部与 design D-S19），所以本 change 阶段一排在 critic 的 SAO/CompactionRL 之后。

**已满足**
- 1.2 前置状态核实（S14 完成）。
- 1.1 的前置决策：卡型（A10G 优先，Q1）、范围（4.x/5.x 全做，Q2）、pin 合并（Q3）已由用户给出；但 1.1 本身（计划文件 + 用户确认）未做。

**不需上卡、不需用户确认（CPU，可直接派子 agent）**
- 1.3 测试基线：改用本机安全测试集（`/home/michael/work/yeto-test-venv`，docs/TESTING.md），排除会拉 Ray 的用例；原命令里的 `/tmp/yeto-venv` 全量不再适用。
- 2.1 clipfrac compile/eager 定性；2.2（只在 2.1 确认是 fork 缺陷时做）；2.3 dual_clipfrac；2.4 超采样计数；2.6 指标接入。
- 3.1 不可声明名单；3.2 文档。
- 1.1 计划文件的起草（确认要用户做，见下）。

**需要用户确认**
- 1.1 `evidence/phase1-plan.md` 定稿后需用户确认，确认前不得开始第 2 组以后的 apply（本 change 原规定）。建议并入第三批合并上卡统一报批。
- 2.5 pin 更新：fork 推送仍按 S14"overlay 补丁代替 push"，若要真推 `yeto/ports` 需用户另批；镜像重建由 IMG 做。与 critic 6.3 合并为一次。
- 6.2 CISPO 两岛（可选），需用户在 1.1 中确认。
- 阶段一总预算（design D9 上限 $35，A10G 方案约 $26）需在第三批中报批。

**需要上卡**（A10G 优先，前置为 1.1 已确认、2.5 新 pin 就绪）
- 4.1 对照、4.2 CISPO/SAPO/GMPO、4.3 变体+TIS、4.4 dual_clip、4.5 KL×4、4.6 opsm_rollout、4.7 norm/whiten。
- 5.1 over_sampling、5.2 clip_higher、5.3（条件）。
- 6.1 两岛 G3、6.2（可选）。
- 2.5 的 parse_args 检查（T4，约 $0.5）。

**建议顺序**：critic 6.3 + 本 change 2.5 合并 pin → critic SAO/CompactionRL → 本 change 1.3/2.x/3.x（CPU，可与前一步并行）→ 1.1 定稿报批 → 4.x/5.x → 6.x。
