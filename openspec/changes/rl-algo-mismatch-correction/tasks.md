约定：
- 测试命令为 `/tmp/yeto-venv/bin/python -m pytest -q`；全量测试以"改动前后失败集合相同"为准。
- 依赖 Miles 源码的 CPU 测试与 upstream 参数解析在 `/home/michael/work/miles-next-venv` 中运行，miles 不可用时 skip；任务说明中记录命令与结果。
- 不修改 legacy `build_miles_argv`，不修改 Miles/SGLang fork，默认 GRPO argv 与哈希不变。
- 状态严格区分"已实现""CPU 测试通过""GPU 验证通过"；fake 测试不能替代 GPU 验收。
- commit 和 push 需要用户确认。

## 1. 基线

- [x] 1.1 改动前运行 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败与错误集合保存到 `openspec/changes/rl-algo-mismatch-correction/baseline-failures.txt`。验证：文件存在，条目数与 pytest 汇总一致。
  - 完成记录（CPU 通过）：基线取 algo-cap 3d1b466 的干净 worktree，共 94 条（68 failed + 26 errors），与 pytest 汇总一致，见 `baseline-failures.txt`。
- [x] 1.2 确认 P0 已提供的接口（`CorrectionSpec`、映射表中的修正参数、`EngineCapabilities.corrections`、`expects_gradient()`、`masked_fraction`、`PluginRef`）在当前分支可用。验证：在 `progress.md` 记录各接口所在文件；缺失项列出并暂停，先向用户报告。
  - 完成记录（已实现）：各接口所在文件见 progress.md“1.2 接口核对”。合入 algo-cap 2f9f02c 后 `TrainStepMetrics.masked_fraction` 也已可用。

## 2. 源码核对与只观测插件（design D2）

- [x] 2.1 在 Miles `9e4260d` 中核对：custom-tis 函数签名与返回值、是否只在 `--use-tis` 时调用、`--get-mismatch-metrics` 输出的指标键全集、vanilla TIS 与 IcePop 的截断/屏蔽行为。验证：结论与行号写入 `progress.md`，并据此关闭 design Open Question 1。
  - 完成记录（已实现）：结论与行号见 progress.md“2.1 源码核对”，Open Question 1 已关闭。
- [x] 2.2 新增 `yeto/rl/algos/mismatch_observe.py`：权重恒为 1、`loss_masks` 原样返回，输出与 Miles 同名的 mismatch 指标。验证：新增 `tests/test_rl_mismatch_observe.py`，在 miles-next-venv 中比对插件指标与 Miles 内置 TIS 函数在相同输入上输出的同名指标（数值一致）。
  - 完成记录（CPU 通过）：`tests/test_rl_mismatch_observe.py::test_observe_metrics_match_vanilla_tis`，`tis`/`tis_abs` 用 `torch.equal` 比对，在 miles-next-venv 中通过。
- [x] 2.3 梯度不变对照：在 CPU 上以相同 logprob、advantage、masks 分别走"无修正"与"只观测"路径调用 Miles 的 losses/corrections，反传。验证：同一测试文件中断言 loss 与梯度 `torch.equal`；在 miles-next-venv 实际运行通过。
  - 完成记录（CPU 通过）：`test_observe_gradient_identical`，共 12 个参数组合（3 seed × 有/无 rollout_mask_sums × 有/无 --use-tis）；调用真实 `policy_loss_function` 并反传，loss 与 logits 梯度均 `torch.equal`。
- [x] 2.4 在 `CorrectionSpec` 中接入只观测模式的校验与翻译（`--custom-tis-function-path <yeto 插件> --get-mismatch-metrics`，按 2.1 结论决定是否附 `--use-tis`），插件哈希进入身份。验证：单测覆盖翻译结果、哈希变化、与其他修正组合被拒；在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv 通过。
  - 完成记录（CPU 通过）：P0 对 custom 函数总是输出 `--use-tis`，CPU 对照已证明加不加 `--use-tis` 梯度都相同。翻译、哈希和拒绝规则见 `tests/test_rl_mismatch_correction.py`，upstream parser 解析见 `test_upstream_parser_accepts_fragment[observe]`。只观测与其他修正的组合由结构和冲突检查拒绝。

## 3. TIS 与 IcePop（design D3/D4/D7）

- [x] 3.1 TIS：要求显式 `tis_clip` 与 `tis_clip_low`，校验有限、下界 ≥ 0 且小于上界，翻译为 `--use-tis --tis-clip H --tis-clip-low L`；核对 `low=0` 时的零梯度判定（design D7）。验证：参数化单测覆盖缺失、倒置、非有限、正常翻译；与 `use_rollout_logprobs` 同开仍被 P0 规则拒绝；upstream `parse_args` 解析通过。
  - 完成记录（CPU 通过）：参数化单测覆盖缺失、倒置、非有限和正常翻译，P0 的 rollout_logprobs 互斥规则仍生效；upstream 解析通过。`low=0` 时权重为 `clamp(ratio,0,H)`，只有 ratio 下溢为 0 才会是 0，因此按默认 GRPO 判定（不屏蔽）。
- [x] 3.2 IcePop：要求显式上下界，翻译为内置 `icepop_function` 路径加 `--tis-clip-low/--tis-clip`，PluginRef 哈希取 Miles 源文件。验证：单测覆盖翻译与哈希；在 miles-next-venv 中 CPU 调用 Miles `icepop_function`，区间 [0.5,5] 内权重等于比率、区间外为 0。
  - 完成记录（CPU 通过）：`test_icepop_weights`：区间 [0.5,5] 内权重等于 ratio，区间外为 0；`ICEPOP_SOURCE_SHA256` 与 Miles 源文件一致。
- [x] 3.3 实现"至多一种修正方式"的拒绝（只观测、TIS、IcePop、MIS 两两互斥；OPSM 可组合，只观测除外）。验证：参数化单测，报错列出冲突项，均在构建 GPU 进程前失败。
  - 复审后完成（CPU 通过）：合入 algo-cap fefcbe0（含 1a-shared.patch）后，`test_opsm_combines_with_tis_but_not_with_observe` 不再 skip，已通过。
  - 状态（已实现，部分）：只观测、TIS、IcePop、MIS 两两互斥，冲突在构建 GPU 进程前失败，报错列出两个 flag。“OPSM 可与其他修正组合”需要 `1a-shared.patch`（P0 的 `method` 是单值），相关测试在未打补丁时 skip。
- [ ] 3.4 IcePop 的 `expects_gradient`：`masked_fraction == 1.0` 时不期望梯度；adapter 从 Miles 指标读取或推导屏蔽比例。验证：fake 测试覆盖全屏蔽不失败、部分屏蔽零梯度失败、屏蔽比例缺失时按期望梯度判定、grad_norm 非有限失败。
  - 状态（已实现 + CPU 通过，adapter 接线未完成）：已有 fake driver 测试 `test_fake_driver_full_mask_round_is_not_a_failure`（全屏蔽不失败；部分屏蔽或缺失时零梯度失败）和 `masked_fraction_from_metrics`。但 trainer 目前只读 `masked_fraction` 键，还没有调用这个 helper，属于 INFRA 需求。

## 4. OPSM（design D5/D7）

- [x] 4.1 `CorrectionSpec` 新增 `opsm_old_logprob_source ∈ {trainer, rollout}`，默认 `trainer`，选 OPSM 时总是写入规范化 JSON；要求显式 δ。验证：单测覆盖两种来源哈希不同、缺 δ 被拒、未选 OPSM 时哈希不受该字段影响。
  - 复审后完成（CPU 通过）：合入 always_emit 后 `test_opsm_source_always_in_canonical_json` 通过。注意：所有已有 OPSM 规格的哈希都因 `opsm_old_logprob_source` 字段而改变，这是 4.1 要求的预期结果。
  - 状态（已实现，部分）：来源字段、两种来源哈希不同、缺 δ 被拒、未选 OPSM 时哈希不变都已实现。“选 OPSM 时总是写入 JSON”需要 `1a-shared.patch` 的 `always_emit`。
- [x] 4.2 翻译：`--use-opsm --opsm-delta δ`，`rollout` 来源附 `--use-rollout-logprobs`，与 TIS 同开时被拒且报错说明该参数同时改变 PPO ratio 的 π_old。验证：单测；upstream `parse_args` 解析通过。
  - 复审后完成（CPU 通过）：`test_opsm_rollout_with_tis_rejected_with_pi_old_note` 通过。
  - 状态（已实现，部分）：翻译和 upstream 解析通过。来源与 use_rollout_logprobs 不一致时拒绝，报错说明它也改变 PPO ratio 的 π_old。与 TIS 同开目前由 P0 的 rollout 互斥规则拒绝；说明 π_old 的专用报错需要 `1a-shared.patch`。
- [x] 4.3 OPSM 数值与判定：CPU 调用 Miles `math_utils.py` 的 OPSM 实现核对屏蔽条件；OPSM 的 `expects_gradient` 按屏蔽类判定。验证：miles-next-venv 中数值测试通过；fake 测试覆盖全屏蔽不失败。
  - 完成记录（CPU 通过）：`test_opsm_mask_condition` 和全屏蔽零梯度数值测试；fake driver 全屏蔽测试通过。注意 Miles 的 `opsm_clipfrac` 不是比例，adapter 推导结果为 None，按严格规则判定。

## 5. MIS / geo-MIS（design D6/D7）

- [x] 5.1 在运行镜像（R0 冒烟所用镜像）中核实 `examples/infra_features/train_infer_mismatch_helper/mis.py` 能否 import，并核实 Miles 仓库许可证。验证：命令与输出写入 `progress.md`，关闭 design Open Question 2。此项只需 CPU 容器，不需要 GPU。
  - 完成记录（已实现）：镜像中 `import examples.infra_features.train_infer_mismatch_helper.mis` 成功，见 `evidence/2026-09-29-g1/five_one.log`；Miles LICENSE 为 Apache-2.0，Open Question 2 已关闭。
- [ ] 5.2 若不能 import：vendor 到 `yeto/rl/algos/vendor/miles_mis.py`，文件头记录来源仓库、commit `9e4260d`、原路径、许可证，只允许改 import。验证：CPU 测试比对副本与 Miles 原文件在相同输入上的输出一致；PluginRef 哈希取副本。能 import 时跳过本项并在 `progress.md` 说明。
  - 复审撤销勾选（未完成，偏离原文）：原文写“能 import 时跳过本项”，而 5.1 已证实可以 import。仍保留 vendor 副本的原因是 P0 插件命名空间只允许 yeto./miles.；是否接受这一偏离需用户批准（见 progress 待批准）。
  - 完成记录（CPU 通过，偏离 D6）：虽然原文件可以 import，但 P0 D7 只允许 `yeto.`/`miles.` 插件命名空间，所以仍用 vendor 副本（待批准，见 progress）。副本正文与原文件逐字相同，输出比对在 18 种组合上 `torch.equal`，PluginRef 取副本哈希。
- [x] 5.3 MIS 字段与翻译：`mis_mode`（截断/屏蔽、几何平均变体，以 `mis.py` 实际提供为准）与显式阈值，翻译为 custom-tis 路径及其参数；屏蔽变体按屏蔽类判定零梯度。验证：单测覆盖翻译、缺阈值被拒、零梯度判定；upstream `parse_args` 解析通过。
  - 完成记录（CPU 通过）：`mis_level`（token/sequence/geometric）、`mis_mode`（truncate/clip/mask）、显式上下界与 `mis_batch_normalize` 已实现。参数通过 `register_runtime_attrs` 写入 Miles namespace；mask 变体按屏蔽类判定；upstream 解析通过。

## 6. 能力声明（fake）与文档

- [x] 6.1 fake engine 声明全部修正机制（OPSM 按来源），Miles adapter 声明保持空集。验证：fake 组合根测试中各机制能启动；同一描述在 Miles adapter 的 `check()` 下被拒，报错列出已支持项。
  - 复审后完成（CPU 通过）：fake.py 已由 ALGO-CAP 合入全部修正声明；fake 组合根测试通过，Miles adapter 拒绝未声明项的测试针对具体集合断言。
  - 状态（未完成）：`fake.py` 由 ALGO-CAP 负责，声明补丁见 `/home/michael/work/infra-drafts/1a-declare.patch`。目前测试通过覆盖 `fake_capabilities(corrections=…)` 实现各机制启动；Miles adapter 拒绝未声明项的行为已测。
- [x] 6.2 默认不变回归。验证：`tests/test_rl_argv_snapshot.py`、`tests/test_rl_miles_adapter_config.py`、`tests/test_rl_engine_algorithm.py` 不改即通过；legacy argv 快照逐字节一致。
  - 完成记录（CPU 通过）：三个文件本 change 未修改（只有合入 P0/rl-integ 带来的改动），49 passed、2 skipped。
- [x] 6.3 更新 `docs/MILES_RL.md`：每个修正的用法与示例描述、阈值含义（论文取值标注来源与"未核实"）、OPSM 的 π_old 来源与 `--use-rollout-logprobs` 的影响范围、MIS 来源（原路径或 vendor）、每项当前验证层级，不写效果收益。验证：文档中的示例描述用 `--dry-run` 在 fake 下执行，结果与描述一致。
  - 完成记录（CPU 通过）：`docs/MILES_RL.md` 新增“Train/inference mismatch corrections”一节；6 个示例由 `test_doc_example_dry_run` 用 P0 的 `--dry-run` 执行（未声明时拒绝，带放行时接受，argv 与哈希一致）。

## 7. GPU 验证（需用户批准卡数与预算后执行）

> 本组的 G1（1 卡冒烟）统一使用 P0 提供的 `--rl-allow-unverified-mechanism <机制名>` 放行（见 `rl-algorithm-capabilities` design D11），只在单岛运行中生效；G1 通过后再在 adapter 中正式声明支持；两岛 G3 只用正式声明，不带放行参数。

- [x] 7.1 准备：复用 R0 冒烟小模型与 harness，证据目录 `openspec/changes/rl-algo-mismatch-correction/evidence/<日期>-<名称>/`，含 `YETO_SHA`、argv、事件与指标 jsonl。Modal 使用 `H100!:N` 并在启动时断言 GPU 名；日志与证据中不打印凭据。验证：dry-run 输出的资源请求为 1 卡；凭据扫描（grep token/key 模式）无命中。
  - 补做完成（已实现）：`yeto launch --rl-single-island-no-sync --controller local --dry-run`（algo-cap 319d974）的输出为 `total_gpus: 1`、`islands: 1`、`syncer: null`、`outer_sync: false`，并列出 unverified_mechanisms，见 `evidence/2026-09-29-dryrun/dryrun.log`（凭据扫描无命中）。
  - （已被下方“补做完成”取代）复审撤销勾选（未完成）：原文要求 launcher dry-run 输出资源请求为 1 卡，但当前 `yeto launch` 没有 `--dry-run` 选项（`launch --help` 中无此项），无法按原文补做；已作为接口需求上报。
  - 完成记录（已实现）：计划与 harness 见 `evidence/2026-09-29-g1/plan.md` 和 `evidence/2026-09-29-g1b/plan.md`；Modal `H100!`×1，启动前断言 GPU 名称（`runs/gpu_name.txt`）；凭据扫描无命中。注：资源请求是 sandbox 的 `gpu="H100!"`，没有经过 launcher 的 dry-run。
- [x] 7.2 G1（1 卡）：只观测、TIS、IcePop、OPSM(trainer)、MIS 各 2–3 轮；OPSM(rollout) 可选。验证：每项指标键存在且有限、不变量无误报、policy token 与 receipt 正常；结果逐项写入 `progress.md`。
  - 完成记录（GPU 验收通过，G1）：Modal H100 单卡，单岛、无 syncer，每项 3 轮，预先声明的判据全部通过。通过的项有 tis、icepop、opsm_trainer（`evidence/2026-09-29-g1/runs/*/check.json`，YETO_SHA bc7a330）和 mismatch_observe、mis、mis_mask（`evidence/2026-09-29-g1b/runs/*/check.json`，YETO_SHA 3b6dfa9；round 2 另验 `rl/outer_sync=false`）。observe 第一次尝试在 harness 下载数据时因网络失败（Miles 未启动）；修复为预下载后重跑一次。opsm_rollout（可选）未跑。launcher 入口 `yeto launch --rl-single-island-no-sync` 未经实跑。
  - 复审补记：round 1（tis/icepop/opsm-trainer）用的是 bc7a330 代码和裸名放行，没有判据 5，入口也不是 launcher，所以不能作为声明依据，改由 launcher 入口按新计划重新验证（见 `evidence/2026-09-29-g1c/`）。原计划没有显式的“receipt 正常”判据；receipt 只由 driver 失败即报事件间接覆盖，没有显式判据，不据此宣告 receipt 通过。GPU 上所有截断/屏蔽分支都未触发（clipfrac 与 mask fraction 均为 0）；每轮一步时 OPSM 结构上不可能触发；ess_ratio/ois 每轮一步恒为 1，不反映训推差异。上传时刻：round 1 的 yeto 代码在 16:4x 以 `git archive bc7a330` 上传（observe 第一次尝试之前重新上传过一次），round 2 在 17:09 后以 `git archive 3b6dfa9` 上传，harness 取自 c8ec241。精确时刻没有另行记录。
  - 入口复验（`evidence/2026-09-29-g1c/`）：tis、icepop、opsm-trainer 走 `yeto launch --rl-single-island-no-sync --controller local` 真实入口复验，判据 1-6 全部通过（判据 6 为显式的 receipt 统计判据）。另有生效验证（`evidence/2026-09-29-trigger/`）：在事先固定的小阈值下，tis、icepop、mis_mask 的截断/屏蔽比例均 >0，OPSM 在 2 步/轮时 opsm_clipfrac >0，均通过。
  - 注：g1c 中 icepop 与 tis 在自然训推差异下数值逐位相同（没有 token 越界），所以生效以 trigger 运行为准。check.json 中 `rc0` 键的含义是“退出码可接受”（0，或 2 且同时打印不可取回的说明），新运行将改名为 `rc_ok`。
- [ ] 7.3 对 G1 通过的每项，在 Miles adapter 的 `EngineCapabilities.corrections` 中加入声明（每项单独变更）。验证：adapter 的 `check()` 单测接受已声明项、仍拒绝未通过项；`progress.md` 引用对应证据目录。
  - 进展（更正）：Miles adapter 声明了 {none, tis, opsm, opsm_trainer}。其中 `opsm` 是内置的“使用了 OPSM（设置了 opsm_delta）”维度，任何 OPSM 规格都要求它，它本身不放行任何来源；来源由 `opsm_trainer` 或 `opsm_rollout` 各自决定，`opsm_rollout` 未声明，也没有 G1 或生效验证。opsm_trainer 的生效证据（trigger 中 opsm_clipfrac>0）同时覆盖 `opsm` 这一维度在 trainer 来源下的代码路径。tis 的生效也已验证。mismatch_observe、icepop、mis、mis_mask 尚未声明：icepop 与 mis_mask 由集成分支统一处理（icepop 需要 corrections:custom 粒度）；mis（截断/裁剪变体）若要声明，需先补一次事先提交计划的触发验证。“对 G1 通过的每项声明”尚未全部满足，保持未勾选。
  - 状态（未完成）：`entry.py` 由 ALGO-CAP 负责。G1 已通过的 tis、opsm_trainer 的声明补丁在 `1a-declare.patch`；observe、icepop、mis、mis_mask 需要先合入 `1a-shared.patch`（custom 粒度），否则声明 custom 会放开任意函数。
- [x] 7.4 G2（1 卡）：只观测约 20 轮，产出报告（每轮 `train_rollout_kl`、`tis_abs` 分位数、`ess_ratio`、[0.5,5] 区间外 token 比例），注明模型与配置、不外推。验证：报告文件存在于证据目录，数据可由指标 jsonl 重新生成；是否推荐默认开启交用户决定，不在本 change 内修改默认。
  - 完成记录（GPU 验收通过，G2）：只观测 20 轮，报告见 `evidence/2026-09-29-g1b/runs/g2-observe/report.md`，可由 `metrics.jsonl`/`miles.log` 用 `harness/report_g2.py` 重新生成。报告注明 colocated-serial / CUDA IPC、模型与配置，并声明不外推；不给推荐，默认值不改。
  - 未达成的设计意图：design D2/G2 把 `ess_ratio` 列为训推差异指标，但 Miles 的 `ess_ratio` 是 π_θ/π_old，每轮一步时恒为 1，不反映训推差异。报告中已注明，这一点记为未达成意图。
- [ ] 7.5 G3（1+1 卡）：两岛 strict-avg，TIS 与 IcePop 各一次，约 3 轮。验证：两岛算法哈希一致、外层同步后权重 hash 一致、不变量无失败。不做 decoupled 对比（须等 `fix-decoupled-lr-schedule` 合入）。
  - G3 已运行一次（`evidence/2026-09-29-g3/results.md`），**未通过**：判据 1、2、5 通过，判据 3、4 无法成立，原因是 island-1 的磁带拉取截断（v2/v3 缺失），不是机制失败。重跑需要先提交 harness 修复（改用 launcher 回传磁带）。
  - （更正：已被下一行取代）早先记录的“G3 两岛这一轮没有运行”已不成立：G3 后来跑过一次，但未通过。
- [x] 7.6 拆除与费用：每次运行后拆除全部资源。验证：provider 侧列出 app/实例/卷为空的输出存入证据目录（"无残留"证明）；按运行汇总卡时与费用写入 `progress.md`。
  - 完成记录（已实现）：两个 sandbox 均已 terminate，`algo1a-g1`/`algo1a-g1b` 均为 stopped、0 tasks（`*/modal_app_list_after_stop.txt`）；没有卷，没有命名 secret；watchdog 已停止。卡时约 0.83 H100·h，估算约 $4（未经账单确认），明细见 progress.md。
- [ ] 7.7 （可选，需另行申请预算）G4 效果 A/B：不在本 change 验收范围内，仅在用户批准后记录方案。

## 8. 集成检查

- [x] 8.1 全量 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败集合与 1.1 相同。验证：集合 diff 为空，结果写入 `progress.md`。
  - 完成记录（CPU 通过）：合入后全量运行 68 failed + 26 errors；按测试 id 与 1.1 基线（94 条）比对，diff 为空，见 `evidence/2026-09-29-cpu/`。
- [x] 8.2 `openspec validate rl-algo-mismatch-correction --strict` 通过。验证：命令输出无错误。
  - 完成记录：`openspec validate rl-algo-mismatch-correction --strict` 的输出为 valid。
- [x] 8.3 `progress.md` 逐项列出任务状态，严格区分"已实现""CPU 测试通过""GPU 验证通过（G1/G2/G3）"，并列出仍为 ⚙ 的机制。验证：文件存在且与任务逐项对应。
  - 完成记录：progress.md 逐项列出状态，并列出仍为 ⚙ 的机制。
