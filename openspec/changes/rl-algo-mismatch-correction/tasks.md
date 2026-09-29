约定：
- 测试命令为 `/tmp/yeto-venv/bin/python -m pytest -q`；全量测试以"改动前后失败集合相同"为准。
- 依赖 Miles 源码的 CPU 测试与 upstream 参数解析在 `/home/michael/work/miles-next-venv` 中运行，miles 不可用时 skip；任务说明中记录命令与结果。
- 不修改 legacy `build_miles_argv`，不修改 Miles/SGLang fork，默认 GRPO argv 与哈希不变。
- 状态严格区分"已实现""CPU 测试通过""GPU 验证通过"；fake 测试不能替代 GPU 验收。
- commit 和 push 需要用户确认。

## 1. 基线

- [ ] 1.1 改动前运行 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败与错误集合保存到 `openspec/changes/rl-algo-mismatch-correction/baseline-failures.txt`。验证：文件存在，条目数与 pytest 汇总一致。
- [ ] 1.2 确认 P0 已提供的接口（`CorrectionSpec`、映射表中的修正参数、`EngineCapabilities.corrections`、`expects_gradient()`、`masked_fraction`、`PluginRef`）在当前分支可用。验证：在 `progress.md` 记录各接口所在文件；缺失项列出并暂停，先向用户报告。

## 2. 源码核对与只观测插件（design D2）

- [ ] 2.1 在 Miles `9e4260d` 中核对：custom-tis 函数签名与返回值、是否只在 `--use-tis` 时调用、`--get-mismatch-metrics` 输出的指标键全集、vanilla TIS 与 IcePop 的截断/屏蔽行为。验证：结论与行号写入 `progress.md`，并据此关闭 design Open Question 1。
- [ ] 2.2 新增 `yeto/rl/algos/mismatch_observe.py`：权重恒为 1、`loss_masks` 原样返回，输出与 Miles 同名的 mismatch 指标。验证：新增 `tests/test_rl_mismatch_observe.py`，在 miles-next-venv 中比对插件指标与 Miles 内置 TIS 函数在相同输入上输出的同名指标（数值一致）。
- [ ] 2.3 梯度不变对照：在 CPU 上以相同 logprob、advantage、masks 分别走"无修正"与"只观测"路径调用 Miles 的 losses/corrections，反传。验证：同一测试文件中断言 loss 与梯度 `torch.equal`；在 miles-next-venv 实际运行通过。
- [ ] 2.4 在 `CorrectionSpec` 中接入只观测模式的校验与翻译（`--custom-tis-function-path <yeto 插件> --get-mismatch-metrics`，按 2.1 结论决定是否附 `--use-tis`），插件哈希进入身份。验证：单测覆盖翻译结果、哈希变化、与其他修正组合被拒；在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv 通过。

## 3. TIS 与 IcePop（design D3/D4/D7）

- [ ] 3.1 TIS：要求显式 `tis_clip` 与 `tis_clip_low`，校验有限、下界 ≥ 0 且小于上界，翻译为 `--use-tis --tis-clip H --tis-clip-low L`；核对 `low=0` 时的零梯度判定（design D7）。验证：参数化单测覆盖缺失、倒置、非有限、正常翻译；与 `use_rollout_logprobs` 同开仍被 P0 规则拒绝；upstream `parse_args` 解析通过。
- [ ] 3.2 IcePop：要求显式上下界，翻译为内置 `icepop_function` 路径加 `--tis-clip-low/--tis-clip`，PluginRef 哈希取 Miles 源文件。验证：单测覆盖翻译与哈希；在 miles-next-venv 中 CPU 调用 Miles `icepop_function`，区间 [0.5,5] 内权重等于比率、区间外为 0。
- [ ] 3.3 实现"至多一种修正方式"的拒绝（只观测、TIS、IcePop、MIS 两两互斥；OPSM 可组合，只观测除外）。验证：参数化单测，报错列出冲突项，均在构建 GPU 进程前失败。
- [ ] 3.4 IcePop 的 `expects_gradient`：`masked_fraction == 1.0` 时不期望梯度；adapter 从 Miles 指标读取或推导屏蔽比例。验证：fake 测试覆盖全屏蔽不失败、部分屏蔽零梯度失败、屏蔽比例缺失时按期望梯度判定、grad_norm 非有限失败。

## 4. OPSM（design D5/D7）

- [ ] 4.1 `CorrectionSpec` 新增 `opsm_old_logprob_source ∈ {trainer, rollout}`，默认 `trainer`，选 OPSM 时总是写入规范化 JSON；要求显式 δ。验证：单测覆盖两种来源哈希不同、缺 δ 被拒、未选 OPSM 时哈希不受该字段影响。
- [ ] 4.2 翻译：`--use-opsm --opsm-delta δ`，`rollout` 来源附 `--use-rollout-logprobs`，与 TIS 同开时被拒且报错说明该参数同时改变 PPO ratio 的 π_old。验证：单测；upstream `parse_args` 解析通过。
- [ ] 4.3 OPSM 数值与判定：CPU 调用 Miles `math_utils.py` 的 OPSM 实现核对屏蔽条件；OPSM 的 `expects_gradient` 按屏蔽类判定。验证：miles-next-venv 中数值测试通过；fake 测试覆盖全屏蔽不失败。

## 5. MIS / geo-MIS（design D6/D7）

- [ ] 5.1 在运行镜像（R0 冒烟所用镜像）中核实 `examples/infra_features/train_infer_mismatch_helper/mis.py` 能否 import，并核实 Miles 仓库许可证。验证：命令与输出写入 `progress.md`，关闭 design Open Question 2。此项只需 CPU 容器，不需要 GPU。
- [ ] 5.2 若不能 import：vendor 到 `yeto/rl/algos/vendor/miles_mis.py`，文件头记录来源仓库、commit `9e4260d`、原路径、许可证，只允许改 import。验证：CPU 测试比对副本与 Miles 原文件在相同输入上的输出一致；PluginRef 哈希取副本。能 import 时跳过本项并在 `progress.md` 说明。
- [ ] 5.3 MIS 字段与翻译：`mis_mode`（截断/屏蔽、几何平均变体，以 `mis.py` 实际提供为准）与显式阈值，翻译为 custom-tis 路径及其参数；屏蔽变体按屏蔽类判定零梯度。验证：单测覆盖翻译、缺阈值被拒、零梯度判定；upstream `parse_args` 解析通过。

## 6. 能力声明（fake）与文档

- [ ] 6.1 fake engine 声明全部修正机制（OPSM 按来源），Miles adapter 声明保持空集。验证：fake 组合根测试中各机制能启动；同一描述在 Miles adapter 的 `check()` 下被拒，报错列出已支持项。
- [ ] 6.2 默认不变回归。验证：`tests/test_rl_argv_snapshot.py`、`tests/test_rl_miles_adapter_config.py`、`tests/test_rl_engine_algorithm.py` 不改即通过；legacy argv 快照逐字节一致。
- [ ] 6.3 更新 `docs/MILES_RL.md`：每个修正的用法与示例描述、阈值含义（论文取值标注来源与"未核实"）、OPSM 的 π_old 来源与 `--use-rollout-logprobs` 的影响范围、MIS 来源（原路径或 vendor）、每项当前验证层级，不写效果收益。验证：文档中的示例描述用 `--dry-run` 在 fake 下执行，结果与描述一致。

## 7. GPU 验证（需用户批准卡数与预算后执行）

> 本组的 G1（1 卡冒烟）统一使用 P0 提供的 `--rl-allow-unverified-mechanism <机制名>` 放行（见 `rl-algorithm-capabilities` design D11），只在单岛运行中生效；G1 通过后再在 adapter 中正式声明支持；两岛 G3 只用正式声明，不带放行参数。

- [ ] 7.1 准备：复用 R0 冒烟小模型与 harness，证据目录 `openspec/changes/rl-algo-mismatch-correction/evidence/<日期>-<名称>/`，含 `YETO_SHA`、argv、事件与指标 jsonl。Modal 使用 `H100!:N` 并在启动时断言 GPU 名；日志与证据中不打印凭据。验证：dry-run 输出的资源请求为 1 卡；凭据扫描（grep token/key 模式）无命中。
- [ ] 7.2 G1（1 卡）：只观测、TIS、IcePop、OPSM(trainer)、MIS 各 2–3 轮；OPSM(rollout) 可选。验证：每项指标键存在且有限、不变量无误报、policy token 与 receipt 正常；结果逐项写入 `progress.md`。
- [ ] 7.3 对 G1 通过的每项，在 Miles adapter 的 `EngineCapabilities.corrections` 中加入声明（每项单独变更）。验证：adapter 的 `check()` 单测接受已声明项、仍拒绝未通过项；`progress.md` 引用对应证据目录。
- [ ] 7.4 G2（1 卡）：只观测约 20 轮，产出报告（每轮 `train_rollout_kl`、`tis_abs` 分位数、`ess_ratio`、[0.5,5] 区间外 token 比例），注明模型与配置、不外推。验证：报告文件存在于证据目录，数据可由指标 jsonl 重新生成；是否推荐默认开启交用户决定，不在本 change 内修改默认。
- [ ] 7.5 G3（1+1 卡）：两岛 strict-avg，TIS 与 IcePop 各一次，约 3 轮。验证：两岛算法哈希一致、外层同步后权重 hash 一致、不变量无失败。不做 decoupled 对比（须等 `fix-decoupled-lr-schedule` 合入）。
- [ ] 7.6 拆除与费用：每次运行后拆除全部资源。验证：provider 侧列出 app/实例/卷为空的输出存入证据目录（"无残留"证明）；按运行汇总卡时与费用写入 `progress.md`。
- [ ] 7.7 （可选，需另行申请预算）G4 效果 A/B：不在本 change 验收范围内，仅在用户批准后记录方案。

## 8. 集成检查

- [ ] 8.1 全量 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败集合与 1.1 相同。验证：集合 diff 为空，结果写入 `progress.md`。
- [ ] 8.2 `openspec validate rl-algo-mismatch-correction --strict` 通过。验证：命令输出无错误。
- [ ] 8.3 `progress.md` 逐项列出任务状态，严格区分"已实现""CPU 测试通过""GPU 验证通过（G1/G2/G3）"，并列出仍为 ⚙ 的机制。验证：文件存在且与任务逐项对应。
