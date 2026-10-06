# Tasks

执行约定：
- 规划文档以 `rl-infra-spec` 分支上的副本为准；实现 worktree、分支与可修改路径按 `../rl-infra-spec/alignment.md` 的工作包派发（原 `/home/michael/work/rl-algos` 分支 `rl-algorithms` 不再作为规划来源），测试命令为 `/tmp/yeto-venv/bin/python -m pytest -q`，upstream 参数解析在 `/home/michael/work/miles-next-venv` 中运行。
- 不改 legacy `build_miles_argv`，默认 GRPO 不变。
- fork 改动只能提交到 `michaellchung/miles` 的 `yeto/ports` 分支，且必须先得到用户同意；绝不向 radixark/miles 或 sgl-project/sglang 提 PR。
- commit 与 push 需要用户确认。

## 1. 决策门与基线

- [x] 1.1 在 pin 的 Miles 提交（`9e4260d` 或当前 `MILES_NEXT_COMMIT`）上重新核对 design Context 中的事实：`compute_policy_loss` 的导入方式、`policy_loss_function` 的结构、是否已有 `policy_objective` 分支、batch 白名单。验证：把核对结果（含文件行号）写入 `openspec/changes/rl-algo-loss-variants/progress.md`。
- [x] 1.2 向用户呈交 design D1 的路线对比与推荐（路线 B），由用户选择路线 A 或 B；如果选 B，还要确认用户同意向 `michaellchung/miles` `yeto/ports` 提交。验证：在 progress.md 中记录用户决定的原文和日期；决定之前不开始第 3 组及以后的实现任务。
- [x] 1.3 记录 pytest 失败基线：`/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败集合保存到 `openspec/changes/rl-algo-loss-variants/baseline-failures.txt`。验证：文件存在，条目数与 pytest 汇总一致。

## 2. 参考公式与 spec 字段（与路线无关）

- [x] 2.1 新建 `tests/rl_loss_variant_reference.py`，用独立的 torch 小张量写出 CISPO、SAPO、GMPO 的参考实现（design D3），在注释中写明论文出处和公式编号，且不 import 被测代码。验证：自检用例用手算值核对 3–5 个元素。
- [x] 2.2 在 `yeto/rl/engine/algorithm.py` 的 `loss` 组中加入变体参数（design D2）：校验、默认值、"只在匹配的 variant 下进入规范化"、不匹配时拒绝。验证：新增 `tests/test_rl_loss_variants_spec.py`，覆盖默认哈希不变（P0 golden 用例不改就能通过）、每个变体的哈希不同、非法 τ/δ 报错并指明字段、参数与 variant 不匹配被拒。
- [x] 2.3 实现拒绝规则（design D4）：变体与 GSPO 组合、变体与 dual-clip 组合。验证：参数化单测，都在 fake 组合根中、在创建 GPU 进程之前失败，报错中给出替代方案。
- [x] 2.4 实现各变体的 `expects_gradient` 判定（design D5）。验证：单测覆盖 CISPO 全部越界仍期望梯度、SAPO 同 CISPO、GMPO 在 clip 比例为 1 时允许零梯度、GMPO 读不到 clip 比例时退回 GRPO 判定、任何变体下 grad_norm 非有限都判失败。
- [x] 2.5 保持 adapter 的 `losses` 声明不变（三个变体仍是"可表达未开放"）。验证：fake 测试确认选用变体会被拒，报错为"可表达未开放"。

## 3. 路线 A：yeto custom loss 插件（仅当选择路线 A 时执行）

> 2026-10-06 复核：用户 2026-09-30 已选路线 B（progress.md 1.2），且 D1 判定路线 A 违反"不重复实现引擎"原则。路线 B 的变体已在 fork 5c1b49eb 实现、pin 与镜像已更新（4.3/4.4），再实现路线 A 会造成同一语义两份实现与两种身份（fork 提交号 vs PluginRef 哈希）。故 3.1–3.4 标为**不适用（N/A，路线未选）**，不实现；若将来改选路线 A，按原文执行。

- [ ] 3.1 新建 `yeto/rl/engine/miles_adapter/loss_variants.py`，在完整复刻 pin 提交上 `policy_loss_function` 的前提下，把 pg_loss 的计算替换为变体，并尽量复用 Miles 的公共函数（`get_log_probs_and_entropy`、`get_sum_of_sample_mean`、`corrections` 等）。**不得使用任何 monkeypatch。** 验证：新增静态检查单测，用 AST 扫描该文件，确认没有对任何 `miles` 模块属性赋值或调用 `setattr`。
- [ ] 3.2 增加守护测试：记录 pin 提交上 `policy_loss_function` 的源码 SHA256，与 miles-next-venv 中实际源码比较，不一致时失败并提示人工 diff。验证：该测试在 miles-next-venv 中通过；手动改一个字节时失败。
- [ ] 3.3 在 miles-next-venv 中，用 CPU 小张量（CP=1，并用模拟切分覆盖 CP>1）比对插件输出与参考实现，覆盖正负优势、ratio 越界、全 mask，以及与 TIS、IcePop、KL loss、entropy 的组合；同时比对 variant=policy_loss 时插件与原 `policy_loss_function` 逐元素一致。验证：新增 `tests/test_rl_loss_variants_plugin.py`，miles 不可用时 skip，在 miles-next-venv 中实际运行通过，在任务说明里记录命令和结果。
- [ ] 3.4 翻译：variant≠policy_loss 时生成 `--loss-type custom_loss --custom-loss-function-path ...`，插件以 PluginRef 源码哈希进入算法身份。验证：翻译单测；在 miles-next-venv 中用 upstream `parse_args` 解析生成的 argv 通过。

## 4. 路线 B：fork variant 分支（仅当选择路线 B 且用户已同意 fork 提交时执行）

- [x] 4.1 在 `michaellchung/miles` 的 `yeto/ports` 上实现：`math_utils.py` 增加 CISPO、SAPO 的逐 token 函数和 GMPO 的序列级函数（CP 全收集方式仿照 `compute_gspo_kl`）；`losses.py` 在 pg_loss 处按 `--policy-loss-variant` 分支；`arguments.py` 增加 variant 与变体参数，默认值保持原行为。验证：fork 上的 diff 只涉及这三个文件和测试；默认参数下原有 fork 测试全部通过。
- [x] 4.2 在 fork 上补 CPU 测试：逐元素比对参考公式（正负优势、越界、全 mask、GMPO 模拟 CP 切分），以及与 TIS、IcePop 的组合；验证 variant 缺省时 `compute_policy_loss` 路径逐元素不变。验证：fork 测试命令和结果记录在 progress.md。
- [x] 4.3 经用户确认后，把提交 push 到 `michaellchung/miles` `yeto/ports`（不开任何 PR），更新 yeto 中的 `MILES_NEXT_COMMIT` pin，并按现有流程重建镜像。验证：pin 指向新提交；镜像 tag 与 digest 记录在 progress.md；`git remote -v` 与 push 目标只包含 michaellchung/miles。
- [x] 4.4 yeto 侧翻译：在映射表中登记 `--policy-loss-variant` 与变体参数（含吸收与冲突检测），并把 fork 提交号写入来源记录。验证：翻译、吸收、冲突单测；在 miles-next-venv（安装新 pin）中用 upstream `parse_args` 解析生成的 argv 通过；来源记录测试确认包含提交号。

### 完成记录（4.1/4.2，LOSS-VAR，2026-10-06）
- 4.1：fork `michaellchung/miles` 提交 540650723 → c45155776 → 5c1b49eb（已在 origin `yeto/ports` 上，本地 `yeto/ports` 03947150 是其祖先；本轮无新 fork 提交、无 push）。`git diff --stat 0af62f4d 5c1b49eb` 只涉及 `loss_hub/losses.py`、`loss_hub/math_utils.py`、`utils/arguments.py` 与 `tests/fast/backends/training_utils/loss/test_policy_loss_variants.py`（+677/−4）。
- 4.2：在 5c1b49eb 的 detached worktree 中，`PYTHONPATH=<wt> /home/michael/work/miles-next-venv/bin/python -m pytest -q tests/fast/backends/training_utils/loss/test_policy_loss_variants.py` → 37 passed（覆盖正负优势、越界、全 mask、GMPO 模拟 CP 切分、TIS/IcePop 组合、缺省 variant 逐元素不变）。整个 `tests/fast/backends/training_utils/loss` 目录：88 passed / 13 failed，13 个失败在 0af62f4d 上完全相同（miles-next-venv 缺 `megatron.core`），新增失败 0。

## 5. 开放声明与文档（路线实现完成后执行）

- [x] 5.1 fake engine 声明 `cispo`、`sapo`、`gmpo`，供 CPU 测试使用；Miles adapter 暂不声明，第 6 组 G1 使用 P0 的 `--rl-allow-unverified-mechanism` 放行。验证：单测确认 adapter 未放行时拒绝、单岛放行时可以启动；默认 GRPO 不受影响（`tests/test_rl_argv_snapshot.py` 不改就能通过）。
- [x] 5.2 更新 `docs/MILES_RL.md`：写明变体的公式、参数、默认值、拒绝规则、所选路线及其升级注意事项（路线 A 的守护测试，或路线 B 的 pin 更新），并注明外层同步的正交性和"未做效果 A/B"。验证：文档中的示例命令用 `--dry-run` 执行，结果与描述一致。

## 6. GPU 验证（需用户批准卡数与预算后执行）

> 本组的 G1（1 卡冒烟）统一使用 P0 提供的 `--rl-allow-unverified-mechanism <机制名>` 放行（见 `rl-algorithm-capabilities` design D11），只在单岛运行中生效；G1 通过后再在 adapter 中正式声明支持；两岛 G3 只用正式声明，不带放行参数。

- [ ] 6.1 （计划已更新，待用户批准机型/卡数/预算）向用户提交计划：三个变体各 1 卡冒烟 2–3 轮，CISPO 两岛 strict-avg 1+1 卡 2–3 轮，同时给出预计时长与费用。Modal 使用 `H100!:N`。验证：在 progress.md 中记录用户批准的卡数与预算。
- [ ] 6.2 每个变体做 1 卡冒烟 2–3 轮：先断言 `nvidia-smi --query-gpu=name` 为 H100，记录 driver；检查 loss、grad_norm 有限，`expects_gradient` 判定无误报，来源记录包含变体参数和插件哈希或 fork 提交号。日志中不打印凭据。验证：每个变体的运行 ID、指标摘要和 GPU 名记录在 progress.md；全部通过后在 Miles adapter 中正式声明该变体，把它改为"GPU 验证通过（冒烟）"。
- [ ] 6.3 CISPO 两岛 strict-avg（1+1 卡）2–3 轮：两岛算法哈希一致，外层平均后的权重哈希在两岛一致。验证：哈希比对结果写入 progress.md。
- [ ] 6.4 拆除所有实例与卷，证明没有残留（Modal app 列表、sky status 等为空），并报告实际费用。验证：无残留的命令输出和费用写入 progress.md。

> 6.1 上卡计划草案（2026-10-06，估算未经实测）：
> - G1：CISPO、SAPO、GMPO 各 1×H100 冒烟 3 轮（Qwen2.5-0.5B LoRA，单岛 colocated-serial，CP=1，`--rl-allow-unverified-mechanism losses:<v>`，镜像 5c1b49e-9f29303）；可在同一容器内串行跑三个变体以省冷启动。估时：冷启动+加载 ~10 min + 每变体 ~8 min → ~35 min。
> - G3：CISPO 两岛 strict-avg 1+1×H100 3 轮（需先按 6.2 正式声明 cispo）；估 ~30 min × 2 卡。
> - 费用估算（Modal H100 约 $4/卡·时）：G1 ≈ $2.5，G3 ≈ $4，合计 ≈ $7，建议上限 $15（含失败重试 1 次）。终态探针放容器内，FAILED 风险高的用例放链尾。
> - 机型授权：用户当前授权的机型只有 8 卡 L40S/H200/H100；本计划的 1 卡（H100!:1 / H100!:2）需用户另行批准。若只能用 8 卡机，则把 G1 三变体与 G3 两岛合并到同一台 8 卡机分卡并行，以单次开机时长计费。
> - GMPO CP>1 仍标"CPU 测试通过，未经 GPU 验证"。

## 7. 集成检查

- [x] 7.1 跑全量 `/tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`，失败集合与 1.3 的基线相同。验证：两次集合的 diff 为空，结果写入 progress.md。
- [x] 7.2 `openspec validate rl-algo-loss-variants --strict` 通过。验证：命令输出无错误。
- [x] 7.3 在 progress.md 中逐项列出任务状态，严格区分"已实现""CPU 测试通过""GPU 验证通过"，并写明所选路线、未执行的路线分支组，以及未经 GPU 验证的项（如 GMPO 在 CP>1 下）。验证：文件存在且与任务逐项对应。

## 完成记录（ALGO-2b，2026-09-30，分支 algo-2b）

- 1.1 / 1.2 / 1.3 / 2.2–2.5 / 5.1 / 7.1–7.3：核对结果、证据与命令见 `progress.md` 的 2026-09-30 ALGO-2b 条目。
- 未勾选：2.1（参考实现与手算自检已完成，但离线无法核对论文公式编号，原文要求写明公式编号）；第 3 组（路线 A，未选）；4.1/4.2（FORK-2b 负责）；4.3/4.4（需 fork 审查、快进 yeto/ports、IMG 更新 pin 后才能做）；5.2（文档与 dry-run 已写并核对，但所选路线的 pin/提交号尚未落地）；第 6 组（GPU 暂停，见 progress.md 的"待本地 GPU 验证"计划）。
- 5.1 注：仅在能力检查层面放行（fake 声明、`--rl-allow-unverified-mechanism` 单岛放行通过 capability check）；真实 Miles 启动仍被 `[loss_variants]` pin 检查阻止，直到 `FORK_COMMITS` 含钉住的提交。

### 完成记录（ALGO-2b 第三轮，2026-09-30，基于 integ-decl cbf3d22，pin = fork 5c1b49eb）
- 2.1：三方都已核对。论文 CISPO（arXiv:2506.13585，式 4–5）、SAPO（arXiv:2511.20347v2，式 5–6）、GMPO（arXiv:2507.20673v3，式 4），逐式记录在 `tests/rl_loss_variant_reference.py` 注释中，差异也写在那里；GMPO 由主 agent 按官方代码 callsys/GMPO `train_zero_math_gmpo.py:675-688` 逐行核对；fork 5c1b49eb 与参考实现一致。手算自检见 `tests/test_rl_loss_variants_spec.py::test_reference_*`。
- 4.3：pin `MILES_NEXT_COMMIT` = 5c1b49eb（IMG，cbf3d22）；镜像 tag 5c1b49e-9f29303，digest sha256:17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef；fork 仓库 `git remote -v` 只有 michaellchung/miles，`yeto/ports` = 5c1b49eb；未开任何 PR（用户 2026-09-30 同意路线 B 与 fork 提交）。
- 5.2：`docs/MILES_RL.md` 的 "Policy-loss variants" 小节；三条示例命令按顺序执行，输出与注释一致（`evidence/2026-09-30-dryrun/dryrun.json`）。
- 4.4 未勾：yeto 侧映射、吸收、冲突与来源记录（miles_commit=5c1b49eb）的单测都已通过；在 miles-next-venv 中以镜像内的 /root/miles（5c1b49eb）运行了 Miles 自己的参数 provider + `validate_policy_loss_variant_args`，全部通过（`evidence/2026-09-30-parse-5c1b49e/`）。但没有跑完整的 upstream `parse_args`（其中 Megatron 那一半需要 CUDA 容器，本机无 docker、无 GPU），不满足原文，由主 agent 决定是否按 §7b.7 的口径另行补跑。

### 完成记录（4.4，ALGO-2b-T4，2026-09-30）
- 在钉住镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:17d428a2…`（/root/miles HEAD = 5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba，Megatron = 镜像自带 /root/Megatron-LM）的 Modal T4 容器内，对 yeto `translate_run_config` 为 CISPO、SAPO、GMPO 生成的完整 argv（原文见 result.json）运行完整 upstream `parse_args`（Miles + Megatron 两半，含 Miles validate_args）+ yeto `validate_parsed_args`：三者通过，解析值与期望精确相等。应拒绝组合全部被拒：yeto 侧 3 项（gmpo+token、cispo+default 聚合、cispo 缺 eps），Miles 完整 parse_args 侧 5 项（GMPO+`--calculate-per-token-loss`、SAPO+`--advantage-estimator gspo`、CISPO+`--eps-clip-c`、`--sapo-tau-pos 0`、`--gmpo-log-clip-low -0.1`，均为 fork 的 AssertionError）。
- 环境差异（按 rl-infra-spec alignment §7b 第 7 条口径）：原文写 "miles-next-venv（安装新 pin）"，该 venv 缺 `megatron.training`；改在钉住镜像中跑完整 parse_args，比原文更严格，视为满足原文意图。翻译、吸收、冲突与来源记录（miles_commit=5c1b49eb）单测见前一条完成记录（本轮复跑 `tests/test_rl_loss_variants_spec.py` 100 passed）。
- 证据：`evidence/2026-09-30-t4-parse/`（plan.md、result.json、run3.log、teardown_proof.txt、billing）。
