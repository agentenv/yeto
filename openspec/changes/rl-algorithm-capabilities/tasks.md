# Tasks

执行约定：
- 规划文档以 `rl-infra-spec` 分支上的副本为准；实现 worktree、分支与可修改路径按 `../rl-infra-spec/alignment.md` 的工作包派发（原 `/home/michael/work/rl-algos` 分支 `rl-algorithms` 不再作为规划来源）。
- 测试命令为 `/tmp/yeto-venv/bin/python -m pytest -q`。全量测试以"改动前后失败集合相同"为准，改动前的基线要先记录下来（见 0.1）。
- 不修改 legacy 的 `build_miles_argv`，不修改 Miles/SGLang fork。
- 本 change 不需要 GPU。
- `yeto/rl/engine/driver.py` 同时被 4.2、`fix-decoupled-lr-schedule` 2.1 与 `rl-infra-spec` 1.7/2.2 修改；按 alignment.md 的顺序由单一写入者提交，4.2 只改 `_check_gradient` 的判定调用。
- commit 和 push 需要用户确认。

## 0. 基线

- [ ] 0.1 在改动前跑一遍全量 `pytest tests/ --continue-on-collection-errors`，把失败和错误集合保存到 `openspec/changes/rl-algorithm-capabilities/baseline-failures.txt`。验证：文件存在，且条目数与 pytest 汇总一致。

## 1. AlgorithmSpec v2 结构与规范化（design D1/D2/D7）

- [ ] 1.1 在 `yeto/rl/engine/algorithm.py` 中实现分组冻结数据类（Advantage/Loss/Kl/Correction/Sampling/Execution、`entropy_coef`、`PluginRef`），并做字段级校验：类型、有限性、范围、枚举。验证：新增 `tests/test_rl_algorithm_spec_v2.py`，非法值用例逐一报出字段名。
- [ ] 1.2 实现 v1/v2 的 `from_dict` 以及 v1 字段到新结构的映射（`kl_coef` 按 D5 处理）。验证：用 R0 已有的 v1 JSON 做往返测试，结果一致。
- [ ] 1.3 实现"v1 可表达时输出 v1 形态"的规范化和 v2 规范化。验证：`tests/test_rl_engine_algorithm.py` 原有的 golden 哈希和 JSON 用例全部不改就能通过；新增用例覆盖：设置任一新字段后 schema 变为 v2、哈希改变；`0` 与 `0.0` 等价；字段顺序无关。
- [ ] 1.4 实现 `PluginRef` 的源码哈希计算与启动时核对，只允许 `yeto.` 或 Miles 内置命名空间。验证：单测覆盖哈希匹配、不匹配、导入失败、命名空间不允许，共四种情况。
- [ ] 1.5 实现派生方法 `required_mechanisms()`（供 D4 使用）和 `expects_gradient()`，默认 GRPO 的判定等价于 R0。验证：单测对默认 GRPO 的判定与 R0 `driver.py` 的逻辑逐一比对。

## 2. 参数映射、吸收与翻译（design D3/D5）

- [ ] 2.1 新建 `yeto/rl/engine/miles_adapter/algorithm_flags.py`，写入映射表（design D3 列出的参数）和"影响训练目标的参数"清单。验证：单测保证映射表是清单的子集，且每行都有解析器和翻译函数。
- [ ] 2.2 在 miles-next-venv 中核对清单和映射表里的每个参数都存在于 upstream parser。验证：新增测试 `tests/test_rl_algorithm_flags_upstream.py`，在 miles 不可用时 skip，并在 `/home/michael/work/miles-next-venv` 中实际运行通过。在任务说明里记录运行命令和结果。
- [ ] 2.3 把映射表中的参数全部加入 `ADAPTER_OWNED_FLAGS`；让 `check_extra_argv` 改为吸收已映射参数、对冲突报错、拒绝清单内未映射的参数。验证：单测覆盖吸收、冲突、拒绝三种情况，报错内容包含参数名和取值。
- [ ] 2.4 让 `translate_run_config` 按映射表从 spec 生成算法参数，与 v1 默认值对应的参数不输出。验证：`tests/test_rl_argv_snapshot.py` 与 `tests/test_rl_miles_adapter_config.py` 不改就能通过（默认 GRPO 的 argv 逐字节不变）；新增用例覆盖每个映射字段的非默认值翻译。
- [ ] 2.5 实现 KL 放置规则（D5）：grpo/gspo 配 reward 中的 KL 且系数大于 0 时拒绝；`placement=loss` 翻译为 `--use-kl-loss` 系列参数；`placement=none` 时不输出 `--kl-coef`。验证：参数化单测覆盖上述情况，以及 v1 `kl_coef` 为 None、0.0、大于 0 这三种输入。
- [ ] 2.6 用 upstream `parse_args` 解析非默认映射所生成的 argv。验证：在 miles-next-venv 中逐字段解析通过；把测试与运行结果写进任务说明。

## 3. 能力声明、执行要求与拒绝矩阵（design D4）

- [ ] 3.1 为 `EngineCapabilities` 增加机制维度和 `execution` 字段，并实现序列化和反序列化。验证：单测覆盖往返一致；`yeto.rl.elastic_benchmark.capabilities.attestation_from_dict` 读取新格式后，旧字段结果与之前相同。
- [ ] 3.2 让 `check()` 按 `required_mechanisms()` 与执行要求逐项比对，把所有问题汇总后一次报出。验证：单测覆盖以下场景：可表达但未开放的机制被拒、需要 critic 被拒（报错提示改用 legacy）、陈旧度要求不满足被拒，报错中列出已支持的选项。
- [ ] 3.3 实现拒绝矩阵：TIS 与 `use_rollout_logprobs` 同时开、reward KL 与 loss KL 同时开、序列级 ratio 未显式给出 clip、要求二值奖励的机制配了非二值奖励。验证：参数化单测，每项都在构建 GPU 进程之前（fake 组合根）失败。
- [ ] 3.4 更新 Miles adapter（`miles_adapter/entry.py`）和 fake engine（`engine/fake.py`）的声明：机制维度与 R0 相同，`execution` 为 critic=False、staleness=0、rollout_logprobs=True。验证：fake 组合根测试中，默认 GRPO 能启动，其他机制被拒。

## 4. 零梯度不变量按算法判定（design D6）

- [ ] 4.1 `TrainStepMetrics` 增加可选字段 `masked_fraction`；adapter 从 Miles 训练指标中读取，读不到时为 None。验证：单测覆盖有、无两种情况。
- [ ] 4.2 把 `driver.py` 的判定改为调用 `spec.expects_gradient()`。验证：R0 已有的零梯度注入测试不改就能通过；新增 fake 测试：声明合法全屏蔽的机制在 `masked_fraction=1.0` 时不判失败；grad_norm 非有限时仍判失败。

## 5. 用户输入、岛间一致与来源记录（design D8/D9）

- [ ] 5.1 learner 和 launcher 新增 `--rl-algorithm-spec PATH`，只在 ports 下生效，与 legacy 组合时启动前拒绝，并按 D8 的优先级构建 spec。验证：单测覆盖以下情况：文件给出 v1、给出 v2、不给（与 R0 相同）、与 legacy 组合被拒。legacy 的 argv 快照测试不改就能通过。
- [ ] 5.2 launcher 在 ports 模式下计算 `expected_algorithm_sha256`，经 `--rl-expected-algorithm-sha256` 下发；learner 在连接 bridge 之前核对，不一致时写入 `rl_algorithm_mismatch` 事件并退出，缺参数时写警告事件。验证：launcher 的 argv 测试确认该参数只在 ports 模式出现；learner 的单测覆盖一致、不一致、缺失三种情况。
- [ ] 5.3 `export.py` 的 ports 分支写入 `algorithm_spec` 与 `algorithm_spec_sha256`。验证：导出测试确认 ports 的来源记录包含这两项且与事件一致；legacy 导出测试不改就能通过（逐字节一致）。
- [ ] 5.5 实现 `--rl-allow-unverified-mechanism NAME`（design D11）：只在 ports 单岛运行中生效；多岛、外层同步或 legacy 下启动前拒绝；事件和来源记录写入 `rl/unverified_mechanisms`；不影响算法哈希；不绕过拒绝矩阵。验证：fake 组合根单测覆盖单岛放行成功、两岛被拒、legacy 被拒、放行时仍触发拒绝矩阵、哈希不变。
- [ ] 5.4 运行事件中记录吸收来的参数（`rl/algorithm_absorbed_flags`）。验证：单测确认事件字段内容正确。

## 6. 文档

- [ ] 6.1 更新 `docs/MILES_RL.md` 的 Engine selection 一节：`--rl-algorithm-spec` 用法、吸收与拒绝规则、KL 放置规则、能力与执行要求、"可表达未开放"的含义，以及 Miles 升级时重新审查参数清单的步骤。验证：文档中的示例命令用 `--dry-run` 执行，结果与描述一致。
- [ ] 6.2 更新 `docs/research/rl-algorithms/research.md` 的 §8 和 §11，改为已确认的方案（吸收取代一律拒绝；execution 与 staleness；KL 放置）。验证：研究文档中不再出现与本 change 矛盾的表述。

## 7. 集成检查

- [ ] 7.1 跑全量 `pytest tests/ --continue-on-collection-errors`，失败集合与 0.1 记录的基线相同。验证：两次集合 diff 为空，结果写入 `openspec/changes/rl-algorithm-capabilities/progress.md`。
- [ ] 7.2 `openspec validate rl-algorithm-capabilities --strict` 通过。验证：命令输出无错误。
- [ ] 7.3 在 `progress.md` 中列出每项任务的状态，严格区分"已实现""CPU 测试通过""未做 GPU 验证（本 change 不需要）"。验证：文件存在且逐项对应。
