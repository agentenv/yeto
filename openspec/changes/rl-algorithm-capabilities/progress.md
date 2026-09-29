# rl-algorithm-capabilities 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)。

## 2026-09-29（Agent ALIGN，阶段 0）

- 规划文档从 `/home/michael/work/rl-algos`（分支 `rl-algorithms` HEAD 18695ae，含未跟踪文件）原样复制到分支 `rl-infra-spec`（提交 `5753e30`），此后以本分支副本为准；rl-algos worktree 未改。
- P0 框架；注册入口先于四个子 change。与 infra 的接口：A1（execution 能力由执行模式给出，算法哈希即 ExecutionProfile 契约身份）；`driver.py` 4.2 按 alignment §7 顺序。
- 任务状态：全部未完成（0 勾选），没有代码改动，没有使用 GPU 或云资源。
- `openspec validate rl-algorithm-capabilities --strict`：见 rl-infra-spec/progress.md 的 ALIGN 条目。

## 待批准

见 alignment.md §8（与本 change 相关的是：GPU 预算、`miles_adapter/config.py` 与 `entry.py` 的归属、“commit/push 需用户确认”一句的澄清）。

## 下一步

按 alignment.md §7 的工作包派发。

## 2026-09-29（Agent ALGO-CAP，WP-CAP）

**分支与提交**：worktree `/home/michael/work/algo-cap`，分支 `algo-cap`，基于 `origin/rl-integ` c5e05f4，采用普通 push（非强推）。
- `3d1b466` 接口冻结：AlgorithmSpec v2、注册入口、能力维度、映射表、learner 参数。
- `ebd436b` 合入 1b-shared 扩展点；launcher/export/cli 接线；补全测试。
- `d5af9df` 文档、research 更新、tasks/progress、证据。
- `62647fb` 合入 1b-p0tests.patch：只改测试夹具，并改为与分支无关的写法，algo-cap 与 origin/algo-1b 479f73e 上都通过。已核对 1b spec 原文：KL loss 必须记录参考模型身份，超采样只能配合动态过滤，reward 后处理只能用 yeto 分派器；这些规则只作用于启用了对应机制的描述。

**任务状态**（逐项）：
| 任务 | 状态 | 证据 |
|---|---|---|
| 0.1 | CPU 通过 | `baseline-failures.txt`（94 条） |
| 1.1–1.5 | CPU 通过 | `tests/test_rl_algorithm_spec_v2.py` |
| 2.1、2.3–2.5 | CPU 通过 | `tests/test_rl_algorithm_flags.py`；argv 快照未改动，照样通过 |
| 2.2、2.6 | CPU 通过（miles-next-venv，fsdp 后端，见 tasks 完成记录的说明） | `evidence/2026-09-29-upstream-parser/pytest.txt` |
| 3.1–3.4 | CPU 通过 | `tests/test_rl_algorithm_capabilities.py` |
| 4.1、4.2 | 已实现（补丁），未完成（未合入本分支，未勾选） | `/home/michael/work/infra-drafts/p0-driver.patch`，打补丁后 69 passed，全量失败集合不变 |
| 5.1–5.5 | CPU 通过 | `tests/test_rl_algorithm_provenance.py`、`tests/test_rl_algorithm_capabilities.py` |
| 6.1 | CPU 通过（dry-run 实跑） | `evidence/2026-09-29-dry-run/` |
| 6.2 | 已实现 | `docs/research/rl-algorithms/research.md` |
| 7.1 | CPU 通过 | `baseline-failures.txt` 与 `after-failures.txt` 的 diff 为空（均为 94 条；2213 passed, 42 skipped） |
| 7.2 | CPU 通过 | `openspec validate rl-algorithm-capabilities --strict`：valid |
| 7.3 | 已实现 | 本表 |

GPU 验证：本 change 不需要，也没有做。没有使用任何云资源，费用 $0，无残留。

**测试命令**：
- 全量：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/ --continue-on-collection-errors`。基线（rl-integ c5e05f4，在 detached worktree `/tmp/algocap-base` 上跑）为 68 failed / 2062 passed / 41 skipped / 26 errors；algo-cap 为 68 failed / 2213 passed / 42 skipped / 26 errors。两边按测试 id 去重后的失败集合完全相同。这 94 项是环境性失败（缺 syncer 二进制、miles、boto3 等），本 change 没有新增或消除任何一项。
- upstream：`PYTHONPATH=/home/michael/work/miles-next:$PWD:$PWD/tests /home/michael/work/miles-next-venv/bin/python -m pytest -q tests/test_rl_algorithm_flags_upstream.py`，20 passed。

**接口（供四个子 change 使用）**：
- `yeto/rl/engine/algorithm.py`：`AlgorithmSpec`（`advantage/loss/kl/correction/sampling/execution/entropy_coef/plugins`，`sha256()`，`canonical_json()`，`required_mechanisms()`，`rejections()`，`expects_gradient(batch, metrics)`，`verify_plugins()`），以及 `PluginRef` 和 `register_field/register_mechanism/register_rejection/register_launch_check/register_island_check/register_runtime_attrs/register_gradient_rule`。
- `yeto/rl/algos/__init__.py::EXTENSION_MODULES`：子 change 在此加一行注册。
- `yeto/rl/engine/miles_adapter/algorithm_flags.py`：`register_flag`、`absorb_extra_argv`、`algorithm_argv`。
- `EngineCapabilities`：`losses/loss_aggregations/kl_placements/corrections/reward_postprocessors/features`、`execution`、`unverified_mechanisms`、`with_unverified()`，以及 `check(..., max_policy_age=None)`。

**与 INFRA 的约定**（已经主 agent 转达）：ADAPTER_OWNED_FLAGS 加入 `--update-weight-transfer-mode` 与 `--yeto-placement-map`；learner/cli/launcher 透传 `--rl-placement`、`--rl-standby-gpus`（只在非默认值时透传，legacy 下拒绝）。

## 待批准（ALGO-CAP）

1. **合入 driver 补丁**：`p0-driver.patch` 修改 driver.py、miles_adapter/trainer.py，并新增测试，需主 agent 或 INFRA 合入。合入后才能勾选 4.1/4.2。
2. **5.5 中“外层同步”的解读**：当前按“岛数 > 1 则拒绝”实现，单岛带 1 成员 syncer 允许放行。如果要求严格到“任何 outer sync 都拒绝”，G1 冒烟就必须用无 syncer 的单岛模式，而当前 learner 入口不支持这种模式。
3. **`ssh_harness.py` 导出接线**：`export_rl_checkpoint` 的调用点在 ssh_harness，需要传入 `algorithm_spec`（可从事件 `rl/algorithm_spec` 读取）；该文件不归 ALGO-CAP。
4. **`yeto/cli.py`**：为 launcher 新增了 `--rl-algorithm-spec`、`--rl-allow-unverified-mechanism`、`--rl-placement`、`--rl-standby-gpus` 四个参数。该文件不在派发列出的路径中，但 launcher 的参数解析就在这里，请确认。
5. **与 infra-a 的合并冲突**：infra-a（ef3e149）也改了 `miles_adapter/config.py` 与 `entry.py`，合并时需要主 agent 处理。另外 `selection.require_ports_supported` 仍然会拒绝带 `rollout_num_gpus` 的 ports 运行，fixed-partition 要跑起来需要 INFRA 调整。

## 下一步
- 主 agent 合入 `p0-driver.patch`，之后勾选 4.1/4.2。
- 子 change 基于 `algo-cap` 最新提交 rebase，按 `EXTENSION_MODULES` 接入。

## 2026-09-29（ALGO-CAP，合并 infra-a 并按独立审查修订）

- 已合并 infra-a；调用 selection 时传入 `placement=args.rl_placement`（learner、launcher）；`miles_args.yeto_rl_expected_algorithm_sha256` 写入 learner。
- 处理的审查意见：F2（research §1/§5 标为历史基线）、F4（注册字段必须可哈希）、F5（`load_extensions` 只在导入成功后置位）、F6（梯度规则绑定到机制，默认 GRPO 的判定不变）、F7（launcher 在起云资源前运行 launch_problems 与 capability check）、F8（`dimension:name`）、F9/F10（存在任何外层同步即拒绝放行）、F11（夹具按字段注册情况分支；1b 字段存在时真实测试往返）、F12（见下）、F14（launcher 与 learner 哈希口径一致；learner 吸收参数后哈希变化会被拒绝，已加测试）。F15/F16/F17 在 driver.py/trainer.py 中，改动放在 `p0-driver-2.patch`，algorithm.py 侧（`gradient_expectation`、`valid_masked_fraction`）已直接实现。
- F12（ports 默认路径上可见的变化）：island learner 命令多了 `--rl-expected-algorithm-sha256`；`rl_engine_selected` 事件多了 `rl/algorithm_spec` 与 `rl/algorithm_absorbed_flags`；默认 GRPO 的 Miles argv 仍与 R0 逐字节相同。
- 任务状态有变化：2.6、5.3、5.5 改回未勾选，4.1、4.2 勾选（详见 tasks.md 的修订记录）。

### 待批准（新增）
- **G1 与放行开关：已新增单岛无外层同步运行模式（主 agent 决定，用户可推翻）；放行开关口径未放宽。** 原问题：D11 按原文执行后，learner（`--syncer` 必填）和 launcher 两个入口都带外层同步，`--rl-allow-unverified-mechanism` 在所有真实入口上都会被拒绝。需另批：要么放宽口径（允许单岛带 1 成员 syncer），要么新增一个无 syncer 的单岛入口。在批准之前，各子 change 的 G1 无法通过放行开关运行。
- 2.6 补救需要一次 Modal CPU 运行，按计划执行，费用 < $1。

## 2026-09-29（ALGO-CAP，新增单岛无外层同步运行模式；主 agent 决定，用户可推翻；放行开关口径未放宽）

- 新增显式入口 `--rl-single-island-no-sync`（learner、cli/launcher 都有），只用于 ports。行为：单岛、不连 syncer、`yeto_policy_sync=False`（LocalOnlySync，不做外层同步）。launcher 在这种模式下不启动 syncer 集群；Modal 岛不需要解析 syncer 地址，结果从岛本身取回。
- 启动前拒绝的组合：`--syncer`、`--num-learners` ≠ 1、`--learner-id` ≠ 0、非默认 sync preset、initial adapter、legacy；launcher 还拒绝多个 `--gpu` 条目和 external learners。
- 这是 D11 放行开关唯一合法的使用场景，满足原文“只在单岛运行中生效，与多岛或外层同步组合时拒绝”；没有放宽任何验收。
- 事件中带 `rl/outer_sync=false`；有放行时带 `rl/contains_unverified_mechanisms=true` 和 `rl/unverified_mechanisms`。
- 测试：`tests/test_rl_algorithm_provenance.py` 中的 no-sync 用例。全量失败集合与基线逐 id 相同。尚未做真实 GPU 运行，第一次真实使用由子 change 的 G1 完成。

## 2026-09-29（ALGO-CAP，复审 bddb58a..2f9f02c 的修复）

- **E1/E2/E3**：2.6 改回未勾选。result.md 已如实写明比较口径的偏离，以及原计划口径本身有误；两份运行日志已 `git add -f` 入库（入库前扫描无凭据）；yeto 版本按推断记录；第 3 次运行的计划 `attempt3-plan.md` 已先于运行提交。
- **N1**：`--rl-single-island-no-sync` 在非 RL 模式下，`run()` 一开始就拒绝，先于其他任何准备步骤，已加测试。**N2**：措辞已改。**N3**：`rl/outer_sync` 现在总是写入事件。**N4**：modal_runner 归 IMG，补丁放在 `infra-drafts/algocap-modal-runner-nosync.patch`，已本地验证。
- **G1**：固定优先级：只要有规则要求期望梯度就期望（收紧优先），之后才考虑放宽；规则返回值按 bool 转换，None 视为弃权；名不副实的旧测试已重命名；design D6 追加了一句说明收紧方向。**G2**：放在 `infra-drafts/p0-driver-3.patch`（基于 infra-a 39fa0ac）。在 infra-a 上单独套用本分支的 algorithm.py 时，`test_rl_algorithm_spec_v2` 中有一个用例会因为来源字符串改为 `tightened:` 而失败；algo-cap 合入 infra-a 后这个用例同步更新，失败随之消失。
- **F-a**：映射表必须等于 D3（`BUILTIN_FLAGS`）与扩展声明的映射（`EXTENSION_FLAGS`）之并；`--gamma` 恢复为未映射参数的参数化用例，扩展注册了它的映射时跳过。
- **S2**：docs 中子 change 的两个小节加了注：由对应子 change 提供，在集成分支生效。
- **1b-p0tests-f1**：夹具改为只要 `grpo_knobs` 在 `EXTENSION_MODULES` 中就调用 `with_pipeline_plugins`，不再靠 import 异常回退；algo-cap 与 origin/algo-1b 2722cad 上都通过。

## 2026-09-29（ALGO-CAP，no-sync 事件磁带、声明策略）

- `171408b`：no-sync 岛通过日志回传事件磁带。
  - learner 的 `install_event_echo()` 把每条磁带记录原样输出为 `YETO_RL_EVENT <磁带行>`，前缀与 INFRA driver 的实验 echo 相同。
  - launcher 的 `EventCollector` 从 Modal/sky 日志流重建 `<run dir>/events/<island>.jsonl`，流重放和 driver echo 造成的重复会被去掉；拆除前最多等 120 s，让日志流结束。
  - 端到端测试：岛侧 echo → fake Modal 日志流 → 本地磁带与岛磁带逐行相同，然后用 `--rl-event-tape` 导出。
  - adapter 声明加入 maxrl、mapo（gdpo 暂缓）。
- 声明策略（主 agent 决定，用户可推翻，见 alignment §7b）：只有 GPU 上机制确实生效的证据才能声明。
  - tis、opsm_trainer 暂时保留声明，但注明“G1 只证明能运行，是否生效待触发运行”；1a 触发运行没能证明生效就撤回。
  - `test_rl_algorithm_capabilities.py` 中的 4 个用例改为从候选机制与 `declared_mechanisms()` 的差集中动态选取未声明项，并断言差集非空；“一次列出全部问题并给出可选项”等断言都保留。另外新增一个用例：Miles adapter 接受每个已声明项，拒绝 overlong_filter。
