# Tasks

## 0. 在途 PR 与分支准备

- [x] 0.1 合入 #64、#65、#59，关闭已失效的 #43；如果 #62、#63 先合入，记录它们对远端准备脚本的改动。验证：`gh pr list -R agentenv/yeto --state open` 中已经没有这三个 PR 和 #43；main 上 `python -m pytest -q tests/test_rl_*.py` 通过。
  - 完成记录（2026-09-29）：#64、#65、#59 以及 #58、#60–#63、#66 已合入 agentenv/yeto main（#61、#63 经 #67、#68 重新合入）。#43 是他人（AlexEisie）的 PR，本人无权关闭，经用户决定视为不在本 change 范围内。main 上 `tests/test_rl_*.py` 与整合分支的失败集合完全相同（均为缺 syncer 二进制、miles 模块等环境原因），无新增失败。
- [x] 0.2 创建 `openspec/changes/rl-engine-ports/migration-ledger.md`，登记以下各项，每项都写明 PR、行为、ports 实现位置和验证方式：
  - #64：梯度流不变量，以及零梯度时拒绝提交；
  - #65：GDN recipe；
  - #59：数据列名；
  - #66：capability 格式。

  验证：清单覆盖以上 4 项，每项的字段都齐全。
- [x] 0.3 从 main 开出 `rl-engine-ports` 分支；在 `michaellchung/miles` 上基于 `9e4260d` 开出 `yeto/ports` 分支；新建 `michaellchung/sglang`（fork 自 `sgl-project/sglang`），并基于 Miles pin 所用的 `sglang-miles` commit 开出 `yeto/ports` 分支。验证：三个分支都存在，两个 fork 分支的基底 commit 记录在 design D6 与 D11 中。
  完成记录（2026-09-29）：`git ls-remote` 确认三个分支存在：`michaellchung/yeto` rl-engine-ports=`1143117`（基底 agentenv/yeto main `e21a7ff`，即 #64/#65/#59 合入后的 main）；`michaellchung/miles` yeto/ports=`0394715`（基底 `9e4260d`，ahead 2/behind 0）；`michaellchung/sglang` yeto/ports=`9f29303`（基底 sglang-miles `571212b`，ahead 4/behind 0）。`gh api .../compare` 的 merge_base 均等于所记基底。HEAD 与 `yeto/rl/__init__.py` 的 `MILES_NEXT_COMMIT`/`SGLANG_NEXT_COMMIT` 一致。基底与来源已写入 design D6、D11。

## 1. 固定 fork 版本与依赖验证

- [x] 1.1 在 `yeto/rl/__init__.py` 中新增 `MILES_NEXT_*`（repository 为 `michaellchung/miles`）和 `SGLANG_NEXT_*`（repository 为 `michaellchung/sglang`）两组常量，以及 `MILES_NEXT_IMAGE`；旧常量不变。验证：`tests/test_rl_launcher.py` 中现有的 legacy 断言全部通过，新增测试断言新旧常量组互相独立，且新常量组中不引用 bundle。
- [x] 1.2 让 `verify_miles_revision` 接收期望常量组作为参数，legacy 路径传入旧常量组。验证：现有的 revision 校验测试通过；新增测试覆盖 ports 常量组的 commit、origin 不匹配以及工作区不干净时的拒绝。
- [x] 1.3 在 `michaellchung/sglang` 的 `yeto/ports` 分支上逐个移植 `agentenv/sglang` 的 5 个补丁（`b34df47`、`95d4d69`、`8db3711`、`c2cb40a`、`e1b57eb`），每个补丁单独提交并带上原有测试；`95d4d69` 按 upstream Miles 的 LoRA 请求格式重新对齐。验证：
  - SGLang 侧移植后的单元测试全部通过；
  - `openspec/changes/rl-engine-ports/sglang-patch-port.md` 逐个补丁记录：移植后的 commit、上游是否已有等价实现、测试结果；
  - `SGLANG_NEXT_COMMIT` 固定在该分支的最新提交上。
- [x] 1.4 在 `michaellchung/miles` 的 `yeto/ports` 分支上实现通用插件调用入口：train actor 增加 `run_plugin`，`TrainGroup` 增加透传方法；只提交到 fork，不向官方 `radixark/miles` 提 PR。验证：Miles 侧单元测试通过；`MILES_NEXT_COMMIT` 固定在包含该提交的 commit 上。
- [x] 1.4b 检查 `agentenv/miles` 中的端口隔离提交（`5494a6ce`）和 TITO Qwen3.8 提交（`5a9cf0d0`、`e2ad83d8`）在 upstream 上是否已有等价实现；没有的话就移植到 `yeto/ports`，提交说明注明来源 SHA。验证：在两岛同机的冒烟测试中没有端口冲突；Qwen3.8 的 TITO 单元测试通过。
- [x] 1.5 在 `launcher.py` 与 `ssh_harness.py` 的远端准备脚本中，按 `--rl-engine` 分支 checkout 与安装源码。验证：新增 launcher 和 harness 单元测试，断言 legacy 生成的脚本与改动前逐字节一致，ports 生成的脚本从 `michaellchung/miles` 与 `michaellchung/sglang` 获取固定 commit，包含 origin、commit 与工作区干净检查，并且不包含任何 bundle 步骤。

## 2. 端口定义与配置拆分

- [x] 2.1 新建 `yeto/rl/engine/ports.py`，定义 `RolloutPool`、`TrainerGroup`、`PolicyState`、`Publisher`、`Placement`、`AlgorithmSpec`、`EngineCapabilities`、`RolloutBatchHandle`、`TrainableState`。返回类型复用 `contracts.py`，并为 E1–E3 的动词预留签名注释。验证：`python -c "import yeto.rl.engine.ports"` 不导入 torch、ray 或 miles，类型检查测试通过。
- [x] 2.2 把 `CanonicalLoraState` 泛化为带 layout 字段的 `TrainableState`（LoRA / full / fragments），LoRA 路径的 hash 与现有 `policy_hash` 保持一致。验证：`tests/test_rl_core.py` 全部通过，新增测试断言同一张量集合在新旧表示下 hash 相同，且未实现的布局会被拒绝。
- [x] 2.3 实现 `AlgorithmSpec` 的规范化与 SHA256（GRPO + 有界非零方差过滤）。验证：单元测试覆盖哈希稳定性、未知字段拒绝，以及与 legacy 参数的对应关系。
- [x] 2.4 把 `build_miles_argv` 拆成两层：引擎无关的 RL 运行配置，以及 legacy 参数翻译。其中包含 #59 的数据列名和 #65 的 GDN recipe。验证：新增快照测试，对现有测试中的全部配置断言 legacy 生成的 argv 与拆分前逐项一致。
- [x] 2.5 实现 `EngineCapabilities` 的序列化，格式与 #66 的 capability 认证一致。如果 #66 尚未合入，就先按其 PR 版本实现，并在它合入时复核。验证：测试中用 #66 的读取函数解析该声明成功。

## 3. MilesAdapter

- [x] 3.1 `miles_adapter/config.py`：把 RL 运行配置和 `AlgorithmSpec` 翻译成 upstream Miles 参数，未映射的配置项拒绝启动；单 cell 断言；禁用 FT 相关参数。验证：单元测试覆盖参数翻译、未映射项拒绝和 FT 参数拒绝。在 upstream 源码上跑 Miles 的 `parse_args`，确认没有未知参数。
- [x] 3.2 `miles_adapter/rollout.py` 与元数据提取回调：`generate` 返回 `RolloutBatchHandle`，元数据通过 `--rollout-all-samples-process-path` 在 rollout 进程内提取。验证：GPU 冒烟测试中一轮生成后，yeto 进程只持有元数据，policy token 与期望快照一致；token 不匹配的注入测试会在训练前被拒绝。
- [x] 3.3 `miles_adapter/trainer.py`：`train_step`、`onload`、`offload`，返回 `LocalStepReceipt`；训练后释放 rollout 引用。验证：GPU 冒烟测试完成一次训练步，receipt 字段完整。
- [x] 3.4 `miles_adapter/state_plugin.py` 与 `state.py`：基于 1.4 的插件入口，在 upstream 的 LoRA 结构上实现导出和应用（保留或重置 optimizer），遵守 D4 的梯度流约束。验证：
  - `test_rl_grad_accumulator_hook` 通过；
  - GPU 测试完成"导出 → 应用 → 再导出"，hash 一致；
  - 重置模式下 moments 被清零且 scheduler 对齐；
  - 在同一 checkpoint 上与 legacy 的导出结果逐张量比对，全部一致。
  - 完成记录（2026-09-29）：`test_rl_grad_accumulator_hook`（#64）CPU 通过；GPU（923b304，`evidence/2026-09-29-rerun-smoke-normal/probe.jsonl`）导出→应用（preserve/reset）→再导出 hash 一致，reset 后 moments 清零、scheduler 对齐；同一初始 checkpoint 上两条引擎导出的 LoRA（392 张量，TF 的 audit `base.f32`）逐位一致（relL2=0、layout_hash 相同），见 `evidence/2026-09-29-eq62-v3/report.md`。
- [x] 3.5 `miles_adapter/publish.py`：基于 upstream 的 `update_weights` 完成发布，生成带成员集合与 payload hash 的 `InferencePublicationManifest`。验证：GPU 测试中发布后所有 engine 的权重 checksum 与清单一致；注入单个 engine 失败时返回错误，而不是一份清单。
- [x] 3.6 `miles_adapter/placement.py`：返回共置和启动时固定分区的只读描述；检测到 Miles 参数规范化改写了请求的放置时拒绝启动。验证：单元测试覆盖两种放置和改写检测。

## 4. IslandDriver 与同步集成

- [x] 4.1 `yeto/rl/engine/driver.py`：实现 colocated-serial 循环（生成 → 训练 → 安全边界同步 → 完整发布），包括 offload/onload 时序、eval 与进度保存；每轮检查 grad_norm 不变量。验证：GPU 冒烟测试在无外层同步的单岛上跑完 3 轮，事件顺序符合 spec；注入零梯度时该轮失败且不提交。
- [x] 4.2 把 strict-avg bridge 接到 driver 的安全边界，用端口的 `PolicyState` 和 `Publisher` 代替 `MilesPolicySync` 中对 Miles 内部的调用。验证：`tests/test_rl_integration.py` 中 strict 的 CPU/伪引擎用例在 ports 路径上通过；GPU 上两岛 strict-avg 完成 3 轮，两岛同步后的 hash 相同。
- [x] 4.3 把 decoupled bridge 接到 driver，包括 run-until-stop、在安全边界 drain BCAST/PULL、finalization 以及"发布一次后停止"。验证：`tests/test_rl_decoupled.py` 在 ports 路径的伪引擎上通过；GPU 上两岛 decoupled 跑到最终 cut 并导出 PEFT。
- [x] 4.4 island 进度 checkpoint 与恢复在 ports 路径上沿用现有格式（不含 LoRA 和 optimizer），重启后以重置方式应用权威 cut。验证：GPU 测试在第 2 轮 kill 掉 learner 并重启，恢复后的轮次与 group 复用规则与 legacy 一致。

  - 完成记录（2026-09-29）：`evidence/2026-09-29-kill44-legacy-vs-ports/report.md`（代码 2bb651d，H100，同一 kill 点：第 2 轮训练中）。两边重启后都从 rollout 2 继续、以 reset 应用 kill 前相同的权威 cut v2、丢弃被杀轮次的轨迹并重新生成、此后每轮 4 组、syncer 4 个 outer step 完成并确认最终 cut。发现并修复 ports 的记录缺陷：重启后未覆盖被杀轮次的 `rollout_metrics`（`bridges.py` `StrictIslandProgress.after_generate`，新增测试 `test_strict_progress_replaces_a_killed_learners_record_for_the_same_round`）。规格允许的差异：over-sampling 大于 batch 时 ports 不复用剩余 group（legacy 注释标明该队列可丢弃）。
## 5. 路径选择

- [x] 5.1 在 CLI、launcher 与 learner 中加入 `--rl-engine {legacy,ports}`，默认 `legacy`。选择写入事件与来源记录；ports 遇到不支持的组合（SAO、dense-full、DSV4、critic、分区）时在启动前拒绝。验证：launcher 与 learner 单元测试覆盖默认值、记录和拒绝矩阵；legacy 的全部现有测试不变。
- [x] 5.2 更新 `docs/MILES_RL.md`，新增"Engine selection"一节：说明两条路径、ports 的边界、端口职责表，以及 Miles 侧唯一的补丁。验证：文档中的命令示例在 `--dry-run` 下可以执行。

## 6. 等价性验收

- [x] 6.1 编写对照脚本 `scripts/rl_engine_equivalence.py`，按 spec 的四层口径（design D12）判定：第 1 轮严格、teacher forcing、第 2 轮起按 seed 汇总的置换检验、路径内 hash。脚本负责多 seed（默认 17–21）运行编排（`plan`/`run`）、对已有运行目录的分析（`analyze`）和报告输出；loss/grad_norm 缺失时从 `miles.log` 补齐并记录来源；阈值为脚本常量，写入报告头。teacher forcing 入口为 `yeto.rl.teacher_forcing.replay_generate`（经 `benchmark_rl.py --custom-generate-function-path` 与 `YETO_RL_REPLAY_ROLLOUTS` 回放 legacy 第 1 轮 rollout）。验证：`plan` 能输出 legacy/ports × seed 与 teacher forcing 的命令；`fake` 模式生成带 FAKE 标记的分层报告；`tests/test_rl_engine_equivalence.py` 覆盖分层判定、置换检验判定、teacher forcing 报告解析与回放；在已提交的 legacy-baseline-v2 与 strict2 证据上，第 1 层复现 0.6%/1.0% 的 grad_norm 相对差并通过。
- [x] 6.2 用小模型（Qwen3-0.6B + LoRA）跑两岛 strict-avg 3 轮：legacy 与 ports 各 5 个 seed（17–21），并以 legacy seed 17 第 1 轮的 rollout 做 teacher forcing。验证：`analyze --preset strict-avg` 的报告结论为 PASS，即
  - 第 1 层：seed 17 第 1 轮 group/token/reward 完全相等，grad_norm 相对差 ≤ 3%；
  - 第 2 层：回放复现记录，初始 LoRA 一致，loss 绝对差 ≤ max(1e-6, 1e-3×|legacy|)，grad_norm 相对差 ≤ 3%；optimizer 之前的 LoRA 梯度（`optimizer.step()` 入口、DP 规约后、裁剪前，全部 LoRA 张量按 canonical 名对齐拼接；`YETO_RL_AUDIT_GRADS=1` 时由 `yeto.rl.grad_audit` 写出 `audit/round-00000001.grad.{f32,json}`，TF 运行由 `run` 自动设置）以 CPU fp32 参考为锚（`yeto.rl.fp32_reference`，由 `analyze` 调用；同一初始 LoRA `base.f32`、同一回放 batch、固定 revision 的模型、与引擎一致的 GRPO 损失）：每个岛 `relL2(ports, fp32) ≤ relL2(legacy, fp32) + 0.05` 且 `cos(ports, fp32) ≥ cos(legacy, fp32) − 0.005`；报告记录回放样本、初始 LoRA、模型权重的来源与 sha256，fp32 参考缺输入或计算失败判未完成；跨路径梯度相对 L2/余弦与 LoRA 更新量相对 L2、余弦、符号翻转比例、范数比只报告（D12，2026-09-29 两次经用户批准的口径修改）；
  - 第 3 层：每个 seed 把第 2 轮起的 reward_mean、loss、grad_norm、action_tokens 分别平均，legacy 与 ports 各 5 个值做精确双侧置换检验（252 种划分，均值差），Bonferroni α=0.0125，四个指标 p 均 ≥ 0.0125；报告列出各 seed 原始值、效应量与检出力局限；
  - 第 4 层：每条路径每个 seed 内各岛同一轮 hash 一致；
  原始记录与报告存放在本 change 目录下。
  - 完成记录（2026-09-29）：`evidence/2026-09-29-eq62-v3/report.md` 结论 PASS（H100；第 1、2 层用 18695ae 的 legacy-s17/ports-s17/tf-*，原始记录在 `evidence/2026-09-29-eq62-v2/`；第 3 层 seeds 18–21 用 4bc018e 的 `evidence/2026-09-29-eq62/`）。第 2 层按用户批准的 fp32 锚定口径（design D12 第二次口径修改）。限制：仅覆盖单轮任务，见 design 风险。
- [x] 6.3 用小模型跑两岛 decoupled（P=8、tau=2、H=4）到导出：legacy 与 ports 各 5 个 seed，并做 teacher forcing。验证：`analyze --preset decoupled` 的报告结论为 PASS，即第 2 层（loss、grad_norm，以及与 6.2 相同的以 fp32 参考为锚的 LoRA 梯度判定；decoupled 的 teacher forcing 未写出梯度审计文件时继续引用 6.2 的 strict-avg teacher forcing；跨路径梯度距离与更新量只报告）、第 3 层（同 6.2 的置换检验口径）、第 4 层（非部分应用的版本与最终 cut）通过，第 1 层只报告；两条路径全部 seed 导出的 PEFT 都能被标准 PEFT 加载；报告给出 seed 17 两条路径最终 LoRA 的相对 L2 距离和 legacy 跨 seed 的同一距离，不设硬阈值（第 2 轮起采样分叉，该距离反映轨迹差异而非 trainer 误差）。

  - 完成记录（2026-09-29）：`evidence/2026-09-29-eq63-v3/report.md` 结论 PASS（H100，4bc018e 的 5 seed；第 2 层引用 6.2 的 strict TF）；10/10 个 PEFT 可被标准 PEFT 加载；最终 LoRA 跨路径相对 L2 0.0083（legacy 跨 seed ≈1.414）。两条路径同样存在 decoupled 学习率衰减到 0 的已知问题（由 `fix-decoupled-lr-schedule` 另行修复）。
## 7. 切换与退役

- [x] 7.0 检查迁移清单：R0 期间合入的所有 RL PR 都已登记，且每一项都已关闭。验证：`migration-ledger.md` 中没有未关闭项；对照 `gh pr list --state merged --search "rl:"` 在 R0 期间合入的列表，没有遗漏。
  - 完成记录（2026-09-29）：对照 2026-09-24 之后合入的 #48–#68（按改动文件筛选），`migration-ledger.md` 已登记全部触及 RL 行为的 PR 并全部关闭；补登记 #60（外部 SGLang router：ports 显式不复刻，由 `require_ports_router_mode` 拒绝预设地址、Modal 仅 legacy 设置该 env，测试见 `tests/test_rl_engine_selection.py`）；#58、#61/#67、#68 与 #48–#57 登记为与引擎无关或已含于基线。遗留：yeto 自身 Modal launcher 路径上尚无 ports 的真实运行（验收中的 Modal 运行走的是 sandbox harness）。
- [x] 7.1 把默认值切换为 `ports`，legacy 仍可显式选择；更新文档。验证：launcher 测试断言新的默认值；用默认参数完成一次真实运行。
  - 完成记录（2026-09-29）：默认值切换见提交 d6248ae（CLI/launcher/learner/harness/export/benchmark 默认 ports，legacy 显式可选，测试断言新默认值）。默认参数真实运行：Modal 2 岛 × 1×H100、strict-avg 3 轮，未传 `--rl-engine`/`--rl-image`，事件 `rl_engine_selected=ports`、默认镜像为 `MILES_NEXT_IMAGE`、fork 校验通过、两岛每轮 hash 一致、grad_norm 与 delta 非零、默认参数导出 PEFT 成功（`evidence/2026-09-29-default-params-modal/`）。head 放在本机（本机 SkyPilot 无 AWS 凭据、Nebius 公网 IP 被跨云测试占用、Verda 不支持开端口）。Nebius 的佐证为 head 模式运行（2cbd45c，当时显式 `--rl-engine ports`，`evidence/2026-09-29-head-ports/`）；Verda 的默认参数运行推迟到 change `fix-verda-provider` 任务 6.5。
- [ ] 7.2 至少一个真实运行周期之后，删除 legacy 路径：旧 `MILES_*` 常量、`miles-*.bundle`、`MilesPolicySync` 回调集成、legacy 参数翻译，以及 `--rl-engine` 参数本身。仅 legacy 支持的功能（SAO、dense-full、DSV4）在文档中标注为待迁移。验证：全部测试通过，仓库中搜索 `MILES_BASE_COMMIT`、`external-policy-sync-path` 与 `--rl-engine` 均无结果，`docs/MILES_RL.md` 已更新。
