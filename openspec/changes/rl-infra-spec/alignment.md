# 阶段 0：infra 与算法对齐（Agent ALIGN，2026-09-29）

范围：`rl-infra-spec` 与五个算法 change（`rl-algorithm-capabilities`=P0、`rl-algo-mismatch-correction`=1a、`rl-algo-grpo-knobs`=1b、`rl-algo-seq-and-adv`=2a、`rl-algo-loss-variants`=2b）。本文件是派发、顺序与待批准事项的唯一来源；各 change 的 `progress.md` 链接到这里。

基线：分支 `rl-infra-spec`（起点 7fffbfd）。五个算法 change 与 `docs/research/rl-algorithms/` 从 worktree `/home/michael/work/rl-algos`（分支 `rl-algorithms` HEAD 18695ae，含未跟踪文件）原样复制，提交 `5753e30`；此后的修订都在本分支上，用 `git diff 5753e30 -- openspec/changes/rl-algo*` 审阅。rl-algos worktree 未改动。

本轮只改规划文档，没有改代码，没有使用 GPU 或云资源。

## 1. 对齐结论

1. **没有语义冲突。** 五个算法 change 都把 `execution.max_policy_staleness` 固定为 0，并且明确拒绝 staleness>0。rl-infra-spec 在严格 profile 下禁止旧版本生成，2.3 不擅自开放 one-step-off-policy。二者方向一致。有两处措辞会让读者以为 infra 会提供异步模式（P0 proposal 的“需要 rl-infra-spec 提供异步模式”、1a design D1 的“rl-infra-spec 异步模式的基础”），已改为“需另立独立算法契约 change”（A6）。
2. **重复定义了同一个概念（已统一）。** infra 1.4 的 `ExecutionProfile.algorithm_contract`/`max_policy_age` 与 P0 的 `AlgorithmSpec.execution.max_policy_staleness`、`EngineCapabilities.execution.max_policy_staleness` 表达的是同一件事。统一规则：算法契约身份就是 `AlgorithmSpec` 的规范化哈希；`max_policy_age ≤ spec.execution.max_policy_staleness`；执行能力中的 staleness 由已认证执行模式决定（A1）。
3. **缺失项（已补进 tasks/design）。**
   - 账本没有区分“算法有意丢弃”和“丢失”（A2）。
   - cut 不含算法身份、插件与 ref 模型（A3）。
   - E3 变 DP 的认证与 loss 归一化机制无关联（A4）。
   - 算法指标没有 profile/epoch/传输标签；分区模式下的 policy token 校验没有写进验收（A5）。
   - M1–M6 的 fork 任务未写入，任务依赖也没接上（M1–M6 提交、A9）。
4. **写入冲突（通过顺序与负责人解决，见 §7）。**
   - `driver.py` 同时被 lr-fix 2.1、P0 4.2、infra 1.7/2.2 修改。
   - `miles_adapter/config.py`、`entry.py` 同时被 P0 与 infra 修改。
   - `rollout_meta_hook.py::record_trained_groups` 被 1b 6.3 与 infra 3.6 共用。
   - Miles fork `yeto/ports` 同时承载 M1–M6 与 2b 路线 B，二者共用同一个 pin。
5. **上游 PR 作废。** rl-engine-ports/progress.md 中“向 radixark/miles 提 run_plugin PR”已改为作废说明。rl-infra-spec design 的 Risks 原文“评估提交 upstream”改为永不向上游提 PR（提交 `c122f1a`）。

## 2. 依赖与能力矩阵

列说明：“提供者”指由哪条 task 提供；“消费者”指哪个算法 change 或 infra task 使用它。一致性：✔ 表示一致；△ 表示已修订对齐；✖ 表示仍有未决问题（见 §8）。

| 能力 / 挂载点 | 提供者 | 消费者 | 接口 | 缺失 / 重复 / 冲突 | 状态 |
|---|---|---|---|---|---|
| ExecutionProfile + readiness + 策略版本 | infra 1.4（`execution_profile.py` 已实现 + CPU 通过，未勾选） | P0 D4 `execution` 能力；1a D1；2b D6 | `algorithm_contract`/`max_policy_age` ↔ `AlgorithmSpec.execution.max_policy_staleness` | 重复定义，已统一（A1）；比对在创建 GPU 进程前完成 | △ |
| 严格 profile 禁止旧版本偷跑 | infra 1.4/2.2/2.3（X9） | 1a（π_behav=π_old 前提）、2a GSPO、2b | `generate_blockers`，R0 `driver.py` 每组 policy token 校验 | 分区模式的验收原先没有写 policy token 校验，已补（A5） | △ |
| one-step-off-policy / staleness>0 | 无（infra 2.3 明确不开放；算法侧全部拒绝） | P0 能力表“异步目标 ⛔” | — | 措辞暗示由 infra 提供，已改（A6） | △ |
| 时间线观测 | infra 1.7（纯计量已实现；事件发射被 lr-fix 阻塞） | 1a G2 报告、1b/2a/2b 每轮指标、infra 6.1 | 事件带 profile hash/epoch；算法指标 `masked_fraction`/clipfrac/mismatch/过滤计数 | 缺少算法指标的标签，已补（A5）；关闭观测兼容旧路径的验证仍待接入 | △ |
| 端口 `Placement.describe/reconfigure` | infra 2.1、3.4、4.7；fork M1/M6（2.1a/4.6a） | 无算法消费者 | `reconfigure(plan, epoch)`（design D1），与 ports.py 预留签名 `reconfigure(target: PlacementDescription, *, epoch)` 不同 | 签名在 3.4a 更新 ports.py 时以 design 为准定稿；预留注释标为 E3，需提前到 E1（3.4a） | △ |
| 端口 `RolloutPool.add_engines/remove_engines/drain` | infra 3.4；M2/M3（3.3a/3.3b） | 无 | `add_engines(count,*,epoch)`、`remove_engines(members,*,epoch)`、`drain(members, deadline)` | `drain` 原先未预留，由 3.4a 补上 | △ |
| 端口 `Publisher.publish(policy, members)` | infra 3.5；M4（3.5a） | 1a（发布路径决定训推差异） | 现有 `publish(state)`，扩展成员集合 | upstream 没有 payload 级 ACK，由 yeto 读回校验补齐（3.5a）；分区模式的 LoRA 必须用 NCCL broadcast（出处：`protocol.py:73-89`、`protocols/broadcast.py:27`、`protocols/cuda_ipc.py:43`、`arguments.py:3581/:3595`，见 upstream-mechanisms.md E0；对训推 logprob 差异的影响属推断，待 GPU 核实），与 colocate 的 CUDA IPC 不同（A5） | △ |
| 新 engine 准入（payload 校验后才进 router） | infra 3.5；fork M4（3.5a） | 无 | 现实现（9ba38f6d）在 `end_update_weights` 内注册 router，`check_weights` 同锁，只能进 router 后读回 | 按现实现达不到 3.5 验收；3.5a 已写明两种补充机制（cordoned 加入后 uncordon，或 M4b 延迟准入），未实现 | ✖ |
| 端口 `TrainerGroup.save_cut/restore_cut/rebuild` | infra 4.2/4.3；M5（4.2a）、M6（4.6a） | 无直接消费者；算法状态须进入 cut | `save_cut(*,epoch)->id`、`restore_cut(id,*,epoch)`、`rebuild(plan)` | 缺算法相关状态，已补（A3） | △ |
| group/batch/update 账本 | infra 3.6 | 1b 动态过滤/超采样/overlong 过滤；2a 多段 rollout | `prepared→optimizer_applied→outer_recorded`，新增终态 `filtered` | 原先无法区分有意丢弃与丢失，已补（A2）；审查 F5 进一步区分终态 `filtered` 与非终态余量 `carried_over`，Miles buffer 回收行为在 4.1 核实；与 1b 共用 `record_trained_groups` hook | △ |
| 完整 cut 状态 | infra 4.1/4.2 | 1b KL loss（ref 模型）、1b/2a 插件（PluginRef、`yeto_algo_plugins`）、P0 算法哈希 | cut manifest 记录 `algorithm_spec_sha256` 与插件哈希，不一致就拒绝恢复 | 已补（A3） | △ |
| E3 变 DP 与算法归一化 | infra 4.6/4.8 | 1b token 级聚合与 Dr.GRPO 常数分母（CP>1 拒绝）；2a `--normalize-advantages`（DP 组内 all-reduce）与 GSPO；2b GMPO（CP 全收集） | 认证绑定 `algorithm_spec_sha256` | 已补（A4）。GRPO 以外的算法是否认证属于预算问题，待批准 | △/✖ |
| IslandDriver 挂载点 | `driver.py` `_generate`（policy token 检查）、`_check_gradient`（零梯度）、`phase()` 事件 | P0 4.2 `expects_gradient`；lr-fix 2.1 学习率指标；infra 1.7/2.2/3.1 | 单文件 | 写入冲突，需按顺序单写（§7） | ✖（顺序已定，待派发） |
| bridges 挂载点 | `bridges.py` outer phase | infra 1.5/3.1/3.8 | — | 算法 change 不改 bridges（P0 不改 bridge 元数据）。无冲突 | ✔ |
| miles_adapter 挂载点 | `config.py` 翻译、`entry.py` 能力声明、`rollout_meta_hook.py`、`state_plugin.py`（`run_plugin`） | P0/1a/1b/2a/2b（翻译与声明）；infra 2.1（LoRA 分区参数）、3.x、4.2（cut 经 `run_plugin`） | — | `config.py`/`entry.py` 多方写入；`run_plugin` 被 cut 与 lr-fix 学习率记录器共用 | ✖（负责人待批准，§8-4） |
| logprob 与训推 mismatch 数据通路 | SGLang `return_logprob=True` → Miles custom-tis 函数 → 指标 | 1a 全部；1b OPSM rollout 来源；P0 拒绝矩阵 | `EngineCapabilities.execution.rollout_logprobs=True` | 分区模式改变权重传输方式，1a 的 G2 结论不能外推（A5）。infra 3.5 可以用 1a 的只观测指标作新 engine 权重正确性的辅助信号（建议，不作为验收） | △ |
| 序列级 / 优势挂载点 | 1b reward 分派器（`--custom-reward-post-process-path`）；Miles loss 路径 | 2a（依赖 1b 分派器）；2b（路线 A 用 custom loss，路线 B 走 fork） | — | 与 infra 无接口交叉；只有 E3 认证（A4）会涉及 | ✔ |
| 确定性对比 | infra X3/4.3、4.6（Modal `H100!:N`，同一次租用内跑两个 arm）；1a 2.3 CPU `torch.equal` | 1a/1b/2a CPU 逐元素对照；各算法 G1/G3 | — | 口径一致。算法 G1/G3 在 serial-colocated 下跑，不依赖 infra 新能力 | ✔ |
| 能力认证格式 | PR #66 `capabilities.py`（infra 1.6 已扩展） | P0 3.1（新维度作为顶层新键，`attestation_from_dict` 忽略未知键） | 同一份 attestation | 一致；P0 3.1 的验收已要求旧字段读取不变 | ✔ |

## 3. 修订清单

每条修订单独提交，但并非都能单独 `git revert`：多条修订改了 tasks.md 中相同或相邻的行，git 三方合并会冲突。回退规则：

- 最稳妥的做法是按提交的逆序回退，回退到目标修订为止。
- A9 改了 2.1/3.3/3.4/3.5/4.2/4.6/4.7 的依赖括号，与 A2–A5、M1–M6 的行相同或相邻：回退 A2–A5 或 M1–M6 中任何一条之前，先回退 A9。
- F1–F3 改了 M1–M6 新增的 3.3a/3.3b/3.5a/4.2a 行及 3.7 行：回退 M1–M6 之前，依次回退 F3、F2、F1，再回退 A9。
- A6、A7、A8、A10、R0-PR 与其他修订不共用行，可以单独回退。F5–F9 的回退规则见各自说明（F5 与 A2 同行，回退 A2 前先回退 F5）。
“验收映射”一列说明该修订对应哪个验收项，或给哪个验收项增加了条件。所有修订都没有放宽验收，也没有改变算法语义。

| ID | 提交 | 文件 | 修改前 → 修改后 | 理由 | 验收映射 | 受影响 task / 代码 |
|---|---|---|---|---|---|---|
| C0 | `5753e30` | 5 个算法 change 与 docs/research | 原样复制 | 便于审阅后续 diff | — | — |
| R0-PR | `c122f1a` | rl-engine-ports/progress.md；rl-infra-spec/design.md Risks | “向 radixark/miles 提 run_plugin PR”；“评估提交 upstream” → 作废说明；永不向上游提 PR | 用户指示 | — | 无代码 |
| M1–M6 | `15e864d` | rl-infra-spec/tasks.md | 无 → 新增 2.1a/3.3a/3.3b/3.4a/3.5a/4.2a/4.6a，并修改表头 | 按 upstream-mechanisms.md 的缺口，以审查后的 tasks-m1-m6.patch 为底稿（用户已授权） | 各 fork 任务都要求 CPU 单测，外加对应 GPU 验收（X2/X3/X4/4.7） | Miles fork `yeto-elastic-m1-m6`；`ports.py` |
| A1 | `919bfbb` | infra tasks 1.4、design D0、spec“显式执行模式”（新增场景）；P0 design D4 | 两套 staleness/契约 → 契约哈希 = `AlgorithmSpec` 哈希；`max_policy_age ≤ max_policy_staleness`；执行能力由执行模式给出 | 消除重复定义；把“严格 profile 禁止偷跑”与算法要求连成一条校验 | X9；P0 spec“陈旧度要求超出执行模式”；新场景“执行模式年龄超出算法要求” | `execution_profile.py`（需增加 `algorithm_spec_sha256`）、`capabilities.py`、P0 3.4 |
| A2 | `5620d8e` | infra tasks 3.6、design D3；1b design D7 | 账本只有消费/丢失 → 增加终态 `filtered`；共用 `record_trained_groups` hook | 动态过滤、超采样、overlong 过滤是合法丢弃，否则会触发“无静默丢样本”误报，或者让真正的丢失被掩盖 | 3.6 验收（更严格，不放宽） | `rollout_meta_hook.py`；1b 6.3 |
| A3 | `538586b` | infra tasks 4.1/4.2、design D5 表 | cut 状态族 → 增加 `algorithm_spec_sha256`、插件哈希、ref 模型身份；不一致就拒绝恢复 | KL loss 需要 ref 模型，插件配置经 runtime attrs 下发，缺少这些状态时恢复后的目标可能静默改变 | 4.1、4.2 验收（增加拒绝条件） | `state_plugin.py`/cut 插件；1b D3/D5 |
| A4 | `18aa6e7` | infra tasks 4.6、design D5 | DP 边的认证没有算法维度 → 认证绑定 `algorithm_spec_sha256`，改变归一化的机制须单独认证 | token 级聚合、常数分母、DP 组内白化、序列级 CP 收集都与 DP/CP 布局有关 | X4 | 4.6/4.7 白名单；1b D2、2a D3、2b D3 |
| A5 | `4102b4c` | infra tasks 1.7/2.2；1a design G2 | 算法指标无标签，分区模式未写 policy token 校验 → 算法指标带 profile/epoch/传输标签；2.2 保留每组 policy token 校验；G2 报告注明执行模式与传输方式，不外推 | 1a 的前提（π_behav=π_old）在分区模式下必须仍然成立；LoRA 分区必须用 broadcast（出处：`protocol.py:73-89`、`protocols/broadcast.py:27`、`protocols/cuda_ipc.py:43`、`arguments.py:3581/:3595`，见 upstream-mechanisms.md E0；对训推 logprob 差异的影响属推断，待 GPU 核实） | 1.7、2.2 验收（更严格）；1a 7.4 报告内容 | `driver.py`、`timeline.py`、`miles_adapter/rollout.py` |
| A6 | `aab5ea8` | P0 proposal 能力表；1a design D1 | “需要 rl-infra-spec 提供异步模式” → “需另立独立算法契约 change” | 与 infra 2.3、design D0 一致，避免误派 | 无验收变化 | 无 |
| A7 | `12d01b5` | 2b design Migration | 路线 B 自行更新 pin → 与 M1–M6 串行合回 `yeto/ports`，只由 Agent IMG 更新 pin/digest | 同一个 fork 分支、同一个 pin，只能有一个写入者 | 2b 4.3 | `yeto/rl/__init__.py` pins |
| A8 | `ffe1141` | 4 个算法 tasks.md 的执行约定 | 在 `/home/michael/work/rl-algos` 工作 → 以本分支副本为准、按工作包派发；P0 增加 `driver.py` 单写说明 | 规划来源唯一；避免与 lr-fix、infra 在同一文件冲突 | 无 | `driver.py` |
| A9 | `a7ac491` | infra tasks 依赖 | 2.1/3.3/3.4/3.5/4.2/4.6/4.7 的依赖中没有 fork 任务 → 加上 2.1a/3.3a/3.3b/3.4a/3.5a/4.2a（LoRA）/4.6a；3.4a 从“同步”改为“前置” | 让派发顺序可以机械推导 | 无验收变化 | — |
| A10 | `95f4cd6` | infra tasks 第 5/6/7 组标题 | 未标注边界 → C 只做 5.1 选中的路径；D1=6.1–6.3，D2=6.4–6.7；F 本轮只交付 7.1/7.2 设计文档，7.3 只写占位 | 用户指示与派发需要 | 不放宽：未选中的 5.x 与 7.3 保持未勾选 | — |
| F1 | `53b5bc4` | tasks 3.5a；alignment 矩阵 | 3.5a 只写读回校验 → 如实写明现实现在读回前已进 router，给出 cordoned 加入后 uncordon 或 M4b 延迟准入两种机制；矩阵新增 ✖ 行 | 审查 F1 | 3.5 验收不变 | fork M3/M4 |
| F2 | `2ccfb78` | tasks 4.2a | 列出 M5 的全部拒绝条件，写明对 E2/E3 profile 的影响，以及 `keep_on_dp_change` 的种子映射 | 审查 F2 | 4.2/4.6 增加拒绝条件 | 1.2 profile 选择、4.1 |
| F3 | `a89290c` | tasks 3.3a/3.3b/3.7 | 按 M2/M3 实际语义写入 stop 半失败 `incomplete`、`wait_cells_tracked`、registration generation；3.7 覆盖该状态 | 审查 F3 | 3.7 增加失败项 | fork M2/M3 |
| F4 | `85adf81` | alignment §3 | “可单独回退” → 如实给出回退顺序 | 审查 F4 | — | — |
| F5 | `e6c10c7` | tasks 3.6/4.1、design D3、1b design D7 | `filtered`（终态）与 `carried_over`（非终态余量）分开；4.1 核实 Miles buffer 回收行为 | 审查 F5 | 3.6 更严格 | `rollout_meta_hook.py` |
| F6 | `44c3749` | design D0、tasks 1.4、P0 design D4 | 本 change 所有模式（含 overlap）年龄均为 0；大于 0 只能来自另立 change | 审查 F6 | — | — |
| F7 | `c274306`、`0b0d202` | 1a design G2；alignment | broadcast 论断补 file:line；logprob 影响标为推断 | 审查 F7 | — | — |
| F8 | `5a2e42f` | progress.md | 删除“M1–M6 未写入”这一过时陈述 | 审查 F8 | — | — |
| F9 | `680edf0` | tasks 第 7 组、design D12、alignment §5 | 7.3 注明出处，改为占位段并覆盖原三个要点；范围延后，待用户审阅 | 审查 F9 | 7.3 保持未勾选 | — |
| GP | `b3b975d` | gpu-plan.md（新增）、progress、alignment | 纳入版本管理；“D1” → “DEV-GATHER” | 避免与阶段 D1 撞名 | — | — |
| DEC | `697e38a` | alignment §7b/§8 | 记录主 agent 决定 | 主 agent 指示 | — | — |

## 4. D1 / D2 划分

- **D1 = 6.1–6.3**：shadow 负载归因与收益预测（X8）、半自动建议（带有效期、expected_epoch、profile hash）、执行前重验。依赖 1.7、2.4、5.7。
- **D2 = 6.4–6.7**：默认关闭的 auto 模式、disabled/manual/recommend/auto 切换、四场景对比、综合故障/学习/回退验收。依赖 D1 通过，以及至少一条实测净收益边（6.4 原文）。
- 如果 2.4 或 5.7 给出“尚无净收益边”的合法否定结论，D2 不开工，系统保持 manual/recommend，该结论照原文交付。
- 命名冲突已处理：GPU 计划原先用“D1”表示 E3 的 DistOpt gather 开发调试，已改名为 “DEV-GATHER”。计划已纳入版本管理，见 [`gpu-plan.md`](gpu-plan.md)（草稿源 `/home/michael/work/infra-drafts/gpu-plan.md` 同步修改）。

## 5. C 与 F 的范围

- **C（第 5 组）**：5.1 实测后只选一个瓶颈路径（host staging 对应 5.2、通信对应 5.3、进程复用对应 5.4、延迟恢复对应 5.5）；5.2–5.5 只做被选中的那一项。未选中的项记为“未选中”并附 5.1 证据，不勾选。5.6/5.7 按原文执行；5.7 允许“无收益不默认启用”的否定结论。
- **F（第 7 组）**：7.1（IslandStatus/pool epoch/pause-budget/veto）与 7.2（运行中扩池/缩池的顺序）只交付设计文档，不实现，不做实际云扩缩。
- **7.3 岛间扩展（后续占位）**：范围延后，待用户审阅。出处：用户 2026-09-29 无人值守轮指示：F 只做单岛 7.1/7.2 设计，7.3 跨岛分配、多云、DiLoCo 成员变更留待以后，只写后续占位。占位段见 design D12，已覆盖原 7.3 的三个要点。原因：
  - 首轮是单岛、固定 DiLoCo 成员；
  - 跨岛分配与成员变更需要修改外层协议；
  - 需要与在途的云接入 change（`add-nebius-verda-modal-clouds`、fix-verda-provider）对照，而后者尚未合入；
  - Verda 与 Modal 暂时不能承载 head。

  前置条件：
  - E1 的 3.8 通过（含两小岛 strict 暂停 X6）；
  - 7.1/7.2 的设计文档通过评审；
  - fix-verda-provider 合入；
  - 用户批准在线云扩缩的范围。

  7.3 保持未勾选。

## 6. M1–M6 映射

| fork | 任务 | 前置于 | 上游缺口（upstream-mechanisms.md） | 草稿状态 |
|---|---|---|---|---|
| M1 显式 role→bundle 映射 | 2.1a | 2.1、4.7 | E0 “Missing for 2.1” | 本地分支 `yeto-elastic-m1-m6`，正在另行独立审查 |
| M2 公开 cell 启停 + epoch | 3.3a | 3.4 | E1 fork-M2 | 同上 |
| M3 router cordon/in-flight/drain | 3.3b | 3.3、3.4 | E1 fork-M3 | 同上 |
| （yeto）ports.py 预留注释 | 3.4a | 3.4 | “Gaps: port verbs” | 未做（INFRA 负责） |
| M4 成员过滤的权重发布 | 3.5a | 3.5 | E1 fork-M4；没有 payload 级 ACK | 同上 |
| M5 LoRA DP 不变 optimizer + RNG | 4.2a | LoRA 的 4.2/4.6 | E2/E3 LoRA 缺 RNG、不能重分片；DistOpt gather 未实现，完成前 LoRA+DistOpt 的 E3 为 no-go | 同上 |
| M6 cell 重绑 bundle + trainer handle 重建 | 4.6a | 4.7 | E3 GPU handoff | 同上 |

上线顺序：独立审查 → 合回 `yeto/ports` → Agent IMG 更新 `MILES_NEXT_COMMIT` 与镜像 digest → 生成 1.1 runtime manifest → 才能声明对应的 capability。2b 路线 B 的 fork 提交与此串行（A7）。

## 7. 执行顺序与工作包

### 顺序原则

1. 先固定被算法依赖的 infra 接口：A1（执行能力 staleness 与 `algorithm_spec_sha256`）、A2（账本终态 `filtered`）、A5（指标标签）。这些已在文档中定稿。代码落地时，P0 3.4 声明的 `execution` 值必须按 A1 的规则给出。
2. P0 的注册入口先于四个子 change。1a 与 1b 可以并行；2a 在 1b 的分派器完成后开始；2b 等用户选定路线。
3. 不依赖 infra 新能力的算法工作与 infra 并行。所有算法的 CPU 工作和 G1/G3 都在 serial-colocated 上进行，不等 E0。
4. `driver.py` 按以下顺序由单一写入者提交：lr-fix（已提交，待合入）→ P0 4.2（只改 `_check_gradient`）→ infra 1.7 事件发射 / 2.2 / 3.1。

### 工作包

| 包 | 负责人（建议） | 内容 | 可修改路径（唯一写入） | 依赖 |
|---|---|---|---|---|
| WP0 | Agent ALIGN（本轮） | 规划对齐 | `openspec/**`、`docs/research/**` | — |
| WP-LR | 现 lr-fix 负责人 | fix-decoupled-lr-schedule 3.2（head 模式 GPU 实跑），然后合入 | lr-fix 分支已列路径；`driver.py` 在合入前归它 | GPU 批准；head 放置（memory：Verda/Modal 不能承载 head） |
| WP-VERDA | 现 verda 负责人 | fix-verda-provider 4.x（等 PR #69）、6.x 真实运行 | 该分支已列路径 | PR #69 合并（需批准） |
| WP-CAP | Agent ALGO-CAP | P0 全部任务 | `yeto/rl/engine/algorithm.py`、`capabilities.py`、`miles_adapter/algorithm_flags.py`（新）、`miles_adapter/entry.py`（能力声明）、`miles_adapter/config.py` 中的 `translate_run_config`/`check_extra_argv`/`ADAPTER_OWNED_FLAGS`、`engine/fake.py`、`yeto/rl/export.py`（ports 分支）、`yeto/rl/learner.py`（ports 分支参数）、`yeto/launcher.py`（算法哈希下发）、`driver.py`（仅 `_check_gradient`，在 WP-LR 合入之后、WP-INFRA2 之前）、对应 tests、P0 的 `docs/MILES_RL.md` 小节 | rl-engine-ports 基底；WP-LR 合入（仅 4.2） |
| WP-1a | Agent ALGO-MIS | 1a | 新文件 `yeto/rl/algos/mismatch_observe.py`、`yeto/rl/algos/vendor/**`、tests；共享注册文件（algorithm.py/algorithm_flags.py/entry.py/fake.py）的改动交 WP-CAP 负责人合并，或在 WP-CAP 完成后按时间片独占 | WP-CAP |
| WP-1b | Agent ALGO-KNOBS | 1b | 新文件 `yeto/rl/algos/reward_pipeline.py`、`reducers.py`、`examples/rl_algorithms/**`、tests；`rollout_meta_hook.py` 的过滤标记（6.3），须先于 infra 3.6；共享注册文件同 WP-1a | WP-CAP |
| WP-2a | Agent ALGO-SEQ | 2a | 在 `reward_pipeline.py` 注册变换（1b 完成后由 2a 接手该文件）、tests；共享注册文件同上 | WP-1b 分派器 |
| WP-2b | Agent ALGO-LOSS | 2b | 路线 A：`miles_adapter/loss_variants.py`；路线 B：Miles fork `yeto/ports`（与 WP-FORK 串行） | 用户路线决定；WP-CAP |
| WP-FORK | 独立审查后的负责人 | M1–M6 审查与合回 `yeto/ports` | Miles fork | 独立审查 |
| WP-IMG | Agent IMG | pins、镜像 digest、launcher/harness 镜像接线、1.1 manifest 的镜像部分 | `yeto/rl/__init__.py` pins 与镜像接线 | WP-FORK；2b 路线 B |
| WP-INFRA1 | Agent INFRA（第一批，只做不涉冻结文件的部分） | 4.1 审计文档、7.1/7.2 设计文档、1.3 计划定稿、1.8（等 PDF） | `openspec/changes/rl-infra-spec/**`（在 ALIGN 结束后接手）、新增 `yeto/rl/engine/*.py` 纯模块 | 无 |
| WP-INFRA2 | Agent INFRA（第二批） | 1.4 接 `algorithm_spec_sha256`（A1）、1.7 事件发射、2.1/2.2/2.3、3.x、4.2–4.5 | `driver.py`、`bridges.py`、`ports.py`、`miles_adapter/{rollout,placement,publish,trainer,state_plugin,rollout_meta_hook 的账本部分}`、新增 `controller.py` | WP-LR 合入、WP-CAP 的 driver 改动已落地、WP-IMG 镜像、1.2 baseline |
| WP-GPU | 主 agent 批准后指派 | 1.1/1.2、E0–E3 实验（按 gpu-plan） | 证据目录 | 预算批准 |

同一文件在同一时刻只有一个写入者。共享注册文件（`algorithm.py`、`algorithm_flags.py`、`entry.py`、`fake.py`、`docs/MILES_RL.md` 的算法小节）在 P0 完成后仍归 WP-CAP 负责人；子 change 以补丁形式提交，由该负责人合并，或者主 agent 按时间片轮流授予独占。

## 7b. 主 agent 决定（2026-09-29，可被用户推翻）

1. **GPU 预算**：用户已授权“执行必要 GPU 实验不再逐轮批准”，原待批准项“GPU 预算”删除。预算与每轮计划仍按 [`gpu-plan.md`](gpu-plan.md) 和 BRIEF 的 GPU 规则（事先写计划、容差与硬超时，并有回收机制）执行。
2. **推送授权**：用户已授权推送 michaellchung/yeto、michaellchung/miles、michaellchung/sglang 的功能分支。算法 tasks 中“commit 和 push 需要用户确认”按此解释为已授权，原文不改。
3. **`miles_adapter/config.py` 与 `entry.py` 按函数划分**：ALGO-CAP 拥有 `config.py` 的算法翻译函数（`translate_run_config` 的算法部分、`check_extra_argv`、`ADAPTER_OWNED_FLAGS`）以及 `entry.py` 的能力声明；INFRA 拥有两文件的其余部分。同一时间只有一个写入者，由主 agent 调度。
4. **`driver.py` 的唯一负责人是 INFRA**。P0 4.2 对 `_check_gradient` 的改动由 ALGO-CAP 以补丁形式交给主 agent，主 agent 在 INFRA 空档合入。§7 顺序原则第 4 条与 WP-CAP 行中“`driver.py`（仅 `_check_gradient`）”以本条为准。
5. **E3 认证范围**：首轮只认证默认 GRPO 属于缩窄认证范围，仍保留为待批准（§8 第 3 项）；在批准前，E3 先按默认 GRPO 执行。
6. **infra 代码工作的基底**：集成分支 `rl-integ` = `rl-infra-spec` 6fca4a8 + `fix-decoupled-lr-schedule` 63ea45a（merge `c5e05f4`，已推送；CPU 失败集合按测试 id 与基线相同）。WP-INFRA2 与 WP-CAP 的代码改动基于 `rl-integ`；规划文档仍在 `rl-infra-spec`。

7. **rl-algo-grpo-knobs 的解析口径**（ALGO-1b 追加）：2.3、4.3、7.1 在钉住镜像中运行完整 `parse_args` + `validate_parsed_args`，miles 解析器用 0394715，megatron 用镜像自带的。这比原文要求的 "miles-next-venv"（该环境缺 `megatron.training`）更严格，视为满足原文意图，完成记录中注明环境差异。5.2 另在钉住镜像内用镜像自带的 Miles 补跑一次 equivalence，作为同源证据。

8. **rl-algo-grpo-knobs 的能力声明标准**（主 agent 决定，ALGO-1b 追加）：只有 GPU 上确实生效的机制才声明。首批声明 token、drgrpo（constant 聚合、custom_pg_loss_reducer、no_grpo_std_normalization）、kl_k3（kl_placements:loss、kl_loss_ref_model）、entropy_bonus、overlong_penalty 与 reward_postprocessors:custom_reward_postprocess。clip_higher、dual_clip（clipfrac 为 0）和 over_sampling（没有补采）暂不声明，等拿到"clipfrac>0 / 补采次数>0"的证据后再声明。

## 8. 待批准事项

1. （已删除：GPU 预算，见 §7b 第 1 条。）
2. **2b 路线 A/B 的决定**；若选 B，还需同意向 `michaellchung/miles` 的 `yeto/ports` 提交（2b 1.2）。
3. **E3 认证范围**：A4 使变 DP 的认证绑定到算法描述。建议首轮只认证默认 GRPO，其他算法请求 E3 边时拒绝。批准前 E3 先按默认 GRPO 执行（§7b 第 5 条）。
4. **`miles_adapter/config.py`、`entry.py` 的归属**：BRIEF 规定 `miles_adapter/*` 归 INFRA（第二批），但 P0 必须修改这两个文件的算法部分。已由主 agent 按函数划分决定（§7b 第 3 条），不再待批准。
5. （已由主 agent 决定，见 §7b 第 2 条。）
6. **1.8** 需要 DynaResize 原文 PDF（本地没有）。
7. **lr-fix 与 fix-verda-provider 的实际状态与用户所说“已完成”不符**：
   - lr-fix（`fix-decoupled-lr-schedule`，HEAD 63ea45a）为 9/10，3.2 两岛 decoupled 需要在 head 模式实跑；
   - fix-verda-provider（`fix-verda-provider-r0`，HEAD 3091721）为 11/19，4.x 等 PR #69（rl-engine-ports → agentenv/yeto main，仍 OPEN），6.x 真实运行未做。

   本轮没有重复实现这两项。PR #69 的合并需要另批。
8. **M1–M6 合回 `yeto/ports` 与镜像重建**：等独立审查结论。
9. **F 7.3 延后**（用户指示，已记录）；实际云扩缩和岛间扩展的实现需要另批。

## 9. validate 与推送

- `openspec validate <change> --strict`：6 个 change 的结果见本分支 rl-infra-spec/progress.md 的 2026-09-29 ALIGN 条目。
- 推送：普通 push 到 `origin rl-infra-spec`（不强推）；推送的 HEAD 即包含本文件的提交。

## 10. Agent INFRA 追加（2026-09-29，分支 `infra-a`）

- **A1 已落地为代码**（bf47641/115ee9a/e4d227a）：`ExecutionProfile.algorithm_spec_sha256`、`check_algorithm_contract`、`execution_max_policy_staleness`（所有模式均为 0），ports 入口在 GPU init 之前执行 preflight。读取 `AlgorithmSpec.execution.max_policy_staleness` 时按 P0 design 的字段路径以鸭子类型处理，v1 spec 视为 0。**需要与 ALGO-CAP 对齐**：v2 `ExecutionSpec` 冻结后，确认字段名与类型。
- **A4 已落地为 schema**：certified edge 可以带 `algorithm_spec_sha256` 列表；trainer-dp/role-transfer 边只对列表中的哈希有效。
- **A5**：driver 在 `observe=True` 时给每轮算法指标打 profile/epoch/权重传输标签（colocated 为 `cuda-ipc`，fixed-partition 为 `nccl-broadcast`）；2.2 保留每组 policy token 校验。
- **§8 第 6 项（1.8 缺原文）已解除**：原文为 arXiv:2607.22614，假设见 `dynaresize-hypotheses.md`。关键发现：论文以 one-step-off-policy 异步流水线为前提（p.2–3），与本 change 的 age 0 不同，因此论文结论不能外推，也不能作为开放 overlap 的依据。
- **entry.py 的能力声明**：infra 的 placement/mode 声明放在单独的 `with_partitioned_serial()` 中，没有修改 ALGO-CAP 负责的 `miles_capabilities()`。

## 追加待批准（Agent ALGO-CAP，2026-09-29）

- **P0 D11 放行开关与 G1**：按 spec 原文，“多岛或外层同步”即拒绝。当前 learner 与 launcher 两个入口都带外层同步，所以放行开关在真实运行中无法使用，各算法 change 的 G1 也就无法借助它运行。需另批二选一：(a) 允许单岛带 1 成员 syncer 时放行；(b) 新增无 syncer 的单岛 learner 入口。ALGO-CAP 没有自行放宽。
- **更新**：新增了单岛无外层同步运行模式（主 agent 决定，用户可推翻，见 §7b）；放行开关口径未放宽。ALGO-CAP 已实现 `--rl-single-island-no-sync`（单岛、无 syncer、无外层同步），放行开关只在这个入口上可用；F9 的严格拒绝保持不变。

## 11. Agent INFRA 追加（2026-09-29，fork 事实更新）

- §2 矩阵中"新 engine 准入（payload 校验后才进 router）"一行原为 ✖。现状：fork `yeto-elastic-m1-m6` 1a68f893 已实现方案 (a)（`start_update_weights(admit_cordoned=True)` → 调用方 `check_weights` 读回 → `admit_cells(cell_ids, expected_epoch)`）。该行改记为 **△（机制已在 fork 实现，CPU 单测；第三轮独立审查中，未合回 yeto/ports、未进镜像，3.5 GPU 未验证）**。3.5 的验收不变。
- M5（4.2a）现在支持 bf16 与 DistributedOptimizer，但 GPU 未验证，E3 LoRA+DistOpt 仍按 DEV-GATHER 的结论处理；M6 rebuild 失败语义（`TrainerRebuildError`，不自动回滚）已写入 4.6a，3.7/4.7 失败矩阵需覆盖。

## 12. 待批准追加（2026-09-29 INFRA，来自 f-design.md 接口缺口）

以下三项需要用户另行批准，本轮不实施：

- **G4 运行中云扩缩与节点追加**：包括按 ID 释放资源。这属于实际的云扩缩实现，按 BRIEF 需要另批。
- **G6 运行中增删 bundle/cell**：需要新的 Miles fork M 项（现有 M1–M6 只支持启动时预声明的 bundle/cell），需要另批。
- **G11 fixed-roster 重启与扩缩后的池形状**：首版重启时恢复到启动时的池形状；在支持按新形状重建之前，不启用扩池。需要确认该约定。

## 7b 追加（2026-09-30）
- 2026-09-30 用户：2b 选路线 B；暂停 GPU 验证，待自有卡本地验证；同意 INFRA 按阶段拆分并行（INFRA-E1: 3.x；INFRA-E2: 4.1–4.5；INFRA-A: 2.3/1.4/1.7/5.1；DEV-GATHER 延后）。
