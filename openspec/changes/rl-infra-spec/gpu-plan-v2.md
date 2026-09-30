# rl-infra-spec GPU 验收计划 v2：A1–A9，硬上限 $300（2026-09-30，PLAN-V2；未启动任何资源）

取代 [`gpu-plan.md`](gpu-plan.md) 中 A1–A9 的部分（§2 F 阶段、§3 A1–A9 各行、§4 预算）。gpu-plan.md §6（A5 head 放置/NAT）和 §8（池身份、租用记录字段、回收流程）继续有效，本文件引用它们。

## 0. 用户决定与硬约束

- **2026-09-30 用户决定**：GPU 验收先做 A1–A9（含 A4b、A6a、A6b、DEV-GATHER），**A10（4.8 学习验证）暂不做**，原则是"尽可能节省"。
- **2026-09-30 用户追加**：A1–A9 的全部 GPU 费用**硬上限 $300**。上限已经包含失败重跑，不再另加 25% 余量。每批有独立预算，累计达到上限前必须停下并报告。
- 4.8 不做：A10 与 4.8 的状态为"未完成（用户决定暂缓）"。按 4.8 原文，"未通过不开放 trainer 边"，暂缓期间同样**不开放 trainer 边**，也不得把 E3 记为 trainer 完成。4.7 即使 GPU 验收通过，也只证明"角色转移机制正确"，不代表 trainer 边可以开放。E1（rollout 边）可以独立交付。
- 规则沿用 BRIEF：状态只有五种；判据、容差和比较口径在运行前固定，事后不改；同一失败只有查明原因并修复后才重跑；不挑 seed；逐位比较用 Modal `H100!:N`，并在运行前断言 GPU 型号。

## 1. 口径与价格

- **单价**（均未核实当日价，下单前核对）：Modal `H100!` 为 $3.95/GPU·h（按秒计费），Modal L40S 为 $1.95/GPU·h，Nebius H100 为 $2.95/GPU·h（SkyPilot）。
- **时长依据**：evidence/infra-a 的实测数据。0.6B LoRA 在 4×H100! 上，partitioned/colocated 每轮 75–110 s（2.4 `RESULT.md`：T1R3 为 74.5/75.5 s，T2R2 为 89.3/86.9 s，C4 为 109.7/110.1 s）。一次 6 轮运行从 launch 到结束共 20–26 分钟，约 0.4 h，其中约 10 分钟是镜像拉取与预热（`2.4/*/start_utc.txt`、`end_utc.txt`）。1.2 的两岛 3 轮运行用了 15 分钟，2.1a 的 2 轮运行用了 12–13 分钟。本计划的估算规则如下：
  - 3 轮以内的运行：0.35 h；
  - 5–6 轮的运行：0.4 h；
  - 含两次切换的 6 轮运行：0.45 h；
  - 同一次租期内顺序执行多个实验时，只计一次 10 分钟启动。
- **GPU·h** 按整池卡数 × wall 计算（空闲卡和备用卡都计入）。
- **Modal 与 Nebius 的选择**：名义价格 Nebius 便宜 25%。但本 change 的全部 GPU 证据（1.1/1.2/2.1a/2.2/2.3 X9/2.4）都是在 "Modal island + 本机 head" 路径上得到的。"Nebius island + 本机 head" 在 yeto RL launcher 上从未跑通过：gpu-plan §6 记录了公网 IPv4 配额只有 3、head 放置不确定；VM 开通时间也要计费。首次使用需要一次验证冒烟，还要承担排错风险。非逐位行全部改用 Nebius 最多省约 $29（28.75 GPU·h × $1）。扣除一次验证冒烟（约 $3）和排错风险后，净收益不确定。因此**默认全部用 Modal `H100!`**，非逐位行改用 Nebius 作为用户可选项（§8 第 4 条）。
- **费用门控（硬上限的执行方式）**：每次 launch 前计算"本批已实际花费 + 本次运行的最坏费用（卡数 × 硬超时 × 单价）"。只要超过本批预算或全局 $300，就不启动，停下来报告。实际费用按 Modal 账单或按秒 wall 计算，记入 `rental.json`。
- **唯一前缀**：`infra-v2-<实验ID>-<UTC日期>-<序号>`，例如 `infra-v2-A4-20261001-1`。Modal app 名为 `yeto-infra-v2-…`，同时作为 `pool_id`（gpu-plan §8.2）。只操作这个前缀或已记录 ID 的资源。
- **回收机制**（每次都执行，参照 gpu-plan §8.4）：
  1. 云端限时：launcher 路径设置 `ModalIslandConfig.timeout_s` = 硬超时；自写 Modal 脚本或 Sandbox 用 `timeout=` = 硬超时。
  2. 外层包裹 `timeout <硬超时>`。
  3. 独立 watchdog（`setsid nohup`，与终端和 agent 无关），在硬超时后 5 分钟执行 `modal app stop -y yeto-infra-v2-<…>`（Sandbox 则执行 `modal container stop <id>`）。
  4. 结束或失败时，先拉日志、事件磁带、journal 和 ledger，再释放；用 `modal app list --json` 核实状态为 stopped 且 0 tasks，并存档；停止 puller 和 watchdog。
  - Nebius（仅在用户选择后使用）：`sky launch --down --idle-minutes-to-autostop 10`，watchdog 按集群名执行 `sky down infra-v2-…`，并用 `sky status` 和 Nebius API 核实。
- **镜像与 manifest**：使用运行当日集成分支 `MILES_NEXT_IMAGE` 的 digest。每批第一次运行时生成 1.1 runtime manifest 并与 pin 比对，不一致就不跑。
- **模型**：全部为 Qwen3-0.6B LoRA r16 all-linear、GRPO 默认 spec、bf16、TP=PP=CP=EP=1。唯一的例外是 E2 plan-v2 已事先登记的 C3（1.7B，见 A6b/A7 行）：该判据已提交，改动它等于改事先登记的配置，而两者费用差不到 $1，所以保留。

## 2. 逐项计划

"卡"一列均为 Modal `H100!`，另有注明的除外。费用列为期望值，Nebius 价格写在括号里。

| ID | task / 验收原文 | 复用证据 | 配置与卡 | 期望时长 | GPU·h | 期望费用 | 硬超时 |
|---|---|---|---|---|---|---|---|
| A1 | 1.2 "完成一轮生成、更新、外层同步与权重确认，形成兼容baseline" | **已完成**：1.2 已勾选（`evidence/infra-a/1.2/attempt2`，2×H100!，两岛 strict-avg） | 不运行 | 0 | 0 | $0 | — |
| A2 | 2.3 X9 "延迟发布不能触发旧版本生成，队列有界"，以及合法重叠 eval‖train/outer_sync；2.1、2.2 的原文验收 | 2.1 已有 GPU 证据：T1R1、T2R2、T1R3、T1R1S1 固定分区启动，colocated 不变。2.2 第四轮 A=B 通过（`round4-check.json`）。2.3 X9 guard 第三轮 C 通过（`round3-check.json`，2 卡）。这些**都不重跑**，只剩 age 0 重叠部分 | local-gpu-plan **L-2.3** 原文：T1R1，arm S/O/OD 各 3 轮，seed 17；2 卡 | 3×0.35 h | 2.1 | $8.3（$6.2） | 每 arm 45 min |
| A2+ | 1.7（**不在 A1–A9 字面范围内，需用户确认**）：观测 GPU 验证 | — | local-gpu-plan **L-1.7** 原文：T1R1，W-tool / W-gen / W-gen observe=False；2 卡 | 3×0.35 h | 2.1 | $8.3（$6.2） | 每次 45 min |
| A3 | 2.4 "同profile公平比较、全池/备用GPU-hours和原始trace齐全，可得'尚无净收益边'的结论" | **合法否定结论已得出**（`evidence/infra-a/2.4/RESULT.md`，4×H100!，T2R2/T1R3/C4 × seed 17/29） | **不重跑** | 0 | 0 | $0 | — |
| F0 | 门：镜像能否在 sm_89 上运行（只为下面的 F-E1 服务） | — | 1×L40S，1.1 manifest 加一次 0.6B 单卡 1 轮 | 0.25 h | 0.25 L40S | $0.5 | 30 min |
| F-E1 | A4 首次上 GPU 前的集成冒烟（**不作为验收证据**） | — | 3×L40S，T1R1S1↔T1R2S0，一次 up/down，只看控制流是否跑通 | ≤1 h | 3 L40S | $5.9 | 75 min（整个租期） |
| A4 | 3.4 X2 "trainer不动、未占用池外资源，不等待故意停用engine，备用卡计入成本"；3.5 "旧generation ACK、错误payload、迟到请求都不能污染新配置"；3.6 "重试/部分组/publish失败无重复消费、无静默丢样本"（旁证）；3.7 "启动/通信/发布失败有界处理……stop 半失败 incomplete…" | 2.4 的 T2R2 数据只作时长依据；3.4a 已勾选 | **4 卡 T2R1S1↔T2R2S0**（原文点名 T4R2S2↔T4R4S0，**见 §2.1，待用户确认**）。E1-A：6 轮（up 发生在第 3 轮前，down 发生在第 5 轮前）；基线为固定 T2R1S0，3 卡 6 轮；E1-B：3 轮；E1-D：5 次运行 | E1-A 0.45 h ×4 卡；基线 0.4 h ×3 卡；E1-B 0.35 h ×4 卡；E1-D 5×0.35 h ×4 卡 | 1.8+1.2+1.4+7.0=11.4 | $45.0（$33.6） | 每次 60 min |
| A4b | 3.3 X5 "active请求为0但tool-wait>0时保留旧路由，超时取消切换而不重放外部副作用" | — | 同 A4 的 4 卡边，E1-C（工具固定 sleep 30 s，`T_drain`=5 s） | 0.4 h | 1.6 | $6.3（$4.7） | 60 min |
| A5 | 3.8 X6 "样本/step/policy/roster不变，quorum超时/PULL重发正确，finalization拒绝切换" | 1.2 的两岛 + 本机 head 路径 | **两小岛**：岛0 为 3 卡 T1R1S1↔T1R2S0，岛1 为 2 卡 T1R1，strict-avg，本机 head。三次运行：基线（两岛都是 T1R1，共 4 卡，6 轮）；切换加 finalization（5 卡，6 轮）；quorum（5 卡，`--quorum-timeout-s 120`） | 0.4+0.45+0.4 h | 1.6+2.25+2.0=5.85 | $23.1（$17.3） | 每次 60 min |
| A6a | 4.2 "坏checksum、截断、step不一致拒绝，源释放前恢复依据完整；……与当前运行不一致即拒绝" | — | **E2 合租**：一次 Modal Sandbox，`H100!:2`，按 `evidence/infra-e2/4.2-4.5/plan-v2.md` 顺序执行 G-4.2（C1、C2）→ G-4.3（C1、C2）→ G-4.4（C3）→ G-4.5（C3，6 项注入），外加 §3 的 L2 项 | 合计约 3.0 h | 6.0 | $23.7（逐位项必须用 Modal） | 整个租期 5 h；每项 90 min（plan-v2） |
| A6 | 4.3 X3 "训练2步后重建，对冻结下一batch比较RNG/计数/moments/参数更新，无额外reset" | — | 同上（同一容器内两个 arm，**逐位**） | 包含在上面 | — | — | — |
| A6b | 4.4 "不重复initialize/after_local_train，外层进度不因重建重放" | — | 同上（C3） | 包含在上面 | — | — | — |
| A7 | 4.5 "rank失败、collective超时、迁移中断、controller crash提交不确定均有界恢复或RECOVERY_REQUIRED" | — | 同上（C3）。4.5 依赖 3.8：A5 通过前可以先跑，但不勾选 | 包含在上面 | — | — | — |
| DEV-GATHER | 4.2a "DistOpt gather 实现后由 4.3 X3 与 4.6 X4 GPU 实验验证"（M5 v2：bf16 + DistOpt，CPU 已逐位） | E2 合租中 G-4.2/G-4.3 的 C2（DP=2、DistOpt）与 plan-v2 §3 的 L2 项就是 DP 不变情形下的 GPU 确认 | **不单独租卡**。DP 变化情形并入 A8 的状态转移检查。只有 C2 或 A8 暴露 gather 缺陷时才启用应急调试：若 F0 通过，用 2×L40S ≤3 h；否则用 `H100!:2` ≤1.5 h | 0（应急时 ≤3 h） | 0（应急 6 L40S / 3 H100） | $0（应急时 ≤$12） | 应急租期 3 h |
| A8 | 4.6 X4 "master/optimizer/RNG/样本映射和下一步数值比较给出go/no-go" | E2 合租中得到的 C2 DistOpt 保存/恢复结论 | `H100!:2` 一个 Sandbox，四个 arm（见 §3 A8） | 1.25 h | 2.5 | $9.9（逐位项必须用 Modal） | 整个租期 2.5 h |
| A9 | 4.7 "实际GPU从trainer转给rollout及反向，复杂并行维度固定，双向成功/失败恢复、batch语义和epoch都正确"（原文允许"P62↔P44或更小等价边"）。**只在 4.6 为 go 时运行** | 2.4 已证明 T2R2/T1R3 等 4 卡配置合法 | **3 卡 T2R1↔T1R2**，即 4.6 所认证的 DP2↔1 边，属于原文允许的"更小等价边"。运行：双向切换 6 轮一次、固定 T2R1 基线一次、失败恢复三次 | 0.45+0.4+3×0.35 h | 5.7 | $22.5（$16.8） | 每次 60 min |
| A10 | 4.8 | — | **不做：未完成（用户决定暂缓）** | — | — | — | — |

### 2.1 A4 缩小卡数（实质改变验收，待用户确认）

3.4 原文写的是"先测T4R2S2↔T4R4S0"。本计划依据用户"尽可能节省"的原则改用 4 卡 T2R1S1↔T2R2S0，被测机制完全相同：rollout 增减一个 engine，trainer 不动，备用卡计入成本。**这属于实质改变验收，由主 agent 依据用户的"尽可能节省"采用，需用户确认。** 确认之前，A4 的结论只能记为"4 卡等价边 GPU 通过"，不勾选 3.4。
- **8 卡原文版本**：A4 与 A4b 全部改为 8 卡，E1-A、E1-B、E1-D 与 A4b 用 T4R2S2↔T4R4S0，基线用 T4R2 6 卡，GPU·h 从 13.0 变为 26.0，**增加 $51.4**；F-E1 改为 7×L40S，**增加 $7.8**。合计**增加约 $59**。这一档直接使用 `evidence/infra-e1/plan.md` 的原判据，不需要换算。
- **更省的 3 卡方案**（可选）：T1R1S1↔T1R2S0，与 A5 岛0 的边相同。比 4 卡再省约 $13.2（13.0 → 9.65 GPU·h）。同样需要用户确认。
- 4 卡档对 `evidence/infra-e1/plan.md` 判据的换算（只换卡号和成员数，判定逻辑不变）：
  - 池 = 4 张 UUID；trainer 用 G0–G1，rollout 用 G2，standby 用 G3；启动时声明 cell c0–c1，只启动 c0。
  - E1-A (c)："第 3–4 轮 publication_members = 2，其余轮 = 1"。
  - E1-A (d)(e)："G0–G1"。
  - 轮数由 12 轮缩短为 6 轮（第 1–2 轮 R1，第 3–4 轮 R2，第 5–6 轮 R1），因为判据不依赖轮数。
- E1-D 的合并：①②③ 的终态都不中断训练，按事务序号合并为 1 次运行（D1）。④ 单独运行（终态为 RECOVERY_REQUIRED）。⑤⑥⑦ 各自单独运行。共 5 次运行，每项判据按原文不变。
  - 如果故障注入文件不支持按事务序号调度，①②③ 拆成 3 次运行，**增加约 $11**（从 B1 预算中支出）。
  - ⑤⑥⑦ 需要"kill learner 后用同一 `--rl-elastic-state-dir` 重启"。启动前必须在 CPU 上确认 Modal 路径的 state dir 位于持久卷上、能原地重启。不满足时，⑤⑥⑦ 判为**环境阻塞、不运行**，3.7 保持未完成，并如实报告。
- 已知风险（integ-s2 progress）：3.7 watchdog 默认没有接 kill（`on_watchdog` 未接线）。启动 E1-D 前如果仍未接线，涉及阻塞引擎调用的项可能超出 deadline。**不为此放宽判据**，失败即如实记录。 （**已过时**：watchdog kill 已由 2b67145 实现并合入；见 §8.7(2)）

## 3. 预先固定的判据（本文件提交后即冻结；没有在此写出的项，以所引计划原文为准）

- **A2 / A2+**：`local-gpu-plan.md` L-2.3 判据 1–6（含"追加"中运行前修正的口径）；L-1.7 判据 1–5。判据 6 不满足时，按原文处理为合法否定结论。
- **F0**：manifest 与 pin 一致；0.6B 单卡完成 1 轮，退出码 0。任一项不满足，判"镜像在 sm_89 不可用"，F-E1 和 DEV-GATHER 的 L40S 应急都取消（回退方式见 §5）。F0 与 F-E1 的结果**都不作为验收或性能证据**。
- **F-E1**：只看 up/down 两个事务能否到达终态并且没有 Python 异常。不设通过/失败判定，也不计入任何 task。
- **A4 / A4b**：`evidence/infra-e1/plan.md` E1-A…E1-E 原判据（4 卡档按 §2.1 换算）。全部为精确相等或精确出现，没有数值容差，每项 1 个 seed。 （8 卡原文档执行、E1-D 另加 watchdog 判据，见 §8.7(2)(5)）
- **A5（3.8 X6）**： （补充判据见 §8.7(1)）
  1. 切换运行中岛0 完成 up（第 3 轮前）和 down（第 5 轮前），两个事务的终态都是 `SUCCEEDED`。两岛的逐轮 `trained_sample_ids_sha256`、组数、样本数与基线运行逐轮相等（样本不变）。每岛每轮恰好一次 optimizer 步（step 不变）。两岛在每个 policy_version 上的 `sync/global_policy_hash` 相同，紧随其后的 publication token 与 apply 一致（policy 身份链不变，但不要求与基线的哈希相等，因为 rollout engine 数量不同，生成文本可以不同）。syncer 磁带中每轮 roster 都是 {岛0, 岛1}，与基线相同。
  2. quorum 运行：`--quorum-timeout-s 120`，岛0 的 up 事务在 `start_cells` 前注入 150 s 延迟。syncer 磁带中该轮至少有 1 次 PULL 重发，岛1 没有退出，roster 没有变化，该轮最终完成，没有 `rl_strict_failure`。
  3. finalization：在最后一轮（第 6 轮）的 finalization 阶段提交 up 请求，事务被拒绝（journal 中为拒绝或取消，原因是 finalization），config_epoch 不变。
  4. 运行前按 gpu-plan §6 做 30 分钟空闲流探测（只用 CPU，无 GPU 费用）。若测得路径会在 150 s 内丢流，quorum 用例判为**环境阻塞、不运行**，不改为其他网络路径冒充。
  5. 报告明确写明"仅完成 rollout 能力"。
- **A6a/A6/A6b/A7**：`evidence/infra-e2/4.2-4.5/plan-v2.md` 的 G-4.2、G-4.3、G-4.4、G-4.5 与 §3 L2 原判据（C1、C2、C3 不变）。确定性设置中任一项不可用即判为环境阻塞，不降级。 （A6b 补充判据见 §8.7(1)）
- **A8（4.6 X4）**：`H100!:2`，C2 配置（0.6B，bf16，DistOpt，GBS=16，确定性设置同 plan-v2 §0）。同一容器内依次运行以下 arm：
  - R：DP2 连续训练 3 步，第 3 步用落盘的冻结 batch；
  - D21：DP2 训练 2 步 → cut → 以 DP1 重建 → 恢复 → 在同一冻结 batch 上训练第 3 步；
  - D12：DP1 训练 2 步 → cut → 以 DP2 重建 → 恢复 → 第 3 步；
  - R1：DP1 连续训练 3 步（作为 D12 的对照）；
  - N：同 R，但 `NCCL_ALGO=Tree`（DP2 归约顺序的噪声底）。
  - go 必须同时满足：
    1. 状态转移精确：恢复后、第 3 步前，按参数名拼接的 FP32 master、exp_avg、exp_avg_sq、step 与 cut 前的聚合值 `torch.equal`；scheduler `num_steps` 相等。
    2. 样本映射精确：第 3 步全局 batch 的 sample ID 集合与顺序与对照 arm 相同，每个新 rank 的样本分配符合 manifest 记录的映射。
    3. RNG：manifest 记录了每个新 rank 的 RNG 来源（恢复或 fresh）及种子推导；没有记录即判 no-go。
    4. 下一步数值：设 Δ 为第 3 步 LoRA 参数更新量。rel-L2(Δ_D21, Δ_R) ≤ max(3×rel-L2(Δ_N, Δ_R), 1e-6)；D12 对 R1 用同一公式，噪声底取 rel-L2(Δ_N, Δ_R)。grad_norm 的相对差按同一规则判定；loss 在同一冻结 batch 上的相对差 ≤ 1e-3。
  - 任一不满足即为 **no-go**（合法否定结论，4.6 如实交付）。原因如果是实现缺陷而不是原理问题，在结论中写明，并按 §5 处理。
  - go 结论只对本次的 `algorithm_spec_sha256` 成立（默认 GRPO），不直接加入白名单。
- **A9（4.7）**：3 卡 T2R1↔T1R2，6 轮。 （**拓扑与资源被 §8.7(4) 取代**；以下原文保留）第 3 轮前 trainer→rollout（T2R1→T1R2），第 5 轮前反向。通过需同时满足：
  1. 两个事务终态都是 `SUCCEEDED`，config_epoch 为 0→1→2；GPU UUID 在 trainer 和 rollout 之间实际转移（按 bundle↔UUID 记录）；TP/PP/CP/EP 不变。
  2. 逐轮 sample-id 哈希、组数、GBS 与固定 T2R1 基线相等；每轮一次 optimizer 步；epoch 与数据游标连续，不回卷。
  3. 失败恢复三次运行：(i) 方向 T2R1→T1R2 时 `create_training_models` 抛 `TrainerRebuildError`，结果为用旧参数重建并从 cut 恢复，终态 `REBUILT_OLD`；有 cleanup_error 时判 RECOVERY_REQUIRED；(ii) 反向时新 rollout engine `start_cells` 被 kill，终态 `REBUILT_OLD`，trainer 以原 DP 继续，下一轮 token 校验通过；(iii) commit CAS 之后 kill controller 再重启，按 journal 对账，不回滚，也不重复消费。
  4. 三次失败运行都满足：`optimizer_applied` 不重复，没有组被消费两次。
- **每批通用**：在 `evidence/infra-v2/<ID>/<run>/` 下保存 rental.json（gpu-plan §8.3 的字段）、GPU 名断言、manifest、磁带、journal、ledger 和 RESULT.md（逐条对照判据）。

## 4. 批次、依赖与预算（硬上限 $300，已含重跑）

代码就绪情况（2026-09-30）：
- A2（2.3）代码已就绪，但经 launcher 运行还需要 eval 配置接线（进行中；integ-s2 已接入 `--rl-overlap-eval`，eval 源接线仍有缺口）。 （**已过时**：launcher eval 接线已合入集成分支；见 §8.7(3)）
- A4/A4b（3.3–3.7）已就绪。
- A6a/A6（4.2/4.3）已就绪；A7（4.5）部分就绪。
- 以下正在由其他 agent 实现：A5 需要 3.8，A6b 需要 4.4，A8 需要 4.6，A9 需要 4.7（且只在 4.6 为 go 时运行）。

| 批次 | 启动条件 | 内容（顺序） | 期望费用 | 本批硬预算（4 卡档） | 本批硬预算（8 卡原文档） | （按 8 卡原文档执行，见 §8.7(5)）
|---|---|---|---|---|---|
| B1 | A4 代码与 elastic 接线在集成分支上；A2 等 eval 接线完成（未完成时 A2/A2+ 顺延，不阻塞 A4） | F0 → F-E1 → A4（E1-A/基线/E1-B/E1-D）→ A4b；A2 → A2+ | 4 卡：$74.3（其中 A2+ $8.3）；8 卡：$133.5 | **$115** | **$170** |
| B2 | 3.8、4.4 代码合入集成分支；A4 已出结论 | A5（空闲流探测 → 基线 → 切换 → quorum）；E2 合租（A6a→A6→A6b→A7） | $46.8 | **$95** | **$80** |
| B3 | 4.6 代码合入，A7 已出结论；4.7 代码合入且 4.6=go 才进入 A9 | A8（内含 DEV-GATHER 的 DP 变化确认）→ go 时 A9；应急调试 ≤$12 | go：$32.4；no-go：$9.9 | **$65**（A8 加应急 $25，A9 $40） | **$50**（A8 加应急 $22，A9 $28） |
| 未分配 | 只有主 agent 向用户报告后才能动用 | — | — | $25 | $0 |
| **合计** | | | 4 卡：**≈$154**；8 卡：**≈$213**；4.6 no-go 时分别为 ≈$131 / ≈$191 | **$300** | **$300** |

- 以上期望费用都不含重跑，重跑从本批预算余量中支出：4 卡档余量约 $146，8 卡档约 $87。
- 期望费用的 GPU·h：H100 为 4 卡档 37.25（8 卡档 50.25），L40S 为 3.25（8 卡档 7.25）。
- 任何一批的"已花费 + 下次最坏费用"超过本批预算时，停下来报告，不从其他批挪用；B3 的 A9 子预算同样如此。
- **4.6 no-go 时的降级**：不运行 A9，B3 剩余预算不再使用。4.7、4.8 保持未完成；4.6 以合法否定结论交付。E3 trainer 边不开放，E1 rollout 能力可以独立交付。如果 no-go 的原因是 gather 实现缺陷，修复后**只重跑一次** A8，费用计入 B3 应急额度；超过额度就停下报告。
- **超出预算时先砍的顺序**（最后一项最先砍）：
  1. A9（依赖 4.6 go，每美元的产出最低）；
  2. 8 卡原文档退回 4 卡档（前提是用户已确认 4 卡）；
  3. F-E1 与 F0（便宜卡冒烟，不产生验收证据）；
  4. A2+（L-1.7）；
  5. A5 的 quorum 用例（只在路径会丢流时才被环境阻塞）。

  A2、A4/A4b、E2 合租和 A8 不砍。砍掉的项如实记为"未完成（预算）"，不降低判据。

### 4.1 按"每美元勾选/交付的 task 数"排序（期望费用，4 卡档）

| 顺位 | 项 | 期望费用 | 通过后可勾选或交付的 task（依赖链满足时） |
|---|---|---|---|
| 1 | A1、A3 | $0 | 1.2（已勾选）；2.4 的合法否定结论（勾选还依赖 2.2 与 1.7） |
| 2 | A2 + A2+ | $16.6 | 2.3，并解开 1.4 → 1.5/1.6 → 2.1/2.2 → 2.4 的勾选链，外加 1.7，最多约 8 项 |
| 3 | E2 合租 | $23.7 | 4.2、4.3、4.4、4.2a（部分），以及 A5 之后的 4.5，共约 4–5 项 |
| 4 | A4 + A4b | $51.3 | 3.3、3.4、3.5、3.7，以及 3.6 旁证；另有 3.3a/3.3b/3.5a 的 GPU 旁证 |
| 5 | A5 | $23.1 | 3.8，并解开 4.5 的依赖 |
| 6 | A8 | $9.9 | 4.6（go 或 no-go 都算交付）、4.2a 的 GPU 验证 |
| 7 | A9 | $22.5 | 4.7、4.6a（只在 go 时） |

实际执行顺序按代码就绪情况排批（§4 表），本排序只用于决定超出预算时砍哪些项。

## 5. F0 门与便宜卡

- 便宜卡只保留两处：F-E1 冒烟，以及 DEV-GATHER 的应急调试。
  - F-E1 的理由：E1 事务代码从未在 GPU 上运行，首跑集成缺陷的概率高。每次在 H100 上失败约花 $6（4 卡 × 0.4 h），换成 L40S 约 $2（3 卡）。
  - 其余项目直接按最小规模在 H100 上跑，因为 F 阶段的额外启动费用抵不过节省下来的钱。
- **F0 未通过时的回退**：取消 F-E1。A4 首跑直接上 H100!，排错费用从 B1 预算余量中支出；B1 预算不变，费用门控照常执行。DEV-GATHER 应急改用 `H100!:2` ≤1.5 h（约 $12）。

## 6. 需用户确认

1. **A4 用 4 卡 T2R1S1↔T2R2S0 代替原文的 T4R2S2↔T4R4S0**。这属于实质改变验收。8 卡原文档增加约 $59，两档都在 $300 之内。另有可选的 3 卡档，再省约 $13。
2. **A2+（1.7 的 L-1.7，$8.3）**：不在 A1–A9 字面范围内。但 2.3、2.4 的勾选链依赖 1.7，建议在 B1 与 A2 一起运行。
3. **A5 的两岛不对称**：岛0 用 3 卡切换，岛1 用 2 卡固定。3.8 原文只要求"两小岛"，没有指定边，因此本条不属于改变验收，列在这里供知悉。
4. 是否改用 Nebius 跑非逐位行：名义上最多省约 $29，需先花约 $3 验证 Nebius island + 本机 head 路径，默认不采用。
5. 未分配的 $25 是否授权主 agent 在某批超出预算时动用（默认不授权，超出即停下报告）。

## 7. 与 v1 的差异摘要

- A1 与 A3 复用已有证据，不再运行（v1 分别为 $35 和 $283）。
- A2 缩小为只做剩余的 L-2.3（2 卡）。
- A4 与 A4b 改为 4 卡（待确认）。
- A5 从 2×8 卡改为 3+2 卡加本机 head。
- E2 四项合并为一次 `H100!:2` 租期。
- DEV-GATHER 不再单独占用 20 h，改为并入 E2 的 C2 和 A8，并保留应急额度。
- A8 从 4 卡改为 2 卡。
- A9 改为 3 卡的等价小边。
- A10 暂缓。
- F 阶段只保留 F0 和 F-E1。
- v1 的 A1–A9 为 $1,500 以上，v2 的期望费用约 $154（4 卡档）。

## 8. 2026-09-30 用户决定与修订（主 agent 维护，INTEG 代笔；本节为修订，不改 §3 任何判据文字）

### 8.1 用户决定（2026-09-30）
- A4 按 3.4 原文 **8 卡 T4R2S2↔T4R4S0** 实测（即 §2.1 的"8 卡原文版本"，直接使用 `evidence/infra-e1/plan.md` 原判据，不做 4 卡换算）。
- 加跑 A2+（L-1.7）。
- 非逐位比较项改用 Nebius：先做一次路径验证冒烟，不通则退回 Modal。逐位项（A6/A8 等）仍用 Modal `H100!`。
- 超预算前停下，报告并阐述进度。
- 未分配余额不得动用。

### 8.2 A5 修订（INFRA-E1 提出，运行前）
- quorum 用例（§3 A5 第 2 条）须带 `--rl-elastic-pause-margin 2.0`，up 请求 deadline 设 230 s；否则 `--quorum-timeout-s 120` 下默认暂停预算 60 s，150 s 注入延迟会在 plan 阶段被 pause 审计拒绝，第 2 条无从执行。详见 `evidence/infra-e1/plan-3.8-4.4-v2.md` §1 第 5 条。
- "`start_cells` 前注入 150 s"的注入点已由 infra-e1 9a5f181 实现并随本次集成合入：launcher `--rl-test-inject-start-delay-s 150`（须同时带 `--rl-elastic`；岛上 `export YETO_RL_TEST_INJECT_START_DELAY_S=150.0`；默认不设置，无影响）。quorum 用例参数为 `--rl-elastic-quorum-timeout-s 120 --rl-elastic-pause-margin 2.0`，请求 deadline 230 s。

### 8.3 A8/A9 规模（计划口径，采纳 INFRA-E3 `evidence/infra-e3/plan-v2.md`） （被 §8.7(4) 取代：以 plan-v3 为准）
- A8：2×H100!，上限 $15.8。
- A9：4×L40S，T2R2↔T1R3，上限 $27.3；仅在 A8 为 go 时运行（F-R1 已获用户批准，见 8.4）。
- DEV-GATHER：A10G，上限 $3.3。
- 注意：E3 分支尚在复审、未合入集成分支；此处仅记录计划口径，判据以 E3 plan-v2 合入后的文本为准。§2 表中 A8/A9/DEV-GATHER 的旧配置与费用被本条取代（§3 A9 判据中的"3 卡 T2R1↔T1R2"拓扑文字未改动，差异待主 agent 裁定）。

### 8.4 A9 前置
- fork 需求 F-R1（启动时可声明不启动、延迟绑定的停止 cell）：**用户 2026-09-30 已批准**（FORK-FR1 已开工，miles 分支 `yeto-deferred-cell`）。A9 仍须 A8 为 go 才运行。

### 8.5 变 DP 认证范围
- 变 DP 认证仅覆盖 dropout=0（`lora_dropout=hidden_dropout=attention_dropout=0`，代码在 DP 变化时拒绝非 0 或未知）：**用户 2026-09-30 已接受**。

### 8.6 2b 的 4.4 补跑
- 用户 2026-09-30 批准 rl-algo 2b 的 4.4 用 Modal T4 补跑，单独记账，不占 A1–A9 的 $300 预算。

### 8.7 主 agent 裁定：计划统一（2026-09-30，运行前）
本条只补充与澄清，不放宽任何判据；被影响的原文保留并标注"见 §8.7"。
1. **A5 与 A6b**：`evidence/infra-e1/plan-3.8-4.4-v2.md` 是本计划的组成部分，其开关、前置条件、补充观测与判据一并生效：
   - A5：quorum 用例用 `--rl-elastic-quorum-timeout-s 120 --rl-elastic-pause-margin 2.0`，请求 deadline 230 s，注入开关 `--rl-test-inject-start-delay-s 150`（§8.2）；岛0 每个执行事务的 `pause_decision`（`outer_phase=round-boundary-published`、`allowed=true`、`stalls_peers=true`、`budget_s` 与输入一致）；quorum 该 step 岛0 只有一次 PUSH，bridge 磁带中没有 "conflicting PULL permits" 和 "invalid PULL permit"。
   - A6b：cut manifest `progress.local_step = 3`、`outer.settled = true`、ledger `carried_over = 0` 且 `ready_unconsumed = 0`；`rl_driver_start` 1 条、`rl_trainer_rebuilt` 1 条（`policy_version = 3`、成员齐全）；重建后第 1 轮的 `trained_sample_ids_sha256` 与数据游标对 B1 精确相等；`SwappableActor.generation` 在 RESTORED 时为 1，在 REBUILD_OLD 时为 2。
   - 被杀 cell 所在 GPU 的空闲检查（plan-3.8-4.4-v2 §4 第 4 条）。
   - 两者冲突时取更严者。逐条核对的结果：没有互相矛盾的判据，plan-3.8-4.4-v2 各条都是在 §3 之上追加的条件，所以两边同时生效。唯一的口径差异是：§3 A5 第 2 条只写了 `--quorum-timeout-s 120`，没写暂停预算；按 plan-3.8-4.4-v2 执行，这是让该判据能够执行的必要条件，不是放宽。
2. **E1-D watchdog**：plan-3.8-4.4-v2 §4 的 5 条判据纳入 A4 的 3.7 部分，与 E1-D 原判据同时满足才算通过。该用例依赖"阻塞 `update_weights`"注入点（INFRA-E1 正在实现），注入点合入集成分支前不得运行。§2.1 中"3.7 watchdog 默认没有接 kill"已过时（2b67145 已实现并合入）。
3. §4 中"A2 eval 源接线仍有缺口"已过时，launcher eval 接线（infra-e1 f79e016）已合入。
4. **A8/A9/DEV-GATHER 以 `evidence/infra-e3/plan-v3.md` 为准**（infra-e3 870a329 已合入集成分支）：
   - A8：`H100!:2`，上限 $15.8。
   - A9：4×L40S，T2R2↔T1R3，上限 $27.3。前置条件：A8 为 go；F-R1 的 fork 实现完成并重建镜像；以及 plan-v3 §5 所列的其余前置（controller/elastic-wiring 补丁合入、E1 的 `bind_members`/`MilesTrainerOps` 接线）。
   - DEV-GATHER：Modal 2×A10G，上限 $3.3。
   - §2 表中与 §3 A9 的 3 卡 T2R1↔T1R2 拓扑被本条取代；判据以 plan-v3 为准。
5. **批次硬预算按 8 卡原文档执行**：B1 $170 / B2 $80 / B3 $50，全局 $300。非逐位项已改用 Nebius（§8.1），实际费用按台账 `infra-drafts/gpu-spend.md` 计。
## 9. 批次 1 执行计划（用户决定后更新，2026-09-30，GPU-B1；本节提交后判据冻结）

用户决定（2026-09-30）：A1–A9，A10 暂缓；全部 GPU ≤ $300（含重跑）；**A4 按 3.4 原文 8 卡 T4R2S2↔T4R4S0**（§2.1 的 8 卡原文档，判据直接用 `evidence/infra-e1/plan.md` 原文，不换算）；加跑 A2+（L-1.7）；**非逐位行改 Nebius**，先做路径验证冒烟，不通则退回 Modal；未分配余额不得动用。代码基线 gpu-b1 = integ-decl a303cbb，镜像 pin 5c1b49e-9f29303（`yeto/rl/__init__.py` 的 MILES_NEXT_IMAGE digest）。台账 `infra-drafts/gpu-spend.md`；证据 `evidence/infra-v2-b1/<run>/`。

### 9.1 价格与本批硬预算

- Nebius H100：v2 §1 写 $2.95/GPU·h；**运行前核对 SkyPilot 目录（`sky show-gpus H100 --cloud nebius`，2026-09-30）为 $3.85/GPU·h**（`gpu-h100-sxm_8gpu-128vcpu-1600gb` $30.80/h，1 卡 $3.85/h，eu-north1），与 Modal `H100!` $3.95 基本持平。Nebius 只有 1 卡与 8 卡两种 H100 规格，6 卡基线也只能租 8 卡。费用门控一律按 $3.85 计最坏费用。
- Modal L40S $1.95/GPU·h（v2 §1）。
- **本批硬预算**：v2 8 卡档 B1 $170 按 v2 所写 Nebius/Modal 价比折算：$170 × 2.95/3.95 = **$127**（取整）。用目录价 $3.85 折算会得到 $166，取较小者 $127。全局累计仍 ≤ $300。
- 门控：每次启动前 本批已花 + 本次最坏费用（卡数 × watchdog 硬上限时长 × 单价）> $127 或全局 > $300 → 不启动，报告。

### 9.2 代码就绪核查（a303cbb，CPU 只读核查，运行前）

| 项 | 需要 | a303cbb 现状 | 处理 |
|---|---|---|---|
| E1-A/E1-E（3.4 X2，3.1/3.2/3.6 旁证） | `--rl-elastic` 经 launcher、CommandInbox、ElasticPlacement、declared_cells | 已接线（launcher `_ports_infra_flags`，entry `elastic_wiring_for`/`ElasticPlacement`）；8 卡 Nebius dry-run 通过 | **运行** |
| attestation 指纹 | attestation `runtime_fingerprint` 必须等于岛上 `sha256(miles_commit + Miles argv)` | 无离线计算入口（需镜像内 Megatron bridge）；同参数两次运行指纹相同（infra-a s4/s5 t2r2-s17 均为 213ae47a…） | 先跑**无请求的 T4R2S2 基线**（本来就是 E1-A 的基线），从其 `rl_driver_start` 取指纹写入 E1-A 的 attestation；elastic/attestation 标志不进入 Miles argv |
| E1-B（3.5） | 在 `check_weights` 前用 `update_weights_from_disk` 覆盖一个新 engine 的故障注入；旧 epoch `start_update_weights` 调用入口 | `load_fault_injection` 只认 `publish_delay_s`，fork 无对应注入 | **等待代码**，不运行 |
| E1-C / A4b（3.3） | 工具等待负载（`tool_wait_scope` 的 generate 函数）+ 岛上 ToolWaitBoard 传入 elastic wiring | `elastic_wiring_for` 未传 `tool_wait_board`（恒 None，drain 只靠串行轮边界）；仓库内无使用 `tool_wait_scope` 的 generate 函数 | **等待代码**，不运行 |
| E1-D（3.7） | ③④ stop_cells 半失败注入；⑤⑥ 原地重启 learner（同一 state dir）；⑦ fork InferenceController 重启 | ③④ 无注入入口；state dir = 容器内 `~/yeto-rl/elastic-state`（非持久卷），Modal 函数 retries 在新容器重跑、Nebius 经 launcher 无原地重启 learner 入口；⑦ 无入口；`on_watchdog` 未接 kill | ③④⑦ **等待代码**；⑤⑥ **环境阻塞**（CPU 前置不满足）；①② 单独跑不能使 3.7 满足原文，为省钱不运行，待注入代码合入后与 ③④ 同批 |
| A2（L-2.3） | launcher 传 `--eval-interval`/eval 集 | launcher 无 eval 配置（`_check_ports_infra_switches` 注释明确） | **等待代码** |
| A2+（L-1.7） | `observe=True` 经 launcher 开启；W-tool 工具负载 | `yeto_rl_observe_timeline` 无任何 CLI/launcher 入口；无工具负载 | **等待代码** |

### 9.3 本批运行（顺序、卡、时长、费用、硬超时）

| 序 | 运行 | 前缀 | 云/卡 | 期望时长 | 硬超时（外层 timeout / watchdog 释放） | 最坏费用 | 判据 |
|---|---|---|---|---|---|---|---|
| 1 | Nebius 路径冒烟 | `infra-v2-b1-nsmoke-20260930-1` | Nebius 1×H100（`gpu-h100-sxm_1gpu`），sky `--down` + autostop 10 min | 0.4 h | 45 min / 50 min `sky down` | 1×3.85×50/60 = **$3.2** | 见 9.4-1 |
| 2 | F0 | `infra-v2-b1-f0-20260930-1` | Modal 1×L40S | 0.25 h | 30 min / 35 min `modal app stop` | **$1.2** | v2 §3 F0 原文 |
| 3a | F-E1 指纹 | `infra-v2-b1-fe1fp-20260930-1` | Modal 3×L40S，T1R1S1，无请求，取到 `rl_driver_start` 即停 | 0.25 h | 25 min / 30 min | **$2.9** | 仅取指纹，不判 |
| 3b | F-E1 | `infra-v2-b1-fe1-20260930-1` | Modal 3×L40S，T1R1S1↔T1R2S0，一次 up/down，6 轮 | ≤1 h | 70 min / 75 min | **$7.3** | v2 §3 F-E1 原文（不判通过/失败，不计入 task） |
| 4 | E1-A 基线 | `infra-v2-b1-a4base-20260930-1` | Nebius 8×H100，T4R2S2 elastic，无请求，12 轮 | 0.7 h | 60 min / 65 min `sky down` | 8×3.85×65/60 = **$33.4** | 基线（同时提供指纹） |
| 5 | E1-A | `infra-v2-b1-a4x2-20260930-1` | Nebius 8×H100，第 3 轮前 up T4R2S2→T4R4S0，第 8 轮前 down，deadline 各 600 s，12 轮 | 0.75 h | 60 min / 65 min | **$33.4** | infra-e1/plan.md E1-A (a)–(g)、E1-E 原文 |
| | 合计最坏 | | | | | **$81.4** | 余 $45.6 只用于"查明原因并修复后"的重跑 |

- F-E1 用 3×L40S（v2 §2 原 F-E1 规格）而不是 §2.1 所写 7×L40S：F-E1 只验证事务控制流，不产生证据，3 卡更省；判据不受影响。
- Nebius 冒烟不通时：拉日志 → `sky down` → Nebius API 核实 → 写结论，第 4/5 项改在 Modal `H100!:8`（$3.95，最坏 8×3.95×65/60 = $34.2/次，合计最坏 $83，仍在 $127 内）。
- 若 F0 不通过：按 v2 §5 取消 F-E1，E1-A 直接上 H100。
- 固定参数（所有 H100 运行）：Qwen3-0.6B（rev c1899de2…）LoRA r16 all-linear，GRPO 默认 spec，gsm8k（rev 0cbd9f31…），`--rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17`，`--rl-placement fixed-partition --rl-rollout-gpus 2 --rl-standby-gpus 2 --rl-single-island-no-sync --controller local --rl-elastic --rl-elastic-cells c0,c1,c2,c3 --rl-elastic-initial-config T4R2S2`，资源清单 `evidence/infra-v2-b1/cfg/resources-8.json`（T4R2S2、T4R4S0 与双向 `rollout-only` 边）。E1-A 另带 attestation（两条边、`execution_modes: [partitioned-serial]`、指纹取自第 4 项）。
- 请求提交：写入岛内 `~/yeto-rl/elastic-state/inbox/<request_id>.request.json`（`CommandInbox` 原格式），经 `ssh <cluster>` 在第 2 轮 / 第 7 轮 generate 期间提交，使其在第 3 / 第 8 轮前的安全点执行；`<request_id>.status.json` 与 journal 取回存档。
- 采集：puller 每 ≤10 s 取事件磁带、`nvidia-smi --query-compute-apps=pid,gpu_uuid`、`nvidia-smi -L`，结束前取 journal/epochs/ledger。

### 9.4 判据（冻结）

1. **Nebius 冒烟**（路径验证，不计入 task）：通过 = (a) sky 集群 UP 且岛容器内 `nvidia-smi` 为 H100；(b) 岛连上本机 head syncer（本机 syncer 日志有该岛连接、磁带有外层同步完成事件）；(c) 1 轮完成，launcher 退回码 0；(d) 运行期间无 keepalive/连接中断导致的失败。任一不满足 = 不通，按 9.3 退回 Modal，不在 Nebius 上排错。公网 IPv4 用量记录（配额 3，本次 1）。
2. **F0 / F-E1**：v2 §3 原文。
3. **E1-A / E1-E**：`evidence/infra-e1/plan.md` §1 E1-A (a)–(g) 与 E1-E 原文，逐字适用，无数值容差，1 个 seed。3.4 只有 E1-A 全部满足才勾选；3.1/3.2/3.6 旁证只记录，不据此勾选。E1-A (e) 的"池"= 该 8 卡 VM 的 8 张 UUID。
4. 3.3、3.5、3.7、2.3、1.7 本批不运行，状态按 9.2 记录，不勾选。

### 9.5 合并 integ-decl 15d88bd（含 47c625e）后的就绪复核（运行前；判据不变）

主 agent 2026-09-30 指示：未开跑项合并 integ-decl 47c625e 后再跑，判据以已提交为准。gpu-b1 已合并 origin/integ-decl 15d88bd（47c625e 为其祖先）。已完成项的代码 SHA：Nebius 冒烟、F0 = a303cbb。其后各运行用本合并提交（证据中逐次记录 SHA）。复核结果（只改就绪判断，不改 §9.4 与 §8.7 任何判据）：

| 项 | 15d88bd 现状 | 处理 |
|---|---|---|
| A2（L-2.3） | launcher 已有 `--rl-eval-interval/-data/-dataset-name/-samples-per-prompt` 与 `--rl-overlap-eval`；**但 launcher 不转发 `--eval-temperature`**，Miles `eval_temperature` 缺省回落到 `rollout_temperature`（默认 1.0，`miles/utils/eval_config.py`），L-2.3 判据 5 要求的贪心 eval（temperature=0）经 launcher 无法设置 | **等待代码**（launcher 需转发 eval temperature），不运行 |
| A2+（L-1.7） | `yeto_rl_observe_timeline` 仍无 CLI/launcher 入口；无工具负载 | **等待代码** |
| E1-B（3.5） | `load_fault_injection` 仍只认 `publish_delay_s`；无权重覆盖注入 | **等待代码** |
| E1-C / A4b（3.3） | `elastic_wiring_for` 仍不传 `tool_wait_board`；无工具负载 | **等待代码** |
| E1-D（3.7） | watchdog kill 已合入（2b67145/5946ffd）；§8.7(2) 所需"阻塞 update_weights"注入点未合入；③④ 的 stop_cells 半失败注入、⑦ 的 fork 重启入口仍无；⑤⑥ state dir 仍在容器内非持久路径 | ①②③④⑦ 与 watchdog 用例 **等待代码**；⑤⑥ **环境阻塞** |
| E1-A/E1-E（3.4） | 不变，可运行 | **运行**（9.3 第 3–5 项，代码 = 本合并提交） |
| 本批硬预算 | §8.7(5) 写 B1 $170；本节 9.1 按用户"按 Nebius 价折算"取 **$127**（更严），维持 | — |

### 9.6 合并 integ-decl a5123ca 后的补充（运行前；判据不变）

- gpu-b1 已合并 origin/integ-decl a5123ca（含 `--rl-test-inject-update-weights-block-s`）。此后的运行（F-E1 起）使用本合并提交；F-E1 指纹运行（9.3 第 3a 项）已在 05822ff 上启动，不中断；Miles argv 相关代码（`miles_adapter/config.py`、`run_config.py`）在 05822ff→本提交之间无改动，指纹不受影响。
- **Nebius 冒烟不通**（`evidence/infra-v2-b1/nsmoke/`），按 9.3 退回 Modal：E1-A 基线与 E1-A 用 Modal `H100!:8`（`--modal-gpu-exact`，运行前断言 GPU 名），最坏每次 8×3.95×65/60 = $34.2。
- **新增运行 6：E1-D watchdog 用例**（按 §8.7(2) 与 `evidence/infra-e1/plan-3.8-4.4-v2.md` §4、§6，判据原文适用，未改）：前缀 `infra-v2-b1-a4wd-20260930-1`，Modal `H100!:8`，T4R2S2 起始，`--rl-test-inject-update-weights-block-s 600`，第 3 轮前 up（deadline 120 s），共 4 轮；期望 0.5 h；硬超时 外层 45 min / watchdog 50 min；最坏 8×3.95×50/60 = **$26.3**。E1-D 其余 ①–⑦ 仍按 9.5（等待代码/环境阻塞），因此即使本用例通过，3.7 也不勾选。
- 更新后本批最坏合计：已花（≤$0.42 + 指纹运行）+ F-E1 $7.3 + 基线 $34.2 + E1-A $34.2 + watchdog $26.3 ≈ $105，< $127。顺序：F-E1 → 基线 → E1-A → watchdog；每次仍按门控计算。

### 9.7 A2（L-2.3）执行安排（主 agent 2026-09-30 指示；合并 integ-decl 6838fe9 后，运行前；判据 = local-gpu-plan L-2.3 1–6 原文，不改）

- Nebius 不再使用（主 agent 决定），全部 Modal。本轮只做 A2。
- 本地端到端 dry-run（`evidence/infra-v2-b1/a2/test_a2_dryrun.py`，3 passed）：真实 CLI → 岛 run 命令 → 临时 HOME 执行 prelude → `learner.parse_args` → `_resolve_eval` → `resolve_rl_run_config`/`build_ports_launch`（translate_run_config）→ `apply_ports_infra_switches` → `execution_profile_for`。确认：eval interval 1、Miles argv `--eval-temperature 0.0`、fixed-partition `--rollout-num-gpus 1`、observe 开启、S=partitioned-serial、O/OD=partitioned-overlap（允许 eval‖train、eval‖outer_sync）、heldout 文件逐字节送达。
- dry-run 发现：`--rl-single-island-no-sync` 与 eval 不兼容（`_resolve_eval`："Yeto evaluation requires the external policy boundary"），因此三个 arm 都用 `--rl-sync-preset strict-avg` 单岛 + 本机 head（与 infra-a 2.x 的 GPU 证据同一路径）。
- 事件来源：`rl_eval`（含 `overlapped`、`rl/policy_token`）、`rl_eval_overlap_start`、`rl_timeline_span(task=eval)`（LoopEvalHandle 真实区间，需 observe）、train span、`rl_publication`、`rl_fault_injected` 在磁带；**eval 分数不在磁带**（`evaluate()` 返回 {}），取自 Miles 日志行 `eval <rollout_id>: {...}`（`miles/ray/rollout/metrics.py` log_eval_rollout_data），判据 5 的分数条件按该行计算。
- 配置：Modal `H100!:2`（`--modal-gpu-exact`），T1R1，`--total-steps 3 --seed 17`，`--rl-observe-timeline --rl-eval-interval 1 --rl-eval-data heldout-gsm8k-test-first32.jsonl（gsm8k test 前 32 条，N=32）--rl-eval-samples-per-prompt 1 --rl-eval-temperature 0 --rl-eval-max-response-len 384`；O/OD 加 `--rl-overlap-eval`；OD 另在代码快照根放 `yeto-rl-fault-injection.json` = `{"publish_delay_s": 30}`。三个 arm 依次运行，前缀 `infra-v2-b1-a2{s,o,od}-20260930-1`。
- 每 arm 硬超时：外层 45 min / watchdog 50 min `modal app stop`；最坏 2×3.95×50/60 = **$6.6**，三 arm 最坏 $19.8；本批已花 ≤$2.91，合计 < $127。

### 9.8 A2+ 与 F-E2（主 agent 2026-09-30 指示；合并 integ-decl b075524 后，运行前）

- **A2+（L-1.7）本地核查不通，不上卡**：真实路径上 `rl_load_sample` 由 `MilesRolloutPool.load_sample()` 产生，字段只有 `active_requests/workers/cordoned`；`timeline.classify_load` 需要的 `queued_requests`、`tool_wait_trajectories`、`ready_groups`、`engine_capacity` 在任何事件中都不出现（工具负载只把等待写进 ToolWaitBoard 和 `non_generation_time`）。L-1.7 判据 1（`classify_load == "tool-wait"` 样本且 in-flight=0；W-gen 有 saturated/busy、无 tool-wait）在该路径上无从计算 → **等待代码**（load sampler 需带上 tool-wait 板计数与 queued/capacity）。
- **F-E2 冒烟**（不计入任何 task，不产生验收证据）：目的：在 E2 harness 前，经 launcher 走 elastic 路径发一次 4.4 `rebuild-trainer`，暴露真实路径问题。配置：Modal 3×L40S，0.6B LoRA，T2R1S0（trainer DP2 + 1 rollout），`--rl-single-island-no-sync --controller local --rl-placement fixed-partition --rl-rollout-gpus 1 --rl-elastic --rl-elastic-resources fe2/resources-T2R1S0.json --rl-elastic-initial-config T2R1S0 --rl-elastic-cells c0`，4 轮；第 2 轮 train 期间由容器内触发器写入 `rb1.rebuild.json`（`--expected-epoch 0 --deadline-s 900`）。本地 dry-run：`evidence/infra-v2-b1/fe2/test_fe2_dryrun.py` 1 passed（argv DP2/rollout 1、无 `--load`、rebuild 前置条件为空、partitioned-serial、真实 controller 接受 rebuild 请求）。
  - 观察项：事务终态是否 `SUCCEEDED`；是否出现 `rl_trainer_rebuilt` 事件；`<state>/cuts` 下是否有 cut 落盘（结束前拉取 manifest）。
  - 停止条件：事务到达任一终态且再完成 1 轮后停止；或出现 Python 异常/RECOVERY_REQUIRED 即拉证据停止；硬超时 外层 40 min / watchdog 45 min，最坏 3×1.95×45/60 = **$4.4**。前缀 `infra-v2-b1-fe2-20260930-1`，代码 = 本提交的合并基线。

### 9.9 F-E1 重跑（F-R1；主 agent 2026-09-30 指示；合并 integ-decl e052dc8 后，运行前）

- 代码 1a5ccd5（= 合并 e052dc8），镜像 `sha256:db815884…`（miles 2f23a0fc，含 F-R1）。本地 dry-run `evidence/infra-v2-b1/fe1r/test_fe1r_dryrun.py` 1 passed：pin digest/commit 正确；`--rl-elastic-declare-cells --rl-elastic-cells c0,c1` 生成 placement map `rollout_cells = [{c0, bundles [1], start true}, {c1, bundles [2], start false}]`（先 rollout 启动 cell，再 standby 停止 cell）。
- 运行：(a) 指纹运行 `infra-v2-b1-fe1rfp-20260930-1`，3×L40S，同参数无 attestation，取 `rl_driver_start.runtime_fingerprint` 后立即停止；(b) F-E1 `infra-v2-b1-fe1r-20260930-1`，3×L40S，T1R1S1↔T1R2S0，6 轮，第 2 轮 train 时 up（deadline 600 s），第 4 轮 train 时 down，容器内触发器提交。
- 判据（主 agent 指定，冻结）：up 与 down 两个事务 journal 终态都为 `SUCCEEDED`；journal 中成员使用 fork cell id；旧成员（c0 对应的 fork cell）全程不变。另须在容器启动日志/manifest 中确认实际镜像 digest = db815884…、镜像内 miles HEAD = 2f23a0fc，不一致即停。
- 预算：两次合计 ≤ $3（硬超时 (a) 20 min、(b) 40 min；最坏 3×1.95×(25+45)/60 = $6.8 超过 $3 → 两次运行的实际停止由本 agent 在达到判据后立即执行，若累计实际费用将超 $3 即停止并回报）。失败即停并回报 journal。
- 9.9 补充（F-E1 重跑启动前）：指纹运行实际 ≤$1.17（指纹 `sha256:88ef2727…`；manifest 镜像内 miles = 2f23a0fc，launcher 拉取 digest = db815884…）。为守住 $3，F-E1 改为第 1 轮 train 时 up、第 3 轮 train 时 down，两事务终态出现后立即停止；watchdog 18 min（最坏 3×1.95×18/60 = $1.76，累计 ≤ $2.93）。判据不变。
- 9.8 补充（F-E2 启动前，判据不变）：代码 ec249ef（合并 integ-decl 3f88c1d）；运行带 `--modal-retries 0 --modal-timeout-s 2700`（外层硬超时 40 min + 5 min），并加进度看门狗（20 分钟无新事件即拉证据后 `modal app stop`）。按 F-R1 后的约定，不再传 `--rl-elastic-cells c0`（未 declare 时该名字会被岛上拒绝），改为缺省"fork 声明的全部 cell"；launcher dry-run rc=0（3 卡）。

### 9.10 F-E1 第三次（合并 integ-decl c28aaf1 后，运行前；判据同 §9.9，不变）
- 代码 f76bf1d；dry-run `fe1r/test_fe1r_dryrun.py` 追加断言 Miles argv 含 `--use-miles-router`，1 passed。Miles argv 变化 → 重新取指纹（前缀 `infra-v2-b1-fe1r2fp-20260930-1`），再跑 `infra-v2-b1-fe1r2-20260930-1`（第 1 轮 train up，第 3 轮 train down）。均带 `--modal-retries 0 --modal-timeout-s`、进度看门狗 20 min。两次合计 ≤ $3：指纹运行 watchdog 20 min，正式运行 watchdog 18 min；累计实际将超 $3 即停。

### 9.11 L-D0（主 agent 批准，不计入 task；判据 = local-gpu-plan.md L-D0 原文，不改）
- 代码 aeaf6e2（合并 integ-decl a382490；launcher `--modal-retries 0` 自动关闭 relaunch）。dry-run `evidence/infra-v2-b1/ld0/test_ld0_dryrun.py` 1 passed：learner 收到 `--rl-deterministic-trainer`，Miles argv 含 `--deterministic-mode`（及 `--sglang-enable-deterministic-inference`），`NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` 写入环境并由 connect_island_ray 转发给 Ray worker。
- 前缀按本 agent 资源纪律用 `infra-v2-b1-d0-{1,2}-20260930`（L-D0 原文示例为 `infra-a-d0-…`，仅名字不同）。Modal `H100!:2`，strict-avg 单岛 + 本机 head，无 eval，3 轮 seed 17，依次运行；外层 30 min，`--modal-timeout-s 2100`，watchdog 35 min，进度看门狗 20 min；最坏合计 $9.2。
- F-E1/F-E2 因 Modal L40S 无容量暂停（`modal app logs`：waiting to be scheduled on a GPU_L40S worker）。
- 9.11 结果：L-D0 判据 1、2 通过（`evidence/infra-v2-b1/ld0/RESULT.md`）。按结论规则重跑 A2 三 arm：§9.7 配置 + `--rl-deterministic-trainer --modal-retries 0 --modal-timeout-s 3000` + 进度看门狗，代码 aeaf6e2，前缀 `infra-v2-b1-a2{s,o,od}-20260930-2`；判据 L-2.3 1–6 不变；dry-run（a2/test_a2_dryrun.py 加开关）3 passed；最坏 3×$6.6。

### 9.12 冒烟改用 A10G（主 agent 2026-09-30 裁定；不计入 task，判据不变）
- Modal L40S 无容量，F-E1、F-E2 改用 Modal A10G，卡数不变（F-E1 3×A10G）。单价按 Modal A10G $1.10/GPU·h（运行前未核实当日价）。
- F-E1 第四次：代码 037d4f5（合并 integ-decl aaebc90），镜像仍为 db815884…（F-E1 不涉及 cut）。dry-run（fe1r/test_fe1r_dryrun.py，卡改 3xa10g）1 passed。先取指纹（watchdog 20 min），再正式运行（第 1 轮 train up、第 3 轮 train down，watchdog 25 min）；最坏 3×1.10×45/60 = $2.5。判据同 §9.9：up 与 down 都 SUCCEEDED、成员为 fork cell id、旧成员不变。
- F-E2 等新镜像 pin 合入后再跑（A10G，卡数 3）。

### 9.13 A2 第三次（三 arm 同 SHA）与 F-E1 第四次（主 agent 裁定；判据不变）
- 代码 37155d8（合并 integ-decl b2fe5dd：launcher 磁带按行切分修复；镜像 pin `sha256:2cc5cc52…`，Miles e3a11ab3）。dry-run：a2/test_a2_dryrun.py 3 passed，fe1r/test_fe1r_dryrun.py 1 passed（pin 与 commit 从代码读取核对）。
- A2：S、O、OD 全部在 37155d8 重跑（不复用 S2），配置同 §9.11 A2 重跑（含 `--rl-deterministic-trainer`），前缀 `infra-v2-b1-a2{s,o,od}-20260930-3`，每 arm 最坏 $6.6。
- F-E1：3×A10G，037d4f5 上取得的指纹因镜像 pin 变化作废，在 37155d8 上重取（`infra-v2-b1-fe1r5fp-…`）后正式运行（`infra-v2-b1-fe1r5-…`）；最坏 $2.5。与 A2 并行（互不共享端口/资源）。

### 9.14 A4 / A4b 执行计划（主 agent 批准；运行前提交；判据以 §3、§8.7、`evidence/infra-e1/plan.md` E1-A…E1-E、`evidence/infra-e1/plan-3.8-4.4-v2.md` §4/§6/§7.1/§8 为准，不放宽）

- 代码：d613130（合并 integ-decl a016f7f），镜像 = 该提交 `MILES_NEXT_IMAGE`（运行前 manifest 核对）。本批预算按主 agent：B1 $170；已花 ≤$33.6（+A2 OD 补跑）。
- 公共配置：Modal `H100!:8`（`--modal-gpu-exact`），`--rl-single-island-no-sync --controller local`，T4R2S2（trainer G0–G3，rollout G4–G5，standby G6–G7），`--rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2,c3`（c0/c1 在 G4/G5 启动，c2/c3 在 G6/G7 声明不启动），resources `a4/resources-8.json`（T4R2S2、T4R4S0、双向 rollout-only 边），`--total-steps 12 --seed 17`，`--rl-observe-timeline`，`--modal-retries 0 --modal-timeout-s <硬超时+300>`，进度看门狗 20 min。所有用例（含基线）都带 `--rl-elastic`（§8 Miles router 一致）且 Miles argv 相同 → 共用一个 attestation 指纹。
- 本地 8 卡 dry-run：`evidence/infra-v2-b1/a4/test_a4_dryrun.py` 9 passed（rollout_cells、`--use-miles-router`、declare-cells、各用例注入开关/restart loop 到达岛上环境）。
- 顺序、费用（8×$3.95 = $31.6/h）、硬超时：
  1. **调度探测 + 指纹**（`a4fp`）：同公共配置加 `--rl-print-attestation-fingerprint`，learner 打印指纹后退出（Ray/GPU 之前）；兼作 8 卡 H100 调度探测：10 分钟内无容器即停止回报。硬超时 20 min，最坏 $10.5。（主 agent 要求"CPU 容器取指纹"：launcher 无 CPU 岛入口，指纹运行在同一次调度探测中完成，不再单独开 H100。）
  2. **E1-A 基线**（`a4base`）：无请求，12 轮。硬超时 45 min，最坏 $23.7。
  3. **E1-A 切换 + E1-E 旁证**（`a4x2`）：第 2 轮 train 时（第 3 轮前）up→T4R4S0，第 7 轮 train 时（第 8 轮前）down，deadline 各 600 s；up 生效后再提交一次相同 request_id（幂等核对）；采样 router `/worker_inflight`、`nvidia-smi` compute-apps（E1-A (d)(e)）。硬超时 45 min，最坏 $23.7。判据 E1-A (a)–(g)、E1-E。
  4. **E1-B**（`a4e1b`）：`--rl-test-inject-weight-override <岛上 base 模型快照路径>`，第 2 轮 train 时 up；判据 E1-B (a)–(d)（(c)(d) 用容器内脚本调用 router/fork）。硬超时 30 min，最坏 $15.8。
  5. **E1-D ③**（`a4d3`）：先 up 成功，再 down 时 `--rl-test-inject-stop-failures 1`；④（`a4d4`）：`--rl-test-inject-stop-failures` 足以超过 T_recovery。各硬超时 30 min，最坏各 $15.8。①② 以容器内触发器在 `fork_op start issued` / `VERIFYING` 时 kill 新 cell 的 SGLang 进程（同一次运行按事务顺序，`a4d12`）。
  6. **E1-D ⑤⑥⑦**（按 §7.1 裁定 (c)）：⑤ 先 up（无 kill）→ 以同一 state dir 重启并带 `--rl-test-kill-learner-at COMMITTED` 发 down；⑥ epoch 0 的 up 在 QUIESCING kill；⑦ 首个事务前 kill learner（容器内 kill，restart loop 重启）后发 up。各硬超时 30 min。
  7. **watchdog 用例**（`a4wd`，§8.7(2)）：`--rl-test-inject-update-weights-block-s 600`，up deadline 120 s。硬超时 25 min，最坏 $13.2。
  8. **E1-C / A4b**：`Timeouts.drain`（T_drain）在 launcher/learner 无配置入口（默认 120 s），E1-C 要求 T_drain=5 s + 工具等待 30 s，按原参数不可执行 → **等待代码/裁定**，不运行。
- 门控：每次启动前 本批已花 + 本次最坏 ≤ $170；预计全部执行（不含 E1-C）最坏约 $200 > 余额，按顺序执行并在余额不足时停止回报（优先 1–4，其次 7、5、6）。失败即停。

### 9.15 A5（3.8 X6）执行计划（运行前提交，未上卡；判据 = §3 A5 第 1–5 条 + §8.7(1) + `evidence/infra-e1/plan-3.8-4.4-v2.md` §1、§2、§8，不放宽）

- 拓扑（主 agent 裁定：launcher 的 elastic/placement 参数对所有岛全局生效，异构岛不可表达）：两岛各 3 卡 `H100!:3`，均为 T1R1S1（`--rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1`：c0 在 G1 启动，c1 在 G2 声明不启动），strict-avg，本机 head（`run_local_head.py`，`SYNCER_PUBLIC_IP`），6 轮，seed 17。只有岛0 收到切换请求；岛1 全程 T1R1S1。基线同样两岛 3+3、带 `--rl-elastic`（§8 Miles router 一致），不发请求。
- 本地端到端 dry-run：`evidence/infra-v2-b1/a5/test_a5_dryrun.py` 4 passed——两岛运行命令经 prelude + `learner.parse_args`：rollout_cells、`--use-miles-router`、declare-cells 到达两岛；quorum 用例两岛 learner 收到 `--rl-elastic-quorum-timeout-s 120 --rl-elastic-pause-margin 2.0`，`YETO_RL_TEST_INJECT_START_DELAY_S=150.0` 导出，syncer 命令带 `--quorum-timeout-s 120`（非 quorum 用例不带）；指纹开关到达。
- 用例、卡、费用（H100! $3.95/GPU·h，6 卡 = $23.7/h）：

| 序 | 用例 | 前置/操作 | 期望 | 硬超时（外层 / watchdog） | 最坏 |
|---|---|---|---|---|---|
| 0a | 指纹 | 与正式运行同参数 + `--rl-print-attestation-fingerprint`，在 Modal CPU 容器（同镜像）执行 learner 命令，Ray/GPU 之前退出；三种参数组合（基线/切换、quorum）各取一次 | ≈$0.1 | 15 min | <$0.5 |
| 0b | 空闲流探测（§3 A5 第 4 条） | Modal CPU 函数对本机 `SYNCER_PUBLIC_IP:29400` 建 TCP 连接，60/180/350/600 s 空闲后检查；工具待写（约 30 行） | ≈$0.05 | 35 min | <$0.2 |
| 1 | 基线 B | 两岛 3+3，无请求，6 轮 | 0.45 h，$10.7 | 60 / 65 min | $25.7 |
| 2 | 切换 + finalization | 岛0 第 2 轮 train 时 up（第 3 轮前），第 4 轮 train 时 down（第 5 轮前），第 6 轮 train 时再发 up（finalization 应拒绝/取消） | 0.5 h，$11.9 | 60 / 65 min | $25.7 |
| 3 | quorum | 同 2 的 up，岛0 带 quorum/pause-margin/start-delay，up deadline 230 s；0b 测得 150 s 内会丢流则本用例判环境阻塞、不运行 | 0.5 h，$11.9 | 60 / 65 min | $25.7 |
|  | **合计** |  | ≈$35 | | **≈$77.6** |

- 可勾选：3.8（全部满足后；并解开 4.5 的依赖）。报告写明"仅完成 rollout 能力"。

### 9.16 A4 / A4b 用例费用与可勾选 task 对照（供用户取舍；8×H100! = $31.6/h；判据见 §9.14）

| 用例 | 用途 | 期望 | 最坏（硬超时） | 通过后可勾 |
|---|---|---|---|---|
| 调度探测 + 指纹（CPU 容器取指纹可替代，≈$0.1） | 前置 | $4 | $10.5（20 min） | — |
| E1-A 基线（12 轮） | 3.4 必需 | $14 | $23.7（45 min） | 与下一行合起来：**3.4** |
| E1-A 切换 + E1-E 旁证 | 3.4 必需 | $15 | $23.7（45 min） | **3.4**（3.1/3.2/3.6 仅旁证） |
| E1-B 权重覆盖 + 迟到 ACK/旧 epoch | 3.5 | $8 | $15.8（30 min） | **3.5** |
| watchdog 用例（§8.7(2)） | 3.7 的一部分 | $6 | $13.2（25 min） | 3.7 需与下面全部一起 |
| E1-D ①② kill 新 SGLang（启动/发布中） | 3.7 | $8 | $15.8 | 〃 |
| E1-D ③ stop 半失败后重试成功 | 3.7 | $8 | $15.8 | 〃 |
| E1-D ④ stop 持续失败超过 T_recovery（900 s） | 3.7 | $12 | $21.1（40 min） | 〃 |
| E1-D ⑤ down COMMITTED 后 kill learner（裁定 (c)） | 3.7 | $8 | $15.8 | 〃 |
| E1-D ⑥ epoch 0 up QUIESCING kill | 3.7 | $8 | $15.8 | 〃 |
| E1-D ⑦ 首个事务前重启 learner/fork | 3.7 | $8 | $15.8 | 〃 |
| E1-C / A4b 工具等待 drain（T_drain 入口待 INFRA-E1） | 3.3 | $8 | $15.8 | **3.3** |

- 分组合计（最坏）：只做 3.4 ≈ $58（含探测，用 CPU 指纹则 $47.4）；+3.5 ≈ $74；+3.7 全部 ≈ $187；+3.3 ≈ $203。期望费用约为最坏的 55–60%。
- 可降费选项（需用户/主 agent 决定，不影响判据）：E1-D 各项轮数压到 4 轮、硬超时 25 min（每项最坏 $13.2）；①②③ 合并为一次运行（按事务顺序），约省 $31；④ 可与 ③ 分开但共享一次 up 前缀。
