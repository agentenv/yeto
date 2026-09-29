# rl-infra-spec E0–E3 GPU 实验计划与预算（1.3 定稿，2026-09-29 INFRA；未启动任何 GPU）

作者：Agent I，2026-09-29。基于 `openspec/changes/rl-infra-spec/{tasks.md,design.md,upstream-mechanisms.md}`。本计划是 task 1.3 的交付草案，**需用户确认后才租卡**。

## 0. 前提与口径

- 所有实验在 ports 路径、`MILES_NEXT_IMAGE`（按 digest 固定）上进行。E1 需要 Miles fork M2–M4，E2 需要 yeto 侧 `save_cut/restore_cut`（和 LoRA 下的 M5 RNG），E3 需要 M1/M5/M6。这些提交目前只在本地分支 `yeto-elastic-m1-m6`，**没有进镜像**。每个 GPU 阶段开始前，先把对应 fork 提交打进一个新镜像 digest，并生成 1.1 manifest。
- 模型：功能验证用 Qwen3-0.6B LoRA；验收用 Qwen3-1.7B LoRA（TP=PP=CP=EP=1，strict-avg，GRPO）。4B（#66 示例）只在 2.4 收益面需要时才上。
- 云：本计划只使用**已支持**的 Nebius（SkyPilot）与 Modal（`yeto/modal_runner.py`）。Verda 的 provider 修复（change `fix-verda-provider`，PR #69 未合入）仍在途，**不视为已支持，不出现在任何实验行里**。约束（见 memory）：Verda 与 Modal 不能承载 head。两岛实验 head 放置见 §6。
- **逐位/数值对比规则**：凡是需要逐位对比或"下一步数值对比"的实验（X3、4.3、4.6），必须在**同一次租用**内同时跑两个 arm。用 Modal 时写成 `H100!:N`（禁止 H100→H200 自动升级），并在启动时断言 `nvidia-smi` 报告的 GPU 名称。统计比较（2.4、4.8）要求同一 provider、同一 GPU 型号，并同样断言 GPU 名称。
- **H100 验收默认用 Nebius（约 $2.95/GPU·h）；逐位/数值对比与 DistOpt 调试用 Modal `H100!:N`（约 $3.95）。Verda（约 $2.30）只在 change `fix-verda-provider` 合入后作为可选降价方案，不计入默认预算。** L40S（Modal）约 $1.95。单价是估算，下单前核对当日价格。GPU-hours 按整池 × wall time 计（备用卡也计入）。时长已包含镜像拉取和预热。

## 1. 分阶段策略

1. **F 阶段（便宜卡功能验证）**：4×L40S 或 A100-80GB（Modal），小模型，只验证控制流和故障路径：能启动、能切换、失败能恢复、账本正确。结果**不能**作为 GPU 验收或性能证据。
   - 进入 F 之前有一道门 F0：确认镜像的 SGLang/Megatron kernel 支持 sm_80/sm_89。如果不支持，F 阶段改用 4×H100（Nebius），F 预算约 ×1.5。
2. **A 阶段（H100 验收）**：只跑 tasks.md 验收原文所要求的实验。每个实验都要等对应的 F 实验通过后才开始，避免在 H100 上排错。
3. **门控**：只有 4.6 的结论是 go，才跑 4.7/4.8；如果是 no-go，就省掉约 370 H100 GPU·h。4.8 先用 2 个 seed 做 gate，指标在容差内再补第 3 个 seed。

## 2. 功能验证（便宜卡）

| ID | 目的 | 对应任务 / 验收原文 | 卡 | 时长 | 云 | GPU·h | 估价 |
|---|---|---|---|---|---|---|---|
| F0 | 镜像在 Ampere/Ada 上能跑；生成 runtime manifest | 1.1 "manifest与`MILES_NEXT_*`/`SGLANG_NEXT_*`一致，缺接口组合拒绝认证" | 4×L40S | 1.5 h | Modal | 6 | $12 |
| F1 | 小模型串行共置一轮（兼容冒烟） | 1.2 "完成一轮生成、更新、外层同步与权重确认"（仅冒烟，不算验收） | 4×L40S | 1.5 h | Modal | 6 | $12 |
| F2 | 显式分区启动 + partitioned-serial 跑 N 步（T2R2） | 2.1 "小模型固定分区正常启动，旧共置配置保持原行为"；2.2 功能 | 4×L40S | 3 h | Modal | 12 | $23 |
| F3 | rollout 增减 T2R1S1↔T2R2S0；旧 ACK、迟到请求；启动/发布失败注入；工具等待轨迹 drain 预演 | 3.3 X5、3.4 X2、3.5、3.7 功能路径 | 4×L40S | 7 h | Modal | 28 | $55 |
| F4 | cut 保存/销毁/同形重建；坏 checksum 和截断拒绝 | 4.2 "坏checksum、截断、step不一致拒绝"；4.3 功能 | 4×L40S | 4 h | Modal | 16 | $31 |
| F5 | 故障矩阵预演（rank kill、collective 超时、controller crash） | 4.5 功能路径 | 4×L40S | 4 h | Modal | 16 | $31 |
| F6 | DP1↔2 重分片和 T3R1↔T2R2 角色转移，功能跑通 | 4.6/4.7 功能路径 | 4×L40S | 5 h | Modal | 20 | $39 |
| **小计** | | | | ~26 h | | **104** | **≈$203** |

## 3. GPU 验收（H100）

| ID | 目的 | 对应任务 / 验收原文 | 卡 | 时长 | 云 | 逐位/`H100!:N` | GPU·h | 估价 |
|---|---|---|---|---|---|---|---|---|
| A1 | 兼容 baseline | 1.2 "完成一轮生成、更新、外层同步与权重确认，形成兼容baseline" | 4×H100 | 3 h | Nebius | 否（后续统计比较同在 Nebius H100） | 12 | $35 |
| A2 | E0 固定分区 + partitioned-serial（T4R4、T4R2S2）+ X9 延迟注入 | 2.1、2.2 "partitioned-serial完成固定算法步数，不因分卡改变sample IDs/optimizer时序"；2.3 X9 "延迟发布不能触发旧版本生成，队列有界" | 8×H100 | 5 h | Nebius | 否 | 40 | $118 |
| A3 | E0 固定配置扫描 P62/P44/P26/P422/legacy × 2 seeds | 2.4 "同profile公平比较、全池/备用GPU-hours和原始trace齐全，可得'尚无净收益边'的结论" | 8×H100 | 12 h | Nebius | 否（同型号，断言 GPU 名称） | 96 | $283 |
| A4 | E1 双向 rollout 切换 + 故障注入 | 3.4 X2 "trainer不动、未占用池外资源，不等待故意停用engine，备用卡计入成本"；3.5 "旧generation ACK、错误payload、迟到请求都不能污染新配置"；3.6、3.7 | 8×H100 | 14 h | Nebius | 否 | 112 | $330 |
| A4b | X5 工具轨迹 drain：两条轨迹，一条等工具且 engine active=0 | 3.3 X5 "active请求为0但tool-wait>0时保留旧路由，超时取消切换而不重放外部副作用" | 4×H100 | 3 h | Nebius | 否 | 12 | $35 |
| A5 | E1 两小岛 strict 暂停兼容 | 3.8 X6 "样本/step/policy/roster不变，quorum超时/PULL重发正确，finalization拒绝切换" | 2 岛 × 8×H100 + head（见 §6） | 5 h | Nebius | 否 | 80 | $236 |
| A6a | E2 cut 导出/加载与拒绝路径（真 trainer 状态） | 4.2 "坏checksum、截断、step不一致拒绝，源释放前恢复依据完整" | 8×H100 | 3 h | Nebius | 否（checksum 比对，不涉跨卡数值） | 24 | $71 |
| A6 | E2 同形重建数值一致（训练2步→cut→重建→冻结下一 batch vs 连续运行） | 4.3 X3 "对冻结下一batch比较RNG/计数/moments/参数更新，无额外reset" | 8×H100 | 4 h | **Modal `H100!:8`** | **是**（同容器两 arm） | 32 | $126 |
| A6b | E2 driver 替换端口实现、保留 bridge、重发权重（单岛 strict，本地 syncer） | 4.4 "不重复initialize/after_local_train，外层进度不因重建重放" | 8×H100 | 3 h | Nebius | 否 | 24 | $71 |
| A7 | E2 同形恢复故障矩阵（约 8 例） | 4.5 "rank失败、collective超时、迁移中断、controller crash提交不确定均有界恢复或RECOVERY_REQUIRED" | 8×H100 | 8 h | Nebius | 否 | 64 | $189 |
| DEV-GATHER | **DistOpt gather/reshard 开发调试**（fork-M5 缺口；含全参 dist-ckpt 对照） | 4.2a 前置；为 4.6 提供可测实现 | 2×H100 × 20 h | 20 h | **Modal `H100!:2`** | 是（调试需确定性） | 40 | $158 |
| A8 | E3 DP1↔2 重分片 spike，下一步数值对比 | 4.6 X4 "master/optimizer/RNG/样本映射和下一步数值比较给出go/no-go" | 4×H100 | 7 h | **Modal `H100!:4`** | **是** | 28 | $111 |
| A9 | E3 P62↔P44 双向角色转移 + 失败恢复（4.6 go 才跑） | 4.7 "实际GPU从trainer转给rollout及反向……双向成功/失败恢复、batch语义和epoch都正确" | 8×H100 | 10 h | Nebius | 否 | 80 | $236 |
| A10 | E3 学习验证：固定/同形恢复/变 DP，先 2 seeds 再补第 3 | 4.8 "预声明数值/学习容差、heldout/reward与NaN/发散检查" | 8×H100 | 36 h | Nebius | 否（统计，同型号） | 288 | $850 |
| **小计** | | | | | | | **932** | **≈$2,849** |

## 4. 总预算与分阶段额度

| 阶段 | 内容 | 便宜卡 | H100 GPU·h | 额度 |
|---|---|---|---|---|
| S0/E0 | F0–F2 + A1–A3 | 24 GPU·h | 148 | ≈$483 |
| E1 | F3 + A4、A4b、A5 | 28 | 204 | ≈$656 |
| E2 | F4、F5 + A6a、A6、A6b、A7 | 32 | 144 | ≈$519 |
| E3 spike | DEV-GATHER + F6 + A8 | 20 | 68 | ≈$308 |
| E3 完整（4.6 go 才解锁） | A9、A10 | — | 368 | ≈$1,086 |
| **合计** | | 104 | 932（≈117 个 8 卡节点·h） | **≈$3,052** |

- 失败重跑余量 +25%：上限 ≈$3,815。
- 4.6 no-go（不解锁 E3 完整）：≈$1,966。
- 可选降价：fix-verda-provider 合入后把 Nebius 行（832 GPU·h）改到 Verda，约省 $540，合计 ≈$2,510。
- 全部改 Modal：≈$3,880。

## 5. 顺序与并行

F0 → F1 → A1 → F2 → A2 → A3。之后 F3→A4→A5（E1）与 F4→A6→F5→A7（E2）两条线可以并行，E1 和 E2 可以共用同一次 8×H100 租期；A6a→A6→A6b→A7 顺序。E2 通过后 DEV-GATHER→F6→A8，4.6 为 go 时再跑 A9→A10。

每次租用都记录 region、GPU 型号/UUID/NUMA/互联、镜像 digest、fork 提交、租期和清理确认（task 1.3）。每次都用 `yeto` 已有 launcher 启动，并用 pool_id/pool_epoch 登记到 #66 manifest 的 `resources` 段。

## 6. A5 head 放置与 NAT/keepalive

- 约束：Verda 与 Modal 不能承载 head（memory）；rl-engine-ports 7.1 的做法是 head 放本机。
- 首选：head（syncer）放在与两岛同 region 的一台 Nebius CPU VM（公网 IP 或同 VPC 内网），岛与 head 之间不经过家用/企业 NAT。若 Nebius 公网 IP 仍被跨云测试占用，退回本机 head（同 7.1）。
- NAT/keepalive 风险（pause-audit.md）：learner socket 没有 SO_KEEPALIVE，strict 暂停期间唯一流量是每 quorum_timeout_s 一次的 PULL 重发；会丢空闲流的 NAT 可能掐断连接，`max_reconnects=0` 时 learner 退出。处理：
  1. A5 开始前做 30 分钟空闲流探测（同路径 TCP 连接，按 60/180/350/600 s 空闲点检查存活），测得 `idle_flow_timeout_s`；
  2. 暂停时长按 `pause_audit.pause_decision(..., idle_flow_timeout_s=测得值)` 取上限，"跨 quorum timeout"用例把 `--quorum-timeout-s` 调小（如 120 s）而不是把暂停拉长到 900 s 以上；
  3. 若测得路径会在目标暂停内丢流，只在同 VPC head 上跑跨 quorum 用例；SO_KEEPALIVE 的代码修复作为独立后续项，不在本实验里偷改。

## 7. 未决 / 风险

- 镜像能否在 L40S/A100 上运行尚未验证（F0 门）。
- E1 依赖 fork M2–M4 合回 yeto/ports 并进镜像；M4 目前没有 payload 级 ACK，需 yeto Publisher 补读回校验。
- M5 的 DistributedOptimizer gather/reshard 未实现（DEV-GATHER 预算）；未完成前 LoRA+DistOpt 的 E3 为 no-go。
- DynaResize 原文本地未找到，1.8 不影响本预算。
- 单价未核对当日报价。

## 8. 1.3 定稿：池身份、launcher 映射与每次租用记录（Agent INFRA，2026-09-29）

本节是 task 1.3 的交付物。它补齐 1.3 验收要求而前文未写全的部分：如何映射到现有 launcher/harness 与 #66 pool 身份，每次租用必须记录哪些字段，以及清理。前文 §0–§7 的实验行与预算保持不变，只改了 §0 中对 Verda 的表述（改为：不视为已支持）。

### 8.1 与现有接入的映射

| 云 | 启动入口（现有代码） | 限时/自动回收（机制） | 本计划用途 |
|---|---|---|---|
| Nebius H100 | `yeto` launcher → SkyPilot（`yeto/launcher.py` 云凭据表 `:651-660`，Nebius 每个 region 需要一个 project，`:746-756`） | `sky launch --down` + `--idle-minutes-to-autostop`；另起独立 watchdog 按集群名 `infra-a-*` 执行 `sky down <name>`（只处理本计划记录的名称） | 1.2 A1、E0 A2/A3、E1、E2 的统计类实验 |
| Modal L40S/A100/H100 | `yeto/modal_runner.py`（`validate_modal_shape`，`ModalIslandConfig.timeout_s`） | Modal 函数 `timeout=`（`modal_runner.py:313`），结束后 `modal app stop -y infra-a-*` 并用 `modal app list` 核实 | F 阶段功能验证 |
| Modal `H100!:N`（逐位比较） | **缺口**：`MODAL_GPUS["H100"] = "H100"`（`modal_runner.py:59-60`），没有 `!`，Modal 可能升级为 H200。在修复前，逐位实验（A6/A8/DEV-GATHER）不走 launcher 的 Modal 路径，改用独立的 Modal 脚本：`gpu="H100!:N"`，启动时断言 `nvidia-smi --query-gpu=name` 全部为 H100，否则退出。launcher 的修复归 Agent IMG / launcher 负责人，已列入待办。 | 同上 | 4.3 X3、4.6 X4 |
| Verda | — | — | **不使用**（在途 provider） |
| 本机 | — | — | 只跑 CPU 测试；本机 GPU 占用不作为约束，也不参与实验 |

镜像：一律使用 `MILES_NEXT_IMAGE` 的 digest 形式（`--rl-image docker:<repo>@sha256:<digest>`，`launcher.py:1243` 会校验）。每个 GPU 阶段开始前，由 Agent IMG 提供包含所需 fork 提交的新 digest；1.1 manifest 与这个 digest 一一对应。

### 8.2 pool 身份（写入 #66 study manifest 的 `resources`）

- `pool_id` = `infra-a-<实验 ID>-<UTC 日期>-<序号>`，例如 `infra-a-A2-20261001-1`；同时用作 SkyPilot 集群名或 Modal app 名的前缀，以便按名回收。
- `pool_epoch`：池内 GPU 集合每变化一次加 1；同一次租用内切换配置不改变 epoch（配置 epoch 由 3.2 的 journal 管理）。
- `gpus`：`[{uuid, model, node, index}]`，由租用后执行 `nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv` 得到，不预先填写。
- `standby`：备用卡按 config 的 `standby` 计入 GPU·h（§0 口径）。E0 的 T4R4/T4R2S2 固定配置中的 S 就是这里的备用卡。

### 8.3 每次租用必须记录的字段（写入该实验的证据目录 `rental.json`）

`pool_id`、云与 region、实例/集群/app ID、owner（`infra-a`）、用途（task 编号）、创建时间与释放时间（UTC）、GPU 型号与各卡 UUID（断言输出原文）、NUMA 与互联（`nvidia-smi topo -m` 原文、`lscpu` 的 NUMA 段）、主存（`free -g`）、镜像 digest、yeto commit、miles fork commit、备用卡数量、租期上限（硬超时）、预估费用与实际费用、回收方式与核实输出（`sky status` / `modal app list` / Nebius API）。无法核实的项写“未确认”。

### 8.4 租期与清理（机制优先）

1. 启动前，把本次的 task 编号、验收项、模型、GPU 型号与数量、预计时长、费用估算、事先登记的成功/失败条件与容差、硬超时写进计划文件并提交。
2. 云端限时：Modal 用 `timeout=`；SkyPilot 用 `--down` 与 autostop。
3. 独立于终端与 agent 的回收：用 `timeout <硬超时> <启动命令>` 包裹，再加一个后台 watchdog 脚本，到期后按记录的 ID 执行 `sky down <name>` 或 `modal app stop -y <name>`，并把核实结果写入 `rental.json`。
4. 结束或失败时：先拉日志、事件磁带和结果，再释放资源；删除本次创建的卷；核实没有残留资源。
5. 只操作 `infra-a-` 前缀或已记录 ID 的资源；不执行 `sky down -a` 这类全局命令；共享配置只读。

### 8.5 1.3 验收对照

- “计划可映射现有 launcher/harness 和 pool 身份”：见 §8.1、§8.2。
- “记录 GPU/NUMA/互联、主存、镜像、备用卡、租期及清理”：§8.3 规定每次租用记录的字段（实际值只有租用后才有），§8.4 规定租期与清理。
- “不把本机占用当限制”：见 §8.1 最后一行。
- “不把在途 provider 当已支持”：Verda 已从所有实验行和默认预算中排除（§0、§8.1）。
- “本任务交付计划，实际租用按实验执行范围进行”：本节只是计划，没有租用任何资源。
