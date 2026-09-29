# 岛内训推弹性：修订后的阶段拓扑与思路

目标是 **yeto + miles** 的分区执行、GPU角色转移与控制闭环。原串行共置保留为兼容基线，不再限定目标架构；允许必要的driver、worker和通信结构性重构。本文图示是设计建议，未认证的DP/重叠能力不是当前源码事实。

对照：[设计](design.md)、[行为spec](specs/island-elastic-reconfiguration/spec.md)、[实施任务](tasks.md)、[引擎端口与版本固定](../rl-engine-ports/design.md)。A完成源码/论文/资源与依赖调查，下面五图对应E0、E1–E3、C、D1、D2；F仅设计后续云扩缩与岛间接口。

图用纯ASCII，`-->`为标注的控制/状态/数据流，`<-->`为协议通信。T是trainer rank，R是TP1 engine，S是池内备用容量。8卡是通过yeto租用的示例池，不是硬编码上限；所有卡含备用卡计入成本。角色框并列只表示物理分区，能否并发由ExecutionProfile决定。

## 阶段1：固定训推分区基线（E0）

```text
                [Yeto existing cloud provisioning]
                       | before experiment
                       v
               [Ready pool + topology + identity]
                       |
+----------------------v----------------------------------------+
| One island / one logical learner                              |
|                                                               |
| [Existing syncer] <--> [Yeto algorithm bridge]                  |
|                              |                                |
|                     [ExecutionProfile + readiness]            |
|                              |                                |
|        [Yeto IslandDriver] -> [Engine ports] -> [MilesAdapter] |
|                          |             |                      |
|                     ready task     ready task                 |
|                          v             v                      |
| [Trainer partition: T0..T5]    [Rollout partition: R0 R1]      |
| [G0 G1 G2 G3 G4 G5]            [G6 G7]                         |
|           ^                         |                         |
|           |                         v                         |
|           +------------ [Complete groups / rewards]            |
|           |                                                   |
|           +--> [Weight publish + exact policy ACK] --> Rollout |
|                                                               |
| [Metrics: actual dependencies / waits / memory / throughput]   |
+---------------------------------------------------------------+
```

先建立明确的分区映射和任务依赖，再谈动态调整。`partitioned-serial`可以先跑通：即使两组GPU独立，下一batch依赖新策略时仍等待更新和发布。`partitioned-overlap`只对同一算法契约允许的ready任务开放；有资源不代表有合法工作可执行。

| 运行模式 | 定位 | 首阶段验收 |
|---|---|---|
| serial-colocated | 当前兼容参考 | 原有样本、step、外层行为不变 |
| partitioned-serial | 目标runtime的基本能力 | 固定分区可运行且保持同一算法语义 |
| partitioned-overlap | 有条件的目标能力 | 依赖、策略版本、缓冲/反压和时间线验证通过 |

DynaResize主要研究分离的异步流水线。若本项目要获得同样重叠只能改变陈旧度/更新规则，则单列算法设计，不以调度开关隐式实现。本阶段应交付固定配置收益面，允许“该profile尚无有利转换边”的结论。

思路：把“架构能分区”和“算法允许并发”拆开验证，性能对比才有解释力。任务对应2.1–2.4。

## 阶段2：手动重配置与完整恢复（E1/E2/E3）

```text
 [Manual target] --> [Yeto controller + profile/pause guard]
                                 |
                                 v
        [Yeto IslandController + IslandDriver; port verbs]
                                 |
                   [Fence admission / quiescent cut]
                                 |
                 +---------------+----------------+
                 |                                |
                 v                                v
    E1: [Rollout membership]        E2/E3: [Complete state cut]
    [Router + payload ACK]                     |
                 |                 [Verified recovery backing]
                 |                             |
                 |                 [Release / restore workers]
                 |                             |
                 +---------------+-------------+
                                 v
                [Validate / commit epoch / resume]
                                 |
                    [Journal + sample/update ledger]

  E1 (trainer unchanged, S is reserved capacity):
    [T0 T1 T2 T3 | R0 R1 | S S]
             --> [T0 T1 T2 T3 | R0 R1 R2 R3]

  E2 (same layout, new worker generation):
    [Trainer gen=a] --> [Complete cut] --> [Trainer gen=b]

  E3 (certified example, same 8-GPU pool):
    GPU       G0  G1  G2  G3  G4  G5  G6  G7
    P62       T0  T1  T2  T3  T4  T5  R0  R1
                         |
                         v
    P44       T0  T1  T2  T3  R0  R1  R2  R3
```

三个子里程碑解决不同问题，可分开验收：

- **E1**：保留trainer，从备用池增加rollout副本或退回备用池，验证drain、路由、权重发布和事务。不是trainer弹性。
- **E2**：卡数不变，销毁/重建trainer并完整恢复参数、master、moments、scheduler、RNG、样本和外层进度，建立可靠恢复路径。
- **E3**：在E2证据上改变DP，并把实际GPU在训推角色间转移。P62/P44需独立认证，恢复spike可以先用DP1↔2；不能把论文比例当作已合法配置。

固定TP=PP=CP=EP=1的非打包示例：GBS=48，DP6×microbatch1×accum8与DP4×microbatch1×accum12消费相同逻辑batch。实际optimizer分片、loss normalization和RNG映射另验。

图中R编号是目标逻辑索引，不代表复用旧进程/权重；目标generation和物理映射必须更新。新旧worker默认顺序占用冲突GPU，若用备用卡重叠初始化要显式计入容量。learner/bridge保持，原通信组不跨进程继承。

思路：手动指定目标，把决策与执行分离。除GPU健康还检查样本、更新、策略和外层一致性。释放前可取消，释放后从cut重建；外层提交不确定时停止而非盲目重放。对应任务3.1–4.8。

## 阶段3：按实测瓶颈降低切换成本（C）

```text
             [Validated manual transition + profiler]
                              |
                         choose bottleneck
                              v
          [Yeto driver transaction -> port verbs]
                              |
                  [Complete cut / state ownership]
                              |
            +-----------------+-------------------+
            |                 |                   |
            v                 v                   v
    [Durable backing] [Host state backing] [Live source state]
            |                 |                   |
            +-----------------+-------------------+
                              |
                  [Bounded transport buffers]
                              |
                              v
                  [Target runtime generation]
                   |                    |
         [Model / compute ready] [Optimizer state ready]
                   |                    |
                   +----------+---------+
                              v
             [Per-operation dependency fences]
                              |
          [Commit profile startup conditions / RESUMING]
                              |
         [First-step validation + all restore obligations]
                              |
                           SUCCESS

 Optional experiments, selected by measured bottleneck:
 [Worker process reuse] [Weight-group prewarm] [Init overlap]
```

完整状态的载体与传输buffer是两件事。图中的backing可组合使用，不代表任意一种都具有同等故障保证：释放源前须证明状态完整、校验通过且有满足故障域要求的恢复依据。只用小buffer不能在源全部退出后凭空恢复完整状态。

E基线全状态ready后运行；C才可认证延迟optimizer materialization。先查Megatron后端最早依赖点，不假设只在optimizer.step需要。若开放早期计算，未ready操作被fence；配置epoch已提交也不等于恢复已成功，事务保持RESUMING，直到首step与恢复义务验证完成。失败遵守update/外层账本，不倒退已提交epoch。

论文还描述保留进程；本方案把它列为独立实验，区别于保留Ray placement group。预热限于已验证通信组，目标worker实际存在才可建立communicator。每项优化与hard rebuild比较，包括后台工作、首step拖慢、峰值和额外备用GPU成本。

思路：优化关键路径，但不把成本移出计时范围。只选择证据支持的瓶颈，不一次实现全部论文机制。对应任务5.1–5.7。

## 阶段4：shadow与半自动建议（D1）

```text
 [Runtime metrics] --> [Cause classification + profile model]
                                  ^
 [Measured transition costs] -----+
 [Certified configs / edges] -----+
                                  |
                                  v
                        [Shadow recommendation]
                                  |
                      [Explainable recommendation]
                      source / target / epoch
                      gain / cost / expiry
                                  |
                                  v
                        [Human approval]
                                  |
                                  v
            [Recheck expiry / epoch / load / pause guard]
                         |                     |
                       valid                 invalid
                         |                     |
                         v                     v
          [Same manual transaction]      [Reject with reason]
                         |
                 [Result / learning record]
```

shadow只记录建议。半自动才允许人选择建议并触发，但人工批准不能绕过安全条件；批准时负载、epoch或profile已经变化，就拒绝该建议，不替换成另一个目标自动执行。

建议必须区分GPU饱和、工具等待、长尾与同步阻塞，使用当前执行模式的端到端模型。首step的后台恢复代价属于切换成本。还没有可靠收益边时可给出“保持当前配置”的建议，无需为了演示而强制切换。

思路：在自动执行前让人检查判断是否合理，并校准收益预测。此阶段独立交付，不把shadow等同半自动。对应任务6.1–6.3。

## 阶段5：岛内自动控制（D2）

```text
                 [Yeto algorithm bridge / pause veto]
                                  |
 [Metrics] --> [Profile-aware prediction] <-- [Cost records]
      ^                           |
      |              [Certified profitable transition]
      |                           |
      |             [Dwell / cooldown / failure gate]
      |                           |
      |   [Manual override] ------+
      |   [Mode / disable] -------+
      |                           v
      |               [Single-transaction controller]
      |                           |
      |                           v
      |          [Yeto driver safe point -> port verbs]
      |                     |              |
      |                     v              v
      +--------------- [Trainer]       [Rollout]
      +--------------- [Journal / recovery / first-step metrics]

 [Yeto cloud lifecycle] --> [Already ready pool + optional standby]
                                      |
                                      v
                        [Certified island configurations]
```

自动化改变触发方式，执行协议不变。开启条件是当前profile至少一条认证转换边具有可重复净收益；论文百分比、功能运行成功和瞬时GPU利用率都不能替代该证据。

```text
conservative_saved_time(window)
    > upper_bound_full_switch_cost + safety_margin
```

持续失衡、最短停留、cooldown、频率限制和失败停用共同避免抖动。收益不确定或训练即将结束保持现状。禁用auto阻止新自动请求，已有事务继续完成或恢复。

云侧在实验前供给资源，图中没有自动扩池通道。后续扩池要先云就绪、认证拓扑/版本并提交pool epoch，再允许岛内使用；缩池先排空。它们不等于增加或减少DiLoCo成员。

思路：以实测收益和完整故障处理为自动控制依据。比较兼容默认、同profile最佳固定与动态，分别归因执行模式变化和resize收益。对应任务6.4–6.7。

## 依赖与可并行工作

| 工作 | 必须先取得的证据 | 可并行推进 |
|---|---|---|
| 固定分区E0 | A的资源/版本与依赖契约 | 完整状态保存研究 |
| 手动rollout E1 | 分区映射、安全点与权重身份 | 同形恢复E2 |
| trainer/角色转移E3 | E2完整恢复与分片/batch实验 | 已通过rollout边的成本优化 |
| 成本优化C | 对应边可工作且有profiling | 建议界面与trace分析 |
| 半自动D1 | 合法边、可解释成本/收益模型 | 按新数据校准模型 |
| 自动D2 | 半自动验收、重复净收益与恢复门槛 | F的接口设计 |

采用云租卡意味着可以安排合适资源做这些实验；固定池是单次验证边界，不是永久部署上限。F只规划在线云扩缩、跨岛分配与成员协议，保持当前单岛实现范围清晰。
