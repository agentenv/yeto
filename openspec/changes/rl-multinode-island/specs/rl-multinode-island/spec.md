# rl-multinode-island Specification

## Purpose

把 yeto 的 RL learner island 从单节点 sky 集群扩展为 N 节点 sky 集群：以 (节点, GPU) 建模资源与放置，Ray head 固定在岛内 node0，弹性 cell 可跨节点声明但单个引擎与 trainer 节点内组（`tp*cp`）不跨节点（EP/PP 允许跨节点），任一节点失联进入 `RECOVERY_REQUIRED`，回收必须逐节点确认。单节点岛与旧资源 cfg 的行为保持不变。

## ADDED Requirements

### Requirement: 岛是 N 节点 sky 集群且 Ray head 在 node0
launcher SHALL 把 `--gpu cloud:NxGxgpu` 的一个条目解析为一个 N 节点、每节点 G 卡的 sky 集群作为一个 learner island。node rank 0 MUST 启动 Ray head 并运行 learner 进程；其余节点 MUST 以 worker 身份加入同一 Ray 集群并在 head 退出后自行清理本节点的 Miles Ray 进程。系统 MUST NOT 为 Ray head 单独分配节点。

#### Scenario: 两节点岛启动
- **WHEN** 以 `--gpu nebius:2x8xh100 --rl-engine ports` 启动
- **THEN** 生成的 sky task `num_nodes == 2`、`network_tier == best`
- **AND** rank 0 的 run 脚本包含 `ray start --head` 与 `yeto.rl.learner`，rank 1 的分支包含 `ray start --address` 且在加入后立即注册退出清理

#### Scenario: 单节点岛行为不变
- **WHEN** 以 `--gpu nebius:8xh100` 启动
- **THEN** 生成的 task 与本 change 之前逐字节相同（除版本号外）

### Requirement: 资源 cfg 以 (节点, GPU) 建模并向后兼容
弹性资源 cfg SHALL 接受可选的 `nodes` 与 `gpus_per_node` 字段；`placement` 条目 SHALL 接受 `"n{k}:{g}"`、逻辑 bundle 整数、GPU uuid 三种写法并归一为 `(node_index, local_gpu)`。不含 `nodes` 的 cfg MUST 与之前解析结果完全一致。

#### Scenario: 旧 cfg 原样通过
- **WHEN** 解析不含 `nodes` 的 `resources-8.json`
- **THEN** `parse_configs` 的输出与改动前一致，且不出现任何节点约束拒绝

#### Scenario: 节点数与池不一致
- **WHEN** cfg 声明 `nodes: 2, gpus_per_node: 8` 而 `gpus` 列出 12 个条目
- **THEN** 解析以 `ManifestError` 拒绝，错误信息给出期望 16 与实际 12

#### Scenario: 混用写法
- **WHEN** 同一 `placement` 列表内既有 `"n0:1"` 又有整数 `3`
- **THEN** 解析拒绝并指出混用

### Requirement: 逻辑 bundle 与节点的对应是被断言的不变量
逻辑 bundle `p` 的节点 SHALL 定义为 `p // gpus_per_node`。岛启动时系统 MUST 用 Miles placement group 返回的 (node, gpu) 序列核对该分块；核对失败 MUST 以错误终止启动，不得猜测映射。

#### Scenario: PG 排序与分块不符
- **WHEN** `get_pg_view` 返回的 bundle 节点序列不是按 `gpus_per_node` 整块分组
- **THEN** `StartupBundles` 抛出 `BundleMapError`，learner 不进入训练

### Requirement: 放置约束默认不允许引擎或 trainer 节点内组跨节点，可显式放行
默认（无显式放行）每个 rollout 引擎的 `gpus_per_engine` 张卡 MUST 位于同一节点；trainer 的每个节点内组（连续 `tp*cp` 个 trainer rank，TP 默认留在节点内）MUST 位于同一节点，且 `tp*cp` MUST ≤ `gpus_per_node` 并整除之；standby 卡 rebind 到 cell 的目标卡 MUST 与该 cell 其余卡同节点。trainer 占用的 (节点数 × 每节点卡数) MUST 为矩形。违反者 SHALL 在启动前（launcher）或解析时（cfg）被拒绝。"TP 留节点内"只是默认偏好（用户裁定 2026-10-04 v2）：cfg `parallel.allow_cross_node_tp: true` 或 CLI `--rl-allow-cross-node-tp` MUST 放行 trainer `tp*cp` 组跨节点（保留整除与矩形约束）；cfg `parallel.allow_cross_node_engine_tp: true` 或 CLI `--rl-allow-cross-node-engine-tp` MUST 放行 rollout 引擎跨节点，且该引擎 MUST 由整节点组成。放行时系统 MUST 在启动日志 WARN 并在 journal `topology.layout` 记录 `cross_node_tp` / `cross_node_engine_tp`。同一 `(node, gpu)` 槽位 MUST NOT 在 trainer/rollout/standby 中出现两次（混布默认不重叠占卡）。

#### Scenario: 引擎跨节点
- **WHEN** `gpus_per_node=8, gpus_per_engine=4`，cell 声明 bundles `[6,7,8,9]`
- **THEN** 以 `ValueError` 拒绝，信息含 "spans nodes"

#### Scenario: trainer 非矩形
- **WHEN** 16 卡岛要求 trainer 12 卡（n0 全部 8 + n1 的 4）
- **THEN** launcher 拒绝，提示 trainer 每节点卡数必须相等

#### Scenario: 合法的两节点分区
- **WHEN** 16 卡岛 `T8R8S0`，`rollout_engine_gpus=8`，`parallel tp=2 pp=1`
- **THEN** trainer 映射为 n0 的 8 卡（`--actor-num-nodes 1 --actor-num-gpus-per-node 8`），rollout 一个 cell 映射为 n1 的 8 卡

#### Scenario: 同节点混布（Q2 裁定 2026-10-04）
- **WHEN** 2×2 岛 cfg `T2R1S1` 的 `placement` 为 trainer `["n0:0","n1:0"]`、rollout `[["n0:1"]]`、standby `["n1:1"]`，`--tensor-parallel 1 --pipeline-parallel 2 --rl-rollout-gpus 1 --rl-standby-gpus 1`
- **THEN** 接受：rollout 引擎与 trainer rank 0 共用 n0 的不同卡；launcher 由 cfg placement 推出 `--actor-num-nodes 2 --actor-num-gpus-per-node 1` 并透传 `--rl-island-bundle-map {"trainer":[0,2],"rollout":[1],"standby":[3]}`；learner 侧 `PlacementRequest(bundle_map).trainer_shape()` 与之一致，否则启动前拒绝

#### Scenario: 混布非矩形被拒
- **WHEN** 2×2 岛 cfg 的 trainer 为 `["n0:0","n0:1","n1:0"]`（n0 两卡、n1 一卡）
- **THEN** launcher 在任何云操作前以 `ValueError` 拒绝，信息含 "trainer GPUs per node must be equal"

#### Scenario: TP 组跨节点被拒
- **WHEN** 24 卡岛（3×8）`T16R8S0`，`parallel tp=16 pp=1`（或 `tp*cp` 不整除 8，例如 `tp=3`）
- **THEN** 以 `ValueError`/`ManifestError` 拒绝，信息含 "in-node (tp*cp) group ... spans nodes"（或 "not divisible by ... tp*cp"）；launcher 侧 `tp*cp > gpus_per_node` 或不整除时拒绝，信息含 "RL TP*CP must fit and divide one node"

#### Scenario: 显式放行 trainer TP 跨节点（v2 裁定）
- **WHEN** 2×4 岛 `parallel tp=8`，cfg `parallel.allow_cross_node_tp: true`（或 `--rl-allow-cross-node-tp`）
- **THEN** 接受，trainer 形状 2×4；launcher 输出 WARN；journal `topology.layout.cross_node_tp = 1`；不带开关时拒绝，信息含 "spans nodes" 与 "--rl-allow-cross-node-tp"

#### Scenario: 显式放行 SGLang TP8 跨两台四卡节点
- **WHEN** 4×4 岛 `T8R8S0`，`rollout_engine_gpus=8`，rollout 引擎 placement 为 n2 与 n3 的全部 8 卡，`parallel.allow_cross_node_engine_tp: true`
- **THEN** 接受，`rollout_cells` 把该引擎切为一个 8 bundle 的整节点块 cell；learner 收到 `--rl-allow-cross-node-engine-tp`；同一引擎若只取 n2 的 4 卡 + n3 的 2 卡则拒绝，信息含 "whole 4-GPU nodes"

#### Scenario: 混布槽位重复被拒
- **WHEN** 2×2 岛 cfg 的 rollout 为 `[["n0:1"]]` 且 standby 为 `["n0:1"]`
- **THEN** 以 `ManifestError` 拒绝，信息含 "n0:1 more than once"

### Requirement: 跨节点 TP 引擎按完整副本扩缩容
允许跨节点的 rollout 引擎 SHALL 作为一个整体副本（其全部节点的卡一起）上线或下线。rollout-only 边的源与目标 rollout 卡数 MUST 均为 `rollout_engine_gpus` 的整数倍；cfg 中放行跨节点引擎的 config 其 `rollout` MUST 为 `rollout_engine_gpus` 的整数倍；`bind_members` 的跨节点目标 MUST 为整节点块。系统 MUST NOT 支持通过单独摘除引擎的一个节点实现缩容。

#### Scenario: 只摘一个节点被拒
- **WHEN** 引擎 8 卡跨 2×4，请求从 rollout 8 卡到 rollout 4 卡的 rollout-only 边
- **THEN** `plan()` 以 `Rejected` 拒绝，信息含 "single node of it cannot be removed"；cfg 解析时该目标 config 以 `ManifestError` 拒绝，信息含 "whole number of 8-GPU cross-node engine replicas"

#### Scenario: 整副本上下线
- **WHEN** rollout 8 → 16 卡（新增一个跨节点 8 卡引擎）
- **THEN** 接受，plan 的 target_engines = source_engines + 1

### Requirement: trainer 的 EP 与 PP 组允许跨节点
一个 learner 的 trainer MAY 占用多个节点、每节点多卡；其 EP 组与 PP 组 MAY 跨节点（用户裁定 2026-10-04）。放置校验 MUST NOT 对 EP 组施加节点对齐要求（仅要求 `tp*cp*ep` 整除 trainer 卡数），MUST NOT 对 PP 组施加节点规则；trainer 仍 MUST 满足矩形与 `tp*cp` 节点内规则。

#### Scenario: 跨节点 PP
- **WHEN** 24 卡岛（3×8）`T16R8S0`，`rollout_engine_gpus=8`，`parallel tp=8 pp=2`
- **THEN** 接受；trainer 占 n0+n1（`--actor-num-nodes 2 --actor-num-gpus-per-node 8`），两个 PP stage 各在一个节点，rollout cell 在 n2

#### Scenario: 跨节点 EP
- **WHEN** 24 卡岛（3×8）`T16R8S0`，`parallel tp=2 pp=1 ep=8`
- **THEN** 接受；EP 组由 8 个 TP 组构成、横跨 n0 与 n1；若 `tp*cp*ep` 不整除 trainer 卡数（如 `T8` 配 `tp=2 ep=8`）则拒绝，信息含 "expert parallel 8 needs trainer GPUs divisible by tp*cp*ep = 16"

### Requirement: 弹性 cell 可跨节点声明但每个 cell 不跨节点
`--rl-elastic-declare-cells` 在多节点岛上 SHALL 可用。`placement_map_arg` 生成 `rollout_cells` 时 MUST 先按节点分块再按 `gpus_per_engine` 切分；跨节点剩余不足一个引擎的卡 MUST 进入 unbound。`bind_members` MUST 拒绝目标卡跨节点的绑定。

#### Scenario: 跨节点 standby 绑定
- **WHEN** 调用 `bind_members({cell}, gpus)` 且 `gpus` 分属两个节点
- **THEN** 抛出 `MembershipPlanError`，fork 的 `set_pg_view`/`rebind_cell` 未被调用

### Requirement: 每个 learner 以 recipe 推导的最少节点起步
launcher SHALL 按 recipe 的并行度（trainer `tp*cp*ep*pp` 最小副本 + rollout 引擎 `sglang.tp` + standby）推导每 learner 最少节点数；最小副本 MAY 大于一个节点（EP/PP 跨节点），此时 MUST 为整节点倍数，且 `tp*cp` MUST ≤ `gpus_per_node` 并整除之；`--gpu` 低于最小值 MUST 拒绝；未显式给出节点数时默认取最小值。

#### Scenario: 低于最少节点
- **WHEN** recipe 要求 trainer 8 卡 + rollout 8 卡且 `gpus_per_node=8`，而 `--gpu` 给出 1 节点
- **THEN** 启动前拒绝并给出"至少 2 节点"

#### Scenario: 跨节点最小副本
- **WHEN** recipe `tp=2 pp=2 ep=4`（最小副本 16 卡）+ rollout 引擎 8 卡，`gpus_per_node=8`
- **THEN** 最少节点为 3；`tp=16` 则拒绝，信息含 "TP stays inside a node"；`tp=2 pp=3 ep=2`（12 卡）则拒绝，信息含 "not a whole number"

### Requirement: 任一节点失联进入 RECOVERY_REQUIRED
岛内 driver SHALL 监测 Ray 节点存活。任一节点失联时，岛 MUST 进入 `RECOVERY_REQUIRED` 终态并以非零码退出，MUST NOT 以部分节点继续训练或 rollout。重启恢复前置检查 MUST 要求存活节点数等于声明节点数且拓扑同形（每节点卡数一致，主机名可变），否则拒绝进入成员差分恢复。

#### Scenario: worker 节点失联
- **WHEN** 训练中 node1 的 raylet 停止
- **THEN** 60 s 内 journal 记录 `node_lost` 与 `RECOVERY_REQUIRED`，learner 非零退出

#### Scenario: 重启时节点不足
- **WHEN** restart loop 重启 learner 而 `ray.nodes()` 存活节点为 1 < 2
- **THEN** 前置检查失败，不调用 `restore_membership_state`

### Requirement: 多节点回收逐节点确认
`yeto down` 对多节点岛 SHALL 以 cluster 为单位 down，并对每个节点实例做云端核实。任一节点未确认 MUST 导致非零退出并列出该节点实例 ID；不得把 sky 对集群的"不存在"视为所有节点已释放。

#### Scenario: 一个节点未确认
- **WHEN** 2 节点岛 down 后云端仍能查到 node1 实例
- **THEN** 命令非零退出，输出含 node1 实例 ID 与下一步

### Requirement: 多节点岛在运行时对账 GPU UUID
多节点岛启动时（拓扑预检之后、placement group 之前）系统 MUST 在每个节点上采集 `nvidia-smi --query-gpu=index,uuid` 并与声明的池（cfg `gpus[*].uuid` 和/或 journal 中上一化身的基线）逐卡比对，结果写入 journal `gpu_pool` 记录（incarnation、每节点 uuid、source、rebind、mapping、diffs、accepted）。形状不符 MUST 拒绝；uuid 不符默认 MUST 拒绝；仅当 `--rl-elastic-accept-rebind` 给出且形状一致时 MAY 接受并记录旧→新映射。首次无任何声明时 SHALL 把观测写入基线。单节点岛不采集、行为不变。

#### Scenario: UUID 不符默认拒绝（Q6 裁定 2026-10-04）
- **WHEN** journal 基线为 2×2 的 4 个 uuid，重启后 worker 被换机、其 2 个 uuid 不同，且未给 `--rl-elastic-accept-rebind`
- **THEN** 启动以 `RuntimeError` 拒绝（信息含 "gpu_pool" 与 "--rl-elastic-accept-rebind"），journal 追加 `gpu_pool{accepted:false, diffs:[...]}`，不写 RECOVERY_REQUIRED 终态（允许带 flag 重启）

#### Scenario: 带 accept-rebind 接受并记录
- **WHEN** 同上，但给出 `--rl-elastic-accept-rebind`，节点数与每节点卡数一致
- **THEN** 接受；journal `gpu_pool{accepted:true, rebind:true, mapping:{旧uuid:新uuid}}`，后续 placement 按新 uuid 绑定

### Requirement: GPU UUID 重绑定防重复占用、角色冲突与旧进程误重入
在 UUID 对账接受（含 `--rl-elastic-accept-rebind` 重绑定）之后、placement group 之前，系统 MUST 再做三项检查并把 uuid → 角色映射写入 journal `gpu_pool.roles`：(a) 重复占用——观测到的 uuid 在池内唯一，且不与另一活跃岛绑定的 uuid 重叠；(b) 角色冲突——按 bundle map（或 leading 布局）每个 uuid 恰属一个角色（trainer/rollout/standby）；(c) 旧进程误重入——journal 基线或观测池中的任一 uuid 若仍被旧化身的存活进程持有（节点上 `/tmp/yeto-rl-incarnation/<uuid>.json` 记录的化身 id 与 pid 存活），MUST 拒绝并要求先 `yeto down`。任一项失败 → journal `gpu_pool{accepted:false}` + tape `rl_reconfiguration RECOVERY_REQUIRED`，不写 journal 终态（允许处理后重启）。接受后系统 SHALL 在每个节点为所绑定的 uuid 写入本化身标记。

#### Scenario: 旧化身仍占卡
- **WHEN** 新化身启动，n1 的 uuid 标记文件记录的旧化身 pid 仍存活
- **THEN** 启动以 `RuntimeError` 拒绝，信息含该 uuid、旧化身 id 与 "yeto down"；journal `gpu_pool.accepted=false`

#### Scenario: 一卡两角色
- **WHEN** bundle map 使 bundle 2 同时属于 trainer 与 rollout
- **THEN** 拒绝，信息含 "both trainer and rollout"

#### Scenario: 正常重绑定
- **WHEN** 换机后 uuid 不同、带 `--rl-elastic-accept-rebind`，无旧化身存活，无重叠
- **THEN** 接受；journal `gpu_pool.roles` 为 4 个 uuid 各一角色；各节点写入化身标记
