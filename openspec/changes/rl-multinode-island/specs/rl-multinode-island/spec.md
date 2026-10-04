# rl-multinode-island Specification

## Purpose

把 yeto 的 RL learner island 从单节点 sky 集群扩展为 N 节点 sky 集群：以 (节点, GPU) 建模资源与放置，Ray head 固定在岛内 node0，弹性 cell 可跨节点声明但单个引擎/模型并行组不跨节点，任一节点失联进入 `RECOVERY_REQUIRED`，回收必须逐节点确认。单节点岛与旧资源 cfg 的行为保持不变。

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

### Requirement: 放置约束不允许引擎或模型并行组跨节点
每个 rollout 引擎的 `gpus_per_engine` 张卡 MUST 位于同一节点；trainer 的每个 `tp*pp*cp` 模型并行组 MUST 位于同一节点；standby 卡 rebind 到 cell 的目标卡 MUST 与该 cell 其余卡同节点。trainer 占用的 (节点数 × 每节点卡数) MUST 为矩形。违反者 SHALL 在启动前（launcher）或解析时（cfg）被拒绝。

#### Scenario: 引擎跨节点
- **WHEN** `gpus_per_node=8, gpus_per_engine=4`，cell 声明 bundles `[6,7,8,9]`
- **THEN** 以 `ValueError` 拒绝，信息含 "spans nodes"

#### Scenario: trainer 非矩形
- **WHEN** 16 卡岛要求 trainer 12 卡（n0 全部 8 + n1 的 4）
- **THEN** launcher 拒绝，提示 trainer 每节点卡数必须相等

#### Scenario: 合法的两节点分区
- **WHEN** 16 卡岛 `T8R8S0`，`rollout_engine_gpus=8`，`parallel tp=2 pp=1`
- **THEN** trainer 映射为 n0 的 8 卡（`--actor-num-nodes 1 --actor-num-gpus-per-node 8`），rollout 一个 cell 映射为 n1 的 8 卡

### Requirement: 弹性 cell 可跨节点声明但每个 cell 不跨节点
`--rl-elastic-declare-cells` 在多节点岛上 SHALL 可用。`placement_map_arg` 生成 `rollout_cells` 时 MUST 先按节点分块再按 `gpus_per_engine` 切分；跨节点剩余不足一个引擎的卡 MUST 进入 unbound。`bind_members` MUST 拒绝目标卡跨节点的绑定。

#### Scenario: 跨节点 standby 绑定
- **WHEN** 调用 `bind_members({cell}, gpus)` 且 `gpus` 分属两个节点
- **THEN** 抛出 `MembershipPlanError`，fork 的 `set_pg_view`/`rebind_cell` 未被调用

### Requirement: 每个 learner 以 recipe 推导的最少节点起步
launcher SHALL 按 recipe 的并行度（trainer `tp*pp*cp*ep` 最小副本 + rollout 引擎 `sglang.tp` + standby）推导每 learner 最少节点数；`--gpu` 低于最小值 MUST 拒绝；未显式给出节点数时默认取最小值。

#### Scenario: 低于最少节点
- **WHEN** recipe 要求 trainer 8 卡 + rollout 8 卡且 `gpus_per_node=8`，而 `--gpu` 给出 1 节点
- **THEN** 启动前拒绝并给出"至少 2 节点"

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
