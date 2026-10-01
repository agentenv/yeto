# Design: rl-multinode-island

状态：规划稿（CPU，未实现任何代码）。本文先给代码现状追踪，再给设计决策 D1–D12，最后列待用户决策项。

## 0. 现状追踪（只读，2026-10-01，integ-decl 8347351）

| 层 | 位置 | 现状 | 多节点缺口 |
|---|---|---|---|
| `--gpu` 语法 | `yeto/gpu_spec.py` | `cloud:[N x]G x gpu[@region]`，`ClusterSpec(num_nodes, gpus_per_node)`，`total_gpus = N*G` | 语法已支持 N；无"最少节点"校验 |
| 岛构造 | `yeto/launcher.py` L2172–2530 `make_miles_island_task` | `sky.Task(num_nodes=spec.num_nodes)`；`MASTER_ADDR=$(SKYPILOT_NODE_IPS|head -1)`；rank0 `ray start --head --port=6379 --temp-dir=$HOME/miles-ray` 后跑 `python -m yeto.rl.learner`；非 0 rank `until ray start --address=$MASTER_ADDR:6379`，然后 `while ray status; do sleep 5`；`num_nodes>1` → `network_tier=best`；`--actor-num-nodes {spec.num_nodes}` 透传 | 骨架可用。缺：worker 侧 `trap` 只在加入后才装；worker 不感知 learner 退出码；GPU/NCCL 环境变量（`NCCL_SOCKET_IFNAME` 等）未按云设置；重启循环 `yeto_rl_restart_loop` 只在 rank0 |
| 参数校验 | `yeto/launcher.py` L1620–1665 | TP×PP 整除 `total_gpus`、EP 整除、`fixed-partition` 要求 `spec.num_nodes == 1` 且 `actor = gpus_per_node − rollout − standby` | **硬限单节点**；`rl_actor_gpus_per_node` 把 rollout/standby 当作"本节点内"扣除 |
| 弹性 cfg | `yeto/rl/elastic_benchmark/capabilities.py::parse_configs/_parse_config_extras/placement_rejection/validate_pool` | config = `{trainer, rollout, standby, rollout_engine_gpus, parallel{tp,pp,cp,ep?}, placement{trainer:[uuid], rollout:[[uuid]], standby:[uuid]}, capacity}`；pool = `resources.gpus[{uuid, model, node, index}]`；`placement_rejection` 已禁止 rollout 引擎与 trainer 模型并行组跨节点 | pool 在岛内从未解析（cfg 的 `gpus` 为空，只有 `--rl-elastic-trainer-edges` 要求 `manifest_pool_gpus`）；无 `nodes` 段；无 cell→节点约束 |
| placement | `miles_adapter/placement.py` | `PlacementRequest(kind, trainer_gpus, rollout_gpus, gpus_per_engine, standby_gpus, bundle_map, rollout_cell_names)`；`placement_map` = 角色→逻辑 bundle 序号 `[0..N)`；`placement_map_arg` 把 cell 按 `gpus_per_engine` 连续段切 | 逻辑 bundle 无节点含义；连续段切分在多节点下可能跨节点 |
| 逻辑 bundle↔物理 | `miles_adapter/bundles.py::StartupBundles` | `pool_gpus[p]` = 逻辑 bundle p；通过 fork `get_pg_view` 得到 `(pg, reordered_bundle, gpu_id)` | 无节点字段；`view_for(gpus)` 不检查同节点 |
| Miles PG | `~/miles/miles/ray/placement_group.py` | `placement_group(bundles, strategy="PACK")`，InfoActor 取 `(node_ip, gpu_id)`，按 `(ip, gpu)` 排序得 `pg_reordered_*` | 排序后逻辑 bundle 天然按节点分块（node0 的 0..G-1，node1 的 G..2G-1）——这是可依赖的不变量，但当前代码没有断言 |
| 弹性 placement | `miles_adapter/elastic_placement.py::ElasticPlacement` | E1 从不移动 trainer GPU；plan 以 GPU id 元组表达 | 跨节点 plan 需节点约束 |
| cell 声明 | `run_config._rollout_cell_names`、`learner.py` L173/333/347、`rollout.py::bind_members/unbind_members` | `--rl-elastic-declare-cells` 需 `fixed-partition`；`bind_members` 要求 `per*len(cells)` 个不同 GPU，经 `StartupBundles.view_for` 生成 `set_pg_view` + `rebind_cell` | 不检查目标 GPU 同节点 |
| 执行 profile | `yeto/rl/engine/execution_profile.py` | `check_elastic_placement` 只看 `placement_kind` | 无节点维度；无需大改 |
| 恢复 | `evidence/infra-e1/recovery-design.md`、`controller.py::_recover_membership`、`trainer_transition.py`(`RECOVERY_REQUIRED`) | 重启后按启动形状重新声明 cell，差分恢复到 journal 提交成员；`RECOVERY_REQUIRED` 为终态 | 无"节点失联"事件；重启循环只覆盖 rank0 进程 |
| head/syncer | `launcher.py` L4340–4410 | head 模式 = 本机 syncer；否则独立 syncer 集群 `{prefix}-syncer`，`syncer_handle.head_ip` | 与岛内节点数正交，不改 |
| 回收 | `launcher.py` L4261–4300 `down + 云端核实`；`openspec/specs/head-run-teardown` | 按 cluster 名 down，再云端确认实例消失 | 多节点时云端核实需覆盖**每个**节点实例 |
| Modal | `yeto/modal_runner.py::validate_modal_shape` | 多容器岛要求整节点 GPU 型号 | 保持 |

## 1. 设计决策

### D1 岛 = N 节点 sky 集群，head 在 node0（保持现有拓扑）
- 沿用 `sky.Task(num_nodes=N)`，node0 = Ray head = learner 进程 = trainer rank0 所在；worker 节点只 `ray start --address` 并常驻。
- 不引入独立 head 节点：Flash-Next 的 trainer 是显存瓶颈，没有理由给 head 空出一台 GPU 机；Ray head 开销可忽略。
- worker 节点改为：`ray start --address` 成功后立即 `trap stop_miles_ray EXIT`；循环改为 `while ray status ... && ! [ -f $HOME/yeto-rl/learner-exit ]`，由 rank0 在 learner 退出时写 `learner-exit` 文件（经 sky 共享的 `~/sky_workdir` 不可靠，改用 Ray KV：rank0 退出时 `ray.experimental.internal_kv` 写标记，或直接依赖 head 退出使 `ray status` 失败）。基线：**依赖 head 退出**（当前行为），加一个 60 s 内重试上限，避免 worker 永远 sleep。
- 重启循环 `yeto_rl_restart_loop` 保持只在 rank0（重启的是 learner 进程，不是 Ray）；worker 节点在 Ray 存活期间不受影响。

### D2 资源 cfg schema：向后兼容的节点扩展
```json
{
  "nodes": 2,                         // 可选；缺省 1
  "gpus_per_node": 8,                 // 可选；缺省 = 池大小 / nodes 或由岛注入
  "configs": {
    "T8R8S0": {"trainer": 8, "rollout": 8, "standby": 0, "rollout_engine_gpus": 8,
               "parallel": {"tp": 2, "pp": 1, "ep": 4},
               "placement": {"trainer": ["n0:0","n0:1",...,"n0:7"],
                             "rollout": [["n1:0",...,"n1:7"]],
                             "standby": []}}
  },
  "edges": [...]
}
```
- `placement` 条目允许三种写法并统一解析成 `(node_index, local_gpu)`：`"n{k}:{g}"`（新）、整数逻辑 bundle `p`（旧，`node = p // gpus_per_node`）、uuid（现有 pool 写法，经 `gpus[*].node/index` 反查）。
- 无 `nodes` 字段的 cfg 行为完全不变（现有 `resources-8.json` 与全部 A4 证据照旧）。
- `validate_pool` 增加：有 `nodes` 时 `gpus` 若非空必须覆盖 `nodes*gpus_per_node` 且每节点 `index` 为 `0..G-1`。

### D3 逻辑 bundle 的节点不变量
- 定义：逻辑 bundle `p` 的节点 = `p // gpus_per_node`，本地卡 = `p % gpus_per_node`。
- 依据：Miles `_create_placement_group` 的 PACK + 按 `(node_ip, gpu_id)` 排序；sky 的 `SKYPILOT_NODE_IPS` 顺序 = node rank。实现阶段在 `StartupBundles.__init__` 增加断言：`get_pg_view` 返回的 `(node_ip, gpu)` 序列按节点分块且块长 = `gpus_per_node`，否则 `BundleMapError`（fail closed，不猜）。
- 注意 IP 排序 ≠ node rank 排序：PACK 排序按 IP 数值，sky node0 不一定 IP 最小。因此"node_index"以 **PG 排序后的块序号** 为准，并在 journal 里记录 `node_index → (sky_rank, ip, hostname)` 映射；Ray head 所在节点由 `ray.nodes()` 的 `is_head` 判定，不假设它是块 0。

### D4 放置约束（节点感知）
在 `PlacementRequest.__post_init__`/`validate_bundle_map`/`capabilities.placement_rejection` 统一执行：
1. 每个 rollout 引擎（`gpus_per_engine` 张卡）同节点；
2. trainer 的模型并行组 `tp*pp*cp` 同节点（Megatron TP 走 NVLink；PP 跨节点在本 change 不开放，待裁定 Q3）；
3. EP：`ep` 组可跨节点，但要求 `ep % (gpus_per_node / (tp*pp*cp)) == 0` 或 `ep` 整除单节点内组数（整节点对齐）；
4. standby 卡 rebind 到 cell 时目标卡同节点；
5. 任一角色的 GPU 集合不要求整节点，但 **trainer 集合必须是整数个"模型并行组"且每组不跨节点**。
违反 → 启动前 `ValueError`（launcher 侧）或 `ManifestError`（cfg 侧），不进入 GPU。

### D5 launcher 校验与 `rl_actor_gpus_per_node` 的重定义
- `fixed-partition` 去掉 `num_nodes == 1` 限制；trainer 卡数 = `total_gpus − rollout_num_gpus − standby`，要求能被 `tp*pp*cp` 整除且按 D4 可放置；`--actor-num-nodes/--actor-num-gpus-per-node` 的推导改为"trainer 占用的节点数与每节点卡数"——当 trainer 不是整节点时（例如 16 卡岛 T8R8：trainer 占 node0 全部 8 卡），`actor_num_nodes=1, actor_num_gpus_per_node=8`；当 trainer 跨节点且每节点占用数不等时拒绝（Miles `actor_num_nodes*actor_num_gpus_per_node` 必须是矩形）。
- 新增 `--rl-min-nodes-per-learner`（默认由 recipe 推导，见 D8），`spec.num_nodes` 低于最小值即拒绝。
- Modal：保持 `validate_modal_shape`；多容器岛同样走 D4。

### D6 Ray/NCCL 环境
- rank0/worker 统一导出：`NCCL_SOCKET_IFNAME`（按云：nebius=`eth0`，aws/gcp 用默认探测），`NCCL_IB_DISABLE`（无 IB 时置 1），`GLOO_SOCKET_IFNAME`；由 launcher 根据 `spec.cloud` 选择，写入 run 脚本 prelude（与现有 `_ELASTIC_TEST_EXPORTS` 同一机制）。
- 这些值作为"待 GPU 验证确认"项，CPU 阶段只保证脚本生成正确。

### D7 弹性 cell 跨节点声明
- `--rl-elastic-cells` 名称不变；`placement_map_arg` 的 `rollout_cells[*].bundles` 切分改为"按节点分块后再按 `gpus_per_engine` 切"，禁止一段跨节点；跨节点剩余不足一个引擎的卡进入 unbound。
- `rollout.py::bind_members` 增加同节点检查（通过 `StartupBundles.node_of(gpu)`）。
- `describe_cells` 输出增加 `node` 字段（yeto 侧从 bundles 推导，不要求 fork 改接口）。
- fork 接口需求：**无新增**。`set_pg_view`/`rebind_cell`/`rollout_cells` 现有语义足够；只在 yeto 侧收紧输入。

### D8 Flash-Next recipe 并行度表达与"最少节点"
- `parallel` 段新增 `ep`（已在 `PARALLEL_DIMS` 则沿用）与 `sglang: {tp, ep, dp}`；recipe 通过 `--rl-model-recipe` 给出默认值（LoRA：按 NEXT-WEEK-PLAN 的 TP2/PP1/EP?，SGLang TP8/EP8；精确值须对照 pin 的 Miles `scripts/models/*flash-next*` 核对——本机 `~/miles` 无该文件，待实现阶段从镜像内 Miles 读取）。
- 最少节点推导：`min_nodes = ceil((trainer_min_gpus + rollout_min_gpus + standby) / gpus_per_node)`，其中 `trainer_min_gpus = tp*pp*cp*ep_lcm`（模型并行最小副本），`rollout_min_gpus = sglang.tp`。Flash-Next LoRA：trainer 8 + rollout 8 → 2 节点；全参按 32 卡 recipe → 4 节点。
- "每 learner 先起最少节点"：launcher 默认 `num_nodes = min_nodes`，更大需显式 `--gpu N x`。

### D9 故障域与恢复语义
- 新事件：`node_lost(node_index, reason)`，来源：(a) Ray `NodeDiedEvent`/`ray.nodes()` alive=false 轮询（driver 每 10 s）；(b) trainer 集体通信超时（`--rl-distributed-timeout-minutes`）；(c) sky job 的节点失败。
- 规则：**任一节点失联 → 岛进入 `RECOVERY_REQUIRED`**（事务层终态，与 E3 一致），learner 进程以非零码退出；不做部分节点续跑。理由：trainer 或 rollout 的模型并行组一定损失一个成员；而且 Flash-Next 规模下任何降级配置都无法装下权重。
- head 节点失联 = learner 失联，由外层（restart loop / sky job 失败）处理，与单节点语义相同。
- 重启恢复（E1-D）：restart loop 只重启 learner 进程；若 `ray.nodes()` 的 alive 节点数 < N，则重启前置检查失败 → 直接 `RECOVERY_REQUIRED`，不进入差分恢复。恢复要求"同形"：`node_index → gpus_per_node` 结构一致（主机名可变）；journal 增加 `topology` 记录。
- 节点重建（sky 自动恢复 spot 节点）不在本 change 自动处理：需要用户裁定 Q4。

### D10 回收原子性
- `sky down <island>` 对多节点集群是整体操作；现有"云端核实"改为对集群的**全部**实例 ID 逐个核实（launcher 在 up 后记录 `node_instance_ids[]`）。任一节点未确认 → 保留记录、非零退出、列出节点（与 `head-run-teardown` 规则一致）。
- spot：多节点 spot 任一节点被抢占 = 节点失联（D9）。

### D11 本地演练（CPU，多进程模拟多节点 Ray）
- 用单机 `ray start --head` + `ray start --address` 两个进程（同机不同 `--temp-dir`，`--num-gpus=8 --resources='{"node:n1":1}'` 伪造 GPU 资源）模拟 2 节点；`InfoActor` 返回的 node ip 相同，因此用 **自定义资源标签 `yeto_node:k`** 作为演练中节点身份；`StartupBundles` 的节点分块断言在演练模式下读取该标签（`YETO_MULTINODE_SIM=1`），生产模式读取 node ip。
- 演练覆盖：PG 创建与分块、placement map 跨节点 cell 切分、`bind_members` 同节点拒绝、`node_lost` 注入（kill 第二个 ray 进程）→ `RECOVERY_REQUIRED`、恢复前置检查拒绝。
- 不启动 sglang/Megatron；用现有 `ForkMembership/ElasticFakePool` 假件。

### D12 GPU 验证（最后，单独审批）
- 2 节点 × 最小卡数（nebius 2×1×H100 若可申请，否则 2×2）。判据与硬超时见 tasks §3。预算上限 $60。

## 2. 数据流（新增/修改处）
```
--gpu cloud:2x8xh100 ──parse_gpu_spec──> ClusterSpec(N=2,G=8)
   ├─ launcher 校验: min_nodes(recipe) ≤ N; D4 放置约束(CPU)
   ├─ cfg: nodes=2, placement "n0:*"/"n1:*" ──parse_configs──> ResourceConfig(+node 解析)
   ├─ sky.Task(num_nodes=2, run: rank0 head+learner / rank>0 join)
   └─ 岛内: Miles PG(PACK) ──StartupBundles(断言分块)──> 逻辑 bundle p → (node p//G, p%G)
          ├─ placement_map_arg: rollout_cells 按节点分块切分
          ├─ bind_members: 同节点检查
          └─ driver: node_lost 轮询 → RECOVERY_REQUIRED
```

## 3. 待用户决策
- **Q1 EP 跨节点**：Flash-Next 512 专家，EP 组是否允许跨节点（D4 规则 3）？允许则需确认 Miles recipe 的 EP 值与节点对齐；不允许则 trainer 最少卡数上升。
- **Q2 rollout 与 trainer 是否同节点混布**：16 卡 LoRA 最小配置建议 node0=trainer 8、node1=rollout 8（引擎 TP8 整节点）；若用户要 colocated（同卡训推），则多节点 colocated 另行设计（本 change 只做 fixed-partition 多节点，colocated 多节点仅保留参数校验）。
- **Q3 PP 跨节点**：本 change 禁止（TP×PP 组同节点）。全参 32 卡若 recipe 用 PP>1 跨节点，需放开并补 NCCL 验证。
- **Q4 节点失联后的自动重建**：方案 a）整岛 `RECOVERY_REQUIRED` 退出，由人工/外层 `yeto up` 重建（本 change 基线）；b）restart loop 等待 sky 自动恢复节点后重入恢复（需节点自愈检测，+1 天）。
- **Q5 GPU 验证规格**：2×1×H100（≈$5/h，验证 Ray/PG/cell/故障域）还是 2×8×H100（≈$62/h，顺带验 Flash-Next 4 层变体跨节点）？建议前者，后者并入 S2。
- **Q6 pool 解析**：岛内是否要求运行时解析 `resources.gpus`（nvidia-smi uuid）并与 cfg 对账？建议：多节点时必解析（fail closed），单节点保持可选。
