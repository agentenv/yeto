# Proposal: rl-multinode-island

## Why

yeto 的 RL learner island 目前在实践中是**单节点** sky 集群：`--gpu` 语法虽允许 `cloud:NxGxgpu`（`yeto/gpu_spec.py::ClusterSpec.num_nodes`），launcher 也会在 `num_nodes>1` 时生成多节点 sky task（node0 `ray start --head`、其余节点 `ray start --address`，`yeto/launcher.py` L2450–2500），但往下所有与岛内布局有关的层都把岛当成一台机器：

- `--rl-placement fixed-partition`（弹性与 E1/E2/E3 的前提）在 `yeto/launcher.py` L1636 硬性要求 `spec.num_nodes == 1`；
- placement/profile 以**扁平 bundle 序号**建模（`miles_adapter/placement.py::PlacementRequest`、`placement_map`、`bundles.py::StartupBundles`），没有节点维度；
- 弹性资源 cfg（`--rl-elastic-resources`，如 `cfg/resources-8.json`）只有 `trainer/rollout/standby` 计数；`capabilities.placement_rejection` 虽已检查"引擎/模型并行组不得跨节点"，但依赖的 `resources.gpus[*].node` 池从未在岛内解析；
- 弹性 cell（`--rl-elastic-declare-cells` → fork-F-R1 `rollout_cells`）按连续 bundle 段切分，隐含同节点；
- E1-D 重启恢复（`evidence/infra-e1/recovery-design.md`）假设每次重启按"启动形状"重新声明 cell，没有"某个节点失联"的故障域。

下周主线基模 `Qwen/Qwen3.8-Flash-Next`（125B 总参/6B 激活，bf16 ≈360 GB）LoRA 最少需要 2 节点（16×H100），Miles 全参 recipe 需要 32 卡（4 节点）。没有多节点岛，S2（全尺寸 LoRA 试跑）与 S3（全参准备）都无法启动（见 `infra-drafts/NEXT-WEEK-PLAN.md` S1/S2/S3）。

用户决定（NEXT-WEEK-PLAN §2b）：多节点岛立项为新 change；**每个 learner 先起最少节点/卡**；**先把代码逻辑与设计捋顺（design.md + CPU 测试 + 本地演练），再上卡**；禁止贸然跑测试。

## What Changes

- **岛 = N 节点 sky 集群**：`ClusterSpec.num_nodes` 成为 ports 路径的一等参数；launcher 生成的 run 脚本保持 node0 为 Ray head、worker 节点加入并"等 head 退出"，补齐 worker 节点侧的 Ray 清理、GPU/NCCL 环境与多节点 `network_tier`；Modal 多容器岛维持 `validate_modal_shape` 的整节点规则。
- **(node, gpu) 资源建模**：弹性资源 cfg 增加可选 `nodes` 段与 `gpus[*].node`；`placement`/`rollout_cells` 可用 `(node, local_gpu)` 或逻辑 bundle 序号表达；旧的纯计数 cfg（无 `nodes`）**原样有效**，等价于 `nodes=1`。
- **placement/profile 节点感知**：`PlacementRequest`/`placement_map`/`StartupBundles` 增加节点维度，规则：rollout 引擎、trainer 的 TP×PP(×CP) 模型并行组、standby→rollout 的 rebind 目标均不得跨节点；trainer DP 维度可跨节点；EP 组按 recipe 允许跨节点但必须整节点对齐。
- **Ray head 位置**：head 固定在岛内 node0（当前行为），`--pin-rollout-manager-to-head` 不变；明确 head 节点既是 Ray head 也承载 trainer rank0；不引入"独立 head 节点"。
- **弹性 cell 跨节点声明**：cell 名绑定到 `(node, gpu...)` 段；`bind_members`/`rebind_cell` 的目标 GPU 列表必须同节点；`describe_cells` 增加节点字段。
- **故障域与恢复**：新增"节点失联"故障域：任一 worker 节点失联 → 岛进入 `RECOVERY_REQUIRED`（不做部分节点续跑）；head 节点失联 = 岛失联（现有 learner 失联语义）。重启恢复（E1-D）在**整岛同形重建**后按原规则恢复成员；记录节点身份（sky node rank + 主机名）以便判定"同形"。
- **成本与回收**：多节点 `sky down` 的原子性要求——岛的回收必须按 cluster 名整体 down，并对每个节点实例做云端核实；任何节点未确认 → 非零退出并列出。
- **最少起步**：按 recipe 推导"每 learner 最少节点数"（LoRA Flash-Next：2 节点；全参：4 节点），launcher 在 CPU 侧拒绝低于最小值的 `--gpu`。
- 不改变：legacy 路径、单节点岛的全部行为与测试、外层 syncer 协议。

## Capabilities

### New Capabilities

- `rl-multinode-island`：多节点 learner island 的资源建模、放置约束、Ray 拓扑、弹性 cell 跨节点声明、节点故障域与回收语义。

### Modified Capabilities

- 无（`island-elastic-reconfiguration` 仍是 `rl-infra-spec` 的 delta spec，尚未进主 specs；本 change 对其的影响以"兼容约束"写在本 change 的 spec 与 design 中，待其归档后再做 MODIFIED delta）。

## Non-goals

- 跨岛扩展、自动增减节点（云扩缩容需另批）。
- 独立 head 节点、Ray head 高可用。
- 部分节点失联后的"降级续跑"（trainer DP 缩减跨节点）——本 change 只做整岛 `RECOVERY_REQUIRED`。
- 多模态、MTP、长上下文。
- Qwen3.8-Flash-Next 的 LoRA 布局（M3）与 HF→torch_dist 转换（M4/S3）。

## Impact

- 代码（实现阶段）：`yeto/gpu_spec.py`、`yeto/launcher.py`（岛构造、校验、down）、`yeto/rl/elastic_benchmark/capabilities.py`（cfg schema）、`yeto/rl/engine/miles_adapter/{placement,elastic_placement,bundles,rollout,entry}.py`、`yeto/rl/engine/run_config.py`、`yeto/rl/engine/controller.py`（节点故障域）、`yeto/rl/learner.py`。
- 文档：`docs/MILES_RL.md` 岛拓扑一节。
- 测试：新增 `tests/test_rl_multinode_*.py`（CPU）；本地多进程 Ray 演练脚本（CPU）；GPU 验证放最后、需另批。
- 费用：GPU 验证 2 节点预算 ≈$60（硬上限），在全局 $600 内。
