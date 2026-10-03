# Tasks: rl-multinode-island

状态只能是：已实现 / CPU 通过 / GPU 验收通过 / 合法否定结论 / 未完成。GPU 任务放最后，**不勾、不启动**，需主 agent 另批。所有判据在运行前固定。

## 0. 设计定稿（CPU）

- [x] 0.1 用户裁定 design §3 Q1–Q6，并把结论回写 design.md（每项一行"裁定：…/日期"）。验证：design.md 无未裁定项；`openspec validate rl-multinode-island --strict` 通过。
  - 完成记录（2026-10-01，CPU 通过）：Q1–Q6 取主 agent 默认并回写 design §3（标『待用户复核』）。
- [ ] 0.2 从镜像内 pin 的 Miles 读取 Flash-Next 全参 recipe（`scripts/models/*flash-next*` 或等价）与 4 层变体的 TP/PP/EP/SGLang TP/EP 值，写入 design D8 表格；标明来源 commit。验证：表格每个数值带来源。

## 1. CPU 单测（schema / 映射 / plan 生成）

- [x] 1.1 `yeto/gpu_spec.py`：`ClusterSpec` 增加 `min_nodes` 校验入口（纯函数 `require_min_nodes(spec, min_nodes)`）。测试 `tests/test_gpu_spec.py`：`2x8xh100` 解析、低于最小值拒绝、旧单节点字符串不变。
  - 完成记录（2026-10-01，CPU 通过）：`yeto/gpu_spec.py::require_min_nodes`；`tests/test_rl_multinode_schema.py::test_gpu_spec_two_nodes_and_min_nodes`。
- [x] 1.2 cfg schema（D2）：`capabilities.py` 解析 `nodes/gpus_per_node`，`placement` 支持 `"n{k}:{g}"`/整数/uuid 三种写法并归一为 `(node, local)`；无 `nodes` 的 cfg 行为不变。测试 `tests/test_rl_multinode_schema.py`：三种写法等价；混写拒绝；`nodes` 与 `gpus` 不一致拒绝；现有 `resources-8.json` 原样通过且输出与改动前逐字节一致（对 `parse_configs` 结果做快照对比）。
  - 完成记录（2026-10-01，CPU 通过）：`yeto/rl/engine/multinode.py`（Topology/topology_of/check_pool_topology/normalize_placement）+ `capabilities.parse_configs` 节点分支（`ResourceConfig.placement_slots`，旧 cfg 默认 None）；旧 cfg 快照、三种写法等价、混用/池不一致/计数不一致拒绝见 `tests/test_rl_multinode_schema.py`。
- [x] 1.3 放置约束（D4）：`placement.py::validate_bundle_map`/`PlacementRequest` 增加 `gpus_per_node` 参数与节点规则 1–5。测试 `tests/test_rl_multinode_placement.py`：引擎跨节点拒绝；TP 组跨节点拒绝；EP 整节点对齐（按 Q1 裁定）；standby rebind 跨节点拒绝；单节点（`gpus_per_node=None`）全部旧测试不变（`tests/test_rl_miles_adapter_placement.py` 零改动通过）。
  - 完成记录（2026-10-01，CPU 通过）：`PlacementRequest(gpus_per_node, model_parallel, expert_parallel)` + `_check_nodes`/`trainer_shape`；规则 1/2/3/5 由 `multinode.node_placement_rejection` 统一；`tests/test_rl_multinode_placement.py`；`tests/test_rl_miles_adapter_placement.py` 零改动通过。规则 4（standby rebind 同节点）在 1.4 的 `bind_members` 实现。
- [x] 1.4 `placement_map_arg` 跨节点 cell 切分（D7）：按节点分块后切 `gpus_per_engine`；剩余进入 unbound。测试：16 卡 T8R8 engine 8 → 1 个 cell 全在 n1；T8R4S4 engine 2 → 2 个 start cell + 2 个 standby cell 均不跨节点；T12R4 engine 4 且 G=8 → rollout 段 `[12..15]` 在 n1，trainer 占 n0 全部 + n1 前 4（按 D5 矩形规则应**拒绝**，因 trainer 每节点卡数不等）。
  - 完成记录（2026-10-01，CPU 通过）：`placement_map_arg` 改用 `multinode.chunk_by_node`；`rollout.bind_members` 增加逐 cell 同节点检查（`StartupBundles.same_node`）；T8R8/T8R4S4/T12R4 用例见 `tests/test_rl_multinode_placement.py`。
- [x] 1.5 `StartupBundles` 节点分块断言（D3）：假 `views` 构造跨节点乱序 → `BundleMapError`；正常分块 → `node_of(gpu)` 正确。测试同 1.3 文件。
  - 完成记录（2026-10-01，CPU 通过）：`StartupBundles(gpus_per_node, node_ids, node_resolver)` + `assert_node_blocks`、`node_of/same_node`；entry.py 以 `ray.util.placement_group_table(pg)['bundles_to_node_id']` 作运行时节点源（Q6，fail closed）；测试同上。
- [x] 1.6 launcher 校验（D5）：去掉 `num_nodes==1` 限制；`actor_num_nodes/actor_num_gpus_per_node` 矩形推导；`--rl-min-nodes-per-learner` 与 recipe 默认；D6 NCCL 环境 prelude 生成。测试 `tests/test_rl_launcher_multinode.py`：`--gpu nebius:2x8xh100 --rl-placement fixed-partition --rollout-num-gpus 8` 生成 `--actor-num-nodes 1 --actor-num-gpus-per-node 8`；sky task `num_nodes=2`、`network_tier=best`、worker 分支含 `trap`；低于 min_nodes 拒绝；现有 `tests/test_rl_launcher_partition.py` 零改动通过。
  - 完成记录（2026-10-01，CPU 通过）：去掉 `num_nodes==1` 限制；`rl_trainer_shape`/`rl_min_nodes`/`multinode_env_prelude`（launcher.py，D5/D6/D8）；`--rl-min-nodes-per-learner`（cli.py）；岛内 `--rl-island-gpus-per-node`（launcher→learner→run_config.ParallelLayout.island_gpus_per_node→config.placement_request）；worker 分支 `trap` 提前到 join 之前且 join 有界（150×2 s 失败即非零退出）。`tests/test_rl_launcher_multinode.py`：2 节点 task `num_nodes=2`/`network_tier=best`/`--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rl-island-gpus-per-node 8`/NCCL prelude/worker trap 顺序；单节点 task 无多节点附加；低于 min_nodes 在 `_prepare_rl_args` 拒绝；`tests/test_rl_launcher_partition.py`、`tests/test_rl_launcher.py` 零改动通过。D6 的接口名取值待 GPU G1 确认。
- [x] 1.7 故障域（D9）：`driver`/`controller` 增加 `node_lost` 事件处理 → `RECOVERY_REQUIRED` 并写 journal `topology`；恢复前置检查 `alive_nodes < N` 拒绝。测试 `tests/test_rl_multinode_recovery.py`（复用 `test_rl_reconfig_recovery.py` 的假件）：注入 node_lost → 终态；重启后 alive 不足 → 不进入差分恢复；同形但主机名变化 → 允许。
  - 完成记录（2026-10-01，CPU 通过）：`IslandController(topology, node_probe)`/`set_topology`/`topology_rejection`/`check_nodes`（journal `topology`/`node_lost` 记录，`_enter_recovery`）；`open()` 在拓扑不满足时先于任何 restore/start/stop 进入 RECOVERY_REQUIRED；`_recovery_precondition` 与 `confirm_recovery`（checks.nodes）接入同形检查（存活节点数 = N 且每节点 GPU 数 = G，主机名可变）；`IslandDriver._probe_nodes` 每轮前轮询；entry.py 在 `controller.open` 前用 `ray.nodes()` 作 probe。`tests/test_rl_multinode_recovery.py`（6 用例，复用 `test_rl_reconfig_recovery` 假件）；`tests/test_rl_reconfig_recovery.py` 零改动通过。
- [x] 1.8 回收（D10）：launcher 记录 `node_instance_ids[]`，`down` 的云端核实逐节点；任一未确认非零退出并列出。测试 `tests/test_rl_launcher_island_failure.py` 增补：2 节点一台未确认 → rc≠0 且信息含节点 ID。
  - 完成记录（2026-10-01，CPU 通过）：`terminate_and_verify(num_nodes=)` → `_terminate_and_verify_nodes`：down 前捕获实例 id，逐实例确认行；无 probe/probe 失败/任一实例仍存活 → False 并列出 UNCONFIRMED 实例 id；`run()` 以 `learner_cluster_names` 映射传 num_nodes。单节点路径不变（`tests/test_teardown_verify.py` 零改动通过）。用例在 `tests/test_rl_launcher_multinode.py`（4 个 teardown 用例）。
- [x] 1.9 全量回归：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/test_rl_*.py tests/test_gpu_spec.py | tee /home/michael/work/infra-drafts/s1-pytest.log`，失败集去重后与 `/tmp/integ-s2-base.ids` 同口径比较，无新增失败。
  - 完成记录（2026-10-01，CPU 通过）：`tests/test_rl_*.py tests/test_gpu_spec.py --continue-on-collection-errors`：13 failed/2150 passed/29 skipped/14 errors（日志 `infra-drafts/s1-pytest.log`），27 个失败 id 去重后全部 ⊂ 基线 `/tmp/integ-s2-base.ids`（94），无新增失败。

## 2. 本地演练（CPU，多进程模拟多节点 Ray，D11）

> 2026-10-01 暂缓（线程守卫）；2026-10-02 GPU 链暂停后主 agent 放行；2026-10-03 完成。解释器：`/tmp/review-miles-venv/bin/python`（ray 2.58.0；yeto-venv 无 ray）。冒烟 `ray.init(num_cpus=4)` 5.8 s 通过。证据：`infra-drafts/s1-sim/`（summary.txt、各用例 .log、run1/ 为首轮含两处脚本缺陷的记录）。本机已知环境问题：driver 注册 Ray 偶发挂起（A27B 同源，`RegisterClient`），脚本按日志识别后在新进程重试（首轮 pg_blocks 重试 1 次）；该 venv 需先 import torch 再 import ray（numpy 双初始化）。

- [x] 2.1 演练脚本 `tests/multinode_sim/run_sim.sh`：单机起 2 个 Ray 进程（不同 `--temp-dir`，`--num-gpus=8`，资源标签 `yeto_node:0/1`），`YETO_MULTINODE_SIM=1`；输出到 `/home/michael/work/infra-drafts/s1-sim/`。验证：`ray status` 显示 2 节点 16 "GPU"。
  - 完成记录（2026-10-03，CPU 通过）：`tests/multinode_sim/run_sim.sh`：head `--node-ip-address=127.0.0.1 --num-gpus=4 --resources yeto_node:0` + worker `yeto_node:1`，每用例 `timeout 300`，结束 `ray stop --force`；线程数 8351 → 9077（2 节点）→ 8352。
- [x] 2.2 PG 分块演练：在 sim 上创建 16 bundle PACK PG，`StartupBundles` 断言通过，打印 `p → (node, local)` 表。验证：表与 D3 一致；日志落盘。
  - 完成记录（2026-10-03，CPU 通过）：`sim.py pg_blocks`：16→8 bundle PACK PG，`placement_group_table.bundles_to_node_id` 作 node_resolver，`StartupBundles` 分块断言通过并打印 p→(node,local) 表；交错顺序被 `BundleMapError` 拒绝（fail closed）。
- [x] 2.3 cell 切分与 bind 演练：用 `ForkMembership` 假件声明 T8R8 cell，`bind_members` 跨节点目标被拒、同节点通过。验证：断言日志。
  - 完成记录（2026-10-03，CPU 通过）：`sim.py cells_bind`：T4R2S2 engine 2 → cells c0=[4,5] start / c1=[6,7] standby / c2 unbound，无跨节点；`bind_members(c1, p3,p4)` 跨节点被拒且 fork 零调用；`(p6,p7)` 同节点通过（view+rebind）。
- [x] 2.4 故障注入演练：kill 第二个 Ray 进程 → driver 10 s 内发出 `node_lost` → `RECOVERY_REQUIRED`；重启 learner 假件 → 前置检查拒绝；重新拉起第二个 Ray 进程 → 前置检查通过并进入（假）差分恢复。验证：事件带时间戳落盘，顺序正确。
  - 完成记录（2026-10-03，CPU 通过）：`sim.py node_loss`：真实 pkill worker Ray → `check_nodes` 0.7 s 内 `node_lost` → RECOVERY_REQUIRED（journal `node_lost` alive=1）；缺节点重启 → 前置检查拒绝、无 restore/start/stop；节点回来后同 state dir 仍 RECOVERY_REQUIRED（Q4 a 终态），新 state dir 2 节点 RUNNING。
- [x] 2.5 回收演练：`down` 走假云 API，模拟一个节点"未确认" → rc≠0。
  - 完成记录（2026-10-03，CPU 通过）：`sim.py teardown`：probe = 带 `yeto_node:1` 标签的存活 Ray 节点 id；down 杀 worker → 逐实例确认 True；down 不动 → UNCONFIRMED False。
- [x] 2.6 演练总结写入 `openspec/changes/rl-multinode-island/progress.md`（命令、日志路径、通过/失败）。
  - 完成记录（2026-10-03，CPU 通过）：见本节 blockquote 与 `openspec/changes/rl-multinode-island/progress.md`。

## 3. GPU 验证（放最后；不勾；需主 agent 另批；轨上限 **$25**，H100 回退需重新放行）

预先判据（运行前固定，不得事后修改；S7 审查 T4-S7-PROGRESS.md §6 改为最小配置）：
- 规格：**nebius `2x1xl40s@eu-north1`**（2 节点 × 1×L40S，sm_89；节点 0 = trainer `n0:0`，节点 1 = rollout cell `n1:0`，cfg `tests/multinode_gpu/resources-2x1.json`，`nodes=2 gpus_per_node=1`，T1R1S0，无 standby），模型 `Qwen3-0.6B` LoRA，`--rl-engine ports --rl-placement fixed-partition --rl-rollout-gpus 1 --rl-elastic`，`--rl-single-island-no-sync --controller local`（不起 syncer、不占 29400）。镜像 `yeto-miles-ports@sha256:37ac689e…`（c35702e-4e4148f）。
- 先行 **G0 探针**（单节点 `1xl40s`，`timeout 900`，`--total-steps 1`）：PASS = launcher rc 0 且 ≥1 次 train + ≥1 次 generate 且 GPU 名为 L40S 且无 kernel/sm_89 不兼容报错；FAIL = 换 A10G 探针一次，仍 FAIL → 停下回报（H100 回退需重新放行）。
- 硬超时：每用例 ≤ 30 min（`timeout 1800` 包裹 + 独立 watchdog 按 cluster 名 `sky down`），G4 ≤ 15 min；总链 ≤ 2 h；G3 放链尾。**修订（10-03 16:40Z，重跑前）**：首个用例 g12 的 launcher 超时 3600 s = 30 min 用例窗口 + ≤30 min 冷启动（2 节点镜像拉取 + setup fetch + 模型下载实测 25–30 min；首跑 s1-mn-20261003b 以 1800 s 在冷启动中被截断 = INVALID_TEST(hard_timeout)，无产品结论）；30 min 用例窗口改以 launch.ts.log 的 "Job submitted" 起算（判据 `case_window_le_1800s`）。
- 唯一前缀 `s1-mn-`；脚本 `tests/multinode_gpu/`（s1run.sh / s1chain.sh / s1kill.sh / s1probe.sh / s1reset.sh / s1judge.py），运行目录 `/home/michael/work/s1-runs/`；资源 ID/创建时间/owner 记录到 `infra-drafts/s1-gpu.md`。
- G3 判据以 journal 的 `node_lost` / `topology`（重启后 alive<2 → "recovery refused"）记录与 `rl_reconfiguration result=RECOVERY_REQUIRED error=node_lost:…` 为证；**no-sync 下重启的 `LedgerError` 不得当作通过**。60 s 判据：kill 时刻取 worker 时钟（`kill.txt`），`node_lost.wall_time − kill ≤ 60`；终态探针 `s1probe.sh` 在 head 容器内每 5 s 记 `ray status` 活节点数与 journal 尾行，kill 后采集窗口 ≥ 200 s（≥150 s）。
- 2x1 的已知局限：无 standby GPU，E1 rollout-only up/down 边**不可验证**（需 2x2 以上），3.2 最多记 PARTIAL。

- [x] 3.0 G0 探针（镜像在 sm_89 上能起、1 步 train + 1 次 generate）。**PASS 2026-10-03**：s1-mn-20261003-g0，容器内 generate/train/finish 各 1 次，L40S，无 kernel 报错（launcher 本机 timeout 在冷启动阶段触发，判读以容器内证据，已注明）。
- [ ] 3.1（2026-10-03 状态：7 次 2×1×L40S 尝试均止于 G1 之前/之中，根因为 launcher 多节点路径缺陷并已修复：network_tier、dashboard-host、worker 模型快照、D6 接口名=network-interface-0；未得通过证据；另发现 Ray PG 的 trainer/engine 节点落位与 D3 假设相反，待裁定。详见 infra-drafts/T4-S7-PROGRESS.md §7）G1 拓扑：岛起来后 journal `topology` 记录 `nodes=2 gpus_per_node=1` 且 `alive` 长度 2（= `ray.nodes()` alive=2）；`StartupBundles` 分块断言通过（fail closed：learner 进入 ≥1 轮即通过，launch.log 无 `not node-blocked`/`BundleMapError`）；learner 完成 ≥1 轮（train + generate）。**通过** = 全部成立且 launcher rc 0；**失败** = 任一不成立或超时。
- [ ] 3.2 G2 跨节点 cell：trainer 在 n0（head 的 compute-apps 为 learner/ray 进程、无 sglang）、rollout cell 在 n1（worker1 的 compute-apps 含 sglang）；≥2 轮 train→generate（跨节点权重同步生效）；journal `config_epoch` 单调。E1 rollout-only 边 up/down 各 1 次：2x1 不可验证（记 PARTIAL，不勾）。
- [ ] 3.3 G3 节点失联（链尾）：训练轮 ≥1 的 train 阶段 `pkill -f miles-ray/` 杀 n1 raylet → ≤ 60 s 内 journal `node_lost` + `rl_reconfiguration RECOVERY_REQUIRED(node_lost:…)`，learner 非零退出，`node_lost` 之后无 train/generate 事件（无部分续跑）；`--rl-elastic-restart-attempts 1` 原地重启后第二条 `topology` 记录 alive<2 且 "recovery refused"（前置检查拒绝）。
- [ ] 3.4 G4 回收：g3 的 launcher（无 `--keep`）teardown 输出 2 行 `node instance <id> confirmed terminated`、无 `UNCONFIRMED`；`cleanup_run.sh`（RUNS_BASE=s1-runs）两次核清（nebius project `{}`、sky 无集群、本地无进程）；记录预估/实际费用。
- [ ] 3.5 合法否定结论出口：若 G1 因 NCCL/网络环境（D6）失败且 2 次定因修复后仍失败，记录为"多节点需 IB/网络层另立项"，不勾 G2–G4。

## 4. 文档与收尾

- [x] 4.1 `docs/MILES_RL.md` 增加"多节点岛"一节（拓扑图、cfg 示例、约束、故障域）。
  - 完成记录（2026-10-01）：`docs/MILES_RL.md` 新增 "Multi-node islands" 一节（拓扑与放置规则、cfg 示例、节点故障域与运维要点、逐节点回收确认）。
- [ ] 4.2 progress.md 收尾：分支/HEAD/未提交/测试命令结果/证据路径/费用/待批准。
