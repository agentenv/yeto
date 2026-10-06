# D1-2/D1-3 上卡前逻辑复核（2026-10-05，代码 = d1-gpu，已 merge integ-decl 6324cd4b）

链脚本：`tests/multinode_gpu/s11d1chain.sh <prefix>`。顺序：T2R1S1 s17（冷启）→ T2R2S0 s17 → T2R1S1 s29 → T2R2S0 s29 → d1e1（KEEP_M3=1）→ T1R3S0 s17 → T1R3S0 s29（风险最高的放链尾）→ trap：杀所有 watchdog、`sky down <本集群>`、cleanup_run.sh 跑两遍、sky status 检查。每次运行之间：先确认集群 UP，再 s1reset.sh（两节点都要求无计算进程且显存 <1.5G，不干净就停止）；每次运行后 kill 它自己的 watchdog（否则会把共享集群 down 掉）；预算守卫为 3.5 h 后不再开新运行。

## (a) 用例参数
- d1sweep：`--keep`；seed 用 `SEED`（`--seed ${SEED:-17}`）；`--total-steps 6`；T2R1S1 = ELASTIC22（rollout 1 + standby 1，PP2）；T2R2S0 = rollout 2，无 standby，PP2；T1R3S0 = rollout 3，PP1，用 resources-2x2-t1r3.json；三个配置都开 `--rl-observe-timeline`。不需要 attestation（不发请求）。
- d1e1：沿用 m3 参数，加 6 个触发器（up@train rid1/5/9，dn@3/7/11；expected_config_epoch 0..5 递增）。
- **发现并已修复的问题 1**：fp_local22 算 attestation 指纹时写死 `--total-steps 6`。实测指纹随 total-steps 变化（6 → 0d17e24b…，14 → ae19cdd6…），因此 d1e1 改了步数后，所有请求都会被 controller 以"no capability attestation"拒绝（与 s8-m1m3a M3 FAIL 同一根因）。已修复：s1run 把 `--total-steps $STEPS --seed $SEED` 传给 fp_local22。核对：在 HEAD 上用 6/17 重算，仍得到 0d17e24b，说明 merge 没有改变指纹。
- **发现并已修复的问题 2**：fp_local22 的 stdout 现在第一行是 launcher 的提示行（`[launcher] nebius/eu-north1: VM image … pre-pulled`），原先的 `| python3 json.load` 会解析失败（exit 67）。已改为 `| tail -1`。
- **发现并已修复的问题 3**：12 步时 dn3 在 rid 11 触发，安全点在 rid 12，此时训练已结束，dn3 拿不到 SUCCEEDED。改为 14 步（dn3 之后还有 rid 12、13 两轮）。
- 复用 m3 的方式：attestation 写入 `$R/attestation-m3.json`；分支条件扩为 `m3 || d1e1`。

## (b) 布局
- 在 launcher 层做了 CPU 干跑：T2R2S0 的 trainer shape 为 (2,1)，T1R3S0 为 (1,1)，都通过校验，bundle map 由 cfg placement 推导。controller 要求 initial_config 必须存在于 resources 中，三个配置都满足。
- T1R3 的运行时风险：rollout 引擎落在 n1:0、n1:1，与 trainer 不在同一节点，需要跨节点发布权重。这条路径已被 T2R2S0（n1:1 引擎）和 M3 的 up 边覆盖过；新增的风险点是 n1 上同时有 2 个引擎，以及 n1 上没有 trainer。回退方案：T1R3 放在链尾，失败就记为无效，不重跑，2.4 只用 T2R1S1 和 T2R2S0 两个配置下结论（与计划一致）。

## (c) 判据与采集
- 2.4 有效性与轮时由 `s1cost.py sweep` 判定：rc0、rl_round_trained 数 = steps、gpu-*.txt 为 L40S 且 4 个 UUID 与 gpu_pool 一致、无 RECOVERY_REQUIRED。每轮时长 = 该 rid 的 generate+train+outer_sync 加上随后一次 publish，取 rid≥1 的中位数。用 M3 旧数据验证可跑通：中位数 13.5 s。
- 5.1：`s1cost.py <run>` 按请求输出 wait_safe/drain/init/verify/resume/阻塞合计/首轮额外。所需字段都在 journal 与 rl-island tape 中。
- 采集窗口 ≥ 采集延迟：puller 每 10 s 拉取一次，launcher 退出后还有一次终拉（rc.txt 延迟 20 s 写出）。M3 旧证据里的 journal 含 finalization 记录，说明终拉可以拿全。触发器是容器内 0.5 s 轮询，相对约 13 s 的轮时足够。终态探针（s1probe）在容器内运行。
- 1.7 load sample：M3 在 `--rl-observe-timeline` 下产生了 rl_load_sample；本链所有用例都开了该开关。

## (d) 代码逻辑（HEAD 6324cd4b）
- 8036c769 到 6324cd4b 的变更：auto.py、recommend.py、controller（recommend_mode 默认 disabled，只新增 inbox 的 `mode` 动词）、timeline 小改。E1 的 request/plan/fork 路径没有行为变化，指纹也不变（见 a）。
- s1reset 会清掉 ~/yeto-output 与 elastic-state，上一次运行的 tape 和 inbox 不会误触发 s1inwatch。

## (e) 时长与费用
- 历史参照：M1 冷启 18 min；暖启的 M3 用 8 min。本链共 7 次运行：18 + 5×8 + 10（d1e1）+ 清理 ≈ 75 min，约 $12。
- 硬顶：每次 HARD 3000 s；预算守卫 3.5 h（$32），低于预登记上限 $37（含重跑 ≤$57）。

---
# S11 合并链复核（2026-10-05 第二版，代码 d1-gpu，已 merge integ-decl eb94f35c）——取代上面 L40S-only 链

用户裁定：批准 H200；L40S 若有容量就优先用 L40S；阶段总额度 $300，**本链上限 $150**（链内预算守卫）；一次上机尽量多采数据；不得编造数据，没跑到的段在结果文件里写"未执行"。

## SKU 与时价（`sky show-gpus H200 --cloud nebius`，2026-10-05 实查）
| SKU | 时价（按需） |
|---|---|
| nebius eu-north1 `gpu-h200-sxm_8gpu-128vcpu-1600gb`（8×H200 141GB） | **$36.00/h** |
| nebius eu-north1 2×`gpu-l40s-*_1gpu`（2×2 L40S） | $9.14/h（gpu-spend.md 既有 run） |

## 链 `tests/multinode_gpu/s11h200chain.sh <prefix>`
| 段 | 内容 | 集群 | 预计 |
|---|---|---|---|
| 1 sweep (2.4) | T2R1S1/T2R2S0 × seed 17/29，6 轮 | L40S `<P>-l`（PREFER_L40S=1）；首次 provision 失败**立即**回落 H200 `<P>-h`（不等待） | L40S 冷启 18 min + 3×8 min；H200 同量级 |
| 2 e1 (5.1) | d1e1：3 对 up/down，14 轮，attestation 按本次 steps/seed/GPU/resources 重算 | 同上 | 约 10 min |
| 2b lp | 岛停止后，用 sglang 跑基座模型的 teacher-forced logprob：tp1 分别在各张卡上、tp2、tp4（L40S 只用 head 的 2 卡：tp1×2 + tp2），输出 lp.json 与 lp-summary.json | 同上 | 约 10 min |
| 2c T1R3S0 | seed 17/29 | 同上 | 2×8 min |
| 3a fnboot | fnrun fn8s（--keep）：拉起挂载 FS 的 H200 `<P>-f`。torch_dist 尚不存在，learner 预期在 ref-load 检查处快速失败；若 torch_dist 已存在，这次就直接作为 fnA | H200 `<P>-f`（FS 只能在 provision 时挂上，所以不能复用 sweep 集群，sweep 集群先 down） | 冷启约 15 min + 失败约 5 min |
| 3b fnprep | s11fnprep.sh：先 populate 4layer snapshot 到 FS 的 hub 布局并写 marker，再 convert 生成 torch_dist。df 闸门：populate 前要求剩余 ≥2.1×snapshot，convert 前 ≥1.1×snapshot；不满足即 FAIL 并跳过 fnA；不删除任何数据。耗时与 df 写入 fnprep.json | 同上 | 约 20–30 min |
| 3c fna | fnrun fn8s，STEPS=6（链首先在本地用 fp_fn 按 steps 6 重算指纹并存档）；跑 judge_qwen3_8_next_lora_log.py；另存 lora.txt、torchdist-load.txt、terminal.txt（容器内 df 与显存） | 同上 | 约 30 min |

费用估算：
- **L40S 路径**：段 1–2c 约 85 min × $9.14 ≈ $13；段 3 约 80 min × $36 ≈ $48；合计约 **$61**。
- **H200 回落路径**：段 1–2c 约 90 min × $36 ≈ $54，加段 3 $48，合计约 **$102**。
- **最坏**：两条路径都被链内守卫封顶在 **$150**。守卫按"已花费（各集群 UP 墙钟 × 时价）+ 下一步估算 ≤ CAP"判断，不满足的步不再启动，结果文件中记为"未执行"。
- 硬顶：每次运行 HARD 3000 s（fna 为 3600 s），每次运行都有 watchdog。

## 复核项
- **H200 资源文件** `resources-1x4-h200.json`：1 节点 4 卡；T2R1S1 为 trainer n0:0–1、rollout n0:2、standby n0:3；T2R2S0 为 rollout n0:2、n0:3；T1R3S0 为 trainer n0:0、rollout n0:1–3。launcher 参数加 `--rl-island-use-gpus-per-node 4 --rl-island-network-tier none`；后者避免 H200:8 默认 "best" 网络档去申请 IB 集群。
- **单测** `tests/test_rl_multinode_h200.py`：三个配置的 island spec 都是 4 卡，trainer shape 为 (1,2)/(1,2)/(1,1)，`CUDA_VISIBLE_DEVICES=0,1,2,3`。全部相关测试共 161 passed（multinode_*、launcher_multinode、fn_align）。其中有一次运行出现 4 failed，重跑后全过，疑似并发运行 fp_fn 导致的不稳定，原因未查明。
- **UUID 对账**：s1cost sweep 改为检查 gpu_pool 恰有 4 个 UUID，且都包含在 nvidia-smi 列出的 UUID 中（H200 会列出 8 个物理卡）；GPU 型号按 `--gpu` 断言。
- **attestation 指纹**（d1e1，用 fp_local22 加额外参数重算）：L40S 6 步为 0d17e24b，与历史 M3 一致；H200 alloc4、14 步为 0c779f5b。**风险**：H200 指纹从未在真机上与运行时 rl_driver_start 的值对过。若不一致，e1 的请求会被拒绝，可从 journal 看出来，事后用 rl_driver_start 对照。
- fn8s 指纹（fp_fn，steps 6，seed 17）为 abe7500d，仅作信息，fn8s 不做认证。
- **fnboot 是有意的"预期失败"**：它的作用是拿到挂好 FS 的节点。它的 rc 非 0 不算失败，只看 fna。
- `--rl-heartbeat-interval` 与 `--rl-resource-sample-interval` 只是 learner 参数，launcher 不转发。开了 `--rl-observe-timeline` 后默认值为 30 s / 60 s，因此不需要额外传参。另外 puller 每 10 s 拉一次 nvidia-smi 快照（gpu-*.txt、apps-*.txt）。
- **每次运行保存的数据**：完整 pulled/（tape、journal、run.log、inwatch、gpu 与 apps 快照）、args.txt（argv）、yeto_sha、attestation-m3.json、sweep.json（有效性与每轮时长）、cost.json/tsv（e1）、coldstart.txt（launch 时间戳中的拉镜像、加载、ready 相关行）、perf.txt（tok/s 相关行）、dashboard.html（yeto dashboard export）、sky status -v（SKU 实况）、lp.json；fn 段另存 fnprep.json、lora.txt、torchdist-load.txt、judge-fn.out、terminal.txt。
- **未覆盖**：
  - logprob 对照的是基座模型在不同卡和 TP 形状下的数值差异，不是各配置训练后的策略；
  - FN 阶段 A 测不到 full 模型的加载时间（见 FN-A-PRELAUNCH-REVIEW §1）；
  - `--sglang-enable-deterministic-inference` 与 GDN 能否共存，只能在真机上看。

## e1 单节点缺陷：现象/定因/修复/测试（S11 H100，run s11-h200-20261005m-e1）

- **现象**：1×8 H100、岛分配 GPU 0–3（`--rl-island-use-gpus-per-node 4`，resources-1x4-h200.json）。sweep 四个 run 正常；e1 的 up1（T2R1S1→T2R2S0）wait_safe/drain/init/VERIFYING 均通过，COMMITTED 后 RESUMING 立即 RECOVERY_REQUIRED：`placement bookkeeping failed after commit: rollout GPUs ['n0:2', 'n0:3'] are outside the pool`，rc=4。随后 launcher 判定 job FAILED，即使带 `--keep` 也拆掉了集群；链在 c17 前报 "cluster not UP"，停止。
- **定因**（类别 (c)，与 gpu_pool 对账、资源文件索引都无关）：`ElasticPlacement._resolve_one` 只有在 `gpus_per_node` 不为空时才把配置里写的 `n<k>:<g>` 槽位解析成 `bundle<b>`。单节点不传 `--rl-island-gpus-per-node`，`MilesPlacementSpec.topology` 为 None，entry.py 传进来的 `gpus_per_node=None`，于是 `n0:2`/`n0:3` 按原字符串保留，与 pool（`bundle0..3`）比对时被判为 pool 外。M3（2×2）有拓扑，所以能通过。CPU 回放的 sweep 不做 reconfigure，所以碰不到这条路径。
- **修复**（8443a8fd）：没有拓扑时，岛只有一个节点，`n0:<g>` 直接解析为逻辑 bundle g；`n<k≥1>:<g>` 照旧原样保留，由 pool 检查拒绝。多节点路径不变。
- **测试**：`tests/test_rl_elastic_placement_resolve.py::test_single_node_island_resolves_n0_slots_without_topology`，使用真实的 resources-1x4-h200.json 跑 up、down、restore_committed。修复前失败，修复后通过。
- **链**（b64ef992）：`START_AT=e1` 跳过 sweep，由 e1 负责 provision。某个 run 失败、集群被 launcher 拆掉后，下一步先把已拆集群的费用计入，再重新 provision；lp 改在 c17 之后跑。
