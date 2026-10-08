# 任务

通用要求（每项都适用）：只用 CPU，不碰云/GPU；每阶段合并前跑 ①标准样本比对 ②静态边界检查 ③现有 CPU 测试全过（miles-next-venv 下排除会拉起本机 Ray 的 `mismatch_tape`、`upstream_parse_args` 等）。审计编号指 `infra-drafts/YETO-DECOUPLING-AUDIT-S16.md`。

先后关系：verl 适配层开工前完成 1–4 组（阶段 0–3）；6 组（阶段 5）在 verl 首次真跑前完成；5 组（阶段 4）建议在 verl 开工前完成；7、8 组（阶段 6、7）可与 verl 并行。

## 1. 阶段 0：护栏（标准样本 + 边界检查）

- [x] 1.1 选 6–8 个典型配置（GRPO 默认、decoupled、CISPO、critic、mismatch 修正、codex 奖励、多节点放置），写生成脚本录 Miles 命令行、`AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash`、插件源码哈希、strict（schema 3）/decoupled（schema 4）进度文件、假引擎驱动下的 tape 片段，存 `tests/golden/decoupling/`。验收：新测试 `test_decoupling_golden.py` 在当前代码上通过；连跑两次结果一致；不起 Ray、不访问 GPU。
- [x] 1.2 边界检查测试 `test_import_boundaries.py`（纯 `ast`）：规则按 spec `rl-framework-neutral-core`；初始白名单按审计 E1–E9、A1–A9、R1–R7、R14、C7 等实际扫描结果生成，记录上限条数。验收：当前代码通过；在临时文件里注入一条 `import miles` 测试失败；人为删一条白名单对应代码后测试提示"请删白名单条目"。
- [x] 1.3 在 `yeto/rl/engine/README.md` 写"核心 / 适配层"约定与边界规则；Miles 适配层 README 写清"翻译什么、声明什么、不支持什么"。验收：文档存在并被 1.2 测试的报错信息引用。

  - 1.1 完成情况（2026-10-08，分支 s16-decouple-p0）：实际录 8 个配置（按用户指定清单）：grpo_default、grpo_tis、decoupled、drgrpo、seq_adv_maxrl、codex_harness、elastic、fn_2x8，外加假引擎 tape（local-only colocated、strict schema 3、decoupled schema 4）。生成脚本 `tests/decoupling_golden.py`。偏差如实记录：①CISPO、critic 两类未单独录（用户清单未含，留后补）；②codex 配置的推理/工具调用解析器名由 Miles 函数决定，本机不得 import Miles，两值以占位符录入；③进度文件的字节含挂钟耗时字段，比对的是解码后去掉时间字段的内容，不是文件字节；④`ExecutionProfile` 的 5 个 Miles 参数取自翻译后的 argv（未经 Miles 解析器）；⑤`session_contract_hash` 取决于运行时 LoRA 张量布局，离线不可得，未录。
  - 1.2 完成情况：白名单 `tests/import_boundary_allowlist.txt` 初始 31 条，上限 `ALLOWLIST_CAP = 31`。C7（launcher `import sky`）按 spec 规则不算违规（启动器属云层），未入白名单；另扫出审计未列的 `harness/codex/tb2_provider.py` import `modal`，已入白名单并注明。

## 2. 阶段 1：引擎核心切断反向依赖

- [x] 2.1 `RecoveryRequired` 移入核心异常模块（E2，`trainer_transition.py:342`），Miles 适配层改 import 核心版本。验收：`test_rl_engine_ports` 等现有测试通过；白名单删对应条目。
- [x] 2.2 `CutContext` 移进 ports/核心，可选方法升级为 `CuttableTrainer` 协议（E21，`ports.py:205-213`）；`ReshardPlan` 提到核心，重分片可行性改经 `reshard_problems` 可选方法（E1，`trainer_transition.py:42,162-163`）。验收：假引擎与 Miles 适配层都通过重分片测试；白名单删 E1 条。
- [ ] 2.3 核心事件写入器（后端注入实现），替换 `driver.py:278` 与 `reward_pipeline.py:384` 对 `_append_rl_event` 的调用（E3、A4）。验收：标准样本 tape 片段逐字节一致；白名单删两条。
- [x] 2.4 strict/decoupled 进度格式与 `_record_local_round/_record_final_payload/_save_progress` 抽到核心进度模块，旧版引擎反过来 import（E6，`bridges.py:26,519-521`）。验收：标准样本进度文件逐字节一致；读旧进度文件的测试通过。
- [x] 2.5 运行时清单：探针表、版本模块表、镜像清单与 commit 来源、补丁记录由适配层提供（E7–E9，`runtime_manifest.py:34-57,98,132-188,146-157`）；Miles 下字段名 `miles_overlay` 保留。验收：`test_rl_runtime_manifest` 通过，Miles 清单与改动前字段与值相同。
- [x] 2.6 权重传输中立名（E4/V2，`driver.py:108`、`miles_adapter/placement.py:357`），由适配层声明并翻译；tape 中旧值保持。验收：标准样本一致；新增单测验证中立名到 Miles 名的映射。
- [x] 2.7 "睡眠时能否发布"等改为能力声明字段，核心注释改中立措辞（E5、E17、E18、E20）；切点后端私有部分放 `backend_state`（E17，Miles 下序列化不变）。验收：标准样本一致；能力声明测试覆盖新字段。
- [x] 2.8 假引擎改为"能力参数化"（E22），能用同一套中立测试跑不同能力组合。验收：驱动器测试在 Miles 能力与"仅五端口"能力两种参数下都通过；在未安装 Miles 的环境导入核心成功。

## 3. 阶段 2：奖励 / 过滤 / harness 中立化

- [ ] 3.1 新建中立轨迹、奖励结果、过滤决定类型与 `yeto/rl/rewards/` 目录（审计 §2）。验收：类型单测；边界检查覆盖新目录。
- [ ] 3.2 math/gsm8k/length/gdpo 奖励与非零方差过滤器改为中立形式，Miles 包装保留原签名（R2、R3、R15，`math_reward.py:51`、`filters.py:19-62`、`gsm8k_reward.py:35`、`length_reward.py:34`、`algos/gdpo_reward.py:39,49`）；过滤状态由过滤器对象持有。验收：录制样本集上包装前后数值、状态、元数据、保留与原因全部相同（新对照测试）；`test_rl_math_reward` 通过。
- [ ] 3.3 codex 奖励：验签逻辑不动，"中止"改为返回奖励结果，由 Miles 包装翻成 `Sample.Status.ABORTED` / `DynamicFilterOutput`（R4、R5，`codex/reward.py:159-162,222-224,268`、`tbench_reward.py:32,101`）。验收：codex 奖励现有测试全过；新增"验签失败→中止"对照测试。
- [ ] 3.4 核心轮次元数据/策略令牌/计数器接口，算法扩展与 harness 改用它（A2 `seq_adv.py:438,463`、A5 `teacher_forcing.py:118,129`、A7 `codex_openenv_*`）；Miles 实现仍在 `rollout_meta_hook`。验收：标准样本一致；白名单删对应条目。
- [ ] 3.5 codex 预检中成员 ID 规则、奖励作用域检查、看板句柄改经核心接口（A6，`preflight.py:35,167,275`）。验收：预检测试通过；白名单删条。
- [ ] 3.6 Miles 专用 harness 胶水（`codex/generate.py` 全文件、`codex_openenv_generate.py:52,64,130-142` 收样本段、`tool_wait_workload.py:72-73`）移入 Miles 适配层（此时可暂放 `engine/miles_adapter/harness_glue/`，阶段 4 随整体搬），轨迹记账（`:42-127`）留中立。验收：`test_harness_codex_openenv` 等 codex CPU 测试通过。
- [ ] 3.7 改名不改行为：`codex_harness_agent.py` 中 `miles` 相关名、`compaction_bridge.py` 的 `miles_base_url`、`tb2_provider.py:686-700` 的 `miles_args`（R9–R11）；tape 字段 `tito_session_mismatch` 不改。验收：codex harness 测试通过；标准样本一致。
- [ ] 3.8 会话服务协议文档 + 测试替身（R8，`codex_openenv_agent_function.py:252-345`；路径以 Miles pin 的 `sessions.py` 为准核实）。验收：用替身跑 codex harness CPU 测试通过；协议文档写明哪些路径已核实。
- [ ] 3.9 math 判分工具只作测试用：把 Miles `math_utils` 的 `extract_answer/grade_answer_mathd/grade_answer_sympy` 拷到 `tests/vendor/miles_math_utils.py`，文件头写"许可证待核实、不进 main"；运行时 math 奖励仍需装 Miles，白名单保留 `math_reward.py:24`（R1，D9c）。验收：边界检查确认 `yeto/` 下无对该测试文件的 import；无 Miles 环境下用拷贝版跑 math 奖励对照测试，与装 Miles 时逐例相同。
- [ ] 3.10 用户自定义奖励环境接口（D9b）：中立奖励注册/加载（中立名或 `模块:函数`，源码哈希进插件身份）、Miles 自动包装、示例 `examples/custom_reward/`（最小自定义奖励 + 测试）。验收：示例奖励不 import 任何框架即可经 Miles 包装被调用，结果与直接调用中立函数相同；未注册名报错；源码改动后插件身份哈希变化。

## 4. 阶段 3：配置与算法扩展中立化

- [ ] 4.1 学习率计划保留在核心，"翻译成 Miles/Megatron 旗标"一步移入 Miles 适配层（E11，`run_config.py:244-352`）。验收：标准样本命令行一致；`decoupled` 学习率相关测试通过。
- [ ] 4.2 `RunConfig` 拆中立部分与 Miles 部分（E10、E12，`run_config.py:37,53-60,394-395,439,498-512,595-603,648`）。验收：标准样本一致；Miles 专有校验（ref-load release、TP×PP 整除）只在 Miles 部分出现。
- [ ] 4.3 算法扩展只注册中立字段 + 校验 + 默认值，"字段→Miles 旗标"表搬进 Miles 适配层（A1，`seq_adv.py:47`、`critic.py:42-43`、`sao.py:46`、`loss_variants.py:46`、`critic_warmup.py:492`、`miles_overlay.py:72`）；argv 生成离开 `algorithm.py`（E14，`:1394-1404,1584-1678`）；`selection.py:68-71` 随之（E15）。验收：`test_rl_algorithm_flags*`、`test_rl_algorithm_spec_v2` 通过；标准样本命令行与哈希一致；白名单删 A1 各条。
- [ ] 4.4 过滤器/插件中立名 + 旧名规范化映射（E13，`algorithm.py:48,103`，D6）。验收：所有已知旧名都能规范化为新名的单测；新旧名得到同一个新哈希；`hash-migration.md` 对照表覆盖全部标准样本配置（旧哈希、新哈希、CPU 逐位一致结果）；新旧版本岛混跑与旧哈希检查点续训被拒的单测。
- [ ] 4.4a 训练端绑定核对（D6a，启动前执行，不一致即启动失败）：①映射后 Miles 命令行与改名前逐字相同；②加载的插件模块路径、函数限定名、源码哈希与改名前相同；③固定输入经映射路径实际调用一次插件，与改名前路径调用结果逐位相同；结果写入运行时清单。验收：标准样本全部配置三项都通过；人为把一个中立名映射到另一函数、或改一个参数拼写，各自导致启动失败并指出哪一项；verl 映射表的同类核对留接口与待实现测试桩。
- [ ] 4.5 后端能力声明加"支持的算法字段"；启用不支持字段时启动前拒绝。验收：用"仅五端口"假后端声明的单测正反例。
- [ ] 4.6 奖励/优势变换拆纯函数（`reward_pipeline.grpo_default`、`seq_adv` 的 MaxRL/MAPO/GDPO、超长惩罚、超长过滤），Miles 插件改薄包装（RL-ALGO-LOCATION §4 A1）。验收：`test_rl_reward_pipeline_equivalence`、`test_rl_seq_adv_miles`、`test_rl_seq_adv` 保持 `torch.equal`；插件源码哈希前后值与书面论证写入文档（**待确认**沿用 GPU 证据）。
- [ ] 4.7 中立逐词元损失参考函数（以 `tests/rl_loss_variant_reference.py` 为蓝本：PPO clip/dual-clip、CISPO、SAPO、KL k1/k2/low_var_kl、TIS/IcePop 权重与掩码），只用于 CPU 对照 Miles fork 函数；不改 fork。验收：CPU 上与 fork 对应函数逐位一致，不一致项列表写入文档；测试在未安装 Miles 时跳过而非失败。
- [ ] 4.8 归约器与 megatron 依赖（A3，`reducers.py:35,41`、`vendor/miles_mis.py:350`、`grad_audit.py:209,320`）本 change 不迁，只确认在白名单并注明"暂不动"。验收：白名单条目带注释。
- [ ] 4.9 中立 `--deterministic` 开关，各后端×设备给环境变量（P4，`cli.py:226,381`、`miles_adapter/entry.py:1046`）。验收：Miles×NVIDIA 下环境变量与改动前相同。

## 5. 阶段 4：Miles 代码归位

- [ ] 5.1 `yeto/rl/engine/miles_adapter/` → `yeto/rl/adapters/miles/`，原位置留转发模块（同一对象）。验收：旧路径与新路径导入得到同一对象的单测；全部现有测试通过；标准样本一致。
- [ ] 5.2 `rl/miles.py` → `adapters/miles/legacy/`；`rl/learner.py` 的 Miles 岛入口 → `adapters/miles/island_entry.py`，中立的参数与 run_config 解析上提（L1、L2，`learner.py:1314,1347`）。验收：`--rl-engine legacy` 与 `ports` 的标准样本一致。
- [ ] 5.3 Miles 专用模型文件 → `adapters/miles/models/`（M1–M4）；`rl/__init__.py:12-69` 常量 → `adapters/miles/pins.py`（L4）；`miles_overlay.py`、`overlays/` → `adapters/miles/overlay.py`（L3）；harness 胶水 → `adapters/miles/harness_glue/`。验收：import 冒烟测试；overlay sha256 校验测试通过。
- [ ] 5.4 后端注册表与 `--rl-backend {miles,verl}`（默认 miles；verl 此时未注册则报"未注册"）；启动器与 `cli.py:405`、`launcher.py:455,1389,1640,1791,1836,1851`、`stage_w_entry.py:31`、`launcher.py:2895,3604,6941` 改经注册表（A8、A9、L5、P3）。验收：默认命令行与标准样本一致；未知后端报错单测。
- [ ] 5.5 bundle map 等 Miles 专用拼写从 `launcher.py:313-515` 移入 Miles 适配层，拓扑计算留中立（P2）。验收：多节点放置标准样本一致。
- [ ] 5.6 边界检查规则加"适配层互不 import"。验收：注入反例失败。

## 6. 阶段 5：后端身份进契约

- [ ] 6.1 `BackendIdentity{engine, engine_commit, device_family, param_map_sha256}` 与独立哈希；Miles 适配层给固定值（E23，`execution_profile.py:237-239`、`algorithm.py:1230`）。验收：旧两种哈希等于标准样本；身份哈希同配置两次相同、改任一字段即变。
- [ ] 6.2 岛握手比较身份哈希，不同即拒绝；`local_learner.py` 与 `sao_streaming_runtime.py:203,642` 的训练契约输入引用该身份哈希。验收：单测——Miles 与模拟 verl 身份握手被拒并报双方身份；两个 Miles 岛握手不受影响（标准样本一致）。

## 7. 阶段 6：硬件层（可与 verl 并行）

- [ ] 7.1 `yeto/hw/device.py` 设备族表（NVIDIA、昇腾；AMD/TPU/摩尔线程预留行标"未核实"），以 `accel.py:22-30` 为起点；`accel.py` 改读此表（H1、V1）。验收：`accel.py` 现有用法测试通过；表单测。
- [ ] 7.2 卡型目录合并 `yeto/hw/catalog.py`（H4：`launcher.py:224`、`gpu_spec.py:26-35`、`modal_runner.py:62-87`）。验收：各卡显存与 Modal 钉卡规则查询结果与改动前相同。
- [ ] 7.3 通信与确定性环境变量查表（H3、H5，`ssh_harness.py:3559-3563,3631-3632`、`launcher.py:278-302`、`cli.py:298-308`）；`ssh_harness.py:1075` 与 `profiles/qwen3_8_next.py:534` 写死 H200 改为参数（H2、V7、M5）。验收：NVIDIA 下生成的前导脚本与环境变量与改动前逐项相同。
- [ ] 7.4 `yeto/hw/topology.py`：`multinode.py` 中立部分（节点×卡），框架约束改为后端 `layout_rules`（E16、H7）。验收：`multinode` 现有测试通过；多节点标准样本一致。
- [ ] 7.5 镜像三方合成接口（后端给软件需求、硬件给基底、云给仓库与凭据，V5）。验收：现有各云镜像解析结果与改动前相同。
- [ ] 7.6 能力求交：后端支持的设备族 ∌ 请求设备族时启动前拒绝。验收：Miles×昇腾拒绝单测。
- [ ] 7.7 跨卡型/跨厂商合并接口（D9a 第一期）：契约加"兼容组"字段（卡型+厂商）与比较函数，第一版兼容组要求完全相同；阈值表加"卡型对/厂商对容差"键（无值即视为未标定、拒绝合并）；文档写明 syncer 只交换中立格式增量（扁平 f32/bf16、规范参数名）。验收：同卡型同厂商合并通过、H100 与 H200 拒绝并报"未标定容差"、NVIDIA 与昇腾拒绝的单测；第二、三期的前提与验收写入 design D9a，不在本 change 实现。

## 8. 阶段 7：云层与按任务分 spot/按需（可与 verl 并行）

- [ ] 8.1 `yeto/cloud/provider.py` 云提供方接口与能力声明（计费方式、回收提前通知秒数、能否当头节点、网络档位、卡数上限），以 `modal_runner.py:1-25` 为样板；`CloudSignals` 并入（C2、C3）。验收：接口单测；Modal 实现通过现有 modal_runner 测试。
- [ ] 8.2 各云实现，把 `launcher.py` 中 `if spec.cloud ==` 分支（C1：`:326,1200,1242,1254,3392,3718,4291,4586,6432,6531-6556`）、`MULTINODE_IB_CLOUDS`/`NETWORK_TIER_BEST_SHAPES`（C4）、云×卡→镜像表（C5，`:2494-2507`）、模型存储/预烘镜像/头节点限制（C6）移入；直接 `sky.Task` 只在 SkyPilot 实现内（C7）。验收：对现有各云配置干跑（不开卡）生成的 SkyPilot 任务与 Modal 调用参数与改动前一致的对照测试；边界检查"云名分支只在 `yeto/cloud/`"。
- [ ] 8.3 自有集群只留接口：SSH 与 k3s 云提供方仅定义能力声明骨架与"未实现"报错（C8、C9）；实际接入归 rl-local-cluster-deploy，等 verl 完成且新集群到手后再做。验收：选择 ssh/k3s 时报"未实现"的单测；`ssh_harness` 现有路径不受影响。
- [ ] 8.4 任务可中断性四级（用户 2026-10-08 已确认，见 design）；syncer 改为声明 I0（B2，`launcher.py:882,911`）；`--spot` 改为"允许调度层用 spot"的上限，实际开卡行为暂不变（B1，`cli.py:1016-1025,1249-1250`）。验收：开 `--spot` 时 syncer 仍按需的单测；干跑配置与改动前一致。
- [ ] 8.5 检查点持久存储改由云提供方 `durable_store()` 给出，与计费方式解耦（B3，`launcher.py:2415-2416,3873-3880,1517`）；回收通知秒数进能力声明（B4、B6）。验收：干跑对照一致。
- [ ] 8.6 调度建议（只出建议）：读能力声明 + `shape/plan.py:255-272` 价格 + 可中断性等级，输出岛级资源建议；训练岛 spot 两个条件不满足时建议按需并给原因（B5）。验收：单测覆盖 I0–I3 各一例与训练岛 spot 条件正反例。
- [ ] 8.7 spot 回收 → 退岛/`save_cut`/`remove_engines` 的接缝只定义接口与文档，对接岛间调度 change（B7）。验收：接口文档与假实现单测。

## 9. 收尾

- [ ] 9.1 白名单清点：列出剩余条目与"暂不动"理由（预期剩 A3、R1 等）。验收：清单写入 `yeto/rl/engine/README.md`。
- [ ] 9.2 design「待确认」各项逐项由用户回复并回写 design.md「待确认」。验收：每项标"已确认/改为…"。
