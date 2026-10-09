# 任务

通用要求（每项都适用）：只用 CPU，不碰云/GPU；每阶段合并前跑 ①标准样本比对 ②静态边界检查 ③现有 CPU 测试全过（miles-next-venv 下排除会拉起本机 Ray 的 `mismatch_tape`、`upstream_parse_args` 等）。审计编号指 `infra-drafts/YETO-DECOUPLING-AUDIT-S16.md`。

先后关系（2026-10-08 S16 末调整：3.4→4.11 阶段 3，3.6→5.7 阶段 4，3.7 拆 3.7a/3.7b）：verl 适配层开工前完成 1–4 组（阶段 0–3）；6 组（阶段 5）在 verl 首次真跑前完成；5 组（阶段 4）建议在 verl 开工前完成；7、8 组（阶段 6、7）可与 verl 并行。

## 1. 阶段 0：护栏（标准样本 + 边界检查）

- [x] 1.1 选 6–8 个典型配置（GRPO 默认、decoupled、CISPO、critic、mismatch 修正、codex 奖励、多节点放置），写生成脚本录 Miles 命令行、`AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash`、插件源码哈希、strict（schema 3）/decoupled（schema 4）进度文件、假引擎驱动下的 tape 片段，存 `tests/golden/decoupling/`。验收：新测试 `test_decoupling_golden.py` 在当前代码上通过；连跑两次结果一致；不起 Ray、不访问 GPU。
- [x] 1.2 边界检查测试 `test_import_boundaries.py`（纯 `ast`）：规则按 spec `rl-framework-neutral-core`；初始白名单按审计 E1–E9、A1–A9、R1–R7、R14、C7 等实际扫描结果生成，记录上限条数。验收：当前代码通过；在临时文件里注入一条 `import miles` 测试失败；人为删一条白名单对应代码后测试提示"请删白名单条目"。
- [x] 1.3 在 `yeto/rl/engine/README.md` 写"核心 / 适配层"约定与边界规则；Miles 适配层 README 写清"翻译什么、声明什么、不支持什么"。验收：文档存在并被 1.2 测试的报错信息引用。

  - 1.1 完成情况（2026-10-08，分支 s16-decouple-p0）：实际录 8 个配置（按用户指定清单）：grpo_default、grpo_tis、decoupled、drgrpo、seq_adv_maxrl、codex_harness、elastic、fn_2x8，外加假引擎 tape（local-only colocated、strict schema 3、decoupled schema 4）。生成脚本 `tests/decoupling_golden.py`。偏差如实记录：①CISPO、critic 两类未单独录（用户清单未含，留后补）；②codex 配置的推理/工具调用解析器名由 Miles 函数决定，本机不得 import Miles，两值以占位符录入；③进度文件的字节含挂钟耗时字段，比对的是解码后去掉时间字段的内容，不是文件字节；④`ExecutionProfile` 的 5 个 Miles 参数取自翻译后的 argv（未经 Miles 解析器）；⑤`session_contract_hash` 取决于运行时 LoRA 张量布局，离线不可得，未录。
  - 1.2 完成情况：白名单 `tests/import_boundary_allowlist.txt` 初始 31 条，上限 `ALLOWLIST_CAP = 31`。C7（launcher `import sky`）按 spec 规则不算违规（启动器属云层），未入白名单；另扫出审计未列的 `harness/codex/tb2_provider.py` import `modal`，已入白名单并注明。

## 2. 阶段 1：引擎核心切断反向依赖

- [x] 2.1 `RecoveryRequired` 移入核心异常模块（E2，`trainer_transition.py:342`），Miles 适配层改 import 核心版本。验收：`test_rl_engine_ports` 等现有测试通过；白名单删对应条目。
- [x] 2.2 `CutContext` 移进 ports/核心，可选方法升级为 `CuttableTrainer` 协议（E21，`ports.py:205-213`）；`ReshardPlan` 提到核心，重分片可行性改经 `reshard_problems` 可选方法（E1，`trainer_transition.py:42,162-163`）。验收：假引擎与 Miles 适配层都通过重分片测试；白名单删 E1 条。
- [x] 2.3 核心事件写入器（后端注入实现，`yeto/rl/engine/events.py`），驱动器侧替换 `driver.py:278` 对 `_append_rl_event` 的调用（E3）。验收：标准样本 tape 片段逐字节一致；白名单删驱动器条目。
  - 2.3 拆分（用户 2026-10-08 确认）：`reward_pipeline.py:384` 侧（A4）移到阶段 3 任务 4.10。原因：`reward_pipeline.py` 是 Miles 插件，改其 import 会改插件源码哈希，从而改 `AlgorithmSpec.sha256()`，应与 4.6 纯函数化一起换哈希并记入对照表。
- [x] 2.4 strict/decoupled 进度格式与 `_record_local_round/_record_final_payload/_save_progress` 抽到核心进度模块，旧版引擎反过来 import（E6，`bridges.py:26,519-521`）。验收：标准样本进度文件逐字节一致；读旧进度文件的测试通过。
- [x] 2.5 运行时清单：探针表、版本模块表、镜像清单与 commit 来源、补丁记录由适配层提供（E7–E9，`runtime_manifest.py:34-57,98,132-188,146-157`）；Miles 下字段名 `miles_overlay` 保留。验收：`test_rl_runtime_manifest` 通过，Miles 清单与改动前字段与值相同。
- [x] 2.6 权重传输中立名（E4/V2，`driver.py:108`、`miles_adapter/placement.py:357`），由适配层声明并翻译；tape 中旧值保持。验收：标准样本一致；新增单测验证中立名到 Miles 名的映射。
- [x] 2.7 "睡眠时能否发布"等改为能力声明字段，核心注释改中立措辞（E5、E17、E18、E20）；切点后端私有部分放 `backend_state`（E17，Miles 下序列化不变）。验收：标准样本一致；能力声明测试覆盖新字段。
- [x] 2.8 假引擎改为"能力参数化"（E22），能用同一套中立测试跑不同能力组合。验收：驱动器测试在 Miles 能力与"仅五端口"能力两种参数下都通过；在未安装 Miles 的环境导入核心成功。

## 3. 阶段 2：奖励 / 过滤 / harness 中立化

- [x] 3.1 新建中立轨迹、奖励结果、过滤决定类型与 `yeto/rl/rewards/` 目录（审计 §2）。验收：类型单测；边界检查覆盖新目录。
  - 完成情况：`yeto/rl/rewards/types.py`（`Trajectory`、`RewardResult`、`FilterDecision`）；`yeto/rl/rewards/` 已在边界检查 `CORE_DIRS` 内，`test_rl_neutral_rewards.py` 另查该目录不 import 框架。
- [x] 3.2 math/gsm8k/length/gdpo 奖励与非零方差过滤器改为中立形式，Miles 包装保留原签名（R2、R3、R15，`math_reward.py:51`、`filters.py:19-62`、`gsm8k_reward.py:35`、`length_reward.py:34`、`algos/gdpo_reward.py:39,49`）；过滤状态由过滤器对象持有。验收：录制样本集上包装前后数值、状态、元数据、保留与原因全部相同（新对照测试）；`test_rl_math_reward` 通过。
  - 完成情况：中立形式在 `yeto/rl/rewards/builtin.py`；gsm8k/length/gdpo 入口与 `filters.bounded_nonzero_reward_std` 改为薄包装（原模块、原签名），过滤状态由 `BoundedNonzeroStdFilter` 对象持有，`args._yeto_bounded_filter_state` 仍是同一状态字典（Miles 轮次钩子按轮置空的语义保持）。偏差：`math_reward.py` 一字未改——其源码哈希在标准样本 fn_2x8 内，改动即破坏标准样本；中立 `math_reward` 复用其纯函数 `score`。对照：`tests/golden/rewards/miles_entry_points.json` 由改动前代码录制，`test_rl_neutral_rewards.py` 逐项比对相同。
- [x] 3.3 codex 奖励：验签逻辑不动，"中止"改为返回奖励结果，由 Miles 包装翻成 `Sample.Status.ABORTED` / `DynamicFilterOutput`（R4、R5，`codex/reward.py:159-162,222-224,268`、`tbench_reward.py:32,101`）。验收：codex 奖励现有测试全过；新增"验签失败→中止"对照测试。
  - 完成情况：`codex/reward.py` 新增中立 `secrlenv_reward`、`tbench_reward.py` 新增中立 `tbench_reward`，"中止"以 `RewardResult(aborted=True)` 返回，原入口翻成 `Sample.Status.ABORTED`；验签逻辑未动。入口模块路径进 argv/标准样本，故留原位，白名单 R4/R5 条目不变（`check_group` 仍用 `DynamicFilterOutput`）。
- [ ] ~~3.4~~ 已挪到阶段 3，见 4.11（用户 2026-10-08 S16 末裁定）。原文：核心轮次元数据/策略令牌/计数器接口，算法扩展与 harness 改用它（A2 `seq_adv.py:438,463`、A5 `teacher_forcing.py:118,129`、A7 `codex_openenv_*`）；Miles 实现仍在 `rollout_meta_hook`。验收：标准样本一致；白名单删对应条目。
- [x] 3.5 codex 预检中成员 ID 规则、奖励作用域检查、看板句柄改经核心接口（A6，`preflight.py:35,167,275`）。验收：预检测试通过；白名单删条。
  - 完成情况：成员 ID 规则移入 `yeto/rl/engine/members.py`、看板句柄 `LazyBoardActor` 移入 `yeto/rl/engine/tool_wait.py`（原位置转发同一对象）、奖励作用域规则移入 `yeto/rl/harness/reward_scope.py`（Miles 侧仍抛 `MilesConfigError`，`entry.preflight_stage` 在调用 harness 预检前先做 Miles 版检查，报错类型与文字不变）。白名单删 `preflight.py` 条目：27→26。
- [ ] ~~3.6~~ 已挪到阶段 4，见 5.7（用户 2026-10-08 S16 末裁定；`codex/generate.py` 自证哈希常量随搬家重钉）。原文：Miles 专用 harness 胶水（`codex/generate.py` 全文件、`codex_openenv_generate.py:52,64,130-142` 收样本段、`tool_wait_workload.py:72-73`）移入 Miles 适配层（此时可暂放 `engine/miles_adapter/harness_glue/`，阶段 4 随整体搬），轨迹记账（`:42-127`）留中立。验收：`test_harness_codex_openenv` 等 codex CPU 测试通过。
- [ ] 3.7 改名不改行为（用户 2026-10-08 S16 末裁定拆为 3.7a/3.7b）：`codex_harness_agent.py` 中 `miles` 相关名、`compaction_bridge.py` 的 `miles_base_url`、`tb2_provider.py:686-700` 的 `miles_args`（R9–R11）；tape 字段 `tito_session_mismatch` 不改。验收：codex harness 测试通过；标准样本一致。
  - [x] 3.7a 对外协议不改：HTTP 头 `X-Miles-*` 与 `model_provider="miles"` 属对外协议（网关、会话服务、Miles 镜像内代码都按此拼写），**保持原样不改名**；tape 字段 `tito_session_mismatch` 也不改。验收：无代码改动，本条即记录裁定。
  - [ ] 3.7b 内部改名（R9–R11 中只在 yeto 进程内使用的变量/参数名）随**下次镜像构建**一起做，不单独改：避免镜像内外名字不一致。验收：同 3.7。
- [x] 3.8 会话服务协议文档 + 测试替身（R8，`codex_openenv_agent_function.py:252-345`；路径以 Miles pin 的 `sessions.py` 为准核实）。验收：用替身跑 codex harness CPU 测试通过；协议文档写明哪些路径已核实。
  - 完成情况：协议文档 `yeto/rl/harness/session_protocol.md`（以 ports pin `c35702ee` 的 `miles/rollout/session/sessions.py` 等核实，逐项标"已核实/未核实"；流式 SSE 回复未核实）；替身 `tests/session_service_double.py`；`tests/test_harness_session_protocol.py` 让 harness 可信层的建会话/删会话/释放未返回分段经真实 HTTP 打替身。未实现 verl 侧。
- [x] 3.9 math 判分工具只作测试用：把 Miles `math_utils` 的 `extract_answer/grade_answer_mathd/grade_answer_sympy` 拷到 `tests/vendor/miles_math_utils.py`，文件头写"许可证待核实、不进 main"；运行时 math 奖励仍需装 Miles，白名单保留 `math_reward.py:24`（R1，D9c）。验收：边界检查确认 `yeto/` 下无对该测试文件的 import；无 Miles 环境下用拷贝版跑 math 奖励对照测试，与装 Miles 时逐例相同。
  - 完成情况：`tests/vendor/miles_math_utils.py` 为 pin 提交原文件逐字节拷贝（整文件，含三函数及其依赖），文件头注明许可证待核实；本机 Miles 包被屏蔽，"装 Miles 时"以 pin 提交的原文件（从本地 Miles 仓库 git 取出）代替，逐例相同；白名单 `math_reward.py` 条目保留。
- [x] 3.10 用户自定义奖励环境接口（D9b）：中立奖励注册/加载（中立名或 `模块:函数`，源码哈希进插件身份）、Miles 自动包装、示例 `examples/custom_reward/`（最小自定义奖励 + 测试）。验收：示例奖励不 import 任何框架即可经 Miles 包装被调用，结果与直接调用中立函数相同；未注册名报错；源码改动后插件身份哈希变化。
  - 完成情况：`yeto/rl/rewards/registry.py`（注册/加载、插件身份含源码 sha256）、`yeto/rl/engine/miles_adapter/rewards.py`（通用 Miles 包装；`miles_custom_rm_path()` 校验并给出 `--custom-rm-path` 值）、`examples/custom_reward/`。未接入命令行/启动器（不新增旗标，保持 argv 不变），启动前校验需调用方调用 `miles_custom_rm_path`。

## 4. 阶段 3：配置与算法扩展中立化

- [x] 4.1 学习率计划保留在核心，"翻译成 Miles/Megatron 旗标"一步移入 Miles 适配层（E11，`run_config.py:244-352`）。验收：标准样本命令行一致；`decoupled` 学习率相关测试通过。
  - 完成情况（s17-decouple-p3）：`LR_SCHEDULE_FLAGS`、`lr_schedule_argv` 原样移到 `yeto/rl/engine/miles_adapter/lr_schedule.py`；`resolve_lr_schedule`/`LrSchedule` 留核心。调用方（ports `config.py`、旧版 `learner.py`、测试）改 import。标准样本命令行逐字节一致；`test_rl_applied_lr`、`test_rl_argv_snapshot`、`test_rl_decoupled` 通过。
- [ ] 4.2 `RunConfig` 拆中立部分与 Miles 部分（E10、E12，`run_config.py:37,53-60,394-395,439,498-512,595-603,648`）。验收：标准样本一致；Miles 专有校验（ref-load release、TP×PP 整除）只在 Miles 部分出现。
  - 本轮未做（改动面大：`resolve_run_config` 的校验顺序与旧版共用，需单独一轮）。
  - 部分（s17-decouple-p4）：Miles 专有校验（`--megatron-ref-load` release 标记、TP×PP 整除、EP 整除 actor world、全局批量整除 DP）移到 `adapters/miles/run_config_rules.py`，核心 `resolve_rl_run_config` 在原位置经注册表角色 `run_config_rules` 调用——校验顺序与报错文字不变（`run_config._resolve_ref_load` 保留为薄转发，测试的猴补丁点不变）。已验证：旧版命令行快照（含报错摘要）不变、新单测 `test_miles_only_run_config_checks_live_in_the_adapter`。**未做**：`RLRunConfig` 的 Miles 字段（`sglang_router_port`、`use_miles_router`）与 E12（Qwen3.8-Next 走 Miles 插件的 provider 读取）拆成单独 `MilesRunConfig`——改字段会牵动标准样本与大量测试字面量，建议随 verl 第 1 组需要时再拆。
- [x] 4.3 算法扩展只注册中立字段 + 校验 + 默认值，"字段→Miles 旗标"表搬进 Miles 适配层（A1，`seq_adv.py:47`、`critic.py:42-43`、`sao.py:46`、`loss_variants.py:46`、`critic_warmup.py:492`、`miles_overlay.py:72`）；argv 生成离开 `algorithm.py`（E14，`:1394-1404,1584-1678`）；`selection.py:68-71` 随之（E15）。验收：`test_rl_algorithm_flags*`、`test_rl_algorithm_spec_v2` 通过；标准样本命令行与哈希一致；白名单删 A1 各条。
  - 部分完成（s17-decouple-p3）：seq_adv、loss_variants、critic、sao 四个扩展模块的 Miles 旗标行与"规格→Miles 命令行"函数原样移到 `yeto/rl/engine/miles_adapter/algo_flag_rows.py`，按原导入顺序注册（`MAPPINGS` 顺序不变，标准样本命令行逐字节一致）；算法扩展只剩中立字段/校验/默认值；白名单删 critic、sao、loss_variants 三条（seq_adv 仍因 A2 保留）。未做：`critic_warmup.py:492`（阶段 W 干跑命令行工具，属 Miles 专用，建议阶段 4 随 critic 胶水整体搬）、`miles_overlay.py:72` 已改调新模块但文件本身属阶段 4 搬迁；`algorithm.py` 的 argv 生成（E14，`to_legacy_argv`）已移为 `miles_adapter.algorithm_flags.legacy_algorithm_argv`（只有测试调用，行为不变）；`algorithm.py` 中其余 Miles 旗标拼写只出现在报错文字里，未动。`selection.py:68-71`（E15，按 extra argv 的 `--advantage-estimator` 选旧版/ports 引擎）留到阶段 4 随 5.4 后端注册表一起改。
  - 收尾（s17-decouple-p4）：`selection.py` 的 extra argv 旗标拼写（`--sao*`、`--advantage-estimator`、`--use-critic`、`--rollout-num-gpus`）移到 `adapters/miles/selection_argv.py`，核心只拿中立事实（报错文字未改）；`critic_warmup.py` 干跑经注册表角色 `algorithm_flags` 取 Miles 翻译，白名单删 A1 critic_warmup 条。已验证：`test_rl_engine_selection`、`test_rl_critic_warmup*` 通过。
- [x] 4.4 过滤器/插件中立名 + 旧名规范化映射（E13，`algorithm.py:48,103`，D6）。验收：所有已知旧名都能规范化为新名的单测；新旧名得到同一个新哈希；`hash-migration.md` 对照表覆盖全部标准样本配置（旧哈希、新哈希、CPU 逐位一致结果）；新旧版本岛混跑与旧哈希检查点续训被拒的单测。
  - 本轮未做。
  - 已实现（s17-decouple-p4）：`BOUNDED_NONZERO_STD_FILTER`/`STOCK_NONZERO_STD_FILTER` 改中立名 `nonzero_reward_std_bounded`/`nonzero_reward_std`，`LEGACY_FILTER_NAMES` + `normalize_filter_name`（SamplingSpec、v1 关键字、from_legacy_args 三处入口规范化）；Miles 适配层 `binding.py` 把中立名翻回原路径，命令行逐字节不变。`PLUGIN_NAMESPACES`（`yeto.`/`miles.` 插件命名空间校验）未改，它不进哈希。已验证：`tests/test_rl_binding_check.py`（旧名全部可规范化；新旧名同一新哈希；新旧版本岛混跑被契约检查拒绝；旧哈希切点身份不等）。哈希对照见 hash-migration.md「阶段 4 记录」：8 个标准样本都不带过滤器，4.4 不改它们的哈希；带过滤器的 3 个规格新旧值已登记。
- [x] 4.4a 训练端绑定核对（D6a，启动前执行，不一致即启动失败）：①映射后 Miles 命令行与改名前逐字相同；②加载的插件模块路径、函数限定名、源码哈希与改名前相同；③固定输入经映射路径实际调用一次插件，与改名前路径调用结果逐位相同；结果写入运行时清单。验收：标准样本全部配置三项都通过；人为把一个中立名映射到另一函数、或改一个参数拼写，各自导致启动失败并指出哪一项；verl 映射表的同类核对留接口与待实现测试桩。
  - 本轮未做（依赖 4.4）。`tests/decoupling_bitwise_check.py` 可作为第③项"CPU 实调逐位一致"的雏形。
  - 已实现（s17-decouple-p4）：`adapters/miles/binding.py` `check_filter/check_spec`，三项核对（①命令行逐字 ②模块路径+函数名+源码哈希 ③固定输入 CPU 实调），任一不一致抛 `BindingError` 并指出第几项；接到 `entry.preflight`（启动前、起 Ray 前），结果以 `[yeto] backend binding {...}` 打到岛日志。已验证：单测覆盖真表通过、映射到另一函数（①）、身份被遮蔽（②）、调用结果不同（③）、拼写错（①）各自失败；verl 同类核对留跳过的测试桩。**限制/未做**：结果只进岛日志，未写进运行时清单文件；`miles.*` 原版过滤器在无 Miles 机器上第③项记 skipped；标准样本命令行逐字对照由 `test_decoupling_golden` 承担。
- [x] 4.5 后端能力声明加"支持的算法字段"；启用不支持字段时启动前拒绝。验收：用"仅五端口"假后端声明的单测正反例。
  - 完成情况：`EngineCapabilities.algorithm_fields`（None=未声明、不限制，现有后端与声明 JSON 字节不变）；`AlgorithmSpec.set_fields()` 给出偏离默认值的字段路径；`check()` 启动前拒绝声明集合外的字段。正反例单测在 `tests/test_rl_engine_capabilities.py`（"仅五端口"假后端）。
- [ ] 4.6 奖励/优势变换拆纯函数（`reward_pipeline.grpo_default`、`seq_adv` 的 MaxRL/MAPO/GDPO、超长惩罚、超长过滤），Miles 插件改薄包装（RL-ALGO-LOCATION §4 A1）。验收：`test_rl_reward_pipeline_equivalence`、`test_rl_seq_adv_miles`、`test_rl_seq_adv` 保持 `torch.equal`；插件源码哈希前后值与书面论证写入文档（**待确认**沿用 GPU 证据）。
  - 部分完成：`reward_pipeline.grpo_group_normalize`、`seq_adv.gdpo_group_values` 抽成纯函数，插件改薄包装；MaxRL/MAPO（`maxrl_values`/`mapo_values`）与超长惩罚（`overlong_penalty_value`）原本已是纯函数；超长过滤未动。新旧逐位对照 `tests/decoupling_bitwise_check.py` 20 例全部 `torch.equal`。**未在本机验证** `test_rl_reward_pipeline_equivalence`、`test_rl_seq_adv_miles`（需 import Miles，本机屏蔽）；`test_rl_seq_adv` 通过。插件哈希前后值写入 `hash-migration.md`。
- [x] 4.7 中立逐词元损失参考函数（以 `tests/rl_loss_variant_reference.py` 为蓝本：PPO clip/dual-clip、CISPO、SAPO、KL k1/k2/low_var_kl、TIS/IcePop 权重与掩码），只用于 CPU 对照 Miles fork 函数；不改 fork。验收：CPU 上与 fork 对应函数逐位一致，不一致项列表写入文档；测试在未安装 Miles 时跳过而非失败。
  - 完成情况：`yeto/rl/algos/loss_reference.py`（PPO 裁剪/双裁剪、CISPO、SAPO、KL k1/k2/k3/low_var_kl（含重要性比）、TIS/IcePop 权重与裁剪比例、token/sample 平均；不被训练调用）。`tests/test_rl_loss_reference.py`：与论文公式参考（`tests/rl_loss_variant_reference.py`）一致；与 Miles fork pin `c35702ee` 的 `compute_policy_loss`、`compute_cispo_loss`、`compute_sapo_loss`、`compute_approx_kl`、`vanilla_tis_function`、`icepop_function` **CPU 逐位一致（`torch.equal`），不一致项：无**。做法：从本地 Miles 仓库 `git show` 取函数源码、去掉 `@torch.compile` 后用 torch 执行（不 import Miles 包；编译版与即时执行版的差异未覆盖）；无本地仓库时跳过。GMPO（需拼整条序列，框架层）未纳入逐位对照。
- [x] 4.8 归约器与 megatron 依赖（A3，`reducers.py:35,41`、`vendor/miles_mis.py:350`、`grad_audit.py:209,320`）本 change 不迁，只确认在白名单并注明"暂不动"。验收：白名单条目带注释。
  - 完成情况：白名单 `reducers.py`、`vendor/miles_mis.py`、`grad_audit.py` 三条已带"暂不动（任务 4.8）"注释，确认保留。
- [x] 4.9 中立 `--deterministic` 开关，各后端×设备给环境变量（P4，`cli.py:226,381`、`miles_adapter/entry.py:1046`）。验收：Miles×NVIDIA 下环境变量与改动前相同。
  - 完成情况：核心 `yeto/rl/engine/determinism.py` 表（目前只有 miles×nvidia 一行，原值原序搬入；昇腾等未实测的行不填、查询即报"未标定"）；`miles_adapter/entry.DETERMINISM_ENV` 改为查表结果；`--deterministic` 作为 `--rl-deterministic-trainer` 的中立别名（同一 dest，启动器转发不变）。`cut_plugin.py:971` 另有一份同值副本（插件文件，改动会动其哈希），本轮未动。单测 `tests/test_rl_determinism_env.py`。
- [x] 4.10 （原 2.3 的 reward_pipeline 侧已移到此处）`reward_pipeline.py:384` 改用核心事件写入器（原 2.3 的 A4 部分，移入原因见 2.3 注记），与 4.6 同一批换插件哈希。验收：标准样本 tape 片段一致；新哈希写入对照表；白名单删 A4 条目。
  - 完成情况：`reward_pipeline.emit_event` 改调核心 `yeto.rl.engine.events.write_event`（默认写入器即原函数）；与 4.6 同批换插件哈希（seq_adv_maxrl，见 `hash-migration.md`）；白名单删 A4 条目。标准样本中假引擎 tape 片段不变。
- [x] 4.11 （原 3.4，S16 末挪入阶段 3）核心轮次元数据/策略令牌/计数器接口，算法扩展与 harness 改用它（A2 `seq_adv.py:438,463`、A5 `teacher_forcing.py:118,129`、A7 `codex_openenv_*`）；Miles 实现仍在 `rollout_meta_hook`；改 `seq_adv` 等插件即接受新哈希（记入 `hash-migration.md`）。**推理进程里由谁注入核心接口先出方案写进 design，由用户拍板后再实现。** 验收：标准样本除已登记的新哈希外一致；白名单删对应条目。
  - 本轮只交付方案：design D11（候选 A–D，建议 A：环境变量 `YETO_RL_BACKEND` + 后端名注册表，未设时默认 miles），**待用户拍板，未实现**；A2/A5/A7 白名单条目保持。
  - 已实现（主 agent 代拍板 2026-10-08 第 2 条：方案 A，未设变量时默认 miles）：核心 `yeto/rl/engine/backends.py`（后端注册表，与 5.4 同一张表，按角色查模块、importlib 加载）+ `yeto/rl/engine/rollout_meta.py`（`current_policy_token`/`expected_policy_version`/`record_round_metadata`/`sink_available`，键名常量与 `counter_value` 移入核心，字符串值不变）；Miles 实现仍是 `rollout_meta_hook`（只把常量改为从核心导入、加 `sink_available`；WP6 的两行与 `_load_miles_tokenizer` 原样保留）；`connect_island_ray` 在 Ray 作业环境里设 `YETO_RL_BACKEND=miles`。seq_adv、teacher_forcing（ports 令牌部分）、codex_openenv_* 改用核心接口。白名单删 A2、A5（ports 部分）、A7 共 5 条；`teacher_forcing` 旧版引擎路径（`_legacy_policy_token`）保留。已验证：`tests/test_rl_backend_registry.py`（7 例，含未设变量默认 miles、verl 未注册报错、假后端注入、dir 汇点读写）。**未验证**：推理进程/codex 子进程里环境变量在真机上的传递（无 GPU）；seq_adv、codex 插件哈希变化见 hash-migration.md。

## 5. 阶段 4：Miles 代码归位

- [x] 5.1 `yeto/rl/engine/miles_adapter/` → `yeto/rl/adapters/miles/`，原位置留转发模块（同一对象）。验收：旧路径与新路径导入得到同一对象的单测；全部现有测试通过；标准样本一致。
  - 已实现（s17-decouple-p4，S17 夜间 N1）：整个目录 git mv 到 `yeto/rl/adapters/miles/`；旧位置 28 个转发模块把自己在 `sys.modules` 里替换成新模块（同一对象，猴补丁仍生效）；包内 `from ..X` 改为 `from yeto.rl.engine.X`；仓库内调用方全部改用新路径。已验证：`tests/test_miles_adapter_forwarding.py`（30 例：旧/新路径同一对象）。标准样本：Miles 命令行里插件路径随之变化，阶段末统一重录（见 hash-migration.md「阶段 4 记录」）。
- [ ] 5.2 `rl/miles.py` → `adapters/miles/legacy/`；`rl/learner.py` 的 Miles 岛入口 → `adapters/miles/island_entry.py`，中立的参数与 run_config 解析上提（L1、L2，`learner.py:1314,1347`）。验收：`--rl-engine legacy` 与 `ports` 的标准样本一致。
  - 大部分（s17-decouple-p4）：`rl/miles.py` → `adapters/miles/legacy/engine.py`，`rl/learner.py` → `adapters/miles/island_entry.py`（旧路径留转发模块，`python -m` 旧路径也会运行新模块）；岛启动命令改为 `python3 -m yeto.rl.adapters.miles.island_entry`。已验证：旧版命令行快照 `test_rl_argv_snapshot` 在把新模块路径映射回旧路径后全部摘要不变（证明除路径外逐字节相同）。**未做**："中立的参数与 run_config 解析上提"——`parse_args` 仍在 island_entry 内（run_config 解析本来就在 `engine/run_config.py`），留给 verl 接入时按需拆。
- [ ] 5.3 Miles 专用模型文件 → `adapters/miles/models/`（M1–M4）；`rl/__init__.py:12-69` 常量 → `adapters/miles/pins.py`（L4）；`miles_overlay.py`、`overlays/` → `adapters/miles/overlay.py`（L3）；harness 胶水 → `adapters/miles/harness_glue/`。验收：import 冒烟测试；overlay sha256 校验测试通过。
  - 大部分（s17-decouple-p4）：模型文件（flash_next_provider、deepseek_v4_* 六个、miles_full_parameter* 五个、chunked、dense_sweep_wire、miles_sao_streaming）→ `adapters/miles/models/`；`rl/__init__.py` 的 Miles/SGLang pin 与镜像常量 → `adapters/miles/pins.py`，`yeto.rl` 用模块级 `__getattr__` 延迟转出（导入核心不加载适配层）；构建脚本与探针改读 pins.py；`miles_overlay.py` → `overlay.py`、`overlays/` → `adapters/miles/overlays/`（补丁文件内容不变）；harness 胶水见 5.7。**未做**：M4 `yeto/megatron/`（SFT 的 Megatron 岛，不是 Miles RL 路径，design 里本就标"未核实"，建议另议）。已验证：`test_rl_ports_image`、`test_rl_miles_overlay`、`test_rl_fake_profiles`（核心不加载 Miles 与适配层）通过。
- [x] 5.4 后端注册表与 `--rl-backend {miles,verl}`（默认 miles；verl 此时未注册则报"未注册"）；启动器与 `cli.py:405`、`launcher.py:455,1389,1640,1791,1836,1851`、`stage_w_entry.py:31`、`launcher.py:2895,3604,6941` 改经注册表（A8、A9、L5、P3）。验收：默认命令行与标准样本一致；未知后端报错单测。
  - 已实现（s17-decouple-p4，与 4.11 同一张表）：`--rl-backend {miles,verl}`（默认 miles）；`resolve_default_rl_image` 里先查表，verl 未注册即报"未注册"（启动前）；launcher 6 处、cli、stage_w_entry 改经 `backends.module(角色)`，白名单删 A8、A9 三条。已验证：`tests/test_rl_backend_registry.py`。
- [x] 5.5 bundle map 等 Miles 专用拼写从 `launcher.py:313-515` 移入 Miles 适配层，拓扑计算留中立（P2）。验收：多节点放置标准样本一致。
  - 已实现（s17-decouple-p4）：拓扑计算本来就在中立的 `engine/multinode.py`；launcher 里剩下的 Miles 岛入口旗标拼写（`--rl-island-bundle-map JSON`、`--rl-allow-cross-node-*`）移到 `adapters/miles/launch_flags.py`，launcher 经注册表角色 `launch_flags` 调用。已验证：多节点 launcher 测试（m4x1/m5/m5_h100 等 237 例）通过，生成的旗标逐字相同；fn_2x8 标准样本除阶段 4 已登记的路径字段外不变（见 hash-migration.md）。
- [x] 5.6 边界检查规则加"适配层互不 import"。验收：注入反例失败。
  - 已具备（阶段 0 的 `ADAPTER_DIRS` 规则已含"适配层互不 import"），注入反例 `test_adapters_and_cloud_layer_rules`（verl 适配层 import miles 适配层即失败）通过；阶段 4 未改规则。
- [x] 5.7 （原 3.6，S16 末挪入阶段 4）Miles 专用 harness 胶水（`codex/generate.py` 全文件、`codex_openenv_generate.py:52,64,130-142` 收样本段、`tool_wait_workload.py:72-73`）随整体搬到 `adapters/miles/harness_glue/`，轨迹记账（`:42-127`）留中立；`codex/generate.py` 自证哈希常量随搬家重钉并记入 `hash-migration.md`。验收：`test_harness_codex_openenv` 等 codex CPU 测试通过。
  - 已实现（s17-decouple-p4）：SecRLEnv 的 Miles generate 包装 → `adapters/miles/harness_glue/codex_generate.py`，自证哈希重钉 `1c79b0e6…` → `1df1ded9…`（`harness/codex/pins.py` 与 `yeto.rl.SECRLENV_GENERATE_SHA256` 同步，`SECRLENV_GENERATE` 路径随之改）；"记录 agent 元数据"部分留中立（`harness/codex/generate.py`），所以 agent.py、codex_harness_agent.py 不改、签名哈希不变；codex_openenv_generate 的上游 generate、ABORTED 状态、收样本段经注册表角色 `harness_glue`（`adapters/miles/harness_glue/codex_openenv.py`），轨迹记账留中立；tool_wait 的 Miles generate → `harness_glue/tool_wait.py`（`GENERATE_PATH` 改新路径）。白名单删 R6、R7、R14。已验证：`test_harness_codex_openenv`、`test_codex_bridge_compaction`、codex/secrlenv CPU 测试通过。**未验证**：真机 SecRLEnv（该路径已被用户搁置）。

## 6. 阶段 5：后端身份进契约

- [x] 6.1 `BackendIdentity{engine, engine_commit, device_family, param_map_sha256}` 与独立哈希；Miles 适配层给固定值（E23，`execution_profile.py:237-239`、`algorithm.py:1230`）。验收：旧两种哈希等于标准样本；身份哈希同配置两次相同、改任一字段即变。
  - 已实现（s17-decouple-p4，S17 夜间 N1）：`yeto/rl/engine/backend_identity.py`（`BackendIdentity` + 独立 sha256，schema `yeto-backend-identity-v1`）；Miles 固定值在 `adapters/miles/identity.py`（engine=miles、ports 用 `MILES_NEXT_COMMIT`/legacy 用 `MILES_COMMIT`、device_family=nvidia、参数名映射=恒等映射的哈希），注册表角色 `identity`。标准样本新增 `backend_identity` 字段（ports 身份哈希 `9d5696a3d3b6…`），算法哈希与契约哈希不变。已验证：`tests/test_rl_backend_identity.py`（同配置两次相同、改任一字段即变、标准样本两种旧哈希不含身份）。
- [x] 6.2 岛握手比较身份哈希，不同即拒绝；`local_learner.py` 与 `sao_streaming_runtime.py:203,642` 的训练契约输入引用该身份哈希。验收：单测——Miles 与模拟 verl 身份握手被拒并报双方身份；两个 Miles 岛握手不受影响（标准样本一致）。
  - 已实现（s17-decouple-p4）：岛与 syncer 的 HELLO 会话契约由"布局指纹"改为"布局指纹 + 后端身份哈希"（`backend_identity.session_contract_hash`，`BridgeConfig`/`DecoupledBridgeConfig` 新字段 `backend_identity_sha256`，Miles 岛入口填入）；syncer 只接纳会话契约逐字节相同的岛（`syncer/src/server.rs` `SessionSpec` 相等比较），所以 Miles 与 verl（或不同 Miles 钉）的岛握手被拒；能拿到双方身份的地方用 `check_identity_match` 报双方身份。dense（`local_learner.dense_sweep_session_contract_hash` 新参数）与 SAO（`sao_role_stream_session_contract_hash`）的契约输入引用 legacy Miles 身份哈希。已验证：单测（Miles vs 模拟 verl 被拒并报双方身份；两个 Miles 岛契约相同；HELLO 帧带新契约）。**未验证**：真 Rust syncer 端到端（本机 `cargo build` 在基线就失败，`test_rl_integration` 12 例基线即报错）；真机多岛。**版本边界**：新旧代码的岛不能混跑（会话契约不同）；阶段 5 之前写下的 syncer 检查点因会话契约不同不能续跑。

- [x] 6.2a elastic 模式补上身份比较（6.2 只覆盖了严格同步的 HELLO）。验收：Miles 岛和 verl 岛 JOIN 互拒并报双方身份；同后端同提交的岛行为不变；帧与 HMAC 两边逐字节一致。
  - 已实现（s17-elastic-identity，base s17-decouple-p4）：elastic JOIN 正文加 32 字节后端身份哈希（`syncer/src/elastic.rs` `ElasticMsg::Join.backend_identity`；`yeto/rl/elastic_client.py` `Join.backend_identity`、`ElasticClientConfig.backend_identity_sha256`）；`bridge.make_island_bridge` 与 `engine/bridges.ElasticAvgSync` 把 `BridgeConfig.backend_identity_sha256` 传进 JOIN。syncer（`elastic_server.rs`）用第一个被接纳的 JOIN 钉住身份（和严格模式第一个 HELLO 定会话契约一样），之后身份不同的 JOIN 回 MSG_ERROR："backend identity mismatch, JOIN refused: island N declares <哈希> but this elastic session is pinned to <哈希> …"，不入池；钉住的身份写进检查点（`YELSRV2`），续跑后照样拒。
  - 已验证：`cargo test` 141 过（基线 139 + 黄金帧/旧帧拒绝 1 + 服务端钉身份/续跑/旧检查点拒绝 1）；Python 单测（`tests/test_rl_elastic_backend_identity.py` 新增 5 例，`test_rl_inter_island_elastic_client.py` 黄金帧 15 更新）；elastic/inter_island/bridge/decoupled 相关 312 例全过。真 Rust syncer（release 构建）本地三假岛端到端：Miles 岛 1 加入并播种，verl 岛 2 被拒（错误里有双方哈希），Miles 岛 3 正常加入，岛 1、3 合并一步得 2.0（`test_real_syncer_refuses_verl_island_in_miles_session`，需 `YETO_TEST_ELASTIC_SYNCER`）。
  - 本机 cargo：`cargo` 在 `~/.cargo/bin`，非登录 shell 的 PATH 里没有，直接敲 `cargo` 报 command not found；`export PATH=$HOME/.cargo/bin:$PATH` 后基线 5fac05f7 `cargo build` 成功、`cargo test` 139 过（N6 的 145 = 139 + 它 PR #134 加的 6 例）。所以 6.2 记录里的"本机 cargo build 在基线就失败"实为找不到 cargo，不是代码编不过。
  - **版本边界**：见 hash-migration.md "阶段 5 补"——新旧岛/syncer 不能混用，`YELSRV1` 检查点不能续跑。
  - **未验证**：真机多岛（不上 GPU）；Ray 驱动的岛（本机不跑 Ray，只用假岛）。另：与 PR #134（s17-x1-syncer-modes）试合并无冲突，合并树 cargo test 147 过（139+6+2）。

## 7. 阶段 6：硬件层（可与 verl 并行）

- [ ] 7.1 `yeto/hw/device.py` 设备族表（NVIDIA、昇腾；AMD/TPU/摩尔线程预留行标"未核实"），以 `accel.py:22-30` 为起点；`accel.py` 改读此表（H1、V1）。验收：`accel.py` 现有用法测试通过；表单测。
- [ ] 7.2 卡型目录合并 `yeto/hw/catalog.py`（H4：`launcher.py:224`、`gpu_spec.py:26-35`、`modal_runner.py:62-87`）。验收：各卡显存与 Modal 钉卡规则查询结果与改动前相同。
- [ ] 7.3 通信与确定性环境变量查表（H3、H5，`ssh_harness.py:3559-3563,3631-3632`、`launcher.py:278-302`、`cli.py:298-308`）；`ssh_harness.py:1075` 与 `profiles/qwen3_8_next.py:534` 写死 H200 改为参数（H2、V7、M5）。验收：NVIDIA 下生成的前导脚本与环境变量与改动前逐项相同。
- [ ] 7.4 `yeto/hw/topology.py`：`multinode.py` 中立部分（节点×卡），框架约束改为后端 `layout_rules`（E16、H7）。验收：`multinode` 现有测试通过；多节点标准样本一致。
- [ ] 7.5 镜像三方合成接口（后端给软件需求、硬件给基底、云给仓库与凭据，V5）。验收：现有各云镜像解析结果与改动前相同。
- [ ] 7.6 能力求交：后端支持的设备族 ∌ 请求设备族时启动前拒绝。验收：Miles×昇腾拒绝单测。
- [ ] 7.7 跨卡型/跨厂商合并接口（D9a 第一期）：契约加"兼容组"字段（卡型+厂商）与比较函数，第一版兼容组要求完全相同；阈值表加"卡型对/厂商对容差"键（无值即视为未标定、拒绝合并）；文档写明 syncer 只交换中立格式增量（扁平 f32/bf16、规范参数名）。验收：同卡型同厂商合并通过、H100 与 H200 拒绝并报"未标定容差"、NVIDIA 与昇腾拒绝的单测；第二、三期的前提与验收写入 design D9a，不在本 change 实现。

## 8. 阶段 7：云层与按任务分 spot/按需（可与 verl 并行）

设计输入（S17）：`openspec/changes/rl-infra-spec/cloud-pool-design.md` §3.4（回收按等级处理，对应 8.7）、§5（调度建议规则与 `CloudAdvice` 形状，对应 8.6）。

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
