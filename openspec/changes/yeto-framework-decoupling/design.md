## Context

- 动机见 proposal.md。事实依据：`infra-drafts/YETO-DECOUPLING-AUDIT-S16.md`（审计，编号 E/A/L/P/R/M/C/H/B/V）、`infra-drafts/YETO-DESIGN-PHILOSOPHY-S16.md`（设计稿）、`infra-drafts/RL-ALGO-LOCATION-S16.md`（算法位置调研）。
- 引擎端口已基本中立：`engine/ports.py` 五个角色、`RolloutBatchHandle` 不透明（`ports.py:80`），驱动器不碰词元与张量。漏点集中在少数文件：`trainer_transition.py:42,342`（E1/E2）、`driver.py:108,278`（E4/E3）、`bridges.py:26,519-521`（E6）、`runtime_manifest.py:34-57,146-157`（E7–E9）。
- 配置与算法层不 import Miles，但口径是 Miles 的：`run_config.py:244-352,394-648`（E10–E12）、`algorithm.py:48,103`（E13，默认过滤器路径与插件命名空间，**进哈希**）、算法扩展经 `miles_adapter.algorithm_flags.register_flag` 注册（A1）。
- 主要张量计算在 Miles 进程里（含 fork 补丁约 500 行 + 2786 行 overlay）；yeto 真正参与计算的只有奖励后处理/优势变换、Dr.GRPO 归约器、mismatch 观测插件、vendor 的 MIS（RL-ALGO-LOCATION §0）。
- harness + 奖励过滤约 8300 行，绑 Miles 的约 350–450 行，集中在 `codex/generate.py`、`codex_openenv_generate.py:52,64,130`、`codex/reward.py:159-162,222-224,268`、`tbench_reward.py:32,101`（审计 §3）。
- 云/硬件/计费：`launcher.py` 中 `if spec.cloud ==` 10+ 处（C1）；`modal_runner.py:1-25` 已是"提交/查状态/重启/取消+日志"样板（C2）；`shape/providers.py:33 CloudSignals` 已是每云一类（C3）；`accel.py:22-30` 有 cuda/npu 抽象但 RL 路径未用（H1）；全局 `--spot`（`cli.py:1016-1025`，B1），syncer 写死按需（`launcher.py:882,911`，B2）。

## Goals / Non-Goals

**Goals**
- 每阶段可独立合并，合并前标准样本逐字节一致。
- verl 适配层开工时，只需依赖中立接口，不必改核心、不必复制 Miles 代码。
- 边界由自动检查守住，不靠评审记忆。

**Non-Goals**
- `engine/` 改名为 `core/`（本 change 以"`engine/` 下除适配层外即核心"的约定加边界检查达到同样效果）。
- 改 Miles fork 的损失分派；迁移 critic、GAE 变体、归约器、需拼回整条序列的算法（GMPO/GSPO/OPSM 在上下文并行下）、熵。
- 调度层自动执行；k3s；跨后端/跨卡型合并本身。
- 修改 `gateway/`、`envs/`、`shape/` 的选型算法、`diffusion/`。

## Decisions

**D1 去耦合单独成 change，作 rl-verl-backend 前置。** 备选：并进 rl-verl-backend。不选，因为去耦合的回归门槛（逐字节不变）与 verl 的验收（新功能、要 GPU）性质不同，混在一起无法独立合并。依赖关系：verl 开工前完成阶段 0–3；阶段 5 在 verl 首次真跑前完成；阶段 4 建议先完成；阶段 6、7 与 verl 并行。

**D2 目录：`yeto/rl/adapters/miles/`、`yeto/rl/adapters/verl/`；`engine/` 不改名。** 搬迁时原 `yeto/rl/engine/miles_adapter/*.py` 留转发模块（`from yeto.rl.adapters.miles.x import *` 并保持同一对象），旧 `yeto/rl/miles.py` 同理。转发模块保留到所有内部 import 与测试改完，并经边界检查确认无新用法后再删（删的时间另议）。Miles 适配层内部分：`legacy/`（← `rl/miles.py`，L1）、`island_entry.py`（← `rl/learner.py` 中 Miles 部分，L2）、`models/`（← `flash_next_provider`、`deepseek_v4_*`、`miles_full_parameter*`、`sglang_deepseek_v4_clone`、`dense_sweep_wire`、`yeto/megatron/`，M1–M4）、`harness_glue/`（← `codex/generate.py`、`codex_openenv_generate.py` 收样本段、`tool_wait_workload.py`，R6/R7/R14）、`pins.py`（← `rl/__init__.py:12-69`，L4）、`overlay.py`（← `rl/miles_overlay.py`，L3）、`rewards_glue.py`（中立类型 ↔ Miles `Sample`）。

**D3 阶段 0 先录标准样本，再动代码。** 样本内容：选 6–8 个典型配置（GRPO 默认、decoupled、带 CISPO、带 critic、带 mismatch 修正、带 codex harness 奖励、多节点放置），存 Miles 命令行、`AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash`、插件源码哈希、strict/decoupled 进度文件、假引擎驱动下的 tape 片段。全部由 CPU 测试生成，存入 `tests/golden/decoupling/`。之后每阶段同一测试比对。备选：只靠现有测试。不选，因为现有测试不覆盖"字节级"的命令行与进度文件。

**D4 静态边界检查。** 一个纯 `ast` 测试（不 import 被检模块、不起 Ray）。规则见 spec `rl-framework-neutral-core`。白名单格式为"文件路径 + 被禁模块"列表并记录上限条数；条数超过上限、或条目已不存在于代码时都失败——保证只减不增且及时删除。初始白名单按审计 E1–E9、A1–A9、R1–R7、R14、C7 等逐条生成。

**D5 中立类型最小化。** 中立轨迹、奖励结果、过滤决定字段取 Miles 现用字段并集（设计稿 §5.2–5.3），确保 Miles 包装可无损来回转换；驱动器不使用中立轨迹（批次句柄仍不透明）。

**D6 接受新哈希 + 旧→新对照表。** `algorithm.py:48` 默认过滤器路径与 `:103` 插件命名空间改中立名、插件搬位置后，`AlgorithmSpec.sha256()` 按新名与新源码重新计算，不刻意保持旧值。为让旧证据仍可引用：阶段 0 标准样本中每个典型配置记录"旧哈希 → 新哈希"并附 CPU 逐位一致对照结果，写入对照表 `hash-migration.md`；旧名配置仍可读入（规范化为新名），但算出的是新哈希。版本边界一次性切换：新旧版本代码的岛哈希不同、不能混跑，旧检查点按旧哈希续训会被拒，写入发布说明。真机证据不单独开卡，由下一次本来要开的卡（如 2×8 早门或 verl 单卡）顺带以新哈希跑。备选：映射旧名保持哈希不变。不选（用户 2026-10-08 裁定），因为兼容代码更复杂，且"哈希不变"并不保证训练端绑对，真正保证靠 D6a。

**D6a 改名后训练端绑定必须逐项核对（用户强调为关键）。** 哈希相同或不同都不等于训练端用的是同一实现。因此每个中立名在每个后端都有"绑定核对"，启动前执行，任一项不一致即启动失败：
1. 命令行逐字对照：映射后传给 Miles 的参数（以后 verl 的配置键）与改名前生成的逐字相同（标准样本 + 运行时比对）。
2. 插件身份对照：映射后加载的插件模块路径、函数限定名、源码哈希与改名前相同。
3. CPU 端到端调用：用固定输入经映射后的加载路径实际调用一次插件，与改名前路径调用结果逐位相同，证明生效的是同一实现。
核对结果写入运行时清单；verl 适配层落地时为其映射表提供同样的核对。

**D7 后端身份并列哈希。** 新增 `BackendIdentity{engine, engine_commit, device_family, param_map_sha256}` 的独立哈希，不并入旧哈希；Miles 下取值固定。岛握手时比较，不同即拒绝。与 rl-verl-backend D6 的"后端身份进训练契约哈希"对齐：rl-verl-backend 的契约哈希输入（`local_learner.py`、`sao_streaming_runtime.py:203,642`）改为引用此处的身份哈希，而不是各自拼。

实施记录（阶段 5，2026-10-08 夜）：岛握手用的是 syncer HELLO 的会话契约——实现为 `sha256("yeto-rl-session-contract-v2\0" + 布局指纹 + 身份哈希)`，syncer 侧不用改（它只比较字节相等）。代价：与阶段 5 之前的代码、以及之前写下的 syncer 检查点不兼容（版本边界，同 D6）。Miles 的参数名映射是恒等映射，仍单独哈希，以后改映射即改身份。

**D8 算法分层（按 RL-ALGO-LOCATION §4）。**
- 第 1 步（阶段 3 内，零 GPU）：`reward_pipeline.grpo_default`、`seq_adv` 的 MaxRL/MAPO/GDPO、超长惩罚、超长过滤拆成纯函数；Miles 插件改薄包装；等价测试保持 `torch.equal`。
- 第 2 步（阶段 3 内）：中立逐词元损失接口与参考实现（以 `tests/rl_loss_variant_reference.py` 为蓝本），只做 CPU 对照 Miles fork 的 `compute_policy_loss/cispo/sapo/gmpo` 与 TIS/IcePop/MIS。
- 推迟：Miles fork 改调 yeto 函数（改 pin、需小规模 GPU 对照），到 verl 第一阶段打通后另议。
- 不动：critic、GAE 变体、VAPO 正例损失、价值分类损失、归约器（含 Dr.GRPO）、上下文并行下需拼回整条序列的算法、熵。

**D9 硬件层与云层只做"查表 + 接口"，NVIDIA 与现有云输出不变。** 硬件层以 `accel.py` 为起点建 `yeto/hw/device.py`、`catalog.py`（合并 `launcher.py:224 GPU_MEM_GB`、`gpu_spec.py:26-35`、`modal_runner.py:62-87`）、`topology.py`（`multinode.py` 中立部分）。云层以 `modal_runner.py` 为样板抽 `CloudProvider`，`sky.Task` 继续作为 SkyPilot 系云的内部描述，中立层用 yeto 自己的岛描述。验收靠"干跑生成配置与改动前一致"。

**D9a 跨卡型/跨厂商合并：理想状态都能合并，分三期实现，接口现在留好。**
- 第一期：同卡型同厂商（现状）。前提：后端身份一致。验收：现有多岛测试。
- 第二期：同厂商不同卡型（如 H100 与 H200）。前提：①按卡型对标定数值容差（同一批次在两卡型上的 logprob/梯度差），写入阈值表；②契约字段加入卡型（设备族之外的卡型字段，作为"兼容组"而非"必须相同"）；③syncer 只交换中立格式增量（扁平 f32/bf16，按规范参数名）。验收：小模型两卡型岛合并的 CPU/GPU 对照在容差内（GPU 部分另行申请）。
- 第三期：跨厂商（NVIDIA 与昇腾，同后端 verl）。前提：第二期全部前提 + 参数名映射表哈希一致 + 跨厂商容差标定 + 两侧通信不直连、只经 syncer 中立增量。验收：同上，另加跨厂商容差表。
- 本 change 只做：契约里的"兼容组"字段与比较函数（第一版兼容组 = 卡型+厂商完全相同，即仍拒绝），以及阈值表与 syncer 增量格式的接口说明；第二、三期另开任务。

**D9a-1 任务 7.7 细化：卡型兼容组（10-09 用户确认）。**
- 现状：岛身份 `BackendIdentity` 只含引擎、引擎 commit、设备族、参数名映射哈希（`yeto/rl/engine/backend_identity.py:44-47`），不含卡型。H100 岛与 H200 岛今天会被当成同一身份接纳。
- 字段：岛身份加 `compat_group`，取值为"厂商-卡型"的小写串，例如 `nvidia-h100`、`nvidia-h200`。取值来自硬件层卡型目录（7.2），不从云名推。卡型取不到时启动前报错，不填默认值。
- 进哈希：`compat_group` 进身份哈希，所以也进 syncer 会话契约（D7 实施记录里的 `sha256("yeto-rl-session-contract-v2\0" + 布局指纹 + 身份哈希)`）。这会改变所有现有身份哈希与 golden 值，属于版本边界。迁移记录写进 `hash-migration.md`：列出旧值、新值、原因、日期。
- 比较规则（第一版）：两岛 `compat_group` 必须完全相同。不同就走现有"身份不符"规则：syncer 只拒绝这个连接，不退出，不影响其他岛（与 elastic 下"被拒只拆该岛、记 island_lost、不重开"一致）。拒绝原因写"兼容组不同：<a> 对 <b>，容差未标定"。
- 阈值表：留"卡型对容差"键的接口，第一版表为空。表里没有这一对就视为未标定，按上条拒绝。
- 第一版兼容组只含"厂商-卡型"。是否再加驱动版本、CUDA 版本：待定，待用户拍板。
- 第二期（放开同厂商不同卡型）：先标定容差，再允许在表里有值的卡型对合并。调研结论（10-09 主 agent 转达，未逐篇核对原文）：
  - 逐位审计层风险高：跨卡型逐位哈希必然不同，须改为带容差比对。可参考 TOPLOC 局部敏感哈希（arXiv 2501.16007）。
  - 外层平均对收敛的影响：估计低到中，但未查明。OpenDiLoCo（arXiv 2407.07852）只测了同卡型。
  - RL 训推不一致层风险中：生成与训练在不同卡型上时，TIS 截断比例可能上升。
  - 放开前的验证实验列在 tasks 7.8，全部不在本 change 上卡，需报批。
- 与 rl-spot-cost-saving 的关系：该 change 换区域/换云重开岛时，候选只取与现有成员 `compat_group` 相同的卡型（第一期限制）。

**D9b 用户自定义奖励环境接口。** 中立奖励接口（输入中立轨迹，输出奖励结果）+ 注册/加载机制（按中立名注册，或以 `模块:函数` 加载，加载时记录源码哈希进插件身份）+ 示例（一个最小自定义奖励及其测试）。Miles 包装自动适配任何已注册的中立奖励；以后 verl 同样。用户代码不需要 import 任何框架。

**D9c math 判分工具只作测试用。** Miles `math_utils` 三个函数可拷进 yeto，但只放测试目录（如 `tests/vendor/miles_math_utils.py`）用于对照，不进 main 运行代码，文件头写明"许可证待核实"。运行时 math 奖励仍是"装了 Miles 才能用"，边界检查对 `math_reward.py:24` 保留白名单。

**D10 可中断性四级与调度建议。** 等级定义见 spec `yeto-resource-dimensions`。第一版调度层只输出建议（与岛间调度 change 的 Non-Goals 一致），岛间调度与云层可与 verl 并行。自有集群接入（rl-local-cluster-deploy）推后，等 verl 完成且新集群到手后再做，本 change 只留 SSH/k3s 云提供方接口，不实现。实际开卡仍按现有 `--spot`/`--on-demand` 行为；把建议接到真实开卡另开任务并经用户确认。spot 回收通知 → 岛间调度的退岛 + 引擎 `save_cut`/`remove_engines` 的接缝只定义接口，不实现自动化。

**D11 推理进程里的"轮次元数据/策略令牌/计数器"核心接口由谁注入（任务 4.11，原 3.4）——方案，待用户拍板，未实现。**

背景：下列代码运行在**推理进程**（Miles 的 rollout Ray actor，codex 还会再起子进程）里，直接 import Miles 适配层的 `rollout_meta_hook`：
- `algos/seq_adv.py` `_current_round_id`/`_report_round`（A2）：读当前策略令牌 `current_policy_token()`、写本轮计数 `record_round_metadata(args, round_id, nonzero_advantages=…)`；
- `teacher_forcing.py` `_ports_policy_token`（A5）：读策略令牌；另有旧版引擎路径 `yeto.rl.miles._policy_token_for_rollout`；
- `harness/codex/codex_openenv_generate.py`、`codex_openenv_subprocess_agent_function.py`（**子进程**）：`expected_policy_version(sample)` 与两个键名常量；`codex_openenv_agent_function.py`：`counter_value`（纯函数，可直接挪核心，不涉及注入）。

拟定的核心接口（新文件 `yeto/rl/engine/rollout_meta.py`，只定义、不含任何框架代码）：`current_policy_token() -> str | None`、`expected_policy_version(sample) -> str | None`、`record_round_metadata(args, round_id, **counters)`、键名常量（`expected_policy_version`、`policy_age_violation` 等，字符串值不变，tape 字段不变）、`counter_value`（纯函数，搬家即可）。Miles 实现仍是 `rollout_meta_hook`（不改其源码，避免改其插件哈希）。

待定问题：推理进程（含 codex 子进程）里，核心怎么拿到"Miles 实现"。候选：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A 环境变量 + 后端名注册表（**建议**） | 适配层开岛时在推理进程环境里设 `YETO_RL_BACKEND=miles`（Miles：经 Ray `runtime_env.env_vars` 或岛启动脚本导出；核实：当前仓库里没有代码显式设置 `YETO_ROLLOUT_META_SINK`，推理进程用默认 Ray 具名 actor，因此落点需实施时确认）；核心 `rollout_meta.port()` 首次调用时按名字查表 `{"miles": "yeto.rl.engine.miles_adapter.rollout_meta_hook"}`，用 `importlib` 加载（字符串，不是静态 import，边界检查通过）。与阶段 4 任务 5.4 的 `--rl-backend` 注册表同一张表 | 子进程自动继承环境变量；与 5.4 合一；verl 只需加一行 | 需要决定"变量未设时"的行为（见下） |
| B 由 Miles 先加载的插件在 import 时自注册 | `rollout_meta_hook` 被 Miles 作为 `--buffer-filter-path` 等加载时调用 `rollout_meta.install(self)` | 不需要环境变量 | 依赖加载顺序；**codex 子进程里不会加载该插件，拿不到实现**——不可行 |
| C 经 `args` 运行时属性传模块路径 | 适配层用已有的运行时属性机制（`register_runtime_attrs`）把实现模块路径塞进 Miles `args` | 显式、进哈希 | `teacher_forcing`、codex 子进程等调用点拿不到 `args`，要改签名；路径进 argv/契约，改动面大 |
| D 显式依赖注入（调用方传入端口对象） | 插件包装层把端口对象作为参数传给中立函数 | 最干净 | 插件签名由 Miles 决定，传不进去，只能在包装层做——等于方案 B/C 的变体 |

方案 A 需要一并拍板的细节：
1. 变量未设时：(a) 默认当作 `miles`（与现状行为完全相同，过渡期安全；建议到 verl 落地后改为必须显式设置）；或 (b) 视为"无实现"，读令牌返回 None、写计数被忽略（与现在"没有 sink 时返回 None"一致，但若启动器漏设变量，GDPO/MaxRL 的非零优势计数会静默丢失）。**建议 (a)**。
2. 哈希：`seq_adv.py`、`codex_openenv_subprocess_agent_function.py` 是插件，改调用点即改插件源码哈希 → seq_adv_maxrl、codex_harness 两个标准样本换新哈希，记入 `hash-migration.md`（与 D6 一致）。
3. 旧版引擎（`--rl-engine legacy`）的 `teacher_forcing` 路径不动，仍走 `yeto.rl.miles`（阶段 4 随 legacy 整体搬）。

在用户拍板前，4.11 只交付本方案，`seq_adv`/`teacher_forcing`/codex 的 A2/A5/A7 白名单条目保持。

**拍板与实施（2026-10-08 夜，主 agent 代拍板第 2 条）**：采用方案 A，变量未设时默认 miles（细节 1 选 (a)）。实现：`yeto/rl/engine/backends.py`（与 5.4 同一张注册表，按角色查模块）、`yeto/rl/engine/rollout_meta.py`；Miles 适配层在 `connect_island_ray` 的 Ray 作业环境里设 `YETO_RL_BACKEND=miles`（codex 子进程经环境继承）。哈希：seq_adv_maxrl 换新算法哈希；codex_harness 只有插件源码哈希变，算法哈希与契约哈希不变（hash-migration.md「阶段 4 记录」）。到 verl 落地后再议是否改为必须显式设置。

## Risks / Trade-offs

- [搬迁改动 import 面大，易漏] → 转发模块 + 边界检查 + 标准样本三重兜底；每阶段只搬一类文件。
- [进度文件抽到核心时字节变化（字段顺序、换行）] → 进度文件属于标准样本，逐字节比对；抽取时原样复用序列化代码。
- [中立名映射正确但训练端加载了另一实现] → D6a 三项绑定核对，不一致即启动失败。
- [插件源码哈希随代码挪位改变，影响已有 GPU 证据] → CPU 逐位一致 + 书面论证后沿用旧证据（**待确认**）；跨岛比较需两岛同时升级，写入发布说明。
- [中立名映射漏掉某个旧名导致哈希变] → 标准样本覆盖所有内置过滤器/插件名；加"所有已知旧名都能映射"的单测。
- [边界检查白名单变成永久豁免] → 白名单记录上限条数，只减不增；条目对应代码消失即失败。
- [硬件/云层抽象过早，昇腾等实测前接口不准] → 只为已有实现（NVIDIA、现有云）落代码，其他厂商只留表行并标"未核实"。
- [verl 等不及去耦合] → 阶段 0–3 量为 S+M+M+L；阶段 6、7 不阻塞 verl。

## Migration Plan

1. 阶段 0 合并：只加测试与样本，零行为变化。
2. 阶段 1–5 依次合并，每次跑：标准样本比对、边界检查、现有 CPU 测试（miles-next-venv 下排除会拉起本机 Ray 的 `mismatch_tape`、`upstream_parse_args` 等）。
3. 阶段 6、7 与 verl 并行，独立合并。
4. 回退：每阶段是单独提交，可整体回退；转发模块保证回退中途旧 import 仍可用。
5. 不需要 GPU、不碰云；如某阶段想补 GPU 证据，另行申请。


## 算法放在哪一层（用户 2026-10-08 确认）

原则：**算法的数学在 yeto，接入点在适配层，深度绑定框架的留在框架。**
- 判定标准：算法是否只依赖"每词元张量 + 每样本标量"（logprob、旧/参考 logprob、优势、掩码、奖励）。是 → yeto 纯函数（附 CPU 参考测试作为标准答案）；还需要模型结构、优化器或并行切分 → 留在框架侧。
- yeto 层：奖励/优势变换（GRPO 归一化、MaxRL、MAPO、GDPO、超长惩罚等）与逐词元损失项（PPO 裁剪、CISPO、SAPO、TIS、IcePop、MIS、KL 估计）。
- 适配层：每框架一次接入点（verl `@register_adv_est`/`@register_policy_loss`；Miles 现有插件路径加载机制），工作量由"算法数 × 框架数"变为"算法数 + 框架数"。
- 框架层：跨卡归约语义、需拼回整条序列的算法（GSPO、GMPO、OPSM、序列级 MIS）、critic 模型与训练循环、熵。框架有原生实现直接用（如 verl 自带 critic）；没有的才打补丁，公式以 yeto 参考函数为准并做 CPU 对照。
- 落地节奏：从现在起新算法默认写在 yeto 并经适配层接入；Miles fork 已有补丁先不动（有 GPU 证据），verl 第一阶段打通后逐个迁到 yeto，每迁一个与原实现 CPU 逐位对照；critic、归约器、需拼回整条序列的算法、熵暂不动。一次训练只设一种 RL 公式，两后端公式对不上即拒绝该组合。

## 待确认

以下按建议默认写入，等用户确认：

- **D11（任务 4.11）推理进程里核心接口由谁注入**：建议方案 A（环境变量 `YETO_RL_BACKEND` + 后端名注册表，变量未设时默认 miles），待用户拍板；拍板前不实现。

- 2026-10-08 用户决定：任务 2.3 拆分，驱动器侧在阶段 1 完成；`reward_pipeline.py` 侧（A4）移到阶段 3（任务 4.10），因为它是插件，改 import 会改插件源码哈希与算法哈希，应与 4.6 纯函数化同批换哈希。

已由用户 2026-10-08 裁定（原待确认）：接受新哈希并维护旧→新对照表、必须做训练端绑定核对（D6、D6a）；插件哈希变化用"CPU 逐位一致 + 对照表"沿用旧 GPU 证据，新哈希的真机证据由下一次本来要开的卡顺带取得；math 判分工具只作测试用拷入并预留用户自定义奖励接口（D9b、D9c）；不同卡型/厂商的岛最终都要能合并、分三期（D9a）；云层与岛间调度与 verl 并行、自有集群推后只留接口（D10）；可中断性四级（I0 不可中断、I1 可中断需切点、I2 可中断无状态、I3 可随时重跑）与训练岛上 spot 的两个条件（回收提前通知 ≥ 切点保存耗时、已配持久存储）按现方案执行；verl 原生损失与 Miles fork 公式不一致时在算法配置里拒绝该组合（一次训练只用一种公式）；critic 等 verl 原生能直接用的在适配层直接调用，verl 没有的优先在 yeto 侧实现；旧路径转发模块保留到阶段 4 后所有内部 import 改完，再单独提交删除。

## Open Questions

- `ssh_harness.py`（4960 行）、`learner.py`、`launcher.py` 只按关键词抽样审过（审计 §4），阶段 4、7 开工时可能发现新漏点——按"加入白名单再逐步删"处理，不改方案。
- NPU 上与 `NCCL_ALGO=Ring` 对应的 HCCL 确定性变量未核实，阶段 6 只留空行。
