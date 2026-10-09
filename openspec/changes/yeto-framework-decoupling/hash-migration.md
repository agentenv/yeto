# 哈希对照表（旧 → 新）

design D6：去耦合后 `AlgorithmSpec.sha256()` 按中立名与新源码重新计算，不刻意保持旧值。本表为阶段 0 标准样本中每个配置记录"旧哈希 → 新哈希"及 CPU 逐位一致结果，供引用旧 GPU 证据。

- 旧哈希：阶段 0（分支 `s16-decouple-p0`，基于 agentenv/main `62cec689`）用 `tests/decoupling_golden.py` 在 CPU 上算出，原值存于 `tests/golden/decoupling/<配置名>.json`。
- 新哈希、CPU 逐位一致结果：阶段 3（任务 4.4、4.4a）/阶段 4（任务 5.x）填写。未列新哈希的配置表示到该阶段为止哈希未变（标准样本逐字节相同）。
- 旧 GPU 证据：在 `/home/michael/work/s1-runs`、仓库 `openspec/changes/*/evidence`、`infra-drafts` 中按算法哈希全文检索（2026-10-08）。**只有算法哈希能对上**：标准样本的 `ExecutionProfile.contract_hash` 来自 CPU 小配置（批量、卡数与真机不同），检索结果为 0 条，因此本列不表示"同一契约的真机运行"，只表示"同一算法规格跑过 GPU"。

| 配置名 | 说明 | 旧 `AlgorithmSpec.sha256()` | 旧 `ExecutionProfile.contract_hash` | 新哈希 | CPU 逐位一致 | 旧 GPU 证据（算法哈希相同） |
|---|---|---|---|---|---|---|
| grpo_default | GRPO 默认，ports，固定分区，strict-avg | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:27bb768f7462a3ff3b660d9ba9cf00477c16a840f2c8d9c00e259cec01ced533` | 待阶段 3/4 | 待填 | 大量，例如 `s1-runs/s1-mn-20261003-g0/pulled/rl-island-0.jsonl`、`s1-runs/verda-g0-20261004c/.../rl-island-0.jsonl`、`s1-runs/s15-island1b-20261008g/head/sky_logs/2-yeto-head-job/run.log` |
| grpo_tis | GRPO + TIS（tis_clip 2.0，tis_clip_low 0） | `5a8a5a9ff5abb317fc8159ab56c07293dc3bc99e084542880b6d32cc5a02b441` | `sha256:f30ac4010eb5b88fac0261507eb951da18718d2bacd512f2d29a79ad84269355` | 待阶段 3/4 | 待填 | `openspec/changes/rl-algo-mismatch-correction/evidence/2026-09-29-g1/runs/tis/`、`.../2026-09-29-g1c/runs/tis/`、`.../2026-09-29-g3b/run/`；另 `s1-runs/s15-fncodex-full-modal-20261007{a,b,c}`、`s15-fncodex-4layer-modal-20261007{a,b,c}` |
| decoupled | decoupled 外层同步，每次同步 2 个本地轮 | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:634583f6f0d29e91a0cb9882b42d732d2adea65ab9c8d37c3c4f87ec3e3904b5` | 待阶段 3/4 | 待填 | 算法规格同 grpo_default；decoupled 真机运行：`s1-runs/s14-dlr-ports-20261007a/head/sky_logs/2-yeto-head-job/run.log`（契约哈希未核对） |
| drgrpo | Dr.GRPO 常数分母归约（G1 证据 spec 原文件） | `0b000b1c9cc850271ee20120fc2b3b8fbf624545d0f8a73cc21aa3389aaf6e85` | `sha256:94562ea539b9602ae900036c53941673ae4b49962631298cc80cb6f3cdeebb48` | 待阶段 3/4 | 待填 | `openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1/out/drgrpo/`（`island-0/events.jsonl`、`g1_meta.json`） |
| seq_adv_maxrl | MaxRL 优势变换 + 奖励流水线插件 | `e4b213355a0ddfbe0c94a7ac5c1806f16bc3ac0009829c260d80dd9af7b0f4f9` | `sha256:4b7c610f458b454dee649a9b3c030f8339823858fde700c28d051a8206aefc25` | 阶段 3（s17-decouple-p3）：算法 `7458121d46104f8e9eb635b14b20daa8fd415147451ff3c2e69c06b09b41ef37`；契约 `sha256:cfc65138416e06d544a70b24cd021cec1072ceab305229e8a677a9abf1f2c66c`；插件源码 `reward_pipeline` `cfbc099a…`→`b7af1fb4…`、`seq_adv` `2c710922…`→`6935e992…` | 一致（`tests/decoupling_bitwise_check.py`，20 例 `torch.equal`，见下文阶段 3 记录） | `openspec/changes/rl-algo-seq-and-adv/evidence/g3/rerun/`（`events/algo2a-g3-l{0,1}-modal.jsonl`、`check.json`） |
| codex_harness | codex harness，Qwen3.5-0.8B，假 bundle 合同 | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:27bb768f7462a3ff3b660d9ba9cf00477c16a840f2c8d9c00e259cec01ced533` | 待阶段 3/4 | 待填 | 无同配置 codex 运行（`s15-fncodex-*` 用的是 TIS 规格 `5a8a5a9f…`，见 grpo_tis 行）；算法规格本身同 grpo_default |
| elastic | 岛内弹性模式（`--rl-elastic`） | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:27bb768f7462a3ff3b660d9ba9cf00477c16a840f2c8d9c00e259cec01ced533` | 待阶段 3/4 | 待填 | 算法规格同 grpo_default；弹性真机运行例如 `s1-runs/s11-h200-20261005n-e1/`、`s1-runs/s15-island1a-20261007c/`（契约哈希未核对） |
| fn_2x8 | Flash-Next 2×8 正式训练形状 | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:18dda066767289bca9563ba12c23e8f20206421b23a492431f0810e14adce84e` | 待阶段 3/4 | 待填 | 算法规格同 grpo_default；FN 真机运行例如 `s1-runs/s14-fnsmoke-modal-20261007a/launch.log`、`s1-runs/s16-fn2x8-modal-20261008a/launch.log`（契约哈希未核对） |

## 未录或有限制的项

- `session_contract_hash`：由运行时 LoRA 张量布局算出（`yeto.protocol.layout_fingerprint`），离线不可得，标准样本中为 `null`。
- codex_harness 的推理/工具调用解析器名由 Miles 函数决定；本机不得 import Miles，标准样本中这两个值是占位符 `<MILES-RESOLVED-...>`，其余参数为 yeto 自己生成。
- `ExecutionProfile` 所需的 5 个 Miles 参数取自翻译后的命令行，未经 Miles 解析器。
- strict（schema 3）/decoupled（schema 4）进度文件：比对解码后的内容（去掉挂钟时间字段），不比对文件字节——文件里有每轮耗时等挂钟字段。
- CISPO、critic 两类典型配置本阶段未录（用户指定清单未含），需要时补到标准样本并在本表加行。

## 阶段 3 记录（分支 s17-decouple-p3，2026-10-08）

哈希变化只有一处：**seq_adv_maxrl**（其余 7 个配置与假引擎 tape 标准样本逐字节不变）。原因是两个插件模块的源码变了，插件源码哈希进 `AlgorithmSpec.sha256()`：

| 插件模块 | 改了什么 | 任务 | 旧源码哈希 | 新源码哈希 |
|---|---|---|---|---|
| `yeto/rl/algos/reward_pipeline.py` | 写事件改经核心写入器 `yeto.rl.engine.events.write_event`（不再 import `yeto.rl.miles`；默认写入器就是原函数，行为不变）；GRPO 组内归一化抽成纯函数 `grpo_group_normalize`，`grpo_default` 改为薄包装 | 4.10、4.6 | `cfbc099ad9a117de6836a0cb48feb0d5f5d059d9265faa5884bd0ef9a489c162` | `b7af1fb4212366fcc8c76118b1dd35ff4f87742d684a6b23bed775e17d23c8cf` |
| `yeto/rl/algos/seq_adv.py` | `--gamma` 的 Miles 旗标行移到 `miles_adapter/algo_flag_rows.py`；GDPO 组内合成抽成纯函数 `gdpo_group_values` | 4.3、4.6 | `2c710922aa44d7e0e6108450dc0fac6128063714385501c528def06c1ea4e656` | `6935e992f7665822ccde0f2559b460625bd3d90c9089be62196260b3c1548162` |

- CPU 逐位一致：`PYTHONPATH=/tmp/s15-noray python tests/decoupling_bitwise_check.py s16-decouple-p2` 从阶段 2 分支取出旧函数体，与新代码在同一批固定输入上比较（`grpo_default` 4 组输入 × 4 种估计器/标准差组合、`gdpo` 白化开/关、`maxrl` 2 组），20 例全部 `torch.equal`，0 处差异（本机 2026-10-08 运行）。
- 未在本机验证：`tests/test_rl_reward_pipeline_equivalence.py`、`tests/test_rl_seq_adv_miles.py`（需 import Miles，本机按规则屏蔽 Miles/Ray，收集阶段即报错，未运行）。
- 同步更新的样本：`tests/golden/decoupling/seq_adv_maxrl.json`、`examples/rl_algorithms/dapo-like.json`（插件源码哈希）、`openspec/changes/rl-algo-seq-and-adv/examples/{maxrl,mapo,gdpo}.json`（由该 change 的 `make_examples.py` 重新生成）。`openspec/changes/rl-algo-seq-and-adv/evidence/g3/` 下的历史证据保留旧哈希不改。
- 旧 GPU 证据：算法规格（除插件源码哈希外）不变，按 D6 以"CPU 逐位一致 + 本表"引用 `rl-algo-seq-and-adv/evidence/g3/rerun/`；新哈希的真机证据未取得，由下一次本来要开的卡顺带取得。
- 预告：任务 4.11（原 3.4）若按 design D11 实施，`seq_adv.py` 还会再改一次（`rollout_meta_hook` 调用改走核心接口），seq_adv_maxrl 与 codex_harness（`codex_openenv_subprocess_agent_function.run` 是插件）的哈希届时再变，并在本表追加。

## 合入 main 后的标准样本更新（S17 合并，2026-10-08）

去耦合叠层（#118→#119→#120→#126）合进 main 后，main 上 `tests/test_decoupling_golden.py` 8 个配置样本不再一致（假引擎 tape 样本不变）。原因不是去耦合改了代码，而是阶段 0 分支（809b2cbb）的起点早于 main 上已合入的两个 PR，录样本时没包含它们的改动：

| 变化字段 | 原因（已在 main 上的提交） | 旧值 | 新值 |
|---|---|---|---|
| `plugins`：`yeto/rl/engine/miles_adapter/rollout_meta_hook.py` 的 3 个入口（全部 8 个配置） | #116 codex/TB2 修复（`2d208ae5`、`60b2b8e1`、`9bd2a2ec`：tape 带 Codex 退出状态、结束原因、判分输出尾部） | `82c1c88c…` | `5b4d802c…` |
| `plugins`：`codex_openenv_subprocess_agent_function.run`（仅 codex_harness） | #116（`ba7f95e7`：TB2 任务提示改发 instruction.md） | `7bd38698…` | `dab8efda…` |
| `ports_runtime_fingerprint`（全部 8 个配置） | #121 方案 A 镜像钉：`MILES_NEXT_COMMIT` 由 `c35702ee…` 改为 `8bc52237…`，该指纹 = Miles 提交 + Miles 命令行 | 各配置旧值 | 各配置新值 |

- `algorithm_sha256`、契约哈希、Miles 命令行均未变（上表 rollout_meta_hook / codex agent 函数是命令行插件，不进 `AlgorithmSpec.sha256()`）。
- 核对：把 `MILES_NEXT_COMMIT` 临时改回 `c35702ee…` 重算，8 个配置的 `ports_runtime_fingerprint` 与旧样本逐一相等，确认指纹变化只来自 Miles 提交钉。
- #124（发布提速）rebase 到 main 之后，与 main 重算出的样本逐字节相同，#124 自身不改任何标准样本。
- 已用 `python tests/decoupling_golden.py --write` 重新生成 8 个配置样本。

## #127 合入（codex 思维链不计入损失开关，S17 合并，2026-10-08）

- 唯一变化：`yeto/rl/engine/miles_adapter/rollout_meta_hook.py` 源码哈希 `5b4d802c…` → `be53a608…`（全部 8 个配置的 `plugins` 里该文件 3 个入口）。原因：`record_trained_groups` 增加调用 `yeto.rl.harness.reasoning_loss_mask.apply_from_args`（默认不开时直接返回，不改样本）。`algorithm_sha256`、契约哈希、`ports_runtime_fingerprint`、Miles 命令行均不变。
- 合并时顺带的边界修正：#127 原版在中立核心 `yeto/rl/harness/reasoning_loss_mask.py` 里 import 了 `miles.utils.processing_utils`，违反 import 边界（白名单只减不增）。改为由 Miles 适配层 `rollout_meta_hook._load_miles_tokenizer` 加载分词器并以 `tokenizer_loader` 传入；中立核心未拿到加载器时报错拒绝（不静默把思维链算进损失）。行为不变。

## 阶段 4 记录（分支 s17-decouple-p4，起点 main 80e944b6，2026-10-08 夜）

标准样本已用 `python tests/decoupling_golden.py --write` 重录；假引擎 tape 样本逐字节不变。

### 8 个配置的变化一览

| 配置 | `algorithm_sha256` | Miles 命令行摘要 | 契约哈希 |
|---|---|---|---|
| grpo_default | 不变 `27df1133c924e7a3…` | `8e29812ab25b71d9…` → `cd9c51c2946a4e45…` | `27bb768f7462a3ff…` → `27bb768f7462a3ff…` |
| grpo_tis | 不变 `5a8a5a9ff5abb317…` | `499a99adb1b26e80…` → `06b407103acaff82…` | `f30ac4010eb5b88f…` → `f30ac4010eb5b88f…` |
| decoupled | 不变 `27df1133c924e7a3…` | `78727298ee39cb9a…` → `1108c08ed4e9ef28…` | `634583f6f0d29e91…` → `634583f6f0d29e91…` |
| drgrpo | 不变 `0b000b1c9cc85027…` | `9ae71f2b9407fb46…` → `7ee222703a50f20c…` | `94562ea539b9602a…` → `94562ea539b9602a…` |
| seq_adv_maxrl | `7458121d46104f8e9eb635b14b20daa8fd415147451ff3c2e69c06b09b41ef37` → `0cbaa8212dad1654e09c8b349e7c323600749810f3da3335dfefb23c1960a815` | `67ed96627b700f65…` → `8acfb1140db90c4d…` | `cfc65138416e06d5…` → `2497445bf264fd1d…` |
| codex_harness | 不变 `27df1133c924e7a3…` | `18fafcaafda2c19c…` → `259d2fdeff47128f…` | `27bb768f7462a3ff…` → `27bb768f7462a3ff…` |
| elastic | 不变 `27df1133c924e7a3…` | `28bb08ef1bf161f9…` → `c88e8996324d0261…` | `27bb768f7462a3ff…` → `27bb768f7462a3ff…` |
| fn_2x8 | 不变 `27df1133c924e7a3…` | `9dc4312965e48edd…` → `8d056171747e96ed…` | `18dda066767289bc…` → `18dda066767289bc…` |

`ports_runtime_fingerprint`（= Miles 提交 + Miles 命令行）随命令行全部变化；`plugins` 字段全部变化（见下）。

### 原因（逐项）

| 变化 | 原因 | 任务 | 旧 | 新 |
|---|---|---|---|---|
| Miles 命令行：3 个插件路径 `yeto.rl.engine.miles_adapter.rollout_meta_hook.{record_trained_groups,extract_rollout_metadata,policy_buffer_filter}` → `yeto.rl.adapters.miles.rollout_meta_hook.*`（全部 8 个配置；命令行其余部分逐字节相同） | 目录搬迁 | 5.1 | — | — |
| `rollout_meta_hook.py` 源码哈希（全部 8 个配置的 `plugins`，文件路径也变为 `yeto/rl/adapters/miles/rollout_meta_hook.py`） | 4 个键名常量与 `counter_value` 改为从核心 `engine/rollout_meta.py` 导入；新增 `sink_available()`。`record_trained_groups` 里 WP6 的两行与 `_load_miles_tokenizer` 原样保留 | 4.11 | `be53a608…` | `af10161a148d3059…` |
| `seq_adv.py` 源码哈希 → **seq_adv_maxrl 的 `algorithm_sha256` 与契约哈希变化** | `_current_round_id`/`_report_round` 改走核心接口 `engine.rollout_meta`（读令牌/写本轮计数），数学部分未动 | 4.11 | `6935e992f7665822ccde0f2559b460625bd3d90c9089be62196260b3c1548162` | `c292fe3356eb3ca14fa93960c2da93303edf8d2755e62fef6b479bd833a885b8` |
| `codex_openenv_subprocess_agent_function.py` 源码哈希（codex_harness 的 `plugins`） | `expected_policy_version` 改走核心接口 | 4.11 | `dab8efda…` | `22df84e9bbb22675…` |

- 预告里说 codex_harness 的哈希会变：实际只有 `plugins` 字段（命令行插件，不进 `AlgorithmSpec.sha256()`）与命令行摘要变，`algorithm_sha256`、契约哈希不变。
- CPU 逐位一致：`PYTHONPATH=/tmp/s15-noray python tests/decoupling_bitwise_check.py agentenv/main`：20 例全部 `torch.equal`，0 处差异（本机 2026-10-08 夜）。
- 旧版引擎命令行快照（`tests/test_rl_argv_snapshot.py`）：把阶段 4 搬家后的新模块路径映射回旧路径后，全部摘要与原快照相同（证明旧版命令行除路径外逐字节不变）。
- 同步更新：`examples/rl_algorithms/dapo-like.json`（seq_adv 源码哈希、过滤器改中立名）、`openspec/changes/rl-algo-seq-and-adv/examples/{maxrl,mapo,gdpo}.json`（`make_examples.py` 重新生成）。

### 4.4 过滤器中立名

| 旧名（Miles 路径） | 新名（中立） |
|---|---|
| `yeto.rl.filters.bounded_nonzero_reward_std` | `nonzero_reward_std_bounded` |
| `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` | `nonzero_reward_std` |

8 个标准样本都不带动态采样过滤器，所以 4.4 本身不改它们的哈希（WP7 交接里"全部 8 个样本换哈希"的估计不成立）。带过滤器的规格换新哈希（旧名、新名读入后得同一个新哈希；旧值取自 agentenv/main 80e944b6 同一构造）：

| 规格 | 旧 `algorithm_sha256` | 新 |
|---|---|---|
| v1 有界过滤器、最多替换 2 次 | `32ba45a6d3e466e6b664f938779959fdae8767236d5177eabf59f62957e5a47a` | `b669e6c4de10a52d95ecc45294bbfeba857dddd0769abc358528729407bae37e` |
| v2 原版过滤器 | `1129f5c0c80f08bf3d69c08e14e1c803c867e6508e1d55fc53934e671a774e2a` | `05d7bc4b978f11caf6489e530081dcc62ab2131b2261445ee10eefebeb012bb8` |
| v2 有界过滤器、最多替换 4 次（`test_rl_engine_algorithm` 的 R0 样本同此） | `f25cf271d45d38253ea1f255095fe5c0a9b478841a6f90270533382b7adb41dc` | `ac623863613bc25aa7506f31b50e799b5a272919ecdcf05885ff72a9e38d9038` |

- 传给 Miles 的 `--dynamic-sampling-filter-path` 仍是旧路径，逐字节不变（适配层 `adapters/miles/binding.py` 翻译）。
- 版本边界：新旧代码的岛对同一带过滤器配置算出不同哈希，契约检查拒绝混跑；旧哈希的切点续训被 `verify_cut` 以"算法身份不同"拒绝（单测 `tests/test_rl_binding_check.py`）。
- 4.4a 绑定核对（命令行逐字、插件身份、CPU 实调）在启动前执行，结果以 `[yeto] backend binding {...}` 打印到岛日志；原版过滤器（`miles.*`）在本机无 Miles，第③项记为 skipped。

### 5.7 SecRLEnv 自证哈希

`codex/generate.py` 的 Miles 包装搬到 `yeto/rl/adapters/miles/harness_glue/codex_generate.py`：`GENERATE_SHA256`（`harness/codex/pins.py`）与 `yeto.rl.SECRLENV_GENERATE_SHA256` 由 `1c79b0e678b8681b5bd6221b5a4bbc6adbe7a1413b688e248cb930eb3e456cca` 改为 `1df1ded9c6c81404a6101e1821aefc3129d207a3150241980306d6494ee4a2be`，`SECRLENV_GENERATE` 路径改为 `yeto.rl.adapters.miles.harness_glue.codex_generate.generate`。`agent.py`、`codex_harness_agent.py` 未改，签名哈希不变。

### 未验证
- 新哈希与新插件路径没有真机证据（本轮不上 GPU），由下一次本来要开的卡顺带取得。
- 需 import Miles 的等价测试（`test_rl_reward_pipeline_equivalence`、`test_rl_seq_adv_miles`、`test_rl_algorithm_flags_upstream`）本机屏蔽未跑。

## 阶段 5 记录（同分支 s17-decouple-p4，2026-10-08 夜）

- `algorithm_sha256`、契约哈希、Miles 命令行：8 个标准样本全部不变（身份哈希是并列的第三个哈希，不并入前两个）。
- 标准样本新增字段 `backend_identity`：Miles ports 身份 `{engine: miles, engine_commit: 8bc52237…, device_family: nvidia, param_map_sha256: 36c37d69…}`，哈希 `9d5696a3d3b6e6d802115ef3deb970b5e4d9206d1751d71f9075a849d2909f8d`。
- 运行时变化（离线样本记不到）：RL 岛发给 syncer 的会话契约从"布局指纹"改为"布局指纹 + 身份哈希"的 sha256；dense 与 SAO 的会话契约输入加入 legacy Miles 身份哈希。新旧版本岛混跑会被 syncer 拒绝；阶段 5 之前的 syncer 检查点不能续跑。未取得真 syncer 与真机证据。

## C6b 评测岛（分支 s17-eval-island，基于 s17-decouple-p4 f7fe923e，2026-10-08 夜）

只变一处：`yeto/rl/adapters/miles/rollout_meta_hook.py` 加了训练批次按难度分桶（rl-eval-difficulty-buckets 4.1），8 个标准样本里该插件的 `source_sha256` 由 `af10161a148d3059148342896778fec5b915c90ed96258ccab1754f1805cda61` 改为 `605e0ee1e23444cd4710350fe64f59808658605ef0096ccad03d4e7a61585fca`。该哈希不进 `AlgorithmSpec.sha256()`，8 个配置的算法哈希、契约哈希不变；`fake_engine_tapes.json` 逐字节不变。与阶段 5 合并时如两边都改了 rollout_meta_hook，以合并后源码重新生成（`python tests/decoupling_golden.py --write`）。

## 阶段 5 补：elastic 模式 JOIN 带后端身份（分支 s17-elastic-identity，2026-10-08 夜，任务 6.2a）

- 算法哈希、契约哈希、8 个标准样本：都不变（这次只改 elastic 帧和 elastic 检查点，不碰哈希输入）。
- 帧格式变了：elastic JOIN（消息 15）的正文在 `capacity` 后面多 32 字节 `backend_identity`（岛的 `BackendIdentity.sha256()` 原始字节，没声明时全 0），HMAC 照旧覆盖"类型字节 + 正文"。黄金帧（密钥 `k1`，身份 `0xab`×32）由 Rust `ElasticMsg::encode` 自己输出：`03000000000000000100000007000000000000000000000000000440` + `ab`×32 + `3d2faaa5418d42914e93471c8f79f008699aee2190bd1d505af97476e9ec21a7`，Rust 单测和 Python 单测都比对这串字节。其余 elastic 消息的黄金帧不变。
- elastic syncer 检查点格式：魔数 `YELSRV1` 改 `YELSRV2`，魔数后面加"有无身份 1 字节 + 身份 32 字节"。
- 版本边界（新旧不能混用）：
  - 旧岛（JOIN 没有身份字段）连新 syncer：JOIN 被拒，报 "JOIN without backend identity (island older than the syncer?)"。
  - 新岛连旧 syncer：旧 syncer 解码时报 "trailing bytes in elastic frame type 15"，同样被拒。
  - 旧检查点（`YELSRV1`）新 syncer 不能 `--resume`，报 "predates the backend identity field"。
  - 同一种后端、同一个提交的岛，新代码之间行为和以前一样（JOIN_ACK、合并、权重都不变）。

## 阶段 5 补：严格模式拒绝不一致的 HELLO 不再致命（分支 s17-strict-reject，2026-10-08 夜，任务 6.2b）

- 算法哈希、契约哈希、8 个标准样本、帧格式、检查点格式：都不变。
- 只变了 syncer 对错误岛的处理和 MSG_ERROR 文本：契约不一致时错误文本改为 `session mismatch (HELLO refused, session keeps running): expected session_contract_hash=… layout_fingerprint=… …, got …`（仍以 `session mismatch` 开头，旧客户端照样识别为被拒）。新旧代码可以混用。

## S17 N16 学习率调度开关（分支 s17-lr-constant，基线 s17-decouple-p4 @5fac05f7）

新增 `--rl-lr-schedule {auto,linear,constant}`，默认 auto。用 `tests/decoupling_golden.py` 的 `record_config` 走完整的 launcher → 岛 → Miles 命令行链路实测（grpo_default 配置）：

| 配置 | `algorithm_sha256` | Miles 命令行摘要 | 契约哈希 | 学习率参数 |
|---|---|---|---|---|
| grpo_default（auto） | `27df1133c924e7a3…` 不变 | `cd9c51c2946a4e45…` 不变 | `27bb768f7462a3ff…` 不变 | `--lr-decay-style linear --lr-decay-iters 3` |
| grpo_default + `--rl-lr-schedule linear` | 同上 | 同上（与 auto 逐字节相同） | 同上 | 同上 |
| grpo_default + `--rl-lr-schedule constant` | `27df1133c924e7a3…` 不变 | `1108c08ed4e9ef28…`（与 decoupled 标准样本的摘要相同：两者命令行只差这一个值） | `27bb768f7462a3ff…` 不变 | `--lr-decay-style constant --lr-decay-iters 3` |

- 8 个标准样本全部不变（auto 不往命令行加任何参数；`tests/test_decoupling_golden.py` 12 过）。
- 变的只有 Miles 命令行摘要和 `ports_runtime_fingerprint`（`sha256:0ad73504…` → `sha256:cbe1cd06…`）。学习率调度不进 `AlgorithmSpec.sha256()`，也不进 `ExecutionProfile.contract_hash`。
- 行为变化（随 auto 规则并入 N5 a73ab1b2）：`--rl-island-scheduling elastic` 的岛由线性改为常数学习率，命令行摘要随之变化；两个哈希不变。标准样本里的 `elastic` 配置是岛内弹性 `--rl-elastic`，不是跨岛 elastic，所以不受影响。
- 已知限制（待主 agent 定）：因为契约哈希不含学习率调度，同一个同步服务下一个岛用 linear、另一个岛用 constant 不会被握手拒绝。launcher 给同一次运行的所有岛下发同一个值，只有手工拼命令或续训时换了参数才会出现；如果要堵死，需要把调度写进契约，这会让所有配置的契约哈希都变，本次没做。

## S17 C9 codex 事件字段（PR #141，分支 s17-codex-events，2026-10-09 合进 main 时迁移）

- 原因：C9 在 `trajectory_diagnostics` 里加了 `end_kind` 和逐回合长度（`turn_completion_tokens`、`turn_context_tokens`、`turn_tool_output_bytes`），原改在旧路径 `yeto/rl/engine/miles_adapter/rollout_meta_hook.py`；阶段 4/5 已把该文件搬到 `yeto/rl/adapters/miles/rollout_meta_hook.py`（旧路径只剩转发），合并时把这 8 行原样移到新文件。
- 变化：8 个标准样本里 rollout_meta_hook 插件的 `source_sha256` 由 `605e0ee1e23444cd4710350fe64f59808658605ef0096ccad03d4e7a61585fca`（C6b 后）改为 `cab11ca4d4082524f113f3aa28963af7394767b9252c431c88088e466b23b760`。只有这一个字段变；`algorithm_sha256`、契约哈希、Miles 命令行、`backend_identity`、`fake_engine_tapes.json` 都不变。已用 `python tests/decoupling_golden.py --write` 重新生成。

## S17 N13 evaluate_time（PR #142，分支 s17-codex-closeout，2026-10-09 合进 main 时迁移）

- 原因：N13 在 `trajectory_diagnostics` 里加了 `evaluate_time`（判分耗时，只观测），原改在旧路径 rollout_meta_hook；合并时把这 3 行移到 `yeto/rl/adapters/miles/rollout_meta_hook.py`。
- 变化：8 个标准样本里 rollout_meta_hook 插件的 `source_sha256` 由 `cab11ca4…`（C9 后）改为 `6192a70410e8f2a24fe76321dcd35036f16f7c679d91a4f4866d5081693e6ed6`，其他字段都不变；已重新生成。
- 同时合并了 `codex_openenv_agent_function.finish_trusted` 里 N13 与 M1 各自加的判分计时：共用一次计时，`evaluate_time`（N13）与 `YETO_TIMING` 的 `verify_s`（M1）取同一个值。
