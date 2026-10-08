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
