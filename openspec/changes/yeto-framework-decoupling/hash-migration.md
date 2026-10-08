# 哈希对照表（旧 → 新）

design D6：去耦合后 `AlgorithmSpec.sha256()` 按中立名与新源码重新计算，不刻意保持旧值。本表为阶段 0 标准样本中每个配置记录"旧哈希 → 新哈希"及 CPU 逐位一致结果，供引用旧 GPU 证据。

- 旧哈希：阶段 0（分支 `s16-decouple-p0`，基于 agentenv/main `62cec689`）用 `tests/decoupling_golden.py` 在 CPU 上算出，原值存于 `tests/golden/decoupling/<配置名>.json`。
- 新哈希、CPU 逐位一致结果：阶段 3（任务 4.4、4.4a）/阶段 4（任务 5.x）填写。
- 旧 GPU 证据：在 `/home/michael/work/s1-runs`、仓库 `openspec/changes/*/evidence`、`infra-drafts` 中按算法哈希全文检索（2026-10-08）。**只有算法哈希能对上**：标准样本的 `ExecutionProfile.contract_hash` 来自 CPU 小配置（批量、卡数与真机不同），检索结果为 0 条，因此本列不表示"同一契约的真机运行"，只表示"同一算法规格跑过 GPU"。

| 配置名 | 说明 | 旧 `AlgorithmSpec.sha256()` | 旧 `ExecutionProfile.contract_hash` | 新哈希 | CPU 逐位一致 | 旧 GPU 证据（算法哈希相同） |
|---|---|---|---|---|---|---|
| grpo_default | GRPO 默认，ports，固定分区，strict-avg | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:27bb768f7462a3ff3b660d9ba9cf00477c16a840f2c8d9c00e259cec01ced533` | 待阶段 3/4 | 待填 | 大量，例如 `s1-runs/s1-mn-20261003-g0/pulled/rl-island-0.jsonl`、`s1-runs/verda-g0-20261004c/.../rl-island-0.jsonl`、`s1-runs/s15-island1b-20261008g/head/sky_logs/2-yeto-head-job/run.log` |
| grpo_tis | GRPO + TIS（tis_clip 2.0，tis_clip_low 0） | `5a8a5a9ff5abb317fc8159ab56c07293dc3bc99e084542880b6d32cc5a02b441` | `sha256:f30ac4010eb5b88fac0261507eb951da18718d2bacd512f2d29a79ad84269355` | 待阶段 3/4 | 待填 | `openspec/changes/rl-algo-mismatch-correction/evidence/2026-09-29-g1/runs/tis/`、`.../2026-09-29-g1c/runs/tis/`、`.../2026-09-29-g3b/run/`；另 `s1-runs/s15-fncodex-full-modal-20261007{a,b,c}`、`s15-fncodex-4layer-modal-20261007{a,b,c}` |
| decoupled | decoupled 外层同步，每次同步 2 个本地轮 | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:634583f6f0d29e91a0cb9882b42d732d2adea65ab9c8d37c3c4f87ec3e3904b5` | 待阶段 3/4 | 待填 | 算法规格同 grpo_default；decoupled 真机运行：`s1-runs/s14-dlr-ports-20261007a/head/sky_logs/2-yeto-head-job/run.log`（契约哈希未核对） |
| drgrpo | Dr.GRPO 常数分母归约（G1 证据 spec 原文件） | `0b000b1c9cc850271ee20120fc2b3b8fbf624545d0f8a73cc21aa3389aaf6e85` | `sha256:94562ea539b9602ae900036c53941673ae4b49962631298cc80cb6f3cdeebb48` | 待阶段 3/4 | 待填 | `openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1/out/drgrpo/`（`island-0/events.jsonl`、`g1_meta.json`） |
| seq_adv_maxrl | MaxRL 优势变换 + 奖励流水线插件 | `e4b213355a0ddfbe0c94a7ac5c1806f16bc3ac0009829c260d80dd9af7b0f4f9` | `sha256:4b7c610f458b454dee649a9b3c030f8339823858fde700c28d051a8206aefc25` | 待阶段 3/4 | 待填 | `openspec/changes/rl-algo-seq-and-adv/evidence/g3/rerun/`（`events/algo2a-g3-l{0,1}-modal.jsonl`、`check.json`） |
| codex_harness | codex harness，Qwen3.5-0.8B，假 bundle 合同 | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:27bb768f7462a3ff3b660d9ba9cf00477c16a840f2c8d9c00e259cec01ced533` | 待阶段 3/4 | 待填 | 无同配置 codex 运行（`s15-fncodex-*` 用的是 TIS 规格 `5a8a5a9f…`，见 grpo_tis 行）；算法规格本身同 grpo_default |
| elastic | 岛内弹性模式（`--rl-elastic`） | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:27bb768f7462a3ff3b660d9ba9cf00477c16a840f2c8d9c00e259cec01ced533` | 待阶段 3/4 | 待填 | 算法规格同 grpo_default；弹性真机运行例如 `s1-runs/s11-h200-20261005n-e1/`、`s1-runs/s15-island1a-20261007c/`（契约哈希未核对） |
| fn_2x8 | Flash-Next 2×8 正式训练形状 | `27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea` | `sha256:18dda066767289bca9563ba12c23e8f20206421b23a492431f0810e14adce84e` | 待阶段 3/4 | 待填 | 算法规格同 grpo_default；FN 真机运行例如 `s1-runs/s14-fnsmoke-modal-20261007a/launch.log`、`s1-runs/s16-fn2x8-modal-20261008a/launch.log`（契约哈希未核对） |

## 未录或有限制的项

- `session_contract_hash`：由运行时 LoRA 张量布局算出（`yeto.protocol.layout_fingerprint`），离线不可得，标准样本中为 `null`。
- codex_harness 的推理/工具调用解析器名由 Miles 函数决定；本机不得 import Miles，标准样本中这两个值是占位符 `<MILES-RESOLVED-...>`，其余参数为 yeto 自己生成。
- `ExecutionProfile` 所需的 5 个 Miles 参数取自翻译后的命令行，未经 Miles 解析器。
- strict（schema 3）/decoupled（schema 4）进度文件：比对解码后的内容（去掉挂钟时间字段），不比对文件字节——文件里有每轮耗时等挂钟字段。
- CISPO、critic 两类典型配置本阶段未录（用户指定清单未含），需要时补到标准样本并在本表加行。
