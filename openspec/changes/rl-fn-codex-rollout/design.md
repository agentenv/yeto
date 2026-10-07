# Design: FN × codex harness

## Context

事实来源（均为 S14 CPU 核对，file:line 见 `infra-drafts/CODEX-STATUS-S14.md`）：

- codex 唯一 GPU 证据：`codex-smoke-20261003-12`（Modal 1×H100，Qwen3.5-0.8B LoRA，`codex-bundle/launch-9.1.sh`）：`--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate --custom-agent-function-path yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run --codex-backend-profile qwen35_08b --codex-reasoning-effort xhigh --use-session-server --tito-model qwen35 --tito-allowed-append-roles tool user --agent-max-seq-len 8192 --seq-len 8192 --rollout-max-response-len 4096 --lora-targets attention --rollout-batch-size 6 --n-samples-per-prompt 4`，env `YETO_CODEX_BUNDLE_DIR`、`YETO_HARNESS_ENVIRONMENT_PROVIDER=…tb2_provider:modal_provider`、`YETO_HARNESS_TB2_TASKS_DIR=/opt/yeto/codex/tb2-tasks`、`YETO_HARNESS_TB2_MODAL_APP=yeto-tbench2`、`TBENCH_REWARD_HMAC_KEY`、`MODAL_TOKEN_ID/SECRET`、`SECRLENV_MAX_TURNS=12`。
- FN 单岛形态：`s1-runs/s14-fnsmoke-modal.sh`——Modal 8×H200（`--modal-gpu-exact`）、Volume `yeto-fn-models`（torch_dist ref-load）与 `yeto-event-tapes`、TP2 PP4 EP2、SGLang `--rollout-num-gpus-per-engine 8`、colocated + `--rl-offload-train`、`--sglang-mem-fraction-static 0.7`（回退 0.55）、`--lora-targets all-linear --rl-lora-expert-rank 8`、seq-len 8192 / response 4096、$46.0/h。
- FN 模板：HF `chat_template.jinja`（de4b8e4d）只认 `enable_thinking`，默认开 thinking；recipe 传的 `thinking_mode` 不被读取（`FN-MODAL-SMOKE-REVIEW.md` §5）。fork pin `c35702e`（`MILES_NEXT_COMMIT`）里 `TITOTokenizerType.QWEN4_EXP="qwen4exp"` 与 `QWEN38_SMALL` 共用 `Qwen38SmallTITOTokenizer`：固定模板 `qwen3.8_small_and_flash_next_fixed.jinja`，`extra_kwargs={"preserve_thinking": True}`，`consistant_kwargs=["add_vision_id","enable_thinking","reasoning_effort"]`，`allowed_append_roles={tool,user,assistant}`（`miles/utils/chat_template_utils/tito_tokenizer.py:455-465`）。
- 现有 `qwen38` profile（`codex_backend.py:29-43`）指向 `Qwen/Qwen3.8-27B`，kwargs `{enable_thinking, preserve_thinking, reasoning_effort: xhigh}`，`rl_model_recipe=generic`；`validate_stock_codex_fields` 在带 `model_identifier` 的 profile 上要求 `lora_targets=="attention"` 且 `expert_full_count==0`（`:160-171`）。
- 9.2 失配：smoke-12 step 1 上游 `rollout/tito_session_mismatch_rate/v1`=0.1667（4/24：`assistant_text` 3、`special_token_count` 1），step 2 = 0；`train/tis_clipfrac` 两步 0.0（`CODEX-92-PRELAUNCH-REVIEW.md` §1）。
- A16：`preflight.resolve_member` 读 `miles_args.yeto_rl_member_id` / `yeto_rl_cell_id` / `YETO_RL_CELL_ID`，yeto 内无产出者（`CODEX-STATUS-S14.md` §5）。

## Goals / Non-Goals

Goals：FN（4 层与全尺寸）能用与 smoke-12 相同的 A 路径跑 Terminal-Bench 2 多轮轨迹，奖励 `tbench_reward`，训练一次有效更新；多轮 TITO 一致性可按原因分类判定；为 FN-TRAIN-PLAN 正式训练提供可复用的 profile、脚本与判据。Non-Goals 见 proposal。

## Decisions

### D1. FN profile：`qwen38_next` / `qwen38_next_4layer`，`tito_model=qwen4exp`

- 两个 profile 都 `"model": "qwen4exp"`、`"tito_model": "qwen4exp"`（与 `qwen35_08b` 同样"独立身份、复用家族级 TITO"的写法），`rl_model_recipe` 取 FN 现有 recipe 名（launcher 当前未传 `--rl-model-recipe`，默认 `generic`；以 `s14-fnsmoke-modal.sh` 实际渲染为准，CPU 任务 0.1 核对），`identity_label` 分别 `Qwen3.8-Flash-Next` / `Qwen3.8-Flash-Next-4layer`，`model_identifier/model_revision` = `HF_REPO_FULL@HF_REVISION_FULL` / `HF_REPO_4LAYER@HF_REVISION_4LAYER`（直接 import `yeto/rl/profiles/qwen3_8_next.py` 常量，单一真相）。
- `chat_template_kwargs = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}`，与 `qwen38` 一致：`enable_thinking` 是 HF 模板唯一读取的键；`preserve_thinking` 与 `reasoning_effort` 由 fork 固定模板消费（`consistant_kwargs` 要求跨轮一致）。**风险点**：SGLang 侧生成用的是 HF 模板还是 fork 固定模板，决定了 `reasoning_effort` 是否真的进入 prompt；若两侧模板不同，首轮前缀就会出现 `special_token_count` 类失配。阶段 0 任务 0.3 用 fork `chat_template_verify.py` 对两份模板同输入渲染比对，结论写进 progress，再决定是否把 kwargs 收窄为 `{"enable_thinking": True}`（备选 D1-alt）。
- `tito_allowed_append_roles = ["tool", "user"]`（与现有 profile 一致，不开放 `assistant`：R-D5a 禁止修补）。
- 新增字段 `lora_targets: "all-linear"`、`lora_expert_rank: 8`；`validate_stock_codex_fields` 改为 `lora_targets != profile.get("lora_targets", "attention")` 比对，并新增 `lora_expert_rank` 入参（launcher 透传 `args.rl_lora_expert_rank`）。现有四个 profile 行为不变（默认 `attention` / 0）。
- 不复用 `qwen38`：它是 27B dense 的身份，改它会破坏已签名的 allowlist 语义。

### D2. thinking 与多轮一致性：`keeps_history_reasoning` 按 TITO 固定模板声明

- 5.1 的声明挂在 codex profile 上（`keeps_history_reasoning: bool`），由 `GatewayConfig.keeps_history_reasoning` 读取。`qwen35`（`clear_thinking: False`）与 `qwen4exp`（`preserve_thinking: True`）声明 `True`；离线测试用 fork `chat_template_verify.py` 渲染"system/user/assistant(think+text)/tool/assistant(think+text)"五段历史，断言第 2 段 assistant 的 think 文本仍在渲染结果中。
- 5.2：`keeps_history_reasoning=False` 时 `ChainRegistry` 已有 `_only_reasoning_dropped` 分支（`chains.py:129`），补 `template_drops_reasoning` 计数与 mask=1 断言的单测；FN 预期不会走到这个分支，走到即判为"声明错误"而非正常断链。
- 9.2 失配 4/24 对 FN 的含义：`assistant_text`（3 条）= session server 用模板重渲染上一轮 assistant 后，文本与模型实际生成 token 不等——候选根因 (a) 模板对 think 段两端空白/换行规范化，(b) `</think>` 后的空行处理，(c) 工具调用 JSON 重序列化；`special_token_count`（1 条）= 重渲染多/少了特殊 token——候选根因 (d) 生成在 `<|im_end|>` 前被 `max_tokens` 截断、重渲染补了结束 token。FN 与 Qwen3.5 共享 Qwen3 token 边界（`Qwen3TITOTokenizer` 子类），(a)(b)(d) 同样适用；(c) 取决于 `tool_call_parser=qwen3_coder` 相同，也适用。所以根因分类是 FN 的前置，不是 Qwen3.5 独有问题。阶段 0 任务 0.4 从 smoke-12 岛内 `sample.metadata` 的 `compute_session_mismatch` list[dict] 取证（若 tape 不含，从 `gpu-b1-runs/codex-smoke-20261003-12/runs/.../events` 与 launcher.log 回读；取不到就标"不可取证，阶段 1 开 `--rl-observe-timeline` 落 tape 后再判"）。
- 判定口径（spec）：失配率阈值 **≤ 5%** 且每条失配都有分类原因；未分类失配 >0 则阶段 2 不启动。阈值由主 agent 定，待用户拍板（Q2）。

### D3. 先 4 层变体，再全尺寸；两阶段共用一个 argv 骨架

- 阶段 1：`qwen38_next_4layer`，模型 `CharyZeng/Qwen3.8-Flash-Next-4layer@d19a6b60`（HF 30.5 GB，需先下到 Volume `yeto-fn-models`，并转 torch_dist——S11/S12 的 4layer 流程），`--tito-model qwen4exp`，其余 codex 参数与 smoke-12 一致，数据 `tbench2_smoke6.jsonl`，`--rollout-batch-size 6 --n-samples-per-prompt 4 --total-steps 1`，加 `--rl-observe-timeline --modal-tape-volume yeto-event-tapes`（harness 计数落 tape，补 9.2 判据 1/2 的取证能力）。
- 卡：首选 Modal `H100!:1`，colocated + `--rl-offload-train`，`--sglang-mem-fraction-static 0.35`。**未验证**：4 层变体此前只在 8×H100/H200 节点跑过（`s1-runs/s11-*fnboot`、`s13-h100-*fnboot`），1 卡上 30.5 GB bf16 ×（SGLang + Megatron ref）+ KV + LoRA 是否放得下是估算；PLAN_ONLY 阶段若 Megatron 侧估算 >70 GB，则改 `H100!:2`（TP2，$8.78/h，费用翻倍仍 <$8 上限）。不为此上 8 卡。
- 阶段 2：复制 `s14-fnsmoke-modal.sh` 的 MODEL/LORA/PAR/COLO/MODALX 五段不动，只替换 DATA（tbench2 子集 + `--rl-allow-local-data`）、REWARD、加 codex 五件套（generate/agent/profile/effort/session-server/tito/append-roles/agent-max-seq-len）与 harness env 透传；`--total-steps 2`；`--rollout-batch-size 6 --n-samples-per-prompt 4`（24 条/轮，与 smoke-12 同口径便于对照）。SGLang TP8 下 session server 的多轮 append 由 agentic_tool_call 走同一引擎，无额外配置。
- 序列长度：TB 多轮 + FN thinking，`--agent-max-seq-len 8192 --seq-len 8192 --rollout-max-response-len 4096` 作为阶段 1/2 的保守值；`max_seq_len` 策略边界会签名 reward 0（上游 7.1），比例在判据里单列。阶段 3 可按实测提到 16384（显存需重估）。

### D4. codex 环境放置：全部在 Modal，岛容器起 Sandbox

- 复用 smoke-12 的形态：Codex CLI 0.145.0 二进制与 TB2 任务目录在 bundle（`scripts/fetch_codex_bundle.py` → `YETO_CODEX_BUNDLE_DIR`），launcher 校验 pins 后以 `add_local_dir` 挂进岛镜像 `/opt/yeto/codex`（`launcher.py:4508-4545`）；Codex 子进程（`codex_openenv_subprocess_agent_function.run`）在**岛容器内**运行，指向进程内 `_ResponsesBridge` → Session Server → SGLang，不需要任何 Codex / OpenAI 账号或出网。
- 任务容器 = `tb2_provider.ModalSandboxBackend`：rollout worker 在岛容器内用 `MODAL_TOKEN_ID/SECRET`（`HARNESS_PASSTHROUGH_ENV` 透传）在 app `yeto-tbench2` 下按任务官方镜像 `Sandbox.create`，`ttl=YETO_HARNESS_TB2_SANDBOX_TTL_S`，一沙箱一 episode（`EpisodeBinding`）。verifier `bash /tests/test.sh` 在沙箱内跑，结果经可信层签 HMAC。
- 密钥：`TBENCH_REWARD_HMAC_KEY` 透传到岛进程，agent 子进程环境由 `preflight` 的 scrub 列表（`:204-206`）剥掉；Modal token 同样不进 Codex 子进程。网络：岛容器需出网到 Modal API（起沙箱）与 ghcr（镜像），沙箱默认 Modal 网络策略（上游 8.2 的"默认拒绝出网"尚未实现，记为已知限制）。
- 8×H200 容器的 CPU（32 核）/内存（1024 GiB）足够同时跑 24 个 Codex 子进程 + relay；但 Sandbox 创建延迟与 `yeto-tbench2` app 的并发沙箱数是阶段 2 时长的主要不确定项，预留 1.5 h。

### D5. A16：INFRA 暴露成员键

- 产出侧（接口请求 IR-5，给 INFRA）：ports `entry.py` 在 `preflight_stage` 之前把岛自身的 `cell_id` 写到 `miles_args.yeto_rl_cell_id`，并导出 `YETO_RL_CELL_ID` 给 rollout worker 子进程；来源 = launcher 已有的岛编号（`learner_id` / `island_id`，`rl_engine_selected` 事件中已有 `island_id`）。单岛 `--rl-single-island-no-sync` 时也写（值 0），harness 侧 `resolve_member` 自然得到 `engine:0`。
- 消费侧不变（`preflight.resolve_member`）。CPU 单测：多岛 fake 下 `close_admission([member])` 只关该成员，其它岛的 `allow_new_session` 仍为真。
- 不在 GPU 验收：阶段 1/2 都是单岛，只断言 tape 里 harness 计数带 `engine:0` 键。

### D6. 与 CompactionRL 共享

- `COMPACTIONRL-CANDIDATES.md` 推荐 A（TB2 / TB2.1）。本 change 的 provider（`tb2_provider`）、bundle（tasks dir）、数据生成（pin 内 `make_tbench2_data.py`）、奖励与签名全部可直接复用；CompactionRL 只多开 `YETO_CODEX_COMPACTIONRL=1`（`compaction_bridge.py`）与 `gae_variant=cross_segment_per_sample`。本 change 的脚本保留 `COMPACTION=0/1` 开关位但默认 0，且 launcher 现有的 `YETO_CODEX_COMPACTION_ENABLED` 拒绝逻辑不变。
- 待用户拍板 Q3：TB2（89 题，R-TB 已锁 commit `2fd12b88`）还是 TB2.1（修了 26 题）。本 change 默认 TB2 子集（已有数据文件），阶段 3 扩量时再换。

### D7. 判据

| 判据 | 阶段 1（4 层） | 阶段 2（全尺寸 2 轮） | 数据来源 |
|---|---|---|---|
| preflight 通过（bundle 签名、profile 身份、provider） | 必须 | 必须 | launcher.log |
| 轨迹数 = batch×n，全部有 HMAC 或 ABORTED 分类 | 24/24 | 2×24 | tape `rl_load_sample` / rollout perf |
| 非零奖励 | 不要求（4 层模型几乎不可能解题；记录"合法否定") | ≥1 条 reward=1 **或** 记录合法否定结论并由主 agent 裁定是否仍算 PASS | `rollout/episode_raw_reward` |
| 失配率 `tito_session_mismatch_rate` | ≤5% 且全部有分类 | ≤5% 且全部有分类 | 上游指标 + HarnessBoard tape |
| 断链 `tito_chain_breaks{template_drops_reasoning}` | 0 | 0 | tape |
| logprob：`train/tis_clipfrac` | 记录 | ≤1%（≥99% 落入 TIS 范围） | launcher.log |
| 训练更新 | grad_norm>0、policy v1 | 两轮 grad_norm>0、publication payload hash 两轮不同（沿用 `s14-fnsmoke-judge.py`） | tape / judge |
| 计数归零 | `harness_in_flight`、`env_live` 结束 0 | 同 | tape |
| 显存 | 无 OOM | 无 OOM（第 2 轮 SGLang resume 时） | NVML 采样 |

### D8. 费用（Modal 2026-10-07 价，估算）

- 阶段 1：`H100!:1` $4.39/h（含 CPU/内存按 smoke-12 配置），预计 0.5–0.75 h ≈ **$2–3.3**；若改 2 卡 ≈ $4.5–6.6；上限 **$8**，HARD 3600 s。
- 阶段 2：8×H200 $46.0/h，预计 1–1.5 h ≈ **$46–70**；最坏 2 h $92；上限 **$92**。
- 阶段 0 $0。总上限 $100；超出 `gpu-budget` 当前额度部分需用户另批（Q1）。

## Risks / Trade-offs

- 1×H100 跑 4 层变体未验证可能 OOM → PLAN_ONLY 先估，必要时 2 卡；同配置 OOM 不重试。
- 4 层变体权重 / torch_dist 不在 Volume → 阶段 1 前置 ≈$1–2 的下载/转换（或在阶段 1 同容器内先转，HARD 相应加 900 s）。
- 新镜像 `37ac689e` 上 codex preflight 未验证（smoke-12 用 `12fcd9e5`）→ 阶段 1 兼做镜像适配验证；失败即停，不直接上 8×H200。
- FN thinking 很长，4096 response 下 `max_seq_len` 边界占比可能很高，组内全 0 → 无有效更新 → 以"合法否定"记录，阶段 3 前重估 seq-len。
- `reasoning_effort` 两侧模板不一致（D1 风险点）→ 阶段 0 必须给出结论，否则阶段 1 不开。
- Codex 0.145.0 对 Qwen3.8 工具调用格式（`qwen3_coder` parser）的兼容性只在 Qwen3.5 上验证过。

## Open Questions（需用户拍板）

- Q1 预算：本 change 总上限 $100 是否批准；是否允许阶段 2 在阶段 1 PASS 后立即连跑。
- Q2 失配率阈值 5% 与"未分类失配 >0 则不进阶段 2"的硬门槛。
- Q3 数据：阶段 3 用 TB2 还是 TB2.1；是否引入 Terminal-Bench Pro 扩量。
- Q4 阶段 2 非零奖励是否为硬 PASS 条件（FN 全尺寸在 6 题子集上拿不到 reward=1 时如何处置）。
- Q5 正式训练序列长度（8192 保守值 vs 16384）。

## S14 用户裁定（2026-10-07，对 Open Questions）
- Q1 预算：$100 为硬上限；阶段 1 通过且余额能覆盖阶段 2 时才自动继续；当前余额不足以授权阶段 2，需另批。
- Q2 失配：5% 作为小规模冒烟的临时门槛，同时四类已知失配必须为 0；报告须给样本量、分母与每类计数，不能只报百分比。
- Q3 数据：本轮固定 TB2；TB2.1 留作后续独立对比。
- Q4 非零奖励：不设为硬 PASS；全零只算链路通过、不算学习有效；另加一个已知可得分样例验证奖励链路。
- Q5 序列长度：先 8192，按截断统计再决定是否升 16384。
- 4 层变体进 Modal Volume：作为 GPU 阶段启动前的必备条件（上传、版本核对、加载验证先于计费）。
