# Proposal: Flash-Next 正式 RL 训练以 codex harness 为 rollout / 奖励来源

## Why

用户 2026-10-07（S14）原话："A7 当然要恢复成目标，FN 训练的时候用的 harness 就是 codex，你得赶紧把这块落地了。"

现状（`infra-drafts/CODEX-STATUS-S14.md` §2，可指到 file:line）：

- codex harness（`rl-codex-harness-rollout`）只在 Qwen3.5-0.8B 上跑通过一次（`codex-smoke-20261003-12`，Modal 1×H100：9.1 PASS、9.2 部分）；10-03 之后无任何 codex GPU 运行。
- Qwen3.8-Flash-Next（FN）S10–S14 的所有运行都用 `yeto.rl.math_reward:reward_func` + `zhuzilin/dapo-math-17k`（`s1-runs/s14-fnsmoke-modal.sh:38`）。FN 单岛链路（Modal 8×H200、TP2 PP4 EP2、colocated+offload、2 轮）在 S14 已 PASS，具备把奖励源换成 codex 的前提。
- 代码层没有 FN 的 codex 后端 profile：`yeto/rl/codex_backend.py` 只有 `deepseekv4` / `qwen38`（指向 `Qwen/Qwen3.8-27B`，不是 FN）/ `qwen35` / `qwen35_08b`；`validate_stock_codex_fields` 还硬性要求 `lora_targets == "attention"`、`expert_full_count == 0`，与 FN 原生 per-expert LoRA（`--lora-targets all-linear --rl-lora-expert-rank 8`，`yeto/rl/profiles/qwen3_8_next.py::LORA_TARGET_SUFFIXES` 14 个叶子）直接冲突。
- 上游 change 的 5.1 / 5.2（`keeps_history_reasoning` 声明与按模板原因断链）、A16（harness 成员键无产出者）、9.2 step-1 失配 4/24 的根因都未做，这些是 FN 多轮 TITO 一致性能否判定的前置。

因此 A7 不是"再跑一次"，而是一条从 CPU profile 到全尺寸单岛的完整落地链，需要独立 change 管理。

## What Changes

- **新增 FN codex 后端 profile**：`qwen38_next`（全尺寸 `Qwen/Qwen3.8-Flash-Next@de4b8e4d`）与 `qwen38_next_4layer`（`CharyZeng/Qwen3.8-Flash-Next-4layer@d19a6b60`），`tito_model=qwen4exp`（fork pin `c35702e` 的 `Qwen38SmallTITOTokenizer`，固定模板 `qwen3.8_small_and_flash_next_fixed.jinja`）；profile 新增 `lora_targets` / `lora_expert_rank` 声明，身份校验改为按 profile 声明比对而不是写死 `attention`。
- **补齐上游 change 的 5.1 / 5.2**：每个受支持 profile 声明 `keeps_history_reasoning`，离线用 fork `chat_template_verify.py` 渲染多轮带 think 历史比对；网关按 `template_drops_reasoning` 原因断链。首批覆盖 `qwen35` / `qwen4exp`。
- **9.2 step-1 失配根因（CPU）**：从 smoke-12 的 session mismatch 记录（`assistant_text` 3、`special_token_count` 1）分类根因，并给出对 FN 的推论；复跑前不上卡。
- **A16 接线**：INFRA 在 ports entry 为每个岛注入 `miles_args.yeto_rl_member_id`（或 `YETO_RL_CELL_ID`），harness 准入落到成员键而不是全局键；单岛允许为 None。
- **数据与奖励**：`--reward-function yeto.rl.harness.codex.tbench_reward:reward_func`，数据为 Terminal-Bench 2 任务行（首批沿用 `codex-bundle/data/tbench2_smoke6.jsonl` 的 R-TB 六任务子集；正式训练扩到 TB2/TB2.1 全量，与 CompactionRL 候选 TB2 共享 provider、bundle 与数据）。
- **分阶段上卡**：阶段 1 四层变体冒烟（Modal 1×H100!，≈$2–3）→ 阶段 2 FN 全尺寸单岛 8×H200 两轮（≈$46–70）→ 阶段 3 并入 `infra-drafts/FN-TRAIN-PLAN.md` 正式训练参数（`tests/multinode_gpu/fntrain.sh`），math_reward 仅保留为管线冒烟用途。

## Non-Goals

- 不修改 fork Miles / SGLang；不升级 `MILES_NEXT_COMMIT` pin（`c35702e` 已含 `qwen4exp` TITO）。
- 不实现真实 `SandboxBroker`（上游 8.x）、网关 HTTP 外壳（10.1/10.2）、partial rollout / 续跑；沿用 A 路径（进程内 `_ResponsesBridge` + `tb2_provider` Modal Sandbox）。
- 不做 CompactionRL 训练本身（`rl-algo-critic-family` 9.4/9.5），只保证环境/数据可共享。
- 不做多岛 FN + codex 的 GPU 验收（A16 只做接线与 CPU 单测；多岛验收随 `rl-infra-spec` X5）。
- 不改 Codex CLI 版本（0.145.0 签名锁定）；不替换 Modal Sandbox 为其它沙箱。
- 不在本 change 内决定正式训练的总轮数、学习率与预算（属 FN-TRAIN-PLAN）。

## Capabilities

### New Capabilities

- `rl-fn-codex-rollout`：Flash-Next 模型族接入签名 codex harness 的契约——profile 身份与 LoRA 声明、模板 / thinking 一致性声明、失配分类、成员键接线、分阶段上卡判据与费用上限。

### Modified Capabilities

（无。`rl-agentic-harness-rollout` 的 5.1 / 5.2 / A16 在本 change 的 tasks 中承接实现，其要求文本不变；完成后由上游 change 勾选。）

## Impact

- 代码（apply 时）：`yeto/rl/codex_backend.py`（新 profile、LoRA 声明字段、校验放宽）、`yeto/launcher.py:2022-2055`（透传 `lora_expert_rank` 给校验）、`yeto/rl/harness/gateway/{core,chains}.py`（5.2）、`yeto/rl/harness/codex/preflight.py:265-283`（A16 消费侧不变）、INFRA `yeto/rl/engine/miles_adapter/entry.py` / `rollout.py`（A16 产出侧，接口请求）、`tests/test_codex_backend*.py`、`tests/test_harness_gateway.py`、`tests/test_harness_codex_openenv.py`。
- 运行脚本：新增 `s1-runs/s15-fncodex-4layer-modal.sh`、`s1-runs/s15-fncodex-full-modal.sh`（由 `s14-fnsmoke-modal.sh` 与 `codex-bundle/launch-9.1.sh` 合成，不进仓库主干，与现有 s1-runs 一致）。
- 依赖：`rl-codex-harness-rollout` 已完成部分（A 路径、网关库、HarnessBoard、tb2_provider）；FN 单岛链路（S14 PASS）；Modal Volume `yeto-fn-models` 上的 full torch_dist（已存在，S14 使用）与 4layer HF / torch_dist（**待下载 / 转换**，SESSION13-HANDOFF §6 待决项）。
- 费用：阶段 1 ≈$2–3（上限 $8）；阶段 2 ≈$46–70（上限 $92）；总上限 $100，需在 `infra-drafts/gpu-spend.md` 预登记。阶段 3 费用归 FN-TRAIN-PLAN。
