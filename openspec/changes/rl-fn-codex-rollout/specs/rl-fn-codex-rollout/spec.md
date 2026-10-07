# Spec Delta

## Purpose

Qwen3.8-Flash-Next（4 层变体与全尺寸）接入签名 codex harness 的行为契约：profile 身份、模板一致性声明、失配分类、成员键、分阶段上卡判据与费用上限。本 spec 不改变 `rl-agentic-harness-rollout` 的既有要求。

## ADDED Requirements

### Requirement: Flash-Next codex 后端 profile
系统 SHALL 提供 `qwen38_next` 与 `qwen38_next_4layer` 两个签名 codex 后端 profile，`tito_model` MUST 为 fork pin 中的 `qwen4exp`，`model_identifier` / `model_revision` MUST 与 `yeto/rl/profiles/qwen3_8_next.py` 的 HF 仓库与 revision 常量相同。profile MUST 声明 `lora_targets` 与 `lora_expert_rank`，身份校验 MUST 按 profile 声明比对而不是固定值；未声明的 profile 默认 `attention` / 0，现有 profile 行为 MUST 不变。

#### Scenario: FN 全尺寸 LoRA 配置通过校验
- **WHEN** `--codex-backend-profile qwen38_next --tito-model qwen4exp --model Qwen/Qwen3.8-Flash-Next --model-revision de4b8e4d… --lora-targets all-linear --rl-lora-expert-rank 8`
- **THEN** 校验通过，`apply_chat_template_kwargs` 等于 profile 声明

#### Scenario: LoRA 目标漂移被拒
- **WHEN** profile 为 `qwen38_next` 而 `--lora-targets attention`
- **THEN** 启动失败，错误包含 `Qwen3.8-Flash-Next` 身份漂移

#### Scenario: 旧 profile 不受影响
- **WHEN** profile 为 `qwen35_08b` 且 `--lora-targets attention`、expert rank 0
- **THEN** 校验结果与修改前一致

### Requirement: 模板一致性声明
每个受支持的 codex profile SHALL 声明 `keeps_history_reasoning`，网关 MUST 以该声明初始化 `ChainRegistry`。系统 SHALL 提供离线测试：用 fork pin 的 `chat_template_verify.py` 渲染至少两轮带 reasoning 的历史，断言渲染结果与声明一致；首批 MUST 覆盖 `qwen35` 与 `qwen4exp`。`keeps_history_reasoning=false` 时断链原因 MUST 只为 `template_drops_reasoning`，且 reasoning 生成段 mask MUST 为 1。

#### Scenario: qwen4exp 保留历史 reasoning
- **WHEN** 用 `qwen3.8_small_and_flash_next_fixed.jinja` 渲染含两段 assistant think 的历史
- **THEN** 第一段 think 文本仍出现在渲染结果中，声明为 `true` 的测试通过

#### Scenario: 声明与模板不符
- **WHEN** 某 profile 声明 `true` 但渲染丢弃了历史 think
- **THEN** 离线测试失败，profile 不得进入 GPU 阶段

### Requirement: 失配分类与门槛
对每条 `tito_session_mismatch`，系统 SHALL 记录上游 `compute_session_mismatch` 的分类（至少 `assistant_text`、`special_token_count`）与根因标签；阶段进入下一阶段前，失配率 MUST ≤ 5% 且未分类失配数 MUST 为 0。

#### Scenario: 失配率超阈值
- **WHEN** 阶段 1 的 `tito_session_mismatch_rate` 为 8%
- **THEN** 阶段 2 不启动，progress 记录分类与下一步 CPU 分析

#### Scenario: 失配全部可分类且低于阈值
- **WHEN** 24 条中 1 条失配，分类为 `special_token_count` 且根因已记录
- **THEN** 该判据通过

### Requirement: harness 成员键由 INFRA 产出
ports 引擎 SHALL 在 harness preflight 前为每个岛设置 `miles_args.yeto_rl_cell_id`，并向 rollout worker 导出 `YETO_RL_CELL_ID`；单岛运行时 MUST 同样设置。harness 侧 `close_admission(members)` MUST 只影响对应成员键下的会话。

#### Scenario: 单岛
- **WHEN** `--rl-single-island-no-sync` 启动
- **THEN** harness 计数的成员键为 `engine:0`，而非全局键

#### Scenario: 多岛按成员关闭准入
- **WHEN** 两岛 fake 下关闭成员 1 的准入
- **THEN** 成员 0 的 `allow_new_session` 仍为真

### Requirement: 分阶段上卡与费用上限
FN × codex 的 GPU 验证 SHALL 按阶段 1（4 层变体，Modal `H100!`，上限 $8）→ 阶段 2（全尺寸 8×H200 两轮，上限 $92）顺序进行；每阶段 MUST 在上卡前完成复核文档与 `gpu-spend.md` 预登记，前一阶段的全部判据未通过（或未记录合法否定结论）时后一阶段 MUST 不启动。同配置 OOM 或 preflight 失败 MUST 不重试。

#### Scenario: 阶段 1 preflight 失败
- **WHEN** 新镜像上 codex bundle 校验失败
- **THEN** 停止，不启动阶段 2，费用回填

#### Scenario: 阶段 2 PASS
- **WHEN** 两轮 grad_norm>0、payload hash 两轮不同、失配率 ≤5%、`env_live`/`harness_in_flight` 归零、无 OOM
- **THEN** 本 change 的 GPU 判据通过，FN-TRAIN-PLAN 可把奖励源切为 `tbench_reward`

### Requirement: 正式训练的 rollout / 奖励来源
FN-TRAIN-PLAN 的正式训练参数 SHALL 以 codex harness（`agentic_tool_call.generate` + `codex_openenv_subprocess_agent_function.run` + `tbench_reward`）为 rollout 与奖励来源；`math_reward` MUST 只用于管线冒烟并在脚本中标注。

#### Scenario: fntrain.sh 渲染
- **WHEN** 阶段 3 渲染正式训练 argv
- **THEN** 含 codex 五件套与 `tbench_reward`，不含 `math_reward`
