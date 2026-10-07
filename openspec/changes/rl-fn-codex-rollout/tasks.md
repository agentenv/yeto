# Implementation tasks

标记：`[Y]` yeto 自有文件；`[IR]` 需 INFRA 修改（接口请求）；`[GPU]` 需上卡，执行前按 design D7/D8 判据与上限获批并预登记；`[CPU]` 本机可完成。估算为 CPU 工时（agent 小时）或 GPU 费用。全部任务不改 fork Miles/SGLang，不动 `MILES_NEXT_COMMIT`。

## 0. 阶段 0：CPU（$0）

- [x] 0.1 [Y][CPU] `codex_backend.py` 新增 `qwen38_next` / `qwen38_next_4layer`（D1），常量 import 自 `yeto/rl/profiles/qwen3_8_next.py`；新增 `lora_targets` / `lora_expert_rank` 字段，`validate_stock_codex_fields` 加 `lora_expert_rank` 参数并按声明比对；`launcher.py:2036-2051` 透传 `rl_lora_expert_rank`；核对 FN 实际 `rl_model_recipe`（PLAN_ONLY 渲染 `s14-fnsmoke-modal.sh`）。验收：`tests/test_codex_backend.py`（或现有同名测试文件）新增 spec 三个 Scenario 的用例；现有 profile 用例全部不变地通过。依赖：无。估算：1.5 h。
- [x] 0.2 [Y][CPU] 5.1：profile 加 `keeps_history_reasoning`，`GatewayConfig` 从 profile 读取；离线测试用 `~/work/miles-fr1`（HEAD 需 == `MILES_NEXT_COMMIT` 或含同一 jinja）的 `chat_template_verify.py` 渲染 `qwen35` 与 `qwen4exp` 两轮带 think 历史，断言保留。验收：CPU 测试通过，且在 fork checkout 缺失时 skip 而非失败。依赖：0.1。估算：2 h。
- [x] 0.3 [Y][CPU] D1 风险点：同一多轮消息分别用 FN HF `chat_template.jinja`（de4b8e4d）与 fork `qwen3.8_small_and_flash_next_fixed.jinja` 渲染，kwargs 取 profile 声明，比对 token 序列；给出 `reasoning_effort` / `preserve_thinking` 是否只在一侧生效的结论，决定 kwargs 维持或收窄（D1-alt）。验收：结论与 diff 写入 progress.md（S14：infra-drafts/FNCODEX-TEMPLATE-DIFF.md，结论=维持 kwargs，不收窄）；若收窄，0.1 的 profile 与测试同步改。依赖：0.1。估算：1.5 h。
- [x] 0.4 [Y][CPU] 9.2 失配根因：从 `gpu-b1-runs/codex-smoke-20261003-12` 的 events / launcher.log（或岛内 `sample.metadata` 若可取）提取 step-1 的 4 条 `compute_session_mismatch` 记录，按 D2 的 (a)–(d) 分类，写明每条对 FN 是否适用；取不到则标"不可取证"，并在阶段 1 脚本中加 `--rl-observe-timeline --modal-tape-volume` 保证取证。验收：progress.md 表格 4 行，每行有分类或"不可取证"（S14：infra-drafts/FNCODEX-MISMATCH-92.md，1 条 (d) 间接取证、3 条 (a)/(b) 不可区分，逐条 dict 不可取证）。依赖：无。估算：1.5 h。
- [x] 0.5 [Y][CPU] 5.2：`template_drops_reasoning` 断链计数与 reasoning mask=1 单测；FN 声明 `true` 路径下断言该计数恒 0。验收：`tests/test_harness_gateway.py` 新用例通过。依赖：0.2。估算：1 h。
- [x] 0.6 [IR][CPU] A16 产出侧（D5）：INFRA 在 ports `entry.py` 设 `miles_args.yeto_rl_cell_id` 并导出 `YETO_RL_CELL_ID`；harness 侧单测两岛 fake 按成员关闭准入。验收：`test_rl_ir_harness.py` / `test_harness_codex_openenv.py` 新用例；单岛 `resolve_member()==engine:0`。依赖：INFRA 同意接口请求。估算：1.5 h（含接口请求文本）。
- [x] 0.7 [CPU] 运行脚本与复核：`s1-runs/s15-fncodex-4layer-modal.sh`（由 `codex-bundle/launch-9.1.sh` + `s14-fnsmoke-modal.sh` 的 watchdog/judge 合成，`H100!:1`，PLAN_ONLY 打印显存估算与 argv）与 `s1-runs/s15-fncodex-full-modal.sh`；`infra-drafts/FNCODEX-PRELAUNCH-REVIEW.md`（上卡前复核：参数逐项、判据、停机条件、费用）；`gpu-spend.md` 预登记行。验收：PLAN_ONLY 两脚本都通过 `launcher._prepare_rl_args` + `build_modal_island_config().validate()`。依赖：0.1–0.6。估算：2 h。 **（S14 V1 已就绪：Volume `yeto-fn-models` 已有 4 层 HF `/hf/Qwen3.8-Flash-Next-4layer/d19a6b60`（22 文件 30.52 GB，MANIFEST all_ok）与 torch_dist `/torch_dist/qwen3.8-flash-next-4layer_torch_dist`（tracker=release，21.8 GiB，1×H100 转换 207 s ≈$0.33）；PLAN_ONLY ref-load 路径与 MANIFEST 校验一致；详见 infra-drafts/FNCODEX-PRELAUNCH-REVIEW.md §5。）**
- [x] 0.8 [CPU] 4 层变体资产：确认 Volume `yeto-fn-models` 是否已有 4layer HF 与 torch_dist；没有则给出下载/转换命令（S11/S12 流程），并决定是在阶段 1 容器内先转还是单独无卡任务。验收：progress.md 记录 Volume 现状与方案。依赖：无。估算：0.5 h（查询）+ 转换若需 GPU 另计 ≈$1–2。 **（S14 P3b 勾选口径：Volume 现状已查明——4 层 HF 与 torch_dist 均不在 `yeto-fn-models`；命令与费用已给（`infra-drafts/fn-modal/dl_hf_4layer.py` <$0.2，`conv_fn_4layer.py` ≈$1/上限 $1.87，均未运行）；**资产仍未就绪，是阶段 1 开卡前置**。）**

阶段 0 合计 ≈ 11.5 h CPU；0.3 与 0.4 结论为阶段 1 的开卡前提。

## 1. 阶段 1：4 层变体冒烟（Modal 1×H100!，上限 $8）

- [ ] 1.0 [Y][CPU] 放开 Codex OpenEnv 适配器的 profile pin：`forward_legacy_openenv_preflight`（preflight.py:306）、`codex_openenv_agent_function.BACKEND_PROFILE_NAME`/模块级 qwen35_08b 断言、`yeto/rl/__init__.py` identity env `YETO_CODEX_OPENENV_BACKEND_PROFILE` 改为按 `--codex-backend-profile` 允许 `qwen38_next_4layer`（及阶段 2 的 `qwen38_next`），镜像 env 与 `tests/test_harness_codex_openenv.py` 同步；验收：CPU 测试通过，且 PLAN_ONLY 或新增离线 preflight 检查能覆盖该分支。依赖：1.1 首跑结论（S14 L1）。估算：2 h。
- [ ] 1.1 [GPU] 运行 `s15-fncodex-4layer-modal.sh`：**（S14 L1 2026-10-07 首跑 `s15-fncodex-4layer-modal-20261007a` INCOMPLETE，0 轮，≈$0.31：learner preflight `forward_legacy_openenv_preflight` 抛 "the Codex OpenEnv adapter requires backend profile qwen35_08b"——OpenEnv 子进程 agent 路径对 profile 硬编码 qwen35_08b（pins.OPENENV_BACKEND_PROFILE、镜像 env YETO_CODEX_OPENENV_BACKEND_PROFILE），0.1 的 `qwen38_next_4layer` 未放开该 pin；H100/Volume/tape 均正常，模型未加载。需新增代码任务 1.0 放开 OpenEnv profile pin 后再开卡；详见 infra-drafts/FNCODEX-PRELAUNCH-REVIEW.md §9。）** `qwen38_next_4layer`、`--tito-model qwen4exp`、tbench2_smoke6、batch 6 × n 4、`--total-steps 1`、`--rl-observe-timeline`。判据：D7 阶段 1 列全部满足（preflight 通过、24/24 轨迹分类、失配 ≤5% 且全分类、断链 0、grad_norm>0、计数归零、无 OOM）；非零奖励不要求。停机：preflight 失败 / OOM / 任一对齐断言失败 / HARD 3600 s / 费用 $8。依赖：阶段 0 全部。估算：≈$2–3.3（2 卡 ≈$4.5–6.6）。
- [ ] 1.2 [CPU] 结果归档：原始数据按 GPU 清单保存到 `s1-runs/<run>/`（launcher.log、tape、judgment.json、NVML、费用回填）；失配分类表更新；9.2 判据 1/2 的取证结论回写上游 change 的 tasks.md 9.2 行。验收：progress.md 阶段 1 节 + `gpu-spend.md` 回填。估算：1 h。

## 2. 阶段 2：FN 全尺寸单岛两轮（Modal 8×H200，上限 $92）

- [ ] 2.1 [GPU] 运行 `s15-fncodex-full-modal.sh`：`s14-fnsmoke-modal.sh` 的 MODEL/LORA/PAR/COLO/MODALX 不变，DATA/REWARD 换 tbench2 + `tbench_reward`，codex 五件套，`--total-steps 2`，batch 6 × n 4。判据：D7 阶段 2 列（含两轮 payload hash 不同、第 2 轮无 OOM、tis_clipfrac ≤1%）；非零奖励按 Q4 处置。停机：同 1.1 + 第 2 轮 OOM 不重试 + 费用 $92。依赖：1.1 PASS、Q1/Q2/Q4 拍板、预登记。估算：≈$46–70，最坏 $92。
- [ ] 2.2 [CPU] 归档与判定：同 1.2；额外记录每轮 rollout 时长分解（沙箱创建 / Codex 轮次 / verifier）、沙箱并发峰值、`env_live` 曲线，供阶段 3 估时。估算：1 h。

## 3. 阶段 3：并入 FN-TRAIN-PLAN 正式训练

- [ ] 3.1 [CPU] `tests/multinode_gpu/fntrain.sh` 与 `infra-drafts/FN-TRAIN-PLAN.md` §2 参数表：奖励源改 codex 五件套 + `tbench_reward`，数据改 TB2/TB2.1 全量（Q3），`math_reward` 标注为冒烟专用；按 2.2 的时长分解重估首跑健康门费用。验收：`fntrain.sh plan` 渲染 argv 含 codex 五件套且 `_prepare_rl_args` 通过；FN-TRAIN-PLAN 预算表更新并标 [估算]。依赖：2.1 PASS。估算：2 h。
- [ ] 3.2 [CPU] 多岛前置：A16 多岛 GPU 验收与 X5 drain 归 `rl-infra-spec` 3.3；本 change 在 progress 登记依赖与解除条件。估算：0.5 h。
- [ ] 3.3 [CPU] 收尾：上游 `rl-codex-harness-rollout` tasks.md 5.1 / 5.2 / 9.2 按本 change 证据勾选或改写；`rl-algo-supplement` §14 行状态更新；`CODEX-STATUS-S14.md` A7 行改为"已恢复为目标，进度见本 change"。估算：0.5 h。
