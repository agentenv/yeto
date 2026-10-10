## Context

- 前置：`yeto-framework-decoupling` 阶段 0–3（开工前）、阶段 5（首次真跑前）；适配层落点 `yeto/rl/adapters/verl/`，与 `yeto/rl/adapters/miles/` 同级，只依赖中立端口与类型，不 import Miles 适配层（由边界检查保证）。

- yeto 引擎端口定义在 `yeto/rl/engine/ports.py`，现唯一实现为 Miles 适配层（VERL-ASCEND-DESIGN.md:34-49）。
- verl 已提供：推理 `BaseRollout`/`RolloutReplica`、训练 `BaseEngine` + `EngineRegistry`（按 device/vendor 选实现，NPU 上 megatron 落到 MindSpeed）、权重同步 `CheckpointEngine`、内置训推修正 `rollout_corr`（VERL-ASCEND-DESIGN.md:13-21）。
- 不落盘 LoRA 热更新 verl 已有（`TensorLoRARequest` + vLLM hijack），读回校验 verl 没有（VERL-ASCEND-DESIGN.md:131-133）。
- 岛间 syncer 协议只交换扁平 f32/bf16 增量，与硬件无关（VERL-ASCEND-DESIGN.md:66-72）。
- 实测基线（verl FSDP2 + vLLM 0.29、H100、Qwen3-0.6B LoRA r32、T=1、top_p=1）：abs_diff≈0.017、k3≈0.0008、tis_clipfrac 0–5e-5、10 步无漂移；训练端 logits 为 bf16，同批两次重算逐位一致（VERL-MODAL-CHECK-S16.md §7.2-7.4）。
- 注意：S16 实测走的是 verl `TaskRunnerV1` / `verl/trainer/ppo/v1/trainer_base.py::_compute_old_log_prob`，不是 `ray_trainer.py`（VERL-MODAL-CHECK-S16.md §7.1）。适配层挂钩点须以 v1 路径为准。

## Goals / Non-Goals

**Goals**
- 本周：Modal 单卡上 yeto driver 通过五端口驱动 verl（FSDP2 + vLLM）跑 Qwen3-0.6B LoRA GRPO 冒烟，tape 事件、不一致指标、发布校验齐全。
- 训练后端可替换：同一适配层后续接 Megatron(MindSpeed)。
- 判据口径与后端无关，阈值按后端+硬件+版本分别记录。
- 预留 NPU 迁移与混合岛接口，但不实现。

**Non-Goals**
- 弹性（ElasticRolloutPool、ReconfigurablePlacement、MemberPublisher）、切点保存恢复。
- Miles 岛与 verl 岛混合合并。
- SGLang 路线（verl 中 SGLang 的 LoRA 必须合并，不能发 adapter，VERL-ASCEND-DESIGN.md:125）。
- Flash-Next 全尺寸、2 岛 syncer 跑 verl（可在本周后做）。

## Decisions

**D1 版本：钉用户 fork 的 acad9875a8bdfc81afbcb0a50d146b2630f44093。** 不跟主干、不用 v0.9.1（1876b06d）。理由：用户裁定（SESSION16-HANDOFF.md:376），S16 实测即基于该 commit。适配层只依赖 `BaseEngine` / `CheckpointEngine` / `RolloutReplica` / agent loop 这几层（VERL-ASCEND-DESIGN.md:97）。

**D2 路线：第一步 FSDP2 + vLLM，后续 Megatron(MindSpeed) + vLLM(-Ascend)。** FSDP LoRA 用 HF PEFT，名字天然匹配 yeto 规范名正则（VERL-ASCEND-DESIGN.md:124）；vLLM 支持不合并 adapter。训练后端在适配层内部通过一个"训练后端"选择点（对应 verl `EngineRegistry` 的 backend/device）切换，端口语义不变；参数名映射表按训练后端各自提供。

**D3 五端口映射**（VERL-ASCEND-DESIGN.md:40-46）
- 推理池：包 verl agent loop / 推理副本；推理侧后处理产出每组 GroupMetadata（policy_token、reward 均值/方差、token 数），payload 存 verl 批数据引用，yeto 不解引用。可选字段（tool_wait、trajectory_rewards）先报"未报告"。
- 训练组：train_step = 训练侧重算 old logprob + 优势 + 训练一步；onload/offload 对应引擎搬设备。
- 策略状态：export 取可训练张量（LoRA 时只取 adapter），按规范名排序；apply 写回。
- 发布器：经 checkpoint_engine 推权重（首步 base，之后只推 adapter），随后读回校验（D5）。
- 放置：从 verl 资源池读卡号，设备名用通用前缀（cuda/npu）。

**D4 LoRA 配置：`lora_alpha = rank`、dropout 0、bias none**，显式覆盖 verl 默认 alpha 16（VERL-ASCEND-DESIGN.md:134）。`rollout.load_format=safetensors`（FSDP LoRA 要求，VERL-ASCEND-DESIGN.md:126）。merge 默认 false；S16 未见 merge 对不一致的影响（§7.3 第 2 点），大 adapter 情形未测。

**D5 LoRA 发布与读回校验。** 发布器背后有两种传送方式，用配置切换，对外是同一发布接口：
- 内存（第一版默认）：verl `TensorLoRARequest` 不落盘热更新（VERL-ASCEND-DESIGN.md:131）。
- 落盘（退路，用于 vLLM-Ascend 不支持内存热更新时）：写到按版本号命名的本地目录再加载，加载成功后删除旧版本目录。
读回：自写推理进程扩展方法（经 collective_rpc），从已注册 adapter 按规范名（拆回 q/k/v、gate/up 打包）算校验和，与发送端比较。第一版只证明"推理端已收到这一版"，结果中写明校验层级；做不到时显式返回 `LORA_UNVERIFIABLE`（VERL-ASCEND-DESIGN.md:133）。"推理槽里实际生效的 adapter"层不做。

**D6 契约哈希：后端身份进训练契约哈希，不改 HELLO 线格式。** 基础身份哈希（engine、commit、设备族、参数名映射表哈希）由 `yeto-framework-decoupling` 阶段 5 提供，本 change 在其上补 lora_config_hash、wire_dtype。 输入 `{engine, engine_commit, device_kind, param_name_map_hash, lora_config_hash, wire_dtype}`（VERL-ASCEND-DESIGN.md:148）。第一版 engine/device 直接进哈希 → Miles 岛与 verl 岛必然握手失败（即禁止混合）。预留：哈希输入构造集中在一处，并带"合并兼容模式"字段（第一版只有 `strict`），将来混合模式改为只放规范化后的映射结果。

**D7 训推不一致：修正用 verl 原生，判据指标按统一口径计算，verl 原生指标原样落 tape 作补充。**（VERL-ASCEND-DESIGN.md:236）
- 判据新增"带正负号的平均 logprob 差（训练−推理）"。
- 旁路模式（old=rollout）与任何不一致判据互斥，配置期拒绝。
- `full_determinism` 只允许在诊断/复现模式下打开，且必须同时 `enforce_eager=True`（否则 vLLM 起不来，VERL-MODAL-CHECK-S16.md §7.1、§7.3 第 4 点）；训练模式下拒绝。
- 采样默认 T=1、top_p=1、top_k=-1；非默认 top_p/top_k 时判据必须知情（会有系统偏移）。
- 判据指标统一在 yeto 侧按 Miles 口径重算（逐词元 |Δ| 先样本内再跨样本平均、k3 截 ±10、tis_clipfrac、带符号平均差），挂在 verl 训练侧 old logprob 重算之后；verl 原生 `rollout_corr/*` 只存档作参考，不进判据。
- 修正只开放两边公式等价的部分：TIS 下界 0、IcePop。其余（TIS 下界>0、MIS 各模式、veto、OPSM 等）选 verl 后端时启动即报错"verl 后端不支持此修正"（VERL-ASCEND-DESIGN.md:139-144、222-235）。

**D8 TITO 与 codex 接入：yeto 自有会话服务。** verl agent loop 没有会话服务 HTTP 路由，codex 网关无法直接对接（VERL-ASCEND-DESIGN.md:299）。做法：在 yeto 内实现与 Miles 会话服务同接口的小会话服务——对外仍提供 codex 网关现用的聊天接口（网关不改）；对内首轮用固定模板渲染，后续只对新追加后缀分词，以词元 ID 调 vLLM 生成，保存累积词元/logprob/掩码，拼成训练样本交给 verl。TITO 检查放在 yeto 这层、Miles 与 verl 共用：Qwen3.8 模板构建器、追加角色白名单 tool/user、标准模板重渲染比对（移植 Miles token_seq_comparator 思路），产出同名 `tito_session_mismatch` 进 `rl_harness_mismatch`。不依赖 verl 自带连续词元构建器（其无 Qwen3.8 家族、角色白名单更宽，VERL-ASCEND-DESIGN.md:297）。

**D10 镜像**：yeto 默认镜像继续用 ghcr.io/michaellchung/yeto-miles-ports；verl 引擎用另建的 Modal 镜像（基于钉定 commit）。

**D9 运行时清单**：GPU 记 verl commit、vllm、torch、transformers、peft；NPU 另记 torch_npu、CANN、vllm_ascend、mindspeed，cuda/nccl 字段按设备改记 CANN/HCCL（VERL-ASCEND-DESIGN.md:60）。

**D11 算法如何映射到 verl**（依据 `infra-drafts/RL-ALGO-LOCATION-S16.md` §2、§4）
- 挂接点：verl fork acad9875 `core_algos.py:50-85` 的 `POLICY_LOSS_REGISTRY`/`@register_policy_loss`、`:113-150` 的 `ADV_ESTIMATOR_REGISTRY`/`@register_adv_est`；归约 `agg_loss`（`:1140-1204`，乘 dp_size、需全局计数）；张量为填充后的 (bs, L)。
- 奖励与优势：yeto 已有的奖励后处理/优势变换（GRPO 归一化、MaxRL、MAPO、GDPO、超长惩罚、超长过滤）由 `yeto-framework-decoupling` 阶段 3 拆成纯函数；verl 适配层用 `@register_adv_est` 注册一层薄包装调用同一组纯函数，Miles 与 verl 共用同一实现。验收：同一组奖励输入，经 verl 包装与经 Miles 插件得到的优势逐位相同（CPU）。
- 逐词元策略损失（PPO clip/dual-clip、CISPO、SAPO、GMPO→verl `geo_mean`、GSPO）：第一阶段用 verl 原生实现，不注册 yeto 版本；用 yeto 中立参考函数在 CPU 上对照 verl 原生函数。逐位一致（或在书面给定容差内）才在 verl 能力声明里开放该字段；**不一致即在算法配置里拒绝该组合**（启动前报"verl 后端此损失与 Miles 口径不一致"）。逐项差异（如 CISPO 下界、GMPO 裁剪口径）未核实，对照前一律不开放。
- KL、熵、归约（按词元/按样本、Dr.GRPO 常数分母）：用 verl 原生；归约语义不同（verl 乘 dp_size）属适配层职责，不进中立函数；Dr.GRPO 等 yeto 自写归约器第一阶段在 verl 上不开放。
- 训推修正：同 D7，只开放 TIS 下界 0 与 IcePop（映射到 verl `rollout_corr`）。
- critic 族（PPO/GAE、VAPO、SAO）：若启用，用 verl 自带 critic，只做容差/趋势对照，不要求与 Miles 逐位可比；第一阶段不开放。
- 能力声明：verl 适配层给"算法字段 → verl 配置键"映射表（`yeto/rl/adapters/verl/`），缺映射的字段启动前拒绝；映射后同样执行去耦合 change 的训练端绑定核对（配置键逐字、注册函数身份与源码哈希、CPU 实调一次）。

## Risks / Trade-offs

- GPU 跑通 ≠ NPU 跑通：vLLM-Ascend 上 LoRA hijack 是否生效未核实，verl NPU 测试中无 RL+LoRA（VERL-ASCEND-DESIGN.md:132）。→ NPU 到货后首项验证；不生效则降级为合并模式或 `LORA_UNVERIFIABLE`。
- vllm-ascend 0.23 `enable_reduce_sample=true` 时 logprob 可能错误 → 必须关闭并断言（VERL-ASCEND-DESIGN.md:247）。
- NPU 上 fp16 出现 NaN → 坚持 bf16（VERL-ASCEND-DESIGN.md:249）。
- 跨后端绝对差值不可比 → 阈值按后端+硬件+版本分别标定（VERL-ASCEND-DESIGN.md:243）。
- 参数名映射错误导致"同名不同义"静默合并 → 映射表哈希进契约哈希（VERL-ASCEND-DESIGN.md:96）。
- Flash-Next 不在 verl NPU 支持表（VERL-ASCEND-DESIGN.md:28, 95）。
- verl 推理端用 processed_logprobs，开 top_p 必带偏移（VERL-ASCEND-DESIGN.md:214）。
- NPU 上引擎只能睡到 level 1：vllm-ascend 不支持 `sleep_mode=2`（verl fork acad9875 `verl/third_party/vllm/__init__.py:41-44`，已核实）。→ NPU 岛休眠态仍占 HBM，弹性显存估算不能照搬 GPU 岛的数，到货后单独量一次（S19-NPU-910B4-BRINGUP.md §5 第 5 步）。
- 昇腾机器可能是 aarch64：fork 的昇腾 Dockerfile 同时处理 aarch64 与 x86_64 的 CANN 库路径（已核实），但轮子与镜像缓存不能跨架构复用。→ 构建脚本要求显式 `--arch` 并拒绝跨架构构建（`scripts/build_verl_npu_image.sh`）；到货当天先跑 `uname -m`。
- NPU 依赖栈与 GPU 栈无交集：NPU 为 vLLM 0.23 + torch 2.10 + CANN 9.1.0，GPU 为 vLLM 0.29 + torch 2.13（已核实，出处见 `yeto/rl/adapters/verl/pins.py`）。→ 两个镜像、两套版本断言（`pins.expected_versions(family)`），不共用。

## Migration Plan

引擎选择为显式配置项，默认仍为 Miles；verl 路径完全新增，不改 Miles 路径行为。回退 = 切回 Miles。

## 待定

1. NPU 机器型号与到货时间、镜像选择（910B / A3 / 950，CANN 版本，torch_npu post2 vs post4，VERL-ASCEND-DESIGN.md:27）。
2. 阈值按后端+硬件+版本分别标定的具体数值（GPU 初值参考 VERL-MODAL-CHECK-S16.md §7.4，NPU 无数据）。

已于 2026-10-08 由用户拍板（原待定）：判据统一 yeto 侧按 Miles 口径重算；修正只开放 TIS 下界 0 与 IcePop；LoRA 内存热更新 + "已收到"层读回、落盘退路；codex 经 yeto 会话服务接 verl；默认镜像不变。

## Open Questions（未核实，非决策）

- verl RS 被拒词元的分母处理；SGLang 返回 logprob 是否含温度/top_p（跨后端比较前需确认）；vllm-ascend 0.23 下批不变性是否生效；syncer torch-svd 工作进程在 npu 上能否用（VERL-ASCEND-DESIGN.md:150, 271）。
