## Why

用户要接入昇腾 NPU 算力（SESSION16-HANDOFF.md §0.1），而 yeto 当前唯一的训练后端 Miles（Megatron + SGLang）只支持 CUDA。verl 在 NPU 上有现成的 FSDP2/MindSpeed + vLLM-Ascend 路线（VERL-ASCEND-DESIGN.md:23-30）。用户裁定"本周要能跑的 verl 后端"（SESSION16-HANDOFF.md §6 第 13 条）。S16 Modal 单卡实测确认 verl 自身训推一致在默认设置下正常（VERL-MODAL-CHECK-S16.md §7.2、§7.4：abs_diff≈0.017、k3≈0.0008、tis_clipfrac≈0，10 步无漂移），具备写提案的前提。

本草稿只写文档、不改代码，未拍板处标"待定"，等用户审阅。

## 前置依赖

本 change 依赖 `yeto-framework-decoupling`：
- verl 适配层开工（tasks 1.2 起）前，须完成其阶段 0（标准样本与边界检查）、阶段 1（引擎核心切断反向依赖）、阶段 2（奖励/过滤/harness 中立化与会话服务协议）、阶段 3（配置与算法字段中立化、奖励/优势纯函数、逐词元损失参考函数）。
- verl 首次真跑（tasks 2.x）前，须完成其阶段 5（后端身份进契约）；本 change 的契约哈希任务改为在该身份哈希上扩展。
- 建议先完成阶段 4（Miles 代码归位与后端注册表），使 verl 直接落在 `yeto/rl/adapters/verl/` 并经注册表以 `--rl-backend verl` 选择。
- 其阶段 6（硬件层）、阶段 7（云层与按任务分 spot/按需调度）可与本 change 并行。

## What Changes

- 在 yeto 引擎端口层新增第二个引擎 `verl`：由 yeto driver 通过五个端口（推理池、训练组、策略状态、发布器、放置）驱动 verl，弹性/切点类端口第一版不声明（VERL-ASCEND-DESIGN.md:49）。
- verl 来源固定为用户 fork https://github.com/michaellchung/verl ，钉 commit `acad9875a8bdfc81afbcb0a50d146b2630f44093`（SESSION16-HANDOFF.md:376）。
- 路线：第一步 FSDP2 + vLLM，Qwen3-0.6B LoRA 先在 Modal GPU 跑通，再迁 NPU（vLLM-Ascend）；Flash-Next 180B 全尺寸后续走 Megatron(MindSpeed) + vLLM(-Ascend)。训练后端在 verl 适配层内必须可替换（SESSION16-HANDOFF.md:377）。
- LoRA 配置遵守 yeto 规则 `lora_alpha = rank`（VERL-ASCEND-DESIGN.md:134）；采样默认温度 1、top_p=1、top_k=-1。
- 训推不一致：禁止 verl 旁路模式与不一致判据同时开启；判据新增"带正负号的平均 logprob 差（训练−推理）"（实测 top_p=0.9 时 −0.032 而 k3/tis_clipfrac 识别不出，VERL-MODAL-CHECK-S16.md §7.2 D 行、§7.3 第 3 点）；`full_determinism` 只允许诊断复现，不允许训练（会使组内采样完全相同，§7.3 第 4 点）。
- 契约哈希：第一版禁止 Miles 岛与 verl 岛混合合并——后端、设备、参数名映射表哈希进入训练契约哈希；预留混合模式接口（SESSION16-HANDOFF.md:378）。
- 算法映射：奖励后处理/优势变换用 yeto 纯函数，经 verl `@register_adv_est` 挂接；逐词元策略损失用 verl 原生实现，以 yeto 参考函数在 CPU 上对照，公式不一致的组合在算法配置里拒绝（见 design D11）。
- 判据统一在 yeto 侧按 Miles 口径重算，verl 原生指标只存档作参考；修正只开放两边公式等价的部分（TIS 下界 0、IcePop），其余在选 verl 后端时启动即报错"verl 后端不支持此修正"。
- LoRA 发布：第一版走内存热更新（verl TensorLoRARequest），自写推理进程扩展方法按规范名算校验和读回，只证明"推理端已收到这一版"，做不到时显式 LORA_UNVERIFIABLE；退路为落盘到按版本号命名的本地目录再加载、成功后删旧版本；两种方式在同一发布接口后用配置切换。
- 词元进词元出（TITO）与 codex 接入：在 yeto 内实现与 Miles 会话服务同接口的小会话服务，对外仍提供 codex 网关现用的聊天接口（网关不改），对内用词元 ID 调 vLLM 生成，保存累积词元/logprob/掩码并拼训练样本交给 verl。TITO 检查（标准模板重渲染比对、追加角色白名单 tool/user、Qwen3.8 模板构建器）放在 yeto 这层，Miles 与 verl 共用（VERL-ASCEND-DESIGN.md:296-299）。
- 镜像：yeto 默认镜像继续用 ghcr.io/michaellchung/yeto-miles-ports；verl 另建 Modal 镜像。
- 运行时清单补 verl/vllm（NPU 上另加 torch_npu、CANN、vllm_ascend、mindspeed）版本（VERL-ASCEND-DESIGN.md:60）。

非目标（本 change 不做）：弹性成员变更、切点保存恢复、Miles/verl 混合岛合并本身、SGLang 路线、Flash-Next 在 NPU 上的全尺寸训练、修改 codex 网关本身。

## Capabilities

### New Capabilities
- `rl-verl-engine-backend`: verl 作为 yeto 第二个引擎的端口实现、版本钉死、可替换训练后端、LoRA 配置、发布与读回校验、契约哈希中的后端身份、运行时清单。
- `rl-mismatch-judging`: 训推不一致判据与修正在跨后端时的口径（带符号平均差、旁路互斥、确定性只诊断、阈值按后端+硬件+版本标定、修正只开放公式等价子集）。
- `rl-tito-session-service`: yeto 自有会话服务（与 Miles 会话服务同接口），承担词元序列对齐与 TITO 检查，Miles 与 verl 共用；codex 网关经它接 verl。

### Modified Capabilities
（无。现有主 spec 只有 head-run-teardown，与本 change 无关。）

## Impact

- 代码（后续实现时）：新增 `yeto/rl/adapters/verl/`、yeto 会话服务；`yeto/rl/engine/capabilities.py` 能力声明；训练契约哈希输入（`yeto/rl/local_learner.py`、`yeto/rl/sao_streaming_runtime.py`）；`runtime_manifest.py`；新 Modal 镜像（基于 fork 钉的 commit）。
- 依赖：verl fork、vLLM（实测 0.29.0）、torch 2.13.0+cu130、transformers 5.12.1、peft 0.19.1（VERL-MODAL-CHECK-S16.md §7.1）。NPU 侧版本组合待定。
- 费用：本周 Modal 单卡（H100）若干小时，按 GPU 预算记账。
- 待定项汇总见 design.md「待定」。
