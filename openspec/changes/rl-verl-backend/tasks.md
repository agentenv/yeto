# 任务

说明：一、二部分为"本周 Modal GPU 可跑通"，三部分为"等 NPU 机器"。GPU 一律 Modal、最小卡数（`H100!:1` 并断言 GPU 名），上卡前先复核代码并写入文档，记账用途。

## 0. 前置（`yeto-framework-decoupling`）

- [ ] 0.1 确认 `yeto-framework-decoupling` 阶段 0–3 已合并（1.2 起的前提）。验收：其 tasks 1–4 组全勾、标准样本与边界检查在 main 上通过。
- [ ] 0.2 确认其阶段 5 已合并（2.x 首次真跑前提）。验收：其 tasks 6 组全勾。

## 1. 本周：无卡准备（CPU）

- [ ] 1.1 verl Modal 镜像（yeto 默认镜像仍为 ghcr.io/michaellchung/yeto-miles-ports，不变）：基于 fork commit acad9875a8bdfc81afbcb0a50d146b2630f44093 构建，复用 S16 版本组合（torch 2.13.0+cu130、vLLM 0.29.0、transformers 5.12.1、peft 0.19.1）。验收：CPU 干跑 hydra 组装通过；记镜像 digest 与 pip freeze。
- [ ] 1.2 `yeto/rl/adapters/verl/` 骨架 + CPU 假实现，能力声明只含五端口、引擎名 verl。验收：端口一致性单测全绿，不起 Ray（排除会拉本机 Ray 的测试）。
- [ ] 1.3 配置校验：alpha≠rank 拒绝；commit 不符拒绝；旁路模式+判据拒绝；训练模式开 full_determinism 拒绝、诊断模式未开 eager 拒绝；TIS(下界 0)/IcePop 以外的修正启动即报"verl 后端不支持此修正"；采样默认 T=1/top_p=1/top_k=-1。验收：每条各有单测（正反例）。
- [ ] 1.4 契约哈希（在去耦合 change 阶段 5 的后端身份哈希上扩展）：输入加入 engine/engine_commit/device_kind/param_name_map_hash/lora_config_hash/wire_dtype，兼容模式字段固定 strict。验收：单测证明同形状 Miles 与 verl 哈希不同、同配置两次哈希相同。
- [ ] 1.4a 算法映射（design D11）：verl 适配层用 `@register_adv_est` 包装 yeto 奖励/优势纯函数；"算法字段→verl 配置键"映射表与训练端绑定核对。验收：同一奖励输入经 verl 包装与经 Miles 插件的优势逐位相同（CPU）；缺映射字段启动前拒绝的单测。
- [ ] 1.4b 逐词元损失对照：yeto 参考函数对 verl 原生 vanilla/dual-clip、cispo、sapo、geo_mean、gspo 做 CPU 对照，结果写入文档；不一致项在算法配置里拒绝。验收：每种损失一条对照测试；不一致组合的拒绝单测。
- [ ] 1.5 参数名映射（FSDP2 PEFT 名→规范名）。验收：单测覆盖 Qwen3-0.6B 全部 LoRA 键，映射表哈希稳定。
- [ ] 1.6 判据在 yeto 侧按 Miles 口径重算（abs_diff、k3、tis_clipfrac、带符号平均差），verl 原生指标只存档。验收：用 S16 原始数据（s1-runs/s16-verl-mismatch-20261008/…/dump）离线重算，与 VERL-MODAL-CHECK-S16.md §7.2 表一致（A1、D 行）；D 行带符号判据报警。
- [ ] 1.7 阈值表按（引擎、训练后端、推理引擎+版本、硬件）为键；缺键报"未标定"。验收：单测。具体数值待定。
- [ ] 1.8 运行时清单加 verl/vllm/transformers/peft 字段。验收：单测。
- [ ] 1.9 yeto 会话服务（与 Miles 会话服务同接口）+ 共用 TITO 检查（Qwen3.8 构建器、角色白名单 tool/user、标准模板重渲染比对）。验收：单测——故意插特殊词元计数≥1、追加 system 被拒、Qwen3.8 不回退默认构建器；同一段录制的 codex 轨迹经 Miles 后端与 verl 后端的会话服务组装，词元序列与损失掩码逐个相同。
- [ ] 1.10 发布器两种传送方式（内存 / 按版本号目录落盘）配置切换。验收：CPU 假推理端单测——落盘方式下新版本加载成功后旧目录被删；不可校验时返回 LORA_UNVERIFIABLE。

## 2. 本周：Modal 单卡

- [ ] 2.1 推理池 + 训练组接真 verl（v1 训练入口，参照 VERL-MODAL-CHECK-S16.md §7.1）。验收：driver 跑 3 步，tape 有 rl_publication / rl_policy_apply，各组 reward/token 非空。
- [ ] 2.2 策略状态 export/apply + 发布器（内存热更新 TensorLoRARequest）+ 自写推理进程扩展方法按规范名读回校验（"推理端已收到这一版"层）。验收：export→apply→export 逐位相等；读回校验和一致；故意改一个张量被查出；切到落盘方式同样通过。
- [ ] 2.3 TIS(下界 0) 与 IcePop 映射到 verl 原生修正。验收：开 TIS 上界 2.0 跑通，tis 指标落 tape。
- [ ] 2.4 codex 网关经 yeto 会话服务接 verl（网关不改）。验收：单卡多轮小样本跑通，rl_harness_mismatch 事件出现且正常样本失配为 0；同一段 codex 轨迹经两种后端会话服务得到相同词元序列与掩码。（若 Qwen3.8 单卡放不下，以 1.9 单测 + 0.6B 家族跑通为准，注明。）
- [ ] 2.5 10 步冒烟（Qwen3-0.6B LoRA r32 alpha32，T=1，top_p=1）。验收：无 NaN；abs_diff/k3/tis_clipfrac 与 S16 基线同量级（≈0.017/≈0.0008/≈0），相对第 1 步无跳变；按 GPU 必存清单存原始数据；费用记账。

## 3. 等 NPU 机器

- [ ] 3.1 确定机型（910B/A3/950）、CANN 与镜像 tag、torch_npu 版本（待定）。验收：清单写入文档并记镜像 digest。
- [ ] 3.2 vLLM-Ascend 上 LoRA 内存热更新是否生效。验收：发布后读回校验通过；不生效则配置切到落盘方式并通过读回校验。
- [ ] 3.3 断言 vllm-ascend `enable_reduce_sample=false`，坚持 bf16。验收：启动断言生效（反例单测）。
- [ ] 3.4 NPU 单卡重复 S16 的 A/B/C 臂（lr0 基线、确定性+eager、LoRA merge 对照）+ top_p=0.9 对照。验收：全部指标非 NaN；带符号判据识别 top_p 偏移；得到 NPU 组合的阈值初值并写入阈值表（数值待定）。
- [ ] 3.5 HCCL 权重同步（训推分离时）。验收：发布读回校验通过。
- [ ] 3.6 运行时清单 NPU 字段（torch_npu、CANN、vllm_ascend、mindspeed、HCCL）。验收：真机清单字段齐全。
- [ ] 3.7 Megatron(MindSpeed) 训练后端接入与其参数名映射表（为 Flash-Next 180B 准备）。验收：切换训练后端后五端口单测不变；小模型 3 步冒烟；Flash-Next 可行性另行评估。
- [ ] 3.8 （可选）syncer torch-svd 工作进程在 npu 设备上可用性。验收：`--iso-worker-device npu:0` 能起并完成一次 SVD，否则记录不支持。

## 4. 待定事项跟进（需用户拍板，非实现任务）

- [ ] 4.1 NPU 型号/到货/镜像；4.2 各后端+硬件+版本组合的阈值数值。验收：用户逐项裁定并回写 design.md「待定」。
