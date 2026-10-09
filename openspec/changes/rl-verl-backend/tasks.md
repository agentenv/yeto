# 任务

说明：一、二部分为"本周 Modal GPU 可跑通"，三部分为"等 NPU 机器"。GPU 一律 Modal、最小卡数（`H100!:1` 并断言 GPU 名），上卡前先复核代码并写入文档，记账用途。

## 0. 前置（`yeto-framework-decoupling`）

- [x] 0.1 确认 `yeto-framework-decoupling` 阶段 0–3 已合并（1.2 起的前提）。验收：其 tasks 1–4 组全勾、标准样本与边界检查在 main 上通过。
  - 2026-10-08 主 agent 代拍板（S17 夜间）：去耦合 4.2、4.6、5.2/5.3 只部分完成，不挡 verl，按部分完成放行；verl 分支 s17-verl-g1 叠在 s17-decouple-p4@5fac05f7（PR #136，未合）上开发。标准样本与边界检查在该分支上通过（tests/test_import_boundaries.py、test_rl_backend_identity.py）。
- [ ] 0.2 确认其阶段 5 已合并（2.x 首次真跑前提）。验收：其 tasks 6 组全勾。
  - 2026-10-08：阶段 5（6.1/6.2 后端身份、会话契约绑定身份）在 s17-decouple-p4@5fac05f7，未合 main；V1/V2 基于该分支真跑（主 agent 代拍板）。合并后再勾。

## 1. 本周：无卡准备（CPU）

- [x] 1.1 verl Modal 镜像（yeto 默认镜像仍为 ghcr.io/michaellchung/yeto-miles-ports，不变）：基于 fork commit acad9875a8bdfc81afbcb0a50d146b2630f44093 构建，复用 S16 版本组合（torch 2.13.0+cu130、vLLM 0.29.0、transformers 5.12.1、peft 0.19.1）。验收：CPU 干跑 hydra 组装通过；记镜像 digest 与 pip freeze。
  - 2026-10-08：`yeto/rl/adapters/verl/image.py`（fork acad9875 + uv.lock 全量 + 一行本地读回补丁 patch_verl.py，不推 fork）；`--rl-image verl-build:<commit>` 由 Modal 构建（modal_runner 接受这种引擎自建镜像）。CPU 干跑 `scripts/verl_modal_dryrun.py` 通过：版本 torch 2.13.0+cu130 / vLLM 0.29.0 / transformers 5.12.1 / peft 0.19.1，hydra 组装并逐项读回断言键一致，补丁在位，Qwen3-0.6B 规范 LoRA 392 个名与期望一致。证据 s1-runs/s17-verl-dryrun/dryrun.json；镜像 id im-QlL7BxGCpsJguDbsemHSGX（Modal 镜像 id，非 registry digest）；pip freeze 随每次真机运行落盘（verl-pip-freeze-<岛>.txt）。
- [x] 1.2 `yeto/rl/adapters/verl/` 骨架 + CPU 假实现，能力声明只含五端口、引擎名 verl。验收：端口一致性单测全绿，不起 Ray（排除会拉本机 Ray 的测试）。
  - 2026-10-08：`yeto/rl/adapters/verl/` + 注册表 `verl` 行（rollout_meta/entry/elastic_hook/launch_flags/run_config_rules/binding/identity/image）；五端口实现在 `ports_impl.py`（在 verl 任务进程里运行，CPU 侧只测纯函数部分）；launcher 按 `launch_flags.ISLAND_ENTRY_MODULE` 换岛入口、verl 不做 Miles 源码准备。单测 tests/test_rl_verl_adapter.py（不起 Ray）。
- [x] 1.3 配置校验：alpha≠rank 拒绝；commit 不符拒绝；旁路模式+判据拒绝；训练模式开 full_determinism 拒绝、诊断模式未开 eager 拒绝；TIS(下界 0)/IcePop 以外的修正启动即报"verl 后端不支持此修正"；采样默认 T=1/top_p=1/top_k=-1。验收：每条各有单测（正反例）。
  - 2026-10-08：`config.py` validate/build_overrides，每条规则正反例单测。IcePop：设计里开放，但到 verl rollout_corr 的映射尚未核实，**暂时拒绝**（报"映射尚未核实"），列入剩余。
- [ ] 1.4 契约哈希（在去耦合 change 阶段 5 的后端身份哈希上扩展）：输入加入 engine/engine_commit/device_kind/param_name_map_hash/lora_config_hash/wire_dtype，兼容模式字段固定 strict。验收：单测证明同形状 Miles 与 verl 哈希不同、同配置两次哈希相同。
  - 未做（V1/V2 最小子集外）：后端身份已含 engine/commit/设备族/参数名映射哈希（阶段 5），lora_config_hash 与 wire_dtype 进身份哈希尚未加。
- [ ] 1.4a 算法映射（design D11）：verl 适配层用 `@register_adv_est` 包装 yeto 奖励/优势纯函数；"算法字段→verl 配置键"映射表与训练端绑定核对。验收：同一奖励输入经 verl 包装与经 Miles 插件的优势逐位相同（CPU）；缺映射字段启动前拒绝的单测。
  - 部分：GRPO 用 verl 原生 `grpo`（未用 @register_adv_est 包 yeto 纯函数）；"算法字段→verl 配置键"表与三项绑定核对在 `binding.py`（配置逐字、注册函数身份+源码哈希、CPU 实调；Modal 干跑里 verl 可 import 时三项全过）；未映射字段在 island_entry 启动前拒绝（单测）。优势逐位对照未做。
- [ ] 1.4b 逐词元损失对照：yeto 参考函数对 verl 原生 vanilla/dual-clip、cispo、sapo、geo_mean、gspo 做 CPU 对照，结果写入文档；不一致项在算法配置里拒绝。验收：每种损失一条对照测试；不一致组合的拒绝单测。
  - 未做。已知差异待核：verl 默认 clip_ratio_high=0.2、loss_agg_mode=token-mean，与 Miles 默认是否一致未逐位对照。
- [x] 1.5 参数名映射（FSDP2 PEFT 名→规范名）。验收：单测覆盖 Qwen3-0.6B 全部 LoRA 键，映射表哈希稳定。
  - 2026-10-08：`param_names.py`（训练侧去掉 `.default`；vLLM 侧 qkv_proj/gate_up_proj 按序拆包），映射表哈希进后端身份；单测覆盖 Qwen3 28 层 392 个键双向映射。
- [x] 1.6 判据在 yeto 侧按 Miles 口径重算（abs_diff、k3、tis_clipfrac、带符号平均差），verl 原生指标只存档。验收：用 S16 原始数据（s1-runs/s16-verl-mismatch-20261008/…/dump）离线重算，与 VERL-MODAL-CHECK-S16.md §7.2 表一致（A1、D 行）；D 行带符号判据报警。
  - 2026-10-08：中立实现 `yeto/rl/engine/mismatch_criteria.py`（abs_diff/k3/tis_clipfrac/带符号均值 + 阈值表 + judge）；用 S16 原始数据重算 A1、D 两步与 §7.2 表一致（abs_diff 差 <0.0005、k3 差 <5e-5），D 行带符号判据报警（abs_diff 也报，0.035>0.03），A1 不报。verl 侧在 old logprob 重算后由 yeto 计算并落 tape（`rl_mismatch_criteria`、rl_round_trained.mismatch）。
- [ ] 1.7 阈值表按（引擎、训练后端、推理引擎+版本、硬件）为键；缺键报"未标定"。验收：单测。具体数值待定。
  - 部分：阈值表按（引擎、训练后端、推理引擎+版本、硬件）为键、缺键报"未标定"（单测）；只有 verl/fsdp2/vllm-0.29.0/H100 一行，数值为暂定（0.03/0.01/1%/带符号 0.01）。
- [ ] 1.8 运行时清单加 verl/vllm/transformers/peft 字段。验收：单测。
  - 部分：岛启动写 verl-runtime-<岛>.json（verl commit、torch/vllm/transformers/peft、nvidia-smi）+ pip freeze；未并入 yeto 统一运行时清单 schema。
- [ ] 1.9 yeto 会话服务（与 Miles 会话服务同接口）+ 共用 TITO 检查（Qwen3.8 构建器、角色白名单 tool/user、标准模板重渲染比对）。验收：单测——故意插特殊词元计数≥1、追加 system 被拒、Qwen3.8 不回退默认构建器；同一段录制的 codex 轨迹经 Miles 后端与 verl 后端的会话服务组装，词元序列与损失掩码逐个相同。
  - 未做（codex 接入属 2.4，本轮不在 V1/V2 范围）。
- [x] 1.10 发布器两种传送方式（内存 / 按版本号目录落盘）配置切换。验收：CPU 假推理端单测——落盘方式下新版本加载成功后旧目录被删；不可校验时返回 LORA_UNVERIFIABLE。
  - 2026-10-08：`publish.py`（bf16 校验和、compare → VERIFIED / LORA_MISMATCH / LORA_UNVERIFIABLE、DiskTransport 按版本目录且加载成功后删旧）+ `vllm_readback.py`（vLLM worker 内从已注册 LoRAModel 按规范名算校验和）；单测含假 vLLM LoRAModel 打包拆包、转置记录、出错不抛。落盘方式只有 CPU 假推理端验证，真机只跑内存方式。

## 2. 本周：Modal 单卡

- [x] 2.1 推理池 + 训练组接真 verl（v1 训练入口，参照 VERL-MODAL-CHECK-S16.md §7.1）。验收：driver 跑 3 步，tape 有 rl_publication / rl_policy_apply，各组 reward/token 非空。
  - 2026-10-08 V1（s1-runs/s17-verl-v1-20261008c，judgment.json 全过）：`yeto launch --rl-backend verl` → verl island_entry → verl_main（yeto TaskRunner）→ 中立 IslandDriver + verl 五端口（ports_impl）；5 轮，tape 有 rl_publication ×6、rl_round_trained ×5，每轮 32 组 reward/token 非空。rl_policy_apply 在单岛不同步模式下不出现（LocalOnlySync 不写回），V2 两岛覆盖。
- [x] 2.2 策略状态 export/apply + 发布器（内存热更新 TensorLoRARequest）+ 自写推理进程扩展方法按规范名读回校验（"推理端已收到这一版"层）。验收：export→apply→export 逐位相等；读回校验和一致；故意改一个张量被查出；切到落盘方式同样通过。
  - 2026-10-08 V1：发布自检（导出→写回→导出逐位相等；篡改一个张量后读回报 LORA_MISMATCH 且只指出该张量）；v0–v5 读回校验和全部一致（层级：推理端已收到）。落盘方式只在 CPU 假推理端测过，真机未测（未验证）。
- [x] 2.3 TIS(下界 0) 与 IcePop 映射到 verl 原生修正。验收：开 TIS 上界 2.0 跑通，tis 指标落 tape。
  - 2026-10-08 V1：TIS 上界 2.0、下界 0 映射为 verl `rollout_is=token, threshold=2.0`，跑通，verl rollout_corr/* 落 tape（只存档）。IcePop 未映射（暂拒绝，未验证）。
- [ ] 2.4 codex 网关经 yeto 会话服务接 verl（网关不改）。验收：单卡多轮小样本跑通，rl_harness_mismatch 事件出现且正常样本失配为 0；同一段 codex 轨迹经两种后端会话服务得到相同词元序列与掩码。（若 Qwen3.8 单卡放不下，以 1.9 单测 + 0.6B 家族跑通为准，注明。）
- [x] 2.5 10 步冒烟（Qwen3-0.6B LoRA r32 alpha32，T=1，top_p=1）。验收：无 NaN；abs_diff/k3/tis_clipfrac 与 S16 基线同量级（≈0.017/≈0.0008/≈0），相对第 1 步无跳变；按 GPU 必存清单存原始数据；费用记账。
  - 2026-10-08 V1：缩为 5 轮（主 agent 定的 V1 范围）：无 NaN；abs_diff 0.011–0.013、k3 0.0005–0.0006、tis_clipfrac 0、带符号均值≈−0.0006，相对第 1 轮无跳变（≤1.16 倍）；与 S16 同量级（偏低，回复更短、LoRA fp32 主参数）。原始数据 s1-runs/s17-verl-v1-20261008c/tape-direct；费用见台账（V1 合计≈$1.7 估算）。10 步未跑。

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

## 5. V2：两个 verl 岛（S17 夜间追加）

- [x] 5.1 两个 verl 岛接严格同步 syncer（Nebius head + 2×Modal H100!），外层合并后每版两岛全局策略哈希一致；岛 1 在 v2 后退出、launcher 同 id 重启重入。验收：S17-V2-VERL-PRELAUNCH-REVIEW.md §4 J1–J6。
  - 2026-10-08 V2（s1-runs/s17-verl-v2-20261008c，judgment.json J1–J6 全过，rc=0）：v0–v6 每版两岛全局策略哈希相同；岛 1 在 v2 后退出、launcher 同 id 重启、新连接代际重入并跑完；两岛每版发布读回 VERIFIED；判据无报警。前两次 a/b 失败原因（Ray CPU 槽不足、Ray 重启脚本错误）与修复见 S17-V2-VERL-PRELAUNCH-REVIEW.md §8。剩余：重入岛的数据游标不续位（N16 已在 CPU 修，见 6.4）。
- [x] 5.2 Miles 身份 HELLO 进 verl 会话被真 Rust syncer（严格参数）拒绝，反向同样被拒，同为 verl 接受。证据 s1-runs/s17-verl-handshake/result.json（布局哈希与 V1 真机一致）。注：严格模式下被拒会让 syncer 以 layout_hash_mismatch 致命退出；elastic 模式 JOIN 带身份由 N12（PR #140）补。

## 6. 剩余（S17 夜间 V1/V2 之外，未验证）

- [ ] 6.1 IcePop 到 verl rollout_corr 的映射与对照（现为拒绝）。
- [ ] 6.2 1.4 lora_config_hash / wire_dtype 进身份哈希；1.4a 用 @register_adv_est 包 yeto 纯函数并逐位对照；1.4b 逐词元损失对照（含 clip_ratio_high、loss_agg_mode 与 Miles 默认是否一致）。
- [ ] 6.3 1.7 阈值表正式数值；1.8 运行时清单并入 yeto 统一 schema；1.9/2.4 codex 会话服务。
- [ ] 6.4 落盘发布方式真机验证；verl 岛重入后数据游标续位；elastic 模式 verl 岛（JOIN 带身份见 PR #140）。
  - 2026-10-08 N16（分支 s17-rejoin-cursor，CPU 已实现并单测，**未上卡验证**）：数据游标续位已做。查明这是中立层缺陷，不是 verl 独有：驱动只在有批次账本或同步方式提供"按整轮跳过"兜底时才恢复数据位置，而这个兜底只有 elastic 有；严格同步且没有账本的岛（V2 的 verl 岛；Miles 严格模式不带 --rl-elastic 时同理，按代码推断、未上卡）在新容器里重启后数据源从 0 开始。Miles elastic 的 N3/N5 运行中退出重入是同进程暂停后重入，数据源没有重置，事件记录里也没有 cursor_restored，不存在同样的重复。改法：① 中立层 `bridges.whole_round_restart_cursor` 抽成共用函数，StrictAvgSync / DualStrictAvgSync 也提供兜底；驱动只在账本里没有 v-1 这一轮记录时才用兜底（账本记了 v-1 却没游标仍按原规则拒绝）。② verl 岛 `VerlRolloutPool` 提供 data_cursor / seek_data_cursor（`adapters/verl/data_cursor.py`：数 `_fetch_one_gen_batch` 取过的题数，向前跳整块，不能后退），每轮游标写进 verl-rollout 记录。证据：tests/test_rl_restart_data_cursor.py 新增 3 条（无账本严格重启跳过 2 轮，修前失败）、tests/test_rl_verl_adapter.py 新增 2 条（跳过后取到的题与不中断时相同）；39 个相关测试文件 919 过，唯一失败 test_rl_ir_harness 在基线同样失败。待下次 verl 两岛上卡时用 prompts_sha256 复核。
- [ ] 6.5 单岛不同步 + Modal 时 launcher 以 exit 2 收尾（与 Miles 相同的既有行为），是否改为成功由用户定。
