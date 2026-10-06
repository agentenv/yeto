# Flash-Next 阶段 A（fnA）真机结果（try27，2026-10-06）

纯 CPU 离线分析，没有做任何 sky 或 GPU 操作。原始数据见 `try27-s12-h100-20261006a/`，校验和见其中的 `SHA256SUMS`（51 个文件，共 5.4 MB，单个文件都不超过 2 MB）。
标注约定：**已验证** = 原始日志或事件里有直接证据；**推断** = 由代码阅读得出，没有直接日志；**未验证** = 没有证据。

## 0. 结论

**按 §4 严格判定：不通过（部分通过）。** 管道端到端跑通：qwen4_exp SGLang 引擎、torch_dist 加载、原生 LoRA、首次导出与发布、6 轮 GRPO 都完成了，rc=0。但有两处不满足：

1. §4 要求"observe 事件（span/load_sample）≥3 个完整窗口且归因非空"，其中 `rl_load_sample` 为 **0 条**，load 归因（queued/active/tool-wait/long-tail）全部为空。span 部分满足。
2. 6 轮 reward、loss、grad_norm 全部为 0，`delta_l2_norm=0`，`zero_variance_group_ratio=1.0`，**没有学习信号**。loss/logprob judge 按字面通过，但 judge 文档自己写明：grad 全为 0 时 logprob 检查是空检查（vacuous），LoRA B 保持为 0。因此"训练正确性"没有得到实质验证。

链内 judge 报的 `trainable FAIL` **是判据调用错误，不是 LoRA 有问题**（见 §2）。改用正确参数重跑后，全部 7 项 PASS（`fna/judge-fn-r16.{out,json}`）。

| §4 子项 | 结果 | 证据 |
|---|---|---|
| 6 轮训练完成 | **PASS（已验证）** | run.log 中 `step 0..5`；rl-island-0.jsonl 中 6 条 `rl_local_round`；rc=0 |
| expected_rank_trainable_4layer(16,8,8) | **PASS（已验证）** | 8 个 rank：4×15746048 + 4×15416832，与公式完全一致 |
| expected_lora_keys = 54 | **间接通过，未直接验证** | `check_lora_weight_equal=True`，日志中没有 `[LORA-CHECK]`/`end_weight_update failed`；SGLang 加载了全部 7 个 leaf。日志没有打印 54 这个计数（§3） |
| loss/logprob judge | **字面 PASS，实质为空** | logprob_abs_diff 6 轮在 0.0360–0.0365 之间且稳定，但 grad 全为 0 |
| observe ≥3 个完整窗口且归因非空 | **部分通过：span 满足，load_sample 不满足** | 37 条 span，60 s 窗口共 9 个，train/rollout/publish 归因非空；`rl_load_sample` 为 0 条 |
| VM→容器时间、torch_dist 加载时间 | **PASS（已记录）** | §5 |

## 1. 运行信息

- 前缀 `s12-h100-20261006a`。Nebius eu-north1，1×8 H100 80GB HBM3（gpu-h100-sxm_8gpu-128vcpu-1600gb，$30.8/h），按需实例。
- 代码 yeto 72be54c2；镜像 `yeto-miles-ports@sha256:37ac689e…`（miles c35702e，sglang 4e4148f）。
- 模型 CharyZeng/Qwen3.8-Flash-Next-4layer@d19a6b60。ref-load 为 `/mnt/yeto-models/torch_dist/qwen3.8-flash-next-4layer_torch_dist`（release 标记，try24 转换，本次 fnprep 跳过）。
- 并行与配置：TP2 PP2 EP4，2 个 TP4 引擎，colocated + offload，r16 / r_e 8，gsm8k，rollout 8×8，max response 1024，seq 2048，lr 1e-5，seed 17，6 步，`--no-sglang-deterministic-inference`，`--rl-observe-timeline`。指纹 fp_fn fn8s = sha256:f70d5baf…。
- 与 §4 预登记的偏差：(a) GPU 从 H200 改为 H100（用户授权，台账有记录）；(b) launcher 打印 `YETO_NEBIUS_NO_BAKED_IMAGE set; using the stock VM image`，即**没有使用**预登记中的烘焙 VM 镜像。容器镜像 digest 与预登记一致。

## 2. trainable FAIL 的原因：judge 调用参数错误

- run.log 中有 8 行 `lora.py:410 - Qwen3.8-Next native LoRA applied: rank=16 expert_rank=8 alpha=16 trainable=N total=…`，rank 0–7 各一行。行首多了 `Qwen3.8-Next ` 前缀，但 `TRAINABLE_RE` 用 `search` 匹配，不受影响。格式没有变化，actor 的日志也已经拉取到。
  - rank 0–3（PP stage 0）：trainable=15746048
  - rank 4–7（PP stage 1）：trainable=15416832
- 公式 `expected_rank_trainable_4layer(16, 8, 8)` 的结果为 (15746048, 15416832)，**与日志逐项一致**。
- 离线重跑时用了 `--num-gpus 8 --rollouts 6`，没有传 `--rank`，judge 的默认值是 `--rank 32`，期望值因此变成 (17074176, 16415744)。同时 judge 只统计 rank= 与参数相同的行，所以 `seen={}`。
- 加上 `--rank 16 --expert-rank 8` 重跑：trainable PASS（exact_count=true，dedup_repeats=0），verdict 为 PASS。
- 结论：**属于判据调用问题，不是真实的 LoRA 错误。**
- 建议（只是建议，没有改代码）：
  1. 链脚本 `tests/multinode_gpu/s11h200chain.sh:141` 调 judge 时漏了 `--num-gpus 8 --rollouts 6 --rank 16 --expert-rank 8`，链内因此 rc=2。
  2. judge 可以去掉 `--rank` 默认值 32（M4 时代的值），改为必填，或者从 run.log 的参数表读取 `lora_rank`。
  3. judge 遇到"rank 不匹配而跳过的 trainable 行"时应当报出来，不应静默显示 `seen={}`。

## 3. 首次导出 38 个张量与 profile 中 54 的差异

结论：两个数字的统计口径不同，彼此不矛盾。38 是**最后一个 PP stage 的 HF 导出张量数**，54 是**SGLang 每个引擎回读的 `lora:*` 键数**。完整模型的 HF 导出应为 78。

**38 的来源：已验证代码，并用计数吻合交叉核对。**
- yeto 对 Flash-Next 使用 Miles c35702e 的 `export_qwen3_8_next_lora_hf_chunks`（`miles_plugins/models/qwen3_8_next/lora.py:628`）。它只遍历**本 rank 的 model chunks**：在 TP/EP 维度做 gather，**不在 PP 维度汇总**。
- yeto 只保留"主 rank"的导出（`state_plugin._is_main_rank` → `actor._is_first_replica_megatron_main_rank`）。Miles 对主 rank 的定义（`initialize.py:160`）是 `dp==0 ∧ tp==0 ∧ pp.rank == pp.size-1`，也就是**最后一个 PP stage**。
- 各类 adapter 的导出张量数：
  - GDN：`_GDN_PROJECTIONS` 共 5 个（in_proj_qkv/z/b/a、out_proj），每个导出 A 和 B，共 10 个。
  - QSA：q/k/v/o 各导出 A 和 B，共 8 个。其中 q/k/v 共享同一个 A，按名字仍算 3 个张量。
  - 共享专家：gate/up/down 各导出 A 和 B，共 6 个。gate 和 up 共享 fc1 的 A。
  - 路由专家：gate_up/down 各导出 A 和 B，共 4 个（按 EP gather 打包，按 --lora-rank 补零）。
- PP stage 1 负责第 2–3 层（GDN+MoE、QSA+MoE）：10 + 8 + 2×(6+4) = **38**，与日志一致。
- PP stage 0 负责第 0–1 层（GDN+MoE、GDN+MoE）：2×(10+6+4) = 40，未导出。全模型合计 78。
- 计数吻合是强证据，但 run.log 没有打印张量名清单，所以"38 恰好是 stage 1 的那些张量"属于**推断，逐名未确认**。

**54 的来源（profile 注释、`tests/test_rl_qwen3_8_next_profile.py:115`）：**
- 54 = 3 层 GDN×10 + 1 层 QSA×8 + 4 层 MoE×4，口径是 SGLang 每个引擎的 `lora:*` 回读键数。
- 运行时没有任何代码调用 `expected_lora_keys()`。
- 本次的直接证据只有两条：
  1. 各 TP rank 都打印了 `loaded weights for target modules ['down_proj','gate_up_proj','in_proj_ba','in_proj_qkvz','o_proj','out_proj','qkv_proj']`，正好是 `SGLANG_LORA_LEAVES` 的 7 个 leaf。
  2. `check_lora_weight_equal=True`，且没有出现 `[LORA-CHECK]` 失败行。
- 键数 54 本身**未直接验证**：日志没有输出这个计数，也没有查过 Miles 校验是否会在通过时打印。

**影响：**
- 本次 no-sync，导出只用于指纹，不会 apply 回去，所以只导出 stage 1 不影响训练。
- 以后如果要用这个导出做 policy sync 或 checkpoint，必须补上 PP 汇总，否则会丢掉第 0–1 层的 LoRA。这一点记为已知限制。
- 发布 payload 为 466,769,920 字节（每次 publish）。它走的是 Miles 自己的权重更新路径，与这 38 个张量无关。

## 4. 学习信号为 0

**事实（已验证，来自 run.log 与 jsonl）：**

| 轮 | rewards | truncated | resp_len 均值 | grad_norm | loss | logprob_abs_diff |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.953 | 1000.8 | 0 | 0 | 0.03607 |
| 1 | 0.0 | 0.969 | 1013.8 | 0 | 0 | 0.03648 |
| 2 | 0.0 | 1.000 | 1024.0 | 0 | 0 | 0.03616 |
| 3 | 0.0 | 0.984 | 1020.2 | 0 | 0 | 0.03650 |
| 4 | 0.0 | 0.969 | 1000.0 | 0 | 0 | 0.03626 |
| 5 | 0.0 | 0.969 | 999.0 | 0 | 0 | 0.03596 |

- 每轮 64 条轨迹（8 组 × 8 条），`zero_variance_group_ratio=1.0`，`delta_l2_norm=0.0`。第 0 轮最短响应 75 token，说明有少数样本正常结束，但奖励仍为 0。

**对判据的含义：**
- 管道跑通，但没有学习信号：GRPO 组内奖励方差为 0，advantage 为 0，所以 pg_loss 为 0、grad 为 0。
- LoRA B 保持零初始化，第 0–5 轮的策略与基座完全相同（LoRA 增量为 0）。
- 因此 logprob 检查只验证了"基座的训练与推理 logprob 一致（约 0.036）"，没有验证"LoRA 更新后训练与推理仍一致"。M3 #2 的 QSA de-interleave 端到端检查在这一轮也是空检查。

**原因分析（推断，未逐条验证）：**
1. 4-layer 是从全模型截出的 4 层调试切片，不是能正常工作的语言模型。大约 95% 的样本写到 1024 上限都不出 EOS，说明生成基本不连贯。这是主因。
2. gsm8k 加上 `gsm8k_reward:score` 只有答案完全匹配才给分，对这个模型来说几乎不可能得分。
3. 即使把 max response 调高，也只会产生更长的无效文本，**不会**带来奖励。

**获得非零信号的可选方案（只列出，不实施）：**
- (a) 保持 4-layer，换一个容易得分的奖励：格式或长度奖励（例如"在 N token 内输出 EOS"、"含数字"），或者对 logprob/长度给连续值奖励，确保组内有方差。目的只是让 grad≠0、LoRA B≠0，以便实质验证 logprob 一致性和导出变化。
- (b) 换一个极简合成任务（复制或重复固定 token），设 max response 64–128，用匹配率给连续奖励。
- (c) 判据中加一项"至少 1 轮 grad_norm>0 且导出 hash 发生变化"。judge 已经有 `nonzero_grad_rounds` 字段，可以把它升级为必过项。
- (d) 真正的学习信号要靠完整模型（fn32s/fn32b），这不属于阶段 A。

## 5. 其余数据

- **VM→容器（fnboot 冷启动，已记录）**：
  - 10:54:12 Launching
  - 10:57:17 Instance is up（3 min 05 s）
  - 11:05:22 Docker container is up（又过了 8 min 05 s，使用 stock VM 镜像，镜像在 VM 上现拉）
  - 11:05:56 Cluster launched
  - 合计 **11 min 44 s**
  - fna 复用热集群：11:09:07→11:09:25，18 s。setup 首行出现在 11:09:57。
- **torch_dist 加载（已记录，精度约 ±1 s）**：rank 0 从 `loading release distributed checkpoint` 到 `successfully loaded`，11:11:39→11:11:58，约 **19 s**。时间戳是 launcher 收到日志的时间；日志里只看到一对，即 stage 0 的 rank 0。转换本身在 try24 完成：190 s，23.4 GB。
- **每轮时长（sweep.json）**：
  - 第 0 轮 186.6 s，其中 train 150.3 s 含首次编译和 warmup；publish span 143 s，包含首次导出与发布。
  - 第 1–5 轮 37.3–37.9 s，中位数 37.6 s。
  - 第 1–5 轮的 span：offload 约 5.5 s，generate 约 4.9 s，onload 约 1.4 s，train 约 10.3 s，outer_sync 约 5.6 s。
  - 吞吐 tok/s 约 5.9k–6.5k（rl_round_trained）。
  - 整个 fna 运行 11:08:59→11:21:29。
  - sweep 的 `valid=false` 只是因为 D1 专用检查 `uuids_4_in_pool` 不满足，与 fnA 无关。
- **显存（rl_resource_sample，60 s 采样一次，共 8 次）**：
  - 采样峰值为单卡 79,767 MiB / 81,559 MiB（97.8%），出现在第 3 轮与第 4 轮之间，所有卡都在 77.7–79.8 GiB 之间。
  - 其余采样值在 13–69 GiB 之间。
  - 采样过稀，**真实峰值未知**，可能更高。
  - `terminal.txt` 中 nvidia-smi 全部为 0 MiB，那是运行结束后的读数，不是峰值。
  - 没有出现 OOM。
- **observe**：
  - 37 条 `rl_timeline_span`：每轮 offload/generate/onload/train/outer_sync 各 1 条，外加 7 条 publish，role 为 `trainer+rollout`。
  - 用 `timeline.load_windows(events, 60)` 离线切出 9 个窗口。第 3–9 个窗口的 gpu_busy/train/rollout/publish 比例都非空，例如窗口 6：gpu_busy 0.677、train 0.59、publish 0.212。
  - **`rl_load_sample` 为 0 条**，queued/active/tool_wait/tail_wait 全部为 None。
  - 原因未确认。推断：`MilesRollout.load_sample` 在路由的 `/worker_inflight` 不可用时会静默返回 None（`rollout.py:670-683`），而 run.log 里完全没有 `worker_inflight` 字样。另外第 1–5 轮 generate 只有约 5 s，与 5 s 的采样间隔相当，即使 probe 正常也很难采到。
  - D1 PRELAUNCH-REVIEW 记录 M3 在 D1 上产生过 load_sample，所以这次缺失可能与 Flash-Next/ports 的路由配置有关，**未验证**。
- **花费（gpu-spend.md 估算）**：
  - try27 为 $14.34（fnboot 约 $7.3，fna 累计约 $14.29）。
  - try22–27 合计 $68.78。
  - 链结束时两次 cleanup 都是 clean，集群已删除。sky down 过程中出现过一次 security group 删除重试的告警，最终 rc=0。

## 6. try22–27 问题链

| try | 前缀 | 失败点 | 根因 | 修复提交 | 花费 |
|---|---|---|---|---|---|
| 22 | s11-h200-20261005v | fnboot：`model type qwen4_exp` | learner 在 Miles 注册 HF 别名之前直接调用 megatron.bridge AutoBridge | 5810d11d（不走 AutoBridge 构建 provider）+ e1c29130（`--rl-boot-only`） | $10.39 |
| 23 | s11-h200-20261005w | fnprep 转换：`torchrun: command not found` | ssh 非登录 shell 的 PATH 不含镜像 venv | 7604f2e8（运行时 preflight：PATH、Miles/Megatron 根目录） | $8.95 |
| 24 | s11-h200-20261005x | fnA：`canonical LoRA layout is empty` | Flash-Next 的 specs=() 仍去求 hash；Bridge 导出不认识原生 adapter | 461f378b（从首次原生导出学习 layout） | $11.97 |
| 25 | s11-h200-20261005y | fnA trainer：`No module named megatron.post_training` | editable finder 只映射 core/training，island PYTHONPATH 缺 `/root/Megatron-LM` | f970f0c7 | $10.89 |
| 26 | s11-h200-20261005z | 8 个 SGLang 引擎：FlashInfer GDN prefill 与确定性推理冲突 | yeto 默认开确定性推理，与 Flash-Next 配方不一致 | 72be54c2（Flash-Next 强制 `--no-sglang-deterministic-inference`） | $12.24 |
| 27 | s12-h100-20261006a | 6 轮完成；链内 judge rc=2 | 链脚本调 judge 缺参数；离线重跑时 rank 用了默认值 32 | 无代码修复（建议见 §2） | $14.34 |

详细复盘见 infra-drafts/FN-A-PRELAUNCH-REVIEW.md §9–§15，以及 FN-IMAGE-PLAN.md、FN-4LAYER-PROVENANCE.md。

## 7. 下一步建议（都未实施）

1. 修链脚本的 judge 调用和 judge 的 rank 默认值（§2）。
2. 查清 `rl_load_sample` 为 0 的原因：用 CPU 核对 Flash-Next ports 下的 `sglang_router_ip/port` 和 `/worker_inflight` 是否可用。必要时让 probe 在返回 None 时打一次诊断日志。
3. 补 PP 汇总导出，或者至少在文档中注明"导出只覆盖最后一个 PP stage"（§3）。
4. 如果要实质验证 LoRA 训练正确性：在 4-layer 上用简单奖励（§4 方案 a/b）再上卡一次，并把"grad_norm>0 且导出 hash 变化"加入判据。上卡前按惯例先复核并预登记。
5. 再上卡时考虑把 resource sample 的采样间隔降到 ≤10 s，以便得到可信的显存峰值。
