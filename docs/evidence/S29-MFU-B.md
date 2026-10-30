# S29 MFU-B：真实长度（262k + 压缩切段）RL 实测报告

数据：Modal base 采样 68 组 / 262 条（`npu1:/data/work/s29-mfu/sample-buffer-modal68`），未截断。
真实长度分布：p50 28.4k、p90 116.5k、max 208.6k，总 12.22M token；prompt max 32.4k（压缩切段样本），response max 200.6k。
集群：npu1+n2+n3 共 24× 910B3（64 GiB/卡），Nemotron 120B，TP4/PP3/EP8 为基准布局。

## 1. 结论（先说结果）

- **B 方案在本集群上跑通了，但覆盖不到真实长度的上限。** 可稳定训练的配置是单卡 micro-batch 预算 28672 token；29k 以上的样本无法进入训练。
- **瓶颈是显存，不是算力。** 单卡 micro-batch 预算每提高一档就 OOM，实测上限落在 28672~36864 之间。
- **CP 三档全部实测失败。** CP8/TP1、CP4/TP2、CP2/TP4 在单卡 28672~40960 预算下都 OOM。
- **实测 MFU 8.5%**（真实长度、与 SFT 同口径），低于 A 方案的 11.2%、SFT 的 18.6%。

## 2. 显存上限实测（倒序：预算从大到小）

全部为 24 卡实跑，"差额"是 OOM 时最后一次申请失败的大小；OOM 时 already-allocated 一律约 57 GiB/61 GiB。

| 运行 | 布局 | max_token_len | 单卡 token | 结果 | 差额 |
|---|---|---|---|---|---|
| s29-b1 | TP4/PP3/CP1 | 262144 | 262144 | OOM | 8.37 GiB |
| s29-b4 | TP4/PP3/CP1 | 131072 | 131072 | OOM | 3.17 GiB |
| s29-b5 | TP4/PP3/CP1 | 98304 | 98304 | OOM | 960 MiB |
| s29-bg | TP4/PP3/CP2 | 81920 | 40960 | OOM | 954 MiB |
| s29-b6 | TP4/PP3/CP1 | 81920 | 81920 | OOM | 2.01 GiB |
| s29-b7 | TP4/PP3/CP1 | 73728 | 73728 | OOM | 1.77 GiB |
| s29-b8 | 同上 + 全阶段 E=1 专家重计算 | 73728 | 73728 | OOM | 698 MiB |
| s29-bh | TP4/PP3/CP2 | 65536 | 32768 | OOM | 822 MiB |
| s29-bn | TP2/PP3/CP4 | 114688 | 28672 | OOM | 584 MiB |
| s29-bf | TP1/PP3/CP8（全覆盖 208.6k） | 262144 | 32768 | OOM | 2.05 GiB |
| s29-b9 | TP4/PP3/CP1 | 57344 | 57344 | OOM | 1.37 GiB |
| s29-bi | TP4/PP3/CP1 | 49152 | 49152 | OOM | 856 MiB |
| s29-bj | TP4/PP3/CP1 | 40960 | 40960 | OOM | 338 MiB |
| s29-bk/bl | TP4/PP3/CP1 | 36864 | 36864 | OOM | 926 MiB |
| **s29-bm** | **TP4/PP3/CP1** | **28672** | **28672** | **跑通** | — |
| **s29-bo** | **TP4/PP3/CP1（加大步）** | **28672** | **28672** | **跑通** | — |

对照实验（排除环境因素）：s29-bl 用与 A6 **完全相同**的环境（不加 E=1、开 comm overlap），只把数据换成 B 的真实长度，36864 一样 OOM。说明上限由数据长度与 token 预算决定，不是这些开关。

## 3. OOM 的显存大头

OOM 栈稳定落在 MoE 专家 FC1 的激活上：
`experts.py:777 bias_act_func → ascend_moe_fused_relu2.py:84 torch.empty_like`。
原因是 MoE 的 permute 激活按 `tokens × top_k(22) / EP` 放大，长序列下它比 attention 大得多。
全阶段打开专家重计算（`ASCEND_RECOMP_POLICY` 的 `E=1` 加到 pp0/pp1）确实有效：同样 73728 预算下，失败申请量从 1.77 GiB 降到 698 MiB，但不足以翻越上限。

## 4. CP 为什么救不了

一个被推翻的判断：先前认为"EP8 与 CP 互斥"。核对 `megatron/core/parallel_state.py:780-800` 后作废——Megatron 0.18.2 用 MoE Parallel Folding，专家组由 `ETP × EP × PP × EDP = world_size` 独立构建（`RankGenerator(..., cp=1)`），与 dense 的 CP/DP 无关。TP4×PP3×CP2 下 EP8 完全合法。

CP 真正的问题是**它没有降低专家侧每卡的 token 数**：实测 OOM 的差额随 `max_token_len`（= MAXTOK×CP）变化，而不是随单卡 MAXTOK 变化。CP2（单卡 40960/32768）与 CP4（单卡 28672）都在 OOM，且 CP 自身还要额外的通信缓冲。
三档都试过：

| 布局 | 可覆盖的最长样本 | 结果 |
|---|---|---|
| TP1×PP3×CP8 | 262144（全部） | OOM 2.05 GiB（TP1 下 dense 权重与 vocab 131072 的 logits 不分片） |
| TP2×PP3×CP4 | 114688 | OOM 584 MiB（TP2 使 dense 权重较 TP4 翻倍） |
| TP4×PP3×CP2 | 65536 | OOM 822 MiB |

PP6 不可行：`world_size % (ETP × EP × PP) == 0` 要求 EP8+PP6 需要 48 卡。

## 5. MFU（同一口径：tokens × 7.34e10 / (s × 24 × 313e12)）

| 运行 | 数据 | tokens/步 | 步时 | MFU |
|---|---|---|---|---|
| SFT 18.60 配方 | 28k binpack | 2.642M | 138.8s | 18.59% |
| A6 | 预筛短样本，28k | 2.646M | 230.3s | 11.23% |
| **s29-bo** | **真实长度 ≤28672** | **3.012M** | **345.1s** | **8.53%** |
| s29-bm | 真实长度 ≤28672，小步 | 1.506M | 259.6s | 5.68% |

口径说明：
- A6 与 B 的步时都是 train step（含 old_log_prob 的额外一次前向），与 SFT 的 138.8s 不完全同构；A 系列 profiling 测得 A6 的 update_actor 单独为 187.7s（13.77%）。B 没有拿到分阶段计时。
- **7.34e10 常数不含 attention 的二次项。** B 的序列更长，真实 FLOPs 更高，所以上表低估了 B。按本步实际的 Σ L²=5.49e10、8 个全注意力层、hidden 4096、全重计算估算，attention 额外贡献约 6.5%，即 s29-bo 的 attention-inclusive MFU 约 **9.1%**。这是估算值，不是实测。

## 6. 为什么 B 比 A 慢

在同样的单卡 token 预算（28672）下，B 比 A 慢约 24%（按 token 归一）。已确认的因素：
1. **序列更长**：B 的均长 17.1k vs A 预筛后更短，单个 micro-batch 装的序列条数更少，attention 与 Mamba 的长度相关开销上升。
2. **micro-batch 数更少**：B 每步 3.01M token / 2 DP / 28672 ≈ 52 个，与 A 相当；但 B 的长序列让 binpack 的装填更不均匀。
3. **A 系列 profiling 已确认的两项同样存在**：old_log_prob 每轮约 42s；host 侧每个 micro-batch 多出约 1.3s。

**未完成**：B 的 msprof + py-spy 倒序瓶颈表没有做。时间都花在上面 15 次 OOM 的二分上，拿到可跑配置时已接近收尾。这是本次最大的缺口。

## 7. 要达到真实 262k 需要什么

按实测，单卡 micro-batch 预算上限 ≈ 28672 token，而覆盖 208.6k 的样本需要 `MAXTOK × CP ≥ 208589`，即 CP ≥ 8。CP8 唯一可行的 24 卡布局是 TP1×PP3×CP8，而它因 TP1 不分片 dense 权重与 logits 而 OOM。因此：

- **24 卡放不下。** 需要更多卡（例如 48 卡做 TP2×PP3×CP8，或 TP4×PP3×CP4），或
- 把 dense 侧显存再压一档：梯度/优化器更激进的 offload、logits 分块（vocab 131072 是大头）、激活 CPU offload，
- 或继续沿用压缩切段，把超长轨迹在采样端就切到 29k 以内（当前 88% 的样本已在 131k 以内，但 29k 以内只有约 1/3）。

## 8. 产物

- 运行目录：`npu1:/data/work/s29-mfu/runs/s29-b{1,4,5,6,7,8,9,f,g,h,i,j,k,l,m,n,o}`
- store 构建脚本：`npu1:/data/work/s29-mfu/build_b.py`（按长度上限筛选、每组取最长 k 条）、`filter_store.py`
- 参数化启动脚本：`npu1:/data/work/s29-mfu/island_b.sh`（TP/PP/EP 改为环境变量）及各 `b*_chain.sh`
- yeto 源码未做任何修改，因此没有新建本地分支提交。

---

## 9. B 的倒序瓶颈表（实测，补充于剖析之后）

剖析方法：复用 s29-prof 的 tap（`megatron.core` 的 forward_step/backward_step 钩子），B 臂与 SFT 臂同窗口（2 个 micro-batch）、同卡位（每个 PP stage 一张）、同 START=70（第二步）。B 臂 trace：`npu1:/data/work/s29-prof/traces/b/pp{0,1,2}`，对比结果 `cmp_b_vs_sft.json`。

### 窗口墙时（区间并集，互斥）

| stage | 臂 | 窗口墙时 s | 纯计算 % | 纯通信 % | 重叠 % | 未覆盖空转 % |
|---|---|---|---|---|---|---|
| pp0 | B | 29.406 | 7.8 | 87.6 | 4.1 | 0.5 |
| pp0 | SFT | 33.245 | 9.1 | 85.2 | 5.4 | 0.3 |
| pp1 | B | 16.009 | 14.3 | 77.5 | 7.4 | 0.8 |
| pp1 | SFT | 18.531 | 16.2 | 73.2 | 9.7 | 0.9 |
| pp2 | B | 3.823 | 63.6 | 0.1 | 31.0 | 5.3 |
| pp2 | SFT | 5.154 | 58.3 | 5.7 | 34.2 | 1.8 |

pp0 设备时间构成（核时求和，可重叠）：B 的每一类计算算子都只有 SFT 的 0.59~0.74 倍（matmul 0.67、moe_gmm 0.72、mamba 0.72、attention 0.74），唯独 elementwise 1.52、reduce 2.26。

### 倒序瓶颈（按对 MFU 的影响从大到小）

| # | 瓶颈 | 量级 | 证据 | 可否优化 |
|---|---|---|---|---|
| 1 | **binpack 装填率只有 71.4%** | 直接浪费约 29% 算力 | 实测装箱：176 条样本装进 176 个 micro-batch，即每条独占一个。真实样本均长 17.1k，任意两条相加都超过 28672 预算。这正是 trace 里 B 每类算子都是 SFT 0.7 倍的原因。 | 要提高装填就得提高 MAXTOK，而 MAXTOK>28672 必 OOM（见第 2 节）。**在 24 卡上无解** |
| 2 | **host 侧每 micro-batch 约 1.3s** | B 约 114s / 338s 步（34%） | A 系列 profiling 实测每 micro-batch 多出 1.3s 且不在设备 trace 内。B 每 DP 有 88 个 micro-batch（A6 为 47），所以这项在 B 里几乎翻倍 | 可优化，但本轮未定位到具体函数（py-spy 未做） |
| 3 | **old_log_prob 额外一次完整前向** | A 系列实测约 42s/步 | 步时包含它；B 的序列更长，这一项只会更大 | **不可用 bypass 规避**：yeto 明确拒绝（会令 old_log_probs=rollout_log_probs 而丢失 pi_prox） |
| 4 | GRPO loss 的 elementwise / reduce | pp0 上 1.52x / 2.26x，绝对值约 0.15s/窗口 | 见上表 | 收益小，不优先 |

### 已尝试并失败的优化

| 优化 | 结果 |
|---|---|
| 提高 MAXTOK（131072→28672 共 10 档） | 全部 OOM，上限 28672 |
| CP2 / CP4 / CP8（三种 TP 搭配） | 全部 OOM |
| 全阶段 MoE 重计算 E=1 | 有效但不足（失败申请量 1.77 GiB → 698 MiB） |
| PP 层切分 42/42/5、24/x/30 | 仍 OOM，且差额变大（换成别的 stage 吃紧） |
| 关 comm overlap / expandable_segments | 本就已生效，无额外收益 |
| old_log_prob bypass | 被 yeto 设计性拒绝 |

## 10. 最终数字

| 运行 | 数据 | tokens/步 | 步时 | MFU |
|---|---|---|---|---|
| SFT 18.60 配方 | 28k binpack，装填约 100% | 2.642M | 138.8s | 18.59% |
| A6 | 预筛短样本 | 2.646M | 230.3s | 11.23% |
| **s29-br（B 最终）** | **真实长度，装填 71.4%** | **3.012M** | **338.2s** | **8.70%** |

未达成 18.6% 的目标。根因是第 1 项：真实长度样本在 28672 的单卡预算下装不满，而预算受显存硬限制无法提高。在 24 卡上这是物理上限，不是调参问题。

## 11. 装填率是主因的直接证据

用同一套装箱近似（首次适配降序，预算 28672）算两臂实际的 micro-batch 构成：

| 臂 | 样本数 | 均长 | micro-batch 数 | 装填率 | 步时 MFU |
|---|---|---|---|---|---|
| A6 | 168 | 24.3k | 168 | 84.6% | 11.23% |
| B（s29-br） | 176 | 20.5k | 176 | 71.4% | 8.70% |

两臂都是**一条样本占一个 micro-batch**（168 条装 168 箱、176 条装 176 箱）：agentic 轨迹长度在 17~28k，任意两条相加都超过 28672 的预算，verl 的装箱无从合并。于是单卡每个 micro-batch 的有效 token 就等于该样本自身长度，**MFU 近似正比于 均长/预算**：
- A6：24.3/28.672 = 84.6% → 11.23%
- B： 20.5/28.672 = 71.4% → 8.70%
- 比值 0.84 vs 0.77，装填率解释了两臂差距的绝大部分，余下由长序列的 attention/Mamba 长度相关开销与 micro-batch 数差异构成。

对照组本想用"样本全部 ≤14336、使每箱能装两条"来正面验证，但真实数据里满足该条件的只有 3 组 12 条，样本量不足以成步，故未做。

**推论**：在本集群上，提升 B 的 MFU 只有两条路——(a) 提高单卡 token 预算（受显存硬限，28672 已是上限）；(b) 把超长轨迹在采样端切短，使多条能装进一个预算（即继续加大压缩切段的力度）。调度与算子层面的优化（comm overlap、插件栈、重计算策略）在 A 系列已证明只有零点几个百分点的空间。

## 12. host 侧 114s 的去向（py-spy 实测）

方法：`py-spy dump` 对 `ray::WorkerDict` 进程连续采样 92 次（3s 间隔），只取活跃线程（Ray actor 方法跑在 "AsyncIO Thread: default"，不是 MainThread——前两次采样失败就是因为只看了 MainThread）。`--native` 在 aarch64 上不支持，用普通 dump。采样落点：83 次在 `actor_update_actor`，8 次在 `actor_compute_log_prob`。

| 占比 | 叶子帧 | 调用链 | 性质 |
|---|---|---|---|
| 30%（28/92） | `synchronize (torch_npu/npu/streams.py:258)` | `_maybe_dtoh_and_synchronize (moe/token_dispatcher.py:930)` ← `token_dispatch` ← `moe_layer.dispatch` | MoE all-to-all 前要把每专家 token 数 D2H 到 host，每个 MoE 层、每个 micro-batch 一次全设备同步 |
| 24%（22/92） | `synchronize (torch_npu/npu/utils.py:75)` | `_communicate_shapes (p2p_communication.py:263)` ← `_communicate` ← `send_forward` | 流水 P2P 发送前的形状握手，带一次全设备同步作为竞态保护 |
| 13%（12/92） | `_fused_adam (torch/optim/adam.py:879)` | `adam` ← `optimizer.step` | 优化器 offload 到 CPU 后，Adam 在 host 上串行执行 |
| 4%（4/92） | `_engine_run_backward` | 重计算的 backward | 正常计算 |
| 其余 | MoE permute / mamba / TE gemm | — | 正常计算 |

三项合计占 host 线程 67%。

**与 SFT 的关系**：这三项 SFT 同样要付（同一套配方、同样 offload 优化器）。真正让 B 更贵的是**付的次数**——前两项都是"每 micro-batch × 每层"的固定开销，而 B 因为装填率只有 71.4%，同样的 token 量要切成 88 个 micro-batch/DP（SFT 为 48），于是这些 host 同步被多付了约 1.8 倍。这就把第 2 项瓶颈又收敛回第 1 项：**装填率是总根因**。

**可优化方向（未实施，供下一轮）**：
1. `_maybe_dtoh_and_synchronize` 的全设备同步可否换成 stream 级等待，或把每专家计数在设备上保留（MindSpeed 已有 fused permute，但计数仍需回 host）。这是最大的单项。
2. `_communicate_shapes` 的同步：现有插件 `ascend_sync_p2p_shape.py` 只缓存了形状张量的构造，注释里明说"没有改动 `torch.cuda.synchronize()` 竞态保护"，所以这 24% 还在。
3. CPU Adam 13%：可考虑把优化器 offload 比例从 1.0 降下来，用显存换时间——但显存正是当前的硬约束，两者冲突。

## 13. 优化前后对比表（逐项实测）

基线重复性：同一配置跑了三次，步时 338.2 / 345.1 / 345.5s，run-to-run 波动约 2%。判定优化是否有效要超过这个带宽。

| # | 优化 | 做法 | 步时 | 相对基线 | 结论 |
|---|---|---|---|---|---|
| 0 | 基线（s29-br/bo/bu） | — | 338.2 / 345.1 / 345.5s | — | MFU 8.53~8.70% |
| 1 | 消除 MoE 的 D2H 同步（host 第 1 项，30%） | `moe_pad_expert_input_to_capacity=True` + `moe_expert_capacity_factor=1.25`（drop-and-pad，dispatcher 走静态计数、完全不做 D2H 同步） | **351.3s** | **+1.7%（在噪声带内，无收益）** | **无效**。而且 capacity padding 会增加专家侧算力，并引入 token drop 影响正确性，不建议采用 |

**这条负面结果很重要**：host 线程里 54% 的 `synchronize` 并不是 host 自身的开销，而是 **host 在等设备**。把 MoE 的同步点彻底去掉后步时没有改善，说明设备侧才是瓶颈，A 系列由"步时反推"得到的"每 micro-batch 多 1.3s host 开销"不应直接解读为可回收的 host 时间。

由此修正第 9 节的倒序表：host 侧三项（MoE 同步 30%、P2P 形状同步 24%、CPU Adam 13%）中，至少第 1 项已证实不可回收。第 2、3 项大概率同理（都是与设备计算重叠的等待），但未单独验证。

**结论不变且更强**：真正的瓶颈是第 1 项——binpack 装填率 71.4%，即设备在每个 micro-batch 里只做了 71% 的有效工作。要提升只能提高单卡 token 预算（受显存硬限 28672）或在采样端把轨迹切短。

## 14. 重要更正：装填率假设被实验否定

第 9、11、13 节把"binpack 装填率 71.4%"当作头号瓶颈，这个判断**是错的**，已由对照实验推翻。

做法：用 `build_b_split.py` 把每条真实样本按 token 位置截成 ≤14336 的一段（prompt 原样保留，response 截断，`loss_mask` 与 `rollout_log_probs` 按同一下标同步截断，三个数组始终对齐），复现"更细的压缩切段"之后的长度分布。装填率由此从 71.4% 提到 98.6%，micro-batch 数从 176 降到 114。

| 运行 | 均长 | 装填率 | micro-batch 数 | tokens/步 | 步时 | MFU |
|---|---|---|---|---|---|---|
| 基线 s29-bo/br/bu | 20.5k | 71.4% | 176 | 3.01M | 338.2 / 345.1 / 345.5s | 8.53~8.70% |
| **s29-s2（模拟细切段）** | **14.1k** | **98.6%** | **114** | **3.17M** | **364.3s** | **8.50%** |

**装填率提升 38%、micro-batch 数减少 35%，MFU 没有任何改善。**

原因：verl 开了 `use_remove_padding=True`（THD 打包），没装满的 micro-batch 本来就只在真实 token 上计算，**不存在 padding 浪费**。所以 trace 里"B 的算子只有 SFT 的 0.7 倍"只是反映 B 的 micro-batch 携带的 token 少，不是浪费。我把它读成浪费，是推断错误。

至此两个主要假设都已被实验否定：

| 假设 | 验证方式 | 结论 |
|---|---|---|
| host 侧 synchronize 是可回收开销（第 12 节） | drop-and-pad 消除 MoE 的 D2H 同步 | 否定：351.3s vs 338~346s，无收益。那些 synchronize 是 host 在等设备 |
| binpack 装填率是头号瓶颈（第 9/11/13 节） | 模拟细切段把装填率提到 98.6% | 否定：364.3s，无收益。THD 打包下本就没有 padding 浪费 |

**尚未解释的差距**（按每秒有效 token 计）：

| 臂 | tokens/步 | 步时 | tok/s | 相对 SFT |
|---|---|---|---|---|
| SFT | 2.642M | 138.8s | 19035 | 1.00 |
| A6 | 2.646M | 230.3s | 11490 | 0.60 |
| B（s29-s2） | 3.17M | 364.3s | 8706 | 0.46 |

A6 对 SFT 的 1.66x 里，A 系列 profiling 已归因：update_actor 187.7s（1.35x）+ old_log_prob 约 42s。B 相对 A6 还要再慢 1.32x，在 micro-batch 更少更满、单条更短的情况下仍然更慢，**原因未定位**。下一轮应当直接对 B 与 A6 两臂做同口径的分阶段计时（old_log_prob / 前后向 / 优化器 / 其余），而不是继续从 trace 的比值去推断——本轮两次推断都被实验证伪了。

## 15. 三项补充核查

### 15.1 相位拆解（py-spy 采样法）

第 12 节那 92 个活跃线程采样本身就给出了相位分布（采样落在哪个 Ray actor 方法上）：

| 相位 | 采样占比 | 按 345s 步时折算 |
|---|---|---|
| `actor_update_actor`（训练前后向 + 优化器） | 90%（83/92） | 约 311s |
| `actor_compute_log_prob`（old_log_prob 的额外一次前向） | 9%（8/92） | 约 34s |

与 A 系列 profiling 给 A6 的 old_log_prob 约 42s 同量级。**结论：差距不在 old_log_prob，而在 update_actor 本身。** B 的 update_actor 约 311s / 3.01M token，SFT 整步 138.8s / 2.64M token，按 token 算 update_actor 慢约 2.0 倍。

### 15.2 FLOPs 口径：attention 的 O(L²) 不是解释

Nemotron 120B 的 89 层里只有 8 层是全注意力（`layers_block_type` 实测），hidden 4096，全重计算下每注意力层约 `8·L²·d`，即全模型约 `2.62e5 · ΣL²`。

| 臂 | ΣL²（每步） | attention FLOPs | 线性项 FLOPs | attention 占比 | 含 attention 的 MFU |
|---|---|---|---|---|---|
| B（s29-s2，224 条 × 14.1k） | 4.45e10 | 1.17e16 | 2.33e17 | +5.0% | 8.50% → **8.93%** |
| SFT（28k binpack，按文档均长 8k 估） | 2.11e10 | 5.5e15 | 1.94e17 | +2.8% | 18.59% → **19.11%** |

两种口径下差距几乎不变（2.19x → 2.14x）。**长序列的 attention 二次项解释不了这个差距。** SFT 那一行的 ΣL² 用的是文档均长 8k 的估计值，属于估算，但即使取极端值也不足以改变量级。

### 15.3 配置 diff：recompute 等关键项两边一致

RL 实际生效的 `override_transformer_config`（从 s29-s2 的 hydra 覆盖项提取，共 13 项）：
`recompute_granularity=full`、`recompute_method=uniform`、`recompute_num_layers=1`、`moe_token_dispatcher_type=alltoall`、`moe_permute_fusion=True`、`moe_grouped_gemm=True`、`moe_router_dtype=fp32`、`moe_aux_loss_coeff=0.0`、`moe_router_bias_update_rate=0.0`、`gradient_accumulation_fusion=False`、`tp_comm_overlap=True`、`num_layers_in_first_pipeline_stage=29`、`num_layers_in_last_pipeline_stage=30`。

SFT 配方（`/data/work/sft-run/go_pp3.sh`）的默认值是 `RECOMPUTE=full`、`RECOMPUTE_METHOD=uniform`、`RECOMPUTE_NUM=1`，并同样叠加 `ASCEND_RECOMP_POLICY`。**两边的重计算配置一致**，不是差异来源。插件栈 RL 侧用的就是 `/data/work/sft-run/plugins`，同一套。

唯一已知未对齐项：`ASCEND_STAGE_TIMER` 的 `stage_timer.py` 在 RL 侧被插件栈排除（它在模块级 import ms-swift，RL 路径没有），所以 RL 拿不到配方自带的分阶段计时——这也是本轮只能用 py-spy 采样法做相位拆解的原因。

### 15.4 结论与下一步

update_actor 内部按 token 慢 2.0 倍，而 recompute/插件/并行形状/attention 口径都已排除。剩下最可能的方向（未验证）：
1. verl 的 Megatron 训练循环与 ms-swift 的 SFT 循环在**每个 micro-batch 的固定开销**上不同（数据搬运、TensorDict 切片、loss 构建）。注意第 14 节已证明这不是 micro-batch *数量* 的问题——B 的 micro-batch 更少更满仍然更慢，所以是**每个 micro-batch 内部**的差异。
2. GRPO 的 loss 路径（TIS 校正、advantage、按 token 的比值）比 SFT 的 CE 重。trace 里 elementwise 1.52x、reduce 2.26x 已指向这里，但绝对值当时看着小——在排除其他项之后，这项值得重新量化。
下一轮应当直接在 verl actor 的 micro-batch 循环里加计时器，把一个 micro-batch 拆成 数据准备 / 前向 / 反向 / loss / 其余，与 SFT 的同口径对比。
