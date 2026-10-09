# Design

## Context

见 proposal.md 的 Why。下面只写现状与约束。

- 起机入口：`yeto/launcher.py` 的 `launch`（6915 起）。它先调 `sky_patches.install()`（6925）、`prepare_launch_args`（6926）、`_write_run_manifest`（6927），再解析 `--gpu`（6935），在 6959–6960 调 `require_modal_for_gpu_exact` 与 `warn_if_model_wont_fit`，之后才创建云资源。head 模式另有入口：`yeto/cli.py:1733`、`yeto/cli.py:2032` 调 `prepare_launch_args`。
- 现有显存检查：`warn_if_model_wont_fit`（launcher.py:5162）只拿 `MODEL_WEIGHT_GB`（models.py:151）和 `GPU_MEM_GB`（launcher.py:224）比权重总量，只打印警告。它看不到上下文、logits 块、优化器状态和并行布局。
- 现有线程检查：launcher 里没有。只有上卡链脚本手写等待循环（rl-infra-spec/evidence/s17-i3-45/chain.sh:6：`ps -L -u michael | wc -l`，<2800 才继续，最多 60×30 s）。真正拆运行的是 s1run 的线程守卫（THREAD_MAX 3300，ulimit -u 4096）。
- 岛任务生成：`launch` 内 7070–7095 用同一个 `args` 对每个岛调 `task_factory(args, spec, m, num_learners, syncer_addr)`（RL 时为 `make_miles_island_task`，launcher.py:3809）。Modal 岛再经 `build_modal_island_config`（5002）把 sky task 的 run 脚本与 envs 转成 Modal 配置。dry-run 在 7616–7636 用同样的循环。所以"按岛换参数"只要在这一处给每个岛一份改过的 `args` 副本，sky 岛和 Modal 岛都会跟着变。
- 已有先例：`--rl-elastic-debug-delay-s ISLAND:S`（cli.py:493，launcher.py:641 `elastic_debug_delay_env`）已经是按岛号生效的测试开关，但它只通过环境变量注入延迟，不改岛参数。
- 负例要用的参数：`--rl-lr-schedule`（cli.py:481）、`--rl-max-policy-age`（cli.py:476）、`--rl-max-carry-lag`（cli.py:513）。岛身份由 `yeto/rl/engine/backend_identity.py` 计算，`island_contract_sha256`（108 行）把身份与学习率调度绑在一起。

## Goals / Non-Goals

**Goals:**
- 在 `launch` 创建云资源之前完成线程与显存两项检查，失败时零花费退出。
- 显存估算公式可回测：用已知运行的真实峰值或真实 OOM 校准常数，并写清误差。
- 单岛换参数只改白名单参数，sky 岛和 Modal 岛同一处生效，dry-run 能看到差异。

**Non-Goals:**
- 不在运行中持续监控线程数，也不在运行中拆机（拆机正是要避免的损失）。
- 不做精确的显存模拟，只做偏保守的上界估算。
- 不改 s1run 守卫本身（它在本仓库之外）。
- 不支持单岛换模型、换数据、换算法规格。

## Decisions

### 决定 1：线程预检放在 `launch` 开头，读 /proc
launcher 在 `sky_patches.install()` 之前读线程数。计数办法：遍历 /proc/<pid>/status，累加属于当前 uid 的进程的 `Threads:` 字段，等价于 `ps -L -u <user> | wc -l`（后者多一行表头）。SkyPilot API 服务按命令行包含 `sky.server` 识别，单独累加。head 模式（cli.py:1733、2032）同样调用。
- 配置：`--preflight-threads {wait,error,off}`（默认 wait）、`--preflight-thread-start N`（默认 2800）、`--preflight-thread-hard N`（默认 3000）、`--preflight-thread-wait-s S`（默认 1800，每 30 s 重读）。
- 备选：调用 `ps` 子进程。没选，因为 /proc 不依赖外部命令，单测可以喂假的 /proc 目录。
- 备选：达到门槛自动重启 SkyPilot API 服务。没选，因为重启会取消进行中的 sky.launch（memory never-stop-agent-with-live-launch），只提示办法，由人决定。

### 决定 2：显存估算公式（每张卡，单位 GiB）
峰值 = 训练侧 + 推理侧（同卡时） + 固定开销。各项如下。记号：P＝参数量，V＝词表大小，L＝层数，H＝隐藏维，T＝单个微批的 token 数（上下文长度 × 微批样本数，按最长序列算），TP/PP/CP/DP＝并行度，C＝logits 分块 token 数。

1. 权重：`2·P / (TP·PP)`（bf16）。
2. 梯度与优化器：
   - 全参：梯度 `4·P/(TP·PP)`（fp32 主梯度）＋ 优化器 `12·P/(TP·PP·DP_shard)`（fp32 主权重与 Adam 两个矩）。DP_shard 在开分布式优化器时等于 DP，否则为 1。
   - LoRA：底座只算权重项。适配器参数 `P_lora = Σ r·(d_in+d_out)`，梯度与优化器按 `16·P_lora` 计。
3. 激活：`k_act · L/PP · T/(TP·CP) · H · 2`。开全重算时 k_act 取校准值（只保留每层输入），不开时取较大的校准值。
4. 词表 logits 块：`k_logit · min(T, C) · V/TP · 4`（fp32）。k_logit 计入 logits、softmax 与反向各一份。S17 M1 run c 实测这一块 7.6 GiB，用它校准 k_logit。
5. 推理侧（训练与推理同卡且不卸载时）：`mem_fraction_static · 卡显存`，按 SGLang 配置读取。卸载时只算推理侧常驻部分（校准值）。
6. 固定开销：CUDA 上下文、NCCL 缓冲与碎片。实现时拆成两项：`R0`（常数）＋ `FRAG ×（第 1–4 项之和）`（分配器碎片随训练分配量增长）。理由：run c 的 OOM 信息直接给出了碎片 21.04 GiB 与已分配 107.75 GiB，按比例建模比单一常数更贴近实测。
7. 同卡且卸载（colocate + offload，默认 1 卡岛）时训练与推理交替：峰值取"训练阶段（1–6 项）"与"推理阶段（`mem_fraction_static × 卡显存 + R0`）"的较大者。理由：Qwen3-0.6B 运行的实测峰值（35–39 GiB）来自推理阶段的 KV 池，训练阶段远小于它。

阈值：峰值 ≤ `0.9 × 卡显存` 通过（`--preflight-memory-margin` 可调）。卡显存用 torch 报告值：H100 79.18 GiB、H200 139.80 GiB（M1 OOM 信息），其他卡 `GPU_MEM_GB × 0.98`。超限时依次试三种建议并重算：上下文按 1024 一档往下找能通过的最长值、回复长度按同比例缩短（按 1024 取整）、同卡型加一倍卡（TP 或 DP 翻倍）、换下一档显存卡型（GPU_MEM_GB 中更大者），只输出能通过的建议。
- 模型结构（L、H、V、P）从 `yeto/models.py` 的模型表或已下载的 HF config 读取。读不到时打印"显存未估算"并继续（spec 要求）。
- 备选：直接起一张卡跑一步测峰值。没选，因为每次都要花钱，而 S17 两次 OOM 正是"起了才知道"。
- 常数 k_act、k_logit、R、卸载常驻量在任务 2.2 回测前均为未验证。

### 决定 3：回测数据集
用下面五个已知运行校准与检验，原始数据都在 s1-runs 下：
- M1 run b：Qwen3.5-4B、单卡 H100、上下文 16k，首步训练 OOM。估算必须超限。
- M1 run c：同上换 H200，仍 OOM，logits 块 7.6 GiB。估算必须超限，logits 项与 7.6 GiB 相差 ≤10%。
- M1 run d：H200、上下文 12288、回复 6144，峰值 125–126 GB。估算必须通过，且与实测相差 ≤10%。
- N17：多发 A/B 与负例运行（Qwen3.5-4B 单岛 1×H200）。峰值需从原始数据取，取不到记"无数据"，不编造。
- ARU-2：Qwen3-0.6B gsm8k 单卡 H100。峰值同样从原始数据取。
校准只用 b、c、d，N17 与 ARU-2 用来检验。回测结果写进本 change 的 evidence 目录。

### 决定 4：单岛换参数在岛任务循环里换 `args` 副本
新增 `--rl-island-override ISLAND:KEY=VALUE`（可重复）和 `--rl-negative-test-run`。在 `launch` 的岛循环（launcher.py:7086–7088）与 dry-run 循环（7616–7630）里，对被点名的岛先 `copy.copy(args)` 再 setattr，然后交给 `task_factory`。sky 岛与 Modal 岛都从这份 task 生成，所以一处生效。
- 白名单：`rl_lr_schedule`、`rl_max_policy_age`、`identity_test_salt`。最后一项是新的仅测试参数，岛侧把它混进 `island_contract_sha256` 的输入，只为造"身份不符"。
- `rl_max_carry_lag` 不在白名单：它只传给 syncer（launcher.py `_ISLAND_SCHEDULING_PARAMS` → syncer `--max-carry-lag`），岛命令行不带它，按岛换不改变任何岛的行为。传了就起机前报错，写明"syncer 参数，不能按岛换"。（10-09 主 agent 代用户拍板）
- 先在 `prepare_launch_args` 之后对全局参数做完全部校验，再对每个改过的副本单独跑一遍与该参数相关的校验（例如 `--rl-max-policy-age` 的后端支持检查），避免副本绕过检查。
- 记录：`_write_run_manifest`（7542）写 `negative_test: true` 与 `island_overrides` 列表。岛环境变量带 `YETO_ISLAND_OVERRIDE`（JSON），岛入口启动后在 tape 写 `rl_island_override` 事件。看板 reducer 读到该事件后给岛加标记。
- 正式训练禁用（10-09 主 agent 代用户拍板：原文 `--rl-resume` / `yeto export` 在 main 不存在，换成实际接入点）：
  - 续训：续训靠 `--rl-checkpoint-store`。负例运行的每个岛带 `YETO_NEGATIVE_TEST_RUN=1`，启动时在 checkpoint store 根目录写 `YETO_NEGATIVE_TEST` 标记。非负例运行在 store 里发现该标记就报错：launcher 起机前检查本机可读的 store 路径，岛启动时再检查一次（跨机器、Modal Volume 都有效）。本机 runs 目录下 `negative_test: true` 且 `checkpoint_store` 相同的运行清单作为补充检查。
  - 导出：对外导出权重的命令是 `yeto merge --adapter-dir`。adapter 目录及其上两级有 `YETO_NEGATIVE_TEST` 标记，或有 `negative_test: true` 的 run_manifest.json 时报错。
- 备选：给每个岛单独一份完整配置文件。没选，因为改动面大，而负例只需要少数参数。

## Risks / Trade-offs

- [显存估算偏保守会误拦能跑的配置] → 报错里给出各分项与建议，并提供关闭开关；回测要求 run d 不被误拦。
- [估算常数只用 Qwen3.5-4B 校准，换模型可能失准] → 回测里加入 Qwen3-0.6B（ARU-2）检验；误差超过 10% 时只告警不拦截，直到补齐数据。
- [线程等待把上卡链拖长] → 等待上限可配，超时报错不花钱。
- [单岛换参数被误用于正式运行] → 双开关、终端警告、清单标记、续训与导出拒绝。
- [副本 args 与全局校验不一致] → 对副本重跑相关校验，并加单测覆盖每个白名单参数。

## Migration Plan

- 新检查默认开启。上卡链脚本里的手写线程等待可以保留到本 change 通过后再删。
- 回滚：传关闭开关即可，不涉及数据迁移。
- 默认不传单岛换参数时，岛命令行标准样本（tests/test_decoupling_golden.py）必须不变。

## Open Questions

- SkyPilot API 服务在本机的进程命令行特征需实现时确认（暂定包含 `sky.server`）。
- N17 与 ARU-2 是否记录了显存峰值需翻原始数据确认。没有就只用 b、c、d 三点。

## 附表：显存估算回测（任务 2.2）

数据：evidence/memory-backtest.json（脚本 evidence/memory_backtest.py）。实测峰值取岛 tape 里 `rl_load_sample.peak_gpu_mem_bytes` 的最大值（NVML 整卡已用显存，每 10 s 采一次，是真实峰值的下界）。原始数据在 /home/michael/work/s1-runs/ 下（s17-m1-20261008b/c/d、s17-n17-*、s18-aru2-*）。

校准常数：K_LOGIT=0.5（run c 申请 7.58 GiB＝16384×248320×2 字节，一份 bf16 logits），FRAG=0.1953、R0=4.97 GiB（run c OOM 信息），K_ACT（不重算）=45.62（由 run d 实测峰值 123.27 GiB 反解），K_ACT（全重算）=2.0（未验证）。

| 运行 | 用途 | 请求 | 估算 GiB | 实测 | 误差 |
|---|---|---|---|---|---|
| M1 run b | 校准 | Qwen3.5-4B H100 上下文 16384 | 159.6 | OOM，需求 ≥ 78.1 GiB | 估算超限，起机前拦下 |
| M1 run c | 校准 | Qwen3.5-4B H200 上下文 16384 | 159.6 | OOM，需求 ≥ 141.3 GiB | 估算超限，起机前拦下（logits 项误差 -0.0%） |
| M1 run d | 校准 | Qwen3.5-4B H200 上下文 12288 | 123.3 | 123.3 GiB | -0.0% |
| N17 A/B a (a) | 检验 | Qwen3.5-4B H200 上下文 12288 | 123.3 | 126.8 GiB | -2.8% |
| N17 A/B a (b) | 检验 | Qwen3.5-4B H200 上下文 12288 | 123.3 | 127.1 GiB | -3.0% |
| N17 A/B b (a) | 检验 | Qwen3.5-4B H200 上下文 12288 | 123.3 | 123.8 GiB | -0.4% |
| N17 A/B b (b) | 检验 | Qwen3.5-4B H200 上下文 12288 | 123.3 | 126.9 GiB | -2.9% |
| N17 codex | 检验 | Qwen3.5-4B H200 上下文 12288 | 123.3 | 119.5 GiB | +3.1% |
| N17 elastic | 检验 | Qwen3-0.6B H100 上下文 1024 | 36.6 | 35.0 GiB | +4.7% |
| N17 strict | 检验 | Qwen3-0.6B H100 上下文 1024 | 36.6 | 35.0 GiB | +4.7% |
| ARU-2 M0 | 检验 | Qwen3-0.6B H100 上下文 2560 | 36.6 | 36.8 GiB | -0.3% |
| ARU-2 M1 | 检验 | Qwen3-0.6B H100 上下文 2560 | 36.6 | 38.9 GiB | -5.8% |
| ARU-2 MB | 检验 | Qwen3-0.6B H100 上下文 2560 | 36.6 | 56.0 GiB | -34.5% |

判读：
- b、c 估算超限，c 的 logits 项与 7.6 GiB 相差 0%，d 估算通过（123.3 ≤ 125.8 GiB）。三条验收全过。
- run d 是校准点，误差 0 是构造出来的，不算检验。S17 早报写 d 峰值 125–126 GB；tape 最大值是 131.3–132.4 GB（122.3–123.3 GiB）。估算与两种读数都在 10% 以内。早报数字的出处没找到。
- 检验点：N17 五个 4B 运行误差 −3.0%～+3.1%，N17 两岛 0.6B 运行 +4.7%，ARU-2 M0/M1 −0.3%/−5.8%。
- ARU-2 MB 低估 34.5%：tape 峰值随轮次增长（35→56 GiB），配置与 M0 相同（micro batch 1、无动态批），原因未知。因只有 Qwen3.5-4B 校准过，其他模型超限只告警不拦截（design 风险一节）。
- N17 A/B 的实测峰值（126.8–127.1 GiB）超过 0.9 阈值（125.8 GiB），但运行没有 OOM。估算 123.3 GiB 让它通过，结论一致。
