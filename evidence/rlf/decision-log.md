# RLF implementation decision log

- 2026-10-11: Reduced sub-agent concurrency to one after provider returned HTTP 429. Work is now sequential.
- 2026-10-11: Codex is the only online RL harness. AgentFlow remains offline task production/acceptance metadata and is never trusted as grader evidence.
- 2026-10-11: Qwen3-30B-A3B thinking mode is the planned baseline (32K window, 1024 generation tokens/turn, 48 turns, 8192-byte tool output). GPU launch is gated on CPU evidence.
- 2026-10-11: The legacy `rl-agentic-reward-env` change remains deferred; this change owns the RLF bridge.

## 2026-10-11 trusted reference flag 修复（通过）

- 假设检验：先前失败归类为 solver_control_flow_or_transport。本机 gcc 13.3 复现证明 offset 72 直接跳 win 可拿 flag，控制流无误。
- 根因：旧 solver 只 `recv(4096)` 一次，printf 的 "hello ..." 与 win() 写的 flag 分两个 TCP segment 到达，单次 recv 永远看不到 flag。
- 修复：recv 循环读到 EOF/超时（`modal_ctf_trusted_recvfix.py`）；另备 ret gadget 对齐回退（未用到）。
- 证据：`modal-ctf-trusted-recvfix2.log`，sandbox sb-01M4MB62RAE1FR0Y3HDGWZJRX7（已 terminate），direct-win 首次尝试即得 `flag{trusted_vm_acceptance_7b3d}`，与目标容器注入的 FLAG 一致，SOLVER_RC:0，CLEAN_RC:0。
- 结论：trusted_reference_flag = pass；failure_class 改为 resolved_transport_single_recv。image build、GPU probe 仍然不算 RL 通过。

## 2026-10-11 打包 CTF 编排的真实 VM E2E（通过）

- 缺口修复：W4 打包的 `RlfModalProvider(ctf=...)` 首次上真实 Modal VM 时 dockerd rc=127——`vm_exec` 用 `bash -lc` 而 docker:27-dind 是 alpine 无 bash。
- 修复：`RlfModalVmConfig.image_run_commands`，CTF 路径在镜像构建期 `apk add --no-cache bash`（acquire_vm 内惰性应用，CPU 测试无 modal 依赖）。
- E2E 证据：`modal-ctf-lease-e2e.log`，VM sb-01M4MEKE66M30GA31EZBGEBRR0，26 s 完成 acquire→dockerd→build→target healthy→relay exploit→submit→trusted grade(reward=1.0, RLF 三元组在 grader_evidence)→destroy→describe=gone。
- trusted flag 由可信侧 env 注入，日志中只出现 agent exploit 自己读出的 flag。

## 2026-10-11 Qwen3-30B-A3B baseline 首跑（链路通过，预算不可行）

- 运行：A100-80GB bf16，SGLang 157s + session server 48s 就绪；episode 走完 acquire→Codex drive→trusted grade 全链路（evaluated，非 infra/rejected）。
- 途中修复：①`modal run -m` 在容器内补水时重跑 spec_from_env 崩溃 → 改 driver 模式（同 eval island）；②SessionServerConfig lora_rank=None 被 pydantic 拒绝 → 先误改 32，触发 session server 请求 miles_lora adapter 400 → 正确修法 lora_rank=0（is_lora_enabled = rank>0 or path 非空）；③harness 非 200 不带 body → 加有界诊断读取（503 不读的行为钉不变），pin 重钉 d2ebc3d9。
- 结果：success_rate=0.0，reward=0.0，end_reason=max_seq_len（response_truncated）：模型第一轮把 1024 token 上限全部花在 thinking（reasoning_chars=4606，finish_reason=length，content_chars=0，0 tool calls）。reasoning 内容方向正确（在找 win 地址）。
- 结论：32K 上下文没问题（max_model_total_tokens=1694）；1024/turn + xhigh thinking 对该模型不可行。5.7 的 compaction 验证需要"能跑多轮"的 baseline，当前预算下没有真实截断样本可验。
- 花费：4 次 A100 运行合计约 24 min ≈ $1.0 [估算]（3 次启动期失败 + 1 次完整 episode）。

## 2026-10-11 baseline 放宽预算后（用户批准 1024→8192）

- 用户批准：单轮上限放宽到 8192，加简易题验证 reward 链路。
- run8（8192 cap）：ret2win 真实 36 轮 36 tool calls、31K tokens，max_seq_len 截断——5.7 的真实触发条件已出现。bad case：模型用 32 位地址（\x0b\x85\x04\x08）+ 线性 padding 爆破，从未命中 flag。ezshell 因 stage_package 只发 Dockerfile/entrypoint.sh/public 而缺 service.sh 构建失败 → fixture 改为 public/service.sh。
- run9：发现任务陈述没给目标地址，模型全程猜错地址（$TARGET_URL、localhost:1234），两题都在 6–8 轮后提交错 flag。修复：catalog.prompt_rows 从 task.yaml target_port 注入 "The target service listens on 127.0.0.1:31337"（RLF CTF target 固定 host network，地址恒为 127.0.0.1:<port>）。另发现 run.json 顶层 task_ids 不生效（需放 baseline 键下），已在新 run 配置修正。

## 2026-10-11 5.7 compaction 真实验证（通过）

- 24K profile（qwen3_cyber）三次真实跑：run14 10 轮死于单轮 8192 截断（未达触发线 20480，gate 正确不动）；run15 1 轮 protocol error；run17 6 轮早提交。profile 注册与 gating 真实有效。
- 机制验证另用 12K 窗口/4K reserve/单轮 4096（触发线 8192）：run18 第 8 轮 10320 tokens 触发，summary_ok=true，segment 0 kept_steps=1，context 8210→2176 切换后继续 4 轮并以 submit 正常结束。segment identity 与共享 reward 合并由 CPU 测试锁定（test_codex_bridge_compaction / test_rlf_rollout_entry）。
- 途中修复：baseline runner 未预创建 segment sessions（已补，含 teardown 清理）；for_profile 与覆盖参数冲突（kwargs 优先）；budget==max_seq_len 桥约束（profile 24576 需配 context_length 24576）。
- 观察：该模型 thinking 轮常超 2K tokens，单轮截断（harness TITO 完整性规则）是长 episode 的主要死因；8K 窗口对该模型不可行。

## 2026-10-11 5.8 GPU RL gate 首跑（失败：训练端 cutlass 编译，非 RLF 链路）

- 配置：L40S×1，qwen35_08b，GRPO 2 轮×4 组×4 样本，LoRA r16，rlf_success reward，rlf_modal_provider，任务 ezshell+ret2win。
- 进展：岛启动、RLF provider 环境装配、rollout 正常产样本（prefill batch 4 并发）——RLF→Codex rollout→reward 链路在训练路径上真实工作。
- 失败点：Megatron 训练步 cutlass MLIR 编译错误 `unable to compute crd2idx ... cute.layout "(?):(?{i64 div=8})"`，island rank 0 exit 1，launcher 按固定名单 0s 恢复超时停跑（exit 4）。
- 教训：岛停止前 tape/checkpoint 未拉回（"no pulled Modal tape file"），本跑 rollout 的 reward 证据随岛销毁。下次开卡：跑挂时先拉回 ~/yeto-output/ 与 ~/yeto-rl/ 再停。
- failure class：trainer_kernel_compile（与 RLF 无关；与模型/LoRA shape 相关）。
- 途中修复（已提交 3ff8e93e）：launcher 透传 YETO_RLF_* 前缀、rlf_modal_provider 入 MODAL_SANDBOX_PROVIDERS、岛侧任务目录 /root/sky_workdir 路径、sandbox token 走 Modal Secret。

## 2026-10-11 gate 第三跑（rlf-gate-20261011-5，失败，双根因均已定位）

代码 d15f6e1b（session-server-workers=8 生效：48s 就绪，远小于 300s 窗），unfused 后端，约 16 min ≈$0.55。
进展链完整走到训练步：v0 rl_publication 落磁带（policy update 证据已可收集）、rollout 0 收齐 16 样本进训练转换。两个失败：

1. **训练步 CUDA OOM**（unfused 后端，seq 8192：注意力分数矩阵全程驻留，与 LoRA 无关）。下一跑改 `--rl-training-attention-backend fused`（cuDNN，显存与 flash 同量级且完全绕开 flash-attn cute JIT 编译失败）。
2. **RLF CTF 沙箱被 Modal 300s 空闲回收**：`RlfModalVmConfig` 写死 `idle_timeout=min(ttl_s,300)`，模型长生成轮（exec 空窗 >300s）后 VM 消失，下一 exec 抛 NotFoundError → episode 无 trusted finish → preflight fail-closed 丢弃该组（3 组被丢、由补位凑齐 16 样本；reward 全 0 与 0.8B 能力一致，非管线错误）。修复 de0a7b7a：idle_timeout 默认跟随 ttl_s，可显式覆盖。判据：fail-closed 行为本身正确，属环境 TTL 配置缺陷，非代码竞态。

待用户重启后第四跑：fused + 沙箱修复，PREFIX 递增，预算 ≤$3。

## 2026-10-11 gate 第四跑（rlf-gate-20261011-6，失败）与第五跑决策（代拍板）

第四跑（fused/L40S/0.8B，约 12 min ≈$0.4）：沙箱修复生效（NotFoundError 归零）、fused 旗标生效（Megatron 打印 AttnBackend.fused）、rollout 0 收齐 16 样本。训练步失败：TE 报 "No dot product attention backend is available"——cuDNN fused 对 L40S/sm89 + 该输入组合不可用。三条后端全灭：flash（cute JIT 编译错）、unfused（seq8192 OOM：mbs 已为 1、recompute 未启用且无透传旗标、单样本分数块 7.34 GiB、总缺口 ~5.4 GiB）、fused（不支持）。trusted-fields 拒收 2 组为 ret2win 超时中止的 by-design 行为。

第五跑决策（代拍板，留痕）：
1. 卡型换 H200×1（141 GiB）+ unfused：4B 在 seq 8192 mbs1 无 recompute 估算峰值 ~87 GiB（据 0.8B 实测 42.5+7.3 外推），141 GiB 有充足余量；不加 recompute 透传（无旗标、加管道有风险）。
2. 模型换 Qwen3.5-4B（rev 851bf6e8，M1 生产钉版，profile qwen35）：0.8B 连续两跑 16+ 样本 reward 全 0，gate 五判据中 std>0/success_count/advantage 三项都依赖至少一次 ezshell 成功，0.8B 不可行。4B 是 M1 agentic 跑验证过的形状（H200 + session server + TorchMemorySaver 共置）。
3. 其余不变（LoRA r16 attention、seq 8192、GRPO 2×4×4、worker 8）。预算预登记 ≤$6。

第五跑（rlf-gate-20261011-7）head 形态（代拍板）：本地 preflight 线程守卫把内核线程计入 root（~1000），真实用户线程仅 ~1841 仍卡 2800 线；按用户指示改用 `--controller head`（Nebius eu-north1 无卡 head）+ `--preflight-threads off`——理由：guard 的目的是保护跑 driver 的机器，head 模式下 driver 在 Nebius VM，本地只剩瘦客户端，本地计数是不适用的机器。Modal 令牌按 s19-critic-g3.sh 模式走环境变量进 head（不落盘）。跑完先从 head 拉回产物再 sky down。
