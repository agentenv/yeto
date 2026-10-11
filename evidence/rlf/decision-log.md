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
