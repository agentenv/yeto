# A4 E1-B（3.5）：Nebius 8×H100，代码 47efd25，2026-09-30 —— 未通过（注入无效，失败即停）

配置：公共 A4 配置 + `--rl-test-inject-weight-override /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de2…`（岛上 base 模型快照）；第 2 轮 train 时提交 up→T4R4S0。判据 infra-e1/plan.md E1-B (a)–(d)。

- 注入确实执行：launch.log 第 5096 行 `[yeto] TEST INJECTION YETO_RL_TEST_INJECT_WEIGHT_OVERRIDE_PATH: reloading inference-engine-all-0-0-00002 from …/snapshots/c1899de2…`。
- 但随后 `weights_checker` 全部 200，事务 VERIFYING → weight_admission → COMMITTED → **SUCCEEDED**（journal seq 10–14），成员 00000–00003。判据 (b) 要求终态 `REBUILT_OLD` → **(b) 未通过**。
- 原因（推测，交代码负责人核实）：本配置为 LoRA（r16 all-linear），发布/校验的对象是 LoRA 适配器权重；用 base 模型快照重载 engine 的 base 权重与原本一致（同一 revision），`check_weights` 的 checksum 不受影响 → 该注入在 LoRA 配置下不产生可检测的"错误 payload"。需要改为在已加载适配器之后覆写适配器（或用不同 revision/扰动过的权重）。
- (a) 采样工具问题（本 agent 侧）：容器内 router 采样指向 `<node>:20035/worker_inflight` 得到 404/超时（该端口本次不是 Miles router），未取得 cordon/in-flight 样本 → (a) 无证据。
- (c)(d) 探针：`e1b_probe.py` 用 ssh 默认 `python3` 执行，找不到 `ray`（岛 learner 所用解释器环境不同）→ 未执行。
- 结论：3.5 **未完成**；按"失败即停"停止 A4 其余用例，等待代码修复（注入）与本 agent 的探针修正（router 端口从 launch.log/Miles 参数读取；probe 用 learner 同一解释器）。
- 费用 ≤$11.49（14:09:28–14:31:50，22.4 min×$30.8/h）。释放：nstop 按集群名 `sky down`，nebius API 中该前缀实例数 0。
