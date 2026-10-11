# Design: RLF Cyber to Codex RL

## Context

rlenvfactory 的 cyber/CTF pipeline 以 pinned challenge archive、Creator 生成 task package、静态 materialization、reference solve quality gate 和 publish 组成任务生产链；其原生 target/grader 依赖隔离 Docker、随机 FLAG 和 trusted host grader。AgentFlow 是一个 Codex/Claude/Kimi/Pi 的 DAG 编排器，适合承载 Creator、reference solver、QA 和 repair 的离线 fanout/merge/loop，但它的 Codex adapter 只产生普通 CLI event trace，不提供 Yeto 所需的 token/logprob/loss-mask、policy version、TITO 或 signed reward。因此 AgentFlow 的边界止于发布已验收任务包。Yeto 已有 Codex subprocess harness、`EnvironmentProvider` lease、trusted verifier、neutral trajectory/reward types 和 Modal provider，但现有 reward environment 主要是 TB2。Modal GPU Sandbox 使用 gVisor，不能可靠承载 nested Docker；因此 RLF Docker 生命周期放在 Modal VM Sandbox，训练仍运行在 Modal GPU 资源上。

## Decisions

1. **新 capability，旧 change 搁置。** 新建 `rlf-codex-rl`；不扩展 `rl-agentic-reward-env`。后者若未实施保持原状，避免把 benchmark-neutral 预备工作变成隐式前置依赖。
2. **AgentFlow 只负责离线生产。** 若采用 AgentFlow，其 DAG 负责 Creator、materialization、reference solve、QA/repair 和 publish 编排；Yeto 只接收带完整身份和验收证据的发布包。AgentFlow 的 node result、success criterion 或普通 trace 不能直接作为 RL sample/reward。
3. **导入边界。** 先导入已 publish/验收的任务包，不在 rollout worker 内运行 Creator。RLF 当前 CTF deliverable 的最小公开交付物是 `public/`、`task.yaml` 和确定性的 `MANIFEST.json`；`task.yaml` 至少提供 `task_id`、`domain`、`target_service`、`target_port`、`source_digest`、`grader_mode` 和 runtime secret 名称。Yeto admission 还要在旁路 manifest 中固定 rlenvfactory commit、package/image digest、验收证据和 schema version，而不能把 Creator 或 AgentFlow 自己写入的字段当作可信身份。
4. **provider 分层。** `RLFTaskCatalog` 负责 admission；`RLFEnvironmentProvider` 负责每 trajectory lease、隔离网络/secret、工具代理和 teardown；trusted verifier 负责 grader 与证据。worker API 不暴露 private solution、FLAG 或 grader。
5. **Codex 复用。** 复用现有 subprocess harness 的 token/logprob/loss-mask/segment 采集，不新建 SWE 或其他 solver 抽象；将 episode/task/policy identity 放入 trajectory metadata。
6. **判分状态机。** worker completion、timeout、grader result、cleanup result 是独立字段。timeout 只影响 episode status；只要状态仍可判分就执行 trusted grader。admission/infrastructure/cleanup error 进入 rejected-data 分类。
7. **训练适配。** reward 实现遵循中立 `RewardResult` 与 success declaration；Miles 和 generic verl adapter 都必须保留 metadata。现有只返回 float 的兼容路径不得用于 RLF gate。
8. **Modal 拓扑。** CPU/VM Sandbox 负责 dockerd、RLF target/grader 和 episode cleanup；GPU 任务负责 Codex rollout/learner。两者通过受保护的 lease/endpoint 和短期 secret 连接，所有资源带 episode label 并在 finally 路径回收。
9. **今晚的模型与上下文策略。** 首选 `Qwen/Qwen3-30B-A3B` thinking mode；先运行 32K 单窗口 baseline（每轮 1024 tokens、48 turns、工具输出 8K、episode 20–30 分钟），先观察成功率和 `max_seq_len_hit`。如果 trace 在提交 flag 前耗尽窗口，再启用 24K compaction window、4K–8K reserve、最多 2–3 次 compaction。Qwen context budget 必须显式注册；不把 64K 作为未经验证的硬上限。
10. **GAE 证据。** RL gate 要求 group 内 reward 非恒定、至少出现成功/失败差异，并保存 reward sum/std、success count、advantage absolute sum 和 policy update；全零 reward 或全零 advantage 只能作为失败诊断，不得伪装成通过。

## Data flow

`RLF/AgentFlow offline DAG -> published RLF package -> admission manifest -> Modal VM episode -> Yeto Codex subprocess harness -> worker trajectory -> trusted RLF grader -> RewardResult(success + evidence) -> Miles/verl adapter -> RL learner -> gate evidence`。

## Failure handling

admission、unsupported_harness、timeout、grader_error、infrastructure_error、cleanup_error 使用稳定枚举；只有 grader 成功且 cleanup 已确认的样本可进入训练数据。每个 episode 写 append-only lifecycle events，包含 identities、时间、状态转换和 teardown confirmation，不记录 private solution 或 secret。

## Validation plan

CPU fake provider 覆盖状态机和 metadata；本地 Docker 覆盖 task/grader contract；Modal VM smoke 覆盖 dockerd/secret/isolation/cleanup；先用 Qwen3-30B-A3B 32K 单窗口做 cyber 能力 baseline，再按截断率启用 compaction，最后运行极小 Codex-only RL gate，核对 trace、reward 分布、success、非零 GAE、policy update 和回收证据。完整 paid GPU run 不作为 proposal 阶段执行内容。
