# yeto.rl.harness — agentic harness integrations

| Package | Role |
|---|---|
| `harbor/` | Harbor as the agent harness: task loading, sandbox (boat provider), agent, verifier. See `harbor/README.md`. |
| `gateway/` | Records proxy (Responses / Chat -> token-recorded chat) in front of the sampler, prefix hash chains. |
| `reward_env/` | Benchmark-neutral hold-out helpers (`benchmark`, `swebench_verified`). |
| `chat_templates.py`, `reasoning_loss_mask.py`, `placeholder_logprob_mask.py`, `reward_scope.py` | Shared token / mask / reward-scope helpers. |

The self-written Codex harness (`codex/`: Terminal-Bench task loading, sandbox
broker, tool execution, grading, Codex bridge) was removed in S26 E0.5. The last
commit that has it is tagged `s26-pre-harness-removal`. `session_protocol.md`
documents the session-service protocol that harness used; it is kept as the
contract of `tests/session_service_double.py`.
