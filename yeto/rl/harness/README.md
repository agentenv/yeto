# yeto.rl.harness — agentic harness integrations

| Package | Role |
|---|---|
| `harbor/` | Harbor as the agent harness: task loading, sandbox (boat provider), agent, verifier. See [`harbor/README.md`](harbor/README.md). |
| `gateway/` | Records proxy (Responses / Chat -> token-recorded chat) in front of the sampler, prefix hash chains. |
| `reward_env/` | Benchmark-neutral hold-out helpers (`benchmark`, `swebench_verified`). |
| `chat_templates.py`, `reasoning_loss_mask.py`, `placeholder_logprob_mask.py`, `reward_scope.py` | Shared token / mask / reward-scope helpers. |

The earlier self-written Codex harness has been removed.
[`session_protocol.md`](session_protocol.md) documents the session-service
protocol it used; it is kept as the contract of `tests/session_service_double.py`.
