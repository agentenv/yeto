# yeto.rl.harness — agentic harness integrations (ports path)

Change: `openspec/changes/rl-codex-harness-rollout`. Packages: `codex/` (Codex
Terminal-Bench harness, moved from `yeto_miles_secrlenv@5bfc011` + rewritten
OpenEnv adapter) and `gateway/` (protocol gateway core: Responses / Chat /
Messages → TITO chat, prefix hash chains).

## Trust boundary (task 6.3, design D7)

| Item | Trusted layer | Untrusted layer |
|---|---|---|
| Processes | training driver / `agentic_tool_call` worker; `codex_openenv_subprocess_agent_function.run`; verifier runner | `codex_openenv_agent_worker` subprocess (own session, `start_new_session=True`); Codex 0.145.0 binary (isolated `CODEX_HOME`); task container |
| Reward HMAC key | read from `TBENCH_REWARD_HMAC_KEY(_FILE)` (0400/0600, non-symlink, 32–4096 B; `tbench_outcome.validate_hmac_key_source`) only in the trusted process | scrubbed from the worker env (`scrubbed_environment`); the worker refuses to start if `TBENCH_REWARD_HMAC_KEY*`/`SECRLENV_REWARD_HMAC_KEY` is visible (`assert_no_reward_key`) |
| Verifier assets (`tests/test.sh`, solution) | injected by the sandbox backend only at `verify` time (`TB2_WITHHOLD_TESTS` semantics); `evaluate` is not routed on the worker wire (`environment.serve_environment`) | never readable before verification |
| Network | gateway / session-server endpoint; environment control endpoint (bearer token per lease) | task container: default deny egress except the gateway session endpoint (sandbox backend policy, D9); worker reaches only `env_url` and `base_url` |
| Signing points | `tbench_reward.reward_func`, parent `tbench_outcome.verified_outcome`, `trajectory_evidence._verified_outcome` — all three verify; any failure rejects the sample | cannot forge: no key |
| Failure classes | infrastructure (acquire/lease/worker protocol): unsigned, sample ABORTED, no 0 reward; policy boundary (timeout / max_turns / max_seq_len) and wrong answer: signed, reward 0 | — |
| Cleanup | lease `destroy` + `describe()=="gone"` before return; worker SIGTERM→SIGKILL on process group; tool-wait scope balanced on cancel | — |

CPU evidence: `tests/test_harness_codex_openenv.py` (`*_scrubs_key_*`, `*_infrastructure_failures_*`, `*_cancellation_*`).

## Gateway (tasks 4.1/4.2)

`gateway.Gateway.handle(entry, body, trajectory_id)`: entry ∈ {responses, chat,
messages} → canonical chat → `ChainRegistry.locate` (continue / fork
`retry_fork` / break `history_rewrite|template_drops_reasoning|compaction_window`)
→ `SessionBackend.chat` on the chain's session → entry response. Signed sampling
fields cannot be overridden; unsupported shapes fail closed. `max_chains=1` is
the first-batch Codex assertion. `ContextProvider` (identity by default) is the
CompactionRL hook (task 7.2).
