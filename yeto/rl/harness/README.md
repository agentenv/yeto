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

## Island wiring (stage 4: IR-1..IR-4 as implemented by INFRA, `integ-decl` 8347351)

| Concern | Interface (INFRA) | Harness side |
|---|---|---|
| Preflight before any allocation (2.3) | `entry.preflight_stage(miles_args, launch, ...)` runs `HarnessPreflight = Callable[[miles_args, launch], None]`, resolved from `miles_args.yeto_harness_preflight` / `YETO_HARNESS_PREFLIGHT`, before `connect_island_ray` | `codex.preflight.harness_preflight` (`HARNESS_PREFLIGHT_SPEC`): `check_harness_reward_scope`, agent-function check, `preflight_codex_openenv`, then installs the `EnvironmentProvider` (`miles_args.yeto_harness_environment_provider` / `YETO_HARNESS_ENVIRONMENT_PROVIDER`) and the island boards into `codex_openenv_subprocess_agent_function.configure` |
| Drain probe (3.2) | `tool_wait.drain_blockers(router, tool_wait, harness: HarnessSnapshot)`; pool reads `HarnessBoard` via `entry.harness_source` (named actor `yeto-rl-harness-<learner>`); `MilesRolloutPool.drain` closes admission first | agent entry: `allow_new_session(member)` → `enter_session`/`exit_session`, `lease_acquired(id, deadline=)`/`lease_released` (idle sandboxes count; hard deadline bounds the wait); gateway: same admission + sessions, `AdmissionClosed` on a draining member |
| Target policy version (4.4) | `RolloutPool.generate(rollout_id, *, expected_policy_version)` → `MilesRolloutPool` publishes the token via the metadata sink; `rollout_meta_hook.expected_policy_version(sample)` reads it (metadata first) | `codex_openenv_generate.generate` and the agent entry refuse (`PolicyVersionMissing`) without a token; drift → `policy_age_violation=1` → `harness_counters` → driver `PolicyIdentityError` |
| 1.7 metrics (4.5) | `timeline.LOAD_SAMPLE_SCHEMA`: `harness_in_flight`, `env_live` (gauge), `tito_session_mismatch`, `tito_chain_breaks{reason}`, `policy_age_violation` (counter; no `_total` suffix), labels `profile_hash`/`epoch` | gateway mirrors `record_session_mismatch` / `record_chain_break(reason)` onto the `HarnessBoard`; generate wrapper writes `policy_age_violation` into sample metadata |
| `reward_scope` (7.1) | `config.check_harness_reward_scope(miles_args.yeto_harness_reward_scope)` in `validate_parsed_args` | re-checked in `harness_preflight`; the harness itself only emits `reward_scope=trajectory` |

CPU evidence: `tests/test_harness_codex_openenv.py` (`*_harness_board_*`, `*_admission_*`, `*_policy_token_*`, `test_entry_preflight_stage_*`, `test_ir4_schema_names_*`), `tests/test_harness_gateway.py::test_gateway_mirrors_counters_*`, `tests/test_rl_ir_harness.py` (INFRA side).
