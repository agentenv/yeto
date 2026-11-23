# Breakers: reference

A breaker is anything in yeto that automatically stops a run or a process,
pauses production, refuses an order or a start, or drops data. Every breaker is
declared once in the registry (`yeto/breakers/`). This page is the reference.
The customer guide explains how to use it:
[KIMI_K31_ASYNC_RL_GUIDE.md, section 7](KIMI_K31_ASYNC_RL_GUIDE.md#7-breakers-and-cost-protection).

## Kinds

| Kind | Meaning | Configurable in `breakers:` |
|---|---|---|
| `breaker` | Stops a component or the run at a limit. | yes |
| `guard` | Drops or cuts part of the data. The run goes on. | yes |
| `retry_budget` | A retry count or time. When it is spent, data is dropped or a component stops. | yes ("off" means no retries) |
| `alert` | Alerts only. | yes |
| `invariant` | A data-correctness check. | no: shown, never disabled |
| `semantic` | A training parameter the dependency rules read. | no: set in its usual place |

## Layers

| Layer | Runs in |
|---|---|
| `run` | the `yeto run up` monitor |
| `launch` | the host that runs `yeto launch` (cost guard, fleet) |
| `trainer` | the trainer island (buffer reader, driver) |
| `sampler` | the sampler process (engine sampler, agentic round, dispenser) |
| `harness` | the agentic harness inside the sampler (Harbor, sandbox providers) |
| `pool` | the rollout pool controller |
| `monitor` | read-only checks (`yeto breaker check`, dashboard alerts) |

## Where a value comes from

For each parameter, in order:

1. the `breakers:` section (`breakers.pools.<name>` first for a pool entry);
2. a legacy knob (a flag, a run config key, a spec field, a pool field, an
   environment variable), in the order the registry declares them;
3. the parent value for an inherited parameter (`run.budget.usd`), only when
   the parent was set in the `breakers:` section;
4. the registry default.

Two sources that disagree are a conflict. The config is refused, and both
locations are named. Two sources that agree are fine; `show` marks the second
one as redundant.

`yeto breaker show --config run.json` prints the result. `--json` prints the
`breakers.resolved.json` format, which `yeto run up` also saves in its state
directory.

## Events

Every trip writes one `rl_breaker` line to the component's existing event tape
(trainer `YETO_RL_EVENT` lines, the sampler tape, the pool tape,
`cost-events.jsonl`, and `events.jsonl` in the `run up` state directory). The
old events stay unchanged; `rl_breaker` is an extra line.

```json
{"event": "rl_breaker", "schema": "yeto-breaker-v1",
 "name": "trainer.sample_wait", "layer": "trainer", "kind": "breaker",
 "state": "trip", "action": "stop",
 "value": 3601.2, "limit": 3600, "unit": "s", "direction": "max",
 "scope": {"island": "0"}, "count": 1,
 "detail": "rollout 37: 3/8 groups after 3600 s without a new group",
 "legacy_event": "SampleBufferRolloutError", "config_source": "trainer.argv --rl-sample-wait-s",
 "stop_reason": "breaker:trainer.sample_wait", "time_unix": 1795400000.0}
```

`state` is `warn`, `trip`, `clear`, `shadow` (recorded only, for `action: warn`
on an entry that would otherwise act) or `config_mismatch`.

Components also send `rl_breaker_status` snapshots with the current value and
the limit of each entry. The dashboard's Breakers panel reads them.

Exit codes do not change. A breaker that stops the run adds
`stop_reason=breaker:<name>` to the launcher's last line, the run record and
the `run up` `state.json`.

## Golden lock

`tests/golden/breakers` records the effective value of every breaker for every
example config and for edge configs (no budget, `--no-cost-guard`, environment
overrides, legacy off spellings). The test asserts that today's code and the
registry both give those values, so a config without `breakers:` keeps its
behavior.

```bash
PYTHONPATH=. python -m pytest tests/golden/breakers -q
```

Regenerate `golden.json` only for a new case or a key rename
(`python -m tests.golden.breakers.capture --write`), never to accept a changed
value.

## Registry

The tables below are generated from the registry by
`python3 scripts/gen_breaker_tables.py` (the same output as
`yeto breaker list --markdown`). `tests/test_breakers_docs.py` fails when they
drift. "Legacy knobs" are the old spellings. They are deprecated from 0.2.0
and print a deprecation note; they will be removed in 0.4.0, two minor releases
later (see [CHANGELOG.md](../CHANGELOG.md)).

<!-- BEGIN GENERATED: scripts/gen_breaker_tables.py -->

### run

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `run.budget` | breaker | usd=null warn_ratio=0.8 force_pools_ratio=1.2 release_pools_on_trip=false | stop | yes | run config budget.usd, run config budget.release_nc_on_breach | stops the trainer when the estimated spend passes the cap; the pools stay unless release_pools_on_trip |
| `run.wall_clock` | breaker | hours=3 | stop | yes | run config budget.wall_h | stops the trainer after this many hours of one run up; on when not written (3 h). Its pool release and forced pool-stop follow run.budget.release_pools_on_trip / force_pools_ratio |
| `run.trainer_silence` | breaker | limit_s=null floor_s=1800 stall_factor_n=4 | stop | yes | run config budget.silence_min | stops the trainer when its launcher log and progress paths stop growing while the launcher runs |
| `run.first_group` | breaker | timeout_s=5400 | stop | yes | run config first_group.timeout_s | ends run up (exit 1, the pools kept) when no group reaches the store in time |
| `run.starvation` | breaker | alert_s=1800 stop_s=7200 unhealthy_s=600 trainer_wait_max_s=7200 stop_grace_s=1200 check_s=300 | stop | yes | run config starvation.alert_min, run config starvation.stop_min, run config starvation.unhealthy_min, run config starvation.trainer_wait_max_min, run config starvation.stop_grace_min, run config starvation.check_s | the trainer runs but no new group reaches the buffer: alert, then stop through the STOP flag when no sampler is healthy |
| `run.rollout_stall_alert` | alert | limit_s=1800 | warn | yes | run config rollout.stall_alert_s | the in-process controller logs a rollout stall (controller: in_process only) |
| `run.sandbox_containers` | alert | max_run_n=null max_per_sampler_n=null | warn | yes | run config sandbox.max_run_containers, run config sandbox.max_containers_per_sampler | a warning line when the run's sandbox containers pass the larger of the two limits |
| `run.release_stall` | breaker | limit_s=null check_s=300 stop_grace_s=1200 | warn | yes | - | after the first release, no new release for the limit: an alert (stop: the STOP flag); a trainer restart starts the clock again |
| `run.trainer_exit` | invariant | - | stop | no | - | a launcher that exits without a result, or finishes FAILED / non-zero, is a breach |
| `run.controller_exit` | invariant | - | stop | no | - | the in-process rollout controller stopped driving the pools: a breach |

### launch

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `launch.budget` | breaker | usd=null warn_usd=null stop_grace_s=1200 | stop | yes | trainer.argv --budget-usd, trainer.argv --budget-warn-usd, trainer.argv --budget-stop-grace-minutes | the launch's spend (its resource ledger, priced per resource); with a breakers: section that sets run.budget.usd it inherits that value unless written here |
| `launch.idle` | breaker | warn_s=900 stop_after_warn_s=900 load_grace_s=null max_load_s=7200 | stop | yes | trainer.argv --idle-warn-minutes, trainer.argv --idle-stop-minutes, trainer.argv --load-grace-minutes, trainer.argv --max-load-minutes | no progress event (or 0 % GPU on every reporting GPU) outside a loading phase |
| `launch.nc_lease_overdue` | alert | grace_s=300 | warn | yes | trainer.argv --nc-lease-overdue-minutes | alert only: the NC pool owns its leases; off with the cost guard |
| `launch.stall` | breaker | timeout_s=900 | stop | yes | trainer.argv --rl-stall-timeout | RunStalled, exit 6; the launcher tears the run down |
| `launch.trainer_failures` | retry_budget | max_n=3 | stop | no | trainer.argv --rl-trainer-max-failures | the trainer scheduler releases the trainer and stops the run (exit 8) |
| `launch.provider_timeout` | breaker | timeout_s=86400 retries_n=10 | stop | no | trainer.argv --modal-timeout-s, trainer.argv --modal-retries | enforced by the provider, not by yeto: no rl_breaker event |
| `launch.teardown_verify` | invariant | - | stop | no | - | after teardown every Modal app row of the run is stopped with 0 tasks (else exit 5) and every sky cluster is confirmed gone (else the head stays up) |

### trainer

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `trainer.sample_wait` | breaker | timeout_s=3600 | stop | no | trainer.argv --rl-sample-wait-s, trainer env YETO_SAMPLE_BUFFER_WAIT_S | the buffer reader ends the run when no new group arrives for timeout_s while a round fills |
| `trainer.policy_age` | guard | max_versions=null | drop | yes | trainer.argv --rl-max-policy-age, trainer env YETO_SAMPLE_BUFFER_MAX_POLICY_AGE | groups older than max_versions are consumed, not trained (0 / unset: every sample from the policy just published, no buffer trainer) |
| `trainer.nan` | breaker | skip=false skip_max_consecutive_n=2 skip_max_fraction=0.05 skip_min_rounds_n=20 | stop | no | trainer env YETO_RL_NAN_SKIP, trainer env YETO_RL_NAN_SKIP, trainer env YETO_RL_NAN_SKIP, trainer env YETO_RL_NAN_SKIP | a non-finite grad_norm stops the trainer (strict); with skip, the round is skipped and the trainer stops past the skip limits (nan_skip_breaker) |
| `trainer.xhw_guard` | guard | warn_abs_diff=0.1 act_abs_diff=0.15 warn_kl=0.03 act_kl=0.06 warn_clip=0.01 act_clip=0.03 window_n=3 min_rounds_n=2 quarantine_rounds_n=8 weight=0.5 max_cond=20 confidence_z=2 | quarantine | yes | trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD, trainer env YETO_RL_XHW_GUARD | per-source train/rollout log-prob mismatch: warn, quarantine or downweight a source, or stop |
| `trainer.buffer_damaged` | breaker | max_run_n=8 | stop | yes | trainer env YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN | a systematic payload fault stops the take instead of consuming the backlog |
| `trainer.release_drain` | breaker | timeout_s=1800 | stop | no | trainer env YETO_TRAINER_RELEASE_DRAIN_S | a commanded island stops without the upload still running after timeout_s (the samplers keep the previous release) |
| `trainer.group_integrity` | invariant | - | drop | no | - | a group with fewer than 2 or more than n trials, or a non-finite reward or log-prob, is consumed, not trained |
| `trainer.sample_alignment` | invariant | - | drop | no | - | a group whose samples do not convert to the trainer's form (R3 routing rows, segment layout) leaves the round; takes in a row without a convertible group stop the trainer |
| `trainer.compaction_round` | invariant | - | stop | no | - | a round of compacted trials the trainer cannot train whole (DP schedule, --seq-length, segment siblings) is refused |
| `trainer.nonfinite_reward` | invariant | - | stop | no | - | a batch with a non-finite group reward is refused before training |
| `trainer.gradient_flow` | invariant | - | stop | no | - | grad_norm 0 while the round's advantages (or a mechanism) require a gradient |
| `trainer.policy_hash` | invariant | - | stop | no | - | the trainer's policy differs from the cut it applied or was rebuilt from |
| `trainer.strict_invariant` | invariant | - | stop | no | - | any other strict trainer check (layout hash, LR 0 before the final round, rollout statistics) ends the trainer abnormally |

### sampler

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `sampler.load_failures` | breaker | max_consecutive_n=3 | stop | no | spec max_consecutive_load_failures | the sampler exits (SamplerLoadFatal) after this many failed weight loads in a row |
| `sampler.store_read_outage` | breaker | limit_s=1800 max_damaged_latest_n=120 | stop | no | spec max_read_outage_s, spec max_damaged_latest_reads | the sampler exits (StoreReadOutage) when the store has not answered a poll for limit_s; null: no bound |
| `sampler.write_defer` | guard | limit_s=600 | drop | no | spec write_defer_max_s | past the limit a waiting group is given up and its task released (not counted as a redo) |
| `sampler.incarnation_lease` | guard | max_unread_s=60 | pause | yes | spec incarnation_lease, spec incarnation_lease.max_unread_s | group writes wait while the lease is unread for longer than max_unread_s (null: never); off (incarnation_lease: false) also turns the incarnation fence off |
| `sampler.drain` | guard | timeout_s=null reload_max_defer_s=600 abandon_grace_s=120 | drop | no | spec agentic.drain_timeout_s, spec reload_max_defer_s, spec agentic.abandon_grace_s | a drain-mode weight switch cancels the trials still running at min(timeout_s, reload_max_defer_s) (agentic_round.effective_drain_timeout); both null: no cut |
| `sampler.min_samples` | guard | min_n=2 | drop | no | spec agentic.min_samples | a group with fewer kept trials is not written; its task goes out again (counted as a redo) |
| `sampler.infra_retries` | retry_budget | max_n=2 | drop | no | spec agentic.infra_retries | per trial slot, at once (no backoff); 0: no retry. A spent budget loses the trial |
| `sampler.trial_wall_clock` | guard | limit_s=7200 | drop | no | spec agentic.task_defaults.wall_clock_s | a trial past it ends ABORTED (wall_clock_cap), dropped and not retried; a task row's own wall_clock_s overrides it for that task |
| `sampler.dispense_redo` | retry_budget | max_n=2 | drop | no | spec prompts.dispense.max_redo, pool sampler.dispense max_redo, pool argv --dispense-max-redo | the first sampler's value is kept in plan.json and every later sampler adopts it |
| `sampler.dispense_lease` | guard | lease_s=900 | drop | no | spec prompts.dispense.lease_s, pool sampler.dispense lease_s, pool argv --dispense-lease-s | a claim not renewed for lease_s ends expired (counted as a redo) and is dispensed again. Each sampler uses its own spec's lease (plan.json keeps the first one for reference); a row's metadata.dispense_lease_s overrides it for that task |
| `sampler.vram_guard` | alert | high_frac_ratio=0.97 realert_s=600 interval_s=null | warn | no | spec vram_guard.high_frac, spec vram_guard.realert_s, spec vram_guard.interval_s | rl_sampler_vram_high / rl_sampler_vram_ok on the tape of a card sampler |
| `sampler.infra_streak` | breaker | max_n=5 cooldown_s=300 | warn | yes | spec agentic.infra_streak.enabled, spec agentic.infra_streak.max_n, spec agentic.infra_streak.cooldown_s, spec agentic.infra_streak.action | warn: rl_breaker only. pause: the store dispenser claims no new task (trials in flight go on) until a group is written or cooldown_s has passed |
| `sampler.loop_guard` | guard | max_period_n=64 repeats_n=8 | drop | yes | - | cuts a looping completion (finish loop_abort); off by default, set only by code (EngineSamplerConfig.loop_guard); thresholds not measured |
| `sampler.token_alignment` | invariant | - | - | no | - | a trial whose recorded calls do not align with its tokens ends ABORTED (token_alignment) |
| `sampler.compaction_startup` | invariant | - | - | no | - | a compaction config whose budgets do not fit max_model_len refuses the sampler's start |
| `sampler.max_prompt_len` | semantic | - | - | yes | - | a first prompt longer than this ends the trial ABORTED |
| `sampler.agent_timeout` | semantic | - | - | yes | - | a trial past it ends POLICY_LIMIT and is scored (kept) |
| `sampler.dispense_passes` | semantic | - | - | yes | - | passes over the data |
| `sampler.max_backlog_groups` | semantic | groups_n=null | - | yes | spec max_backlog_groups | unread groups at which the sampler pauses generating (rl_sampler_backlog_full); "auto": derived by yeto run up from the release cadence (sample_roles) |
| `sampler.max_compactions` | semantic | - | - | yes | - | compactions of one trial before ContextFull (POLICY_LIMIT, kept) |

### harness

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `harness.setup_retries` | retry_budget | max_n=0 backoff_initial_s=5 backoff_max_s=60 | - | yes | env HARBOR_SETUP_RETRIES (deprecated), env HARBOR_SETUP_BACKOFF_S, env HARBOR_SETUP_BACKOFF_MAX_S | sandbox start, image pull and agent install failures are retried on a new sandbox; once spent the trial is ABORTED (infra) |
| `harness.reach_check` | guard | attempts_n=15 timeout_s=5 interval_s=2 | drop | yes | env YETO_AGENTENV_SKIP_REACH_CHECK (deprecated), env YETO_AGENTENV_REACH_TRIES (deprecated), env YETO_AGENTENV_REACH_TIMEOUT_S (deprecated), env YETO_AGENTENV_REACH_INTERVAL_S | an AgentENV microVM that cannot reach its judge / records endpoints fails its start (NetworkError, counted as infra) instead of scoring a silent zero |
| `harness.sandbox_caps` | breaker | max_sandboxes=null agentenv_max_active_n=64 agentenv_starts_per_minute_n=60 agentenv_starts_per_hour_n=null agentenv_create_timeout_s=120 boat_max_active_n=100 boat_starts_per_minute_n=12 boat_starts_per_hour_n=null | pause | no | env HARBOR_MAX_SANDBOXES (deprecated), env YETO_AGENTENV_MAX_ACTIVE (deprecated), env YETO_AGENTENV_STARTS_PER_MINUTE (deprecated), env YETO_AGENTENV_STARTS_PER_HOUR (deprecated), env YETO_AGENTENV_REQUEST_TIMEOUT_S (deprecated), env YETO_BOAT_MAX_ACTIVE (deprecated), env YETO_BOAT_STARTS_PER_MINUTE (deprecated), env YETO_BOAT_STARTS_PER_HOUR (deprecated) | a sampler whose max_in_flight exceeds a cap refuses to start; at run time a create above a cap or start rate waits |
| `harness.sandbox_ttl` | breaker | limit_s=null margin_s=900 default_s=10800 | - | no | env HARBOR_SANDBOX_TTL_S, env HARBOR_SANDBOX_TTL_MARGIN_S, env HARBOR_SANDBOX_TTL_DEFAULT_S | the provider reclaims a sandbox after its TTL: the backstop for a lost delete (a killed sampler); HARBOR_ENV_KWARGS ttl_s still wins over it |
| `harness.turn_max_tokens` | semantic | - | - | yes | - | the records proxy's signed max_tokens of one model call; a longer reply is cut (finish length) and the agent goes on |
| `harness.agent_call` | semantic | - | - | yes | - | pi's own per-call idle timeout (1800 s, 3600 s with proxy compaction) and model-call retries (8); a retried call forks the token chain |

### pool

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `pool.provision_breaker` | breaker | failures_n=3 cooldown_s=600 max_cooldown_s=3600 max_trips_n=3 | halt | no | pool elastic breaker_failures, pool elastic breaker_cooldown_s, pool elastic breaker_max_cooldown_s, pool elastic breaker_max_trips | an order that fails, a ticket lost before it is granted or a node whose sampler never comes up counts a failure; each trip is a cooldown (warn), the last one halts ordering (trip) |
| `pool.spend_cap` | breaker | usd=null ucr=null | pause | yes | pool elastic max_run_spend_usd, pool elastic max_run_spend_ucr | refuses a new order when the spend plus one node hour passes the cap (the nodes up keep running) |
| `pool.node_cap` | breaker | max_nodes_n=null max_orders_n=null | pause | no | pool elastic max_nodes, pool elastic max_orders | refuses a new order past the node or order cap |
| `pool.crash_loop` | breaker | - | replace | no | pool elastic crash_loop_before_ready | a sampler that crashed past pool.sampler_restarts on a reachable node; one that never joined: replace the node, or halt the pool like a configuration error (nothing is ordered until pool-retry / pool-stop; a node that joined before is always replaced) |
| `pool.sampler_restarts` | retry_budget | max_n=2 reset_s=1800 | replace | no | pool elastic max_sampler_restarts, pool elastic sampler_restart_reset_s |  |
| `pool.engine_stall` | breaker | limit_s=600 max_restart_failures_n=3 | replace | yes | pool elastic engine_stall_s, pool elastic max_stall_restart_failures | fences and restarts a stalled engine in place, then retires the node |
| `pool.sampler_stall` | breaker | limit_s=null | replace | yes | pool elastic stall_s | the sampler counts as crashed |
| `pool.start_timeout` | breaker | limit_s=3600 | replace | no | pool elastic start_timeout_s | a provisioning failure: the node is replaced and the provision breaker counts it |
| `pool.unreachable` | breaker | grace_s=300 | replace | no | pool elastic unreachable_grace_s |  |
| `pool.fence_hold` | breaker | limit_s=7200 | - | yes | pool elastic fence_max_hold_s | releases the node (the fence is still retried) |
| `pool.lease_overdue` | breaker | grace_s=0 | replace | no | pool elastic overdue_grace_s | a lease-bounded provider only; deletes this pool's own ticket |
| `pool.control_plane_alert` | alert | limit_s=600 | warn | no | pool elastic control_plane_alert_s |  |
| `pool.fence_alert` | alert | failures_n=5 limit_s=600 | warn | no | pool elastic fence_alert_failures, pool elastic fence_alert_s | a shard fence that keeps failing (it is retried every tick) |
| `pool.lease_end` | breaker | replace_lead_s=3600 end_margin_s=600 | replace | no | pool config replace_lead_s, pool config end_margin_s | lease-bounded providers only (National Compute); a provider without leases never drains for it |

### monitor

| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |
|---|---|---|---|---|---|---|
| `monitor.dashboard_heartbeat` | alert | heartbeat_warn_s=60 heartbeat_severe_s=300 | warn | yes | - | an island's last heartbeat (or event) is older than this |
| `monitor.dashboard_startup` | alert | startup_warn_s=1200 startup_severe_s=1800 startup_init_warn_s=2700 startup_init_severe_s=3600 startup_step_warn_s=900 | warn | yes | - | an island still starting writes no event, or one startup step is slow |
| `monitor.dashboard_missed` | alert | missed_consecutive_warn=2 missed_consecutive_severe=5 | warn | yes | - | an island missed this many synchronous rounds in a row |
| `monitor.dashboard_quorum` | alert | quorum_recent_rounds=7 quorum_min_rounds=10 quorum_factor=3 | warn | yes | - | the recent median quorum latency is this many times the baseline |
| `monitor.dashboard_grad_norm_spike` | alert | grad_norm_window=8 grad_norm_spike_factor=5 | warn | yes | - | grad_norm above this many times the median of the previous window (NaN / Inf: critical) |
| `monitor.dashboard_clip_kl` | alert | clipfrac_info=0.2 kl_warn=0.1 | warn | yes | - | the latest clip fraction (info) or KL (warning) above its line |
| `monitor.dashboard_budget` | alert | budget_warn_pct=60 budget_severe_pct=80 | warn | yes | - | the cost estimate against the dashboard's --budget |
| `monitor.dashboard_sft_reference` | semantic | sft_ref_step_s=0 sft_hbm_limit_gib=0 sft_syncer_rss_limit_bytes=0 sft_slow_factor=1.1 sft_stop_factor=1.5 sft_stop_slack_s=300 | - | yes | - | reference lines and the stopped status of the SFT page; no alert |
| `monitor.dashboard_compaction` | alert | compaction_window_rounds=5 compaction_min_trials=5 compaction_summary_ok_warn=0.9 compaction_summary_ok_severe=0.5 compaction_truncated_warn=0.2 compaction_frac_info=0.5 compaction_long_loss_gap=0.2 compaction_fork_lost_warn=0.05 | warn | yes | - | context compaction of agentic trials: failing summaries, truncation, lost long trials |
| `monitor.dashboard_trial_fail` | alert | trial_fail_window_rounds=10 trial_fail_min_trials=8 trial_fail_warn=0.5 | warn | yes | - | agentic trials ending on infrastructure |
| `monitor.dashboard_trainer_missing` | alert | trainer_missing_warn_s=1200 trainer_missing_severe_s=1800 | warn | yes | - | samplers running without a trainer |
| `monitor.dashboard_trainer_failed` | alert | trainer_fail_window_s=3600 trainer_fail_loop=2 | warn | yes | - | failed trainer launches in the window |
| `monitor.dashboard_stale_read_position` | alert | stale_read_recent_s=300 | warn | yes | - | a sampler waits on a read position the trainers have moved past |
| `monitor.dashboard_release_stalled` | alert | release_stall_warn_s=600 | warn | yes | - | a policy release stuck uploading (critical when samplers wait on it) |
| `monitor.dashboard_guard_drops` | alert | guard_drop_window_rounds=10 guard_drop_min_groups=20 guard_drop_min_rounds=3 guard_drop_warn_ratio=0.1 guard_drop_alert_ratio=0.25 | warn | yes | - | share of the groups a data-dropping guard dropped over its recent rounds |
| `monitor.health_logprob` | alert | amd_warn_abs_diff=0.1 amd_act_abs_diff=0.15 nvidia_warn_abs_diff=0.04 nvidia_act_abs_diff=0.052 ascend_warn_abs_diff=0.05 ascend_act_abs_diff=0.075 | warn | yes | - | train/rollout log-prob mismatch per sampler hardware class (amd: the in-run xhw guard's defaults) |
| `monitor.health_is_clipped` | alert | max_ratio=0.05 | warn | yes | - | sc_is_clipped, mean over the window |
| `monitor.health_is_weight` | alert | min_ratio=0.9 | warn | yes | - | sc_importance_weight, mean over the window |
| `monitor.health_stale` | alert | max_ratio=0.3 | warn | yes | - | stale groups / (stale + trained) |
| `monitor.health_flips` | alert | max_ratio=0.01 | warn | yes | - | changed / total weights of the last release (--flips) |
| `monitor.health_aborted` | alert | max_ratio=0.2 | warn | yes | - | agentic trials lost / trials |
| `monitor.health_zero_call` | alert | max_ratio=0.05 | warn | yes | - | agentic trials retried before any model call / trials |
| `monitor.health_nan_skip` | alert | max_consecutive_n=2 max_fraction=0.05 min_rounds_n=20 | warn | yes | - | NaN-skipped rounds: in a row, and as a share |

### switch groups

| Group | Members | Legacy switch |
|---|---|---|
| `cost_guard` | launch.budget, launch.idle, launch.nc_lease_overdue | trainer.argv --no-cost-guard |

<!-- END GENERATED -->
