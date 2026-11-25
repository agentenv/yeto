# RL fleet dashboard (`yeto dashboard`)

A read-only dashboard. It reduces the learner tapes, the syncer tape, the
controller journal and the head-side `fleet.jsonl` into views with one reducer
(`yeto/dashboard/reducer.py`), and has two outputs:

- **Live**: `yeto dashboard serve`, a stdlib HTTP server that binds only to
  127.0.0.1, accepts only GET, and is reached over an SSH tunnel.
- **Offline**: `yeto dashboard export`, one self-contained HTML file (the view
  JSON is inlined; no fonts or CDN are loaded and opening it makes no network
  requests).

Layout: alert cards (critical / warning / info; clicking one jumps to the
island, round or metric) -> per-island overlaid training curves (12 metric tabs)
-> syncer round table (derived, with an "only anomalies" filter) -> island
health cards (clicking one expands the island detail on the same page: progress
and policy, cell table, reconfiguration transactions, recent events, Ray embed)
-> cost-efficiency table. The page has no action that changes run state; the
only action is "copy CLI command".

W&B remains the curve archive; it is not a data source of the dashboard.

## Live: serve on the head, view through a tunnel

```bash
# on the head (the run name is the --cluster-prefix)
yeto dashboard serve --run <run> [--tapes <extra tapes or directories>...] [--port 8787] [--budget 300] [--head <head-host>]
# prints: ssh -N -L 8787:127.0.0.1:8787 <head-host>  then open http://127.0.0.1:8787/

# on your laptop
ssh -N -L 8787:127.0.0.1:8787 <head-host>
# open http://127.0.0.1:8787/ ; the page polls /api/overview every 5 s
```

`--run <run>` follows `~/.yeto/runs/<run>/events/*.jsonl` (the reassembled
learner tapes echoed back), `~/.yeto/runs/<run>/fleet.jsonl`, and the syncer tape
on the head, `~/yeto-output/yeto-tape.jsonl`, when it exists. `--tapes` adds
files or directories (a directory is searched two levels deep for `*.jsonl`;
when a directory holds exactly one `rl-island-N.jsonl`, its `journal.jsonl` is
attributed to island N). Every `--poll` seconds (default 2) the server reads
each file incrementally by byte offset and consumes only complete lines;
re-reading the same offset never counts twice. Offsets live in memory only and
are independent of the W&B forwarder's `OffsetStore`.

`--host` accepts only loopback addresses (127.0.0.1 / ::1 / localhost);
`--host 0.0.0.0` refuses to start. 127.0.0.1 is still reachable by other users
on the same machine: the tool assumes a single-user head and does no
authentication.

Endpoints (all GET; any other method gets 405):

| Endpoint | Content |
|---|---|
| `/` | The page (the same static assets as the export) |
| `/api/overview` | Run, global state, alerts, cost summary, island cards, overlaid curve series, round markers |
| `/api/islands/<id>` | Island detail: card, metric series, resource samples, cell table, reconfiguration transactions, recent events, `ray_embed` |
| `/api/rounds?only_bad=1` | Derived syncer round table |
| `/api/fleet` | `fleet.jsonl` lifecycle and `cost_tick`, price table source, cost view |
| `/api/events?island=&type=&after=&limit=` | Paged events (`after` is the previous page's `cursor`) |

## Run store: tapes uploaded by the islands (`--store`)

```bash
yeto dashboard serve --run <run> --store s3://<bucket>/runs/<run> --port 8787 [--store-poll 10] \
    [--up-state ~/.yeto/rl-up/<run>/state.json] [--budget 50]
yeto dashboard export --store s3://<bucket>/runs/<run> -o run.html
```

`--store` (an `s3://` URI or a directory) reads every island's
`tapes/<island>/<incarnation>/<seq>.jsonl` chunk, the tapes the samplers and the
trainer upload every `YETO_RL_TAPE_UPLOAD_S` seconds: islands on leased nodes and
replacement nodes appear without a per-node mirror. The store is polled every
`--store-poll` seconds (default 10). Credentials come from the environment, as for
the launcher (`YETO_RL_STORE_S3_*`, else `AWS_PROFILE` / the AWS variables).
`--up-state` adds the cumulative runtime and spend of a `yeto run up` run; the file
is read on every view and may appear later.

A failing store read is shown, not hidden: `/api/overview` and `/api/view` carry
`overview.store` (`sources`, `requests`, `errors`, `failing`, `last_error`,
`last_ok_at`); while `failing` is non-zero the page raises the warning "run store
unreadable" and, with no island yet, the global status says so instead of "no
data". The "Islands now" note shows the tapes read and the last clean read.

One record often reaches the dashboard more than once: the launcher's echoed tape
(`~/.yeto/runs/<run>/events`) and the trainer's store upload carry the same records,
and a restarted trainer uploads its whole local tape again under its new launch. A
timestamped event equal to one already folded for the same island is dropped
(`overview.duplicates` counts them), so counts are not inflated and a restart is
dated by its own first record.

A dashboard may start before the run: the run's own paths under `--run`
(`~/.yeto/runs/<run>/events`, `fleet.jsonl`) and the `--tapes` paths are followed
once they appear. `export` still refuses when none of its paths exists and no
`--store` is given. The page's "Commands" box repeats this dashboard's own
arguments (port, store, tapes ...), so a copied command reproduces it.

## RL metrics page: fleet charts

- **Window**: the header's `run` button (the RL default) spans the run itself, first event to
  now; `1h` ... `30d` are fixed windows. The choice is remembered per browser.
- **Marks**: restarts, relaunches and NC pool events are small triangles in a strip above every
  fleet chart, one per island color; marks within 2 minutes merge into one (the count is shown)
  and their labels are in the hover tooltip only.
- **Line breaks**: a line breaks where two points are more than 3x the series' median spacing
  apart, or where the sampler incarnation changes; node memory charts follow the same rule.
- **Samplers**: up to two samplers are drawn one line each. With more, a chart draws one
  aggregate (reward: the group-weighted mean of the last 8 x N groups over all samplers; round
  time, response length, HBM: p50 with a p10-p90 band) and each sampler's line opens from the
  legend toggle. The sampler reward is a group-weighted rolling mean of its last 8 groups; the
  tooltip names the round's own mean.
- **Incarnations** are read from the tape that carried a record (a run store lists incarnation
  folders by name, not by time). An incarnation starts at its engine start (before its join).
- **Card time** of a sampler outside `fleet.jsonl` is the union of its incarnations' first ->
  last event: a lower bound, since boot before the first event is not on the tapes.
- **Hot reload**: each release's load lists its parts (resolve, the drain of in-flight trials it
  waited for from `rl_agentic_drain`, load, pause) in Islands now and the release table.
- **Trainer waiting for data** includes the round still waiting (Islands now shows it as
  "current round N waiting for data").
- **Zero-variance groups**: from samplers' drops when they report them, else the trainer's
  `rl_local_round.zero_variance_group_ratio` x the round's trained groups (trained with zero
  advantage, not dropped). **Policy age** of trained groups: `rl_round_trained.policy_age_groups`.
- **Round tooltip, async buffer mode**: a sampler row lists what the trainer took from it in that
  round (groups, generation versions and age, mean reward), matched by the incarnation prefix of
  `rl_trajectory_reward.trajectory_id`; samplers keep their own round numbers, so there is no
  "missed" row (that and the consecutive-missed alert belong to synchronous multi-island runs).
- **Task dispense** (Async GRPO section): the slot ledger as the samplers' `rl_dispense_*` events
  show it: slots settled by state, attempts ended without settling (expired lease, released,
  requeued, lost), re-claimed slots, and every claim still held with its holder. A holder whose
  island is stale / lost / stopped is flagged: its claims come back only when their lease expires
  and another sampler claims them again (attempt + 1). The lease itself is not on the tapes.

## RL metrics page: details

Below the fleet charts, the details follow an island switch: a fleet of async islands shows
`Island 0 / Island 1 / ... / Fleet`, an async run with several trainers one button per trainer,
a synchronous multi-island run one per island; a run with one trainer has no switch. The pick is
kept per browser. For the picked island (async):

- the island header: trainer, machine, store lane, release cadence, max policy age, policy
  version, startup steps, status;
- Trainer rounds (totals, sparklines, newest rows) next to Releases: the island's releases only,
  one load tag per sampler of the island (`vLLM 157s`, `SGL 1 at start`; a grey dashed tag = not
  loaded; hover a tag for resolve / drain / load / pause);
- Samplers & pools: one row per sampler, grouped by pool (source, engine and cards, status,
  version, policy age, groups, truncation, backlog cap, the pool's logprob |diff| / TIS
  clipfrac); a column no sampler reports is hidden and named under the table;
- Backlog (with the samplers' `backlog_full` / `admitted` reports) next to the release timeline;
- the per-round drill-down of the island's trainer (metric, round time, node memory, phase
  timeline from the first round; the startup before it is a note).

`Fleet` holds one summary row per island, every trainer's metric overlaid, and the run-wide
ledgers: syncer merges, the task dispense ledger (recover / expired events are listed; slots
handed back unwritten on a reload are counted only) and NC pool events. Island health is drawn for
synchronous runs; an async run carries it in the island header and the sampler table.

## Offline export

```bash
yeto dashboard export --run <run> -o run.html
yeto dashboard export --tapes <pulled tape directory> -o run.html --budget 1000
```

The header shows "offline export, generated at ...". The offline page takes
"now" from the newest timestamp in the tapes (not the time it is opened), so
heartbeat ages of a finished run do not grow over time. For the same tapes, the
overview inlined by the export equals `/api/overview` served with the offline
clock (`tests/test_dashboard_serve.py`). The live server computes ages from the
wall clock. In the export, the Ray iframe shows only a placeholder and the
tunnel command.

## ssh_harness runs: mirroring remote tapes

```bash
yeto dashboard mirror --target user@learner-host --remote ~/yeto-output/rl-island-0.jsonl \
    --local ~/dash/<run>/rl-island-0.jsonl --island 0 &
yeto dashboard serve --tapes ~/dash/<run>
```

The mirror runs `ssh <target> tail -c +<mirrored bytes + 1> -F <remote>`. After a
disconnect it reconnects with exponential backoff (from 1 s up to 30 s) and
continues from the end of the local mirror, with no loss and no duplicates.
Disconnects and recoveries go to the side file `<mirror>.source.jsonl`
(`dashboard_source_lost` / `dashboard_source_restored`), from which the dashboard
raises a "tape source lost" alert for that island. Modal islands keep using the
`TapeCollector` echo and need no mirror.

## Cost (an estimate, not a bill)

Cost = price-table rate ($/GPU-hour) x GPU count x wall clock (from
`island_ready` to `island_lost` / `island_stop` / now; restarts accumulate
again). The default price table is yeto's unified price table
(`yeto/cost/prices.json`: list prices, each row with its source and date,
**not billing data**); replace it with `--prices my.json` (a JSON object with
`prices` of `cloud:GPU` -> $/GPU-hour, `*:GPU` is a wildcard, and optional
`host_prices`). An island
missing from the table shows "unpriced" rather than $0; local islands count as
$0 and are left out of the efficiency ranking. With `--budget`, reaching 60% /
80% of the budget raises a warning / critical alert, and the projected time to
reach the budget is shown. A run without `fleet.jsonl` (older runs, or runs not
started through the FleetController) shows "no data" in the cost panel.

The head-side `fleet.jsonl` is written by `FleetController`
(`runs/<run>/fleet.jsonl`): `island_ready` (first start and restarts),
`island_lost`, `island_stop`, and one `cost_tick` every 5 minutes. The price table
and the budget can be passed to the launcher with the environment variables
`YETO_DASHBOARD_PRICES` / `YETO_DASHBOARD_BUDGET_USD`.

## Alert rules and thresholds

| Rule | Default threshold | Level |
|---|---|---|
| Heartbeat / last event age | >60 s warning, >300 s critical | 1/0 |
| Island still starting (no `rl_driver_start` yet) | >1200 s warning, >1800 s critical | 1/0 |
| One startup step (Ray up / engines ready / weights loaded) stuck while heartbeats arrive | >900 s | 1 |
| Consecutive missed rounds | >=2 warning, >=5 critical | 1/0 |
| Quorum latency | median of the last 7 rounds >= 3x baseline (after >=10 rounds) | 1 |
| grad_norm spike / NaN | > 5x the median of the previous 8 rounds; NaN/Inf critical | 1/0 |
| clipfrac / KL | clip > 0.2 info; KL > 0.1 warning | 2/1 |
| Budget | 60% / 80% | 1/0 |
| RECOVERY_REQUIRED (journal phase or tape `rl_reconfiguration`) | — | 0 |
| ssh mirror source lost | — | 1 |
| Run store unreadable (`--store` reads failing since the last clean poll) | — | 1 |
| A trainer skipping releases while the previous upload is in flight (`rl_policy_release_skipped` upload_in_flight) with no `rl_policy_released` since, for over 10 min | samplers waiting on a stale read position: critical, else warning | 0/1 |
| A sampler waiting on a full backlog (as it counts it, reported in the last 5 min) while the trainers' read position leaves less than its cap: it learns the read position only from a policy release or the trainer's read heartbeat | — | 1 |
| Trainer launches ending in a failure leave (`island_leave` not STOPPED / DONE ...) in the last hour; the detail names the root-cause error line | 1: warning, >= 2: critical | 1/0 |
| Samplers up without any trainer island on the tapes (async run; since the first sampler event) | > 1200 s warning, > 1800 s critical | 1/0 |
| Agentic sampler trials (current incarnation, last 10 rounds in which trials ended, >= 8 trials): none with a verdict (ok / policy limit); aborted (infrastructure) share | none: critical; > 50%: warning | 0/1 |
| Compaction `summary_ok` share (per sampler, last 5 rounds with compaction, >= 5 summaries) | < 90% warning, < 50% critical | 1/0 |
| Compacted trials ending truncated (same window, >= 5 compacted trials) | > 20% | 1 |
| Trials that compacted at all (same window, >= 5 trials) | > 50% info | 2 |
| Trials invalidated for a fork / history rewrite of a compacted session (`compaction_fork_lost`) as a share of compacted trials (same window, >= 5) | > 5% | 1 |
| Multi-segment trials not written minus one-segment trials not written (same window, >= 5 multi-segment trials) | > 20 points | 1 |

Levels: 0 critical, 1 warning, 2 info. Override thresholds with
`--thresholds t.json` (keys are in `yeto/dashboard/alerts.py:DEFAULT_THRESHOLDS`;
an unknown key is an error).

## Context compaction (CompactionRL segments)

Shown only when the tapes carry compaction data; without it the page, the view
JSON and the events keep their old fields. With `agentic.compaction` on, a
compacted trial is written as one buffer sample per segment
(`metadata.compaction`), every segment with the trial's reward.

- Sampler: each `rl_sampler_engine_round` carries `agentic.compaction`, the
  summary of the round's written samples (`yeto/rl/engine/timeline.py:
  CompactionTally`); the sampler log prints the same as one
  `agentic round N compaction: ...` line. The round event also counts the
  segments whose rebuilt context cut the task message or the kept last step's
  tool output, as `agentic.compaction_task_truncated` and
  `agentic.compaction_last_step_truncated` (absent when zero).
- Trainer: `rl_round_trained.compaction` is the same summary over the step's
  trained samples; W&B gets its scalars as `train/compaction/<key>` on
  `train/step`.

| Key | Meaning |
|---|---|
| `trials` / `samples` / `samples_per_trial` | trials (distinct trajectories), samples (segments), samples per trial |
| `compacted_trials` / `compacted_frac` | trials with at least one compaction, and their share of `trials` |
| `compactions` / `compactions_mean` / `compactions_max` | summary turns in total, per trial, the most in one trial (`max_compactions`: the configured limit) |
| `segments_hist` | trials by segment count (`{"1": n1, "2": n2, ...}`) |
| `segment_samples` / `segment_sample_frac` | samples of compacted trials, and their share of `samples` |
| `summaries` / `summary_ok` / `summary_ok_frac` | summary turns, those with a complete `<summary>` block, the share |
| `summary_truncated` / `summary_truncated_frac` | summary turns that hit the summary turn's `max_tokens` before `</summary>`, their share of `summaries`; null when the samples do not report it. The summary alert names this share and suggests a larger `summary_max_tokens` |
| `truncated_trials` / `truncated_compacted_trials` / `truncated_compacted_frac` | trials whose last segment is `truncated`; those that compacted; their share of `compacted_trials` |
| `exhausted_trials` / `exhausted_truncated_trials` | trials that used every allowed compaction (and also ended truncated); null when the limit is unknown |
| `context_full_trials` | trials the records proxy stopped for a full context; null when not reported |
| `trainable_tokens` / `summary_tokens` / `summary_token_frac` | loss-mask tokens, summary-turn tokens, the share; null when the samples carry no `summary_tokens` |
| `fork_lost_trials` / `compaction_exposed_trials` / `fork_lost_frac` | trials the gateway invalidated because the harness forked or rewrote the history of a compacted session (`compaction_fork_lost`, lost before the group was written); written compacted trials plus those; the share |
| `retry_replays` / `retry_replay_trials` | lost replies the gateway replayed to a harness retry, and the trials that had any; null when the samples do not report them |
| `reward_mean_trial` / `reward_mean_segment` | reward mean per trial (the GRPO unit) and per segment sample; with compaction on, the round's `agentic.reward_mean` / `policy_tokens_mean` are the per-trial values |
| `policy_tokens_mean_trial` / `policy_tokens_mean_segment` | loss-mask tokens per trial and per segment sample |
| `by_segments` | per segment-count bucket (`0` = given up by a drain before it started, `1`, `2`, `3`, `4+`, `unknown`): trials `written`, `lost` before the group was written by reason (`drain_timeout`, `served_by_other_version`, aborted ...), `dropped` with a group that was not written by reason, `not_written_frac`; the trainer adds `stale` from `rl_buffer_stale_skipped.stale_by_segments` when reported |
| `xhw_downweighted_groups` / `_trials` / `_samples` / `xhw_trial_mean_shift` | groups xhw_guard downweighted (`metadata.yeto_xhw_downweight`) and the mean absolute move of their per-trial reward mean (the downweight keeps the per-sample mean, not the per-trial one) |

With compaction on, each `rl_agentic_drain` also carries `drained_by_segments`
(the trials the drain cut, by the segment count the records proxy held when it
cut them), `drained_compacted_trials` (2 segments or more) and
`drained_uncompacted_trials` (1 segment); the same cut trials appear in the
round's `by_segments` as `lost: {"drain_timeout": n}`.

The "Context compaction" table in the Async GRPO section sums the counts over
all rounds per island (ratios are recomputed from the sums), shows the trial
fate per segment-count bucket (are long trials dropped more often than short
ones?) and lists the samplers' last 20 rounds. The alert thresholds are the `compaction_*` keys of
`DEFAULT_THRESHOLDS`.

## Missing data

A field that is not in the tapes is shown as "no data", never drawn as 0. The
ports-path learner tape does not yet emit `rl_heartbeat`, `rl_resource_sample`,
`train_metrics{}`, reward quantiles, response length, truncation ratio and
similar fields; the matching curve tabs and card fields show "no data" and will
appear without dashboard changes once they are emitted. Without `tok_per_s`,
tok/s is derived as `action_tokens / (rollout_seconds + train_seconds)`. Without
`rl_cell_snapshot`, the cell table is derived from the journal's `gpu_pool`
records (with the source labelled).

## Generation-phase utilization

The "generation-phase utilization" block below the per-round panel reads the
learner tape:

- Cutoff drop / carry-over table: from `rl_rollout_cutoff`, the submitted groups,
  target groups, dropped groups, dropped samples, dropped tokens and filtered
  groups; empty fields show "unknown".
- Trajectory completion curve: `trajectory_ended_at` of `rl_trajectory_reward`
  minus the start of the generation span (`rl_timeline_span` with
  task=generate); the dashed line is the cutoff (end of the generation span).
  Trajectories dropped at the cutoff have no completion event and are only
  counted in the table.
- Four-part timing: per-round totals of `generation_seconds` / `tool_seconds` /
  `evaluate_time` / `sandbox_start_seconds` and the tool-execution share;
  selecting a round lists each trajectory.
- KV usage and queueing: `kv_used_tokens/kv_capacity_tokens` and
  `queued_requests` from `rl_load_sample`, shaded by generation span.
- Cross-round carry-over: shown only when trajectory events carry a
  `started_rollout_id` that differs from the training round, or
  `policy_versions` with more than one version.
- A chart without any data is not drawn; it is named, with the event it needs, in the one-line
  "Panels not used in this run" fold at the bottom of the page (the whole block is hidden when
  none of its charts has data, as on older tapes).

Comparing two runs (this block only; the `--compare` tapes are read once at
start and not followed):

```bash
yeto dashboard export --tapes <A tapes...> --compare <B tapes...> --label "A oversample 6" --compare-label "B oversample 12" -o ab.html
```

The page switches between "side by side" and "overlay" (overlay applies only to
the completion curve), and between "all rounds overlaid" and "round N".

## SFT runs (ms-swift Megatron + decoupled DiLoCo)

`yeto/dashboard/sft.py` adapts the native files of ms-swift islands and the
syncer into `sft_*` records for the same reducer; the page switches to the
metrics layout (KPI row, "Islands now", fleet charts, distributions,
cost / efficiency, syncer events) when the view carries an `sft` key. RL runs
keep their panels with the same styling plus a round-duration distribution.
The header range (1h / 6h / 24h / 7d / 30d) filters every SFT chart and
percentile client-side; cost KPIs compare with the previous window of the same
length.

Local directory layout read by `yeto dashboard serve|export --tapes <dir>`:

```
<dir>/island-<N>/logging.jsonl  yeto_steps.jsonl  bench_metrics.jsonl  health.rank*.jsonl
                 args.json  island.json (name, nodes)  hostmem.jsonl (sft_host_mem)
<dir>/syncer/    syncer.log  tape.jsonl  cmdline.txt (syncer argv)  rss.jsonl (sft_syncer_rss)
```

Pull the files with any read-only incremental copier (byte offsets); the
dashboard itself never connects to the nodes. `bench_metrics` / `health` step
numbers restart at 1 per launch and are anchored at the first step logged by
ms-swift / the yeto plugin. Reference lines are threshold keys, unset (0, no
line) by default: `sft_ref_step_s` (the recipe's expected step time),
`sft_hbm_limit_gib` (the accelerator's HBM limit), `sft_syncer_rss_limit_bytes`
(the syncer host's memory cap); `sft_slow_factor` (1.1) marks steps slower than
the reference.

A resumed launch re-runs the steps after its checkpoint: a step at or below the
newest one supersedes everything from that step on, so each island has one row
per iteration (the newest launch's). An island is `stopped` (greyed out, not
counted as active) once no step arrived for `sft_stop_factor` (1.5) x its step
time p50 + `sft_stop_slack_s` (300 s).

Token tiles (whole run, all islands, one value per island iteration so a step
re-run after a restart counts once): "Input tokens" sums the bench `tokens`
(packed input tokens without padding, including context that carries no loss);
"Loss tokens" sums the yeto plugin's `target_tokens` (the tokens the loss is
computed on). The cost / efficiency tiles use the same two sums over the
selected time range.

Chart semantics worth knowing:

- "Tokens / s" interpolates each island's bench `tokens_per_s` linearly onto
  one time grid and stacks the interpolated values, so islands that step at
  different moments do not produce a saw-tooth. Samples just outside the time
  range anchor its edges (no ramp from zero at the left edge). Across a gap
  longer than ~2.5 of the island's step times the island counts as down (zero),
  except for one step after its last and before its next sample.
- Syncer round timing: a merge record's `ms` is the round's open-to-done wall
  time = quorum wait (`quorum_ms`) + grace wait for stragglers + the merge
  (`sync/merge_seconds`, falling back to `sync_ms`). "Syncer merge time" draws
  only the merge; "Syncer round wait" draws the quorum wait and the grace wait
  (`ms - quorum_ms - sync_ms`); the syncer distribution card shows merge and
  wait (`ms - sync_ms`) per round. When a tape record lacks `ms` / `quorum_ms`
  they are taken from syncer.log (`outer step ... ms=`, `grace window armed ...
  quorum_secs=`). A few rounds far above the rest (the first round waits hours
  for the first push) are marked off scale instead of setting the axis.
- Duration axes label every tick in one unit (s, min or h) with round values.
- The live server sends the page build (hash of the page code) with each
  `/api/view`; a tab opened before a redeploy reloads itself once.
- "Sync overhead per step" plots `export_s` on every step with `export_s > 0`.
  The plugin pushes from a background thread, so `pushes_sent` (pushes that
  finished inside the step) is almost always 0 and cannot mark push steps.
