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
again). The default price table `yeto/dashboard/prices.example.json` holds
**example values, not billing data**; replace it with `--prices my.json` (same
format as the example, keys are `cloud:GPU`, `*:GPU` is a wildcard). An island
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

Levels: 0 critical, 1 warning, 2 info. Override thresholds with
`--thresholds t.json` (keys are in `yeto/dashboard/alerts.py:DEFAULT_THRESHOLDS`;
an unknown key is an error).

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
- Older tapes without these events show "no data" in this block.

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
ms-swift / the yeto plugin. Reference lines are threshold keys:
`sft_ref_step_s` (571), `sft_hbm_limit_gib` (53.9),
`sft_syncer_rss_limit_bytes` (1.4e12), `sft_slow_factor` (1.1).

Chart semantics worth knowing:

- "Tokens / s" holds each island's last bench `tokens_per_s` on a shared time
  grid (zero before an island's first step and after it goes quiet for about
  three of its own step times) and stacks the held values, so islands that
  step at different moments do not produce a saw-tooth.
- "Syncer merge time" is `sync/merge_seconds` (falling back to `sync_ms`) per
  merge record: the merge itself. The record's `ms` is the round's
  open-to-done wall time, which includes waiting for the quorum; it is drawn
  with `quorum_ms` on the separate "Syncer round time" chart.
- "Sync overhead per step" plots `export_s` on every step with `export_s > 0`.
  The plugin pushes from a background thread, so `pushes_sent` (pushes that
  finished inside the step) is almost always 0 and cannot mark push steps.
