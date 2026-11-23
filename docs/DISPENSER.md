# Task dispenser over a shared bucket

This page covers the **store dispenser**: how agentic samplers that share nothing but one
object-store bucket (for example, A-card samplers on separate cloud accounts writing one R2 bucket)
divide the training tasks between them. There is no syncer, no TCP endpoint and no coordinator
process. The whole state lives in the bucket.

If all your samplers can reach one syncer over TCP, the syncer's task dispenser
(`--rl-data-dispense`) is still available. The two dispensers give you the same guarantees.

## Guarantees

- **Each task goes to one sampler at a time.** Two samplers can never hold the same attempt of a
  task.
- **Tasks are not lost.** A task held by a sampler that crashed, was evicted or was fenced goes out
  again.
- **A trained task is not dispensed again**, even when every trajectory scored 0. A wrong answer is
  training signal.
- **Infrastructure failures are retried, with a bound.** A task whose group could not be written
  because its trials failed on infrastructure (sandbox errors, aborted generations) goes out again,
  at most `max_redo` more times. After that it is marked `exhausted`.
- **Restarts are safe.** A restarted sampler first marks as trained the tasks whose groups it had
  already written. Then it releases the rest of its old claims.
- **The data ends cleanly.** Once every task of every pass is settled, every sampler prints
  `[engine-sampler] data ended`, writes `rl_sampler_data_ended` and exits with code 0. The NC pool
  ([NC_POOL.md](NC_POOL.md)) takes exit 0 as the normal end of the run: it does not restart the
  sampler or replace its node. The trainer then trains what is left and exits with code 0 as
  well (see [When the data ends](#when-the-data-ends)).

## Requirements

- The bucket must support conditional writes (`PutObject` with `If-None-Match: *`). Cloudflare R2
  and AWS S3 both support them. We measured R2 directly: in 32-way races on one key, exactly one
  writer won every time, and every loser got HTTP 412. The dispenser refuses to run on a store
  without atomic create, such as a FUSE bucket mount or an endpoint that answers `NotImplemented`.
- Every sampler of the run must use **the same prompt file** (byte for byte), the same seed, the
  same number of passes and the same hold-out list. The first sampler writes a plan file, and any
  sampler that disagrees with it refuses to start.

## Configuration

The dispenser lives in the `prompts.dispense` section of the sampler spec:

```json
"prompts": {
  "path": "/work/sampler/prompts.jsonl",
  "input_key": "messages",
  "groups_per_round": 8,
  "format": "agentic",
  "dispense": {
    "backend": "store",
    "namespace": "<run-prefix>/task-dispense",
    "lease_s": 900,
    "max_redo": 2,
    "passes": 1,
    "seed": 0,
    "holdout_ids": ["<task id>", "..."],
    "require_holdout": true
  }
}
```

| Key | Meaning | Default |
|---|---|---|
| `backend` | `"store"` selects this dispenser. | `tcp` |
| `namespace` | Path of the ledger under the run's store. Use **the same value on every sampler** of the run. Give a new run a new namespace. | `task-dispense` |
| `store` | Another store root for the ledger. | the spec's `buffer_root` / `store` |
| `lease_s` | Claim lease. The holder renews it every `lease_s / 3`, so a trajectory may run far longer than the lease. A claim counts as expired only after the holder stops renewing for `lease_s`. The lease does **not** limit trial time; it only sets how long a dead sampler's tasks wait before another sampler takes them. Keep 900. A value above 1800 s gets a warning (dry run and sampler start). See [Changing the lease](#changing-the-lease). | 900 |
| `max_redo` | How many `infra` or `expired` endings a task may have before it is marked `exhausted`. Releases (fenced, restarted, shut down) do not count. | 2 |
| `passes`, `seed` | Number of passes over the data. Pass 0 is in file order; each later pass is reshuffled by `seed`. | 1, 0 |
| `holdout_ids` / `holdout` | Task ids that must never be trained on, given inline or as a file path on the sampler. A training row that has one of these ids makes the sampler refuse to start. | none |
| `require_holdout` | Refuse to start without a hold-out list. | false |
| `mix` | `--data-mix` source weights (rows carry `metadata.yeto_data_source`). | none |
| `by_label`, `labels` | One stream per task label (`task_label`). `labels` limits this sampler to those labels. | none |
| `scan_s`, `max_scan_s` | How often an idle sampler lists the ledger (backs off up to `max_scan_s`). | 30, 120 |
| `full_scan` | `true`: every scan lists the whole ledger. By default a scan lists only the tasks that are not settled yet. | false |

A single task can override the lease through `metadata.dispense_lease_s` in its row.

### Changing the lease

`lease_s` is a runtime setting. It is not part of the plan:

- Each sampler renews by the `lease_s` of its own spec. `plan.json` records the first sampler's
  value for reference only. A sampler whose `lease_s` differs from the plan prints both values at
  start; `rl_dispense_lease` on its tape has `lease_s` and `plan_lease_s`.
- To change it, edit `lease_s` in the spec (or `--dispense-lease-s`) and restart the samplers. The
  new value applies from their next claim. The plan, its fingerprint and the task order do not
  change.
- Each claim records its holder's lease (`lease_s` in the claim and in `status --json`).
  Another sampler ends a claim as `expired` only when **both** its own lease and the holder's
  recorded lease have run out. So samplers with different leases are safe: a short-lease sampler
  never takes a task from a live long-lease holder, and a long-lease sampler only waits longer.
- A row's `metadata.dispense_lease_s` still overrides the lease for that task, on every sampler.
- Samplers of older yeto versions judge every claim by the plan's lease. If one may still run,
  raise `lease_s` above the plan's value on all samplers at the same time.

The NC sampler launcher sets the section with flags. Pass them identically to every sampler:

```
--dispense-namespace <run-prefix>/task-dispense \
--dispense-lease-s 900 --dispense-max-redo 2 \
--dispense-holdout ./holdout.json        # read locally, embedded as holdout_ids
[--dispense-mix 0.7,0.3] [--dispense-labels swe,terminal] [--dispense-by-label]
```

In a pool config ([NC_POOL.md](NC_POOL.md)), put the same object under `sampler.dispense`. Don't combine the dispenser with
`--data-slot`: the dispenser replaces fixed slots.

## Behaviour

Each task (a "slot" of the global sequence) moves through these states:

| Outcome of an attempt | What happens next |
|---|---|
| Group written to the buffer, any reward | `trained`. The task is not dispensed again. |
| Trials lost to infrastructure (group too small to write) | `infra`. Dispensed again, counted toward `max_redo`. |
| Holder stopped renewing (sampler died) | `expired` after `lease_s` of server time. Dispensed again, counted. |
| Holder fenced, restarted, shutting down, or round not written | `release`. Dispensed again at once, not counted. |
| More than `max_redo` counted failures | `exhausted`. Not dispensed again. |
| Row is malformed (no prompt or no task) | `dropped`. |

- **Time.** Leases are measured in the bucket's time: the object `LastModified` and the listing
  `Date` header. The sampler hosts' own clocks are never used.
- **Ownership check.** Right before a sampler writes a group, it checks that it still holds the
  task: no newer attempt exists, its attempt was not ended, and its incarnation is not fenced. If
  the check fails, the group is dropped with reason `dispense_lost`.
- **Fencing.** When the pool fences a sampler's shard (the node is taken for gone), that
  sampler's incarnation lease moves to the fence. From then on:
  - the fenced sampler renews nothing, writes no group and takes no new task;
  - any other sampler that lists the ledger releases the fenced sampler's claims immediately, without
    waiting for the lease to run out.

  If a fenced node writes a group anyway, the trainer rejects it, because the trainer's
  incarnation gate drops groups of a superseded incarnation.
- **Mix and labels.** The weighted interleave of streams is fixed in the plan, so the mix holds
  over the whole run, whichever sampler takes which task. A sampler limited to some labels takes
  only those tasks and leaves the rest to the other samplers.
- **One task at a time where possible.** With `passes` > 1, the passes of one task can meet at a
  pass boundary (later passes are reshuffled), and a redo can meet a later pass. When a sampler
  claims, it takes the tasks that have no attempt in flight first (on any sampler). A task that is
  already running is taken only when nothing else is left, so no task waits for long and the last
  tasks of a run may still run twice at once. This changes only the claim order: the plan, its
  fingerprint and the mix stay the same.

## Checking status

```
python -m yeto.rl.engine.store_dispense status --store s3://<bucket>/<run-store> \
    --namespace <run-prefix>/task-dispense
```

The command prints a totals line and one row per task. The totals count tasks in each state
(`trained`, `exhausted`, `dropped`, `running`, `stale`, `queued`) plus `unclaimed`, `slots` and
`ended`. Each task row has these columns:

| Column | Meaning |
|---|---|
| `slot` | Position of the task in the global sequence. |
| `state` | `trained` / `exhausted` / `dropped`: settled. `running`: held, lease alive. `stale`: lease ran out, not yet reclaimed. `queued`: waiting for its next attempt. |
| `attempts` | Number of attempts so far. |
| `task_id` | The task's id. |
| `holder` | Sampler of the latest attempt. |
| `elapsed_s` | Duration of the settling attempt. |
| `beat_age_s` | Time since the last renewal. |
| `lease_s` | The holder's lease for this claim (`--json` only). A claim with none (older yeto) uses the plan's. |
| `failure_reasons` | One entry per failed attempt, as `attempt:kind:reason`. |

Options:

- `--json` prints the full report.
- `--log FILE` writes one `rl_dispense_task` JSON line per task, for dashboards.
- `--brief` uses one listing request plus one read of each task's latest claim (so `task_id` and
  `holder` are shown); it skips the settlement and failure reads (`elapsed_s`, `failure_reasons`).

The command needs read access to the bucket only. Credentials come from the usual AWS environment
or profile variables.

Samplers also record dispenser events on their event tape:

| Event | When |
|---|---|
| `rl_dispense_claim` | A sampler took a task. |
| `rl_dispense_settle` | A task was settled: trained, exhausted or dropped. |
| `rl_dispense_requeue` | An infrastructure failure sent a task out again. |
| `rl_dispense_release` | A claim was released (unwritten, closing, or the holder was fenced). |
| `rl_dispense_expired` | A claim expired. |
| `rl_dispense_lost` | This sampler lost a task it held: superseded or fenced. |
| `rl_dispense_recover` | The sampler restarted and recovered its earlier claims. |
| `rl_dispense_written` | Per-round summary. |
| `rl_dispense_close` | The sampler shut down. |
| `rl_dispense_lease` | At start: the sampler's `lease_s`, `renew_s`, `plan_lease_s` when it differs, and the long-lease `warning`. |

## Limits

- **Request cost.** A sampler scans at most once per `scan_s`. A scan lists the tasks that are not
  settled yet: 5 listing requests and 1 read when they lie together at the front of the ledger
  (the usual case), 4 more for each group of old unsettled tasks far behind, and at worst one
  request per 1000 keys of the ledger when unsettled tasks are spread over all of it. The first
  scan of a process lists every settlement once. A held task costs one renewal write every
  `lease_s / 3` and three existence checks every third renewal, plus three checks before its
  group is written. `status --open` lists only the unsettled tasks and reads only their claims.
- **Expired holders.** An expired holder that is still alive (cut off from the bucket for longer
  than the lease) may write its group in the short window before it notices it lost the task. The
  task is then trained twice, which only happens past a full lease of lost connectivity. Fenced
  holders are covered by the trainer's incarnation gate.
- **Ledger size.** The ledger keeps one object per claim, renewal, failure and settlement. Delete
  the namespace when the run is archived.

## When the data ends

1. Once every task is settled, each sampler ends its buffer shard with an end record that names
   the dispenser plan, writes `rl_sampler_data_ended` and exits 0. This is a state of the ledger
   that never changes back. A sampler that only stops (preempted, scaled down, restarted) writes no
   end record.
2. The trainer's merger reads that end record, reads every shard once more, and then ends the
   data. Shards of `--rl-buffer-shards` that wrote no end record do not hold it back if they have
   no groups (no sampler took that shard index) or only groups of the settled plan (their sampler
   stopped and never came back). A shard with groups of another source keeps the data going until
   it writes its own end. The merger logs `the task dispenser settled every task` and reports
   `data_end_by_dispenser`.
3. The trainer trains the groups that are left. The last round may be short of groups; it is
   padded to the batch size and trained, not dropped. The next read finds the end
   (`rl_buffer_exhausted`).
4. The trainer releases the last trained policy (a release skipped by `--rl-release-every` or
   `--rl-release-skip-busy` is published now), writes `rl_run_end` with `reason: "data_ended"` and
   exits 0. A run that reaches `--total-steps` first writes `rl_run_end` with
   `reason: "total_steps"`.

A run that uses several dispensers (several pools, each with its own namespace) ends when every
dispenser that wrote groups is settled. A dispenser that has not written a single group by then is
not waited for.

## A new task set or more passes

The prompt file, `passes`, `seed`, the streams and the hold-out list are part of the plan. A
sampler that disagrees with the plan refuses to start (`dispense mismatch`). Once the data ended,
the buffer shards are ended too, and the trainer does not read past that end.

To continue with new tasks or more passes, start a new run under a new prefix: a new `namespace`,
a new buffer and a new trainer store. Do not edit or delete `plan.json` of a running ledger, and do
not reuse a namespace for another task set.
