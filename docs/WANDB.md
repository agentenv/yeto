# W&B telemetry

Opt-in Weights & Biases streaming for a fleet: one W&B **group** per run,
one **run** per learner island, plus one for the syncer's merge record.
Off by default; nothing in this document changes a run that does not pass
`--wandb`.

    export WANDB_API_KEY=...
    yeto launch --gpu aws:8xa100@us-east-2,runpod:8xh100@CA \
      --model qwen35-9b --data org/chat-traces \
      --wandb --wandb-project fleet-lab

## Why

Yeto's existing observability is a log line every ten steps plus the
syncer's JSONL event tape, both of which need `sky logs` and a script to
turn into an answer. That is workable for one island and painful for the
case Yeto exists to serve: eight islands in four regions, some of them
spot, merging asynchronously. The questions that matter — *which island is
falling behind*, *is the grace window absorbing a slow link or hiding a
dead one*, *did that island's loss diverge or did it just stop pushing* —
are all comparisons across islands over time.

## Run topology

    group = <--cluster-prefix>          # the run's name
    ├── job_type="syncer"   name="syncer"      <- event-tape metrics
    ├── job_type="learner"  name="learner-0"   <- rank 0 of island 0
    ├── job_type="learner"  name="learner-1"
    └── ...

**One run per island, not one per fleet.** Island-local steps advance
independently under async DiLoCo; a shared run would log a non-monotonic
step series and W&B would silently drop points. Grouping recovers the
fleet view without that cost.

**Only rank 0 of an island logs.** Every other rank holds the same
all-reduced loss and the same merged fragments.

**A preempted island keeps its curve.** The run id is
`<group>-learner-<id>` with `resume="allow"`, so when the fleet controller
relaunches a spot island the new process reattaches to the run it left
behind instead of starting a second one.

## Metrics

Learner runs (`yeto/learner.py`, and the diffusion / Megatron / MLX peers),
on the same ten-step cadence as the existing log line, x-axis `local_step`:

| metric | meaning |
|---|---|
| `train/loss_per_token` | the island-global SUM loss over its target tokens |
| `train/lr` | inner AdamW learning rate |
| `train/sec_per_step`, `train/tokens_per_sec` | throughput for *that window* |
| `train/raw_tokens_total`, `train/target_tokens_total` | running totals |

Sync-boundary metrics, emitted from `yeto/diloco_sync.py` — the one module
every backend's sync goes through — x-axis `global_step`. Only boundaries
that actually merged or pushed are logged; the rest would bury them:

| metric | meaning |
|---|---|
| `sync/staleness_max`, `sync/staleness_mean` | versions this island is behind the newest global fragment |
| `sync/merges_applied` | broadcasts applied at this boundary |
| `sync/pushes`, `sync/push_bytes` | pushes answered, and their wire size after q4 |
| `sync/push_delta_norm` | ‖local − raw anchor‖, the size of what this island contributed |
| `sync/pending_pulls` | pulls the island could not answer yet |

Syncer run, from the event tape (`yeto/wandb_tape.py`):

| metric | meaning |
|---|---|
| `sync/gnorm` | outer-gradient norm of the merge |
| `sync/quorum_ms`, `sync/grace_ms`, `sync/sync_ms`, `sync/merge_ms` | rendezvous timings; a null is omitted, not logged as zero |
| `sync/expected`, `sync/responded`, `sync/missed`, `sync/participation` | who made the round |
| `learner/<id>/staleness`, `.../contribution`, `.../weight`, `.../c_steps`, `.../c_tokens` | per-island accounting, straight from the RDA weights |

`wandb.config` carries the launch flags plus the provenance pins (model and
data identifier/revision), the island's backend, world size, and fragment
count — so a curve can be filtered by cloud, region, or GPU.

## Debugging map

| symptom | series |
|---|---|
| one island dragging the merge | `learner/*/contribution` diverging, `learner/*/staleness` climbing |
| merges missing islands | `sync/missed` > 0 with `sync/grace_ms` at its ceiling |
| WAN is the bottleneck | `sync/push_bytes` against `sync/sync_ms` |
| outer step degenerate | `sync/gnorm` spiking or collapsing to 0 |
| data skew across islands | `train/loss_per_token` fanning out between islands |
| spot preemption | a gap in an island's curve, a step down in `sync/responded` |

## The syncer stays Rust

The syncer writes its merge record already; nothing was added to its hot
path. Instead of shipping the tape to a reader, the reader goes to the
tape — in both controller modes:

| mode | where the syncer runs | who tails the tape |
|---|---|---|
| `head` (default) | a subprocess on the head VM | `LocalSyncer.start_tape_forwarder`, a daemon thread in the controller |
| `local` | its own `<prefix>-syncer` cluster | `yeto.wandb_tape --follow`, a sidecar process on that VM |

The sidecar (`syncer_tape_sidecar` in `yeto/launcher.py`) is backgrounded
on purpose. The syncer has to stay the job's **foreground** process,
because FleetController reads that job's exit code as the syncer's health;
and as a separate process the forwarder cannot take the syncer down with
it. Telemetry also pulls the repo workdir onto the syncer VM, which the
cross-build path already did — `yeto.wandb_tape` and `yeto.wandb_logger`
are stdlib-only, so the syncer VM never needs the training stack. A test
pins that.

`tests/test_wandb_tape.py` parses the Rust format literal in
`syncer/src/server.rs` and fails if a field is added there without a
decision on the Python side.

### Resuming, not replaying

A forwarder records the byte offset of the last record it forwarded next
to the tape (`~/yeto-tape.jsonl.offset`). A restarted forwarder — a
restarted sky job, a reused head — picks up from there instead of logging
every past merge a second time. Writes are throttled to one every two
seconds and flushed on shutdown; **sky ends a job with SIGTERM, so the
follow loop handles it** rather than dying with unflushed offsets. A tape
shorter than the stored offset is a different tape and is read from the
top.

The bounded cost of a hard kill (SIGKILL, VM loss) is up to two seconds of
re-logged merges. `--from-start` ignores a stored offset deliberately.

### Replaying a finished tape

    python -m yeto.wandb_tape ~/yeto-tape.jsonl --wandb-group <run-name>
    python -m yeto.wandb_tape ~/yeto-tape.jsonl --follow    # live

## Failure policy

Telemetry never costs a run:

- no `wandb` package, no `--wandb`, or rank > 0 → a no-op sink
  (`yeto.wandb_logger.NullRun`), with no cost at the call sites;
- a failed `wandb.init` → warning, training continues;
- a failed `wandb.log` → the run downgrades itself to the no-op sink for
  the rest of training rather than retrying into a WAN partition;
- the `pip install wandb` step on a learner is `|| echo`, not fatal.

`WANDB_API_KEY` follows the same path as `HF_TOKEN`: read from the
submitting machine's environment, forwarded through sky `envs` to the head
(or the syncer cluster) and to every learner, and only when `--wandb` was
passed. It is never
written into `wandb.config` (`build_config` drops anything whose name looks
like a credential).

**Offline islands.** With no key and no `~/.netrc` entry, an online run
would block or fail on a headless spot VM, so `yeto.wandb_logger` forces
`mode="offline"` instead: metrics buffer to disk and `wandb sync` recovers
them. `--wandb-mode offline` selects this deliberately, which is the safer
choice for a WAN-distant island.
