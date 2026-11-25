# Trainer islands at run time

An island fleet (`islands` + `fleet` in the run config, [MILES_RL.md](MILES_RL.md), "Island
fleet") is by default one `yeto launch` of every trainer island and the syncer
(`fleet.launch: synced_group`). That launch ends when one island is lost for good, and no
island can join while it trains.

With `fleet.launch: per_island`, every trainer island is a launch of its own. The islands
share one syncer. While the fleet trains you can:

- **add** an island with its own sampler pools,
- **remove** an island together with its pools,
- **move** an island to other machines or to another cloud, under the same learner id.

The other islands keep training through each of these changes. A single trainer run (no
`islands`) is not affected by anything in this page.

## Turning it on

```json
{
  "fleet": {"launch": "per_island", "max_islands": 4, "dispense": "fleet", "trainer_argv": ["..."]},
  "rollout": {"control_plane": "store"},
  "controller": "in_process",
  "islands": ["..."]
}
```

| Key | Meaning |
|---|---|
| `fleet.launch` | `synced_group` (default, one launch) or `per_island`. |
| `fleet.max_islands` | The most islands the fleet can ever hold. It becomes `--rl-sync-max-learners` on every launch: the syncer admits learner ids `0 .. max_islands-1`. Default: the declared islands, so nothing can be added. It must be at least the declared islands. |
| `rollout.control_plane` | Must be `store`. The islands are members of the fleet's desired state ([FLEET_CONTROL_PLANE.md](FLEET_CONTROL_PLANE.md)). |
| `controller` | Must be `in_process`. The trainer worker runs in the rollout controller of `yeto run up`. |

`yeto run up --dry-run` refuses a per-island fleet without the store control plane. An
island with Miles samplers (`kind: miles`) is refused too: it is one launch with its
trainer, so it stays on `synced_group`.

The flags in `fleet.trainer_argv` are the same as for `synced_group`. A count quorum
(`--rl-sync-quorum`) is refused for a full-parameter fleet whose `max_islands` is above
the declared islands, because a count does not follow islands that join and leave. Leave
both quorum flags out to get `--rl-sync-quorum-theta 1.0`, or give a theta.

A round that was open when an island was removed or moved can stall the merge, see
[Limits today](#limits-today).

## What runs

`yeto run up` starts:

- **The syncer launch.** It is what `run up` starts as its trainer. It runs the syncer only
  and records its address and term in the run store (`fleet/island_syncer.json`). A
  syncer cluster whose status cannot be read is asked through the cluster's job table
  first. The syncer launch ends when the job ended in failure, or when the cluster stayed
  unreadable or down for 300 s in a row (`--rl-fleet-syncer-grace-s` in
  `fleet.trainer_argv`; only the syncer launch gets it). That time includes time the sky
  API itself does not answer, so a sky API outage that long ends the fleet even when the
  syncer is healthy.
- **One island launch per island.** The trainer worker starts each one as soon as the
  syncer record exists. Island k is learner k, trains from `trainer<k>/` in the store,
  consumes the shards of its own pools (`<shard_base>-i<k>-<n>`) and follows only their
  member tables (`YETO_RL_MEMBER_LANES=i<k>`). Its cluster prefix is `<run>-i<k>`.
- **The trainer worker.** It runs under the rollout controller's supervisor, next to the
  pool workers. Every 15 s it reads the newest desired state and reconciles each trainer
  member. A step that fails (a store read error) is retried with a backoff of up to
  300 s. If its thread dies the supervisor restarts it, and it adopts the island launches
  that are still running. It never starts a second launch of an island. Before each step
  it checks that `fleet/applied.json` still names its controller. Once another controller
  has taken the fleet over, the worker starts, stops and relaunches no island any more.
  When that file cannot be read, the worker skips the step.

Each island launch is supervised on its own. An infrastructure failure relaunches that
island only, at most 3 times. The other launches never notice. A configuration error or
a strict failure marks the island `failed`. A normal end marks it `done`. The worker keeps
its record in `fleet/trainers/<k>/launch.json`, next to the island's lifecycle record
`state.json` and its learner id claim `learner.json`.

The island launches are processes of the host that runs `yeto run up`. Each record names
that host. **Restart `yeto run up` on the same host.** A worker on another host cannot see
those processes, so it leaves every island recorded by another host alone and logs an
error naming that host. It does not adopt, relaunch or start those islands a second time.

If that host is lost or renamed, an operator can hand its islands to another host, see
[Operations](#operations-lost-control-host).

The syncer launch is not adopted by a later one either. A new syncer launch starts the
syncer again, from its checkpoint. On a new cluster that syncer has a new address. The
island launches started for the old syncer lose it, end, and are relaunched for the new
one like any infrastructure failure, which uses up their relaunch budget.

The run's spend (`run.budget`) bills each island launch from its record, each incarnation
at the price it ran at. An island added while the fleet trains counts from its start, and
a removed island counts up to its end. With `budget.usd` set, `add` needs `--usd-per-h`.

`yeto run stop` stops every island launch. `yeto run down` downs the syncer launch and
every island launch.

## Commands

Each command takes the run config, writes one new generation of the desired state, and
exits. The trainer worker and the pool workers of the running `yeto run up` apply it
within seconds. `--dry-run` shows the change and runs the island launch's own dry run, and
writes nothing.

```bash
yeto fleet island add --config run.json --name north --gpu modal:1x2xh100 --usd-per-h 8 --samplers north.samplers.json --dry-run
yeto fleet island add --config run.json --name north --gpu modal:1x2xh100 --usd-per-h 8 --samplers north.samplers.json
yeto fleet island move --config run.json --name sglang --gpu aws:2xh100@us-east-2 --usd-per-h 6 --dry-run
yeto fleet island move --config run.json --name sglang --gpu aws:2xh100@us-east-2 --usd-per-h 6
yeto fleet island remove --config run.json --name north --dry-run
yeto fleet island remove --config run.json --name north
yeto fleet show --config run.json
```

| Exit | Meaning |
|---|---|
| 0 | Written (or, with `--dry-run`, the change is valid). |
| 2 | Refused, nothing written: a bad name or samplers file, no learner id left, no `--usd-per-h` while the run has a budget, a trainer shape that differs from the fleet's, an island's launch that fails its dry run. |
| 3 | The desired state changed under the command three times in a row. Run it again. |

### Adding an island

1. Write the island's samplers list to a file, for example `north.samplers.json`. Use the
   same entries as an island's `samplers` in the run config. Each pool needs its own
   `shard_base` and `shard_limit`. A pool that shares a port range with other pools
   (records relay or tunnel) cannot be added at run time.
2. Run `yeto fleet island add ... --dry-run`. It names the learner id the island would get,
   its pools (`i<k>-<name>`), and it runs the island launch's dry run.
3. Run the same command without `--dry-run`.

The island gets the lowest learner id never used in this run. The claim is written to the
store before the desired state, so two adds at the same time never get the same id. An id
is never handed out again, not even after its island was removed. One exception: if the
desired state write of an `add` failed (exit 3), running the same `add` again takes the
same id, as long as the island never started and nothing was written under `trainer<k>/`.
When no id below `fleet.max_islands` is left, `add` exits with 2.

The new island joins the running syncer session. It catches up to the current merged
weights first, then answers rounds like the others. Its pools are compiled for its id:
shards `<shard_base>-i<k>-<n>`, store `trainer<k>/`, and the fleet's one task ledger.

### Removing an island

`remove` drops the trainer member and its sampler pools from the desired state. The
worker writes the island's STOP (`fleet/trainers/<k>/STOP`). It writes the STOP again
while the launch runs, because a launch that was still starting clears it once. The
island's launch then tears its machines down. The pools drain: their claims go back to
the ledger and their shards are retired. The other islands keep merging. The merge can
stall once, see [Limits today](#limits-today).

### Moving an island

`move` changes the island's `--gpu` (and price). The worker writes the island's STOP; the
launch on the old machines ends. Then it starts the next launch on the new machines, under
cluster prefix `<run>-i<k>-m<n>`, with the same learner id and the same `trainer<k>/`
releases. The syncer sees the island away for a short time and back under its id; it
catches up to the merged weights before it answers rounds again. Its pools and their
shards stay as they are.

A move costs the island's training since its last policy release
(`--rl-release-every`): the new launch starts from the newest release in `trainer<k>/`.
It also stalls the merge once, like a remove.

### Shape checks

An added or moved island must have the fleet's trainer shape. The full-parameter merge
needs the same fragments on every island (`--fragments`, TP and PP). The worker passes the
fleet's shape (`--rl-fleet-trainer-shape`) to every launch it starts for an added or moved
island. The command's dry run checks it, so an island of another shape (for example TP 2
in a TP 1 fleet) is refused with exit 2 before anything is written or ordered.

## Limits today

- **No LEAVE on remove and move.** Every remove and every move tears the island's
  machines down without the island telling the syncer it leaves. The syncer sees the
  connection drop and takes the island out of the pool, so the following rounds no longer
  expect it. A round that was already open and had asked the island for its delta still
  waits for it. Until the syncer frees an island's place in open rounds when its lease
  runs out, that round merges without the island only at its soft deadline
  (`--rl-sync-soft-deadline-s`). With no soft deadline set, it waits for the quorum
  timeout, 900 s by default. With the default `--rl-sync-quorum-theta 1.0`, the fleet's
  outer merge stalls that long once per remove or move. Training on the other islands
  goes on.

  `yeto run up` and its dry run print a warning about this when `fleet.trainer_argv`
  sets no soft deadline. If you need a shorter stall, set `--rl-sync-soft-deadline-s` in
  `fleet.trainer_argv`. Make it longer than the slowest island's local training between
  two syncs. A round opens before the islands train for it, so a shorter deadline also
  cuts a slow island out of rounds while every island is healthy. Its late delta is
  carried into a later round, and it is refused once its base lags more than
  `--rl-sync-max-base-lag`. The soft deadline also becomes the default of
  `--rl-sync-final-grace-s`, how long the syncer waits for the islands to take the final
  cut after the last step. Set `--rl-sync-final-grace-s` too if that should differ.
- **A move loses progress.** It loses the island's training since its last release (see
  [Moving an island](#moving-an-island)).
- **Same host.** `yeto run up` must be restarted on the host that started the island
  launches (see [What runs](#what-runs)).
- **The syncer is not adopted.** A new syncer launch restarts the syncer, and the island
  launches relaunch for it (see [What runs](#what-runs)).

## Operations: lost control host

This section is for operators of the fleet. Use it only when the host that ran
`yeto run up` is lost or was renamed. On the new host:

```bash
yeto run up --config run.json --take-over-islands-from <old-host>
```

The take-over has two guards:

1. **It waits until this run up holds the fleet.** This run up's controller first claims
   `fleet/applied.json`. That fences the old host's controller, and the old host's trainer
   worker stops at its next step if it still runs.
2. **It asks each island's source first.** For each island the old host started, the
   island's record (`fleet/trainers/<k>/state.json`) is checked against its source (sky
   or Modal):
   - **Still running**: a launch is started under the same prefix and incarnation, and it
     adopts the running machine. Nothing new is ordered.
   - **Gone** (or its record has ended): the island starts again as its next incarnation
     (`<run>-i<k>-m<n>`, same learner id, from its last release). An island removed
     meanwhile is recorded removed.
   - **Unknown** (the source does not answer, or cannot be asked from this host): nothing
     is ordered for that island. An error is logged and the island is asked again at the
     next step.

Every island launch first takes its island's records lock (`fleet/trainers/<k>/lock`). If
the old host was not lost after all (a network partition) and its launcher still holds
that lock, the new launch hands off with exit 12. The island then goes back to the old
host, and this run up does not take it over again. Two incarnations of one island never
run at once.

The new host does not stop launches the old host started. When the old host's launcher
processes are really gone, `yeto down` their clusters if any remain (the names are in
`fleet/trainers/<k>/launch.json`, `prefix_now`).

## Code

| Piece | Module |
|---|---|
| Per-island compile, trainer members, launch argv | `yeto/rl/island_launches.py`, `yeto/rl/fleet.py` |
| Trainer worker | `yeto/rl/trainer_worker.py` |
| `yeto fleet island` | `yeto/rl/fleet_islands.py` |
| Syncer launch | `yeto/launch/fleet_syncer.py` |
| Learner id claims | `yeto/launch/learner_ids.py` |
| Island lifecycle and STOP | `yeto/launch/island_lifecycle.py` |
