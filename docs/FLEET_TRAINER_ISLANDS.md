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

## What runs

`yeto run up` starts:

- **The syncer launch.** It is what `run up` starts as its trainer. It runs the syncer only
  and records its address and term in the run store (`fleet/syncer.json`).
- **One island launch per island.** The trainer worker starts each one as soon as the
  syncer record exists. Island k is learner k, trains from `trainer<k>/` in the store,
  consumes the shards of its own pools (`<shard_base>-i<k>-<n>`) and follows only their
  member tables (`YETO_RL_MEMBER_LANES=i<k>`). Its cluster prefix is `<run>-i<k>`.
- **The trainer worker.** It runs under the rollout controller's supervisor, next to the
  pool workers. Every 15 s it reads the newest desired state and reconciles each trainer
  member. If its thread dies the supervisor restarts it, and it adopts the island launches
  that are still running. It never starts a second launch of an island.

Each island launch is supervised on its own. An infrastructure failure relaunches that
island only, at most 3 times. The other launches never notice. A configuration error or
a strict failure marks the island `failed`. A normal end marks it `done`. The worker keeps
its record in `fleet/trainers/<k>/launch.json`, next to the island's lifecycle record
`state.json` and its learner id claim `learner.json`.

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
| 2 | Refused, nothing written: a bad name or samplers file, no learner id left, a trainer shape that differs from the fleet's, an island's launch that fails its dry run. |
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
is never handed out again, not even after its island was removed. When no id below
`fleet.max_islands` is left, `add` exits with 2.

The new island joins the running syncer session. It catches up to the current merged
weights first, then answers rounds like the others. Its pools are compiled for its id:
shards `<shard_base>-i<k>-<n>`, store `trainer<k>/`, and the fleet's one task ledger.

### Removing an island

`remove` drops the trainer member and its sampler pools from the desired state. The
worker writes the island's STOP (`fleet/trainers/<k>/STOP`), and the island's launch
stops it and releases its machines. The pools drain: their claims go back to the ledger
and their shards are retired. The other islands keep merging.

The syncer frees the island's place in the rounds in flight at once when the island sent
LEAVE. When the island was torn down without it, those rounds wait up to its lease
(`--rl-sync-island-lease-s`) first, and the syncer drops the island when the lease runs
out.

### Moving an island

`move` changes the island's `--gpu` (and price). The worker writes the island's STOP; the
launch on the old machines ends. Then it starts the next launch on the new machines, under
cluster prefix `<run>-i<k>-m<n>`, with the same learner id and the same `trainer<k>/`
releases. The syncer sees the island away for a short time and back under its id; it
catches up to the merged weights before it answers rounds again. Its pools and their
shards stay as they are.

### Shape checks

An added or moved island must have the fleet's trainer shape. The full-parameter merge
needs the same fragments on every island (`--fragments`, TP and PP). The worker passes the
fleet's shape (`--rl-fleet-trainer-shape`) to every launch it starts for an added or moved
island. The command's dry run checks it, so an island of another shape (for example TP 2
in a TP 1 fleet) is refused with exit 2 before anything is written or ordered.

## Code

| Piece | Module |
|---|---|
| Per-island compile, trainer members, launch argv | `yeto/rl/island_launches.py`, `yeto/rl/fleet.py` |
| Trainer worker | `yeto/rl/trainer_worker.py` |
| `yeto fleet island` | `yeto/rl/fleet_islands.py` |
| Syncer launch | `yeto/launch/fleet_syncer.py` |
| Learner id claims | `yeto/launch/learner_ids.py` |
| Island lifecycle and STOP | `yeto/launch/island_lifecycle.py` |
