# Fleet control plane: sampler pools in the run store

The fleet control plane drives a run's sampler pools from state kept in the run store
(`store` of the run config) instead of files on the control host. It lets you:

- add, remove, rescale and reconfigure sampler pools while the run trains, without
  restarting the trainer or touching the other pools;
- stop the controller (`yeto run detach`) and start it again, even on another host,
  without draining a single sampler;
- keep one stuck pool from holding the others.

It is **opt-in**. A run config without it behaves exactly as before. The single-pool
customer path (`examples/k31/`) does not use it.

## Turning it on

```json
"controller": "in_process",
"rollout": {"control_plane": "store", "drain_on_exit": false, "lock_ttl_s": 600}
```

| Key | Meaning |
|---|---|
| `rollout.control_plane` | `"local"` (default: local state files, as before) or `"store"`. `"store"` needs `controller: "in_process"`. |
| `rollout.lock_ttl_s` | Lease of each pool's store lock, in seconds (default 600, at least 60). Keep it well above the clock skew between control hosts. |
| `rollout.drain_on_exit` | `false` is recommended: a process exit detaches (the samplers keep running) instead of draining every pool. |

## What lives in the store

| Key | Written by | Content |
|---|---|---|
| `fleet/desired.json`, `fleet/desired/<generation>.json` | `yeto fleet apply`, `yeto run scale`, the first `run up` | The fleet's desired state (below). |
| `fleet/applied.json`, `fleet/applied/<seq>.json` | The controller | The generation it applied last, its pools, and the pools it is still removing. Each record is created once: of two controllers that start at once, one claims it and the other stops applying. |
| `fleet/pools/<pool>/state.json` | The holder of the pool's lock | The pool state: members, orders, shards, budget use, pending handoffs. |
| `fleet/pools/<pool>/lock/<token>.json`, `.lease` | The controller | The pool's single-writer lock. The token is a fencing token. |
| `fleet/shards/<shard_base>/<n>.json`, `ended/<shard>.json` | The pool's lock holder | The shard number allocator. Each number is issued once. End records of shards that ended for good. |
| `fleet/members/<pool>.json` | The pool's lock holder | The member table the trainer reads. It lives in the trainer's buffer store. |

### The lock

Each pool has one writer. The controller claims `fleet/pools/<pool>/lock` with an
exclusive create (`If-None-Match: *`) and renews the lease every tick.

- **A second controller** is refused while the lease runs. Once the lease has run out,
  the next claimer gets a higher token.
- **Before every action that changes something** (an order, a sampler start, a fence, a
  release, a drain request), the pool lists the tokens.
  - If a higher token exists, the pool *detaches*: it stops acting and drains nothing.
  - If the store does not answer, the pool does nothing until it does.
- **Stores without atomic exclusive create** are refused (an S3 endpoint that rejects
  `If-None-Match`, or a bucket mount).

### One-time import

On the first start on the store control plane, a pool imports its local state file
(`~/.yeto/nc/pool-<shard_base>.json`) once. The file is only read. From then on the
store holds the state, and the local file is written only as a mirror for local readers
(`run status`, the `run up` monitor).

### Shard numbers

Shard names stay `<shard_base>-<n>`. `n` now comes from the store allocator:

- **Each number is issued once**, across controllers and control hosts. The lifelong
  `shard_limit` order cap does not apply.
- **The tunnel port range** (`agentic.setup.tunnel.listen`) still bounds the index: one
  port per index. When no index is left, the pool writes `pool_shard_budget_exhausted`
  (ended, used, limit) and orders nothing more.
- **A number reserved for an order that failed** is reused for the next order. So is the
  number of a ticket lost before any grant (a spot launch with no capacity) when no
  sampler started and its shard holds no record (`shard_given_back`). A number a member
  was granted, started or wrote on stays that member's.
- **Numbers that were issued but never ordered** (a controller that died or detached in
  between) are ended at the next takeover. Each gets an end record in the allocator and a
  data end record in its empty shard, so the trainer never waits for them. They are never
  reused.

### The member table

A pool on the store control plane publishes every shard it has ever assigned in
`fleet/members/<pool>.json`. `epoch` goes up by one on each change.

`run up` starts the trainer with `--island-env YETO_RL_MEMBER_TABLE=1`. With that
variable set, the trainer reads the tables (at most every 15 s) and adds new shards to the
shards it consumes. `--rl-buffer-shards` is still the initial set. The trainer never
drops a shard, and a shard listed as `retired` never holds the data end.

The trainer follows only the tables of its own lanes:

- `YETO_RL_MEMBER_LANES` lists them, separated by commas, or `*` for every table in the
  trainer's store. An island lane `i<k>` matches a `shard_base` that ends with `-i<k>`
  (a fleet names island k's pools `<base>-i<k>`). Any other lane `L` matches `L` and
  `L-...`.
- Without the variable, the lanes are the bases of the initial `--rl-buffer-shards`.
- A single trainer gets `*` from `run up`: its store prefix is its one lane, and a pool
  added while it trains has a new `shard_base`. A fleet's island k trainer gets `i<k>`.
- A table outside the lanes is ignored and reported once (`rl_member_table_ignored`).

At the end of the data, `run up` also ends every member-table shard that did not end (a
sampler that was fenced or lost).

## Pool workers, blocked pools, detach

On the store control plane the controller is a supervisor:

- **One worker thread per pool.** A pool whose provider call hangs is reported once as
  `pool_blocked` (`run status`: `blocked_s`; the dashboard shows a warning until
  `pool_unblocked`) while the others tick on.
- **A worker that dies is restarted** after a backoff, without holding the other pools.
  The pool reloads its state from the store and keeps every member it holds that the store
  copy does not (one ordered after a failed save).
- **Releases run in the background.** A tick only decides. A stop waits at most 60 s for a
  release still running; a hung one stays draining and is reported as left.
- **A stop or detach that gives up on a blocked tick** (after `join_s`) detaches that pool:
  when the tick returns it orders, starts and fences nothing more. Its state is saved and
  its store lock released only then, so whatever the tick did is recorded.

```bash
yeto run detach --config run.json   # the controller stops driving the pools; no drain
yeto run up --config run.json       # adopts every member under its shard; no sampler restarts
```

With `rollout.drain_on_exit: false`, a process exit also detaches and releases the store
locks, so the next `run up` does not wait for the lease to run out.

## Starting with the pools that pass

A `samplers` entry may set `"required": false`. If such a pool fails its pool dry-run
(credentials, capacity, the provider not answering), it is **degraded**: it is left out of
this `run up`, and the run starts with the other pools. A config error (the shard, field or
consistency checks) still refuses the start. The key works on both control planes.

You can see a degraded pool in:

- the `run up` log;
- `state.json` under `degraded_pools`;
- the `rl_pool_degraded` event;
- a dashboard warning.

`run up` still refuses to start when a `required` pool fails (the default), when no pool
is left, or when a fleet island is left without any pool.

The dashboard reads both the legacy and the neutral event names (below), so a tape written
before, during or after the compatibility window folds the same way.

## Pool event names

Pool events keep their `nc_elastic_<event>` name. On the store control plane every record
also carries `neutral_event: "rl_pool_<event>"`. The dashboard and the breaker registry
accept both names.

## The desired state

`fleet/desired.json` is the copy of the newest generation. Each generation is
`fleet/desired/<generation>.json`, created once.

- **To change it**, read generation `g` and write `g + 1`.
- **If two writers start from the same `g`**, the second write is refused with "read
  again".
- **`run up` seeds generation 1** from the run config. From then on the desired state wins
  over the run config's `pool` / `samplers`.

```json
{
  "schema": "yeto.fleet.desired/v1",
  "generation": 3,
  "run": "my-run",
  "members": [
    {"kind": "sampler_pool", "name": "nc", "required": true,
     "pool": {"provider": "nc", "config": "pool.nc.json", "target": 2}},
    {"kind": "sampler_pool", "name": "modal", "required": false,
     "pool": {"provider": {"kind": "modal", "image": "sglang:latest", "secret": "store", "bucket": "b"},
              "target": 1, "slots": [1, 2], "shard_base": "my-run-modal", "spec": "spec.modal.json",
              "run_prefix": "my-run", "r2_env_file": "store.env"}},
    {"kind": "trainer", "name": "island-1", "learner_id": 1,
     "source": {"kind": "modal", "gpu": "L40S"}, "shape": {"nodes": 1, "gpus_per_node": 1, "tp": 1, "pp": 1},
     "mode": {}, "fixed_roster": false}
  ]
}
```

| Field | Meaning |
|---|---|
| `members[].kind` | `sampler_pool` or `trainer`. |
| `sampler_pool.name` | The pool name: unique, no `/`. |
| `sampler_pool.pool` | A `samplers` entry of the run config without `name`: `provider`, `config`, `target`, `spec`, `interval_s`, `usd_per_h`, `data_path` and inline pool keys. |
| `sampler_pool.required` | As in a `samplers` entry. |
| `sampler_pool.retire`, `mode` | One-shot. They apply only with the generation that carries them (`yeto run scale --remove`). |
| `trainer` | Reserved for trainer islands (`learner_id`, source, shape, mode, fixed roster). The sampler control plane keeps these members as written and only checks `name` and an integer `learner_id`. The trainer worker reconciles them. |

### What the controller does with a new generation

The controller compares the new generation with the one it applied before, pool by pool:

| Change | Action |
|---|---|
| A pool added | A worker starts and orders. The trainer consumes the pool's shards from the member table without a restart. |
| `target` changed | The new target goes through `pool-scale` with its usual checks. |
| Anything else of the pool entry changed | **Rolling replacement.** New members start with the new config. Old members are replaced one at a time: make before break when a slot is free, otherwise one member is fenced and handed on at a time. At most one member is short at any time. The `shard_base`, the provider and the slots (`accounts`) cannot change in place: remove the pool and add a new one. |
| A pool removed | Its target goes to 0, and the scale-down fences and drains its members (their claims go back to the ledger). Then every shard the pool was issued is ended for good: a number never ordered gets a data end record in its shard, every shard gets an allocator end record and `retired` in the member table. Then its machines are released and confirmed ended, and the pool is dropped. |

The controller records each applied generation in `fleet/applied.json`. A controller that
starts after a detach or a crash builds the pools of that generation, including the ones it
was still removing, and then applies every newer generation. A change made while no
controller ran is never lost: a removed pool is still drained, retired and released.

The reconcile never waits for a pool's tick. A new config waits until the pool's tick lock
is free. The end of a removal runs on the pool's own worker.

A pool's own target changes between two generations are not undone. Examples are a
manual release the pool detected, or an automatic scaler.

## Commands

```bash
yeto fleet show --config run.json                         # the newest generation
yeto fleet show --config run.json --generation 2          # an older one
yeto fleet show --config run.json > desired.json          # edit this file, then:
yeto fleet apply --config run.json --file desired.json --generation 3 --dry-run   # what would change
yeto fleet apply --config run.json --file desired.json --generation 3 --note "add a Modal pool"
yeto run scale --config run.json --pool nc --samplers 4   # writes the pool's target in the desired state
yeto run scale --config run.json --pool nc --remove <ticket>
```

`yeto fleet apply` exits with 3 when the store is no longer at `--generation`. Run
`yeto fleet show` again and apply your change to that generation.

A scale up (a target raised, a pool added) goes through the run's own check first, like
`yeto run scale`: the run's breaker and the budget. If the check refuses it, `fleet apply`
exits with 4 and writes nothing.

`yeto run down` stops every pool the fleet drives: the desired state's, the ones it is
still removing, and the run config's. A pool state in the store with live members that
none of them names is reported, and `run down` is then not verified.

### Adding a pool

1. Run `yeto fleet show --config run.json > desired.json`.
2. Append a `sampler_pool` member. Give it a new `name` and a `shard_base` no other pool
   uses.
3. Run `yeto fleet apply --config run.json --file desired.json --generation <g> --dry-run`.
   It should say `add pool <name>`.
4. Run the same command without `--dry-run`.

### Removing a pool

Delete the pool's member from the file and apply it. The pool drains its members, retires
its shards and releases its machines. `run status` no longer lists it.

### Changing a pool's config

Edit the member's `pool` entry (for example its `spec`) and apply it. The members are
replaced one by one.

## Code

- `yeto/launch/pool/store_lock.py`: the lock.
- `yeto/launch/pool/store_state.py`: the pool state.
- `yeto/launch/pool/shard_alloc.py`: the shard allocator.
- `yeto/rl/engine/member_table.py`: the member table.
- `yeto/launch/pool/desired.py`: the desired state.
- `yeto/launch/pool/fleet_reconcile.py`: the reconcile.
- `yeto/launch/pool/fleet_cli.py`: `yeto fleet`.
