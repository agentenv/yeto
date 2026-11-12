# National Compute sampler pool

`python -m yeto.launch.nc pool` keeps a set of sampler nodes on National
Compute (NC) running for a whole RL run. Each node runs one engine sampler
that writes its own shard of the run's sample buffer in your object store.
The pool orders nodes, starts the samplers, replaces nodes before their
leases end, and recovers from evictions, lost nodes and crashed samplers.

> **Billing warning.** NC does **not** end a lease at `lease_until`. When
> `lease_until` passes, the ticket stays `active` and keeps billing until it
> is deleted. The pool always deletes its nodes before `lease_until`, but
> only while its control loop is running. If the loop is gone, the nodes
> keep billing until you run `pool-stop` (see [Stopping](#stopping-a-pool)).

This page covers:

- [Before you start](#before-you-start)
- [The control host](#the-control-host)
- [The pool config](#the-pool-config-pooljson) and every field's default
- [Commands](#commands): `--dry-run`, `pool`, `pool-status`, `pool-stop`
- [What the pool does](#what-the-pool-does)
- [Known limitations](#known-limitations)
- [Billing](#billing)

## Before you start

You need:

- **NC accounts.** An NC account holds **one node at a time**. To run N
  samplers you need at least N accounts. To have a replacement ready before
  each lease ends (with no gap in sampling), you need N + 1. Provide
  **N + 2**: the pool takes an account out of use (a
  [halt](#a-halted-account-clear-halt)) when it cannot prove what one of its
  deletes released, until a person has checked it.
- **The NC order tools** `nc-manage` and `nc-query`, plus the controller tool
  directory, which provides the workspace relay. The pool reaches nodes over
  SSH through a small relay in each account's NC workspace.
- **An object store bucket** (S3 or R2) that is shared with the trainer.
- **A sampler spec** for the engine sampler. Start from
  `examples/k31/sampler-spec.nc-mi355x-vllm.json`.

Set these environment variables on the control host:

| Variable | Meaning | Default |
|---|---|---|
| `YETO_NC_ACCOUNTS` | Comma-separated list of the NC accounts this tool may order on. Required. An account in `pool.json` that is not on this list is refused. | none |
| `NC_MANAGE`, `NC_QUERY` | Command lines of the order tools. Can be overridden in `provider`. | `nc-manage`, `nc-query` |
| `NC_TOOL_DIR` | Directory of the controller tool (`tool/core.mjs`, `tool/wsrun.mjs`). | none |
| `NC_NODE_BIN` | The `node` binary that runs the tool. | `node` |
| `YETO_NC_STATE_DIR` | The pool's state directory: ticket ledger, leases, pool state, events, pid and stop files. | `~/.yeto/nc` |
| `YETO_RESOURCE_LEDGER` | The run's cost ledger (set by `yeto launch` for its processes; set it yourself for a pool started by hand, e.g. `~/.yeto/runs/<run>/resources.jsonl`). Every order, grant and release of an NC ticket is recorded there as an `nc_lease`, so `yeto cost`, the budget and `yeto cleanup` see the pool's nodes. | none (not recorded) |
| `YETO_NC_RELAY_DIR` | Directory of the relay inside the NC workspace. | the workspace home `/data/home/.yeto-nc` |

The **store env file** (`r2_env_file`) must be mode `0600`. It holds the
sampler's store credentials, in this format:

```
AWS_ACCESS_KEY_ID=<key id>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_ENDPOINT_URL=https://<endpoint>
AWS_REGION=auto
R2_BUCKET=<bucket>
```

The bucket variable is named `R2_BUCKET` by default. If your env file names
it differently, set `bucket_env` in the pool config. The samplers write to
`s3://<bucket>/runs/<run_prefix>`.

## The control host

The pool is a **foreground loop**. It decides every order, replacement and
release, so it must run for the whole RL run. Run it:

- on an always-on machine (a small VM is enough; it needs no GPU), not on a
  laptop;
- under `tmux`, `screen` or a `systemd` unit, so an SSH disconnect does not
  kill it.

Once the loop stops, nothing replaces or releases the nodes. What happens
depends on how it stopped:

| How the loop ends | What happens to the nodes |
|---|---|
| `pool-stop`, `SIGTERM`, `SIGINT` (Ctrl-C), `SIGHUP` (the terminal closed) | The loop drains every node (stops the sampler, deletes the ticket, confirms it reads `ended`), then exits. If a drain keeps failing, it prints `NODES STILL BILLING` with the tickets and the stop command. |
| The process crashes or is killed with `SIGKILL`, or the host loses power or network | **The nodes keep billing**, past `lease_until`. Run `pool-stop --config <file>` from the same host and state dir as soon as you can. `pool-status` reports this case with a `NODES STILL BILLING` banner. |
| The control host is gone for good | The state dir is gone with it. Release each ticket in the NC console, or run `python -m yeto.launch.nc down --account <account> --ticket <ticket>` from a host with the same ledger. Keep `YETO_NC_STATE_DIR` on a disk that outlives the host. |

The pool refuses to start a second loop on a pool whose loop is still
running.

### The state files

The state dir holds two files that the pool and every manual command on the
same state dir write:

- `tickets.json`, the **ticket ledger**: one row for each ticket this tool
  created. `down`, `pool-stop` and the patrol release only tickets in it.
  An order is written to it before it is sent, as a row with
  `"ticket": null` (a *pending order*). The ticket id is filled in from the
  order's own answer, and from nowhere else.
- `leases.json`: the address and lease end of each granted node.

Each change is one read-modify-write under a lock (`tickets.lock`,
`leases.lock`), so the pool and a manual command can run on the same state dir
at once and no row is lost. Each write goes to a temporary file of its own,
which then replaces the old file. Both files are mode `0600`. Leave the
`.lock` files in place.

A ledger that exists but cannot be read (not JSON, or not a list of ticket
rows) is an error. It is never taken for an empty ledger. Every command that
needs the ledger fails with `ticket ledger <path>: ...`, and the file is left
as it is. The tickets in it may still be billing. To recover:

1. Stop the pool loop if it runs (it cannot release anything meanwhile).
2. Copy `tickets.json` aside, then repair it by hand. It is a JSON list of
   rows; each row needs `account` and `ticket`, and `run` for the pool to find
   it. Mark a ticket that has ended with `"ended": true`.
3. If it cannot be repaired, read the open tickets from
   `pool-<shard_base>.json` and the NC console, and release them in the
   console. Then move the broken file away. The next order starts a new
   ledger.

The trainer has the same requirement: a no-sync trainer runs with
`--controller local`, so its control loop also lives on your host for the
whole run.

## The pool config (`pool.json`)

A minimal config:

```json
{
  "target": 2,
  "accounts": [<account-a>, <account-b>, <account-c>],
  "shard_base": "nc",
  "run": "<run>",
  "machine_type": "mi355x",
  "spec": "sampler-spec.json",
  "run_prefix": "<run-prefix>",
  "r2_env_file": "<path>/store.env",
  "sampler": {"dispense": {"backend": "store", "namespace": "<run-prefix>/task-dispense"}},
  "model_pull": {"src_key": "models/<model-dir>"},
  "elastic": {"max_run_spend_ucr": 2000000000}
}
```

The pool refuses unknown keys, at the top level and inside `sampler`,
`model_pull`, `node_hooks`, `backlog`, `elastic` and `provider`. A typo is an
error, never silently ignored.

### Top-level fields

| Field | Meaning | Default |
|---|---|---|
| `target` | Samplers to keep running. Must be at least 1 and at most the number of `accounts` (one node per account). | required |
| `accounts` | NC accounts the pool may order on. No repeats; each must be in `YETO_NC_ACCOUNTS`. Accounts beyond `target` are used for replacements. | required |
| `shard_base` | Prefix of the buffer shards (`<shard_base>-0`, `-1`, ...). Also names the state file `pool-<shard_base>.json`. | required |
| `run` | Run tag written to every ticket in the ledger. The pool and `pool-stop` only ever delete tickets that carry this tag. One word, no spaces. | `shard_base` |
| `machine_type` | NC machine type to order. This is **not** a Hugging Face model; the served model is set in the sampler spec. A value containing `/` is refused. | `mi355x` |
| `model` | Deprecated name of `machine_type`. It still works but prints a deprecation note; it cannot be set together with `machine_type`. | none |
| `route` | `ssh`: SSH + docker on the node (the verified route). `sky`: SkyPilot SSH node pool (refused on nodes with the provider's health k3s). | `ssh` |
| `spec` | Sampler spec file. `__RUN_PREFIX__` in it is replaced by `run_prefix`. | required (ssh) |
| `run_prefix` | The run's path under `s3://<bucket>/runs/`. | required (ssh) |
| `r2_env_file` | Store env file (mode `0600`), see above. | required (ssh) |
| `bucket_env` | Name of the bucket variable in the store env file. | `R2_BUCKET` |
| `image` | Sampler container image. | the pinned digest of the K3.1 ROCm vLLM image (`yeto.rl.sampler_pins.VLLM_SAMPLER_IMAGE_ROCM`) |
| `protected_prefixes` | Run prefixes the sampler must never write to (for example, another live run's prefix in the same bucket). | `[]` |
| `agentic_env_file` | Second env file for agentic samplers (sandbox hosts, records proxy). | none |
| `sampler` | Per-run sampler knobs, see below. | `{}` |
| `model_pull` | Pulls the model from the bucket onto each new node, see below. | none (the model must already be on the node) |
| `backlog` | Sizes `max_backlog_groups` from the sampler layout, see below. | none (the spec's value) |
| `node_hooks` | Commands to run on each new node and before each in-place restart, see below. | `[]` |
| `data_slots` | Split the prompts into N fixed slots (`i % N`). Must be at least `target`. Leave it unset when you use the dispenser. | none |
| `replace_lead_s` | How long before a lease ends the pool orders its replacement. Cover node setup, the model pull and the engine start. | `3600` |
| `end_margin_s` | A node is drained this long before `lease_until`, whether or not its replacement is ready. | `600` |
| `stop_below_ucr` | When the shared balance drops below this, the pool stops ordering and drains every node. | `20000000` |
| `min_balance_to_order_ucr` | Minimum balance for each new order. | `stop_below_ucr` + one node hour |
| `event_command` | Shell command that gets each pool event as JSON on stdin (a pager, a chat hook). A failure is logged and never stops the pool. | none |
| `probe_timeout_s`, `probe_connect_timeout_s` | Time limits of the per-tick SSH probe of each node. | `90`, `20` |
| `provider` | `{"kind": "nc", "manage": ..., "query": ..., "tool_dir": ..., "node_bin": ...}` overrides the NC tools. `{"factory": "package.module:function"}` plugs in another supply of nodes. | `{"kind": "nc"}` |
| `model_src`, `model_dest`, `workdir` | Model pull and work directory of the `sky` route. | none |
| `comment` | Free text, ignored. | none |

### `sampler`

These knobs override the sampler spec on every node of the pool:

| Field | Meaning |
|---|---|
| `dispense` | The store task dispenser (`prompts.dispense`), the same on every sampler. See [DISPENSER.md](DISPENSER.md). |
| `n` | Samples per prompt. |
| `max_tokens` | Response length cap. |
| `thinking_effort` | Thinking effort passed to the engine. |
| `drop_zero_variance` | Drop groups whose rewards are all equal. |
| `reload` | How a new policy release is loaded: `hot` or `restart`. |

### `model_pull`

| Field | Meaning | Default |
|---|---|---|
| `src_key` | Model directory under the bucket. You upload it there once, before the run. | required |
| `dest` | Path on the node. | `/mnt/nvme/models/Kimi-K3.1-Pro` |
| `include` | Glob of the files to pull. | all files |
| `max_mb_s` | Pull rate limit. | `2500` |
| `timeout_s`, `poll_s` | Pull deadline and status poll interval. | `3600`, `30` |

### `backlog`

The fields `synced` (required), `trainers`, `fragments`, `local_rounds`,
`pipeline` and `explicit_rounds` size `max_backlog_groups`, the same way the
launcher sizes it (`--synced`, `--trainers`, ... of `ssh-sampler start`).

### `node_hooks`

Each hook is `{"name": ..., "command": ..., "where": "node" | "local",
"timeout_s": 600}`. The command is formatted with `{alias}`, `{ip}`,
`{ticket}`, `{account}`, `{user}`, `{shard}` and `{run}`. Each hook must be
idempotent and exit 0; a non-zero exit fails the node start.

### `elastic`

| Field | Meaning | Default |
|---|---|---|
| `max_run_spend_ucr` | Cap on the run's estimated spend; no order is placed past it. Set it. | none (no cap) |
| `max_nodes` | Most billed nodes at once. | `2 x target` (in practice at most one per account) |
| `max_orders` | Most orders over the whole run. | none |
| `unreachable_grace_s` | How long probes must fail before a node counts as lost. Shorter failures count as network blips. | `300` |
| `fence_alert_failures`, `fence_alert_s` | A shard fence that fails is tried again every tick. After this many failures, or this long without success, the pool sends `fence_stuck` once and keeps trying. | `5`, `600` |
| `fence_order_ahead_s`, `fence_blocked_hold_s` | While a shard's fence is owed, a node to continue the shard is ordered only in the first `fence_order_ahead_s` of the failure. A granted node whose start the fence holds back is released after `fence_blocked_hold_s` (`fence_blocked_node_released`) and ordered again once the fence succeeded. | `300`, `1800` |
| `fence_max_hold_s` | How long a node is kept (and billed) while its shard fence fails. After that the node is released (`fence_abandoned_node_released`); the fence is still tried and nothing continues the shard before it succeeds. `null`: no limit. | `7200` |
| `control_plane_alert_s` | How long NC status calls must fail before an alert. The pool never acts on a failed status call alone. | `600` |
| `stall_s` | A joined sampler whose event tape has not grown for this long counts as crashed. Set it above your longest round. | none (off) |
| `max_sampler_restarts` | In-place restarts of a crashed sampler before its node is replaced. Restarts before the node's first join and before a `pool-retry` do not count. | `2` |
| `sampler_restart_reset_s` | A sampler that ran this long since it joined starts a fresh restart count when it crashes; `null`: the count never resets. | `1800` |
| `start_timeout_s` | Time allowed from the order to a joined sampler. | `3600` |
| `overdue_grace_s` | How long a ticket may read `active` past `lease_until` before the pool deletes it. | `0` |
| `lost_handoff_wait_s` | How long the replacement of a lost node waits for its predecessor's handoff. | `60` |
| `breaker_failures`, `breaker_cooldown_s`, `breaker_max_cooldown_s`, `breaker_max_trips` | Circuit breaker on consecutive provisioning failures: it opens for a cooldown that doubles on each trip; after `breaker_max_trips` trips, ordering stops for good. | `3`, `600`, `3600`, `3` |
| `stop_on_finish` | A sampler that exits 0 (its data ended) stops the whole pool. | `true` |

## Commands

### Check a config: `pool --dry-run`

```
python -m yeto.launch.nc pool --config pool.json --dry-run
```

The dry run validates the config, renders the sampler spec, checks the store
env file (it exists, has mode `0600`, and contains the bucket key) and the
order tools, then prints:

- the orders the pool would place at start;
- the accounts left over for replacements;
- the estimated spend per hour: steady, and the peak while replacements
  overlap.

It orders nothing and calls no NC API. Exit codes: 0 (ok), 1 (problems
found), 2 (invalid config).

### Run: `pool`

```
python -m yeto.launch.nc pool --config pool.json [--interval 60]
```

This runs the control loop in the foreground, one tick every `--interval`
seconds. The loop writes:

- the state to `$YETO_NC_STATE_DIR/pool-<shard_base>.json`;
- events to `events-<run>.jsonl`;
- its pid to `pool-<shard_base>.pid`.

If the state file says the pool stopped, the loop exits at once. Move the
file away to start a new pool with the same `shard_base`.

### Inspect: `pool-status`

```
python -m yeto.launch.nc pool-status --config pool.json [--events 20]
```

This prints the loop's process (`alive`), the members that may still hold a
ticket (`billed`, with `lease_until` and `past_lease_until`), the state file
and the last events. It is read-only. If the state file holds members and no
loop is alive, it prints the `NODES STILL BILLING` banner and exits 1.
It also prints `target` (the target the loop holds, from `pool-scale` when one
was given, else `pool.json`), `nodes` (members that may hold a ticket) and
`samplers` (`nodes/target`). `yeto run status` shows the same pair under
`samplers`.

### Change the number of samplers while the run trains: `pool-scale`

```
python -m yeto.launch.nc pool-scale --config pool.json (--target N | --add K | --remove <ticket>) \
    [--retire <ticket> ...] [--drain | --fence] [--force-static] [--dry-run]
yeto run scale --config run.json (--samplers N | --add K | --remove <ticket>) ...   # from the run config
```

`--add` and `--remove` are relative to the target recorded when the command
runs (`--remove T` releases T and lowers the target by one). A scale needs the
store task dispenser (`sampler.dispense` or the spec's `prompts.dispense`);
with static `data_slots` it needs `--force-static` (a removed node's slot then
waits in the pool's pending list until a scale up continues it).

`pool-scale` writes the wanted target to `pool-<shard_base>.scale.json` in the
state dir. It never takes the pool lock: a running loop keeps it and reads the
file at its next tick (within `--interval` seconds). A loop started later reads
it at start, so the target outlives a restart; it takes precedence over
`target` in `pool.json`. Concurrent `pool-scale` commands serialize on
`pool-<shard_base>.scale.lock` (checks included); relative requests add up.

- **Scale up.** The loop orders the missing nodes like any fill. `pool-scale`
  refuses (exit 2, nothing changed) before anything is ordered when the new
  target is above the usable accounts (`accounts` inside `YETO_NC_ACCOUNTS`; one
  node per account), `data_slots` or `elastic.max_nodes`, when the order breaker
  is tripped, or when the new nodes would pass `elastic.max_orders`,
  `elastic.max_run_spend_ucr` (one node hour each) or `shard_limit` (the
  trainer's `--rl-buffer-shards` / the tunnel port range; shard indexes are
  never reused). The loop's own order gates still apply.
- **Scale down.** The loop picks the members to release: the `--retire`
  tickets first, then members not yet ready, then the newest, cheapest and
  shortest-lease ones. In `drain` mode (`elastic.scale_down`, default, or
  `--drain`) a ready member is asked through the dispenser ledger
  (`<namespace>/drain/<shard>`) to claim no new task (`nc_elastic_scale_draining`);
  it keeps counting as billed but not toward the target, and goes on once its
  sampler answers that it holds no claim (`<namespace>/drained/<shard>`) or
  after `elastic.scale_drain_timeout_s`. A raised target meanwhile keeps a
  draining member instead of ordering (`nc_elastic_scale_drain_cancelled`). In
  `fence` mode (`--fence`), and for a member not ready, it goes at once. Going
  means: fence the shard (its dispense claims go back at once), stop the
  sampler, release and delete the ticket, check it reads `ended` and emit
  `nc_elastic_scaled_down` with the `outcome`. A replacement in flight for
  that member goes with it. Nothing is handed on and nothing is ordered for
  it; the other members are not touched. `--target 0` releases every node and
  keeps the loop running; a negative target is refused. `--retire` needs a
  target below the current member count.

Manual operations take priority over the loop's automation. `nc down --ticket T`
on a member of a running pool becomes `pool-scale --remove T --fence` (with no
loop alive it deletes the ticket and lowers the recorded target). A member
whose ticket ends `released` without this tool deleting it (the NC console)
lowers the target by one and is not replaced (`nc_elastic_manual_release_detected`;
`elastic.external_release: "replace"` restores replacing it); evictions
(`preempted`, `lost`, `admin`) are replaced as before. `pool-pause --config F
--ticket T [--hold-tasks]` / `pool-resume` stop and restart every automatic
action on a member (it keeps counting toward the target). A request with
`--source auto` is refused for `elastic.manual_hold_s` after a manual one.
All of it is kept in the same control file and logged as `nc_elastic_manual_*`;
`pool-status` prints it under `manual`. The customer guide describes it:
[KIMI_K31_ASYNC_RL_GUIDE.md, section 5.6](KIMI_K31_ASYNC_RL_GUIDE.md#56-manual-operations-take-priority).

The pool process must run code that has `pool-scale` (restart an older loop
once; H1 adopts its nodes). After that, no restart is needed to scale.

### Stopping a pool

```
python -m yeto.launch.nc pool-stop --config pool.json [--wait 900]
```

- **The loop is running:** `pool-stop` writes a stop file. The loop picks it
  up within one tick, drains every node and exits. `pool-stop` waits up to
  `--wait` seconds for that.
- **No loop is alive** (it crashed, was killed, or the host rebooted):
  `pool-stop` releases the nodes itself. It releases the members recorded in
  the state file, plus any open ticket the ledger holds for this run's tag.

Either way, `pool-stop` only deletes tickets that this tool created, are
tagged with this run, and are still the account's current ticket. It deletes
them by id and confirms each one reads `ended`. It never uses `--all` and
never touches another run's or another person's ticket. Exit code 0 means
every node is released. Exit code 1 prints the banner with the tickets left.

### An unrecorded order: `adopt`

```
python -m yeto.launch.nc adopt --account <account> --ticket <ticket> --yes
```

Use it after an `nc_elastic_order_unrecorded` event whose `action` names the
`adopt` command, or after an order that failed with `ticket <id> may be that
order ... not recorded`. The account shows an open ticket, the ledger has a
pending order of this tool on that account, and nothing ties the two
together. Only a person can decide:

1. Open the ticket in the National Compute console. Check that it was placed
   at the time of the event and that nobody else ordered on the account.
2. If it is this tool's order, run `adopt`. The pending order becomes the
   ticket's row (`adopted_by: "operator"`, with the run tag of the order). A
   running pool's patrol then finds it as an open ticket of the run that no
   member owns: it takes the node in as a member, or releases it when the pool
   is stopping or has `release_orphans`. `pool-stop` releases it too. Without
   a pool, release it with `down --account <account> --ticket <ticket>`.
3. If it is someone else's ticket, leave it. If it is nobody's, release it in
   the console.

`adopt` refuses unless all of this holds:

- the ticket is the account's current open ticket and is not in the ledger;
- the ledger has a pending order of the account, its call is over, and this
  tool saw this very ticket for it within 30 minutes of the order (that is the
  ticket the event named);
- it is not the ticket the account showed before the order;
- if the ticket has a grant time: it was granted no earlier than 5 minutes
  before the order was sent and no later than 4 hours after the order call
  ended.

Without `--yes` it only prints what it would do. It never deletes anything.

A pending order is closed, and can no longer make any ticket adoptable, when:

- the account reads free twice in a row after the order call (the order
  placed nothing that is open); or
- a later order on the account succeeds.

A ticket first seen more than 30 minutes after the order is reported once,
with `stale: true` and no `adopt` command. It cannot be adopted: handle it in
the console.

The ledger keeps one pending row per account (`"ticket": null` with an
`order` id): `owner` is the host and process that sent the order,
`created_at` the last order sent, `since` the first one without a recorded
outcome, `call_ended` when the failed call returned (absent: the process died
inside the call; the call then counts as over 6 minutes after `created_at`),
`error` its error, `free_reads` the free reads counted so far, and
`observed_ticket` / `observed_at` the unrecorded ticket reported.

### A halted account: `clear-halt`

```
python -m yeto.launch.nc clear-halt --account <account> [--yes]
```

The order tool's delete cannot name a ticket (see
[Known limitations](#known-limitations)). When the pool cannot prove that a
delete it sent released its own ticket, it **halts the account**: a halt row
in the ledger. While the row is there, every order, delete and adopt on the
account is refused, by the pool and by the CLI. The other accounts keep
working. A halt never ends by a timeout.

There are two reasons. Each comes with an event (`severity: critical`) that
carries `account`, `reason`, `expected` (the ticket the delete was meant for)
and `action` (the command to run):

| Event | What happened | How it ends |
|---|---|---|
| `nc_elastic_foreign_ticket_released` (`ticket`) | A delete of this tool released `ticket`, which is not its own: the pool's ticket ended on its own and someone else ordered in the instant before the delete. | Only by `clear-halt --yes`. |
| `nc_elastic_delete_unconfirmed` (`delete_sent_at`) | A delete was sent, and the pool's ticket is gone from the account without ever reading `ended` (the account shows no order, or a newer ticket). The pool cannot tell which ticket the delete released. Most often it was the pool's own. | By `clear-halt --yes`, or by itself when a later status read shows the `expected` ticket, by id, ended (event `halt_cleared`). |

What to do:

1. Run `clear-halt --account <account>` without `--yes`. It prints the halt
   (reason, `expected_ticket`, `released_ticket` or `delete_sent_at`) and the
   ticket the account holds now (id, state, `lease_until`). It changes nothing.
2. In the National Compute console, look at the account's tickets around the
   time of the halt:
   - `foreign_ticket_released`: find who placed `released_ticket` and tell
     them their node was released. Check that the `expected` ticket is ended.
   - `delete_unconfirmed`: check that the `expected` ticket is ended and that
     no other ticket of the account was released at `delete_sent_at`. If one
     was, treat it as the first case.
   - If the `expected` ticket is still active, release it in the console (the
     pool will not touch the account while it is halted).
3. Run `clear-halt --account <account> --yes`. The pool uses the account
   again at its next tick.

**Which halts to expect.** A `delete_unconfirmed` halt after a normal release
needs two things at once: the delete's answer does not name the ticket, and
the ended ticket is never shown by a status read afterwards. How often
National Compute does that is not measured. It is expected to be rare, by
inference from a small sample: in the ticket ledgers of earlier runs on one
control host, all 19 tickets this tool released had read back as `ended`,
`released` (19 of 19; no larger sample exists). In the first days of a run,
read the fields of each `delete_unconfirmed` event (they are in the halt row
too) to see which case it was:

| Field | Values |
|---|---|
| `delete_answer` | `received`: the delete answered ok, without a ticket id. `none (<error>)`: the call failed or its answer was lost. `unknown`: the process ended inside the call. |
| `seen` | `no_order`: the status shows no ticket. `other_ticket`: it shows `seen_ticket` in `seen_state`. |
| `delete_attempts`, `delete_sent_at` | How many times the delete was sent for this ticket, and when first. |

Many halts with `delete_answer: received` and `seen: no_order` would mean
that National Compute stops showing a released ticket at once; report that,
it is what a `--ticket` argument of the order tool would fix.

While an account is halted (or a delete sent on it is still unconfirmed), it
does not count as free: the pool orders on its other accounts and says so
once (`account_unavailable`).

Until then the pool has one account less. With no spare account, a node of
the pool that reaches its lease end is not replaced. See
[Before you start](#before-you-start) for how many accounts to provide.

### Retry after a configuration error: `pool-retry`

```
python -m yeto.launch.nc pool-retry --config pool.json
```

Fix `pool.json` or the sampler spec first. `pool-retry` checks the fixed
config the same way `--dry-run` does and refuses (exit 2) if it is still
wrong. If the config passes, it asks the loop to re-read the config and
restart the sampler on the same node(s) at its next tick. The shard, the data
and the model already on the node are kept. If no loop is alive, start
`pool` again; it picks up the request.

To release a single ticket:

```
python -m yeto.launch.nc down --account <account> --ticket <ticket>
```

### Upgrading and rolling back yeto: the state file format

The pool state file carries a `format` number. Format 1 (no `format` key) is
what every yeto before the fence retry writes. Format 2 adds the member state
`fencing` and a few member fields (the fence still owed, grace and restart
bookkeeping).

- **Upgrade.** Nothing to do: a newer yeto reads the older file and writes
  the newer format at its next save.
- **Rollback.** A yeto that writes format 1 cannot read a format 2 file: it
  fails on the unknown member fields with a `TypeError`, in `pool` and in
  `pool-stop` alike. Before you install the older version, stop the pool
  loop and run, with the newer version:

  ```
  python -m yeto.launch.nc pool-state-downgrade --config pool.json
  ```

  It writes the state file as format 1 and keeps the original next to it
  (`<name>.format2.bak`). It is refused while a pool loop is alive, while a
  member is `fencing` (wait for the fence, or run `pool-stop` first), and
  while a shard fence is still owed; `--drop-owed-fences` gives those fences
  up and prints each shard left unfenced.
- A yeto that meets a state file of a format newer than its own says so and
  names this command instead of guessing.

## What the pool does

On each tick, the pool reads every member's ticket and probes its node with
one SSH round trip. Then it acts:

- **Lease end.** A replacement is ordered on a free account `replace_lead_s`
  before the end. It continues the old shard's prompts. Once the replacement
  is ready, the old node is drained. A node is always drained `end_margin_s`
  before `lease_until`. With no free account, there is a gap until the old
  node is gone.
- **Eviction, lost node, overdue lease.** The node's shard is fenced, so a
  node that comes back cannot add groups the trainer would take. The ticket
  is deleted. The next order continues the shard.

  The fence is a record in the shard itself: the pool appends an ownership
  record `pool-fence:<ticket>` at the shard's next position
  (`sample-buffer/shards/<shard>/<position>.json`, created only if the
  position is free; event `rl_shard_owner`). Every group the node created
  before that record is trained; nothing it writes after it is, and it cannot
  mark the shard as ended. The sampler's own write reads the shard first and
  is refused, and the trainer applies the same rule when it merges. No clock
  is compared, so a clock difference between the control host and a node does
  not matter. The pool also retires the sampler in the dispenser ledger, so
  its open claims go back at once. A sampler restarted in place after an
  engine stall is fenced the same way, without the stop request; the
  restarted process takes the shard again with its own ownership record.

  On a run whose trainer runs older code (no trainer generation lease in the
  store), on a shard that holds no ownership record yet (its sampler joined
  before the new trainer, or runs older code, and may not read one), and on
  the `syncer` data path, the pool writes no ownership record:
  the fence is the lease record `sample-buffer/leases/<shard>.json` and the
  syncer's member table, as before. See "Upgrading the code of a running
  run" in `KIMI_K31_ASYNC_RL_GUIDE.md` for mixed versions.
- **A fence that fails.** The fence comes before everything else and is tried
  again every tick (`fence_failed`, then `fence_stuck`). Until it succeeds the
  member shows as `fencing`: its node is kept (and billed), nothing is handed
  on and no replacement is ordered. A node that is gone anyway (ticket ended,
  lease over, released by a person, `pool-stop`, or kept for
  `fence_max_hold_s`) is released without the
  fence; the fence is still tried every tick, and the node that continues the
  shard is ordered but its sampler starts only after the fence succeeded
  (`start_waits_for_fence`).
- **Configuration error.** Before it orders anything, `pool` (and
  `--dry-run`) renders the sampler spec and builds locally every part the
  sampler builds at start: the island config (required keys such as
  `max_backlog_groups`, sampling parameters), the engine adapter from the
  `engine` section (unknown section keys, server fields and engine options
  are refused; no engine process starts), the `rollout_fields`, and the reward
  (`reward_kind` and the import of `reward`). A spec that fails is refused, so
  you never wait for node setup to find out. The reward must be importable on
  the control host too. If a sampler still exits with code 2 on a node (its spec or
  arguments were refused), the pool treats it as a configuration error, not a
  crash. It does not restart the sampler, fence its shard or replace the node.
  It logs `CONFIGURATION ERROR`, emits `sampler_config_error`, and orders
  nothing more. The node is kept (and billed) until you fix the config and run
  `pool-retry`, or until its lease end, when it is drained as usual.
- **Crashed sampler.** The sampler is restarted in place, up to
  `max_sampler_restarts` times. After that, the node is replaced. A
  `docker inspect` that fails (sudo, dockerd not answering) is not a crash:
  it counts as a probe that did not reach the node (`unreachable_grace_s`).
  A ready sampler whose container is reported missing is restarted when the
  next probe reports the same.
- **Sampler exits 0: the data ended.** When the store dispenser has settled
  every task, the sampler prints `data ended` and exits 0. The pool treats
  this as the normal end of the run. It does not restart the sampler, does
  not replace the node and places no new order. With `stop_on_finish`
  (default), it drains every node and stops. Without it, the other samplers
  finish their data or go at their lease end, and then the pool stops.
  A fenced sampler also exits 0, so on `data_path: r2` the pool first reads
  the run store: with a stop request (handoff record) on the shard, a pool
  fence on its incarnation lease, or a `FENCED` leave of its newest
  incarnation, the data has not ended (`sampler_exit_not_data_end`). The node
  is retired and its shard continued (stop request), or the sampler is
  restarted in place like a crashed one. If the store does not read, nothing
  is decided (`exit_evidence_unreadable`) until it does.
- **A halted account.** When a delete of this tool released, or may have
  released, a ticket that is not its own, the ticket ledger halts the account:
  every order and delete on it is refused until an operator runs
  `python -m yeto.launch.nc clear-halt --account N --yes`. The pool leaves a
  halted account out (`slot_halted`, once) and orders on the other accounts;
  a refusal by the guard is not a provisioning failure and never opens the
  breaker. A member on a halted account is kept as it is, still billed
  (`release_blocked`); the pool releases it once the halt is lifted, and
  lets it go at once when its ticket has ended anyway (nothing to delete). With
  fewer usable accounts than the target it sends `slots_below_target`
  (critical): the pool runs short and a node at its lease end has no account
  for its replacement. `pool-status` and `yeto run status` list halted
  accounts, deletes not confirmed yet and orders without a recorded outcome
  under `accounts_attention` (and as `ATTENTION:` lines).
- **Stray tickets.** A patrol deletes any open ticket of this run that no
  live member owns.
- **Only its own tickets.** The pool and the CLI delete, stop and reach a
  ticket only when the ledger proves it is theirs: the row was written before
  the order was sent and holds the ticket id from that order's answer, or an
  operator adopted the ticket (`adopted_by: "operator"`). Any other ticket on
  the account is only read and reported.
- **An order whose answer was lost.** If the order call fails (a timeout,
  output that cannot be parsed, an error after the request), National Compute
  may still have placed the order. The pool does not guess. The pending order
  stays in the ledger, and when the account then shows an open ticket that no
  ledger row holds, the pool emits `order_unrecorded` (with the account and
  the ticket id) once, and the order fails with `not recorded`. The node of
  such a ticket is never looked up or reached. That ticket is
  **billing and nobody will release it**: the pool places no further order on
  the account and never deletes the ticket. See
  [`adopt`](#an-unrecorded-order-adopt). When the order tool answers
  `created: false` (someone else's order is on the account), nothing is
  recorded either.
- **Releasing a node.** The order tool's delete takes no ticket id: it
  releases whatever ticket the account holds when it runs. So before each
  delete the pool reads the account twice, the second time right before the
  call, and sends the delete only if both show its own ticket, open. The
  delete is written to the ledger before it is sent (a row with
  `pending_delete`; `phase` is `intent` until the call is about to start,
  then `sent`) and stays there until the account shows that ticket
  `ended` (the ticket's row then gets `release_confirmed_by`). A record
  still at `intent` means no delete was sent; it is dropped without a halt. Every later status read of the account, by any command and however
  much later, settles it; until then the account takes no other write. It sends
  no delete within 120 s (`YETO_NC_DELETE_GUARD_S`) of the ticket's
  `lease_until` or `preempt_at`: the release fails with `no delete sent now`
  and is retried after that point. Keep `end_margin_s` well above this guard
  (the default is 10 minutes), so the normal drain before the lease end is
  not deferred.
- **A status answer the pool cannot use.** The pool acts on an NC status
  answer only when the order tool succeeded (exit 0) and the answer has a
  `ticket` field that is `null` (the account has no order) or a ticket with a
  `state`. A failed call, no output, an error document, an answer without a
  `ticket` field and a ticket without a `state` all count as a failed status
  call: nothing is decided on it, a release in progress fails and is retried,
  and after `control_plane_alert_s` the pool alerts. A valid answer that does
  not show the member's ticket (no order, or another ticket id) is read a
  second time. The lease counts as ended, and a release records the ticket as
  already ended, only when both answers agree. After a delete, the release is
  confirmed when the ticket reads `ended`, or when two answers in a row show no
  order. A delete call the tool refused is a failed release.

Each decision is an `nc_elastic_*` event, for example `ordered`,
`member_ready`, `released`, `evicted`, `node_lost`, `lease_overdue`,
`sampler_restart`, `sampler_crash_loop`, `sampler_finished`,
`order_refused`, `order_unrecorded`, `foreign_ticket_released`,
`delete_unconfirmed`, `halt_cleared`, `account_unavailable`,
`legacy_ledger_rows`, `breaker_open` and `pool_stopped`.

Each sampler writes its event tape as `<stem>.<incarnation>.jsonl` (from the
spec's `event_tape`), one file per incarnation. To read the tapes in the
dashboard, pass the directory: `yeto dashboard export --tapes <dir>`.

The tapes stay on the nodes, under `/mnt/nvme/work/...` (the spec's `/work/`
path). Each node's local disk is destroyed when its lease is released, so
copy the tapes off while the node is up. The pool writes an SSH alias
`yeto-nc-<ticket>` for each node. The alias goes through the workspace relay,
and `~/.ssh/config` includes it, so you can copy the tapes with plain `scp`:

```
scp 'yeto-nc-<ticket>:/mnt/nvme/work/sampler/events*.jsonl' <tapes-dir>/
yeto dashboard export --tapes <tapes-dir>
```

## Known limitations

- **A delete cannot name its ticket.** `nc-manage delete --account N`
  releases the account's current ticket. The pool checks the ticket right
  before the call and stays away from the lease end and from an announced
  preemption, but one window is left: between that last read and the delete,
  the pool's ticket ends on its own (an unannounced preemption, a release in
  the console) *and* another user of the account places an order. The delete
  then releases their ticket. The pool cannot prevent this. What it does:
  - If the account then shows another ticket as `releasing` or released, the
    pool names it (`foreign_ticket_released`) and halts the account.
  - If the released ticket is no longer shown (the account shows no order, or
    a newer ticket), the pool only knows that its delete was never confirmed
    on its own ticket. It halts the account all the same
    (`delete_unconfirmed`). This also happens, rarely, after a normal release:
    the delete's answer did not name the ticket and the ended ticket was never
    shown. A halt for a person to check is the price of never writing on
    without knowing.
  - Either way the pool places no order on the account in between.

  *The fix is in the order tool:* give `nc-manage delete` a `--ticket <id>`
  argument that calls the release-by-id endpoint the tool already wraps
  (`release(ticketId)` in `tool/core.mjs`), and have it refuse when the id is
  not the session's ticket. The pool would then pass the id; this window and
  the halts above go away. Until the tool has it, the pool stays strict: there
  is no setting that relaxes the halt.
- **A queued ticket has no time.** National Compute reports `granted_at` for
  an active ticket and no order time at all. An unrecorded order whose ticket
  is still queued cannot be checked against the time of the order, so
  [`adopt`](#an-unrecorded-order-adopt) relies on the operator's check in the
  console. Do not adopt a ticket you cannot place.
- **Tickets recorded by an earlier version.** Ledger rows written before
  orders were recorded with their order (no `order` field) are still treated
  as this tool's, so a running node can be released after an upgrade. The
  earlier version could record a ticket that the order tool merely found on
  the account. At its first patrol, a pool lists its open rows of that kind
  once in a `legacy_ledger_rows` event. Check that each ticket listed is one
  this tool ordered; if one is not, set `"ended": true` on its row.
- **Two wrong "no order" answers in a row** for an account that has a ticket
  are taken as true: the lease counts as ended.

## Billing

- **A lease is not ended at `lease_until`.** The ticket stays `active` and
  billed until it is deleted. The pool drains nodes before that point, and
  its patrol deletes any ticket of the run found `active` past it. Both only
  happen while the loop runs.
- A lease is granted for up to 4 hours (`lease_until`) and cannot be
  renewed, but it does not end there by itself. Renewal is a replacement
  order on another account.
- A node that drops off the network can stay `active` and billed. The pool
  deletes it after `unreachable_grace_s`.
- The spend shown by `--dry-run` and tracked under `max_run_spend_ucr` is an
  estimate: active time × the ticket's rate. Your account balance is the
  real figure, and it is shared by everything on the organisation.
- Before you leave a run unattended:
  1. Set `elastic.max_run_spend_ucr`.
  2. Run the loop under tmux or systemd on an always-on host.
  3. Check `pool-status` now and then.
  4. When the run ends, confirm `pool-status` shows no `billed` members.
