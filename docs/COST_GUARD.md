# Cost protection for unattended runs

yeto protects every `yeto launch` run against three ways of spending money while nobody watches:
a run that costs more than planned, GPUs that sit idle, and resources that outlive the run.

## 1. Resource ledger (always on)

Every billed resource a run creates is appended to `~/.yeto/runs/<run>/resources.jsonl` the moment
it exists: Modal apps (by app id), Modal FunctionCalls, National Compute leases (ticket + account), and sky
clusters. A second line is written when a resource is released. Writes are fsynced, so the ledger
stays correct even if the launcher is killed with SIGKILL.

Processes started for the same run find the ledger through `YETO_RESOURCE_LEDGER`, which `yeto launch`
sets for its worker; a process that records into it writes an `nc_lease` entry with `account`, the ticket
as id, `lease_until` and a rate.

> The NC lease pool (`python -m yeto.launch.nc pool`) does not write this ledger yet: it keeps its own
> ticket ledger (`YETO_NC_STATE_DIR`). Its leases are not in `yeto cost`, the budget or `yeto cleanup`;
> stop its nodes with `python -m yeto.launch.nc down --account N --ticket T`.

## 2. Budget breaker

| Flag | Default | Meaning |
|---|---|---|
| `--budget-usd` | none (off) | When the run's cost reaches this, stop gracefully: the trainer takes a cut, the fleet ends, and every resource is torn down. |
| `--budget-warn-usd` | 80 % of `--budget-usd` | Emit `cost_budget_warn` once. |
| `--budget-stop-grace-minutes` | 20 | If the run is still up this long after a graceful stop, release every ledger resource by force. |
| `--cost-poll-seconds` | 60 | How often the guard checks. |
| `--cost-prices` | none | Price table used to estimate time that has not been billed yet. |

A reused run name (`--cluster-prefix`) keeps its ledger. The budget of a new launch counts the
resources still open plus those created since it started; what an earlier launch of that name released
before is not counted (`yeto cost --run` still reports the whole ledger). A new launch also removes the
`STOP` flag an earlier launch of that name left, so a budget or idle stop does not end the next launch.

How cost is computed:
- **Modal**: hourly rows from the Modal billing API, matched by exact app id, plus an estimate for the
  time after the newest billed hour (billing data lags behind). The estimate uses the rate recorded in
  the ledger, then the price table, then the app's newest complete billed hour.
- **National Compute**: lease duration × rate. A lease past its `lease_until` keeps billing until it is
  deleted, and it is counted that way. Set `YETO_NC_USD_PER_GPU_HOUR` (or `YETO_NC_USD_PER_UCR`) to price it.
- A resource without a known rate is reported as **unpriced** (`cost_unpriced` event), never as $0, and
  is not counted toward the budget. Configure a rate for it.

## 3. Idle shutdown

| Flag | Default | Meaning |
|---|---|---|
| `--idle-warn-minutes` | 15 | `idle_warn` after this long with no progress event, or with every reporting GPU at 0 %. `0` turns idle shutdown off. |
| `--idle-stop-minutes` | 15 | Stop the run (same graceful-then-forced path as the budget) this long after `idle_warn` if it is still idle. |
| `--load-grace-minutes` | 45 | No idle verdict for this long after the start or after any loading event (weight loading, a trainer restart). Loading the K3.1 weights takes about 20 minutes with 0 % GPU. |
| `--max-load-minutes` | 120 | Loading events alone cannot hold off the idle verdict for longer than this since the last progress (catches a loader stuck in a restart loop). |
| `--cost-tapes` | none | Extra event tape globs the detector reads, for example sampler tapes pulled from NC nodes. |

Progress means trained rounds, sampled rounds, cuts and policy releases. Heartbeats are not progress,
because a hung process still sends heartbeats. A sampler that waits while the trainer trains does not
make the run idle, because idleness is judged for the run as a whole.

## 4. Lease expiry

An NC lease still open 5 minutes after its `lease_until` is released (`nc_lease_overdue`). The
lease pool normally drains nodes before that point; this check is a safety net.

## 5. Cleanup

- When the run ends normally or with an error, the launcher tears down its resources. The worker then
  releases anything the ledger still lists.
- On SIGTERM (`yeto down`), the launcher unwinds through its teardown.
- If the launcher is killed outright, a detached **reaper** process (one per run) sees the worker exit and
  releases what the ledger still lists.
- Manual cleanup: `yeto cleanup --run <run> [--dry-run]`. It releases only what the ledger lists, each by
  its exact id. It never matches by name pattern and never uses `--all`. Running it again is safe.
- `--keep` runs are left up by the reaper and the backstop. Release them with `yeto down <run>`, which
  also cleans the ledger.

### Control host

The guard and the reaper run on the machine that ran `yeto launch`. If that machine goes offline
(sleep, network loss, shutdown), no budget, idle or lease-expiry check runs and nothing releases the
run's resources; Modal apps and NC leases keep billing. Run long launches from a host that stays up
(a small always-on VM, in `tmux` or under `systemd`), and after an outage run `yeto cost --run RUN`
and `yeto cleanup --run RUN`.

## 6. Cost report

`yeto cost --run <run> [--json] [--no-billing] [--prices FILE] [--tapes GLOB...]` shows:
- cost per resource: billed $, estimated $ and total
- cost per UTC hour
- **GPU effective utilization**: productive time ÷ billed compute time. Productive time is the time
  between progress events that are at most `--gap-minutes` (default 10) apart. This is an estimate taken
  from event tapes. When hardware utilization samples are available, their mean is shown next to it.

## Events

All events go to `~/.yeto/runs/<run>/cost-events.jsonl` and to the launcher log (`[cost-guard] ...`):
`cost_tick`, `cost_budget_warn`, `cost_budget_exceeded`, `cost_unpriced`, `idle_warn`, `idle_cleared`,
`idle_shutdown`, `cost_guard_stop` (mode graceful / forced), `nc_lease_overdue`.

`--no-cost-guard` turns the guard off. The ledger, the reaper and `yeto cleanup` keep working.
