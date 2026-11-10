# Kimi K3.1 async GRPO (agentic RL): end-to-end guide

This guide takes you from an empty control host to a running, monitored and
cleanly stopped asynchronous GRPO run of Kimi K3.1 with agentic (Harbor)
environments. One run config drives everything through one command,
`yeto run up`.

Detailed references live in their own documents and are linked rather than
repeated here:

| Topic | Reference |
|---|---|
| NC sampler pool: every `pool.json` field, `pool-*` commands, billing | [NC_POOL.md](NC_POOL.md) |
| Store task dispenser (`sampler.dispense`) | [DISPENSER.md](DISPENSER.md) |
| Sandbox hosts, `yeto sandbox`, `agentic.envs` | [SANDBOX_HOSTS.md](SANDBOX_HOSTS.md) |
| Run health breakers, log-prob guards | [BREAKER.md](BREAKER.md) |
| Resource ledger, budget breaker, cleanup | [COST_GUARD.md](COST_GUARD.md) |
| Trainer flags, rounds, cuts, releases, padding | [MILES_RL.md](MILES_RL.md) |
| Dashboard details, alert thresholds | [DASHBOARD.md](DASHBOARD.md) |
| Every `yeto` command and flag | [CLI_REFERENCE.md](CLI_REFERENCE.md) |

## Contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Configuration](#3-configuration)
4. [Launch](#4-launch)
5. [Operating a running job](#5-operating-a-running-job)
6. [Dashboard](#6-dashboard)
7. [Breakers and cost protection](#7-breakers-and-cost-protection)
8. [Troubleshooting FAQ](#8-troubleshooting-faq)
9. [Trainer image (Miles)](#9-trainer-image-miles)
10. [Full-parameter parallel layout (64 and 128 experts)](#10-full-parameter-parallel-layout-64-and-128-experts)

---

## 1. Architecture

The run is split into four independent parts that only talk through the
object store and a few well-defined network paths:

- **Trainer island** on Modal B300 GPUs (`yeto launch --training-mode rl
  --rl-sample-role trainer ...`). It reads sample groups from the buffer,
  trains, and publishes policy releases.
- **Sampler islands** on National Compute (NC) nodes, one sampler per node,
  managed by the NC pool loop (`python -m yeto.launch.nc pool`). Each sampler
  runs the inference engine described by the sampler spec, claims tasks from
  the dispenser, runs agentic trials and writes groups to its own buffer
  shard.
- **Object store** (R2 or any S3-compatible bucket): the sample buffer
  (`sample-buffer/shards/<shard>`), the task dispenser ledger, the policy
  releases, the run's event tapes and the trainer's checkpoint store.
- **Sandbox host(s)** running the Harbor task containers. Samplers reach them
  through the reverse tunnel configured in the spec (`agentic.setup.tunnel`)
  and the Docker endpoint (`agentic.setup.sandbox.docker_host`).

The control host (your workstation or a small VM) runs `yeto run up`, which
in turn runs the pool loop, the trainer launcher, the dashboard and the
monitor.

```
                         +-----------------------------------------------+
                         |  Control host                                 |
                         |  yeto run up --config run.json                |
                         |   |- preflight (dry runs, argument checks)    |
                         |   |- NC pool loop  (order / setup / replace)  |
                         |   |- trainer launcher (yeto launch)           |
                         |   |- dashboard (loopback HTTP, reads store)   |
                         |   '- monitor (budget, wall clock, silence)    |
                         +------+-------------------+--------------------+
                                |                   |
              orders, SSH setup |                   | launches
                                v                   v
 +------------------------------------+    +---------------------------------+
 |  NationalCompute sampler islands   |    |  Modal B300 trainer island      |
 |  node <shard_base>-0 .. -N         |    |  Miles trainer                  |
 |  engine (vLLM) + agentic sampler   |    |  reads groups, trains,          |
 |  claims tasks, runs trials         |    |  publishes policy releases,     |
 +----+------------------+------------+    |  writes cuts                    |
      |                  ^                 +---------+---------------+-------+
      | trials           | releases                  ^               |
      v                  |                           | groups        | releases,
 +------------------+    |      +--------------------+---------------v-----+
 | Sandbox host(s)  |    +------+  Object store (R2 / S3): <bucket>          |
 | Harbor task      |           |   sample-buffer/shards/<shard>  (groups)   |
 | containers       |           |   task dispenser ledger         (tasks)    |
 +------------------+           |   policy releases               (weights)  |
          ^                     |   event tapes, checkpoint store            |
          '-- reverse tunnel ---+--------------------------------------------+
              (sampler <-> sandbox)
```

Data flow per step: samplers claim tasks from the dispenser, generate
trajectories against the sandboxes, and write complete groups to their shard;
the trainer reads groups up to `--rl-max-policy-age` versions old, trains,
and releases a new policy; each sampler loads the release and continues.

## 2. Prerequisites

### Accounts and credentials

All values below are placeholders. Keep real values in files outside the
repository and point the configs at them.

| What | Used by | How it is provided |
|---|---|---|
| Modal account and token | trainer | Modal CLI login on the control host; `modal` SDK installed (preflight checks it) |
| Modal secret holding the store credentials | trainer | trainer flag `--modal-store-secret <modal-secret>` |
| Object store bucket and credentials | everything | run config `env.AWS_PROFILE: "<profile>"`; pool `r2_env_file: "<store-env-file>"` for the nodes |
| NC accounts | pool | `YETO_NC_ACCOUNTS` in the run config `env` and `accounts` in `pool.json` (`<account-id>` list) |
| NC tool directory | pool | `env.NC_TOOL_DIR` (see [NC_POOL.md, Before you start](NC_POOL.md#before-you-start)) |
| SSH keys for sandbox host and tunnel | samplers | spec `agentic.setup.sandbox.key.path`, `agentic.setup.tunnel.key.path` |
| Agentic environment file | samplers | pool `agentic_env_file: "<agentic-env-file>"` (HARBOR_*, DOCKER_HOST, ...) |

An NC account holds one node at a time. To run `N` samplers and keep a
replacement ready, provide at least `N + 1` accounts.

### Images

- **Sampler image** on NC: the vLLM build the sampler spec pins
  (`engine.server.engine_version`, `engine.options.pin_version`). The pool
  pulls it during node setup; override with `image` in `pool.json`.
- **Trainer image** on Modal: selected by the trainer flags (see
  `examples/k31/launch-trainer-modal-b300.sh` and [MILES_RL.md](MILES_RL.md)).
  The run's code is shipped as a code pack (`--rl-code-pack`, default `auto`).
  Building and pinning the image: [section 9](#9-trainer-image-miles).

### Model weights

- **Trainer**: the base checkpoint on a Modal volume
  (`--model <model-id>`, `--model-revision <revision>`,
  `--modal-model-volume <volume>`).
- **Samplers**: the MXFP4 checkpoint in the bucket. The pool pulls it onto
  each new node through `model_pull` (`src_key: "<models/prefix>"`,
  `max_mb_s`, `timeout_s`); the spec's `engine.server.base_dir` points at the
  local copy.
- **Expert selection**: the trainer's `--rl-expert-selection <file>` and the
  spec's `expert_selection_sha256` must describe the same file. Format and
  generation: [section 10.3](#103-expert-selection-file).

### Sandbox host

Prepare one or more hosts with Docker reachable over SSH. Task sets and
readiness are described in [SANDBOX_HOSTS.md](SANDBOX_HOSTS.md). `yeto run up`
calls `yeto sandbox status` in preflight and `yeto sandbox prepare` before
ordering any node.

## 3. Configuration

A run uses three files:

- `run.json`: the run config read by `yeto run up / status / down / scale`;
- `pool.json`: the NC pool config ([NC_POOL.md](NC_POOL.md));
- the sampler spec: the engine, sampling, prompts and agentic sections of each
  sampler (`pool.json` `spec`). Starting points are in `examples/k31/`
  (`sampler-spec.nc-mi355x-vllm.json`).

### 3.1 `run.json`

```json
{
  "comment": "Kimi K3.1 async GRPO, agentic",
  "run": "<run>",
  "store": "s3://<bucket>/runs/<run>",
  "env": {
    "AWS_PROFILE": "<profile>",
    "YETO_NC_ACCOUNTS": "<account-id>,<account-id>,<account-id>",
    "YETO_NC_STATE_DIR": "<nc-state-dir>",
    "NC_TOOL_DIR": "<nc-tool-dir>"
  },
  "preflight": {"commands": [["python", "<trainer-argument-check.py>"]]},
  "sandbox": {
    "envs": "<envs.json>",
    "docker_host": "ssh://<sandbox-host>",
    "max_run_containers": 40
  },
  "pool": {"config": "<pool.json>", "interval_s": 30},
  "trainer": {
    "argv": ["--training-mode", "rl", "--gpu", "modal:<nodes>x8xb300", "..."],
    "usd_per_h": 0,
    "progress_paths": []
  },
  "first_group": {"timeout_s": 5400, "poll_s": 60},
  "dashboard": {"port": 8795},
  "budget": {
    "usd": 0,
    "wall_h": 3.0,
    "nc_usd_per_h": 0,
    "silence_min": 30,
    "release_nc_on_breach": false
  }
}
```

Fill the `0` prices with your contracted rates; they drive the spend estimate.

| Field | Meaning |
|---|---|
| `run` | Run name. Also the state directory `~/.yeto/rl-up/<run>/` (pids, logs, step times) and the `ai.yeto.run` label of sandbox containers. Required. |
| `store` | Run store URI. The dashboard reads it; the first-group wait lists `sample-buffer/shards/` under it. Required. |
| `env` | Environment for every subprocess (paths may use `~`). `YETO_RESOURCE_LEDGER` defaults to `~/.yeto/runs/<run>/resources.jsonl`. |
| `preflight.commands` | Extra checks run during preflight; any non-zero exit stops `up`. |
| `sandbox.envs` | The `agentic.envs` file for `yeto sandbox status / prepare / teardown`. Optional. |
| `sandbox.docker_host` | Docker endpoint used to count and remove this run's trial containers. |
| `sandbox.max_run_containers` | Warning threshold for this run's container count (a warning, not a breaker). |
| `pool.config` | Path of `pool.json`. Required. |
| `pool.interval_s` | Seconds between pool control ticks (default 30). |
| `trainer.argv` | The `yeto launch` arguments of the trainer. Required. |
| `trainer.usd_per_h` | Trainer price per hour for the spend estimate. |
| `trainer.progress_paths` | Extra files whose growth counts as trainer progress for the silence backstop. |
| `first_group.timeout_s`, `poll_s` | How long `up` waits for the first buffer object before starting the trainer (default 5400 s, polled every 60 s). |
| `dashboard.port` | Loopback port of the dashboard (default 8795). |
| `budget.usd` | Spend cap (trainer plus NC estimate). |
| `budget.wall_h` | Wall-clock cap from the pool start (default 3). |
| `budget.nc_usd_per_h` | Price of one NC node hour for the estimate. |
| `budget.silence_min` | Minimum silence window of the backstop; the effective window is `max(4 x --rl-stall-timeout, silence_min)`. |
| `budget.release_nc_on_breach` | `true`: a budget or wall-clock breach performs the full `down` including the NC pool. Default `false` (see section 7). |

### 3.2 Trainer arguments (`trainer.argv`)

The full trainer command, with comments for every flag, is in
`examples/k31/launch-trainer-modal-b300.sh`; all flags are documented in
[MILES_RL.md](MILES_RL.md). The ones you will tune most:

| Flag | Meaning |
|---|---|
| `--rl-sample-role trainer` | This island trains only; samplers live elsewhere. |
| `--rl-buffer-shards <shard_base>-0,<shard_base>-1,...` | The buffer shards the trainer reads. List every shard index the pool may ever use (see `shard_limit` below). |
| `--rollout-batch-size` | Prompts (groups) per training round. |
| `--n-samples-per-prompt` | Group size; must match the spec's `sampling.n`. |
| `--rl-max-policy-age` | How many versions old a group may be; needs a matching algorithm spec (`--rl-algorithm-spec`, e.g. `examples/k31/algorithm-spec.async-age2.json`). |
| `--rl-checkpoint-store` | Where round cuts and the ledger are kept; enables resume across launches. |
| `--rl-cut-every N` | Keep a round cut every `N` rounds (plus the last round). See also `--rl-cut-keep`, `--rl-cut-auto-mtbf-s`. |
| `--rl-release-every N` | Release every N-th policy version to the samplers (default 1). Needs `--rl-max-policy-age >= N`. |
| `--rl-stall-timeout` | Trainer hang detection in seconds. `yeto run up` refuses `0`. |
| `--rl-stop-after-rounds N` | Stop after N rounds with a final cut. |

### 3.3 `pool.json`

```json
{
  "run": "<run>",
  "route": "ssh",
  "target": 2,
  "accounts": ["<account-id>", "<account-id>", "<account-id>"],
  "shard_base": "<shard-base>",
  "shard_limit": 8,
  "machine_type": "mi355x",
  "spec": "<sampler-spec.json>",
  "run_prefix": "runs/<run>",
  "r2_env_file": "<store-env-file>",
  "agentic_env_file": "<agentic-env-file>",
  "model_pull": {"src_key": "<models/prefix>", "max_mb_s": 2500, "timeout_s": 7200},
  "replace_lead_s": 3600,
  "end_margin_s": 600,
  "sampler": {"dispense": {"backend": "store", "namespace": "task-dispense"}},
  "elastic": {"max_nodes": 3, "max_orders": 10, "engine_stall_s": 600}
}
```

Key fields (the complete list is in
[NC_POOL.md, the pool config](NC_POOL.md#the-pool-config-pooljson)):

| Field | Meaning |
|---|---|
| `target` | Samplers to keep running. At most the number of `accounts`. Change it at run time with `yeto run scale`. |
| `accounts` | NC accounts the pool may order on; each must also be in `YETO_NC_ACCOUNTS`. Accounts beyond `target` are used for replacements. |
| `shard_base` | Shard prefix: nodes write `<shard_base>-0`, `<shard_base>-1`, ... Also names the pool state file `pool-<shard_base>.json`. |
| `shard_limit` | How many shard indexes the run may take; set it to the number of shards in the trainer's `--rl-buffer-shards`. The usable range is also bounded by the port range of `agentic.setup.tunnel.listen` (one port per shard). Shard indexes are not reused, so replacements consume new indexes. |
| `elastic.max_nodes` | Most billed nodes at once (default `2 x target`). |
| `elastic.max_orders` | Most orders over the whole run. |
| `elastic.engine_stall_s` | Seconds without generated tokens while requests are pending before an engine counts as stalled (default 600). |
| `elastic.max_sampler_restarts` | In-place restarts of a crashed sampler before its node is replaced (default 2). |
| `sampler.dispense` | Task dispenser settings ([DISPENSER.md](DISPENSER.md)). |
| `model_pull` | Model download from the bucket onto each new node. |

`pool --dry-run` (part of preflight) refuses unknown keys, so a typo is
reported before anything is ordered.

### 3.4 The sampler spec

Start from `examples/k31/sampler-spec.nc-mi355x-vllm.json`. Placeholders such
as `<nc-node-id>` are filled per node by the pool. Key fields:

| Field | Meaning |
|---|---|
| `engine.adapter` | `vllm-server` for the K3.1 sampler. |
| `engine.server.base_dir` | Local path of the MXFP4 checkpoint on the node. |
| `engine.server.max_model_len` | Engine context length. |
| `engine.server.extra_args` | Extra engine flags. They come after the preset's flags and take precedence. |
| `engine.options.k31_preset` | Serving preset (`amd-article`, `rl-precision`, `vllm-recipe`, `rl-precision-cuda`). The ROCm presets set `--gpu-memory-utilization 0.88`. |
| `engine.options.tp`, `max_num_seqs` | Tensor parallel degree and concurrent sequences per engine. |
| `sampling.n` | Samples per prompt (group size). |
| `sampling.max_tokens`, `temperature`, `top_p`, `top_k` | Generation settings. |
| `max_backlog_groups` | Groups a sampler may hold ahead of the trainer before it pauses. |
| `prompts.groups_per_round` | Groups a sampler produces per round. |
| `reward`, `reward_kind` | Reward function import path and kind. |
| `agentic.*` | Agentic trial settings: `max_in_flight`, `turn_max_tokens`, `max_response_len`, `infra_retries`, `task_defaults`, `setup` (sandbox, tunnel, tasks), `envs` ([SANDBOX_HOSTS.md](SANDBOX_HOSTS.md)). |

**GPU memory utilization.** To change it, add the flag to
`engine.server.extra_args`, for example
`["--gpu-memory-utilization", "0.85"]`.
Recommended practice: keep the preset's 0.88 on MI355X TP8. The collectives'
scratch memory lives outside vLLM's budget, and the preset keeps that
headroom free.

## 4. Launch

### 4.1 Dry run

```
yeto run up --config <run.json> --dry-run
```

Preflight runs every check and orders nothing:

1. the Modal SDK check (when the trainer targets `modal:`), and a refusal of
   `--rl-stall-timeout 0`;
2. `python -m yeto.launch.nc pool --config <pool.json> --dry-run`: the pool
   config, the accounts, and the sampler spec built locally the way the
   sampler builds it at start (island config, engine adapter, rollout fields,
   reward import, `agentic.setup`);
3. `yeto sandbox status` (when `sandbox.envs` is set);
4. `yeto launch <trainer argv> --dry-run`;
5. every `preflight.commands` entry.

Output goes to `~/.yeto/rl-up/<run>/preflight.log`. Each step logs its return
code (`rc=0` passes). The run ends with either
`dry run: preflight ok, nothing ordered` or
`preflight failed: nothing ordered (see preflight.log)`. If an earlier trainer
launch of this run exists, preflight also notes that the trainer will start
with `--rl-ledger-set-aside-before` (see section 5.3).

### 4.2 Start

```
yeto run up --config <run.json>
```

Steps: preflight, `yeto sandbox prepare`, the pool loop (detached), the wait
for the first sample group, the trainer (detached), the dashboard, then the
monitor in the foreground. `up` is resumable: running it again skips steps
whose process is still alive.

Options:

| Option | Use |
|---|---|
| `--dry-run` | Preflight only. |
| `--skip-preflight` | Resume after a preflight that already passed. |
| `--trainer-now` | Start the trainer immediately instead of waiting for the first group, so the Modal GPUs and the NC node setup come up in parallel. This shortens time to first step. The trainer may idle until samplers deliver groups. |
| `--no-monitor` | Return after the dashboard starts. Recommended practice: keep the monitor running, since it enforces the budget, wall-clock and silence guards. |

Status at any time:

```
yeto run status --config <run.json>
```

It prints the step history, process liveness, buffer object count, the spend
estimate, `pool-status`, the dispenser status, and the sampler target/nodes.

## 5. Operating a running job

### 5.1 Scaling sampler islands

```
yeto run scale --config <run.json> --samplers <N>
yeto run scale --config <run.json> --samplers <N> --retire <ticket>   # pick which node leaves
```

This runs `python -m yeto.launch.nc pool-scale --config <pool.json> --target <N>`
on the running pool. You can also call `pool-scale` directly.

A scale request is checked before anything is ordered. It is refused (and
nothing changes) when the target exceeds the number of `accounts`,
`data_slots`, `elastic.max_nodes`, `elastic.max_orders`,
`elastic.max_run_spend_ucr`, or the shard range (`shard_limit` /
tunnel ports). On scale down the pool fences the leaving node's shard first,
so its claims return to the dispenser. Details:
[NC_POOL.md, pool-scale](NC_POOL.md#change-the-number-of-samplers-while-the-run-trains-pool-scale).

### 5.2 Automatic node replacement

The pool loop probes every node each tick and replaces lost capacity without
operator action:

1. **Detect.** A node that is evicted, unreachable for longer than
   `elastic.unreachable_grace_s`, or past its lease is treated as lost.
2. **Fence.** The node's shard is fenced (`nc_elastic_shard_fenced`), so a
   node that comes back cannot add groups the trainer would take; its
   dispenser claims are released.
3. **Release and re-order.** The ticket is deleted and a new node is ordered
   on a free account (`nc_elastic_ordered`). Before a planned lease end, the
   replacement is ordered `replace_lead_s` ahead so there is no gap.
4. **Setup.** The new node runs `agentic.setup` (sandbox access, tunnel,
   tasks), the `model_pull`, `node_hooks`, and the sampler start.
5. **Ready.** The sampler joins (`nc_elastic_member_ready`) and continues the
   shard's prompts on a new shard index.

The whole sequence appears in the dashboard's NC pool events. Event names and
the per-case behaviour are listed in
[NC_POOL.md, What the pool does](NC_POOL.md#what-the-pool-does).

### 5.3 Resuming training

- **From the latest cut.** With `--rl-checkpoint-store`, the trainer keeps
  round cuts (`--rl-cut-every`, `--rl-cut-keep`). Run
  `yeto run up --config <run.json>` again with the same run config; the
  trainer resumes from the latest cut. A changed run configuration is refused
  unless you add `--rl-resume-allow-config-change` (the differences are
  recorded in the `rl_resume` event).
- **From a policy release.** Add `--rl-init-from-release latest` (or
  `r<seq>-v<version>` / a version) to `trainer.argv`. The trainer loads that
  release instead of the base checkpoint, and version numbering continues
  after it. Current scope: the optimizer state starts fresh on this path.
- **From the base, on an existing run.** When `yeto run up` restarts a trainer
  of a run that launched before, it adds
  `--rl-ledger-set-aside-before <now>`. A start from the base then moves the
  earlier batch ledger aside (`ledger.before-<ts>`) instead of stopping on it.
  No manual cleanup is needed.

### 5.4 Upgrading the code of a running run

The run's code is pinned as a code pack at the first start. Resumed and
replacement islands get the same pack, so a relaunch never picks up local
changes by accident.

To move the run to the current code, add
`--rl-code-pack-upgrade "<reason>"` to `trainer.argv` (with `--rl-code-pack`
`git-patch` or `release`), then run `yeto run up` again. The upgrade is
accepted when the algorithm's semantic hash is unchanged, and the manifest
records the old and new pack with your reason. To resume deliberately on the
pinned pack, use `--rl-code-pack-keep` instead. For algorithm changes,
Recommended practice: start a new run (another `--cluster-prefix`).

### 5.5 Stopping

```
yeto run down --config <run.json>
```

`down` performs, in order:

1. writes a read-only dashboard export to `~/.yeto/rl-up/<run>/dashboard.html`;
2. stops the dashboard and the trainer (`yeto down <run>`);
3. releases the NC pool (`pool-stop`);
4. tears down the sandboxes (`yeto sandbox teardown`) and removes this run's
   trial containers;
5. verifies with `pool-status` and reports `down: verified` or
   `down: NOT verified, see down.log`.

Recommended practice: run `down` when you finish. NC leases bill until their
tickets are deleted. `pool-stop` alone releases only the NC nodes
([NC_POOL.md, Stopping a pool](NC_POOL.md#stopping-a-pool)).

## 6. Dashboard

`yeto run up` starts `yeto dashboard serve --store <store> --port <port>` on
the control host. It binds to loopback only and prints
`http://127.0.0.1:<port>/`. If the control host is remote, open a tunnel from
your laptop:

```
ssh -N -L <port>:127.0.0.1:<port> <control-host>
```

Then browse to `http://127.0.0.1:<port>/`. For an offline copy, use
`yeto dashboard export --store <store> -o <file.html>`. `down` writes one
automatically. More detail: [DASHBOARD.md](DASHBOARD.md).

All numbers come from the event tapes; `n/a` means the value is not on the
tapes.

### Async GRPO panel

**Trainer rounds.** One row per trained round:

| Column | Meaning |
|---|---|
| reward mean +- std, p10 / p50 / p90 | Reward of the trained samples. The main learning signal. |
| train-rollout KL, logprob \|diff\| | Gap between trainer and sampler log-probs. Breaker limits: [BREAKER.md](BREAKER.md). |
| TIS clipfrac | Fraction of importance weights clipped. |
| grad_norm | Gradient norm. |
| resp len mean / p95, truncated | Response lengths and the share cut at the token limit. |
| groups | Groups consumed (trained) this round. |
| stale dropped, policy ages | Groups dropped as older than `--rl-max-policy-age`, and the age histogram of trained groups. |

The totals line above the table sums trained groups, stale drops,
off-policy-gate drops, short groups, carried-over groups and **padded
samples**. Padding tops up a round that cannot be filled with whole groups,
and the real samples' gradient is scaled by the padded fraction.
The trainer reports the ratio as the metric `sample_buffer/padded_fraction`
([MILES_RL.md](MILES_RL.md)). Current scope: the dashboard shows the padded
sample count; the fraction is read from the trainer metrics. A fraction that
stays high means samplers are not keeping up. Add samplers or raise
`--rl-max-policy-age`.

**Samplers.** One row per sampler island:

| Column | Meaning |
|---|---|
| status | `healthy`; `starting`; `waiting: backlog full (trainers behind)`; `waiting: waiting for a policy release`; `stale`; `lost`; `left the run (<reason>)`; `finished`; `stopped`. |
| policy | `v<policy_version>` the sampler currently generates with. |
| policy age p50 / max, histogram | Trainer version minus the version a round was generated with. |
| generate s p50 / max | Seconds per generation round. |
| groups written | Groups the sampler wrote to its shard. |
| zero-variance seen / dropped, truncated, dropped groups | Group quality counters. |
| trials retried / lost | Agentic trials retried, and trials lost (`rl_agentic_trial_lost`). |
| reward | Last reward the sampler reported. |

**Buffer backlog.** Groups written by all samplers minus the trainers' read
position. A growing backlog means trainers are behind. A backlog near zero
means samplers are the bottleneck.

**Policy releases.** Each trainer release and when each sampler finished
loading it. NC pool events appear as dashed lines.

**NC pool events.** The `nc_elastic_*` stream (ordered, member_ready,
shard_fenced, evicted, node_lost, engine_stalled, released, ...). Use it to
follow a replacement end to end.

### Other panels

- **Island health**, **Event stream**: per-island state and the raw events.
- **Cost (estimate, not billing data)**: spend per resource from the resource
  ledger and the price table (`--prices`). The cost bar uses `--budget`.
  Current scope: the figures are estimates; your provider invoices are
  authoritative. `yeto run status` shows the same spend estimate the monitor
  uses (`spend_usd_estimate`).

## 7. Breakers and cost protection

The monitor in `yeto run up` checks the run every minute:

| Guard | Trigger | Action |
|---|---|---|
| Budget | spend estimate > `budget.usd` | Stop the trainer. |
| Wall clock | time since pool start > `budget.wall_h` | Stop the trainer. |
| Trainer exit | launcher finished `FAILED` or non-zero | Stop the trainer path cleanly. |
| Silence backstop | trainer log and `progress_paths` unchanged for `max(4 x --rl-stall-timeout, silence_min)` | Stop the trainer. |
| First group | no buffer object within `first_group.timeout_s` | Stop before the trainer starts. |
| Sandbox containers | count > `sandbox.max_run_containers` | Warning only. |

**Trainer first, NC kept.** By design, a breach stops only the trainer
(`yeto down <run>`). The NC pool, its nodes and the sandboxes stay up, so a
resume does not repeat node setup and model pull. The monitor prints which
nodes still bill and the release command.

**Forced NC release at 1.2x.** After a budget or wall-clock breach, the
monitor stays up and prints the billing banner on every poll. Once the spend
passes `1.2 x budget.usd`, or the wall time passes `1.2 x wall_h`, it runs
`pool-stop` itself. It exits when no node bills.

**Opt-in full stop.** Set `"release_nc_on_breach": true` in `budget` to make a
budget or wall-clock breach perform the full `yeto run down` immediately.

The pool also has its own guards (`elastic.max_nodes`, `max_orders`,
`max_run_spend_ucr`, order breaker); see [NC_POOL.md](NC_POOL.md). Log-prob
and health breakers inside the run are described in [BREAKER.md](BREAKER.md).
The resource ledger, idle shutdown and cleanup are described in
[COST_GUARD.md](COST_GUARD.md).

## 8. Troubleshooting FAQ

Logs are in `~/.yeto/rl-up/<run>/` (`preflight.log`, `sandbox.log`,
`pool.log`, `trainer.log`, `dashboard.log`, `down.log`).

**A sampler stays `waiting`.**
Check the reason shown in its status:
- `backlog full (trainers behind)`: the sampler has `max_backlog_groups`
  ready and pauses by design. The trainer is the bottleneck. Check the trainer
  rounds and `trainer.log`. Scaling samplers down saves cost while this
  persists.
- `waiting for a policy release`: the sampler cannot generate more within
  `--rl-max-policy-age` until a newer release arrives. Check the Policy
  releases panel for the latest release and load time. If the trainer has
  not released recently, look at the trainer first.

**A sampler shows `starting` for a long time.**
Node setup includes the image pull, `model_pull` and engine start. Follow the
NC pool events and `pool.log`. The pool gives up after
`elastic.start_timeout_s` and orders a replacement.

**Trials are lost (`trials retried / lost` grows).**
Each lost trial emits `rl_agentic_trial_lost` with the error, the call, the
token window and the trial directory. Typical causes: a model reply that
cannot be aligned to the recorded tokens, or a prompt longer than the
trainer's prompt width. Infrastructure failures are retried up to
`agentic.infra_retries`. A small, steady rate is normal for agentic tasks.
A sudden rise across all samplers usually points at the sandbox host:
run `yeto sandbox status --envs <envs.json> --store <store>`
([SANDBOX_HOSTS.md](SANDBOX_HOSTS.md)).

**`engine_stalled` in the NC pool events, then the sampler recovers.**
The engine had requests pending but produced no tokens for
`elastic.engine_stall_s`. The pool fenced the shard and restarted the engine
in place. This is the automatic recovery path and needs no action. After
`max_stall_restart_failures` consecutive restarts without progress, the node
is retired and replaced (`engine_stall_loop`). An idle engine (waiting for the
trainer) is never counted as stalled. Current scope: the separate
event-tape stall detector (`elastic.stall_s`) is off by default; set it above
your longest round if you want it.

**A release is not loaded by a sampler.**
The sampler emits `rl_sampler_release_not_loaded` once per release and cause,
with `reason` and `detail`. Check, in order: that the release finished
uploading (Policy releases panel), that the node can read the store
(`r2_env_file`), and that the release matches the sampler (the reason names
the mismatch). The sampler keeps generating with its current version
within `--rl-max-policy-age` and retries on the next poll.

**The pool logs `CONFIGURATION ERROR`.**
A sampler exited with code 2 because its spec or arguments were refused. The
pool keeps the node and orders nothing more. Fix the spec, then run
`python -m yeto.launch.nc pool-retry --config <pool.json>` to restart the
sampler on the same node(s)
([NC_POOL.md, pool-retry](NC_POOL.md#retry-after-a-configuration-error-pool-retry)).

**`yeto run scale` is refused.**
The message names the limit (accounts, `max_nodes`, `max_orders`, spend, or
the shard range). Shard indexes are not reused. For a long run with many
replacements, set a generous `--rl-buffer-shards` list and `shard_limit` up
front.

**`up` stops at the first group.**
No buffer object appeared within `first_group.timeout_s`. Check `pool.log`
for node setup progress and the samplers' event tapes. On the next attempt,
consider `--trainer-now` so the trainer start overlaps the node setup.

**The trainer stopped with `BREAKER: trainer: log / tape silent`.**
The launcher was alive but produced no output for the silence window.
Inspect `trainer.log` and the Modal app. Resume with `yeto run up` (section
5.3). NC nodes are kept, so the restart is quick.

**`down` reports `NOT verified`.**
Read `down.log`. Usually a node release is still in progress. Run
`python -m yeto.launch.nc pool-status --config <pool.json>` and, if nodes are
listed, `pool-stop` again.

## 9. Trainer image (Miles)

The trainer runs from a container image that carries Miles, SGLang,
Megatron-LM and their dependencies at fixed commits. Every default image is
pinned by digest.

### 9.1 Default images and `--rl-image`

The defaults are defined in `yeto/rl/adapters/miles/pins.py`:

| Constant | Used for |
|---|---|
| `MILES_K31_FULL_IMAGE` | Default `--rl-image` for `--rl-model-recipe kimi_k31 --tuning full` (the full-parameter, partial-experts trainer) |
| `MILES_K31_IMAGE` (= `MILES_NEXT_IMAGE`) | Default for every other recipe and the LoRA form |

Both resolve to `docker:<registry>/yeto-miles-k31@sha256:<digest>`. Only the
registry prefix is configurable:

- `YETO_IMAGE_REGISTRY=<registry>/<org>` (in the run config `env`) points the
  defaults at a mirror that carries the same digests. Because the reference is
  a digest, a mirror cannot serve different bytes under the same pin.
- `--rl-image docker:<registry>/<repo>@sha256:<digest>` overrides the image for
  one run. Recommended practice: always reference by digest, not by tag.
- `--rl-image-private` pulls with a registry login
  (`SKYPILOT_DOCKER_USERNAME` / `SKYPILOT_DOCKER_PASSWORD`; see
  [CLOUDS.md](CLOUDS.md)).

When the image's Miles commit differs from the default pin, the island
checks its checkout against the commit recorded for that digest
(`IMAGE_MILES_COMMITS` in `pins.py`).

### 9.2 Building the Miles K3.1 image

Source: `docker/miles-k31/` in this repository.

- `docker/miles-k31/versions.lock.json` pins the upstream base image by index
  and manifest digest. It also pins the base's build arguments (the Miles,
  SGLang and Megatron-LM commits, CUDA, NCCL, FlashInfer) and the expected
  package versions.
- `docker/miles-k31/build.sh` builds without a Docker daemon. It uses `crane`
  to append one overlay layer to the pinned base. The layer holds the Miles
  fork checkout, the SGLang fork checkout, the `peft` and `pyrage` wheels, the
  Megatron patches, `/opt/yeto/image-manifest.json` and the image-assert lock
  under `/opt/yeto/image`.

```
MILES_SRC=<miles-fork-clone> SGLANG_SRC=<sglang-fork-clone> \
MILES_BRANCH=<branch-or-commit> SGLANG_BRANCH=<branch-or-commit> \
DEST_REPO=<registry>/<org>/yeto-miles-k31 \
  docker/miles-k31/build.sh --push
```

Before it builds anything, the script compares the base image's recorded build
arguments with the lock and stops on any mismatch. It resolves the fork
branches to commits and tags the image `k31-<miles7>-<sglang7>` (plus
`TAG_SUFFIX` when set). After a push it prints the digest and writes
`build-record.json`. Use that digest in `--rl-image`.

**How versions are fixed.** The Miles commit inside the image is the commit
`MILES_BRANCH` resolved to at build time. It is recorded in
`/opt/yeto/image-manifest.json`, in the `ai.yeto.miles.commit` image label and
in the image-assert `commits.json`. The pinned digest therefore fixes Miles,
SGLang, Megatron-LM, Megatron-Bridge and torch_memory_saver together.

### 9.3 Code packs: `git-patch` and `release`

The trainer image fixes the frameworks. The yeto code that runs on the island
is pinned separately as the run's **code pack** (`--rl-code-pack`):

| Mode | What runs on the island | Pinned by |
|---|---|---|
| `git-patch` (the `auto` default with a bucket or `modal-volume://` checkpoint store) | Your local git tree (commit plus a patch of uncommitted and untracked files), packed once at the first start and stored in the checkpoint store | The pack digest in the run manifest |
| `release` | `/opt/yeto/src` inside a yeto release image; no local tree is shipped | The release image digest |
| `off` | The local tree at each launch, unpinned | Nothing |

A release image is built with `scripts/build_yeto_release_image.sh [--push]`
(`DEST_REPO=<registry>/<org>/yeto-release`). It adds one layer with
`git archive` of one clean commit to the image pinned as `MILES_NEXT_IMAGE`,
and it refuses a dirty work tree. Launch it with
`--rl-code-pack release --rl-image docker:<registry>/<repo>@sha256:<digest>`.
At start the island checks its source tree hash against the release manifest.
See `docker/yeto-release/README.md`.

Current scope: release images are built on `MILES_NEXT_IMAGE`, while the K3.1
full-parameter recipe defaults to `MILES_K31_FULL_IMAGE`. Release mode has been
launched on Modal only. Recommended practice for K3.1 full-parameter runs:
`git-patch` (or `auto`) on the full-parameter image. Use `release` once a
release image built on the matching base is available.

Switching a running run between packs or images goes through
`--rl-code-pack-upgrade "<reason>"` ([section 5.4](#54-upgrading-the-code-of-a-running-run)).

## 10. Full-parameter parallel layout (64 and 128 experts)

### 10.1 What is trained

Kimi K3.1 has 93 decoder layers. The first is dense, so 92 layers are MoE,
each with 896 routed experts (top-16). The full-parameter form trains a
selected subset of routed experts in every MoE layer in place. Typical subsets
are 64 or 128 per layer, and any count that EP divides is accepted. The
other experts stay frozen in MXFP4, and the non-expert weights stay frozen in
bf16. LoRA is intended for infrastructure tests only.

### 10.2 Parallelism knobs

| Flag | Meaning and constraints |
|---|---|
| `--gpu modal:<nodes>x8xb300` | Island size: whole B300 nodes of 8 GPUs. |
| `--tensor-parallel` (TP) | 8 in the provided examples (within one node). |
| `--pipeline-parallel` (PP) | Splits the 93 layers into stages. An uneven split is derived automatically (e.g. PP6: first 17, four middle stages of 15, last 16), and a split that does not add up is refused before any GPU starts. |
| `--expert-parallel` (EP) | Routed experts sharded over EP ranks. EP shares ranks with TP x DP; it is not an additional multiplier. EP must divide both 896 and the trainable count, so 8, 16, 32 or 64 for 128 experts. It must equal the selection file's `ep_size`. |
| Expert tensor parallel (ETP) | Fixed at 1 by the Miles backend. |
| `--context-parallel` (CP) | Splits long sequences. Recommended above 64k tokens. |
| `--rl-optimizer-offload-fraction F` | Keeps the fraction F of the FP32 Adam state (master, m, v) in host memory. `1.0` moves all of it to the host. |
| `--modal-memory-gib` | Host memory requested per Modal container (default 32 GiB per GPU). Size it for the offloaded optimizer state. |

GPU count = nodes x 8 = TP x PP x DP.

### 10.3 Expert selection file

`--rl-expert-selection <expert_selection.json>` (schema 1) lists, for every MoE
layer, the `trainable` expert ids and a `slot_order` (the slot-to-expert
permutation of the 896 experts for a given EP size). It also carries `sha256`
and `trainable_sha256`. Each MoE layer selects its own experts. The sha256 is
written into every release manifest and must match the sampler spec's
`expert_selection_sha256`.

Generate it with the selection tool:

```
# first run, before routing statistics exist: uniform selection
python -m yeto.rl.adapters.miles.models.partial_experts.selection --model kimi_k31 \
  --num-experts 896 --num-trainable 128 --ep-size <EP> --moe-layer-offset 1 \
  --mode uniform --num-moe-layers 92 --output expert_selection.json

# from routing statistics: sample-buffer group payloads or router counts
python -m yeto.rl.adapters.miles.models.partial_experts.selection --model kimi_k31 \
  --num-experts 896 --num-trainable 128 --ep-size <EP> --moe-layer-offset 1 \
  --payload group-*.json --output expert_selection.json      # or --counts routed_counts.json

# a new EP size for the same trainable set (keeps trainable_sha256)
python -m yeto.rl.adapters.miles.models.partial_experts.selection \
  --input expert_selection.json --ep-size <EP> --output expert_selection.ep<EP>.json

# check a file
python -m yeto.rl.adapters.miles.models.partial_experts.selection --validate expert_selection.json
```

**Moving from 64 to 128 experts** means generating a new selection with
`--num-trainable 128` (and the `--ep-size` of the new layout). Then update
`--rl-expert-selection` and `--expert-parallel` in `trainer.argv` and
`expert_selection_sha256` in the sampler spec. The trainable set changes,
so this is a new run, not a resume of a 64-expert run. `yeto launch` (and
`--dry-run`) validates the selection against `--expert-parallel` and, when the
checkpoint's `config.json` is readable, against the model's layer count
(`yeto/launch/k31_checks.py`).

### 10.4 Memory estimator

The trainer dry run (`yeto launch ... --dry-run`, part of `yeto run up`
preflight) prints a per-GPU estimate for the busiest pipeline stage
(`kimi_k31_full.memory`, from `memory_estimate` in
`yeto/launch/k31_checks.py`). It breaks HBM down into trainable experts,
frozen MXFP4 experts, non-expert weights, activation checkpoints, the MoE
working set and the logits. It also reports `host_optimizer_gib_per_gpu`.
The estimate warns above 85% of the card's capacity and never refuses a
launch.

The estimate is first-order. It leaves out allocator fragmentation, NCCL
buffers and CUDA graphs. Observed on the validated 64-expert layout below:

| | Estimator | Observed |
|---|---|---|
| HBM peak per GPU | ~162 GiB | ~251 GiB (about 1.55x) |
| Host memory per node | ~1.1 TiB (8 x 135.8 GiB offloaded optimizer) | ~1.9 TB of 2 TB |

Recommended practice: plan HBM at about 1.6x the estimate and host memory at
about 1.8x the offloaded-optimizer estimate, and keep at least 10-15% free on
both.

### 10.5 Recommended layouts

Validated: 64-expert full-parameter on 2×8 B300; 128-expert layout per
estimator.

All rows use B300 (about 282 GiB usable per GPU), TP8, CP1,
`--seq-len 32768`, `--rl-optimizer-offload-fraction 1.0`.

| Trainable experts | GPUs | Layout | Estimator HBM / GPU | HBM planned at 1.6x | Host optimizer / node (estimator) | Status |
|---|---|---|---|---|---|---|
| 64 | 16 (2x8) | TP8 PP1 EP16 (DP2) | 162 GiB | ~260 GiB (observed ~251) | ~1.1 TiB (observed ~1.9 TB) | Validated |
| 128 | 16 (2x8) | TP8 PP1 EP16 | 224 GiB | ~360 GiB | ~2.1 TiB | Exceeds a B300; not recommended |
| 128 | 32 (4x8) | TP8 PP2 EP16 (DP2) | 116 GiB | ~185 GiB | ~1.1 TiB (same as the validated 64-expert host load) | Minimum |
| 128 | **48 (6x8)** | **TP8 PP3 EP16 (DP2)** | **81 GiB** | **~130 GiB** | **~0.7 TiB** | **Recommended** |

Recommended for 128 experts: **6 nodes x 8 B300 = 48 GPUs, TP8 PP3 EP16.**
The 93 layers split evenly (31 per stage). The selection is built with
`--ep-size 16`. The estimate leaves room on HBM for longer sequences, and
host memory has comfortable headroom against the 2 TB of a node.

The 4x8 layout fits the estimate, but its host memory load equals the
validated 64-expert run, which used about 1.9 TB of 2 TB. Use it only with
care, or with `--rl-optimizer-offload-fraction` below 1.0, which trades
host memory for HBM.

For longer contexts, raise `--context-parallel` or `--pipeline-parallel` and
re-read the estimator's output (it warns for `--seq-len` >= 65536 with CP1).

Current scope: the 128-expert layouts above come from the estimator and the
observed 64-expert ratios. Recommended practice: run a short
`--rl-stop-after-rounds` job on the chosen layout and check the HBM and host
peaks before a long run.

## Planned enhancements

- The padded-sample fraction as a dashboard column next to the padded sample
  count.
- Pool configuration changes (accounts, limits) without a pool restart. Today,
  adding accounts takes a pool restart; the target can change at run time.
- Release-mode images on the K3.1 full-parameter base.
- On-hardware validation of the 128-expert full-parameter layout.
