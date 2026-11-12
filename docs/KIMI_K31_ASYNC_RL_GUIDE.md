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
| 128-expert full-parameter layout on 6×8 B300 | [K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md) |
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
9. [Images](#9-images)
10. [Full-parameter parallel layout](#10-full-parameter-parallel-layout)
11. [Daytona sampler sandboxes](#11-daytona-sampler-sandboxes)
12. [Sky and Modal rollout pools](#12-sky-and-modal-rollout-pools)

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

### Control host software

Install yeto from a clone of this repository into its own virtual
environment, and run every `yeto` and `python -m yeto...` command of this
guide with that environment's interpreter:

```
git clone <yeto-repo> && cd yeto
uv venv -p 3.12 .venv            # or: python3 -m venv .venv
uv pip install -p .venv/bin/python -e ".[launcher,modal]"
.venv/bin/yeto --help
```

Also needed on the control host: `ssh`, the AWS CLI (`aws`, for the model
upload below), the Hugging Face CLI (`hf`, part of `huggingface_hub`), the
Docker CLI (`yeto sandbox` talks to the sandbox host's Docker over
`ssh://`), and Node.js for the National Compute tool. When `https_proxy` is
set, Modal connects through the proxy and needs `aiohttp_socks`
(`uv pip install -p .venv/bin/python aiohttp_socks`); preflight stops with a
message naming it otherwise. Run the pool's `event_command` (if you set one)
with the same interpreter.

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
  Building and pinning the image: [section 9](#9-images).

### Model weights

The bucket starts empty: no step of `yeto run up` downloads the model for you.
Before the first run, put the base checkpoint where each part reads it.

| Reader | Reads from | When |
|---|---|---|
| Trainer (Modal) | The Hugging Face Hub (`--model <org>/<name>`, `--model-revision <40-hex sha>`), or a Modal volume when `--modal-model-volume <volume>` is set | At every trainer launch, inside the Modal container |
| Samplers (NC nodes) | The bucket, `s3://<bucket>/<model_pull.src_key>/`, copied onto the node by `model_pull` | Once per new node, before the engine starts |
| Control host | File listings only (Hub, Modal volume or bucket) and the small `config.json` / `*.index.json`, for the preflight model checks | During preflight |

**1. Download the checkpoint once**, on any machine with fast egress (a
cloud VM is better than a workstation for a large model), pinned to the
revision you pass the trainer:

```
hf download <org>/<name> --revision <40-hex sha> --local-dir <dir>/<name>
```

Gated models need `hf auth login` first, and the trainer needs the same Hub
token in its Modal environment.

**2. Upload it to the bucket** with the credentials of the store env file
(or an AWS profile for the same bucket). Leave out the Hub's local cache
directory:

```
aws s3 sync <dir>/<name> s3://<bucket>/models/<name> --exclude ".cache/*" \
  --endpoint-url https://<endpoint>
aws s3 ls --recursive --summarize s3://<bucket>/models/<name>/ --endpoint-url https://<endpoint>
```

The prefix must hold the files directly (`config.json`, `*.safetensors`,
tokenizer files), not a nested folder. Any prefix works, but keep it outside
`runs/<run>/` if several runs share the model.

**3. Point the pool at it**: `"model_pull": {"src_key": "models/<name>"}` in
`pool.json`. Leave `model_pull.dest` unset. The node copies the model to
`/mnt/nvme/models/Kimi-K3.1-Pro`, and the sampler container mounts that
directory at `/models/kimi-k3.1-pro` for any model. So the spec's
`engine.server.base_dir` and `agentic.tokenizer` must be
`/models/kimi-k3.1-pro`, even for a model that is not K3.1.

`pool --dry-run` (part of the `yeto run up` preflight) lists
`s3://<bucket>/<src_key>/` once with the store env file (every page of the
listing, so a checkpoint of more than 1000 files is covered) and stops when:

- `config.json` is missing, or the credentials cannot list the prefix;
- `model.safetensors.index.json` is present and a shard its `weight_map`
  names is not in the listing (the message gives the count and the first five
  missing names, e.g. `2 of 64 weight shards named by
  model.safetensors.index.json are missing: model-00007-of-00064.safetensors, ...`);
- there is no index and no single-file `model.safetensors`.

Only the index is read; no weight file is downloaded. Without this check a
missing or partial upload would surface only after a node was ordered and
billed.

**Trainer model check.** `yeto launch --dry-run` (step 4 of the `yeto run up`
preflight) and every real `yeto launch` check the trainer's `--model` the
same way before any order:

| `--model` | Checked |
|---|---|
| Hub id (`<org>/<name>`, with `--model-revision`) | The repository file list at that revision. A missing repository, revision or a gated model without a token stops the launch. A control host that cannot reach the Hub only warns. |
| Hub id with `--modal-model-volume <volume>` | `hf/<name>/<revision[:8]>/` of the volume, when its `MANIFEST.json` matches. Otherwise a warning (the trainer falls back to the Hub) and the Hub check. |
| `s3://` / `r2://` URI | One listing with the control host's store credentials. |
| Local directory | The directory, when it exists on the control host; otherwise a warning (it must exist on the trainer node). |

A problem ends the dry run with `dry run rejected: trainer model: ...`.
`--preflight-model warn` prints the problems and continues; `off` (or
`YETO_PREFLIGHT_MODEL=off`) skips the check. The dry-run plan records the
result under `trainer_model`.

**Permissions.** The store env file's key needs read on the model prefix and
read/write/delete on `runs/<run_prefix>/` (buffer, dispenser ledger, fences,
event tapes). The Modal secret named by `--modal-store-secret` needs
read/write on `runs/<run>/` (checkpoint store, releases). The control host's
profile (`env.AWS_PROFILE`) needs read on both, for the dashboard and the
first-group wait.

**Sizes and times** (measured; your links will differ):

| Model | Size | Hub download | Upload to the bucket | Node pull (`model_pull`) |
|---|---|---|---|---|
| Qwen3-0.6B (bf16), 10 files | 1.5 GB | 22 s | 228 s from a workstation link (~7 MB/s) | under 1 min on an MI355X node |
| Kimi K3.1 Pro (MXFP4) | 1.56 TB | hours; use a cloud VM | hours; use a VM in the bucket's region | ~95 min for a single `aws s3 sync` at ~275 MB/s; the parallel pull at the default 2500 MB/s cap is much faster |

**Trainer from a Modal volume (optional).** With `--modal-model-volume
<volume>` the trainer mounts the volume read-only at `/mnt/yeto-models` and
serves the model from `hf/<name>/<revision[:8]>/`. That directory must hold
the checkpoint files and a `MANIFEST.json` with `"repo": "<org>/<name>"`,
`"revision": "<40-hex sha>"` and `"all_ok": true`. If the manifest is missing
or does not match, the trainer logs `[yeto-model-volume] WARNING` and
downloads from the Hub instead.

**Expert selection** (K3.1): the trainer's `--rl-expert-selection <file>` and the
spec's `expert_selection_sha256` must describe the same file. Format and
generation: [K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md#3-expert-selection-file).

### Sandbox host

Prepare one or more hosts with Docker reachable over SSH. Task sets and
readiness are described in [SANDBOX_HOSTS.md](SANDBOX_HOSTS.md). `yeto run up`
calls `yeto sandbox status` in preflight and `yeto sandbox prepare` before
ordering any node.

**SSH keys the samplers use.** Every sampler node reaches the sandbox host
with two keys from the spec, and the host must accept both before the first
node starts. Nothing in `yeto run up` installs them:

| Spec field | Used for | Host side (`~/.ssh/authorized_keys` of the sandbox user) |
|---|---|---|
| `agentic.setup.sandbox.key.path` | `docker -H ssh://...` from the node (only `docker system dial-stdio`) | `restrict,command="<dial-only script>" <public key>` |
| `agentic.setup.tunnel.key.path` | the reverse tunnel `-R <listen>` from the node back to its records proxy | `restrict,port-forwarding,permitlisten="<ip>:<port>",... ,command="<dial-only script>" <public key>`, with one `permitlisten` for **every** port of `agentic.setup.tunnel.listen` (shard `k` listens on `lo + k`, and replacements take new shard indexes) |

The dial-only script is a forced command that allows only the Docker CLI
transport and the tunnel's empty command:

```
#!/bin/sh
case "$SSH_ORIGINAL_COMMAND" in
  "docker system dial-stdio") exec docker system dial-stdio ;;
  "") exec sleep infinity ;;
  *) echo "refused: $SSH_ORIGINAL_COMMAND" >&2; exit 1 ;;
esac
```

Keep exactly one line per key. sshd applies the first line that matches a
key and ignores later ones, so an older line for the same key (for example
one that allows only the first port) silently narrows the range: shard 0
starts, and the first replacement node, on a new shard and port, fails with
`the sandbox host refused the reverse forward`. Every replacement takes a new
shard index, so the range must be open before the run, or replacements
cannot finish unattended.

The sandbox host's sshd also needs `GatewayPorts clientspecified`, so the
tunnel can listen on the private address. If a key is missing, node setup
stops with `the sandbox host refused the docker key` (or `the tunnel key`)
and the key's fingerprint. The pool reports `CONFIGURATION ERROR`, keeps the
node (it still bills), and orders nothing more. Add the line on the host,
then run `python -m yeto.launch.nc pool-retry --config <pool.json>`.

**A host that was powered off.** Bare-metal sandbox hosts are often stopped
between runs to save cost. Power the host on through your provider (for
Cherry Servers: `cherryctl server start <server-id>`, then wait until
`ssh <sandbox-host> true` succeeds, usually 2 to 5 minutes), then check it
before `yeto run up`:

1. **The Docker data disk is mounted.** Run
   `ssh <sandbox-host> 'docker info | grep -E "Docker Root|Images"; df -h <docker-root>'`.
   If the data disk did not mount at boot (for example a RAID array that came
   up under another device name than the one in `/etc/fstab`), Docker starts
   on an empty root directory: `yeto sandbox status` then shows
   `images 0/<n>` although the images were pulled before, and
   `yeto sandbox prepare` tries to pull or build every image again. Stop
   Docker, mount the disk, start Docker, and fix `/etc/fstab`.
2. **Sidecars are stopped.** A power cycle stops every container. `yeto
   sandbox prepare` adopts a sidecar that someone else started only while it
   is running; a stopped one is refused with
   `container <name> exists and is not this run's ... free the name or
   declare another container_name`, and `up` stops with
   `sandbox prepare failed: nothing ordered`. Either start those containers
   again (as their owner), or drop `container_name` from the sidecars file so
   that `prepare` starts run-labelled sidecars of its own, which `down`
   removes again.
3. Run `yeto sandbox status --envs <envs.json> --store <store>` and confirm
   the images and sidecars counts.

Recommended practice: after the run, `yeto run down` and only then power the
host off.

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
    "max_run_containers": 40,
    "max_containers_per_sampler": 20
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

**Several sampler pools.** Instead of `pool`, a run can declare a `samplers`
list, one entry per pool (set one of the two keys, not both):

```json
"samplers": [
  {"name": "nc", "provider": "nc", "config": "<pool.json>", "interval_s": 30},
  {"name": "modal", "provider": {"kind": "modal", "image": "<sampler-image>", "secret": "<modal-secret>",
                                 "bucket": "<bucket>", "gpu": "B300", "gpus": 8,
                                 "model_volume": "<volume>", "price_usd_h": 0},
   "data_path": "r2", "usd_per_h": 0, "target": 1, "slots": [1, 2],
   "shard_base": "<run>-modal", "spec": "<sampler-spec.json>", "run_prefix": "<run>",
   "r2_env_file": "<store.env>", "sampler": {"dispense": {"backend": "store", "namespace": "<run>/dispense"}}}
]
```

| Entry key | Meaning |
|---|---|
| `name` | Pool name for `--pool` (default: the provider kind). Unique in the run. |
| `config` | A `pool.json` file. Keys given inline in the entry override the file's; the file itself is not changed (the merged config is written to `~/.yeto/rl-up/<run>/pools/<name>.json`). |
| `provider` | A kind (`nc`, `daytona`, `sky`, `modal`) or a full provider section. A kind must match the provider of `config`. |
| any `pool.json` key | `target`, `accounts` (NC) or `slots` (other providers), `shard_base`, `spec`, `sampler`, `elastic`, ... (section 3.3). |
| `data_path` | `r2` (default: buffer shards and policy releases in the run store) or `syncer` (the sampler spec takes weights from the syncer and returns samples through the relay; the pool then fences nothing in the store, the syncer retires a stopped member). All pools of a run use one data path. |
| `interval_s` | Seconds between the pool's control ticks (default 30). |
| `usd_per_h` | Price of one island hour of this pool, for the spend estimate and the scale-up budget check (default `budget.nc_usd_per_h`). |

Each pool keeps its own state under its `shard_base`, so `shard_base` must
differ between pools, and the trainer's `--rl-buffer-shards` must name the
shards of every pool (preflight checks each pool). `yeto run up` dry-runs,
starts, reports and stops every pool; `yeto run down` releases all of them.

| Field | Meaning |
|---|---|
| `run` | Run name. Also the state directory `~/.yeto/rl-up/<run>/` (pids, logs, step times) and the `ai.yeto.run` label of sandbox containers. Required. |
| `store` | Run store URI. The dashboard reads it; the first-group wait lists `sample-buffer/shards/` under it. Required. |
| `env` | Environment for every subprocess (paths may use `~`). `YETO_RESOURCE_LEDGER` defaults to `~/.yeto/runs/<run>/resources.jsonl`. |
| `preflight.commands` | Extra checks run during preflight; any non-zero exit stops `up`. |
| `sandbox.envs` | The `agentic.envs` file for `yeto sandbox status / prepare / teardown`. Optional. |
| `sandbox.docker_host` | Docker endpoint used to count and remove this run's trial containers. |
| `sandbox.max_run_containers` | Warning threshold for this run's container count (a warning, not a breaker). |
| `sandbox.max_containers_per_sampler` | Optional. Raises the warning threshold to this many containers per billed NC node, so it follows `yeto run scale`. |
| `pool.config` | Path of `pool.json`. Required unless `samplers` is set (see below). |
| `pool.interval_s` | Seconds between pool control ticks (default 30). |
| `controller` | `subprocess` (default: one pool process per pool) or `in_process` (every pool in the `yeto run up` process; see section 12). |
| `rollout.drain_on_exit` | `in_process` only. `true` (default): the exit of `yeto run up` drains every pool. `false`: the machines stay up for the next `yeto run up` to adopt (section 12). |
| `rollout.stall_alert_s` | `in_process` only. Seconds every ready sampler may stay silent (no group, no trainer-wait heartbeat) before a rollout stall warning (default 1800, `0`: off). An alert only. |
| `trainer.argv` | The `yeto launch` arguments of the trainer. Required. |
| `trainer.usd_per_h` | Trainer price per hour for the spend estimate. |
| `trainer.progress_paths` | Extra files whose growth counts as trainer progress for the silence backstop. |
| `first_group.timeout_s`, `poll_s` | How long `up` waits for the first buffer object before starting the trainer (default 5400 s, polled every 60 s). |
| `dashboard.port` | Loopback port of the dashboard (default 8795). |
| `dashboard.prices` | Price table JSON for the dashboard's cost panel (default: the built-in example table, see section 6). `up` also passes the run name and `budget.usd` to the dashboard. |
| `budget.usd` | Spend cap (trainer plus NC estimate). The NC part counts the nodes actually billed (grant to release), so it follows `yeto run scale`. |
| `budget.wall_h` | Wall-clock cap from the pool start (default 3). |
| `budget.nc_usd_per_h` | Price of one NC node hour for the estimate. |
| `budget.silence_min` | Minimum silence window of the backstop; the effective window is `max(4 x --rl-stall-timeout, silence_min)`. |
| `budget.release_nc_on_breach` | `true`: a budget or wall-clock breach performs the full `down` including the NC pool. Default `false` (see section 7). |
| `budget.release_nc_on_finish` | `true`: the full `down` (every sampler pool, any provider) runs as soon as the trainer finishes (`SUCCEEDED`). Default `false` (see section 5.5). |

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
  "run_prefix": "<run>",
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
| `target` | Samplers at the start. At most the number of `accounts`. Change it at run time with `yeto run scale` (section 5.1); the scaled value then takes precedence. |
| `accounts` | NC accounts the pool may order on; each must also be in `YETO_NC_ACCOUNTS`. Accounts beyond `target` are used for replacements. |
| `shard_base` | Shard prefix: nodes write `<shard_base>-0`, `<shard_base>-1`, ... Also names the pool state file `pool-<shard_base>.json`. |
| `shard_limit` | How many shard indexes the run may take; set it to the number of shards in the trainer's `--rl-buffer-shards`. The usable range is also bounded by the port range of `agentic.setup.tunnel.listen` (one port per shard). Shard indexes are not reused, so replacements consume new indexes. |
| `elastic.max_nodes` | Most billed nodes at once (default `2 x target`). |
| `elastic.max_orders` | Most orders over the whole run. |
| `elastic.engine_stall_s` | Seconds without generated tokens while requests are pending before an engine counts as stalled (default 600). |
| `elastic.max_sampler_restarts` | In-place restarts of a crashed sampler before its node is replaced (default 2). Restarts during node setup (before the first join) and before a `pool-retry` do not count, and the count starts again after `elastic.sampler_restart_reset_s` (default 1800) of healthy running. |
| `elastic.scale_down` | `drain` (default) or `fence`: how a scale down removes a node (section 5.1). |
| `elastic.scale_drain_timeout_s` | Longest drain of a removed node before it is fenced (default 1800). |
| `elastic.external_release` | `respect` (default): a node deleted in the NC console lowers the target and is not replaced. `replace`: it is replaced like an eviction (section 5.6). |
| `elastic.manual_hold_s` | Seconds after a manual operation during which an automatic scale request is refused (default 3600). |
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
| `engine.options.k31_preset` | Serving preset (`amd-article`, `rl-precision`, `vllm-recipe`, `rl-precision-cuda`). The ROCm presets set `--gpu-memory-utilization 0.84` and refuse a value above 0.86 (`engine.options.allow_high_gpu_memory` overrides). |
| `engine.options.tp`, `max_num_seqs` | Tensor parallel degree and concurrent sequences per engine. |
| `sampling.n` | Samples per prompt (group size). |
| `sampling.max_tokens`, `temperature`, `top_p`, `top_k` | Generation settings. |
| `max_backlog_groups` | Groups a sampler may hold ahead of the trainer before it pauses. |
| `reload_max_defer_s` | Longest time, in seconds, a downloaded policy release may wait for running agentic trials before the sampler loads it anyway (default 600; `null` waits indefinitely). A forced load is reported as `rl_sampler_policy_loaded` with `forced: true` and `deferred_s`; trials that span the switch are excluded from training (lost as `served_by_other_version`, or aborted when the switch falls inside a model call) and counted in `rl_agentic_group`. |
| `prompts.groups_per_round` | Groups a sampler produces per round. |
| `reward`, `reward_kind` | Reward function import path and kind. |
| `agentic.*` | Agentic trial settings: `max_in_flight`, `turn_max_tokens`, `max_response_len`, `infra_retries`, `task_defaults`, `setup` (sandbox, tunnel, tasks), `envs` ([SANDBOX_HOSTS.md](SANDBOX_HOSTS.md)). |

**GPU memory utilization.** To change it, add the flag to
`engine.server.extra_args`, for example
`["--gpu-memory-utilization", "0.82"]`.
Recommended practice: keep the preset's 0.84 on MI355X TP8. Measured on an
agentic CyberGym sampler (TP8, R3, top-128 log-probs, 20 trials in flight):
at 0.88 each GPU started at 256 GiB and grew to 275 GiB of 288 (95.5%)
within 30 minutes, outside vLLM's budget, at both `max_model_len` 262144 and
1048576; an in-place expert reload stages experts on top of that. The
sampler refuses a value above 0.86 unless
`engine.options.allow_high_gpu_memory` is set. At 0.84 the KV cache still
holds about 2.5M tokens, far more than 20 agentic trials of 32K tokens use.

**GPU memory guard.** The sampler samples every GPU's memory once a minute.
When a GPU is above 97% it writes `rl_sampler_vram_high` to its event tape.
While the GPU stays high it writes the event again every 10 minutes. When every
GPU is back under the threshold it writes `rl_sampler_vram_ok`. The pool
reports the alert as the pool event `sampler_vram_high`, and the dashboard
shows it on the sampler row. You can change the threshold and intervals with
the spec section `vram_guard`: `{"high_frac": 0.97, "realert_s": 600,
"interval_s": 60}`.

Before an in-place expert reload, the sampler estimates the staging memory it
needs: the largest layer of released experts, narrowed over TP, times two (the
staging copy plus the kernel-format copy), plus `engine.options.reload_reserve_gib`
(default 2). It compares this with the free memory of the fullest GPU. If
there is not enough room, the sampler first empties the engine's allocator
cache on every rank and measures again. If there is still not enough room, it
reloads each layer in smaller chunks of experts. If not even one expert fits,
it reloads by restarting the server. `memory_guard` in the
`rl_sampler_policy_loaded` event records which path was taken.

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

Add or remove NC sampler nodes while the run trains, with one command. The
trainer keeps training; nothing is restarted.

```
yeto run scale --config <run.json> --add 1                  # one more sampler node
yeto run scale --config <run.json> --remove <ticket>        # release this node (target - 1)
yeto run scale --config <run.json> --samplers <N>           # an absolute count (0 or more)
yeto run scale --config <run.json> --samplers <N> --retire <ticket>   # pick which node leaves
yeto run scale --config <run.json> --remove <ticket> --fence          # leave now, see below
yeto run scale --config <run.json> --add 2 --dry-run        # check only; nothing changes
```

`yeto run scale` checks the run's own guards, then runs
`python -m yeto.launch.nc pool-scale --config <pool.json> ...` (same flags,
with `--target` for `--samplers`). The running pool applies the request at its
next tick (within `pool.interval_s`). The new target outlives a pool restart. The `target` in `pool.json` is only the starting value: once a scale has run, the scale file's target wins, also after a restart of the same run, and editing `pool.json` changes nothing (the pool logs `the scale file wins` and records a `config_target_overridden` event). Change the number of samplers with `yeto run scale --samplers N`. A new run (another run id) ignores an older run's scale file.
`yeto run status` shows the target, the billed nodes and each node's state
(`retiring: true` while it drains); the dashboard shows the same (section 6).

**Requirements.** Scaling needs the store task dispenser
(`sampler.dispense` in `pool.json`, or `prompts.dispense.backend: "store"` in
the sampler spec). With it, every node takes its tasks from one shared ledger,
so a new node never repeats a task and a removed node's tasks go back to the
others. Without a dispenser every sampler reads every prompt, so a scale is
refused. With static `data_slots` a scale is refused unless you pass
`--force-static`: a removed node's slot then pauses (its prompts wait) until a
later scale up continues it, and the count can never pass `data_slots`.

**Scale up.** The pool orders the missing nodes like any replacement. Each new
node takes a new buffer shard (`<shard_base>-<n>`), sets up, loads the newest
released policy and starts claiming tasks. Its groups are trained on as soon
as they are written. If a node is still draining from an earlier scale down,
the pool keeps that node instead of ordering a new one.

**Scale down.** The pool picks the node to remove: the `--remove` /
`--retire` ticket, else a node not ready yet, else the newest, cheapest and
shortest-lease one. Two modes, set per run by `elastic.scale_down` in
`pool.json` and per request by `--drain` / `--fence`:

| Mode | What happens to the removed node |
|---|---|
| `drain` (default) | Its sampler claims no new task. Its trials in flight finish and their groups are written and trained on. Once it holds no task (or after `elastic.scale_drain_timeout_s`, default 1800 s), the pool fences its shard, stops the sampler, deletes the ticket and checks that it reads `ended`. |
| `fence` | Fenced at once: its task claims go back to the dispenser and the other nodes take them at their next claim, without waiting for the claim lease. Its trials in flight are lost (the tasks are redone elsewhere). Then the same release as above. |

In both modes the groups the node wrote before the fence stay in its shard and
the trainer reads them as usual; a group written after the fence is dropped by
the trainer's incarnation gate. No task is given out twice and none is lost.
No replacement is ordered for a removed node, and the other nodes are not
touched. A removed node that is evicted, crashes or reaches its lease end
while it drains is released at once, again with no replacement. The events
are `nc_elastic_scale_draining`, `nc_elastic_scaled_down` (with `outcome`:
`drained`, `drain timeout`, `fenced`, `evicted`, `lease end`, ...) and
`nc_elastic_scale_drain_cancelled`.

**Cost.** A node bills from its grant to its release, whatever the target
says. The budget estimate, the `budget.usd` breaker and the forced release
count the nodes the pool state bills, never the target. A draining node bills
until it is released, so `drain` costs up to `scale_drain_timeout_s` of one
node per removed node; use `--fence` when that matters more than the trials in
flight. A scale up is refused when the run's spend plus one node hour per new
node would pass `budget.usd`, or when the run's breaker has fired. A scale
down is never refused for cost. The sandbox container warning line follows the
billed nodes when you set `sandbox.max_containers_per_sampler`.

**Limits.** A scale is refused (exit 2, nothing changed) before anything is
ordered when:

- the target is above the usable accounts: the pool's `accounts` that are also
  in `YETO_NC_ACCOUNTS` (one node per account). Adding accounts needs a pool
  config change and a pool restart;
- the target is above `data_slots` or `elastic.max_nodes`;
- the new nodes would pass `elastic.max_orders`, `elastic.max_run_spend_ucr`
  (one node hour each) or the shard range;
- the pool's order breaker is tripped (scale up only);
- a `--remove` / `--retire` ticket is not a live node of the pool.

The shard range is the limit to plan for. The trainer reads only the shards
listed in `--rl-buffer-shards`, so `shard_limit` must equal that list and every
`<shard_base>-<i>` below `shard_limit` must be in it; `yeto run up` preflight
refuses a pool that could order a shard the trainer does not read. Shard
indexes are never reused: every new node and every lease-end replacement takes
one. For a long run that scales, list enough shards up front (largest node
count x number of 4 h leases, plus headroom). A target above the free accounts
at a given moment (for example while a replacement holds one) is not an
error: the pool reports `nc_elastic_order_refused` ("no free National Compute
account") once and orders when an account frees.

Concurrent scale requests are serialized: two `--add 1` from two shells make
+2. To remove a node, use `--remove` (or `nc down`), never the NC console; see
section 5.6 for manual operations and pauses. Details of the control file:
[NC_POOL.md, pool-scale](NC_POOL.md#change-the-number-of-samplers-while-the-run-trains-pool-scale).

**`yeto rollout`: one entry for every provider.** `yeto rollout scale` takes
the same flags and runs the same checks as `yeto run scale` (the target is a
positional argument); `yeto rollout status` lists every pool of the run:

```
yeto rollout status --config <run.json>                        # every pool: provider, target, billed islands
yeto rollout status --config <run.json> --detail               # plus each pool's full pool-status
yeto rollout scale <N> --config <run.json>                     # the run's only pool
yeto rollout scale --add 1 --config <run.json> --pool <name>   # one pool of a samplers list
yeto rollout scale <N> --config <run.json> --provider modal    # the pool of that provider
yeto rollout scale --remove <ticket> --config <run.json>       # the pool holding the ticket
yeto rollout scale <N> --pool-config <pool.json>               # a pool file directly (pool-scale only)
```

With several pools (section 3.1, `samplers`), name the pool with `--pool`, or
with a `--provider` that matches exactly one pool; `--remove` finds the pool
that holds the ticket. Otherwise the request is refused with the list of
pools. `yeto run scale`, `yeto run pause` and `yeto run resume` take the same
`--pool` / `--provider`. `status` exits 1 when an island bills with no pool
loop alive and prints the `pool-stop` command that releases it.

Provider support:

| Provider (`provider.kind`) | Island | Scale / status | Eviction refill | Data path | Verified on real machines |
|---|---|---|---|---|---|
| `nc` (default) | 8 x MI355X node, Docker over SSH | yes | yes (lease end and eviction) | `r2` | yes |
| `daytona` | GPU sandbox, process over SSH | yes | yes (spot deletion) | `r2` | no (simulated end to end) |
| `sky` | One-node SkyPilot cluster, Docker over SSH, offer rotation | yes | yes (preemption, stuck start) | `r2`; `syncer` with a syncer sampler spec | no (CPU tests with a fake sky) |
| `modal` | Modal GPU function call, no SSH | yes | yes (call exited or failed) | `r2` | no (CPU tests with a fake client) |

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

Replacement is unattended only when everything a new node needs is in place
before the run: the model in the bucket, the sandbox host's keys with the full
tunnel port range (section 2, Sandbox host), and free accounts. A
`CONFIGURATION ERROR` on a replacement halts the pool until `pool-retry`.

**What you see after a sampler is killed for memory (OOM).** Measured on a
one-node run (Qwen3-0.6B, one MI355X node):

| Time | Event | Meaning |
|---|---|---|
| t | the kernel kills the engine or sampler process (`Memory cgroup out of memory` in the node's `dmesg`) | |
| t + 7 s | `nc_elastic_sampler_restart` (`attempt` n), or `nc_elastic_sampler_crash_loop` with `detail=sampler exited exit=137` once `elastic.max_sampler_restarts` is used up | in-place restart, or node replacement |
| t + 8 s | `nc_elastic_shard_fenced why=sampler crash loop` | the shard takes no more groups; its claims return to the dispenser |
| t + 34 s | `nc_elastic_released` | the ticket is deleted and confirmed ended |
| t + 70 s | `nc_elastic_ordered ... continues_shard=<old shard>` | a replacement on a new shard index |
| t + 9 to 12 min | `nc_elastic_member_ready`; on the node `rl_sampler_start_on_latest` (`base_version` = the newest release) | the new island generates with the latest policy; the interrupted tasks are claimed again with the next attempt number, never twice at once |

The trainer keeps training on groups already in the buffer (within
`--rl-max-policy-age`) and waits otherwise. No action is needed. If OOM
repeats on every node, lower the engine's memory use (`--max-num-seqs`,
`--gpu-memory-utilization`, `agentic.max_in_flight`) and resume.

**What you see after a node is reclaimed by the provider.** Measured on the
same run:

| Time | Event |
|---|---|
| t | the ticket ends outside yeto |
| t + 30 s | `nc_elastic_released_externally ... state_was=ready`, then `nc_elastic_shard_fenced why=released` |
| t + 60 s | `nc_elastic_released` (the pool confirms the ticket is ended) |
| t + 100 s | `nc_elastic_ordered ... continues_shard=<old shard>` |
| t + 10 min | `nc_elastic_member_ready`, `rl_sampler_start_on_latest` with the newest release |

The reclaimed node's claims are released by the first sampler that scans the
dispenser after the fence (a fenced holder's claim is released at once, not
after `lease_s`). With a single sampler that is the replacement, so the tasks
continue when it joins.

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
   trial containers (label `ai.yeto.run=<run>`) and their compose networks;
   containers of other runs are not touched;
5. verifies with `pool-status` and reports `down: verified` or
   `down: NOT verified, see down.log`.

Recommended practice: run `down` when you finish. NC leases bill until their
tickets are deleted. `pool-stop` alone releases only the NC nodes
([NC_POOL.md, Stopping a pool](NC_POOL.md#stopping-a-pool)).

**When the trainer finishes.** A trainer that reaches `--rl-stop-after-rounds`
ends with `finished: SUCCEEDED`, but the NC nodes and the sandboxes keep
billing: the samplers wait for releases that no longer come. By default the
monitor logs `trainer finished (SUCCEEDED); the NC pool and the sandboxes
still bill` once and keeps its guards (the wall clock eventually stops the
run). With several `samplers` pools the banner names every pool with its
provider (`the sampler pools nc (nc), modal (modal) ...`); the same applies
to Daytona, Sky and Modal pools. Recommended practice: set `"release_nc_on_finish": true` in `budget` so
the monitor runs the full `down` as soon as the trainer finishes, or run
`yeto run down` yourself.

### 5.6 Manual operations take priority

The NC pool keeps the sampler count at its target and replaces lost nodes on
its own. A person's decision always wins over that automation: every manual
operation goes through the pool's control file (it outlives a pool restart),
is logged as an `nc_elastic_manual_*` event and is marked "manual" on the
dashboard.

**To remove one node, use one of these commands. Do not delete it in the NC
console.**

```
yeto run scale --config <run.json> --remove <ticket>          # drain it, then release it
yeto run scale --config <run.json> --remove <ticket> --fence  # release it now
python -m yeto.launch.nc down --account <id> --ticket <ticket>   # the same as --remove --fence
```

`nc down --ticket T` on a node of a running pool does not delete the ticket
behind the pool's back: it becomes the pool request `pool-scale --remove T
--fence` (target minus one), and the pool releases the node and confirms the
ticket ended. With no pool loop running, `nc down` deletes the ticket itself
and lowers the pool's recorded target, so a later pool start does not order a
node in its place. `--direct` deletes the ticket even when a pool runs.

**A node deleted in the NC console anyway** (a ticket that ends `released`
without this tool deleting it) is taken as a manual removal: the pool lowers
its target by one, orders no replacement, hands nothing on and emits
`nc_elastic_manual_release_detected` (shown in red on the dashboard). If that
node had a replacement in flight, the replacement takes its place and the
target is unchanged. A node the platform takes back (ticket ended
`preempted`, `lost` or `admin`) is still replaced automatically. To get the
old behaviour (a console deletion is replaced too), set
`"elastic": {"external_release": "replace"}` in `pool.json`.

**Pause a node for maintenance.**

```
yeto run pause --config <run.json> --ticket <ticket> [--hold-tasks]
yeto run resume --config <run.json> --ticket <ticket>
```

(or `python -m yeto.launch.nc pool-pause / pool-resume --config <pool.json>
--ticket <ticket>`). While a node is paused the pool takes no automatic action
on it: no probe verdict, no sampler restart, no retire, no fence, no
lease-end drain, no replacement order, and it is never picked by an automatic
scale down (naming it in `--remove` ends the pause). It keeps counting toward
the target, so its absence is not filled, and it keeps billing. Past its lease
end the pool only warns (`nc_elastic_manual_paused_past_lease_end`): resume or
remove it. `--hold-tasks` also stops its sampler from claiming new tasks (its
trials in flight finish), so you can inspect it without losing work; resume
lifts the hold. If the platform ends the ticket during a pause, the pause ends
and the node is handled as usual. `pool-status` lists the paused nodes
(`manual.paused`, and `paused: true` per billed node).

A pause, resume or scale takes effect within about a second: the loop's wait
between ticks ends as soon as the control file changes, and a tick re-reads the
file right before it would restart a sampler, so a sampler you stop after
`pause` returns is never restarted behind your back. On resume, a sampler that
is not running (e.g. a container you stopped during the pause) is restarted in
place once; that restart does not count toward `max_sampler_restarts`, so
repeated pause/resume cycles never trigger a crash-loop replacement.

**Manual before automatic scaling.** A scale request carries its source:
`--source manual` (the default for every command a person types) or
`--source auto` (for an automatic scaler you run yourself). An automatic
request is refused for `elastic.manual_hold_s` (default 3600 s) after the last
manual operation. `pool-status` shows who set the target (`manual.target_source`:
`manual`, `auto`, `manual_release` or `config`) and when
(`manual.manual_at`).

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
| status | `healthy`; `starting`; `waiting: backlog full (trainers behind)`; `waiting: waiting for a policy release`; `stale`; `lost: <reason> at <HH:MM>`; `left: <reason> at <HH:MM>[, replaced by <ticket>]`; `finished`; `stopped`. |
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
loading it, with the latency from the release. A replacement island that
started on the newest release shows `(at start)` instead of a latency. A
release no running sampler loaded shows `not loaded` (with
`--rl-max-policy-age` above 0 a sampler may skip versions). NC pool events
appear as dashed lines.

**NC pool events.** The `nc_elastic_*` stream (ordered, member_ready,
shard_fenced, evicted, node_lost, engine_stalled, released, target_changed,
scale_draining, scaled_down, ...). Use it to follow a replacement or a scale
end to end. Above it, one line shows the pool's sampler count: ready nodes,
nodes draining after a scale down, nodes paused by hand, the target (marked
"set by hand" after a manual change) and the shards. Manual operations are
labelled "manual" in the event list. A sampler that
left by a scale down shows as left (`scaled down`), not as lost; one still
draining keeps its normal health.

A sampler card that left names the cause from the pool events, the time it
happened and the ticket ordered in its place, for example
`lost: evicted (preempted) at 05:09, replaced by <ticket>`. The cause is the
first exit event of the node, not the fence and release that follow it:

| Card reason | Pool event |
|---|---|
| `evicted (<end reason>)` | `evicted` (preempted, lost, admin) |
| `lease expired (<end reason>)` | `expired` (lease end, replaced in turn) |
| `released outside the pool`, `released by hand (console / nc down)` | `released_externally`, `manual_release_detected` |
| `scaled down (<outcome>)` | `scaled_down` (drained or fenced) |
| `sampler crash loop after <n> restarts: <detail>` | `sampler_crash_loop` (for example an out-of-memory loop) |
| `configuration error: <detail>` | `sampler_config_error`; the card returns to healthy after `pool-retry` |
| `start timeout`, `node lost: <detail>`, `pool stopped: <why>` | `start_timeout`, `node_lost`, `pool_stopped` |

### Other panels

- **Island health**, **Event stream**: per-island state and the raw events.
- **Cost (estimate, not billing data)**: spend per resource from the resource
  ledger and the price table (`--prices`, from `dashboard.prices` in
  `run.json`). The cost bar uses `--budget` (`budget.usd`). An NC sampler is
  priced as a whole 8-GPU node, whatever TP its engine uses, from the
  ticket's order to its release (pool events); a pool node whose provider
  states its machine-hour rate (`hourly_cost`, carried on the pool's
  `member_ready` event) is priced at that rate instead of the price table. The trainer is priced from the
  launcher's record of its GPU type; a GPU type missing from the price table
  shows as unpriced, never $0.
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
| Trainer finished | launcher finished `SUCCEEDED` (exit code 0) | Banner: the sampler pools (every provider) still bill; with `budget.release_nc_on_finish` the full `down`. |
| Silence backstop | trainer log and `progress_paths` unchanged for `max(4 x --rl-stall-timeout, silence_min)` | Stop the trainer. |
| First group | no buffer object within `first_group.timeout_s` | Stop before the trainer starts. |
| Sandbox containers | count > `sandbox.max_run_containers` (or `max_containers_per_sampler` x billed nodes, if larger) | Warning only. |
| Scale up | spend + one node hour per new node > `budget.usd`, or a breaker fired | `yeto run scale` refuses; nothing changes. |

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

**Preflight stops with `model_pull: no s3://<bucket>/<src_key>/config.json`.**
The model is not in the bucket under `model_pull.src_key`, or the store env
file's key cannot read it. Upload it (section 2, Model weights) and run the
dry run again. Nothing was ordered.

**Preflight stops with `... weight shards named by model.safetensors.index.json are missing`.**
The upload is partial. Run the `aws s3 sync` of section 2 again (it copies
only the missing files) and compare the file count with the local copy.

**`dry run rejected: trainer model: ...`.**
The trainer's `--model` is missing, unreadable or incomplete where the trainer
reads it (section 2, Model weights, Trainer model check). Fix the name,
`--model-revision`, the Hub token or the volume snapshot.

**`sandbox prepare failed: nothing ordered`.**
Read `sandbox.log`. After a sandbox host reboot the usual causes are an
unmounted Docker data disk and stopped sidecars (section 2, A host that was
powered off).

**Every trial is lost with `RewardFileNotFoundError` within seconds.**
Open a trial directory on the node (`/mnt/nvme/work/harbor/trials/<trial>/`,
`verifier/test-stdout.txt`). `task-server unreachable at http://<ip>:<port>`
means the task files name a fixed address of the sandbox host's Docker
bridge, which can change when the host restarts. Write
`{{yeto.sidecar.<name>.url}}` (or `{{yeto.host.bridge_ip}}`) in the task files
instead; `yeto sandbox prepare` renders the current address
([SANDBOX_HOSTS.md](SANDBOX_HOSTS.md)). Run `prepare` again; replacement
nodes pick up the rendered tasks. The dispenser settles tasks that fail
`max_redo` times as `exhausted`; later passes dispense them again.

**`the sandbox host refused the docker key` or `refused the reverse forward`.**
The key line on the sandbox host is missing, or an older line for the same
key wins (section 2, SSH keys the samplers use). Fix the line, then
`python -m yeto.launch.nc pool-retry --config <pool.json>`.

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
The message names the limit (usable accounts, `max_nodes`, `max_orders`,
spend, the run budget, a breaker, the shard range, or a missing task
dispenser). Shard indexes are not reused. For a long run with many
replacements, set a generous `--rl-buffer-shards` list and `shard_limit` up
front. Try a request first with `--dry-run`.

**A removed node is still billed.**
In `drain` mode it bills until its trials in flight finish (at most
`elastic.scale_drain_timeout_s`). `yeto run status` shows it with
`retiring: true`. To release it now, repeat the request with `--fence`, for
example `yeto run scale --config <run.json> --samplers <N> --fence`.

**`up` stops at the first group.**
No buffer object appeared within `first_group.timeout_s`. Check `pool.log`
for node setup progress and the samplers' event tapes. On the next attempt,
consider `--trainer-now` so the trainer start overlaps the node setup.

**The trainer stopped with `BREAKER: trainer: log / tape silent`.**
The launcher was alive but produced no output for the silence window.
Inspect `trainer.log` and the Modal app. Resume with `yeto run up` (section
5.3). NC nodes are kept, so the restart is quick.

**The tape shows `rl_buffer_damaged_payload_skipped`.**
The trainer passed over groups whose payload object is missing or is not a
group (cut short, not JSON, another schema). Each is consumed, not trained,
and named in the event (`position`, `sample_id`, `payload_ref`, `reason`);
the metric is `sample_buffer/damaged_payload_groups`. One or two are a damaged
or deleted object. A payload the store failed to return (a throttle, a 5xx, a
timeout) is never counted here: that read fails the round and is read again.

**The trainer stopped with `DamagedPayloadLimit`.**
8 payloads in a row (or 8 that are more than half of what one round read)
did not decode: not single damaged objects but a systematic cause, usually
samplers of another code version writing another payload schema, or a
lifecycle rule deleting payloads. Nothing was consumed; the trainer stops at
the same position on every restart until the cause is fixed. To pass the
groups over anyway (their tasks are settled in the dispenser and are not
generated again), set `YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN` in the trainer's
environment (`0`: no limit).

**`rl_buffer_exhausted` carries `passed_over`.**
The last round found no trainable group; `passed_over` counts what it
consumed without training (`stale`, `stale_payload`, `stale_incarnation`,
`quarantined`, `dropped_small`, `nonfinite_reward`, `damaged_payload`,
`duplicate_task`). The
same counts end the error text of a round that waited `wait_s` without a new
group. `stale_payload` are groups whose index record passed the age gate but
whose payload holds an older policy version: the payload decides.

**The tape shows `rl_buffer_duplicate_task_skipped`.**
Two groups of one store-dispenser task reached the buffer (a sampler stalled
past its dispense lease and wrote after another sampler had taken the task
over, or a written group was not yet settled when its node was retired). The
trainer trains the first group of a task, in buffer order, that it accepts for
training and passes over the later ones; each is named in the event (`slot`,
`plan`, both positions and sample ids) and counted in
`sample_buffer/duplicate_task_groups`. A task is a slot of one dispenser
namespace (`plan` is the namespace and a digest of its plan), so the same data
dispensed again under another namespace is trained again. Occasional events
need no action. `unverified_groups` counts the verdicts whose first group lies
below the watermark a restarted trainer resumed from: the trainer takes such a
group as trained, though the earlier process may have passed it over (too old,
too small, a damaged payload). In that case the task is trained zero times,
never twice. `YETO_SAMPLE_BUFFER_ONE_GROUP_PER_TASK=0` in the trainer's
environment turns the rule off.

**`down` reports `NOT verified`.**
Read `down.log`. Usually a node release is still in progress. Run
`python -m yeto.launch.nc pool-status --config <pool.json>` and, if nodes are
listed, `pool-stop` again.

## 9. Images

### 9.1 Trainer image (Miles)

Use this image for the full-parameter recipe:

```
docker:ghcr.io/agentenv/yeto-miles-k31@sha256:2de5d796de19d38f4e3165efce2a9528a7f7d3e60e3cb32fe29fd00c6abf9b5b
```

It is the default `--rl-image` for `--rl-model-recipe kimi_k31 --tuning full`
(`MILES_K31_FULL_IMAGE` in `yeto/rl/adapters/miles/pins.py`). It is also the
image the trainer ran in the validated Modal end-to-end run. It carries Miles
`k31/integ-trainer` `1f29e43043`. Recommended practice: pin this digest
explicitly in `trainer.argv`
(`--rl-image docker:ghcr.io/agentenv/yeto-miles-k31@sha256:2de5d796...`, the
full value above) so the run does not depend on a default.

The image `yeto-miles-k31@sha256:63ad0d2a...` (`MILES_K31_IMAGE`) is for the
LoRA recipe.

**Registry.** The default registry is `ghcr.io/agentenv`, and no setting is
needed to use it when its packages are readable by your Modal workspace. If
the trainer fails to pull the image (an anonymous pull of a private package
answers `403`), ask for read access, or use `--rl-image-private` below, or a
mirror. Set `YETO_IMAGE_REGISTRY=<registry>/<org>` only when you
pull from a mirror that carries the same digests. Because images are
referenced by digest, a mirror cannot serve different bytes under the same
pin. For a private registry, add `--rl-image-private` with
`SKYPILOT_DOCKER_USERNAME` / `SKYPILOT_DOCKER_PASSWORD` (see
[CLOUDS.md](CLOUDS.md)).

The island checks its Miles checkout against the commit recorded for the
image digest (`IMAGE_MILES_COMMITS` in `pins.py`).

### 9.2 Sampler image (vLLM ROCm)

NC MI355X samplers run the upstream vLLM ROCm image, pinned in
`yeto/rl/sampler_pins.py` (`VLLM_SAMPLER_IMAGE_ROCM`):

```
docker:docker.io/vllm/vllm-openai-rocm@sha256:5f47cc1004d9a7ac65f8a913725edb20c62c99a07e31cbf4ba21741bd0b4f9b1
```

The image reports vLLM version `0.0.0+mum.14d1ec2336`. The sampler spec's
`engine.server.engine_version` and `engine.options.pin_version` refer to it.
The node installs yeto into the image at start.

### 9.3 Building your own Miles image

Source: `docker/miles-k31/` in this repository. Most customers use the
published digest above. This section applies when you rebuild it.

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

Before it builds anything, the script compares the base's recorded build
arguments with the lock and stops on any mismatch. It tags the image
`k31-<miles7>-<sglang7>`. After a push it prints the digest and writes
`build-record.json`. The Miles commit inside the image is the commit
`MILES_BRANCH` resolved to at build time. It is recorded in
`/opt/yeto/image-manifest.json`, in the `ai.yeto.miles.commit` label and in
the image-assert `commits.json`.

### 9.4 Code packs: `git-patch` and `release`

The image fixes the frameworks. The yeto code on the island is pinned
separately as the run's **code pack** (`--rl-code-pack`):

| Mode | What runs on the island | Pinned by |
|---|---|---|
| `git-patch` (the `auto` default with a bucket or `modal-volume://` checkpoint store) | Your local git tree (commit plus a patch of uncommitted and untracked files), packed once at the first start | The pack digest in the run manifest |
| `release` | `/opt/yeto/src` inside a yeto release image; no local tree is shipped | The release image digest |
| `off` | The local tree at each launch | Nothing |

`scripts/build_yeto_release_image.sh [--push]`
(`DEST_REPO=<registry>/<org>/yeto-release`) builds a release image. It adds a
`git archive` of one clean commit to the image pinned as `MILES_NEXT_IMAGE`,
and refuses a dirty tree. See `docker/yeto-release/README.md`.

Current scope: release images are built on `MILES_NEXT_IMAGE`, which is the
LoRA-recipe image. Recommended practice for K3.1 full-parameter runs: use
`git-patch` (or `auto`) with the full-parameter image in section 9.1.

Moving a running run between packs or images goes through
`--rl-code-pack-upgrade "<reason>"` ([section 5.4](#54-upgrading-the-code-of-a-running-run)).

## 10. Full-parameter parallel layout

The recommended 128-expert layout, its measured data, the selection file
format, the full launch command and the validation steps are in
**[K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md)**. This section is a
summary. Where the two differ, that document applies.

### 10.1 Validation status

- Validated: 64-expert full-parameter on 2×8 B300 (end to end with NC
  samplers).
- Measured: 128-expert at 524k total context on 6×8 B300, PP split 32/28/33.
- Recommended for production: 128-expert at 262k total context on 6×8 B300,
  PP split 32/29/32. This split is derived from the memory model fitted to
  the 524k measurement.

### 10.2 Layouts

| Trainable experts | GPUs | Layout | Context | Status |
|---|---|---|---|---|
| 64 | 16 (2×8 B300) | TP8 PP1 EP16 CP1 | 32k | Validated |
| 128 | 48 (6×8 B300) | PP3 x TP8 x CP2 x EP16, DP1, split 32/29/32 | 262k | Recommended |
| 128 | 48 (6×8 B300) | PP3 x TP8 x CP2 x EP16, DP1, split 32/28/33 | 524k | Measured upper bound |

The 128-expert layout needs `--rl-allow-cross-node-tp`,
`--rl-max-tokens-per-gpu` = `seq-len / CP`, the split flags and the stage-0
gradient hook in `YETO_MILES_TRAIN_EXTRA_ARGV`, and
`--rl-distributed-timeout-minutes 60`. See
[K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md#4-layout).

Parallelism notes, from the launcher:

- `--expert-parallel` shares ranks with TP x DP; it is not a further
  multiplier. It must divide both 896 and the trainable count, and it must
  equal the selection file's `ep_size`.
- Expert tensor parallel is fixed at 1 by the Miles backend.
- An uneven PP split is passed with
  `--decoder-first/last-pipeline-num-layers` and checked before any GPU
  starts.

### 10.3 Expert selection and moving from 64 to 128

The selection file format and commands are in
[K31_FULL_262K_LAYOUT.md, section 3](K31_FULL_262K_LAYOUT.md#3-expert-selection-file).
To move from 64 to 128 experts:

1. Generate a new selection with `--num-trainable 128 --ep-size 16`.
2. Update `--rl-expert-selection` and `--expert-parallel` in `trainer.argv`.
3. Update `expert_selection_sha256` in the sampler spec.

The trainable set changes, so this starts a new run rather than resuming a
64-expert one.

### 10.4 Memory estimator and headroom

The trainer dry run prints a first-order per-GPU estimate
(`kimi_k31_full.memory`, from `memory_estimate` in
`yeto/launch/k31_checks.py`). It includes `host_optimizer_gib_per_gpu` and
warns above 85% of the card's capacity. The estimate leaves out allocator
fragmentation, NCCL buffers and CUDA graphs. On the validated 64-expert
layout:

| | Estimator | Observed |
|---|---|---|
| HBM peak per GPU | ~162 GiB | ~251 GiB |
| Host memory per node | ~1.1 TiB offloaded optimizer | ~1.9 TB of 2 TB |

Recommended practice: plan HBM at about 1.6x the estimator and host memory
at about 1.8x the offloaded-optimizer estimate. For the 128-expert layout,
size from the measured per-stage figures in
[K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md#7-measured-data-6-x-8-b300),
and confirm headroom on the first step as described there.

## 11. Daytona sampler sandboxes

The sampler pool can take its nodes from Daytona GPU sandboxes (AMD MI355X,
spot) instead of National Compute. Only the supply changes: eviction handling,
shard fencing, claim release, refill, the budget, the breaker and the orphan
patrol are the same pool code as for National Compute.

Status: the provider is covered by CPU tests against a fake Daytona SDK; it has
not yet run against a live Daytona account. Starting the sampler inside a
sandbox (no Docker in a sandbox) is a separate step of the node route.

### Configuration

In `pool.json`, set `provider` to the Daytona kind. `slots` (`accounts` is
accepted as an alias) are plain indexes (`[1, 2, ..., max_nodes]`), not accounts:
one sandbox per slot at a time. Logs, pool events, `yeto rollout status` and the dashboard name these `slot` (for example `slot=2`); only National Compute pools say `account`, because there the slot is an NC account. Event JSON also keeps an `account` field with the same value for older readers.

```json
"provider": {
  "kind": "daytona",
  "image": "rocm/vllm:<tag>",
  "gpus": 8,
  "gpu_type": "MI355X",
  "spot": true,
  "cpu": 128,
  "memory_gib": 1536,
  "disk_gib": 4096,
  "name_prefix": "yeto-smp",
  "ssh_token_minutes": 1440
}
```

| Key | Meaning |
|---|---|
| `image` / `snapshot` | Exactly one. With `snapshot`, the snapshot's own resources apply (build it with MI355X x 8); `gpus`, `cpu`, `memory_gib` and `disk_gib` apply to `image` only. |
| `gpus` | 1 to 8. |
| `gpu_type` | `MI355X` (other Daytona GPU types are accepted). |
| `spot` | `true` (default) or `false` for on-demand. |
| `ttl_minutes` | Wall-clock limit per sandbox; `0` (default) is none. |
| `queue_timeout_min` | Minutes an on-demand create may wait for capacity. |
| `create_timeout_s`, `delete_timeout_s` | Client-side waits (default 900 s / 300 s). |
| `env` | Environment variables of the sandbox. |
| `host_setup` | Commands run over SSH once a sandbox is up. |
| `target` | Daytona region, if the organization has one. |
| `api_key_env`, `api_url_env` | Names of the environment variables holding the API key and URL (default `DAYTONA_API_KEY`, `DAYTONA_API_URL`). |

Every sandbox is created with auto-stop off (`auto_stop_interval=0`) and
auto-delete on stop: Daytona deletes a stopped GPU sandbox, so the default
15-minute idle stop would end a sampler that waits for the trainer.

`pool --dry-run` checks the section, the API key variable and that the
`daytona` Python package is installed; it makes no Daytona call.

### Account requirements

- An API key in `DAYTONA_API_KEY` on the pool host (`DAYTONA_API_URL` only for a
  non-default endpoint). Never put the key in `pool.json`.
- Spot GPU sandboxes need Daytona Tier 2 or higher and GPU (or spot GPU)
  credits. Lower tiers restrict outbound network: the sampler must reach the
  object store, the model source and the trainer, so confirm the tier's network
  policy before the first run.
- Check MI355X spot capacity in the Daytona dashboard or with the
  `gpu-capacity` endpoint; a spot create with no capacity fails at once.

### Spot reclaim semantics

Daytona deletes a spot sandbox without notice when an on-demand user needs the
capacity (Started, Deleting, Deleted, with `spotEvictedAt` recorded). The pool
reads that as an eviction, the same as a National Compute eviction:

1. the shard is fenced, so the trainer frees its claims at once and ignores
   anything the lost sandbox still writes;
2. the sandbox is confirmed deleted;
3. a replacement sandbox is created and continues the shard and its data slot.

There is no preemption notice and no lease end, so only eviction-driven
replacement applies. A sandbox that reads stopped, archived or failed is
treated as lost the same way.

### Orphans and teardown

Every sandbox carries the labels `yeto-run=<run>` and `yeto-slot=<slot>` and a
name `<name_prefix>-<run>-<n>`, and is recorded locally before the create call.
The patrol lists sandboxes by the run label, so a sandbox whose create outlived
the pool process is adopted (or deleted by `pool-stop`). The provider deletes
only sandboxes with its name prefix that carry this run's label or record.

### Differences from National Compute

| | National Compute | Daytona |
|---|---|---|
| Node | Bare-metal 8 x MI355X, Docker | VFIO GPU VM sandbox, no Docker |
| Lease | Time-limited lease, replaced before its end | No lease end; optional `ttl_minutes` |
| Reclaim | Eviction with a ticket end reason | Spot deletion without notice (`spotEvictedAt`) |
| Access | SSH through the workspace relay | SSH through `ssh.app.daytona.io` with a per-sandbox token (renewed automatically) |
| Balance guard | Account balance in ucr | None. `elastic.max_run_spend_ucr` still caps node-hours, priced at the National Compute MI355X rate; also set a Daytona spend limit |
| Accounts | NC accounts in `YETO_NC_ACCOUNTS` (`accounts`) | Slots, one sandbox each (`slots`; shown as `slot=`) |

## 12. Sky and Modal rollout pools

Two more providers run under the same pool policy (scale, eviction ->
fence -> refill, breaker, budget) and the same `yeto rollout` entry
(section 5.1). Neither has run on real machines yet; treat them as preview.

**Sky** (`"provider": {"kind": "sky", ...}`): one SkyPilot cluster per island,
reached over SSH with sky's key, running the Docker sampler route.

```json
"provider": {"kind": "sky",
             "candidates": {"gpus": ["H100", "H200"], "spot_only": true, "infra": "nebius"},
             "cluster_prefix": "yeto-smp", "ssh_key": "<sky-ssh-key>",
             "start_timeout_s": 1800, "host_setup": ["sudo docker info"]}
```

- Machines come from `offers` (`[{"resources": {<sky Resources args>}, "price_usd_h": <usd>}]`),
  or `candidates` (the spot watchdog's Nebius list: on-demand H100 first, then
  spot cheapest first; `prices` names a price file), or one `resources` block
  with `price_usd_h`. An order is a (slot, offer) pair. A failed launch, or a
  cluster that never finishes starting, cools that offer down (10 min, doubling
  per repeat, at most 80 min) and the next order tries the next offer; a
  success clears the offer and moves back to the cheapest. The pool's breaker
  counts a failure only when every offer is cooling down.
- A cluster still starting after `start_timeout_s` is released and replaced; a
  stopped or missing cluster reads as preempted and is refilled.
- Only clusters named `<cluster_prefix>-<run>-<n>` that this pool recorded or
  sky lists under that prefix are ever taken down.
- Orders and releases are written to the run's resource ledger (`sky_cluster`).
- `slots` (alias `accounts`) are plain indexes `1..N` (one cluster or Modal
  island each), not accounts; logs and status show them as `slot=`.
- `python -m yeto.rl.spot.watchdog` still works; `--via-provider` sends its
  machine calls through this provider. For new runs, prefer a sky pool.

**Modal** (`"provider": {"kind": "modal", ...}`): one Modal GPU function call
per island, no SSH. The pool deploys one app (`app_name`, default
`yeto-rollout-<run>`) with a `rollout_sampler` function and spawns one call per
island with the island's rendered sampler spec.

| Key | Meaning |
|---|---|
| `image` | Sampler image (required). |
| `secret` | Modal Secret with the store credentials (required). |
| `bucket` | Bucket of the run store (required). |
| `gpu`, `gpus` | GPU type and count of the function (default `B300`, 8). |
| `model_volume`, `model_mount` | Modal volume with the weights, mounted read-only (default mount `/mnt/yeto-models`). |
| `timeout_h` | Function timeout; a call that reaches it is refilled as a lost island (default 24). |
| `cpu`, `memory_gib`, `env`, `python` | Function resources, extra environment, sampler Python. |
| `price_usd_h` | Price of one island hour (resource ledger). |

- An order bills nothing; an island bills from its spawn until its cancel.
- A running call is not a joined sampler. Each call writes a ready marker
  (`pool-ready/<shard>/<ticket>.<n>.json` in the run store) once its sampler's
  event tape has `rl_sampler_join` (model loaded, lease claimed); only then is
  the island ready. A call that never reports ready within
  `elastic.start_timeout_s` (default 3600) is a failed start: it is cancelled
  and replaced, and counts toward the breaker.
- `model_pull` and `node_hooks` are not available (no node); put the weights
  in a Modal volume.
- Calls and the app are written to the run's resource ledger
  (`modal_call`, `modal_app`), so `yeto cleanup --run` finds them.

**Daytona** sandboxes are written to the run's resource ledger
(`daytona_sandbox`); set `price_usd_h` in the provider section to price them.
Like NC leases they belong to the pool, as do the Modal calls and Sky clusters
of a sampler pool: a plain `yeto cleanup --run <run>` keeps them, and
`yeto cleanup --run <run> --include-pools` releases the ones still open (every
pool provider; `--include-nc` is its deprecated earlier name).

**Spend caps.** Each provider prices its machines in its own currency (NC in
ucr, Sky / Modal / Daytona in USD from `price_usd_h`); the pool converts every
rate to one unit before it adds them up, so a mixed pool's spend is correct.
`elastic.max_run_spend_ucr` and `elastic.max_run_spend_usd` both cap the
pool's orders (the USD-to-ucr rate is `YETO_NC_USD_PER_UCR`, default `1e-6`).
Also set `usd_per_h` on the `samplers` entry (it drives `budget.usd` of
`yeto run up`) and a spend limit at the cloud provider.

**Syncer data path.** `"data_path": "syncer"` on a `samplers` entry (or the
pool file) sends weights and samples through the actor syncer instead of the
run store. Set `syncer: {"address": "host:port", "key_env": "<env var with the
island HMAC key>"}` on the entry or once at the top of the run config. The
pool writes `data_path` and `syncer` into each sampler spec, fences and drains
islands through the syncer's member table, and marks a replacement sampler
with `predecessor_retired: true` when the pool has already retired the island
it replaces. The syncer keeps its drain set in memory only; the pool re-sends
every drain it still wants (scale-down victims, `--hold-tasks` pauses) on each
tick, so a restarted syncer is drained again within one pool interval. The
syncer dispenser retries a row at most `--dispense-max-redo` times (default 2,
the same as the store ledger's `max_redo`); a lease expiry counts, so a row
outlives two holders that died with it and is settled `exhausted` at the third.

**One controller for every pool.** By default `yeto run up` starts one
`python -m yeto.launch.nc pool` process per pool. With `"controller":
"in_process"` in the run config, all pools run in one rollout controller
inside the `yeto run up` process (state files, scale files, locks and
`pool-stop` work the same way). That process must stay up, so `--no-monitor`
is refused in this mode. `yeto launch --rollout-pool <pool.json>` (repeatable)
does the same inside the launcher's fleet controller. `yeto rollout status`
reports each pool's controller view (target, members, pending handoffs,
breaker) read from its saved state, without touching any machine.

The controller also watches the rollout side: when every ready sampler with a
progress signal (its event tape grew, or its engine generated tokens) has been
silent for `rollout.stall_alert_s` (default 1800; `0` turns it off), and none
of them is waiting for the trainer, `yeto run up` (and the launcher's fleet
controller, event `rollout_stalled` in `fleet.jsonl`) logs a rollout stall
warning once. It is an alert only: nothing is stopped. A sampler blocked on
the trainer writes wait heartbeats and is never counted as stalled. Modal
calls give no progress signal and are not part of this check.

**The in-process controller needs a host that stays up.** With
`"controller": "in_process"` the sampler pools are driven by the `yeto run up`
process itself. Run it under tmux or systemd on a host that stays up for the
whole run, not in an interactive SSH session that can drop. What happens when
that process exits is set by `rollout.drain_on_exit`:

| `rollout.drain_on_exit` | When the `yeto run up` process exits (normal exit, Ctrl-C, SIGTERM) |
|---|---|
| `true` (default) | Every pool is drained: samplers stopped, machines released and confirmed ended. |
| `false` | Nothing is drained. The machines keep running and billing, and no pool drives them (no refill, no breaker, no budget cap) until the next `yeto run up --config <run.json>` adopts them from the pool state files. The exit prints the tickets still billing. |

Use `false` only with a supervisor that restarts the process at once, for
example this systemd unit:

```ini
# /etc/systemd/system/yeto-run-<run>.service
[Unit]
Description=yeto run up <run> (in-process rollout controller)
After=network-online.target
Wants=network-online.target

[Service]
User=<user>
WorkingDirectory=<yeto-checkout>
EnvironmentFile=<env-file-with-credentials>
ExecStart=<yeto-checkout>/.venv/bin/yeto run up --config <run.json>
Restart=on-failure
RestartSec=30
KillSignal=SIGTERM
TimeoutStopSec=1800

[Install]
WantedBy=multi-user.target
```

```json
"controller": "in_process",
"rollout": {"drain_on_exit": false, "stall_alert_s": 1800}
```

`TimeoutStopSec` gives a `drain_on_exit: true` drain time to confirm every
release before systemd kills the process. `systemctl stop` sends SIGTERM,
which runs the same exit path as Ctrl-C. To end the run for good, use
`yeto run down --config <run.json>`, not `systemctl stop` with
`drain_on_exit: false`.

## Planned enhancements

- The padded-sample fraction as a dashboard column next to the padded sample
  count.
- Pool configuration changes (accounts, limits) without a pool restart. Today,
  adding accounts takes a pool restart; the sampler count can change at run
  time (`yeto run scale`).
- Automatic scaling from the buffer backlog and the trainer's wait time.
- Release-mode images on the K3.1 full-parameter base.
- A measured first step of the 262k layout (split 32/29/32) under production conditions.
