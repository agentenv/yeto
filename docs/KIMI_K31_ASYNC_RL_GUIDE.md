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
replacement ready, you need `N + 1` accounts. Provide `N + 2`. The pool takes
an account out of use (it *halts* it) when it cannot prove that a node release
it sent hit its own node: National Compute's delete command cannot name the
node. A halted account stays out until a person has checked it; the events
`nc_elastic_foreign_ticket_released` and `nc_elastic_delete_unconfirmed`
announce it. The steps are in
[NC_POOL.md, "A halted account"](NC_POOL.md#a-halted-account-clear-halt):
run `python -m yeto.launch.nc clear-halt --account <account-id>`, check the
account in the National Compute console, then run it again with `--yes`.

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

The tunnel forwards to the sampler's records proxy, so the spec must give
that proxy a fixed port: set `agentic.proxy.port` (for example
`"proxy": {"port": 18080}` under `agentic`), or `agentic.setup.tunnel.target_port`
to the same value. Without it, node setup refuses the spec with
`agentic.setup.tunnel needs a fixed records proxy port`.

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
                                 "model_volume": "<volume>", "price_usd_h": "<USD per island hour: required>"},
   "data_path": "r2", "target": 1, "slots": [1, 2],
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

**Prices of Modal / Daytona / Sky pools must be real.** These providers bill in USD,
and the run's spend estimate counts their islands at the entry's `usd_per_h`, else the
provider section's `price_usd_h` (Sky: or `offers[0].price_usd_h`). The placeholder
above is not a price: replace it with your actual rate (Modal: the GPU price per
second on modal.com/pricing x 3600 x `gpus`, plus CPU / memory; Daytona: your contract
or dashboard rate; Sky: the instance price). When `budget.usd` is set, `yeto run up`
preflight refuses a USD-priced pool whose price is missing, not a number, or `<= 0`
(it would be counted at $0 and the budget guard would never stop it). National
Compute pools are priced by `budget.nc_usd_per_h` and are not affected.
`"budget": {"usd": null}` (or no `usd`) means no spend cap: preflight says so and
only the wall clock stops the run.

Each pool keeps its own state under its `shard_base`, so `shard_base` must
differ between pools, and the trainer's `--rl-buffer-shards` must name the
shards of every pool (preflight checks each pool). `yeto run up` dry-runs,
starts, reports and stops every pool; `yeto run down` releases all of them.

| Field | Meaning |
|---|---|
| `run` | Run name. Also the state directory `~/.yeto/rl-up/<run>/` (pids, logs, step times; set `YETO_STATE_DIR` to use another directory in place of `~/.yeto/rl-up`) and the `ai.yeto.run` label of sandbox containers. Only `yeto run up` creates the state directory: `run status` on a run without one exits 2 (most likely a misspelled name); `run down` warns and still stops what the config names (trainer, pools, sandboxes), reporting NOT verified. Required. |
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
| `budget.usd` | Spend cap (trainer plus NC estimate). The NC part counts the nodes actually billed (grant to release), so it follows `yeto run scale`. Cumulative over the whole run: a `yeto run down` followed by `yeto run up` does not reset it. |
| `budget.wall_h` | Wall-clock cap in hours of the current `yeto run up`, counted from the moment that `up` starts the pools; a new `yeto run up` after `down` (or a stop or crash) starts it again, a trainer restart inside the same `up` does not (section 5.5, "Clocks across down and up"). Default 3 when the key is absent: the trainer is stopped 3 h after `up` starts the pools, and preflight (and `yeto run up --dry-run`) prints a WARNING when the key is missing. `null` means no wall-clock cap (same meaning as `budget.usd: null`); a long production run (50-100 h) should set its hours or `null`. A value that is not a positive number fails preflight. |
| `budget.nc_usd_per_h` | Price of one NC node hour for the estimate. |
| `budget.silence_min` | Minimum silence window of the backstop; the effective window is `max(4 x --rl-stall-timeout, silence_min)`. |
| `budget.release_nc_on_breach` | `true`: a budget or wall-clock breach performs the full `down` including the NC pool. Default `false` (see section 7). |
| `budget.release_nc_on_finish` | `true`: the full `down` (every sampler pool, any provider) runs as soon as the trainer finishes (`SUCCEEDED`). Default `false` (see section 5.5). |

Both the budget breaker and the wall-clock breaker work on yeto's own estimate,
not on the platform's live bill: the spend is node uptime as yeto sees it times
the configured hourly prices (`trainer.usd_per_h`, `budget.nc_usd_per_h`, the
pool prices). The estimate drifts from the real bill when a price is wrong or
missing, when Modal bills time yeto does not count (cold start, image pull), and
when a crashed process leaves containers or nodes running that yeto no longer
tracks. Check the provider's billing page for the authoritative number.

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
| `--rl-round-cuts {auto,off}` | `auto` (default): round cuts as the store and the cut flags give them. `off`: no round cut is wired, written or read; a relaunch continues from the newest policy release. The K3.1 examples use `off` (one full-parameter cut is about 5 TB). Refused together with `--rl-cut-every`, `--rl-cut-auto-mtbf-s`, `--rl-cut-max-interval-s`, `--rl-cut-keep`, `--rl-stop-after-rounds` and `--rl-resume-allow-config-change`. See section 5.3. |
| `--rl-cut-every N` | Keep a round cut every `N` rounds (plus the last round). Not with `--rl-round-cuts off`. See also `--rl-cut-keep`, `--rl-cut-auto-mtbf-s`. |
| `--rl-release-every N` | Release every N-th policy version to the samplers (default 1). Needs `--rl-max-policy-age >= N`. |
| `--rl-release-skip-busy` | Skip a release while the previous upload is still in flight (latest wins; the trainer never waits for an upload). Needs staged full-weight releases (`--tuning full`, no `--rl-release-inline`) and `--rl-max-policy-age >= 3`. See section 3.5. |
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
| `elastic.max_orders` | Most orders over the whole run, counted cumulatively across every `yeto run up` of the run (the count lives in the pool state, so a restart does not reset it; `max_run_spend_ucr` / `max_run_spend_usd` likewise). |
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
| `agentic.weight_switch` | How an agentic sampler switches weights: `drain` (default on engines without live updates) stops starting new trials, waits for the running ones, then loads the release; `live` loads into the running engine. |
| `agentic.drain_timeout_s` | Longest a `drain` switch waits on running trials before it cancels them (lost as `drain_timeout`). The effective limit is min(`agentic.drain_timeout_s`, `reload_max_defer_s`): with `reload_max_defer_s` 600 and `drain_timeout_s` 3600, a drain ends after 600 s. A larger `drain_timeout_s` (or none) is reported as a warning by `pool --dry-run` and at sampler start (not for a spec that resolves to `live`). **Behaviour change:** a drain-mode spec without `drain_timeout_s` used to wait for its trials to end by themselves; it is now cut at `reload_max_defer_s` (default 600 s) and the trials still running are cancelled and lost. Set `reload_max_defer_s: null` (and no `drain_timeout_s`) to keep the old wait. Must be a number >= 0 or `null`. Keep it at or below `reload_max_defer_s`: while a drain waits the engine starts no new trial, and one trial running into an hour-long agent timeout otherwise idles the node for that hour. |
| `agentic.abandon_grace_s` | After a drain's timeout cancels its trials, the longest wait (default 120 s) for a trial that does not honour the cancel, for example one blocked in a long sandbox tool call. It is then counted lost (`drain_timeout`), detached, the drain ends and the release loads; its late result is dropped and its recorder session is closed (no further model output); its sandbox is left to the trial's own teardown. A detached trial keeps its sandbox and its concurrency slot until it ends by itself (at most its agent timeout): until then that many fewer new trials start; `agentic.detached_in_flight` in `rl_sampler_engine_round` and `detached_in_flight` in `rl_agentic_drain` count them. A number >= 0 (`null` is refused). |
| Release pickup in drain mode | The sampler downloads a new release in the background as soon as `LATEST` changes and keeps starting trials meanwhile; the drain starts once the release is on disk. The drain is for the release that is on disk, and it always ends in a load of that release: a newer release published during the drain is downloaded, drained for and loaded afterwards (releases published faster than a drain do not starve the sampler; it stays behind by at most the releases of one drain). The cut and the detach are timers, so writing groups does not delay them: the last trial ends at most min(`agentic.drain_timeout_s`, `reload_max_defer_s`) + `abandon_grace_s` (only if a trial ignores the cancel) after the drain began (`trials_ended_s` of `rl_agentic_drain`). The groups that ended are written before the drain ends and the release loads, so a sampler that dies in the load loses nothing it finished; `waited_s` is `trials_ended_s` plus the time to write what was still unwritten. From `LATEST` to loaded takes about download + `waited_s` + load. A call that waits (every slot busy or held by detached trials) writes `rl_agentic_waiting` every 60 s. A failed download is reported as `rl_sampler_release_not_loaded` (`download_failed`) and retried; a release of another weight scheme is not downloaded. |
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

### 3.5 1M context, concurrency and release cadence

Measured on one 8 x MI355X sampler (TP8, `--gpu-memory-utilization` 0.84,
`max_model_len` 1048576, R3, top-128 log-probs, fp32 head) with a fixed
Terminal-Bench set of 8 tasks x 8 trials. Each item is marked *measured* or
*not verified*.

**Context limits.** `agentic.max_prompt_len` and `agentic.max_response_len`
only cut the training sample written to the buffer: a sample whose first-turn
prompt is longer than `max_prompt_len` is ABORTED, and a response is cut at
`max_response_len` (later turns are not trained on). They do not limit a
running trial; only `engine.server.max_model_len` does, so a trial can reach
1M with both set to 16384 (*measured*). For 1M training (*not verified* end to
end), set `max_prompt_len` to the real first-turn bound and `max_response_len`
to `max_model_len - max_prompt_len`, and raise the trainer's `--seq-len` and
`--rollout-max-response-len` to match.

**Cost of 1M requests** (*measured*):

| Item | Value |
|---|---|
| One 1.04M-token request | 201-250 s (prefill about 180 s), KV 42-52% |
| KV capacity at gmu 0.84 | 2,550,775 tokens: about 2 full-length 1M requests at once (2: KV 82%, no preemption; a 3rd queues) |
| R3 payload per turn | the routing of the whole context, about 3.9 GB at 1.04M tokens (0.96 GB at 250k); a 30-turn trial growing to 1M moves about 60 GB (estimated) |
| Other trials on the node during a 1M prefill | request e2e 11.5 -> 61 s (about 5x), ITL 29 -> 121 ms, generation 357 -> 145 tok/s |
| `/metrics` | blocked while the API server serializes a GB-sized reply; the pool reports `engine_metrics_unreadable`, which is not by itself a stall |

**Concurrency** (`agentic.max_in_flight`, *measured*, 64 trials per level):

| `max_in_flight` | Done | trial/h | Trial wall mean / p90 s | ITL ms | KV peak | VRAM peak GiB |
|---|---|---|---|---|---|---|
| 20 | 64/64 | 137.8 | 360 / 866 | 26.5 | 10% | 264.2 |
| 32 | 63/64 | 141.9 | 476 / 980 | 32.6 | 15% | 256.4 |
| 48 | 53/64 | >= 182 (lower bound) | 405 / 589 | 40.5 | 25% | 267.3 |

The limit is ITL growing with the decode batch and long-tail trials, not KV
and not the sandbox host (CPU <= 14% busy); about two thirds of trial time is
tool execution. Recommended: 48; use 32 when 1M-long trials are mixed in or
per-trial latency matters. Keep `max_num_seqs` >= `max_in_flight`.

**One weight switch in drain mode** (*measured* under full load at 48 in
flight, `drain_timeout_s` 300): `LATEST` seen after about 8 s; the release is
prefetched in the background for 138 s while trials keep starting; once it is
on disk (`release_ready`) the drain starts and is cut at
min(`agentic.drain_timeout_s`, `reload_max_defer_s`); cancelled trials that
ignore the cancel are detached after `abandon_grace_s` (120 s). The drain took
498 s (10 trials abandoned, 0 detached), then the in-place load 50 s (engine
pause 8.1 s), VRAM peak 267.2 GiB. That run predates two changes that are
covered by tests only (*not measured on hardware*): the cut and the detach
are timers that group writes no longer delay (in that run the writes ran
about 160 s behind and the cut was issued at about 404 s instead of 300 s);
and a release published during a drain no longer cancels that drain's load. **Each release costs a sampler about 9-12
minutes of useful time**, and concurrency falls off during the drain.

**Release cadence.** If the release interval is under about an hour, that
loss is above 15%. A sampler loads at most one release per drain, so with
releases more frequent than a drain plus a load (about 8-10 minutes here) it
skips the ones published meanwhile and serves weights up to one drain old;
releasing more often than that buys nothing on a drain-mode sampler. `--rl-release-every N` (needs `--rl-max-policy-age >= N`)
or `--rl-release-skip-busy` (needs `--rl-max-policy-age >= 3`; the lag is
about ceil(upload_s / step_s) + ceil(download_s / step_s) + backlog groups /
groups per step versions, so a 12-20 minute K3.1 upload over ~200 s steps is
4-6 versions alone) release less often; both need `--rl-sample-role trainer`
on the Miles backend. Skipped versions are reported as
`rl_policy_release_skipped`. Raise `--rl-max-policy-age` with them and watch
`rl_buffer_stale_skipped` and KL (throughput gain *not verified*).

**4 h lease.** An NC node is released at its 4 h lease end. A pool with only
one account has no replacement and the node is simply gone (*measured*). For
anything longer than 4 h minus 10 minutes, list replacement accounts
(`replace_lead_s` orders ahead); the dry run must not print
`replacement accounts: none`.

**Trainer at 1M.** A full-parameter trainer at 1M needs
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: without it reserved
memory exceeds allocated by 73-89 GiB and 1M does not fit within PP3. With
it, 32 or 64 experts fit on PP2 x TP8 x CP4 = 64 GPUs and 128 experts want 96
GPUs or 64 GPUs with activation offload (measured on 4- and 16-layer cut
models, extrapolated to 93 layers; full 1M training *not verified*). The
launcher allows expandable segments on a full-parameter trainer island
(#353). See also section 10.4.

## 4. Launch

### 4.1 Dry run

```
yeto run up --config <run.json> --dry-run
```

Before preflight, the run config itself is loaded: a missing file, invalid
JSON, an unknown key or a malformed `first_group` / `budget` / `dashboard` /
`rollout` / `starvation` value ends `yeto run` with one line naming the file
and the field (`run config <file>: ...`), exit 2.

Preflight then runs every check and orders nothing:

1. the Modal SDK check (when the trainer targets `modal:`), then the checks
   that need no subprocess: `budget.usd` / `budget.wall_h` (a WARNING when
   `wall_h` is absent; a refusal for a value that is not a positive number or
   null, for `budget.usd` 0 with prices set, and for a USD-priced pool without
   a price), the `starvation` section, a refusal of `--rl-stall-timeout 0`, and
   per pool the shard range and the trainer/sampler consistency checks;
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

It prints the step history, process liveness, buffer object count (`null` when the store listing failed, e.g. no store credentials on this host), the spend
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
the trainer reads them as usual; nothing the node writes after the fence is
trained (see "What a fence is" in section 5.2). No task is given out twice and
none is lost.

**The gate fails closed, and pauses training.** To judge a group the gate reads
its shard's incarnation lease (`sample-buffer/leases/<shard>.json`). A lease
that is absent (no such key) accepts the group. A lease that *cannot be read*
(a timeout, a 5xx, a reset connection, a truncated body) is no verdict: that
group is neither trained nor counted stale, and the trainer reads the buffer in
strict order, so **every group behind it waits too, from every shard**. The
round then holds only the groups before it, and its committed watermark is
exactly that group's position, so a restore never trains a group twice. The
price: one shard whose lease stays unreadable pauses the whole training (the
samplers keep writing; the groups wait in the buffer). The trainer logs, at
most once a minute, `incarnation gate: training paused: lease of shard <X>
unreadable since <T>`, and the next take counts the waiting groups in
`sample_buffer/lease_deferred_groups`. While it waits, neither the no-new-group
timeout nor the data end applies, and nothing stops the wait except the lease
becoming readable again or the trainer process being stopped (the reader has
no stop or drain signal of its own). A sampler whose previous lease cannot be
read at its start waits for it (up to 10 minutes, ending early on a reclaim or
stop) rather than overwrite it.
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

The reclaimed node's claims are released by the fence itself (the pool retires
the sampler in the dispenser ledger; a fenced holder's claim is released at
once, not after `lease_s`), and the other samplers take them at their next
claim. With a single sampler that is the replacement, so the tasks continue
when it joins.

**What a fence is.** Each sampler writes its groups into its own shard of the
buffer, `sample-buffer/shards/<shard>/<position>.json`, one object per record,
each created only if the position is still free. A fence is one more record in
that sequence, an *ownership record*
(`{"schema": "yeto.rl.sample-buffer/v2", "pos", "owner": {"incarnation",
"epoch", "by", "claimed_at"}}`): a sampler writes one for itself when it joins
(`by: sampler`), the pool writes `pool-fence:<ticket>` when it fences a shard
or restarts a stalled sampler in place (`by: pool`). One rule decides what is
trained, the same for every reader: a group counts if the last ownership
record before it names the sampler process (incarnation) that wrote it, or if
no ownership record precedes it. So:

- every group the node created before the fence record is trained, including
  the ones written in the last seconds before it;
- nothing a fenced process writes afterwards is trained. Its write reads the
  shard first and is refused (`ShardFenced`, nothing is created; the task goes
  back to the dispenser), and the trainer applies the rule again when it
  merges the shards, so a process that skipped the check gains nothing
  (`sample_buffer/stale_incarnation_groups`, event `rl_buffer_merge_skipped`
  with `reason: stale_incarnation`, the shard and the owner's epoch);
- a sampler that cannot read its shard several times in a row (5 by default)
  writes no group until a read succeeds again; such a group's task goes back
  to the dispenser uncounted (`release`, like a fenced one), so a store
  outage does not use up `max_redo`;
- a fenced process cannot mark its shard as ended either. A shard the pool
  fenced therefore never reads as ended (before, a drained sampler that was
  then fenced could still write its end mark). Nothing in a K3.1 run waits for
  that: the trainer is started with the list of shard names
  (`--rl-buffer-shards`), and the names not in use never end, so the buffer's
  "every shard has ended" state is not reached by scaling down. The run ends
  by the trainer's own step count and `yeto run down`;
- no clock takes part: the control host's and the nodes' clocks may differ by
  any amount. `claimed_at` is information only.

`rl_shard_owner` (shard, incarnation, epoch, by) is emitted by the sampler at
its join and by the pool at a fence. A sampler that joined before the trainer
of a new run was up starts under the older lease record and writes its
ownership record at one of its next fence polls (`adopted: true` in the
event; every `incarnation_lease.poll_s`, 30 s by default). `sample-buffer/leases/<shard>.json` is
still written, for the dashboard and for older readers; it no longer decides
anything on such a shard.

Storage: one ownership record (about 200 bytes) per sampler start and per
fence, in the shard it concerns; the merged index holds none. A sampler's
fence check is the incremental listing of its own shard that every write does
anyway. A pool fence lists the shard from the last ownership record on (one
request per 1000 records) and reads one or two records.

The merge also passes over a shard record it cannot read (not JSON, another
schema, a missing field): `sample_buffer/bad_shard_records`, event
`rl_buffer_merge_skipped` with `reason: bad_record`; the other groups of that
shard and all other shards are trained as usual. A shard position that is
missing while later ones exist (an object deleted from the bucket by hand or by
a lifecycle rule) is not skipped: the trainer prints a warning and emits
`reason: missing_shard_position`, and the shard's later groups wait until the
object is back. A sampler restarted on a shard with a damaged record passes
over it the same way instead of failing at its start.

### 5.3 Resuming training

- **K3.1 default: round cuts off.** The K3.1 trainer examples
  (`examples/k31/launch-trainer-modal-b300.sh`,
  `examples/k31/launch-convertible-modal.sh`) pass `--rl-round-cuts off`. A
  full-parameter K3.1 round cut holds the bf16 weights, the fp32 master copy
  and both Adam moments of every trained expert: about 5 TB per cut, too much
  to write at a useful interval. With `off` the trainer wires no round cut: it
  writes none, reads none, and logs `round cuts are off` with the event
  `rl_round_cuts_off` at start. The cut flags (`--rl-cut-every`,
  `--rl-cut-auto-mtbf-s`, `--rl-cut-max-interval-s`, `--rl-cut-keep`,
  `--rl-stop-after-rounds`, `--rl-resume-allow-config-change`) are refused with
  `off` before anything starts; so is a convertible node
  (`--rl-sample-role convertible`), which hands its trainer state over through
  round cuts. Other runs keep the default `auto`, which behaves as before.
  `off` is refused with `--rl-backend verl`: the verl island has no
  `--rl-init-from-release` and does not read the buffer on from a release's
  trained position, so a relaunch could not continue correctly.
  Resume with `off`:
  - Every launch path continues from the newest release on its own; no flag
    has to be added by hand:
    - `yeto launch` (also a plain re-run of
      `examples/k31/launch-trainer-modal-b300.sh`): every trainer node of a
      full-parameter trainer island runs
      `python -m yeto.rl.engine.init_from_release --release latest --if-released`
      before its Ray starts. When the store holds a trained policy release it
      prepares the newest one and logs `adding --rl-init-from-release latest`;
      when it holds none (the run's first launch) it logs `no policy release
      yet` and the trainer starts from the base checkpoint. This is decided on
      the node, not on the launcher: the launcher may have no store credential
      (a Modal Secret), and a dry-run command is used long after the launch.
    - Convertible spec (`examples/k31/launch-convertible-modal.sh`): the
      `role_switch.trainer.command` from the dry run carries
      `--rl-round-cuts off`; before running it, the container prepares the
      newest release the same way and adds `--rl-init-from-release-dir`.
    - `yeto run up`: when it restarts the trainer of a run that launched
      before, `trainer.argv` has `--rl-round-cuts off`, the store holds a
      trained policy release and `trainer.argv` names no
      `--rl-init-from-release` itself, it adds `--rl-init-from-release latest`
      and logs that the optimizer state starts fresh (step
      `trainer_init_from_release` in the run state). If the store cannot be
      listed, the restart stops with an error instead of guessing.
    "Holds a trained policy release" is judged the same way everywhere: a
    readable `RELEASE.json` or `LATEST.json` under `policy-releases/` that is
    not a derived (re-quantized) release. A `RELEASE.json` whose `LATEST.json`
    was never written counts; a torn record does not.
  - Safety net in the trainer: with `off`, a trainer that finds releases
    `v1..vK` in the store but loaded the base checkpoint (no
    `--rl-init-from-release`, for example an island command started by hand)
    stops before training with `RoundCutsOffRestartError`: it would continue
    the numbering at `vK` and read the buffer from `vK`'s trained position, so
    the groups before that position would never reach the weights. Relaunch
    through `yeto launch`, add `--rl-init-from-release latest`, or start a new
    run under a fresh run prefix / store.
  - The trainer loads the newest release, continues the version numbers after
    it, and reads the sample buffer from the position that release was trained
    up to (`buffer_trained`). If the buffer refuses that position (beyond the
    index the store holds), the trainer emits
    `rl_buffer_start_position_refused` and reads from the start, as with round
    cuts; the `--rl-max-policy-age` gate passes over the stale groups. (A
    trainer that cannot be told the start position at all stops with an error
    instead of reading from the start.)
  - Cost of this resume: the Adam moments and step count start from zero and
    the fp32 master copy is rebuilt from the bf16 release weights; the LR
    schedule position is not restored (warmup and decay restart from the
    beginning of the schedule); the RNG state is reset. At most one release
    interval of training is lost (`--rl-release-every` versions): the rounds
    trained after the newest release are trained again from the buffer.
  - To resume from a specific release instead, put
    `--rl-init-from-release r<seq>-v<version>` in `trainer.argv` (it must be
    the store's newest; see the next item).
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
- **Without a cut, on a store that holds releases** (round cuts `auto`; with
  `off` see the first item). A trainer that stopped
  before its first cut was committed leaves releases `v1..vK` and no cut. The
  next launch starts from the base checkpoint, continues the version numbers
  at `vK`, and reads the sample buffer from the position the weights of `vK`
  are trained up to (`buffer_trained` in the release record and in the
  `rl_buffer_resume_versions` event), not from the start. Groups trained into
  `vK` are not read again; groups the stopped trainer selected after it
  published `vK` (their update never reached the store) are trained again.
  The earlier batch ledger is moved aside as above.
  A release record written by an earlier yeto has no `buffer_trained`. Its
  `buffer_selected` is the trainer's read position when the upload of the
  release ended, which can lie past groups `vK` never trained, so it is not
  used: the launch reads the buffer from the start. Of the groups it reads
  again:
  - groups older than `--rl-max-policy-age` versions are passed over as stale
    (`rl_buffer_stale_skipped`);
  - groups inside the window of a version whose release is no longer in the
    store are passed over as unidentified (next item);
  - groups of `vK-1` and `vK` (the two releases the store keeps) are trained.
    Those the stopped trainer had already trained are trained a second time:
    at most the groups of two versions.
  This happens once: after the new launch has released one policy, the newest
  record has `buffer_trained` and the fallback is not taken again.
- **Groups without a policy identity after a resume.** A relaunched trainer
  starts at `vK` and reads the buffer on from an earlier position (a cut's
  watermark, `buffer_trained`, or the start). The store keeps the newest two
  releases, so a group inside the `--rl-max-policy-age` window can be of a
  version below `K` that no record names any more. The trainer cannot tell
  which weights generated it: it consumes the group without training it and
  counts it (`rl_buffer_unidentified_policy_skipped` per round, with the
  versions; `sample_buffer/unidentified_policy_groups`; the dashboard's
  trainer totals; `trainer_buffer` in `yeto run status`). Only versions below
  the start version `K` of a relaunched trainer are treated this way. A group
  of a version from `K` on that was never published, or a label with another
  hash, still stops the trainer with `not a published policy`.
  The base policy `v0` has no release either (the samplers start on the base
  checkpoint), but it is named while it is inside the window: by the weights
  the trainer loaded when it starts without a cut from the base checkpoint,
  or by the batch ledger restored with a cut (`rl_base_policy_restored`).
  Groups of `v0` are then trained. They are passed over only when neither
  exists: a start with `--rl-init-from-release` and no cut, or a cut whose
  ledger was started after round 0 (the gate event then has
  `base_version_named: false`).
  When the reader reads groups and keeps none 3 selections in a row
  (`YETO_BUFFER_UNIDENTIFIED_STARVED_SELECTS`), it logs a warning and emits
  `rl_buffer_unidentified_policy_starved`; it keeps waiting for groups of a
  release the store names.
  Scope: the resume behavior of this section (`buffer_trained`, the batch
  ledger moved aside, groups without a policy identity passed over, no cut
  before the first trained round) is the Miles trainer's, the one this guide
  runs. A verl buffer trainer does not have it.
- **Behind the releases.** When the latest cut is older than the newest
  release, the trainer restores the cut and continues the version numbers at
  the newest release (`rl_round_cut_behind_releases`). It keeps no cut before
  it has trained a round (`rl_round_cut_skipped`); the next due cut is kept.

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

Mixed versions on one run store (the fence format of section 5.2):

| Trainer | Samplers / pool | What happens |
|---|---|---|
| new | new | Ownership records fence the shards. A shard that already holds groups from before the upgrade keeps them: its part before the first ownership record is read under the older rule (lease record, 60 s clock margin). |
| old (no trainer generation lease in the store) | new | The samplers and the pool write no ownership record and use the lease record as before. Nothing changes for the old trainer. |
| new | old, on a run started by old code | Works as before the upgrade (lease record). Restart the samplers on the new code to get the fence above; the dispenser plan of the run stays at its version. |
| any | old, on a run started by new code | Refused at the sampler's start: `dispense mismatch: version is 2 here, 3 in .../plan.json`. Use the new sampler image. |
| old, after a new trainer ran on the store | - | Not supported, and refused: see "No way back" below. |
| new | new samplers, old pool | Not supported. The old pool fences by the lease record only, which an owned shard ignores: groups the fenced process writes before it stops are trained. Upgrade the pool together with the samplers. |
| new | old, still running when the new trainer starts | Not supported (see the upgrade order below): a pool fence or a new sampler writes an ownership record into the shard, and the old sampler stops at its next read of it. Nothing is misread. |

**No way back to older trainer code.** Once a shard of the run holds an
ownership record, code from before ownership records cannot read the run's
buffer, and it says so instead of misreading it:

| Older code | What it does on such a store |
|---|---|
| trainer (and the head's view of the merged index) | Stops at its first read of the merged index: `SampleBufferError: sample buffer: record out of order: {"marker": "shard-ownership", "note": "written by yeto with shard ownership records: code older than them cannot read this buffer", ...}`. It merges and trains nothing, on every restart. The new trainer writes that marker into the merged index before the first group it merges from an owned shard. |
| sampler on a shard with an ownership record | Stops at its start with the same `record out of order` error, showing the ownership record. With a store dispenser it is refused even earlier: `dispense mismatch: version is 2 here, 3 in .../plan.json`. |
| the head's count of unmerged shard records (`UnmergedShards`) | `KeyError: 'sample_id'` on the ownership record. Not a clear message; it is not a misread either. |
| trainer, in the short window after a new sampler joined and before the new trainer merged anything | No marker exists yet. The older trainer copies the shard's records, the ownership record included, into the merged index and then stops with `record out of order` (showing that record). The copied groups were not checked against the ownership rule. The current code reads on from such an index (it passes over the copied record and counts it, `foreign_merged_records`), so going forward again works; do not rely on it. |

The merged group records themselves are unchanged for older readers (two more
fields, `shard_epoch` and `shard_passed`, that older code ignores). To go back
to older code, start a new run prefix.

**Upgrade order for a running run: samplers and pool first, the trainer
last.** Ownership records need an atomic exclusive create: on a store without
one (an endpoint that refuses `If-None-Match`, the OSS relay) the trainer claims
no generation and everything stays on the lease record fence. The trainer is the switch: the new format starts when the new trainer
claims its generation lease in the store (`trainer-lease/<generation>.json`,
one more at every start of a trainer that merges the shards of an object
store; a generation is only this format switch, it does not keep two trainers
from running at once), and from then on the new samplers and the pool write ownership records.

1. Bring the samplers up on the new image (drain the old ones, below) and
   restart the pool on the control host with the new code. With the old
   trainer still running they write no ownership record and keep the run's
   dispenser plan at its version: nothing in the store changes format, and
   going back to the old samplers and pool is still possible.
2. Then restart the trainer on the new code (`--rl-code-pack-upgrade`). At its
   start it claims the generation lease; every new sampler takes its shard with
   an ownership record at its next lease poll (event `rl_shard_owner`,
   `adopted: true`; within `incarnation_lease.poll_s`, 30 s by default), and
   the pool's fences of those shards become ownership records too (a shard
   that holds none yet is fenced by its lease record, as before).

The other order (trainer first) is not supported: once the new trainer ran, a
new sampler takes its shard with an ownership record (and the pool's later
fences of that shard are ownership records too). An old sampler started on such
a shard cannot read the record: it stops at its start (or at its next write)
without writing past it, and every restart of it on old code stops again until
it runs the new image: safe, but the shard does not continue.

**Rollback.** Before step 2: roll the samplers and the pool back as usual.
After step 2 there is no rollback to older code on the same run store, of the
trainer, the samplers or the pool (an older pool's lease-only fence does not
fence an owned shard) ("No way back" above): older code stops on the new
records instead of misreading them, so a rollback costs the run, not its data.
To go back, start a new run prefix (another `--cluster-prefix`), from the last
released policy if you want to keep the training so far.

**Upgrading the samplers: drain the old ones first.** The trainer trains one
group per dispensed task (`rl_buffer_duplicate_task_skipped`, section 8). It
recognises a task by two fields the sampler writes into each group's index
record, `dispense_attempt` and `dispense_plan`. Samplers on code from before
these fields write groups without them, and such a group is not judged: if an
old sampler and a new one both write a group of the same task (the old one
stalled or was retired with the group written but not settled, and the task
was given out again), both groups are trained. So do not run old and new
samplers side by side longer than needed:

1. Scale the old samplers down with `drain` (section 5.1): each claims no new
   task, finishes its trials in flight, writes and settles their groups, and
   is then fenced and stopped. Wait for `nc_elastic_scaled_down` with outcome
   `drained` for every node; a `drain timeout` or `fenced` outcome means that
   node's open tasks were released and will be generated again.
2. Start the samplers on the new code in one step.

Groups already in the buffer from the old samplers are trained as before;
they are only not protected against a second group of the same task. The
trainer may be upgraded before or after the samplers: it reads records with
and without the fields.

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

**Clocks across down and up.** Each `yeto run up` process records its period
(start, last monitor poll, end) in `ups` of `~/.yeto/rl-up/<run>/state.json`
and sets `t0` to its own start. The two caps differ:

| | Counts from | Reset by a new `yeto run up` |
|---|---|---|
| `budget.wall_h` (and the 1.2x wall cap of the NC breaker) | `t0`, the start of the current `up` | yes |
| Starvation watch; first-group wait (skipped once the run has a first group) | the current `up` process | yes |
| `budget.usd` (spend estimate) | `run_t0`, the first start of the run; NC nodes billed grant to release; the trainer from its first launch, inside the up periods only (a stopped gap bills no trainer) | no |

A restart inside the same `up` process (for example the trainer restarted by
its scheduler) keeps the current `up`'s clock. A run whose state predates the
periods keeps its old `t0` as `run_t0`, so its spend stays counted.

`yeto run status` reports under `uptime` the number of ups, the cumulative
runtime (`cumulative_runtime_h`: the up periods summed, the stopped gaps
between `down` and the next `up` excluded), the current up's runtime, the
time left on `budget.wall_h`, and the cumulative spend estimate. The
dashboard started by `up` (`--up-state <state.json>`) shows the same as the
"Run uptime" card. An `up` that ended without `down` (crash, kill) counts
until its last monitor poll.

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

### 5.7 Object store requests, retries and limits

Every island talks to the run's bucket with one S3 client. This section says how
many requests a run sends, what happens when the store refuses one, and which
settings change that.

**What a refusal looks like.** R2 limits concurrent requests. Under load it
answers `503 ServiceUnavailable: Reduce your concurrent request rate for the same
object` (seen on `ListObjectsV2` while other clients wrote and deleted keys under
the same prefix). The client retries by itself; the island sees an error only when
every retry was refused.

**Retries and timeouts (defaults).**

| Setting | Default | Meaning |
|---|---|---|
| Retry mode | `standard` | Retries 500 / 502 / 503 / 504, connection errors and throttling codes. The wait before retry *n* is `random() * min(2^(n-1), 20)` seconds, so islands refused together do not return together. |
| Requests per call | 9 (the first plus 8 retries) | 8 waits: 45 s on average, 91 s at most. Measured against an endpoint that refuses everything: 26 to 42 s of wall time per call. |
| Connect timeout | 10 s | A connection that is not open after 10 s is retried. |
| Read timeout | 60 s | Longest silence inside a response. |
| Connection pool | file concurrency x part concurrency (128) | One connection per transfer thread. |

Each client also has a retry quota: about 100 retries without a success in
between use it up, and a refused request then fails at once until successes
refill it. In a long refusal storm expect errors after seconds, not after 91 s.

`adaptive` mode adds a client-side rate limit, but it starts only after a
throttling error *code* such as `SlowDown`. R2's `503 ServiceUnavailable` is not
one, so `adaptive` does not slow a client down for it. Keep `standard`.

To change the defaults, set these in the environment of the launcher (they are
copied to every island with the other store settings) or in the pool's store env
file:

| Variable | Effect |
|---|---|
| `YETO_RL_STORE_MAX_ATTEMPTS` (or `AWS_MAX_ATTEMPTS`) | Requests per call, the first included. `12` waits up to 151 s. |
| `YETO_RL_STORE_RETRY_MODE` (or `AWS_RETRY_MODE`) | `standard`, `adaptive` or `legacy`. |
| `YETO_RL_STORE_CONNECT_TIMEOUT_S`, `YETO_RL_STORE_READ_TIMEOUT_S` | The two timeouts, in seconds. |
| `YETO_RL_STORE_MAX_CONNECTIONS` | Connection pool size. |
| `YETO_RL_STORE_PART_CONCURRENCY`, `YETO_RL_STORE_FILE_CONCURRENCY`, `YETO_RL_STORE_DOWNLOAD_PROCESSES`, `YETO_RL_STORE_PART_MB` | Parallelism of large transfers (policy releases). |

The `YETO_RL_STORE_*` name wins over the `AWS_*` name. Each variable is read on
its own: `AWS_MAX_ATTEMPTS` alone keeps `standard` mode.

Keep one call shorter than a third of the dispenser lease (`lease_s / 3`, 300 s
by default): a sampler renews its tasks one request at a time, and a renewal
round that takes longer than the lease loses the tasks to other samplers.

**Policy release downloads.** A release is 100 to 200 GB, read as 64 MiB ranges.
An NC sampler reads with 8 processes x 4 files x 16 ranges (512 concurrent
requests, about 1.4 GB/s on an MI355X node), so 16 samplers that pick up the same
release at once send about 8000 concurrent range reads, up to 256 of them on one
file. Two things keep that from failing a download:

- *Start spread.* Each sampler waits a fixed part of `YETO_RL_RELEASE_SPREAD_S`
  seconds before it starts a release download (30 on NC samplers, 5 elsewhere).
  The part is derived from the sampler's host name
  (`YETO_RL_RELEASE_SPREAD_KEY` overrides it), so it is the same at every release
  and two samplers keep their distance. The samplers still overlap for most of a
  download; the spread only keeps them from opening all their reads in the same
  second. It delays the release on a sampler by up to the spread, so keep it
  well below the trainer's release interval (for example 30 s against a release
  every few minutes); otherwise the last sampler serves one more version behind.
- *Automatic slow-down.* When the store refuses a range read for its request
  rate (`503 ServiceUnavailable`, `503 SlowDown`, `429`) after the client's own
  retries, the download halves the number of range reads that process runs at
  once, waits (1 to 30 s, with jitter) and reads the range again, up to 6 times.
  Each 8 reads without a refusal give one read back. The sampler log shows
  `[s3-store] WARNING: ... the store refused a range read for its request rate`.
  A download fails only when one range is refused 6 times in a row.

If those warnings come at every release, lower `YETO_RL_STORE_FILE_CONCURRENCY`
(32 on NC samplers): the download slows down in proportion, and each file still
has at most 16 reads per sampler.

**How many requests a run sends.** With 16 samplers and about 100,000 tasks in 24
hours:

| Source | Requests per hour | Notes |
|---|---|---|
| Dispenser scans | about 10,000 `LIST`, 2,000 `GET` | 5 listings and 1 read per scan, a scan per sampler every 30 s at most. Independent of the ledger's size. |
| Lease renewals | 12 `PUT` and 12 `HEAD` per task in flight | One write per task every `lease_s / 3`. |
| Claims and settlements | about 8,500 `PUT`, 12,500 `HEAD` | Per task: one claim, one settlement, three checks before its group is written. |
| Tape upload | 120 `PUT` per island | One chunk every 30 s (`YETO_RL_TAPE_UPLOAD_S`). |
| Dashboard | about 6,500 `LIST` | One poll of the store every 10 s (`--store-poll`), one listing per island tape. |

A scan lists only the tasks that are not settled yet. It costs more when old
unsettled tasks lie far behind the newest ones: 4 more listings for each such
group, and up to a listing of the whole ledger (one request per 1000 keys) when
unsettled tasks are spread over all of it, for example tasks whose sandbox is
never ready or a task label that no sampler serves.
`python -m yeto.rl.engine.store_dispense status --open` shows the unsettled tasks
with far fewer requests than the full table.

Set `prompts.dispense.full_scan` to `true` in the sampler spec (or
`YETO_RL_DISPENSE_FULL_SCAN=1` on a sampler) to make every scan list the whole
ledger again; use it only to rule the range scan out while debugging.

### 5.8 Re-running tasks that are already settled

The dispenser never hands out a settled task again. To re-run one by hand, delete
its settlement record `<namespace>/done/<slot, 12 digits>.<kind>` from the store
(`<namespace>` is `prompts.dispense.namespace`, by default `task-dispense`; the
slot is in `store_dispense status`).

A running sampler does not notice the deletion. Each sampler remembers which
tasks it has seen settled and does not list them again (section 5.7), so the
task is handed out only by a sampler process that started after the deletion.
Restart the samplers that should take it (pause the node with `--hold-tasks`,
wait for its trials to finish, stop its sampler container, then resume: section
5.6); a restarted sampler lists every settlement once. It finds the task's last
claim without a settlement, ends that attempt as `expired` once its lease has run
out, and hands the task out again. The expiry counts toward `max_redo`: a task
that had already used its retries is settled `exhausted` instead of re-run.

With `full_scan` on (section 5.7) every scan lists the whole ledger, and a
running sampler sees the deletion at its next scan, at most `max_scan_s` (120 s)
later. Turning `full_scan` on needs a restart of the sampler as well, so it only
helps when it was on before the deletion.

Do not delete records of tasks that are in flight, and do not delete the
`claims/` records: the attempt numbers continue from them.

### 5.9 Robustness behaviour of the samplers, trainer, store and pool

This section lists what the samplers, the trainer, the object store client and
the pool now do on their own when a part of the run fails, so you can tell
expected recovery from a real fault. Each item names its settings, the events
or metrics to look for and what to do. Where another section already explains
it in detail, it is linked. It covers every robustness change merged from
#352 to #406; section 5.11 describes the end-to-end rehearsal on the merged code. Pool events below are shown by their short name;
on the NC event tape they carry the `nc_elastic_` prefix.

#### Samplers (agentic)

| Behaviour | Settings (default) | Events / metrics | What to do |
|---|---|---|---|
| The sampler checks at start that the sandbox host's docker is reachable (`docker info` over the configured `DockerHost` / ssh) and exits with `SamplerSpecError` instead of claiming nothing forever (#401). | none | `SamplerSpecError` at sampler start | A lost docker key, a bad `known_hosts` or a wrong `docker_host` now fails the start with an actionable message: fix the host access, then `pool-retry`. |
| Records proxy: with Qwen3 `enable_thinking: false`, a multi-turn trial no longer fails its second turn with `RecordsMismatch` (the generation prompt's empty think block, which the history form omits, is accepted as a prefill); a real rewrite of an earlier turn still fails closed (#400). | none | `RecordsMismatch` only on a real rewrite | Small-model (Qwen3) agentic rehearsals get real rewards. Kimi K3.1 is not affected (its generation prompt is a prefix of the next rendering). |
| A `drain` weight switch is cut at min(`agentic.drain_timeout_s`, `reload_max_defer_s`). The release is downloaded in the background first (trials keep running), and every drain ends in a load of the release on disk. Trials that ignore the cancel are detached after `abandon_grace_s`. Details: section 3.4. | `reload_max_defer_s` (600; `null` waits indefinitely), `agentic.drain_timeout_s` (unset: the defer limit applies), `agentic.abandon_grace_s` (120) | `rl_agentic_drain` (`timed_out`, `abandoned_trials`, `detached_trials`, `trials_ended_s`, `detached_in_flight`, `waited_s`), `rl_agentic_waiting` (every 60 s while a call waits for a slot), `rl_sampler_release_not_loaded` (`download_failed`) | Keep `drain_timeout_s` <= `reload_max_defer_s`; `pool --dry-run` warns otherwise. Many `abandoned_trials` per drain: the agent tool calls are longer than the cut; raise the cut or release less often (section 3.5). |
| R3 routing rows of each model call are cut to the trial's training window (first prompt + `max_response_len` - 1 rows) as soon as the call returns, instead of holding the whole context's routing per live trial. The trained `routed_experts` are byte-identical. | none (follows `max_response_len`) | turn records carry `routed_experts_cut` | Nothing. A `TokenAlignmentError` about a short R3 window is a bug report, not a configuration problem. |
| Release downloads start at a fixed offset inside a spread window per sampler, and a range read the store refuses for its rate lowers the process's concurrent range reads (halved per refusal, one back per 8 successes, up to 6 tries per range). Details: section 5.7. | `YETO_RL_RELEASE_SPREAD_S` (30 on NC samplers, 5 elsewhere; `0` off), `YETO_RL_RELEASE_SPREAD_KEY` (NC: `<run prefix>/<shard>`, else host name) | sampler log `[s3-store] WARNING: ... the store refused a range read for its request rate` | Warnings at every release: lower `YETO_RL_STORE_FILE_CONCURRENCY`. Keep the spread well below the release interval. |
| The dispenser scan lists only unsettled slots (range listings) instead of the whole ledger; lease renewal is one write per held slot. Details: sections 5.7, 5.8. | `prompts.dispense.full_scan` / `YETO_RL_DISPENSE_FULL_SCAN=1` (off) | - | A `done/` record deleted by hand is seen only by a restarted sampler (section 5.8). |
| A failed store **read** never ends a sampler or disarms a fence: lease, handoff, ledger, `LATEST.json` and buffer reads are polls, read again at the next poll; the sampler keeps its weights and its tasks. A finished group whose hold cannot be checked waits instead of being written or dropped, and is given up (task released, not counted toward `max_redo`) after `write_defer_max_s`. Only a failed **write** ends the process (`INFRA_FAILED`, restarted by the pool). Two exceptions end the process at once: a configuration error of the store (403 / `AccessDenied`, `InvalidAccessKeyId`, `SignatureDoesNotMatch`, `NoSuchBucket`, missing credentials) is fatal and not retried, and one poll that keeps failing for longer than 1800 s (`max_read_outage_s`, fixed in code) ends the sampler with `StoreReadOutage`. Details: section 5.10. | `write_defer_max_s` (600), `incarnation_lease.max_unread_s` (60), `prompts.dispense.lease_s` (900) | `rl_sampler_store_read_failed` / `rl_sampler_store_read_ok`, `rl_dispense_scan_failed`, `rl_sampler_write_deferred` (`claim_unread`, `lease_unread`, `given_up`), `rl_sampler_release_not_loaded` (`latest_unread`) | Short outages: nothing. A sampler that exits on a 403 or `NoSuchBucket`: fix the store credentials or bucket name in the store env file; the pool restarts it. `StoreReadOutage`: the store has been unreadable for 30 minutes; fix the store access. |
| Store dispenser retire and lost claim answers: `store_dispense retire` writes the retired marker **before** it lists the claims to release, and a sampler checks its marker once after a claim pass that won anything; a claim taken while the retire ran is released at once. A claim whose create was applied but whose answer was lost (timeout) is tracked as unsure; each scan reads it back and releases it if it is this holder's, and the sampler does not answer a drain while a claim is unsure. | none | claim release reasons `retired`, `claim_answer_lost` | Nothing. A drained or retired node no longer leaves a claim behind until its lease expires. |
| An agentic group whose trials ran across an in-place weight load that failed after touching the engine (or that started while the loader was not ready) is dropped instead of being written with the old `behavior_version`. | none | groups dropped as `unchecked_weights_in_flight` | Occasional drops after a failed hot reload are expected; frequent drops mean the reload itself fails: read the sampler log. |
| At sampler close, the agent error of a detached trial that is cancelled again is retrieved and recorded instead of lost (#406). | none | the trial's error record | Nothing; the real error is there when you troubleshoot. |

#### Object store client

| Behaviour | Settings (default) | What it prevents |
|---|---|---|
| Explicit SDK retries: `standard` mode, 9 requests per call, 10 s connect and 60 s read timeout (section 5.7). | `YETO_RL_STORE_RETRY_MODE` / `AWS_RETRY_MODE`, `YETO_RL_STORE_MAX_ATTEMPTS` / `AWS_MAX_ATTEMPTS`, `YETO_RL_STORE_CONNECT_TIMEOUT_S`, `YETO_RL_STORE_READ_TIMEOUT_S` | An unreachable endpoint holding one call for minutes, longer than the dispenser lease renewal period. |
| A conditional create refused on the SDK's own retry (`PreconditionFailed` after a lost answer) reads the object back; equal bytes mean the create succeeded. | none | A group appended twice to a sampler shard, a merged record appended twice, a claim left orphaned until it expires and counts toward `max_redo`. |
| A whole-object read whose body is cut short is read again (up to 5 tries with jittered backoff). Error answers (403, a 5xx left after the SDK retries) still raise. | none | One reset connection ending the trainer (payload, index record, `LATEST.json`) or a sampler (lease, handoff, claim). |
| A multipart upload whose complete was applied but not answered is recognised from the part journal (size plus composite SHA-256 or multipart ETag), so a 100-200 GB release file is not uploaded twice. What cannot be proven is uploaded again. | none | Double uploads of release files. |
| Downloads write to a private partial file (named by host, process and call) and rename; a partial file left by a killed download is removed after an hour by the next download of that target. | none | Cut-short files at the final name; two downloads of one target corrupting each other. |

#### Trainer

| Behaviour | Settings (default) | Events / metrics | What to do |
|---|---|---|---|
| **Format change (#398).** Shard ownership fence: a fence is an ownership record in the shard itself, and a second sampler start on a shard leaves it resumable instead of fencing it for good. Mixed versions, the upgrade order (samplers and pool first, the trainer last) and why there is no way back to older code on the same store: sections 5.2 ("What a fence is") and 5.4. | none | `rl_shard_owner` (`adopted: true`) | Follow the upgrade order of section 5.4; to go back to older code, start a new run prefix. |
| A damaged payload (missing, cut short, other schema) is passed over and the round is filled from the groups after it; a failed read (throttle, 5xx, timeout) still raises and the next take starts with the groups it had selected. After decoding, the payload's own policy version is checked against `--rl-max-policy-age` again. A run of damaged payloads stops the trainer with `DamagedPayloadLimit` without consuming anything. | `YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN` (8 in a row, or 8 once more than half of a take; `0`: no limit) | `rl_buffer_damaged_payload_skipped`, `sample_buffer/damaged_payload_groups`, `sample_buffer/stale_payload_groups`, `passed_over` in `rl_buffer_exhausted` | See section 8, "The trainer stopped with `DamagedPayloadLimit`". |
| The shard merge into `sample-buffer/index/` is idempotent by `(shard, shard_pos)`: two mergers (an old trainer not dead yet and its replacement), a lost create answer or a half-failed merge batch never train a group twice. A restarted merger resumes each shard at its highest merged position + 1. | none | `duplicate_merged_records` (repeats already in an index written before this change: passed over, one warning), `unmerged_shard_gaps` (records an earlier merger skipped: reported, left unmerged) | A non-zero `unmerged_shard_gaps` means some groups of a bucket written by older code are not trained; nothing to fix in the running run. |
| Every round cut is written behind a first-cut mark until the store holds a committed `LATEST` (one restored at start, or a cut of this process that committed), not only the store's very first background cut: an inline cut (background commits off, the final cut) and a later cut while an earlier store sync failed or still runs are marked too. The mark is uploaded when it is written. A trainer that dies before the first `LATEST` is committed restarts at rollout 0 instead of failing with `StoreIntegrityError: ... holds round cuts but no LATEST`. The local cut pointer is written after its staged copy, so a failed staging never points at dropped files. | none | - | Nothing. A bucket written by older code that already fails this way still needs the unmarked `round-cuts/<cut_id>/` objects removed once. |
| A trainer that resumes above the store's releases (no cut, or a cut older than the newest release) starts at the newest release vK. Main change: a buffer group whose policy version is below K and is named by no release record any more (the store keeps only the newest releases) is consumed, counted and not trained, instead of stopping every start with `PolicyIdentityError`. An unknown version at K or above is still an identity error. In addition the earlier ledger is set aside, the buffer is read from the position recorded with vK, and a due cut whose ledger lacks the previous round is skipped. Details: section 5.3. | `--rl-max-policy-age`, `--rl-ledger-set-aside-before`, `--rl-cut-every`, `--rl-cut-auto-mtbf-s` | `rl_buffer_unidentified_policy_gate` (at start), `rl_buffer_unidentified_policy_skipped`, metric `sample_buffer/unidentified_policy_groups`, round metadata `buffer_unidentified_groups` / `buffer_unidentified_versions` / `buffer_unidentified_positions`; also `rl_round_cut_behind_releases`, `rl_round_cut_skipped`, `rl_ledger_set_aside`, `rl_buffer_start_position_refused` (falls back to position 0) | Nothing; the skipped groups are not trained (their tasks are settled). A `PolicyIdentityError` for a version >= K is a real fault: report it. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is accepted on a full-parameter trainer island (`--rl-sample-role trainer`, no rollout engine); it is still refused on a colocated island with engines and on any island with `--rl-offload-train`, before any GPU starts. Measured: peak reserved memory 251 to 174 GiB on a 16-layer cut model, bit-identical loss, about 8 s more per step. | trainer env | launch check message | Set it for 1M or other tight layouts (section 3.5, section 10.4). |
| **The incarnation gate fails closed and pauses training.** A shard's lease that cannot be read (timeout, 5xx, reset connection, cut-short body) is no verdict: the group is neither trained nor counted stale, and because the buffer is read in strict order every later group of every shard waits too. The round holds only the groups before it, and its committed watermark is that group's position. Nothing ends the wait but a readable lease or stopping the trainer. While a shard's reads fail, an older cached lease read does not answer. Details: section 5.1, "The gate fails closed, and pauses training". | none | trainer log (at most once a minute) `incarnation gate: training paused: lease of shard <X> unreadable since <T>`; metric `sample_buffer/lease_deferred_groups` | Fix the store access for that lease (`sample-buffer/leases/<shard>.json`). This is not starvation: the starvation monitor names it `trainer side: ... fail-closed pause` and does not stop the trainer for it. |
| **The gate judges a round again right before it is trained.** A fence written while a take waited for later groups keeps an earlier selected group out: each chosen group's shard lease is read again (at most once per shard per second); a group fenced since its selection is dropped (counted stale, its task freed, and a group passed over for that task is judged again); from the first unreadable lease on, the groups wait as above. The same recheck runs in the verl buffer feed. | none | counted in the stale and `lease_deferred_groups` metrics | Nothing. |
| The trainer trains only the first accepted group of each store-dispenser task; later groups of the same task (a sampler that wrote after its lease expired, a retire before settlement) are consumed and counted, not trained. | `YETO_SAMPLE_BUFFER_ONE_GROUP_PER_TASK` (on; `0` off) | `rl_buffer_duplicate_task_skipped`, `sample_buffer/duplicate_task_groups`, `duplicate_task` in `passed_over` | Occasional events: nothing (section 8). |
| Staged release writer across restarts: a restarted trainer lists the release directories in the store (not only its local stage, which no longer holds earlier uploads), so it prunes the releases of earlier processes to the newest `keep` and removes a partial upload of a killed attempt before writing under the same `r<seq>-v<version>` name. `LATEST` is read again after that listing, so a release another writer published meanwhile is not taken for a partial upload. A partial directory at or above the new seq is pruned first and never takes a kept place. | release `keep` | - | Nothing. Before this change a restarted trainer left old releases in the bucket and could serve a mix of two uploads under one name. |
| **Trainer scheduler restarts stop on a repeated crash.** A trained round resets the failure count only when it reaches a version no earlier incarnation of this loop trained. A trainer that crashes deterministically after the same cut (it restores the cut, retrains the same rounds, crashes again) now gives up after `--rl-trainer-max-failures` incarnations instead of restarting forever. | `--rl-trainer-max-failures` (3) | trainer journal `gave_up`, `TrainerLoopGaveUp` | Read the trainer log of the last incarnation; fix the cause, then `yeto run up` resumes from the newest cut. |
| **A fatal store error at the incarnation gate ends the trainer.** A lease read refused for configuration reasons (403 / `AccessDenied`, `InvalidAccessKeyId`, `SignatureDoesNotMatch`, `NoSuchBucket`) is no longer a fail-closed pause: the trainer exits through its failure path (failure count, `TrainerLoopGaveUp`). `ExpiredToken`, 5xx and timeouts remain pauses. | none | trainer log `incarnation gate: fatal store error: lease of shard <s> refused by the store (<code>)`; `yeto run up` names it "the trainer exits on it: not starvation, not a gate pause" | Fix the store credentials or bucket in the trainer's store env, then `yeto run up`. |
| A failed staged release upload (export, full disk, throttling past retries, `LATEST` put) stops the trainer at its very next publish, whatever `--rl-release-every` is (before, a cadence-skipped publish could hide it for N-1 rounds while the samplers stayed on the old policy). | `--rl-release-every` | trainer failure with the upload error | Read the trainer log, fix the cause (disk, store), `yeto run up` resumes from the newest cut. |
| `--rl-round-cuts {auto,off}`; the K3.1 examples pass `off` (one full-parameter cut is about 5 TB). With `off` no round cut is written or read, and a relaunch continues from the newest policy release: `yeto run up` and every trainer node add `--rl-init-from-release latest` on their own. A trainer that finds releases but started from the base stops with `RoundCutsOffRestartError`. `off` is refused with the cut flags, a convertible node and `--rl-backend verl` (#405). | `--rl-round-cuts` (`auto`) | `rl_round_cuts_off`, `adding --rl-init-from-release latest`, `no policy release yet`, `RoundCutsOffRestartError` | See section 5.3. Cost of the resume: optimizer state, LR schedule position and RNG restart; at most one release interval is trained again. |

#### Pool and National Compute

| Behaviour | Settings (default) | Events | What to do |
|---|---|---|---|
| A sampler exit 0 ends the pool ("the data ended") only when the run store says so. A handoff record retires the member and a new shard continues; a fence or a `FENCED` leave (also one the pool wrote before it restarted, or an operator's `store_dispense retire`) restarts the sampler in place, counted toward `max_sampler_restarts`; an unreadable store decides nothing that tick. | `max_sampler_restarts` | `sampler_left_fenced`, `sampler_restart` | Nothing. |
| A failed fence is retried every tick (member state `fencing`, kept across a pool restart); the node is kept and no successor starts on that shard until it succeeds. A node that must go (ticket ended, lease end, pool stop) is released at once and the fence stays owed. | `elastic.fence_order_ahead_s` (300), `elastic.fence_blocked_hold_s` (1800), `elastic.fence_max_hold_s` (7200; `null`: keep until fenced), `elastic.fence_alert_failures` (5), `elastic.fence_alert_s` (600) | `fence_failed` (first and every tenth failure), `fence_stuck` (once) | `fence_stuck`: the store refuses the lease write; check the store credentials and its error rate. |
| Unreachable grace counts failed probes only; a failed `docker inspect` reads `unknown`, not absent (a ready container must read absent twice before a restart); a `pool-scale` request written during a tick is kept; a stale handoff of an earlier pool no longer stops a new sampler on that shard. | - | - | Nothing. |
| The pool state file now has `"format": 2`. An older pool cannot read it. | - | - | Before rolling the code back run `python -m yeto.launch.nc pool-state-downgrade --config pool.json` (refuses while the pool runs or a fence is owed; see [NC_POOL.md](NC_POOL.md)). |
| NC status answers are trusted only when they report the ticket field; any other answer is a control-plane blip and decides nothing. An answer that does not show the expected ticket is read a second time before a lease is called ended. | - | - | Nothing; a release that read an untrusted answer is retried. |
| The ticket ledger and lease file are written under a cross-process lock through a private temp file. A ledger that cannot be read is an error (`LedgerError`) and is left as it is, instead of being treated as empty. | - | - | `LedgerError`: restore the file from its last good copy or fix it by hand; do not delete it (every ticket in it would lose its ownership proof). |
| Ticket ownership: the client changes only a ticket whose ledger row was written before the order and holds the id from that order's own answer, or one an operator adopted. A delete re-reads the status first, is refused within the guard of the lease end or preemption, and is journaled until the ticket reads ended. | `YETO_NC_DELETE_GUARD_S` (120) | `order_unrecorded` (with the `adopt` command), `legacy_ledger_rows`, `foreign_ticket_released`, `delete_unconfirmed`, `halt_cleared` | `order_unrecorded`: check the account in the console; if the node is this run's, `python -m yeto.launch.nc adopt --account <id> --ticket <ticket> --yes`, otherwise leave it. |
| Halt: an account whose delete released, or may have released, another ticket is halted until `clear-halt --account <id> --yes` (a `delete_unconfirmed` halt lifts by itself when that exact ticket reads ended). The pool skips a halted account instead of counting it free, a guard refusal (nothing sent) is not a breaker failure (the slot is held out 10 min and the next slot is ordered in the same tick), and a member on a halted account is kept, not re-drained. A failed ledger read keeps the known halt. | - | `slot_halted`, `slot_halt_cleared`, `slots_below_target` (critical), `release_blocked`, `release_hit_foreign_ticket` (critical); `pool-status` and `yeto run status` list them under `accounts_attention` | Section 2 (Accounts) and [NC_POOL.md, "A halted account"](NC_POOL.md#a-halted-account-clear-halt). Provide spare accounts so a halt does not drop the pool below target. |
| Teardown of a Sky trainer cluster is confirmed only by the cloud. A cloud probe that fails after the cloud once reported a live instance, or instances still live after every retry, leave the cluster **not confirmed** (before, both counted as down). A launch that is still running after its cancel also leaves the release unconfirmed, because the cluster may come up after the down. | - | launcher log `NOT confirmed down`, `teardown NOT confirmed (the cluster may still come up: ...)`; `down` reports `NOT verified` | Check the cloud console and rerun `yeto down <run>` (section 8). |
| Trainer starvation: with no new buffer object for `starvation.alert_min` the monitor logs a critical alert; at `stop_min`, and only while no sampler has been healthy for `unhealthy_min`, it stops the trainer through the STOP flag and `yeto down`. The sampler pools are kept. Details: section 7, "Trainer starvation". | `starvation`: `alert_min` (30), `stop_min` (120), `unhealthy_min` (10), `check_s` (300), `stop_grace_min` (20) | `CRITICAL trainer_starvation_alert`, `CRITICAL trainer_starvation_stop` | Find out why no sampler writes (pool status, halted accounts); once a sampler is healthy, `yeto run up --config <run.json>` resumes from the latest cut. |
| Model pull on an NC node: a refused listing (bad credentials or endpoint) or an empty one (wrong `src_key`) fails the pull (`YETO_NC_PULL fail`, exit 3) instead of starting the sampler on an empty model directory, and any pull container exit without `done` ends the wait at once instead of polling up to an hour on a billed node. | - | pull failure through the pool's transient-retry / retire path | Check `model_pull.src_key` and the store env file. |
| Lease end without a ready replacement: when a lease-bounded node reaches its end margin and its make-before-break replacement is not ready (order refused, balance below the order floor, replacement stuck), the pool raises an alert and fences the shard before the drain (unless a successor sampler already started). A balance below the order floor is an event, not only a log line. | `end_margin_s` | `lease_end_without_ready_replacement` (alert, once per member), `order_refused` | The slot samples nothing until a node is back: top up the account or add accounts. |

#### `yeto run up`, preflight and cost

| Behaviour | Settings (default) | Events / messages | What to do |
|---|---|---|---|
| `yeto sandbox prepare` fails closed when it cannot health check an adopted sidecar, when the sidecar is published on a stale docker bridge address, or when it cannot read the bridge address; the error names both addresses and the container is left alone (#399). | `hosts[].ssh` | prepare error naming both addresses | A host whose docker bridge moved (daemon restart) no longer reports "sidecars N/N" while every trial ends `RewardFileNotFoundError`: recreate the sidecars, then rerun prepare (section 8). |
| Trainer/sampler consistency check before anything is ordered: group size (`sampling.n` vs `--n-samples-per-prompt`), response caps, prompt + response vs `--seq-len`, R3 routed experts, expert selection hashes, store paths, the release directory, and the sampler tunnel port range vs the trainer's buffer shard count. The stage-1 example trainer launcher uses `SEQ_LEN` 12288. | - | preflight refusal naming both values | Make both sides agree, then rerun. |
| Price preflight: with `budget.usd` set, a USD-priced pool (Modal, Daytona, Sky) whose price is missing, not a number or `<= 0` is refused (section 3.1). Fill a real price; for Modal, the GPU price per second x 3600 x `gpus` plus CPU / memory. `budget.usd: null` means no cap. | `usd_per_h`, `price_usd_h`, `budget.usd` | preflight refusal | Fill the price and rerun. |
| Modal / Daytona / Sky sampler pools are counted at their own price (`usd_per_h`, else the provider's `price_usd_h`), not at `budget.nc_usd_per_h`, by the budget breaker, the forced pool stop and `run scale`; the cost guard counts priced Modal pool calls before Modal's billing report shows the app. | - | - | Nothing, once the prices are real. |
| Safe re-entry: one `up` per run (`~/.yeto/rl-up/<run>/up.lock`; a second `up` exits), a leftover controller `STOP` flag is removed and logged before the pool starts, and the trainer is stopped by its `--cluster-prefix` (not the run name), so a breach or starvation stop does not miss it. | - | message from the second `up` | Rerunning `yeto run up --config run.json` on a stopped, crashed or running run is safe. |
| Data end: once every sampler pool is finished and holds no live member, the monitor writes the end record of every shard the trainer reads that has not ended (fenced, lost or never used shards), so the trainer finishes on the data end; starvation is not checked after that. | - | `state["data_ended"]` | Nothing. |
| `yeto run down` takes the run's `up.lock` first: a running `run up` of the same run gets SIGTERM and up to `YETO_RUN_DOWN_GRACE_S` seconds to exit before the teardown starts; a `down` that meets another `down` only waits for it (#402). | `YETO_RUN_DOWN_GRACE_S` (60) | the `down` message that it interrupts the running `up` | Nothing: `down` and a running `up` no longer race. To keep training after a `down`, run `yeto run up` again. |
| The run's `state.json` is read, merged and written under a lock and replaced atomically, so `yeto run scale` and `yeto run up` writing at the same time no longer overwrite each other (#403). | none | - | Nothing. |
| Local disk check before a release lands on a node: a sampler hot load and a trainer `--rl-init-from-release` (download and delta rebuild) compare the free space with the release record's sizes times a headroom and stop before the first byte instead of failing with ENOSPC part way (#458). A trainer node drops surplus mirror copies before a delta rebuild, so the peak is one release-sized copy lower (#459). | `YETO_RL_DISK_HEADROOM` (1.05; values below 1, `inf` or `nan` fall back to the default) | `InsufficientDiskSpace: <what>: needs about <N> GB on <path> (... headroom), only <M> GB free` | Free the disk or point the cache at a larger one, then retry. |
| A trainer node that cannot prepare its init release ends with `[yeto-init-release] FAILED: <type>: <message>`, and `yeto run up` quotes that line (credentials, presigned URL queries and URL userinfo redacted, at most 500 characters) (#460, #461-#463). | none | `[yeto-init-release] FAILED: ...` | Fix the cause named in the line. |
| The `run up` monitor survives a full or read-only control-host disk: a failed `state.json` write is logged once and the state stays in memory, so the budget breaker and the 1.2x forced pool stop keep working (#469). The pool loop emits `state_save_failing` once per streak of failed state saves (and `state_save_recovered`), keeps running, and fails closed before a step that depends on a fence's saved success (#467, #468). | none | `CRITICAL state_write_failed`, `state_save_failing`, `state_save_recovered` | Free the control-host disk. |
| A commanded trainer island (head-scheduled trainer) waits at most `YETO_TRAINER_RELEASE_DRAIN_S` for its background policy release upload before it reports STOPPED, so an upload that hangs no longer keeps the island billing; past the bound STOPPED goes out with the error (#465). | `YETO_TRAINER_RELEASE_DRAIN_S` (1800) | STOPPED with the upload error | Read the island log for the upload error. |

#### CI

A newer push to a pull request cancels the CI run of its older head; runs on
`main` are never cancelled. The store fault simulation (real sampler, trainer
and pool store code on a fault-injecting S3) runs green on `main` in its full
mode, and flaky throttling and timing tests were isolated. Since #398 the
gate mutations are skipped on the fence-zombie preset; a unit test now covers
the gate's seal boundary instead and fails under both mutations (#404).

### 5.10 Object store outages

Every island reads and writes the run's bucket. A sampler treats its reads as
polls and its writes as the work:

- **A read that fails is no verdict.** The incarnation lease, the handoff
  record, the dispenser ledger, `LATEST.json` and the sampler's own buffer
  index are read again at the next poll. The sampler keeps the weights it
  serves and the tasks it holds; nothing is given back to the dispenser. The
  tape marks the outage: `rl_sampler_store_read_failed` /
  `rl_sampler_store_read_ok` (`what`: `lease`, `handoff`, `backlog`, `buffer`),
  `rl_dispense_scan_failed` (`state`: `failed`, `recovered`) and
  `rl_sampler_release_not_loaded` with `reason: latest_unread`.
- **A group that is ready during the outage waits.** Before a group is written
  the sampler reads whether it still holds the task and the shard. While the
  store does not answer, the group is neither written nor dropped
  (`rl_sampler_write_deferred`, `reason: claim_unread` or `lease_unread`) and
  is written once the store answers. Lease renewals are writes, so the claims
  of the tasks in flight stay alive through an outage of the reads. The wait
  ends after `write_defer_max_s` without an answer, or when the sampler is
  asked to stop: the group is then given up (`given_up: true`), its task is
  released without counting toward `max_redo`, and the sampler goes on.
- **Only a failed write ends a round.** A group write that fails ends the
  sampler process as `INFRA_FAILED`; the pool restarts it and the restart
  settles what was written.

The invariant, checked by `yeto.launch.checks.store_outage_check`
(`read_outage_problems`): while every read (GET, LIST, HEAD) of the store fails
for up to **10 minutes** (`max_outage_s`, the check's default; the sampler's
`write_defer_max_s`) and writes still work, no sampler exits and no task in
flight is released; afterwards every sampler goes on. The settings behind it:

| Setting | Default | Meaning |
|---|---|---|
| `write_defer_max_s` (sampler spec) | 600 | The longest a sampler holds finished groups back while the store does not answer whether it still holds their tasks. Past it the waiting groups are given up and their tasks released. Raise it together with `max_outage_s` to ride out longer read outages. |
| `incarnation_lease.max_unread_s` (sampler spec) | 60 | A sampler that could not read its incarnation lease for this long holds its group writes back until it reads it again (it cannot know whether the pool fenced it). `null`: no bound. Inside this window a fenced sampler whose reads fail can still write a group; the trainer then drops it by the fence's time stamp. |
| `prompts.dispense.lease_s` (sampler spec) | 900 | An outage of reads **and** writes longer than this lets the claims in flight run out: their tasks are given out again and the groups of the first holder are not written. |
| `prompts.dispense.scan_s` / `max_scan_s` | 30 / 120 | A failed ledger scan is repeated after this backoff. |

A `LATEST.json` that is present but is not a release record is not an outage:
the sampler keeps its weights for 120 polls in a row (10 minutes at `poll_s`
5), then stops, because the trainer rewrites the pointer with every release.

On the trainer, a sampler's incarnation lease that cannot be read keeps the
lease read last; a lease never read so far is read again every 2 s for up to
10 minutes before any group of that sampler is trained or dropped.


### 5.11 End-to-end rehearsal on the merged code (2026-11-13)

A full `yeto run up` with Qwen3-0.6B on the code of `main` at that time
(#352 to #393): a Modal 1xH100 trainer, one NC MI355X agentic sampler island,
cybergym tasks on one sandbox host, a shared R2 store.

| Check | Result |
|---|---|
| One command | `yeto run up --config run.json` ran preflight, sandbox prepare, pool, first group, trainer, dashboard and monitor |
| Training | 10 of 10 steps, trainer `SUCCEEDED`; policy age never above `--rl-max-policy-age 2` |
| Hot reload | 4 in-place loads after drain plus one `start_on_latest` after a restart; the sampler followed v0 to v10; no forced loads, rebuild failures or lineage resets |
| Fault injection | `docker restart` of the sampler container: a new session started on the newest release after about 27 s, claims recovered, training never paused |
| Alerts | no false alert: no pause, starvation, `unchecked_weights_in_flight`, breaker or unconfirmed teardown |
| Teardown | `release_nc_on_finish` released the node; `down: verified` |
| Cost | about $23 (budget $60) |

The #388 consistency check caught a real mismatch in the dry run (8 tunnel
ports for 4 buffer shards). The two blocking problems were in the
environment, not in `main`: a docker key missing on the sandbox host, and a
docker bridge address that had moved under existing sidecars (both in
section 8; since #399 and #401 they fail at once). The #395 price check was
not yet on `main` during this run.

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
| Wall clock | time since pool start > `budget.wall_h` (default 3 h; `null`: never) | Stop the trainer. |
| Trainer exit | launcher finished `FAILED` or non-zero | Stop the trainer path cleanly. |
| Trainer finished | launcher finished `SUCCEEDED` (exit code 0) | Banner: the sampler pools (every provider) still bill; with `budget.release_nc_on_finish` the full `down`. |
| Silence backstop | trainer log and `progress_paths` unchanged for `max(4 x --rl-stall-timeout, silence_min)` | Stop the trainer. |
| First group | no buffer object within `first_group.timeout_s` | Stop before the trainer starts. |
| Trainer starvation | no new buffer object for `starvation.alert_min` / `stop_min` (see below) | Alert; at `stop_min`, with no healthy sampler, stop the trainer with the STOP flag. |
| Sandbox containers | count > `sandbox.max_run_containers` (or `max_containers_per_sampler` x billed nodes, if larger) | Warning only. |
| Scale up | spend + one node hour per new node > `budget.usd`, or a breaker fired | `yeto run scale` refuses; nothing changes. |

**Trainer first, NC kept.** By design, a breach stops only the trainer
(`yeto down <run>`). The NC pool, its nodes and the sandboxes stay up, so a
resume does not repeat node setup and model pull. The monitor prints which
nodes still bill and the release command.

**Trainer starvation.** When every sampler is gone (all died, the pool cannot
replace them, every account is halted) the trainer keeps waiting for groups in
its buffer reader (with `--rl-trainer-scheduler`, the head keeps waiting for the
RUN ack) while its GPUs bill. The monitor watches two signals it already has:
new object keys under the store's `sample-buffer/shards/` (listed every
`check_s`), and the pool state files (members in `ready`, the pool loop running).

```json
"starvation": {"alert_min": 30, "stop_min": 120, "unhealthy_min": 10,
               "check_s": 300, "stop_grace_min": 20, "trainer_wait_max_min": 120}
```

| Key | Default | Meaning |
|---|---|---|
| `alert_min` | 30 | Minutes without a new buffer object before `CRITICAL trainer_starvation_alert` is logged (once per starved stretch; the line names the duration, the healthy / live / paused sampler counts and the pool notes). `0`: no alert. |
| `stop_min` | 120 | Minutes without a new buffer object before the trainer is stopped, **only** while no sampler is healthy (no ready, unpaused member in a pool whose loop runs) for at least `unhealthy_min`. `0`: never stop. Must be >= `alert_min`. |
| `unhealthy_min` | 10 | How long the pools must report no healthy sampler before a stop. A short gap (a lease-end handoff) does not count. |
| `check_s` | 300 | Seconds between two incremental listings of the buffer shards. |
| `stop_grace_min` | 20 | After the STOP flag, how long the monitor waits for the trainer launcher to exit before `yeto down`. |
| `trainer_wait_max_min` | 120 | While every healthy sampler waits for the trainer (its newest tape line is an `rl_sampler_waiting` heartbeat: backlog at `max_backlog_groups`, e.g. a restarted trainer loading its weights), the starvation clock is held. The hold lasts this long at most from the start of the wait (only a new group ends the wait), so a trainer hung without exiting, or samplers held by a failed post-load check, still alert `alert_min` later. `0`: no hold. |

What never stops the trainer: a healthy sampler (long agentic trials can go
an hour without a group), a member paused with `yeto run pause` (in any state),
pool state that cannot be read (of any pool), and a failed buffer listing; these
alert only. A replacement under way (unpaused `provisioning` / `starting`
members within their pool's `elastic.start_timeout_s`) also holds the stop, for
at most that timeout from the first replacement of the stretch (the alert says
`recovering: N member(s) starting`). A stop is decided on a fresh listing of the
buffer. The listing is incremental: one page after the last key seen per known
shard each `check_s`, and every 6th check the `shard-names/` prefix for new
shards; any failed request counts as no listing. The incarnation gate's
fail-closed pause (`incarnation gate: training paused`, a lease that cannot be
read) is not starvation: the groups are in the buffer and the trainer waits for
the lease. When the trainer log shows that pause, the alert names it
(`trainer side: ... fail-closed pause`).

The stop is the budget stop's path: `CRITICAL trainer_starvation_stop`, the
run's STOP flag (`yeto stop-run <--cluster-prefix>`; with the trainer scheduler
the trainer drains with a cut, a fixed trainer island keeps its latest round
cut), then `yeto down <run>` once the launcher exited (or after
`stop_grace_min`). The sampler pools are kept (with `controller: in_process`,
`rollout.drain_on_exit` decides, as after every breach). Once a sampler is healthy again,
`yeto run up --config <run.json>` starts the trainer again and it resumes from
its latest cut (rounds after that cut are trained again; section 5.3). The
pool's own state is not touched. `run up` refuses a bad `starvation` section in
preflight.

**Forced NC release at 1.2x.** After a budget or wall-clock breach, the
monitor stays up and prints the billing banner on every poll. Once the spend
passes `1.2 x budget.usd`, or the wall time passes `1.2 x wall_h`, it runs
`pool-stop` itself. It exits when no node bills.

The forced stop is recorded in the run state (`nc_forced_stop`). A later
`yeto run up` of the same run refuses to start until you confirm it with
`yeto run up --config <run.json> --allow-after-forced-stop` (raise `budget.usd` /
`budget.wall_h` first if the run should go on). The confirmation is recorded,
so later restarts go on until the breaker forces another stop.

**Opt-in full stop.** Set `"release_nc_on_breach": true` in `budget` to make a
budget or wall-clock breach perform the full `yeto run down` immediately.

The pool also has its own guards (`elastic.max_nodes`, `max_orders`,
`max_run_spend_ucr`, order breaker); see [NC_POOL.md](NC_POOL.md). Log-prob
and health breakers inside the run are described in [BREAKER.md](BREAKER.md).
Run `yeto breaker` on a K3.1 run with `--sampler-tapes` (the MI355X sampler
tapes) or `--sampler-hardware amd`: without either the sampler hardware is
unknown and the breaker fails closed to the `nvidia` limits (act 0.052), so the
healthy MI355X mismatch band (0.076-0.099) trips `logprob_diff`.
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
*What you see.* The trainer's round fails with `DamagedPayloadLimit: N damaged
payloads in this take (...)`, and it fails again at the same buffer position
after every restart or resume. The limit is 8 damaged payloads in a row, or 8
once they are more than half of the payloads the round decoded (default;
`YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN`). Nothing was consumed: the watermark has
not moved past these groups.

*What it means.* Single damaged or deleted payloads are passed over and
counted (`rl_buffer_damaged_payload_skipped`, above). This many together is a
systematic cause: samplers of another code version writing another payload
schema, a lifecycle or cleanup rule deleting `sample-buffer/payloads/`, or a
trainer image that cannot decode what the samplers write.

*How to look.* The error text names the last buffer position, its
`sample_id` and the reason (`payload is missing`, `payload schema ... != ...`,
`payload does not decode: ...`). Earlier rounds' events
(`rl_buffer_damaged_payload_skipped`) and the trainer log name each group with
its `position`, `sample_id`, `payload_ref` and `reason`. Look at those objects
in the bucket under `sample-buffer/<payload_ref>`: whether they exist, their
size, and the `schema` field of their JSON. A `sample_id` starts with the
sampler's id and incarnation, which tells you which samplers wrote them.

*What to do.* Fix the cause (bring the samplers and the trainer to the same
code pack, section 5.4; remove the rule that deletes payloads) and resume. If
the groups are lost for good and you want the run to go on without them, set
`YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN` in the trainer's environment to a number
above the damaged run, or to `0` for no limit, and resume. Passing them over
means these tasks are not trained: they are already settled in the dispenser
and are not generated again in this run. Set the variable back afterwards, so
the next systematic fault stops the trainer again.

**`rl_buffer_exhausted` carries `passed_over`.**
The last round found no trainable group; `passed_over` counts what it
consumed without training (`stale`, `stale_payload`, `stale_incarnation`,
`quarantined`, `dropped_small`, `nonfinite_reward`, `damaged_payload`,
`duplicate_task`). The
same counts end the error text of a round that waited `wait_s` without a new
group. `stale_payload` are groups whose index record passed the age gate but
whose payload holds an older policy version: the payload decides.

**Quick reference: store, pool, trainer and drain faults.**

| Symptom | Likely cause | What to do |
|---|---|---|
| Sampler idle for minutes at each release; `rl_agentic_drain` shows large `waited_s` | Drain cut too long, or group writes behind | Lower `reload_max_defer_s` / `agentic.drain_timeout_s` (section 3.4); check `trials_ended_s` vs `waited_s` |
| `rl_agentic_drain` with many `abandoned_trials` / `detached_trials`, fewer trials starting afterwards | Trials blocked in long sandbox tool calls ignore the cancel; detached trials keep their slot until they end | Expected; if constant, raise the cut or release less often (section 3.5) |
| `rl_sampler_release_not_loaded` `download_failed` | Store refused the download repeatedly | Check `[s3-store] WARNING ... request rate`; lower `YETO_RL_STORE_FILE_CONCURRENCY`, keep `YETO_RL_RELEASE_SPREAD_S` on |
| Calls fail after about 30-90 s with 503 under load | Store request-rate limit after all SDK retries | Section 5.7: raise `YETO_RL_STORE_MAX_ATTEMPTS` modestly, keep one call below `lease_s / 3` |
| Trainer start fails with `StoreIntegrityError: ... holds round cuts but no LATEST` | A first cut written by older code died while uploading | Remove the unmarked `round-cuts/<cut_id>/` objects once; current code drops such a cut by itself |
| Trainer log warns `duplicate_merged_records` or `unmerged_shard_gaps` | Index written before the idempotent merge | Nothing to do; gaps are groups not trained |
| `fence_stuck` alert | Lease writes to the store fail | Check store credentials and error rate; the node is kept until the fence succeeds or `fence_max_hold_s` |
| `slot_halted` / `slots_below_target` | A delete may have hit a ticket that is not this run's | Check the account in the console, then `clear-halt --account <id> --yes` (section 2) |
| `order_unrecorded` | An order's answer was lost, or a node on the account was not ordered by this tool | Adopt only if it is this run's node; otherwise leave it alone |
| `LedgerError` from an NC command | The ticket ledger file is damaged | Restore or repair it by hand; never delete it |
| Pool refuses to start, names `pool-state-downgrade` | State file of a newer format than this code | Use the newer code, or downgrade the file with `pool-state-downgrade` before rolling back |
| Trainer log `incarnation gate: training paused: lease of shard <X> unreadable since <T>`; `sample_buffer/lease_deferred_groups` grows; no rounds | The trainer cannot read that shard's lease and fails closed; every later group waits (section 5.1) | Fix the store access for `sample-buffer/leases/<shard>.json`; training resumes by itself once it reads. Not starvation |
| `CRITICAL trainer_starvation_alert` | No new buffer object for `starvation.alert_min` minutes: samplers dead, pool not replacing them, accounts halted | Check `pool-status` and halted accounts; see section 7, "Trainer starvation" |
| `CRITICAL trainer_starvation_stop`, trainer stopped | No new buffer object for `stop_min` and no healthy sampler for `unhealthy_min` | Bring a sampler back, then `yeto run up --config <run.json>`; it resumes from the latest cut |
| Sampler exits at once on a 403 / `AccessDenied` / `NoSuchBucket` | Store configuration error: fatal, not an outage | Fix the credentials or bucket in the store env file |
| Sampler exits with `StoreReadOutage` | One store poll failed for more than 1800 s | Fix the store access; the pool restarts the sampler |
| `rl_sampler_write_deferred` with `given_up: true` | The store did not answer whether the sampler still holds the task for `write_defer_max_s` | Expected after a long read outage; the task is redone without counting toward `max_redo` |
| Trainer journal `gave_up` / `TrainerLoopGaveUp` | The trainer crashed `--rl-trainer-max-failures` times in a row without reaching a new version | Read the last incarnation's log and fix the cause before `yeto run up` |
| Launcher log `NOT confirmed down` / `teardown NOT confirmed` | A cloud probe failed after instances were seen, instances were still live after the retries, or a launch outlived its cancel | Check the cloud console and rerun `yeto down <run>` |
| Groups dropped as `unchecked_weights_in_flight` | A hot weight load failed after touching the engine while trials ran | Expected after a failed reload; frequent drops mean the reload itself fails: read the sampler log |
| `run up` preflight refuses with a trainer/sampler mismatch | Group size, response cap, `--seq-len`, R3, expert selection hash, store path or release dir differ between trainer and sampler | Make both sides agree as the message says, then rerun |
| `run up` preflight refuses a pool price | `budget.usd` is set and a USD-priced pool (Modal, Daytona, Sky) has no real price | Fill `usd_per_h` / `price_usd_h` with your rate (section 3.1), or set `budget.usd: null` for no cap |
| Every trial ends `RewardFileNotFoundError`, although prepare reported all sidecars up | The sandbox host's docker bridge address changed (daemon restart, e.g. the default bridge to another subnet); existing task sidecars are bound to the old address and published nowhere, and tasks hardcode the old URL | Check `docker network inspect bridge` on the host; recreate the sidecars (or give the run its own sidecar names), use `{{yeto.sidecar.task-server.url}}` in tasks instead of a fixed address, set `hosts[].ssh` so prepare can health check, rerun `yeto sandbox prepare`. Since #399 prepare refuses such a sidecar |
| Agentic setup fails `the sandbox host refused the docker key`; pool stops in CONFIGURATION ERROR | The run's docker key is no longer in the sandbox host's `authorized_keys` (removed or overwritten) | Add the key back on the host, then `pool-retry` (section 2, Sandbox host); the node is reused. Since #401 the sampler start fails on an unreachable docker as well |
| Multi-turn trials end on turn 2 with `RecordsMismatch: the rendering of the earlier turns changed (template rewrites history)` | Qwen3 with `enable_thinking: false`: the template drops the empty think block from history. Not K3.1 | Fixed by #400 (update the code); a K3.1 run or a real history rewrite showing it is a bug report |
| `lease_end_without_ready_replacement` alert | A node reached its lease end and its replacement is not ready (order refused, low balance) | Top up or add accounts; the shard was fenced and continues when a node is back |
| A second `yeto run up` of the same run exits at once | Another `up` of this run still holds `up.lock` | Use the running one, or stop it first |
| The trainer stops at start with `RoundCutsOffRestartError` | `--rl-round-cuts off`, the store holds policy releases, but the trainer loaded the base checkpoint (no `--rl-init-from-release`, e.g. an island command started by hand) | Relaunch through `yeto run up` / `yeto launch`, or add `--rl-init-from-release latest`; to start over, use a new run prefix / store (section 5.3) |
| A running `yeto run up` ends on SIGTERM | A `yeto run down` of the same run interrupted it (#402) | Expected; run `yeto run up` again after the `down` if training should go on |
| `CRITICAL: the exit drain of the sampler pools FAILED` / `may STILL BE BILLING` | The in-process exit drain raised (provider client error, network) | Run `yeto run down --config <run.json>` and check it reports verified; the drain is not retried at interpreter exit |
| `SIGTERM during the exit drain ... ignored` | A stop signal arrived while `run up` was draining its pools | Expected; wait for the drain to end (allow for it in `TimeoutStopSec`) |
| `run up refused: pool <name> is stopped ... but still holds live member(s)` | An earlier stop has not finished releasing its machines | Run `yeto run down --config <run.json>`, then `yeto run up` again |
| `run up refused: pool <name> would refuse every order: N orders placed, max_orders M` (or `max_run_spend_*`) | The pool limits count over the whole run, every earlier `run up` included | Raise `elastic.max_orders` / `max_run_spend_*` in the pool config, or start a new run name |
| `run up` exits with `an earlier run up ... FORCED the sampler pool down` | The 1.2x budget breaker fired (`nc_forced_stop` in the run state) | Check the spend, raise `budget.usd` / `wall_h` if needed, rerun with `--allow-after-forced-stop` |
| Pool log `pool: N orders placed, max_orders M: not ordering` (or `max_nodes`, `max_run_spend_*`) | A pool budget limit refuses new orders | Raise the limit in the pool config |
| Trainer exits with `incarnation gate: fatal store error` | The store refuses the trainer's credentials or the bucket does not exist | Fix the trainer's store env; this is not a pause or starvation |

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

1. the shard is fenced (section 5.2, "What a fence is"), so its claims are
   freed at once and nothing the lost sandbox still writes is trained;
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
| `true` (default) | Every pool is drained: samplers stopped, machines released and confirmed ended. A SIGTERM during the drain is logged and ignored until the drain ends. If the drain fails, the exit prints a `CRITICAL ... may STILL BE BILLING` banner and does not retry; run `yeto run down`. |
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
