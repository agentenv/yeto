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
| Control host | The Hub config only (`config.json`, for the trainer dry run) | During preflight |

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

`pool --dry-run` (part of the `yeto run up` preflight) reads
`s3://<bucket>/<src_key>/config.json` with the store env file, and stops when
the model is missing or the credentials cannot read it. Without this check a
missing upload would surface only after a node was ordered and billed.

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
| `dashboard.prices` | Price table JSON for the dashboard's cost panel (default: the built-in example table, see section 6). `up` also passes the run name and `budget.usd` to the dashboard. |
| `budget.usd` | Spend cap (trainer plus NC estimate). |
| `budget.wall_h` | Wall-clock cap from the pool start (default 3). |
| `budget.nc_usd_per_h` | Price of one NC node hour for the estimate. |
| `budget.silence_min` | Minimum silence window of the backstop; the effective window is `max(4 x --rl-stall-timeout, silence_min)`. |
| `budget.release_nc_on_breach` | `true`: a budget or wall-clock breach performs the full `down` including the NC pool. Default `false` (see section 7). |
| `budget.release_nc_on_finish` | `true`: the full `down` runs as soon as the trainer finishes (`SUCCEEDED`). Default `false` (see section 5.5). |

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
| `target` | Samplers to keep running. At most the number of `accounts`. Change it at run time with `yeto run scale`. |
| `accounts` | NC accounts the pool may order on; each must also be in `YETO_NC_ACCOUNTS`. Accounts beyond `target` are used for replacements. |
| `shard_base` | Shard prefix: nodes write `<shard_base>-0`, `<shard_base>-1`, ... Also names the pool state file `pool-<shard_base>.json`. |
| `shard_limit` | How many shard indexes the run may take; set it to the number of shards in the trainer's `--rl-buffer-shards`. The usable range is also bounded by the port range of `agentic.setup.tunnel.listen` (one port per shard). Shard indexes are not reused, so replacements consume new indexes. |
| `elastic.max_nodes` | Most billed nodes at once (default `2 x target`). |
| `elastic.max_orders` | Most orders over the whole run. |
| `elastic.engine_stall_s` | Seconds without generated tokens while requests are pending before an engine counts as stalled (default 600). |
| `elastic.max_sampler_restarts` | In-place restarts of a crashed sampler before its node is replaced (default 2). Restarts during node setup (before the first join) and before a `pool-retry` do not count, and the count starts again after `elastic.sampler_restart_reset_s` (default 1800) of healthy running. |
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
| `reload_max_defer_s` | Longest time, in seconds, a downloaded policy release may wait for running agentic trials before the sampler loads it anyway (default 600; `null` waits indefinitely). A forced load is reported as `rl_sampler_policy_loaded` with `forced: true` and `deferred_s`; trials that span the switch are excluded from training (lost as `served_by_other_version`, or aborted when the switch falls inside a model call) and counted in `rl_agentic_group`. |
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
   trial containers;
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
run). Recommended practice: set `"release_nc_on_finish": true` in `budget` so
the monitor runs the full `down` as soon as the trainer finishes, or run
`yeto run down` yourself.

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
loading it, with the latency from the release. A replacement island that
started on the newest release shows `(at start)` instead of a latency. A
release no running sampler loaded shows `not loaded` (with
`--rl-max-policy-age` above 0 a sampler may skip versions). NC pool events
appear as dashed lines.

**NC pool events.** The `nc_elastic_*` stream (ordered, member_ready,
shard_fenced, evicted, node_lost, engine_stalled, released, ...). Use it to
follow a replacement end to end.

### Other panels

- **Island health**, **Event stream**: per-island state and the raw events.
- **Cost (estimate, not billing data)**: spend per resource from the resource
  ledger and the price table (`--prices`, from `dashboard.prices` in
  `run.json`). The cost bar uses `--budget` (`budget.usd`). An NC sampler is
  priced as a whole 8-GPU node, whatever TP its engine uses, from the
  ticket's order to its release (pool events). The trainer is priced from the
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
| Trainer finished | launcher finished `SUCCEEDED` (exit code 0) | Banner: the NC pool still bills; with `budget.release_nc_on_finish` the full `down`. |
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

**Preflight stops with `model_pull: no s3://<bucket>/<src_key>/config.json`.**
The model is not in the bucket under `model_pull.src_key`, or the store env
file's key cannot read it. Upload it (section 2, Model weights) and run the
dry run again. Nothing was ordered.

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

## Planned enhancements

- The padded-sample fraction as a dashboard column next to the padded sample
  count.
- Pool configuration changes (accounts, limits) without a pool restart. Today,
  adding accounts takes a pool restart; the target can change at run time.
- Release-mode images on the K3.1 full-parameter base.
- A measured first step of the 262k layout (split 32/29/32) under production conditions.
