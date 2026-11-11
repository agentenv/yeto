# Running the K3.1 vLLM sampler in a Daytona GPU sandbox

A Daytona GPU sandbox is a VM booted from an OCI snapshot, with the MI355X GPUs passed
through. It has no docker daemon, Daytona owns the GPU driver, and you reach it through
SSH (Daytona's gateway) or the SDK's `process.exec`. The National Compute sampler route
(`yeto/launch/nc/ssh_sampler.py`) runs docker on the host. The Daytona route
(`yeto/launch/daytona_sampler.py`) runs the same sampler entry point as a plain process.

| Step | National Compute (bare metal) | Daytona sandbox |
|---|---|---|
| Reach the node | Workspace relay (`tunnel.py`, `relay.mjs`, `port_open.mjs`) as the ssh `ProxyCommand` | Daytona SSH gateway (provider ssh alias) or SDK exec |
| Host setup | Docker on NVMe, passwordless sudo, docker group | None: the snapshot is the image |
| Image | `docker pull` of the pinned `vllm-openai-rocm:kimi-k31` digest | Snapshot built from `docker/daytona-sampler/Dockerfile` (same digest) |
| Card check | `rocm-smi` on the host | `preflight`: `/dev/kfd`, `/dev/dri`, torch HIP, gfx950 |
| Base model | aws-cli container, capped parallel pull to `/mnt/nvme/models` | Same pull shell, run by the sandbox's awscli into `/models` |
| Code | `git archive` to `/mnt/nvme/yeto/<sha>`, mounted at `/yeto` | `git archive` to `/work/yeto/<sha>` |
| Secrets | `--env-file` on `docker run` | Env files (0600) sourced by the start script |
| Agentic setup | Helper steps in the image via `docker run`; tunnel helper container | Helper steps in the sandbox; tunnel is a detached process |
| Sampler | `docker run -d yeto-sampler-<shard>` | `setsid bash /work/sampler/<shard>.start.sh` (pid, exit and log files) |
| Probe / stop | `docker inspect` / `docker stop` | pid file + `/proc` / process-group TERM then KILL |

The sandbox paths are the container paths of the NC route (`/work`, `/models/kimi-k3.1-pro`),
so one sampler spec serves both routes.

## Network

The sampler only makes outbound connections:

- the object store (releases, buffer shards, dispenser ledger, event tapes);
- the sandbox host's docker daemon over SSH (`agentic.setup.sandbox.docker_host`);
- the reverse tunnel `ssh -R` to the sandbox host (`agentic.setup.tunnel`), which carries
  the sandboxes' calls back to the sampler's records proxy.

Nothing has to connect into the sandbox. The vLLM server listens on the loopback address
for the sampler. The pool reads `/metrics` and the event tapes through the exec handle.
On NC, the relay exists only because a node's SSH port is reachable from inside National
Compute and not from the control host. Daytona's SSH gateway replaces it.

Daytona network tiers: Tier 1 and 2 organisations have restricted egress that a sandbox
cannot override, and spot GPU sandboxes need Tier 2 or higher. With a domain allow list,
Daytona's egress proxy re-signs TLS with `/etc/daytona/netleash/ca.crt`. The start script
then exports a combined CA bundle (`AWS_CA_BUNDLE`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`).
A CIDR allow list (Tier 3+) must include the sandbox host for the SSH connections.

## Sizing

The release cache (`/work/cache/yeto`, set by `YETO_CACHE_DIR`) holds at most the served release
plus the one being downloaded. The
follower prunes every other download first. Overlays are symlinks.

| Item | GiB |
|---|---|
| Base MXFP4 checkpoint | B (Kimi-K3.1-Pro: 1.56 TB, about 1,453) |
| Releases on disk | 2 x R (R about 193 for a 128-expert release, about 96 for 64 experts) |
| Snapshot root file system | about 60 |
| vLLM / AITER / triton caches | 64 |
| Harbor site, tasks, trial logs, tapes | 64 |
| Headroom | 15 % |

`daytona_sampler.k31_disk_plan()` computes this. With 128-expert releases the sandbox needs
2,332 GiB; request at least 2,400 GiB of disk. GPU sandboxes default to 512 GiB whatever
the GPU count, and the maximum is 512 GiB per GPU (4,096 GiB for 8).

Memory: Daytona allows up to 192 GiB per GPU (1,536 GiB for 8). The preflight requires
512 GiB by default (`Requirements.memory_gib`). `/dev/shm` must be at least 64 GiB, the size
the NC route gave the container. vLLM's tensor-parallel workers exchange data through it.

## Preflight

Each start runs `daytona_sampler.preflight` in the sandbox before anything else starts. A
failed check raises `PreflightError`, which names the check, what was found and the remedy:

| Check | Requires |
|---|---|
| `kfd` | `/dev/kfd` exists and is read-write for the sandbox user |
| `dri` | At least one `/dev/dri/renderD*` per GPU |
| `rocm` | torch initialises HIP. The driver line (`/sys/module/amdgpu/version`, kernel) and the image's ROCm are printed for comparison. |
| `gpus` | torch sees the required GPU count |
| `arch` | Every GPU is `gfx950`. A failure only warns when `in_place_reload` is off. |
| `shm`, `memory`, `disk` | The sizes above |
| `write:/work`, `write:/models` | The snapshot made both writable |
| `python`, `modules`, `aws` | The snapshot is the yeto sampler image |
| `vllm` | Pinned vLLM version. A mismatch is a warning. |
| `memlock` | `ulimit -l` unlimited. A mismatch is a warning. |
| `store` | The object store endpoint answers |
| `ssh:<host>:<port>` | The sandbox host's SSH port is reachable (agentic specs) |

A failed check that depends on the host (`kfd`, `dri`, `rocm`, `gpus`, `arch`) makes the pool
replace the sandbox. Any other failed check (snapshot, size or network policy) would fail the
same way on every new sandbox, so the pool keeps the sandbox and halts until `pool-retry`.

The in-place expert reload (`vllm_expert_reload`) still checks the MXFP4 backend
(`AITER_MXFP4_BF16`) inside the engine. The preflight can only make sure the cards are gfx950.

## Pool configuration

Set `"sampler_route": "process"` in `pool.json`, or use a provider that exposes
`sampler_route`. The pool then starts, probes, stops and pulls through
`DaytonaSamplerRoute` instead of docker. `model_pull.dest` is a sandbox path under `/models`.

## Security difference

On NC, the tunnel key sits in a directory only the helper container mounts. In a sandbox
the sampler runs as the same user as the tunnel process, so the sampler can read the key
in `/tunnel`. Give the tunnel key only `permitlisten` on the sandbox host, as the onboarding
checklist already requires.

## Running 16 samplers

This section covers a fleet of 16 Daytona MI355X x8 spot sandboxes feeding one trainer
(Modal 6 x 8 B300). Everything below was simulated with 16 fake sandboxes (directories and
processes), a fake ssh, a fake object store and a stub engine. It has not run on 16 real
sandboxes. Run the checks in "Scaling up step by step" on your account before you rely on
16.

### Account prerequisites

| Item | Needed for 16 samplers |
|---|---|
| Daytona tier | Tier 2 or higher (spot GPU sandboxes). With a CIDR allow list (Tier 3+), add every sandbox host. |
| GPU quota | 16 x 8 = 128 MI355X at once, plus 8 per replacement in flight. Ask for 136 or more. |
| Disk | At least 2,400 GiB per sandbox (`daytona_sampler.k31_disk_plan()`, section Sizing). |
| API rate | 16 creates in the first minute, then one `get` per sandbox per pool tick (`pool.interval_s`, default 30 s) and one ssh-token call per sandbox every 14 to 16 hours (24 h tokens). |
| Object store | 16 readers at once (see Bandwidth). Ask your provider whether the account has an egress or request-rate limit. |

Pool configuration for 16:

```json
{"target": 16, "slots": [1, 2, 3, "...", 24], "shard_base": "<run>",
 "provider": {"kind": "daytona", "image": "<sampler image>", "gpus": 8, "spot": true,
              "disk_gib": 2400, "memory_gib": 1536, "price_usd_h": <your spot price>,
              "create_wait_s": 5, "api_attempts": 6}}
```

`slots` (`accounts` is accepted as an alias) are plain indexes, not accounts: give at least `target` + the number of replacements you allow in
flight (24 for 16 samplers). Each replacement takes the next shard number, so the trainer's
`--rl-buffer-shards` and the tunnel port range must cover every shard you expect over the run. Logs, pool events, `yeto rollout status` and the dashboard name these `slot` (for example `slot=2`); only National Compute pools say `account`, because there the slot is an NC account. Event JSON also keeps an `account` field with the same value for older readers.

### What the pool does at 16

- **One tick watches all samplers at once.** The pool reads the 16 sandbox states and probes
  the 16 samplers over ssh in parallel (`watch_parallel`, default 16). The decisions then run
  in order, sandbox by sandbox. A tick costs about two ssh round trips, not 32. One sandbox
  that does not answer cannot delay the others.
- **Creates do not block the tick.** `Daytona.create` waits until the sandbox has started,
  which can take minutes for a GPU sandbox. The provider runs it in the background and waits
  `create_wait_s` for a fast failure. A spot create without capacity fails at once and still
  counts toward the pool's breaker. A create that is still running reads as `waiting`.
- **Rate refusals are retried.** A Daytona call refused with 429 or "rate limit" is retried
  up to `api_attempts` times, with exponential backoff and full jitter (at most 30 s
  apart).
- **SSH token renewals are spread out.** Every sandbox renews its gateway token once less
  than a third of it is left, plus an offset of up to a twelfth that differs per sandbox.
  Sixteen sandboxes created together therefore renew over about two hours instead of in one
  tick.
- **Release downloads survive store throttling.** A ranged GET whose body is cut is
  retried with jittered backoff. The S3 client uses botocore's `standard` retry mode with
  8 attempts, which also retries 429 and 503 SlowDown. `AWS_RETRY_MODE` and
  `AWS_MAX_ATTEMPTS` override it. If a download still fails, it resumes: every whole file
  stays, so only the missing files are fetched again. A failed download is retried after
  45 to 60 s (`download_retry_s`, jittered), so the samplers do not retry in lock step. Each
  sampler also starts a new download after a random 0 to 5 s delay.

### Sandbox hosts

Sixteen samplers that each run about 20 trials at once need about 320 trial containers.
Spread them over several sandbox hosts (two hosts of 160, see the sizing below). Every sampler drives exactly one host.

1. List every host in `agentic.envs.hosts`, each with its `max_sandboxes`. Optionally set
   `tunnel_listen_host` when a host's docker bridge address is not the one in
   `tunnel.listen`:

   ```json
   "envs": {"run": "<run>", "hosts": [
     {"name": "sbx-a", "docker_host": "ssh://sbx@192.0.2.11", "max_sandboxes": 160,
      "trial_memory_gib": 4},
     {"name": "sbx-b", "docker_host": "ssh://sbx@192.0.2.12", "max_sandboxes": 160,
      "trial_memory_gib": 4, "tunnel_listen_host": "192.0.2.101"}], ...}
   ```

2. Set `"docker_host": "envs"` in `agentic.setup.sandbox`, and leave `agentic.setup.tunnel.host`
   unset. Shard `k` then drives one host, chosen by weighted round robin over the hosts with
   `max_sandboxes` as the weight. Its reverse tunnel goes to the same host and listens on
   port `lo + k` of `tunnel.listen`. The sampler's environment carries
   `YETO_SANDBOX_HOST=<name>`, so the dispenser gate counts that host's containers. The gate
   counts the containers of every sampler on that host together against `max_sandboxes`.
3. A replacement sampler continues its predecessor's shard and keeps that shard's host.
   `yeto run scale --add` continues the sequence, so a running sampler never moves. To move
   one shard, pin it with `"assign": {"<sampler id>": "<host name>"}` in `agentic.envs`.
   This applies at the sampler's next start.
4. Install both keys on **every** host. Use one `authorized_keys` line per key, and give
   the tunnel key a `permitlisten` for the whole port range (sshd uses the first line that
   matches, so remove older lines for the same key). Each host needs `GatewayPorts
   clientspecified`, the sidecars and the task images. `yeto sandbox prepare --envs <spec>
   --store <store>` prepares every host (`--host <name>` for one).
5. Check the assignment before you start. No host is contacted:

   ```
   yeto sandbox assign --envs <spec> --islands 16 --shard-base <run> --per-sampler 20
   ```

   It lists the samplers per host and exits 1 when a host would need more than
   `max_sandboxes`. While the run trains, `yeto sandbox status` prints running/max per host
   and a warning at 90 %. A full host makes its samplers wait for room
   (`rl_dispense_wait_capacity` on their tapes): they claim no tasks they cannot run.
6. In the run config, `"sandbox": {"envs": "<spec>", "docker_host": "envs"}` makes the
   container count, its alert line and `yeto run down` cover every host.

Task images. Harbor starts every trial in a new container from its task's image, so pull
every image before the run. Otherwise the first trial of each task waits for its pull.
`yeto sandbox prepare` already does this on every host of `agentic.envs.hosts` (`--host
<name>` for one). It pulls or builds each task image, bakes in the agent runtime when the
taskset asks, and pulls the `prefetch` images of the sidecar files. Images that are already
present are skipped. Up to `pull_parallel` pulls run at once (default 4) and
`build_parallel` builds. A failed pull (a registry blip or rate limit) is tried again, up to
3 times, with backoff.

Every host gets every image. A sampler may take any task, and with `"affinity": "strict"`
it takes only tasks that are ready on its own host. A per-host subset would save disk only
by narrowing which tasks each sampler can run.

Before pulling, `prepare` counts the distinct images and the missing ones. It estimates
their size from the images already present (5 GiB each when none are), compares that with the
free space on docker's disk, and warns when they would not fit with 10 % to spare. The
estimate is printed per host as `images: N needed, M were missing (about X GiB)`.
`yeto sandbox status` prints the missing images per host. To pull them, run `prepare` again;
it is idempotent. How long a full pull takes depends on the registry's bandwidth to the
host, so time one task image on your host first and scale by the missing GiB.

With `"affinity": "prefer"` a sampler may take a sidecar-free task that is ready only on
another host. Its tape then records `rl_dispense_task_off_host`, because that task's image
may be pulled at trial start. Run `prepare` for that host if the event shows up often.

Host sizing, from a stress test of one 64-core, 754 GB, 7 TB host:

- **Bridge addresses come first.** With docker's default bridge (a /24) the host stopped at
  about 250 containers (`no available IPv4 addresses on this network's address pools:
  bridge`). With a /20, 320 containers started with 0 failures, at 11 % load, 2 % memory and a
  steady 0.044 s `docker exec`. Widen the pool before the run: set `bip` or
  `default-address-pools` in the daemon's `daemon.json` (for example a /20), or give the run its
  own large network. `yeto sandbox prepare` and `yeto sandbox status` read each host's bridge
  subnets and warn when they hold fewer addresses than `max_sandboxes`.
- **Memory comes next.** Give every trial container a `--memory` at the task's p95 and set
  `trial_memory_gib` to it. `status` then reports the room as the smallest of
  `max_sandboxes`, the bridge addresses and memory / `trial_memory_gib`. For example, at 4 GiB
  per trial, 754 GB holds about 188 trials at their peak.
- **Recommendation:** about 160 trials per host (8 samplers of 20). 16 samplers need **2
  hosts**.

After a host loses power, check the following before `prepare`. Docker and containerd must
both be running again (restart them together). The data disk or md array must be mounted
where docker's root is. The docker0 address may have changed (tasks must use the
`{{yeto.sidecar.<name>.url}}` placeholders, never a fixed bridge IP). The sidecars must be
running.

### Bandwidth, cold start and cost

`yeto rollout plan` prints the estimate offline. Every default names its source: measured,
or an assumption that you replace with your own numbers.

```
yeto rollout plan --islands 16 --island-usd-h <your spot price> --step-s <your step time>
```

With the defaults (16 islands, 1.56 TB base, 207 GB 128-expert release):

| Item | 16 samplers | Source |
|---|---|---|
| Cold start (base pull at about 1.5 GB/s, then engine start) | about 32 min per sampler; 25 TB pulled by the fleet | measured on NC MI355X (1.2 to 1.8 GB/s under a cap); engine start assumed |
| Release fan-out | 3.3 TB per step (16 x 207 GB) | measured release size |
| Release apply | about 7 min (1.5 GB/s download plus a 5 min reload); the K3.1 sampler fetched 103.3 GB in 106.6 s (0.97 GB/s) | measured download; reload time assumed |
| Policy age | about 3 versions at a 5 min step | trainer step assumed |
| Sandbox hosts | 320 containers: 2 hosts of 160 | host stress test; trials per sampler assumed |
| Cost per hour | 16 x your spot price + about $408 trainer | trainer measured; spot price yours |

Per-sampler download speed is the key input. An R2 test with 16 concurrent readers scaled
linearly to 4.36 GB/s in total, with no 429, 503 or SlowDown, and 596 metadata requests per
second without errors. One Modal container was the bottleneck (140 to 380 MB/s). A single
MI355X sandbox reaching 1 GB/s or more has not been measured. At 275 MB/s (one `aws s3 sync` process) a 207 GB release
takes about 12.5 min per step, which holds back training. At 1 GB/s it takes about 3.5 min,
and at 2.5 GB/s about 83 s. `yeto rollout plan` warns when `--release-mb-s` is below
1000. In that case, use delta releases or fan the release out inside the fleet.

Keep each sandbox at about 256 concurrent range reads and no more, for example
`YETO_RL_STORE_DOWNLOAD_PROCESSES=16`, `YETO_RL_STORE_FILE_CONCURRENCY=16` and
`YETO_RL_STORE_PART_CONCURRENCY=16`, with 32 to 64 MB parts (`YETO_RL_STORE_PART_MB`,
default 64). Set them in the pool's store env file (`r2_env_file`), which the sampler and the base pull both read. The base pull's `aws s3 sync` takes
`AWS_RETRY_MODE=standard` and `AWS_MAX_ATTEMPTS=8` from the same env file.

The store sees 16 readers at once: about 24 GB/s at 1.5 GB/s each. Daytona's per-sandbox
limit is unknown; measure it on your first sandbox and pass it as `--pull-mb-s` and
`--release-mb-s`. Do not push one node past about 3 GB/s: an MI355X node reading R2 at
about 7 GB/s dropped off the network after 1 to 2 minutes, which is why yeto caps the base
pull at 2500 MB/s. If your account caps egress, pass
`--store-gb-s` to the plan: every island then gets its share, and the cold start and policy
age grow with it. The samplers always load the newest release and skip older ones. A
release that takes longer to apply than a trainer step therefore makes the samplers serve an
older policy. It does not queue releases. Release every k steps, or release fewer experts,
when the plan warns about policy age.

### Scaling up step by step

Start with 4 samplers and double the fleet only after each check passes:

```
yeto run up --config <run.json>                       # pool target 4
yeto run scale --config <run.json> --add 4            # 8
yeto run scale --config <run.json> --add 8            # 16
```

At each size, before you add more, check the following:

- `yeto rollout status --config <run.json>`: every sampler is ready, with no `order_failed`
  or breaker events. Daytona quota and spot capacity show up here first.
- the dashboard: every sampler's policy version follows the trainer within the plan's
  policy age, and groups arrive steadily.
- `yeto sandbox status --envs <spec> --store <store>`: every host is below 90 % of
  `max_sandboxes`, and no sampler tape shows `rl_dispense_wait_capacity` for long.
- the sampler logs: no repeated store errors (`download_failed`) during a release.
- the pool log: no `pool: tick took` warning (a tick over 60 s).

If a step fails, scale back with `yeto run scale --config <run.json> --samplers <N>`. Fix the
cause (quota, host size, store limit), then add again.
