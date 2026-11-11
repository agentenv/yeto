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

The release cache holds at most the served release plus the one being downloaded. The
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
