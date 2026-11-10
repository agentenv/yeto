# Clouds

Which clouds `yeto shape` can plan across and `yeto launch` can provision
on, where each one's credentials live, what "region" means to it, and
which behaviours have been verified on real machines. Yeto keeps no
credentials of its own: every cloud's own CLI writes them to a standard
place on the **submitting machine**, SkyPilot and Yeto read them there,
and in head controller mode the launcher copies only the files of the
clouds a fleet actually touches onto the head VM. Learner islands never
receive cloud credentials (only the Hugging Face token, and W&B /
CyberGym keys when those features are on).

On Windows, run everything from WSL: SkyPilot imports the POSIX-only
`resource` module and fails on native Windows. Credential files must be in
the WSL home directory (`~` inside WSL), not the Windows user profile.

## Credentials

| Cloud | Set up with | File Yeto/sky read | Env-var alternative |
|---|---|---|---|
| AWS | `aws configure` | `~/.aws/` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| RunPod | `runpod config` | `~/.runpod/config.toml` | `RUNPOD_API_KEY` |
| Nebius | `nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt` and `nebius --format json iam whoami \| jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt` | `~/.nebius/` | `NEBIUS_IAM_TOKEN` + `NEBIUS_TENANT_ID` |
| Verda | console → Credentials → Cloud API Credentials, saved as JSON | `~/.verda/config.json` (`client_id`, `client_secret`) | `VERDA_CLIENT_ID` + `VERDA_CLIENT_SECRET` |
| Modal | `modal token new` | `~/.modal.toml` | `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` |
| GCP (optional, `gs://` outputs only) | `gcloud auth application-default login` | `~/.config/gcloud/` | — |

`sky check` reports each sky cloud as enabled or explains what is missing.
Modal is not a sky cloud; `modal token` shows the active token.

**Nebius binds one project to one region.** Every Nebius region a fleet
touches needs `nebius.region_configs.<region>.project_id` in
`~/.sky/config.yaml`; `yeto launch` refuses a fleet that names a region
without one, before any VM is created. The same project ids let the
planner ask Nebius for live preemptible prices.

## What each cloud means by "region"

| Cloud | `--regions` entry | Meaning | Notes |
|---|---|---|---|
| AWS | `aws:us-east-1` (or bare `us-east-1`) | data-center region | quotas, placement scores and prices are per region; default `us-east-1,us-east-2,us-west-1,us-west-2` |
| Nebius | `nebius:eu-north1` | region | eu-north1, eu-west1, me-west1, uk-south1, uk-south2, us-central1; H100×8 only in eu-north1, B200 only in me-west1/us-central1 |
| Verda | `verda:FIN-03` | location code | FIN-01/02/03 (Helsinki), ICL-01 (Iceland); the planner reads the live list, falling back to these |
| RunPod | `runpod:CA` | country code | one global pool; stock is not per country |
| Modal | `modal:us` / `modal:us-west` | placement hint with a surcharge | broad (`us`, `eu`, `ap`) ×1.15, narrow (e.g. `us-west`, `eu-north`) ×1.75; leave it out to run unpinned at the base price |

Clouds you do not name in `--regions` are unrestricted, except AWS, which
stays on its default list.

## What the planner knows per cloud

| Cloud | Catalog and price | Capacity signal | Spot |
|---|---|---|---|
| AWS | sky catalog (refreshed every few hours) | vCPU quota − usage, spot placement score (AWS APIs) | yes |
| RunPod | sky catalog (no H200 rows — the planner says so) | stock High/Medium/Low (RunPod GraphQL) | catalog spot = on-demand |
| Nebius | sky catalog with spot column; live preemptible price from the billing calculator when a project id is configured | Capacity API resource-advice, per region/platform/preset | yes; dynamic spot pricing — the live price overrides the catalog |
| Verda | Verda's public `/v1/instance-types` (sky's catalog is too thin) × locations | one bulk `/v1/instance-availability` call | flat 50% of on-demand |
| Modal | static dated table in `yeto/shape/providers.py` (GPU + reserved CPU + memory); the planner warns when it is more than 90 days old | none; constant "will run, may queue" | none |

## Modal islands

Modal has no VMs and no SSH, so a `modal:` island is a Modal function call
(`yeto/modal_runner.py`), not a sky cluster. It runs the same `run`
script the sky task would run, with the `SKYPILOT_*` variables provided
from Modal's cluster info. Facts to keep in mind:

- The syncer must be reachable from Modal's network. Head controller mode
  (the default) puts it on a public head VM; under `--controller local`
  pass `--syncer-public-addr HOST:PORT` when the syncer address is private.
- Multi-container islands (`modal:2x8xh100`) must use whole nodes
  (`H100:8` per container, a Modal rule) and get RDMA.
- RL islands need `--rl-image docker:<repo>@sha256:<digest>` — the same
  digest the sky islands use.
- Object-store data (`s3://`, `gs://`) cannot be mounted; use an HF dataset
  id or a local path.
- `yeto down <run>` stops the run's Modal app (`yeto-<prefix>`); every
  island's containers end with it, and the app must then list as
  `stopped` with 0 tasks or the run is reported not fully down.
- An all-Modal SFT fleet's model is not fetchable over ssh; recover it
  from the syncer checkpoint with `yeto-export`, or keep at least one sky
  island in the fleet.
- Manual join: launch with `--external-learners 1`, then
  `python -m yeto.modal_runner --learner-id N --num-learners M --syncer-addr H:P --run-script <file> ...`.

## Default container images

RL islands and engine samplers run digest-pinned images (`yeto-miles-k31`,
`yeto-miles-ports`, `yeto-sampler-k31`) from `ghcr.io/agentenv` by default.
The digest is fixed in the code; only the registry prefix is configurable:

```bash
export YETO_IMAGE_REGISTRY=<registry>/<org>   # a mirror holding the same digests
```

| Image | Used by |
|---|---|
| `yeto-miles-k31@sha256:63ad0d2a…` | RL trainer default (Kimi-K3.1 LoRA and other recipes) |
| `yeto-miles-k31@sha256:2de5d796…` | Kimi-K3.1 `--tuning full` trainer default |
| `vllm-openai-rocm@sha256:5f47cc10…` | Kimi-K3.1 A-card vLLM sampler |

`--rl-image docker:<repository>@sha256:<digest>` replaces the image of one
run, and a sampler spec's `image` replaces the sampler's.

**The packages are private for now.** Pulls need a registry token with
`read:packages` on the organization:

```bash
export SKYPILOT_DOCKER_SERVER=ghcr.io          # or your mirror's host
export SKYPILOT_DOCKER_USERNAME=<github-user>
export SKYPILOT_DOCKER_PASSWORD=<token-with-read:packages>
yeto launch ... --rl-image-private             # forward the login to the islands
```

The same variables (or a `ghcr.io` entry in `~/.docker/config.json`) serve SkyPilot
VMs and Modal islands (yeto passes them to Modal as a registry secret). An A-card
sampler node pulls with `docker login <registry>` using the same token before
`docker pull`. Without credentials the pull fails with an authorization error.

## Modal B300 clusters (Kimi K3.1 trainer)

A K3.1 trainer island is a Modal clustered function on whole B300 nodes,
`--gpu modal:6x8xb300` (`MODAL_GPUS["B300"]`, 8 per container like the other
whole-node types). Facts to keep in mind:

- The decorator is `modal.clustered(size=N, rdma=True)` (Modal Clusters GA);
  a client that only has `modal.experimental.clustered` is used through the
  same path (`modal_runner.clustered_decorator`). Modal sets the NCCL
  interface variables for RDMA itself; yeto exports `NCCL_IB_DISABLE=0` and
  nothing about `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`.
- Every Modal island gets `NCCL_RAS_ENABLE=0` (`MODAL_ISLAND_DEFAULT_ENVS`):
  NCCL 2.30's RAS thread crashes the process after the first engine memory
  release. `--modal-env KEY=VALUE` overrides any default; VM-passthrough hosts
  need `--modal-env NCCL_NVLS_ENABLE=0`.
- A preempted or failed container restarts the whole cluster. With a
  `modal-volume://` or `s3://` checkpoint store the re-run is a resume from
  the store's `LATEST` (`storage.resumes_in_new_container`), not a failure.
- Image: `docker/miles-k31/` (one overlay on `radixark/miles:kimi-k31`, CUDA
  13.0.3, NCCL 2.30.7). Modal's GPU page says "B300 requires CUDA 13.1+"
  while the image toolkit is 13.0.3; the Modal host driver (610.x, CUDA 13.3)
  runs 13.x runtimes. Whether the 13.0.3 toolkit satisfies the B300 JIT paths
  (flashinfer, deep_gemm, triton) is open until the first smoke test on two
  `B300:8` containers; the fallback is a 13.1 toolchain layer with
  `CUDA_HOME` pointed at it. The same test decides IB verbs vs EFA hosts
  (`experimental_options={"efa_disabled": True}` forces IB).
- Shape planning (`yeto shape`) does not price B300 on Modal: the price table
  (`MODAL_GPU_USD_PER_SECOND`) has no entry and none is guessed; `yeto launch
  --gpu modal:...xb300` works.

## S3-compatible stores (Cloudflare R2) on Modal

A Modal island takes an `s3://BUCKET[/PREFIX]` `--rl-checkpoint-store`. Modal
mounts no bucket, so every node keeps a local stage of the store and moves
the whole store by object (`yeto.cloud.s3_store`, `STAGE_ENV`): round cuts,
`LATEST`, the sample buffer, policy releases and trainer commands. The head
(trainer scheduler, convertible pool) reads the same store by object key.

- Endpoint: `--rl-object-store-endpoint https://<account>.r2.cloudflarestorage.com`
  (or `AWS_ENDPOINT_URL` in the launcher environment). The region becomes
  `auto` unless one is configured; the launcher exports the endpoint to its
  own environment too so every launcher-side boto3 client (head object API,
  code pack store) talks to the same endpoint.
- A standard AWS setup works as is. The endpoint is, in order:
  `YETO_RL_STORE_S3_ENDPOINT_URL`, `AWS_ENDPOINT_URL_S3`, `AWS_ENDPOINT_URL`,
  then `endpoint_url` of the store's AWS profile (top level or in its `s3`
  section). The region is `YETO_RL_STORE_S3_REGION`, `AWS_REGION` /
  `AWS_DEFAULT_REGION`, the profile's `region`, else `auto` for an R2 endpoint.
  The profile is `YETO_RL_STORE_S3_PROFILE`, else `AWS_PROFILE`; `yeto launch
  --rl-object-store-profile NAME` sets it for a launch. The head reads with the
  profile, and without `--modal-store-secret` its key, endpoint and region are
  shipped to the islands (an island has no profile to read). A profile for R2:

  ```ini
  # ~/.aws/config
  [profile my-r2]
  region = auto
  endpoint_url = https://<account>.r2.cloudflarestorage.com
  ```

  with the key pair under `[my-r2]` in `~/.aws/credentials`. Store CLIs
  (`python -m yeto.rl.engine.store_dispense status`) take `--endpoint-url` and
  `--profile`. A credential or endpoint error names what to set: for example,
  an R2 key sent to AWS S3 because no endpoint was set reports
  `AWS S3 (no endpoint set) refused the credentials (InvalidAccessKeyId)` and
  the endpoint variables.
- Credentials: `--modal-store-secret NAME` attaches a Modal Secret holding
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (and optionally
  `AWS_ENDPOINT_URL`, `AWS_REGION`) to the island function; the island reads
  that pair because the launch says so (`YETO_RL_STORE_S3_CREDENTIALS=aws-env`).
  Create it once:
  `modal secret create NAME AWS_ACCESS_KEY_ID=<key id> AWS_SECRET_ACCESS_KEY=<secret> AWS_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com AWS_REGION=auto`
  (an R2 API token with Object Read & Write on the bucket).
  Without the flag the launcher's store credential (`YETO_RL_STORE_S3_*`) is
  shipped as on sky. The launcher's own `AWS_*` variables are never a store
  credential, and no island uses the default credential chain.
- Uploads: a file larger than one part (`YETO_RL_STORE_PART_MB`, default 64)
  goes as a resumable multipart upload. The UploadId and acknowledged parts
  are journaled next to the source (`<file>.upload.tmp`); a failed try
  continues from the parts already stored, up to `YETO_RL_STORE_UPLOAD_ATTEMPTS`
  tries (default 3), and a later push or a restarted rank continues it too.
- Part checksums, `YETO_RL_STORE_PART_CHECKSUM` (forwarded to every island):
  `1` sends a SHA-256 with every part and checks the store's composite
  checksum after the upload; `0` sends none; `auto` (the default) sends them
  except to a Cloudflare R2 endpoint (`*.r2.cloudflarestorage.com`), which
  answers `NotImplemented` to a checksummed `UploadPart`. Any other endpoint
  that answers `NotImplemented` turns checksums off for that store with one
  warning per process (`does not take SHA-256 part checksums`) and the file
  is uploaded again without them; set `0` for such an endpoint to skip the
  probe. Without part checksums a cut is still checked by size and by the
  manifest's SHA-256 when it is read.
- Account ids, bucket names and endpoints live in the launch flags, the
  environment or the Secret, never in the code.
- Open until a run: the per-node stage of a multi-node trainer (each node
  pushes its own shards and pulls every subtree it does not have) and the
  store throughput from Modal to R2.

## National Compute sampler nodes (MI355X, ROCm)

An account holds one 8-card node at a time. A lease is granted for up to 4
hours and cannot be renewed. Nodes can be preempted, and the local disk is
destroyed at release. The rollout side of a K3.1 run is therefore a set of
short-lived sampler islands over the object store.

> **Billing warning.** A lease does **not** end on its own at `lease_until`.
> When `lease_until` passes, the ticket stays `active` and **keeps billing**
> until it is deleted. Each node must be released explicitly:
>
> - `python -m yeto.launch.nc pool-stop --config <pool.json>` releases every
>   node of a pool;
> - `python -m yeto.launch.nc down --account <account> --ticket <ticket>`
>   releases a single node.
>
> The pool does this for you only while its control loop runs. A lease recorded
> in a run's resource ledger is also shown by `yeto cost --run RUN`, deleted by
> `yeto cleanup --run RUN` / `yeto down RUN`, and deleted by the cost guard 5
> minutes after its `lease_until` ([COST_GUARD.md](COST_GUARD.md)).

**Recommended path:** the vLLM sampler pool, `python -m yeto.launch.nc pool`.
It orders the nodes, starts and replaces the samplers, and releases the nodes
before their leases end. The pool config, `--dry-run`, `pool-status`,
`pool-stop` and the control-host requirement are covered in
[NC_POOL.md](NC_POOL.md). The task dispenser that its samplers share is
covered in [DISPENSER.md](DISPENSER.md).

**Legacy path (manual SGLang node scripts).** `scripts/sampler_node/` holds
scripts that set up one node by hand with the SGLang server. Each value is a
parameter; no site, account or model name is written in them. Nothing
releases a node set up this way except `down` (see the billing warning
above). The scripts:

- `nc_lease.sh --account N` polls `nc-query status` until the ticket is
  active, writes `node.json` (ip, user, host key, lease_until, rate) and pins
  the host key.
- `bootstrap.sh --image IMAGE --model MODEL` reports the host (a node without
  `amd-smi` or `rocm-smi` is refused: the launcher cannot check its cards),
  makes the NVMe directories, pulls the SGLang ROCm image and fetches the
  base model (HF repo, HF bucket, or `--peer user@host:/path` from a node
  that has it) in the background. With `--release-store DIR --release-prefix
  RUN [--release-peers host:port,...] [--serve-port P]` it also starts the
  policy release puller (`python -m yeto.rl.spot.s3_delivery pull-loop`):
  releases come from the peer nodes first (near-region fan-out,
  `yeto.rl.spot.peer_fanout`) and from the store only when no peer has them,
  and the node serves its own placed releases on `--serve-port`. The bucket,
  endpoint and credentials come from `--r2-env-file` (mode 0600; rclone takes
  the endpoint through `RCLONE_S3_ENDPOINT`).
- `run_sglang.sh --image IMAGE --model-dir DIR --tp 8` starts the server in
  docker with the K3.1 contract on: `--enable-return-routed-experts`,
  `--enable-fp32-lm-head`, `--attn-dp-size TP`, `--moe-a2a-backend deepep`
  (each a parameter), and the ROCm environment
  `ROCM_QUICK_REDUCE_QUANTIZATION=NONE SGLANG_ROCM_RUNTIME_RESERVE_GB=10
  AITER_SITUV2_A8W4=1 HSA_NO_SCRATCH_RECLAIM=1` (no `MX_PATCH`); the store env
  file goes in as the container environment and must be mode 0600.
- `run_engine_sampler.sh --spec SPEC.json` starts
  `python -m yeto.rl.engine.engine_sampler` with the spec; the store
  credentials come from `--r2-env-file`; the interpreter must import
  `safetensors` and `boto3` (`--pip-install` adds them).

Launcher side, an MI355X node of an SSH inventory is a `rocm` device-family
node (`yeto/gpu_spec.py`): the card preflight runs `amd-smi static --json`,
then `rocm-smi --showproductname --showmeminfo vram --json`, and compares the
cards, count and HBM (288 GiB) with the inventory (`yeto/hw/card_tools.py`).
The JSON shapes follow the ROCm documentation and are confirmed on the first
lease. `scripts/prune_layers.py` cuts the first N layers of a checkpoint
(K3.1: `--layers 4` keeps the dense layer, two KDA layers and the first
MLA/DSA layer with all 896 experts, 7 of 96 shards) for CPU and single-node
checks.

## Verified on real machines

Verification results decide runtime behaviour: the planner only plans RL
islands on clouds in `VERIFIED_DOCKER_IMAGE_CLOUDS`, prices RL spot only
on clouds in `VERIFIED_SPOT_STORAGE_CLOUDS`, and assumes an RDMA fabric
for multi-node islands only on `RDMA_CLOUDS` (all in
`yeto/shape/catalog.py`). Update the constants and this table together.

| Cloud | SFT island, 2+ sync rounds | RL island via `docker:` image | Spot checkpoint store | Multi-node fabric |
|---|---|---|---|---|
| AWS | yes | yes | yes (`sky.Storage`) | EFA on p4/p5 |
| RunPod | yes | yes | not verified | single node |
| Nebius | yes | not verified | not verified (the S3-compatible store needs `aws configure --profile nebius`) | not verified (InfiniBand expected on 8-GPU SXM) |
| Verda | not verified | not verified | not available in sky | single node |
| Modal | yes | yes (8xH100, single container) | not verified (Modal Volume); `s3://` (R2) wired, not verified | not verified (RoCE, whole nodes; B300 clusters wired, not verified) |

A mixed fleet (learner islands on Nebius and Modal, syncer head on AWS) and a
manual Modal join next to a Nebius island have also run to completion, with
every island answering every outer step.

Practical notes from those runs:

- With more than one island and the default `--quorum 1`, the islands race
  and each answers a different outer step. Pass `--quorum <islands>` when
  every round should include every island.
- The head installs sky's `nebius` / `verda` / `runpod` extras, plus `modal`
  when the fleet has a Modal island, and needs `~/.sky/config.yaml` for the
  Nebius project ids.
- A Nebius tenant's default public-IPv4 quota is 3, i.e. at most three VMs
  across all runs.
- A tokenizer whose chat template has no `{% generation %}` block (for
  example Qwen3-0.6B) needs `--assistant-mask-mode legacy` for assistant-only
  SFT.
- A sky island's training log stays on its own cluster and is lost at
  teardown; run `yeto logs <prefix>` before `yeto down` if you need per-step
  learner metrics. A Modal island's stdout streams back through the SDK.
- The syncer log formats gradient norms with `{gnorm:.4}`; tiny values print
  as `0.0000`. Read the tape for the real value. `sync_ms` in the tape is the
  syncer's merge time, not a WAN measurement; WAN time is folded into
  `quorum_ms`.
- Under `strict-avg` the run has one fragment and `--pipeline` is clamped to
  `--fragments`, so it has no effect. To hide WAN round trips, use the
  decoupled preset with more fragments; the default `--pipeline 2` is a sound
  start.
- Planner limits: Verda's stock signal is binary, so the planner may propose
  more 1-GPU islands than Verda has; the planner warns about, but still plans
  in, Nebius regions without a project id, which `yeto launch` then refuses.
- Manual Modal join: the run script must be built from the fleet's own
  arguments and source (`launcher.make_learner_task` for a `modal:` spec at the
  seat's learner id); the learner refuses to start when its source SHA256
  differs from the fleet's. The launch log prints only an MLX join command.

### Log streaming: sky heartbeats are not the end of the stream

The launcher follows every sky job with `sky.tail_logs(..., follow=True,
preload_content=False)` from a tail thread (`launcher._tail`), and the RL
event tape is echoed over that stream. The SkyPilot API server's
`log_streamer` appends a `<heartbeat></heartbeat>` control payload after every
30 s without a new log line, and the client's `decode_rich_status` yields
`None` for every control payload when it runs off the main thread. The sdk
docstring describes `None` as "the log has been completely streamed", so a
loop that stops on `None` ends at the first quiet 30 s (model load,
generation). The launcher skips `None`; the end of the stream is the iterator
being exhausted. The island-file recovery of the tape (`_recover_echo_tape` /
`settle_echo_tapes`) stays as the fallback.

## Thread preflight before launch

`yeto launch` counts the threads of the current user before it creates any
cloud resource (the same number as `ps -L -u $USER | wc -l` without the header
line; read from `/proc`). A per-user thread limit (`ulimit -u`, or a watchdog
that enforces one) can tear down a live run when the user goes over it, so a
launch must not start near it. The check runs in `launcher.run` before
`sky_patches.install()`, in `yeto launch` before the worker is spawned, and in
the head path before the head VM is provisioned.

| Threads | `wait` (default) | `error` | `off` |
|---|---|---|---|
| below `--preflight-thread-start` (default 2800) | continue | continue | no check |
| start .. below `--preflight-thread-hard` (default 3000) | re-read every 30 s; continue when below start; stop after `--preflight-thread-wait-s` (default 1800) | stop | no check |
| hard limit or more | stop | stop | no check |

A stop creates no cloud resource. The message gives the total, the
threshold, the hard limit and the threads of the SkyPilot API server
(`sky.server` process, its `SkyPilot:executor` children and their
descendants) with their share, or "SkyPilot API server: not found". To free threads:
`sky api stop && sky api start` -- only when no launch is in progress (a
restart cancels a running `sky.launch`). Do not run local tests that start
Ray while a GPU launch is live.

Switch off: `--preflight-threads off` (prints a warning; the run manifest
records `preflight.threads.disabled: true`). The readings are in the run
manifest (`preflight.threads`).

## Tearing a run down

`yeto down <prefix>` only says `run '<prefix>' is down` (exit 0) once every
cluster of the run is confirmed gone; anything less exits 1, leaves the run
in state `TEARDOWN_INCOMPLETE` with the unconfirmed clusters recorded, and
a rerun of `yeto down` continues from there. The order matters:

1. **Learners of a head run are torn down from the head.** Only the head's
   sky knows them: this machine's sky launched the head, the head's sky
   launched the learners, and a local `sky down <learner>` just answers
   "does not exist". `yeto down` cancels the controller job on the head
   (so it cannot relaunch what is being removed), then runs the downs over
   ssh on the head, where each learner is also checked at the cloud before
   it counts as down. Three attempts, 20 s apart, because the head's own
   sky answers 500 while it is busy.
2. **The head is deleted only after every learner is confirmed.** If any
   learner is still unconfirmed the head is kept, the command exits 1 and
   names the learners: rerun `yeto down`, or delete them in the cloud
   console. Deleting the head first orphans the learners.
3. **Modal islands end with the run's app**, then the app must list as
   `stopped` with 0 tasks.
4. **Everything this machine's sky launched (the head, local-mode
   learners) is checked at the cloud after the down**, using sky's own
   per-cloud instance query; a surviving instance is retried and, if it
   outlives the retries, printed by id so it can be deleted by hand. A
   cloud that cannot be queried is reported as "not cloud-verifiable;
   trusting sky", and then only a clean down or "does not exist" counts.

If the head is already gone (deleted by hand, or by an older `yeto down`),
step 1 cannot run: the command exits 1 listing the learners, and the only
way to find them is the cloud's own listing, e.g.
`nebius compute instance list` — look for instances named after the
learner cluster.
