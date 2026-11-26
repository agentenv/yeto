# Kimi K3.1 full-parameter trainer on your own 6 x 8 B300 cluster

This runbook brings up the Kimi K3.1 async GRPO trainer on a bare-metal
cluster that you own: six HGX B300 nodes with eight GPUs each, reached over
SSH, running Docker, connected by InfiniBand or RoCE. The samplers stay on
National Compute MI355X nodes, and the object store connects the two sides.
Everything that is not specific to bare metal (samplers, pool, sandbox host,
dashboard, breakers) is described in the end-to-end guide,
[KIMI_K31_ASYNC_RL_GUIDE.md](KIMI_K31_ASYNC_RL_GUIDE.md). The layout and its
measurements are in [K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md).

Templates:

| File | Purpose |
|---|---|
| [`examples/k31/inventory.b300-6x8-baremetal.yaml`](../examples/k31/inventory.b300-6x8-baremetal.yaml) | The six nodes: SSH target, cluster address, NIC, GPUs, fabric, Docker flags, mounts and per-node environment (section 4). |
| [`examples/k31/run.baremetal-b300-262k.json`](../examples/k31/run.baremetal-b300-262k.json) | Run config: the 262k trainer on the inventory above, the NC MI355X pool `pool.nc-mi355x-agentic-262k.json`, budget and dashboard (section 10). |
| [`examples/k31/algorithm-spec.async-age12.json`](../examples/k31/algorithm-spec.async-age12.json) | Algorithm spec for `--rl-max-policy-age 12` with `--rl-loss score_centering --rl-score-centering-is tis` (the loss carries the importance correction). |

Both templates pass `yeto launch --dry-run` as shipped once the placeholders
are filled (`tests/test_k31_baremetal_kit.py` runs this check).

## Contents

1. [Status](#1-status)
2. [Architecture](#2-architecture)
3. [Cluster requirements](#3-cluster-requirements)
4. [Node inventory](#4-node-inventory)
5. [Trainer image](#5-trainer-image)
6. [Model weights](#6-model-weights)
7. [Expert selection file](#7-expert-selection-file)
8. [Parallel layout and memory](#8-parallel-layout-and-memory)
9. [Bring-up order](#9-bring-up-order)
10. [Launch](#10-launch)
11. [Monitoring](#11-monitoring)
12. [Failure modes](#12-failure-modes)

## 1. Status

| Item | Status |
|---|---|
| Layout PP3 x TP8 x CP2 x EP16, 128 trainable experts | Measured at 524k on 6 x 8 B300 (Modal, IB); 262k recommended from the fitted memory model ([K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md#1-which-262k)) |
| SSH + Docker trainer island on NVIDIA nodes | CPU-tested only (fake SSH clusters, the launcher end to end). It has not run on a bare-metal NVIDIA cluster yet: the first bring-up follows section 9 step by step |
| Resident trainer island (a trainer-only launch supervised by the fleet: relaunch, reattach, teardown per node) | In this repository, unit-tested |
| Inventory, Docker flag checks, image pre-pull, per-node environment, node scratch, object-store staging, code pack on the nodes | In this repository, unit-tested |
| Cluster check (`yeto preflight`) | In this repository, unit-tested against recorded tool output |
| Overlay-safe trainer image | `yeto-miles-k31@sha256:3a0c5445...` (next9, the `--tuning full` default; section 5); the overlay-safe base was pulled and run on overlay2 and on the containerd snapshotter (Docker 29.9 / 29.10) |

Each bring-up step lists what it must establish and the command that checks
it.

## 2. Architecture

```
 control host (yeto run up)
   |  SSH (BatchMode, keys)                          object store (S3 API)
   v                                                  ^   sample buffer, releases,
 6 x 8 B300 (your cluster)                            |   leases, event tapes
   one Docker container per node  <-- IB / RoCE -->   |
   node 0: Ray head + rendezvous  ---- HTTPS ---------+
   nodes 1..5: Ray workers                            |
                                                      v
                                    NC MI355X sampler nodes (vLLM, TP8)
```

- `yeto run up` runs on a control host. It starts the NC sampler pool, waits
  for the first sample group in the store, then starts the trainer on the
  inventory nodes over SSH (`docker run --network host --ipc host` on each
  node), and monitors budget, stalls and starvation.
- The trainer is one 48-GPU island: PP3 x TP8 x CP2 x EP16, DP 1. It reads
  the samplers' groups from the store and writes each policy release back to
  it. The samplers load each release.
- Inter-node traffic: CP all-gather and point-to-point, the EP16 all-to-all,
  and PP point-to-point. There is no data-parallel gradient all-reduce, so
  SHARP is not used.

## 3. Cluster requirements

### 3.1 Hardware and host software

| Item | Requirement | How to check |
|---|---|---|
| Nodes | 6 x HGX B300, 8 GPUs each, x86_64 | `nvidia-smi -L` |
| OS | Ubuntu 22.04 or 24.04 (other distributions work if the items below hold) | `cat /etc/os-release` |
| Kernel | 5.12 or later for DMA-BUF GPUDirect RDMA; older kernels need `nvidia-peermem` loaded | `uname -r`; `lsmod \| grep peermem` |
| NVIDIA driver | 580 or later, open kernel modules. The image is CUDA 13.0 (`NVIDIA_REQUIRE_CUDA=cuda>=13.0`); no CUDA toolkit is needed on the host | `nvidia-smi` (driver version) |
| NVLink fabric | Fabric Manager (and NVLSM on HGX B300) installed and running, matching the driver version | `systemctl status nvidia-fabricmanager`; `nvidia-smi -q \| grep -A2 Fabric` shows `Completed` |
| GPU memory | `nvidia-smi` reports about 275,000 MiB per GPU (the inventory declares 288 GiB; an 8% tolerance applies) | `nvidia-smi --query-gpu=memory.total --format=csv` |
| Host memory | At least 1.6 TB (1,600 GiB) per node; 2 TB (2,048 GiB) recommended. The optimizer state is offloaded to host memory (`--rl-optimizer-offload-fraction 1.0`); the fitted estimate for this layout is about 1,550 GiB per node at 262k, and the 64-expert run used 1.9 TB of 2 TB | `free -g` |
| `/dev/shm` | Leave it at the distribution default (half of RAM). The containers share the host IPC namespace (`--ipc host`), so Ray's object store and NCCL use the host `/dev/shm` | `df -h /dev/shm` |
| Local NVMe | At least 6 TB per node, one file system mounted at the same path on every node (the template uses `/mnt/nvme`). Holds the weights (1.56 TB), the store stage (releases about 207 GB each; a round cut, when enabled, about 0.9 TB per node), the Docker / containerd image store (about 100 GB unpacked), Ray spill and logs | `df -h /mnt/nvme` |

### 3.2 Docker

| Item | Requirement |
|---|---|
| Docker Engine | 24 or later; 29 is fine with the rebuilt image (section 5.1). The SSH user runs `docker` without `sudo` (member of the `docker` group). |
| Storage | `overlay2` graph driver or the containerd `overlayfs` snapshotter. Put the data root on NVMe. With the containerd image store (the Docker 29 default), `/var/lib/containerd` does **not** follow Docker's `data-root`: bind-mount it to NVMe as well, or the 24 GB image fills the root disk. |
| NVIDIA Container Toolkit | Installed and configured for Docker, so `docker run --gpus all` works (CDI devices also work). |
| Locked memory | Start `dockerd` with `LimitMEMLOCK=infinity` (systemd drop-in). The inventory also passes `--ulimit memlock=-1:-1`; NCCL registers its RDMA buffers with `ibv_reg_mr` and fails or falls back without it. `--privileged` alone does not lift the limit. |
| Check | `docker info` (Storage Driver, Docker Root Dir), `docker run --rm --gpus all <image> nvidia-smi -L` |

### 3.3 Network

| Item | Requirement |
|---|---|
| Compute fabric | One 400 Gb/s or faster ConnectX-7 / ConnectX-8 port per GPU (8 per node), rail-optimized. InfiniBand (NDR / XDR) or RoCE v2. For RoCE, a lossless configuration (PFC and ECN) on the traffic class NCCL uses, and the GID index of the RoCE v2 address. |
| Verbs devices | `/dev/infiniband` present on the host; `ibv_devinfo -l` lists the compute HCAs (for example `mlx5_0` .. `mlx5_7`); ports `PORT_ACTIVE`. Storage or management HCAs are left out of `NCCL_IB_HCA`. |
| Cluster network | One IP per node on an interface every node reaches (the inventory `address` and `interface`). All TCP ports open between the six nodes on that network: Ray GCS 6379, Ray dashboard 8265, Ray worker ports, the NCCL bootstrap (ephemeral ports) and the torch rendezvous (29500). Do not expose these ports outside the cluster. |
| Control host to nodes | SSH (port 22, or through a jump host: `proxy_jump`), key login, host keys already in `known_hosts` (yeto runs `ssh -o BatchMode=yes`). |
| Egress from every node | HTTPS to `ghcr.io` (the 24 GB image, first pull) and to the object store endpoint. Per node: the weights once (1.56 TB, unless a shared file system holds them), the sample groups the trainer reads (up to about 10 GiB per group at full 262k length), and the release uploads (about 207 GB per release, split across the ranks). Measured object-store upload rate on a previous cluster: about 270 MB/s; 1 GB/s or more is recommended. A proxy must allow large objects and long-lived connections. |
| Egress from the control host | The object store, National Compute, the sandbox host, GitHub / PyPI for installing yeto. |

### 3.4 Information to send before bring-up

Collect this from every node and share it with the yeto team:

```
uname -r; cat /etc/os-release | head -2; ulimit -l
nvidia-smi; nvidia-smi topo -m; nvidia-smi -q | grep -i -A3 fabric
systemctl is-active nvidia-fabricmanager
ibv_devinfo; ibstat
docker info; docker version
free -g; df -h /mnt/nvme /var/lib/docker /var/lib/containerd /dev/shm
ip -br addr
```

For RoCE also the output of `show_gids` (or `ibv_devinfo -v`), and the PFC /
ECN and traffic class settings. Add the switch model and topology (number of
tiers, rail alignment, oversubscription).

## 4. Node inventory

The inventory is a YAML file passed to the trainer with
`--rl-node-inventory`. Every field is checked before any SSH connection
(`yeto launch --dry-run` reports the first problem). Start from
[`examples/k31/inventory.b300-6x8-baremetal.yaml`](../examples/k31/inventory.b300-6x8-baremetal.yaml):
it describes one node in full and derives the other five from it with a YAML
merge key, changing only `name`, `host` and `address`.

| Field | Value for this cluster | Notes |
|---|---|---|
| `images.x86_64` | Optional | The run config's `--rl-image` decides the image (section 5); an inventory image must name the same one. |
| `name` | `b300-0` .. `b300-5` | Unique. Node 0 (the first entry) is the Ray head and the rendezvous node. |
| `host` | SSH target, `user@ip` or an alias in the control host's SSH config | Unique. |
| `proxy_jump` | Optional jump host | Passed as `ssh -o ProxyJump=...`. |
| `address` | The node's IP on the cluster network | The other nodes dial node 0 at this address; Ray workers register with it (not with a guessed management IP). |
| `interface` | NIC name that carries `address` (for example `bond0`) | Sets `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`. |
| `device_family`, `cpu_arch`, `card`, `cards_per_node`, `hbm_gib` | `cuda`, `x86_64`, `B300`, `8`, `288` | The preflight compares them with `uname -m` and `nvidia-smi` on every node. |
| `host_memory_gib` | Installed memory, for example `2048` | |
| `fabric` | `ib` or `roce` | Sets `NCCL_IB_DISABLE=0` for both. |
| `runtime` | `docker` | |
| `docker_args` | `--gpus all --device /dev/infiniband --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK` | Refused if the GPUs are missing, or (multi-node ib / roce) the verbs devices or the unlimited memlock. `--network host --ipc host` are always added. |
| `scratch` | An NVMe directory, for example `/mnt/nvme/yeto` | See below. |
| `scratch_min_free_gib` | For example `2000` | The trainer preflight refuses a node with less free space under `scratch`. |
| `store_env_file` | Optional: an owner-only (`0600`) env file on the node with the store credential | When every trainer node has one, the control host ships no store credential to the nodes. |
| `mounts` | Host path to container path | The weights copy (section 6). |
| `env` | Per-node environment | See below. |

**Scratch.** The container's root file system lives in Docker's image store.
Everything large goes to `<scratch>/<run>` on NVMe instead, mounted into the
container as `/yeto-scratch`: the store stage (cut shards and releases before
their upload), the release staging (one mount with the stage, so a release is
hard linked, not copied), temporary files, the init release (reused by the
next start) and the Ray session directory. Point Ray spill there too
(`--rl-island-ray-spill-dir /yeto-scratch/ray-spill`). The weights copy is a
plain mount (`/mnt/nvme/models: /data/models`).

**Code.** The yeto code reaches the nodes as the run's code pack: the control
host uploads it to the s3 store, every node downloads it, checks its sha256 and
runs from it. No yeto checkout is needed on the nodes
(`--rl-ssh-trainer-workdir` is only for runs without an s3 store).

**Environment.** The inventory `env` holds the site's fabric settings; it is
exported in every node's container after the fabric defaults. The model and
run settings, which a Modal run passes with `--modal-env`, go in the run
config as `--island-env KEY=VALUE` (one copy for all nodes); they are exported
after the inventory `env`.

| Variable | Value | Why |
|---|---|---|
| `NCCL_IB_HCA` | `=mlx5_0,...,mlx5_7` | Only the compute HCAs. The leading `=` makes the names exact. |
| `NCCL_IB_GID_INDEX` | RoCE only, usually `3` | The RoCE v2 GID of the cluster address (`show_gids`). Wrong value: NCCL cannot connect, or uses a RoCE v1 GID. |
| `NCCL_IB_TC` | RoCE only | The lossless traffic class of your switches. |
| `NCCL_SOCKET_IFNAME` | Set from `interface` | Override here only if NCCL's bootstrap must use another NIC. |
| `NCCL_RAS_ENABLE` | `0` (`--island-env`) | As in the measured runs. The rows below are `--island-env` too. |
| `OMP_NUM_THREADS` | `8` | |
| `MILES_MXFP4_QAT_TRITON` | `1` | MXFP4 QAT kernels for the routed experts. |
| `MILES_K31_INDEXER_QUERY_CHUNK` | `1024` | DSA indexer query block (stage-0 memory). |
| `MILES_K3_ATTN_RES_TOKEN_CHUNK` | `2048` | Attention-residual token block. |
| `YETO_SAMPLE_BUFFER_MAX_INFLIGHT_BYTES` | `64GiB` | Bounded read-ahead of 262k groups. |
| `YETO_RL_NAN_SKIP` | `max_consecutive=2,max_fraction=0.05,min_rounds=20` | A non-finite step becomes a counted skip. |
| `YETO_MILES_TRAIN_EXTRA_ARGV` | `--log-probs-chunk-size 16384 --no-check-for-nan-in-loss-and-grad --custom-megatron-before-train-step-hook-path yeto.rl.adapters.miles.models.kimi_k31.first_stage_grad.before_train_step` | Log-prob chunking, NaN skip, the stage-0 gradient hook. No split flags: 93 layers split 31/31/31 by default (section 8). |
| `PYTORCH_CUDA_ALLOC_CONF` | Not set by default; `expandable_segments:True` when memory is short (section 8) | |
| `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` | `INFO`, `INIT,NET` for bring-up only | Shows which transport NCCL picked (section 12). yeto sets `NCCL_DEBUG=WARN` otherwise. |

The variable `YETO_DEVICE_FAMILY` is refused in `env`: `device_family`
decides it.

Check the file without contacting any node:

```
python -c "from yeto.launch.node_inventory import load_inventory as l; print(len(l('inventory.yaml')))"
```

## 5. Trainer image

### 5.1 Use the overlay-safe image

Use the full-parameter K3.1 trainer default, next9:

```
ghcr.io/agentenv/yeto-miles-k31@sha256:3a0c544558b67c587f460d1ab87c0c19fa1d578d5d77a9346a1729113a3b0225
```

Put it in `--rl-image docker:<ref>` of the run config (it is also what an
omitted `--rl-image` resolves to for `--rl-model-recipe kimi_k31 --tuning
full`). It carries Miles `f61eb35d` (the trainer MFU work behind switches that
default off; [KIMI_K31_ASYNC_RL_GUIDE.md](KIMI_K31_ASYNC_RL_GUIDE.md) section
9.1.1) on the overlay-safe base, so no repack is needed.

The repack of the former Modal default `2de5d796...`,
`yeto-miles-k31@sha256:2abbbb00...`, holds the same files and the same Miles
commit (`1f29e43043`) as that image; only one layer is packed differently. An
SSH trainer island given `2de5d796...` (by `--rl-image` or by the inventory)
runs this repack instead and says so on stderr. The same holds for the
LoRA / generic default `028e505d...` and its repack `32626906...` (Miles
FULL_D `9fa600b9`), on ssh nodes and on sky VMs. The previous full default
`fa70fe28...` (Miles `aa6b6ff5`) has no overlay-safe repack.

Why: the earlier build of the image
(`ghcr.io/agentenv/yeto-miles-k31@sha256:2de5d796...`) has one layer that
deletes a directory (a whiteout) and recreates it in the same layer. Docker's
`overlay2` driver, the containerd overlayfs snapshotter and podman all fail
to unpack it, on any Docker version:

```
failed to register layer: openat .../sglang-...dist-info/INSTALLER: not a directory
mkparent .../sglang-...dist-info: not a directory
```

The image build (`docker/miles-ports/make_layer.py`) now resolves such
whiteouts before it writes a layer.

### 5.2 Pre-pull

The trainer preflight runs `docker pull -q <image>` on every node (timeout
1 h) before it starts anything, and reports a failed pull with Docker's
message. Pull once by hand during bring-up so the first launch does not wait
for 24 GB per node:

```
docker pull <image>          # on every node; about 24 GB compressed, 85-100 GB in the image store
```

`yeto preflight` (section 9) pulls the image on every node and probes it
with the node's `docker_args`; the trainer preflight pulls it again before
the first start.

## 6. Model weights

The trainer loads `moonshotai/Kimi-K3.1-Pro` (MXFP4, 1.56 TB) at the pinned
`--model-revision` from a local copy on every node. The trainer preflight
makes that copy itself, on all nodes at once, before the first start:

```
--model-store local:///data/models                              # container path, under an inventory mount
--model-store-source s3://<bucket>/models/Kimi-K3.1-Pro         # or a directory with yeto-weights-manifest.json
--model-store-shards auto                                       # auto | all | stage
```

- `local:///data/models` names the container path; the node's `mounts`
  map it to NVMe (template: host `/mnt/nvme/models`). A path no mount covers
  is refused before any SSH.
- With `--rl-sample-role trainer` and PP > 1 (`auto`), each node copies only
  the safetensors files of its pipeline stage plus the shared files (about
  0.5 TB instead of 1.56 TB per node; 3.15 TB instead of 9.4 TB for six
  nodes). Stage `s` runs on the two nodes with the `2s`-th and `2s+1`-th
  smallest cluster `address`.
- Every file is checked against its sha256 (the source manifest, else the
  object's `x-amz-meta-sha256`) before it enters the snapshot; a restart skips
  the verified files in seconds. The node refuses to start the copy with less
  than 1.2 times the missing bytes free.
- The store credential is the run's (`YETO_RL_STORE_S3_*`) unless
  `YETO_WEIGHTS_S3_*` names another; the control host's own `AWS_*` never
  reaches a node.
- With the copy in place the trainer reads it with `local_files_only` and
  never contacts the Hugging Face Hub.

Time per node at 1 GB/s: about 9.5 min for a stage, 26 min for the full
model. Size the site egress for six parallel copies.

## 7. Expert selection file

The trainer trains 128 of the 896 routed experts in each of the 92 MoE
layers. `--rl-expert-selection` names the JSON file that lists them
(format: [K31_FULL_262K_LAYOUT.md, section 3](K31_FULL_262K_LAYOUT.md#3-expert-selection-file)).
The same file's sha256 goes into the sampler spec's
`expert_selection_sha256`. Changing the trainable set later starts a new
run, so settle it before the formal run.

**Recommended file.** Use the routing-frequency selection we provide with
the delivery (128 per layer, built with `--ep-size 16` from the routed-expert
counts of an agentic rollout; selection sha256 `880eeec8...`,
trainable_sha256 `4b7c5881...`). It covers about 39% of routed tokens on
agentic workloads, against 14% for a uniform selection. Validate it:

```
python -m yeto.rl.adapters.miles.models.partial_experts.selection --validate <selection.ep16.json>
```

The output must show `"layers": 92` and `"ep_size": 16`. A file built for
another EP size is refused at launch; re-derive it with
`--input <file> --ep-size 16 --output <selection.ep16.json>`.

**Recheck it with your formal rollout.** The recommended file comes from a
different task mix. Before the formal run, compare it with the routing of
your own tasks:

1. Generate a few dozen groups of the formal task set with R3 on (the kit's
   sampler spec has `rollout_fields.routed_experts: true`): for example a
   short rehearsal run (section 9.4) under its own run name and store prefix.
2. Copy a sample of the groups from the store. Each group carries
   `rollout_routed_experts`; one full-length group can be about 10 GiB, so
   use a machine with enough memory and disk:

   ```
   aws s3 ls s3://<bucket>/runs/<rehearsal-run>/sample-buffer/payloads/ --endpoint-url <store-endpoint> | head
   aws s3 cp --recursive s3://<bucket>/runs/<rehearsal-run>/sample-buffer/payloads/ formal-payloads/ \
     --endpoint-url <store-endpoint> --exclude "*" --include "<first group names>"
   ```

3. Build a selection from them and read its coverage:

   ```
   python -m yeto.rl.adapters.miles.models.partial_experts.selection --model kimi_k31 \
     --num-experts 896 --num-trainable 128 --ep-size 16 --moe-layer-offset 1 \
     --payload formal-payloads/* --output selection.formal.ep16.json
   ```

   The summary on stderr has `coverage_mean` and `coverage_min`.
4. Measure the recommended file on the same counts:

   ```
   python - <<'EOF'
   import glob
   from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection, counts_from_payloads
   counts, tokens = counts_from_payloads(glob.glob("formal-payloads/*"), num_experts=896, moe_layer_offset=1)
   cov = ExpertSelection.load_file("<selection.ep16.json>").coverage(counts)
   print(tokens, round(cov["mean"], 4), round(cov["min"], 4))
   EOF
   ```

5. If the formal selection covers clearly more (for example 5 points or more
   on `coverage_mean`), use it for the formal run: set it as
   `--rl-expert-selection`, update `expert_selection_sha256` in the sampler
   spec, and validate it as above.

For a first smoke test before any rollout exists, a uniform selection works:
`--mode uniform --num-moe-layers 92` with the flags of step 3.

## 8. Parallel layout and memory

| Item | Value |
|---|---|
| Parallelism | PP3 x TP8 x CP2 x EP16, DP 1, sequence parallel. `--rl-allow-cross-node-tp` is required: the TP x CP group (16 GPUs) spans two nodes. |
| Pipeline split | **31/31/31.** 93 decoder layers divide evenly over PP3, so no split flag is passed. |
| Rank placement | Megatron's rank order is tp-cp-ep-dp-pp: stage 0 is ranks 0-15 (nodes 0 and 1), stage 1 ranks 16-31 (nodes 2 and 3), stage 2 ranks 32-47 (nodes 4 and 5). Each TP group stays inside one node; CP pairs and EP16 cross between the two nodes of a stage. |
| Context | `--seq-len 262144`, `--rl-max-tokens-per-gpu 131072` (= seq-len / CP). |
| Microbatches | 8 prompts x 8 samples = 64 samples per step, one per microbatch at full length (m = 64). The log line `num_microbatches=[m]` shows the actual value. |
| Distributed timeout | `--rl-distributed-timeout-minutes 60`. The first step's JIT and autotune took 662 s at 524k. |
| Optimizer | Fully offloaded to host memory (`--rl-optimizer-offload-fraction 1.0`). |

**Why 31/31/31.** The refined memory model fitted to the d1 and 524k
measurements puts the torch-allocated peak per stage at about 157 / 157 /
147 GiB (median) and 167 GiB (conservative) for 31/31/31, below
32/29/32 on stage 0 (160 / 169 GiB), and the even split is about 3% faster
once the indexer runs TP-split [estimate]. The highest allocated peak
observed to pass on a B300 without expandable segments is 166.3 GiB, so the
headroom is about +9 GiB (median) to -1 GiB (conservative) [estimate]. This
is tighter than the earlier 67 GiB estimate in
[K31_FULL_262K_LAYOUT.md](K31_FULL_262K_LAYOUT.md#8-current-scope-and-recommended-practice),
which did not model the per-layer recompute working set.

Watch the first step closely (section 11) and, if a stage runs short:

1. **Expandable segments.** Add `PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True`
   to the inventory `env`. It removes most fragmentation (the allocated
   budget rises to about 213 GiB [estimate]) and is accepted on this trainer
   (`--rl-sample-role trainer`, no engine, no `--offload-train`). It was on in
   the 524k measurement.
2. **Chunk sizes** (stage 0 first): `MILES_K31_INDEXER_QUERY_CHUNK` 1024 to
   512, `MILES_K3_ATTN_RES_TOKEN_CHUNK` 2048 to 1024.
3. **Log-prob phase**: `--log-probs-chunk-size 4096` in
   `YETO_MILES_TRAIN_EXTRA_ARGV`.
4. **Move a layer.** The most memory-safe split is 31/30/32
   (`--decoder-first-pipeline-num-layers 31 --decoder-last-pipeline-num-layers 32`
   in `YETO_MILES_TRAIN_EXTRA_ARGV`): no stage needs expandable segments in
   either estimate, at the speed of 32/29/32. Each layer is about 2.8 GiB.
   The launcher checks any split before a GPU starts.
5. If memory is still short, use a shorter context.

Treat less than 15 GiB of `nvidia-smi` headroom on any GPU as a stop signal.

## 9. Bring-up order

Run the steps in order. Do not start the next one until the current one
passes.

### 9.1 One node

On `b300-0`:

1. Host checks of section 3.4: driver, Fabric Manager `active`, all eight
   GPUs, HCAs `PORT_ACTIVE`, `ulimit -l` unlimited, Docker root on NVMe.
2. Pull the image (section 5.2) and start it with the inventory's Docker
   flags:

   ```
   docker run --rm --network host --ipc host --gpus all --device /dev/infiniband \
     --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK <image> \
     bash -c 'nvidia-smi -L; ibstat -l; ulimit -l; all_reduce_perf -b 1G -e 8G -f 2 -g 8'
   ```

   Expect 8 GPUs, the compute HCAs, `unlimited`, and a stable NVLink
   all-reduce bus bandwidth (record it as the single-node baseline). The
   image ships `all_reduce_perf`, `ib_write_bw` and `ibstat`.
   `all_reduce_perf` links the system NCCL (2.28.3), the trainer uses
   torch's NCCL (2.30.7): for anything across nodes, trust `yeto preflight`.
3. Node checks and TCP ports over the whole inventory, no bandwidth
   benchmark. Host and container side: driver, GPUs, NVSwitch fabric, NVLink
   matrix, GPU-NIC affinity, HCAs and ports, MTU, GID index,
   `/dev/infiniband` and memlock in the container, memory, disks, Docker
   storage driver, the image pull, clock, addresses, and the Ray and torch
   ports between nodes:

   ```
   yeto preflight --gpu ssh:6x8xb300 --rl-node-inventory inventory.yaml --cluster-prefix <run> --no-bench
   ```

   It writes a PASS / WARN / FAIL table with a fix per finding to
   `~/.yeto/runs/<run>/preflight/<time>/` and exits 1 on any FAIL. It checks
   the image the trainer runs: `--image` (the run's `--rl-image`) else the
   inventory's, an overlay-rejected pin as its repack (section 5.1).
   `yeto launch` and `yeto run up` run this same check automatically before
   an ssh trainer starts (section 10.3); `--no-ports` also skips the port
   check.

### 9.2 Two nodes

Between two nodes that will hold one pipeline stage (`b300-0` and `b300-1`):

1. RDMA per rail between every pair of nodes:

   ```
   yeto preflight --gpu ssh:6x8xb300 --rl-node-inventory inventory.yaml --cluster-prefix <run> --no-nccl
   ```

   For each node pair it runs `ib_write_bw` on all rails at once (HCA i to
   HCA i, eight in parallel), inside the trainer image with the inventory's
   Docker flags. On RoCE the GID index is `NCCL_IB_GID_INDEX`, else the
   node's RoCE v2 IPv4 GID. Six nodes are 15 pairs in 5 rounds. Each pair
   reports GB/s per rail and its slowest rail as a share of the link rate:
   80% or more PASS, 50-80% WARN, below 50% or no connection FAIL. Add
   `--ib-pairs ring` to test only neighbouring pairs.
2. The NCCL check: `yeto preflight` with no `--no-*` flag runs everything:
   TCP reachability of the Ray and torch ports, `ib_write_bw` on every node
   pair, `all_reduce` / `all_to_all` / `sendrecv` on disjoint node pairs
   (0+1, 2+3, 4+5), then the same collectives over all six nodes. It fails
   when any rank reports `NET/Socket` instead of `NET/IB`, and warns on a
   bus bandwidth below the rail count's expectation. This is the check that
   catches a silent TCP fallback before a 48-GPU launch.

### 9.3 Six nodes, dry run

1. All six nodes pass 9.1; every node pair passes the `ib_write_bw` check
   of 9.2 (this covers the pairs across stages; the node-pair NCCL groups
   are 0+1, 2+3, 4+5, and the six-node round covers the stage boundaries).
2. The image is pulled on every node, and the weights are copied and verified
   (section 6) on every node.
3. The full `yeto preflight` of 9.2 step 2 (it ends with the six-node
   collectives) passes with no FAIL.
4. Fill the run config and run the preflight (nothing is ordered, no trainer
   starts):

   ```
   yeto run up --config run.json --dry-run
   ```

   Read `preflight.log` (guide section 4.1). It must end with
   `dry run: preflight ok, nothing ordered`; the trainer part reports
   `"total_gpus": 48` and one island `"cloud": "ssh"`, `"num_nodes": 6`.
   A dry run does not run the full cluster preflight; it logs
   `would run the trainer's cluster preflight: yeto preflight ...`. The real
   `yeto run up` runs it before anything is ordered. The dry run does check,
   over SSH, that every trainer node reaches the run store's endpoint and the
   trainer image's registry: one `[PASS]` or `[FAIL]` line per node and
   endpoint (any HTTP answer counts, through the node's proxy too); on a
   `[FAIL]` it logs `cluster preflight failed: nothing ordered` and exits 2.
   `--skip-preflight` skips this check with the rest of the preflight.

### 9.4 Rehearsal, then the formal run

1. **Rehearsal.** Copy the run config, give it its own `run` name and store
   prefix, and set `--total-steps 2`. Launch it (`yeto run up --config
   rehearsal.json`) and check the first steps as in section 11: memory per
   stage, `num_microbatches=[64]`, stage-0 gradients, finite loss, one release
   reaching the samplers. Use its groups for the selection recheck
   (section 7).
2. **Formal run.** `yeto run up --config run.json`.

## 10. Launch

### 10.1 Control host

Install yeto as in the guide (section 2), with the launcher extras:
`uv pip install -p .venv/bin/python -e ".[launcher]"`. The control host also
needs the SSH keys of the inventory, the AWS CLI, and the NC tool.

### 10.2 Run config

Start from
[`examples/k31/run.baremetal-b300-262k.json`](../examples/k31/run.baremetal-b300-262k.json).
It is the Modal 262k kit (`run.agentic-262k.json`) with the trainer moved to
the inventory. Differences:

| Setting | Bare metal | Why |
|---|---|---|
| `--gpu` | `ssh:6x8xb300` | The inventory's six trainer nodes. |
| `--rl-node-inventory` | `<inventory.b300-6x8-baremetal.yaml>` | Section 4. |
| `--island-env` | The model and run settings (section 4) | One copy for all nodes. |
| `--modal-*` flags | none | Their settings are `--island-env`; host memory is the node's own. |
| `env.YETO_RL_STORE_S3_PROFILE` | An AWS profile on the control host | Its key, limited to the store's bucket, is handed to the trainer containers (the trainer never uses the control host's other credentials). `YETO_RL_STORE_S3_ACCESS_KEY_ID` / `..._SECRET_ACCESS_KEY` also work. Or give every node a `store_env_file` (section 4). Every node stages the store under its scratch and uploads by object API. |
| `--rl-island-ray-spill-dir` | `/yeto-scratch/ray-spill` | Ray spill on the node's scratch (NVMe). |
| `--idle-warn-minutes` | `0` | Idle shutdown off: at 262k a step (about 25-35 min at m = 64 [estimate]) or a wait for groups can exceed the default 15 + 15 min idle window. Hang detection stays on (`--rl-stall-timeout`, the run monitor). On paid hardware use `--idle-warn-minutes 45 --idle-stop-minutes 45`, as the Modal kit does. |
| `trainer.usd_per_h` | `0` | Your own hardware. Set an internal rate to have it counted in `budget.usd`. |

Fill every `<...>` value. The ones that must agree across the trainer, the
pool and the sampler spec are listed in the guide's checklist (section 3.7);
the expert selection and its sha are in section 7 here.

### 10.3 Start and stop

```
yeto run up --config run.json --dry-run     # preflight only
yeto run up --config run.json               # pool, first group, trainer, dashboard, monitor
yeto run status --config run.json
yeto run down --config run.json             # trainer, pool and sandboxes
```

Before anything is ordered, `yeto run up` runs the cluster preflight of
section 9.1 step 3 (node checks and TCP ports) on the trainer's nodes, with
the trainer's `--rl-image`; on a FAIL it logs
`cluster preflight failed: nothing ordered` and exits 2. The trainer's own
`yeto launch` then gets `--skip-preflight`, so the check runs once. A direct
`yeto launch` with an `ssh:` island runs the same check and refuses to start
on a FAIL (warnings are printed and the launch goes on). `--skip-preflight`
on either command skips it. `yeto run up` (the dry run too) also checks over
SSH that every trainer node reaches the run store's endpoint and the image's
registry, one `[PASS]` or `[FAIL]` line per node and endpoint, and refuses the
same way on a `[FAIL]`; `--skip-preflight` skips that check as well.

The trainer is a resident SSH island: `yeto launch` starts one container per
node, follows every node, and treats any rank that exits non-zero, a process
that vanishes, or a node that stops answering SSH as a failure of the whole
island; it then stops every node (and confirms it) before it starts the next
incarnation. A restarted launcher reattaches to a running incarnation instead
of starting a second one. `yeto run down`, a stop and the end of the run stop
the containers node by node; `yeto cleanup` stops them from the resource
ledger if the launcher itself died. The expert selection and the algorithm
spec travel in the start script.

A relaunch continues from the newest release in the store with a fresh
optimizer (`--rl-round-cuts off`); budget about 20 minutes of initialization
for every start.

## 11. Monitoring

| Where | What to watch |
|---|---|
| Dashboard (`yeto run up` serves it on `dashboard.port`) | Steps, releases, sampler health, buffer depth, starvation. |
| `yeto run status --config run.json` | Process states, uptime, spend. |
| Node 0 log on the control host | Streamed by the SSH session (trainer events). The other nodes' logs are copied as `<island>-i<n>-node<rank>.log` into the run's events directory. |
| On each node | `~/.yeto/trainer-islands/<run>/<island>-i<n>/trainer.log` (log), `.../rc` (exit code); container `yeto-<run>-<island>-i<n>-n<rank>` (`docker ps`, `docker logs`). |
| GPU memory | `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 30` on one node per stage (nodes 0, 2, 4). |

First step (rehearsal and formal run):

1. `num_microbatches=[64]` in the log.
2. Per stage: torch `max_memory_reserved` / `max_memory_allocated`, and the
   `nvidia-smi` peak. Less than 15 GiB of headroom on any GPU: stop and apply
   section 8.
3. The trainable experts of stage-0 layers have non-zero gradient norms
   ([K31_FULL_262K_LAYOUT.md, section 9](K31_FULL_262K_LAYOUT.md#9-validating-on-your-cluster),
   step 3), loss and `grad_norm` are finite, and
   `train_rollout_logprob_abs_diff` stays at the sampler's noise floor.
4. Step time from step 2 on (step 1 includes JIT and autotune).
5. With `NCCL_DEBUG=INFO` set for bring-up: `NET/IB` on every rank. Remove it
   for the formal run.

## 12. Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| `torch.OutOfMemoryError: CUDA out of memory` on some GPUs | A pipeline stage is over its memory | Find the stage from the global rank (stage = rank // 16, section 8). Apply section 8 in order: expandable segments, chunk sizes, log-prob chunk, split 31/30/32. Relaunch. |
| Container exits 137, `Killed`, or `oom-kill` in `dmesg` | Host memory exhausted (optimizer offload, buffer read-ahead) | Check `host_memory_gib` against `free -g`; lower `YETO_SAMPLE_BUFFER_MAX_INFLIGHT_BYTES`; make sure nothing else runs on the node. |
| One rank exits and the whole trainer stops | Any rank's failure ends the incarnation | Read every node's `trainer.log` and find the earliest traceback by timestamp: the first failure is the cause, the others follow. Check `dmesg` for GPU Xid errors and `nvidia-smi -q` for retired pages. Relaunch continues from the newest release. |
| NCCL watchdog timeout after 60 min | A rank stopped (see above) or the fabric hangs | Find the rank that did not enter the collective. On a long first step only, raise `--rl-distributed-timeout-minutes` to 120. |
| Steps far slower than expected; `NET/Socket` in the NCCL log | NCCL fell back to TCP sockets | Check: `/dev/infiniband` in the container, `NCCL_IB_HCA` names, `NCCL_IB_GID_INDEX` on RoCE, unlimited memlock, `nvidia-peermem` or DMA-BUF. Rerun the two-node check (9.2). yeto refuses a multi-node ib / roce island without verbs devices or memlock, but cannot see a wrong HCA list or GID index. |
| `failed to register layer: ... not a directory` or `mkparent ...: not a directory` during pull or start | The image with the same-layer whiteout | Use the rebuilt image (section 5.1). |
| `no space left on device` during pull | Image store on the root disk | Move Docker's data root and `/var/lib/containerd` to NVMe (section 3.2). |
| Preflight: `docker pull ... failed` | No route to `ghcr.io`, proxy, or disk | Pull by hand on that node and read Docker's message. |
| `cluster preflight refused the launch: ...` | `yeto launch` / `yeto run up` found a FAIL in the node checks or TCP ports | Read `preflight.txt` in the report directory the message names and apply its FIXES. To launch anyway, add `--skip-preflight`. |
| Preflight refuses `docker_args` | GPU flag, `/dev/infiniband` or memlock missing | Use the inventory template's `docker_args`. |
| Trainer stopped as idle (`idle_warn`, then `idle_shutdown`) | The cost guard's idle rule: no progress event and no loading event for its window. Loading the weights takes about 20 min at 0% GPU; a 262k step or a wait for groups can take longer than 30 min | The template turns idle shutdown off (`--idle-warn-minutes 0`). If you turn it on, set `--load-grace-minutes` (default 45) above the start time including pulls, and `--idle-warn-minutes` above the longest step plus buffer wait. |
| Trainer exits with code 6 | `--rl-stall-timeout` (7200 s): no island event for 2 h | A hang. Read the node logs as for a rank exit. The run monitor's own backstop is `max(4 x --rl-stall-timeout, budget.silence_min)` without log growth. |
| Starvation alert; trainer waits | No new groups in the store | A sampler-side problem; see the guide, section 7. The monitor stops the trainer only when no sampler has been healthy for `starvation.unhealthy_min`. |
| Launch refused: `slot_order was built for ep_size 8, the run uses 16` | Selection built for another EP size | Re-derive with `--input <file> --ep-size 16` (section 7). |
| Preflight: store credentials missing | No `YETO_RL_STORE_S3_PROFILE` or key pair on the control host | Set it in the run config `env` (section 10.2). |
| Preflight: `uname -m reports ...` or a card mismatch | The inventory does not match the node | Fix the inventory entry; the preflight compares `cpu_arch`, `card`, `cards_per_node` and `hbm_gib` with the node. |
