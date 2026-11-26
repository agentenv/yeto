# Own clusters

A batch of your own bare-metal machines joins yeto as an SSH node inventory
(`yeto/launch/node_inventory.py`; the trainer's `--rl-node-inventory` format).
The same machines can be the trainer, the sampler pool, or both, in one run
config driven by `yeto run up`: the same control plane, breakers and budget as
any cloud run. Own nodes bill nothing; the run's spend counts the cloud parts
only.

The templates are in `examples/own-cluster/`:

| file | what it is |
|---|---|
| `inventory.own-nodes.yaml` | one 8 x B300 trainer node (`roles: [trainer]`; add nodes like it for a larger trainer) and three 8 x H100 sampler nodes (`roles: [sampler]`) |
| `run.own-trainer-sky-sampler.json` | own trainer, sky spot L40S samplers |
| `run.own-trainer-modal-sampler.json` | own trainer, Modal L40S samplers |
| `run.own-sampler-pool.json` | own sampler pool, Modal trainer |
| `run.fleet-own-and-cloud.json` | a fleet of one own island and one cloud island |
| `sampler-spec.qwen3-4b-vllm*.json` | the Qwen3-4B vLLM sampler of each route (sky, Modal, own nodes) |

Every value in angle brackets is yours to fill in: `<run>`, `<bucket>`,
`<store-endpoint>`, `<store.env>` (the samplers' store env file, mode 0600,
with `R2_BUCKET` and the endpoint), `<model-revision>`, `<prompts.jsonl>`, the
sky ssh key, the Modal secret and volume, the sampler image, and the paths of
the other template files. `tests/test_own_cluster_templates.py` fills them with
test values and runs every template through `yeto run up --dry-run`.

## Trainer with cloud samplers

The trainer runs on your nodes (`--gpu ssh:<N>x<M>x<card>` with
`--rl-node-inventory`); the sampler pools come from any registered source: sky
spot VMs, Modal function calls, National Compute, or other own nodes over SSH.
The samplers and the trainer meet in the run store: the pools write sample
groups there and read the trainer's policy releases from it.

```bash
yeto run up --config run.own-trainer-sky-sampler.json --dry-run
yeto run up --config run.own-trainer-sky-sampler.json
```

The dry run orders nothing and contacts no node. It runs:

- each pool's own dry run (`python -m yeto.launch.nc pool --dry-run`): the
  offers it would order with their prices, the sampler spec checks, the
  model in the bucket;
- the trainer's launch dry run (`yeto launch --dry-run`): the node layout and
  ranks on the inventory nodes;
- the checks between the pools and the trainer: the rollout fields the
  samplers record against the ones the trainer trains on (`top_logprobs_num`,
  routed experts), and, in a fleet, the release form each pool's engine loads
  (weight format, reload mode) and the trainer shape of every island.

The real `yeto run up` then runs the trainer's cluster preflight and the
**egress check of your nodes** before it orders anything (any pool: National
Compute, sky, Modal). `--dry-run --check-nodes` runs the same egress check in
a dry run. Over SSH, on every own node of the run, it checks whether the node
reaches the run store's object API and the trainer image's registry from its
own network:

- The probe is an HTTPS request with `curl` (each attempt bounded by `-m`;
  retried twice on a timeout, a 5xx or a refused connect, not on a DNS
  failure): any HTTP answer counts (401, 403 and 404 included:
  the network path works), and the certificate is not verified, so a proxy
  that intercepts TLS passes.
- It runs with the inventory env exported, so it goes through the node's
  proxy (`HTTPS_PROXY`/`https_proxy` and the other proxy variables there);
  for the registry, when the env sets no proxy, through the docker daemon's
  (`docker info`, bounded by `timeout 5`; what `docker pull` uses). A registry the daemon lists in
  `insecure-registries` (`/etc/docker/daemon.json`), or a loopback one, is
  requested over `http`; a `daemon.json` that does not read means `https`.
- A node without `curl` gets a TCP connect, or, when a proxy is set, a
  `[WARN]` that the path is not checked.
- A store not reached is a `[FAIL]`: the run up orders nothing and names the
  node and the endpoint. A registry not reached is a `[WARN]` (dockerd may pull
  through a proxy of its own, a systemd drop-in, that the probe does not
  see): the image pull reports it if the pull fails. A node that takes no SSH
  fails its `ssh` check. Each FAIL names its own fix (egress, the CA bundle
  for a TLS failure, the clock, the SSH login).
- A registry is checked only for nodes with `runtime: docker` (a `rootfs`
  node pulls no image).

```
cluster preflight: own nodes own-b300-0: checking over ssh that they reach store https://<endpoint>, registry https://<registry>/
cluster preflight:   [FAIL] own-b300-0 reach_store: own-b300-0 cannot reach the store endpoint https://<endpoint> (s3://<bucket>/runs/<run>) (rc 6): http 000 curl: (6) Could not resolve host ... -> open the node's egress to it (firewall, proxy: https_proxy in the node's environment, DNS)
cluster preflight:   [PASS] own-b300-0 reach_registry: the registry endpoint https://<registry>/ (<image>): http 401
cluster preflight:   1 pass, 0 warn, 1 fail, 0 skip
cluster preflight: an own node fails the egress check (the lines above name each node and endpoint): open the node's egress to it (...); nothing is ordered (--skip-egress-check skips this check only)
cluster preflight failed: nothing ordered
```

The store endpoint is the one the launch's store client takes
(`store_client_environ`): the environment's `YETO_RL_STORE_S3_ENDPOINT_URL`,
else `--rl-object-store-endpoint` / `AWS_ENDPOINT_URL`, else the endpoint of
the store profile (`--rl-object-store-profile`), else AWS S3 in the store's
region. The registry is the one of `--rl-image` (else the inventory's image,
else the pinned default; Docker Hub is `registry-1.docker.io`).
`--skip-egress-check` skips this check only; `--skip-preflight` skips every
preflight, this one included.

The check is reachability only: it does not check the store credentials, the
bucket, or the registry login, and it does not reach a registry's blob CDN
(for example the host ghcr.io serves layers from, or Docker Hub's auth and
CDN hosts), so a firewall that allows only the registry host passes it and
fails at the image pull.

The spend line counts the cloud parts only: `trainer.usd_per_h` is 0 for a
trainer on own nodes, and each pool is counted at its node price. It is
logged for a run with own nodes (an own trainer or an own sampler pool) whose
pools are not National Compute ones; a run with an NC pool logs no spend line:

```
dry run: spend estimate $1.50/h = pools $1.50/h (sky-l40s (sky) 2 x $0.7492) + trainer $0/h (own nodes: ssh:1x8xb300); over the 6 h wall clock $8.99 (budget.usd $30)
```

A fleet logs its own estimate instead (`fleet: budget estimate ...`), with
the own island's trainer at `$0`.

Notes for the cloud pools:

- sky: the pool's `model_pull` puts the weights on each VM before its sampler
  starts; on the node runtime (the sky route's default) `dest` is any
  `/models/...` path the spec names (`/models/Qwen3-4B` in the template).
  `prompts_file` copies the prompts to the VM at every start.
- Modal: the model and the prompts are read from the pool's `model_volume`
  (mounted at `/mnt/yeto-models`); set the function's `cpu` and `memory_gib`,
  which Modal bills next to the GPU (the defaults are large).

A fleet of an own island and a cloud island (`run.fleet-own-and-cloud.json`)
is an island fleet as in [MILES_RL.md](MILES_RL.md) ("Island fleet"): island 0
trains on the own nodes, island 1 on Modal, each with its own pools, and one
syncer merges them. The syncer merges islands of one card type (its compat
group) and one trainer shape: the same TP/PP/EP/CP/DP, since the shape-bound
full-parameter layout names every tensor by its rank's parallel position.
Put the cloud island on the same card and the same nodes x GPUs as the own
island (the template: 1 x 8 x B300 on both), or declare one group with
`--rl-compat-group` and set `--rl-sync-layout shape-free` for the whole fleet.
The dry run refuses islands of two shapes, naming both.

## Own sampler pools

Your nodes serve as a sampler pool with `provider.kind: ssh`
(`yeto/launch/ssh_sampler_pool.py`): the pool takes the inventory nodes named in
`provider.nodes` (or every node with `roles: [sampler]`) as its slots, one
sampler per node, and orders nothing.

```bash
yeto run up --config run.own-sampler-pool.json --dry-run
```

- **Node runtime.** An ssh pool runs on the node runtime by default: the
  node runs `python -m yeto.node run` (docs/NODE_RUNTIME.md), which starts the
  sampler; its ready record, heartbeats and exit go to the run store, as on
  every other source. `"node_runtime": false` in the pool's `provider` keeps
  the pool's own sampler start.
- **Spares.** List more nodes than `target`: the template has three nodes for
  two samplers. When a node is lost under its sampler, the pool fences its
  shard and continues it on the spare; the lost node's slot is held for
  `provider.lost_node_hold_s` (default 1800 s) while another node is free.
- **Store.** The samplers write to the run store through `r2_env_file` (the
  store env file the cloud pools use). A node-local store (`provider.store`)
  serves one sampler (`target: 1`).
- **Image.** `provider.container.image` names the sampler image (or `auto`:
  each node's image from the engine image catalog for what the spec asks);
  the dry run refuses a node the image does not serve.
- **Weights.** Put the model and the prompts on every node once (the
  template's sampler nodes mount `/data/models`); the pool does not pull them.
- **Spend.** The pool bills nothing (`price_usd_h` 0 unless you state one), so
  a spend cap (`budget.usd`) counts the other parts only: in the template, the
  Modal trainer.
- **Egress.** `yeto run up` (and a dry run with `--check-nodes`) checks over
  SSH that every pool node reaches the run store and the registry of its
  image, as for an own trainer. The store endpoint is the one the node's
  sampler takes: the `AWS_ENDPOINT_URL` of `r2_env_file` (AWS S3 in the file's
  region when it is empty).

## Node leases

Several runs can share one inventory. Add a `lease_store` at the top of the
inventory, and each node is leased to one run at a time:

```yaml
lease_store: s3://BUCKET/yeto-node-leases/site-a   # or oss://..., or a shared directory
nodes:
  - name: b300-0
    ...
```

- Before an `ssh:` trainer island or an ssh sampler pool starts on a node, it takes
  the node's lease in the store (`nodes/<node>/lease/`). A node another run holds is
  skipped.
- When the inventory has `lease_store`, it may list more trainer nodes than the
  island needs. The launch picks nodes in this order: the nodes this run already
  holds first, then free nodes in inventory order. If too few nodes are free, it
  waits. Every 5 minutes it prints a line with how long it has waited and writes a
  `node_lease_waiting` event. Waiting ends at the run's STOP flag (`yeto stop-run`) or
  after `--rl-node-lease-wait-s` (default 3600 s).
- If the island's jobs journal holds an earlier incarnation, a restarted launch uses
  that incarnation's nodes again. If another run holds one of those nodes, the launch
  is refused; the island never moves to other nodes this way.
- The launch gate's preflight checks the nodes the leases picked. A dry run takes no
  lease: its `ssh_island.node_leases` entry says it checked the first candidates.
- A lease runs for 10 minutes past its last renewal. The launcher renews it from a
  keeper thread and on every supervision tick (the island watch, the pool's ticket
  reads). If the store does not answer past a lease's end, the node is marked
  `unconfirmed` (an alert event) and is never treated as free.
- A lease that ran out, for example because its launcher died, may be taken over by
  another run. Before the takeover, the claimer checks over SSH whether anything of
  the previous run still runs on the node: its trainer containers or process groups,
  or its samplers (by their start marker). If something does, or if the node does not
  answer, the takeover is refused and a `node_lease_blocked` event is written.
  A successful takeover writes `node_lease_taken_over` under `nodes/<node>/events/`.
- A released lease does not prove the node is empty: the next run of another holder
  runs the same check before it claims the node.
- Sampler containers of a pool whose inventory has `lease_store` carry the label
  `yeto.run=<run>`; the check finds them by it (`ps --filter label=yeto.run=<run>`). A
  `yeto-sampler-*` container with no label (started before labels, or by a pool
  without `lease_store`) may be the old run's, and refuses the takeover.
- Each engine is listed as the SSH user and, where `sudo -n` works, through sudo.
  docker has one daemon, so either listing is enough; if neither works the takeover is
  refused. A docker daemon that is not running ("Cannot connect to the Docker daemon")
  counts as no containers; "permission denied" does not. podman's rootless and rootful
  containers are separate: the user's listing must work, and for a pool with
  `container.sudo` the `sudo -n` listing must work too.
- Holders are named `<run>/trainer@<identity>` and `<run>/pool@<identity>`. The
  identity is a nonce made at the run's first launch and kept in the run directory
  (`node-leases.json`; a pool on the store control plane keeps it in the run store).
  Only the same identity takes a running lease back, so two runs with the same name
  on different machines never share a node. A node held by the same run under another
  identity (a run directory restored without `node-leases.json`) is an error at once,
  naming that file: restore it, or end the earlier launch's leases with
  `yeto cleanup --run <run>`.
- A trainer scheduler's island holds its leases for the whole run, across
  incarnations; they are released when the run's loop ends and every incarnation was
  confirmed stopped. When a resident island's release is confirmed, its leases are
  released. A launcher that ends with `--keep`, or whose release was not confirmed,
  keeps the leases without expiry (`kept: true`, `until: null`).

Releasing kept leases: run `yeto cleanup --run <run>`. After it confirmed the stop of
the run's islands on their nodes, it releases the leases those islands kept (only the
ones of that run, and only when kept or run out). A node that stays unreachable keeps
its lease kept; see "Spare nodes" (`yeto cluster release-node`) once that section is
in this guide. Until then, release such a node by hand:

1. Make sure nothing of the run is left on the node: no `yeto-<run>-*` or
   `yeto-sampler-*` containers (`docker ps`, `podman ps`, with `sudo -n` too where
   used), no live process group under `~/.yeto/trainer-islands/<run>/`, no process with
   `yeto-ssh-sampler:<run>:` in its arguments. If the node cannot be reached, make sure
   it is powered off or out of the network.
2. In the lease store, list `nodes/<node>/lease/` and delete the `.lease` object of the
   highest token (`<token>.lease`, eight digits). Keep the `<token>.json` record: the
   next claim takes token + 1.

Clocks: on an `s3://` store, or an `oss://` store the launcher reaches directly,
expiry is computed from the store's own times (the lease object's `LastModified` and
the listing's `Date` header). A directory store, an `oss://` store reached through the
OSS relay, and a listing without a `Date` header give no such times: then the
holder's clock writes the lease end and the reader's clock judges it, so all hosts
must run NTP with clock skew well below the 10-minute lease. The renewal cadence
uses each host's monotonic clock, so a wall clock step does not delay a renewal.

The store must create objects atomically (`If-None-Match: *`); a store found
without it is refused.

Without `lease_store`, nothing is leased and behaviour is as before. An ssh pool's
`--dry-run` notes this once. A trainer dry run notes it only when the inventory
uses another own-cluster field (`fabric_group`).

Known follow-up: the nodes do not renew their own leases. When the launcher is gone,
a node stays out of other runs only while its lease is kept, or while the takeover
check finds the old run's processes on it.

## Fabric groups

A multi-node trainer island needs nodes that reach each other at RDMA rate. Two
nodes on different switches, or a node whose HCA is down, still talk over TCP:
NCCL falls back to `NET/Socket` without failing, and the island runs many times
slower. A fabric group is a set of nodes in which every two nodes reach each
other over IB / RoCE at link rate. `yeto cluster probe` measures the groups and
writes them into the inventory as `fabric_group`; the launcher then keeps every
multi-node island inside one group.

### Probing

<!-- doc-test: probe -->
```bash
yeto cluster probe --inventory nodes.yaml --dry-run    # the plan and its time only: no SSH, no file
yeto cluster probe --inventory nodes.yaml              # measure and print the groups
yeto cluster probe --inventory nodes.yaml --write      # ... and write fabric_group into nodes.yaml
yeto cluster check --inventory nodes.yaml --need 2     # the groups the file declares; where 2 nodes go
```

The probe runs on idle nodes only. It starts the trainer image on every node
and runs benchmarks across the fabric, which would disturb a running job and be
disturbed by it. Before anything else it lists the GPU processes
(`nvidia-smi --query-compute-apps`) and the `yeto-` containers (`docker ps`) of
every node. A busy node, or one whose GPU processes cannot be listed, refuses
the whole probe (exit 3) with the node and what runs there. Probe the idle nodes
with `--nodes`, or pass `--force` to probe anyway. The TCP port check of the
preflight runs as well; a port already in use is a WARN, as in `yeto preflight`.

The probe runs the cross-node steps of `yeto preflight` (it reuses them, it has
no benchmark of its own), inside the trainer image with each node's
`docker_args`:

- `ib_write_bw` between every pair of nodes, rail by rail (HCA i to HCA i, the
  HCAs of the node's `NCCL_IB_HCA`);
- the NCCL collectives on every pair of nodes with `NCCL_DEBUG=INFO`, read for
  the transport NCCL picked.

Two nodes share an edge when all of these hold:

- their `ib_write_bw` check passes against a known link rate (every rail at
  least 80 % of its link rate, `Requirements.ib_pass_fraction`); an unknown link
  rate is a WARN and no edge;
- their NCCL run goes over `NET/IB` (without GPUDirect RDMA still counts: the
  transport is IB);
- neither node failed an HCA check of the preflight (`ib_ports`, `ib_hca`, `gid`,
  `dev_infiniband`), and each has as many active rails as HCAs it declares
  (`NCCL_IB_HCA`, else every port of its fabric's link layer).

A group must be a clique, so each connected component is cut until it is one.
The node with the most missing edges leaves (ties: the last by name) until the
rest are all connected to each other. The nodes that left are grouped again among
themselves. The probe prints every missing edge inside a component. A node left
alone is `fabric unavailable`, with its reasons: the failed node check, the pairs
whose `ib_write_bw` failed, or whose NCCL fell back to `NET/Socket`.

```text
fabric group fg-3f9a0c12: b300-0, b300-1 (2 node(s))
missing edge b300-0<->b300-2: NCCL: NCCL runs over NET/Socket (TCP), not NET/IB
fabric unavailable: b300-2: to b300-0: NCCL: NCCL runs over NET/Socket (TCP), not NET/IB
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | every probed node is in a group and no edge is missing |
| 1 | a node is fabric unavailable, an edge is missing, or the deadline cut the probe short |
| 2 | a usage or inventory error |
| 3 | a node is busy |

| Flag | Default | Description |
|---|---|---|
| `--inventory` | required | the SSH node inventory (YAML / JSON) |
| `--nodes` | every node | probe only these nodes (comma-separated names; CUDA nodes only) |
| `--image` | the inventory's | the trainer image (`ib_write_bw`, torch with NCCL) |
| `--no-pull` | off | do not `docker pull` the image |
| `--nccl-pairs` | `all` | NCCL on every pair, or on disjoint pairs only (see below) |
| `--ib-seconds`, `--nccl-sizes`, `--nccl-iters` | `3`, `16M,128M`, `10` | benchmark length |
| `--max-parallel` | `16` | nodes, or node pairs, probed at once (at least 2; a pair runs both ends together) |
| `--deadline` | `3600` | seconds for the whole probe; a probe it cuts short writes nothing and exits 1 |
| `--force` | off | probe busy nodes too |
| `--write` | off | write `fabric_group` of every probed node into the file |
| `--dry-run` | off | print the plan and an estimate of its time; no SSH |
| `--json` | off | print groups, unavailable nodes, edges, missing edges and pair problems as JSON |

`--nccl-pairs disjoint` runs NCCL only on disjoint pairs (0+1, 2+3, ...). A pair
without its own run then counts when both of its nodes ran `NET/IB` with another
node. That is an inference: it misses a fallback to `NET/Socket` that only that
one pair has. `all`, the default, runs every pair and judges each pair by its own
run. `ib_write_bw` runs on every pair either way.

Pairs grow as n(n-1)/2: 6 nodes are 15 pairs in 5 rounds of disjoint pairs, and
the dry run prints an estimate. The fabric rarely changes, so probe once when the
machines arrive and again after recabling or a switch change, not before every
run.

### Group names

Group names stay the same as nodes come and go:

1. old names (the inventory's `fabric_group`) go to new groups by the number of
   nodes they share, most first over every pair of old name and new group, each
   name and each group once: a node added to a probed fabric joins its group's
   name, and removing a node renames nothing;
2. a new group is named `fg-` plus a hash of its fabric: the IB subnet prefixes
   of its compute ports, or the IP networks of its RoCE ports;
3. when the fabric gives no distinct key, `fg-` plus a hash of its sorted node
   names.

A name held by an inventory node outside the probe (`--nodes`) is never reused
for another group. You may also write a name by hand (letters, digits, `.`, `_`,
`-`; at most 64 characters). `unavailable` is reserved.

### Writing the inventory

Only `--write` writes. Only the `fabric_group` lines change: every other byte of
a YAML inventory (comments, anchors, flow mappings) is kept.

- A probed node in a group gets `fabric_group: <name>`.
- A fabric unavailable node gets `fabric_group: unavailable`. Such a node never
  joins a multi-node island; a single-node island takes it as before.
- When the probed node holds an anchor that unprobed nodes merge
  (`- &b300 ...` then `{<<: *b300, ...}`), those nodes get their current value
  written on themselves (`fabric_group: null`) so the anchor's new group does not
  reach them.

The edited text is read back and compared with the original, then written to a
temporary file next to the target, synced to disk and renamed over it. A symlink
is followed and kept, the mode is kept, and the owner too where the writer may
set it. Without `--write` the probe writes no file at all.

```yaml
nodes:
  - name: b300-0
    fabric_group: fg-3f9a0c12
    host: ops@192.0.2.20
    # (the other fields as before)
```

### Placement

The rule applies to multi-node trainer islands (`--gpu ssh:NxMxCARD` with
N > 1, and `--gpu ssh:<inventory file>`); a single-node island has no fabric
constraint.

- Once any node of the inventory has the `fabric_group` key (`null` included),
  the island's nodes must all be healthy nodes of one group. Otherwise
  `yeto launch` (and its dry run) and `yeto preflight` refuse before any SSH,
  naming the groups and their node counts: `the island needs 6 healthy nodes of
  one fabric group; group fg-3f9a0c12 has only 4, group fg-8d21e7b0 has only 2`.
  A node without a group, or `unavailable`, is listed as such: probe it, or
  remove it from the trainer role.
- The choice is fixed (`yeto/launch/fabric_groups.py`, `pick_placement`):
  excluded nodes leave first (not healthy, quarantined, leased by another run).
  Then the group with the most free nodes that holds the island wins (ties: the
  group name), and its nodes go by name. A caller whose node could not be taken
  excludes it and asks again, so the next node of the same group fills in.
- Healthy here means not excluded by the caller. The running health check of
  the nodes is wired into this filter once it lands (the node health change);
  until then a node counts as healthy.
- An inventory without any `fabric_group` key (every inventory written before
  the probe existed, the customer's included) places islands as before, in the
  inventory's order. A real launch prints one note into the launcher log
  suggesting `yeto cluster probe`. A dry run and `yeto preflight` print nothing
  (the note goes to the logger only), so their output stays byte for byte what
  it was. `yeto cluster probe` and `yeto cluster check` always print the note.

To drop the rule, delete the `fabric_group` lines.
