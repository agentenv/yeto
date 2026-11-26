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
disturbed by it. Before anything else it checks every node, and each finding
refuses the whole probe (exit 3), listing the node and what keeps it:

| Finding | Let through by |
|---|---|
| another run's lease holds the node (an inventory with `lease_store`) | nothing |
| a GPU compute process (`nvidia-smi --query-compute-apps`) | `--force-busy-gpu` (with a WARN) |
| a `yeto-` container | `--force` |
| idleness not confirmed: the node did not answer, its GPU processes or containers could not be listed, or its lease could not be read | `--force` (with a WARN) |

Containers are listed with the node's container engine (the node profile's
`container`: docker or podman), as the user and, when `sudo -n` works, through
it. For docker either view listing is enough; for podman the user's own view
must list. A docker daemon that is not running ("Cannot connect to the Docker
daemon", not "permission denied") has no containers, unless a `containerd-shim`
process still runs on the node: live-restore keeps containers running without the
daemon, and a rootless daemon whose socket `DOCKER_HOST` does not point to looks
down too, so the listing is not confirmed. This rule is docker's only. Probe the
idle nodes with `--nodes`.

With `lease_store`, the leases are read again after the benchmarks and before
anything is written. A node another run leased during the probe refuses the
result (exit 3, nothing written), whatever the flags. A lease that cannot be read
then needs `--force`, as before the probe. The TCP port check of the preflight runs as well; a port
already in use is a WARN, as in `yeto preflight`.

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
  `dev_infiniband`), and each has its rails active, on the preflight's terms: with
  `NCCL_IB_HCA`, every port it names; without it, at least
  `min(ports of the fabric's link layer, cards_per_node)` ports (a spare port
  down, a storage NIC for example, is a WARN in the preflight and no block here).

A group must be a clique, so each connected component is cut into cliques. Up
to 16 nodes the cut is exact: the largest clique is taken (ties: the first
sorted list of member names), then the largest of the rest, until no node is
left. A larger component is cut greedily: the node with the most missing edges
leaves (ties: the last by name) until the rest is a clique, the nodes that left
are cut again, and a node left alone then joins the largest clique it reaches
every member of. The probe prints every missing edge inside a component. A node
left alone is `fabric unavailable`, with its reasons: the failed node check, the
pairs whose `ib_write_bw` failed, or whose NCCL fell back to `NET/Socket`.

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
| 2 | a usage or inventory error, including a bad profile key (`container`, `paths`) |
| 3 | a node is busy, or another run leased a probed node during the probe |

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
| `--force` | off | probe nodes not confirmed idle, or with `yeto-` containers |
| `--force-busy-gpu` | off | probe nodes with GPU compute processes (a WARN each) |
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
   name and each group once; removing a node renames nothing. A name that nodes
   outside the probe (`--nodes`) still hold goes only to a new group whose every
   member already carried it: a node never inherits a group it was not measured
   against;
2. a new group is named `fg-` plus a hash of its fabric: the IB subnet prefixes
   of its compute ports, or the IP networks of its RoCE ports;
3. when the fabric gives no distinct key, `fg-` plus a hash of its sorted node
   names.

To add a node to an existing group, probe it together with every member of that
group (`--nodes` naming all of them): it then joins the group's name. A probe of
the new node with only some of the members gives that pair a new name instead,
and the rest of the group keeps the old one. You may also write a name by hand
(letters, digits, `.`, `_`, `-`; at most 64 characters). `unavailable` is
reserved, in any letter case.

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
set it; the file's newlines (LF or CRLF) are kept. A file that cannot be parsed
is refused with an inventory error. Without `--write` the probe writes no file
at all.

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
  `yeto launch` (and its dry run) refuses before any SSH, naming the groups and
  their node counts: `the island needs 6 healthy nodes of one fabric group; group
  fg-3f9a0c12 has only 4, group fg-8d21e7b0 has only 2`. `yeto preflight` reports
  the same as a FAIL check (`inventory fabric_group`) and still checks every
  node, so it can tell what is wrong with the odd one. A node without a group, or
  `unavailable`, is listed as such: probe it, or remove it from the trainer role.
- The choice is fixed (`yeto/launch/fabric_groups.py`, `pick_placement`):
  excluded nodes leave first (not healthy, quarantined, leased by another run).
  Then the group with the most free nodes that holds the island wins (ties: the
  group name), and its nodes go by name. A caller whose node could not be taken
  excludes it and asks again, so the next node of the same group fills in.
  `required` nodes (an island's surviving nodes when one is replaced) stay: only
  their group is considered, and the free nodes of it fill the rest by name.
- With `lease_store`, the lease pick uses the same placement: the free nodes the
  leases leave are placed by group, and a node whose lease is lost is excluded
  and the placement computed again.
- Healthy here means not excluded by the caller. A launch places on the
  inventory as it is; the spare-node replacement excludes the nodes the running
  health check lost or that are quarantined, and keeps the surviving ones as
  `required`.
- An inventory without any `fabric_group` key (every inventory written before
  the probe existed, the customer's included) places islands as before, in the
  inventory's order. A real launch prints one note into the launcher log
  suggesting `yeto cluster probe`. A dry run and `yeto preflight` print nothing
  (the note goes to the logger only), so their output stays byte for byte what
  it was. `yeto cluster probe` and `yeto cluster check` always print the note.

To drop the rule, delete the `fabric_group` lines.

## On-site control plane

A fleet launched island by island (`fleet.launch: per_island`, `rollout.control_plane: "store"`,
see [FLEET_TRAINER_ISLANDS.md](FLEET_TRAINER_ISLANDS.md)) has two long-running processes besides
its islands:

- the **control plane**: `yeto run up` of the run config. It drives the sampler pools and the
  trainer worker, which starts and supervises one `yeto launch` per trainer island;
- the **syncer**: the fleet's outer optimizer, which every island dials.

On your own machines you can run both on site, as systemd services of a machine of the
cluster (or a small VM in the same room), instead of on a laptop or a cloud VM. `yeto` writes
the units.

### The syncer as a service

Name the syncer's host in the run config:

```json
"fleet": {"launch": "per_island", "syncer_service": "<syncer-host>", "trainer_argv": ["..."]}
```

`fleet.syncer_service` is the host name or address the islands dial, without a port (the
syncer listens on its own ports, 29400 and up). With it:

- the **syncer service** (`yeto fleet service-run --role syncer`, what the unit runs) runs the
  fleet's syncer on that host. A syncer process that dies is started again at once and resumes
  from its checkpoint, on the same address and at the same term.
- **One service at a time.** The service holds a lease in the run store
  (`fleet/syncer_service/lock`, 120 s, renewed with each heartbeat). A second service, on this
  host or another, refuses to start (exit 2) while that lease runs. A service restarted on the
  same host takes its own lease back at once. The lease's token is the service's generation.
- **Its records.** The service writes the syncer's record (`fleet/island_syncer.json`, with its
  generation) and a heartbeat every 30 s (`fleet/island_syncer_service.json`). The heartbeat says
  `starting` until the record is written, then `running`, and at the end `finished`, `failed` or
  `stopped`.
- **Its term** (`syncer_epoch`): the fleet's last record's when that record is this host's
  service's (a restart goes on at the same term), one more when another host wrote it (a syncer
  that starts over elsewhere fences the islands of the old one).
- the **syncer launch** of `yeto run up` orders no syncer machine. It waits for the service (up to
  20 minutes) and takes it only when the heartbeat is `running`, names the record's generation,
  and has moved at least once while it watched (a heartbeat left by a service that is gone does
  not move). It checks that it reads the same island HMAC key, records the syncer for its
  islands, then watches the heartbeat. Whether a heartbeat is fresh is timed on the control
  host's own clock: the two hosts' clocks are never compared. The launch ends with the service:
  0 when the run is finished, 9 when it stopped, 1 when it failed, or when its heartbeat did not
  move (or named another host) for `--rl-fleet-syncer-grace-s` (default 300 s) in a row;
- the syncer is billed at $0/h (your own machine). Set `syncer.usd_per_h` to count it.

The service runs on after the control plane is lost, so the islands keep merging while the
control plane is down.

### The island HMAC key

The syncer refuses an island whose HELLO is not signed with the run's island HMAC key
(`YETO_ISLAND_HMAC_KEY`). Without a syncer service, `yeto run up` makes one and keeps it in its
state directory. With one, the key must come from the environment of both units, and nothing
generates it:

- `yeto run up` (its preflight and its start), the syncer service and the syncer launch refuse
  to start without `YETO_ISLAND_HMAC_KEY`;
- the syncer launch refuses a key that is not the service's (exit 10). The run store holds only
  a key id (a SHA-256 prefix of the key), never the key.

### The environment file

Each unit reads its credentials from one environment file on its host, which group and others
cannot access (mode 0600 or 0400): the run store's keys, the clouds' keys if any, `HF_TOKEN`,
and `YETO_ISLAND_HMAC_KEY` (the same value on every host):

```bash
sudo install -d -m 0755 /etc/yeto
sudo install -m 0600 -o <service-user> /dev/null /etc/yeto/<run>.env
printf 'YETO_ISLAND_HMAC_KEY=%s\n' "$(openssl rand -hex 32)" | sudo tee -a /etc/yeto/<run>.env >/dev/null
sudoedit /etc/yeto/<run>.env   # add the run's other credentials
```

The unit names the file and holds no value. It refuses to start while group or others have any
access to the file, and `--install` checks the mode on the host (`sudo stat`) without reading
the file. Both services see the run config's `env` over the file's values, as `yeto run up`
gives them to its children. The syncer service passes that environment to the store client and
the syncer; it never writes it into its own process environment.

### Writing and installing the units

```bash
yeto fleet service --config run.json --role syncer --env-file /etc/yeto/<run>.env --print
yeto fleet service --config run.json --role control --env-file /etc/yeto/<run>.env --print
yeto fleet service --config run.json --role syncer --env-file /etc/yeto/<run>.env --install <syncer-host>
yeto fleet service --config run.json --role control --env-file /etc/yeto/<run>.env --install <control-host>
```

`--print` writes the unit to stdout. `--install HOST` writes it to
`/etc/systemd/system/yeto-<run>-<role>.service` on HOST over SSH (`sudo -n`), then runs
`systemctl daemon-reload` and `systemctl enable --now` there. It does nothing else.

The unit runs on its host what you give it:

| Option | Default |
|---|---|
| `--remote-config` | `--config`, as an absolute path: the run config must be at the same path on the host, with the files it names |
| `--python` | this Python: give the host's Python that has yeto's dependencies |
| `--yeto-dir` | this yeto checkout (the unit's `PYTHONPATH`) |
| `--user` | this user |
| `--workdir` | `/var/tmp/yeto/<run>-<role>`: the working directory, `yeto run up`'s state (`YETO_STATE_DIR`, `<workdir>/rl-up`) and the launch registry (`YETO_RUNS_DIR`, `<workdir>/runs`) |

Paths must be absolute and hold only letters, digits and `._/+-`; the run name only letters,
digits and `_.-`. A value that a unit line cannot hold as is is refused, never escaped.

### The control unit never drains on exit

The control unit runs `yeto run up` with `rollout.drain_on_exit` false: a stop or restart of
the unit leaves the sampler pools (and the island launches) running, and the next `run up`
adopts them. When the run config leaves `rollout.drain_on_exit` out, the unit sets it false
and says so in its log. A run config with `rollout.drain_on_exit: true` is refused for the
control unit (`--print`, `--install` and the unit itself, exit 2): draining would release the
pools at every restart of the unit.

### How the units restart

Both units have `Restart=on-failure` (30 s apart, at most 5 starts an hour) and
`SuccessExitStatus=143`: a stop of the unit (SIGTERM) is not a failure.

| Unit | Exit | What it means | Restarted |
|---|---|---|---|
| control | 0 | `run up` ended the run | no; ended marker |
| control | 75 | `run up` exited 1: it ended the run after a breaker trip or a failure it handled | no (`RestartPreventExitStatus=75`); ended marker |
| control | 2 | `run up` refused (a preflight that failed, another `run up` of the run) | yes |
| control | 70, a signal | a crash | yes |
| syncer | 0 | the run is finished | no; ended marker |
| syncer | 9 | stopped by the fleet's STOP | no; ended marker |
| syncer | 11 | a strict failure of the syncer | no; ended marker |
| syncer | 2, 10 | a refusal (another service holds the lease), a configuration error (the island HMAC key) | no |
| syncer | 1, 70, a signal | the syncer failed or the service crashed | yes, and it resumes from its checkpoint |
| both | 143 | the unit was stopped | no (a success) |

### The ended marker

A service that ended its run (the exits with an ended marker above) writes a marker to its
state directory (`<workdir>/rl-up/<run>/onsite-<role>.ended`) and to the run store
(`fleet/onsite/<role>.ended`): the exit code, the reason and the time. Each start reads it
first, on this host and in the run store. While it exists, the service exits 0 at once and
starts nothing. A reboot of the host, or the unit started on another host, never orders the
run again after a breaker trip or a strict failure. A marker that cannot be read (the run store
is down) is a crash: the unit tries again later. A refusal, a crash and a stop of the unit leave
no marker.

To start the run again, remove the marker:

```bash
yeto fleet service clear-ended <syncer-host> --config run.json --role syncer --env-file /etc/yeto/<run>.env
```

`clear-ended HOST` runs `yeto fleet service-run --role <role> --clear-ended` on HOST as the unit
runs (below). It removes the marker on that host and in the run store; for the syncer, it also
removes the fleet's STOP from the run store. Then start the unit (`systemctl start`).

### Stopping

- **The run.** Write the fleet's STOP flag on the control host, as the control unit runs: its
  user, its environment file, its `YETO_STATE_DIR` and `YETO_RUNS_DIR`:

  ```bash
  yeto fleet service stop <control-host> --config run.json --role control --env-file /etc/yeto/<run>.env
  ```

  It runs `yeto stop-run <run>` on the control host through `systemd-run` (`sudo -n`), with the
  same `--user`, `--python`, `--yeto-dir` and `--workdir` options as `--install`. By hand, on the
  control host:

  ```bash
  sudo systemd-run --wait --pipe --collect --uid=<service-user> --working-directory=<workdir> \
    --property=EnvironmentFile=/etc/yeto/<run>.env --setenv=PYTHONPATH=<yeto-dir> \
    --setenv=YETO_STATE_DIR=<workdir>/rl-up --setenv=YETO_RUNS_DIR=<workdir>/runs \
    <python> -m yeto.cli stop-run <run>
  ```

  `run up` then stops every island launch and ends the run. The syncer launch writes the
  fleet's STOP to the run store (`fleet/onsite/STOP`), and the syncer service, on whatever host
  it runs, stops its syncer (which writes its checkpoint) and exits 9.
- **The control unit** (`KillMode=process`): `systemctl stop` or a restart signals `yeto run up`
  only. The island launches and the sampler pools keep running, and the next `run up` adopts
  them. To end the run, stop it as above first.
- **The syncer unit** (`KillMode=control-group`): `systemctl stop` sends the syncer SIGTERM; it
  writes its checkpoint and is killed only after 170 s (`TimeoutStopSec=180`). The next start
  resumes from the checkpoint.

### Limits

- One syncer service per user and host: the syncer keeps its checkpoint and tape under the
  service user's home (`~/yeto-output/`).
- If the syncer's host is lost, so is its checkpoint. Start the service on another host only for
  a new run.
- `stop` and `clear-ended` need `sudo -n systemd-run` on the host (systemd 240 or later).
- Logs: `journalctl -u yeto-<run>-<role>`; the syncer's own log is `yeto-syncer.log` in the
  syncer unit's working directory.
