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
| `run.own-trainer-own-sampler.json` | own 2-node trainer island and an own one-node sampler pool, nothing billed; on `inventory.own-b300-nodes.yaml` (two 8 x B300 trainer nodes, one 8 x B300 sampler node) with `sampler-spec.qwen3-4b-vllm-own-b300.json` |
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

The trainer image is the one the launch runs: the trainer's `--rl-image`;
without one, for an all-`ssh:` trainer whose inventory names no image, the
site image catalog's verified record for the model on the trainer nodes, else
the pinned default. The dry run prints the cluster preflight command with that
image (`yeto preflight ... --image <image> --no-bench ...`); the preflight
pulls and probes it and the egress check below probes its registry. An
inventory `image:` / `images:` is not a source of its own: it turns the site
catalog off and must name the same image as the run (its `--rl-image`, else
the pinned default), or the launch and the preflight refuse it ("the run's
image decides"). A preflight that has no image for a docker node refuses
(`no trainer image resolved`) and pulls nothing.

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
else the image the launch fills in: the site catalog's or the pinned default;
Docker Hub is `registry-1.docker.io`).
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

A Miles full-parameter island (`--rl-sync-preset full-parameter`, or
`decoupled` with `--tuning full`, which routes to it) also binds its placement
into the session contract it offers the syncer: nodes x GPUs per node, the
trainer's nodes x GPUs and the trainer/rollout split (`--actor-num-nodes`,
`--actor-num-gpus-per-node`, `--num-gpus-per-node`, `--rollout-num-gpus`).
Two such islands of one shape on other machines (2 x 4 and 1 x 8 GPUs, DP 8 on
both) build one layout but two contracts, and the syncer refuses the later
island at HELLO (`session mismatch`). The fleet preflight (`yeto run up`, its
dry run, and `yeto fleet island add/move`) compares the TP/PP/EP/CP/DP of the
islands and, for these islands, their placement, and refuses the fleet before
anything is ordered, naming both. Run every island of such a fleet on the same
nodes x GPUs per node; `--rl-sync-layout shape-free` drops the parallel shape
from the contract, not the placement. LoRA, the `strict-avg` default and verl
islands do not bind their placement and are not compared, nor is a launch that
gives its islands their own placement (`--rl-island-placement`: the islands get
`--rl-per-island-placement`, which leaves the placement flags out of the
contract).

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
- **Weights.** Put the model on every node once (the template's sampler
  nodes mount `/data/models`); the pool does not pull it.
- **Prompts.** With `prompts_file` the pool copies the file to the host path
  behind the spec's `prompts.path` (under the pool's `workdir` or a mount) at
  every start and checks its size and sha256 there. Without it, put the file
  there before the run; the dry run warns when the sampler cannot read the
  path.
- **Nodes.** A node named in `provider.nodes` needs the `sampler` role (or no
  roles), the spec's tensor parallel size must fit its cards
  (`provider.devices` when set, each an index of the node's cards). `yeto run
  up` refuses a pool on a node of the run's `ssh:` trainer islands or their
  spares, unless both lease the node in one `lease_store` and the islands
  still find the nodes they need (then a warning). Two pools of one run on
  the nodes of one `lease_store` skip each other's nodes; they are refused
  when both name one node in `provider.nodes` or their targets exceed the
  nodes they can take (split a node's cards between pools with
  `provider.devices` on an inventory without `lease_store`).
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
its lease kept. A node a spare-pool island lost is quarantined: release it with
`yeto cluster release-node` (see "Spare nodes" below). Release any other such node by
hand:

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

## Spare nodes

A resident `ssh:` trainer island can keep running after one of its nodes is lost,
if its inventory lists one or more spare nodes. A spare has the role
`trainer-spare`:

```yaml
nodes:
  - {name: b300-0, roles: [trainer], fabric_group: A, ...}
  - {name: b300-1, roles: [trainer], fabric_group: A, ...}
  - {name: b300-5, roles: [trainer-spare], fabric_group: A, ...}   # same card, card count, family, runtime
```

A spare is not an island node. The island still takes exactly as many `trainer`
nodes as its `--gpu` says, so an inventory without spares behaves exactly as
before. Every spare must match the island's nodes in device family, CPU
architecture, card, card count, runtime and `store_env_file`, and it must have a
rendezvous `address`. `yeto launch --dry-run` refuses a spare that cannot take a
node's place, and lists the spares and their fabric groups under the island's
`spares`. A device family without a node health checker (see "Node health" in
`docs/NODE_RUNTIME.md`) never gets a spare: its spares do not apply at all. The dry
run and the start print a WARN, and a failed island is relaunched on its own nodes,
as without spares.

### When a node is lost

A node is lost when either of these holds:

- its log stream over SSH ended, and it then failed `unreachable_polls` SSH checks
  in a row (3 by default);
- its node health check was bad twice in a row.

An SSH check that fails while the node's stream still flows is not a loss on its
own: the island fails, and the relaunch first stops it on every node. A node whose
stop is not confirmed (on a power loss, rank 0 usually exits first and the dead
node's stream only ends minutes later) is checked with `ssh true` on each relaunch
poll; twice `unreachable_polls` checks in a row without an answer make it lost. A
node that answers again is relaunched on, as before.

A network partition is enough: a node that cannot be reached for about 2 to 5
minutes (the stream's keepalive timeout plus the checks) is lost and quarantined,
even if it was healthy. Each check of a waiting island tries the stops such a node
still owes again, so its old container stops once it answers. Nothing lifts a
quarantine on its own; release the node by hand (below) once it is reachable again.

When a node is lost, the island writes `island_node_lost` and stops on its other
nodes. The lost node is not waited for; one that still answers (a health loss) is
stopped at once in the background. The node is quarantined:
`fleet/nodes/<name>/quarantine.json` in the run store records why, which stop the
node still owes, and the run's lease holder. With a `lease_store`:

- the record is also written to `<lease_store>/nodes/<name>/quarantine.json`, next to
  the node's `lease/`. Every run's trainer pick and spare pick leave the node out
  until it is released. The pick prints which run and island quarantined each such
  node and how to release it, once and on every waiting line. A new run whose
  inventory lists no other free trainer node waits for one (`--rl-node-lease-wait-s`)
  and then fails with the same note; it does not use a spare in its place;
- a record that cannot be read (the store fails, or the record is not JSON) counts
  as a quarantine, with the reason `quarantine record unreadable: ...`; if the store
  cannot list the records, each node's record is read instead. A running island's
  node is not taken as lost on such a read failure;
- the run keeps its lease on the node (`kept: true`, no longer renewed) while it is
  quarantined, also across a restarted launcher;
- once a spare took a node's place, the run dir's lease record (`node-leases.json`)
  names the spare. It is written before the island starts on the spare, and put back
  if that start fails. A restarted launcher leases the island's current nodes, taken
  from the island's spares journal (`ssh-islands/<island>-spares.json` in the run
  dir), so it follows a replacement the old launcher stopped in the middle of. Every
  start, reattach included, gives back the leases of old nodes the island no longer
  runs on; an old node another run holds by then does not stop the island.

A stop the node still owes stays in the run's ledger as a row of its own, priced as
that node only, so `yeto cleanup --run <run>` still tries to stop it there. A lost
node whose own stop was confirmed owes nothing for that incarnation and has no such
row; picked again later, a loss of its new incarnation owes its stop again.

### Which spare is taken

The spare and the nodes that stay are placed together as one island. The nodes that
stay never move; the spares are tried by node name, and the first one that meets
every condition below is taken:

- it is in the fabric group of the nodes that stay (`fabric_group`); a node whose
  group is `unavailable` takes no multi-node place;
- its node health check ran and passes now; an answer over SSH is not enough;
- no other island of the run holds it (`fleet/nodes/<name>/claim.json`, written
  once with an exclusive create);
- it is not quarantined, by this run or (with a `lease_store`) by another run. The
  wait reason names the run and island of another run's quarantine and how to
  release the node;
- its cross-run lease is free, when the inventory names a `lease_store`.

The claims need a run store with an atomic exclusive create (s3:// or r2:// with
`If-None-Match`, or a local directory). A check-then-write store, such as the OSS
relay, is refused: two islands could take one spare there.

The health checks run before the pick, outside the launcher's lock, so the other
islands' supervision is not held up. The pick then claims the node and reads its
records again.

The spare takes the lost node's place in the same layout. If rank 0 was lost, the
spare becomes rank 0 and the rendezvous address moves with it. The island's next
incarnation starts under the same learner id and resumes from that learner's newest
checkpoint: its round cut (`island-cuts/island<k>/LATEST`), or else its newest
policy release (`trainer<k>/policy-releases/` when its samplers follow it).
`island_node_replaced` records the old node, the new node and the resume step. To
the syncer this is a short absence followed by a re-JOIN: the island catches up to
the newest outer weights before it merges again, and the other islands keep merging
meanwhile.

### No free spare: waiting and backoff

If no spare qualifies, the island waits (`island_waiting_spare`, with the reason,
e.g. `fabric group A has no free healthy spare node (...)`). It checks again after
60 s, then 120 s, doubling up to 900 s, and writes `island_waiting_spare` again
every hour with `elapsed_s`. While it waits, the fleet controller does not give it
up on `--recover-timeout`.

Two separate times apply while an island waits:

| Setting | Default | What it decides |
|---|---|---|
| `fleet.waiting_absent_after_s` (run config) | 3600 s (1 hour) | Whether the fleet goes on. For this long after its wait began, the waiting island still counts toward `fleet.min_islands`; after that it counts as absent, and a fleet left with fewer than `min_islands` islands stops (its samplers stop billing with it). |
| `--spare-max-wait-s` (launch) | 86400 s (24 hours) | When the island is given up: after this long its relaunch is refused and it is abandoned (also under a fixed roster). |

So in a fleet with more islands than `min_islands`, an island can wait up to 24
hours while the others train. A one-island fleet, or one with exactly `min_islands`
islands, is not stopped the moment an island starts waiting, but it stops after 1
hour of waiting. The wait's start is kept in the island's journal, so a restarted
launcher keeps counting; it starts normally, the quarantined node is lost again,
and the island goes on waiting.

### Replacement limit

An island may replace at most 3 nodes within 24 hours. On the next loss its
relaunch is refused, `island_replacement_limit` lists every node it replaced, and
the island is given up. The fleet then goes on if `fleet.min_islands` islands still
run.

### Releasing a quarantined node

Once a lost node is repaired, put it back with:

```bash
yeto cluster release-node b300-1 --inventory inventory.yaml --store s3://RUN-STORE --dry-run   # what would run
yeto cluster release-node b300-1 --inventory inventory.yaml --store s3://RUN-STORE
```

`--store` is the run's `--rl-checkpoint-store`. The command:

1. checks the node: its health check must run and pass (an answer over SSH alone is
   not enough). `--force` skips the check and prints a WARN; a device family without
   a health checker needs it;
2. runs the stops the node still owes and closes each owed ledger row. The ledger is
   the run's (`runs/<run>/resources.jsonl` on the machine that ran the launcher); run
   elsewhere, the command says where the row is. When the launcher picks the node
   again, it runs the row's stop and closes the row (`released elsewhere`) only once
   that stop succeeded, so the node is billed once and the old container is gone.
   A quarantine record that cannot be read needs `--force`: the stops it owes are not
   known, so check the node for the run's containers by hand first;
3. gives up the run's lease on the node (with a `lease_store`). A kept lease ends at
   once. A live one (the run's launcher still renews it) ends only after a probe over
   SSH found nothing of the run on the node; a lease another run holds is not
   touched;
4. removes the quarantine record, in the run store and in the `lease_store`.

`--dry-run` prints the record, the stops it would run, the records it would remove
(in the run store and the `lease_store`) and what it would do with the lease.

From then on any island may take the node again, including one waiting for a spare.
If the check, a stop, the probe or the lease release fails, the node stays
quarantined (exit 1). The launcher never releases a quarantined node on its own.

The run's launcher may still be running during the release. It never writes a
quarantine record back: a stop it confirmed is recorded as a marker of its own
(`fleet/nodes/<name>/settled/`), and the release first writes `released.json`, with
its time, next to the record in both stores. A stop the launcher confirms after the
release therefore records nothing. The launcher reads the record again before each
stop it retries, and skips a stop the record no longer owes. A confirmed stop removes
its incarnation's pid file and leaves a `stopped` marker in its state dir, so a later
stop of that incarnation (release-node, `yeto cleanup`) succeeds without signalling a
process that reused the pid. A ledger row is closed only after its stop was confirmed.

### Known limits

- The K3.1 customer layout has no spare. Its PP3 x TP8 x CP2 x EP16 trainer takes all
  48 cards of the 6 x 8 B300 nodes, so a lost node fails the island as before. To
  keep spares, use smaller islands, e.g. two 2-node islands, one sampler node and
  one spare node.
- A network partition quarantines: a healthy node that cannot be reached for 2 to 5
  minutes is lost and stays quarantined until `yeto cluster release-node`.
- The layout is never shrunk: without a spare the island waits; it does not change
  TP/PP/EP.

## Onboarding a model image

A new model often comes with an image built for it: a trainer image with the
kernels and Miles / Megatron versions the model needs, or a sampler image with the
vLLM or SGLang build that serves it. `yeto image adopt` takes such an image into
your site, checked and smoked on your own nodes, and records it in the **site image
catalog**. From then on a trainer island or a sampler pool of those nodes that
names no image runs it for that model.

```
yeto image adopt REGISTRY/IMAGE:TAG --model ORG/MODEL --role trainer \
    --inventory cluster.yaml --model-config config.json \
    --smoke-store s3://BUCKET/image-smoke --dry-run

yeto image adopt REGISTRY/IMAGE:TAG --model ORG/MODEL --role sampler \
    --inventory cluster.yaml --sampler-spec sampler-spec.json --dry-run
```

`--dry-run` prints every step with its commands and reaches no registry and no
node (`--model-config` gives it the model's `config.json`; without it the real run
reads the local HF cache or fetches `config.json` from the Hub). Drop `--dry-run`
to adopt the image. The steps, in order:

1. **manifest**: yeto checks that the reference is a docker reference, asks the
   registry for the image's manifest (with the registry login a pull would use:
   `SKYPILOT_DOCKER_USERNAME` / `SKYPILOT_DOCKER_PASSWORD` /
   `SKYPILOT_DOCKER_SERVER`, else `~/.docker/config.json`) and pins a tag to its
   digest. A missing manifest stops here, before any node is contacted.
2. **probe**: one node (`--node`, default the first of the role) is checked first:
   SSH, the host probe and room in its image root, and that its cards are idle (no
   process in `nvidia-smi`, `amd-smi` / `rocm-smi` or `npu-smi`, no running `yeto-`
   container; `--allow-busy` goes on with a warning). It then runs the image once,
   with its container engine and without cards, and reads the CPU architecture,
   Python, torch, CUDA or HIP, the card architectures the image's kernels were
   built for (torch's build flags; only when they are empty does it run again with
   the node's cards), the engine packages and the yeto image lock
   (`/opt/yeto/image`).
3. **fit**: the image against every node of the role: CPU architecture, device
   runtime, card architecture, the trainer image floors yeto knows, and the role's
   engine package (a sampler: the engine of the sampler spec, which `--engine`
   must not contradict). Card architectures follow the CUDA binary compatibility
   rules: a cubin `sm_XY` runs on the same major version with an equal or higher
   minor (an `sm_100` image runs on B300, `sm103`), `sm_XYa` only on exactly X.Y,
   PTX `compute_XY` is compiled on any X.Y or newer card (`compute_XYa` only on
   X.Y). A pass by the minor-version rule or PTX is noted in the result and the
   evidence (`compatible by CUDA minor-version rule; the smoke test is the final
   check`): the smoke runs real kernels on the card. `--strict-arch` asks for a
   cubin of exactly the node's architecture. A ROCm node's `gfx` target must be
   built (feature suffixes such as `:xnack-` are ignored). A mismatch names the
   rule and the node's card. A trainer's `--attention-backend flash` needs sm90+
   cards on every node of the role: below that (an A10G, `sm86`) its backward pass
   fails, so the fit refuses it and names `--attention-backend unfused` instead. The
   backend the fit checks is the one the smoke's launch resolves: with the default
   `auto` that is the recipe's own (`deepseek-v4-flash` and Qwen3.8-Flash-Next train
   with flash, so they are held to sm90+ too; those recipes have no other path, so
   the fit names other cards, and another backend for them exits 2). The dry run's
   fit line names the backend and where it comes from. The smoke trains with that
   backend (`--rl-training-attention-backend`, also taken from a `--smoke-arg`,
   written in full; the two must agree). A sampler is not held to this floor.
4. **pull**: every node of the role (spares included, `roles: [trainer-spare]`)
   is checked over SSH, and its image root (the docker data root, or
   `/var/lib/containerd` with the containerd image store, or podman's storage
   graphroot) must have `--min-image-free-gib` free (default 120). A podman node
   whose graphroot `podman info` does not print, or that the disk probe does not
   find, fails: the image store's disk cannot be confirmed. Only when every node
   passes does the pull run, on every node in parallel with its own container
   engine. A node that fails is named with the preflight's finding or the
   engine's error line.

   Each node's container engine is its inventory `container:` (`docker` or
   `podman`; default: `docker` for `runtime: docker`), whatever its `runtime:`:
   a `runtime: rootfs` node with `container: docker` or `podman` is pulled with
   that engine, and one with no container engine is named as not pulled.

   For a sampler image, pass the ssh sampler pool's config with `--pool-config`:
   it is read the way the pool reads it, so the inventory is the pool's
   `provider.inventory` (`--inventory` may be left out; one naming another file is
   an error), the catalog is that inventory's `site_images:`, the one the pool
   resolves `image: auto` from (a `--catalog` naming another file is an error), the
   nodes are the pool's `nodes:` (one not in the inventory is an error), their
   engine is the pool's
   `container.engine` (default podman, whatever the inventory says), and its
   `container.sudo: true` (`sudo -n podman`, the rootful store) makes the host
   checks, the pull, the probe and the smoke all run the engine through
   `sudo -n`, so the image lands in the store the pool starts from. A
   `runtime: venv` pool runs no image and is refused. With `container.pull:
   never` the pool starts only images already on its nodes: the adoption's pull
   is what puts the image there. Without a pool config, each node's engine is
   the inventory's (a NOTE says so) and `--container-sudo` asks for `sudo -n`;
   against a pool config whose `container.sudo` is off it is an error.
5. **smoke**, on the probe node, after its cards are checked idle again:
   - **trainer**: the model is cut to its first N layers (all their experts, the
     embeddings and the head, `scripts/prune_layers.py`; MTP layers are dropped,
     and the result says so), N being the largest, at most 16, whose trainer
     state fits one node in the model registry's sizing (`yeto adapt`'s backend,
     recipe and tuning for the full model; `--smoke-arg=--tuning=full` picks
     another tuning). The node's `scratch:` (required in the inventory) must have
     room for the cut and the shards it downloads, about twice the cut's size;
     the download cache is removed after the cut. A one-node `ssh:` island then
     trains one round on synthetic rollouts (no sampler, no syncer, a small local
     prompt file and a built-in reward), ends it with a cut in `--smoke-store`
     (the trainer's `rl_round_cut` event and the cut's manifest in the store are
     checked), and a second launch resumes from that cut (its `rl_resume` event
     must name the same cut) and trains another round. `--model-path` takes the
     checkpoint from a directory on the node instead of the Hub. A gated model
     needs `HF_TOKEN` in the environment of `yeto image adopt`, and a Hub mirror
     `HF_ENDPOINT`: both are passed to the node's cut through the script it
     reads, never on a command line and never printed. The model needs
     `--trust-remote-code` when its `config.json` has an `auto_map`, or when the
     model registry does not know it; `--smoke-arg=--trust-remote-code` or
     `--smoke-arg=--no-trust-remote-code` decides it instead. Every RL launch
     requires the flag, whatever the model and backend (the pinned trainer's
     loaders always allow remote code), so the smoke always passes it and refuses
     `--no-trust-remote-code`; the evidence notes say what the model alone needs.
     When the registry's flags for the model need
     a file (for example `--rl-expert-selection` for Kimi K3.1), pass it with
     `--smoke-arg=--rl-expert-selection=/path/selection.json`; any other
     `yeto launch` flag the run needs goes the same way. Only the Miles backend
     has a trainer smoke so far, and only on `runtime: docker` nodes (the `ssh:`
     trainer island starts docker containers as the SSH user: for `--role
     trainer`, any podman or `rootfs` node of the role, spares included, or
     `--container-sudo` / `--pool-config`, is refused).
   - **sampler**: the engine of `--sampler-spec` (the pool's sampler spec) starts
     in the image the way the node runtime starts a sampler (the node's device
     flags and directories), generates once, reloads (the weights of `--release`,
     else the spec's `engine.server.base_dir`, as policy version 1; one of the two
     is required) and generates again. The reload checks the engine's hot-reload
     endpoint only, not the format of a trainer release, and it must reload in
     place: an engine whose reload falls back to a restart fails the smoke, and
     the error names its `reload_mode`; so does a load that returns no report of
     its reload mode. A failing step is named, for example
     `smoke failed: engine: reload failed: ...`.

   Whatever the smoke's outcome, a failure or a timeout, its containers are
   removed and its run stopped (`yeto down`), and its artifacts are deleted: the
   cut on the node, the checkpoint under `--smoke-store`, the launcher's files.
   `--keep-smoke-artifacts` keeps them and prints where they are and their size.

Any failed step stops the adoption and leaves the catalog unchanged. The image
stays pulled on the nodes that already have it (the error names them): a retry
reuses it, and `docker rmi` on those nodes frees the room. After a passing smoke
the catalog gets a `verified-smoke` record with the digest, the versions the probe
read and the evidence of the run (time, node, seconds per step, the nodes that
pulled the image, the smoke's layout and tuning, and any compatibility notes). Two
adoptions that write one catalog at the same time take turns on a lock next to it.

### The site image catalog

The catalog is `~/.yeto/site-images.yaml` (`YETO_SITE_IMAGES_PATH` moves it), or
the file the inventory names with its top-level `site_images:` (a relative path is
taken from the inventory's directory):

```yaml
site_images: images/site-images.yaml
nodes:
  - name: b300-0
    ...
```

Each record:

```yaml
version: 1
images:
  - model: ORG/MODEL
    role: trainer                  # trainer | sampler
    engine: miles                  # trainer: the --rl-backend; sampler: vllm | sglang | vllm-ascend
    ref: REGISTRY/IMAGE@sha256:<64 hex>
    device_family: cuda
    cpu_arch: x86_64
    card_archs: [sm103]            # the card architecture the smoke ran on
    versions: {python: 3.12.3, torch: 2.9.1+cu130, cuda: "13.0", megatron-core: 0.15.0}
    evidence: {at: "2026-11-26T10:00:00Z", node: b300-0, seconds: 1834.2, result: pass}
    status: verified-smoke         # verified-smoke | retired
```

One image adopted on two kinds of nodes keeps one record per CPU and card
architecture. `evidence.at` is an ISO-8601 time with its time zone. To stop using a
record, set its `status: retired`. A broken record (bad YAML, a tag instead of a
digest, an unknown card architecture, a time without a zone) is refused with the
file and the record named, but only by a run that reads it; a key the reader does
not know, or a newer catalog `version`, is a warning.

### Which image a run uses

For an `ssh:` trainer island (when every `--gpu` entry is `ssh:`) and an ssh
sampler pool node:

1. the image the configuration names: `--rl-image`, an inventory `image:` or
   `images:`, a pool's `container.image`. The catalog is not read;
2. else the newest `verified-smoke` record of the catalog for the run's model (an
   alias such as `qwen3-8b` is resolved first) and role whose CPU architecture,
   card architecture and device family match every node of the island (for a
   pool: every node of the pool, so a spare starts the same image);
3. else the built-in default (the pinned `--rl-image`, or the engine image catalog's
   choice for a pool's `container.image: auto`).

A fleet that mixes `ssh:` and cloud islands takes the built-in default and says
so. A pool chooses its images once, at its first start, and keeps them in its
state: a replacement node and a restarted pool controller start the same image.
The launcher and `pool --dry-run` print the image and where it came from, for
example:

```
[launcher] ssh trainer image REGISTRY/IMAGE@sha256:... (adopted 2026-11-26T10:00:00Z on b300-0) (source: site image catalog ...)
```

When it falls back to the default, the line names the closest records and why
they do not fit. A configuration that names its image prints nothing new and runs
exactly as before.

## Islands that lag, join or leave

A fleet of `ssh:` trainer islands on your own machines uses the same fleet mechanisms as any
other island fleet ([FLEET_TRAINER_ISLANDS.md](FLEET_TRAINER_ISLANDS.md)). This section
covers what is specific to `ssh:` islands: how a slow or stuck island is reported, and what
happens when one leaves.

### An island that lags

An island can fall behind or stop pushing while its process and its connection stay up: a
slow node, a hung collective, a trainer that waits on something. The syncer does not wait
for it. When a round's quorum deadline passes, the syncer merges the pushes that arrived and
records the silent island as absent. Later rounds do not wait for it either. Once the island
delivers again, its late push is carried into a later round with a discount, or refused when
it lags past `--rl-sync-max-base-lag`. The island then catches up to the newest merged
weights and answers rounds again. The other islands keep merging the whole time.

The island's fleet controller reports such an island. It reads the syncer's merges, and
when none of the island's pushes was merged in `N` merge cycles it writes:

- `island_straggler` in the launch's `fleet.jsonl`, with `missed_cycles`, `merges`,
  `fragments`, `last_merged_step`, `newest_step`, `absent` and `action`;
- the same fields under `straggler` in the island's record (`fleet/trainers/<k>/state.json`,
  shown by `yeto island status`).

A merge cycle is one outer step of every fragment, so with 4 fragments and `N = 3` the
island is reported after 12 merges without it. Counting starts when the island joins the
syncer after a launch or relaunch, so the time an island takes to start is not counted. A
push the syncer refused does not count as merged. The report is written once per stretch.
`island_straggler_cleared` follows once a push of the island is merged again, and its record's
`straggler` is set back to `null`.

`action` is what the controller does about it. The default is `wait`: nothing is stopped or
relaunched, because the syncer already merges without the island. A lost or unhealthy node
is handled by the island's own launch, not by this report.

Set `N` in `fleet.trainer_argv`, so every island launch gets the same value:

```json
"trainer_argv": ["...", "--rl-island-straggler-cycles", "3"]
```

`3` is the default. `0` turns the report off. The value must be a whole number of 0 or more;
`yeto launch --dry-run` refuses anything else.

The controller reads the syncer's tape in one of two places:

- from the syncer's own file, when the syncer runs on the same host as the controller
  (`--controller head`);
- otherwise from its copy in the run store, `tapes/syncer/<incarnation>/<seq>.jsonl`. The
  syncer launch uploads it every 30 s (`YETO_RL_TAPE_UPLOAD_S`) when `--rl-checkpoint-store`
  is an `s3://` store. With another store, or with the upload off (`YETO_RL_TAPE_UPLOAD_S=0`),
  the launch prints `island straggler watch off` and nothing is reported. Training is not
  affected either way.

The report comes at most one upload interval after the merges it counts.

### Removing an island

```bash
yeto fleet island remove --config run.json --name north --dry-run
yeto fleet island remove --config run.json --name north
```

The trainer worker writes the island's STOP. The island's controller drains it, stops the
trainer on every node (and checks that each container is gone), and records the island
`done`. Its incarnation is closed in the run's resource ledger. The syncer sees the island's
connection drop. The round it was asked for waits one island lease and then merges over the
islands left. The island's nodes are free for another island afterwards.

In a fleet launched as one launch (`fleet.launch: synced_group`), islands cannot be added
while it trains. One island can be stopped and started again by its learner id, for example
island 1 with `yeto island stop 1 --config run.json` and `yeto island start 1 --config run.json`.
