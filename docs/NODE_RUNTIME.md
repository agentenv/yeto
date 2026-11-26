# Node runtime: node profiles, the engine image catalog, `python -m yeto.node run`

Every sampler machine, whatever brought it up (a sky VM, an SSH inventory host, a
machine opened by hand), starts its sampler the same way:

```
python -m yeto.node run --assignment <ticket> --store <run store> --token-file <file>
```

The command reads the ticket's **assignment** from the run store (checked against
the ticket's token), and then, in order:

1. runs the node **preflight** (cards, free disk, `/dev/shm`, host memory, the
   interpreter and its modules, the store endpoint) and records what it found;
2. fetches the **code** tree by hash (`node-code/<sha>.tar` in the store, checked by
   sha256);
3. prepares the **model** by the assignment's model plan (`store`, `volume`, `hf`,
   `preset`, or nothing);
4. fills the machine paths of the sampler spec from the **node profile**
   (`__YETO_WORK__`, `__YETO_MODELS__`, `__YETO_CACHE__`, `/work/...` paths);
5. starts the sampler as its child: a process, or a container of the engine image
   with the node family's device flags.

The sampler writes `nodes/<ticket>/g<generation>/{ready,heartbeat,exit}.json` in
the run store; the heartbeat carries the engine's metrics (generated tokens,
running and queued requests, KV cache use). The pool reads a node's state
(assigned, booting, ready, stuck, exited) from the store alone. Each rewrite of an
assignment bumps its generation, and records live under their generation: a node
still running an older assignment of the ticket cannot overwrite the records of
the current one.

## Node profiles

A node profile is a node inventory entry plus five fields that say how a sampler
starts on that node:

| field | values | meaning |
|---|---|---|
| `card_arch` | e.g. `sm90`, `gfx950`, `ascend910b` | derived from `card` when omitted |
| `container` | `docker`, `podman`, `process`, `modal-fn` | how the sampler starts |
| `paths` | `{work, models, cache}` | absolute host directories (defaults `/work`, `/models`, `<work>/cache`) |
| `model_source` | `store`, `volume`, `hf`, `preset` | where the node gets the model |
| `reach` | `direct`, `jump`, `relay`, `outbound`, `function` | how the control host reaches the node |

The device family (`cuda`, `rocm`, `ascend`, `ppu`) comes from the card. It picks the
container's device flags (`--gpus` for CUDA, `/dev/kfd` and `/dev/dri` for ROCm, the
`davinci` devices and driver mounts for Ascend), the visible-cards variable
(`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ASCEND_RT_VISIBLE_DEVICES`) and
`YETO_DEVICE_FAMILY`. An inventory's own `docker_args` replace the derived flags.

An inventory file with profile fields:

<!-- doc-test: inventory -->
```yaml
nodes:
  - name: h100-a
    host: ubuntu@h100-a
    device_family: cuda
    cpu_arch: x86_64
    card: H100
    cards_per_node: 8
    hbm_gib: 80
    host_memory_gib: 1800
    runtime: docker
    container: docker
    model_source: store
    paths: {work: /data/yeto/work, models: /data/yeto/models, cache: /data/yeto/cache}
  - name: mi355x-a
    host: ubuntu@mi355x-a
    device_family: rocm
    cpu_arch: x86_64
    card: MI355X
    cards_per_node: 8
    hbm_gib: 288
    host_memory_gib: 2700
    runtime: docker
    container: docker
```

`yeto.launch.node_inventory.load_profiles(path)` reads it; every check of the
plain inventory applies.

## The engine image catalog

The image a container sampler runs is chosen per node by
`yeto.launch.engine_images.resolve(profile, model, engine_pref, load=, precision=,
rollout=, image=)`: the first catalog image whose device family, card
architecture, CPU architecture, engine, model recipe, reload mode, precision and
rollout fields fit the node. When nothing fits, it raises `ImageIncompatible` with
each image's reason and what would work instead. The sky route checks every offer
of the pool this way before the pool starts (and in `pool --dry-run`, one line per
offer); the node runtime checks
again before it starts a container (it exits 8, `image`, in its exit record). A pool that
names an image (`image`, or `engine` / `platform`) gets that image checked the same
way; an image outside the catalog is used unchecked.

<!-- doc-test: catalog -->
```python
from yeto.launch import engine_images
from yeto.launch.node_inventory import NodeProfile

h100 = NodeProfile.for_card("h100-a", "H100", 8, hbm_gib=80, container="docker")
chosen = engine_images.resolve(h100, None, "vllm")
print(chosen.image.name, chosen.ref, chosen.python)

ascend = NodeProfile.for_card("npu-a", "910B4", 8, hbm_gib=64, container="docker")
try:
    engine_images.resolve(ascend, None, "sglang")
except engine_images.ImageIncompatible as refused:
    print("refused:", str(refused)[:120])
```

`engine_images.spec_needs(spec)` turns a sampler spec into the arguments above
(`engine.adapter`, `engine.server.in_place_reload`, `rollout_fields`, the model).

## Joining a machine by hand

A machine opened by hand joins a pool with a token. The operator writes its
assignment (`yeto.node.assign.assign_manual`), gives the machine the token, and
runs the node command there. The example uses a store directory and the code of
this checkout; on a real pool the store is `s3://<bucket>/runs/<prefix>` with an
env file holding its credentials (`--env-file`).

<!-- doc-test: manual -->
```python
import json, os
from pathlib import Path

from yeto.launch.node_inventory import NodePaths, NodeProfile
from yeto.node import code as node_code
from yeto.node.assign import assign_manual, node_command
from yeto.rl.engine.object_store import DirectoryObjectStore

demo = Path(os.environ.get("DEMO", "/tmp/yeto-node-demo"))
objects = DirectoryObjectStore(demo / "store")
profile = NodeProfile.for_card("h100-a", "H100", 8, hbm_gib=80, container="docker",
                               paths=NodePaths(work=str(demo / "work"), models=str(demo / "models")))
spec = {"sampler_id": "demo-0", "store": "__STORE__",
        "engine": {"adapter": "vllm-server", "server": {"port": 8000}},
        "prompts": {"path": "/work/prompts.jsonl", "groups_per_round": 2}}
code = node_code.publish_code(objects, *node_code.tree_code_tar())
member, assignment, token = assign_manual(objects, {"run_prefix": "demo"}, shard="demo-0", shards=["demo-0"],
                                          profile=profile, code=code, run_id="demo", spec=spec)
(demo / "token").write_text(token)
(demo / "token").chmod(0o600)
(demo / "ticket").write_text(assignment.ticket)
print(node_command(assignment, str(demo / "store")), "--token-file", demo / "token")
```

### What the token protects, and what it does not

The token binds a machine to one ticket; it is not an authentication.

- The store holds only the token's salted sha256. A node must present the token
  to read its assignment, so a machine cannot start on another ticket's
  assignment by mistake, and the token never appears in the store or on a command
  line (it travels on stdin or in a `0600` file).
- The node refuses an assignment that names another ticket than the key it sits
  under (a document copied between tickets) and an assignment past its deadline
  (`expires_at`). A hand-opened machine's token expires after a day by default
  (`assign_manual(..., token_ttl_s=...)`, None for no deadline); issue a new
  assignment to rejoin later. A pool-ordered machine's token has no deadline: the
  pool hands it over at each start and writes a new assignment and token on the
  next one.
- The token stays with the runtime: the sampler's environment never carries
  `YETO_NODE_TOKEN`.
- Anyone who can write the run store can write an assignment, and anyone who
  holds a ticket's token can read that ticket's assignment (spec, code hash, model
  plan; no store credentials are in it). The store credentials are the security
  boundary: give them only to machines you trust.

On the machine (here: a dry run, which prints the plan and starts nothing):

<!-- doc-test: dry-run -->
```bash
DEMO=${DEMO:-/tmp/yeto-node-demo}
python -m yeto.node run --assignment "$(cat "$DEMO/ticket")" --store "$DEMO/store" \
  --token-file "$DEMO/token" --skip-preflight --dry-run
```

The plan shows the `docker run` the runtime would start: `--gpus all` and
`CUDA_VISIBLE_DEVICES` for this CUDA node, the catalog's CUDA vLLM image, the
node's directories mounted at `/work`, `/models` and `/root/.cache`. Without
`--dry-run` the runtime boots and the store gets the node's records.

`--profile <file>` replaces fields of the assignment's profile with the machine's
own (a JSON or YAML map, e.g. `{"paths": {"work": "/data/work"}}`); `--work`,
`--models` and `--cache` replace one path each.

## Pool routes

| route | sampler start | rollback |
|---|---|---|
| SSH + docker pool on `provider.kind: sky` | node runtime (default) | `"node_runtime": false` at the top level of the pool config |
| SSH inventory pool (`provider.kind: ssh`) | node runtime (default) | `"node_runtime": false` in the `provider` section (or the pool config) |
| National Compute | its own docker route | (shadow only; `node_runtime: true` is refused) |
| Daytona | its own process route | (shadow only; `node_runtime: true` is refused) |

On the ssh pool the node runtime keeps the sampler's environment as the route's own
start leaves it: the yeto tree goes in front of the inherited `PYTHONPATH` (the
venv's `pythonpath`, its `source` scripts' entries and an image's own entries such
as CANN's stay), and `provider.env` is applied last, as before.

On the sky route the node's profile comes from the offer's accelerators
(`H100:8` -> family cuda, 8 cards) with the directories under the ssh user's
home (`~/yeto/{work,models,cache}`); the pool config's `node_profile` section
replaces any field. The container keeps the image's own `PYTHONPATH` behind the yeto
tree, gets the docker route's tuning variables (release download, start spread)
and the pool's `agentic_env_file` as a second `--env-file`:

<!-- doc-test: sky-pool -->
```json
{
  "target": 1,
  "slots": [0],
  "shard_base": "smp",
  "run": "demo",
  "spec": "sampler.json",
  "run_prefix": "demo",
  "r2_env_file": "store.env",
  "provider": {"kind": "sky", "resources": {"accelerators": "H100:8", "infra": "nebius"}},
  "model_pull": {"src_key": "models/my-model", "dest": "/models/my-model"},
  "node_profile": {"paths": {"work": "/data/yeto/work", "models": "/data/yeto/models"}},
  "node_runtime": true
}
```

`model_pull` becomes the node's model plan: the node pulls `src_key` from the run
store into `dest` (a `/models/...` path, as the spec names it) before the sampler
starts.

A sky VM is normally autodowned by sky itself at `provider.hard_cap_minutes`
(default 24 h) after its launch. On a cloud sky cannot autodown (Verda; also
Hyperbolic, Prime Intellect and Yotta in SkyPilot 0.14.3) the provider refuses the
offer unless `"allow_no_autodown": true` is set in the `provider` section. With it,
the VM has no cloud-side hard cap: the pool releases it at its lease end (order +
`hard_cap_minutes`, with a replacement ordered before; 15 minutes past it the
ticket reads ended, so a paused member is released too), and every order prints a
WARNING. If the pool process is gone, `yeto cleanup --run <run> --include-pools`
releases it by exact name from the run's resource ledger; it refuses while a pool
loop of the run still holds its lock (stop that loop with `pool-stop` first), and
`yeto down` leaves pool machines to the pool.

A Verda offer that sets no `image_id` orders `24.04.cuda12.9.docker` (SkyPilot 0.14.3's
own Verda default image is no longer offered there); an offer's own `image_id` is kept,
and `pool` / `pool --dry-run` print the VM image of every offer. When sky's cluster handle
names no ssh user (Nebius), the node is reached as the user sky's template for that cloud
logs in as (`ubuntu` on Nebius and AWS, `root` on Verda, ...; on OCI by the image), with
a WARNING naming the user and where it came from.

National Compute (the MI355X vLLM pool) and Daytona keep their own sampler starts.
`yeto.node.shadow` renders the node runtime's launch on the same node and lists
each difference from the route's own (device flags, mounts, env files,
environment, image, interpreter, sampler, spec path); a route switches only when
that list is empty. On a National Compute node it is down to the node's whole model
directory at `/models` (the route mounts each model on its own), the code tree (the
route mounts it at `/yeto` with `-e PYTHONPATH=/yeto`) and the store URI (the route
builds it on the node from the env file's bucket).

## Node health

`yeto preflight` checks a node before a run. While the run goes on, the node is
checked again every 60 s (`yeto.hw.node_health`), so that a node whose GPU or
fabric broke is taken out before NCCL times out. The check reuses the preflight's
own parsers (`yeto.hw.host_probe`).

Checkers are registered by device family. Only `cuda` has one so far. On any other
family nothing extra runs: heartbeats, probes and island records stay as before.

### What is checked

One shell call reads, in this order:

| item | source | bad when |
|---|---|---|
| XID | `NVRM: Xid` lines of `dmesg` (else `journalctl -k`; the last 1 MiB), with the boot id | a hardware XID since the previous check: 48, 61, 62, 64, 74, 79, 95, 119, 120, 140. Application XIDs (13, 31, 43, ...) are only recorded |
| GPUs | `nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total,ecc.errors.uncorrected.volatile.total` | no GPU at all, fewer GPUs answer than the inventory's `cards_per_node` (a sampler: than the first check saw), a GPU nvidia-smi cannot open, a GPU's memory shrank, or nvidia-smi cannot talk to the driver |
| ECC | the same query | the volatile uncorrectable ECC count grew since the previous check (a count that went down is the new baseline) |
| NVLink fabric | the `Fabric` blocks of `nvidia-smi -q` | a GPU that has a fabric state is not `Completed/Success` (a PCIe card's `N/A` passes) |
| fabric manager | `systemctl is-active nvidia-fabricmanager` and its `NRestarts`, what shows the node's own NVSwitch, the boot id | the node must run it (it was active in this boot, or the node has NVSwitch at the first check) and it is not `active`, or it restarted since the previous check; see below |
| IB / RoCE ports | `ibstat` (`ibv_devinfo -v` only when ibstat fails) | a port named by `NCCL_IB_HCA` is missing, not `Active`, or slower than at the first check |

The XIDs and the ECC count only count from the second check on. What the kernel log
holds at the first check is history: the preflight judged the node then.

The fabric manager is judged by whether the node has NVSwitch of its own, and by
what it did in this boot (the boot id of the XID section).

Whether the node has NVSwitch of its own, in this order:

| finding | NVSwitch |
|---|---|
| the driver lists NVSwitch devices in `/proc/driver/nvidia-nvswitch/devices`, or `/dev/nvidia-nvswitch<N>` exist, or `lspci` shows an NVIDIA PCI bridge (class `0680`) | yes (Hopper HGX) |
| a GPU name starts with `GB` | no (an NVL72-class compute tray: its NVSwitches sit in the switch trays) |
| no GPU has a real NVLink fabric state (`N/A`, or no `Fabric` block) | no (PCIe) |
| the VPD of a Mellanox (`15b3`) PCI function, any function, names the NVLink switch management port (`SMDL=SW_MNG`) | yes (HGX B200 / B300) |
| every GPU is a B200 / B300 (not GB, not PCIe) and has a real fabric state | yes (HGX B200 / B300, without root; a VPD without the port does not overrule it) |
| the VPD of every Mellanox function read and none names it | no |
| anything else, or the fabric state did not read | not determined |

On HGX B200 / B300 the NVSwitches are not PCIe devices of the host. They sit behind a
ConnectX bridge that the fabric manager reaches through NVLSM (NVIDIA Fabric Manager
User Guide). The host shows no procfs entry, no device file and no class `0680`
bridge; amazon-eks-ami PR 2840 reports the same on p6-b200. Hence the VPD rule, and
the GPU-name rule for a user that cannot read the VPD (it reads as root only). Each VPD
read is bounded to 1 s (`timeout 1`), all of them to 3 s; the rest are not read, and
without a `timeout` command none is. A node whose NVSwitch is not determined is never
taken for one without. A section cut off after the state line says `FM section timed
out after reading state`. Only a device, a PCI bridge or the VPD port is hard evidence
of NVSwitch on the node; the GPU names are not.

The state of the fabric manager, its process first:

| systemd | `nv-fabricmanager` process | state |
|---|---|---|
| unit `active` | not looked for | `active` |
| anything else | runs | `active` (a GPU Operator driver container runs it, maybe next to a stale host unit that is disabled, inactive or failed) |
| unit `loaded`, `inactive` / `failed` | none | `inactive` / `failed` (it stopped) |
| no unit (`not-found`), or `inactive` / `failed` of a unit that is not loaded (`masked`) | none | not running |
| any but `loaded` | not seen (`/proc/1/comm` not readable: `hidepid`) | not read |

`systemctl is-active` prints `inactive` for a unit that does not exist, hence the
LoadState. The check reads `systemctl show -p LoadState` and cuts the prefix itself
(`--value` needs systemd 230). It counts the processes by `/proc/<pid>/comm`, which the
kernel cuts to 15 characters (`nv-fabricmanage`).

A fabric manager that is not running, at the first check:

- bad when the node has hard evidence of NVSwitch (the processes were seen: it is gone);
- ok when there is no such evidence and every GPU's fabric is `Completed/Success`: the
  fabric is managed off-host (a shared NVSwitch guest VM, whose fabric manager runs on
  the service VM; a guest whose NVSwitch cannot be determined);
- unread otherwise (the fabric state judges the node).

The rules over the checks:

- No NVSwitch of its own (PCIe, a GB tray, VPD without the switch port): the fabric
  manager is never judged, even when it was active earlier in the boot. A tray may
  have the unit installed, and it fails there. A fabric that is not `Completed` makes
  the node bad by its own reason.
- Once the fabric manager was active in a boot (its unit, or its process), every later
  check of that boot that does not read `active` (`inactive`, `failed`, `activating`,
  `deactivating`, not running) is bad,
  until it is active again. This also holds when NVSwitch was not determined: a fabric
  manager that ran has a fabric to manage. One that stays down makes the node `lost`
  after two checks, even while the fabric still reads `Completed`.
- The first check of a new launcher has no memory (it lives in the checker, not on
  disk). On a node with NVSwitch, any state but `active` is bad at once, and that boot
  is remembered. A check where the state did not read or NVSwitch was not determined
  judges nothing and is not counted as the first one. The same rule, on one check
  alone, is `yeto.hw.node_health.fabric_manager_absolute`: `ok`, `bad` or `unread`.
- On a node that must run it, an `NRestarts` that grew since the previous check of the
  same boot is bad for that check. A unit in a restart loop can read `active` at every
  check; a loop that goes on makes the node `lost`. `NRestarts` counts only the
  restarts systemd makes itself, so this needs a `Restart=` setting on the unit (a
  drop-in such as `Restart=on-failure`); restarts by hand do not count. A fabric
  manager without its unit has no `NRestarts`: its process's pid and start time (field
  22 of `/proc/<pid>/stat`) stand in. Another process in the same boot is one restart,
  bad for that check.
- A new boot id forgets all of it. A fabric manager never active since the reboot is
  judged as before: one bad check when it was active before the reboot (on a node with
  NVSwitch), then the fabric state judges the node. A fabric manager that does not
  start leaves the NVSwitch fabric unfinished on an HGX node.
- A systemctl that timed out says nothing. Without a boot id, a stop is bad once.

Not yet seen on our own hardware: the VPD keyword on an HGX B300 node, and that a
GB300 tray's GPU names start with `GB`. The rules above follow the public sources.

Known limit: on an HGX B300 node checked as a user that cannot read the VPD, there is no
hard evidence of NVSwitch, only the GPU names. If a GPU Operator driver container runs
the fabric manager and it dies after the fabric reads `Completed`, a check that never
saw its process takes the fabric for one managed off-host. New CUDA launches fail once
the fabric manager is gone, so the trainer reports the error itself. A check that saw
the process earlier in the boot does catch it (the per-boot memory).

A kernel log without timestamps (dmesg's default has them) is followed by its last
line, how often that line ends the log, and the line before it, so lines that rotate
out of the log's head do not move the cursor. One limit is left: when the log holds
only one repeated XID line and a copy rotates out while the same XID repeats, that
repeat is missed. Memory and
ECC baselines follow each GPU's PCI address, so a GPU that changes index keeps its
baseline.

A single-node job (a sampler) is judged on the GPU items only. Its ports are
recorded as `notes`. A multi-node trainer island is also judged on the ports of its
`NCCL_IB_HCA`. The island's `env` overrides the inventory node's `env`, as for the
job itself. Without `NCCL_IB_HCA` on an `ib` / `roce` fabric, the island is judged
on the ports NCCL picks by the fabric's link layer. This is the preflight's own rule;
the island is bad only when none of these ports is active. Set `NCCL_IB_HCA` to the
compute HCAs as `yeto preflight` recommends, so that each of them counts.

### Time bound

- Every section of the check runs under `timeout -k 5 10`: TERM at 10 s, KILL 5 s
  later.
- The whole check runs in the background on the node and is killed (its process
  group) past 45 s. That is less than the 60 s period.
- A normal check takes about 1-5 s, mostly `nvidia-smi -q`.
- A killed check is a read failure, except that a completed XID section still counts.
- On a trainer island, the check rides on the status SSH call. That call waits at
  most 75 s per node; the fleet loop asks the nodes one after another.
- If the SSH call times out after the node printed its trainer state, the island
  reads that state and the check is a read failure. If nothing came back, the node
  counts as not answering.

### Verdict and events

- `ok`: the check passed. One good check clears a bad streak.
- `bad_once`: one bad check. Nothing is done yet.
- `lost`: two bad checks in a row. The node is treated as a lost node.
- `read_failed`: the check did not read. It counts neither way: a bad streak goes
  on after it. Its reason says which kind it is:

  | reason | what happened |
  |---|---|
  | `nvidia-smi: timed out after 10s`, `nvidia-smi: not found`, `nvidia-smi: exit <rc>: ...` | nvidia-smi did not answer, and the XID section shows no new hardware XID |
  | `nvidia-smi: no output (the check did not run)` | nothing came back (an SSH call that failed) |
  | `node health output incomplete (no <sections>)` | the output was cut off: a section is missing or has no exit code |
  | `node health check timed out after 45s (killed)` | the check ran past its limit on the node and was killed |
  | `node health check cut off: its SSH call ran past its timeout` | the SSH call timed out after the node printed its trainer state |
  | `node health check skipped: the previous one still runs (a tool stuck on the node)` | an earlier check could not be killed (a tool stuck in the kernel); no new one is started while that same process lives (its pid and its start time both match: a reused pid or a reboot does not count) |
  | `IB tools unread (ibstat / ibv_devinfo)` | a multi-node job whose GPUs pass, but its ports could not be read |
  | `health check stale: the last one <N>s ago` | the newest check is older than three periods (the check stopped) |

Three events are written:

- `node_health`, once when a bad streak first reaches two checks, with its reasons
  (e.g. `GPU 5 XID 79`, `mlx5_3 port DOWN`). A read failure inside the streak does
  not start it again.
- `node_health_read_failed`, once when checks stop reading, and again only after a
  check read in between.
- `node_health_degraded`, once per node, for what the check cannot judge: the kernel
  log is not readable, the IB tools did not read, or a multi-node job has no
  `NCCL_IB_HCA` and no RDMA fabric. It is also written once per streak of five read
  failures in a row. It never makes a node lost; a node that stops answering is
  judged by the SSH reply.

### Where it runs

- **Samplers on the node runtime.** While the sampler runs, `python -m yeto.node
  run` checks the node every `--health-every-s` seconds (default 60; `0` turns it
  off). It writes the newest check to `<work>/sampler/<shard>.health.json`, and the
  sampler adds it to each heartbeat as `health`. A pool that reads the store sees it
  in its probe. The ssh pool (`provider.kind: ssh` on the node runtime, its default)
  removes an earlier check at each start and reads the file in its own probe call;
  it ignores a check of another ticket or assignment. A `lost` node is handled as a node that stopped
  answering: `node_health`, then `node_lost`, the shard is fenced, a replacement
  starts, and the node's slot is held for `provider.lost_node_hold_s` while a spare
  is free.
- **Resident `ssh:` trainer islands.** The check rides on the SSH call that asks
  each node for its trainer state every 60 s.
  - When a node turns `lost`, `SshIslandOps._node_health_lost(rec, rank, reasons,
    reading)` is called. It is called once per turn, after the poll released the
    ops' lock. `reading` carries the check's facts; the node's whole record, notes
    included, is `rec.health[rank].record()`.
  - By default it fails the island as a lost node: every node is released, and the
    fleet's recovery takes over.
  - A relaunch on the same nodes starts a new bad streak. The baselines stay, and so
    does the record until the next check.
  - Each node's newest check is in the island's record
    (`fleet/trainers/<k>/state.json`, field `health`), and the events go to the
    fleet log.

### Turning it off

The check is on wherever the device family has a checker. `node_health: off` turns
it off:

- in the inventory, at the top (every node) or on one node (`node_health: auto`, or
  `on`, turns it back on for that node);
- in the ssh pool's provider section (`provider.node_health`). That value wins over
  the inventory's for the pool's nodes, and the node runtime starts with
  `--health-every-s 0`.

When neither the inventory nor the provider says `node_health`, the dry runs print
nothing about it (a DEBUG log line says whether it is on). When one of them says it,
the island dry run has a `node_health` field (`on (cuda)`, `off`, or
`on (cuda), off on [...]`), and `pool --dry-run` prints `ssh pool: node health: ...`.

To run one check by hand on a node (`--dry-run` prints the script and runs
nothing):

<!-- doc-test: node-health -->
```bash
python -m yeto.hw.node_health --hca "=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" --multi --dry-run
python -m yeto.hw.node_health --family npu --dry-run
```

Without `--dry-run`, the command prints the check's record (verdict, reasons,
facts) and exits 0 when the verdict is `ok`.
