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
of the pool this way before it orders the first machine; the node runtime checks
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
| SSH + docker pool on `provider.kind: sky` | node runtime (default) | `"node_runtime": false` in the pool config |
| SSH inventory pool (`provider.kind: ssh`) | its own start; node runtime opt-in | `"node_runtime": true` in the `provider` section (or the pool config) turns it on |
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

National Compute (the MI355X vLLM pool) and Daytona keep their own sampler starts.
`yeto.node.shadow` renders the node runtime's launch on the same node and lists
each difference from the route's own (device flags, mounts, env files,
environment, image, interpreter, sampler, spec path); a route switches only when
that list is empty. On a National Compute node it is down to the code tree (the
route mounts it at `/yeto` with `-e PYTHONPATH=/yeto`) and the store URI (the route
builds it on the node from the env file's bucket).
