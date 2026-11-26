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
| `inventory.own-nodes.yaml` | two 8 x B300 trainer nodes (`roles: [trainer]`) and three 8 x H100 sampler nodes (`roles: [sampler]`) |
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

The dry run orders nothing. It runs:

- each pool's own dry run (`python -m yeto.launch.nc pool --dry-run`): the
  offers it would order with their prices, the sampler spec checks, the
  model in the bucket;
- the trainer's launch dry run (`yeto launch --dry-run`): the node layout and
  ranks on the inventory nodes;
- the checks between the pools and the trainer: the rollout fields the
  samplers record against the ones the trainer trains on (`top_logprobs_num`,
  routed experts), and, in a fleet, the release form each pool's engine loads
  (weight format, reload mode);
- the egress check of your nodes: over SSH, on every own node of the run,
  whether it reaches the run store's object API and the trainer image's
  registry from its own network. The probe is an HTTPS request with `curl`:
  any HTTP answer counts (401, 403 and 404 included: the network path works),
  and the certificate is not verified, so a proxy that intercepts TLS passes.
  The request goes through the node's proxy: the inventory env's
  `HTTPS_PROXY`/`https_proxy` (and the other proxy variables), else, for the
  registry, the docker daemon's proxy (`docker info`, what `docker pull`
  uses). A node without `curl` gets a TCP connect, or, when a proxy is set, a
  `[WARN]` that the path is not checked. One line per node and endpoint, in
  the preflight's format; a node that does not reach one refuses the run up
  before anything is ordered, naming the node and the endpoint:

  ```
  dry run: own nodes own-b300-0, own-b300-1: checking over ssh that they reach store https://<endpoint>, registry https://<registry>/
  dry run:   [PASS] own-b300-0 reach_store: the store endpoint https://<endpoint> (s3://<bucket>/runs/<run>): http 403
  dry run:   [PASS] own-b300-0 reach_registry: the registry endpoint https://<registry>/ (<image>): http 401
  dry run:   [FAIL] own-b300-1 reach_store: own-b300-1 cannot reach the store endpoint https://<endpoint> (s3://<bucket>/runs/<run>) (rc 6): curl: (6) Could not resolve host ... -> open the node's egress to it (firewall, proxy: https_proxy in the node's environment, DNS)
  dry run:   [PASS] own-b300-1 reach_registry: the registry endpoint https://<registry>/ (<image>): http 401
  dry run:   3 pass, 0 warn, 1 fail, 0 skip
  cluster preflight failed: nothing ordered
  ```

  The store endpoint is `--rl-object-store-endpoint`, else the store's
  endpoint variables (`YETO_RL_STORE_S3_ENDPOINT_URL`, `AWS_ENDPOINT_URL`, the
  store profile's `endpoint_url`), else AWS S3 in the store's region. The
  registry is the one of `--rl-image` (else the inventory's image, else the
  pinned default). A node that takes no SSH fails its `ssh` check the same way.
  `--skip-preflight` skips every preflight, this one included.

The spend line counts the cloud parts only: `trainer.usd_per_h` is 0 for a
trainer on own nodes, and each pool is counted at its node price. It is
logged for a run with own nodes (an own trainer or an own sampler pool) whose
pools are not National Compute ones; a run with an NC pool logs no spend line:

```
dry run: spend estimate $1.50/h = pools $1.50/h (sky-l40s (sky) 2 x $0.7492) + trainer $0/h (own nodes: ssh:2x8xb300); over the 6 h wall clock $8.99 (budget.usd $30)
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
group); put the cloud island on the same card as your nodes, or declare one
group with `--rl-compat-group`.

## Own sampler pools

Your nodes serve as a sampler pool with `provider.kind: ssh`
(`yeto/launch/ssh_sampler_pool.py`): the pool takes the inventory nodes named in
`provider.nodes` (or every node with `roles: [sampler]`) as its slots, one
sampler per node, and orders nothing.

```bash
yeto run up --config run.own-sampler-pool.json --dry-run
```

- **Node runtime.** With `"node_runtime": true` the node runs
  `python -m yeto.node run` (docs/NODE_RUNTIME.md), which starts the sampler:
  its ready record, heartbeats and exit go to the run store, as on every other
  source. `"node_runtime": false` keeps the pool's own sampler start.
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
- **Egress.** The dry run checks over SSH that every pool node reaches the run
  store (the endpoint of `r2_env_file`) and the registry of its image, as for
  an own trainer.
