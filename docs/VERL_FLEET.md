# verl fleet: trainer islands on verl FSDP2

An island fleet is one `run.json`, one `yeto run up`, trainer islands merged by one syncer,
and each island with its own sampler pools ([MILES_RL.md](MILES_RL.md), "Island fleet";
[FLEET_TRAINER_ISLANDS.md](FLEET_TRAINER_ISLANDS.md) for islands added, removed and moved
while it trains). Every island of a fleet can train with the verl backend instead of Miles
(`--rl-backend verl` in `fleet.trainer_argv`). The verl trainer is FSDP2 without
TransformerEngine, so it also runs on sm80 and sm89 cards (A100, L40S), where the Miles
trainer's default fused attention is refused.

## Admission

`yeto run up --dry-run`, the launch preflight, and the run-time `yeto fleet island add` /
`move` accept a verl fleet only when all of the following hold. Otherwise they refuse it
before anything is ordered.

| Rule | What is refused |
|---|---|
| Every island trains with the same backend, and that backend declares `fleet_full_parameter` (Miles and verl both do). | A fleet that mixes Miles and verl islands. A verl layout carries no shard identity, a Miles layout names every rank's shard, so the two do not share a syncer session. |
| Full parameters: `--tuning full` and no `--rl-expert-selection`. | Partial experts (`fleet_partial_experts` is a Miles capability) and `--tuning lora`. |
| No CompactionRL: no pool sampler spec with `agentic.compaction`, and no `advantage.gae_variant cross_segment_per_sample`. | The preflight names the island and the pool. A verl trainer cannot train the per-segment samples of a compacted trial as one trial. |
| The layout hash and fragment count come from the offline planner. | `--rl-parameter-layout-sha256 auto` (verl has no probe on the island). |
| The image is the pinned verl image: `verl-build:<VERL_COMMIT>` on Modal, the registry image of the same recipe elsewhere. | A `verl-build` image of another commit, named by its `--gpu` island. |
| The trainer card is sm80 or newer. | A T4 or V100 island. |

The planner reads the model's config on the meta device (no GPU, no weights). The layout
hash binds the matrix merge, so plan with the one the run actually uses. A fleet
(`--rl-sync-preset decoupled` or `full-parameter`) merges with `rda` in the default async
mode, and with `avg` under `--rl-sync-mode strict` or `--rl-sync-full-parameter-rounds sweep`;
`--rl-sync-matrix-merge` overrides both. The dry run prints the value it uses in each
island's command (`--rl-sync-matrix-merge rda`, in the launch log of `run up --dry-run`);
an island command without that flag merges with `avg`.

```bash
python -m yeto.rl.adapters.verl.full_parameter_plan --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca --fragments 2 --matrix-merge rda
# -> "launcher_flags": "--rl-parameter-layout-sha256 <sha256> --fragments 2"
```

Put those two flags in `fleet.trainer_argv` together with `--rl-backend verl`. `yeto run up`
(and its dry run) computes the same layout and refuses a hash of another matrix merge or
fragment count before anything is ordered. The refusal names the matrix merge it planned
with, and says so when the pinned hash is the layout under the other merge.

The check needs the model's `config.json`. Without it in the local files or Hugging Face
cache, the launcher fetches that one file: from the model's store URI or
`--model-store-source`, then from the Hub at `--model-revision`, then from the
`--modal-model-volume` snapshot. When none of them gives it, `yeto run up` and
`yeto fleet island add/move` refuse a fleet that pins a layout hash, because each island
would check the pin only after its paid boot. Their dry runs print
`WARN: verl layout hash not verified offline: <why>` instead. To verify, put the config in
the cache (`hf download <model> config.json --revision <revision>`) and run again; to go
on unverified, set `YETO_PREFLIGHT_LAYOUT=warn`.

## How a verl fleet differs from a Miles fleet

| | Miles fleet | verl fleet |
|---|---|---|
| Training forms | full parameters, partial experts | full parameters only |
| CompactionRL (`agentic.compaction`) | yes | refused by the preflight |
| Islands of both backends in one fleet | no (refused) | no (refused) |
| Samplers | external pools; Miles sampler islands inside the trainer launch | external pools only (in-launch verl sampler islands cannot follow a trainer) |
| Layout hash | `auto` (probed on the island) or from the offline planner | from the offline planner only |
| Releases | each island releases its own local versions to `trainer<k>/`, and its pools follow it | the same; each verl island launch gets `--rl-release-mode per_learner`, `--rl-sample-store <store>/trainer<k>` and `--rl-sample-lanes 1` |
| Release cadence | every N local versions; skip-busy and merged-mode cadences | every N local versions (`--rl-release-every`); no skip-busy or merged-mode cadence |
| Deterministic stop (a trainer breaker, a strict check, a syncer refusal) | exit 11, not relaunched | exit 11, not relaunched |
| Prompt width of the trainer | from the pool | `--seq-len` minus `--rollout-max-response-len` |
| Trainer cards | sm90 and newer with the default fused attention (another training attention backend only warns) | sm80 / sm89 / sm90 |
| Trainer image off Modal | the pinned Miles registry image | the pinned verl registry image (below) |

## Images

| Where the island runs | Image |
|---|---|
| Modal | `verl-build:<VERL_COMMIT>`. Modal builds it from the pinned recipe (`yeto/rl/adapters/verl/image.py`) on first use and caches it. |
| A sky cloud (AWS, Nebius, ...) | The registry image of the same recipe, `yeto-verl-fsdp@sha256:<VERL_REGISTRY_DIGEST>` (tag `fsdp-<commit8>-<recipe8>`). It replaces `verl-build` automatically, so `fleet island move --gpu aws:...` needs no image flag. |
| An `ssh:` island with docker nodes | The registry image, passed explicitly as `--rl-image`. `verl-build` is refused there. |

There is no default registry for the verl registry image. Name the registry that holds the
pinned digest:

```bash
# ECR (private): a 12-hour login from the AWS CLI
export YETO_VERL_IMAGE_REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com
export SKYPILOT_DOCKER_SERVER=$YETO_VERL_IMAGE_REGISTRY SKYPILOT_DOCKER_USERNAME=AWS
export SKYPILOT_DOCKER_PASSWORD="$(aws ecr get-login-password --region <region>)"

# GHCR: a public package needs no login; a private one needs a read:packages token
export YETO_VERL_IMAGE_REGISTRY=ghcr.io/<owner>
```

- The login goes into the sky task's `secrets`, never its `envs` or its scripts. A private
  Miles image takes the same route (`--rl-image-private` reads `~/.docker/config.json`
  instead). A Modal island gets no login.
- Keep the registry in the islands' region. The image is about 8.6 GB compressed (as stored in the registry).
- Before anything is ordered, the launch asks the registry for the image's manifest (a
  `HEAD`, nothing downloaded) with that login. A missing image, or one the login cannot
  read, is refused in `--dry-run` and in the launch preflight. A registry that cannot be
  reached only prints a warning. `YETO_IMAGE_MANIFEST_CHECK=off` skips the check (for a host
  that cannot reach the registry while the nodes can).
- Without `YETO_VERL_IMAGE_REGISTRY`, a verl island that has to pull its image is refused.
- To put the image into another registry, either copy the pinned digest byte for byte
  (`crane copy <registry>/yeto-verl-fsdp@sha256:<digest> <other>/yeto-verl-fsdp:<tag>`),
  or rebuild it on a linux/amd64 docker host with
  `DEST_REPO=<registry>/yeto-verl-fsdp docker/verl-fsdp/build.sh --push`. A rebuild has a
  new digest, which must be pinned (`VERL_REGISTRY_DIGEST`). `docker/verl-fsdp/Dockerfile`
  is generated from the Modal recipe, and a test fails when the recipe changes without a
  rebuild.

## Cards

- The verl FSDP2 trainer may use sm80, sm89 and sm90 cards (`VERL_FSDP_CARD_ARCHES` in
  `yeto/rl/adapters/verl/pins.py`). A card below sm80 (T4, V100) is refused before any order.
- Cards a verl trainer has already trained on are recorded in `VERL_FSDP_VERIFIED_CARDS`,
  each with its evidence (today: H100 and L40S). Any other card prints an `UNVERIFIED`
  warning but is not refused.
- An L40S island trains Qwen3-0.6B with full parameters at a peak of 13.4 GB of its 46 GB.
  On AWS, sky's default of at least 8 vCPUs per island picks `g6e.2xlarge` (64 GiB of host
  memory) for `aws:1xL40S`.
- The sampler pools pick their own cards. One L4 serves Qwen3-0.6B.

## Commands

```bash
yeto run up --config run.json --dry-run    # the plan: each island's backend, learner id, trainer<k>, pools
yeto run up --config run.json

# while it trains (run up seeds the fleet's desired state; each command has a --dry-run)
yeto fleet island add    --config run.json --name i2 --gpu modal:1x1xl40s --usd-per-h 2.94 \
  --samplers samplers.i2.json
yeto fleet island remove --config run.json --name i1
yeto fleet island move   --config run.json --name i0 --gpu aws:1xL40S@us-west-2 --usd-per-h 2.3

yeto run down --config run.json
```

`samplers.i2.json` is the island's `samplers` list, in the same form as an `islands[].samplers`
entry of `run.json`.
