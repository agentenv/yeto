# Miles RL

Miles RL runs causal-language-model reinforcement learning inside independent
Miles islands and uses Yeto to synchronize their complete LoRA policies. Miles
owns SGLang rollout, reward evaluation, GRPO, and Megatron local training. Yeto
owns the cross-island LoRA boundary, the authoritative global checkpoint,
recovery identity, finalization, and PEFT export.

Two explicit synchronization presets are available:

| preset | role | global synchronization |
| --- | --- | --- |
| `strict-avg` | default correctness baseline | one complete LoRA fragment; every island waits for full-roster equal-weight FedAvg after every local round |
| `decoupled` | Decoupled DiLoCo RL | deterministic multi-fragment LoRA; local RL continues while exact-base fragment rounds are in flight |

Selecting `decoupled` is intentional. The launcher never infers it from the
model, island count, or generic DiLoCo flags.

The training objective (GRPO family, mismatch corrections, policy-loss
variants, critic algorithms) is described in
[RL_ALGORITHMS.md](RL_ALGORITHMS.md).

## Supported Boundary

The supported model and runtime contract is deliberately narrow:

| dimension | supported value |
| --- | --- |
| model | causal language model |
| tuning | LoRA only |
| learner | pinned Miles with Megatron-Core |
| rollout | colocated SGLang (`--rl-placement fixed-partition` is also available for LoRA and is used by the multi-node and elastic paths) |
| optimization | GRPO plus the declared GRPO-family mechanisms (see [RL_ALGORITHMS.md](RL_ALGORITHMS.md)); critic algorithms (PPO, VAPO, SAO, CompactionRL) run on a single island without outer sync only |
| island size | one or more nodes and one or more GPUs per node |
| parallelism | the recipe sets TP/PP (the multi-node and Flash-Next profiles use TP > 1); dense EP = 1; MoE EP may divide the island world size when LoRA tensors remain replicated |
| membership | fixed logical island IDs `0..M-1`, `M >= 1` |
| wire format | f32 |
| global weighting | complete roster, equal learner weight |

Model support follows the tensor contract rather than a Yeto model-name list.
The pinned Miles/Megatron-Bridge/PEFT stack must be able to load the model,
create canonical LoRA, map every trainable adapter tensor to standard PEFT
names and shapes, and apply those tensors on every trainer rank. Unsupported
models fail during initialization instead of selecting a model-specific
fallback.

The following remain outside this boundary:

- critic LoRA (`critic.param_mode=lora` is refused) or diffusion RL;
- TP>1 or PP>1 LoRA gather/scatter;
- expert-sharded LoRA tensors;
- trajectory migration across policy snapshots;
- same-fragment stale updates, dynamic quorum, or learner weighting;
- RDA, IsoLoCo, HeLoCo, delta correction, or broadcast blending;
- optimizer-moment federation;
- Miles' experimental fault-tolerant actor path;
- a cross-island controller, a storage system, or a generic recovery
  framework. (The read-only fleet dashboard, [DASHBOARD.md](DASHBOARD.md),
  exists and is not part of the RL contract.) An island-local, Yeto-side
  reconfiguration controller is allowed: it drives the island's own port verbs
  (rollout pool resize, `TrainerGroup.save_cut`/`restore_cut`/same-shape
  rebuild) at safe points and journals them; it does not control other islands
  and is not a general recovery framework. It is opt-in (`--rl-elastic`, see
  [Elastic reconfiguration](#elastic-reconfiguration)).

Local PPO and CyberGym-specific features are separate from this integration.
Miles custom generation and reward callables can still use existing tool or
environment runtimes without Yeto defining another trajectory format.

## Runtime Ownership

```text
                  complete policy snapshot
                           |
              +------------+------------+
              |                         |
       Miles island 0             Miles island 1       ... M
       SGLang rollout              SGLang rollout
       reward + GRPO               reward + GRPO
       Megatron train              Megatron train
              | canonical LoRA delta      |
              +------------+------------+
                           |
                    Yeto Rust syncer
             exact base, full roster, checkpoint
```

Miles remains responsible for rollout generation, complete GRPO groups,
reward and advantage computation, the local optimizer and scheduler, and the
normal trainer-to-SGLang weight publication. Yeto never implements a second
rollout or reward engine.

The Miles boundary provides:

- canonical replicated-LoRA export and apply on every Megatron rank;
- a post-train, pre-SGLang-publication external synchronization hook;
- preservation or reset of LoRA optimizer state as requested by that hook;
- a stop result from the hook that takes effect only after Miles completes one
  full `update_weights()` publication;
- normal final hook cleanup after the loop exits.

Native Miles and `strict-avg` keep their bounded rollout loops. Only the
`decoupled` external-sync path runs until the authoritative final cut asks it
to stop.

## Canonical LoRA Identity

Every trainable LoRA tensor is represented as contiguous CPU f32 in
deterministic PEFT name order. A canonical tensor spec contains name, shape,
dtype, and numel. Base-model revision, effective LoRA-config hash, and the
canonical layout hash bind the state to one model contract.

Multi-fragment RL separates two identities:

- `canonical_layout_hash` identifies the complete Miles/Yeto tensor schema and
  does not change with the fragment count;
- `sync_layout_fingerprint` identifies fragment membership, order, shapes,
  numel, and AVG merge modes, and is used by protocol HELLO and the syncer
  checkpoint.

For `decoupled`, tensors are sorted by `(-numel, name)` and placed into the
currently smallest of `P` bins, with fragment ID breaking ties. Every tensor
appears exactly once and every fragment uses AVG. The learner and exporter use
the same builder, so an altered order or membership fails by fingerprint.

## Strict-AVG

At committed policy version `v`, every island performs:

1. Apply the complete global LoRA to Megatron and SGLang.
2. Generate complete groups whose recorded weight version is exactly `v`.
3. Execute one Miles GRPO optimizer step.
4. Export the complete local LoRA and send `local - global` at base `v`.
5. Wait for every logical island and committed version `v + 1`.

The syncer uses f32, full roster, equal weight, outer LR 1, and momentum 0:

```text
theta_(v+1) = theta_v + mean(theta_i_v - theta_v)
            = mean(theta_i_v)
```

Applying a committed strict policy clears only LoRA optimizer state, preserves
the optimizer object, aligns scheduler progress, and refreshes the actor
backup. No island-local LoRA is published to the next rollout.

## Decoupled Preset

The public parameters are:

| symbol | flag | meaning |
| --- | --- | --- |
| `P` | `--fragments` | deterministic LoRA fragment count, at least 2 |
| `tau` | `--pipeline` | distinct fragment rounds allowed in flight, `1 <= tau <= P` |
| `H` | `--local-rl-rounds-per-sync` | minimum local optimizer steps between valid pushes for the same fragment, at least 2 |
| `N` | `--total-steps` | complete fragment sweeps |
| `T` | internal | outer fragment steps, exactly `N * P` |

The launcher fixes the rest of the algorithm:

```text
quorum                 = M
grace_ms                = 0
max_base_lag            = 0
learner_weight          = equal
fragment_pattern        = binpack
merge_mode              = AVG for every fragment
outer_lr                = 0.7
outer_momentum          = 0.9
delta_correction        = none
merge_alpha             = 0
wire_dtype              = f32
checkpoint_every        = 1
optimizer_steps/rollout = 1
```

`--experimental-rl-sync` cannot be combined with this preset.

### Policy snapshots

A rollout uses one complete, immutable snapshot:

```text
PolicySnapshot {
    rollout_id
    fragment_versions[P]
    policy_hash = sha256(complete canonical f32 LoRA)
}

SGLang token = yeto:<rollout_id>:<policy_hash>
```

The event tape maps the token to the full fragment-version vector and both
layout identities. Missing, stale, malformed, or mixed trajectory tokens fail
before training. Complete oversampled groups are reusable only while their
token equals the current snapshot.

Different fragments may have different committed versions. That is the
expected Decoupled DiLoCo cut; it is not a mixed-policy trajectory because a
rollout sees the resulting complete LoRA atomically.

### Safe-boundary state machine

SGLang generation and Megatron training remain sequential inside an island.
Network receive threads may queue BCAST and PULL messages while they run, but
Yeto changes trainer weights only in the post-train hook:

1. Validate the completed rollout snapshot and collect real RL statistics.
2. Export the complete post-train canonical LoRA.
3. Drain monotonic BCASTs in fragment order and stage them in that full state
   without changing committed bridge anchors or versions.
4. If any fragment changed, apply the full state once with
   `reset_optimizer=False`, then re-export and verify its full-policy hash;
   only then commit the staged anchors, versions, and counters.
5. Drain PULL permits only after BCAST application.
6. For each permit whose exact raw anchor has accumulated at least `H` local
   steps, send `local_fragment - raw_anchor` without waiting for merge or
   quorum.
7. Atomically checkpoint island progress, create the next complete snapshot,
   set its SGLang token, and return to Miles.
8. Miles publishes the complete trainer LoRA before starting another rollout.

Only a committed BCAST replaces a raw anchor and resets its local step/token
counters. Duplicate messages are ignored, conflicting permits fail, and
invalid fragment/version identities fail. Ordinary hooks report zero remote
quorum wait because no hook waits for its own PUSH to merge.

Initial fragments use the same ordering: assemble a staged cut, apply and
hash-check it, then commit its bridge state. The protocol wire is unchanged;
the Python receiver records only a local monotonic arrival time. PULL-to-PUSH
is measured from local PULL receipt to PUSH enqueue, while BCAST queue time is
measured from local receipt to safe-boundary drain.

### Outer update

For fragment `p` and island `i`:

```text
d_i,p = theta_i,p - anchor_i,p
g_i,p = -d_i,p
g_p   = mean_i(g_i,p)

m_p     = 0.9 * m_p + g_p
Theta_p = Theta_p - 0.7 * (g_p + 0.9 * m_p)
```

The Rust syncer owns this f32 Nesterov update, fragment versions,
checkpoint-before-broadcast ordering, and exact full-roster validation. Yeto
adds no RL wire message.

### Inner optimizer and scheduler

An in-process fragment BCAST replaces LoRA masters and model parameters while
preserving Adam moments. It refreshes the actor backup and keeps scheduler
progress at the island-local rollout count. A new process instead applies the
authoritative cut with `reset_optimizer=True`; no unavailable local Adam
history is reconstructed, while the scheduler advances to checkpointed local
progress.

Syncer fragment step and island rollout progress are separate identities.
Miles `TrainableState.policy_version` carries only the latter in decoupled
applies.

### Learning-rate schedule

Yeto sets the inner learning-rate schedule explicitly
(`RLRunConfig.algorithm.lr_schedule`, resolved from the sync preset) instead
of relying on Miles' implicit default:

| Preset | `--lr-decay-style` | `--lr-decay-iters` | `--lr-warmup-iters` | `--min-lr` |
|--------|--------------------|--------------------|---------------------|------------|
| `strict-avg`, `full-parameter` (alias `dense-full`) | `linear` | `global_rounds x optimizer_steps` | `0` | `0` |
| `decoupled` | `constant` | `global_rounds x optimizer_steps` (unused; satisfies Megatron's `lr_decay_steps > 0`) | `0` | `0` |
| eval-only | not passed | | | |

A strict run whose `rollout_batch_size x n_samples_per_prompt` differs from
`global_batch_size x optimizer_steps` is refused, because the schedule
horizon would not match the run. Decoupled uses a constant rate because its
local step count is not known up front. The four flags are adapter-owned;
passing them through extra Miles argv is rejected. The resolved flags are
pinned by `tests/test_rl_argv_snapshot.py` and
`tests/test_rl_miles_adapter_config.py`.

**Zero learning-rate invariant.** Every `rl_local_round` event records
`applied_lr` (minimum over the round) and `applied_lrs` (one per optimizer
step), read from `optimizer.param_groups[*]["lr"]` before Miles advances the
scheduler (`train/lr` is Miles' post-step value). If a round applied learning
rate 0 and the island will keep training, the round fails with
`rl_strict_failure metric=zero_lr_before_final_round` and is not submitted.
The final round is `local_round_id >= global_rounds` for strict-avg; for
decoupled it is a round trained after the syncer announced the final cut (or
the round that exhausts `learner_budget_steps`).

This dry-run plan selects the decoupled preset without GPUs:

```bash
python3 scripts/benchmark_rl.py --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data openai/gsm8k --data-revision e53f048856ff4f594e959d75785d2c2d37b678ee \
  --reward-function project.rewards:score \
  --islands 2 --arms decoupled --dry-run
```

## Launching

A strict two-island run uses the default preset:

```bash
yeto launch \
  --training-mode rl \
  --gpu aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --model org/model \
  --model-revision <immutable-commit> \
  --data org/prompts \
  --data-revision <immutable-commit> \
  --tuning lora \
  --lora-r 8 \
  --lora-targets attention \
  --total-steps 8 \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 512 \
  --reward-function project.rewards:score \
  --seq-len 2048 \
  --inner-lr 1e-5 \
  --seed 17 \
  --trust-remote-code
```

Add the following for decoupled synchronization:

```bash
  --rl-sync-preset decoupled \
  --fragments 8 \
  --pipeline 2 \
  --local-rl-rounds-per-sync 4
```

`P=8`, `tau=2`, `H=4` is a reasonable starting configuration; it is not
claimed to be optimal for every model.

### Which island reads which rows of `--data`

Row `i` is the 0-based number of a row of `--data`, in read order. The
launcher prints the rule (`[launcher] data split (...)`) and records it in the
run manifest and the dry-run plan (`data_split`); each island records its slot
in `rl_driver_start` (`data_slot`). The rules are in
`yeto/rl/engine/data_split.py`. `--rl-data-split` selects one of:

| mode | rows per island |
| --- | --- |
| `auto` (default) | `dispense` when the launch can use it, otherwise `stride`; the launch log names the reason |
| `stride` | island `k` of `N` reads rows with `i % N == k` (`--rl-data-slot k/N`); each kept row carries `metadata.yeto_data_row` (verl: `extra_info.yeto_data_row`) |
| `dispense` | the actor syncer hands out blocks of positions per round (`--dispense`, see "Task dispenser" in [PROTOCOL.md](PROTOCOL.md)) |
| `shared` | every island reads every row in the same order; for controlled experiments only, the launch warns (`--external-learners` also falls back to it) |

`--rl-island-tasks` replaces the split with explicit task lists. Sampler
islands read their own slots (sampler `k` of `S`: `i % S == k`); trainers that
read the sample buffer read buffer lanes. `stride` is used for verl, the SAO
streaming preset, runs without a syncer, `--rl-elastic`, `--syncer-shards > 1`,
Miles fully_async, island round cuts and `--eval-only`.

With `stride`, after its last row an island starts the next pass over its own
rows: pass 0 is data order, every later pass is reshuffled with a seed made
from `--seed` and the pass number (verl keeps its own sampler). Slots are
fixed at launch: a relaunched island keeps its learner id and rows, the rows
of an island that left are read by no one, and an island added later gets no
slot. A full-parameter island with `--rl-checkpoint-store` keeps its data
position under `<store>/data-positions/` and continues at the next row of its
slot (`rl_data_position_restored`).

**Task dispenser** (`dispense`, Miles only; not with fully_async rollouts,
island round cuts or a sharded head). Every island writes every row into its
prompt file, then claims a block of positions per round and trains those rows
only. No row goes to two islands; a fast island claims more often and never
waits for a slow one; rows a relaunched, preempted or silent island did not
finish go first to the next claim. A row counts as trained only after its
round passed the sync boundary (`rl_dispense_commit`). Position `p` is row
`pass_order(rows, --seed, p // rows)[p % rows]`. A row whose group is too
stale can be given out again (`--rl-dispense-max-redo`, default 1).
`--rl-data-passes N` ends the data after N passes; an island that cannot fill
a batch any more ends FINISHED (`rl_data_ended`); this needs islands that may
end on their own (asynchronous rounds or an island pool). The syncer logs
every grant and report in `~/yeto-output/task-dispense.jsonl`;
`python -m yeto.launch.checks.dispense_check` replays it (no row granted or trained
twice, every row accounted for), and the launcher copies the log and its
check into the run directory.

**Data mix and task labels** (`dispense` only). `--data-mix
'a.jsonl=0.7,org/ds@rev=0.3'` (instead of `--data`) writes the rows of several
sources into one prompt file (`data-mix.jsonl`, rows tagged
`metadata.yeto_data_source`); the dispenser keeps one stream per source and
interleaves them by weight (`--rl-dispense-mix`), so any prefix of the rows
given out is within one row of the weights. `--rl-island-task-labels
'l0=web;l1=pwn,crypto'` gives an island only rows whose `task_label` is one of
its labels (`-`: unlabeled rows); a label taken by no island, or a listed
label on no row, fails the launch. `dispense_check` reports per stream and per
island.

**Sampler islands on the dispenser.** In a synced group
(`--rl-sample-role trainer --rl-island-sample-role M:sampler`)
`--rl-data-split dispense` makes the sampler islands claim rows from the
group's actor syncer (`--rl-data-dispense --rl-dispense-addr $SYNCER_ADDR`)
and report a row once its group is in the sample buffer; a preempted
sampler's rows return after `--rl-dispense-lease-s` (at once when the same
sampler id is relaunched). An extra sampler joins with `--rl-sample-role
sampler --rl-single-island-no-sync --rl-data-split dispense
--rl-dispense-addr HOST:PORT` and a new `--rl-sampler-node` id. Refused: a
store-coupled pair or a convertible pool (no syncer), and `--rl-data-passes`.

### Clouds, controller and images

- **Controller.** `--controller head` (default) hands the run to one small
  on-demand head VM that hosts the syncer and the fleet controller;
  `--controller local` runs a detached worker on this machine plus a separate
  syncer VM (this machine must stay up). `--rl-single-island-no-sync`
  launches one island with no syncer and no outer sync, requires
  `--controller local`, and ends with launcher rc 2 by design (no syncer
  checkpoint); its result is judged from the island tape.
- **Clouds** (see [CLOUDS.md](CLOUDS.md)). Modal is the default GPU cloud for
  RL: a `modal:` island is a Modal function (`yeto/modal_runner.py`); pin the
  GPU with `H100!:N`-style entries (`--modal-gpu-exact`), serve model
  snapshots from a Modal Volume (`--modal-model-volume`, read-only at
  `/mnt/yeto-models`), and size the container with
  `--modal-memory-gib/--modal-cpu`. Neither Modal nor Verda (no sky
  `open_ports`) can host the head or syncer; Nebius can host both. A Modal
  island needs a syncer address reachable from Modal (head public IP, or
  `--syncer-public-addr` under `--controller local`).
- **Images and pins.** The default island image, the pinned Miles and SGLang
  fork commits and the baked Nebius VM images are defined in
  `yeto/rl/adapters/miles/pins.py`. Inside the image,
  `/opt/yeto/image-manifest.json` records every SHA. `--rl-image` overrides
  the image; a private one takes its login from `SKYPILOT_DOCKER_*` or
  `--rl-image-private`.
- **Fork overlay.** Some critic-family Miles flags are not in the pinned image
  commit. With `--rl-miles-overlay auto` (default) the island setup applies a
  sha256-checked patch from `yeto/rl/adapters/miles/overlays/` over the
  image's Miles checkout, only when the algorithm spec uses a fork-only flag;
  `critic-c357` forces it, `off` never applies it. The run manifest then
  records the overlay sha256.

### Variance-aware GRPO sampling

A group whose rewards are identical has zero GRPO advantage. Miles' DAPO-style
filter with oversampling replaces such groups before the training batch is
formed:

```bash
  --rollout-batch-size 4 \
  --over-sampling-batch-size 16 \
  --dynamic-sampling-filter-path \
    miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
  --dynamic-sampling-max-replacements 8 \
  --rl-offload-train \
  --rl-distributed-timeout-minutes 10
```

`--over-sampling-batch-size` must exceed the training batch. When the stock
filter path is selected, Yeto uses a bounded equivalent: it rejects at most
`--dynamic-sampling-max-replacements` zero-variance groups, then accepts one
fallback (`rl/dynamic_filter/forced_groups`); generated, accepted, dropped and
replacement counts are recorded per round. `--rl-offload-train` keeps the
trainer onloaded through policy initialization and the first publication,
then offloads it before rollout zero; the distributed timeout is a fail-fast
bound, not a quality setting.

`--trust-remote-code` is required by the pinned Miles model-loading path; keep
model and dataset revisions immutable and use it only for trusted sources.
The reward callable is `package.module:function`; Yeto hashes its source
before provisioning and the learner verifies that digest before import.
Prompt rows provide `messages`, or a string `prompt`/`input` that becomes a
user message; `label`, `metadata` and `tools` are passed to Miles and the
reward callable.

### Launch dry run, event tapes and exit codes

**Launch dry run.** `yeto launch ... --dry-run` validates the whole launch
(arguments, provenance, algorithm and capability checks) and prints JSON with
the resource request, whether a syncer is started, the algorithm hash, the
per-island memory estimate and each learner command. It creates no cloud
resource.

**Event tapes of Modal islands.** A Modal island's `~/yeto-output` cannot be
fetched, so every Modal and every `--rl-single-island-no-sync` island runs
with `--rl-echo-events`: the learner prints each tape record as
`YETO_RL_EVENT <json>` (`yeto/rl/event_echo.py`) and the launcher rebuilds
`<run dir>/events/<island>.jsonl` from the log stream. A tape without
`rl_learner_finalized` gets a `.incomplete` marker and makes the run exit 3;
`yeto-rl-export --rl-event-tape` refuses it unless `--allow-incomplete`.

**Launcher exit codes.**

| code | meaning |
| --- | --- |
| 0 | success |
| 1 | a learner failed (non-RL), no learner succeeded, or a strict syncer/RL failure or internal error |
| 2 | artifact not fetchable (Modal island, or a no-sync single island) |
| 3 | incomplete island event tape |
| 4 | a fixed-roster RL island could not be recovered |
| 5 | the Modal app was not confirmed stopped after teardown (takes precedence over 0/2/3/4) |
| 6 | the run stalled: no island event for `--rl-stall-timeout` seconds (default 900, 0 disables) and not every island finalized |
| 7 | Modal moved an island to a different container mid-run |
| 8 | `--rl-trainer-max-failures` (default 3) trainer incarnations failed in a row |

For exit 5, every `modal app list` row with the run's app name must be
`stopped` with 0 tasks (no row also counts); the launcher checks up to 5
times, prints the `modal app stop` command to run by hand, and writes
`<run dir>/teardown.json`. The stall clock starts at the first received
event; on a stall the launcher drains the tapes, tears everything down and
does not relaunch. An island whose tape already holds `rl_learner_finalized`
counts as succeeded even if its job then ends non-zero, and is not
relaunched.

## Multi-node islands

An RL learner island may span several SkyPilot nodes (`--rl-placement
fixed-partition` only). `--gpu cloud:NxGxgpu` asks for one N-node cluster of
G GPUs per node per learner:

```bash
yeto launch --training-mode rl \
  --gpu nebius:2x8xh100 \
  --rl-placement fixed-partition --rl-rollout-gpus 8 --rollout-num-gpus-per-engine 8 \
  --tensor-parallel 2 --pipeline-parallel 1 \
  --rl-elastic --rl-elastic-resources cfg/resources-2x8.json --rl-elastic-initial-config T8R8S0 \
  ...
```

### Topology and placement rules

- Logical bundle `p` lives on node `p // G` as local GPU `p % G`; the learner
  asserts this against the Miles placement group at startup and refuses any
  other layout.
- Ray head = node rank 0 = the learner process = trainer rank 0. Worker nodes
  join the head's Ray (bounded join, cleanup trap armed first) and leave when
  the head exits.
- `--rl-rollout-gpus` / `--rl-standby-gpus` are **island totals**. Without a
  cfg `placement` the trainer takes the leading `total - rollout - standby`
  bundles; with one, the trainer GPUs are exactly the listed `n<k>:<g>` slots.
  The trainer must occupy the same number of GPUs on every node it uses, as
  one contiguous ascending run per node (the launcher derives Miles'
  `actor_num_nodes x actor_num_gpus_per_node` and forwards
  `--rl-island-bundle-map` for a non-leading layout).
- By default every **TP*CP** group of the trainer, every rollout engine and
  every standby-to-cell rebind stays on one node; **EP and PP groups may span
  nodes**. `--rl-allow-cross-node-tp` (cfg `parallel.allow_cross_node_tp`)
  lets the trainer TP*CP group span nodes; `--rl-allow-cross-node-engine-tp`
  (cfg `parallel.allow_cross_node_engine_tp`) lets a rollout engine span
  nodes as a whole-node replica, which scales only as one unit. Both are
  logged as warnings and recorded in the journal `topology.layout`.
- A `(node, gpu)` slot may appear once across trainer / rollout / standby.
  Rollout and trainer may share a node on different GPUs (e.g. `T2R1S1`:
  trainer `["n0:0","n1:0"]`, rollout `[["n0:1"]]`, standby `["n1:1"]`);
  non-rectangular trainers are refused. Violations fail at `yeto launch`
  before any cloud resource is touched.
- The island needs at least the recipe-derived minimum of nodes (one trainer
  replica `tp*cp*pp*max(1, ceil(ep*etp/tp))` plus one rollout engine and the
  standby reservation, rounded up to whole nodes);
  `--rl-min-nodes-per-learner N` raises that floor.
- GPU UUID reconciliation: the learner reads `nvidia-smi
  --query-gpu=index,uuid` on every node and compares the pool with the cfg
  `gpus[*].uuid` and/or the previous incarnation's `gpu_pool` journal record.
  A different shape or uuid refuses the start; a same-shape pool on replaced
  machines needs `--rl-elastic-accept-rebind`, which journals the mapping.
  After acceptance the learner refuses duplicate occupation, more than one
  role per uuid, and a uuid still held by a live older incarnation (node
  marker `/tmp/yeto-rl-incarnation/<uuid>.json`; run `yeto down` first).
- The launcher forwards `--rl-island-gpus-per-node G`; a single-node island
  sends nothing.

### Resources cfg (`--rl-elastic-resources`)

A single-node cfg (no `nodes`) keeps its original format. A multi-node cfg
adds `nodes`/`gpus_per_node`; `placement` entries may be `"n<k>:<g>"`, logical
bundle integers or pool GPU uuids (one spelling per cfg):

```json
{"nodes": 2, "gpus_per_node": 8,
 "configs": {"T8R8S0": {"trainer": 8, "rollout": 8, "standby": 0, "rollout_engine_gpus": 8,
                        "parallel": {"tp": 2, "pp": 1},
                        "placement": {"trainer": ["n0:0","n0:1","n0:2","n0:3","n0:4","n0:5","n0:6","n0:7"],
                                      "rollout": [["n1:0","n1:1","n1:2","n1:3","n1:4","n1:5","n1:6","n1:7"]],
                                      "standby": []}}},
 "edges": []}
```

Every config must use all `nodes x gpus_per_node` GPUs; a resolved `gpus` pool
must list exactly that many, each node's `index` running `0..G-1`. Cells
declared with `--rl-elastic-cells` are cut per node (a run never straddles two
nodes; leftovers stay unbound), and `bind_members` refuses a cross-node
target.

### Node failure domain

The controller records the island `topology` in the reconfiguration journal
and polls `ray.nodes()` before every round. Any node loss (fewer alive nodes
than declared, or a node with another GPU count) is recorded as `node_lost`
and the island enters `RECOVERY_REQUIRED`: the learner exits non-zero and
nothing trains or generates on the surviving nodes. A restarted learner checks
the same topology before touching the fork's membership epoch. Same shape
with other hostnames is fine. Rebuilding the island is a manual relaunch with
`yeto launch`.

`NCCL_IB_DISABLE=1`, `NCCL_DEBUG=WARN` and (on Nebius)
`NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=<first up non-virtual interface>` are
exported on every node (the job log prints `[yeto-island] socket interface:
...`); set them in the environment to override.

### Node loss and rebuild (checkpoint store)

On node loss the whole island (one learner, one trainer group) stops and is
rebuilt from a **consistent checkpoint** on a same-shape island; a surviving
or returning node never rejoins the old job. The consistent checkpoint is the
elastic state dir (`--rl-elastic-state-dir`, default
`~/yeto-rl/elastic-state`: reconfiguration journal, `epochs.json`, trainer
cuts, batch ledger, inbox) on node0's local disk. Two things make it
rebuildable elsewhere:

- **`--rl-checkpoint-store <s3://…|gs://…|/shared/path>`** (needs
  `--rl-elastic`): after every commit point (startup `topology`, accepted
  `gpu_pool`, every `COMMITTED` / `SUCCEEDED` phase record) the controller
  copies the state dir to the store and writes `STORE-MANIFEST.json` last, so
  a copy without it is never trusted. A bucket URI is mounted on every node at
  `~/yeto-checkpoint-store`; a path is passed through as is. A learner that
  starts with an **empty** state dir restores the store first (`checkpoint_store
  action=restore`); a non-empty local journal always wins. A failed sync is
  logged in `controller.last_store_sync` and never fails the commit. Without
  the flag a multi-node launch warns and keeps state on node0 only.
- **Parallel layout baseline**: the `topology` journal record carries `layout`
  (`tp/pp/cp/ep`, trainer GPUs, nodes x GPUs per node, `bundle_map`). A later
  incarnation with a different layout is `RECOVERY_REQUIRED` (`layout changed:
  ...`) before any fork membership call; a cut written under another Megatron
  layout is not restorable, and there is no override flag.

Rebuild procedure:

1. The island records `node_lost`, enters `RECOVERY_REQUIRED` and exits
   non-zero; in-place retries are refused by the partial-island preflight.
   Collect the journal and tape evidence.
2. `yeto down` the island.
3. `yeto launch` with the **same cfg, recipe, parallel flags and
   `--rl-checkpoint-store`**, plus `--rl-elastic-accept-rebind` for the new
   GPU uuids.
4. The new island restores the store, checks topology and layout against the
   restored baseline, reconciles and journals the GPU rebind, then recovers
   the committed membership and continues from the last cut.
5. A layout or shape mismatch is `RECOVERY_REQUIRED` again; fix the launch
   flags instead of forcing it.

Not supported: a node returning to its old job, partial continuation on
surviving nodes, rebuilding under another parallel layout, and the store on
Modal islands (use a shared path there). Megatron `--save`/`--load`
checkpoints are not managed by the store.

### Teardown confirmation

`yeto down` / the launcher's teardown confirm a multi-node island per node
instance: the instance ids are captured from the cloud before `sky down`, and
the island counts as released only when the cloud reports every one of them
terminated (`node instance <id> confirmed terminated` lines). No cloud probe,
a failing probe or an instance still alive after the retries prints
`UNCONFIRMED node instance(s) <ids>`, the cluster is reported as not verified
and the run's teardown exits non-zero; delete the listed instances in the
cloud console or rerun `yeto down`. `sky down`'s own success is never trusted
for more than one node.

## Checkpoint and Recovery

The syncer checkpoint is the only authoritative global LoRA. In exact-base RL
it is written before every corresponding BCAST and contains f32 parameters,
outer momentum, per-fragment versions, layout fingerprint, and learner ledger.

Each island atomically stores only reconstruction progress:

- immutable model, data, reward, source, topology, LoRA, preset, `P/tau/H/N/T`,
  and logical learner identity;
- next rollout ID, optimizer-step count, and action-token count;
- latest snapshot token, full-policy hash, and fragment-version vector;
- rollout statistics and complete same-token groups.

It does not store local LoRA or optimizer moments. On restart, the island
receives the current authoritative fragment cut, restores scheduler progress,
and applies the full cut with optimizer reset. Completed groups survive only
when the rebuilt token, full-policy hash, and fragment-version vector exactly
match the checkpoint; otherwise they are discarded. A nonzero syncer cut
without a valid island checkpoint fails instead of guessing local scheduler
progress.

Production clients keep `max_reconnects=0`: connection loss exits the island
and relies on the existing launcher/provider task recovery. The feature does
not add roster shrinking or a new fleet restart controller. Syncer restart can
resume when its checkpoint disk survives; losing that VM and disk remains a
deployment-level durability gap.

### Verify-ahead of received broadcasts

A trainer apply re-checks every replaced fragment's stored chunks (finite
FP32 values, chunk hashes, fragment digest) in its prepare. With
`--rl-sync-verify-ahead auto` (default) each island runs that same check
(Miles `verify_stored_fragment_chunks`) on a background thread as soon as a
broadcast is stored, and the apply at the next boundary names the checked
fragments (`verified_fragments`), so the trainer reads none of their chunks;
a fragment whose check is still running is checked by the trainer as before,
and a failed check fails the apply before the trainer changes. The applied
values are the same either way. It needs a trainer Miles with the
`verified_apply` transport; `--rl-sync-verify-ahead off` (island
`--sweep-verify-ahead off`) or `YETO_FULL_PARAMETER_VERIFY_AHEAD=0` keeps the
check on the trainer. Each `rl_full_parameter_round` event reports
`sync/apply_verified_ahead` and `sync/apply_verify_pending`.

## Finalization and Export

At `T=N*P`, the syncer stops ordinary pulls, persists the quiescent cut, and
sends all f32 final fragments plus an exact manifest. (Full-parameter
asynchronous rounds send the manifest only unless islands evaluate or
`--rl-sync-final-cut download` is set; see
[PROTOCOL.md](PROTOCOL.md#manifest-only-final-cut).) At the next safe
boundary, each island stops ordinary fragment submission, applies the
complete final cut, verifies the trainer's full-policy hash, writes its final
progress checkpoint, acknowledges the manifest, and asks Miles to stop only
after one complete SGLang weight publication. A replacement island that joins
while finalization is pending consumes the terminal manifest directly,
applies it with optimizer reset, acknowledges it, publishes it once, and exits
without an extra rollout.

Export needs the original model contract; a decoupled export also needs the
training layout (the last four flags):

```bash
yeto-rl-export \
  --checkpoint ./rl-output/yeto-state.ckpt \
  --model org/model \
  --model-revision <immutable-commit> \
  --lora-r 8 \
  --lora-targets attention \
  --output-dir ./adapter \
  --sync-preset decoupled --fragments 8 --pipeline 2 --local-horizon 4
```

The exporter reconstructs the canonical specs and fragment layout, verifies
the sync fingerprint and terminal version sweep, rejects non-finite or
mismatched tensors, and writes standard `adapter_model.safetensors` and
`adapter_config.json`. Decoupled export also writes `yeto_rl_provenance.json`
with `P/tau/H/N/T`, outer optimizer settings, final fragment versions, both
layout hashes, full-policy hash, and checkpoint SHA256.
`--rl-algorithm-spec PATH` adds `algorithm_spec` and `algorithm_spec_sha256`
to the provenance.

### Starting a fresh phase

A completed Decoupled adapter can seed policy version zero of another
Decoupled run with `--rl-sync-preset decoupled --rl-initial-adapter ./adapter
[--rl-initial-adapter-sha256 <sha256>]`. The flag accepts only a local
directory. The launcher hashes it before provisioning, rejects a differing
supplied digest, and mounts it read-only at the same path on every island.
The learner rehashes it before rollout zero and requires a standard causal-LM
LoRA adapter whose base model, revision, rank, targets, tensor names and
shapes, and finite values match the new run; RS-LoRA, DoRA, fan-in/fan-out
and per-module rank or alpha modifiers are rejected, and its export
provenance and full-policy hash must match the loaded tensors.

This starts a new phase: the policy tensors are preserved exactly, while the
inner optimizer and scheduler, outer optimizer and syncer, rollout IDs,
fragment versions and checkpoints start fresh. Use a new `--cluster-prefix`
for each fresh phase; reusing a run identity keeps the normal in-phase
recovery behavior instead.

## Engine selection

The Miles islands run one engine path, `ports`: Yeto's `IslandDriver`
(`yeto/rl/engine/driver.py`) owns the island loop and uses upstream Miles as a
library through role ports. `yeto launch --rl-engine` still parses and accepts
only `ports`, so existing commands keep working; the launcher no longer
forwards it.

The learner verifies its source (repository, commit, clean detached checkout,
import path) before any model is loaded
(`yeto.rl.adapters.miles.revision.verify_miles_revision`, pins
`MILES_NEXT_PINS` in `yeto/rl/adapters/miles/pins.py`). The island tape starts
with an `rl_engine_selected` event (`rl_engine=ports`, Miles commit,
`rl/algorithm_spec_sha256`, placement), the SSH plan carries
`"rl_engine": "ports"`, launcher provenance carries `rl_engine`, and
`yeto_rl_provenance.json` of the exported adapter contains
`"rl_engine": "ports"`.

### Ports boundary

`yeto.rl.engine.selection.ports_rejections` lists every combination that is
refused before startup, each with its reason: non-causal models,
full-parameter training under elastic island scheduling (the elastic syncer
has no fragment frames yet), expert-full tuning without the `full-parameter`
sync preset, `--use-critic` or an advantage estimator other than GRPO/PPO,
and fixed partition requested through pass-through argv instead of
`--rl-placement fixed-partition`. Full-parameter training
(`--rl-sync-preset full-parameter`, alias `dense-full`), SAO streaming
(`sao-streaming-full`) and the DeepSeek V4 recipe run on `ports`.

The Miles translation also rejects fork-only options with no upstream
equivalent (`--rollout-engine-base-port`, `--train-master-base-port`,
`--sglang-router-prometheus-port`, `--custom-agent-function-path`,
`--tito-allowed-append-roles`) and all Miles fault-tolerance flags.

### Port responsibilities

| port | responsibility | Miles adapter (`yeto/rl/adapters/miles/`) |
| --- | --- | --- |
| `RolloutPool` | generate one rollout of complete groups from the published policy; members | `rollout.py` over `InferenceController` + `RolloutExecutor`; metadata from `rollout_meta_hook.py` inside the rollout process |
| `TrainerGroup` | one optimizer step on an opaque batch; onload/offload; grad norm for the per-round invariant | `trainer.py` over the single-cell actor group |
| `PolicyState` | export/apply the LoRA trainable state (optimizer reset or preserve, scheduler alignment) | `state.py` + `state_plugin.py`, run on every Megatron rank through `run_plugin` |
| `Publisher` | full publication of one exact policy, acknowledged by every rollout member | `publish.py` over upstream `update_weights`, stamping `yeto:<version>:<policy_tensor_hash>` as the SGLang weight version |
| `Placement` | read-only placement description; detects Miles rewriting the request | `placement.py` |

The policy token `yeto:<rollout_id>:<policy_tensor_hash>` has one definition,
`yeto.rl.core.policy_snapshot_token` (also `PolicySnapshot.token`). The driver
hands the published token to the rollout process, where
`--buffer-filter-path` (`policy_buffer_filter`) reuses only complete groups of
exactly that policy. Outer synchronization uses the `strict-avg` and
`decoupled` state machines and island checkpoint formats in
`yeto/rl/engine/bridges.py`.

### Miles-side changes carried by the fork

`ports` needs a generic entry point in Miles: `run_plugin(fn_path, kwargs)` on
the train actor and a pass-through on the trainer group (`TrainerController`),
which loads `fn_path` and calls it with the actor on every rank. The
trainable-state export/apply and grad-norm plugins live in Yeto. This and the
other compatibility changes (elastic verbs, loss variants, Qwen3.8-Next LoRA
layout, engine-failure propagation) are carried as commits on the pinned Miles
fork; they are not proposed upstream. `MILES_NEXT_COMMIT` must name a commit
carrying them, otherwise the learner refuses to start (before any Miles
component or model is created) with an error naming the pin.

**Colocated publish order.** With a colocated trainer and `offload_train`, the
driver offloads the trainer after the optimizer step and before the weight
publication (the upstream Miles `train.py` order;
`MilesTrainerGroup.publish_offloaded`). Publishing first would keep the
trainer resident during publication and can run a large model out of GPU
memory.

## Algorithm specs

The training objective is one `AlgorithmSpec`, selected with
`--rl-algorithm-spec PATH`. The spec format, the absorb/reject rules, the
mechanism declarations and every algorithm family are described in
[RL_ALGORITHMS.md](RL_ALGORITHMS.md). A spec can be checked without GPUs:

```bash
python3 -m yeto.rl.adapters.miles.algorithm_flags --dry-run \
  --rl-algorithm-spec FILE [--extra "<miles argv>"] \
  [--rl-allow-unverified-mechanism DIMENSION:NAME ...]
```

### Mismatch-correction spec examples

These specs illustrate the train/inference mismatch corrections described in
[RL_ALGORITHMS.md](RL_ALGORITHMS.md#train-inference-mismatch-corrections).
Each block is checked by `tests/test_rl_mismatch_correction.py` through the
dry run: it is rejected as undeclared without the listed allowances (unless
the adapter declares it) and accepted with them.

<!-- mismatch-example allow=corrections:custom,corrections:mismatch_observe,features:mismatch_metrics -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "custom", "mismatch_metrics": true,
   "function": {"path": "yeto.rl.algos.mismatch_observe.observe_mismatch",
                "sha256": "5f83da1f138bbb8494748a8da042e9fb99c2f697eb6d71fb616086469801fa8f"}}}
```

<!-- mismatch-example allow=corrections:tis -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0}}
```

<!-- mismatch-example allow=corrections:custom,corrections:icepop,features:mismatch_metrics -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "custom", "tis_clip_low": 0.5, "tis_clip": 5.0,
   "mismatch_metrics": true,
   "function": {"path": "miles.backends.training_utils.loss_hub.corrections.icepop_function",
                "sha256": "971ccb0bf00b43b0582839c5b8dc05e91162c878ab7ec0ca878e3b1e668f5318"}}}
```

<!-- mismatch-example allow=corrections:opsm,corrections:opsm_trainer -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "opsm", "opsm_delta": 0.0001, "opsm_old_logprob_source": "trainer"}}
```

<!-- mismatch-example allow=corrections:opsm,corrections:opsm_rollout,features:rollout_logprobs_as_old -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "opsm", "opsm_delta": 0.0001, "opsm_old_logprob_source": "rollout",
   "use_rollout_logprobs": true}}
```

<!-- mismatch-example allow=corrections:custom,corrections:mis_mask -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "custom", "mis_level": "geometric", "mis_mode": "mask",
   "mis_lower_bound": 0.9999, "mis_upper_bound": 1.0001,
   "function": {"path": "yeto.rl.algos.vendor.miles_mis.compute_mis_weights_with_cp",
                "sha256": "5c1dce0763f645f7491140e45c6c2b47774364f81ecd7c724cc71560424fefb4"}}}
```

## Flash-Next (Qwen3.8-Flash-Next) path

The Flash-Next profile lives in `yeto/rl/profiles/qwen3_8_next.py` (profile
`qwen3_8_next_4layer_lora`; full model `Qwen/Qwen3.8-Flash-Next`, plus a
4-layer CI variant, LoRA over the linear-attention projections via the fork's
Qwen3.8-Next LoRA layout). It needs a Megatron `torch_dist` checkpoint of the
base model; the HF-to-torch_dist conversion is a separate step, not done at
launch. On Modal the HF weights and the converted checkpoint can be served
from a Modal Volume with `--modal-model-volume`. The full model's LoRA island
needs two 8-GPU nodes (trainer 8 + SGLang TP8) and relies on the colocated
publish order described in
[Miles-side changes carried by the fork](#miles-side-changes-carried-by-the-fork).

## GPU memory estimate before launch

`yeto launch` estimates the per-GPU memory peak of every RL island before it
creates any cloud resource (`yeto/memory_estimate.py`, called from
`yeto/launch_preflight.py` `memory_preflight` in `launcher.run`, in
`yeto launch` and in the head path). It is an upper-bound estimate, not a
simulation. The weights-only `warn_if_model_wont_fit` stays as the first
coarse check.

Per GPU, in GiB (P parameters, V vocabulary, L layers, H hidden size, T tokens
of one micro-batch = max(context, response) x micro-batch 1):

1. weights `2P/(TP*PP)`
2. grads + optimizer: full `4P/(TP*PP) + 12P/(TP*PP*DP_shard)`; LoRA `16*P_lora`
3. activations `K_ACT * L/PP * T/(TP*CP) * H * 2`
4. logits block `K_LOGIT * min(T, C) * V/TP * 4`
5. inference engine `mem_fraction_static * GPU memory` when it shares the GPU and is not released
6. fixed overhead `R0 + FRAG * (1+2+3+4)` (CUDA context, residual of the released engine, allocator fragmentation)

A colocated island (train and rollout on one GPU, offload) alternates two
phases; the peak is the larger of the training phase (1-6) and the rollout
phase (`mem_fraction_static * GPU memory + R0`). The usable GPU memory is the
value torch reports for the known GPU types and `GPU_MEM_GB x 0.98`
otherwise. The constants `K_ACT`, `K_LOGIT`, `FRAG` and `R0` and their
derivation are documented in `yeto/memory_estimate.py`.

Behaviour: above `--preflight-memory-margin` (default 0.9) x GPU memory the
launch stops (`--preflight-memory error`, default) and lists the parts and the
suggestions that fit, each with its recomputed peak: the longest context
(multiple of 1024) that fits with the response shrunk in the same ratio, 2x
GPUs of the same type, the next larger GPU type. Only calibrated models can
stop a launch: for other models an over-limit estimate prints a warning and
never stops it. A model whose structure is unknown (no local or cached
`config.json` and not in `memory_estimate.BUILTIN_CONFIGS`) prints a
not-estimated notice and the launch continues. `--preflight-memory warn`
prints and continues; `off` skips the estimate with a warning. The run
manifest records the estimate per island (`preflight.memory`), and
`yeto launch --dry-run` shows it per island (`memory_estimate`).

## Elastic reconfiguration

`--rl-elastic` (off by default; fixed partition; resources manifest
`--rl-elastic-resources`, initial config, rollout cells) enables the
island-local reconfiguration controller:

- **Manual rollout add/remove/drain** with journalled transactions, expected
  epoch, deadlines and a watchdog; the controller restarts the learner in
  place with `--rl-elastic-restart-attempts`.
- **Same-shape cut/restore** (`TrainerGroup.save_cut/restore_cut`, journal,
  checkpoint store for multi-node rebuilds, see
  [Node loss and rebuild](#node-loss-and-rebuild-checkpoint-store)); there is
  no automatic re-entry of lost ranks.
- **Trainer DP change / role transfer** (`--rl-elastic-trainer-edges`): treat
  the flag as experimental.
- **Recommendation and auto mode** exist in code; auto is off by default.
- Fleet-level control across islands is out of scope (see
  [Supported Boundary](#supported-boundary)).

## Benchmark

[`scripts/benchmark_rl.py`](../scripts/benchmark_rl.py) runs up to three
local, equal-hardware real-Miles arms, selected with `--arms` (default
`single,federated,decoupled`; any other name is rejected):

| arm | purpose |
| --- | --- |
| `yeto-single-mM` | one strict Yeto island using all `M*G` GPUs |
| `yeto-federated-mM` | strict full-roster FedAvg across `M` islands |
| `yeto-decoupled-mM` | decoupled fragment synchronization across `M` islands |

`--dry-run` prints the plan followed by `RL_ENGINE ports` without GPUs (same
arguments as the decoupled example above, with `--arms
single,federated,decoupled`).

All arms match model/revision, LoRA, prompt stream, reward, seed, total GPUs,
optimizer steps, groups, trajectories, and action-token limits. Decoupled
learners freeze after the same local step budget; the syncer then performs
exactly one ordinary full-fragment consolidation sweep from the frozen
policies before marking and exporting the final artifact, so the comparison
does not depend on a network-dependent amount of local work. The report
covers held-out reward and pass@k, KL/ESS/clip fraction, phase timings,
throughput, GPU-hours and utilization, realized `H`, PULL-to-PUSH latency,
BCAST queue time, fragment payload traffic, and deltas versus the
single-island and federated controls. The harness uses real SGLang
generation, the real reward callable and real GRPO training only. See also
[RL_BENCHMARK.md](RL_BENCHMARK.md).

A change to the decoupled path is accepted against a `ports` reference run:
the same configuration under `strict-avg`, or the single-island baseline
(`yeto-single-mM`), with equal model, data, reward, seed, GPUs and step
budget (`scripts/benchmark_rl.py --arms single,federated,decoupled` runs
both). Held-out reward and pass@k must not regress beyond seed noise, and the
invariants in the [Development Guide](#development-guide) must hold.

## Buffer-trainer releases and spot rollout islands

These options apply to a buffer trainer (`--rl-sample-role trainer`) that
publishes policy releases to separate rollout islands.

- **Background releases** (`--rl-release-background`, island entry; needs
  `--rl-release-delta-full-every N`): the trainer writes each staged release
  on the training thread, then encodes, digests and uploads it on a worker
  thread while the next rounds train. At most one release is in flight; the
  next publication waits for it, and a failed background release raises
  there. Single trainer only.
- **Sparse weight updates** (sampler spec `engine.options.sparse_update`,
  `off` by default, or `auto`): when a release was rebuilt from the served
  one with change tracking (`YETO_RL_MIRROR_TRACK_CHANGES=1`), the SGLang
  adapter writes only the changed elements on every rank instead of a full
  reload. It falls back to a full reload when the release has no change
  manifest, changes more than `sparse_max_ratio` (default 0.05) of the
  elements, or touches a tensor the startup probe could not map, and reloads
  in full after a failed write or check. bf16 weights only; quantized
  weights and models whose `load_weights` derives tensors are not supported.
- **MFU of a hybrid model**: `python -m yeto.rl.hybrid_flops TAPE --config
  config.json --gpus N --gpu CARD` prints one JSON line per trained round
  (`step_mfu`, `wall_mfu`) from the trainer's event tape, for models whose
  framework MFU counter does not cover them (Nemotron-H).

Spot rollout islands on Nebius have four operator tools in
`yeto/rl/spot/` (each with `--help`):

| module | role |
| --- | --- |
| `python -m yeto.rl.spot.spot_plan` | write the sampler spec and environment of one rollout island VM |
| `python -m yeto.rl.spot.watchdog` | launch and keep one island VM alive: walks GPU x region x on-demand/spot candidates, deletes stuck or preempted VMs, stops at `--budget-usd` or a stop file (`--dry-run` prints the candidate order) |
| `python -m yeto.rl.spot.s3_delivery` | `publish-loop` on the trainer host and `pull-loop` on the VM: deliver releases through S3 with sha256-checked manifests |
| `python -m yeto.rl.spot.patrol` | read trainer/sampler tapes, VM and sandbox usage and spend; exit 0/1/2 for quiet/warn/stop, `--act` runs the stop-loss |

## Observability

Island JSONL records include:

- rollout ID, exact policy token/hash, and full fragment-version vector;
- group, trajectory, action-token, reward, KL, ESS, clip, and timing metrics;
- applied and submitted fragment IDs, fragment tensor payload bytes, delta
  norm, realized `H`, PULL-to-PUSH time, and BCAST queue time;
- full-policy apply time, snapshot publications, and optimizer-reset count;
- hook duration and whether the hook performed finalization;
- `applied_lr` / `applied_lrs`, the learning rate the round's optimizer steps
  applied (see [Learning-rate schedule](#learning-rate-schedule)).

The syncer tape remains authoritative for outer step, fragment, exact base,
round attempt, full responder roster, Nesterov update norm, merge time, and
layout fingerprint. The read-only fleet dashboard (`yeto dashboard
serve/export`, [DASHBOARD.md](DASHBOARD.md)) reduces these tapes, the syncer
tape, controller journals and the head's `fleet.jsonl` into one view; it is
not a data source and changes no run state.

Payload traffic counts PUSH, ordinary BCAST, and ordinary final-cut fragment
tensors. It intentionally excludes message headers, framing, chunks, and
control messages.

## Development Guide

The implementation is intentionally confined to the RL boundary:

| area | responsibility |
| --- | --- |
| `yeto/rl/core.py` | canonical LoRA, deterministic RL fragments, policy snapshots |
| `yeto/rl/decoupled.py` | raw anchors, BCAST/PULL/PUSH state, final cut and benchmark consolidation |
| `yeto/rl/engine/driver.py`, `yeto/rl/engine/bridges.py` | island loop, outer-sync sessions, island progress |
| `yeto/rl/adapters/miles/island_entry.py` | Miles argument mapping and runtime configuration |
| `yeto/rl/adapters/miles/pins.py` | Miles/SGLang fork pins and the island image |
| `yeto/rl/export.py` | authoritative checkpoint to standard PEFT |
| `yeto/rl/initial_adapter.py` | validated Decoupled PEFT policy warm start |
| `scripts/benchmark_rl.py` | equal-work single/federated/decoupled comparison |
| `syncer/src/` | general fragment scheduler and outer optimizer |

Preserve these invariants when extending the path:

1. A trajectory group uses one real complete policy snapshot.
2. A fragment update uses its exact raw anchor and complete fixed roster.
3. Ordinary local hooks never wait for remote quorum or merge.
4. In-process fragment apply preserves inner optimizer moments and scheduler
   progress.
5. Only the exact final syncer cut may become the exported adapter.
6. A new phase may reuse policy tensors, but never prior optimizer or progress
   state.
7. Rust, SFT, diffusion, local PPO, and generic recovery behavior do not change
   without a separately reviewed design.

Focused verification:

```bash
python -m pytest -q \
  tests/test_rl_core.py tests/test_rl_decoupled.py \
  tests/test_rl_export.py tests/test_rl_initial_adapter.py \
  tests/test_rl_launcher.py \
  tests/test_rl_integration.py tests/test_rl_benchmark.py
```

When Miles is upgraded, re-copy the vendored MIS helper
(`yeto/rl/algos/vendor/miles_mis.py`) and re-review the algorithm flag table;
see [RL_ALGORITHMS.md](RL_ALGORITHMS.md#when-miles-is-upgraded).
