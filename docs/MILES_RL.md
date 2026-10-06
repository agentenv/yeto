# Miles RL

Miles RL runs causal-language-model reinforcement learning inside independent
[Miles](https://github.com/agentenv/miles) islands and uses Yeto to synchronize
their complete LoRA policies. Miles owns SGLang rollout, reward evaluation,
GRPO, and Megatron local training. Yeto owns the cross-island LoRA boundary,
the authoritative global checkpoint, recovery identity, finalization, and PEFT
export.

Two explicit synchronization presets are available:

| preset | role | global synchronization |
| --- | --- | --- |
| `strict-avg` | default correctness baseline | one complete LoRA fragment; every island waits for full-roster equal-weight FedAvg after every local round |
| `decoupled` | Decoupled DiLoCo RL | deterministic multi-fragment LoRA; local RL continues while exact-base fragment rounds are in flight |

Selecting `decoupled` is intentional. The launcher never infers it from the
model, island count, or generic DiLoCo flags.

> **Status:** `strict-avg` has real-model GPU and recovery evidence. The
> `decoupled` source implementation and automated protocol/oracle coverage are
> present, and `MILES_COMMIT` pins the reviewed stop-capable Miles commit.
> Release use additionally requires that commit to be available from the
> configured Miles repository and the real GPU matrix in the design to pass.
> The runtime verifier deliberately rejects a dirty or mismatched checkout.

## Supported Boundary

The supported model and runtime contract is deliberately narrow:

| dimension | supported value |
| --- | --- |
| model | causal language model |
| tuning | LoRA only |
| learner | pinned Miles with Megatron-Core |
| rollout | colocated SGLang |
| optimization | GRPO; one optimizer step per rollout/train cycle |
| island size | one or more nodes and one or more GPUs per node |
| parallelism | TP = PP = CP = 1; dense EP = 1; MoE EP may divide the island world size when LoRA tensors remain replicated |
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

- full-parameter, actor/critic, or diffusion RL;
- TP>1 or PP>1 LoRA gather/scatter;
- expert-sharded LoRA tensors;
- trajectory migration across policy snapshots;
- same-fragment stale updates, dynamic quorum, or learner weighting;
- RDA, IsoLoCo, HeLoCo, delta correction, or broadcast blending;
- optimizer-moment federation;
- Miles' experimental fault-tolerant actor path;
- a new dashboard, a cross-island controller, a storage system, or a generic
  recovery framework. An island-local, yeto-side reconfiguration controller is
  allowed (rl-infra-spec design D1): it drives the island's own port verbs
  (rollout pool resize, `TrainerGroup.save_cut`/`restore_cut`/same-shape
  rebuild) at safe points and journals them; it does not control other islands
  and is not a general recovery framework.

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

The existing Rust syncer owns this f32 Nesterov update, fragment versions,
checkpoint-before-broadcast ordering, and exact full-roster validation. Yeto
adds no RL wire message and does not modify the Rust syncer.

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

Yeto decides the inner learning-rate schedule explicitly
(`RLRunConfig.algorithm.lr_schedule`, resolved from the sync preset) and both
engine paths translate it into the same four Miles flags; neither path relies
on Miles' implicit default, whose horizon is
`--num-rollout x rollout_batch_size x n_samples_per_prompt / global_batch_size`
(= global rounds x optimizer steps).

| Preset | `--lr-decay-style` | `--lr-decay-iters` | `--lr-warmup-iters` | `--min-lr` |
|--------|--------------------|--------------------|---------------------|------------|
| `strict-avg`, `dense-full` | `linear` | `global_rounds x optimizer_steps` | `0` | `0` |
| `decoupled` | `constant` | `global_rounds x optimizer_steps` (unused; satisfies Megatron's `lr_decay_steps > 0`) | `0` | `0` |
| eval-only | not passed | | | |

- **strict-avg** runs exactly one local round (`optimizer_steps` optimizer
  steps) per global round, so the explicit linear schedule is bit-identical to
  the previous implicit one. The launcher refuses a strict run where
  `rollout_batch_size x n_samples_per_prompt != global_batch_size x
  optimizer_steps`, because the two horizons would then differ. The last
  optimizer step still trains with a positive learning rate.
- **decoupled uses a constant learning rate starting with this change.** A
  decoupled island runs until the syncer's final cut, so its local step count
  is not known up front; the previous implicit linear schedule reached 0 after
  `global_rounds x optimizer_steps` local steps and every later round trained
  with learning rate 0 (gradients computed, parameters unchanged, global delta
  0). Decoupled runs from before this change are therefore not bit-reproducible
  with the new trajectory; the yeto commit in their provenance tells them
  apart. Strict-avg is unaffected.
- The four flags are owned by the adapter: passing them through extra Miles
  argv is rejected on the ports path.

**Zero learning-rate invariant.** Every `rl_local_round` event records
`applied_lr` (the minimum over the round's optimizer steps) and `applied_lrs`
(one value per optimizer step): the learning rate each `optimizer.step()`
actually applied, read from `optimizer.param_groups[*]["lr"]` before Miles
advances the scheduler. (`train/lr` is Miles' post-step value, i.e. the next
step's learning rate.) Ports records it in the state plugin's
`train_one_step` recorder; legacy through Miles'
`--custom-megatron-before-train-step-hook-path`, using the combined hook
`yeto.rl.applied_lr.before_train_step`, which also runs the gradient audit
hook when `YETO_RL_AUDIT_GRADS=1`. If a round applied learning rate 0 and the
island will keep training, the round fails with
`rl_strict_failure metric=zero_lr_before_final_round` (naming the local round
and the learning rate) and is not submitted to the syncer. The final round is
`local_round_id >= global_rounds` for strict-avg; for decoupled it is a round
trained after the syncer announced the final cut (or the round that exhausts
`learner_budget_steps`).

The dry-run plan below selects the decoupled preset without GPUs; it does not
print Miles argv. The resolved flags for each preset, and their equality across
the two engine paths, are pinned by `tests/test_rl_argv_snapshot.py` and
`tests/test_rl_miles_adapter_config.py`.

```bash
python3 scripts/benchmark_rl.py --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data openai/gsm8k --data-revision e53f048856ff4f594e959d75785d2c2d37b678ee \
  --reward-function project.rewards:score \
  --islands 2 --arms decoupled --rl-engine ports --dry-run
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

The initial validation configuration is `P=8`, `tau=2`, `H=4`; it is not
claimed to be optimal for every model.

### Variance-aware GRPO sampling

CyberGym rewards can be identical across every sample in a group. Such a
zero-variance group has zero GRPO advantage and contributes no useful update.
Enable Miles' DAPO-style filter together with oversampling to replace those
groups before the training batch is formed:

```bash
  --rollout-batch-size 4 \
  --over-sampling-batch-size 16 \
  --dynamic-sampling-filter-path \
    miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

`--over-sampling-batch-size` must be greater than the training batch when the
filter is enabled. For external rewards such as CyberGym, bound the replacement
work and use the colocated weight-sync safety settings:

```bash
  --dynamic-sampling-max-replacements 8 \
  --rl-offload-train \
  --rl-distributed-timeout-minutes 10
```

When the stock Miles filter path is selected, Yeto automatically uses its
bounded equivalent. It prefers non-zero-variance groups, rejects at most the
configured number of zero-variance groups, then accepts one bounded fallback so
the run cannot spend an unbounded time searching for signal. The fallback is
reported as `rl/dynamic_filter/forced_groups`. Yeto also records generated,
accepted, dropped, replacement, and drop-reason metrics in the round evidence,
so the extra rollout work is visible in the benchmark report.

With `--rl-offload-train`, Miles keeps the trainer onloaded through external
policy initialization and the initial weight publication, then offloads it
before rollout zero. This is the safer setting for a large model when a weight
publication has timed out. The timeout is a fail-fast bound for the distributed
barrier, not a quality setting.

`--trust-remote-code` is required by the pinned Miles model-loading path. Keep
model and dataset revisions immutable and enable it only for trusted sources.
The reward callable uses `package.module:function`; Yeto hashes its source
before provisioning and the learner verifies that digest before import.

Prompt rows provide `messages`, or a string `prompt`/`input` that Yeto converts
to a user message. `label`, `metadata`, and `tools` remain available to Miles
and the reward callable. Existing custom generation, session-server, and TITO
arguments are forwarded unchanged.

## Multi-node islands

An RL learner island may span several SkyPilot nodes (`rl-multinode-island`;
ports engine, `--rl-placement fixed-partition` only). `--gpu cloud:NxGxgpu`
asks for one N-node cluster of G GPUs per node per learner:

```bash
yeto launch --training-mode rl --rl-engine ports \
  --gpu nebius:2x8xh100 \
  --rl-placement fixed-partition --rl-rollout-gpus 8 --rollout-num-gpus-per-engine 8 \
  --tensor-parallel 2 --pipeline-parallel 1 \
  --rl-elastic --rl-elastic-resources cfg/resources-2x8.json --rl-elastic-initial-config T8R8S0 \
  ...
```

### Topology and placement rules

- Logical bundle `p` lives on node `p // G` as local GPU `p % G`. The learner
  asserts this against the Miles placement group at startup
  (`StartupBundles`, `ray.util.placement_group_table`) and refuses to train on
  any other layout.
- Ray head = node rank 0 = the learner process = trainer rank 0. Worker nodes
  join the head's Ray (bounded join, cleanup trap armed before joining) and
  leave when the head exits. There is no separate head node.
- `--rl-rollout-gpus` / `--rl-standby-gpus` are **island totals**. The trainer
  takes the leading `total - rollout - standby` bundles and must occupy the same
  number of GPUs on every node it uses (Miles' `actor_num_nodes x
  actor_num_gpus_per_node` rectangle); every TP*PP group, every rollout engine
  and every standby-to-cell rebind stays on one node; EP groups fit inside one
  node. Violations fail at `yeto launch` before any cloud resource is touched.
- The island must have at least the recipe-derived minimum of nodes (one TP*PP
  replica + one rollout engine + standby, rounded up to whole nodes);
  `--rl-min-nodes-per-learner N` raises that floor. Qwen3.8-Flash-Next LoRA
  (trainer 8 + SGLang TP8) needs 2 nodes; the 32-GPU full-parameter recipe 4.
- The launcher forwards `--rl-island-gpus-per-node G` to the learner; a
  single-node island sends nothing and keeps every pre-existing behaviour.

### Resources cfg (`--rl-elastic-resources`)

A legacy cfg (no `nodes`) is parsed exactly as before. A multi-node cfg adds
`nodes`/`gpus_per_node`; `placement` entries may be `"n<k>:<g>"`, logical bundle
integers or pool GPU uuids (one spelling per cfg):

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
nodes; leftovers stay unbound), and `bind_members` refuses a cross-node target.

### Node failure domain

The controller records the island `topology` in the reconfiguration journal
and polls `ray.nodes()` before every round. Any node loss (fewer alive nodes
than declared, or a node exposing another GPU count) is recorded as
`node_lost` and the island enters `RECOVERY_REQUIRED`: the learner exits
non-zero, nothing trains or generates on the surviving nodes. A restarted
learner (E1-D recovery) checks the same topology **before** touching the fork's
membership epoch; with a node still missing it stays `RECOVERY_REQUIRED` and
performs no differential recovery. Same shape with other hostnames is fine.
Rebuilding the island is a manual `yeto up` (design Q4 a).

Operational notes: `NCCL_IB_DISABLE=1`, `NCCL_DEBUG=WARN` and (on nebius)
`NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=<first non-virtual interface that is
up>` are exported on every node (a Nebius node's NIC is `network-interface-0`,
not `eth0`; the job log prints `[yeto-island] socket interface: ...`); set them
in the environment to override. Journal records `topology` / `node_lost`
and the `rl_reconfiguration` event with `result=RECOVERY_REQUIRED` and an
`error` starting with `node_lost:` are the evidence to collect before tearing
down.

### Node loss and rebuild (checkpoint store)

Ruling (2026-10-04, design Q4): on node loss the island stops the affected
training communication group (= the whole island, one learner has one trainer
group), is rebuilt from a **consistent checkpoint** on a same-shape island, and
there is **no automatic re-entry of the original ranks**: a surviving or
returning node never rejoins the old job. The consistent checkpoint is the
elastic state dir (`--rl-elastic-state-dir`, default `~/yeto-rl/elastic-state`:
reconfiguration journal + `epochs.json`, trainer cuts, batch ledger, inbox),
which lives on node0's local disk. Two things make it rebuildable elsewhere:

- **`--rl-checkpoint-store <s3://…|gs://…|/shared/path>`** (needs
  `--rl-elastic`): the controller copies the state dir to the store after every
  commit point (`topology` baseline at startup, accepted `gpu_pool`, every
  `COMMITTED` / `SUCCEEDED` phase record, i.e. after the durable CAS and the cut
  manifest commit) and writes `STORE-MANIFEST.json` last, so a copy without it is
  never trusted. A bucket URI is mounted on every island node at
  `~/yeto-checkpoint-store` (sky Storage MOUNT, persistent) and the learner gets
  `--rl-elastic-checkpoint-store` with that path; an absolute/`~/` path is passed
  through as is (a share every replacement machine also mounts). A learner that
  starts with an **empty** state dir (fresh machines) restores the store into it
  first and journals `checkpoint_store action=restore`; a local journal always
  wins over the store (an in-place restart is at least as new as its last sync,
  and dropping local records could hide a `RECOVERY_REQUIRED` terminal). A
  failed sync is logged and kept in `controller.last_store_sync`, it never fails
  the commit. Without the flag a multi-node launch prints `[launcher] warning:
  --rl-checkpoint-store not set: ... node0's local disk` and keeps the old
  behavior (fail open); single-node islands are unchanged.
- **Parallel layout baseline**: the `topology` journal record now carries
  `layout` (`tp/pp/cp/ep`, trainer GPUs, `nodes` x `gpus_per_node`, the role ->
  logical bundle `bundle_map`, None = leading-bundle layout) and
  `layout_accepted`. Every later incarnation compares its layout against the last
  accepted baseline; a difference is `RECOVERY_REQUIRED` with `layout changed:
  pp 2 -> 1, …` before any fork membership call, in `_recovery_precondition`
  and in `confirm_recovery` (`checks.layout`). A cut written under another
  Megatron layout is not restorable without conversion, so a rebuild must be
  same-shape; there is deliberately no override flag. Like `gpu_pool`, the
  refusal is per incarnation (no journal terminal): relaunching with the
  original recipe/cfg continues.

Rebuild procedure after a node is lost (M4 in the GPU matrix):

1. The driver poll records `node_lost`, the island enters `RECOVERY_REQUIRED`
   and the learner exits non-zero; the restart loop's in-place retries are
   refused by the partial-island preflight (`alive < N`). Collect the evidence
   (journal `node_lost`, tape `rl_reconfiguration result=RECOVERY_REQUIRED`).
2. `yeto down` the island (teardown confirms every node instance).
3. `yeto launch` with the **same cfg, recipe and parallel flags** (same
   `--rl-elastic-resources`, `--rl-elastic-initial-config`, `--tensor-parallel`
   / `--pipeline-parallel` / `--expert-parallel`, `--rl-rollout-gpus`,
   `--rl-standby-gpus`, same `--rl-checkpoint-store`) plus
   `--rl-elastic-accept-rebind`: the new machines expose other GPU uuids, which
   the Q6 reconciliation accepts only with that flag (`gpu_pool rebind=true`
   with the old -> new mapping; shape must still match).
4. Startup order on the new island: restore the state dir from the store
   (`checkpoint_store restore`), `topology` record + layout check against the
   restored baseline, `gpu_pool` reconciliation + rebind record, then the normal
   restart recovery of the committed membership (`RECOVERING` ->
   `confirm_recovery` after the first publication) and training continues from
   the committed epoch / last cut.
5. A layout or shape mismatch at step 4 is `RECOVERY_REQUIRED` again; fix the
   launch flags, do not force it. A changed `--rl-elastic-state-dir` with an
   already populated directory is an in-place restart, not a rebuild.

Not supported: a node coming back under its old job (the journal keeps
`RECOVERY_REQUIRED`), partial continuation on surviving nodes, rebuilding under
another parallel layout, and Modal islands (no sky Storage mount; use a shared
path there). Megatron `--save`/`--load` checkpoints are not managed by the
store (the cut plugin never sets them).

### Teardown confirmation

`yeto down` / the launcher's teardown confirm a multi-node island per node
instance: the instance ids are captured from the cloud before `sky down`, and
the island counts as released only when the cloud reports every one of them
terminated (`node instance <id> confirmed terminated` lines). No cloud probe, a
failing probe or an instance still alive after the retries prints `UNCONFIRMED
node instance(s) <ids>`, the cluster is reported as not verified and the run's
teardown exits non-zero; delete the listed instances in the cloud console or
rerun `yeto down`. `sky down`'s own success is never trusted for more than one
node.

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

## Finalization and Export

At `T=N*P`, the syncer stops ordinary pulls, persists the quiescent cut, and
sends all f32 final fragments plus an exact manifest. At the next safe
boundary, each island:

1. stops ordinary fragment submission;
2. assembles and applies the complete final cut;
3. verifies the trainer's full-policy hash;
4. writes its final progress checkpoint;
5. acknowledges the exact manifest;
6. asks Miles to stop only after one complete SGLang weight publication.

A replacement island that joins while finalization is pending consumes the
terminal manifest directly, applies it with recovery optimizer reset,
acknowledges it, publishes it once, and exits without an extra rollout.

Strict export needs the original model contract:

```bash
yeto-rl-export \
  --checkpoint ./rl-output/yeto-state.ckpt \
  --model org/model \
  --model-revision <immutable-commit> \
  --lora-r 8 \
  --lora-targets attention \
  --output-dir ./adapter
```

Decoupled export additionally receives the training layout:

```bash
yeto-rl-export \
  --checkpoint ./rl-output/yeto-state.ckpt \
  --model org/model \
  --model-revision <immutable-commit> \
  --lora-r 8 \
  --lora-targets attention \
  --sync-preset decoupled \
  --fragments 8 \
  --pipeline 2 \
  --local-horizon 4 \
  --output-dir ./adapter
```

The exporter reconstructs the canonical specs and fragment layout, verifies
the sync fingerprint and terminal version sweep, rejects non-finite or
mismatched tensors, and writes standard `adapter_model.safetensors` and
`adapter_config.json`. Decoupled export also writes
`yeto_rl_provenance.json` with `P/tau/H/N/T`, outer optimizer settings, final
fragment versions, both layout hashes, full-policy hash, and checkpoint SHA256.

### Starting a fresh phase

A completed Decoupled adapter can seed policy version zero of another
Decoupled run:

```bash
yeto launch \
  --training-mode rl \
  --rl-sync-preset decoupled \
  --rl-initial-adapter ./adapter \
  --rl-initial-adapter-sha256 <optional-expected-sha256> \
  ...
```

`--rl-initial-adapter` accepts only a local directory. The launcher hashes the
directory before provisioning, rejects a supplied digest that differs, and
mounts the attested directory read-only at the same path on every island. The
learner rehashes it before rollout zero and requires a standard causal-LM LoRA
adapter whose base model, immutable revision, rank, targets, exact tensor
names and shapes, and finite values match the new run. RS-LoRA, DoRA,
fan-in/fan-out, and per-module rank or alpha modifiers are rejected. Its
Decoupled export provenance and recorded full-policy hash must also match the
loaded tensors.

This starts a new phase rather than extending the terminal phase in place.
The parent policy tensors are preserved exactly, while the Miles inner
optimizer and scheduler, Yeto outer optimizer and syncer, rollout IDs,
fragment versions, and checkpoints all start fresh. Runtime budget changes,
reopening a finalized syncer, and exact optimizer-state continuation are not
provided. Normal checkpoint recovery within the new phase remains unchanged.
Use a new `--cluster-prefix` for each fresh phase; reusing an existing run
identity keeps the normal in-phase recovery behavior instead.

## Engine selection

`--rl-engine {legacy,ports}` selects how an island drives Miles. It is accepted
by `yeto launch`, `python3 -m yeto.rl.learner`, the SSH harness (through the
launch arguments), `scripts/benchmark_rl.py` and `yeto-rl-export`. The default
is `ports`; the choice is never inferred from the model, island count or other
flags, is fixed before startup, and does not change during a run.

`legacy` stays available for one release as an explicit opt-in: pass
`--rl-engine legacy` to keep the agentenv fork path, which is still required
for every combination outside the ports boundary below (SAO, `dense-full` /
full-parameter, the DeepSeek V4 recipe, critic / non-GRPO estimators, fixed
partition, non-causal models) and for the benchmark's `native` arm. Because the
default is `ports`, such a run without the flag now fails before startup with a
message naming `--rl-engine legacy` instead of silently taking the fork path.
The legacy path is deprecated and is removed after task 7.2 of the
`rl-engine-ports` change; migrate LoRA/GRPO runs to `ports`.

| path | engine source | who owns the island loop |
| --- | --- | --- |
| `legacy` (explicit, deprecated) | `agentenv/miles` fork at `MILES_COMMIT`, installed from the bundled git bundle | Miles `train.py`; Yeto plugs in through the external policy sync callback (`MilesPolicySync`) |
| `ports` (default) | `michaellchung/miles` at `MILES_NEXT_COMMIT` and `michaellchung/sglang` at `SGLANG_NEXT_COMMIT`, fetched directly (no bundle) | Yeto's `IslandDriver` (`yeto/rl/engine/driver.py`); upstream Miles is used as a library through role ports |

Each path pins and verifies its own source (repository, commit, clean detached
checkout, import path) before any model is loaded; changing one pin group never
touches the other. With `ports`, the learner calls
`verify_miles_revision(..., expected=MILES_NEXT_PINS)`.

With `legacy`, source preparation, events, plans and exported provenance are
byte-for-byte what they were before the flag existed (a legacy SSH plan has no
`rl_engine` key, so its digest is unchanged, and a legacy benchmark resume
identity has no `rl_engine` field, so existing legacy runs still resume with
`--rl-engine legacy`). The one difference is that the learner command now
carries `--rl-engine legacy` explicitly, because the learner's own default is
`ports`. With `ports` (explicit or default), the selection is recorded: the
learner command carries `--rl-engine ports`, the
island tape starts with an `rl_engine_selected` event (`rl_engine=ports`, Miles
commit, `rl/algorithm_spec_sha256`, placement), the SSH plan carries
`"rl_engine": "ports"`, launcher provenance carries `rl_engine`, and
`yeto_rl_provenance.json` of the exported adapter contains `"rl_engine": "ports"`.

### Ports boundary (R0)

`ports` supports exactly: causal LM + LoRA, GRPO, serial colocated execution,
and the `strict-avg` and `decoupled` presets. Everything else is rejected
before startup with a message pointing at `--rl-engine legacy`:

| rejected on `ports` | detected from |
| --- | --- |
| full-parameter / `dense-full` | `--parameter-mode full`, `--sync-preset dense-full`, `--tuning full` |
| SAO streaming compaction | an `sao*` preset or `--sao-*` Miles arguments |
| DeepSeek V4 recipe | `--rl-model-recipe deepseek-v4-flash`, `attention-routed-experts`, `--expert-full-count > 0` |
| critic / non-GRPO | `--use-critic`, `--advantage-estimator` other than `grpo` |
| fixed partition | `--rollout-num-gpus` (dedicated rollout GPUs) |

The ports Miles translation also rejects fork-only options with no upstream
equivalent (`--rollout-engine-base-port`, `--train-master-base-port`,
`--sglang-router-prometheus-port`, `--custom-agent-function-path`,
`--tito-allowed-append-roles`) and all Miles fault-tolerance flags. The
benchmark's `native` arm measures stock Miles' own loop and is rejected with
`--rl-engine ports`.

### Algorithm specs (`--rl-algorithm-spec`)

On `ports` the training objective is described by one `AlgorithmSpec`
(`yeto/rl/engine/algorithm.py`), and the Miles algorithm flags are generated
only from it. The spec has these groups: `advantage` (estimator,
`std_normalization`, `rewards_normalization`, `whiten`, `reward_postprocess`,
`reward_binary`), `loss` (`variant`, `eps_clip`, `eps_clip_high`,
`eps_clip_c`, `aggregation`, `reducer`, `custom_loss`), `kl` (`placement`,
`coef`, `estimator`, `unbiased`), `correction`, `sampling`, `execution`,
`entropy_coef` and `plugins`. A plugin is written as `{path, sha256}` (the
SHA256 of the plugin module's source file). Only the `yeto.` and `miles.`
namespaces are accepted. The learner re-hashes and imports every plugin before
it joins outer sync, and refuses to start if the hash differs.

Where the spec comes from:

1. `--rl-algorithm-spec PATH`, a v1 or v2 JSON file. `yeto launch` and
   `python3 -m yeto.rl.learner` both accept it; it applies to `ports` only.
2. Without that flag, the legacy CLI builds the spec exactly as in R0.
3. Mapped Miles flags found in the extra argv are absorbed into the spec.

A spec that only uses R0 fields keeps the R0 v1 canonical JSON, so its hash
is unchanged. Setting any new field switches the spec to
`yeto-rl-algorithm-spec-v2`.

**Absorb and reject.** `yeto/rl/engine/miles_adapter/algorithm_flags.py` maps
each Miles flag to a spec field. Every flag in that table is adapter-owned:

- A mapped flag in the extra argv is absorbed. It is recorded as
  `rl/algorithm_absorbed_flags` in the `rl_engine_selected` event.
- A value that disagrees with the spec is refused, and the error shows both
  values.
- A flag that changes the objective but has no mapping yet is refused. These
  are the flags in `UNMAPPED_OBJECTIVE_FLAGS`, for example `--gamma` or
  `--rollout-temperature`.

**KL placement.**

- `kl.placement=loss` translates to
  `--use-kl-loss --kl-loss-coef C --kl-loss-type T`, and `T` must be given.
- `placement=reward` translates to `--kl-coef`. With `grpo` or `gspo` and a
  coefficient above 0 it is refused, because Miles drops a KL placed in the
  reward for these estimators; use `placement=loss` instead.
- An R0 `kl_coef` is read as `placement=reward`. `0.0` still emits
  `--kl-coef 0.0`, so the hash is unchanged. `placement=none` emits no KL flag,
  and Miles loads no reference model.

**Capabilities and execution.** The engine declares what it supports per
mechanism dimension: `advantage_estimators`, `losses`, `loss_aggregations`,
`kl_placements`, `corrections`, `reward_postprocessors`,
`dynamic_sampling_filters` and `features`. It also declares an `execution`
block: `critic=false`, `max_policy_staleness=0` and `rollout_logprobs=true`.

Before any GPU process exists, the driver handshake refuses:

- every mechanism the spec requires that the engine does not declare;
- a critic;
- a policy age above `execution.max_policy_staleness`, which is fixed at 0;
- every entry of the rejection matrix:
  - TIS together with `use_rollout_logprobs`;
  - reward KL together with loss KL;
  - `gspo` without an explicit clip range;
  - a mechanism that needs a binary reward when the reward is not declared
    binary;
  - `reinforce_plus_plus*` without `whiten`.

**Declaration policy** (main-agent decision, may be overridden by the user;
alignment §7b): a mechanism is declared in `miles_capabilities` only on
evidence that it actually takes effect on GPU. The declarations beyond R0,
each with its evidence, are the `MILES_DECLARED` table in
`yeto/rl/engine/miles_adapter/entry.py` (one commit per mechanism):

- corrections: tis, opsm, opsm_trainer, icepop, mis_mask, mismatch_observe
  (rl-algo-mismatch-correction);
- loss_aggregations: constant, token (token only on Miles 0af62f4d+, where
  the LoRA bridge honours calculate_per_token_loss; entry.MILES_DECLARED_PINS
  withholds it under other pins); features: over_sampling (Miles 0af62f4d only;
  weak evidence, see MILES_DECLARED), overlong_filter (Miles 0af62f4d; needs
  the rollout hook of integ-decl 21912fe+), clip_higher (Miles 0af62f4d; run on
  0394715 and transferred by a code diff, see MILES_DECLARED), kl_loss_ref_model, entropy_bonus,
  overlong_penalty, no_grpo_std_normalization (g1c isolated control), eps_clip
  (g1b run A-r1; eps 0.001/0.002 are trigger test values, not
  recommendations); kl_placements: loss; reward_postprocessors:
  custom_reward_postprocess (rl-algo-grpo-knobs);
- advantage estimators: gspo, reinforce_plus_plus,
  reinforce_plus_plus_baseline; features: maxrl, mapo, gdpo
  (rl-algo-seq-and-adv).

Withdrawn after independent review:

- loss_aggregations:token: grad_norm was bit-identical to the baseline.
- features:no_grpo_std_normalization: no run isolates it from the
  `constant` aggregation.
- features:mismatch_metrics: every evidence run already had use_tis, and
  Miles emits the metrics under `get_mismatch_metrics or use_tis`.

Under a correction that makes Miles set use_tis (tis, icepop, mis_mask,
mismatch_observe), `correction.mismatch_metrics` is claimed by that correction
(`CORRECTION_COMPANIONS`; main-agent decision, may be overridden by the user),
because the flag has no effect there. icepop and mismatch_observe specs are
therefore accepted. Under a generic custom function it is still a separate,
undeclared mechanism.

Not declared, pending evidence or approval:

- dual_clip;
- mis, opsm_rollout, generic corrections:custom;
- features:custom_pg_loss_reducer (generic). 1b now allows only its Dr.GRPO
  reducer, and that reducer is claimed by `loss_aggregations:constant`
  (`register_named_reducer`).

Settings an estimator mandates are claimed by that estimator's mechanism in
that combination only (`ESTIMATOR_COMPANIONS`; main-agent decision, may be
overridden by the user): GSPO's explicit clip range and the rpp family's
advantage whitening. The same settings under grpo are still separate,
undeclared mechanisms.

Measured on integ-decl with the committed example specs:

- accepted: gspo, rpp, rpp_baseline, maxrl, gdpo;
- accepted: dapo-like;
- dr-grpo is accepted after the no_grpo_std_normalization re-declaration. Its
  reducer is claimed by `constant` only at the evidenced source hash.

**Combinations are not GPU-verified.** Each declared mechanism has its own
GPU evidence. Combinations such as tis+opsm_trainer or icepop+opsm_trainer
have none, so they are accepted but unverified.

"Expressible, not enabled" means the spec can describe and translate a
mechanism, but `miles_capabilities` does not declare it yet. A follow-up
algorithm change declares it after its single-GPU smoke passes.

`--rl-allow-unverified-mechanism DIMENSION:NAME` (repeatable, for example
`features:clip_higher`) exempts only the named mechanisms from the "not
declared" check, and only on a single-island run **without outer sync**.
Every launched run and every `yeto.rl.learner` run joins a syncer, so today
the allowance is refused on those entry points; only the fake engine and the
dry run (which models a sync-less single island) accept it.
Every other check still applies. The allowance does not change the hash. It is
recorded as `rl/unverified_mechanisms` in the event and in
`yeto_rl_provenance.json`, which is also marked
`contains_unverified_mechanisms`.

**Island consistency and provenance.** On `ports`, `yeto launch` builds the
spec once and sends each island two things:

- its canonical JSON, as `~/yeto-rl/algorithm_spec.json`;
- `--rl-expected-algorithm-sha256`.

Each learner compares its own hash with the expected one before it joins outer
sync:

- on a mismatch it writes an `rl_algorithm_mismatch` event and exits;
- when the expected hash is missing (a learner started by hand) it writes a
  warning event.

Visible default-path changes on `ports` (no algorithm option given):

- the island learner command always carries
  `--rl-expected-algorithm-sha256 <hash>`;
- the tape ends with an `rl_learner_finalized` record;
- the `rl_engine_selected` event additionally carries `rl/algorithm_spec`
  (canonical JSON) and `rl/algorithm_absorbed_flags` (`{}` by default).

The Miles argv of default GRPO is byte-identical to R0.

`yeto-rl-export --rl-algorithm-spec PATH` writes `algorithm_spec`, the
canonical JSON, and `algorithm_spec_sha256` to the ports provenance. The
legacy provenance is unchanged.

**Event tapes of Modal islands.** A Modal island's `~/yeto-output` cannot be
fetched. So every ports Modal island, and every `--rl-single-island-no-sync`
island, runs with `--rl-echo-events`: the learner prints each tape record as
`YETO_RL_EVENT <json>` (`yeto/rl/event_echo.py`), and the launcher rebuilds
`<run dir>/events/<island>.jsonl` from the log stream.

The check fails closed. A tape without `rl_learner_finalized`, for example a
stream cut when the container exited, gets a `.incomplete` marker and makes
the run exit 3. A synced run still fetches its checkpoint first.

`--rl-event-tape` export refuses incomplete tapes unless
`--allow-incomplete` is given.

**Launcher exit codes.**

| code | meaning |
| --- | --- |
| 0 | success |
| 1 | a learner failed (non-RL), or no learner succeeded |
| 2 | artifact not fetchable (Modal island) |
| 3 | incomplete island event tape |
| 4 | a fixed-roster RL island could not be recovered |
| 5 | the Modal app was not confirmed stopped after teardown |
| 6 | the run stalled: no island event for `--rl-stall-timeout` seconds (default 900, 0 disables) and not every island finalized |

For exit 5, every row of `modal app list` with the run's app name must be
`stopped` with 0 tasks; an earlier run's row with the same name counts too.
If no row is listed any more, that also counts as stopped. The rows created
after this run started are this run's app, and their app ids are recorded.
The launcher checks at most 5 times. It prints a WARN
naming the `modal app stop` command to run by hand. The result is written to
`<run dir>/teardown.json`. Exit 5 takes precedence over 0/2/3/4, because a
possibly still-running app matters more than the run's own outcome.

The stall check needs the event echo, so it applies to every ports RL
island. Its clock starts at the first received event, so islands still pulling
their image or loading the model do not count as stalled. On a stall the
launcher drains the tapes (bounded), tears everything down and does not
relaunch. A typical cause is a dead island-syncer connection.

Strict syncer failures, strict RL job failures, "all learners abandoned" and
internal errors propagate as exceptions (exit 1 from the CLI worker).

An island whose tape already holds `rl_learner_finalized` is counted as
succeeded even if its job then ends non-zero, for example an interrupt during
Ray shutdown after the syncer stopped. It is not relaunched, and the syncer is
not restarted once every learner has finalized.

**Launch dry run.** `yeto launch ... --dry-run` validates the whole launch
(arguments, provenance, the ports algorithm and capability checks) and prints
JSON with the resource request (GPU type and count per island, island count,
whether a syncer is started), the algorithm hash and each learner command. It
creates no cloud resource. With `--rl-single-island-no-sync` (which needs
`--controller local`) it shows one island, no syncer and `outer_sync: false`.

**Dry run.** `python3 -m yeto.rl.engine.miles_adapter.algorithm_flags
--dry-run [--rl-algorithm-spec PATH] [--extra "<miles argv>"]
[--rl-allow-unverified-mechanism NAME]` runs the same steps as the ports
learner, with no engine: resolve, absorb, rejection matrix, then the Miles
adapter's declaration. It prints the canonical spec, the hash, the absorbed
flags, the Miles algorithm flags and the verdict. The exit code is 0 only when
the spec is accepted.

```bash
M="python3 -m yeto.rl.engine.miles_adapter.algorithm_flags"
$M --dry-run          # default GRPO: v1 schema, hash 27df1133..., accepted
echo '{"schema":"yeto-rl-algorithm-spec-v2","loss":{"eps_clip_high":0.28}}' > clip_higher.json
$M --dry-run --rl-algorithm-spec clip_higher.json
    # rejected: features mechanism 'clip_higher' not supported (expressible but not enabled)
$M --dry-run --rl-algorithm-spec clip_higher.json --rl-allow-unverified-mechanism features:clip_higher
    # accepted; hash 1b49346c...
$M --dry-run --extra "--eps-clip-high 0.28" --rl-allow-unverified-mechanism features:clip_higher
    # accepted; absorbed {"--eps-clip-high": "0.28"}; same hash 1b49346c...
$M --dry-run --rl-algorithm-spec clip_higher.json --extra "--eps-clip-high 0.3"
    # rejected: ... sets loss.eps_clip_high=0.3 but the algorithm spec has ...=0.28
$M --dry-run --extra "--kl-coef 0.1"   # rejected: ... use kl.placement='loss'
$M --dry-run --extra "--gamma 0.9"     # rejected: --gamma ... not part of the algorithm spec yet
```

The recorded outputs are in
`openspec/changes/rl-algorithm-capabilities/evidence/2026-09-29-dry-run/`.

**Extending (follow-up algorithm changes).** Add a module under
`yeto/rl/algos/` and one line to `yeto.rl.algos.EXTENSION_MODULES`. The module
registers what it needs:

- in `algorithm.py`: `register_field`, `register_mechanism`,
  `register_rejection`, `register_launch_check`, `register_island_check`,
  `register_runtime_attrs`, `register_gradient_rule`;
- in `algorithm_flags.py`: `register_flag`.

A registered field enters the v2 canonical JSON only when it differs from its
default, so registering one never changes an existing hash. The mechanism is
declared in `miles_adapter/entry.py::miles_capabilities` only after its
single-GPU smoke (G1) passes.

**When Miles is upgraded.** Update `MILES_NEXT_COMMIT`, then re-review the
objective-changing flags of the new `miles/utils/arguments.py` (the algorithm,
rollout and reward groups):

1. Add each new flag to the mapping table or to `UNMAPPED_OBJECTIVE_FLAGS`.
2. Run `tests/test_rl_algorithm_flags_upstream.py` in the upstream venv:

   ```bash
   PYTHONPATH=<miles checkout>:$PWD:$PWD/tests \
     <miles venv>/bin/python -m pytest -q tests/test_rl_algorithm_flags_upstream.py
   ```

   It checks that every listed flag exists upstream and that upstream
   `parse_args` accepts every non-default mapping.
3. Re-check the estimator and KL rules in `algorithm.py` against
   `loss_hub/advantages.py` and `miles/ray/specs/train.py`.

### GRPO-family knobs (`rl-algo-grpo-knobs`)

> **Availability.** The modules, fields and mechanisms in this section are
> provided by the `rl-algo-grpo-knobs` change and take effect on the integration branch
> that contains it. The P0 framework branch (`algo-cap`) alone does not ship
> them; there only the extension points described above exist.

Registered by `yeto/rl/algos/grpo_knobs.py`. Every mechanism below can be
expressed and translated, but none is declared in `miles_capabilities` until
its single-GPU smoke (G1) passes. A declared mechanism means only that G1
(and G3 where it applies) passed. It says nothing about training gains.

| Mechanism | Spec | Miles argv | Constraint (checked before any GPU process) |
|---|---|---|---|
| clip-higher | `loss.eps_clip_high` (and `eps_clip`) | `--eps-clip-high` | finite, > 0 |
| dual-clip | `loss.eps_clip_c` | `--eps-clip-c` | > 1 (Miles asserts the same, later) |
| token aggregation | `loss.aggregation="token"` | `--calculate-per-token-loss` | |
| Dr.GRPO, no std | `advantage.std_normalization=false` | `--disable-grpo-std-normalization` | |
| Dr.GRPO, constant denominator | `loss.aggregation="constant"`, `loss.constant_denominator=D`, `loss.reducer=yeto.rl.algos.reducers.constant_denominator_reducer` | `--custom-pg-loss-reducer-function-path` | D finite > 0; not with token aggregation (one enum; `--calculate-per-token-loss` in extra argv conflicts); CP = 1 |
| KL loss | `kl.placement="loss"`, `coef`, `estimator` ∈ k1/k2/k3/low_var_kl, `kl.ref_model={source, revision}` | `--use-kl-loss --kl-loss-coef --kl-loss-type` | `ref_model` required; its revision must equal the island's `--model-revision` |
| entropy | `entropy_coef` | `--entropy-coef` | finite |
| over-sampling | `sampling.over_sampling_batch_size` | `--over-sampling-batch-size` (R0 slot, from the run config) | needs `sampling.filter`; ≥ rollout batch size |
| overlong penalty | `advantage.reward_shapers=[{name: overlong_penalty, max_length, cache_length}]` + the dispatcher | `--custom-reward-post-process-path yeto.rl.algos.reward_pipeline.post_process` | 0 < cache_length ≤ max_length ≤ `rollout_max_response_len` |
| overlong filter | `sampling.overlong_filter=true` | none (the shared `--rollout-sample-filter-path` hook) | |

**Dr.GRPO and RLOO.** The constant-denominator reducer is vendored from Miles
`9e4260d` `examples/experimental/DrGRPO/custom_reducer.py` (blob `96390ac3`).
The only change is that D comes from the spec instead of the constant 1000.
It applies to pg_loss only; clipfrac, KL and entropy keep the default reducer.
RLOO is not implemented separately: with std normalization off its advantage
is G/(G−1) times the Dr.GRPO advantage, and Adam is nearly invariant to a
constant gradient scale.

**KL loss.** Miles loads the `--ref-load` model only when `kl_coef != 0` or
`use_kl_loss` (`miles/ray/specs/train.py:58`), so KL loss adds a reference
model and one extra reference forward pass. G1 records the effect on peak
memory and round time. `kl.ref_model` is part of the algorithm hash, so two
islands with different references disagree before outer sync. The learner
also refuses a reference revision that differs from `--model-revision`
(`rl_algorithm_island_rejected`).

**Reward dispatcher.** `yeto.rl.algos.reward_pipeline.post_process` is the
only reward post-processing hook on `ports`. It is emitted only when the spec
selects a reward shaper or a non-default `advantage.transform`; default GRPO
keeps Miles' built-in path, and its argv is unchanged. The dispatcher runs
`raw → shapers → advantage transform`. `grpo_default` is element-wise equal
(`torch.equal`) to Miles `_post_process_rewards`: prompt groups, one reward
per multi-segment rollout, the error on inconsistent siblings, G=1, std=0,
`+1e-6`, and the estimator and `rewards_normalization` gates. The comparison
is `tests/test_rl_reward_pipeline_equivalence.py`; it pins the SHA256 of
`train_data_conversion.py`. Details:

- Multi-LoRA is refused, because a custom post-process cannot see
  `prompt_group_sizes`.
- A batch with no `group_index` and no fixed fan-out falls back to one
  whole-batch group, as Miles does, and emits `rl_reward_group_fallback`.
- The configuration reaches Miles as `args.yeto_algo_plugins = {config,
  sha256}`, set through the spec's runtime attrs. The dispatcher and reducer
  re-hash it and refuse a mismatch.

**Adding an advantage transform (P2).**

1. Register it in `reward_pipeline.py` with `register_advantage_transform(name,
   fn(args, samples, rewards, groups, params), validate=...)`. This keeps it
   under the dispatcher's PluginRef hash.
2. Select it with `advantage.transform=name` plus `transform_params`.
3. Register its mechanism and leave it undeclared until G1 passes.
4. Add a CPU test that compares it with a hand computation, and with
   `grpo_default` in its degenerate case.

Any edit to `reward_pipeline.py` changes the dispatcher's source hash:
regenerate specs that pin it (for example `examples/rl_algorithms/*.json`).

**Overlong penalty (DAPO).** For a response of length L:

- L ≤ Lmax − Lcache: penalty 0;
- Lmax − Lcache < L ≤ Lmax: penalty (Lmax − Lcache − L)/Lcache;
- L > Lmax: penalty −1.

The penalty is added to the raw reward before group normalization. L is the
summed length of a rollout's segments, so siblings stay consistent. The
shaped reward is what Miles logs as raw reward. The original reward is kept
in `sample.metadata["yeto_raw_reward"]`, and an `rl_reward_shaping` event
summarizes raw against shaped rewards.

**Overlong filter vs DAPO.** Truncated samples get `remove_sample=True` in
`record_trained_groups`. Checked on CPU against Miles:

- Their loss mask becomes all zero, but their reward still enters the group
  mean and std, so the other samples' advantages are unchanged.
- Under the default sample-mean aggregation a removed sample contributes 0 to
  the numerator. It still counts in the `global_batch_size` divisor
  (`loss.py:197-210`), which dilutes the others.
- Under token aggregation it adds 1 to the reported token count
  (`clamp_min`).

DAPO's paper masks truncated samples in the loss but does not say whether
their reward enters the group statistics. The rollout metadata carries
`filtered_samples`, which the ledger records in the terminal state `filtered`.
Over-sampling leaves no reusable remainder: once a rollout has its batch,
Miles does not return further kept groups to the buffer
(`sglang_rollout.py:505-510`); only samples aborted under `--partial-rollout`
go back. A round in which every
non-zero-variance group was filtered completely does not trip the zero-gradient
invariant.

**Over-sampling and outer averaging.** strict-avg and decoupled average the
islands' deltas with equal weights, not weighted by sample count (inferred;
not verified). Islands that train on different numbers of samples therefore
count equally. The weighting rule is unchanged. Recording each island's
trained samples and groups in the per-round event is task 7.2; it needs a
driver-side event field and is still open.

```bash
M="python3 -m yeto.rl.engine.miles_adapter.algorithm_flags"
$M --dry-run --rl-algorithm-spec examples/rl_algorithms/dr-grpo.json
    # rejected: custom_pg_loss_reducer / no_grpo_std_normalization not declared (pre-G1)
$M --dry-run --rl-algorithm-spec examples/rl_algorithms/dr-grpo.json \
   --rl-allow-unverified-mechanism constant --rl-allow-unverified-mechanism custom_pg_loss_reducer \
   --rl-allow-unverified-mechanism no_grpo_std_normalization   # accepted, hash 725e4216...
```

Recorded outputs: `openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-dry-run/`.

### Sequence-level ratio and advantage variants (`rl-algo-seq-and-adv`)

> **Availability.** The modules, fields and mechanisms in this section are
> provided by the `rl-algo-seq-and-adv` change and take effect on the integration branch
> that contains it. The P0 framework branch (`algo-cap`) alone does not ship
> them; there only the extension points described above exist.

Six optional mechanisms, all **expressible but not declared** by the Miles
adapter until their single-GPU smoke (G1) passes. Declared support is not a
claim of benefit: no effect A/B has been run for any of them.

| Mechanism (`--rl-allow-unverified-mechanism dimension:name`) | Spec | Engine argv |
|---|---|---|
| GSPO (`advantage_estimators:gspo`) | `advantage.estimator="gspo"` + explicit `loss.eps_clip` / `loss.eps_clip_high` | `--advantage-estimator gspo --eps-clip .. --eps-clip-high ..` |
| REINFORCE++ (`advantage_estimators:reinforce_plus_plus`) | `advantage.estimator`, `advantage.whiten=true`, optional `kl.placement="reward"` | `--normalize-advantages [--kl-coef ..]` |
| REINFORCE++-baseline (`advantage_estimators:reinforce_plus_plus_baseline`) | same | same |
| MaxRL (`features:maxrl`), MAPO (`features:mapo`) | `advantage.transform`, `advantage.reward_binary=true`, dispatcher + `plugins=[seq_adv ref]` | `--custom-reward-post-process-path yeto.rl.algos.reward_pipeline.post_process` |
| GDPO (`features:gdpo`) | `advantage.transform="gdpo"`, `advantage.gdpo={components:[{name,weight}], whiten:true}`, dispatcher + `plugins` | same |

Example specs (PluginRef SHA256s regenerated by `make_examples.py`) live in
`openspec/changes/rl-algo-seq-and-adv/examples/`. Check one without an engine:

```bash
python3 -m yeto.rl.engine.miles_adapter.algorithm_flags --dry-run \
  --rl-algorithm-spec openspec/changes/rl-algo-seq-and-adv/examples/maxrl.json
# verdict "rejected": features mechanism 'maxrl' not supported ... (not declared yet)
python3 -m yeto.rl.engine.miles_adapter.algorithm_flags --dry-run \
  --rl-algorithm-spec openspec/changes/rl-algo-seq-and-adv/examples/maxrl.json \
  --rl-allow-unverified-mechanism features:maxrl \
  --rl-allow-unverified-mechanism reward_postprocessors:custom_reward_postprocess \
  --rl-allow-unverified-mechanism features:plugins
# verdict "accepted" (single island without outer sync only; recorded as unverified)
```

**GSPO.** The clip range must be explicit (`gspo_noclip.json` is rejected: the
engine default `--eps-clip 0.2` is a token-level value; the GSPO paper uses
3e-4 / 4e-4). yeto sets no default. Watch the per-round clip fraction
(`masked_fraction`, from Miles `pg_clipfrac`): a round whose every sequence is
clipped has a legitimately zero gradient and does not trip the zero-gradient
invariant; an unknown clip fraction keeps the strict rule. With a single
optimizer step per round Miles recomputes the old policy with the current
weights, so the ratio is ~1 and clipping never binds: GSPO then is a
sequence-ratio GRPO (clip fraction ~0). GSPO combined with an advantage
transform is not opened.

**REINFORCE++ family.** `advantage.whiten=true` is required (Miles asserts
it). Whitening is `--normalize-advantages`: token-level, all-reduced inside the
island's DP group only; statistics never cross islands or enter the outer
protocol. Each island's advantages are therefore normalized by its own
statistics; how that interacts with the outer delta average is an inference,
not measured. `kl.placement="reward"` is allowed and enters the advantage
(loads the reference model). `advantage.gamma` (`--gamma`) is mapped but only
1.0 is open (`rpp_gamma.json` is rejected: supported values [1.0]); with any
other estimator a non-default gamma is rejected. `--lambd` stays unmapped.

**MaxRL / MAPO** require a binary {0,1} reward, declared by
`advantage.reward_binary=true` and checked at runtime after reward shaping (a
non-binary reward fails the round); they are refused with the
`overlong_penalty` reward shaper (use `sampling.overlong_filter`). They compute
per rollout (multi-segment rollouts merged, result shared by every segment),
only with `advantage.estimator="grpo"`. MaxRL: `(r - mean)/mean`, 0 for
all-wrong groups and single-sample groups. MAPO:
`(1-lam)(r-mu)/sigma + lam(r-mu)/mu`, `lam = 1-4p(1-p)`, sigma as Miles' GRPO
(unbiased std + 1e-6); at p=0.5 it equals Miles' GRPO normalization exactly.
MAPO's published evidence is weak; it is opened only as an option.

**GDPO reward vector.** The reward function writes
`sample.metadata["yeto_reward_components"] = {name: float}` for exactly the
declared components (example: `yeto.rl.algos.gdpo_reward.reward_func`,
correctness + format); a missing, extra or non-finite component fails the
round (no default fill). Each component is normalized inside its group, the
weighted sum is whitened over this island's whole rollout batch (sample
level, island-local). The scalar `sample.reward` is only for metrics.

All three transforms run in the yeto dispatcher; their module
(`yeto.rl.algos.seq_adv`) must be listed in `plugins` so its source SHA256
enters the algorithm hash. Each round emits `rl_advantage_transform`
(all-wrong / all-right groups, non-zero advantages).

### Policy-loss variants (`rl-algo-loss-variants`)

> **Status: expressible, not opened.** Route B (user decision 2026-09-30):
> the computation is a variant branch in the Miles fork (`michaellchung/miles`
> `yeto/ports`, `--policy-loss-variant`); yeto only describes, translates and
> validates. The pin `yeto.rl.MILES_NEXT_COMMIT` is fork commit 5c1b49eb
> (listed in `yeto.rl.algos.loss_variants.FORK_COMMITS`; on any other pin a
> launch is refused). The Miles adapter does **not** declare `losses:cispo` /
> `losses:sapo` / `losses:gmpo` until their GPU smoke passes; only the
> single-island `--rl-allow-unverified-mechanism losses:<v>` entry can run
> them. GPU validation is paused by the user; the variants are CPU-tested only.
> No effect A/B has been run and none is claimed.

The variants replace only the token-level pg_loss inside `--loss-type
policy_loss`; TIS / IcePop / OPSM weights, KL loss, entropy and the
aggregation are applied afterwards by Miles' unchanged code (variant loss
first, then the correction weight, then aggregation). With rho = pi_theta /
pi_old and A the advantage:

| Variant (`loss.policy_loss_variant`) | Token loss | Parameters (default) | Miles argv |
|---|---|---|---|
| `policy_loss` (default) | PPO clip (unchanged) | -- | nothing emitted (default GRPO argv and hash unchanged) |
| `cispo` | -sg(clip(rho, 1-eps_l, 1+eps_h)) * A * log pi_theta; every token keeps a gradient; token-normalized | `loss.eps_clip` (eps_l) / `loss.eps_clip_high` (eps_h), **required** (no default); `loss.aggregation="token"` **required** | `--eps-clip .. --eps-clip-high .. --calculate-per-token-loss --policy-loss-variant cispo` |
| `sapo` | -sigmoid(tau (rho-1)) * 4/tau * A; tau = tau_pos for A>0, tau_neg otherwise; no hard mask; per-sequence mean (Miles default aggregation) | `loss.sapo_tau_pos` (1.0), `loss.sapo_tau_neg` (1.05) | `--policy-loss-variant sapo --sapo-tau-pos .. --sapo-tau-neg ..` |
| `gmpo` | one-sided (arXiv:2507.20673v3 eq. 4, in log space): l_t = sign(A) min(sign(A) log rho_t, sign(A) clamp(log rho_t, -delta_l, delta_h)); sequence ratio exp(mean_t l_t) over the whole sequence; -ratio * A. For A>0 only log rho > delta_h is clipped, for A<0 only log rho < -delta_l | `loss.gmpo_log_clip_low` (0.4), `loss.gmpo_log_clip_high` (0.4) | `--policy-loss-variant gmpo --gmpo-log-clip-low .. --gmpo-log-clip-high ..` |

`loss.variant` (P0) stays `--loss-type`; the variant is its own field.
A variant's parameters enter the canonical form (and the hash) exactly when
that variant is selected, then always (even at the default value), and are
always emitted explicitly (never the fork's own default). Absorbed from extra
argv, the fork flags follow the usual rule: a value that differs from the spec
fails before launch naming both values.

Refused before any GPU process:

- a variant with `advantage.estimator="gspo"` (both define the ratio);
- a variant with `loss.eps_clip_c` (dual-clip is for the PPO clip objective);
- a variant with `loss.variant="custom_loss"`;
- CISPO without both `loss.eps_clip` and `loss.eps_clip_high` (the clip range must be in the hash);
- CISPO without `loss.aggregation="token"` (see the paper notes below);
- GMPO with `loss.aggregation="token"` (`--calculate-per-token-loss`; the fork refuses it too);
- SAPO / GMPO with `loss.eps_clip` / `loss.eps_clip_high` (no effect there);
- a parameter of another variant (e.g. `loss.sapo_tau_pos` with `gmpo`);
- tau or delta that is not a positive finite number (the error names the field);
- GMPO with context parallel size > 1 (CPU-tested only; ports currently always runs CP 1, so this is a guard);
- any variant on a Miles pin without the fork commit (launch check `[loss_variants]`).

Zero-gradient invariant: CISPO and SAPO keep the GRPO rule (a round whose
every ratio is out of range still expects a gradient). GMPO may legitimately
produce no gradient when every token is clipped in log space: the round is
relaxed only when the round's global GMPO clip fraction is 1:
sum(`gmpo_clip_num`) / sum(`gmpo_clip_den`) over the round's optimizer steps,
the two counts the fork reports under the same final loss mask (tokens with
A != 0; sequences with all-zero advantages count in neither). Only the ratio
is meaningful (Miles' micro-batch averaging scales both; under CP every rank
adds the same counts). Fork `pg_clipfrac` keeps its own per-sequence-mean
definition and is not used by the rule. The fraction travels as
`TrainStepMetrics.clip_fraction`, never as `masked_fraction` (which
corrections fill with their own mask). Missing counts or a zero denominator
keep the GRPO rule. A non-finite grad norm fails under every variant. The
Miles trainer reads the counts with the patch
`infra-drafts/patches/algo-2b-trainer-v2.patch` (INFRA-owned `trainer.py`).

Paper notes (checked 2026-09-30). CISPO (MiniMax-M1, arXiv:2506.13585 eq. 4-5)
divides by the total token count of the group, i.e. per-token loss; Miles'
default is the per-sample mean, so yeto opens CISPO only with
`loss.aggregation="token"` (Miles normalizes by the batch token count -- the
same up to how groups share a batch). The paper sets no effective lower bound
(eps_low large, only eps_high tuned): a paper-like run uses
`loss.eps_clip >= 1`. SAPO (arXiv:2511.20347v2 eq. 5-6) is a per-sequence
mean, which is Miles' default aggregation. GMPO (arXiv:2507.20673v3 eq. 4)
matches the official code (callsys/GMPO).

Outer sync: the variants only change the loss and are orthogonal to strict-avg
and decoupled; ports run serially (staleness 0, pi_old is this round's start).

```bash
python3 -m yeto.rl.engine.miles_adapter.algorithm_flags --dry-run \
  --extra "--policy-loss-variant cispo --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss"
# verdict "rejected": losses mechanism 'cispo' not supported (... expressible but not enabled)
python3 -m yeto.rl.engine.miles_adapter.algorithm_flags --dry-run \
  --extra "--policy-loss-variant sapo" --rl-allow-unverified-mechanism losses:sapo
# verdict "accepted"; miles_argv ends with
#   --policy-loss-variant sapo --sapo-tau-pos 1.0 --sapo-tau-neg 1.05
# and "launch_warnings" is empty on the 5c1b49eb pin (on a pin outside
# FORK_COMMITS it lists "[loss_variants] ... Expressible but not opened")
python3 -m yeto.rl.engine.miles_adapter.algorithm_flags --dry-run \
  --extra "--policy-loss-variant gmpo --sapo-tau-pos 1.2" \
  --rl-allow-unverified-mechanism losses:gmpo
# verdict "rejected": [loss_variant_params] ['loss.sapo_tau_pos'] only apply to ...
```

Upgrading Miles: the variant lives in one branch point of the fork's
`policy_loss_function` (`losses.py`) plus `math_utils.py` / `arguments.py`.
After every rebase of `yeto/ports`, rerun the fork's variant tests, add the
reviewed commit to `FORK_COMMITS` when Agent IMG moves `MILES_NEXT_COMMIT`
and the image digest (this makes the single-island
`--rl-allow-unverified-mechanism` smoke launchable), and declare a variant in
`entry.MILES_DECLARED` only after its GPU smoke passed.

### Port responsibilities

| port | responsibility | Miles adapter (`yeto/rl/engine/miles_adapter/`) |
| --- | --- | --- |
| `RolloutPool` | generate one rollout of complete groups from the published policy; members | `rollout.py` over `InferenceController` + `RolloutExecutor`; metadata from `rollout_meta_hook.py` inside the rollout process |
| `TrainerGroup` | one optimizer step on an opaque batch; onload/offload; grad norm for the per-round invariant | `trainer.py` over the single-cell actor group |
| `PolicyState` | export/apply the LoRA trainable state (optimizer reset or preserve, scheduler alignment) | `state.py` + `state_plugin.py`, run on every Megatron rank through `run_plugin` |
| `Publisher` | full publication of one exact policy, acknowledged by every rollout member | `publish.py` over upstream `update_weights`, stamping `yeto:<version>:<policy_tensor_hash>` as the SGLang weight version |
| `Placement` | read-only placement description; detects Miles rewriting the request | `placement.py` |

The policy token `yeto:<rollout_id>:<policy_tensor_hash>` has one definition,
`yeto.rl.core.policy_snapshot_token` (also `PolicySnapshot.token`). On `ports`
the driver hands the published token to the rollout process, where
`--buffer-filter-path` (`policy_buffer_filter`) reuses only complete groups of
exactly that policy, the ports equivalent of legacy's completed-group queue
filtering. Outer synchronization reuses the legacy `strict-avg` and `decoupled`
state machines and island checkpoint formats (`yeto/rl/engine/bridges.py`).

### The one Miles-side patch

`ports` needs a single generic entry point in Miles: `run_plugin(fn_path,
kwargs)` on the train actor and a pass-through on the trainer group
(`TrainerController`), which loads `fn_path` and calls it with the actor on
every rank. The trainable-state export/apply and grad-norm plugins themselves
live in Yeto. The patch is carried as one commit on `michaellchung/miles`
`yeto/ports` and proposed upstream. Until `MILES_NEXT_COMMIT` points at a commit
carrying it, `--rl-engine ports` refuses to start (before any Miles component
or model is created) with an error naming the pin.

### Dry-run examples

The benchmark plans either path without GPUs:

```bash
python3 scripts/benchmark_rl.py --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data openai/gsm8k --data-revision e53f048856ff4f594e959d75785d2c2d37b678ee \
  --reward-function project.rewards:score \
  --islands 2 --arms single,federated,decoupled --rl-engine legacy --dry-run

python3 scripts/benchmark_rl.py --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data openai/gsm8k --data-revision e53f048856ff4f594e959d75785d2c2d37b678ee \
  --reward-function project.rewards:score \
  --islands 2 --arms single,federated,decoupled --dry-run
```

The second command (default engine, `ports`) prints the same plan followed by
`RL_ENGINE ports`. Without `--arms`, `ports` plans `single,federated,decoupled`
and `legacy` plans all four arms including `native`; explicitly adding the
`native` arm to a ports run exits with an error naming
`--arms single,federated,decoupled`.

### Train/inference mismatch corrections

Change `rl-algo-mismatch-correction` (registration module
`yeto/rl/algos/mismatch_correction.py`). On the serial ports driver the
behavior policy (SGLang generating with `W_r`) and `pi_old` (Megatron
re-scoring with `W_r`) are the **same weights**: the per-group policy token
enforces it. The ratio `exp(train_old - rollout)` therefore measures only the
numeric difference between the two engines (kernels, bf16, the LoRA weight
publish path). These mechanisms correct that difference; they are not an
off-policy license: `execution.max_policy_staleness` stays 0 and a non-zero
value is rejected before any GPU process.

Every threshold is explicit. yeto sets no default and does not inherit Miles'
parser defaults (for example `--tis-clip-low 0`); the translated argv always
carries the values from the spec. At most one importance-weighting correction
(observe-only, TIS, IcePop or MIS) can be selected, because they share
`correction.method` and Miles has one `--custom-tis-function-path`. Selecting
two of them through extra argv is a conflict that names both flags. OPSM alone
uses `correction.method: "opsm"`. Combining OPSM with TIS, IcePop or MIS needs
the shared-interface patch `1a-shared.patch`, which is pending. Observe-only
can never be combined with another mechanism.

| mechanism | spec (`correction`) | Miles argv / attributes | masks tokens | validation |
| --- | --- | --- | --- | --- |
| `mismatch_observe` | `method: custom`, `function: yeto.rl.algos.mismatch_observe.observe_mismatch`, `mismatch_metrics: true` | `--use-tis --custom-tis-function-path ... --get-mismatch-metrics` | no | CPU (loss and gradient `torch.equal` to no correction) |
| `tis` | `method: tis`, `tis_clip`, `tis_clip_low` | `--use-tis --tis-clip H --tis-clip-low L` | no | CPU |
| `icepop` | `method: custom`, `function: miles...corrections.icepop_function`, `tis_clip_low` < `tis_clip` | `--use-tis --tis-clip H --tis-clip-low L --custom-tis-function-path ...` | yes | CPU |
| `opsm_trainer` / `opsm_rollout` | `method: opsm`, `opsm_delta`, `opsm_old_logprob_source` | `--use-opsm --opsm-delta d` (+ `--use-rollout-logprobs` for `rollout`) | yes (sequences) | CPU |
| `mis` / `mis_mask` | `method: custom`, `function: yeto.rl.algos.vendor.miles_mis.compute_mis_weights_with_cp`, `mis_level`, `mis_mode`, `mis_upper_bound` (+ `mis_lower_bound` for clip/mask), `mis_batch_normalize` | `--use-tis --custom-tis-function-path ...` plus namespace attributes `tis_level`, `tis_mode`, `tis_*_bound`, `tis_batch_normalize`, `use_rs=false` | `mis_mask` only | CPU |

Validation levels: "CPU" means numeric tests against the Miles sources at
`MILES_NEXT_COMMIT` (`tests/test_rl_mismatch_observe.py`, run in miles-next-venv)
plus spec, translation and rejection tests (`tests/test_rl_mismatch_correction.py`).
Single-GPU smoke (G1, Modal H100, 3 rounds, Qwen3-0.6B LoRA) passed for all
mechanisms except `opsm_rollout`. On the integration branch the Miles adapter declares
`none`, `tis`, `opsm`, `opsm_trainer`, `mismatch_observe`, `icepop` and
`mis_mask`. Each was verified through `yeto launch --rl-single-island-no-sync`
and, for the correcting mechanisms, by a run that made the branch fire (tasks
7.2). `opsm` is the OPSM dimension and admits no source by itself;
`opsm_rollout` and `mis` (truncate/clip) are not declared. Any other mechanism fails at startup with a
list of the supported ones. For a single-island smoke only,
`--rl-single-island-no-sync --rl-allow-unverified-mechanism corrections:<name>`
(and `features:<name>` where needed) admits them. The two-island run (G3) has not
been done.

Limits of that GPU evidence: no clipping or masking branch fired on GPU (every
ratio stayed inside the bounds, so `tis_clipfrac`, the IcePop and MIS mask
fractions were all 0). With one optimizer step per round, OPSM cannot trigger by
construction, because pi_theta = pi_old. For the same reason `ess_ratio` and `ois`
are always 1: they are pi_theta/pi_old statistics and do not reflect the
train/inference mismatch. Nothing here claims a training benefit.

Threshold meaning. The literature values below have not been verified in this
repository; they are shown only for orientation:

- TIS: the weight is `clamp(ratio, tis_clip_low, tis_clip)` and multiplies the
  per-token PPO loss. No token is masked. Miles' own example uses `C = 2`
  (unverified).
- IcePop: a token whose ratio lies in `[tis_clip_low, tis_clip]` gets the weight
  `ratio`; any other token gets weight 0. The paper's interval is `[0.5, 5]`
  (unverified).
- OPSM: a sequence is masked when its advantage is negative and the
  sequence-level `mean(pi_old - pi_theta) > opsm_delta`. With
  `opsm_old_logprob_source: "trainer"` (the default, as in Miles) `pi_old` is
  the Megatron re-score, so OPSM only masks sequences that the inner
  mini-batches moved too far; it does not cover train/inference mismatch.
  `"rollout"` requires `use_rollout_logprobs: true`
  (`--use-rollout-logprobs`), which **also replaces `pi_old` in the PPO ratio**,
  not only in OPSM. Because of that, it is rejected together with TIS. The
  DeepSeek-V3.2 report uses the inference-side logprobs (unverified).
- MIS: `mis_mode` is `truncate` (cap at the upper bound), `clip` (clamp to
  `[lower, upper]`) or `mask` (zero the tokens outside the interval), and
  `mis_level` is `token`, `sequence` or `geometric` (geometric mean). Miles'
  example suggests `[0.9999, 1.0001]` for the geometric level (unverified).
  Rejection sampling and the veto threshold of Miles' `mis.yaml` are not
  exposed. MIS is a verbatim copy of Miles
  `examples/infra_features/train_infer_mismatch_helper/mis.py` at `9e4260d`
  (Apache-2.0; header in `yeto/rl/algos/vendor/miles_mis.py`). The runtime image
  can import the original (`examples.infra_features...mis`, checked in the
  image), but plugins must live under `yeto.`/`miles.`, so yeto uses the copy. On a Miles upgrade, re-copy it and
  rerun `tests/test_rl_mismatch_observe.py`, which checks that the copy equals
  the original. Do the same with `ICEPOP_SOURCE_SHA256`.

Zero-gradient rule: `icepop`, `opsm_*` and `mis_mask` may legitimately mask a
whole round. A round with zero gradient is accepted only when the trainer
reports `masked_fraction == 1.0`. The helper
`mismatch_correction.masked_fraction_from_metrics` derives it from the Miles
metrics: `tis_clipfrac` for IcePop and `mis_tis_mask_fraction_low + _high` for
MIS. It returns None for OPSM, whose `opsm_clipfrac` is not a token fraction. An
unknown fraction keeps the strict R0 rule. A non-finite `grad_norm` always
fails. The trainer-side wiring is pending INFRA.

Metrics: `tis`, `tis_abs`, `ois`, `train_rollout_kl`,
`train_rollout_logprob_abs_diff` and `ess_ratio`, plus
`mismatch_outside_0p5_5` for observe-only (the fraction of tokens outside the
IcePop reference interval `[0.5, 5]`). They appear in Miles' `train/*` log. The
ports event stream does not carry them yet (pending INFRA). The serial
colocated mode publishes LoRA weights over CUDA IPC. The partitioned mode of
`rl-infra-spec` must use NCCL broadcast, so a mismatch measured in one mode
does not carry over to the other.

Examples. Each block is checked by `tests/test_rl_mismatch_correction.py`
through the P0 dry run (`python3 -m yeto.rl.engine.miles_adapter.algorithm_flags
--dry-run --rl-algorithm-spec FILE [--rl-allow-unverified-mechanism DIMENSION:NAME ...]`):
it is rejected as undeclared without the allowances and accepted with them.

<!-- mismatch-example allow=corrections:custom,corrections:mismatch_observe,features:mismatch_metrics -->
```json
{"schema": "yeto-rl-algorithm-spec-v2",
 "correction": {"method": "custom", "mismatch_metrics": true,
   "function": {"path": "yeto.rl.algos.mismatch_observe.observe_mismatch",
                "sha256": "9d5209db978e940d9b246d6e08dcb56c23e114594da08bb8ac1c88c79b8d6255"}}}
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
                "sha256": "f75f86c302edb7563ae8026b3bf4dda992217d9eed23b5c7ad0ae93936fd9096"}}}
```

## Benchmark

[`scripts/benchmark_rl.py`](../scripts/benchmark_rl.py) runs up to four local,
equal-hardware real-Miles arms. The `native` arm is stock Miles' own loop and
needs `--rl-engine legacy`; with the default `ports` engine the default arm set
is `single,federated,decoupled`:

| arm | purpose |
| --- | --- |
| `native-miles-mM` | native Miles with no Yeto synchronization |
| `yeto-single-mM` | one strict Yeto island using all `M*G` GPUs |
| `yeto-federated-mM` | strict full-roster FedAvg across `M` islands |
| `yeto-decoupled-mM` | decoupled fragment synchronization across `M` islands |

All arms match model/revision, LoRA, prompt stream, reward, seed, total GPUs,
optimizer steps, groups, trajectories, and action-token limits. Decoupled
learners freeze after the same local step budget `R`. The syncer writes an
unmarked cutoff checkpoint, restarts with pipeline 1, performs exactly one
ordinary full-fragment consolidation sweep from the frozen policies, and only
then marks and exports the final artifact. This avoids comparing a
network-dependent amount of local work.

The report includes held-out reward and pass@k, KL/ESS/clip fraction, rollout,
train, hook and finalization time, artifact-ready time, trajectories and action
tokens per second, GPU-hours, time-weighted GPU activity/utilization, realized
`H`, PULL-to-PUSH latency, BCAST queue time, fragment payload traffic,
responder count, and deltas versus native, single-island, and strict federation
controls.

The harness uses real SGLang generation, the selected real reward callable,
and real GRPO training. It does not accept injected rollouts, synthetic
optimizer steps, or fake rewards as benchmark evidence.

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
layout fingerprint. No dashboard is enabled by this feature.

Payload traffic counts PUSH, ordinary BCAST, and ordinary final-cut fragment
tensors. It intentionally excludes message headers, framing, chunks, and
control messages.

## Validation

Automated coverage exercises deterministic binpacking, policy snapshot
identity, mixed-token rejection, BCAST-before-PULL ordering, horizon gating,
staged BCAST commit, duplicate and invalid protocol messages, multi-fragment
deltas, optimizer preservation, scheduler and exact-snapshot group recovery,
unequal fragment cuts, terminal replacement, budget consolidation, f32
two-island Nesterov oracle behavior, terminal export, standard PEFT reload,
fresh-phase adapter validation and initialization, benchmark fairness, and
Miles stop-after-publication ordering.

Existing strict-avg evidence includes real dense and MoE LoRA GRPO runs,
multi-island f32 parity, process and retained-disk syncer recovery, session/tool
rollouts, standard PEFT load/generation, and the Qwen3.6-27B equal-hardware H200
benchmark summarized in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

The decoupled preset must not be described as release-usable until its pinned
Miles commit and required real causal-LM GPU matrix, including a cross-machine
run and failure run, have completed. Diffusion RL is outside the contract and
is not part of that matrix.

## Development Guide

The implementation is intentionally confined to the RL boundary:

| area | responsibility |
| --- | --- |
| `yeto/rl/core.py` | canonical LoRA, deterministic RL fragments, policy snapshots |
| `yeto/rl/decoupled.py` | raw anchors, BCAST/PULL/PUSH state, final cut and benchmark consolidation |
| `yeto/rl/miles.py` | rollout-token validation, safe-boundary hook, island checkpoint |
| `yeto/rl/learner.py` | Miles argument mapping and runtime configuration |
| `yeto/rl/export.py` | authoritative checkpoint to standard PEFT |
| `yeto/rl/initial_adapter.py` | validated Decoupled PEFT policy warm start |
| `scripts/benchmark_rl.py` | equal-work native/strict/decoupled comparison |
| `agentenv/miles:train.py` | optional run-until-stop and stop-after-publication ordering |
| `syncer/src/**` | unchanged general fragment scheduler and outer optimizer |

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

(cd ../miles && python -m pytest -q \
  --confcutdir=tests/fast tests/fast/test_external_policy_sync.py)
```
