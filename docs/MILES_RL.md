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
- a new dashboard, controller, storage system, or generic recovery framework.

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

`--rl-offload-train` rebuilds the colocated training/rollout process groups
before publishing updated weights; this is the safer setting for a large model
when the first post-training `update_weights` call has timed out. The timeout
is a fail-fast bound for the distributed barrier, not a quality setting.

`--trust-remote-code` is required by the pinned Miles model-loading path. Keep
model and dataset revisions immutable and enable it only for trusted sources.
The reward callable uses `package.module:function`; Yeto hashes its source
before provisioning and the learner verifies that digest before import.

Prompt rows provide `messages`, or a string `prompt`/`input` that Yeto converts
to a user message. `label`, `metadata`, and `tools` remain available to Miles
and the reward callable. Existing custom generation, session-server, and TITO
arguments are forwarded unchanged.

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

## Benchmark

[`scripts/benchmark_rl.py`](../scripts/benchmark_rl.py) runs four local,
equal-hardware real-Miles arms:

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
- hook duration and whether the hook performed finalization.

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
