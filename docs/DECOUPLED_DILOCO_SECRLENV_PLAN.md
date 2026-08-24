# Decoupled DiLoCo plan for SecRLEnv agentic RL

Status: architecture proposal; no launch approval

Date: 2026-08-24

## Decision

Do not scale the existing `6 training + 2 inference` single-node job into a
cross-node Megatron job. It is a useful full-parameter systems smoke, but it is
not a DiLoCo RL architecture: it has one tightly coupled trainer, one rollout
engine, no outer learner aggregation, and no critic.

The replacement architecture uses independent learner islands. Tensor,
pipeline, sequence, and data-parallel collectives are confined to the fast
fabric inside an island. Nodes never participate in one cross-node Megatron or
DDP process group. Commodity Ethernet carries only trajectories, versioned
model fragments or sparse patches, and control metadata.

Implement this in two deliberately separated stages:

1. **GRPO + Decoupled DiLoCo correctness track.** Every island contains a full
   policy trainer. GRPO has no learned critic, so there is no omitted critic in
   this track. This is the shortest path to a literature-backed multi-trainer
   RL baseline and matches SecRLEnv's verifiable rewards.
2. **SAO + Decoupled DiLoCo agentic track.** Replace grouped synchronous
   rollouts with single-rollout asynchronous optimization. Every learner owns
   both its policy trainer and its value trainer; both trainable states are
   reconciled through DiLoCo. This is the target for long, highly variable
   SecRLEnv trajectories, but it is a research integration and must not be
   treated as already validated by either source paper.

Do not add PULSE sparsification until dense fragment exchange is correct and
measured. PULSESync and PULSELoCo are subsequent bandwidth optimizations, not
substitutes for a correct actor/critic and staleness design.

## Current implementation audit

The repository already has useful Decoupled DiLoCo infrastructure, but the
current full-parameter run does not exercise it end to end.

- `syncer/src` and `docs/PROTOCOL.md` implement fragment-wise outer updates,
  frozen learner generations, quorum, adaptive grace, token-weighted merging,
  FP32 Nesterov state, recovery, and an event ledger.
- `yeto/rl/decoupled.py` pipelines RL fragment exchange, but its public state is
  `CanonicalLoraState` and its identity contract contains a LoRA configuration
  hash.
- `yeto/rl/core.py` explicitly defines the canonical boundary as PEFT LoRA
  values.
- `yeto/rl/miles.py` exports and applies `actor_model` trainable state. It has no
  value-model/critic role in the DiLoCo protocol.
- The v125 handoff records one n7 trainer/rollout container and no Yeto outer
  syncer or peer learner. It therefore demonstrates local full-parameter
  Megatron + SGLang + SecRLEnv integration, not distributed DiLoCo RL.

## Why the plan changed

- [GRPO](https://cameronrwolfe.substack.com/p/grpo) removes the learned critic
  by deriving advantages within a group. The current `3 samples/prompt` shape
  is therefore an actor-only algorithm, not an actor-critic algorithm with a
  missing service.
- [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428) consists of independent
  learners doing local inner optimization and a central synchronizer doing
  asynchronous, fragment-wise outer optimization with quorum, grace, and
  token-weighted merging. A pool of rollout workers plus one monolithic trainer
  is not this architecture.
- [PULSE/PULSELoCo](https://arxiv.org/abs/2602.03839) is the closest published
  evidence for DiLoCo-style LLM RL. It separates trainer-to-inference weight
  refresh from trainer-to-trainer pseudo-gradient exchange. Its GRPO results
  used four trainers, shared global rollout checkpoints, and modest local
  horizons (`H=8` for tested Qwen models) because larger horizons increase
  policy staleness. It was evaluated only up to 7B and on MATH, not a 27B
  multi-turn agent environment.
- [SAO](https://arxiv.org/abs/2607.07508) argues that group barriers are a poor
  fit for asynchronous agentic RL. It trains from each completed rollout,
  stores rollout-policy token log-probabilities, applies double-sided
  token-level off-policy masking, uses skip-observation GAE, and updates the
  critic twice per policy update. SAO itself is not a DiLoCo paper; combining
  the two is our proposed design and requires new validation.

## Target architecture

```text
                        CPU / large-memory syncer
                  actor outer state + outer optimizer
                 critic outer state + outer optimizer*
                    quorum / grace / version ledger
                         /         |         \
             fragments /          |          \ fragments
                       /           |           \
              learner island A  learner B  learner C ...
              local fast fabric local fabric local fabric
              actor trainer     actor trainer actor trainer
              critic trainer*   critic*       critic*
              local Adam state  local Adam    local Adam
                       \           |           /
                        \ trajectory stream  /
                         rollout / environment pool
                  SGLang replicas + SecRLEnv executors
                  global actor snapshots only; no trainer
                  collectives across the commodity network

* absent in the GRPO correctness track; mandatory and DiLoCo-managed in SAO
```

### Learner island

An island is the failure and synchronization boundary. It must be able to take
an inner optimizer step without contacting another island.

- Run all Megatron collectives within the island's local NVLink/NVSwitch
  domain. Never form TP, PP, EP, FSDP, or DDP groups across ordinary Ethernet.
- Preserve each island's inner Adam state locally. DiLoCo exchanges
  base-relative parameter deltas, not inner optimizer state.
- Apply incoming global fragments only at an explicit safe boundary after
  backward/optimizer completion and before the next forward pass.
- Track, for every actor and critic fragment, the raw anchor version, local
  steps, trained tokens, and latest applied global version.
- Checkpoint local actor/critic weights, inner optimizer states, data cursor,
  and fragment anchors atomically. A recovering island rejoins with a new
  generation; it never silently reuses stale counters.

For Qwen3.8-27B full-parameter GRPO, begin with one 8-H200 node per learner.
Choose the within-node TP/PP layout only after an isolated memory and throughput
proof. The old TP2/PP3 six-GPU result is evidence that the model can train, not
the new topology contract.

For SAO, do not assume that the 27B value model fits beside the policy. First
measure a full memory ledger for policy weights, gradients, optimizer state,
activations, value weights, value optimizer state, and checkpoint buffers. If
one node cannot hold both, define a logical learner as an actor-training node
plus a critic-training node. They exchange trajectories and scalar/value
targets over Ethernet but perform no cross-node model-parallel collective.

### Synchronizer

Reuse the Rust syncer's proven protocol machinery, then generalize its RL state
model.

- Keep fragmented pulls/pushes, frozen generation membership, minimum quorum,
  adaptive grace, token-weighted merging, FP32 outer state, Nesterov outer
  optimization, durable event tape, and full-fragment catch-up.
- Replace the current LoRA-only identity with a role-qualified full-parameter
  layout: `actor/<tensor>` and, for SAO, `critic/<tensor>`.
- Give actor and critic independent fragment versions, local horizons, outer
  optimizers, and checkpoints. A critic merge may never be interpreted as an
  actor publication.
- Publish an atomic actor snapshot manifest only after a complete fragment
  sweep. Rollout workers consume only such manifests, never a mixture of
  fragment versions.
- Make quorum an explicit experiment parameter. Start at all learners for
  parity tests, then test `K=M-1`. Do not start production with `K=1`.
- Record per-merge responders, omitted learners, staleness, token weights,
  payload bytes, queue time, grace time, and global-delta norm.

The current syncer stores full FP32 global state and momentum. A 27B actor is
roughly 108 GB for each FP32 array before scratch/checkpoint overhead; adding a
full critic roughly doubles the persistent model-role state. The syncer host
therefore needs a measured RAM/NVMe budget before 27B SAO.

### Rollout and environment pool

Rollout generation is decoupled from learner execution but is not allowed to
be anonymous or arbitrarily stale.

- Serve only complete global actor snapshots identified by manifest hash and
  monotonically increasing policy version.
- Attach behavior-policy version, snapshot hash, token log-probabilities,
  task identity, environment identity, reward evidence, and action/observation
  masks to every trajectory.
- Keep SecRLEnv provisioning, flagless DinD debug access, cleanup, and
  verifiable reward outside the optimizer. No neural reward model is required
  for the present binary/verifiable reward contract.
- Route trajectories to learners through a durable queue with at-least-once
  delivery plus idempotent trajectory IDs. Credit a trajectory exactly once.
- Enforce a maximum policy-delay gate. Measure first; do not inherit a large
  pre-training DiLoCo horizon. The initial GRPO gate is `H=1`, then `H=2/4`,
  and only then `H=8` if KL, clipping, and held-out evaluation remain stable.
- Initially refresh inference with complete BF16 snapshots. After correctness,
  implement PULSESync-style compute-visible BF16 patches with bit-identical
  reconstruction and periodic full-checkpoint repair.

### GRPO correctness track

This track keeps the current verifiable reward and group size but moves the
trainer into multiple DiLoCo islands.

- Each prompt group is generated entirely from one published actor snapshot.
- A group is trainable only when all retained samples have the same policy
  version and complete reward/cleanup evidence.
- Each island takes local GRPO steps on a disjoint, deterministic prompt shard.
- After `H` local actor steps, it submits actor deltas to the outer syncer.
- There is no critic to synchronize. Adding a nominal critic to GRPO would add
  cost without matching the algorithm.

This is the first end-to-end target because the PULSELoCo paper supplies direct
evidence for GRPO + DiLoCo. It is still only a systems/algorithm bridge at 27B
and SecRLEnv scale, not a guaranteed quality result.

### SAO agentic track

Only begin after the GRPO DiLoCo baseline is stable.

- Consume a rollout as soon as it completes; do not wait for sibling samples.
- Compute the behavior ratio from rollout-time token log-probabilities and the
  current local policy. Mask tokens outside the configured double-sided trust
  interval rather than silently clipping them into the loss.
- Train a value model and update it twice per actor update initially (`K=2`).
- Compute GAE across action-to-action transitions while excluding observation
  tokens generated by SecRLEnv rather than the policy.
- Place the actor trainer and critic trainer inside every logical learner.
  Synchronize both roles through separate DiLoCo outer states.
- Do not copy SAO's frozen-attention critic rule blindly to dense Qwen3.8. The
  paper's reported stability result is architecture-specific. Compare full,
  shared-backbone/value-head, and frozen-submodule critic designs in a smaller
  model before selecting the 27B design.
- Gate training by policy delay, masked-token fraction, importance-ratio
  quantiles, critic explained variance, critic gradient norm, actor/critic
  version skew, and held-out task performance.

## Implementation work

### Milestone 0 — freeze the old result

- Mark v124/v125 as a single-island full-parameter diagnostic, not the template
  for a distributed RL run.
- Preserve its subnet fix, task pack, rollout evidence, and optimizer failure
  diagnostics. Do not reuse its run ID or checkpoints in the new experiment.

### Milestone 1 — full-parameter actor DiLoCo

- Generalize `CanonicalLoraState` and the Miles trainable-state bridge to a
  bounded, chunked, role-qualified full-parameter state.
- Add full-parameter export/apply methods that preserve local Adam state across
  fragment applies and prove safe-boundary application.
- Add actor snapshot manifests and rollout publication independent from local
  learner checkpoints.
- Keep dense FP32 pseudo-gradients initially. Measure real bytes and overlap.

### Milestone 2 — GRPO multi-island proof

- Reproduce a central baseline and two-island DiLoCo on a 1.5B–7B model first.
- Run two Qwen3.8 learner islands plus a separate rollout pool only after the
  small-model parity and failure tests pass.
- Start with `H=1`, full quorum, one outer fragment pipeline, and short context.
  Increase one dimension at a time.

### Milestone 3 — bandwidth work

- Implement bit-identical PULSESync-compatible actor publication.
- Independently implement PULSELoCo-style compute-visible sparse outer deltas
  with FP32 error feedback.
- Compare dense DiLoCo and sparse DiLoCo at identical data, seeds, local
  horizon, reward, and evaluation cadence before enabling sparsity by default.

### Milestone 4 — SAO actor/critic learner

- Add the critic role, optimizer, value pretraining artifact, skip-observation
  masks/GAE, rollout log-probability contract, and token-level trust mask.
- Add separate actor and critic outer synchronization namespaces and recovery.
- Prove one learner against a non-DiLoCo SAO reference, then two learners with
  `H=1`, before testing asynchronous quorum or longer horizons.

### Milestone 5 — 27B/long-context scale-up

- Scale context progressively (for example 16k, 32k, 64k, 128k, then a 262k
  maximum) using observed token lengths and memory ledgers. Do not allocate or
  train every sample at 262k merely because the serving cap supports it.
- Increase learner count only after quality parity, policy-lag, and bandwidth
  gates pass. More GPUs are useful as more independent islands, not as a single
  Ethernet-spanning Megatron group.

## Required tests and gates

### Correctness

- Full-parameter export/apply round-trip is tensor-identical and optimizer
  state is unchanged by a fragment apply.
- A published actor snapshot is complete and bit-identical on every rollout
  worker.
- Every trajectory binds one behavior policy and exact token log-probabilities.
- GRPO groups never mix policy versions. SAO masks or rejects stale tokens under
  the signed trust policy.
- Actor and critic checkpoints cannot cross-load or mix fragment generations.
- Crash/restart at every pull, merge, checkpoint, apply, rollout, reward, and
  cleanup boundary is idempotent.

### Systems

- Killing one learner does not stop other learners; the syncer advances only
  with the configured quorum and reintegrates the learner by full catch-up.
- No cross-node NCCL/model-parallel traffic appears in the launch graph.
- Report trainer utilization, rollout utilization, useful trajectories/hour,
  inner-step time, outer-sync time, bytes by channel, and total goodput.
- Establish measured memory headroom for actor, critic, optimizer, activation,
  checkpoint, and syncer state before every scale increase.

### Learning

- Compare against a same-compute centralized baseline, not only against the
  untrained base model.
- Use a frozen held-out SecRLEnv evaluation split; training reward alone is not
  evidence of improvement.
- Require confidence intervals across at least three seeds for final claims.
- Track pass@1/pass@k, reward, KL/current-vs-rollout drift, importance-ratio
  quantiles, masked-token fraction, gradient norms, critic explained variance
  when applicable, and regression on non-training tasks.

### Initial go/no-go sequence

1. CPU/small-GPU deterministic protocol tests with two actor learners.
2. Small-model centralized GRPO versus two-island dense DiLoCo, `H=1`.
3. Learner-kill and straggler test with full quorum, then `K=M-1`.
4. Small-model `H=2/4/8` staleness sweep with held-out quality.
5. Qwen3.8 one-island parity smoke.
6. Qwen3.8 two-island dense DiLoCo smoke plus separate rollout pool.
7. PULSESync/PULSELoCo A/B only after dense correctness.
8. SAO one-island actor/critic parity, then two-island DiLoCo.
9. Full SecRLEnv run only after every preceding gate is green.

Any failure in identity, snapshot completeness, optimizer preservation,
trajectory attribution, cleanup, staleness bounds, or held-out quality stops the
scale-up. A successful process launch or optimizer step is not an RL-quality
gate.

## What can be reused and what must change

Reusable:

- Rust fragment protocol and syncer checkpointing
- quorum, adaptive grace, token weighting, Nesterov outer state, and event tape
- learner generation/recovery machinery
- SecRLEnv task packs, environment isolation, verifiable reward, and cleanup
- Qwen3.8 model/TITO integration and the static-subnet concurrency fix

Must change:

- the single-node `6T/2I` topology as the distributed-run template
- LoRA-only RL state typing and policy-only synchronization
- any full-parameter run path that bypasses the DiLoCo bridge
- rollout artifacts that omit behavior log-probabilities or exact snapshot IDs
- synchronous grouped execution as the long-term agentic RL architecture
- evaluation that infers learning from smoke completion rather than held-out
  improvement

## Immediate recommendation

Do not launch another 27B full run from v125. First implement Milestone 1 and
run the small-model two-island GRPO parity experiment. That is the minimum test
that answers whether Yeto is actually doing RL through DiLoCo rather than merely
running rollout inference beside a conventional trainer.
