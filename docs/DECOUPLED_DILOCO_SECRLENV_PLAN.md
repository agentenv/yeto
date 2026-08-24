# GRPO, DiLoCo, and PULSE plan for SecRLEnv

Status: architecture proposal; no launch approval

Date: 2026-08-24

## Decision

Build one actor-only RL system in four gated stages:

1. **SecRLEnv GRPO with dense DiLoCo.** Establish the learning and distributed
   update reference using full FP32 pseudo-gradient exchange and complete BF16
   inference checkpoints.
2. **Replace dense trainer exchange with PULSELoCo.** Keep the rollout, reward,
   GRPO, optimizer, and inference-publication paths unchanged. Add
   compute-visible sparse pseudo-gradients with FP32 error feedback.
3. **Add PULSESync for inference refresh.** Keep PULSELoCo between trainers and
   replace complete inference-checkpoint transfers with bit-exact BF16-visible
   patches.
4. **Add Decoupled DiLoCo features incrementally.** Increase the local horizon,
   stream fragments, tolerate stragglers through quorum and grace, weight by
   useful work, and prove learner failure/rejoin recovery.

Dense DiLoCo and complete inference publication remain supported as validation
modes. They are not separate final products. They provide an oracle against
which the PULSE paths can be tested without changing the RL algorithm.

The final operating loop is:
```
SecRLEnv tasks
    -> inference workers generate rollouts
    -> SecRLEnv returns verifiable rewards and cleanup evidence
    -> each learner island performs local GRPO steps
    -> PULSELoCo reconciles trainer pseudo-gradients
    -> the synchronizer publishes a complete global policy version
    -> PULSESync updates inference workers to that version
    -> next SecRLEnv rollout cycle
```

GRPO derives group-relative advantages and therefore needs no learned critic in
this design.

## Existing implementation baseline

This plan extends working components; it does not reimplement GRPO or
Decoupled DiLoCo from scratch.

- The preserved Qwen3.8 path already runs direct, full-parameter Miles/Megatron
  GRPO with `--advantage-estimator grpo`, three samples per prompt, SecRLEnv
  verifiable rewards, and trainer-to-SGLang weight publication.
- That path is one learner island. Its systems smoke reached rollout generation
  and an optimizer step, but it did not demonstrate repeated nonzero updates or
  held-out learning improvement.
- The Rust syncer already implements base-relative fragment updates, frozen
  learner generations, quorum, adaptive grace, AVG/RDA merging, FP32 Nesterov
  outer state, checkpoint recovery, and an event ledger.
- The current Yeto RL bridge exposes `CanonicalLoraState`; its working
  Decoupled DiLoCo route is therefore LoRA-oriented. The direct full-parameter
  Qwen GRPO path bypasses that route.

The missing foundation is the integration boundary: export and apply bounded
full-parameter fragments from the existing GRPO trainer, preserve its training
state correctly, and drive those fragments through the existing syncer.
PULSELoCo and PULSESync are then added behind that same boundary.

## Network and hardware boundary

The final system doesn't require InfiniBand between learner nodes.

- Each learner island is self-contained and holds a full trainable policy plus
  its local optimizer state.
- TP, PP, sequence-parallel, and DDP collectives stay within one node or another
  explicitly fast-fabric island. No NCCL/model-parallel process group spans the
  ordinary inter-node network.
- Local H200s use their fastest available peer fabric for tightly coupled
  training. The preserved 6-training/2-inference run exercised local NCCL
  through TP2/PP3 and TP2 groups, but did not preserve an explicit topology
  attestation proving NVLink/NVSwitch specifically. Future plans must record
  `nvidia-smi topo -m` before selecting a topology.
- Commodity Ethernet carries trajectories, version metadata, PULSELoCo sparse
  trainer updates, PULSESync inference patches, and occasional repair
  checkpoints.
- A CPU/large-memory synchronizer owns the authoritative global outer state and
  does not execute model forward passes.

This makes InfiniBand an optional throughput improvement rather than a
correctness requirement. The claim remains gated on measured payload sizes,
queueing, synchronization time, and policy staleness under the real Ethernet
network.

Decoupling removes the need for a fast fabric between islands; it does not
remove the need for fast collectives within a multi-GPU learner.

Milestones 0-3 and most of Milestone 4 can be developed on one 8-H200 node with
a smaller model. Partition that node into isolated learner and inference GPU
sets. Learners must have separate processes, state, and checkpoints, and must
never join one cross-island NCCL/DDP group. Their only communication uses the
real DiLoCo/PULSE protocol over loopback or the host network.

A representative layout is three 2-GPU learners, one 2-GPU inference worker,
and a CPU synchronizer. A smaller model may instead use one GPU per learner.
Milestone 4 is not complete until the same frozen build passes on at least two
physical nodes; three are preferred to test `K=M-1`, real node loss,
independent storage, and actual Ethernet latency and throughput.

For Qwen3.8-27B scale-up, start with one 8-H200 node per learner island. Select
the within-node trainer/inference split only after a measured memory ledger and
small-model proof. The earlier 6-training/2-inference result proves that one
node can execute the model; it does not validate multi-island DiLoCo.

## Why this order

The stages change one synchronization surface at a time:

| Stage | Local RL | Trainer exchange | Inference refresh | Question answered |
| --- | --- | --- | --- | --- |
| Dense reference | GRPO | Full FP32 pseudo-gradients | Full BF16 checkpoint | Is distributed GRPO correct? |
| PULSELoCo | Same GRPO | Sparse update + FP32 error feedback | Full BF16 checkpoint | Does compression preserve trainer behavior? |
| PULSESync | Same GRPO | PULSELoCo | Sparse BF16-visible patch | Do rollout workers reconstruct the exact policy? |
| Decoupled | Same GRPO | PULSELoCo, asynchronous/fragmented | PULSESync | Does it remain correct under lag, quorum, and failure? |

The code may be structured for all four modes from the beginning, but the
release gates stay ordered. Enabling PULSELoCo and PULSESync simultaneously
before a dense reference would make a learning regression ambiguous among:

- SecRLEnv rollout/reward attribution;
- GRPO advantage or loss construction;
- local optimizer behavior;
- pseudo-gradient construction or outer aggregation;
- PULSELoCo thresholding/error feedback;
- PULSESync reconstruction; or
- stale-policy rollout handling.

The dense and full-transfer modes are therefore test instruments. Production
eventually uses PULSELoCo + PULSESync.

## Literature boundary

- [GRPO](https://cameronrwolfe.substack.com/p/grpo) supplies the local actor-only
  RL objective: sample a group for a prompt, calculate verifiable rewards, and
  normalize advantages within the group instead of fitting a critic.
- [PULSE/PULSELoCo](https://arxiv.org/abs/2602.03839) separates two communication
  channels. PULSELoCo sparsifies trainer-to-trainer DiLoCo pseudo-gradients with
  error feedback; PULSESync sends only BF16-compute-visible new values to
  inference replicas. Its published GRPO evidence is a starting point, not
  proof for Qwen3.8-27B or multi-turn SecRLEnv.
- [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428) supplies independent
  learners, local inner steps, asynchronous fragment exchange, frozen
  generations, quorum, adaptive grace, useful-work weighting, outer momentum,
  catch-up, and an event ledger.

SAO is not included. If asynchronous single-rollout actor-critic work is
reconsidered later, it gets a separate proposal and does not alter the GRPO
correctness contract described here.

## Target architecture

```text
                 authoritative CPU/large-memory syncer
          global policy + FP32 outer state + error feedback
              version ledger / quorum / recovery state
                    ^                       |
       PULSELoCo sparse pseudo-gradients    | global policy publication
                    |                       v
       +------------------------+    +----------------------+
       | independent learners   |    | inference workers    |
       | local full policy      |    | complete BF16 policy |
       | local Adam state       |    | SGLang generation    |
       | local GRPO steps       |    +----------+-----------+
       +------------------------+               |
                    ^                           | rollouts
                    | trajectories/rewards      v
                    +------------------- SecRLEnv executors

                 PULSESync applies each complete published
                 policy version to the inference workers.
```

### Shared interfaces

All stages use the same interfaces so a reference backend can be switched for
the optimized backend without changing learning semantics.

1. **Trajectory envelope**
   - task/environment identity;
   - prompt-group identity and sample index;
   - behavior-policy version and complete snapshot hash;
   - generated tokens and token-level behavior log-probabilities where needed;
   - verifiable reward and cleanup evidence;
   - globally idempotent trajectory ID.
2. **Local-step receipt**
   - learner/generation identity;
   - global anchor version;
   - deterministic input-batch identity;
   - accepted trajectories and trained-token count;
   - local-step count and optimizer-step result;
   - resulting parameter-layout identity.
3. **Trainer update**
   - base and target global versions;
   - tensor/fragment identity;
   - dense pseudo-gradient or PULSELoCo sparse payload;
   - PULSE threshold/error-feedback version;
   - payload hash, byte count, and completeness proof.
4. **Inference publication**
   - base and target global versions;
   - complete target manifest and target model hash;
   - full BF16 snapshot or PULSESync patch;
   - reconstruction receipt from each inference worker.

No rollout may be trained twice, no group may mix behavior-policy versions, and
no inference worker may serve a partially reconstructed policy.

### Learner island

- Take inner optimizer steps without contacting another island.
- Preserve local Adam state locally. Trainer exchange carries parameter-space
  pseudo-gradients, not local optimizer state.
- Construct GRPO groups from a single published behavior-policy version and a
  single immutable reward contract.
- Apply a newly merged global state only at a safe boundary after an optimizer
  step and before the next forward pass.
- Atomically checkpoint local weights, optimizer state, data cursor, current
  anchor, PULSE error-feedback state, and last applied global version.
- Rejoin with a new learner generation and perform a complete state catch-up;
  never silently reuse stale counters or error-feedback buffers.

### Synchronizer

Reuse the existing Rust syncer's protocol strengths while generalizing the RL
state from LoRA-only values to a bounded, chunked full-parameter policy layout.

Dense DiLoCo is a restricted reference profile of this syncer, not a second
implementation. Begin with FP32 wire payloads, `H=1`, fixed membership, full
quorum, one fragment round in flight, no post-quorum omission, and complete
BF16 policy publication. Existing fragmentation, RDA, grace, pipeline, and
recovery code remains present but is enabled and validated incrementally in
Milestone 4.

- Maintain FP32 global policy/outer-optimizer state and a durable version
  ledger.
- Accept either dense or PULSELoCo updates behind the same validated update
  interface.
- Keep PULSELoCo error-feedback state versioned and checkpointed. A dropped or
  duplicated sparse update must not lose or double-apply residual mass.
- Publish an atomic policy manifest only after a complete merge. Inference may
  never consume a mixture of fragment versions.
- Support complete catch-up snapshots even when the steady-state path is
  sparse.
- Record responders, omitted learners, trained-token weights, local horizons,
  staleness, threshold statistics, residual norms, payload bytes, queue/grace
  time, and global-delta norms.

### Inference and SecRLEnv pool

- Inference workers serve only a complete immutable global-policy version.
- PULSESync applies new BF16 values relative to an exact base version. It must
  reject a missing/wrong base and request a complete repair checkpoint.
- Reconstruction must be bit-identical to the publisher's BF16 target before
  the worker becomes ready.
- Rollout admission freezes a policy version for the entire prompt group.
- SecRLEnv provisioning, flagless DinD debugging, reward verification, and
  cleanup remain outside the optimizer and fail closed.
- A durable trajectory queue uses at-least-once delivery plus idempotent IDs;
  training credit remains exactly once.
- Policy delay is measured and bounded before a trajectory is accepted for
  training.

## Implementation plan

### Milestone 0 — freeze contracts and reference fixtures

- Preserve the direct full-parameter Qwen/SecRLEnv GRPO wrapper, arguments,
  run artifacts, and failure evidence as the existing one-island semantic and
  systems reference—not as a distributed-RL topology template.
- Define the four shared envelopes above and one canonical full-parameter
  tensor/fragment layout.
- Capture small deterministic SecRLEnv trajectory fixtures including rewards,
  cleanup receipts, behavior versions, and GRPO grouping.
- Implement a replay mode that performs no generation and feeds identical
  accepted trajectories to every synchronization backend.
- Add feature switches for dense/PULSELoCo trainer exchange and full/PULSESync
  inference publication. A switch must not change the GRPO batch or optimizer.

### Milestone 1 — GRPO with dense DiLoCo

- Reuse the existing direct Miles full-parameter GRPO path on a small model as
  the one-island semantic reference; do not implement another GRPO algorithm.
- Generalize `CanonicalLoraState` and the existing Miles/DiLoCo bridge to
  full-parameter, role-qualified, bounded fragments.
- Prove export/apply at safe training boundaries without changing the GRPO
  batches, loss, model outputs, checkpoint meaning, or retained training state.
- Run at least two independent learner islands with local optimizers.
- Run the existing syncer in its restricted dense reference profile: `H=1`,
  fixed membership, full quorum, synchronous one-fragment-at-a-time merging,
  FP32 pseudo-gradients, and complete BF16 inference checkpoints.
- Replay identical frozen trajectories through centralized GRPO and dense
  DiLoCo. At `H=1`, verify the expected update relationship within a declared
  numerical tolerance rather than inferring correctness from reward curves.
- Then run a small online SecRLEnv experiment and compare held-out evaluation,
  KL, update norms, and accepted-token accounting across multiple seeds.

Exit gate: distributed GRPO completes repeated update/publication cycles; all
model versions, groups, rewards, and tokens reconcile; restart is idempotent;
and small-model learning is not materially worse than the same-compute
centralized reference.

### Milestone 2 — substitute PULSELoCo

- Keep GRPO, trajectories, optimizer settings, local horizon, membership,
  quorum, and full inference-checkpoint publication unchanged.
- Add compute-visibility thresholding to dense FP32 pseudo-gradients.
- Accumulate every unsent value in an FP32 error-feedback buffer and bind that
  buffer to learner generation, global base version, tensor layout, and
  threshold contract.
- Prove duplicate, omitted, reordered, interrupted, and recovered sparse
  submissions are fail-closed or exactly idempotent.
- In deterministic replay, compare dense and PULSELoCo accumulated global
  updates, residual conservation, BF16-visible model values, and total bytes.
- In online RL, compare held-out quality, KL, global/local drift, residual norm,
  convergence, and goodput under matched accepted-token and optimizer budgets.

Exit gate: PULSELoCo reaches the declared quality/update tolerance to dense
DiLoCo, conserves residuals across checkpoint/restart, and materially reduces
trainer-exchange bytes on the real Ethernet path.

### Milestone 3 — add PULSESync

- Retain PULSELoCo between trainers.
- Keep complete policy snapshots as an available repair and oracle path.
- For every publication, calculate changed BF16 indices/new values relative to
  an exact base version.
- Apply a patch only to a worker holding that exact base. Otherwise require a
  complete checkpoint.
- Reconstruct the target offline and on every inference worker; require exact
  BF16 tensor equality and complete target-manifest equality before serving.
- Inject dropped, duplicated, reordered, corrupted, and stale-base patches.
- Confirm rollout content and behavior-policy attribution are unchanged between
  full-checkpoint and PULSESync publication modes for deterministic prompts.

Exit gate: PULSESync reconstruction is bit-exact, recovery from any interrupted
patch converges through a full checkpoint, no partial version serves traffic,
and inference-publication bytes fall materially without reducing goodput or
held-out quality.

### Milestone 4 — add Decoupled DiLoCo features one at a time

Keep PULSELoCo + PULSESync enabled, then introduce one semantic change per
experiment in this order:

1. Increase local horizon from `H=1` to `H=2`, then `H=4`, then at most `H=8`
   if policy lag, KL, and held-out quality remain within gates.
2. Stream balanced tensor fragments while learners continue local work.
3. Add atomic complete-policy publication across asynchronously merged
   fragments.
4. Move from fixed full quorum to `K=M-1` after learner-kill tests pass.
5. Add adaptive grace for stragglers.
6. Add trained-token/useful-work weighting and the selected outer
   RDA/Nesterov configuration.
7. Add learner failure, restart, full catch-up, and generation-safe rejoin.
8. Test heterogeneous learner speed and controlled network degradation.

Each step is compared against the immediately preceding configuration. If it
changes the accepted trajectories, policy-delay distribution, or effective
optimization budget, report that difference rather than calling the runs
identical.

Exit gate: one learner can fail or lag without corrupting global state or
stopping healthy learners; recovery is exact; policy staleness remains bounded;
and held-out learning remains within the declared reference interval. Same-node
fault injection is necessary but insufficient: the final gate requires the
same frozen build on at least two physical nodes, with three preferred for
quorum and node-loss testing.

### Milestone 5 — scale model, context, and workload

- Perform Milestones 1–4 on a 1.5B–7B model before Qwen3.8-27B.
- Scale Qwen3.8 first with short observed contexts and a complete per-role
  memory ledger.
- Increase context progressively, for example 16k, 32k, 64k, 128k, then a
  262k serving maximum. Do not allocate every training sample at 262k merely
  because the endpoint accepts it.
- Increase the number of learner islands only after update correctness,
  Ethernet payload, synchronization latency, policy lag, and quality gates
  pass.
- More nodes become more independent learner islands or inference capacity;
  they do not become one Ethernet-spanning Megatron group.

## Validation strategy

### Deterministic update tests

Freeze trajectories, rewards, behavior log-probabilities, group membership,
advantages, initial weights, and optimizer state. Run the same inputs through:

1. centralized GRPO;
2. dense DiLoCo;
3. PULSELoCo; and
4. PULSELoCo plus PULSESync reconstruction.

These tests isolate update and communication semantics. They are not learning
claims. Require:

- exact input/token/group accounting;
- declared dense numerical tolerance;
- error-feedback conservation over multiple sparse rounds;
- bit-exact PULSESync BF16 reconstruction;
- identical behavior under checkpoint/restart; and
- fail-closed base-version, layout, identity, and completeness checks.

### Online learning tests

Online arms will not consume identical trajectories after their policies
diverge. Match instead:

- immutable initial checkpoint and tokenizer;
- train/held-out task split;
- prompt ordering and seed schedule;
- GRPO group size and reward code;
- accepted trajectory/token and optimizer-step budgets;
- local horizon where applicable;
- GPU allocation and evaluation cadence; and
- at least three seeds for final quality claims.

Report reward, held-out pass@1/pass@k, KL, current-versus-behavior policy lag,
update/residual norms, useful trajectories per hour, trainer and rollout
utilization, bytes by channel, synchronization time, and wall-clock goodput.

### Failure and recovery tests

Test these independently from the quality comparison:

- learner death before/during/after update submission;
- synchronizer restart around merge/checkpoint/publication;
- duplicate or reordered trainer fragments;
- inference patch interruption and stale-base rejection;
- inference worker restart and full-checkpoint repair;
- learner full catch-up and new-generation rejoin;
- rollout queue redelivery with exactly-once training credit; and
- SecRLEnv task failure and cleanup without checkpoint advancement.

## Required invariants

- No cross-node model-parallel or DDP collective.
- No learned critic or neural reward model in the GRPO/RLVR path.
- Every trainable group binds one complete behavior-policy version.
- Every learner update binds one global anchor and one parameter layout.
- Every published policy is complete, immutable, and content-addressed.
- PULSELoCo residuals survive restart and cannot cross learner generations.
- PULSESync patches apply only to their exact base and reconstruct the exact
  target before traffic is admitted.
- A learner or inference worker can always recover via a complete checkpoint.
- SecRLEnv reward and cleanup evidence remain mandatory and exactly credited.
- Process launch, rollout completion, or one optimizer step is not evidence of
  held-out RL improvement.

## Repository work

Reusable:

- Rust fragment protocol, outer checkpointing, version ledger, and recovery;
- quorum, adaptive grace, useful-work weighting, outer momentum, and event tape;
- SecRLEnv task packs, isolation, verifiable rewards, and cleanup;
- Qwen3.8 model/TITO integration; and
- the existing full-parameter systems-smoke diagnostics.

Must change or be added:

- extend the LoRA-only RL state boundary to support bounded full-parameter
  state from the existing direct Miles GRPO trainer;
- expose dense and PULSELoCo implementations behind one trainer-update API;
- expose full-checkpoint and PULSESync implementations behind one inference
  publication API;
- add global policy manifests independent of learner-local checkpoints;
- persist/version PULSE error-feedback state;
- bind rollout artifacts to behavior-policy versions and exact-once IDs;
- prohibit cross-node Megatron/DDP topology in the plan validator; and
- add deterministic replay, bit-exact reconstruction, failure injection, and
  held-out learning evaluation.

## Immediate execution sequence

1. Freeze shared envelopes and deterministic trajectory/update fixtures.
2. Connect the existing small-model-capable full-parameter GRPO path to the
   existing syncer in dense `H=1` reference mode.
3. Prove centralized/dense update semantics and online held-out learning.
4. Substitute PULSELoCo and prove residual conservation plus quality parity.
5. Add PULSESync and prove bit-exact inference reconstruction.
6. Add Decoupled DiLoCo features one at a time with failure tests.
7. Run the complete small-model Ethernet architecture.
8. Scale to Qwen3.8-27B only after all preceding gates pass.

Do not launch another 27B full run as the next architecture experiment. The
next meaningful artifact is a small-model GRPO + dense DiLoCo reference using
the final shared interfaces. That gives PULSELoCo and PULSESync a trustworthy
oracle while avoiding a later architectural rewrite.
