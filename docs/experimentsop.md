# Optimizer Experiment Standard Operating Procedure

Status: active

Applies to: Yeto optimizer discovery, engineering canaries, distributed
screens, same-state CRN evaluation, and final confirmation

Default production control: memoryless outer SGD with exact learning rate
`0.28` and exact positive-zero momentum

Durable campaign log: `docs/optimizerhunt.md`

## 1. Purpose

This SOP defines how an optimizer idea becomes an admissible performance
claim. It exists to prevent four recurring errors:

1. spending GPUs on a candidate that is inactive, unidentified, or already
   contradicted by retained evidence;
2. calling an online trajectory difference a same-state causal effect;
3. changing a mechanism, threshold, seed, workload, or exclusion after seeing
   its outcome; and
4. leaving billable or ambiguously owned cloud resources behind after a run.

The process is deliberately asymmetric: a candidate can be killed cheaply,
but a claim that it beats SGD-0.28 requires progressively stronger evidence.
Formal proofs, direction scores, engineering canaries, online loss differences,
and same-state finite-loss replays are different evidence classes and must
never be described as interchangeable.

## 2. Non-negotiable rules

### 2.1 Use a fresh matched control

Every candidate loss claim requires a fresh live SGD-0.28 control with the
same:

- source commit and runtime image;
- initial model and adapter state;
- model/data/evaluation hashes;
- row order, RNG seeds, and future-group materialization;
- LoRA shape and parameter layout;
- inner optimizer, scheduler, clipping, and local-step schedule;
- learner count, fragment count, quorum, merge rule, and wire dtype;
- H, token budget, evaluation rows, and stopping rule; and
- system instrumentation required by both arms.

Historical losses with a similar label are not controls. In particular,
"SGD-0.28" means the exact outer kernel and exact f32 learning-rate identity;
it does not permit a historical arm with another LR, inner optimizer, H,
dataset order, or runtime.

### 2.2 Freeze before outcomes

Before a development or confirmation outcome is opened, freeze:

- the candidate formula and sign convention;
- all state variables and their causal update timing;
- exact numerical semantics and constants;
- fallback and state-clearing behavior;
- workloads, seeds, sample allocation, and exclusions;
- primary and secondary metrics;
- go/kill/safety thresholds;
- bootstrap or multiplicity procedure;
- required artifacts and integrity checks; and
- the next stage authorized by each possible verdict.

Changing any of these after viewing an outcome creates a new candidate. It
requires a new run ID, specification, artifact prefix, and preregistration.

### 2.3 Fail closed

Missing, stale, ambiguous, malformed, nonfinite, noncanonical, or
checksum-mismatched evidence must produce an explicit failure state. Never
replace missing vectors with norms, infer optimizer state from endpoints,
repair a broken causal sequence from later rows, or silently shrink a gate's
denominator.

### 2.4 Preserve claim boundaries

- A Lean theorem is an algebraic or geometric result under its hypotheses.
  It is not a language-model loss theorem unless the full training process is
  actually modeled.
- Retained-tape replay is valid only for quantities identified by that tape.
- A direction/cosine screen is not finite-loss evidence.
- A one-A100 M=1 canary is not a distributed optimizer comparison.
- An online paired run is not same-state CRN evidence once an arm takes a
  different action.
- A single workload or seed is not a generalizable win.

### 2.5 Own resources by exact identity

Cloud resource names and labels help discovery but do not authorize mutation.
Every owned VM, disk, image, and canary must be recorded by exact provider ID.
Deletion must reauthenticate that identity immediately before mutation.

Never terminate by prefix, tag, machine type, broad query, or "cleanup all"
logic. Inventory resources that predate the run as protected and out of scope
unless the user explicitly brings an exact resource into scope.

### 2.6 Finish with evidence and teardown

Every launched run ends in exactly one of these lifecycle outcomes:

- a completed, checksummed run followed by exact-ID `delete`;
- an incomplete or failed run with preserved evidence followed by exact-ID
  `abandon`; or
- a still-running owned resource with an explicitly documented operator and
  next action.

"The process exited" is not teardown. Provider lookups for both the recorded
VM and its recorded auto-delete disk must return explicit not-found responses.

## 3. Evidence and claim ladder

| Stage | Typical compute | Required result | Maximum admissible claim |
| --- | ---: | --- | --- |
| 0. Candidate dossier | CPU | Frozen mechanism, causal state, failure mode, and gates | Testable hypothesis |
| 1. Offline identification | CPU | Exact retained-evidence audit and deterministic replay where identifiable | Historical mechanism screen |
| 2. Implementation audit | CPU | Independent reference, production implementation, exact fixtures, state and checkpoint tests, formal audit | Implementation/mechanism correctness |
| 3. E1 engineering canary | 1 A100 | Candidate activates, artifacts validate, fallbacks are exact, overhead is acceptable | Engineering readiness |
| 4. E2 distributed screen | 4 A100 | Four-learner matched schedule, nontrivial actions, integrity and exploratory gates pass | Distributed exploratory evidence |
| 5. Same-state CRN replay | 1--4 A100 | Exact restore, identical future groups, A/B and B/A agreement | Causal finite-loss evidence on captured boundaries |
| 6. Core workload gate | 4 A100 per job | Frozen candidate passes H16/H64/H256 product gate | Multi-workload development evidence |
| 7. Final confirmation | Up to 8 A100 concurrently | One frozen candidate, five fresh paired seeds | Confirmation-grade claim |

No stage may be skipped merely because quota or hardware is available. A
permission ceiling such as eight A100s is not a target allocation.

## 4. Verdict vocabulary

Every stage must end with one primary verdict from this closed vocabulary.

### `PASS`

All integrity, activity, safety, and stage-specific statistical gates passed.
Only the next preregistered stage is authorized.

### `FAIL`

The run was scientifically interpretable and one or more frozen gates failed.
Examples include zero non-stock actions, insufficient gain, an unsafe
regression, or excessive overhead. The candidate does not advance.

### `INCONCLUSIVE`

Integrity passed, but a preregistered information or precision condition was
not met. This does not count as a pass or a failure of the underlying
mechanism. A rerun is allowed only when the preregistration already specified
the inconclusive branch or a new protocol is frozen without using hidden
outcomes.

### `UNIDENTIFIABLE`

The available evidence cannot determine the proposed action or metric. No
score, zero, fallback rate, or pseudo-result may be fabricated. New capture is
required.

### `INFRA_FAILURE`

The scientific stage did not start or did not reach its outcome gate because
of provisioning, transport, process, storage, preemption, or controller
failure. Preserve evidence, abandon the exact resources, and use a fresh run
identity for a retry. An infrastructure failure is not optimizer evidence.

## 5. Stage 0: candidate dossier

Create a candidate dossier before reading new development outcomes. It may be
a dedicated preregistration file under `experiments/optimizer/` or a frozen
section of a campaign plan. It must contain the following.

### 5.1 Identity

- candidate name and version;
- source commit and implementation paths;
- stock control identity;
- development versus confirmation status;
- relationship to previously screened candidates; and
- a statement of what changing a coefficient, sign, delay, interlock, or
  proposal source would make a new candidate.

### 5.2 Mechanism

- exact action formula;
- tensor layout and accumulation order;
- state variables per learner/fragment/tensor;
- when each observation becomes causally available;
- preview versus commit behavior;
- action norm, angle, and clipping bounds;
- nonfinite, degenerate, discontinuity, and hash-failure behavior; and
- exact stock fallback requirements.

### 5.3 Falsification

- the specific prior failure the candidate addresses;
- at least one counterexample or regime where it should abstain or lose;
- minimum nontrivial action rate;
- maximum allowed regression and norm/angle change;
- kill conditions; and
- quantities that the available evidence cannot identify.

### 5.4 Experimental design

- stage being requested;
- model, dataset, LoRA, H, token budget, learner/fragment design, and seeds;
- evaluation object and rows;
- primary effect definition;
- uncertainty and multiplicity calculation;
- exact advancement gate;
- required outputs and checksums; and
- maximum cloud allocation and wall-time envelope.

## 6. Stage 1: offline identification and replay

Offline work is the default first experiment.

1. Inventory the retained evidence without opening hidden confirmation
   outcomes.
2. Write the exact fields needed to reconstruct the candidate.
3. Classify each field as directly stored, derivable with a proved identity,
   ambiguous, or absent.
4. If any action-defining quantity is ambiguous or absent, return
   `UNIDENTIFIABLE`.
5. If identifiable, reconstruct the production stock path bit-exactly before
   scoring a candidate.
6. Freeze candidate actions before reading their later outcomes.
7. Preserve event order, fragment identity, causal delay, and discontinuity
   handling.
8. Report action size, action rate, fragment coverage, direction gain, and
   every predefined safety diagnostic, including unfavorable ones.

Retained scalar norms and cosines do not identify full-vector residual banks,
parallel transport, cross terms, Adam state, or later projections. Hashes prove
identity but do not provide the hashed vector.

### Stage 1 advancement gate

Advance only if:

- production reconstruction is exact;
- the candidate is nontrivial under the frozen denominator;
- the directional/mechanism gate passes;
- no safety gate fails;
- all required fragments/workloads are represented; and
- the evidence is rich enough to implement exact production semantics.

Otherwise kill or acquire the specifically missing evidence.

## 7. Stage 2: implementation and formal audit

### 7.1 Independent numerical reference

Implement a small reference that is independent of the production state
machine. Freeze:

- dtype and byte order;
- product, sum, square-root, division, trig, and write order;
- whether FMA is allowed;
- constants by exact bits where relevant;
- normalization and reprojection order;
- signed-zero behavior;
- overshoot and degeneracy thresholds; and
- malformed-input behavior.

The reference must return the original stock byte object on any defined exact
fallback path. Errors and fallbacks must be distinct.

### 7.2 Golden fixtures

Fixtures must cover, as applicable:

- first-step and warm-up behavior;
- active nontrivial action;
- sign convention;
- cap boundary;
- stationary direction;
- reversal or anti-alignment;
- nonplanar geometry;
- nonfinite and degenerate inputs;
- f32-inert candidate;
- exact stock fallback bytes;
- repeated preview purity;
- commit advancement exactly once; and
- checkpoint/resume equivalence.

Cross-runtime fixtures must use identical raw f32 input bits. Reconstructing
inputs independently from angles is not a parity test. One-ULP mismatches must
be frozen and treated as a portability blocker unless one runtime is declared
the sole authoritative evaluator before outcomes.

### 7.3 Production state machine

The production implementation must:

- use a closed optimizer name and configuration schema;
- validate the exact control/treatment LR and momentum identity;
- isolate candidate state from stock optimizer state;
- compute previews without mutation;
- install state only on a successful commit;
- clear invalid causal state rather than invent continuity;
- persist every action-relevant state field in the checkpoint;
- validate checkpoint invariants on load;
- add candidate evidence conditionally without changing legacy tape/action
  seals when inactive; and
- emit hashes for stock, candidate, selected action, and relevant history.

### 7.4 Formal work

Lean is used for exact statements small enough to model honestly, such as:

- norm and angle bounds;
- causal/nonanticipating state transitions;
- accounting identities;
- exact degeneracy to stock;
- constant-motion recovery; and
- constructive counterexamples to unconditional dominance.

Every formal report must list the assumptions absent from the live system.
Never state that a local quadratic, sphere-geometry, or accounting theorem
proves language-model loss improvement.

### Stage 2 advancement gate

Required before a GPU canary:

- reference and production fixtures agree under one authoritative numerical
  contract;
- focused and adjacent regression tests pass;
- preview, commit, tape, checkpoint, and resume tests pass;
- format/lint/diff checks pass;
- formal target passes with no `sorry`, `admit`, or new project axiom;
- a failure/counterexample test exists; and
- the experiment specification pins the implementation commit.

## 8. Preregistration and immutable run specification

Every cloud attempt gets a fresh run ID. Never edit a failed attempt in place
or reuse its artifact prefix.

The JSON specification must pin:

- full 40-character repository commit;
- unique run, instance, checkout, run-directory, and GCS identities;
- cloud project, zone, machine type, accelerator count, Spot policy, provider
  time limit, disk size/type, and campaign accelerator ceiling;
- exact numeric source-image ID;
- required executables and offline assets;
- main command as a JSON argv array;
- model/data/runtime provenance and checksum manifests;
- completion paths and result checksums;
- expected arms and every critical flag;
- strict-quorum step-budget headroom; and
- stage-specific validation command and output.

The source tree must be clean, pushed, and reproducible from the pinned commit.
The experiment entry point must verify that it imported the harness from its
sibling checkout, not an editable installation elsewhere.

## 9. Cloud preflight

Preflight is read-only until the immutable specification passes.

### 9.1 Authenticate safely

- Use an isolated cloud CLI configuration for experiment credentials.
- Never print, copy into logs, or commit service-account keys, access keys,
  tokens, SSH private keys, or credential files.
- Verify the active project/account and principal without displaying secret
  material.

### 9.2 Inventory all relevant compute

List current instances before launch. Classify each as:

- owned by this exact run;
- protected/out of scope;
- unrelated but counted against quota; or
- ambiguous, which blocks mutation until resolved.

The live inventory, not a stale handoff, is authoritative. Instance IDs can
change when Spot workloads are replaced.

### 9.3 Check capacity and quota

Check all resource dimensions required by the machine type. For GCP A2 this
includes both A2 CPU quota and the applicable A100 quota. Accelerator quota
alone does not prove that an `a2-highgpu-*` instance can be created.

The harness must count every live accelerator VM in the project against the
campaign ceiling, including unrelated instances. Available quota does not
authorize skipping an earlier scientific gate.

### 9.4 Validate the exact spec

Use the sibling-checkout CLI from the repository root:

```bash
PYTHON=.venv/bin/python
SPEC=experiments/optimizer/example.json

$PYTHON scripts/optimizer_experiment.py validate "$SPEC"
$PYTHON scripts/optimizer_experiment.py doctor "$SPEC"
$PYTHON scripts/optimizer_experiment.py render "$SPEC" > /tmp/rendered-run.txt
```

Review the rendered launch and remote command. Confirm:

- the run-specific prefix is empty;
- no prior local lifecycle state uses the run ID;
- the source image resolves to the expected numeric ID;
- the command contains the intended arms and exact gates;
- no nested quoting or shell interpolation changes the argv;
- the checkout path, `PYTHONPATH`, interpreter, Rust toolchain, model, and data
  paths are exact; and
- the maximum runtime leaves time for validation, final sync, and teardown.

## 10. Launch and start

### 10.1 Launch

```bash
$PYTHON scripts/optimizer_experiment.py launch "$SPEC" --yes
```

Immediately record from the returned state:

- numeric VM ID;
- numeric boot-disk ID and exact self-link;
- ownership nonce and management labels;
- exact source-image ID/self-link;
- provisioning model and provider termination action; and
- creation time and maximum run duration.

If post-create provenance verification fails, retain the quarantined exact
identity and use `abandon`; do not create a second untracked VM.

### 10.2 Start

```bash
$PYTHON scripts/optimizer_experiment.py start "$SPEC"
```

Start is allowed only after the remote checkout, clean commit, executables,
model/data checksums, runtime metadata, detached environment, and output paths
verify. Record runner and backup PIDs only when their command lines are bound
to the exact run directory.

### 10.3 Monitor

```bash
$PYTHON scripts/optimizer_experiment.py status "$SPEC"
$PYTHON scripts/optimizer_experiment.py logs "$SPEC" --lines 200
$PYTHON scripts/optimizer_experiment.py sync "$SPEC"
```

During a live run, monitor:

- provider state and Spot/preemption events;
- runner and backup process identity;
- input-integrity completion;
- commit count and balanced fragment coverage;
- learner liveness and reconnect count;
- nonfinite values, fallback reasons, and action rate;
- tape/checkpoint growth;
- writer queue, drops, abandoned bytes, and overhead;
- GCS backup freshness; and
- remaining provider runtime.

Do not interpret a partial loss, action rate, or fragment result while the
frozen stage is still collecting.

## 11. Stage 3: one-A100 engineering canary

The standard optimizer E1 profile is:

- one full-model learner on one A100;
- one CPU syncer over localhost TCP;
- four logical fragments, quorum one;
- stock and candidate arms run sequentially;
- identical base model, seed, data order, budget, and evaluation rows;
- f32 wire, overwrite broadcast, deterministic commit order, and no
  reconnects;
- fixed H4;
- 32 syncer commits, eight per fragment;
- 32,768 training tokens; and
- eight evaluation rows.

Candidate-specific preregistration may change these values, but the M=1 stage
remains an engineering canary. It does not measure multi-worker diversity or
quorum-four aggregation.

### E1 required gates

- both arms complete the exact matched schedule;
- every required tape, checkpoint, result, and provenance artifact verifies;
- every fallback action is bit-identical to stock;
- the candidate produces at least the frozen minimum number of non-stock
  actions, normally at least one;
- action state advances only on commit;
- loss is finite and no preregistered catastrophic regression occurs;
- action/ledger overhead is within the frozen bound, normally 2%; and
- the validator publishes an atomic checksummed verdict.

Zero actions is a scientific `FAIL`, not a tie with SGD. It blocks the
four-A100 stage.

## 12. Stage 4: four-A100 distributed screen

The standard E2 profile is conditional on a reviewed E1 `PASS`:

- four learners on four A100s;
- four fragments, strict quorum four;
- fixed H16;
- weighted production RDA, f32 wire, overwrite broadcasts;
- deterministic commit order and no reconnects;
- 700,000 total training tokens;
- 64 evaluation rows; and
- at least 32 balanced commits, eight per fragment.

The candidate and stock arms must differ only in the frozen outer action
selector. Report all learner and fragment effects even when unfavorable.

### E2 claim limitation

The arms are paired by seed and design, but after the candidate's first
non-stock action their model, optimizer, and future pseudo-gradient states
diverge. Terminal loss is therefore an exploratory online trajectory result,
not a same-state causal effect. E2 may authorize CRN capture; it cannot by
itself establish that the candidate action improves the same boundary.

## 13. Stage 5: same-state CRN evaluation

A causal finite-loss comparison requires a pre-outcome authority that binds
one exact plan per captured boundary.

### 13.1 Required captured state

- all trainable fragments and model buffers;
- complete named optimizer, scheduler, and scaler state;
- exact optimizer clocks and parameter-group topology;
- Python, NumPy, Torch CPU, and indexed CUDA RNG states;
- data iterator position and exact future groups 0 through 7;
- syncer pre/post boundary state and fragment versions;
- ordered responders, weights, payload hashes, and merge configuration;
- exact stock and sealed candidate actions;
- fixed evaluation object; and
- source, image, model, data, config, and evaluator provenance.

Opaque hashes or local learner endpoints without restore authority are not
enough.

### 13.2 Execution

For each authorized boundary:

1. restore a fresh stock branch;
2. apply the sealed stock action;
3. evaluate at k=0;
4. consume the exact future groups 0--7;
5. evaluate at k=8;
6. repeat from a fresh restore for the candidate action;
7. execute both A/B and B/A ordering; and
8. require exact restore/application/group-state hashes and order-invariant
   losses before publishing the paired outcome.

No action may read its own evaluation outcome. The action object and outcome
object are separate immutable artifacts.

### 13.3 Advancement

Use only the frozen finite-loss, action-rate, fragment, bootstrap,
multiplicity, and safety gates. Directional cosine may be reported as mechanism
evidence but cannot substitute for k0/k8 loss.

## 14. Stage 6: core workload product gate

For every workload/seed pair define positive-is-better effect:

\[
D_{s,w}=L_{\mathrm{SGD\mbox{-}0.28},s,w}-L_{C,s,w}.
\]

The current optimizer-hunt core workloads are standard H16, H64, and H256.
Stress workloads are secondary and cannot be chosen post hoc as a core win.

Unless a new campaign freezes another gate before development, advancement
requires:

- at least one core gain strictly greater than `0.018`;
- no core regression worse than `0.009`; and
- a plausible second core workload win.

The `0.009` value is a practical decision margin, not an estimated training
seed standard deviation.

## 15. Stage 7: final confirmation

Final confirmation uses one frozen candidate and configuration, not another
optimizer search.

- Use five fresh paired seeds.
- Keep model, data, evaluation, system, and control semantics frozen.
- Do not reopen development seeds or tune on confirmation outcomes.
- Apply the preregistered multiplicity and product gates exactly.
- Publish every seed/workload pair and failure, not only the mean.
- Separate infrastructure exclusions from scientific exclusions using the
  frozen rules.

Only a passing confirmation may support language such as "beats SGD-0.28".
Generalizable means the frozen core workload/seed gate passed; it does not mean
universal dominance over all objectives or training regimes.

## 16. Artifact and validation requirements

Before a run can be called complete, preserve and checksum:

- immutable JSON specification and its SHA-256;
- exact command argv and rendered remote program;
- repository commit, status, and source diff/manifest;
- cloud VM/disk/image numeric identities;
- model, data, runtime, and image provenance manifests;
- stock and candidate tapes;
- final checkpoints and exported adapters;
- result rows and evaluation configuration;
- stage-specific validation report;
- action/ledger/checkpoint hashes;
- writer accounting and timing evidence;
- runner logs and exit status; and
- a canonical completion manifest.

The stage validator must atomically publish both the verdict and a checksum
marker. A half-written report or report without its checksum is incomplete.

Run the analysis only through the exact declared paths:

```bash
$PYTHON scripts/optimizer_experiment.py sync "$SPEC"
$PYTHON scripts/optimizer_experiment.py analyze "$SPEC"
```

The decision record must state which gate determined the verdict and list all
other failed gates. An equality caused by zero candidate actions must be
described as stock equivalence, not an optimizer tie.

## 17. Teardown

### 17.1 Completed run

Use `delete` only after the harness validates every completion path and
checksum:

```bash
$PYTHON scripts/optimizer_experiment.py delete "$SPEC" \
  --instance-id "$EXACT_INSTANCE_ID" \
  --yes
```

### 17.2 Failed or incomplete run

Use `abandon` with a specific scientific or infrastructure reason:

```bash
$PYTHON scripts/optimizer_experiment.py abandon "$SPEC" \
  --instance-id "$EXACT_INSTANCE_ID" \
  --reason "E1 failed: zero non-stock actions; validation and checksums synced" \
  --yes
```

`abandon` must:

1. authenticate the recorded VM, labels, nonce, disk self-link, and numeric
   IDs;
2. stop only PIDs bound to the exact run directory;
3. perform a final artifact sync;
4. write and locally hash `abandonment.json`;
5. upload and round-trip that exact artifact through the run prefix;
6. reauthenticate the VM and disk;
7. delete only those exact resources; and
8. accept only explicit provider not-found results as proof of teardown.

Retain local lifecycle state as `ABANDONED`. Never erase the failed identity or
reuse its GCS prefix.

### 17.3 Final inventory

After teardown, rerun the project/account inventory and quota checks. Record:

- owned resources absent;
- exact disk absent;
- accelerator and machine-family quota usage;
- protected unrelated resources still present; and
- no new unowned resources.

## 18. Reusable image procedure

Imaging is optional and occurs only after a completed or intentionally prepared
source instance is stopped and scrubbed.

1. Final-sync all experiment evidence.
2. Stop the exact runner and uploader.
3. Remove run directories and declared credential/cache/history locations.
4. Scan user/root homes for credential-like files and fail closed on a hit.
5. Write full model/data SHA-256 manifests and runtime metadata.
6. Bind image metadata to the exact source commit, run, instance ID, and disk
   ID.
7. Clear machine identity, sync filesystems, and stop the VM.
8. Reauthenticate the stopped source VM/disk.
9. Create an unpromoted image without assigning a family.
10. Verify its numeric ID, source identities, nonce, labels, and `READY` state.
11. Launch a one-A100 canary from the exact image ID.
12. Recheck credential absence, machine identity regeneration, all model/data
    hashes, CUDA/model forward, focused tests, Rust tests, and GCS access.
13. Promote the exact image to a family only after the stored canary passes.
14. Delete the exact canary VM/disk and verify both absent.

Use `create-image`, `create-canary`, `test-canary`, `promote-image`, and
`delete-canary` through the harness. A moving image family is never experiment
provenance; specifications pin the numeric image ID.

## 19. Incident handling

### 19.1 Spot preemption

- Stop interpreting partial metrics.
- Sync whatever immutable evidence is available.
- If completion/atomic-resume requirements are not satisfied, classify the
  attempt `INFRA_FAILURE`.
- Abandon any remaining exact resources.
- Retry only with a fresh run ID and artifact prefix.

### 19.2 Controller/import drift

If the CLI imports another checkout or editable installation, stop before
launch/start. The controller must fail closed unless the harness module path is
the expected sibling path. Never compensate by manually copying a rendered
command from another checkout.

### 19.3 Remote quoting or detached-environment failure

Treat a command that failed before the scientific runner as
`INFRA_FAILURE`. Preserve the exact rendered command and stderr. Fix the
renderer or declared environment with an executable regression, commit it,
and create a fresh spec.

### 19.4 Checksum or storage delay

Distinguish:

- network ingress;
- image provisioning/restore;
- local sequential integrity reads;
- model loader reads;
- serialization/hashing;
- publication/fsync; and
- GPU compute.

Do not weaken integrity to improve startup time. An optimization such as
`fs-verity` requires a signed, exact-image-bound attestation; mtime/inode or an
unsigned "verified" marker is not sufficient.

### 19.5 Validator failure after training

The validator verdict is authoritative. Preserve the results, tape,
checkpoint, validation report, and checksum. Do not rerun with relaxed gates
under the same candidate identity.

### 19.6 Artifact corruption or missing evidence

Return `INCONCLUSIVE`, `UNIDENTIFIABLE`, or `INFRA_FAILURE` according to the
frozen protocol. Do not produce a partial `PASS`. Preserve the corrupt object
identity and failure details before teardown.

## 20. Security and protected-resource rules

- Never expose credentials in commands, logs, reports, images, or Git.
- Use least-privilege principals and isolated CLI configurations.
- Treat every pre-existing VM as protected until exact ownership is proved.
- Do not SSH into a protected workload merely to determine whether it can be
  reused.
- Do not alter security groups, firewall rules, disks, Spot requests, or
  instance state for unrelated workloads.
- Do not use a protected training server as an optimizer worker without
  explicit authorization for that exact instance.
- Never use destructive bulk cleanup commands.
- Keep experiment GCS prefixes immutable and run-specific.
- Scrub credentials before imaging and verify absence from the image canary.

## 21. Reporting language

### Allowed examples

- "The E1 engineering canary failed because the candidate selected zero
  non-stock actions."
- "The online four-learner screen improved terminal loss, but it is
  exploratory because the trajectories diverged."
- "The same-state k8 CRN effect was positive on the captured boundaries."
- "Lean proves the candidate's angular cap under orthonormal tangent
  assumptions and also proves a reversal counterexample."
- "The retained tape is unidentifiable for this full-vector proposal."

### Prohibited examples

- "The candidate tied SGD" when it never acted.
- "The one-A100 run proves distributed performance."
- "Positive cosine proves lower training loss."
- "Lean proves the optimizer beats SGD" when it proves only local geometry.
- "The historical control is close enough."
- "The run passed except for the validator."
- "Eight GPUs were authorized, so we launched the final screen."

## 22. Operator checklists

### 22.1 Before opening outcomes

- [ ] Candidate identity and formula frozen.
- [ ] Stock control identity frozen.
- [ ] Evidence sufficiency/identifiability audited.
- [ ] Workloads, seeds, exclusions, and gates frozen.
- [ ] Failure and counterexample behavior documented.
- [ ] Python/reference fixtures pass.
- [ ] Production Rust tests pass.
- [ ] Cross-runtime numerical contract passes or has one declared authority.
- [ ] Checkpoint/resume and preview/commit tests pass.
- [ ] Formal target passes and its limits are written.
- [ ] Full Git commit pushed and worktree clean.

### 22.2 Before launch

- [ ] Fresh run ID, VM name, checkout path, run directory, and GCS prefix.
- [ ] No local state or remote objects already use the identity.
- [ ] Active cloud principal/project/account verified.
- [ ] All running instances inventoried and protected resources listed.
- [ ] Accelerator, CPU-family, disk, IP, and regional quotas checked.
- [ ] Exact image numeric ID and `READY` status checked.
- [ ] `validate`, `doctor`, and human-reviewed `render` pass.
- [ ] Required executables and offline assets declared.
- [ ] Runtime envelope includes final validation/sync/teardown time.
- [ ] Current stage is authorized by the previous checksummed verdict.

### 22.3 During run

- [ ] Exact VM/disk/nonce/source-image identities recorded.
- [ ] Input manifests passed before scientific execution.
- [ ] Runner and backup PIDs match the exact run directory.
- [ ] Commit/fragment schedule progresses as frozen.
- [ ] No reconnect, drop, or nonfinite event violates the protocol.
- [ ] Backup prefix is fresh.
- [ ] Provider runtime and Spot state monitored.
- [ ] No partial outcome used to change the run.

### 22.4 Before claiming completion

- [ ] Both stock and candidate arms reached the exact schedule.
- [ ] Final tapes/checkpoints/exports/results exist.
- [ ] Provenance and checksum manifests verify.
- [ ] Stage validator atomically published verdict plus checksum.
- [ ] Verdict uses the closed vocabulary.
- [ ] Claim is no stronger than the evidence stage permits.
- [ ] All negative fragment/workload/seed results are reported.
- [ ] Final sync completed.

### 22.5 Before ending operator session

- [ ] Completed run deleted or failed run abandoned by exact instance ID.
- [ ] Exact VM lookup returns not found.
- [ ] Exact disk lookup returns not found.
- [ ] Post-run inventory shows no new orphan.
- [ ] Protected unrelated resources remain untouched.
- [ ] Run decision and artifact hashes documented.
- [ ] Documentation commit pushed.

## 23. Experiment record template

Copy this section into the candidate preregistration or run report.

```markdown
# <candidate> <stage> <run-id>

## Frozen identity
- Candidate/version:
- Stage:
- Source commit:
- Spec SHA-256:
- Image numeric ID:
- Model/data/eval hashes:
- Stock identity:
- Candidate identity:

## Design
- Model / LoRA:
- Inner optimizer:
- Outer optimizer/LR/momentum:
- Learners / fragments / quorum:
- H / token budget / eval rows:
- Seeds and row order:
- Primary effect:
- Activity gate:
- Safety gates:
- Statistical gate:
- Overhead gate:
- Allowed next stage:

## Cloud identity
- Provider/project/account:
- Region/zone:
- VM name / numeric ID:
- Disk name / numeric ID:
- Source image name / numeric ID:
- Ownership nonce:
- Provisioning / max duration:
- Artifact prefix:

## Outcome
- Lifecycle verdict: PASS | FAIL | INCONCLUSIVE | UNIDENTIFIABLE | INFRA_FAILURE
- Claim scope:
- Commits and fragment coverage:
- Candidate actions / denominator:
- Primary effect and interval:
- Safety results:
- Overhead:
- Failed gates:
- Validation artifact SHA-256:
- Completion/abandonment SHA-256:

## Teardown
- Final sync verified:
- VM not-found verified:
- Disk not-found verified:
- Post-run accelerator usage:
- Protected resources unchanged:

## Decision
- Advance / kill / acquire missing evidence / infrastructure retry:
- Exact scientific reason:
- Changes that would create a new candidate:
```

## 24. Read-only inventory examples

These commands inspect state; they do not authorize mutation.

### GCP Compute Engine

```bash
gcloud compute instances list \
  --project="$GCP_PROJECT" \
  --format='table(name,id,zone.basename(),status,machineType.basename(),scheduling.provisioningModel)'

gcloud compute regions describe "$GCP_REGION" \
  --project="$GCP_PROJECT" \
  --format=json
```

### AWS EC2 across enabled regions

```bash
for region in $(aws ec2 describe-regions --all-regions \
  --query 'Regions[?OptInStatus!=`not-opted-in`].RegionName' \
  --output text); do
  aws ec2 describe-instances \
    --region "$region" \
    --filters Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,Tags[?Key==`Name`]|[0].Value]' \
    --output text
done
```

If a live ID differs from a handoff document, the provider is authoritative.
Update the protected-resource record before taking any action.

## 25. Current campaign defaults versus universal rules

The following are current optimizer-hunt defaults, not mathematical
universals:

- stock outer optimizer: memoryless SGD-0.28;
- core workloads: H16, H64, H256;
- development product margin: `0.018`;
- maximum allowed core regression: `0.009`;
- E1: one A100, M=1, H4, 32 commits;
- E2: four A100s, M=4, H16, 32 commits;
- same-state horizon: k0 and k8 with future groups 0--7;
- final confirmation: five fresh paired seeds; and
- concurrent accelerator ceiling: eight A100s, subject to quota and prior
  gates.

A future campaign may freeze different values before development. It may not
change them after inspecting the outcomes to which they apply.
