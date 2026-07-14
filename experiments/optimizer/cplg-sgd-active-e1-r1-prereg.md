# CPLG-SGD v1 active E1 short-work engineering preregistration — r1

Status: scientific design frozen before any active E1 outcome; non-launchable
until the active pair, acquisition wrapper, CPU validator, clean pushed source
commit, and immutable cloud acquisition JSON have been implemented and
reviewed

Evidence stage: active online M=1/H4 short-work engineering canary. This is a
4,352-token matched screen, not the SOP's 32,768-token standard E1 profile, not
a same-state CRN comparison, and not a test that CPLG beats SGD-0.28.

## 0. Identity and immutable scientific configuration

The candidate is **Causal Phase-Locked Geodesic SGD v1** (`CPLG-SGD-v1`).
The planned fresh run identity is `exp2-cplg-active-e1-m1-r1`. The basename-
bound scientific configuration is
`experiments/optimizer/cplg-sgd-active-e1-r1-config.json`, SHA-256
`5afe2d4900051fda1ac99cc682c489dfeae85f0eb34d1816646b5bff5f0c26df`.
Its sidecar is part of this preregistration.

There is deliberately no immutable cloud acquisition specification yet. Any
in-progress active presets or acquisition wrapper remain uncommitted,
unreviewed implementation work at this freeze, and the CPU-only validator is
still required. A later implementation commit may make the protocol launchable
only if it implements this frozen design without changing candidate arithmetic,
workload, action denominator, gates, outcome vocabulary, phase boundary, or
claim. The future cloud JSON must independently pin the exact clean pushed
40-character source commit, this configuration digest, this dossier digest,
image ID, input manifests, paths, command, fresh object prefixes, and provider
identities. Filling in an unknown source commit now would be false provenance.

Any change to the mechanism, f32 order, authoritative trig runtime, interlock,
fallback, reason vocabulary, arm order, workload, seed, loss contrast,
threshold, activity denominator, overhead interval, artifact boundary, retry
rule, or claim creates a new candidate/run identity and requires review before
launch.

## 1. Required R2 prerequisite and exclusion

This E1 is conditional on the already closed stock-shadow direction run
`exp2-cplg-shadow-direction-r2`. The prerequisite is accepted only with all of
these exact identities:

- terminal verdict `PASS`, stage `completed_analysis`;
- artifact prefix
  `gs://yeto-exp2-52-model-training-497007/exp2-cplg-shadow-direction-r2`;
- implementation commit
  `eb6d21146011112ffe8df5cb518c985e8c0297bd`;
- cloud-spec SHA-256
  `b39ec7d4faec2691895a2ea41d94df0950e130212a26ad8abbe80d5efe989efd`;
- scientific-config SHA-256
  `fb7d4c0539cc8760058e0f0b20101bde7fcbac9224c8b27ca69d9724180aaf96`;
- analysis SHA-256
  `a1143e818580c3b6b463bb180a04fb20991b04498f5fcd4820da2c6a52ada4fc`;
- final-manifest SHA-256
  `cced40ef7bb2890b0ae92755070a259d8099abb3a4200d658f93e9b4db5b085f`;
- stock-tape SHA-256
  `1e939222005f78d9d9711c4558d59b922e566af513485ce231d11ce417838c31`;
  and
- stock-tape-manifest SHA-256
  `1a0d19a27b99c82a0110a3e186269ebd2d76396b5163a625086862aacabac3fd`.

R2 constructed 12 simulated nonstock actions, had four positive fragment
means, mean delayed shadow cosine gain `0.0028530240058898928`, one-sided
bootstrap lower endpoint `0.0027203083038330076`, and matched capture overhead
`0.009156851278518476`. Those values authorize writing and reviewing this
active protocol only.

R2 model state, vector tape, result losses, simulated actions, and bootstrap
samples are permanently excluded from active E1 inputs, controls, thresholds,
tuning, and outcome calculations. Both active arms are fresh. R2 cannot be
reclassified as the E1 stock arm, and its favorable direction result cannot be
combined with the active loss contrast.

## 2. Exact candidate and control

The control is exact memoryless outer SGD-0.28 through the existing `nesterov`
selector, with f32 learning-rate bits `0x3e8f5c29` and exact positive-zero
momentum bits `0x00000000`. The treatment uses the production optimizer name
`cplg-sgd` with the same learning rate, momentum, weighted-RDA merge, inner
optimizer, data, seed, and work. The outer direction selector is the only
scientific arm difference.

For a current same-fragment stock pseudo-gradient `G`, current unit direction
`u`, preceding unit direction `v`, and `rho = dot(u,v)`, CPLG accepts only
finite acute nonstationary geometry `0 < rho < 1`, with dot overshoot tolerance
`2^-20` and squared-norm floor `2^-40`. It constructs the current forward
tangent

`d = normalize(rho*u - v)`.

It transports the preceding forward tangent `h` from `v` to `u` by

`PT(h) = normalize_tangent(h - (dot(h,u)/(1+rho))*(v+u))`.

Let `c = max(+0, dot(d,PT(h)))`. With current and preceding turn angles and
angle-cap f32 bits `0x3e7adbb0`, the command is

`phi = f32(c * min(theta_current, theta_previous, angle_cap))`.

The candidate is

`C = norm_f32(G) * normalize_f32(cosf(phi)*u + sinf(phi)*d)`.

Every product, sum, subtraction, division, square root, reduction, and output
coordinate follows the frozen left-to-right sequential f32 order. FMA is not
permitted. Pinned Rust `libm = 0.2.15` is authoritative for `atan2f`, `sinf`,
and `cosf` through a hash-pinned helper. The independent Python reference owns
state, vectors, score, and evidence validation, not transcendental authority.

Each fragment stores exactly: previous stock bytes, previous forward-tangent
bytes, previous turn-angle f32, one pending candidate, and the newest at most
three resolved f32 scores. Preview is pure; commit installs one preview exactly
once. At a later same-fragment boundary, the pending candidate is resolved
before the new action:

`z = cos(pending_candidate,current_stock) - cos(previous_stock,current_stock)`.

The current nonstock candidate is applied only when the newest three already-
resolved scores all exist and are strictly positive. The current candidate is
still sealed when the interlock is closed. There is no coefficient search,
tie-break, outcome-dependent tuning, or look-ahead.

The closed candidate reason vocabulary is:

- `not_active`;
- `stock_warmup`;
- `phase_warmup`;
- `interlock_closed`;
- `candidate_selected`;
- `degenerate_stock`;
- `nonacute_turn`;
- `invalid_geometry`;
- `invalid_shadow_score`; and
- `zero_or_rounded_phase`.

For `candidate_selected`, `cplg_used_nonstock` must be true, the action hash
must equal the candidate hash, and both must differ from the stock hash. At
every other boundary, `cplg_used_nonstock` must be false and the selected
action must be the original stock byte object, with action and stock SHA-256
exactly equal. A zero or byte-rounded phase seals the original stock object as
the pending candidate; its later exact `+0.0` score closes the strict-positive
interlock. Nonfinite, degenerate, nonacute, invalid-geometry, or invalid-shadow
evidence returns exact stock and clears the entire affected-fragment causal
state. Valid safety fallback does not remove a boundary from the activity
denominator. Missing or malformed action identity is evidence failure, not a
fallback.

## 3. Truthful short-work matched design

The fixed arm order is fresh stock followed by fresh candidate:

1. `cplg_m1_stock`, outer optimizer `nesterov`;
2. `cplg_m1_candidate`, outer optimizer `cplg-sgd`.

The result order is exactly `[base (untrained), cplg_m1_stock,
cplg_m1_candidate]`. The untrained row is report-only. It may not replace the
stock control or be removed because baseline training is skipped.

Each arm is separately restored from the same initial model, adapter,
optimizer, data-order, and RNG state. The candidate cannot resume the stock
arm, consume its checkpoint or tape, or inherit its process state. The arms run
sequentially on one A100 so there is no concurrent-resource or scheduling
difference. Their initialization identity, responder schedule, train/eval row
identity, and all non-selector settings must match exactly.

The workload is deliberately identical to the truthful R2 acquisition:

- one learner, quorum one, four fragments, H4;
- sequence length 128, microbatch one, grad accumulation one;
- exactly 34 terminal local steps per arm;
- exactly 4,352 raw local training tokens per arm;
- compare token budget exactly `4352`;
- learner maximum 96 only as strict-quorum liveness headroom, never counted as
  executed work;
- exactly 32 commits per arm, eight per fragment, in `[0,1,2,3]` repeated
  eight times;
- f32 wire, weighted RDA, merge alpha zero, overwrite broadcast, no delta
  correction, no reconnects, strict quorum, barrier sync, and deterministic
  commit order;
- inner AdamW LR `0.001`, betas `[0.9,0.999]`, epsilon `1e-8`, weight decay
  `0.01`, ten-step linear warmup then constant, clip norm `1.0`;
- Qwen3.5-9B local model, Capybara local data, assistant-only cross entropy,
  LoRA rank two/alpha four/all-linear, no gradient checkpointing;
- row-shuffle seed `271`, training seed `271271`, maximum 5,000 rows;
- eight fixed evaluation rows; and
- CUDA device, one GPU slot, 20-minute timeout per arm.

This is intentionally not the standard 32,768-token E1 profile. The historical
`--token-budget 32768` label did not cause 32,768 tokens to execute when the
32-commit syncer terminated the learner at 34 steps. This protocol reports the
actual 4,352 tokens and makes no longer-work claim.

## 4. Frozen measurements and gates

### 4.1 Activity and exact fallback

All 32 predefined candidate-arm commit boundaries form the activity
denominator. No boundary is removed for warm-up, interlock closure, geometry
fallback, state clearing, malformed evidence, or unresolved tail state.

A valid active boundary has `cplg_used_nonstock = true`, reason
`candidate_selected`, candidate/action hashes equal, and action/stock hashes
different. A fragment is active if at least one of its eight predefined
boundaries is validly active.

The activity gate is inclusive:

- at least eight valid nonstock selected actions among all 32 boundaries; and
- activity in at least three of the four fragments.

Every non-action boundary must satisfy exact stock fallback. One action-hash
contract violation, one missing/extra/reordered boundary, an unknown reason,
or a noncanonical field prevents `PASS`; the denominator is never shrunk.

### 4.2 Terminal loss regression

The primary loss contrast is the unrounded finite binary64 value

`candidate_eval_loss - stock_eval_loss`

from the closed `results.jsonl` rows. `PASS` permits an inclusive maximum of
`0.05` cross-entropy loss/token. Both losses must be finite and bound to the
same eight evaluation rows. The gate is a noncatastrophic engineering screen,
not a superiority test: zero or negative contrast is descriptive, and a
positive contrast up to `0.05` can still pass.

### 4.3 Matched interval overhead

Each arm publishes unrounded producer monotonic integer nanoseconds. The
interval begins after complete global f32 state initialization and identity
hashing, immediately before commit scheduling opens. It ends after commit 32
and durable closure of that arm's event/action evidence. Model loading, input
hashing, export, and terminal evaluation are outside both intervals. Candidate
live geometry, interlock, hashing, and evidence-writing cost is inside its
interval.

The separate CPU analyzer computes

`(candidate_interval_ns - stock_interval_ns) / stock_interval_ns`.

It never trusts a supplied fraction, rounds before comparison, or clamps a
negative value. `PASS` requires the exact computed value to be at most `0.02`,
inclusive, with zero dropped, abandoned, pending, or errored writer items.

### 4.4 Complete PASS rule

`PASS` requires all of the following:

1. exact source/config/image/runtime/model/data/helper provenance;
2. fresh sequential arms from identical sealed initial state;
3. exact 34-step, 4,352-token, 32-commit, four-by-eight matched work;
4. finite unrounded three-row result evidence and identical evaluation rows;
5. a complete closed-vocabulary candidate tape with exact causal state and
   action/fallback hashes at all 32 boundaries;
6. at least eight valid nonstock actions across at least three fragments;
7. `candidate_eval_loss - stock_eval_loss <= 0.05`;
8. exact matched interval overhead `<= 0.02`;
9. zero checksum, path, symlink, schema, writer, manifest, or phase-boundary
   violation; and
10. complete CPU analysis and a checksummed terminal verdict.

A complete, identifiable, valid run that misses a scientific or engineering
gate is `FAIL`, never a favorable partial result.

## 5. GPU acquisition artifact boundary

The GPU acquisition product and CPU analysis product use fresh, distinct,
immutable prefixes. The planned names are:

- acquisition:
  `gs://yeto-exp2-52-model-training-497007/exp2-cplg-active-e1-m1-r1-acquisition`;
- analysis:
  `gs://yeto-exp2-52-model-training-497007/exp2-cplg-active-e1-m1-r1-analysis`.

The future cloud spec may launch only if both prefixes are empty and every
declared local output is fresh. CPU analysis reads but never modifies the
acquisition prefix.

The A100 runner is responsible for work that requires the live model or GPU:
preflight; both arms; terminal GPU evaluation; syncer checkpoints; adapter
export; train/evaluation row identities; event/action tapes; exact work and
interval receipts; writer close; and a canonical acquisition manifest. The
manifest covers the future cloud spec and command, clean/no-diff attestations,
all input provenance, the closed three-row results, both arm tapes/checkpoints/
logs/exports, all 32 candidate boundary records, the interval receipts, every
sidecar, and the GPU acquisition completion receipt.

The acquisition receipt uses state `GPU_ACQUISITION_COMPLETE` and explicitly
stores `scientific_verdict: null`. It is not a sixth scientific verdict. It
means only that the runner exited, every acquisition object is present, the
local manifest verifies, and CPU analysis can proceed from preserved inputs.

Full causal replay, final gate arithmetic, bootstrap or resampling, and final
scientific verdict publication are forbidden on the A100 VM. In particular,
the harness's existing `analysis` hook is not separation because it SSHes into
the still-live owned VM. The new validator must run in a separate pinned local
or zero-accelerator CPU environment after GPU teardown.

CPU analysis must round-trip every acquisition-manifest object, verify exact
SHA-256 and schema, validate causal action/fallback state, compute activity,
loss contrast, and overhead from frozen inputs, and publish a checksummed
analysis report plus terminal verdict to the separate analysis prefix.

## 6. Resource, synchronization, and teardown

GPU acquisition may own exactly one Spot `a2-highgpu-1g` A100 in
`us-central1-c`, maximum one active accelerator campaign-wide, a 250 GB
`pd-ssd` auto-delete boot disk, and the fresh acquisition prefix. The expected
retained image is
`projects/model-training-497007/global/images/yeto-optimizer-a100-20260714`,
exact image ID `7290368630472593484`. Provider maximum duration is 3,600
seconds. A later cloud spec must reverify image readiness, quota, empty prefix,
protected inventory, and exact source/input identities immediately before
launch.

The run directory syncs every 60 seconds as interruption protection. On
successful acquisition, the runner closes writers, verifies local sidecars and
the acquisition manifest, and exits without analysis. The controller performs
a final delta sync, downloads the manifest-listed bundle, verifies every
SHA-256 round trip, then exact-ID deletes the A100 VM and its auto-delete disk
and records provider not-found for both. CPU analysis starts only after that
teardown proof. A bounded final sync/verification tail is permitted; scalar
replay is not.

If acquisition is incomplete or the runner is nonzero, the controller stops
work, preserves available evidence, exact-ID abandons with a concrete reason,
and proves VM/disk absence. It may not create a scientific result from
incomplete acquisition. No unrelated VM or disk may be adopted, used for
analysis, stopped, relabeled, or deleted.

## 7. Closed outcome vocabulary and retry policy

The CPU terminal scientific verdict has exactly this closed vocabulary:

- `PASS`;
- `FAIL`;
- `INCONCLUSIVE`;
- `UNIDENTIFIABLE`; or
- `INFRA_FAILURE`.

`PASS` receives no retry. It establishes only the claim in Section 8 and
requires a separately reviewed higher-work or same-state CRN protocol before
any stronger test.

`FAIL` receives no retry and kills CPLG-SGD-v1 at active E1 under this frozen
gate.

`INCONCLUSIVE` permits no automatic retry. Evidence is preserved; any later
attempt needs a newly reviewed protocol and fresh identity without using the
hidden outcome to change scientific semantics.

`UNIDENTIFIABLE` means a complete evidence inventory cannot identify the
requested action/contrast. It receives no retry under this evidence design.

`INFRA_FAILURE` permits at most one fresh-identity retry with identical
scientific semantics, only after evidence preservation, exact-ID teardown,
and a documented root-cause repair. It cannot change arms, order, work, gates,
or artifact boundary.

A CPU analysis infrastructure failure is retried only on zero-accelerator CPU
from the same verified acquisition-manifest hash. It never retains, resumes,
or relaunches an A100. Every attempt ends with preserved checksummed evidence
and exact owned-resource teardown.

## 8. Claim boundary

If every gate passes, the sole allowed statement is:

> CPLG-SGD-v1 was active across the frozen 4,352-token matched online
> engineering workload, retained exact stock fallback, stayed within the
> absolute 0.05 terminal-loss regression and 2% matched-interval overhead
> limits, and completed the frozen artifact contract.

Even a `PASS` does not establish that CPLG beats SGD-0.28, positive expected
loss improvement, a same-state CRN finite-loss effect, standard 32,768-token
E1 completion, convergence, unconditional dominance, generalization across
seeds/models/H/learner counts/inner optimizers, authorization for E2, or
production replacement. The fixed stock-then-candidate order is a matched
engineering comparison, not order-randomized or boundary-restored causal
evidence.

## 9. Launch blockers at this freeze

No GPU is authorized by this dossier alone. Before an immutable cloud spec can
be written, review must confirm:

1. exact stock and candidate presets with only the outer selector differing;
2. a wrapper that resets both arms, produces the acquisition-only receipt and
   exits before CPU analysis;
3. a closed-schema CPU validator implementing every frozen gate and verdict;
4. tests for candidate activity, active fragments, exact fallback, action-hash
   corruption, loss boundaries, integer-nanosecond overhead boundaries, phase
   separation, artifact freshness, and every lifecycle outcome;
5. a clean pushed implementation commit and full source/input/image binding;
6. an immutable acquisition JSON whose completion paths end at GPU acquisition
   rather than final analysis; and
7. a dry render proving exactly 34 terminal local steps, 4,352 raw tokens, 32
   commits, and eight commits per fragment for both arms.

The R2 `PASS` satisfies only the prerequisite in Section 1. It does not waive
any of these blockers.
