# CPLG-SGD v1 fresh full-vector shadow preregistration — retry r2

Status: frozen before outcome; launch requires a clean pushed implementation
commit and immutable cloud JSON

Evidence stage: GPU-assisted Stage 1 identification and Stage 2 production
parity acquisition. This is not an active optimizer arm, E1, a same-state CRN
loss comparison, or evidence that CPLG beats SGD-0.28.

## 0. Fresh retry identity and r1 exclusion

This is a newly reviewed protocol after r1 ended with a checksummed
`INCONCLUSIVE` before the direction analyzer ran. R1 is permanently preserved at
`gs://yeto-exp2-52-model-training-497007/exp2-cplg-shadow-direction-r1` and is
excluded from every r2 gate, bootstrap sample, and advancement decision. Its
captured vectors will not be replayed to tune CPLG or forecast the r2 direction
outcome. The only scientific-controller repair is to accept the comparison
program's truthful closed result-row schema `[base (untrained), OFF, ON]` rather
than the incorrect `[OFF, ON]`. The harness also writes explicit nonempty clean
Git/no-diff attestations so a successful run can satisfy exact-ID `delete`.

Every candidate mechanism, arithmetic rule, workload, seed, gate, bootstrap
constant, claim boundary, and one-A100 maximum below is unchanged. R2 uses a new
source commit, run ID `exp2-cplg-shadow-direction-r2`, capture UUID
`667f5de8-6d6d-4ce0-9344-efc239583abf`, artifact prefix, cloud identities, and
immutable spec. No automatic retry follows an r2 `INCONCLUSIVE`.

## 1. Candidate identity

The candidate is **Causal Phase-Locked Geodesic SGD v1** (`CPLG-SGD-v1`).
The stock control is exact memoryless outer SGD: f32 learning-rate bits
`0x3e8f5c29` and exact positive-zero momentum bits `0x00000000`.

Implementation paths:

- production selector: `syncer/src/state.rs`, optimizer name `cplg-sgd`;
- treatment validation: `syncer/src/main.rs`;
- independent f32 geometry/state reference: `yeto/cplg_sgd.py`;
- authoritative transcendental helper:
  `syncer/src/bin/cplg_libm_oracle.rs`, pinned `libm = 0.2.15`;
- strict shadow analyzer: `scripts/replay_cplg_shadow.py`;
- formal scope: `lean-mechanism/LeanMechanism/CausalPhaseLockedGeodesic.lean`.

The final immutable spec must bind this dossier to the full 40-character
clean, pushed source commit. Before either stock arm runs, the wrapper builds
the helper with locked dependencies, publishes its SHA-256 and the Cargo.lock
SHA-256 in an atomic checksummed preflight receipt, and rehashes the helper
after the pair; it also requires HEAD to equal the immutable spec's explicit
40-character commit and rejects tracked or untracked nonignored files. The
verdict reports the same executable identity. Changing the
phase sign, transport, cap, interlock length or
inequality, score, zero-phase rule, clearing rule, arithmetic order, trig
runtime, workload, seed, bootstrap, threshold, or denominator creates a new
candidate/run ID and requires a new preregistration.

## 2. Frozen mechanism

For three causal same-fragment stock directions, let `v` be the previous unit
direction, `u` the current unit direction, and
`rho = dot(u,v)`. V1 accepts only finite acute nonstationary geometry
`0 < rho < 1`, with dot overshoot tolerance `2^-20` and squared-norm floor
`2^-40`. Its current forward tangent is

`d = normalize(rho*u - v)`.

The prior forward tangent is parallel transported from `v` to `u` by

`PT(h) = normalize_tangent(h - (dot(h,u)/(1+rho))*(v+u))`.

With `c = max(+0, dot(d,PT(h)))`, current and preceding turn angles
`theta_t`, `theta_t-1`, and f32 cap bits `0x3e7adbb0`, the commanded angle is

`phi = f32(c * min(theta_t, theta_t-1, cap))`.

The candidate is

`C = norm_f32(G) * normalize_f32(cosf(phi)*u + sinf(phi)*d)`.

Products and sums are separate sequential f32 operations in coordinate order;
FMA is not permitted. Pinned Rust `libm = 0.2.15` is the sole authoritative
runtime for `atan2f`, `sinf`, and `cosf`. Python remains independent for all
vector, state, score, and ledger arithmetic and calls only the hash-pinned raw
f32-bit helper for those three primitives.

Per fragment the causal state is exactly: previous stock bytes; previous
forward-tangent bytes; previous turn-angle f32; one pending candidate; and the
last at most three resolved f32 scores. Preview is pure. Commit installs one
preview exactly once.

At boundary `t`, a pending candidate sealed at `t-1` is resolved first:

`z_t = cos(C_t-1,G_t) - cos(G_t-1,G_t)`.

The score window retains the newest three resolved scores. The current
candidate is selected only if all three are strictly positive. It is still
sealed when the interlock is closed.

Valid zero phase or a candidate that rounds byte-identically to stock is
sealed as an exact copy of the original stock byte object. Its later score is
therefore exactly positive zero, which closes the strict-positive interlock.
It is never reconstructed by normalization. Nonfinite, degenerate, nonacute,
shape/layout/hash/sequence-invalid, or invalid-shadow evidence returns the
original stock bytes and clears the complete fragment causal state. A cleared
fragment needs a fresh stock warm-up and phase warm-up.

## 3. Falsification and safety

CPLG addresses PTI-v1's fixed-turn behavior: PTI produced mixed delayed signs
and never opened its three-positive interlock in E1. CPLG shrinks continuously
to exact stock when two consecutive turns lack coherent transported phase.

Counterexamples are part of the candidate identity. Coherent continuation can
be worse after a true reversal or in objectives whose next direction does not
continue the observed spherical phase. The Lean reversal theorem is a geometry
counterexample, not a language-model loss theorem.

This shadow stage kills the candidate if it is inactive, directionally weak,
fragment-local, non-bit-exact, state-divergent, or too expensive. It does not
observe a candidate-arm loss and cannot rescue a failed gate with a favorable
terminal stock loss.

## 4. Fresh acquisition design

The separate frozen scientific configuration is
`experiments/optimizer/cplg-sgd-shadow-direction-r2-config.json`, with
basename-bound SHA-256
`fb7d4c0539cc8760058e0f0b20101bde7fcbac9224c8b27ca69d9724180aaf96`.
The wrapper must hash that exact file and match it to the producer's declared
run-configuration identity before starting the compare process.

The final JSON will pin the exact image, model/data manifests, commit, zone,
machine, disk, run prefix, and command. The frozen scientific profile is:

- one Spot A100 on `a2-highgpu-1g`, maximum one active accelerator;
- exact retained image ID `7290368630472593484` unless a new image is
  separately qualified before finalization;
- one full-model learner and one CPU syncer over localhost;
- stock outer SGD-0.28 only; CPLG actions are simulated offline;
- H4, four logical fragments with exact IDs `[0,1,2,3]`, quorum one;
- f32 wire, overwrite broadcast, deterministic commit order, no reconnects;
- exact seed, model, dataset/order, LoRA rank/layout, inner AdamW, scheduler,
  clipping, 34 executed local steps at sequence length 128 (exactly 4,352 raw
  local training tokens), and eight evaluation rows copied from the final
  matched spec; the compare CLI `--token-budget` is therefore exactly `4352`,
  while `--learner-max-steps 96` supplies strict-quorum liveness headroom and
  is not counted as executed work;
- exactly 32 contiguous commits in the order `[0,1,2,3]` repeated eight
  times; and
- sequential capture-OFF then capture-ON stock arms from identical initial
  model/adapter/optimizer/data/RNG state. Only the ON arm publishes the exact
  stock vectors. Neither arm runs the CPLG action online.

The producer interval begins after the syncer has received, initialized, and
hashed the exact complete global f32 state and immediately before it opens the
commit scheduling loop. It ends after commit 32 and, for the ON arm, after the
vector writer and its manifest/sidecar have durably closed. It excludes model
loading/hashing, export, and terminal evaluation. Rust publishes the unrounded
monotonic duration in nanoseconds as a checksummed completion receipt. Both
arms must report this exact scope, 32 commits, 34 terminal learner local steps,
the same fragment order, responder identities, and input hashes. The overhead
is computed, never trusted from input, as `(on_ns - off_ns) / off_ns` using
exact integer nanoseconds. A negative value is retained rather than clamped.

## 5. Exact vector tape and ledger

For each committed ON boundary the capture path independently materializes the
same deterministic weighted-RDA stock pseudo-gradient consumed by the legacy
stock commit path and writes it as canonical little-endian f32 bytes before
CPLG selection. The committed stock path then recomputes its aggregate from the
same sorted responders and weights; exact OFF/ON trajectory, checkpoint, export,
and loss parity is mandatory, and measured overhead includes this capture-only
rematerialization. Each row binds commit sequence, fragment, fragment version, numel,
layout identity, responder/weight/merge identity, vector path, vector SHA-256,
previous ledger hash, and row ledger hash. The row hash is SHA-256 of canonical
UTF-8 JSON containing every row field except the row hash itself, plus one
newline. The first predecessor is 64 ASCII zeroes.

A final manifest binds the tape SHA-256, ledger head, exact record and fragment
counts, total vector bytes, the producer-derived live-layout and exact initial
f32-state hashes, the frozen scientific run-configuration hash, writer
accounting, and completion status. The wrapper hashes the actual separate
configuration file before launch; the immutable cloud spec independently pins
the source commit and binds that configuration hash plus image/model/data
provenance, avoiding a self-referential source-commit hash.
The OFF and ON initial-state and completion receipts, manifest, overhead
evidence, and verdict are atomically published, fsynced, and accompanied by
basename-bound SHA-256 sidecars. The wrapper also requires exact OFF/ON final
syncer checkpoints, exported adapter trees, normalized event tapes, and
evaluation losses. Missing, duplicate, stale, noncanonical, symlinked,
escaping, nonfinite, checksum-mismatched, dropped, behavior-changing, or
unclosed evidence is an integrity failure. No denominator is reduced.

## 6. Frozen replay and statistical gate

The sole primary directional effect is the f32 delayed score `z_t` above. The
frozen causal accounting is:

- 32 input boundaries, exactly eight per fragment;
- two noncandidate warm-ups per fragment;
- 24 sealed shadows;
- 20 resolved shadows, exactly five per fragment;
- four declared unresolved tail shadows, exactly one per fragment; and
- simulated selection using the last three already resolved scores.

The uncertainty procedure uses exactly 20,000 circular moving-block bootstrap
draws, seed `0x43504c47`, block length two, independently within each fragment.
Each fragment's five chronological scores are resampled to length five by
drawing uniform circular block starts and truncating the concatenated blocks.
The draw statistic is the equally weighted mean of all 20 resampled scores.
The one-sided lower endpoint is the order statistic at zero-based index
`floor(0.05 * 20000) = 1000` after ascending sort. There is one primary
candidate/statistic, so no multiplicity adjustment is applied.

Block starts use a portable SplitMix64 stream. Initialize unsigned 64-bit
`state = 0x0000000043504c47`. For every requested start, set
`state = state + 0x9e3779b97f4a7c15 (mod 2^64)`, then compute
`z = state`; `z = (z xor (z >> 30)) * 0xbf58476d1ce4e5b9 (mod 2^64)`;
`z = (z xor (z >> 27)) * 0x94d049bb133111eb (mod 2^64)`; and
`r = z xor (z >> 31)`. The circular block start is `r mod 5`. Draw order is
bootstrap replicate outermost, then fragments in ascending numeric order,
then block starts until five scores have been produced for that fragment.

`PASS` requires every condition:

1. the source/helper/spec/tape/manifest/vector/ledger/checksum and causal
   reference/production parity checks pass;
2. the exact 32/4x8, 24/20/4 accounting above passes with no state-clearing or
   integrity fallback;
3. at least eight of all 32 boundaries simulate a non-stock selected action;
4. the mean of all 20 resolved scores is strictly greater than `0.001`;
5. the frozen bootstrap lower endpoint is strictly greater than zero;
6. at least three of four fragment mean scores are strictly positive; and
7. matched interval overhead is at most `0.02`, with zero writer drops,
   abandoned bytes, pending items, errors, or residue.

An interpretable gate miss is `FAIL`. Missing or malformed evidence is
`INCONCLUSIVE`, `UNIDENTIFIABLE`, or `INFRA_FAILURE` according to the active
SOP, never a partial pass.

## 7. Allowed next action

`PASS` authorizes only a separately reviewed and preregistered Stage-3 E1
matched active-action canary. That E1 dossier must freeze its own online safety,
activity, overhead, loss, ledger, and Stage-4 advancement gates before launch.
It cannot reuse this stock trajectory as a live loss control.

`FAIL` kills CPLG-v1. `INCONCLUSIVE` permits no automatic retry; any later
attempt requires another newly reviewed protocol and fresh identity without
using hidden outcomes. Infrastructure failure permits at most one
fresh-identity retry without changing scientific semantics. No
shadow verdict authorizes E2, CRN finite-loss language, H16/H64/H256 product
claims, confirmation seeds, or replacement of production SGD-0.28.

## 8. Resource and teardown envelope

The shadow stage may own at most one Spot A100 VM, its exact auto-delete boot
disk, and one immutable run-specific object prefix. Provider maximum duration
is 3,600 seconds with explicit time reserved for validation, final sync, and
teardown. The final spec records exact numeric identities and ownership nonce.
Every outcome ends in checksummed completion plus exact-ID `delete`, or
preserved failure evidence plus exact-ID `abandon`; VM and disk not-found
lookups and the protected unrelated inventory are recorded afterward.
The wrapper's terminal supervisor atomically publishes a separate checksummed
closed-vocabulary verdict on every started attempt: analyzer `PASS`/`FAIL`,
`INCONCLUSIVE` for post-acquisition evidence failure, or `INFRA_FAILURE` for
configuration/build/compare/runtime failure. A nonfresh terminal path prevents
the attempt from starting. `UNIDENTIFIABLE` is reserved for a complete
evidence inventory that cannot identify the requested action; it is not used
to excuse missing artifacts in this fresh capture.

## 9. Pre-acquisition verification record

The inherited candidate implementation at `a4122c5f...` passed 89 focused
CPLG Python tests, 197 production syncer tests, two authoritative-libm helper
tests, the locked release build, Rust formatting, and the shared raw-bit
fixture before r1. After the r1 controller failure and before any r2 outcome,
the fresh result-schema, lifecycle-attestation, config, replay, wrapper,
spec, and harness contract set passed 131 tests. Ruff check/format passed on
every changed Python file. The full repository suite passed 1,285 Python
tests with two skips when run under CPython 3.13 with the declared `dev`,
`launcher`, and `nava` extras.
An earlier dependency-incomplete full invocation reached 1,255 passes and two
skips but failed 12 unrelated SkyPilot/AWS tests because it omitted
`skypilot[aws]` and `boto3`; no CPLG test failed in that invocation. The exact
4,352-token dry render reports 34 steps for both frozen arms.
The wrapper additionally verifies the configuration's basename-bound checksum
sidecar and requires exactly one untrained-base result row followed by exactly
one OFF row and exactly one ON row; duplicate, extra, or reordered results are
fatal. The base, OFF, and ON losses must all be finite, and OFF/ON must remain
bit-identical.

One shared 11-boundary raw-f32 fixture is consumed by both the Rust production
selector and the independent Python reference. At every boundary it freezes
action and candidate bytes, complete causal-state bytes, theta and delayed-score
bits, the score window, reason, interlock, and clearing state; continuation after
the frozen resume boundary must remain identical. The trace includes first
non-stock selection, a nonacute full-state clear, a `-0.0` coordinate,
byte-identical zero-phase sealing, and a subsequent exact `+0.0` score. Its first
run exposed only a diagnostic spelling mismatch (`non_acute_turn` in Rust versus
`nonacute_turn` in Python); Rust now uses the analyzer's canonical spelling, and
the complete trace passes in both runtimes.

Every analysis, overhead, preflight, and terminal output plus sidecar must be a
fresh directory entry; regular files, sidecars, symlinks, and dangling symlinks
all prevent acquisition, and atomic publication never replaces an existing
entry. The terminal supervisor catches unexpected ordinary exceptions at its
outer boundary, preserves the class/message, and emits `INFRA_FAILURE`; valid
scientific `PASS`/`FAIL`, evidence `INCONCLUSIVE`, known infrastructure errors,
unexpected errors, and stale-terminal refusal all have direct tests. Git HEAD,
clean status, `Cargo.lock`, and helper bytes are revalidated after the pair and
after analysis, and the analyzer-reported helper digest must equal preflight.
