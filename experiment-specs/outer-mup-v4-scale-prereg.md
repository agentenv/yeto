# Outer-muP v4 scale addendum: 1.7B raw-arm finite-horizon T-scan

Status: **PREREGISTERED**, prospective for the 48-cell G4 grid. The required
three-cell center pilot is disclosed below and is not part of G4.

Registered 2026-07-26. JSON is authoritative:
`experiment-specs/outer-mup-v4-scale-prereg.json`.

## 1. Question and scope

Does the finite-horizon transient law transfer from SmolLM2-135M to
SmolLM2-1.7B (the requested 12.6x parameter scale-up)? This is a deliberately
minimal **raw-Nesterov-only** test. It does not rerun the v3 correction or SNOO
arms and makes no claim of exact numerical-constant replication.

The two horizons are obtained at fixed `H=512`:

| learner steps S | per-fragment horizon T=S/H | global syncer commits |
|---:|---:|---:|
| 2,560 | 5 | 20 |
| 10,240 | 20 | 80 |

Every cell uses `M=4` learners packed onto one physical H200, one cell per GPU,
full-parameter bf16 training with AdamW, raw production Nesterov, strict quorum,
barrier synchronization, version-matched anchors, RDA matrix merging, learner
broadcast blend `merge_alpha=0.5`, no delta
correction, no injected delay/jitter, and rho telemetry ON.

## 2. Disk and memory gates completed before scientific work

`lsblk`, `vgs`, `lvs`, and `pvs` were run on both nodes before creating any v4
checkpoint. Each node has eight 3.5-TB NVMe PVs fully allocated to one 27.9-TB
linear LV, already formatted XFS and mounted at `/data`. The VGs have zero free
extents because the whole devices are already in the LV; the mounted filesystem
has about 27.5/27.6 TB free. `h200-n1` root was 97% used (55 GB free) and
`h200-n2` root was 88% used (207 GB free). We therefore created
`/data/yeto-results-v4` and the symlink
`/root/yeto-results-v4 -> /data/yeto-results-v4` on each node. The v4 controller
fails closed unless that mapping and at least 1 TB free are still true.

The required four-learner memory smoke ran an exact `[1,128]` bf16
forward/backward/clip/AdamW step in four concurrent full-parameter 1.7B processes
on one H200. All four states were held resident together. Sampled aggregate VRAM
was **71.283 GiB** (`72,994 MiB`) versus 139.81 GiB available. Its aggregate
record SHA-256 is
`292002deb97438d374631bee114b0884e00bd8201d01544c3f16fe14130e95f2`.
The live fixed-window pilot subsequently reached `103,216 MiB` because it also
holds production anchors/snapshots and merge buffers; this remains below the
`143,771 MiB` device total.

## 3. Model and input identity

Both nodes contain the same pinned snapshot:

- model: `HuggingFaceTB/SmolLM2-1.7B`;
- revision: `effd688a12921b4cc83e3312b6feb579f70f9c71`;
- exact parameter count: 1,711,376,384;
- `model.safetensors` SHA-256:
  `1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0`.

The quartermaster scale manifest is identical on both nodes (SHA-256
`b737381d6fbe1ecdfe1c98ae9a4801ef654f0dc65d743a222b7fbbffadc44a36`,
status PASS). It proves byte-identical tokenizer files between the SmolLM2-360M
and SmolLM2-1.7B snapshots. Its four SmolLM2 learner lanes contain 26,727–27,399
complete 128-token blocks each, exceeding the long cell's 10,240 blocks without
wrap. The four materialized `[2560,128]` safetensors were used for the memory
smoke; the registered cells repack the manifest-bound 13,758-row raw training
file through that same tokenizer so they can consume 10,240 blocks. Training and
the fixed 1,024-row development evaluation are disjoint and hash-bound in the
JSON.

## 4. Required three-cell pilot and 135M-informed ladder placement

Before any G4 ladder was frozen, three `S=2560`, `mu=0` cells ran in parallel at
etas `{0.02215, 0.0443, 0.0886}`: a deliberately wide 2x bracket centered on the
disclosed 135M `T=5` mu0 center `0.0443`. The pilot used separate seed 499
(`training_seed=499499`), source commit
`9db854a036b4d53d7ee3424cc52f68a37161c3e0`, and is excluded from G4.

Pilot development losses at ascending eta were **{1.7413021864, 1.9114922716,
5.4440701253}**. The predeclared three-point
quadratic in `x=log2(eta)` was interior and gave
`eta*_pilot = 0.030244878343039974` (readout SHA-256
`bc49135ba53c8accb052bf6cc4441f3bd261419c8ff56e4b45bb845dccfe3014`).
The production-pattern evidence validator independently passed all three cells
(validation SHA-256
`38b92ecb54a9502affec01b5b061ae0d868efe7e4f4671a4e58604a8f83b7bcc`).
The fitted value is the frozen 1.7B `T=5,mu=0` ladder center.

The `T=20,mu=0` center uses no additional 1.7B outcome. It transfers only the
previously disclosed 135M duration ratio:

```text
C20_mu0 = eta*_pilot * (0.0242 / 0.0443) = 0.0165220328645952
```

This is a ladder-placement device, not evidence for transfer. Thus all four
mu0/mu.9 centers are explicitly 135M-informed: the pilot bracket and T20 ratio
come from the 135M mu0 centers, while the momentum centers use the code-true
finite-horizon predictions requested for this addendum:

```text
D_pred(T=5)  = 2.134
D_pred(T=20) = 1.123
C_T_mu.9     = 0.1 * D_pred(T) * C_T_mu0
```

The frozen centers are therefore:

| T | mu=0 center | mu=.9 center | center provenance |
|---:|---:|---:|---|
| 5 | 0.030244878343039974 | 0.0064542570384047305 | pilot; then `0.1*2.134` |
| 20 | 0.0165220328645952 | 0.001855424290694041 | pilot times 135M duration ratio; then `0.1*1.123` |

Every curve uses the same four-point sqrt2 ladder with log2 offsets
`{-0.75,-0.25,+0.25,+0.75}`. The authoritative numeric centers and all 16 eta
values are in the JSON; no ladder may be recentered after a G4 outcome exists.

## 5. Registered grid (48 cells)

```text
model SmolLM2-1.7B
M=4, H=512, raw arm only
S in {2560,10240} -> T in {5,20}
mu in {0,.9}
four registered etas per curve
seeds {501,503,509}; training_seed=int(str(seed)+str(seed))
2 * 2 * 4 * 3 = 48 cells
```

The 13,758-row training order and 1,024-row development set are held fixed;
registered seeds govern learner initialization/dropout. Endpoint development
NLL/token is the primary outcome. Nonfinite loss is scientific divergence and
stays in the denominator.

The 24 `S=10240` cells are assigned first, followed by the 24 `S=2560` cells,
using deterministic shuffle seed 20260726 and greedy cost balancing over the 16
GPU slots (cost ratio 4:1). Each slot is strictly longest-first. A physical GPU
runs one cell at a time; each cell deliberately packs its four learners on that
GPU.

## 6. Frozen analysis and G4

The frozen analyzer is `scripts/analyze_v4.py`, raw-file SHA-256:

```text
654eb63d5a830b275dfb64d2c9e178ff83c2355a7ec88e02c8c46e2cf1300471
```

For each `(T,mu)` curve it fits
`loss = a*log2(eta)^2 + b*log2(eta) + c` to the four exact registered eta
values using the three-seed mean at each eta. An optimum is interior only when
`a>0` and the vertex is strictly inside the numeric grid (a `1e-12` coordinate
tolerance prevents floating-point boundary promotion).

The registered displacement ratio is

```text
D_obs(T) = [eta*(mu=.9,T) / eta*(mu=0,T)] / (1-.9).
```

The monotonicity interval is a paired nonparametric training-seed curve
bootstrap: 10,000 draws, RNG seed 20260726, the same three-index resample applied
to all four curves, and all four quadratics refit in every draw. The coordinate
is `log2 D(T=5) - log2 D(T=20)`. Percentiles are conditional on interior refits;
at least 9,500/10,000 refits must be valid or G4 is NOT_EVALUABLE.

G4 is PASS iff all of the following hold:

1. both mu=.9 pooled optima are interior (mu0 optima are also required as a
   mathematical precondition to define D);
2. `D_obs(T=5)` is in `[1.7,3.2]`;
3. `D_obs(T=20)` is in `[0.8,1.5]`;
4. the paired 95% bootstrap interval for
   `log2 D(T=5)-log2 D(T=20)` has lower endpoint greater than zero.

The bands mean “same law regime as 135M,” not exact-constant replication. A
complete, bracketed result outside any band or without monotonic separation is
G4 FAIL. Missing/invalid work, any ratio-required pooled unbracketed optimum, or
an invalid bootstrap is G4 NOT_EVALUABLE. Both FAIL and NOT_EVALUABLE are
reported; neither authorizes outcome-aware recentering.

The analyzer writes `/root/g4-readout.json` and prints exactly:

```text
G4 VERDICT: <PASS|FAIL|NOT_EVALUABLE> D5=<value|NA> D20=<value|NA>
```

## 7. Evidence and retry contract

The launch manifest binds the pushed registration commit, both contract hashes,
the frozen analyzer hash, every materialized input hash, every initial command,
and one precomputed attempt-2 command per cell. Each node must emit a PASS
node-authority proof covering Git cleanliness, eight H200 UUIDs, the manifest,
contracts, inputs, and the `/data` result mapping before a cell can start.

Completed evidence requires the exact command hash and clean pushed commit,
20/80 event-tape rows and rho rows as appropriate, strict four-responder quorum,
exact `H=512` counters, all 2,560/10,240 learner steps, finite 1,024-row endpoint
loss, checkpoint and evaluation hashes, learner/syncer exit evidence, timestamps,
and an immutable attempt ID. Slot controllers write 30-second GPU/elapsed
heartbeats so 45-second-plus full-fragment merges are not mistaken for stalls.
Per-cell arm timeouts are 180 minutes (`S=2560`) and 420 minutes (`S=10240`).

There is at most one loss-blind retry wave. A retry is permitted only for a
registered infrastructure reason and must include all four eta cells in the
same `(T,mu,training-seed)` curve. Poor finite loss, a boundary optimum, or any
scientific outcome can never trigger retry. Retry commands and attempt-2 output
paths are hash-bound before launch and remain inside the same wall ceiling.

## 8. Twelve-hour wall and rails

The conservative wall clock begins at the first pilot scientific process,
`2026-07-26T15:02:39Z` (`1785078159`), and ends exactly 12 hours later at
`2026-07-27T03:02:39Z` (`1785121359`). The pilot, registration, main grid,
validation, any retry, evidence collection, and final analyzer all count. At the
deadline, controllers terminate their exact process groups, mark remaining work
`NOT_RUN_WALL_CEILING`, and do not extend after seeing outcomes.

No GCP or AWS is used. v1/v2/v3/explore result trees are immutable and must not
be read as v4 outcomes or modified; v4 reads only the prepared input data. All
checkpoints and evidence live under the roomy v4 volume. No wildcard `pkill` is
used (any emergency name match must use the bracket idiom); ordinary termination
targets recorded process groups. GitHub pushes occur only from the Mac clone.
