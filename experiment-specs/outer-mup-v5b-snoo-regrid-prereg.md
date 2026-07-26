# Outer-muP v5b: second SNOO regrid

**Program ID:** `outer-mup-v5b-snoo-regrid`
**Status:** **PREREGISTERED before any v5b attempt directory, GPU work, or endpoint result.**
**Machine contract:** `experiment-specs/outer-mup-v5b-snoo-regrid-prereg.json`

## 1. Complete pre-outcome disclosure of the v5 miss

V5 finished all `90/90` registered cells with valid hash-bound evidence, but G5
was not evaluable. The sealed operator audit is:

```text
h200-n1:/root/yeto-results-v5/_controller/g5-not-evaluable-audit.json
sha256 9251d78a9e88dc98f78ff524814454392633ade4315a464450c8c8ad78cec93b
```

Its v5 launch-manifest SHA-256 is
`b9d54918da4e884f0b82db97df28996b0f88805038cd8d7472f4c244030405e8`.
For all three conditions, every five-seed mean loss strictly increased with eta:
loss therefore decreased toward the low edge throughout each registered grid.

| condition | v5 log2-eta range | fitted vertex | status | low-edge mean loss |
|---|---:|---:|---|---:|
| a | `[-2.000, +0.500]` | `-2.1353914` | UNBRACKETED | `2.5136736` |
| b | `[-4.035, -1.535]` | `-8.7380109` | UNBRACKETED | `2.5071047` |
| c | `[-2.001, +0.499]` | `-2.3326638` | UNBRACKETED | `2.4847845` |

The paired seed bootstrap had `0/10,000` all-three-interior refits, so no closed
G5 scientific verdict was assigned. The unchanged frozen v5 analyzer also
raised `TypeError` in its main routine after correctly finding the unbracketed
fits, because it subtracted `None` tuned losses. The sealed audit used the
unchanged loader, fitter, and bootstrap functions and did not invent a verdict.
The frozen v5b analyzer explicitly emits a complete `NOT_EVALUABLE` readout
without subtracting losses unless all three pooled fits are interior.

These facts are disclosed before any v5b outcome. They fix the second grid only.
No v5 cell is rerun or altered.

## 2. Prospective 75-cell design

The scientific protocol remains SmolLM2-135M full-parameter training at `M=1`,
`H=512`, `S=2560`, `T=5`, 327,680 tokens, sequence length 128, AdamW inner LR
`.001`, strict quorum, barrier synchronization, version-matched anchors, fixed
512-step/65,536-token windows, zero delay or jitter, rho telemetry, raw outer
Nesterov, and no outer bias correction.

The same five seeds `{521,523,541,547,557}` are reused, including their frozen
no-wrap orders and `seed||seed` training-seed rule. This enables paired inference;
every v5b cell is new because none of its etas occurred in v5.

The three conditions are:

- **a:** plain-AdamW-equivalent outer `mu=0`.
- **b:** SNOO-style outer Nesterov with `mu=.9`.
- **c:** outer `mu=0`, shifted two registered log2 rungs above b as the
  prespecified code-true-law control.

Each grid has five exact 2x-spaced points:

| condition | exact log2 etas | exact etas |
|---|---|---|
| a | `{-9,-8,-7,-6,-5}` | `{.001953125,.00390625,.0078125,.015625,.03125}` |
| b | `{-11,-10,-9,-8,-7}` | `{.00048828125,.0009765625,.001953125,.00390625,.0078125}` |
| c | `{-9,-8,-7,-6,-5}` | `{.001953125,.00390625,.0078125,.015625,.03125}` |

This gives exactly `3 * 5 * 5 = 75` cells. The quadratic uses the actual
registered log2 coordinates, following the established 2x-ladder precedent.

The explicit power-of-two grids make c/b exactly `4`, while the finite-`T`
code-true multiplier is `(1-.9^5)/.1 = 4.0951`. Thus registered c is
`0.9767771` times the exact pointwise match, a `2.3223%` downward difference.
The exact numeric grids control; this discrepancy is disclosed prospectively
rather than silently describing `4x` as `4.0951`.

No grid may be extended, removed, narrowed, shifted, or recentered after any
v5b attempt directory or outcome exists.

## 3. Frozen combined v5 + v5b analysis

The frozen analyzer is `scripts/analyze_v5b.py`, raw SHA-256:

```text
b6ed09595961b81ebd9c2b632211d591f67a36c027faea71ecb86de36cb1e0ef
```

G5B requires all 90 disclosed v5 cells and all 75 v5b cells. For each condition,
the five v5b eta means and six v5 eta means form one 11-point curve. V5b supplies
the new downward support; the v5 observations remain additional high-eta
coverage. Each eta has equal OLS weight after averaging the five paired seeds.

At exact `x=log2(eta)`, fit

```text
loss = A*x^2 + B*x + C.
```

The tuned optimum is `INTERIOR` only if `A>0` and
`min(x)+1e-12 < -B/(2A) < max(x)-1e-12`. Tuned loss is the quadratic evaluated
at that vertex. The primary estimands are fitted tuned loss `b-a` and `c-a`.

Confidence intervals use 10,000 paired nonparametric seed-curve bootstrap draws,
RNG seed `20260726`. One common five-index draw is applied across both campaigns,
every eta, and all three conditions. A draw is invalid if any combined-grid
vertex is not interior; at least 9,500 draws must remain valid. Intervals are
equal-tailed percentile 95% CIs.

G5B is evaluable only when all 165 evidence cells are valid, all three pooled
optima are interior, and the bootstrap threshold passes. Its closed verdicts are:

- `SNOO_HELPS`: the `b-a` CI upper endpoint is below zero.
- `SNOO_NULL`: the `b-a` CI contains zero.
- `SNOO_HURTS`: the `b-a` CI lower endpoint is above zero.

Condition `c-a` is co-reported but does not select the SNOO verdict. The output
goes to `/root/g5b-readout.json`; the required note in
`/private/tmp/h200-v5b-note.md` is:

```text
G5B VERDICT: <SNOO_HELPS|SNOO_NULL|SNOO_HURTS> b-a=<point> [<low>,<high>] c-a=<point> [<low>,<high>]
```

## 4. Collision-free sharing with v4b

At registration, the active v4b launch manifest has SHA-256
`f2abf80d975572dde33ee2c750c1fb91598df8bfea5a78696bdc2c5d3608b55b`
and assigns only GPUs `0..5` on each node. GPUs `6,7` are unassigned and idle.
All 75 v5b cells are shuffled with RNG seed `20260726` and round-robin assigned
to exactly four static queues: GPUs 6 and 7 on `h200-n1` and `h200-n2`. Queue
loads are 18 or 19 cells. V5b never claims GPUs 0–5 and never stops, relaunches,
or modifies v4b.

Immediately before authority, each node must hash-verify the active v4b manifest,
prove it assigns no target GPU, and prove GPUs 6 and 7 have no compute process
and at most 16 MiB allocated. The fresh proofs are hash-bound into the immutable
launch authority.

The active v4b checkout `/root/yeto` remains pinned at its registered commit.
Only after the Mac registration push, v5b creates clean isolated checkouts at
`/root/yeto-v5b` on both nodes and executes at the new registration commit. This
prevents a pull from changing code or Git evidence for queued v4b cells.

## 5. Evidence, retry, storage, and rails

V5b reuses the byte-identical v5 input manifest, SHA-256
`3b0461c1daa821cef7c67fa087613abb537c4c5ee3e0f1c1bacf2144875581fe`,
and the frozen development evaluation, SHA-256
`533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc`.
All model, input, command, source-commit, strict-quorum, tape, telemetry, endpoint,
and artifact hashes are validated under the standard evidence rails.

One attempt-2 wave is allowed only for an enumerated, loss-blind infrastructure
failure and retries the full five-eta condition-by-seed curve. Finite unfavorable
loss, scientific divergence, an edge optimum, or any outcome-aware reason cannot
authorize retry.

The 12-hour wall ceiling begins at immutable launch-authority creation. Normal
termination targets recorded process groups only; wildcard `pkill` is forbidden.
V5b writes only to new symlinks
`/root/yeto-results-v5b -> /data/yeto-results-v5b`. V5, v4b, and every earlier
result tree remain read-only. Sparse cross-node evidence mirrors for final
analysis may be created only inside the new v5b controller tree. No GCP command
or resource is allowed.

The preregistration files, frozen analyzer, and hash-bound launch machinery are
committed and pushed from the Mac before any v5b result root, manifest, authority,
attempt directory, or GPU process is created.
