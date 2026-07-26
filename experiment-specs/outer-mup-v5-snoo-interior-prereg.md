# Outer-muP v5: SNOO interior-optimum repair

**Program ID:** `outer-mup-v5-snoo-interior`
**Status:** **PREREGISTERED** before any v5 GPU work or endpoint result.
**Machine contract:** `experiment-specs/outer-mup-v5-snoo-interior-prereg.json`

## 1. Purpose and prior-result disclosure

This lane repeats the 135M, `M=1`, `H=512`, `S=2560` (`T=5`) SNOO-style
comparison with enough learning-rate support to estimate **interior tuned
optima**. It repairs a disclosed defect in v3 ARM S: all three five-seed curves
selected eta index 0, their lower registered boundary. The v3 analyzer therefore
reported every quadratic optimum as `UNBRACKETED` and used a descriptive
best-grid estimand.

The sealed source is `h200-n1:/root/g3-readout.json`, sha256
`d4a3cde6aa47580dff255c7a66030ab997a95f4072b1883bf71aa54d7da744c8`.
The v3 lower-edge winners were:

| condition | definition | v3 winning eta | v3 mean loss |
|---|---|---:|---:|
| a | plain-AdamW-equivalent `mu=0` baseline | 0.5946035575 | 3.0800111 |
| b | SNOO-style outer Nesterov `mu=.9` | 0.1450832680 | 3.1932354 |
| c | code-true eta-matched `mu=0` control | 0.5941304909 | 3.1246390 |

Those outcomes choose only the new grid centers. No v3 loss is pooled into G5.

## 2. Mandatory v4 drain gate

No v5 GPU process may start until both conditions hold:

1. The sum of `evidence.json` files under the two node-local
   `/root/yeto-results-v4` roots is exactly 48.
2. `pgrep -c -f 'run_slot_v[4]'` is zero independently on `h200-n1` and
   `h200-n2`.

The supervisor polls at five-minute intervals. The evidence count is summed
because the v4 manifest assigns disjoint cells to local filesystems (48 cells
total). `scripts/authorize_v5_launch.py` records both node counts and both process
counts before it can issue launch authority.

## 3. Design

SmolLM2-135M full-parameter training uses `M=1`, `H=512`, `S=2560`, `T=5`,
327,680 tokens, sequence length 128, AdamW inner LR `.001`, strict quorum,
barrier synchronization, version-matched anchors, fixed 512-step windows, zero
injected delay/jitter, and rho telemetry. Outer bias correction is off.

Five fresh paired seeds are `{521,523,541,547,557}`. Each condition has six eta
cells per seed, for exactly `3 * 6 * 5 = 90` cells:

- **a:** outer `mu=0`; eta 1 applies each single-worker AdamW delta exactly.
  The surrounding eta curve independently tunes this baseline.
- **b:** SNOO-style outer Nesterov with `mu=.9`.
- **c:** outer `mu=0`, pointwise matched to b by the code-true finite-T law.

The code-true law is

```text
eta_c[i] = eta_b[i] * (1 - .9^5) / (1 - .9) = eta_b[i] * 4.0951.
```

## 4. Frozen eta grids

Each grid has sqrt(2) spacing. With an even six-point grid, the disclosed v3
winner is the geometric midpoint between indices 2 and 3; the endpoints lie 2.5
sqrt(2)-steps below and above it.

| condition | six registered etas |
|---|---|
| a | `.25, .3535533906, .5, .7071067812, 1, 1.4142135624` |
| b | `.061, .0862670273, .122, .1725340546, .244, .3450681092` |
| c | `.2498011, .3532721035, .4996022, .7065442070, .9992044, 1.4130884141` |

No ladder may be extended, narrowed, shifted, or recentered after a v5 endpoint
exists.

## 5. Inputs and execution binding

The five deterministic no-wrap training orders were materialized before
registration, CPU-only, under the new root `/root/yeto-data/outer-mup-v5`. The
combined manifest exists byte-identically on both nodes:

```text
/root/yeto-data/outer-mup-v5/input-manifest.json
sha256 3b0461c1daa821cef7c67fa087613abb537c4c5ee3e0f1c1bacf2144875581fe
```

The frozen 1,024-row development evaluation sha256 is
`533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc`.
The 135M model files manifest sha256 is
`43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132`.

The manifest builder globally shuffles cells with RNG seed `20260726`, then
round-robins over interleaved node/GPU slots. Each node receives 45 cells. Both
nodes must execute at the pushed registration commit with clean worktrees.

## 6. Frozen analysis and G5

The frozen analyzer is `scripts/analyze_v5.py`, raw-file sha256:

```text
e574294d1d31ad9f8e7431168a877bd4fb58ac5c7c628d7e882eb8a89414a97e
```

For each condition, let `x=log2(eta)` and fit
`loss = A*x^2 + B*x + C` by ordinary least squares to the six five-seed mean
losses. An optimum is `INTERIOR` exactly when `A>0` and `-B/(2A)` is strictly
inside the numeric grid, with a `1e-12` coordinate tolerance. Tuned loss is the
fitted loss at that vertex.

The primary estimands are tuned loss `b-a` and tuned loss `c-a`. Confidence
intervals use 10,000 paired nonparametric seed-curve bootstrap draws, RNG seed
`20260726`. One common five-index draw is used across every eta and all three
conditions. A draw is invalid if any fitted optimum is not interior; at least
9,500 draws must remain valid.

G5 is evaluable only if all 90 cells have valid hash-bound evidence, all three
pooled optima are `INTERIOR`, and the bootstrap validity threshold passes. Its
closed scientific verdicts are:

- `SNOO_HELPS`: the `b-a` CI upper endpoint is below zero.
- `SNOO_NULL`: the `b-a` CI contains zero.
- `SNOO_HURTS`: the `b-a` CI lower endpoint is above zero.

If an evaluability condition fails, analysis status is `NOT_EVALUABLE` and no
out-of-vocabulary scientific verdict is invented. The successful note format is:

```text
G5 VERDICT: <SNOO_HELPS|SNOO_NULL|SNOO_HURTS> b-a=<point> [<low>,<high>] c-a=<point> [<low>,<high>]
```

The supervisor writes that line to `/private/tmp/h200-snoofix-note.md`.

## 7. Evidence, retry, wall clock, and rails

Every attempt is command-hash bound and validated with the existing v3/v4 work
evidence checks. One attempt-2 wave is preregistered only for enumerated
infrastructure failures, and retries the entire six-eta condition-by-seed curve.
Finite unfavorable loss, scientific divergence, edge optima, and any
outcome-dependent reason cannot authorize retry. If attempt 2 exists, it replaces
attempt 1 for every cell in its authorized group.

The stage has a 12-hour ceiling beginning at immutable launch-authority creation.
Normal termination targets recorded process groups. Wildcard `pkill` is forbidden;
if process-name matching is unavoidable, use bracketed patterns. No GCP work is
allowed. The registration commit and push originate from the Mac clone. Nodes do
not pull it until v4 drains. Prior result roots are read-only; v5 writes only to
new symlinks `/root/yeto-results-v5 -> /data/yeto-results-v5`.
