# v14: exact-rate transfer matrix

**Status:** `REGISTERED_PRE_OUTCOME`  
**Authority:** design and gatesim only; `NO_LAUNCH_AUTHORITY`

v14 extends G10 from three directions to the complete directed matrix over
`T={2,5,20,40}` in two schedule geometries. The first holds `H=512` and varies
`S=HT`; the second holds `S=2560` and varies `H=S/T`. Every one of the 24
directed context/pairs receives five fresh paired seeds. At each target and seed,
one fresh exact-target-rate comparator is shared by the three incoming transfer
cells. Thus the prospective design is

```text
(12 transfers + 4 target comparators) * 2 contexts * 5 seeds = 160 cells.
```

No v14 result root, attempt directory, launch manifest, controller, or node
command is authorized by this registration.

## Exact prescriptions

| context | T=2 | T=5 | T=20 | T=40 |
|---|---:|---:|---:|---:|
| fixed H=512 | 0.007821882581822885 | 0.003191644884294105 | 0.0008223020084526104 | 0.0003971228256207733 |
| fixed S=2560 | 0.007341476969217584 | 0.003191644884294105 | 0.0009050641916564297 | 0.00045856076934387404 |

The T=5 and T=20 fixed-H values are direct five-seed G4C raw-Nesterov fits.
T=2 is a disclosed power-law extrapolation through those two fits. T=40 is the
independently registered v11 far-horizon prescription; v11 was deferred, so it
is not represented as a measured optimum. Fixed-S values transport the fixed-H
rates with frozen G6/G9 `S`, `H`, and raw-`D` placement coefficients. Those
qualifications are part of the claim: an exact decimal is frozen, but a frozen
prediction is not silently relabeled as a measurement.

Every source decimal is deployed verbatim at the target. Snapping to a grid,
refitting from v14 outcomes, or substituting a nearby target rate is forbidden.

## Pair decisions

The raw paired endpoint penalty is transfer minus fresh target comparator in
bits per token. G10 contained one finite `+9.504814`-bit scientific-divergence
value. v14 therefore registers a bounded-influence primary estimand: each seed
penalty is symmetrically clipped to `[-0.75,+0.75]` for inference, while every
unclipped value, the raw mean, and the raw SD remain mandatory outputs. The
primary mean and two-sided 95% Student interval use the five fresh paired seeds.

Every pair has exactly one label:

- `PAIR_PENALTY` when the interval is wholly above `+0.10` bits/token;
- `PAIR_BENEFIT` when it is wholly below `-0.10` bits/token;
- `PAIR_NO_DECISIVE_PENALTY` otherwise.

The last label is deliberately not called equivalence. Missing or invalid work
emits no scientific label and must be recovered, if eligible, under the
loss-blind whole-target retry unit.

## Preregistered asymmetry

G10 motivates a directional hypothesis, not an all-pairs penalty claim:
short-source-to-long-target transfer is penalized, while downward transfer is
mild. Within each seed, v14 forms equal-weight upward and downward means over
both contexts, then takes their paired difference. The closed vocabulary is
`ASYMMETRY_CONFIRMED / ASYMMETRY_NULL / ASYMMETRY_REVERSED`.

`ASYMMETRY_CONFIRMED` requires all three conditions: the lower 95% endpoint for
upward-minus-downward exceeds `0.10`, the upward lower endpoint exceeds `0.10`,
and the downward point estimate lies inside `[-0.10,+0.10]`. Reversal requires
the difference interval to lie below `-0.10`; all other complete outcomes are
null.

## Gatesim

The deterministic 20,000-replicate CPU gatesim uses the bounded G10 seed
profiles, including the bounded contribution of its extreme seed, and a stress
profile at G4C's maximum measured paired-contrast SD (`0.163605` bits). It
simulates all 24 pairs individually. `P_evaluable=1.000`; the minimum expected
label probability is `0.86795` across upward pairs and `0.99985` across
downward pairs. Under the declared G10-derived alternative,
`P(ASYMMETRY_CONFIRMED)=0.91335`.

The gatesim validates bracketing-free decision mechanics and power under that
alternative. It is not evidence that any v14 transfer penalty exists.

Machine contract:
`experiment-specs/outer-mup-v14-transfer-matrix-prereg.json`.
Gatesim report:
`experiment-specs/outer-mup-v14-transfer-matrix-gatesim.json`.
Reproduce all reports with:

```bash
python3 experiment-specs/v2pack-gatesim.py --verify
```
