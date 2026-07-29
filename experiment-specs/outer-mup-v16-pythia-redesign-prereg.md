# v16: second-family redesign

**Status:** `REGISTERED_PRE_OUTCOME`  
**Authority:** design and gatesim only; `NO_LAUNCH_AUTHORITY`

v16 is a new prospective Pythia-160M/UltraChat program. It does not amend or
rescue G13 or G13B. That distinction matters because v13b's contract prohibited
another in-program regrid after its outcome.

The binding trigger is the complete v13b readout with SHA-256
`868d8aaa3c422bdb4475302d2071ca0082108e813aab70c5f9061ef90f156f48`:
72/72 cells completed, three of six curves were accepted, and zero of 10,000
shared bootstrap draws retained all six fits.

## Six-rung direction-aware grids

Each curve follows one mechanical rule:

- accepted v13b curve: center on its fitted `eta_star`;
- lower-rate bracket failure: center at `failed_low_edge/sqrt(2)`;
- higher-rate bracket failure: center at `failed_high_edge*sqrt(2)`.

The six offsets are `[-2.5,-1.5,-0.5,+0.5,+1.5,+2.5]` bits. Thus a failed
edge is an interior rung and three rungs extend in the observed failure
direction. For T20/no-momentum, the v13b fit had negative curvature; its vertex
is deliberately not used as truth.

| curve | center | exact six-rung range |
|---|---:|---:|
| T2, mu=0 | 0.007959003015874582 | 0.0014069662510022751 ... 0.045022920032072804 |
| T2, mu=.9 | 0.0034223165582977205 | 0.0006049858114348312 ... 0.019359545965914597 |
| T5, mu=0 | 0.0038377496772455835 | 0.0006784247053192091 ... 0.021709590570214692 |
| T5, mu=.9 | 0.0010529756052147737 | 0.0001861415477178439 ... 0.0059565295269710045 |
| T20, mu=0 | 0.2480668464259681 | 0.04385243732384098 ... 1.4032779943629115 |
| T20, mu=.9 | 0.0004249680901238146 | 0.0000751244545786113 ... 0.002403982546515562 |

All 36 exact decimals are enumerated in the machine contract. A large outer
rate is not a license to clip or remove a rung after registration.

## Observed family noise changes the estimator

v13b contained finite NLLs of `135.487805` and `48.135418`, corresponding to
one-sided residuals of `129.757010` and `39.377260` above their rung medians.
Those values are scientific outcomes, not infrastructure failures. A
conventional three-seed mean analysis remains badly underpowered under that
observed profile.

v16 therefore registers 17 fresh seeds and uses the rung-wise median of all 17
finite losses before fitting each six-point quadratic. No finite value is
dropped, winsorized, retried, or replaced. A 10,000-draw shared paired-seed
bootstrap recomputes all rung medians and fits; at least 7,000 draws must retain
all six accepted curves. The complete design is `3*2*6*17=612` cells.

G16's closed vocabulary is
`SECOND_FAMILY_MONOTONE / SECOND_FAMILY_NONMONOTONE /
SECOND_FAMILY_NOT_EVALUABLE`. A monotone verdict still requires
`D(2)>D(5)>D(20)` and both adjacent 95% interval lower endpoints above zero.
An evaluability failure cannot authorize a v16 regrid.

## Gatesim

The 20,000-replicate gatesim includes curve-specific robust core noise and the
two observed one-in-three, one-sided divergence spikes with shared seed-profile
pairing. It gives `P_evaluable=0.82470`, just above the mandatory 0.80 bar; the
limiting T20/no-momentum curve has `P_accepted=0.92250`. Conditional on an
evaluable draw under the declared monotone placement alternative,
`P_monotone=1.000`.

The simulated center placement is only a power device. In particular, its
extreme T20 ratio is not evidence that the Pythia family follows the proposed
law.

Machine contract:
`experiment-specs/outer-mup-v16-pythia-redesign-prereg.json`.
Gatesim report:
`experiment-specs/outer-mup-v16-pythia-redesign-gatesim.json`.
