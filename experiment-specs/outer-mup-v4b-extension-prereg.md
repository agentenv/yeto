# Outer-muP v4b: preregistered downward grid extension

**Program ID:** `outer-mup-v4b-extension`
**Status:** **PREREGISTERED before any v4b GPU work or endpoint result.**
**Machine contract:** `experiment-specs/outer-mup-v4b-extension-prereg.json`

## 1. Trigger and complete outcome disclosure

G4 completed all 48 registered v4 cells, with no evidence errors, but returned:

```text
G4 VERDICT: NOT_EVALUABLE D5=NA D20=NA
```

The sealed readout is `h200-n1:/root/g4-readout.json`, SHA-256
`f2e70767cb06ecc4fcdd7942d4b517eea3c9927159e828f8d6f8ff31379dcded`.
The bound v4 launch manifest SHA-256 is
`150dab251f29ab191aca4bfa8297950f3f22167f5949d1c8795e467706d2fb1e`.

The four pooled v4 fits were:

| S | T | mu | v4 status | vertex in log2 eta | eta star |
|---:|---:|---:|---|---:|---:|
| 2560 | 5 | 0 | INTERIOR | -5.7656256564 | 0.0183811946 |
| 2560 | 5 | .9 | UNBRACKETED LOW | -8.0556114077 | NA |
| 10240 | 20 | 0 | UNBRACKETED LOW | -6.7501104621 | NA |
| 10240 | 20 | .9 | UNBRACKETED LOW | -10.2263682149 | NA |

This registration discloses those outcomes in full. They select exactly the three
LOW curves for a mechanical downward extension. No v4b process, endpoint loss,
or G4b calculation exists at registration, and the already-interior
`S=2560, mu=0` curve is not rerun.

## 2. Frozen extension

For each LOW curve, add exactly two etas: one and two sqrt(2)-steps below the v4
bottom. Equivalently, the new values are `v4_bottom/sqrt(2)` and
`v4_bottom/2`. Values below are shown in increasing order.

| curve | two new etas | complete combined grid used by G4b |
|---|---|---|
| S2560, T5, mu=.9 | .0019188620980, .0027136808034 | .0019188620980, .0027136808034, .0038377241961, .0054273616067, .0076754483921, .0108547232134 |
| S10240, T20, mu=0 | .0049120297592, .0069466591043 | .0049120297592, .0069466591043, .0098240595184, .0138933182085, .0196481190369, .0277866364171 |
| S10240, T20, mu=.9 | .0005516209420, .0007801098174 | .0005516209420, .0007801098174, .0011032418839, .0015602196348, .0022064837678, .0031204392696 |

The seed set is unchanged: `{501,503,509}`, with training seed formed by decimal
concatenation `seed||seed`. The extension therefore has exactly
`3 curves * 2 etas * 3 seeds = 18` new cells. There are six new eta values in
total. No rung may be extended, removed, shifted, narrowed, or recentered after
any v4b endpoint exists.

Every other scientific setting is inherited from v4: SmolLM2-1.7B revision
`effd688a12921b4cc83e3312b6feb579f70f9c71`, `M=4`, `H=512`, raw production
Nesterov without outer bias correction, AdamW inner LR `.001`, strict quorum,
barrier synchronization, version-matched anchors, fixed 512-step/65,536-token
windows, RDA merge, rho telemetry, and the same frozen 1,024-row development
evaluation. Inputs are bound through the hash-locked v4 launch manifest.

## 3. Frozen combined analysis and G4b

The frozen analyzer is `scripts/analyze_v4b.py`, raw-file SHA-256:

```text
9a6bd4110b55a5487501ab4b32eef205854400dd8735c83c88dd7580951cbab5
```

It loads all 48 validated v4 outcomes and all 18 validated v4b outcomes. For
each curve, let `x=log2(eta)` and fit `loss=a*x^2+b*x+c` by ordinary least
squares to the seed-mean loss at each eta. The three repaired curves use their
six-point combined v4+v4b grids. The unaffected `S2560, mu=0` curve uses its
original four-point v4 grid. A vertex is `INTERIOR` only when `a>0` and
`min(x)+1e-12 < -b/(2a) < max(x)-1e-12`.

The original definition remains:

```text
D_obs(T) = [eta_star(mu=.9,T) / eta_star(mu=0,T)] / (1-.9).
```

Confidence intervals use 10,000 paired nonparametric seed-curve bootstrap
draws, RNG seed `20260726`. One common three-index resample is used at every eta
and for all four refits. A draw is invalid if any vertex is not interior or a D
is nonpositive; at least 9,500 shared draws must remain. D5 and D20 receive
equal-tailed percentile intervals in log2 D, exponentiated to D scale. The
monotonicity statistic is `log2 D(T=5)-log2 D(T=20)` on those same draws.

G4b is evaluable only if all 66 combined cells have valid evidence, all four
pooled optima are interior, and the bootstrap validity threshold passes. It is
`PASS` only when all original G4 conditions hold:

- D(T=5) is in `[1.7,3.2]` inclusive.
- D(T=20) is in `[0.8,1.5]` inclusive.
- The paired 95% CI for `log2 D(T=5)-log2 D(T=20)` has lower endpoint above 0.

Complete evaluable evidence violating a condition is `FAIL`; failed evaluability
is `NOT_EVALUABLE`. The readout goes to `/root/g4b-readout.json`. The required
note line in `/private/tmp/h200-v4b-note.md` is:

```text
G4B VERDICT: <PASS|FAIL|NOT_EVALUABLE> D5=<point [low,high]|NA [NA,NA]> D20=<point [low,high]|NA [NA,NA]>
```

## 4. Fleet sharing with v5

The pre-registration live check found no tmux server, no v4/v5 slot controller,
no GPU compute process, and 0 MiB reported on every GPU on both nodes. The v5
result symlinks existed but had zero evidence cells and no launch artifacts.
Immediately before authority, `scripts/capture_v4b_v5_slot_snapshot.py` must
repeat and seal the tmux, slot-process, compute-process, memory, and utilization
state. Authority requires a passing snapshot.

The deterministic collision-free partition is:

- GPUs 0–5 on each node: one S=10240 v4b cell first, accounting for all 12 long
  cells. Six queues then run one S=2560 v4b cell.
- GPUs 6–7 on each node: begin their v5 queues immediately.
- For GPUs 0–5, each GPU begins its v5 queue only after that GPU's v4b
  controller exits and releases it.

Thus v5 starts immediately on four GPUs, no v5 process is killed or displaced,
and no GPU ever carries v4b and v5 scientific work simultaneously. The
approximately ten-minute v5 cell duration is a scheduling estimate only.

## 5. Evidence, retries, storage, and rails

The v4 evidence validator applies unchanged: exact command hashes and clean
source commit, immutable attempt records, complete learner and strict-quorum
steps, exact tape and telemetry, finite endpoint evaluation, and artifact
hashes. One attempt-2 wave is preregistered only for an enumerated infrastructure
reason and retries both new eta cells in a curve-by-seed group. Finite loss,
scientific divergence, an edge optimum, and outcome-dependent reasons cannot
authorize retry.

The 12-hour ceiling begins when immutable v4b launch authority is created.
V4b writes only to the new symlinks
`/root/yeto-results-v4b -> /data/yeto-results-v4b`. Prior v1/v2/v3/v4/v5 and
exploratory result trees are read-only; cross-node analysis mirrors, if needed,
are created only inside the new v4b tree.

Both preregistration artifacts and the frozen analyzer are committed and pushed
from the Mac before any manifest, authority, attempt directory, or GPU process
is created. Nodes only fast-forward pull the pushed commit and execute clean.
No GCP command or resource is allowed. Normal termination uses recorded process
groups; wildcard `pkill` is forbidden, and any unavoidable process-name match
uses the bracket idiom.
