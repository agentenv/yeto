# Outer-muP v4c: preregistered seed-power lane

**Program ID:** `outer-mup-v4c-seedpower`

**Status:** **PREREGISTERED before any v4c result root, manifest, attempt, GPU work, endpoint loss, or G4C calculation.**

**Machine contract:** `experiment-specs/outer-mup-v4c-seedpower-prereg.json`

## 1. Trigger and complete G4B disclosure

G4B completed all 66 v4+v4b cells with no evidence errors. Its sealed readout
is `h200-n1:/root/g4b-readout.json`, SHA-256
`d58a05c46396d94786c6bcdcffa4f9c72abcc47036652a5260ee87256446be97`,
and reports:

```text
G4B VERDICT: NOT_EVALUABLE D5=1.715783 [1.489964,1.784977] D20=1.204749 [1.148001,1.273801]
```

All four combined-grid quadratic point optima are `INTERIOR`:

| S | T | mu | points | vertex log2 eta | eta star |
|---:|---:|---:|---:|---:|---:|
| 2,560 | 5 | 0 | 4 | -5.7656256564 | .0183811946 |
| 2,560 | 5 | .9 | 6 | -8.3086868447 | .0031538137 |
| 10,240 | 20 | 0 | 6 | -7.2521338700 | .0065597936 |
| 10,240 | 20 | .9 | 6 | -10.3053294665 | .0007902904 |

The D5 point is inside the registered `[1.7,3.2]` band and D20 is inside the
registered `[.8,1.5]` band. The paired monotonicity interval over valid draws
is positive (`[.3191842810,.5210186743]`). G4B is nevertheless not evaluable
because only 7,067 of 10,000 shared bootstrap draws kept all four refits
interior, below the registered 9,500 threshold; 2,933 draws were unbracketed.

That single failure selects the prospective seed-power repair below. It does
not select an eta or any v4c outcome. No eta is added, removed, shifted,
narrowed, or recentered.

## 2. Frozen 44-cell completion

Add training seeds `{541,547}` to every eta of all four existing combined
grids. The training RNG seed is the decimal concatenation `seed||seed`, exactly
as in v4/v4b. The existing `{501,503,509}` outcomes remain read-only.

| curve | complete eta grid receiving both added seeds |
|---|---|
| S=2,560, T=5, mu=0 | .0179837122590, .0254328097784, .0359674245179, .0508656195569 |
| S=2,560, T=5, mu=.9 | .0019188620980, .0027136808034, .0038377241961, .0054273616067, .0076754483921, .0108547232134 |
| S=10,240, T=20, mu=0 | .0049120297592, .0069466591043, .0098240595184, .0138933182085, .0196481190369, .0277866364171 |
| S=10,240, T=20, mu=.9 | .0005516209420, .0007801098174, .0011032418839, .0015602196348, .0022064837678, .0031204392696 |

This is exactly `22 etas * 2 seeds = 44` new cells: 24 long `S=10,240`
cells and 20 short `S=2,560` cells. After registration, no scientific
coordinate or analysis rule may change.

Every other setting is inherited from v4/v4b: SmolLM2-1.7B revision
`effd688a12921b4cc83e3312b6feb579f70f9c71`, `M=4`, `H=512`, sequence
length 128, full tuning, AdamW inner LR `.001`, raw production Nesterov without
outer bias correction, RDA merge, strict quorum, barrier synchronization,
version-matched anchors, fixed 512-step/65,536-token windows, rho telemetry,
and the frozen 1,024-row development endpoint. The complete hash-bound v4
materialized-input ledger is reused.

## 3. Frozen five-seed analysis and G4C

The frozen analyzer is `scripts/analyze_v4c.py`, raw SHA-256:

```text
b8e5470d2b512f487948413104a1783fe45adbbe68871f61873d3a9bae73cf27
```

It uses `scripts/analyze_v4b.py` only for evidence loading, quadratic
arithmetic, quantiles, hashing, and atomic output; that dependency remains
sealed at SHA-256
`9a6bd4110b55a5487501ab4b32eef205854400dd8735c83c88dd7580951cbab5`.

The analyzer loads all 48 v4, 18 v4b, and 44 v4c cells. At each eta it takes
the mean over the five fixed seeds, then fits
`loss = a*log2(eta)^2 + b*log2(eta) + c` by ordinary least squares. A fit is
interior only if `a>0` and its vertex lies strictly between the registered grid
endpoints with a `1e-12` margin. The estimands remain:

```text
D_obs(T) = [eta_star(mu=.9,T) / eta_star(mu=0,T)] / (1-.9).
```

The bootstrap has 10,000 replicates and RNG seed `20260726`. Each replicate
draws five indices with replacement and applies that one common draw to every
eta and all four curve refits. A replicate is invalid if any optimum is not
interior or either D is nonpositive. The bootstrap-validity requirement remains
exactly 95%, hence at least 9,500 of 10,000 shared refits. D intervals are
equal-tailed 95% percentiles in log2 D, reported after exponentiation. The
monotonicity coordinate is `log2 D(T=5)-log2 D(T=20)` over those same draws.

G4C is evaluable only if all 110 combined cells have valid hash-bound evidence,
all four five-seed optima are interior, and at least 9,500 bootstrap draws are
valid. It is `PASS` only if every original G4B scientific condition holds:

- D5 is in `[1.7,3.2]` inclusive.
- D20 is in `[.8,1.5]` inclusive.
- The paired 95% CI for `log2 D5-log2 D20` has lower endpoint greater than 0.

Complete evaluable evidence violating any condition is `FAIL`; a failed
evaluability requirement is `NOT_EVALUABLE`. The readout is
`/root/g4c-readout.json`, and `/private/tmp/h200-v4c-note.md` receives exactly:

```text
G4C VERDICT: <PASS|FAIL|NOT_EVALUABLE> D5=<point [low,high]|NA [NA,NA]> D20=<point [low,high]|NA [NA,NA]>
```

## 4. Longest-first fleet sharing

At registration, v4b is complete and GPUs 0–5 are idle on both nodes. V5B
exclusively owns GPUs 6–7 on both nodes; v6 is sealed and waiting for the
fleet. V4C uses an isolated `/root/yeto-v4c` exact-commit checkout and never
changes or stops either lane.

After fresh node proofs, V4C initially claims GPUs 0–5 on both nodes. Every one
of those twelve queues starts with an `S=10,240` cell. Eight queues carry two
long cells. Four queues carry one long cell followed by four short cells. GPUs
6–7 receive no V4C controller or compute process until every V5B controller and
compute process has drained; fresh proofs then start four deferred queues, each
with one long cell followed by one short cell. This schedules the 24 long cells
first within every queue, balances 22 cells per node, and accounts for the V5B
handoff while targeting about eight hours of queue work.

When the initial slots are claimed, the lane appends a claim line to
`/private/tmp/h200-factorial-note.md`. Only after all 44 cells, validation, and
G4C analysis are complete does it append the exact marker `V4C DONE`, releasing
the fleet to v6.

## 5. Evidence, retries, storage, and rails

V4 validation is inherited unchanged: exact clean source commit and command
hash; immutable attempt start/end records; four complete learners; exact
strict-quorum outer steps, tape, and rho telemetry; a finite 1,024-row endpoint;
and artifact hashes. Partial work never counts and scientific divergence is
retained.

At most one loss-blind retry is allowed. Its unit is the entire registered eta
grid for one `(T,mu,added-seed)` curve—four cells for `S=2,560,mu=0`, six cells
otherwise. Only enumerated infrastructure or loss-blind validator failures may
authorize it. Finite unfavorable loss, edge optima, scientific divergence,
bootstrap validity, and any outcome-dependent reason never authorize a retry.

The immutable launch authority starts a 12-hour ceiling. At the deadline,
controllers terminate only their recorded V4C process groups and seal unrun
cells. Results must use a new symlink
`/root/yeto-results-v4c -> /data/yeto-results-v4c` on each node, with at least
1 TB free at preflight. Every earlier result tree remains read-only. No GCP or
AWS resources or commands are permitted, and no wildcard process killing is
used.
