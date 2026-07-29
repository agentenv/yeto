# v15: multi-seed 7B panel and 27B-LoRA extension

**Status:** `REGISTERED_PRE_OUTCOME`  
**Authority:** design and gatesim only; `NO_LAUNCH_AUTHORITY`

v15 repairs the largest limitation in G9: its retained 7B fits had only one
seed, so their bootstrap intervals collapsed to points. The primary panel uses
three new paired seeds for Qwen2.5-7B full tuning at `T={2,5,20}`, with no
momentum and raw `mu=.9` independently fit on six-rung ladders. A separately
labeled Qwen3.6-27B LoRA extension uses the same three seeds at `T={5,20}`.
The prospective total is 180 cells.

No result root, launch manifest, queue, controller, or GPU process is authorized
by this registration.

## Measured common-mode law and two different bands

G9's signed scale errors were approximately `-0.40` bits at 1.7B and `-0.50`
bits at 7B. v15 freezes the two-point line

```text
c(P) = -0.50 + [-0.10/log2(7/1.7)] * log2(P_B/7).
```

It gives `c(27B)=-0.595382` bits. This is an empirical calibration, and its
27B-LoRA use is an explicitly risky extrapolation—not a universal scaling law.

After removing common mode, G9's 7B raw-minus-no-momentum residual was
`-0.051907` bits. The absolute half-width is `0.21` bits, rounded outward from
four times that residual (`0.207628`). The arm-relative half-width is `0.16`
bits, rounded outward from three times it (`0.155721`). The bands are registered
separately because they answer different questions.

## Independent verdicts

The 7B absolute gate passes when at least five of six curve optima are inside
the drift-corrected `+-0.21`-bit bands. The 7B arm-relative gate passes when at
least two of three within-horizon raw/no-momentum ratios are inside `+-0.16`
bits. Each has its own `PASS / FAIL / NOT_EVALUABLE` vocabulary and its own
bootstrap-validity calculation.

Most importantly, the relative verdict is always computed and published even
if the absolute gate fails or is not evaluable. There is no omnibus rule that
turns absolute failure into relative failure. The 27B-LoRA absolute and
relative labels are likewise separate secondary outcomes and cannot rescue or
invalidate the primary 7B claims.

## Placement and claim boundary

The 7B T=5 centers (`0.007827013` for no momentum and `0.002071630` for raw
momentum) are measured G9 calibration points, now repeated with new seeds and a
wider ladder. T=2 and T=20 centers transport the frozen no-momentum slope and
the selected F3 arm-ratio shape. v15 therefore tests fresh-seed replication and
new-horizon transport after measured calibration; it is not described as an
untouched zero-shot 7B prediction.

The 27B LoRA centers start from the registered v7 `0.28` T=5 placement and its
frozen T=20 and raw-arm ratios. Absolute predictions then apply the
`-0.595382`-bit common correction. Every exact rung is enumerated in the
hash-bound gatesim report.

## Analysis and gatesim

Each curve is an OLS quadratic in `log2(eta)` through all six three-seed mean
losses. A positive-curvature vertex may be at most 0.25 bits beyond the ladder;
it is never clipped. Ten thousand shared paired-seed bootstrap refits require
at least 8,000 valid refits. Missing evidence affects only a gate that actually
requires the missing curve.

The 20,000-replicate gatesim places a common random term ahead of arm
subtraction, adds measured G4C/v7 residual noise, and stresses the LoRA-specific
term by 1.5x. It gives:

- `P_evaluable=1.000`;
- 7B absolute pass probability `0.99810`;
- 7B arm-relative pass probability `0.99900`;
- 27B-LoRA absolute pass probability `0.98925`;
- 27B-LoRA arm-relative pass probability `0.99315`.

These are power calculations under the registered calibrated alternative, not
scientific evidence that either scale arm transports.

Machine contract:
`experiment-specs/outer-mup-v15-multiseed-scale-panel-prereg.json`.
Gatesim report:
`experiment-specs/outer-mup-v15-multiseed-scale-panel-gatesim.json`.
