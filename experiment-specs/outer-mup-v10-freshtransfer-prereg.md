# Outer-muP V10 fresh-transfer preregistration

Status: **REGISTERED BEFORE ANY V10 CELL OR RESULT ROOT**.

Program: `outer-mup-v10-freshtransfer`. The machine-readable companion is
`experiment-specs/outer-mup-v10-freshtransfer-prereg.json`, raw SHA-256
`b99486f91ea6264d8877448aa6eb1308b6b9e81d288c9db4e0e9b2c394e81c68`.

## Question and estimand

The current paper table does not run a source-selected learning rate at its
exact value. It snaps that rate to the nearest already measured target-grid
cell, compares against another reused target-grid cell, and evaluates on the
development stream used by the source fits. V10 removes all three shortcuts.

At SmolLM2-1.7B, raw Nesterov momentum `mu=0.9`, `M=4`, and fixed `H=512`,
V10 deploys each source prescription verbatim on three new training seeds and
compares it against a fresh same-seed run at the target-specific fitted
prescription. The endpoint is the reserved confirmation-audit shard. For each
directed pair and seed,

```text
penalty_bits = [NLL_exact_source_eta - NLL_target_eta] / ln(2).
```

Positive values mean that the transferred source rate is worse. No fitted
loss, nearest rung, interpolation, or old endpoint enters this difference.

## Frozen rates and 18 cells

The T=5 and T=20 rates are the exact interior five-seed G4C quadratic
vertices, not rounded displays:

```text
eta*(T=5)  = 0.003191644884294105
eta*(T=20) = 0.0008223020084526104
```

Their source is `h200-n1:/root/g4c-readout.json`, SHA-256
`16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa`.

T=40 had no 1.7B outcome when this registration was frozen. Its comparator is
therefore the independently preregistered, target-specific fitted prescription
already used by V11:

```text
eta*(T=40) = eta0_fit(T=40) * 0.1 * D_fit(T=40)
            = 0.003800560784723474 * 0.1 * 1.044905865516022
            = 0.0003971228256207733.
```

That rule is sealed at commit
`c310dc69f471c60f208995602c144a34e57f86da`, contract SHA-256
`5ea6c31c877be2ec73e432b6b143a53dcd904fbd5dd5db678209d98f460dff81`.
It is a pre-outcome surface prescription, not a claim that T=40 was already
observed.

| directed deployment | target T | S | exact transferred eta | exact fresh comparator eta |
|---|---:|---:|---:|---:|
| T5 -> T20 | 20 | 10,240 | `0.003191644884294105` | `0.0008223020084526104` |
| T5 -> T40 | 40 | 20,480 | `0.003191644884294105` | `0.0003971228256207733` |
| T20 -> T5 | 5 | 2,560 | `0.0008223020084526104` | `0.003191644884294105` |

Every row has both configurations at fresh seeds `{941,947,953}`. Thus the
design has six configurations and 18 cells. Training RNG seeds are the usual
decimal concatenations `{941941,947947,953953}`. All remaining settings match
the v4/v9 1.7B full-parameter path: sequence length 128, AdamW inner LR .001,
RDA merge, strict quorum, barrier synchronization, version-matched anchors,
four learners packed on one H200, no outer bias correction, and no injected
delay.

## Truly held-out evaluation stream

The command's prebound endpoint is

```text
/root/yeto-data/outer-mup-v3/scale-s2560/raw/confirmation-audit.jsonl
bytes   = 4,774,107
sha256  = d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b
rows    = 1,024 evaluated rows
```

This is the reserved `confirmation-audit` shard carried beside the v6/v9-style
train/development split. V4/V4C hash-bound it as input provenance but did not
pass it to the source-rate commands, and neither the G4C eta fits nor the
nearest-grid transfer table read it. The training JSONL remains
`raw/train.jsonl` at SHA-256
`e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf`.

## G10 closed gate

G10 has exactly three scientific verdicts:

```text
PENALTY_CONFIRMED
PENALTY_NULL
PENALTY_REVERSED
```

`NOT_EVALUABLE` is not a verdict. Infrastructure-incomplete work produces no
G10 readout and must first be recovered under the loss-blind registered retry
rule. Every complete finite 18-cell outcome maps to one of the three words.

For each directed pair, the analyzer enumerates all `3^3=27` ordered paired
fresh-seed resamples and reports an equal-tailed 95% interval for mean penalty
in bits/token. The frozen practical threshold is

```text
tau = 0.17854217097363556 bits/token.
```

It is derived before outcomes as one half of the smallest of the three
surface-predicted penalties:

| pair | target curvature donor | eta mismatch (log2 bits) | predicted penalty (bits/token) |
|---|---:|---:|---:|
| T5 -> T20 | T20 | +1.9565598825 | 0.6511960553 |
| T5 -> T40 | T20 | +3.0066429494 | 1.5377622175 |
| T20 -> T5 | T5 | -1.9565598825 | 0.3570843419 |

The T40 placement calculation prospectively uses the nearest registered 1.7B
long-horizon curvature donor, G4C/T20. This power calculation chooses only the
gate band; observed penalties always come directly from fresh held-out runs.

Classification is deterministic:

- `PENALTY_CONFIRMED` iff every directed 95% interval has lower endpoint at
  least `+tau`.
- `PENALTY_REVERSED` iff every directed 95% interval has upper endpoint at
  most `-tau`.
- `PENALTY_NULL` for every other complete finite outcome, including mixed
  directions or intervals entering the practical-null band.

The equal-weight average over the three directions and its paired interval are
reported descriptively but do not change the gate.

## Noise-calibrated band and GATESIM

The band width is checked against banked 1.7B seed noise, not guessed from the
new outcomes. Across every pair of raw eta rungs within each five-seed G4C T5
and T20 curve (30 paired contrasts), the pooled paired-contrast SD is
`0.0681770752` bits, the median is `0.0331643351`, and the maximum is
`0.1636054316`. Thus `tau` is 2.619 pooled SDs, 1.091 times even the worst
observed contrast SD, and still only half the smallest predicted effect.

The frozen measured-noise simulation is
`experiment-specs/outer-mup-v10-freshtransfer-gatesim.json`, SHA-256
`17f5181675eabba78f5697c7c5114afcecaa14cf581d0851ae54a6f0ed6ea131`.
It resamples shared banked seed profiles at the closest available eta
separation, conservatively donates T20 profiles to T40, and applies the exact
27-draw analyzer to 20,000 synthetic datasets. Before any V10 result existed:

```text
V10 GATESIM P_eval=1.000000
mandatory P_eval bar=0.800000
P(PENALTY_CONFIRMED | registered surface prediction)=1.000000
```

No bracket fit exists in V10, so a complete finite synthetic dataset is always
classifiable; the simulation still exercises the exact paired intervals and
closed decision rule.

The two-node zero-outcome snapshot is preserved as
`experiment-specs/outer-mup-v10-preseal-proof.json`: at
`2026-07-28T02:08:00Z`, both `/root/yeto-results-v10` and
`/data/yeto-results-v10` were absent on both nodes and the bracketed V10
priority query returned no process.

## Frozen analysis, execution, and retries

Frozen analyzer: `scripts/analyze_v10.py`, raw SHA-256
`a5f326275c3462cb1663e6d0084b7248d827ee413d524c951b51d921463aff76`.
Its only local analysis dependency is `scripts/v10_common.py`, raw SHA-256
`86749042c13055f4419b647909677f1da8fa741883d4f723bf80850d5c70acbd`.
Outcome-aware edits, eta substitutions, grid snapping, seed substitutions, or
threshold changes are forbidden.

The registration commit must be pushed to
`origin/experiment/outer-mup-v10-freshtransfer` before launch. Nodes use an
isolated exact-commit checkout at `/root/yeto-v10`, leaving the running v9
checkout untouched. Results use the LVM link
`/root/yeto-results-v10 -> /data/yeto-results-v10`. Every process inherits
`HF_DATASETS_CACHE=/data/hf-datasets-cache` and `TMPDIR=/data/tmp`.

`scripts/run_slot_v10.py` registers as tonight85 `PRIORITY`. A slot may launch
only after the containing `run_slot_v9.py` four-GPU island controller is gone
and `nvidia-smi` shows no compute process on the exact GPU. It never stops,
changes, or overlaps a v9 island. Longest work is assigned first across the 16
slots.

At most one loss-blind retry is allowed for host/GPU, framework/driver,
storage/network, or registered timeout failure. Its unit is the transfer and
comparator pair sharing a target horizon and training seed. Finite unfavorable
loss, scientific divergence, or any outcome-aware reason cannot authorize a
retry. The fold-ready output schema is
`yeto_outer_mup_v10_g10_readout_v1`; the final note line begins exactly
`G10 VERDICT:`.
