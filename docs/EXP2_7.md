# EXP2.7: Group-Local Feature Grid and Dense-Capture Prep

## Question

Can richer group-local features and stricter abstain-style policy grids improve over EXP2.6 on the existing offline artifacts, and are the tools ready for denser captures?

Short answer: the tools are ready, but the existing artifacts still do not pass. The current score family remains too weak for online ProbeCommit.

## Inputs

EXP2.7 local smoke used the existing EXP2.5/EXP2.6 artifacts:

```text
experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_53/calibrated_test.jsonl
experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_67/calibrated_test.jsonl
experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_79/calibrated_test.jsonl

experiment-results/EXP2/probecommit-offline/seed53/policy_replay.jsonl
experiment-results/EXP2/probecommit-offline/seed67/policy_replay.jsonl
experiment-results/EXP2/probecommit-offline/seed79/policy_replay.jsonl
```

No new model probes, online training, or AWS jobs were started for this local scaffold run.

## New Tooling

Added:

```text
scripts/build_group_local_features.py
scripts/replay_group_local_policy_grid.py
scripts/aggregate_group_local_results.py
```

The pipeline is now split into three stages:

1. Build reusable group-level feature rows from candidate features and exact policy replay.
2. Tune and evaluate a larger group-local policy grid using held-out seeds.
3. Aggregate feature and grid summaries.

This makes denser captures easier to process because expensive model replay remains separate from cheap policy-grid analysis.

## Feature Builder

The feature builder writes one row per complete `(seed, step, fragment)` group. Each row includes:

- action metrics from exact replay,
- per-score group statistics,
- top/bottom candidate diagnostics,
- score spread, IQR, entropy, top gaps, and z-normalized top gaps,
- pairwise score/utility concordance,
- agreement between score fields,
- token-weighted and oracle-positive reference utilities.

Local output:

```text
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.json
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.md
```

Summary:

| Metric | Value |
|---|---:|
| Complete groups | 614 |
| Groups per seed | 195 / 208 / 211 |
| Token-weighted negative rate | 0.436 |
| Token-weighted strict-negative rate | 0.156 |
| Oracle-positive headroom mean | 0.001113 |

Score diagnostics:

| Score | Candidate good AUROC | Pairwise concordance | Top1 bad | Bottom1 bad | Linear drop25 gain |
|---|---:|---:|---:|---:|---:|
| `probe_grad_dot` | 0.582 | 0.511 | 0.446 | 0.480 | 0.000004 |
| `probe_grad_cosine` | 0.577 | 0.512 | 0.441 | 0.474 | 0.000006 |
| `calibrated_score` | 0.627 | 0.517 | 0.436 | 0.485 | 0.000008 |
| `consensus_cosine` | 0.505 | 0.509 | 0.474 | 0.498 | 0.000002 |
| `freshness` | 0.511 | n/a | 0.469 | 0.477 | 0.000008 |
| `combined_score` | 0.507 | 0.496 | 0.476 | 0.485 | 0.000000 |

The core failure is unchanged: candidate-level AUROC is not enough, and within-group pairwise concordance remains close to chance.

## Policy Grid

The policy grid evaluates train-seed-tuned rules over the reusable group features. Deployable actions exclude oracle and random policies:

```text
token_weighted
freshness_weighted
anchor_drop_bottom25
anchor_positive_threshold
anchor_shrink
probecommit_v1
```

Oracle and random policies are retained only as references.

Policy families include:

- base deployable actions,
- drop bottom 25% if score spread is high,
- drop bottom 25% if score IQR is high,
- drop bottom 25% if top-gap z-score is high,
- drop bottom 25% if entropy is low,
- positive-threshold action if spread and top-gap are high,
- shrink if score mean is low,
- drop-or-shrink combinations,
- agreement-gated drop rules across score fields.

Local output:

```text
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.json
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.md
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid_splits.jsonl
```

Held-out seed result:

| Metric | Value |
|---|---:|
| Mean held-out gain vs token-weighted | 0.000062 |
| All held-out seeds positive gain | true |
| Negative-rate relative drop | 0.037 |
| Strict-negative relative drop | 0.064 |
| Oracle-positive headroom captured | -0.276 |
| Selected mass | 0.925 |
| Act rate | 0.483 |
| Gate pass | false |

Per split:

| Test seed | Rule | Gain | Negative drop | Strict drop | Headroom captured | Act rate |
|---:|---|---:|---:|---:|---:|---:|
| 53 | `probe_grad_dot:drop25_gapz:0.510476` | 0.000106 | 0.053 | 0.031 | 0.222 | 0.482 |
| 67 | `probe_grad_dot:drop25_spread_gapz:0.00293385:0.417615` | 0.000010 | 0.025 | 0.161 | -1.223 | 0.591 |
| 79 | `probe_grad_cosine:drop25_iqr:0.00728958` | 0.000070 | 0.032 | 0.000 | 0.173 | 0.374 |

Interpretation:

- The larger policy grid is more conservative than EXP2.6, with act rate `0.483`.
- It still captures negative oracle headroom on average.
- The utility gain is smaller than EXP2.6 (`0.000062` vs `0.000114`).
- Negative-rate reduction is far below the online-readiness target.
- Seed 67 remains the warning case: positive utility gain but negative oracle-headroom capture.

## Decision

Do not start online ProbeCommit.

EXP2.7 local scaffold passes as tooling, but not as method evidence. The group-local feature grid confirms the earlier diagnosis:

1. Bad merge headroom is real.
2. Candidate-level signal exists.
3. Existing score fields do not produce strong within-group ranking.
4. More policy wrapping does not fix weak group-local evidence.

The next useful work is denser capture and stronger score evidence, not online training.

## Dense-Capture Gate

Before any online run, the next capture stage should meet:

```text
complete groups per seed >= 500
preferably complete groups per seed >= 1000
within-group pairwise concordance >= 0.58
top-score bad rate clearly below bottom-score bad rate
policy captures >= 0.40 oracle-positive headroom
negative-rate drop >= 0.20
strict-negative rate decreases
all held-out seeds positive gain
selected mass >= 0.40
random-count controls do not explain the gain
```

The current local artifacts fail the group-count, within-group concordance, headroom-capture, and negative-rate-drop gates.

## Exact Commands

Build group-local features:

```bash
mkdir -p experiment-results/EXP2/probecommit-offline/exp2_7_local

python3 scripts/build_group_local_features.py \
  --features \
    experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_53/calibrated_test.jsonl \
    experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_67/calibrated_test.jsonl \
    experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_79/calibrated_test.jsonl \
  --policy-replay \
    experiment-results/EXP2/probecommit-offline/seed53/policy_replay.jsonl \
    experiment-results/EXP2/probecommit-offline/seed67/policy_replay.jsonl \
    experiment-results/EXP2/probecommit-offline/seed79/policy_replay.jsonl \
  --out-jsonl experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl \
  --out-summary experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.md
```

Run policy grid:

```bash
python3 scripts/replay_group_local_policy_grid.py \
  --features experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl \
  --out-json experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.md \
  --out-splits experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid_splits.jsonl
```

Aggregate:

```bash
python3 scripts/aggregate_group_local_results.py \
  --feature-summary experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.json \
  --policy-grid experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.json \
  --out-json experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_aggregate.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_aggregate.md
```

## Artifact Paths

```text
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.json
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features_summary.md
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.json
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid.md
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_policy_grid_splits.jsonl
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_aggregate.json
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_aggregate.md
```

## Validation

```text
/tmp/yeto-runtime/bin/python -m py_compile scripts/build_group_local_features.py scripts/replay_group_local_policy_grid.py scripts/aggregate_group_local_results.py scripts/replay_group_local_probecommit.py
/tmp/yeto-runtime/bin/python -m pytest tests/test_smoke_scripts.py
```

Result:

```text
15 passed
```
