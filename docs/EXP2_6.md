# EXP2.6: Group-Local ProbeCommit Offline Replay

## Question

Can group-local confidence rules decide when to use anchor-gradient drop/shrink actions, using only train-seed tuning and no oracle labels at decision time?

EXP2.5 showed the main failure mode: the anchor-gradient signal has useful global tails, but it is weak at ranking candidates inside the same `(step, fragment)` merge group. EXP2.6 tests that failure mode directly.

## Inputs

EXP2.6 reuses the EXP2.5 offline artifacts and does not run model probes or online training.

Candidate feature inputs:

```text
experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_53/calibrated_test.jsonl
experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_67/calibrated_test.jsonl
experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_79/calibrated_test.jsonl
```

Exact policy replay inputs:

```text
experiment-results/EXP2/probecommit-offline/seed53/policy_replay.jsonl
experiment-results/EXP2/probecommit-offline/seed67/policy_replay.jsonl
experiment-results/EXP2/probecommit-offline/seed79/policy_replay.jsonl
```

Output root:

```text
experiment-results/EXP2/probecommit-offline/group-local/
```

## Method

The script joins per-candidate features with exact per-group policy replay rows.

New script:

```text
scripts/replay_group_local_probecommit.py
```

It evaluates:

- group-local score diagnostics,
- within-group pairwise concordance,
- top-score vs bottom-score candidate bad rates,
- linear candidate-utility drop estimates,
- held-out-seed tuning for abstain/drop/shrink rules,
- exact replay metrics by selecting among already evaluated actions.

The deployable action set excludes oracle and random policies:

```text
token_weighted
freshness_weighted
anchor_drop_bottom25
anchor_positive_threshold
anchor_shrink
probecommit_v1
```

Oracle policies are retained only as references in the output summaries. A regression test now checks that oracle/random actions cannot be selected by the train-seed tuner.

## Result Summary

EXP2.6 joined `614` complete groups across 3 seeds.

| Metric | Value |
|---|---:|
| Mean test utility gain vs token-weighted | 0.000114 |
| All held-out seeds positive gain | true |
| Mean negative-rate relative drop | 0.044 |
| Mean strict-negative relative drop | 0.125 |
| Mean selected mass | 0.882 |
| Mean oracle-positive headroom captured | -0.256 |
| Gate pass | false |

Decision: do not start online ProbeCommit from these group-local rules.

The gain is positive but too small, and the policy does not recover oracle headroom. Negative-rate reduction is far below the `20-25%` target.

## Group-Local Score Diagnostics

| Score | Candidate good AUROC | Pairwise concordance | Top1 bad | Bottom1 bad | Top1 strict bad | Bottom1 strict bad | Linear drop25 gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `probe_grad_dot` | 0.582 | 0.511 | 0.446 | 0.480 | 0.182 | 0.202 | 0.000004 |
| `calibrated_score` | 0.627 | 0.517 | 0.436 | 0.485 | 0.174 | 0.199 | 0.000008 |
| `freshness` | 0.511 | n/a | 0.469 | 0.477 | 0.197 | 0.182 | 0.000008 |
| `combined_score` | 0.507 | 0.496 | 0.476 | 0.485 | 0.178 | 0.197 | 0.000000 |

Interpretation:

- `calibrated_score` still has the best candidate-level AUROC.
- Within-group pairwise concordance remains weak: `0.517` for calibrated score and `0.511` for raw grad-dot.
- The top-score candidate has lower bad rate than the bottom-score candidate, but the gap is not strong enough for reliable selection.
- Linear drop estimates are close to zero, which matches the exact replay result: score-based dropping has very limited headroom under these features.

## Held-Out Seed Replay

Each split tunes rules on two seeds and evaluates the selected rule on the held-out seed.

| Test seed | Selected rule | Test groups | Utility gain | Negative drop | Strict drop | Headroom captured | Selected mass | Actions |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 53 | `calibrated_score:drop25_if_iqr_ge:0.00433481` | 195 | 0.000087 | 0.042 | 0.094 | 0.170 | 0.843 | `{'anchor_drop_bottom25': 191, 'token_weighted': 4}` |
| 67 | `probe_grad_dot:drop25_if_top_gap_ge:0.0012454` | 208 | 0.000024 | 0.025 | 0.161 | -1.173 | 0.917 | `{'anchor_drop_bottom25': 118, 'token_weighted': 90}` |
| 79 | `probe_grad_dot:drop25_if_top_gap_ge:0.000940996` | 211 | 0.000231 | 0.064 | 0.121 | 0.237 | 0.886 | `{'anchor_drop_bottom25': 161, 'token_weighted': 50}` |

The tuner mostly chooses `anchor_drop_bottom25` with a small number of token-weighted fallbacks. That is not an effective abstain/wait policy. It still acts on too many groups, and the confidence features do not reliably identify when dropping helps.

Seed 67 is the clearest warning case: the selected rule gives a small positive mean utility gain but negative oracle-headroom capture. It reduces some strict negatives, yet does not move toward the oracle-positive reference.

## Gate

| Gate | Result |
|---|---:|
| Complete groups joined | 614 |
| All held-out seeds positive utility gain | true |
| Mean utility gain positive | true |
| Oracle-positive headroom captured >= 0.40 | false |
| Negative-rate drop >= 0.20 | false |
| Selected mass >= 0.40 | true |
| EXP2.6 gate pass | false |

This fails the online-readiness gate.

## Interpretation

EXP2.6 strengthens the EXP2.5 diagnosis:

1. Bad merge headroom is real.
2. Current-state anchor scores have some signal.
3. The current signal is not strong enough under the group-local decision geometry.
4. Train-seed-tuned abstain/drop rules produce small gains but not enough negative-rate reduction or oracle-headroom recovery.

The right next step is not online training. The next useful offline work is to collect denser captures and improve group-local evidence.

## Next Offline Work

Recommended next steps:

1. Collect denser captures: at least `500` complete groups per seed, preferably `1000+`.
2. Add group-normalized features:
   - score ranks,
   - score gaps,
   - score entropy,
   - consensus distance rank,
   - disagreement measures,
   - selected-mass uncertainty.
3. Add real abstain/wait replay:
   - act only when score separation is strong,
   - otherwise keep token-weighted fallback,
   - report abstain/wait rate explicitly.
4. Try multiple anchor batches or anchor ensembles to reduce split sensitivity.
5. Only rerun exact model replay after a cheap group-local diagnostic shows stronger within-group concordance.

## EXP2.7 Follow-Up

EXP2.7 split the local group analysis into reusable feature-building and policy-grid stages:

```text
scripts/build_group_local_features.py
scripts/replay_group_local_policy_grid.py
scripts/aggregate_group_local_results.py
docs/EXP2_7.md
```

Result on the existing artifacts: still no-go for online ProbeCommit.

| Metric | EXP2.7 local value |
|---|---:|
| Complete groups | 614 |
| Mean held-out gain vs token-weighted | 0.000062 |
| Negative-rate relative drop | 0.037 |
| Strict-negative relative drop | 0.064 |
| Oracle-positive headroom captured | -0.276 |
| Act rate | 0.483 |
| Gate pass | false |

The tooling is ready for denser captures, but the existing score family still fails the group-local evidence gate.

## Exact Command

```bash
mkdir -p experiment-results/EXP2/probecommit-offline/group-local

python3 scripts/replay_group_local_probecommit.py \
  --features \
    experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_53/calibrated_test.jsonl \
    experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_67/calibrated_test.jsonl \
    experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_79/calibrated_test.jsonl \
  --policy-replay \
    experiment-results/EXP2/probecommit-offline/seed53/policy_replay.jsonl \
    experiment-results/EXP2/probecommit-offline/seed67/policy_replay.jsonl \
    experiment-results/EXP2/probecommit-offline/seed79/policy_replay.jsonl \
  --out-json experiment-results/EXP2/probecommit-offline/group-local/group_local_replay.json \
  --out-md experiment-results/EXP2/probecommit-offline/group-local/group_local_replay.md \
  --out-records experiment-results/EXP2/probecommit-offline/group-local/group_local_records.jsonl
```

## Artifact Paths

```text
experiment-results/EXP2/probecommit-offline/group-local/group_local_replay.json
experiment-results/EXP2/probecommit-offline/group-local/group_local_replay.md
experiment-results/EXP2/probecommit-offline/group-local/group_local_records.jsonl
```

## Validation

```text
/tmp/yeto-runtime/bin/python -m py_compile scripts/replay_group_local_probecommit.py
/tmp/yeto-runtime/bin/python -m pytest tests/test_smoke_scripts.py
```

Result:

```text
14 passed
```
