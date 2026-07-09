# EXP2.8: Hard Group-Local Policy Search

## Question

Is there hidden policy signal in the existing group-local features if we search harder than threshold grids?

Short answer: some deployable-action headroom exists, and a harder selector improves mean utility over EXP2.7. But it still does not recover oracle headroom or reduce negative merges enough. The method remains no-go for online ProbeCommit.

## Setup

Input:

```text
experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl
```

No new model probes, online training, or AWS jobs were started for this step.

New script:

```text
scripts/search_group_local_policy.py
```

The script performs a nested held-out-seed search:

1. Outer split: hold out one seed for final evaluation.
2. Inner split: tune kNN action-selector configs across the two train seeds.
3. Refit selector on both train seeds.
4. Evaluate on the held-out seed.

The selector uses only group-local decision-time features. It predicts which deployable action should be used:

```text
token_weighted
freshness_weighted
anchor_drop_bottom25
anchor_positive_threshold
anchor_shrink
probecommit_v1
```

Oracle actions are excluded from deployable tuning and selection. They are reported only as upper bounds.

## Why This Search Is Harder

EXP2.7 used threshold-style rules over individual group statistics. EXP2.8 tries a broader selector:

- feature selection from all numeric group-local stats,
- k-nearest-neighbor action-gain prediction,
- weighted and unweighted neighbors,
- multiple feature counts,
- multiple k values,
- action fallback thresholds,
- optional train-seed reward for oracle-positive headroom capture.

This asks a stronger question:

> If the available features contain enough information to choose among existing deployable actions, can a supervised held-out-seed selector find it?

## Action-Set Headroom

There is substantial oracle headroom inside the deployable action set itself.

| Reference | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass |
|---|---:|---:|---:|---:|---:|
| best deployable action per group | 0.001644 | 0.593 | 0.792 | 1.581 | 0.813 |
| oracle-positive | 0.001113 | 0.728 | 0.854 | 1.000 | 0.523 |
| oracle top-k | 0.000709 | 0.231 | 0.344 | 0.684 | 0.503 |

This is important. The problem is not that the action set is intrinsically useless. If a selector could choose the right existing action per group, the replay would look very strong.

The best deployable action distribution is mixed:

```text
anchor_drop_bottom25: 147
probecommit_v1: 141
freshness_weighted: 104
token_weighted: 95
anchor_positive_threshold: 77
anchor_shrink: 50
```

So the right policy is not simply "always drop" or "always shrink." The selector needs real group-local discrimination.

Seed-level deployable oracle headroom is present in every seed:

| Seed | Groups | Token negative rate | Best deployable gain | Oracle-positive gain |
|---:|---:|---:|---:|---:|
| 53 | 195 | 0.487 | 0.001743 | 0.001342 |
| 67 | 208 | 0.380 | 0.001457 | 0.000969 |
| 79 | 211 | 0.445 | 0.001737 | 0.001043 |

This rules out a trivial explanation where only one seed has action-set headroom. The harder problem is that the best action varies group by group inside each seed.

Fixed deployable actions are not enough:

| Fixed action | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass |
|---|---:|---:|---:|---:|---:|
| `anchor_drop_bottom25` | 0.000114 | 0.034 | 0.094 | -0.161 | 0.845 |
| `anchor_positive_threshold` | -0.000052 | 0.127 | 0.146 | -0.205 | 0.605 |
| `anchor_shrink` | -0.000006 | 0.026 | 0.104 | -0.244 | 1.000 |
| `freshness_weighted` | 0.000026 | 0.071 | -0.063 | -0.245 | 1.000 |
| `probecommit_v1` | 0.000030 | 0.011 | 0.083 | -0.207 | 0.699 |
| `token_weighted` | 0.000000 | 0.000 | 0.000 | 0.000 | 1.000 |

The best constant action is `anchor_drop_bottom25`, but its gain is only `0.000114`, far below the deployable-action oracle gain of `0.001644`. That gap is the selection problem EXP2.8 targets.

## Hard-Search Results

All variants use held-out-seed evaluation.

| Variant | Mean gain | Negative drop | Strict drop | Headroom captured | Selected mass | Gate pass |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.000168 | 0.092 | 0.135 | -0.182 | 0.845 | false |
| risk-biased | 0.000168 | 0.092 | 0.135 | -0.182 | 0.845 | false |
| headroom reward 0.0005 | 0.000212 | 0.092 | 0.134 | -0.162 | 0.843 | false |
| headroom reward 0.002 | 0.000148 | 0.063 | 0.064 | -0.133 | 0.867 | false |
| headroom reward 0.002, smaller k | 0.000200 | 0.057 | 0.208 | -0.213 | 0.832 | false |

The best mean-utility variant is `headroom reward 0.0005`:

| Metric | Value |
|---|---:|
| Mean gain vs token-weighted | 0.000212 |
| All held-out seeds positive gain | true |
| Negative-rate relative drop | 0.092 |
| Strict-negative relative drop | 0.134 |
| Oracle-positive headroom captured | -0.162 |
| Selected mass | 0.843 |
| Gate pass | false |

This is better than EXP2.7's `0.000062` gain, but it is still far below the gate. In particular, the selector still moves opposite the oracle-positive headroom on average.

## Best Variant Split Details

For `headroom reward 0.0005`:

| Test seed | Gain | Negative drop | Strict drop | Headroom captured |
|---:|---:|---:|---:|---:|
| 53 | positive | improved | mixed | positive |
| 67 | positive | improved | improved | negative |
| 79 | positive | improved | improved | weak positive |

Seed 67 remains the blocker. This is the same failure mode seen in EXP2.6 and EXP2.7: train-seed tuning can produce a small utility gain on the held-out seed, but the selected actions do not reliably recover the oracle-positive merge headroom.

## Interpretation

EXP2.8 gives a sharper diagnosis:

1. The existing deployable action set has enough replay headroom to matter.
2. The current group-local features do not expose that headroom in a transferable way.
3. Harder policy search can find small positive utility gains.
4. Those gains are not aligned with oracle-positive headroom and do not reduce negative merges enough.

This is stronger evidence than EXP2.7. The failure is not just that the policy grid was too simple. Even a supervised kNN action selector does not pass held-out-seed gates.

## Decision

Do not start online ProbeCommit from the current score family.

The next experiment should not be more policy search over the same features. It should collect better evidence:

- denser captures,
- anchor ensembles,
- repeated anchor splits,
- group-local score stability,
- direct subset probe approximations,
- richer candidate disagreement features.

The most useful next question is:

> Can better current-state measurements identify which deployable action to take, given that deployable action headroom clearly exists?

## Commands

Default hard search:

```bash
mkdir -p experiment-results/EXP2/probecommit-offline/exp2_8_hard_search

python3 scripts/search_group_local_policy.py \
  --features experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl \
  --feature-counts 16 32 \
  --k-values 5 15 50 \
  --thresholds -0.0001 0 0.0001 \
  --out-json experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_default.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_default.md
```

Best utility variant:

```bash
python3 scripts/search_group_local_policy.py \
  --features experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl \
  --feature-counts 16 32 \
  --k-values 5 15 50 \
  --thresholds -0.0001 0 0.0001 0.0002 \
  --headroom-reward 0.0005 \
  --out-json experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom.md
```

## Artifacts

```text
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_default.json
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_default.md
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_risk.json
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_risk.md
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom.json
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom.md
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom2.json
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom2.md
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom2_smallk.json
experiment-results/EXP2/probecommit-offline/exp2_8_hard_search/hard_policy_search_headroom2_smallk.md
```

## Validation

```text
/tmp/yeto-runtime/bin/python -m py_compile scripts/search_group_local_policy.py scripts/build_group_local_features.py scripts/replay_group_local_policy_grid.py scripts/aggregate_group_local_results.py scripts/replay_group_local_probecommit.py
/tmp/yeto-runtime/bin/python -m pytest tests/test_smoke_scripts.py
```

Result:

```text
16 passed
```
