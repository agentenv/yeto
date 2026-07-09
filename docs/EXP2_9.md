# EXP2.9: Direct Action-Probe Replay

## Question

Can the syncer choose a useful deployable merge action by directly probing each action on a small anchor split, then reporting the chosen action on a disjoint oracle split?

Short answer: no. Direct action-probe replay does not pass. The deployable action set still has large oracle headroom, but a two-batch anchor probe does not select the right action reliably enough.

## Setup

EXP2.9 reuses the existing EXP2.5/EXP2.8 artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m/
experiment-results/EXP2/probecommit-offline/
```

New scripts:

```text
scripts/replay_action_probe_policy.py
scripts/aggregate_action_probe_results.py
```

No online training, new model capture, or AWS job was started. The replay ran locally on CPU because only anchor-split action utilities had to be recomputed; oracle-split action utilities were reused from the already computed disjoint `policy_replay.jsonl` files.

## Action Set

Deployable actions:

```text
token_weighted
freshness_weighted
anchor_drop_bottom25
anchor_positive_threshold
anchor_shrink
probecommit_v1
```

Reference actions:

```text
best_deployable_oracle
oracle_positive
oracle_topk
random_top1_action_count
```

Oracle and random actions are never selectable by action-probe policies.

## Anchor / Oracle Split

EXP2.9 uses the original disjoint split manifests from EXP2.5:

```text
experiment-results/EXP2/probecommit-offline/seed53/disjoint_split_manifest.json
experiment-results/EXP2/probecommit-offline/seed67/disjoint_split_manifest.json
experiment-results/EXP2/probecommit-offline/seed79/disjoint_split_manifest.json
```

The manifest matters because the original split was hash-shuffled with the AWS path string. Reusing the manifest ensures the locally recomputed anchor utilities are paired with the same disjoint oracle utility records already present in `policy_replay.jsonl`.

| Split field | Value |
|---|---:|
| Anchor rows | 64 |
| Anchor batches | 2 |
| Oracle rows | 192 |
| Oracle batches | 8 |
| Anchor/oracle overlap | 0 |
| Complete groups | 614 |

## Policies Tested

`action_probe_top1`:

```text
choose argmax action by anchor utility
```

`action_probe_margin_gated`:

```text
choose anchor top1 only if top1 - second >= 0.0005,
otherwise fall back to token_weighted
```

`action_probe_risk_aware`:

```text
choose anchor top1 only if max anchor utility >= 0,
otherwise fall back to token_weighted
```

## Results

Aggregate over seeds 53, 67, and 79:

| Policy | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass | Gate |
|---|---:|---:|---:|---:|---:|---|
| token_weighted | 0.000000 | 0.000 | 0.000 | 0.000 | 1.000 | baseline |
| best fixed deployable | 0.000118 | 0.049 | 0.063 | -0.191 | 0.898 | fail |
| EXP2.8 hard selector | 0.000212 | 0.092 | 0.134 | -0.162 | 0.843 | fail |
| action_probe_top1 | -0.000041 | 0.096 | 0.124 | -0.064 | 0.533 | fail |
| action_probe_margin_gated | -0.000056 | 0.067 | 0.126 | 0.130 | 0.683 | fail |
| action_probe_risk_aware | -0.000033 | 0.100 | 0.104 | -0.239 | 0.569 | fail |
| best_deployable_oracle | 0.001646 | 0.591 | n/a | 1.585 | 0.813 | upper |
| oracle_positive | 0.001118 | 0.727 | 0.853 | 1.000 | 0.523 | upper |
| oracle_topk | 0.000713 | 0.230 | 0.342 | 0.686 | 0.504 | upper |
| random action-count control | -0.000007 | 0.061 | 0.110 | -0.233 | 0.668 | control |

The main aggregate policy by mean gain is `action_probe_risk_aware`, but its mean gain is negative:

```text
mean gain vs token = -0.000033
negative-rate drop = 0.100
strict-negative drop = 0.104
headroom captured = -0.239
```

This is far below the EXP2.9 pass criteria.

## Per-Seed Results

Main aggregate policy: `action_probe_risk_aware`.

| Seed | Action-probe gain | Negative drop | Strict drop | Headroom captured | Chosen action distribution |
|---:|---:|---:|---:|---:|---|
| 53 | -0.000156 | 0.116 | 0.125 | -0.339 | `{'anchor_drop_bottom25': 5, 'anchor_positive_threshold': 80, 'probecommit_v1': 96, 'token_weighted': 14}` |
| 67 | -0.000010 | 0.025 | 0.065 | -0.537 | `{'anchor_drop_bottom25': 8, 'anchor_positive_threshold': 80, 'freshness_weighted': 2, 'probecommit_v1': 98, 'token_weighted': 20}` |
| 79 | 0.000066 | 0.160 | 0.121 | 0.160 | `{'anchor_drop_bottom25': 4, 'anchor_positive_threshold': 86, 'freshness_weighted': 2, 'probecommit_v1': 102, 'token_weighted': 17}` |

Seed 67 still fails. It has negative mean gain and strongly negative headroom capture under the main aggregate policy.

For completeness, the best per-seed action-probe variants were:

| Seed | Best action-probe variant | Gain | Headroom captured |
|---:|---|---:|---:|
| 53 | `action_probe_margin_gated` | -0.000145 | 0.101 |
| 67 | `action_probe_margin_gated` | 0.000049 | 0.189 |
| 79 | `action_probe_risk_aware` | 0.000066 | 0.160 |

Even the per-seed best variants are too small and inconsistent.

## Decision

Do not proceed to online ProbeCommit.

EXP2.9 fails every aggregate online-readiness gate:

```text
mean_gain_ge_0.0005: false
all_seeds_positive: false
negative_drop_ge_0.20: false
strict_drop_ge_0.20: false
headroom_captured_ge_0.40: false
seed67_positive_gain: false
seed67_nonnegative_headroom: false
beats_random_action_count: false
gate_pass: false
strong_pass: false
```

The important diagnosis is now sharper:

```text
bad merge headroom exists,
deployable action oracle headroom exists,
but two-batch direct action probing does not choose the right action.
```

This means the current online ProbeCommit path should stop. Continuing to online training from this score/action-probe family would burn compute without a clean offline reason.

## Interpretation

EXP2.9 answers the immediate question negatively. The issue is no longer just weak feature search. Even when the syncer directly probes each deployable action on a small anchor split, the selected action does not generalize to the disjoint oracle split.

Likely causes:

- two anchor batches are too noisy for group-level action selection,
- action utility differences are often smaller than probe noise,
- anchor split and oracle split disagree at the group level,
- deployable actions are too coarse and often share similar selected mass,
- shrink/drop decisions need a more stable uncertainty estimate than one anchor evaluation.

The best deployable oracle remains strong:

```text
gain vs token = 0.001646
negative drop = 0.591
selected mass = 0.813
```

So the action space still has headroom. The blocker is deployable measurement quality.

## Next Step

Stop the current ProbeCommit policy path.

The next useful work is measurement redesign, not online training:

```text
1. Estimate anchor/oracle agreement directly across multiple anchor splits.
2. Increase anchor batches: 2 -> 4 -> 8.
3. Test lower-confidence-bound action selection instead of top1 utility.
4. Add repeated anchor ensembles and report action-rank stability.
5. If action rank stability remains weak, pivot to a measurement/benchmark result.
```

A minimal EXP2.10 would be:

```text
Anchor Stability Sweep:
  anchor_batches = 2, 4, 8
  anchor_splits = 3 to 5
  metric = action-rank agreement with oracle action ranking
  stop condition = top1 action agreement remains low
```

Do not run larger models until this measurement stability issue is resolved.

## Exact Commands

Seed 53:

```bash
mkdir -p experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53

/tmp/yeto-runtime/bin/python scripts/replay_action_probe_policy.py \
  --capture-dir experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/work/m12/syncer_probe \
  --features experiment-results/EXP2/probecommit-offline/exp2_7_local/group_local_features.jsonl \
  --candidate-features experiment-results/EXP2/probecommit-offline/calibrated_anchor_gradient_heldout_seed_53/calibrated_test.jsonl \
  --policy-replay experiment-results/EXP2/probecommit-offline/seed53/policy_replay.jsonl \
  --split-manifest experiment-results/EXP2/probecommit-offline/seed53/disjoint_split_manifest.json \
  --model HuggingFaceTB/SmolLM2-135M \
  --data experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/work/eval.jsonl \
  --seq-len 128 \
  --lora-r 2 \
  --lora-alpha 4 \
  --fragments 12 \
  --anchor-batches 2 \
  --anchor-batch-size 1 \
  --anchor-max-rows 64 \
  --oracle-batches 8 \
  --oracle-batch-size 1 \
  --oracle-max-rows 256 \
  --disjoint-anchor-oracle \
  --oracle-source precomputed \
  --device cpu \
  --progress-every 25 \
  --out-jsonl experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_replay.jsonl \
  --out-summary experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_summary.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_summary.md
```

Seeds 67 and 79 used the same command shape with seed-specific paths.

Aggregate:

```bash
/tmp/yeto-runtime/bin/python scripts/aggregate_action_probe_results.py \
  --summaries \
    experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_summary.json \
    experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed67/action_probe_summary.json \
    experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed79/action_probe_summary.json \
  --expected-seeds 53 67 79 \
  --out-json experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/action_probe_aggregate.json \
  --out-md experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/action_probe_aggregate.md
```

## Artifacts

```text
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_replay.jsonl
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_summary.json
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed53/action_probe_summary.md
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed67/action_probe_replay.jsonl
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed67/action_probe_summary.json
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed67/action_probe_summary.md
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed79/action_probe_replay.jsonl
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed79/action_probe_summary.json
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/seed79/action_probe_summary.md
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/action_probe_aggregate.json
experiment-results/EXP2/probecommit-offline/exp2_9_action_probe/action_probe_aggregate.md
```

## Validation

```bash
/tmp/yeto-runtime/bin/python -m py_compile \
  scripts/replay_action_probe_policy.py \
  scripts/aggregate_action_probe_results.py

/tmp/yeto-runtime/bin/python -m pytest tests/test_smoke_scripts.py
```

Result:

```text
20 passed
```
