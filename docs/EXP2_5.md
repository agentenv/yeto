# EXP2.5: Anchor-Gradient ProbeCommit Offline Replay

## Question

Can current-state anchor-gradient scores recover merge utility headroom exposed by oracle filtering, without using oracle labels at decision time?

Short answer: not yet.

The offline measurement confirms that harmful candidates can degrade actual merged updates, and oracle filtering exposes real headroom. However, the disjoint anchor/oracle score is much weaker than the earlier non-disjoint diagnostic, and the fixed `probecommit_v1` policy does not pass the online-readiness gate.

## Input Artifacts

Primary inputs were the three syncer-current equal-token captures:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m/
```

The run used:

| Item | Value |
|---|---:|
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Data | per-seed `work/eval.jsonl` from the EXP2 captures |
| LoRA rank / alpha | 2 / 4 |
| Fragments | 12 |
| Task A probe batches | 4 |
| Task B/C oracle batches | 8 |
| Anchor rows | 64 |
| Oracle rows | 192 |
| Device | AWS `p4de.24xlarge` spot, 8x A100 80GB |

The eval files have 256 rows. With a 64-row anchor split, only 192 rows remain for the disjoint oracle split. This is lower than the requested `oracle-max-rows 256`, but the split is still deterministic and non-overlapping.

Output root:

```text
experiment-results/EXP2/probecommit-offline/
```

## Full Merge Replay

Task A replayed every complete captured `(step, fragment)` group with at least two candidates.

| Seed | Groups | Token utility | Token negative | Token strict negative | Oracle-positive utility | Oracle-positive headroom | Oracle top-k headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 195 | 0.000583 | 0.431 | 0.241 | 0.002724 | 0.002141 | 0.001645 |
| 67 | 208 | 0.001427 | 0.404 | 0.168 | 0.003346 | 0.001919 | 0.001159 |
| 79 | 211 | 0.001617 | 0.360 | 0.133 | 0.002826 | 0.001208 | 0.001255 |

Aggregate:

| Metric | Value |
|---|---:|
| Total complete groups | 614 |
| Min groups per seed | 195 |
| Mean candidate count | 5.667 |
| Bad weight mass | 0.441 |
| Strict-bad weight mass | 0.178 |
| Token-weighted utility | 0.001209 |
| Token-weighted negative rate | 0.398 |
| Token-weighted strict-negative rate | 0.181 |
| Oracle-positive utility | 0.002965 |
| Oracle-positive headroom | 0.001756 |
| Oracle top-k headroom | 0.001353 |
| Random positive-count headroom | 0.000251 |

Interpretation:

- Bad individual candidates are not fully averaged away.
- Token-weighted merged updates are negative in about 39.8% of complete groups.
- Oracle-positive and oracle-top-k filtering both improve mean one-step merge utility across all three seeds.
- Oracle headroom is larger than matched random-count controls.
- Task A still does not pass the preferred group-count gate, because the available captures have only 195 to 211 complete groups per seed, below the requested 500+.

Gate A:

| Gate | Result |
|---|---:|
| `records_gate_500` | false |
| `token_negative_rate_above_20pct` | true |
| `oracle_positive_headroom_positive` | true |
| `oracle_topk_headroom_positive` | true |
| `oracle_positive_beats_random` | true |
| `oracle_topk_beats_random` | true |
| `all_seed_oracle_positive_headroom_positive` | true |
| `gate_a_pass` | false |

Practical decision: merge headroom is real in these captures, but the capture count is not enough to treat Gate A as fully passed.

## Disjoint Anchor / Oracle Scoring

Task B recomputed candidate utility using a deterministic non-overlapping split:

```text
anchor rows: 64
oracle rows: 192
overlap count: 0 for every seed
```

Per-seed scoring:

| Seed | Records | Negative utility | Token AUROC | Hand AUROC | Raw grad-dot AUROC | Raw grad-cos AUROC | Calibrated AUROC | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 1,158 | 0.505 | 0.500 | 0.499 | 0.529 | 0.530 | 0.617 | 0.038 |
| 67 | 1,171 | 0.453 | 0.500 | 0.495 | 0.630 | 0.624 | 0.670 | 0.052 |
| 79 | 1,182 | 0.479 | 0.500 | 0.522 | 0.580 | 0.576 | 0.628 | 0.107 |

Aggregate:

| Metric | Value |
|---|---:|
| Raw grad-dot AUROC | 0.580 |
| Held-out-seed calibrated AUROC | 0.638 |
| Held-out-seed calibrated ECE | 0.066 |
| Token AUROC | 0.500 |
| Hand-score AUROC | 0.505 |
| All split overlap counts | 0 |

Interpretation:

- Disjoint anchor/oracle scoring is above token count and the hand score, but it is below the intended Gate B threshold.
- The earlier non-disjoint/current diagnostic was much stronger. This disjoint run is the more conservative result and should supersede it for online-readiness decisions.
- The seed 67 split is the strongest, but seed 53 is close to weak-signal territory.
- Calibration error is acceptable on average, but AUROC is the blocker.

Gate B:

| Gate | Result |
|---|---:|
| `overlap_count == 0` | true |
| raw grad-dot AUROC mean >= 0.65 | false |
| held-out calibrated AUROC mean >= 0.70 | false |
| every held-out seed AUROC >= 0.65 | false |
| mean ECE <= 0.12 | true |

Decision: Gate B fails. The disjoint current-state signal is promising but not reliable enough to drive an online policy.

## Held-Out Seed Calibration

Calibration used three held-out-seed splits:

```text
train: 67,79 -> test: 53
train: 53,79 -> test: 67
train: 53,67 -> test: 79
```

| Test seed | Train seeds | Train records | Test records | Calibrated AUROC | ECE |
|---:|---|---:|---:|---:|---:|
| 53 | 67,79 | 2,353 | 1,158 | 0.617 | 0.038 |
| 67 | 53,79 | 2,340 | 1,171 | 0.670 | 0.052 |
| 79 | 53,67 | 2,329 | 1,182 | 0.628 | 0.107 |

This confirms that the calibrated score transfers across seeds better than token count, but not strongly enough for the proposed policy gate.

## Offline Policy Replay

Task C evaluated anchor-gradient policies on the disjoint oracle split. The reported `probecommit_v1` used:

```text
score = probe_grad_dot
selected = candidates with score >= percentile(score, 30)
if selected_mass < 0.35: selected = top 50% by score
weights = token_weight * softplus(score / 0.01)
outer_lr_multiplier = 0.5 if selected mean score < 0 else 1.0
```

Per-seed result:

| Seed | Groups | Token utility | Token negative | ProbeCommit-v1 utility | ProbeCommit-v1 negative | Headroom captured | Selected mass | Random-count utility |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 195 | 0.000871 | 0.487 | 0.000910 | 0.492 | 0.226 | 0.682 | 0.000978 |
| 67 | 208 | 0.001209 | 0.380 | 0.001140 | 0.404 | -1.089 | 0.709 | 0.001071 |
| 79 | 211 | 0.000817 | 0.445 | 0.000937 | 0.403 | 0.211 | 0.705 | 0.000580 |

Aggregate policy table:

| Policy | Mean utility | Negative rate | Strict negative | Oracle-positive headroom captured | Selected mass |
|---|---:|---:|---:|---:|---:|
| token-weighted | 0.000966 | 0.437 | 0.157 | 0.000 | 1.000 |
| freshness-weighted | 0.000993 | 0.406 | 0.166 | -0.248 | 1.000 |
| metadata-calibrated | 0.000908 | 0.427 | 0.153 | -0.440 | 1.000 |
| anchor-reweight-sigmoid | 0.000849 | 0.426 | 0.187 | -1.046 | 1.000 |
| anchor-reweight-softplus | 0.000986 | 0.418 | 0.167 | -0.394 | 1.000 |
| anchor-top50 | 0.000903 | 0.427 | 0.174 | -0.703 | 0.504 |
| anchor-drop-bottom50 | 0.000903 | 0.427 | 0.174 | -0.703 | 0.504 |
| probecommit-v1 | 0.000996 | 0.433 | 0.143 | -0.217 | 0.699 |
| oracle-positive | 0.002084 | 0.120 | 0.023 | 1.000 | 0.523 |
| oracle-topk | 0.001679 | 0.336 | 0.103 | 0.686 | 0.504 |
| random-probecommit-count | 0.000876 | 0.429 | 0.169 | -0.599 | 0.699 |

Interpretation:

- Oracle-positive filtering still gives a large upper bound in the policy replay setting.
- The fixed `probecommit_v1` policy gives only a tiny mean utility gain over token-weighted merging: `+0.0000298`.
- Gains are not consistent across seeds; seed 67 is worse than token-weighted.
- Negative merge rate does not drop enough: 0.437 to 0.433.
- Strict-negative rate improves: 0.157 to 0.143.
- Selected mass is healthy at about 0.699, so this is not a trivial drop-most-candidates policy.
- Random-count control does not explain the small mean gain, but the gain is too small and inconsistent to justify moving online.

Gate C:

| Gate | Result |
|---|---:|
| all seeds positive utility gain | false |
| mean utility gain | 0.0000298 |
| headroom captured >= 50% | false |
| negative rate drop >= 25% | false |
| strict-negative rate decreases | true |
| selected mass >= 0.40 | true |
| beats random-count control | true |
| `gate_c_pass` | false |

## Further Investigation

I ran an additional local investigation over the existing EXP2.5 JSONL artifacts, without recomputing model probes. The goal was to explain why candidate-level anchor-gradient scoring looked useful, while merge-policy replay did not pass.

Additional outputs:

```text
experiment-results/EXP2/probecommit-offline/investigation/further_analysis.json
experiment-results/EXP2/probecommit-offline/investigation/further_analysis.md
```

### Probe split caveat

Task A and Task C are not numerically identical replay settings:

| Task | Purpose | Probe rows / batches | Consequence |
|---|---|---|---|
| Task A | full merge headroom | 4 batches, up to 128 rows | used to estimate all available merge groups cheaply |
| Task B/C | disjoint scoring and policy replay | 8 oracle batches, 64 anchor rows + 192 oracle rows | stricter split, lower-noise utility labels |

Because of this, the Task A token-weighted mean utility (`0.001209`) should not be compared directly to the Task C token-weighted mean utility (`0.000966`). The direction of the results is comparable, but exact utility levels are split-dependent.

### Candidate score tails are useful, but group-local ranking is weak

The calibrated held-out score has a visible global tail effect. On all 3,511 candidate records, the top calibrated-score quintile has much higher utility and lower bad rate than the bottom quintile:

| Score | Quintile | Records | Mean utility | Bad rate | Strict-bad rate |
|---|---:|---:|---:|---:|---:|
| `calibrated_score` | bottom 20% | 702 | -0.000731 | 0.600 | 0.261 |
| `calibrated_score` | middle 20% | 702 | 0.000090 | 0.501 | 0.182 |
| `calibrated_score` | top 20% | 703 | 0.004000 | 0.303 | 0.094 |
| `probe_grad_dot` | bottom 20% | 702 | 0.000146 | 0.513 | 0.222 |
| `probe_grad_dot` | top 20% | 703 | 0.003000 | 0.319 | 0.100 |

This means the signal is not useless. The top tail is genuinely enriched for useful fragments.

However, the syncer does not choose from a global candidate pool. It chooses among candidates inside the same `(step, fragment)` group. Within those groups, ranking quality is much weaker:

| Seed | Score | Good AUC | Pairwise concordance within group | Top-1 bad rate | Bottom-1 bad rate | Top-1 strict-bad | Bottom-1 strict-bad |
|---:|---|---:|---:|---:|---:|---:|---:|
| 53 | `probe_grad_dot` | 0.529 | 0.491 | 0.497 | 0.487 | 0.210 | 0.174 |
| 67 | `probe_grad_dot` | 0.630 | 0.520 | 0.413 | 0.471 | 0.173 | 0.202 |
| 79 | `probe_grad_dot` | 0.580 | 0.521 | 0.431 | 0.483 | 0.166 | 0.227 |
| all | `probe_grad_dot` | 0.583 | 0.521 | 0.361 | 0.491 | 0.120 | 0.222 |
| all | `calibrated_score` | 0.627 | 0.508 | 0.421 | 0.449 | 0.167 | 0.148 |

This is the core failure mode. `calibrated_score` improves global candidate AUROC, but it does not reliably order candidates inside each merge group. `probe_grad_dot` has better group-local behavior than the calibrated score, but its aggregate pairwise concordance is still only `0.521`, barely above chance.

Policy implication: a group-local commit rule cannot rely on the current score as a precise ranker. A usable version probably needs either a much stronger within-group score, an abstain/wait action for ambiguous groups, or a conservative high-confidence tail rule rather than top-k selection in every group.

### Freshness cannot rank candidates in this stress setting

Freshness has near-chance AUROC and zero useful within-group ranking in this replay. Under the equal-token freeze-delay setup, many candidates inside a merge group share the same or effectively tied freshness. This confirms that simple freshness weighting is not the missing policy.

### Bad mass strongly predicts merge failure

The full replay also shows that bad candidate mass is not harmless. Binning groups by bad weight mass gives a monotonic degradation in token-weighted merge utility:

| Bad weight mass | Groups | Token utility | Token negative rate | Oracle-positive headroom |
|---|---:|---:|---:|---:|
| `[0.00, 0.25)` | 218 | 0.005109 | 0.060 | 0.000248 |
| `[0.25, 0.50)` | 86 | 0.001608 | 0.279 | 0.001086 |
| `[0.50, 0.75)` | 160 | -0.000198 | 0.512 | 0.001800 |
| `[0.75, 1.00)` | 150 | -0.003124 | 0.833 | 0.004242 |

This strengthens the measurement result: harmful candidates are not always averaged away. When bad mass dominates the candidate set, token-weighted merge utility becomes negative and oracle headroom grows.

### Oracle headroom is intermittent

On the stricter policy split, oracle-positive filtering improves token-weighted utility on only `56.5%` of groups:

| Groups | Mean oracle-positive headroom | Median | p05 | p95 | Positive headroom | Zero/nonpositive headroom |
|---:|---:|---:|---:|---:|---:|---:|
| 614 | 0.001113 | 0.000586 | -0.002110 | 0.005194 | 0.565 | 0.435 |

This matters for policy design. There are many groups where oracle-positive filtering has no headroom because all candidates are already useful, no positive candidates exist, or the aggregate is already better than the filtered subset under probe noise. A policy that always drops candidates can easily harm those groups.

### Policy sensitivity

Across all implemented non-oracle policies, the best mean utility gain is from dropping the bottom 25% by anchor score, not from `probecommit_v1`. It still does not pass the gate.

| Policy | Mean utility | Gain vs token | Gain positive rate | Negative rate | Strict-negative rate | Selected mass |
|---|---:|---:|---:|---:|---:|---:|
| token-weighted | 0.000967 | 0.000000 | 0.000 | 0.436 | 0.156 | 1.000 |
| freshness-weighted | 0.000993 | 0.000026 | 0.531 | 0.406 | 0.166 | 1.000 |
| anchor reweight softplus | 0.000987 | 0.0000195 | 0.493 | 0.417 | 0.166 | 1.000 |
| anchor drop bottom 25% | 0.001081 | 0.000114 | 0.484 | 0.422 | 0.142 | 0.845 |
| anchor positive threshold | 0.000915 | -0.0000518 | 0.362 | 0.381 | 0.134 | 0.605 |
| anchor top 50% | 0.000906 | -0.0000614 | 0.489 | 0.427 | 0.174 | 0.503 |
| metadata calibrated | 0.000904 | -0.0000629 | 0.500 | 0.427 | 0.153 | 1.000 |
| `probecommit_v1` | 0.000997 | 0.0000300 | 0.502 | 0.432 | 0.143 | 0.699 |
| random matched to `probecommit_v1` count | 0.000873 | -0.0000942 | 0.435 | 0.428 | 0.169 | 0.699 |
| oracle-positive | 0.002080 | 0.001113 | 0.565 | 0.119 | 0.023 | 0.523 |
| oracle top-k | 0.001676 | 0.000709 | 0.617 | 0.336 | 0.103 | 0.503 |

Takeaways:

- `anchor_drop_bottom25` is the strongest simple non-oracle policy in this artifact set, but its gain is small and its positive-gain rate is below 50%.
- `anchor_positive_threshold` reduces negative and strict-negative rates more than `probecommit_v1`, but it lowers mean utility. It is acting like a conservative risk-control rule, not a better descent rule.
- `probecommit_v1` beats its matched random-count control, so the score is doing something. The effect is too small and not seed-stable.
- Random selection with the oracle-positive selected count gives a nontrivial utility gain in the broader policy table. That is not a deployable baseline because the count comes from oracle labels, but it warns that part of the oracle gap can come from selected-mass regularization rather than correct identification of useful candidates.

### Updated diagnosis

The EXP2.5 proof chain now looks like this:

| Step | Status | Evidence |
|---|---|---|
| Equal-token candidates have diverse utility | passed earlier | EXP2 syncer-current oracle |
| Bad candidates affect actual merged updates | mostly passed | bad-mass bins and oracle headroom; capture count still below gate |
| Anchor-gradient predicts individual utility on disjoint data | weak partial | global top tail works, AUROC below gate |
| Anchor-gradient ranks candidates within a merge group | fails / weak | pairwise concordance around `0.52` for raw score and `0.51` for calibrated score |
| Offline policy improves merged utility enough | fails | best non-oracle gain small; `probecommit_v1` not seed-stable |

This changes the next research question. The issue is no longer just "find a candidate-level score." The score must be useful under the group-local decision geometry of the syncer.

## Go / No-Go Decision for Online ProbeCommit

Do not proceed to online ProbeCommit from this policy.

Reason:

1. Full merge replay shows real oracle headroom, but the available captures do not meet the preferred 500+ complete groups per seed.
2. Disjoint anchor/oracle scoring drops to AUROC 0.580 raw and 0.638 calibrated, below the offline score gate.
3. The fixed policy does not capture 50% of oracle headroom, does not reduce negative merge rate by 25%, and does not improve all held-out seeds.
4. Further analysis shows the current score has useful global tails but weak within-group ranking, which is exactly the ranking problem the syncer policy needs to solve.

The next step should be policy redesign, not online training. The most useful direction is to keep the merge-headroom result, then improve group-local scoring or policy selection with train-seed-only tuning and stronger current-state features.

Concrete next offline actions:

1. Collect denser captures so each seed reaches at least 500 complete groups.
2. Tune policies on train seeds only, especially bottom-drop percentile, high-confidence threshold, selected-mass floor, and shrink factor.
3. Add calibrated-score policy variants. The score improves global AUROC but was not used as the main `probecommit_v1` decision score here.
4. Build group-normalized features: per-group score ranks, score gaps, consensus distance, selected-mass uncertainty, and candidate disagreement.
5. Test abstain/wait policies that avoid acting when group-local score separation is weak.
6. Increase anchor diversity or use multiple small anchor batches to reduce sensitivity to the 64-row anchor split.

## EXP2.6 Follow-Up

EXP2.6 implemented the first group-local offline replay over the existing EXP2.5 artifacts. It joins calibrated candidate records with exact policy replay rows, tunes abstain/drop/shrink rules on train seeds, and evaluates the selected rule on the held-out seed.

Result: still no-go for online ProbeCommit.

| Metric | EXP2.6 value |
|---|---:|
| Joined complete groups | 614 |
| Mean test utility gain vs token-weighted | 0.000114 |
| All held-out seeds positive gain | true |
| Negative-rate relative drop | 0.044 |
| Strict-negative relative drop | 0.125 |
| Oracle-positive headroom captured | -0.256 |
| Selected mass | 0.882 |
| Gate pass | false |

The key finding is unchanged but sharper: group-local confidence rules produce small positive utility gains, but they do not recover oracle headroom and they do not reduce negative merges enough. The current score is still too weak under the syncer decision geometry.

Full report:

```text
docs/EXP2_6.md
```

New script:

```text
scripts/replay_group_local_probecommit.py
```

## Known Limitations

- The capture cadence produced only 195 to 211 complete groups per seed. This is a full replay of the available captures, but it is not enough for the preferred group-count gate.
- The disjoint split had 192 oracle rows because each eval file has 256 rows and 64 were reserved for anchor batches.
- Policy parameters were fixed from the proposed v1 rule. A proper train-seed-only grid over `tau`, percentile, shrink factor, and selected-mass floor may do better.
- This is one-step offline replay on SmolLM2-135M and Capybara eval rows, not an online training result.
- The policy replay uses held-out calibrated feature files, but the main policy score is raw `probe_grad_dot`; calibrated policy variants should be tested separately.
- Candidate-level AUROC is not sufficient. The current score is much weaker as a within-group ranker, which is the relevant objective for merge selection.
- Utility and policy means differ between Task A and Task C because they use different probe row/batch settings.
- No online training was started.

## Exact Commands

Each seed output directory contains the exact commands and reproducibility snapshots:

```text
experiment-results/EXP2/probecommit-offline/seed53/command.sh
experiment-results/EXP2/probecommit-offline/seed67/command.sh
experiment-results/EXP2/probecommit-offline/seed79/command.sh
experiment-results/EXP2/probecommit-offline/seed53/git_commit.txt
experiment-results/EXP2/probecommit-offline/seed53/git_diff.patch
```

The core command shapes were:

```bash
python scripts/replay_merge_utility.py \
  --capture-dir <seed>/work/m12/syncer_probe \
  --model HuggingFaceTB/SmolLM2-135M \
  --data <seed>/work/eval.jsonl \
  --seq-len 128 --lora-r 2 --lora-alpha 4 --fragments 12 \
  --probe-batches 4 --probe-batch-size 1 --probe-max-rows 128 \
  --device cuda --all-groups --min-candidates 2 --resume
```

```bash
python scripts/evaluate_anchor_gradient_features.py \
  --capture-dir <seed>/work/m12/syncer_probe \
  --model HuggingFaceTB/SmolLM2-135M \
  --data <seed>/work/eval.jsonl \
  --seq-len 128 --lora-r 2 --lora-alpha 4 --fragments 12 \
  --anchor-batches 2 --anchor-batch-size 1 --anchor-max-rows 64 \
  --oracle-batches 8 --oracle-batch-size 1 --oracle-max-rows 256 \
  --disjoint-anchor-oracle --device cuda --max-records 2048
```

```bash
python scripts/calibrate_fragment_score.py \
  seed53/anchor_gradient_disjoint.jsonl \
  seed67/anchor_gradient_disjoint.jsonl \
  seed79/anchor_gradient_disjoint.jsonl \
  --split heldout-seed --test-seed <53|67|79>
```

```bash
python scripts/replay_probecommit_policy.py \
  --features calibrated_anchor_gradient_heldout_seed_<seed>/calibrated_test.jsonl \
  --capture-dir <seed>/work/m12/syncer_probe \
  --split-manifest seed<seed>/disjoint_split_manifest.json \
  --model HuggingFaceTB/SmolLM2-135M \
  --data <seed>/work/eval.jsonl \
  --seq-len 128 --lora-r 2 --lora-alpha 4 --fragments 12 \
  --oracle-batches 8 --oracle-batch-size 1 --oracle-max-rows 256 \
  --device cuda --min-candidates 2 --score-field probe_grad_dot
```

## Artifact Paths

Primary summaries:

```text
experiment-results/EXP2/probecommit-offline/merge_replay_aggregate.json
experiment-results/EXP2/probecommit-offline/merge_replay_aggregate.md
experiment-results/EXP2/probecommit-offline/heldout_seed_calibration_aggregate.json
experiment-results/EXP2/probecommit-offline/heldout_seed_calibration_aggregate.md
experiment-results/EXP2/probecommit-offline/policy_replay_aggregate.json
experiment-results/EXP2/probecommit-offline/policy_replay_aggregate.md
experiment-results/EXP2/probecommit-offline/investigation/further_analysis.json
experiment-results/EXP2/probecommit-offline/investigation/further_analysis.md
```

Per-seed outputs:

```text
experiment-results/EXP2/probecommit-offline/seed53/
experiment-results/EXP2/probecommit-offline/seed67/
experiment-results/EXP2/probecommit-offline/seed79/
```

Per-seed files include:

```text
full_merge_replay.jsonl
full_merge_replay_summary.json
anchor_gradient_disjoint.jsonl
anchor_gradient_disjoint_summary.json
disjoint_split_manifest.json
policy_replay.jsonl
policy_replay_summary.json
command.sh
git_commit.txt
git_diff.patch
stdout.log
stderr.log
summary.json
```

## AWS Run Notes

The offline run used one `p4de.24xlarge` spot instance in `us-east-1d`. Local upload to EC2 over SSH was too slow, so the seed archives were staged through a temporary S3 bucket and downloaded on the instance with presigned URLs.

Cleanup completed:

```text
p4de instance: terminated
temporary EC2 key pair: deleted
temporary SSH ingress rule: removed
temporary S3 bucket and objects: deleted
```

## EXP2.12 Scale Stress Follow-up

After EXP2.5-2.11 showed that small-model group-local policies were brittle, I ran a manual p4de scale sweep on Qwen/Gemma models to test whether the failure was a tiny-model artifact. The p4de instance used a manual DLAMI rather than Sky because the Sky-selected p4de image had a broken CUDA/NVSwitch stack. The full compact table is in `experiment-results/EXP2/scale_policy_summary.md`.

### Setup

- Instance: manual AWS `p4de.24xlarge`, 8x A100 80GB, DLAMI with working Fabric Manager.
- Data: `trl-lib/Capybara`.
- Models: `qwen35-4b`, `qwen35-9b`, `qwen3-8b`, `gemma4`.
- Stress: fixed-token windows with freeze-before-delay push.
- Policies: token-weighted baseline, anchor/drop/reweight variants, direct action-probe variants, candidate-probe variants, oracle references.

### Main Scale Findings

| Run | Records | Best deployable result | Gate status |
|---|---:|---|---|
| Qwen3.5-9B seed31 | 16 | `action_probe_risk_aware`: gain `+0.00197`, headroom captured `40.9%`, selected mass `0.781` | Closest signal; fails negative/drop safety gates. |
| Qwen3.5-9B seed43 | 20 | No replicate: best action-probe gain only `+0.000132`, headroom captured negative | Does not replicate seed31. |
| Gemma4 long | 52 | `action_probe_top1`: gain `+0.02011`, selected mass `0.854`, headroom captured `38.6%` | Strong mean utility; still misses headroom threshold and safety gates. |
| Gemma4 restricted action sweep | 52 | `action_probe_top1`: gain `+0.01505`, selected mass `0.936`, headroom captured `19.1%` | Mean utility robust; safety/headroom weak. |
| Qwen3-8B | 15 | `metadata_calibrated`: gain `+0.00604`, headroom captured `73.3%` in policy replay | Not an action-probe win; safety unchanged. |
| Qwen3.5-4B fast | 20 | Several small positive gains around `+0.001` | Positive but headroom capture often negative. |

### Decision

Do not start online ProbeCommit yet.

The scale runs show the policy problem is not purely a tiny-model artifact: Gemma4 and Qwen3.5-9B both show deployable actions with meaningful one-step utility gain, and the best Qwen3.5-9B action-probe run crossed the mean-gain and headroom-capture thresholds. However, the signal did not replicate on Qwen3.5-9B seed43, and the strongest Gemma4 run still failed the safety gates. The remaining blocker is not finding mean utility; it is reducing negative/strict-negative merge rate reliably across seeds.

### Next Direction

The most promising direction is a conservative safety-first policy rather than more broad action search:

1. Use action-probe only as a veto/shrink rule, not as top-1 action selection.
2. Keep selected mass high (`>=0.85`) unless anchor evidence is very strong.
3. Optimize for negative-rate drop first, mean utility second.
4. Replicate Qwen3.5-9B seed31 with denser groups before any online run.
5. If safety still fails, pivot to a measurement/diagnostic contribution rather than an online policy claim.
