# RL elastic resource benchmark

`scripts/benchmark_rl_elastic.py` plans and reports studies that compare fixed
GPU partitions (trainer/rollout/standby) against scheduled and automatic
resizing inside one RL island. It is separate from `scripts/benchmark_rl.py`;
that harness, its CLI and its result fields are unchanged. The suite reuses
its prompt pairing and file format through `yeto/rl/elastic_benchmark/legacy.py`.

The planning contract lives in the miles repository under
`openspec/changes/rl-elastic-resource-benchmark/`. This document covers what is
implemented today: the study contract, capability gating, evidence, paired work
and reporting. No GPU runner ships yet; matrix items are executed by runners
that register against a runtime capability attestation.

## Commands

```bash
python scripts/benchmark_rl_elastic.py example --mode calibration --output study.json
python scripts/benchmark_rl_elastic.py validate --study study.json
python scripts/benchmark_rl_elastic.py plan --study study.json --capabilities caps.json
python scripts/benchmark_rl_elastic.py report --study study.json --capabilities caps.json --study-dir out/
```

None of these load a model, import torch/ray/miles, or create cloud resources.
`plan` prints the config table (legal configs with gradient accumulation,
illegal ones with the rejection reason), every arm × config × scenario × seed
item with its status, and the runnable budget. `report` exits non-zero while
the matrix is incomplete.

## Study manifest

A manifest is JSON with `schema_version`, `mode` (`calibration` or `formal`)
and seven field groups: `identity`, `profile`, `resources`, `matrix`, `work`,
`evaluation`, `timing`. `dump_manifest` stamps `study_hash`, a sha256 over the
canonical JSON; `load_manifest` refuses a file whose content no longer matches.

Calibration manifests may leave fields as the string `"unresolved"` and may
have an empty GPU list and no quality rules. Formal manifests must have
immutable model and data revisions, reward/source/runtime fingerprints,
physical GPU UUIDs, per-metric `delta_quality` and `catastrophic` bounds,
`delta_stable`, `min_speedup` and a statistics method. Contradictions are
rejected instead of repaired: phase updates must sum to `update_budget`,
`groups_per_update × samples_per_group` must equal `global_batch`, warmup plus
measured updates cannot exceed the budget.

## Configs, edges and capability gating

Configs are named partitions such as `P62` (6 trainer, 2 rollout). The first
round certifies TP=PP=CP=EP=1, so trainer count is the DP size. A config is
illegal when it has no trainer or rollout GPU, does not match the pool size,
uses DP>1 with a dense full-parameter profile, or `global_batch` is not
divisible by DP × micro batch. Illegal configs stay in the table with their
reason; they are never swapped for a neighbour.

Edges are directed and typed: `rollout-only`, `same-shape-restore`,
`trainer-dp`, `role-transfer`, `standby-scale`. Shape checks reject, for
example, a `rollout-only` edge whose trainer count changes.

A capability attestation (`caps.json`) says what the runtime has certified:

```json
{
  "runtime_fingerprint": "sha256:...",
  "execution_modes": ["partitioned-serial"],
  "partitioned_driver": true,
  "certified_edges": [{"source": "P422", "target": "P44", "kind": "rollout-only"}],
  "optimized_paths": [],
  "auto_controller": false
}
```

Each matrix item is `supported`, `unsupported` (illegal config, unknown edge)
or `blocked_dependency` (declared but uncertified edge, unattested execution
mode, missing optimized path or controller). Without an attestation only the
`legacy-fixed` arm is runnable.

## Evidence and resume

Every attempt writes to
`runs/<arm>[@config]/<scenario>/seed-<n>/attempt-<k>/` and ends with
`evidence-index.json` (sha256 and size of every raw file) and `result.json`.
Reuse on resume requires the same `study_hash`, the same matrix key and every
digest to match; a modified, missing or unindexed file is an error rather than
a silent rerun. Results are never overwritten; a new attempt gets a new number
and failed attempts stay on disk.

`result.json` has four layers: `execution` (completed, failed, unsupported,
blocked_dependency, pending), `correctness`, `quality` and `benefit`. Only
completed attempts may carry verdicts. The study summary is `incomplete` while
any supported item lacks a verified completion, and the study-level benefit is
`demonstrated` only when the matrix is complete and every dynamic arm passes
correctness, quality and benefit. Negative results are kept as `negative`.

## Paired work and scenarios

`work.split_rows` draws disjoint train/calibration/test/held-out row sets from
a seeded permutation. `legacy.materialize_paired_inputs` writes the training
stream in the legacy `combined.jsonl` format and refuses any held-out row in
the training assignment. `work.group_seeds` gives every (group, sample) a
stable logical seed so arms can reconcile their inputs.

Phases bind to logical update indexes. `work.phase_schedule` expands them to
one slot per update, and `work.assign_groups` draws prompt ids per update from
calibration-measured length buckets, so a slow and a fast arm see the same
phase sequence. Scenario recipes (`stable`, `phased`, `tail`, `tool-wait`,
`oscillating`) only shape the bucket mix and the environment contract; they
take measured mixes, the measured payback window and the frozen environment
from a calibration record and refuse to run without them. They never change
global batch, group size, optimizer steps or truncation rules.
`work.length_diagnostics` reports real lengths, cap-hit ratio and
zero-advantage ratio from captured samples.

## Negative-test islands (per-island override)

To check on real hardware that a mismatching island is refused, one island
can get another value of a whitelisted parameter. This is for negative tests
only.

```
yeto launch ... --training-mode rl --rl-island-scheduling elastic \
  --gpu modal:1xh100,modal:1xh100 \
  --rl-negative-test-run \
  --rl-island-override 1:identity_test_salt=n12
```

| Key | Effect on the named island |
|---|---|
| `rl_lr_schedule` | another LR schedule (`auto`/`linear`/`constant`); the LR schedule hash is part of the island identity |
| `rl_max_policy_age` | another policy-age limit (落后上限); checked against the backend like the global flag |
| `identity_test_salt` | TEST ONLY: mixed into `island_contract_sha256`, so the island declares a different identity |

`--rl-max-carry-lag` is a syncer parameter and cannot be changed per island
(10-09 ruling of the main agent for the user). Any other key, an island number
out of range, or `--rl-island-override` without `--rl-negative-test-run` stops
the launch before any cloud resource.

**Warnings.** The launcher prints a NEGATIVE-TEST RUN banner. The run manifest
has `negative_test: true` and `island_overrides` (island, key, old, new). The
overridden island gets `YETO_ISLAND_OVERRIDE` (JSON) and writes an
`rl_island_override` event at the head of its tape; the dashboard shows
"负例岛：学习率调度 linear→constant" next to it. Every island of the run gets
`YETO_NEGATIVE_TEST_RUN=1` and writes `YETO_NEGATIVE_TEST` at the root of its
checkpoint store.

**Not for production.** A run without `--rl-negative-test-run` refuses a
`--rl-checkpoint-store` with the `YETO_NEGATIVE_TEST` marker (launcher, when the
store is a path on this machine, plus the local run manifests; the island again
at start, on every cloud). `yeto merge` refuses an adapter directory with the
marker or with a `run_manifest.json` that has `negative_test: true` in it or up
to two levels above. (The original plan named `--rl-resume` and `yeto export`;
they do not exist on main, so the same guard sits on `--rl-checkpoint-store`
and `yeto merge` -- 10-09 ruling of the main agent for the user.)

Without `--rl-island-override` and `--rl-negative-test-run` every island
command and environment is unchanged (`tests/test_decoupling_golden.py`).

## What is not implemented

Runners for the fixed partition, scheduled switching, auto control, the
switching timeline and cost aggregation, the amortization sweep and the
statistical acceptance are pending on the upstream reconfiguration change and
on real calibration runs. See the tasks file in the miles repository for
status.
