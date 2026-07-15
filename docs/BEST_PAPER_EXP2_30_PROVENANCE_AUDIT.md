# EXP2.30 legacy phase-map provenance and evidence audit

**Audit date:** 2026-07-14

**Status:** recovered legacy evidence; admissible for experimental design only

**Auditor:** independent reconstruction from the retained S3 prefix, not from paper tables

## 1. Source recovered

The complete prefix was synchronized read-only from:

```text
s3://yeto-exp-artifacts-533462777468-us-west-2/
  probecommit-resume-20260710/exp2-30-phasemap/
```

AWS account identity at retrieval was `533462777468`. The recovered inventory
contains 27 arm directories, 27 `.arm_done` markers, 27 commands, 27
`results.jsonl` files, and 344 files total (6.8 MB). No missing advertised grid
cell was found.

Checksums of the synchronized snapshot, computed over sorted paths:

| material | aggregate SHA-256 |
|---|---|
| all recovered files | `e128950c24362224a393eee981ddf7af672a6ddbb2233933c50b1e9110efacc6` |
| the 27 `command.sh` files | `63857ec4c0479cdcdab4bc73c2b0f092e901bd60eace9ac32d760d6a9a6e14a7` |
| the 27 `results.jsonl` files | `8281f078eae87731cc70604c19685d6ca75c28fb1d4607ef74c10dee764b9dd1` |

The prefix-level `git_commit.txt` names commit
`89f76a98660669911316ec7e0eab5118d2d03abd`, which exists in the repository.
However, every per-arm `git_commit.txt` says that `git rev-parse HEAD` failed
because the launcher was not inside a Git repository, and every per-arm
`git_diff.patch` contains the corresponding failed `git diff`. The root commit
is useful provenance but does not prove that the exact executed working tree
was clean or identical to that commit.

The prefix reports one `NVIDIA A100-SXM4-80GB`. No immutable machine image,
container digest, model-file digest, dataset digest, tokenized-evaluation hash,
or per-example evaluation artifact was retained.

## 2. Recovered design

All arms used:

- full-parameter `HuggingFaceTB/SmolLM2-135M` training;
- `trl-lib/Capybara`, 5,000 training rows and 64 evaluation rows;
- one development shuffle/training seed, `223 / 223223`;
- four learners, four fragments, strict full quorum;
- inner AdamW at LR `0.001`;
- outer Nesterov with RDA merge and no delta correction;
- bf16 wire values and f32 syncer arithmetic;
- 655,360 total training tokens, sequence length 128, microbatch 1;
- zero injected learner delay and jitter;
- equal total inner work across horizons;
- `H in {16, 64, 256}`, `mu in {0, .5, .9}`, and
  `eta in {.0875, .175, .35}`.

The resulting global commit counts were 320, 80, and 20 for H16, H64, and
H256. With four fragments, H256 therefore supplied only five buffer updates
per fragment.

Important semantic and design limitations:

1. The commands have `--strict-quorum`, but not `--barrier-sync` and not
   `--version-matched-anchor`. This is the older non-barrier/current-anchor
   protocol, not the clean true-lockstep DiLoCo reference.
2. Every command injects `--baseline-loss 0.0`. The report's displayed
   synchronous baseline is fictitious. The actual m4 endpoint remains usable,
   and separate mu=0 cells exist, but there is no live within-run baseline arm.
3. Evaluation has only 64 rows and endpoints are serialized to six decimal
   places. There is no training-seed replication and no empirical
   training-seed variance estimate.
4. The eta blocks were not randomized. S3 timestamps show all `.175` commands
   first, then all `.0875` commands, then all `.35` commands. Eta is therefore
   confounded with launch time/block.
5. `--gpu-slots 1` was used. The exact worker-to-device scheduling differs
   from the proposed four-GPU evidence protocol.
6. Some arms changed `--learner-max-steps` from 30,000 to 2,500, and probe
   capture was enabled only on selected mu=0 arms. The token-budget stop appears
   to have completed, but command parity was not exact.

## 3. Exact endpoint reconstruction

Losses below are ordered as eta `.0875 / .175 / .35`:

| H | mu=0 | mu=.5 | mu=.9 |
|---:|---:|---:|---:|
| 16 | 2.033595 / 2.051313 / 2.109217 | 2.091697 / 2.169113 / 2.286041 | 2.313818 / 2.425225 / 2.523270 |
| 64 | 2.141625 / 2.186610 / 2.298574 | 2.198890 / 2.371415 / 2.520968 | 2.547837 / 2.720555 / 2.873083 |
| 256 | 2.179065 / 2.269186 / 2.398224 | 2.266719 / 2.405362 / 2.619599 | 2.579845 / 2.759056 / 2.964441 |

For every one of the nine `(H, mu)` curves, the lowest loss occurs at the
lowest sampled LR, eta=.0875. All nine curves improve monotonically as eta is
reduced from .35 to .175 to .0875. None has a bracketed LR optimum.

At the common boundary eta=.0875, the momentum penalties relative to mu=0 are:

| H | mu=.5 minus mu=0 | mu=.9 minus mu=0 |
|---:|---:|---:|
| 16 | +0.058102 | +0.280223 |
| 64 | +0.057265 | +0.406212 |
| 256 | +0.087654 | +0.400780 |

Thus the legacy 27-cell result contains **no beneficial momentum crossover,
even at its boundary optimum**. Both tested positive-momentum settings are
worse than mu=0 at all three horizons. The mu=.9 penalty does not shrink from
H64 to H256, and the mu=.5 penalty is largest at H256.

## 4. Evidence judgment

EXP2.30 supports one narrow statement:

> In one non-barrier, current-anchor, seed-223 SmolLM2 run, positive outer
> momentum was harmful throughout the sampled eta range, and every sampled
> LR curve was still improving at the lower eta boundary.

It does **not** show that conventional LR tuning fails to rescue momentum,
because the optimum was not bracketed. It also does not show a helpful-to-
harmful phase transition. Consequently it cannot support paper Section 4.12's
claim that the full-parameter LR gate passes or that the crossover is not an
LR artifact.

The retained commands and tapes are useful for choosing the new low-LR search
range. They are not pooled with new evidence, do not determine a statistical
noise floor, and never count as a confirmation seed.

## 5. Required replacement

The replacement experiment is frozen in
`docs/BEST_PAPER_PHASE_MAP_P0_P1_PREREG.md`. It begins below eta=.0875, uses a
same-protocol upper neighbor, true barrier plus version-matched semantics, a
large locked evaluation set, fresh seeds, randomized Spot blocks, live mu=0
controls, and a deterministic rule that continues downward whenever the
lowest LR remains optimal.
