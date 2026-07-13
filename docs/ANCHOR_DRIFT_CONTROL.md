# The Current-Anchor Confounder and the 3-Arm Causal Control

## What we are actually studying (precise name)
**Strict-quorum, non-barrier, current-anchor streaming DiLoCo variant.**
- SYNC side: quorum=4 of 4 → every commit is an all-worker merge (no
  subset-commit, no straggler-drop). Server commit discipline is synchronous.
- ASYNC side: workers do NOT block on the outer merge — they keep local
  training while the syncer merges/probes; broadcasts applied on arrival;
  worker/syncer compute overlaps. Worker execution is NOT lockstep.
- ANCHOR: syncer differences each pushed model against its OWN CURRENT global
  fragment, not the global the learner started that window from.

Intuition: the server waits for everyone each round, but the workers do not
stand still waiting for the server.

## The confounder (the key point)
learner_upload = old_global + local_change. But the syncer computes
`delta = current_global - learner_upload`, NOT `learner_base - learner_upload`.
So:
```
server_delta = true_local_delta  -  (current_global - learner_base_global)
             = true_local_delta  -  anchor_drift
```
The **anchor-drift term** is NOT a local learning signal — it's contamination
from version mismatch. Even at zero injected delay it can be nonzero whenever
(a) no barrier, (b) merge/broadcast latency is tens of ms, (c) the worker keeps
training in that window. Worse, anchor-drift plausibly correlates with recent
global updates and the momentum buffer — so persistent momentum may repeatedly
accumulate it. The observed poison therefore has ≥3 candidate sources:
1. outer-momentum pathology intrinsic to true barrier DiLoCo;
2. non-barrier execution-overlap / history mismatch;
3. current-anchor differencing (anchor-drift contamination).
Current experiments cannot separate these three.

## The 3-arm control (all else identical: seed, data order, inner steps,
## outer LR, momentum, quorum, total tokens, nominal cadence)
- **A. barrier + version-matched** — original DiLoCo: all workers finish local
  steps, HARD-BLOCK, delta vs the same global each received, merge, outer step,
  resume. Answers: does true vanilla DiLoCo have the crossover?
- **B. non-barrier + version-matched** — current streaming architecture (no
  block, strict quorum, zero injected delay) BUT each learner tags its
  base_version and delta is computed vs its actual training base. Answers: does
  non-barrier overlap alone cause poison even with correct delta semantics?
- **C. non-barrier + current-anchor** — the current implementation. Answers:
  does current-anchor semantics additionally amplify?

## Interpretation table (the reviewer's causal decomposition)
| result | conclusion |
|---|---|
| A, B, C all show the poison | outer-momentum pathology is DiLoCo-family-intrinsic |
| A & B show it, C worse | intrinsic, current-anchor amplifies |
| A clean, B & C show it | caused by non-barrier overlap, not vanilla DiLoCo |
| A & B clean, only C | mainly a current-anchor-delta implementation failure mode |
| A shows it, B/C stronger | barrier DiLoCo has it; streaming nonstationarity worsens |
| none reproduce | effect depends on other impl details; re-localize |

## Required instrumentation (log per push)
- learner base global at window start; syncer current global at receipt.
- `anchor_drift = current_global - learner_base_global`.
- Measure: (1) ||anchor_drift|| / ||true_local_delta||; (2) angle(anchor_drift,
  momentum buffer); (3) overlap of anchor_drift with the transverse component
  (per docs/THEORY.md); (4) does poison grow with the anchor-drift/local-delta
  ratio; (5) can the current-anchor matched-effective-LR residual be explained
  by this term. If the ratio is large in poison cells and small in safe cells →
  a direct implementation-level culprit.

## Claim discipline (paper language)
CAN write: "We study a strict-quorum, non-barrier streaming DiLoCo variant in
which learners keep training while the syncer commits. The short-horizon
outer-momentum crossover persists at zero injected delay, so externally
injected staleness is not necessary. Bounded commit-latency overlap and
current-anchor differencing remain, so these experiments do not yet establish
the same failure in lockstep barrier DiLoCo."
CANNOT write: "Vanilla/fully-synchronous DiLoCo has this problem"; "asynchrony
is innocent"; "variable H / decoupling itself causes the poison."
Best single sentence: *Outer-momentum poison appears even without injected
delay in a strict-quorum streaming DiLoCo variant, ruling out network-induced
staleness as a necessary cause, but leaving non-barrier overlap and
current-anchor differencing as potential contributors.*

## Implementation needed
- `--barrier-sync` (worker blocks after push until broadcast) — being added in
  the barrier workstream (learner.py).
- `--version-matched-anchor` — learner tags base_version on push; syncer
  differences vs that base, not current global. NEW (learner.py push +
  syncer/server.rs delta construction; default byte-identical).
- anchor-drift logging in the merge/commit path.

## Run (scheduled: launch when a GPU frees; do NOT add a node while fleet busy)
Arms A/B/C × crossover corners {H16-mu0, H16-mu09, H256-mu0, H256-mu05},
seed 223/223223 sync, capture ON, >=12sigdigit loss. S3 prefix
exp2-46-anchorctl. On 1x RTXPRO6000 96GB or AWS p4d (fits 4 learners).
