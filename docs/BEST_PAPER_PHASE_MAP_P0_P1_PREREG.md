# Best-paper phase map: frozen P0/P1 preregistration

**Frozen:** 2026-07-14, before any P0 or P1 model-derived evaluation loss exists

**Scope:** full-parameter SmolLM2 low-LR gate

**Machine-readable companion:**
`experiment-specs/best-paper-phase-map-p0-p1-prereg.json`

This protocol replaces EXP2.30 for all claims about tuned outer momentum on a
full-parameter model. Legacy EXP2.30 is design evidence only; its endpoints
must not be pooled into any estimate below.

**Pre-outcome audit-set and packing-canary amendment (2026-07-14).** Independent
statistical and infrastructure review, before any canary or scientific run,
required two additional safeguards. First, development tuning and final
confirmation now use two disjoint 1,024-example sets; the confirmation-audit
set receives no model evaluation until every P3 training cell is resolved and
the complete checkpoint registry is sealed. Second, a short four-A100
packing/barrier canary (P0b) must pass after
the one-A100 semantic canary and CPU replay (P0a), and must itself be torn down
and replayed before P1. These changes alter no scientific axis, estimand,
threshold, development seed, confirmation seed, or analysis rule. Sections
2--4 and 9--11 below and companion schema `0.2` are the amended authority.

## 1. Question and estimands

The first question is deliberately fatal to the current paper framing:

> After independently bracketing outer LR for each `(H, mu)`, does a genuine
> helpful-to-harmful outer-momentum transition remain?

For horizon `H` and momentum `mu`, define the development tuned endpoint

```text
T(H, mu) = minimum locked-eval NLL over the preregistered eta bracket.
```

The two primary development contrasts are:

```text
D_short = T(16, .9) - T(16, 0)
D_long  = min[T(256, .5), T(256, .9)] - T(256, 0)
```

Positive `D_short` means high momentum hurts at short H. Negative `D_long`
means some tested momentum helps at long H after LR tuning. H64 is the
predeclared intermediate-horizon check, not a substitute for either primary
corner.

P1 is development, not confirmation. A one-seed sign pattern can advance the
campaign but cannot appear in the paper as seed-robust evidence.

## 2. Frozen common protocol

The following must be identical across scientific arms except for the
registered `H`, `mu`, `eta`, seed, output path, and randomized order:

| item | frozen value |
|---|---|
| model | `HuggingFaceTB/SmolLM2-135M`, full parameter |
| model revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| data | local pinned `trl-lib/Capybara` parquet; hash bound before P0a |
| train/eval | 5,000 train rows; 1,024 locked development-eval rows; 1,024 disjoint locked confirmation-audit rows |
| sequence / microbatch | 128 / 1 |
| learners / fragments | M=4 / 4, full quorum |
| inner optimizer | AdamW, LR `0.001`, persistent state |
| outer optimizer | repository Nesterov implementation |
| merge | explicit `--matrix-merge rda`; resolved per-fragment modes sealed |
| correction | `--delta-correction none` |
| arithmetic | bf16 wire / f32 syncer |
| schedule | fixed padded windows, zero injected delay/jitter |
| synchronization | `--strict-quorum --barrier-sync --version-matched-anchor` |
| total work | 655,360 training tokens per arm |
| device | one GCP `a2-highgpu-4g` scientific VM, four A100 GPUs |
| provisioning | Spot only; no on-demand fallback |

For H16/H64/H256, respectively:

| H | fixed-window microsteps | fixed-window tokens | global commits |
|---:|---:|---:|---:|
| 16 | 16 | 2,048 | 320 |
| 64 | 64 | 8,192 | 80 |
| 256 | 256 | 32,768 | 20 |

The H256 result has only five buffer updates per fragment. P1 can answer the
fixed-budget LR gate, but it cannot validate a stationary autocorrelation law.
A later equal-outer-commit experiment is mandatory before making stationary or
universal-kernel claims.

Before P0a, bind and seal a clean pushed 40-hex Git commit, numeric image ID
`7290368630472593484`, model-file hash, parquet hash, train-row hash,
the separate development/audit source-index, canonical-row, example-ID,
packed-sequence, and token hashes, full command hash, randomization-plan hash,
and an independent retry-policy hash. A retry authorization must cite the
retry-policy hash; it must never reuse the randomization hash. Placeholders in
the companion JSON are not launch authority.

`frozen.retry_policy_hash` is exactly SHA-256 over the UTF-8 bytes of the final
top-level `retry_policy` object serialized as JSON with lexicographically
sorted keys, no insignificant whitespace (`separators=(",", ":")`), and
`ensure_ascii=false`. It is not the hash of a launcher helper or a reduced
projection of the policy.

**Pre-outcome retry clarification (2026-07-14).** Independent launcher/
validator review found an ambiguity in the whole-block retry rule before P0a or
P1 was launched. This clarification changes no outcome, grid, seed, gate, or
analysis rule. It is part of the authoritative preregistration commit that must
be bound before P0a: a completed peer in a block containing an infrastructure
failure remains `COMPLETED`, with its original loss and artifacts retained. It
must never be relabeled as `INFRA_FAILURE`. The frozen retry-only reason
`peer_block_invalidated_by_infra_failure` permits the mandatory whole-block
rerun under the exact conditions in Section 7.

`frozen.command_hash` is the hash of the canonical campaign command template,
not a claim that all 36 expanded argvs are identical. The bound descendant
must additionally contain a complete immutable `cell_command_hashes` object
mapping every expected `cell_id` to the SHA-256 of its exact argv. Each result
must match its cell-specific hash. Missing, extra, or duplicate map keys fail
before launch.

Every bound manifest also materializes an explicit `expected_cells` coordinate
list. P1-R0 must equal the 36-cell Cartesian grid, but this list—not a Cartesian
axis expansion—is authoritative for coverage. Adaptive and later tuned stages
are ragged across H and eta: they preserve parent coordinates, add only
rule-derived coordinates, and keep complete randomized blocks with a live
mu=0 control. The command-hash registry keys must equal the expected cell IDs
exactly.

**Pre-outcome split and lineage clarification (2026-07-14).** A second
independent launcher/validator review found that applying each study seed to
the whole row population would change the evaluation set across training
seeds. No P0a or scientific outcome existed when this was found. The campaign
therefore freezes `eval_split_seed=331` and the exact split algorithm below.
It also requires every bound manifest to identify the raw authoritative
preregistration blob and every later manifest to cite its sealed parent. This
clarification changes no hypothesis, outcome, grid, gate, or analysis rule.

The authoritative template is the raw bytes at
`experiment-specs/best-paper-phase-map-p0-p1-prereg.json` in a clean pushed
source commit. Each bound descendant records that path, commit, and raw-byte
SHA-256. Validation retrieves the blob with `git show <commit>:<path>` and
requires both its bytes and hash to match; pointing `--prereg-template` at a
different file is forbidden. These bindings live in the descendant, so the
template does not self-reference its own commit or hash.

Parent-manifest hashes use SHA-256 of canonical JSON with the same sorted-key,
no-whitespace, `ensure_ascii=false` convention as the retry-policy hash. P0a is
the only parentless bound canary. P0b must cite the sealed passing P0a manifest
and the raw-file SHA-256 of its sealed post-deletion CPU replay report. P1-R0
must cite the sealed passing P0b manifest and the raw-file SHA-256 of P0b's
sealed post-deletion CPU replay report. Every subsequent registered stage must
cite a sealed parent. Existing parent result rows are immutable. An
adaptive-bracketing descendant is cumulative: its parent rows remain an exact
prefix, its eta and command registries may only be extended by the next
Section 6 boundary/midpoint rule, and only new result rows may be appended.
Changing an old result, an unregistered eta, model/data/evaluation/protocol
semantics, retry policy, or horizon/work invalidates the descendant.

## 3. Locked evaluation

There are two outcome sets with different roles:

- **development evaluation:** 1,024 examples used by P0a/P0b semantic checks
  and by P1/P2 LR tuning and promotion gates;
- **confirmation-audit evaluation:** 1,024 additional examples, disjoint from
  training and development evaluation, used as the sole primary endpoint for
  P3. No model-derived loss, per-example loss, prediction, or other
  outcome-dependent statistic may be computed on this set before the P3 audit
  authorization in Section 9.

Split construction is exact. Load the pinned parquet in canonical source-row
order and take source indices `0,...,7047` (the first
`train_rows + development_eval_rows + audit_eval_rows = 7,048` rows). Create
that integer list and apply `random.Random(331).shuffle` exactly once. In the
resulting order:

```text
positions [0, 5000)    = pre-study-shuffle training pool
positions [5000, 6024) = locked development evaluation
positions [6024, 7048) = locked confirmation-audit evaluation
```

No second split or role-assignment shuffle is permitted. For each registered
study shuffle seed, copy only the 5,000-index training pool and apply
`random.Random(shuffle_seed).shuffle`; neither evaluation index list nor its
order changes. P0a and P0b both use shuffle/training seed `337 / 337337`, P1
uses `347 / 347347`, and later stages use only their registered seeds.

Before P0a, bind separate hashes for the training-pool, development, and audit
source-index arrays. For each evaluation role also bind: (a) the UTF-8 bytes of
canonical compact JSONL source rows in frozen order (sorted object keys, one
object plus `LF` per row); (b) the canonical compact JSON array of example
IDs; (c) the complete packed artifact containing input IDs, labels, attention
masks, supervision weights, lengths, and target-token masks/counts; and (d)
the canonical compact JSON token-ID arrays. These are the distinct
`*_source_indices_hash`, `*_rows_hash`, `*_example_ids_hash`, `*_packed_hash`,
and `*_token_ids_hash` fields in the companion JSON. Before each stage, bind
maps from every expected study seed to its ordered training-index-array hash
and materialized training-JSONL hash. Also bind `audit_access_policy_hash` as
SHA-256 of the exact top-level `confirmation_policy` object serialized with
the same canonical JSON convention as the retry policy.

Loss-free split preparation and hashing do not count as opening the audit set.
"Opened" means that any checkpoint/model is evaluated on an audit example or
that any resulting loss, prediction, or statistic is exposed to a person or
outcome-aware process. P0a, P0b, P1, P2, and all P3 *training* commands must be
structurally incapable of naming the audit artifact or an audit-output path;
their exact command registries are checked for this property. Through P2 the
audit-access log is empty, audit outcome fields and URIs are null, and only the
pre-bound loss-free hashes exist.

Requirements:

- retain every aggregate loss at at least 12 significant digits;
- retain per-sequence loss sum, target-token count, source index, example ID,
  and token hash for row-aligned paired evaluation;
- do not regenerate the evaluation split after a failure or preemption;
- report paired-example uncertainty as evaluation uncertainty only;
- never call evaluation-row uncertainty training-seed uncertainty;
- never reuse `0.009` as a noise estimate.

Development-set endpoint NLL is primary for P1/P2 decisions only. Audit-set
endpoint NLL is primary for P3 confirmation only. Training loss, development
loss on P3 checkpoints, curve AUC, tape geometry, rho, aligned gain, and
transverse energy are secondary diagnostics and cannot replace a failed audit
endpoint gate.

## 4. P0a/P0b: loss-blind semantic and packing canaries

Both canaries are explicitly **non-evidence**. Their losses remain quarantined
and never enter an effect estimate, LR choice, gate, figure, or paper table.
Both use the same shuffle/training seed `337 / 337337`, the same development
evaluation, and exactly one complete block
`H=16, eta=.0875, mu in {0,.5,.9}`. Both use an abbreviated 65,536-token
schedule: 32 global commits at H16, eight applied updates per fragment. Raw
syncer capture is required at every applied step.

**P0a (single-GPU semantic canary).** Run first on one Spot A100 using
`a2-highgpu-1g`, `gpu_slots=1`. After artifacts are sealed, delete the VM,
boot disk, and accelerator; obtain provider not-found/zero-accelerator proof;
then run the independently sealed CPU RDA/Nesterov recurrence replay over every
captured step. P0a is the only parentless stage.

**P0b (four-GPU concurrency/barrier packing canary).** Only after P0a and its
post-deletion replay pass, run the identical canary workload on one Spot
`a2-highgpu-4g` using `gpu_slots=4`. P0b cites the canonical sealed P0a
manifest and raw P0a replay-report hash. P0b may differ from P0a only in its
registered stage/output/lineage identifiers and the machine/gpu-slots fields;
its code, image, model, data, splits, seed, H, eta, mu block, work, optimizer,
merge, barrier, version, dtype, and normalized workload argv must match P0a.
Every P0b attempt must seal:

- the `nvidia-smi` CUDA-index-to-GPU-UUID inventory;
- each learner's resolved CUDA index, GPU UUID, and learner ID;
- proof of exactly four distinct A100 UUIDs and a bijection from the four
  learners to those four UUIDs; and
- the barrier/version trace showing that no learner begins its next window
  before the registered broadcast and all four base versions match.

After P0b artifacts are sealed, delete its VM, boot disk, and all four
accelerators, obtain exact-ID not-found/zero-accelerator proof, and run the
same independent frozen-tolerance CPU replay over every captured step. P1 is
blocked until the sealed P0b manifest and its post-deletion replay both pass.

Together P0a and P0b verify:

1. all four learners participate in every full-quorum commit;
2. barrier execution prevents a learner from starting the next window before
   the registered broadcast;
3. every admitted push is differenced against its recorded base version;
4. fixed-window `c_steps=H` and `c_tokens=128H` hold for every responder;
5. RDA replay and the Nesterov recurrence reproduce the applied step;
6. the resolved merge mode, fragment layout, flags, hashes, and work counters
   appear in sealed artifacts;
7. no injected historical baseline is present;
8. development per-sequence evaluation sums reproduce the aggregate loss and
   no audit-evaluation read or output exists;
9. the Spot VM, GPU, disk, image, command, and teardown evidence are complete.

Any P0a or P0b semantic, packing, replay, or teardown failure blocks P1 until
fixed, followed by a new P0a -> replay -> P0b -> replay chain with new immutable
manifests. Fixes cannot be made silently under the old preregistration hash.

## 5. P1-R0: initial low-LR grid

P1-R0 uses the fresh development shuffle/training seed `347 / 347347`.
Repository search found no prior use of this seed family at freeze time.
It may launch only from the clean commit that produced the passing P0a/P0b
chain and only after the sealed post-deletion P0b replay is cited in its bound
manifest. P1 uses the locked development evaluation; it has no audit-evaluation
path or outcome field.

The exact Cartesian grid is:

```text
H   = {16, 64, 256}
mu  = {0, .5, .9}
eta = {.021875, .04375, .0875, .175}
```

This is 36 expected cells. Eta=.175 is mandatory: it is the new-protocol upper
neighbor needed to avoid declaring eta=.0875 bracketed merely because legacy
EXP2.30 sampled it. Every `(H, eta, seed)` block contains a live mu=0 arm and
the two momentum arms. No `--baseline-loss` or imported absolute loss is
permitted.

## 6. Sequential bracketing rule

Adaptation is allowed only on the P1 development seed and only through this
algorithm. Each adaptive round receives a new immutable expected-grid manifest
before launch; the P1-R0 manifest is never edited.

For every `(H, mu)` independently:

1. After all 36 P1-R0 cells are valid and sealed, find the eta with lowest
   high-precision locked-eval NLL.
2. If eta=.021875 is best, add eta=.0109375. If the new lower boundary is best,
   continue halving through `.00546875`, `.002734375`, and `.0013671875`, one
   sealed round at a time.
3. If the best eta remains the lower boundary after `.0013671875`, label that
   `(H, mu)` **UNBRACKETED-LOW** and fail the tuned-LR gate. Do not extrapolate
   an optimum or open later seeds.
4. If eta=.175 is best, add eta=.35, then eta=.7 if needed. If eta=.7 remains
   best, label the cell **UNBRACKETED-HIGH** and fail the gate.
5. Once an interior point is best, add the geometric midpoint between it and
   each adjacent sampled neighbor exactly once. The final development choice
   is the lowest point estimate on this refined grid; all adjacent losses and
   paired-example intervals are reported.
6. Divergent hyperparameters remain scientific outcomes and are assigned
   infinite loss for tuning. Infrastructure failures remain missing until a
   loss-blind allowed retry succeeds.

No legacy loss enters this algorithm. No seed 359, 373, or confirmation seed
may be evaluated or unblinded while P1 bracketing is in progress.

## 7. Randomization, Spot preemption, and retries

P1 runs on Spot capacity only. There is no on-demand fallback. Scientific
blocks are `(H, eta, seed)` triples containing mu=0/.5/.9. The 12 initial
blocks are put in a committed pseudorandom order; momentum order is independently
permuted within each block. The randomization plan is materialized and hashed
before launch.

This blocking keeps every momentum result close in wall time to its live mu=0
control. Hardware identity, zone, start/end time, attempt, and order index are
recorded so residual time-block or preemption effects are auditable.

Allowed retry triggers are loss-blind and mechanical only:

- provider-reported Spot preemption;
- VM/host/GPU failure;
- process exit before a scientific divergence is recorded;
- missing or checksum-invalid required artifact;
- validator-detected command/provenance mismatch before outcomes are opened.

When any arm has a genuine allowed infrastructure failure, the entire
three-arm block is rerun from identical initial state, seed, command, image,
and data on the same machine/GPU class. Only the genuinely failed arm is marked
`INFRA_FAILURE`; any peer that completed remains `COMPLETED`, and its loss,
status, and artifacts are retained unchanged. A completed peer may be rerun
only with retry reason `peer_block_invalidated_by_infra_failure`. That string
is a retry-authorization reason only: it is never a `failure_reason` and never
licenses relabeling a completed attempt.

Every arm in the repeated block must be present in the same retry round. Each
new attempt is linked by `retry_of` to the immediately prior attempt for that
cell. All three new attempts must carry the same loss-blind
`retry_authorization`, created before any aggregate or per-example outcome in
the prior block is opened, containing at least:

- `loss_blind: true` and the independently frozen `policy_hash`;
- `trigger_attempt_id`, identifying a genuine `INFRA_FAILURE` attempt in the
  immediately prior instance of the same block;
- `trigger_reason`, equal to that attempt's allowed mechanical failure reason;
- `trigger_block_id`, equal to the repeated `(H, eta, seed)` block ID; and
- `prior_manifest_sha256`, binding the sealed prior block-attempt manifest.

`prior_manifest_sha256` is SHA-256 over the UTF-8 canonical JSON of the exact
campaign manifest prefix immediately before the retry block, using the same
sorted-key, no-whitespace, `ensure_ascii=false` convention as the retry-policy
hash. From the final manifest, the preimage is reconstructed by removing the
contiguous three-row retry-block suffix in question and every later result row,
while leaving all non-`results` fields identical. Retry-block rows must
therefore be contiguous and result acquisition is append-only. Mechanical
serialization and sealing of outcomes into that prefix is allowed; "opened"
means exposed to a human or outcome-aware analysis. The shared authorization
must be created after the prefix is sealed but before that exposure.

The validator must establish that the cited trigger exists in the immediately
prior complete block round, has status `INFRA_FAILURE`, and has the cited
direct infrastructure-failure reason; that the prior-manifest hash matches;
and that all mu arms are rerun. An `INFRA_FAILURE.failure_reason` must come only
from `direct_infrastructure_failure_reasons`, never from the peer-only retry
reason. All attempts are retained. A finite completed loss, a poor loss, or a
preregistered hyperparameter divergence is never a direct or peer retry
trigger.

## 8. P1 go/kill decisions

The following numerical margins are prospective resource/importance gates,
not claims about a measured noise floor.

P1 may advance to the three-development-seed stage only if:

1. all semantic, provenance, coverage, evaluation, and live-control validators
   pass;
2. all nine `(H, mu)` LR curves are bracketed under Section 6;
3. `D_short >= +0.020`; and
4. `D_long <= -0.010`.

Interpretation of failures is frozen:

| outcome | required conclusion |
|---|---|
| D_short < +.020 after tuning | the headline short-H effect is substantially an LR-matching effect in this full-parameter setting; kill the claimed tuned poison |
| D_short passes, D_long >= 0 | high momentum can be harmful, but there is no tuned beneficial crossover; kill the phase-transition/controller framing |
| -.010 < D_long < 0 | at most a weak exploratory long-H benefit; insufficient for the best-paper gate |
| any LR curve unbracketed | no tuned comparison is allowed; extend only as Section 6 permits, otherwise stop |
| missing/invalid live control | affected block is inadmissible, not evidence |

Regardless of sign or size, P1 alone cannot corroborate robustness.

## 9. Advancement to three and eight seeds

### P2: three-seed development

If P1 passes, add unopened development seeds:

```text
shuffle/training = 359/359359 and 373/373373
```

For each `(H, mu)`, freeze the P1-selected eta and its immediate lower and
higher bracket neighbors before launching either new seed. Run all expected
cells on both seeds, retain live mu=0 controls, and unblind seeds 359 and 373
only after both complete and validate. P1 seed 347 plus these two seeds form a
three-seed **development** set.

Advance only if the short and long contrasts have their registered signs in at
least two of three seeds and their three-seed means still satisfy
`D_short >= +.020` and `D_long <= -.010`. This is a promotion gate, not final
inference. Any formula, controller, selected momentum, LR, contrast, tolerance,
or analysis changed after P2 requires a completely new confirmation seed set.

### P3: eight fresh confirmation seeds

The untouched confirmation shuffle seeds are:

```text
383, 397, 409, 421, 433, 443, 457, 461
```

Their corresponding training seeds are the duplicated decimal forms
`383383`, ..., `461461`. They are not run, evaluated, or unblinded during P1 or
P2.

Before P3, freeze one short-H pair and one long-H pair from P2, their exact
etas, all code/config/hashes, and a complete expected-grid manifest. Each pair
includes a fresh live mu=0 arm for every seed. P3 then uses a mandatory
two-phase train-then-audit protocol:

1. **Train and seal, without evaluation.** Run every preregistered P3 training
   cell without mounting, naming, or reading either evaluation artifact. A P3
   `COMPLETED` training row has `loss=null`, no per-example loss URI, and a
   sealed final-checkpoint URI/hash plus complete attempt/provenance data. A
   scientifically nonfinite run is retained as `DIVERGED` and is assigned
   infinite loss by the frozen analysis; it is never retried as infrastructure.
   Resolve all loss-blind mechanical retries, complete every one of the eight
   paired seed blocks, and seal a canonical registry of all expected cells,
   final attempt IDs, statuses, checkpoint URIs/hashes (null for registered
   divergences), command hashes, and completion timestamps. No sequential
   stopping is allowed.
2. **Authorize one complete audit batch.** Only after the full training
   registry is sealed, create a shared loss-blind
   `audit_unblind_authorization`. It binds the canonical P3 manifest hash,
   complete checkpoint-registry hash, exact cell-to-audit-command registry
   hash, committed audit randomization-plan hash, the maximum training
   completion timestamp, and its own creation timestamp. The audit start must
   be later than both that creation time and every training completion time.
3. **Evaluate all checkpoints, expose none partially.** A mechanical evaluator
   processes every eligible final checkpoint on the frozen confirmation-audit
   set in the committed order. It records at least 12-significant-digit
   `audit_loss`, row-aligned per-sequence loss sums/counts/identities, exact
   checkpoint and audit-command hashes, and timestamps. The audit result IDs
   must cover the expected P3 cell IDs exactly; registered divergences receive
   explicit divergence audit rows and infinite loss in analysis. No aggregate,
   per-sequence loss, prediction, log excerpt, ranking, or partial summary is
   exposed to a human or outcome-aware process until the entire audit bundle
   validates and is sealed. Only then may one shared unblind timestamp be
   recorded. A mechanical audit failure retains the hidden failed attempt and
   permits only a loss-blind whole-batch rerun from the same checkpoint
   registry; a scientific value never triggers an audit retry.

`audit_loss` is the sole primary P3 outcome. Development loss, if computed
after the sealed audit is unblinded, is labeled post-confirmation secondary
and cannot alter inclusion, tuning, or inference.

The two co-primary seed-level paired **audit-loss** contrasts use
Holm-adjusted two-sided 95% confidence intervals. Confirmation requires:

- short-H mean penalty at least +.020 and adjusted lower endpoint above zero;
- long-H mean benefit at most -.010 and adjusted upper endpoint below zero;
- no provenance, coverage, semantic, or live-control failure.

A failed co-primary contrast is reported as a failure. Evaluation-example
bootstraps may accompany but cannot replace the eight training-seed pairs.

## 10. SNOO and Outer-Momentum Restarting reconciliation

P1 is not a direct replication of either neighboring result, and the paper
must not call them contradictory.

- SNOO (`arXiv:2510.15830`) reports beneficial pseudo-gradient Nesterov in an
  M=1, jointly tuned, long-run C4 pretraining regime over step frequencies up
  to 400 and model sizes from roughly 125M to 1B, with much larger token
  budgets. M=1 removes cross-worker averaging entirely.
- Outer-Momentum Restarting (`arXiv:2605.28585`) uses Llama-150M on C4,
  sequence length 2,048, M=2, about 3.3B tokens, H in
  `{64,128,512,1024,2048}`, outer LR in `{.1,.3,.5,.7,.9,1.1}`, and momentum
  in `{.1,.3,.5,.7,.9}`. Its main claim is restart robustness across the
  hyperparameter region, not that every no-restart momentum arm is optimal.

These regimes differ from legacy EXP2.30 in worker count, data/objective,
sequence length, token budget, H range, LR range, synchronization semantics,
and tuning protocol. A universal paper claim therefore requires two later
reconciliation blocks after P3:

1. **M-axis bridge:** repeat the frozen tuned phase-map comparison at M=1 and
   M=4 on the same model/data, measuring the full temporal kernel and realized
   buffer geometry. This tests whether SNOO's benefit follows the M-induced
   temporal regime rather than contradicting it.
2. **OMR benchmark bridge:** run a faithful C4/Llama-150M, M=2 long-horizon
   block covering at least H64 and H2048, with tuned no-restart Nesterov and
   the authors' hard-restart baseline under equal tuning budgets. The full
   `{64,...,2048}` axis is required before claiming horizon-wide superiority or
   that a new controller subsumes restarting.

Until those blocks run, conclusions are scoped to the frozen SmolLM2/Capybara
M=4 setting.

## 11. Required evidence bundle

Every expected arm, including failures and divergences, must have:

- immutable study/round/cell ID and exact command;
- canonical campaign-command hash plus a complete expected-cell-to-command-
  hash map bound before launch;
- H, mu, eta, shuffle seed, training seed, attempt, block, and order index;
- clean Git commit and image/model/data/split-role/command/randomization hashes;
- explicit evaluation role (`development`, `none`, or `confirmation_audit`)
  with only the fields permitted for that stage;
- machine type, GPU index/UUID identity, Spot status, zone, VM/disk numeric IDs
  and nonce;
- barrier, version-match, strict-quorum, RDA, correction, dtype, and delay flags;
- target and observed tokens, microsteps, global commits, and per-fragment
  update counts;
- start/end timestamps and infrastructure/scientific status;
- high-precision development endpoint and per-sequence artifact for P0a/P0b/P1/P2,
  null outcome fields for P3 training, or high-precision audit endpoint and
  audit per-sequence artifact for the sealed P3 audit phase;
- tape/capture URI and SHA-256;
- paired live-control cell ID;
- retry lineage and mechanical retry reason, if any;
- GPU teardown and zero-accelerator proof after artifact sealing.

P0b additionally requires the sealed four-GPU inventory, per-learner
CUDA-index/UUID bijection, and packing/barrier trace. P3 additionally requires
the complete sealed checkpoint registry, audit-command registry,
randomization plan, shared audit authorization, audit bundle, and timestamps
proving `all training complete < authorization < audit start < bundle seal <
unblind`. The canonical/raw hashes in each lineage hop and both canary replay
reports are retained.

Unexpected, duplicate, missing, silently retried, or hash-mismatched cells fail
validation. Infra failures are not scientific outcomes; scientific divergences
are not infra failures.
