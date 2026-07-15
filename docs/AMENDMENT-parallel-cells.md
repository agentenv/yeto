# Concurrency amendment: parallel scientific cells

**Version:** 1.0, prospective pre-outcome amendment

**Status:** This document amends only the scheduling, ownership, retry execution,
and evidence aggregation rules for the stages named in Section 2. It is not a
cloud-launch authorization, does not revive any stopped attempt, and does not
make an implementation launch-ready. A launch still requires the lineage,
canary, implementation, packet, review, and explicit user/root gates in
Section 11.

## 1. Authority, precedence, and unchanged scientific question

The underlying authority remains the raw JSON blob at commit
`16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80`, path
`experiment-specs/best-paper-phase-map-p0-p1-prereg.json`, raw SHA-256
`7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b`,
and its prose companion at that commit,
`docs/BEST_PAPER_PHASE_MAP_P0_P1_PREREG.md`. Each future bound descendant that
uses cross-cell parallelism MUST identify this amendment by repository path,
reviewed source commit, and raw-byte SHA-256, in addition to the existing
preregistration lineage.

The original preregistration controls every matter not explicitly changed
here. In particular, this amendment does **not** change any model, data,
evaluation split, seed, cell coordinate, optimizer, work budget, estimand,
threshold, divergence rule, live-control requirement, P3 blinding rule, or
analysis rule. It changes the registered maximum cross-cell width from one to
four for the included stages and freezes the previously missing cross-VM
schedule and evidence rules.

The need for an amendment is the binding concurrency ruling in
`STAGE-PLAN.md` lines 47--55: the frozen four-GPU canary is concurrency *within
one cell*, the current runner executes cross-cell work serially, and the
currently legitimate cross-cell width is one. `EXPERIMENT-PROGRAM.md`'s
Statistical protocol requires arm-to-machine, launch-order, and time-block
randomization while preserving hardware/time-block pairing; those requirements
become binding for the included stages through this amendment.

Normative terms `MUST`, `MUST NOT`, `REQUIRED`, and `FAIL` are literal
validation requirements. A missing or unknown required field is a `FAIL`, not
permission for an operator choice.

## 2. Scope and stage ruling

The only stage codes authorized by this amendment are the following closed
vocabulary:

| `stage_code` | Cross-cell ruling | Atomic launch wave |
|---|---|---|
| `p1r0` | Included | One registered `(H, eta, seed=347)` block: all three `mu={0,.5,.9}` cells |
| `p1ad` | Included only within one already-bound adaptive descendant | One newly authorized `(H, eta, seed=347)` block: all three `mu={0,.5,.9}` cells |
| `p2` | Included | One registered `(H, eta, seed)` block: all three `mu={0,.5,.9}` cells |
| `p3t` | Included for training only | All four registered training cells for one fresh seed: the short pair and long pair together |

The following are excluded: P0a, P0b, P3 audit, M-axis work, OMR bridge work,
and every other future stage. P0a remains one cell on one
`a2-highgpu-1g`; P0b remains one cell at a time on one
`a2-highgpu-4g`. P3 audit has a separate hidden whole-batch protocol and no
concurrency authority here. An excluded stage has cross-cell width one if an
existing authority already permits it, and width zero if it is not otherwise
launch-ready.

### P1-R0 shardability ruling

P1-R0 **is shardable under this amendment**. Its 36 cell coordinates are the
complete, non-adaptive Cartesian product frozen before launch. Parallel
execution therefore cannot select which P1-R0 cells exist. It consists of
exactly 12 randomized three-cell waves, one for each `(H, eta, seed=347)`
block. All three live-control arms start in the same wave, so the registered
blocking rationale is preserved. The fourth VM is deliberately idle during
each P1-R0 wave; P1-R0's realized width is three even though the campaign
ceiling is four.

P1 adaptive bracketing is serial **between descendant rounds**. No cell whose
coordinate depends on an unsealed loss may be materialized, provisioned, or
launched. After a prior round is completely sealed and the registered Section
6 rule has deterministically produced a new immutable descendant, the new
three-arm blocks in that one descendant may use the `p1ad` schedule. No wave
from round `r+1` may overlap any work, retry, sealing, unblinding, or analysis
from round `r`. Thus the outcome-dependent boundary remains serial while the
non-adaptive work inside a bound round may be parallel.

P2 uses three-cell live-control waves and therefore has realized width three.
P3 training uses all four cells for a seed in one wave and therefore may reach
realized width four. The eight P3 seed waves are complete before any audit
authorization; no partial training or evaluation outcome may be exposed.

## 3. Hard capacity ceiling

The operator ceiling is frozen as:

```text
logical VM slots                         = {v0,v1,v2,v3}
maximum concurrent scientific cells     = 4
maximum campaign-owned attached A100s   = 16
scientific VM shape                      = a2-highgpu-4g
A100s per scientific VM                  = 4
active scientific cells per VM          = 0 or 1
learners per cell                        = 4
learner-to-GPU allocation                = one learner per distinct A100 UUID
provisioning                             = SPOT
on-demand fallback                       = forbidden
automatic restart                        = false
on-host maintenance                      = TERMINATE
instance termination action              = DELETE
boot disk auto-delete                    = true
project                                  = model-training-497007
zone                                     = us-central1-c
```

“Concurrent” means that the half-open scientific execution intervals
`[scientific_started_at, scientific_ended_at)` overlap. Preparing immutable
inputs does not create an active cell. A process that has begun model training,
checkpoint loading for training, or development evaluation is active until it
has written a terminal attempt record and exited.

Every campaign-owned VM with attached A100s counts toward the 16-accelerator
ceiling whether it is busy, idle, stopped, preempting, or awaiting teardown.
There is no fifth warm spare. A replacement generation MUST NOT be provisioned
until the replaced generation has an exact-ID terminal provider record and the
provider census proves that creating the replacement cannot exceed four
campaign VMs or 16 attached A100s. Before every wave, during every replacement,
and before the campaign seal, the controller MUST prove both caps from its
state and from provider evidence. Any observed excess is `FAIL`; no result
acquired while over either cap is admissible.

At most one attempt of a given `cell_id` may be active. At most one cell may be
active in a logical slot. A new planned wave cannot start until every cell in
the preceding wave has a mechanically sealed terminal attempt record and any
required immediate retry wave has resolved.

## 4. Canonical roster, fixed seed, and deterministic randomization

All scheduling is materialized before the first included scientific VM is
started. The operator may not reroll, rebalance, swap, pack, skip, or reorder a
materialized plan for capacity, convenience, observed duration, failure rate,
or outcome.

The original scientific plan remains authoritative for cell coordinates,
block membership, live-control links, exact commands, and target work. For an
included stage, this amendment supersedes only the serial execution meaning of
its block/within-block order indices: those old indices remain audit metadata,
while `time_block_index`, seeded slot assignment, and launch-order index below
are the temporal execution authority. The newly bound scientific and parallel
plans MUST cross-reference the same cells and blocks exactly; a validator may
not enforce the old one-`subprocess.run` sequence against a parallel launch.

### 4.1 Canonical encoding and roster

`canonical_json(x)` means UTF-8 JSON with object keys sorted
lexicographically, `separators=(",", ":")`, and `ensure_ascii=false`.
`sha256(x)` means the lowercase hexadecimal SHA-256 digest of the specified
bytes.

Before schedule construction, build `parallel_roster_v1` with exactly these
fields:

```text
schema
stage_code
study_id
descendant_kind
authoritative_prereg_template_sha256
parent_manifest_sha256
parent_expected_cell_ids_hash
cumulative_expected_cell_ids_hash
launch_cells
```

For `p1r0`, `launch_cells` is the descendant's 36 expected cells. For `p1ad`
and `p2`, it is the exact set difference between the cumulative child's
expected cell IDs and the sealed parent's expected cell IDs; the parent IDs
and result rows MUST remain an exact immutable prefix. For `p3t`, it is the 32
new P3 training cells and excludes every inherited P2 cell. The two ID hashes
are SHA-256 over canonical JSON arrays of the respective IDs sorted by UTF-8
bytes. This distinction forbids both rerunning inherited scientific rows and
mistaking a cumulative descendant for a launch list.

`launch_cells` is sorted by `cell_id` and contains exactly these fields per
row:

```text
cell_id, block_id, h, mu, eta, seed, training_seed,
paired_control_id, command_hash, normalized_workload_command_hash
```

There may be no duplicate cell ID or coordinate. The roster cell IDs and
launch-command-hash keys MUST equal the derived launch set exactly, and every
key/value MUST match the same row in the cumulative bound descendant.
`roster_hash = sha256(canonical_json(parallel_roster_v1))`.

### 4.2 Seed and rank function

The one 256-bit master seed is fixed as the hexadecimal byte string:

```text
0728fa50c14f4e52113407ab12e173b7ef4eb3b3b36f192ec7b814dd411223c5
```

It is the SHA-256 of the UTF-8 bytes of
`yeto-best-paper-parallel-cells-v1|16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80|7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b`.
It MUST NOT be replaced, salted, sampled again, or chosen after looking at a
result.

For ASCII/UTF-8 strings `domain`, `study_id`, and `token`, define:

```text
rank(domain, token) = SHA256(
    hex_decode(master_seed)
    || 0x00 || utf8(domain)
    || 0x00 || utf8(study_id)
    || 0x00 || utf8(token)
)
```

Ranks are ordered as unsigned 256-bit big-endian integers, ascending. A digest
tie is broken by ascending UTF-8 token bytes. Hash sorting, not a library PRNG
or implementation-defined shuffle, is authoritative.

### 4.3 Atomic groups and time blocks

Group construction is stage-specific and has no operator option:

- For `p1r0`, `p1ad`, and `p2`, group by the preregistered `block_id`. Each
  group MUST contain exactly three cells, exactly one for each
  `mu={0,.5,.9}`, with identical `H`, `eta`, shuffle seed, and training seed.
- For `p3t`, group by training seed. Each group MUST contain exactly the four
  frozen P3 cells for that seed: both arms of the short pair and both arms of
  the long pair. Any seed group with a missing, extra, or duplicate cell is
  `FAIL`.

The group ID is the existing `block_id` for `p1r0`, `p1ad`, and `p2`; for
`p3t` it is exactly `p3t-seed-<shuffle_seed>`, with the seed in ordinary
base-10 notation. Sort groups by `rank("wave", group_id)`. Their zero-based
positions are `time_block_index` and `wave_index`. One atomic group is one
time block and one launch wave; groups are never split across waves. This is
the required time-block randomization.

### 4.4 Seeded arm-to-VM assignment and launch order

For initial wave attempt round `1`, sort that wave's cell IDs by
`rank("arm-order|<group_id>|1", cell_id)`. Independently sort logical slots
`{v0,v1,v2,v3}` by `rank("slot-order|<group_id>|1", slot)`. Zip the two lists.
Unused slots remain idle; another group's arm may not fill an idle slot.

Sort assigned cell IDs by `rank("launch-order|<group_id>|1", cell_id)` to
obtain the exact dispatch order. All assigned VMs MUST report `READY` with
validated provider evidence and immutable inputs before the first dispatch.
The controller issues all start messages in the committed order without
waiting for a cell to complete. The last dispatch timestamp minus the first
MUST be at most 60 seconds, and the last recorded scientific-start timestamp
minus the first MUST be at most 120 seconds. Exceeding either bound is the
loss-blind direct reason `pre_unblinding_validator_provenance_failure` for the
wave; it is never repaired by editing timestamps or redefining the time block.

Retry round `r>1` uses the identical rules after replacing the terminal `1` in
the three domains above with base-10 `r`. Thus retry slot assignment and launch
order are deterministic, but distinct rounds are independently domain
separated. A retry wave is inserted immediately after the failed wave and
before the next planned time block. Original planned groups keep their
materialized `time_block_index`; each retry record additionally carries
`retry_time_block_index`, monotonically increasing in actual wave order.

The complete plan records every group, wave index, cell, logical slot,
launch-order index, exact command hash, and the retry derivation above. It also
records the capacity constants from Section 3. Its canonical hash is
`parallel_plan_hash`. `roster_hash`, `parallel_plan_hash`, the existing
scientific randomization-plan hash, and their raw file hashes MUST be bound in
the launch manifest and cited byte-for-byte by every VM partial manifest.

## 5. Per-VM identity, namespace, and provider-evidence ownership

### 5.1 Logical slot and physical generation

A logical slot is one of `{v0,v1,v2,v3}`. A physical VM generation is a
positive base-10 integer beginning at `1` independently in each slot and
incrementing by exactly one whenever that slot receives a replacement. There
are no leading zeros and no generation reuse.

Let `tag` be the first 16 hexadecimal characters of `roster_hash`. Let
`campaign_attempt` be the smallest positive integer whose campaign state and
artifact namespaces have never existed; it is frozen in the reviewed packet.
Lower values MUST be proven absent or immutably abandoned. The exact per-VM
run ID is:

```text
bp-<stage_code>-<tag>-c<campaign_attempt>-<slot>-g<generation>
```

For example, the grammar permits
`bp-p3t-0123456789abcdef-c1-v2-g1`. The run ID MUST match
`^bp-(p1r0|p1ad|p2|p3t)-[0-9a-f]{16}-c[1-9][0-9]*-v[0-3]-g[1-9][0-9]*$`.

A higher `campaign_attempt` is only a fresh infrastructure namespace. If every
lower attempt stopped before any scientific cell began, cell attempt counters
may still begin at `1`. If any lower attempt began a scientific cell, the new
packet MUST cite and retain its complete attempt/partial/lifecycle registry and
continue the per-cell counters; only groups eligible under Section 6 may be
rerun. A campaign-attempt change never resets scientific lineage, discards an
outcome, or licenses a whole-roster rerun. Any exposed outcome blocks such a
restart unless new pre-outcome authority supplies untouched cells.

The campaign packet freezes one `CAMPAIGN_ARTIFACT_ROOT`, one
`CAMPAIGN_STATE_ROOT`, and one fixed absolute
`SCIENCE_ROOT=/opt/yeto-science/<stage_code>/<tag>`. For each physical
generation:

```text
state path       = <CAMPAIGN_STATE_ROOT>/<run_id>.json
artifact prefix  = <CAMPAIGN_ARTIFACT_ROOT>/vms/<slot>/g<generation>/
attempt prefix   = <artifact prefix>/cells/<cell_id>/attempt-<attempt>/
provider record  = <artifact prefix>/provider/provider-evidence.json
partial manifest = <artifact prefix>/manifests/vm-partial-manifest.json
lifecycle record = <artifact prefix>/manifests/vm-lifecycle-final.json
```

Every state path and artifact prefix is create-only and MUST be proven absent
before provisioning. No VM may write another slot/generation's prefix. The
campaign aggregator alone may write under
`<CAMPAIGN_ARTIFACT_ROOT>/campaign/`. Common source/model/data objects are
immutable and read-only. All VMs use byte-identical files at byte-identical
absolute paths beneath `SCIENCE_ROOT`; a per-VM path MUST NOT enter a
scientific command.

### 5.2 Nonces and exact ownership

Each physical generation receives a newly sampled 128-bit ownership nonce
from the operating system cryptographic random source, encoded as exactly 32
lowercase hexadecimal characters. It is generated before the create request,
persisted in controller state, never reused, and placed on the VM and boot disk
with the exact campaign tag, slot, generation, and run-ID labels. Nonce draws
are operational uniqueness tokens, not scientific randomization and may not be
used in the schedule rank function.

Before any scientific start, the harness MUST persist and hash one provider
record owned by that physical generation. It contains at least:

```text
project, zone, run_id, campaign tag, slot, generation, ownership nonce,
instance name, instance numeric ID, boot-disk name, boot-disk numeric ID,
source-image numeric ID, machine type, provisioning model, termination action,
automatic-restart flag, maintenance action, boot-disk auto-delete flag,
creation timestamp, four CUDA indices, four A100 UUIDs, and the
learner-to-GPU-UUID bijection.
```

The instance numeric ID and disk numeric ID MUST be captured before start;
the four GPU UUIDs and bijection MUST be captured before the first cell. The
record MUST prove the Section 3 provider contract, four distinct A100s, exact
label/nonce ownership, and exclusion of protected numeric instance ID
`3908640733128066700`. Every attempt row cites the exact provider-record raw
SHA-256 and the slot/generation on which it ran. A name, prefix, label alone,
or a provider record from another generation is insufficient.

## 6. Spot preemption and retry semantics

The attempt status vocabulary is exactly:

```text
COMPLETED, DIVERGED, INFRA_FAILURE, FAILED
```

The direct infrastructure-failure vocabulary remains exactly:

```text
provider_spot_preemption
vm_host_gpu_failure
process_exit_before_scientific_divergence
missing_or_checksum_invalid_required_artifact
pre_unblinding_validator_provenance_failure
```

The sole peer retry reason remains
`peer_block_invalidated_by_infra_failure`. It is never a `failure_reason`.
The forbidden retry triggers remain `poor_loss`, `finite_completed_loss`,
`scientific_divergence`, and `post_unblinding_preference`.

Provider Spot preemption is a direct trigger only when the exact numeric-ID
provider lifecycle record reports preemption. A timeout, slow cell, process
exit, missing log, or operator belief MUST NOT be relabeled as provider
preemption. Ambiguity is `FAILED` unless another frozen direct reason is
positively established.

Every cell begins with attempt counter `1`. Its attempt ID is exactly
`<cell_id>-attempt-<attempt>`. A retry increments every cell in the atomic
group by one with no gap, creates a new create-only attempt directory, and
starts from the same frozen initial model state, seed, data order, command,
image, and work budget. An attempt may never resume a partial optimizer,
learner, syncer, tape, result, or checkpoint state. Provider automatic restart
remains false. Files from a failed attempt are evidence and are never treated
as inputs to its successor.

For `p1r0`, `p1ad`, and `p2`, a genuine direct infrastructure failure reruns
the complete three-cell `(H,eta,seed)` group. For `p3t`, it reruns the complete
four-cell seed group. Completed peers retain their original `COMPLETED` rows,
losses/checkpoints, and artifacts and receive the peer retry reason on their
new attempts. A direct-failure row remains `INFRA_FAILURE`. All new group rows
share one loss-blind authorization created after the prior wave prefix is
mechanically sealed and before any outcome is exposed. It contains all fields
required by the original retry policy plus `parallel_plan_hash`, `group_id`,
`retry_round`, and the exact prior wave-manifest canonical SHA-256.

Retry eligibility requires every non-trigger peer in the immediately prior
round to be `COMPLETED` or another direct `INFRA_FAILURE`. If the same round
contains `DIVERGED` or `FAILED` together with `INFRA_FAILURE`, the scientific
row remains terminal and MUST NOT be rerun, while the whole-group retry
precondition cannot be met. That campaign is therefore `FAILED` and cannot
seal; resolving such a mixed terminal round requires new pre-outcome authority,
not an operator exception.

If preemption destroys a physical generation, that generation is hash-locked,
torn down/verified by exact ID, and replaced in the same logical slot with the
next generation. The schedule is not remapped manually. Surviving VMs may
serve the retry assignment produced by Section 4.4; the failed slot must have
a validated replacement before it is assigned work.

Retry rows are appended to the campaign attempt registry as one contiguous
wave in committed launch-order order. No later planned group may be appended
between a triggering wave and its retry. The analysis round for a group is the
smallest retry round in which all group cells are resolved as
`COMPLETED` or `DIVERGED` and no row is `INFRA_FAILURE` or `FAILED`. Earlier
completed peer values are retained but excluded from the primary group
comparison; there is no outcome-based choice among attempts.

`DIVERGED` is a scientific terminal outcome, receives infinite loss under the
frozen analysis, and never triggers a retry. `FAILED` is nonretryable and
blocks a scientific campaign seal. An unresolved `INFRA_FAILURE` also blocks
the campaign seal. An operational stop preserves all evidence and produces an
abandonment inventory, never a scientific seal.

## 7. Per-cell positive-work and finite-result gates

A terminal process exit and the presence of infrastructure artifacts do not
prove scientific work. Before an attempt may be `COMPLETED`, the validator
MUST reconstruct its exact command and observed work from immutable artifacts
and pass every stage-specific predicate below.

For P1-R0, P1-adaptive, P2, and P3 training, a completed full-budget cell has:

```text
observed training tokens                    = 655360
aggregate microsteps                        = 5120
learner IDs                                 = {0,1,2,3}
physical optimizer steps per learner        = 1280 exactly
full quorum                                 = true
fixed-window work                           = exact
version-matched anchor                      = true
barrier trace                               = valid
executed command hash                       = expected cell command hash
provider evidence hash                      = assigned VM generation hash
```

The horizon-specific work MUST also equal:

| H | global outer steps | per-fragment outer updates |
|---:|---:|---:|
| 16 | 320 | 80 |
| 64 | 80 | 20 |
| 256 | 20 | 5 |

For `p1r0`, `p1ad`, and `p2`, `COMPLETED` additionally requires a finite
IEEE-754 development endpoint, finite per-sequence `loss_sum` and
`loss_per_token` for every locked row, positive target-token counts, exact row
identity, and reproduction of the aggregate within relative and absolute
tolerance `1e-12`. NaN and either infinity are not finite. The endpoint is
retained at at least 12 significant digits. A nonfinite scientific endpoint is
`DIVERGED`, never a hollow `COMPLETED` row.

For `p3t`, `COMPLETED` requires all registered training work; exactly one
recorded `training_loss` for each learner step `1..1280` for each learner
`0..3` (5,120 values total), with every value finite; a sealed nonempty final
checkpoint with SHA-256; and finite checkpoint tensors under the frozen
validator. Its result field remains `loss=null`, its evaluation role is
`none`, and all development/audit loss paths remain null or absent. These
training-loss values are work/health evidence only and MUST NOT be exposed or
used for selection or inference. This amendment does not smuggle evaluation
into P3 training.

A `DIVERGED` row MUST contain the exact command/provider hashes, a tape prefix,
the last finite step, the first scientific nonfinite event, and a hashed
scientific-divergence record. It may terminate before full work; that is the
registered scientific outcome. An infrastructure row with zero or partial
work may be retained as `INFRA_FAILURE`, but cannot satisfy an expected cell
or make the campaign sealable until the loss-blind retry rule resolves its
group.

For every `launch_cells` entry, the campaign gate requires exactly one
analysis-round terminal row in `{COMPLETED,DIVERGED}`. Inherited parent cells
are validated by exact prefix/hash equality and are not rerun. `COMPLETED`
means full exact learner steps plus the finite stage-appropriate result above.
Missing, duplicate, zero-work, partial-work-as-completed, hash-mismatched, or
nonfinite-as-completed rows are `FAIL`.

This rule incorporates tonight's hollow-seal lesson as a positive invariant.
`HANDOFF.md` lines 145--161, 200, 235--247, and 484--489 establish that the
stopped attempt ran zero cells and correctly yielded no acquisition/final
manifest or scientific outcome. `STAGE-PLAN.md` lines 263--275 places a new
reviewed production commit with sealgate fixes before the canary/P1 chain.
The current RUNNER makes the latent failure mode concrete: it constructs a
seal at lines 2754--2786 and calls that path at line 3171 before raising for
`FAILED`/`INFRA_FAILURE` rows at lines 3172--3180. Under this amendment, a
zero-cell, infra-only, unresolved, or work-unvalidated inventory may be
hash-preserved as failure evidence but MUST NOT create `campaign-seal.json` or
carry status `sealed_results`.

## 8. Per-VM partial manifests and one campaign seal

The controller assigned to each physical VM generation writes one append-only
`vm-partial-manifest.json` containing only attempts executed by that exact
slot/generation and citing the common roster, parallel plan, bound descendant,
scientific randomization, provider evidence, and exact command hashes. Attempt
rows are ordered by actual wave order and then committed launch order. The
partial manifest is canonicalized and hash-locked after its last attempt and
before teardown, using already synchronized immutable artifacts if the VM was
preempted. Its only terminal status is `vm_partial_hash_locked`; it is not a
scientific result seal and cannot authorize analysis or a descendant.

After hash-locking, the harness performs exact-ID teardown for that physical
generation and writes `vm-lifecycle-final.json`. The lifecycle record cites
the partial-manifest hash and records the explicit instance numeric ID, disk
numeric ID, nonce/labels, deletion request/completion times, independent exact
instance `NOT_FOUND`, exact disk `NOT_FOUND`, and zero attached accelerator
proof. Preempted or otherwise automatically deleted resources still require
the exact-ID absence proofs. Name-only, prefix, wildcard, label-only, or
“delete all campaign resources” operations are forbidden.

The campaign aggregator is read-only with respect to VM namespaces. It MUST:

1. verify the exact bound descendant, roster, parallel plan, scientific
   randomization plan, and command registries;
2. enumerate the controller's exact slot/generation registry and require one
   hash-locked partial manifest and one final lifecycle record for every
   generation, with no unregistered generation;
3. verify pairwise-disjoint namespaces, unique run IDs, unique 128-bit nonces,
   unique instance/disk numeric IDs, and the four-slot/16-A100 cap over the
   full timestamped lifecycle;
4. reconstruct waves and require exact seeded assignment, dispatch order,
   time-block order, start-span bounds, and append-only retry lineage;
5. reconstruct every work/result predicate in Section 7 from the referenced
   artifacts rather than trusting booleans in a partial manifest;
6. require exact launch-cell coverage by the deterministic analysis rounds,
   prove inherited parent cells/results are an immutable prefix, retain every
   failed/retried/diverged attempt, and reject any unresolved `INFRA_FAILURE`
   or `FAILED` row;
7. verify the exact-ID teardown record for every physical generation and a
   final provider census of zero campaign-owned A100s; and
8. for P3 training, build and hash the complete 32-cell checkpoint registry,
   preserve `loss=null`, prove both evaluation artifacts were structurally
   inaccessible, and prove that no partial outcome was exposed.

Only after all eight checks pass may the aggregator write the sole scientific
`campaign-seal.json`, with exactly these fields:

```text
schema = yeto_parallel_campaign_seal_v1
status = sealed_results
stage_code
study_id
authoritative_prereg_template_sha256
amendment_raw_sha256
bound_manifest_canonical_sha256
roster_hash
parallel_plan_hash
scientific_randomization_plan_hash
campaign_manifest_canonical_sha256
vm_registry_canonical_sha256
vm_partial_manifest_hashes
vm_lifecycle_record_hashes
cumulative_expected_cell_count
launch_cell_count
resolved_launch_cell_count
attempt_count
work_evidence_all_pass = true
schedule_all_pass = true
provider_ownership_all_pass = true
exact_id_teardown_all_pass = true
partial_outcomes_exposed = false
sealed_at_utc
```

The two hash arrays are ordered by `(slot, generation)` and contain one object
with `slot`, `generation`, and raw SHA-256 per generation. Counts are integers,
not assertions. The campaign manifest contains the complete canonical attempt
registry and deterministic analysis-round mapping. The campaign seal is
create-only. Any later byte change creates a new descendant with explicit
lineage; it never overwrites or “repairs” the seal.

No human or outcome-aware process may read a partial P1-R0, P1-adaptive-round,
P2, or P3-training result before its complete campaign/round seal. Mechanical
validators may test schema, hashes, work counts, and finiteness loss-blindly.
For P2, both new seeds remain blinded until the whole P2 campaign seals. For
P3, the campaign seal and checkpoint-registry hash are prerequisites to the
separate audit authorization; the seal itself exposes no training or audit
outcome.

## 9. Hash and identity rebind matrix

No scientific identity becomes selectable by VM. Differences are divided into
three exhaustive classes.

### 9.1 Campaign-wide identities that remain binding on every VM

The following remain identical across all slots, generations, waves, and
attempts:

- authoritative preregistration commit/path/raw hash
  `16d27bc...` / the path in Section 1 /
  `7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b`;
- the original retry-policy object, canonical SHA-256
  `41cf8ddbe13538bf111c2198807dd1522ba51f2c21e73269311a66d8ae398f1e`,
  and confirmation-policy object, canonical SHA-256
  `130bb6245ecfb1874b83389852333ef1e0a65e2ec54a2438e4c2c10f9ebd4536`;
- image numeric ID `7290368630472593484`, image projection SHA-256
  `038098c2b5356c9117f1019bf0d19c8999ab50f259dceb041a57fcf657d2620f`;
- model `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, manifest SHA-256
  `43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132`,
  and archive SHA-256
  `53d15a96a333e33c6a7a9224dbe6392a2480420bd40a327588797d03b625e4c3`;
- dataset `trl-lib/Capybara`, revision
  `e235e846458bff3398a88aed812347f7f0756520`, and parquet SHA-256
  `970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409`;
- split algorithm/roles, all reviewed train/development/audit identity hashes,
  and the loss-free freeze summary SHA-256
  `2b0f53ed6861ebdda4a0ebc0ec0b7612dee2ab217906b763b06ec48980b9c8dc`;
- one new clean pushed production commit, one source bundle, one common
  bootstrap template, one scientific protocol, one bound expected-cell
  roster, one base cell-command registry, one normalized workload-command
  registry, one roster hash, and one parallel-plan hash for the campaign; and
- all registered H/mu/eta/seed coordinates, work, pairing, live controls,
  outcomes, thresholds, blinding, and analysis rules.

The same exact scientific cell command bytes and `command_hash` apply wherever
a cell is assigned. VM/campaign namespaces live in the attempt working
directory and harness layer, not in the scientific argv. The only relative
scientific output arguments permitted are the frozen `work` and `report`
values resolved beneath the create-only attempt directory. Model, training,
and permitted evaluation inputs use the common absolute `SCIENCE_ROOT` and are
byte-identical. Any other cross-VM command difference is `FAIL`.

### 9.2 Campaign identities that MUST be regenerated before parallel launch

The old production commit `0af7f4a80426babc14896c7c1f7885abcb331d46`
does not contain the required parallel builder/aggregator or sealgate fixes.
Consequently its source-bundle hash, run-phase argv hashes, P0a randomization
hashes, bound-manifest hashes, campaign/cell-command hashes, harness
spec/bootstrap/render hashes, and pre/postupload packet hashes in `HANDOFF.md`
are historical evidence only. They neither authorize nor validate a parallel
campaign.

A parallel launch MUST regenerate and bind the new production commit and
ancestry proof, source bundle, common bootstrap template, exact base cell
commands, normalized commands, roster, scientific and parallel randomization
plans, bound manifest, campaign harness manifest, all hashes, tests, and review
packet. After that packet is accepted, those campaign-wide scientific bytes
MUST NOT drift because of slot, provider ID, preemption, run ID, or retry.

### 9.3 Fields and hashes that are necessarily per physical VM generation

Only the following operational identities vary by slot/generation:

```text
run ID; controller state path; artifact prefix; physical generation;
ownership nonce and ownership labels; instance name/numeric ID;
boot-disk name/numeric ID; A100 UUIDs and CUDA map;
provider-evidence bytes/hash; rendered per-VM harness argv/spec/bootstrap hash;
attempt working directory and its raw path hash; start/end timestamps;
VM partial-manifest hash; lifecycle-final hash; exact-ID teardown proofs.
```

Each per-VM rendered outer artifact MUST cite the common campaign identities.
No per-VM value may change H, mu, eta, seed, data order, cell argv, work,
evaluation role, or analysis. Zone and machine type are campaign-wide under
Section 3, not per-VM choices.

## 10. Required conformance tests

Before review, the implementation MUST pass deterministic tests that:

1. reproduce the master seed and rank function from independent
   implementations and golden byte vectors;
2. materialize the identical plan regardless of input row/dictionary order;
3. reject a missing/duplicate/extra cell, malformed P1/P2 three-arm block, or
   malformed P3 four-cell seed group;
4. prove P1-R0 yields 12 three-cell waves, never a fourth mixed-block cell,
   and P3 yields eight four-cell seed waves;
5. reject manual slot swaps, dispatch reorder, split groups, start-span excess,
   cross-round P1 overlap, and any concurrent-cell/A100 excess;
6. simulate preemption in every slot and launch position, proving contiguous
   whole-group retries, fresh counters/directories, no resume, retained peers,
   deterministic retry reassignment, and no outcome-triggered retry;
7. reject nonce/ID reuse, cross-namespace writes, provider-record substitution,
   name-only teardown, missing exact-ID `NOT_FOUND`, and protected ID
   `3908640733128066700` in any target set;
8. delete or corrupt each work artifact in turn and prove no completed row and
   no campaign seal can be written; explicitly test zero cells, zero learner
   steps, one learner short by one step, one missing fragment update, NaN,
   `+Inf`, `-Inf`, endpoint/per-sequence mismatch, missing checkpoint, and
   unresolved infra failure;
9. prove a correctly evidenced `DIVERGED` outcome is retained, never retried,
   and remains sealable as divergence rather than completed finite work;
10. aggregate partial manifests in every arrival order and obtain one
    byte-identical canonical campaign manifest/seal preimage;
11. prove a VM partial manifest cannot masquerade as `sealed_results` and that
    there is exactly one create-only campaign seal; and
12. prove P3 training commands and every backup/synchronization path cannot
    mount, name, read, copy, or expose development or audit artifacts, while
    the checkpoint registry still covers exactly 32 expected cells.

A test skip, flaky result, missing negative test, or disagreement between
independent schedule implementations is `FAIL`.

## 11. Dual review and first-launch gate

This amendment grants no launch authority by itself. Before the first
parallel scientific launch, the existing P0a -> exact-ID teardown -> CPU
replay -> P0b -> exact-ID teardown -> CPU replay chain MUST pass from the new
clean production lineage. The work-evidence-before-seal fix and the parallel
implementation MUST be present in that lineage; `STAGE-PLAN.md` lines
266--283 cannot be skipped. P2 additionally requires the reviewed cumulative
parent-lineage builder. P3 additionally requires the reviewed training-only,
checkpoint-registry, audit-authorization, hidden-evaluator, whole-batch retry,
seal, and unblind machinery. This amendment removes only the concurrency
registration blocker.

Two independent reviewers MUST review the same byte-identical launch packet:

1. **Scientific-integrity/ICML review** re-approves authority ancestry,
   unchanged scientific semantics, P0 chain, exact command/input identities,
   P1/P2/P3 scope, audit isolation, work-evidence gates, provider ownership,
   namespace separation, aggregation, preservation, and exact-ID teardown.
2. **Statistics review** re-approves the canonical roster, fixed seed/rank
   implementation, atomic groups, arm-to-machine assignment, launch order,
   time-block randomization, pairing, retry/analysis-round selection,
   blinding, and absence of outcome-dependent schedule choices.

The packet MUST bind at least:

```text
this amendment's raw SHA-256 and reviewed source commit;
the authoritative prereg raw SHA-256;
new production commit, full ancestry proof, source-bundle hash, and test log;
common and per-VM bootstrap/spec/argv hashes;
all immutable input identities and generation-qualified sources;
bound manifest, roster hash, parallel-plan hash, scientific plan hash;
base and normalized cell-command registries;
stage code, expected cells/groups/waves, project/zone, capacity constants;
campaign attempt, state/artifact/science roots, empty/create-only proofs;
provider/nonce/exact-ID/teardown contract;
retry policy and work/seal validator hashes;
P3 no-evaluation and no-partial-exposure proofs when applicable.
```

Each reviewer decision vocabulary is exactly `{PASS, FAIL}`. The packet is
launch-eligible only when both decisions are `PASS`, both decisions cite the
same packet raw SHA-256, and a new explicit user/root launch authorization
cites that packet. An old review, a review of one VM, a review of a serial
plan, or a review of a different hash does not transfer.

Any change to the amendment bytes; source commit; schedule/aggregation code;
rank vectors; roster/plan/command hashes; stage cells; capacity; project/zone;
machine/GPU shape; namespace grammar; retry policy; work gate; P3 isolation;
or teardown contract invalidates both passes and requires a new dual review
before launch. Runtime-only values enumerated in Section 9.3 do not require a
new review when they are generated and validated exactly under the accepted
packet.

## 12. Final binding ruling

Subject to every gate above, the scientifically legitimate maximum is **four
simultaneous cells on four Spot `a2-highgpu-4g` VMs, totaling at most 16
A100s**. P1-R0 and P2 deliberately realize width three because an indivisible
three-arm live-control block occupies one wave; P3 training may realize width
four because all four cells of a fresh-seed paired block occupy one wave.

P1-R0's 36 preplanned cells may therefore be sharded. P1's adaptive logic may
not: only cells already authorized in one immutable descendant may run in
parallel, and every next descendant waits for the preceding round's complete
seal and registered deterministic decision. P0, P3 audit, and all unnamed
future stages receive no parallel authority from this amendment.
