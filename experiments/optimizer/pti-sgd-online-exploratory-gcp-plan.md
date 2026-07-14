# Exploratory online PTI-SGD versus SGD-0.28 GCP plan

Date: 2026-07-14 (America/Los_Angeles)

Status: **DRAFT / BLOCKED / NON-LAUNCHABLE**

This plan freezes the smallest useful online engineering sequence for the
PTI-SGD policy in `pti-sgd-fresh-confirmation-prereg.md`. It is deliberately
not a capture-v2 common-random-numbers (CRN) experiment and cannot produce a
causal PTI superiority claim. The two online arms may share a seed and input
order, but they evolve through different states after the first non-stock
action. Their terminal loss difference is exploratory paired-by-seed evidence,
not a same-state boundary effect.

No JSON launch specification is checked in with this plan. The repository at
the time of this freeze has the PTI f32 policy state machine, capture-v2 action
adapter, and CRN authority contracts, but the production Rust syncer does not
accept `pti-sgd` as an outer optimizer and `scripts/compare_diloco.py` has no
stock/PTI online arm pair. Creating a spec whose command names PTI would
therefore be false provenance. The exact implementation gates below must land
in one reviewed and pushed commit before the two immutable JSON derivatives
are created.

## Fixed cloud envelope

Both stages use:

- project `model-training-497007`;
- Spot provisioning, delete-on-termination, no automatic restart, and a
  provider maximum runtime of exactly 3,600 seconds;
- READY source image
  `projects/model-training-497007/global/images/yeto-optimizer-a100-20260714`
  with exact numeric image ID `7290368630472593484`;
- a 250 GB `pd-ssd` boot disk and `storage-rw` scope;
- checkout source mode pinned to one future full 40-hex implementation commit;
- exact Python `/home/shou/venv/bin/python`, Cargo
  `/home/shou/.cargo/bin/cargo`, and Rust
  `/home/shou/.cargo/bin/rustc` executables;
- offline model/data paths `/home/shou/models/Qwen3.5-9B` and
  `/home/shou/data/Capybara-local/train.parquet`;
- verified `/etc/yeto-model-files.sha256`, `/etc/yeto-data.sha256`,
  `/etc/yeto-runtime.txt`, and `/etc/yeto-optimizer-image.json` provenance;
- a 60-second GCS background sync interval and a run-specific, initially empty
  GCS prefix; and
- a campaign-wide ceiling of eight active accelerators. The two stages are
  strictly sequential and request only one or four A100s respectively.

The declared runtime environment is fixed to:

```text
PATH=/home/shou/.cargo/bin:/home/shou/venv/bin:/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHONUNBUFFERED=1
```

Live image READY status, A2 CPU quota, preemptible A100 quota, active project
accelerators, and zonal Spot capacity must be rechecked immediately before
each launch. Quota and machine-type availability can be checked read-only;
actual Spot stock is not guaranteed by a read-only API and is known only when
the provider accepts or rejects an exact create request.

## Required online implementation before specs exist

One commit must provide all of the following without changing the frozen PTI
coefficient, interlock length, or fallback semantics:

1. A production syncer outer-optimizer mode named `pti-sgd` that consumes the
   exact current and immediately preceding same-fragment production
   pseudo-gradient bytes in commit order and runs the existing deterministic
   coordinate-order f32 kernel with coefficient exactly `-1/4`.
2. A gap-free per-fragment history and sealed-score state machine equivalent
   to `yeto.pti_sgd.PTISGDPolicy`, including the three-positive-score
   interlock, unresolved-tail closure, and history clearing on continuity or
   integrity failure.
3. Bit-identical stock fallback: every ineligible event must pass the original
   stock pseudo-gradient bytes to the existing memoryless SGD-0.28 update
   without decoding and re-encoding the fallback.
4. Durable action evidence for every opportunity, containing commit sequence,
   fragment/version identity, current and previous stock digests, selected
   action digest, `candidate_selected` or closed stock reason code, interlock
   inputs, resolved shadow score when available, and a hash-chain predecessor.
5. Checkpoint and resume support for the PTI history, score windows, open
   shadows, and ledger head, with a test proving uninterrupted and resumed
   online executions are bit-identical.
6. A `compare_diloco.py` pair whose only treatment difference is the stock
   versus PTI outer action. The control must remain RDA plus Nesterov with
   momentum positive zero and outer learning rate 0.28; because momentum is
   zero this is the campaign's SGD-0.28 path. Both arms must have equal
   diagnostic and evaluation work.
7. Fail-closed result validation that rejects missing action opportunities,
   broken ledgers, nonfinite loss, mismatched commit schedules, unequal
   evaluation work, any non-bit-identical stock fallback, missing final
   checkpoint, or an unverified input/runtime/image digest.
8. CLI and harness tests that make every frozen flag below an exact expected
   flag and prove that neither arm silently selects a different optimizer,
   merge rule, seed, H, token budget, LoRA shape, or data order.

Until all eight items exist, both proposed run IDs are reserved names only and
must not be launched, adopted, or used as artifact prefixes.

## Stage E1: one-A100 engineering canary

Reserved run ID and instance name:
`exp2-pti-online-e1-m1-canary`.

Reserved artifact prefix:
`gs://yeto-exp2-52-model-training-497007/exp2-pti-online-e1-m1-canary`.

The launchable derivative will use `us-central1-c`, machine type
`a2-highgpu-1g`, `accelerator_count: 1`, and
`max_total_accelerators: 8`. It will execute the stock and PTI arms
sequentially on the same VM with:

```text
model                         /home/shou/models/Qwen3.5-9B
data                          /home/shou/data/Capybara-local/train.parquet
sequence length               128
micro batch                   1
inner AdamW LR                0.001
LoRA rank / alpha             2 / 4
evaluation rows               8
maximum input rows            5000
row-shuffle seed              271
training seed                 271271
wire dtype                    f32
merge                         weighted RDA
merge alpha                   0
delta correction              none
outer stock kernel            Nesterov, momentum +0.0, LR 0.28
PTI coefficient               -1/4
PTI interlock                 three positive resolved shadows
fixed H                       4 optimizer steps
syncer commits                32
learner maximum steps         96
strict quorum                 enabled
barrier synchronization       enabled
deterministic commit order    enabled
reconnects                    disabled
token budget                  32768
arm timeout                   20 minutes
```

Thirty-two commits give eight predefined opportunities per fragment when the
four-fragment commit order is balanced. This is the smallest round number that
leaves opportunities after one same-fragment warm-up and three resolved
pre-action shadows; a 16-commit smoke would generally terminate before the
interlock could authorize a non-stock action. The learner cap is the ideal 32
steps plus 64 steps of liveness headroom.

E1 is an engineering pass only if both arms complete the identical 32-commit
schedule; every fallback is byte-identical to stock; the PTI ledger and final
checkpoint verify; at least one post-warm-up PTI action is selected; all
losses are finite; no individual evaluation regression exceeds 0.05; action
compute plus mandatory ledger overhead is below 2% of the matched stock
commit interval; and all declared checksum manifests verify after the writer
drains. Its loss sign is reported but is not an advancement threshold and is
not called CRN evidence.

## Stage E2: conditional four-A100 screen

Reserved run ID and instance name:
`exp2-pti-online-e2-m4-screen`.

Reserved artifact prefix:
`gs://yeto-exp2-52-model-training-497007/exp2-pti-online-e2-m4-screen`.

E2 stays `cloud.adopt_only: true` and labeled `draft=true` until a checksummed
E1 engineering PASS is reviewed. Its derivative will use `us-central1-c`,
machine type `a2-highgpu-4g`, `accelerator_count: 4`, and
`max_total_accelerators: 8`. The stages may not overlap.

E2 keeps the exact model, data, LoRA, inner optimizer, RDA, f32 wire,
SGD-0.28, PTI, seed, strict-quorum, barrier, deterministic-order, reconnect,
and validation settings from E1. It changes only the declared learner count
from one to four, GPU slots from one to four, fixed H from 4 to 16, token
budget to 700000, learner cap to 512, evaluation rows to 64, and arm timeout
to 45 minutes. It still requests exactly 32 syncer commits, giving the frozen
minimum of eight opportunities per fragment. The 512-step cap covers the
ideal 128 learner steps plus 384 steps of liveness headroom.

E2 uses the same engineering integrity gates as E1 and additionally requires
all four learners and all four fragment means to be reported even when they
are unfavorable. The exploratory summary must publish the paired-by-seed
terminal and trajectory loss differences, action fraction, per-fragment
effects, worst regression, overhead, and exact stock-fallback count. It must
state prominently that the arms are not restored from the same boundary and
therefore the result is not the preregistered capture-v2 CRN gate.

## Immutable spec derivation and teardown

After the online implementation lands, create two new JSON files rather than
editing this plan. Each must pin the same pushed implementation commit, exact
image path and numeric ID, exact runtime paths, unique run directories, unique
GCS prefix, every expected flag, strict quorum step budget, completion paths,
checksum manifests, and input provenance paths. E1 may become launchable only
after local validation, render review, detached-runtime smoke, live cloud
doctor, and an empty-prefix check. E2 remains non-launchable until E1 passes.

Success uses harness `sync` followed by exact-ID `delete`. Any timeout,
preemption, validation failure, or partial result uses `sync` followed by
exact-ID `abandon` with a nonempty reason. A Spot failure never authorizes
reuse of either run ID or GCS prefix; a retry receives a new immutable suffix.

