# EXP2-54 sequential GCP A100 capture plan

Date: 2026-07-14 (America/Los_Angeles)
Status: **DRAFT / NON-LAUNCHABLE**

This plan turns the exact-state capture work into three strictly sequential
GCP Spot stages. It does not authorize a cloud launch. All three checked-in
specifications retain `cloud.adopt_only: true` and pin audited code commit
`452ebdea30503f10c1eda68e9fdcff704be2a792`.
The seed-239 command additionally contains a deliberately non-integer sentinel,
so confirmation cannot start accidentally even if the cloud guard is removed.

## Fixed infrastructure envelope

Every stage uses the READY source image
`projects/model-training-497007/global/images/yeto-optimizer-a100-20260714`
with expected image ID `7290368630472593484`, `pd-ssd`, Spot provisioning,
and delete-on-termination. Each VM is an `a2-highgpu-4g`: four single-process
learners are pinned across four A100s with `--gpu-slots 4`.

The stages must not overlap. The campaign therefore consumes at most four
A100s at once, even though the broader project is allowed to use eight. The
hard runtime envelopes imply at most 20 A100-hours if every stage runs to its
limit: 4 for the one-hour qualifier safety envelope, 8 for development, and 8 for
confirmation. Spot provisioning, review, and any restart time are outside
those compute caps.

The accelerator quota is not the current blocker: `us-central1` has 16
preemptible A100s available. The regional `A2_CPUS` quota is 12, while this
shape needs 48. Quota preference
`yeto-a2-cpus-us-central1-48-20260714` requested the least value that can run
one stage. Google approved the request to 48 at `2026-07-14T11:58:10Z`; trace
ID `2b5bd8c8-2383-4de4-aa90-44b38f1dbe0c`. The harness doctor now refuses
launch if the live grant later falls below the requested shape.

| Order | Specification | Geometry | Purpose | Earliest successor gate |
|---|---|---|---|---|
| 1 | `exp2-54-smoke-capture-draft.json` | four learners, strict quorum 4/4, H=4, capture off/on | prove that enabling capture preserves behavior and produces self-consistent evidence | checksummed parity PASS and complete capture validation |
| 2 | `exp2-54a-seed223-development-draft.json` | four learners, strict quorum 4/4, H=16, seed 223223 | acquire development capture with at least 32 joined boundaries and at least 8 per fragment | immutable development analysis and winner-selection manifest |
| 3 | `exp2-54b-seed239-confirmation-locked.json` | four learners, strict quorum 4/4, H=16, seed locked | independent confirmation of only the frozen winner | unchanged seed-239 gates pass independently |

## Stage 0: prerequisites and immutable inputs

No stage may be opened until all of the following are true:

1. The capture, audited-transcript, validator, and parity changes are committed,
   reviewed, and pushed as `452ebdea30503f10c1eda68e9fdcff704be2a792`;
   every spec pins that same immutable code commit.
2. The image path and numeric source-image ID have been re-described and still
   identify the READY image above. Do not substitute a family or a newer image.
3. The model and data paths exist on the image and the harness doctor/render
   output agrees with the checked-in spec. Before starting the runner, execute
   `/etc/yeto-model-files.sha256` and `/etc/yeto-data.sha256`; copy them plus
   `/etc/yeto-runtime.txt` and `/etc/yeto-optimizer-image.json` into the run's
   sealed `input-provenance/` tree.
4. The three specs validate using the final checkout, and their rendered
   commands contain no BCMP shadow flags. The qualifier deliberately retains
   syncer-probe capture for both matched arms as an independent parity input;
   the development and confirmation captures do not use that legacy probe.
5. The smoke command actually creates
   `report/optimizer_state_capture_parity.json` and its standard sha256sum
   sidecar using `scripts/validate_optimizer_capture_parity.py`. The compare
   command now invokes this fail-closed gate only when the explicit
   `--optimizer-state-capture-parity` flag is present; the launch spec must
   retain that flag, both fully sampled probe indexes, and both parity output
   completion paths. The successful verdict must also create the portable
   `optimizer_state_capture_parity.inputs.sha256` manifest.
6. Preserve the final specs, rendered commands, and their SHA-256 hashes before
   changing any `draft` or `adopt_only` field. A launchable spec is a reviewed
   derivative, not an in-place reinterpretation of this draft.

Local, non-mutating validation is:

```bash
PYTHONPATH=. python3 scripts/optimizer_experiment.py validate \
  experiments/optimizer/exp2-54-smoke-capture-draft.json
PYTHONPATH=. python3 scripts/optimizer_experiment.py validate \
  experiments/optimizer/exp2-54a-seed223-development-draft.json
PYTHONPATH=. python3 scripts/optimizer_experiment.py validate \
  experiments/optimizer/exp2-54b-seed239-confirmation-locked.json
```

Run `render` and cloud `doctor` only after inserting the final commit SHA.
Neither validation nor rendering is permission to launch.

## Stage 1: H=4 capture-off/capture-on qualifier

The smoke is not a one-GPU schema test. It runs the matched settings
`capture_m4_off,capture_m4_on` sequentially on the same four-A100 geometry.
The two presets are frozen to differ only by arm name and the capture
treatment. Both explicitly disable reconnects and use AdamW learners, strict
4/4 quorum, four fragments, H=4,
float32 wire values, merge alpha zero, no delta correction, outer Nesterov
with momentum zero, and outer LR 0.28.

For parity only, `--syncer-probe-capture --syncer-probe-capture-every 1`
records both `capture_m4_off/syncer_probe/index.jsonl` and
`capture_m4_on/syncer_probe/index.jsonl`. Both files are mandatory completion
artifacts and explicit parity-validator inputs. They supplement rather than
replace the capture-on arm's authoritative audited response transcript.

The smoke requests 16 syncer steps and caps each learner at 80 steps. The
harness derives 16 ideal learner steps and requires 64 additional liveness
steps, so 80 is the exact declared minimum. Capture validation requires at
least four joined boundaries and at least one on every fragment.

The qualifier passes only if all of these conditions hold:

- both arms complete without timeout, reconnect substitution, missing round,
  non-finite value, or unexpected configuration difference;
- the parity producer writes a `PASS` verdict from explicit off/on inputs and
  the verdict's checksum sidecar verifies;
- overhead is at most 2% over the exact syncer-monotonic interval from commit
  sequence 1 to N, covering the same commits 2..N in both arms; cold
  model/data/CUDA startup is excluded, and missing/reordered/non-monotonic
  producer timestamps fail closed;
- `optimizer_state_capture_parity.inputs.sha256` verifies every probe payload,
  event tape, final checkpoint, export file, transcript, and result row
  consumed by the passing decision;
- the capture-on validator returns `PASS`, the audited syncer response
  transcript is present, and its joined-boundary minimum is satisfied;
- all four learner `manifest.json` files and their sidecars verify;
- the capture tree manifest verifies every artifact named by the validator,
  including the authoritative syncer response transcript;
- the validation-summary sidecar and every path listed in
  `execution.checksum_manifests` pass `sha256sum -c`; and
- the harness has uploaded a complete artifact tree to the smoke-specific GCS
  prefix. A nonempty `results.jsonl` alone is not a pass.

Any missing parity output, validator failure, checksum failure, Spot
preemption, or ambiguous off/on comparison is a qualifier failure. Preserve
the artifact prefix for diagnosis and rerun a fresh qualifier after fixing the
cause; never promote partial smoke evidence into the development gate.

## Stage 2: seed-223 H=16 development capture

Development stays closed until the smoke verdict and all smoke checksum
manifests have been reviewed. It runs only `capture_m4`, at H=16, with training
seed `223223` and row-shuffle seed `223`. The command requests exactly 32 total
syncer steps and permits 512 learner steps. Thirty-two is the mechanism-capture
target, not the old 340-commit training-quality campaign: with capture capped
at 64 midpoint windows per learner, extending to 340 could exhaust exact
capture and fall back to legacy pushes that audited transcript mode must
reject. The budget calculation is fail-closed:

```text
ideal learner steps       = ceil(32 / 4) * 16 = 128
declared liveness headroom                         = 384
required learner cap      = 128 + 384 = 512
```

The validator must join at least 32 complete committed boundaries overall and
at least eight for each of the four fragments. It must also validate exact H/2
and H counters, audited request/response identity, payload digests, responder
sets/order, monotone attempts, all four learner manifests, absence of temporary
or orphan files, and finite exact-state tensors. Development is incomplete if
any one of those requirements is absent even when the training result exists.

Acquisition does not yet authorize replay or CRN analysis. The recorder emits
per-learner `.pt` envelopes, whereas the frozen replay contract requires joined
`state.npz`, `optimizer.json`, and `crn.json` boundary bundles. Before scoring,
implement and test a committed-responder materializer and the still-missing
whole-model/optimizer/RNG/data-iterator/next-eight-step CRN capture and restore
layer; freeze their executable and configuration hashes without changing
formulas, thresholds, exclusions, or ranking. Only then can the development
gate in `experiments/optimizer/exp2-54-exact-state-prereg.md` run. If neither
MTRF nor MSTP passes every corrected gate, kill both and do not open
confirmation.

If a winner exists, write one immutable selection manifest containing at
least the selected candidate, complete development outcomes, preregistration
hash, exact command/spec/repository/image/model/data hashes, candidate formula
and executable/config hashes, the committed seed-239 value, and a statement
that no threshold changed after seed-223 results became visible. Store the
manifest and its checksum outside the mutable VM before any seed-239 edit.

## Stage 3: seed-239 confirmation remains locked

The confirmation spec is intentionally invalid as a training command:

```text
--training-seed __OPEN_SEED239_ONLY_AFTER_FROZEN_SELECTION__
```

It also remains `adopt_only`, `draft`, and labeled `seed-lock=unopened`. Only
an immutable GO record referencing a passing smoke, a passing seed-223 winner,
and the hashed selection manifest may authorize a reviewed derivative. At
that point, and not before, replace the sentinel in both the command and
`checks.expected_flags` with the preregistered numeric seed (expected campaign
encoding: `239239`), retain row-shuffle seed `239`, insert the final repository
SHA, and bind the frozen single winner. Revalidate and hash the derivative
before changing its cloud guard.

Confirmation keeps H=16, the same 32/512 liveness calculation, and the same
minimum 32 joined boundaries with at least eight per fragment. It must pass the
unchanged single-hypothesis gates independently. Seed 223 cannot rescue a
seed-239 failure, and confirmation must not be repurposed to compare, tune, or
replace the frozen winner.

## Artifact and checksum contract

Each capture stage requires the following evidence beneath its own run root:

- `report/results.jsonl`;
- for smoke, both matched arms' `syncer_probe/index.jsonl` parity inputs;
- `work/<capture-arm>/syncer_response_transcript.jsonl`;
- `optimizer_state_capture_validation.log` and the validator's JSON summary;
- the summary's `.sha256` sidecar;
- `optimizer_state_capture_artifacts.sha256`, the portable tree manifest that
  covers the authoritative transcript and captured artifacts;
- four learner `manifest.json` files and their `.sha256` sidecars; and
- for smoke, `report/optimizer_state_capture_parity.json` and its `.sha256`
  sidecar plus `report/optimizer_state_capture_parity.inputs.sha256`; and
- `input-provenance.sha256`, sealing the verified model/data manifests and
  copied runtime/image metadata before training starts.

The harness treats every declared checksum file as both a completion path and
a manifest to execute with `sha256sum -c` before recording success. The tree
manifest validates the full captured file set, while the individual manifest
sidecars make learner-local corruption fail independently. Do not weaken this
contract after seeing an incomplete run.

GCS prefixes are distinct (`exp2-54-smoke`, `exp2-54a`, and `exp2-54b`). Never
copy a partial stage into a successor prefix or combine evidence from multiple
attempts under one purported run identity. On Spot preemption, sync/preserve
the partial prefix for diagnosis, retire the exact instance through the
harness ownership checks, and begin a fresh attempt with a new immutable run
identity.

## Timing expectation

Once the final SHA, parity integration, 48-A2-vCPU grant, and Spot capacity
exist, the smoke is expected to take about 30 minutes but has a one-hour
provider safety envelope. Development and confirmation each have two-hour hard
envelopes. The earliest expected compute time remains about 4.5 hours, with a
five-hour worst-case provider envelope, but confirmation cannot be scheduled
as a simple continuation: seed-223 replay, statistical review, winner freezing,
and the immutable GO record sit between the two full captures. Quota review,
Spot queueing, or preemption makes wall-clock ETA open-ended until the doctor
turns green.
