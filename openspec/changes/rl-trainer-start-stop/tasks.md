# Tasks

Codes follow the phase codes of `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §8. This change holds L3, L4, L5, and L7. Effort is in sub-agent person-days and is an estimate. Total estimate: L3 4 d, L4 4–5 d, L5 3.5 d, D3/D4 4 d, L7 4.5 d; about 20–21 d plus GPU cost.

## Prerequisite

- [x] P.1 The S23 integration branch (P0-3, P5, P0-S) is merged to agentenv main, and this change is rebased on it. Acceptance: `recovery.py`, `leave_reason.py`, and schema 4 evidence exist on the base. GPU: no.

## L3. Ports and fakes

- [x] L3.1 Define `TrainerLifecycle`, `DurableSampleBuffer`, `TrainerScheduler` (design D1) in the neutral layer, with fake implementations. Acceptance: CPU tests drive start → train → stop → start with fakes. GPU: no. Estimate: 1.5 d
- [x] L3.2 Add `SCHEDULED` to `recovery.TRIGGERS`; `RoundCutCheckpoint.save` uses it on a scheduler stop; resume treats it as complete (design D3). Acceptance: CPU tests for save and resume. GPU: no. Estimate: 0.5 d
- [x] L3.3 Map lifecycle stops to P5 reason codes and JOIN as a new incarnation per start (design D2). Acceptance: CPU tests on the elastic client fake; tape shows one JOIN and one LEAVE per start. GPU: no. Estimate: 1 d
- [x] L3.4 `TrainerLifecycle.run` wraps `IslandDriver`; check that `rebuild_trainer` works across repeated starts. Acceptance: CPU test with three start/stop cycles on the fake trainer. GPU: no. Estimate: 1 d

## L4. Durable buffer

- [x] L4.1 Durable index next to sample payloads in the existing cut storage; append, select, backlog statistics. Acceptance: CPU tests, including a reopen after process exit. GPU: no. Estimate: 1.5–2 d
- [x] L4.2 Watermark in the cut record (design D4). Acceptance: CPU test of the crash scenario in the spec. GPU: no. Estimate: 1 d
- [x] L4.3 Arrival notification by polling (design D4). Acceptance: CPU test that a callback fires within one interval. GPU: no. Estimate: 0.5 d
- [x] L4.4 Samplers write to the buffer; trainer publishes continuously and samplers load new weights (design D5). Acceptance: CPU test with fake sampler and trainer; behavior versions advance in the buffer. GPU: no. Estimate: 1–1.5 d
- [x] L4.5 Real-path wiring: samplers write to the durable buffer; the trainer Miles rollout reads from it through the `--rollout-function-path` plugin (no fork change); after training, `commit_consumed` runs and `buffer_watermark` is set. Acceptance: CPU tests on the plugin path. GPU: no. (not verified on GPU: L5.4 has not passed) Estimate: 2 d
- [x] L4.6 Sampler-only island mode (generate, no training); trainer to sampler weight publish path through storage (HF export plus `LATEST.json`); `rollout_argv` and `island_env` in `RLRunConfig` and the launch layer; `group_index` unique across restarts. First version supports full-parameter only; LoRA publish is refused at launch. Acceptance: CPU tests. GPU: sampler side verified inside L5.4 (2026-10-15, evidence: infra-drafts/SESSION24/gpu/L5.4-review.md); trainer publish not verified. Estimate: 2 d

## L5. Scheduler S1

- [x] L5.1 S1 with default `Q_on = A_hard + 1`, optional `α` giving `min(Q_s1, A_hard + 1)` and ski-rental stop; `prices` accepted and ignored (design D6). Acceptance: CPU tests for every scheduler scenario in the spec. GPU: no. Estimate: 1 d
- [x] L5.2 Read `T_s` from the last `trainer_start_phase` summary; fall back to configuration. Acceptance: CPU test. GPU: no. Estimate: 0.5 d
- [x] L5.3 Scheduler state in the controller journal; continue after head restart (design D8). Acceptance: CPU test of the head-restart scenario. GPU: no. Estimate: 1 d
- [x] L5.3b Controller main loop calls `S1Scheduler.decide` periodically and runs Start and Stop through `DriverLifecycle`; switch `--rl-trainer-scheduler`, default off. Acceptance: CPU tests. GPU: no (not verified on GPU: L5.4 has not passed). Estimate: 1 d
- [x] L5.3c Remote `TrainerLifecycle` (trainer island started through Modal; command channel is a store command file), `binding_factory`, shared consumption watermark, stop when the data end leaves less than one batch. The sky trainer island is not wired and `check` refuses it. Acceptance: CPU tests. GPU: no (not verified on GPU: L5.4 has not passed). Estimate: 2 d
- [ ] L5.3d Scheduler stops the loop and reports an error after a set number of failed incarnations in a row (default 3). Acceptance: CPU test. GPU: no. (not verified: in progress) Estimate: 0.5 d
- [ ] L5.4 GPU check: small model (Qwen3.5-0.8B), one sampler and one trainer in one island, scheduler S1 with a small `T_s` override so that at least two stop/start cycles happen. Acceptance: two `SCHEDULED` cuts; version continues across starts; no sample trained twice; `offpolicy_gate` events present; start phases recorded; L5.5/L5.6/L5.7 fields present (L5.8). GPU: yes, review before launch. Estimate: 1 d + cost (estimate: 2×1 H100 for about 1.5 h, about $16 at $5.3/h per card-hour; to be fixed at review) (partial: sampler island GPU verified 2026-10-15, evidence: infra-drafts/SESSION24/gpu/L5.4-review.md; the full-parameter trainer failed to publish weights in no-sync mode (LoRA export path), so no SCHEDULED cut was verified; not verified on GPU: two stop/start cycles, no sample trained twice. Fix in progress)

## D3/D4. Sync merge metrics and sync timing

Codes follow `infra-drafts/SESSION24/DISCUSS.md` D3 and D4.

- [x] L5.5 Syncer records per outer step: merge weight share per island, carried-over share with discount, rejected share by reason (design D9). The merge weight, `carry_gamma`, and `max_carry_lag` do not change. Acceptance: Rust unit tests and a real-syncer + fake-island CPU test check the shares against hand-computed values. GPU: no. Estimate: 1 d
- [x] L5.6 Island-side timing: `submit_ts`, `global_ready_ts`, `recv_seconds`, `sync_wait_seconds`; replace the constant 0.0 of `sync/remote_quorum_wait_seconds` (`yeto/rl/engine/bridges.py:1096`) with the measured value (design D10). Acceptance: CPU test with a fake syncer that delays the reply; the measured wait is within tolerance of the delay. GPU: no. Estimate: 1 d
- [x] L5.7 Syncer tape: timestamp on every event; arrival offset on `DeltaAccepted`; step duration and last-arriving island on `OuterStep` (design D10). Acceptance: Rust unit tests; existing tape golden tests updated. GPU: no. Estimate: 1 d
- [ ] L5.8 Collect L5.5, L5.6, and L5.7 fields in the GPU checks L5.4 and L6.10. No separate launch. Acceptance: the run directory has the new fields; a short report splits the sync boundary into transfer and wait, and lists the merge shares. GPU: yes, inside L5.4 / L6.10 (no extra cost). Estimate: 0.5 d (not verified on GPU: L6.10 has run without these fields in a scheduler run; L5.4 has not passed)
- [ ] L5.9 Decide with the L5.8 data (design D10 decision rule): transfer first, or slow-island handling (L7, per-island K, grace window and `q_min`). Also decide whether to discuss linear token weighting or a new `carry_gamma` (design D9). Acceptance: the decision is written back into this change. GPU: no. Estimate: 0.5 d (not done: waits for L5.8 data)

## L7. Convertible nodes

- [x] L7.1 Node class `convertible` in the launcher; refuse it on spot or low-precision rollout nodes. Acceptance: CPU tests. GPU: no. Estimate: 0.5 d
- [x] L7.2 Conversion rollout → trainer and back, using SGLang release and Megatron onload; P0-S cut rule for in-flight trajectories (design D7). Acceptance: CPU tests with fakes; the last rollout node is never converted. GPU: no. Estimate: 2 d
- [x] L7.3 Scheduler counts the rollout capacity lost by conversion, and the idle time of a convertible node counts only time with neither rollout nor training. Acceptance: CPU tests. GPU: no. Estimate: 1 d (ConversionScheduler is not yet wired into TrainerLoop; left to the integration stage)
- [x] L7.4 GPU check: 1 convertible node + 1 spot rollout node (or 2×1 H100 on Modal as stand-in), at least one conversion each way. Acceptance: conversion time recorded; after conversion back, the rollout engine weight checksum equals the published checksum of the same version (text is not compared: H100 runs are not deterministic); training continues. GPU: yes, review before launch. Estimate: 1 d + cost not set (GPU verified 2026-10-15, evidence: infra-drafts/SESSION24/gpu/L7.4-review.md; two conversions each way, engine checksum equals published checksum after conversion back, to_trainer about 1.7 s, to_rollout about 2.1 s; fixed a LoRA warm-up crash found in launch 1)
- [ ] L7.5 S3 price and availability policy: not in this change. Interface only (L5.1).
