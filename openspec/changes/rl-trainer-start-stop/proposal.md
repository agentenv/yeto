# Proposal

## Why

The direction after S23 is a dedicated sampler island: rollout runs on cheap or different hardware (spot, other regions, lower precision), writes trajectories with behavior logprobs, and a trainer consumes them with off-policy correction. Sampler supply and trainer demand do not match in time. A trainer that runs all the time pays for idle GPUs; a trainer that starts only once cannot follow a changing backlog.

The user decided (2026-10-14, `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §9 and the Q5 supplement):
- The trainer is designed to start and stop many times. "Start once" is the special case where the scheduler never stops it.
- Reuse what exists: JOIN/LEAVE and lease (`yeto/rl/elastic_client.py`), the P5 leave reason codes, the P0-3 recovery abstraction, and the P0-S cut and resume.
- The sample buffer is durable. The trainer can also be started when rollouts arrive.
- Q1: while it runs, the trainer keeps publishing weights to the samplers. Version gap is measured from the trajectory start version (D2, 2026-10-14). Samples within `A_hard` are trained with IS correction; groups over `A_hard` are dropped (gate in `rl-offpolicy-correction-passthrough`). `A_soft` is removed.
- Q2: a new cut trigger kind `SCHEDULED`.
- Q5: convertible nodes. A convertible node is in the trainer capability class. By default it does rollout. The scheduler decides when it becomes a trainer. The switch is the existing colocated sleep and onload. The stop rule is ski rental: stop when the idle cost reaches the cost of one start.
- Start/stop policy: build the S1 framework first; `Q_on = A+1` by default; optional `α` gives `Q_on = min(S1, A+1)` (no default for `α`); the price term is only an interface.

Also decided on 2026-10-14 (`infra-drafts/SESSION24/DISCUSS.md` D3, D4):
- D3: an island submits its update to the syncer when it has K trajectories, not on a timer. Merge weight, `carry_gamma`, and `max_carry_lag` stay as they are. New metrics show the merge share per island, the carried-over share, and the rejected share.
- D4: the trainer does not continue local training while it waits for the syncer. First, measure where the sync boundary time goes. Then decide.

Key constraint (explore §3.2, derived): while the trainer is stopped, the policy version does not advance. If the trainer then trains the backlog batch by batch, batch k has gap k. So one start can consume about `A+1` batches of backlog by version rules alone. Q1 (continuous publishing) makes new samples newer while the trainer runs; `Q_on` is capped at `A+1` for the backlog that exists at start.

## What Changes

- New port `TrainerLifecycle`: plan, start (acquire, JOIN, restore cut, warm up), run, drain and cut, stop (LEAVE). States `COLD → ACQUIRING → JOINING → RESTORING → WARMING → TRAINING → DRAINING → STOPPED`.
- New port `DurableSampleBuffer`: append, backlog statistics, select, commit consumed watermark, arrival notification. The watermark is part of the consistent cut.
- New port `TrainerScheduler`: `decide(backlog, lifecycle_state, prices) -> Start | Stop | Hold`. The first implementation is S1 with the `A+1` cap and the ski-rental idle stop. `prices` is accepted and ignored.
- New cut trigger kind `SCHEDULED` for a stop that the scheduler orders.
- Continuous weight publishing from a running trainer to the samplers that read the buffer.
- Convertible nodes: a node class flag; the switch from rollout to trainer uses SGLang release and Megatron onload; in-flight long trajectories go back to the buffer by the P0-S cut rule.
- Scheduler state goes into the controller journal, so that a head restart continues the decision state.
- Sync merge metrics (D3): per step, the merge weight share of each island, the share of carried-over deltas and their discount, and the share of rejected deltas.
- Sync timing measurement (D4): island-side `submit_ts`, `global_ready_ts`, `recv_seconds`, `sync_wait_seconds`; a real value for `sync/remote_quorum_wait_seconds`; timestamps on syncer tape events, arrival offset on `DeltaAccepted`, step duration and last-arriving island on `OuterStep`.
- Non-goal: local training on the trainer while it waits for the global weights (D4). See design "Non-goals and rejected alternatives".
- Non-goal in this change: a cross-cloud sampler island data plane (handoff §6 items A2/A3/A4). This change uses an in-island or same-storage buffer first. The ports are shaped so that a cross-island source can implement them later.
- Non-goal: price and availability policy (S3) beyond the interface.

## Capabilities

### New Capabilities
- `rl-trainer-lifecycle`: repeated trainer start and stop, a durable sample buffer with a cut-safe watermark, a scheduler with the S1 rule, the `SCHEDULED` trigger, continuous publishing, and convertible nodes.

### Modified Capabilities
<!-- None on main. P0-3 recovery (not yet on main) gets one more trigger value; this change states it as an ADDED requirement and does not edit the P0-3 change directory. -->

## Impact

- New: `yeto/rl/engine/trainer_lifecycle.py`, `yeto/rl/engine/sample_buffer.py`, `yeto/rl/engine/trainer_scheduler.py` (names are proposals).
- `yeto/rl/engine/recovery.py` (`TRIGGERS` + `SCHEDULED`), `yeto/rl/engine/resume.py` (watermark in the cut record), `yeto/rl/engine/sample_pool.py` (durable form of `SamplePoolIndex`), `yeto/rl/engine/driver.py` (`IslandDriver` wrapped by `TrainerLifecycle.run`), `yeto/rl/engine/elastic.py` (journal of scheduler state).
- `yeto/rl/engine/bridges.py` (`remote_quorum_wait_seconds`, now hard-coded to 0 at `:1096`, checked on `s23-g3-fp`), syncer `syncer/src/elastic.rs` (tape events) and its merge-share metrics. `syncer/src/merge.rs` (`learner_weight`) does not change.
- Launcher: node class `convertible`; scheduler process on the head.
- Depends on the S23 integration branch (P0-3, P5, P0-S), which is not on `d6c58fb8`. Line references in design.md are from `s23-p0s-fix` `4cc2c28e`.
- Depends on `rl-offpolicy-correction-passthrough` (gate) for the GPU checks, and uses `T_s` from `rl-trainer-cold-start-cost`.
