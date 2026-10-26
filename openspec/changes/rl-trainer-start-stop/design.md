# Design

## Context

Source: `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §3, §4, §9, and the Q5 supplement. Line references are from `s23-p0s-fix` `4cc2c28e` unless marked `d6c58fb8`.

Three layers (memory note "sampler island direction", handoff 4 §6):
1. One island with many rollout nodes and few trainer nodes. Most existing code serves this layer.
2. Cross-cloud sampler islands. Needs a sample data plane and off-island publishing.
3. DiLoCo between trainer islands. Exists.

Node capability classes:
- Trainer class: on-demand instance, fast interconnect, same precision as the trainer.
- Rollout class: may be spot, another region, lower precision.
- Convertible: a trainer-class node that does rollout by default.

Notation (explore §3.1): `T_s` start time, `c` trainer cost per second, `C_s = c·T_s`, `T_b` steady batch time, `Q` backlog in batches, `A` = `A_hard`. The version gap is the current trainer version minus the trajectory start version (`infra-drafts/SESSION24/DISCUSS.md` D2). There is no `A_soft`.

Source for D9 and D10: `infra-drafts/SESSION24/DISCUSS.md` D3 and D4 (decided 2026-10-14). Line references in D9 and D10 are from worktree `s23-g3-fp`.

## Decisions

### D1. Mechanism and policy are separate
`TrainerLifecycle` does what it is told. `TrainerScheduler` decides. `DurableSampleBuffer` holds samples. A test can drive the lifecycle without a scheduler, and the scheduler is a pure function of its inputs.

```text
TrainerLifecycle
  plan_start(spec) -> StartPlan
  start(plan, *, resume_from: CutRef | None) -> TrainerHandle   # acquire → JOIN → restore → warm up
  ready(handle, deadline) -> bool
  run(handle, source: DurableSampleBuffer, until: StopCondition) -> RunReceipt
  drain_and_cut(handle, reason: LeaveReason) -> CutRef           # P0-3 save, trigger SCHEDULED
  stop(handle, reason: LeaveReason) -> None                       # LEAVE and release cards

DurableSampleBuffer
  append(entry: SampleIndexEntry) -> None
  backlog(current_version) -> BacklogStats   # batches, oldest version, version histogram, arrival rate
  select(consumer, limit) -> list[Selected]
  commit_consumed(cursor) -> None            # written into the cut
  on_arrival(callback) -> None

TrainerScheduler
  decide(backlog, lifecycle_state, prices) -> Start | Stop | Hold
```

"Start once" is a scheduler that returns `Hold` until the data ends, then `Stop(FINISHED)`.

### D2. Reuse existing machinery
- Each start is a new incarnation JOIN (`elastic_client.py` `Join`, `JoinAck` with `base_version`, `catch_up`). Lease default 150 s.
- Stop reasons are the P5 codes: scheduler stop `STOPPED` (2), data end `FINISHED` (0), spot reclaim `PREEMPTED` (3), errors by `leave_reason_for`.
- Restore uses `CuttableTrainer.restore_cut` on a new trainer (`ports.py:260`). Cut storage uses `CheckpointStore` (`resume.py`).
- `TrainerLifecycle.run` wraps an `IslandDriver`; `until` maps to the round limit.

### D3. `SCHEDULED` trigger
Add `SCHEDULED = "scheduled"` to `recovery.TRIGGERS` (`recovery.py:39-42`). `RoundCutCheckpoint.save` uses it when the stop comes from the scheduler. `MANUAL` stays for the last round and for a user stop. Resume logic treats `SCHEDULED` like `MANUAL`: the cut is complete and the next start restores it.

### D4. Buffer watermark in the cut
The consumed watermark is written into the same cut record as the trainer state. On restore, the trainer resumes at the watermark. A sample is consumed at most once per cut lineage. Samples appended after the watermark stay in the buffer and go through the version gate at the next start.

First implementation: index in object storage (or the existing bucket mount store) next to the sample payloads; arrival notification by polling the index with a fixed interval. A push notification is a later option.

### D5. Continuous publishing (Q1)
While the trainer is in `TRAINING`, it publishes each new version through the existing publish path. Samplers that read from the buffer load the new weights by the existing rollout weight update. Samples carry their behavior version (P0-S schema 4). When the trainer stops, samplers keep the last published version.

### D6. Scheduler S1
- Start when `Q ≥ Q_on`. Default: `Q_on = A_hard + 1`. This default does not change after D2: the gap is counted from the start version, and only `A_hard` limits the backlog. Optional: when the configuration gives `α` (allowed start cost share), `Q_on = min(Q_s1, A_hard + 1)` with `Q_s1 = ceil(T_s·(1−α)/(α·T_b))`. `α` has no default.
- Stop after the backlog is empty and the idle time reaches `T_idle = T_s` (ski rental: idle cost equals one start cost; worst case at most 2× the optimum, a classical result).
- `T_s` comes from the last measured start (`trainer_start_phase` summary in `rl-trainer-cold-start-cost`). Without a measurement, `T_s` is a configured value.
- Example (estimate, explore §3.3): `T_s ≈ 500 s`, `T_b ≈ 44 s`, `α = 0.2` → `Q_s1 ≥ 45` batches. With `A_hard = 2`, `Q_on = 3`. So for small models the cap decides.
- `prices` is accepted and not used. S2 (rate switch) and S3 (price, availability, preemption) are later work.

### D7. Convertible nodes (Q5)
- Only trainer-class nodes can be convertible. A spot or low-precision rollout node cannot.
- Switch rollout → trainer: stop new requests to the node's SGLang; in-flight long trajectories are cut by the P0-S rule (finished segments go back to the buffer; unfinished background trajectories are dropped and regenerated, P0-S-Q1 ③); SGLang releases memory; Megatron onloads. The measured wake_up is about 1 s (explore §2 H, 0.8B, colocated).
- A multi-GPU trainer needs a group of convertible nodes. The scheduler counts the rollout capacity lost by the switch.
- Switch trainer → rollout: the same ski-rental rule decides, so the node stops later than a pure warm node would.
- The idle cost of a convertible node is near zero while it does rollout, so `T_idle` counts only time when it does neither.

### D8. Scheduler state survives a head restart
The scheduler writes its decisions and inputs to the controller journal, in the same way as `restore_auto_state` (`elastic.py:48`). After a head restart, the scheduler reads the journal and continues.

### D9. Island submit rule and merge weight stay; add merge metrics (D3)
- An island submits its update to the syncer after it trains on K trajectories. There is no time threshold on the island.
- The merge weight stays `w = c_tokens² / c_steps` (`learner_weight`, `syncer/src/merge.rs:26-37`).
- A late delta with lag 1..`max_carry_lag` goes into the next step with weight factor `carry_gamma^lag`. A delta with a larger lag is rejected (`ElasticState::submit`, `syncer/src/elastic.rs:1172-1195`). `carry_gamma` stays 0.5 and `max_carry_lag` stays 2 (values from `DISCUSS.md` D3).
- New metrics per outer step: the merge weight share of each island; the share of deltas that are carried over, with their discount; the share of deltas that are rejected, by reason.
- Known risk (inferred, not measured): K counts trajectories, and agentic trajectory length varies. The squared token count can give an island with many long trajectories a much larger share. When the metrics show this, we discuss linear token weighting (as in Decoupled DiLoCo, arXiv 2604.21428) and a different `carry_gamma`.

### D10. Measure the sync boundary before any change (D4)
Existing evidence (`s1-runs/s23-g3-fp-main-20261014d`, full-parameter sync, 2 islands, 3 rounds): the sync boundary is 92–97% of each round. Training is 10–12 s. The two islands differ by less than 5 s. The existing events cannot split the boundary into transfer and wait: syncer tape events have no timestamp, and `sync/remote_quorum_wait_seconds` is hard-coded to 0.0 (`yeto/rl/engine/bridges.py:1096`). Inferred: most of the time is full-parameter transfer.

New fields:
- Island side: `submit_ts`, `global_ready_ts`, `recv_seconds`, `sync_wait_seconds`. `sync/remote_quorum_wait_seconds` gets the measured value.
- Syncer tape: every event gets a timestamp. `DeltaAccepted` gets the arrival offset from the start of the outer step. `OuterStep` gets the step duration and the last island to arrive.

Collection: the fields are collected in the planned GPU checks L5.4 and L6.10. No separate GPU launch. An elastic run with islands of different speed needs its own budget review.

Decision rule after the data:
- Transfer is most of the boundary → optimize transfer (S23 leftover item 2).
- A slow island holds the others → in this order: convert waiting trainer nodes to rollout (L7); set K per island by island speed (DyLU idea); keep the grace window and `q_min`.

### Non-goals and rejected alternatives
- Time-based or "K or T, first wins" submit (D3): rejected for now. A variable sample count per update needs new learning-rate and loss normalization rules, and there is no data that a timer helps.
- Local training while waiting for the syncer (D4): not done. Reasons: (1) naive delayed updates hurt convergence (Eager Updates, arXiv 2502.12996: 500M model loss 2.67 → 3.01); (2) there is no RL precedent, and we infer the risk is higher in RL; (3) the upper bound of the gain is small, because the measured training time is 10–12 s per round. Eager updates are not a fallback option.

## Risks

- `IslandDriver.rebuild_trainer` (`driver.py:1605`) assumes a driver object life cycle. Repeated starts may break this assumption. L3 starts with a fake trainer to find it.
- Polling arrival adds delay up to one interval. Acceptable for S1 because `T_s` is minutes.
- Convertible switch reduces rollout capacity exactly when backlog is high. The scheduler must not convert the last rollout node.
