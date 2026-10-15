# Spec Delta

## Purpose

Let a trainer start and stop many times against a durable sample buffer, with stop decisions made by a separate scheduler. A trainer that starts once is the special case.

## ADDED Requirements

### Requirement: Trainer lifecycle states
A trainer instance SHALL move only through `COLD → ACQUIRING → JOINING → RESTORING → WARMING → TRAINING → DRAINING → STOPPED`, and from any state to `STOPPED` on failure. Each start SHALL JOIN as a new incarnation, and each stop SHALL send LEAVE with a P5 reason code.

#### Scenario: Scheduled stop
- **WHEN** the scheduler orders a stop while the trainer is in `TRAINING`
- **THEN** the trainer drains, saves a cut with trigger `SCHEDULED`, and sends LEAVE with reason `STOPPED`

#### Scenario: Spot reclaim
- **WHEN** the trainer gets a reclaim notice
- **THEN** it saves a cut with trigger `PREEMPT` if time allows, and sends LEAVE with reason `PREEMPTED`

### Requirement: Restart continues from the last cut
A start with a previous cut SHALL restore the trainer state and the buffer watermark from that cut. Training SHALL continue at the restored policy version.

#### Scenario: Stop then start
- **WHEN** the trainer stops at version v with a `SCHEDULED` cut and later starts again
- **THEN** the first published version after the start is v+1, and no sample before the watermark is trained again

### Requirement: Durable buffer with cut-safe watermark
The sample buffer SHALL keep samples across trainer stops and head restarts. The consumed watermark SHALL be written into the same cut record as the trainer state.

#### Scenario: Trainer crash after consuming
- **WHEN** the trainer consumes samples and fails before the next cut
- **THEN** the next start resumes at the watermark of the last cut, and those samples are consumed again

### Requirement: Scheduler decides start and stop
The scheduler SHALL be a function of backlog statistics, lifecycle state, and prices that returns `Start`, `Stop`, or `Hold`. The S1 implementation SHALL start when `Q ≥ A_hard + 1` by default, and when `Q ≥ min(Q_s1, A_hard + 1)` if the configuration gives `α`. `α` SHALL have no default. The S1 implementation SHALL stop when the backlog is empty and the idle time reaches the measured or configured start time `T_s`. The S1 implementation SHALL ignore prices.

#### Scenario: Cap by version limit
- **WHEN** `α` is set so that `Q_s1 = 45`, `A_hard = 2`, and the backlog is 3 batches
- **THEN** the scheduler returns `Start`

#### Scenario: Default threshold without α
- **WHEN** `α` is not set, `A_hard = 2`, and the backlog is 2 batches
- **THEN** the scheduler returns `Hold`; at 3 batches it returns `Start`

#### Scenario: Ski-rental stop
- **WHEN** the backlog is empty and the trainer has been idle for `T_s`
- **THEN** the scheduler returns `Stop` with reason `STOPPED`

#### Scenario: Start once
- **WHEN** the scheduler is configured as always-on
- **THEN** it returns `Hold` until the data ends, then `Stop` with reason `FINISHED`

### Requirement: Continuous publishing while training
While the trainer is in `TRAINING`, it SHALL publish each new policy version to the samplers that feed the buffer. Each sample SHALL record its behavior policy version.

#### Scenario: Sampler follows the trainer
- **WHEN** the trainer publishes version v+1
- **THEN** samples appended after the sampler loads v+1 record behavior version v+1

### Requirement: Convertible nodes
Only trainer-class nodes SHALL be marked convertible. A convertible node SHALL do rollout until the scheduler converts it. The conversion SHALL release the rollout engine memory before the trainer onloads, and SHALL return in-flight trajectory segments to the buffer by the P0-S cut rule. The scheduler SHALL NOT convert the last rollout-capable node.

#### Scenario: Convert to trainer
- **WHEN** the scheduler converts a convertible node
- **THEN** the node's rollout engine stops taking requests, releases memory, and the trainer onloads on the same GPUs

#### Scenario: Spot node marked convertible
- **WHEN** the launch configuration marks a spot rollout node as convertible
- **THEN** launch fails with an error that names the node and the class rule

### Requirement: Scheduler state survives head restart
The scheduler SHALL record its inputs and decisions in the controller journal, and SHALL continue from the journal after a head restart.

#### Scenario: Head restart while trainer is idle
- **WHEN** the head restarts while the trainer is idle for time t
- **THEN** the scheduler continues with idle time t, not 0

### Requirement: Scheduled cut trigger kind
The consistent cut trigger SHALL be one of `PREEMPT`, `PERIODIC`, `MANUAL`, `SCHEDULED`. `SCHEDULED` SHALL be used only for a stop that the trainer scheduler orders. Resume SHALL treat a `SCHEDULED` cut as complete.

#### Scenario: Resume from a scheduled cut
- **WHEN** the newest cut has trigger `SCHEDULED`
- **THEN** resume restores it in the same way as a `MANUAL` cut

### Requirement: Island submits after K trajectories
An island SHALL submit its update to the syncer after it trains on K trajectories. An island SHALL NOT submit on a timer. The syncer SHALL keep the merge weight `c_tokens² / c_steps`, `carry_gamma` 0.5, and `max_carry_lag` 2 as defaults.

#### Scenario: Slow island does not submit early
- **WHEN** an island has fewer than K trajectories and other islands have already submitted
- **THEN** the island does not submit a partial update

### Requirement: Sync merge metrics
For each outer step, the syncer SHALL record the merge weight share of each island, the share of deltas carried over with their discount, and the share of deltas rejected with their reason.

#### Scenario: Late delta carried over
- **WHEN** a delta arrives with lag 1 and `carry_gamma` is 0.5
- **THEN** the next outer step records that delta as carried over with discount 0.5, and its merge share includes the discount

#### Scenario: Delta over the lag limit
- **WHEN** a delta arrives with lag 3 and `max_carry_lag` is 2
- **THEN** the step records it as rejected with reason `carry_lag_exceeded`

### Requirement: Sync boundary timing
Each island SHALL record `submit_ts`, `global_ready_ts`, `recv_seconds`, and `sync_wait_seconds` for each round, and SHALL report the measured value of `sync/remote_quorum_wait_seconds`. Every syncer tape event SHALL carry a timestamp. `DeltaAccepted` SHALL carry the arrival offset from the start of the outer step. `OuterStep` SHALL carry the step duration and the last island to arrive.

#### Scenario: Wait and transfer are separable
- **WHEN** a round finishes on an island
- **THEN** the round record gives `sync_wait_seconds` and `recv_seconds` as separate values, and `sync/remote_quorum_wait_seconds` is not a constant 0

### Requirement: No local training while waiting for global weights
The trainer SHALL NOT take optimizer steps on the local copy of the weights between submit and the arrival of the global weights.

#### Scenario: Trainer waits
- **WHEN** an island has submitted its update and the global weights have not arrived
- **THEN** the island trainer takes no optimizer step until the global weights arrive
