# Design

## Context

The facts from reading the code are in "Why" of proposal.md. The core facts: an abnormal exit already sends LEAVE through the `finally` at `driver.py:1767-1770` → `bridges.py:655-664`. What is missing is the reason code, the visibility of failures, the lease default and the tests. The user decisions are in "Decided" of proposal.md.

## Goals / Non-Goals

Goals:
- When an island exits, the syncer receives a LEAVE that carries one of the 7 reason codes.
- A failure to send LEAVE is visible.
- In the hard-kill case, the wait before removal shrinks to a 150 s lease.
- Tests lock the behavior above.

Non-Goals:
- The launcher sending LEAVE on behalf of an island (see D5).
- Fault isolation on the strict path (fixed_roster).
- A CLI for adding an island during a run.
- Verification on GPU.

## Decisions

D1: The driver layer decides the reason code. The bridge only sends it.
- How: `ElasticAvgSync` gets a new `abort(reason)` method, or `close()` gets a `reason` parameter (choose one at implementation time; the smaller change wins). In the `except` branch, the driver looks up the code in a mapping table by exception class.
- Reason: only the driver knows why the run exits.

D2: A LEAVE send failure is not re-raised.
- How: emit an `elastic_leave_failed` event with the exception type and text.
- Reason: at this time the driver is propagating the original error. A re-raise would hide it.

D3: `StrictAvgSync` and `DecoupledSync` do not change.
- Reason: the strict path has no LEAVE semantics. The close of the decoupled bridge only closes the bridge (`bridges.py:1162-1164`), and this is out of scope for this change.

D4: On the Rust side, add only constants and log code names. The merge logic does not change.
- For any Leave, the syncer already calls `purge_departed` (`elastic_server.rs:450`). This behavior already meets the requirement.
- An unknown code is recorded as 0 (`elastic.rs:742`).

D5: A code expresses only the broad class. The details go to the event log.
- A code is a 1-byte wire protocol field. The exception class name and the error message go to the island event log. They do not go on the wire.
- The definition of the 7 codes is in proposal.md. Code 1 LEASE_EXPIRED is decided by the syncer itself. An island never sends it.
- First implementation step: list the mapping from all exception classes under `yeto/rl/engine` to codes (for example where CutError and JournalError belong), and write it in tasks.md 1.1. Show the mapping table to the user once, after it is listed.
- An unmapped exception goes to INFRA_FAILED by default (decided, see D7 Q2).

D6: The launcher sending LEAVE on behalf of an island was evaluated and is not done for now.
- Feasibility: the launcher polls every 30 s and can see that an island died (FleetController, `launcher.py:6081`). It also holds the HMAC key, so it can send LEAVE on behalf of the island.
- Reason not to do it: a restarted island uses the same island number. A late LEAVE would wrongly remove the new island.
- Prerequisite work: add a generation field to the LEAVE frame, and change the frame format on both the Rust and the Python side. The change surface is large, so it is postponed.
- Alternative: change the default island lease of the elastic mode from 300 s to 150 s. The heartbeat interval is `lease_s/3` (`elastic_client.py:577`). At 150 s it is 50 s, which tolerates 2 lost heartbeats in a row.
- Three places MUST have the same value: `--island-lease-s` at `syncer/src/main.rs:159-162`, and `--rl-island-lease-s` in the launcher and in `yeto`. A test asserts that the three places are equal.
- 300 was just set by tasks 0.8 of `rl-code-quality-cleanup`. This change changes it again, and does not touch that directory.


D7: Exception → reason code mapping (reviewed by the user on 2026-10-13, all items follow the recommendation)
- The full mapping table is in `leave-code-map.md` in this directory (with file:line numbers). Summary: normal end, or the syncer sent FINISHED first → 0; KeyboardInterrupt and SystemExit → 2; a reclaim notice for a droppable island (AWS, Nebius, Modal) → 3; node loss, reconfiguration or rebuild failure, storage and resume errors, and unmapped exceptions → 4; training checks such as PublicationError, PolicyIdentityError, RoundFailedError and StrictRlInvariantError → 5; config mismatch → 6. Code 1 is decided only by the syncer.
- Q1: Add `IslandRecoveryRequired(DriverError)`. The 6 RECOVERY_REQUIRED raises in driver.py change to raise it. It maps to 4 by class, not by string match.
- Q2: An unmapped exception defaults to 4 INFRA_FAILED.
- Q3: Option a. A non-droppable island that receives SIGTERM keeps the current behavior (it sends no LEAVE, and the syncer removes it as 1). The reclaim SIGTERM of a droppable island sends 3. Writing a stop marker in the CLI to tell 2 and 3 apart is left for later.
- Q4: CutError, the JournalError family, StoreIntegrityError and LedgerError map to 4. The specific class names go to the event log.
- Q5: When the new syncer receives an unknown code, it logs `unknown(<n>)`, removes the island as usual, and raises no error.
- Q6: A failure before JOIN (startup checks, JOIN refused) puts no LEAVE on the wire. Only code 6 is recorded in the event log.
- Island-side lease: the launcher passes the same `--rl-island-lease-s` to the syncer and to every elastic island. `ElasticAvgSync` hands it to the client, and the heartbeat interval is lease/3. The client default is also changed to 150 (`DEFAULT_ISLAND_LEASE_S`), and a test locks the four places to the same value. (The main agent acceptance on 10-13 found that the island side still used the 30 s default.)
- Launcher wrap-up exemption: `rl_learner_finalized` is waived only when the code name of the last `pool_leave` of the syncer is finished, stopped, preempted or lease_expired. For infra_failed, training_failed, config_mismatch, unknown, and requested from an old syncer, the launcher still reports a failure. (Decided by the main agent on 10-13: requested is not waived. The syncer and the launcher are built from the same repository at the same version, so it is accepted that an old test run does not resume; see P0-D5.)
- verl island lease: a verl elastic island and a Miles island are generated by the same launcher function `make_miles_island_task`; only the entry module changes to `yeto.rl.adapters.verl.island_entry`. Both pass the island arguments through `_ports_infra_flags`, so `--rl-island-lease-s` is passed along with them. The verl entry parses this argument into `island_lease_s` in the plan, and `adapters/verl/trainer.py` hands it to `ElasticAvgSync`. A test locks that the verl island command contains this argument.
- Wrapped exception: when a plain `DriverError` carries `__cause__`, the code follows the cause (for example, a wrapped ProfileError → 6). `PublicationError` of the Miles adapter layer (`adapters/miles/publish.py::PublicationError`) maps to 5.
- Event log: `island_leave` (code, code name, exception class name, error message) is written only on an exceptional exit. A normal end does not write it, so the recorded golden tape does not change.
- Interface: the bridge `close(reason=FINISHED)`. One process sends LEAVE only once (decided by `left`; the 3 that the reclaim listener sent first is not overwritten). No bridge may send 1 (if 1 is passed, send it as 4). The code definitions exist in two places only, `elastic_client.LeaveReason` and the constants in `elastic.rs`, and a parity test locks them.

## Risks / Trade-offs

- The old syncer records an unknown code as 0 (FINISHED). This affects only the log text. It does not affect the merge.
- With the lease shortened to 150 s, a healthy island with slow heartbeats may be removed by mistake. The heartbeat is 50 s, so 2 in a row can be lost. Packet loss on real networks is unverified.
- After a hard kill, the wait for removal is still up to 150 s.

## Verification plan

- CPU: Python unit tests, Rust `cargo test`, a reason code parity test, and an end-to-end test (a real Rust syncer plus two simulated islands; one island leaves for each class of reason, and the other island keeps producing rounds).
- Lease timeout: the test verifies the removal with a shortened lease. It does not wait the real 150 s.
- No run on GPU.
