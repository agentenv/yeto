## 0. Decided

- [x] 0.1 The user decided items 1–4 of "Decided" in proposal.md on 2026-10-13

## 1. Reason codes

- [x] 1.1 (Reviewed by the user on 10-13; see `leave-code-map.md` and design.md D7) List the mapping from all exception classes under `yeto/rl/engine` to the 7 codes (including CutError, JournalError and others), and write it in this directory. Show the mapping table to the user once after it is listed, and then continue with the tasks from 1.2
- [x] 1.2 In `yeto/rl/elastic_client.py`, define 7 constants (`FINISHED=0`, `LEASE_EXPIRED=1`, `STOPPED=2`, `PREEMPTED=3`, `INFRA_FAILED=4`, `TRAINING_FAILED=5`, `CONFIG_MISMATCH=6`). Rename `REQUESTED` to `FINISHED`; the value does not change
- [x] 1.3 In `syncer/src/elastic.rs`, define constants with the same values. The log at `elastic.rs:742` uses the code name. An unknown code is logged as `unknown(<n>)` and raises no error (D7 Q5)
- [x] 1.4 Parity test: the numeric values of the seven reason codes are equal in Python and in Rust

## 2. The exit path sends LEAVE

- [x] 2.0 Add `IslandRecoveryRequired(DriverError)` to `driver.py`. The 6 RECOVERY_REQUIRED raises change to raise it (D7 Q1)
- [x] 2.1 Each bridge supports closing with a reason code through `close(reason=FINISHED)` (D1, D7)
- [x] 2.2 `driver.py` selects the code by the mapping table: normal finish FINISHED; user stop (KeyboardInterrupt, SystemExit) STOPPED; SIGTERM keeps the current behavior (D7 Q3 option a); the spot reclaim notice path (`preemption.py`, `droppable.py`) PREEMPTED; the rest follows the mapping table, and an unmapped exception defaults to INFRA_FAILED (D7 Q2)
- [x] 2.3 Write the specific exception class name and error message to the island event log. They do not go on the wire (D5)
- [x] 2.4 When sending LEAVE fails, emit an `elastic_leave_failed` event and do not re-raise (D2)

## 3. Lease default 300 → 150

- [x] 3.1 Change the default of `--island-lease-s` at `syncer/src/main.rs:159-162` to 150
- [x] 3.2 Change the default of `--rl-island-lease-s` in the launcher and in `yeto` to 150
- [x] 3.4 The launcher passes the lease to the elastic islands, and the client on the island side uses the same value (acceptance rework)
- [x] 3.5 At wrap-up, the launcher waives `rl_learner_finalized` only for finished/stopped/preempted/lease_expired (acceptance rework)
- [x] 3.3 Add or change a test that asserts the defaults in the three places are equal (see tasks 0.8 of `rl-code-quality-cleanup`). Do not change the `rl-code-quality-cleanup` directory

## 4. Tests (CPU)

- [x] 4.1 Unit test: the LEAVE reason code for each class of reason (including RECOVERY_REQUIRED → INFRA_FAILED)
- [x] 4.2 Unit test: when sending LEAVE raises, emit `elastic_leave_failed`, and the original DriverError still propagates
- [x] 4.3 End-to-end: a real Rust syncer plus two simulated islands. One island leaves for each class of reason, and the other island keeps producing rounds before the soft deadline. The syncer log contains the matching code name
- [x] 4.4 (Covered by an existing Rust test: `three_islands_late_expiry_and_midrun_join` in `elastic_server.rs`, lease 0.6 s) End-to-end: an island sends no LEAVE (simulated hard kill). The test uses a shortened lease and verifies the removal at lease timeout, and the other island keeps producing rounds
- [x] 4.5 `cargo test` and the local safe test set pass (10-13: cargo 159 passed; full run 16 failed/33 errors, the only differences from the baseline are 7 multinode thread-gate items [environment: threads >2900], and the baseline test_quorum_lease_catch_up passes this time)

## 5. Docs

- [x] 5.1 (End section of `docs/PROTOCOL.md`) Write the manual steps for adding an island (start a separate island process with the same `--rl-syncer-epoch`, `yeto/cli.py:531`). Do not build a CLI for adding an island
