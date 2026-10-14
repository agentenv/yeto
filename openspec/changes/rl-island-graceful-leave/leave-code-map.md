# P5: Island leave paths → LEAVE reason code mapping table (draft, for user review)

Written on 2026-10-13. P5 sub-agent. Code baseline: worktree `/home/michael/work/s23-t9`, branch `s23-t9`, HEAD `f2db57f8`.
This file is only a mapping table and conventions. **No implementation code was written.** Line numbers refer to `f2db57f8`.
The definition of the 7 codes is in `openspec/changes/rl-island-graceful-leave/proposal.md` (decided P5-1).

| Value | Name | Sent by |
|---|---|---|
| 0 | FINISHED | island |
| 1 | LEASE_EXPIRED | decided only by the syncer itself; an island does not send it |
| 2 | STOPPED | island |
| 3 | PREEMPTED | island |
| 4 | INFRA_FAILED | island |
| 5 | TRAINING_FAILED | island |
| 6 | CONFIG_MISMATCH | island |

## 0. Three facts from reading the code

1. **LEAVE reaches the wire only after JOIN succeeds.** `ElasticClient.leave()` sends no frame when `self.ack is None` (`yeto/rl/elastic_client.py:853-856`). So an exception raised before `sync.start()` (startup, argument checks, JOIN refused) produces no LEAVE and needs no code. In the table, such paths are marked "not on the wire".
2. **Many exceptions in the driver are wrapped as a plain `DriverError`.** Node loss (`RecoveryRequired`) is wrapped as `DriverError("island is RECOVERY_REQUIRED: …")`, see `yeto/rl/engine/driver.py:727,740,770,792,1310,1540`. If the mapping uses the exception class only, node loss is wrongly assigned to TRAINING_FAILED. See open question Q1.
3. **SIGTERM does not go through `finally` today.** On a non-droppable island, `install_preempt_handler` only writes the preemption marker, and then re-sends the signal with the default action to kill the process (`yeto/rl/engine/resume.py:484-520`, installed at `yeto/rl/adapters/miles/entry.py:825`). The process dies directly. The `finally` of `driver.run()` (`driver.py:1767-1770`) does not run, no LEAVE is sent, and the syncer finally removes the island as LEASE_EXPIRED.

## 1. Python side: inside the driver (`IslandDriver.run`, `driver.py:1713-1770`)

All the exceptions below go through `finally → self.sync.close()` (`driver.py:1770`) → `ElasticAvgSync.close()` (`yeto/rl/engine/bridges.py:655-663`) to send LEAVE. Today it always sends 0 (`LEAVE_REASON_REQUESTED`).

### 1.1 Normal end

| # | Path (file:line) | Code | Reason |
|---|---|---|---|
| N1 | In the loop, `boundary.stop` is true → `self.sync.finish(self)` (`driver.py:1757`) → `bridges.py:645-653` sends LEAVE | 0 FINISHED | Training ran to the end as planned |
| N2 | The syncer ends first: `ElasticFinished` is caught at `bridges.py:612` → `_finished()` sends LEAVE (`bridges.py:617-637`) | 0 FINISHED | The syncer already sent FINISHED, or disconnected at the final version, and the island finishes normally |
| N3 | `sync.start()` returns `start.finished=True` (`driver.py:1744`) → goes directly to N1 | 0 FINISHED | On resume, the syncer is already at the final version |

### 1.2 User stop

| # | Path | Code | Reason |
|---|---|---|---|
| S1 | `KeyboardInterrupt` (Ctrl-C / SIGINT default handling) → `finally` | 2 STOPPED | The user interrupts on purpose; `_classify_failure` also explicitly assigns no cause (`driver.py:753`) |
| S2 | `SystemExit` → `finally` | 2 STOPPED | Explicit exit; same as above |
| S3 | `yeto down` / `yeto stop-run`: SIGTERM, SIGKILL after 10 s (`yeto/cli.py:2617-2623`) | Today no LEAVE → the syncer records 1 | See fact 3; sending 2 needs a change to signal handling, see Q3 |

### 1.3 Infrastructure failure → 4 INFRA_FAILED

| # | Path | Reason |
|---|---|---|
| I1 | Node loss: `_refuse_if_recovery_required` (`driver.py:727`), `_probe_nodes` (`:740`), node loss detected by `_classify_failure` (`:770`), `_confirm_recovery` (`:792`), the pre-check of `run_round` (`:1310`), and `RecoveryRequired` inside a safe point (`:1540`). The source is 8 places starting at `controller.py:1102,1192,1532` and `adapters/miles/trainer_rebuild.py:258` | The node or the trainer cannot recover; this is a hardware or environment problem |
| I2 | `RecoveryRequired` escapes without being wrapped (two classes with the same name, `controller.py:124` and `engine/errors.py:13`) | Same as I1 |
| I3 | `TransactionFailed` (`controller.py:117`, 17 places starting at `:1061`), `TrainerTransitionFailed` (`trainer_transition.py:53`, raised from `:313`), `RebuildRefused` (`adapters/miles/rebuild_wiring.py:75`, 6 places), `RebuildNotStarted` (`driver.py:1568-1573`) | Engine reconfiguration or trainer rebuild failed; this is a run environment problem |
| I4 | Unclassified third-party exceptions: CUDA OOM, Ray actor death, `OSError`/`ConnectionError`, `ToolWaitError`/`HarnessAdmissionError` (`tool_wait.py:50,292`) | A process or an external dependency crashed; also the default code, see Q2 |
| I5 | `ElasticRejoinError`: 5 re-entry failures in a row (`elastic_client.py:726-731`, thrown by `check()` at `:735-741`) | The syncer is unreachable; note that LEAVE most likely cannot be sent either |
| I6 | `CutError` (`engine/cut.py:40`; 23 places starting at `adapters/miles/trainer.py:545`), the `JournalError` family (`journal.py:40-52`), `StoreIntegrityError` (`resume.py:50`), `LedgerError` (`ledger.py:67`, `island_ledger.py:96`) | Storage, journal or resume state is corrupted; assigned to the environment. This is ambiguous, see Q4 |

### 1.4 Training check failed → 5 TRAINING_FAILED

| # | Path | Reason |
|---|---|---|
| T1 | `PublicationError` (`driver.py:664,682,689,818,1606,1609`) | The publish check is inconsistent |
| T2 | `PolicyIdentityError` (`driver.py:840-884`, 6 places) | The policy token of the rollout does not match |
| T3 | `RoundFailedError`: optimizer step failed (`driver.py:905`) | The training step itself failed |
| T4 | `StrictRlInvariantError`: policy_tensor_hash mismatch (`:625`), reward not finite (`:897`), gradient not finite or zero (`:908,923,928`), hash mismatch after rebuild (`:1580`) | Training numeric checks; the driver also emits `rl_strict_failure` (`:1759-1766`) |
| T5 | `DriverError("generation … admission fenced")` (`:823`), `"run reached max_rollouts without a stop"` (`:1747`), data cursor restore failure (`:1659,1666,1672,1679`) | The driver's own flow checks |
| T6 | `CrossChannelCommitError` (`bridges.py:670`, 4 places starting at `:773`), `WeightsChanged`/`PolicyDigestError` (`policy_digest.py:42-46`), `GradAuditError`, `AppliedLrError`, the `CriticStateError` family, `OverlapGuardError`, `ProvenanceError`, `SampleIndexError`, `PolicyAgeError`, `UntrustedTBenchOutcome` | Consistency checks on training data or state |

### 1.5 Config mismatch → 6 CONFIG_MISMATCH

| # | Path | Reason |
|---|---|---|
| C1 | Driver init checks: execution mode/placement mismatch, `ProfileError`, algorithm declaration mismatch (`driver.py:545,552,559,562,579,581`) | The config of this island is invalid by itself; **but it is raised before `run()`, so it is not on the wire** |
| C2 | `CapabilityMismatch` (`capabilities.py:244`), `ManifestMismatch` (`runtime_manifest.py:228`), `BackendIdentityMismatch` (`backend_identity.py:144`), `UnsupportedPortsCombination`, `AlgorithmSpecError`/`PluginIdentityError`, `TopologyError`, `UnknownBackend`, `UnsupportedLayoutError` | All are startup argument/environment checks. No call site was found that raises after JOIN, so **today they are not on the wire**. The code is set to 6 for now, and kept for future in-run checks |
| C3 | JOIN refused: different compat_group or backend identity (`syncer/src/elastic_server.rs:408-420`); the island side raises `ElasticProtocolError("JOIN refused …")` (`elastic_client.py:767`) | Not compatible; but the island never became a member, so **it is not on the wire** (fact 1) |

### 1.6 Cloud reclaim → 3 PREEMPTED

| # | Path | Reason |
|---|---|---|
| P1 | AWS metadata polling on a droppable island → `handle_notice` → `leave()` (`yeto/cloud/droppable.py:178-184`, `preemption.py:131-135`) | Reclaim notice from the cloud provider |
| P2 | Nebius host SIGTERM on a droppable island → notice file → `leave()` (`droppable.py:185-212`) | Same as above |
| P3 | Modal SIGINT/SIGTERM on a droppable island → `ModalExitHandler.handle` → `leave()` (`preemption.py:159-184`, installed at `droppable.py:172-176`) | Treated as reclaim today; but it cannot be told apart from a user stop, see Q3 |

All three paths call `client.leave()` through the `leave()` closure at `entry.py:845-850`. They do not go through the driver, and today they send 0.
The verl adapter (`adapters/verl/trainer.py:265,329`) has no reclaim listener, so it has only the paths of 1.1–1.5.

## 2. Rust side (syncer)

| # | Path | Code | Reason |
|---|---|---|---|
| R1 | Each tick `expire_leases` (`syncer/src/elastic_server.rs:663`) → `leave(i, LEAVE_REASON_LEASE_EXPIRED)` (`syncer/src/elastic.rs:1071-1079`) | 1 | The only code decided by the syncer |
| R2 | A `Leave` frame from an island → `coord.leave(island_id, reason)` (`elastic.rs:1099-1101`) → `purge_departed` (`elastic_server.rs:450`) | Record the code sent by the island as it is | The syncer does not change the code |
| R3 | Log code name: `elastic.rs:742` today knows only 1, and logs all others as `"requested"` | n/a | At implementation time, change to 7 code names; whether an unknown value is logged as `"unknown(<n>)"` or as 0, see Q5 |
| R4 | A connection-level error (fencing `elastic.rs:1008`, HMAC, frame format) returns `MSG_ERROR` (`elastic_server.rs:539,573`) and does not remove the member | none | The member stays; the heartbeat or the lease decides whether it stays |
| R5 | The syncer exits on its own fatal error (`fatal_rx`, `elastic_server.rs:660`) | none | The syncer exits and sends no LEAVE to the islands; the island side goes through N2 or I5 |

## 3. Questions for the user to decide

- **Q1 How to recognize wrapped node loss.** Node loss is wrapped into a `DriverError` string (fact 2). Recommendation: at implementation time, add a subclass `IslandRecoveryRequired(DriverError)`, change the 6 places to raise it, and map by class. Alternative: match the string `"RECOVERY_REQUIRED"` (`_classify_failure` already does this, `driver.py:755`). The new subclass changes 6 lines of `driver.py`. This is an object shared with the "driver fatal error path", so it must be aligned with P0-1.7b.1 on the coordination board.
- **Q2 Default code for an unmapped exception.** design.md D5 left this empty. Recommendation: default to 4 INFRA_FAILED. All check exceptions that the driver raises itself can be listed explicitly under 5. Most of the remaining unknown exceptions are OOM, Ray and network. Alternative: default to 5.
- **Q3 Whether SIGTERM goes to STOPPED or PREEMPTED.** The same signal comes from `yeto down`/`stop-run` and also from Modal/Nebius reclaim. Today a non-droppable island that receives SIGTERM sends no LEAVE (fact 3), and a droppable island on Modal always sends it as reclaim. Options:
  - (a) Keep the current behavior: SIGTERM on a non-droppable island sends no LEAVE (the syncer records 1); SIGTERM on a droppable island sends 3. This is the smallest change.
  - (b) `yeto down`/`stop-run` first writes a stop marker (a file or an environment variable) and then sends the signal. The island sends 2 by the marker, otherwise 3. This changes the CLI and goes beyond the original scope of this change.
  Recommendation: (a); leave (b) for later.
- **Q4 Whether storage/resume errors go to 4 or 5.** `CutError`, `JournalError`, `StoreIntegrityError`, `LedgerError`. Some are disk or object store failures (like 4), and some are state inconsistencies (like 5, for example `EpochConflict` and `JournalCorrupt`). Recommendation: assign all to 4 (operations check the storage first), and tell them apart by the class name in the event log.
- **Q5 How to record an unknown code in the syncer.** The proposal says "record as 0". Recommendation: the new syncer records `"unknown(<n>)"`, raises no error, and removes the island as before. The old syncer still records 0. This affects only the log text.
- **Q6 Whether it is acceptable that C1/C2/C3 are not on the wire.** All of them fail before JOIN and produce no LEAVE. Should the island record `leave_reason=6` in the event log, so that the launcher can count them? Recommendation: record it in the event log, and do not send it on the wire.

## 4. Convention for `close()` of the new Sync of P0-1.7b.1 (convention only, not implemented)

1. The driver is the only place that decides the code. The bridge (including the new Sync of P0-1.7b.1) only sends it (design.md D1).
2. Interface: `close(reason: int = LeaveReason.FINISHED)`. No separate `abort()`. In the `except BaseException` of `run()`, the driver looks up the code by the table in section 1 and stores it in a local variable. In `finally`, it calls `self.sync.close(reason)`. On the normal end path, `finish()` has already sent 0, and `close()` sees `left=True` and does not send again.
3. The code definition lives in one place only: Python `LeaveReason` (`IntEnum`) in `yeto/rl/elastic_client.py`, and the constants in Rust `syncer/src/elastic.rs`. A parity test locks the values. The new Sync of P0-1.7b.1 imports from `elastic_client` and does not define its own values.
4. When the new Sync is not an elastic bridge and has no LEAVE semantics, it accepts and ignores `reason` (the same as `StrictAvgSync` and `DecoupledSync`, design.md D3).
5. When sending LEAVE fails: emit the event `elastic_leave_failed` (with the code, the exception class name and the text), and do not re-raise (design.md D2).
6. Code 1 LEASE_EXPIRED MUST NOT be sent by any Sync. Passing 1 is a programming error: send it as 4 and record an event.
7. One process sends LEAVE only once, decided by the `left` flag of the bridge. If the reclaim listener (section 1.6) has already sent 3, a later `close()` of the driver does not overwrite it.
8. The specific exception class name and message go to the island event log (event `elastic_leave`, fields `reason`, `reason_name`, `error_type`, `error`). They do not go on the wire.
