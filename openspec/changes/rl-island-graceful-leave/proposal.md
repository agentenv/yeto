# Proposal

## Why

SESSION23 Q1 decided: the failure of a single island must not affect the whole fleet (`infra-drafts/SESSION23/P5-island-failure.md`). P5 left three unverified items. This change first verifies them by reading code, then fills only the gaps. It does not redo existing features.

Code-reading result on 2026-10-13 (branch HEAD `f2db57f8`):

1. **The island sends LEAVE before it exits: this exists, but covers only "exit by a raised error".** P5 states "no active LEAVE found". That is not accurate.
   - On node loss, `controller.check_nodes()` enters RECOVERY_REQUIRED (`yeto/rl/engine/controller.py:664-686`).
   - The driver raises `DriverError("island is RECOVERY_REQUIRED …")` (`yeto/rl/engine/driver.py:740,770`).
   - The `finally` of the driver calls `self.sync.close()` (`driver.py:1767-1770`).
   - When the island has not left, the elastic bridge `ElasticAvgSync.close()` calls `client.leave()` (`yeto/rl/engine/bridges.py:655-664`).
   - `leave()` sends `Leave(epoch, island_id, LEAVE_REASON_REQUESTED)` (`yeto/rl/elastic_client.py:853-856`).
   - After the syncer receives Leave, it calls `purge_departed` (`syncer/src/elastic_server.rs:450`).
   - Gap a: the only reason codes are `REQUESTED=0` and `LEAVE_REASON_LEASE_EXPIRED=1` (`elastic_client.py:62-63`, `syncer/src/elastic.rs:275-276`). The syncer log cannot tell "the island left because of a failure" from "the island left normally".
   - Gap b: `close()` swallows every exception (`bridges.py:657-661`). A LEAVE send failure is not recorded.
   - Gap c: when the process is hard killed (SIGKILL, whole machine down, the node that hosts the driver dies), the `finally` does not run. The only way is still to wait for the lease to expire.
   - Gap d: no test was found that asserts "the RECOVERY_REQUIRED path sends LEAVE". The search covered only files under `tests/` that contain both RECOVERY_REQUIRED and leave. This is not an exhaustive search and is unverified.
2. **A CLI to add an island during a run: does not exist.** The `yeto` subcommands are only launch, shape, merge, rl, sample-diffusion, status, logs, down, stop-run, and the internal `_worker` and `_head` (`yeto/cli.py:1405-1595`). No add-island style command was found in the launcher or the cli. Adding an island still needs a manual start of a separate island process with the same `--rl-syncer-epoch` (`yeto/cli.py:531`).
3. **Current state of single-island failure isolation:** the elastic path is implemented (removal on lease expiry, rounds at a threshold, automatic rejoin, mid-run island add). For the evidence, see P5 section 7.1. This reading did not re-check it. The strict path (fixed_roster) still needs all members present, and a strict failure exits the whole session (P5 section 7.2 item 1, not re-checked in this reading).
4. **The lease default is now 300 s.** `syncer/src/main.rs:159-162` defaults to `300.0`, which matches the launcher `--rl-island-lease-s`. `rl-code-quality-cleanup` tasks 0.8 just unified it to 300. By the user's decision, this change changes it again to 150 s (see What Changes). It does not change the `rl-code-quality-cleanup` directory.

## What Changes

- Expand the reason codes to 7. Python (`yeto/rl/elastic_client.py:62-63`) and Rust (`syncer/src/elastic.rs:275-276`) use the same values, locked by a cross-check test. This replaces the original proposal, which only added `ISLAND_FAILED=2`.

  | Value | Name | Meaning |
  |---|---|---|
  | 0 | FINISHED | Training ran to completion as planned (the old REQUESTED, renamed, value unchanged) |
  | 1 | LEASE_EXPIRED | Decided by the syncer itself: heartbeat timeout, the island vanished silently (hard kill, whole machine down, network cut) |
  | 2 | STOPPED | Stopped by the user: `yeto stop-run`, Ctrl-C/KeyboardInterrupt/SystemExit |
  | 3 | PREEMPTED | The cloud reclaimed a spot instance: the reclaim notification paths in `yeto/cloud/preemption.py` and `droppable.py` |
  | 4 | INFRA_FAILED | Hardware or runtime environment: RecoveryRequired node loss, GPU fault, out of GPU memory, process crash |
  | 5 | TRAINING_FAILED | A training check did not pass: the DriverError family, such as PublicationError, PolicyIdentityError, RoundFailedError, non-finite gradient |
  | 6 | CONFIG_MISMATCH | Found during the run to be incompatible with the other islands: CapabilityMismatch, BackendIdentityMismatch, ManifestMismatch |

- A code expresses only the broad category. The wire protocol uses 1 byte. The exact exception class name and error message go to the island's event log, not the wire protocol.
- An old syncer records an unknown code as 0 (`elastic.rs:742`) and does not report an error. The syncer log records the code name.
- First implementation step: list the mapping from every exception class under `yeto/rl/engine` to a code (for example, where CutError and JournalError belong). After the mapping table is listed, show it to the user once, then continue the implementation.
- On each exit path, the driver picks the code from the mapping table, sends LEAVE first, then closes the bridge.
- A LEAVE send failure in `ElasticAvgSync.close()` now emits the event `elastic_leave_failed` and is no longer swallowed silently. It is still not re-raised, so it does not hide the original error.
- Change the default lease of elastic-mode islands from 300 s to 150 s. Three places MUST change together: `--island-lease-s` at `syncer/src/main.rs:159-162`, and `--rl-island-lease-s` in the launcher and in `yeto`. Add or change a test that asserts the three are equal (follow the way `rl-code-quality-cleanup` tasks 0.8 set 300). The heartbeat interval is `lease_s/3` (`elastic_client.py:577`). At 150 s it is 50 s, which tolerates 2 consecutive lost heartbeats.
- Add CPU tests:
  - Unit test: the LEAVE reason code for each category of reason.
  - Cross-check of the reason code values between Python and Rust.
  - End-to-end: a real Rust syncer and two simulated islands. One island leaves for each category of reason, and the other island keeps producing rounds.
  - Removal on lease timeout: the test uses a shortened lease (it does not wait the real 150 s).
- Not done: the launcher sending LEAVE on behalf of the island (deferred, see D5 in design.md); strict path failure isolation; a CLI to add an island during a run; run on GPU.

## Capabilities

### New Capabilities
- `rl-island-failure-isolation`: The behavior and reason codes of an active LEAVE when an island fails.

### Modified Capabilities
- None.

## Impact

- `yeto/rl/elastic_client.py`, `yeto/rl/engine/bridges.py`, `yeto/rl/engine/driver.py`, and the lease defaults in the launcher and `yeto/cli.py`.
- `syncer/src/elastic.rs` (new constants and code names in logs), `syncer/src/main.rs` (lease default).
- 1-2 new test files. The wire protocol does not change: the reason code field is already a `u8`, and an old syncer records an unknown code as 0 (`elastic.rs:742`) and reports no error.
- Lease default change: after an elastic island vanishes silently, the wait before removal drops from 300 s to 150 s. Unverified: whether a healthy island with slow heartbeats is removed by mistake at 150 s. The heartbeat is sent every 50 s, and 2 consecutive losses are tolerated.

## Decided (user, 2026-10-13)

1. Reason codes: 7, see What Changes. The first implementation step is to list the exception class mapping and show it to the user.
2. Hard kill: do not build the launcher sending LEAVE on behalf of the island. Rely on the lease timeout. Change the default lease of elastic-mode islands from 300 s to 150 s, change the three places together, and add a test.
3. Add island: do not build a CLI to add an island during a run. Only write the manual steps (start a separate island process with the same `--rl-syncer-epoch`, `yeto/cli.py:531`).
4. Verification: do not run on GPU. Run only CPU tests.
