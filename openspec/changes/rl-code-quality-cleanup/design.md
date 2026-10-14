# Design

## Context

The goal is the PM duplicate list of 2026-10-12. The facts below come from code reading by the main agent and sub-agents on the same day (worktree `/home/michael/work/s22-agentic-sft`, HEAD 8f952cca). Where the list and the code differ, the code is the reference. The cleanup goal does not change.

**Repository hygiene**
- 236 tracked files are log-type files, about 3.8MB. Most are in `openspec/changes/add-nebius-verda-modal-clouds/run-logs` and `openspec/changes/rl-engine-ports/evidence/`. The jsonl files under `tests/` and `data/` are test inputs and stay. The largest single file is 1.68MB. `.gitignore` already has `*.log`, `**/*.jsonl` and similar rules. But it has `!` allowlist lines that let some events/tape/log files through, and it has no `run-logs/` rule. Ignore rules do not affect tracked files, so `git rm --cached` is needed.

**Engine**
- There are 46 forwarding stubs, about 614 lines. `yeto/rl/` has 18 stubs of 17 lines each (`sys.modules[__name__] = target`; the `learner` stub also uses `runpy` to support `python -m`, and about 36 files refer to it). `yeto/rl/engine/miles_adapter/` has 28 stubs of 10–11 lines each (about 13 referring files).
- The legacy engine `adapters/miles/legacy/engine.py` has 2289 lines. The SAO streaming code itself is not in legacy (`models/sao_streaming.py` and others, about 2633 lines). It reaches legacy only through two fully qualified names at `sao_streaming_runtime.py:30-31`: `set_current_published_policy_identity` (:225) and `generate_rollout` (:1166-1349, 184 lines in the function body, and ports has no equivalent with that name). Lines :1350-2286 are policy sync classes such as `MilesPolicySync`. They do not belong to `generate_rollout`. Compaction record parsing is at `legacy/engine.py:324-358`. The DSV4 expert full-parameter runtime loading is at :1644. Dense-full is already mostly in `island_entry.py`. `engine/selection.py:54-67` sends the SAO preset and `--sao*` flags (detected at `selection_argv.py:35`), dense-full and DSV4 away from ports.
- The `co_names` probe at `island_entry.py:656` checks whether legacy `_append_rl_event` is an old writer. It does not tell the engines apart. If that code calls `append_record` directly, the probe can be deleted.
- The three verl run paths and the copied mismatch criterion: not checked line by line. Check them again before phase 4 starts.

**Island-side sync**
- `rl/bridge.py:881 ElasticRlBridge` has 0 callers. `:981 make_island_bridge` is used by 2 tests only. `critic_state.py:198 TwoRoleStrictAvg` is used by 1 test only. `DualStrictAvgSync` is in use. In `local_learner.py`, only the PPO role-lane is deleted. The other role-lane code is in use.
- The bridge merge has no parity test between `StrictRlBridge` and `StrictAvgSync`.

**syncer**
- `elastic.rs:483 merge_weight` copies `merge.rs:27 learner_weight` (the step count is u64 in one and u32 in the other). The Python copy is at `engine/island_ledger.py:156-160`.
- The staleness limit is 2 on both sides: Python `island_ledger.py:105,185`, Rust `elastic.rs:29,521`. Line 18 of the golden fixture already has a lag=3 case that is rejected.
- `iso_worker.rs` is not a protocol server. It is a process pool that starts Python subprocesses (frame magic `YETOISO1`). It is a separate thing from `protocol.py`/`protocol.rs`. The "two wire protocol implementations" item is handled as the protocol pair only.

**Algorithm and declarations**
- GRPO group normalization has four copies on our side: `algos/reward_pipeline.py:400`, `algos/seq_adv.py:332`, `fp32_reference.py:112`, `rlf/contracts.py:67` (the last one subtracts the mean only and does not divide by std). The Miles and verl copies are upstream code.
- `MILES_DECLARED` is at `adapters/miles/entry.py:69-155`. `G1_DECLARED` is at `algos/grpo_knobs.py:473-490`, and its comment says it mirrors `MILES_DECLARED`. verl has no `*_DECLARED`. The nearest is `adapters/verl/binding.py:31 CONFIG_GOLDEN`. The claim "G1 contradicts itself" is unverified. A diff is needed.
- The fork lineage appears in three places: `algos/loss_variants.py:72`, `adapters/miles/entry.py:205` (one more commit, 0af62f4d) and `algos/critic_fork.py:47` (the critic is a separate chain). `pins.py` currently holds only the head commit.

**Config**
- `cli.py` has 106 `--rl-` literals (not 74). Most of the 249 in the launcher are hint strings that contain `--rl-`. `run_config.py` has 1149 lines.
- Lease: `syncer/src/main.rs:160` defaults to 30. The `cli.py:526` help text says 300. `launcher.py:679` always passes 300.

**Reward, harness, resume**: six reward definitions, two trust paths, four Codex entries, a separate pin in `ssh_harness.py`, four checkpoint stores and three save paths. These exist according to the PM list. The line-by-line check is left until just before the matching phase starts.

## Goals / Non-Goals

Goals: follow the PM order. Delete the legacy engine. Keep one bridge, one weight definition and one staleness definition. Build one resume abstraction. Roll out the other consolidations step by step. Every PR has zero change in training behavior and can be rolled back alone.

Non-Goals: changing training numerics, changing scheduling semantics, changing the public CLI semantics (except removing legacy options and old import paths), Ascend, boat.dev.

## Decisions

1. **Phase 0 goes first, because phase 1 depends on it.** After the stubs are deleted, the set of references is smaller, and the legacy call sites are easy to see. The log removal must finish before PR #238 merges (P6 order).
2. **How to remove the logs.** Run `git rm --cached` in one commit. In `.gitignore`, delete the `!` lines that let logs through. Add `run-logs/`, `*.stderr`, `*.stdout` and `*metrics*.json`. Move the evidence logs to `/home/michael/work/infra-drafts/evidence/<change>/`. Keep one `EVIDENCE.md` in each original directory to point to the new place. Add a test `test_no_tracked_logs.py` that scans `git ls-files`.
3. **The real work in deleting legacy is moving `generate_rollout`.** It is not "rewrite SAO". Move `generate_rollout` (`legacy/engine.py:1166-1349`) and its 25 private helper functions as one module to `adapters/miles/models/rollout_legacy_port.py` (move first, split later). Point `sao_streaming_runtime.py:30-31` to the new place. Then replace the parts one by one with the equivalent features that ports already has. The other three items (policy identity, compaction parsing, DSV4 runtime loading) each get one small PR. When all moves are done, clean `selection.py:54-67`. At the end, delete the `legacy/` directory and its pin.
   - Before the deletion, record one set of reference outputs (one minimum-card run each for SAO, dense-full and DSV4). This set is the regression baseline for phase 1. Store the baseline data in the evidence directory outside the repository.
4. **Bridge merge order.** First write the parity tests `StrictRlBridge` against `StrictAvgSync`, and `ElasticRlBridge` against `ElasticAvgSync` (`ElasticRlBridge` has no callers. If it is deleted, only the strict parity remains). Keep `engine/bridges.py` as the surviving side, because it is on the production path. `rl/bridge.py` becomes a thin wrapper or is deleted. Keep the decoupled boundary loop at `bridges.py:1097-1125`. The legacy copy goes away with phase 1.
5. **Weight and staleness.** `elastic.rs` calls `merge::learner_weight` (the step count is converted with a relaxed u64 rule; the original plan of `u32::try_from` is dropped; the zero-value branch stays). The Python copy at `island_ledger.py:160` stays, but `test_rl_inter_island_rust_golden.py` locks it by parity (reason: the Python side is the island-local ledger and must not depend on a Rust build). The staleness limit parity test reads the `elastic.rs` constants with a regex.
6. **One table for the three sync modes.** Put the same table (mode / weight / staleness / closing) in the module docstring of `engine/bridges.py` and in this spec. A test asserts that the docstring contains the table header, to stop drift.
7. **Resume abstraction.** Define `ConsistentCut` (outer step number, inner step number of each island, set of completed groups, checkpoint id) and one `ResumeController`. The `--rl-elastic` controller store and the syncer local checkpoint both implement one `CheckpointStore` interface. `InFlightSaver`, `handle_notice` and `round_cut` all call the same `save(cut)`. Write the interface and the parity tests first, then move the paths one by one.
8. **Algorithm, reward and config consolidation.** For each item, run the parity test first and merge after (Decision 9). The config schema has one source, `run_config.py`. The `add_argument` calls of cli, launcher and island_entry are generated from the schema. Generate first, then diff against the current argument list, and switch only when the difference is zero.
9. **Execution rule for zero behavior change.** Every numeric path needs a parity test before it is merged. If the parity fails, stop. Write the difference in "Open Questions" of this file, and the user decides which copy is the reference. Do not change values silently.
10. **Where the PM list and the code differ.** The code numbers are the reference (46 stubs, 106 `--rl-`). The `co_names` probe is deleted as planned. The `iso_worker` dual implementation is not handled as a protocol. The two upstream GRPO copies stay unchanged. All of this is recorded in Context above. We do not tell the PM again that "the list was wrong". We only do the work as it actually is.

11. **Unified full-parameter sync (1.7b, P0-D1…D9, design draft `infra-drafts/SESSION23/P0-1.7b.1-design.md`):**
    - ports adds `FullParameterSync` in `engine/full_parameter_sync.py`. It implements `SyncSession`. It does not import Miles. It depends only on the `RoleStream` protocol in `engine/ports.py`.
    - The single-role streaming sync is based on `MilesFullParameterRoleStream` (formerly `sao_streaming.py:282-960`). Move it to `adapters/miles/full_parameter/role_stream.py`, remove the SAO-specific names, and put `algorithm` into the config.
    - The multi-role pairing rules (same sync config, paired staging, paired apply, when one role fails all roles close together with the same reason code, time-limited PULL deviation) work for N roles. With 1 role they reduce to the simple case naturally, and no special branch is written.
    - The dense strict sweep is a restricted config: 1 role, H=1, pipeline depth 1, `sweep=True`, and the syncer gets `--policy-sweep-fragments P`. No `DenseSweepSync` is added. `DenseSweepWire` is deleted after the wire-level parity test passes.
    - The weight state uses the `fragments` layout of `TrainableState` and carries only fragment handles. The policy_tensor_hash takes the identity of the fragment slices. The whole model is never assembled in one process.
    - The policy token is unified as `yeto:<version>:<weight hash>`. The dense publish goes through the publish check of `driver.publish`.
    - The M1 limits (H=1, start from version 0, fixed `num_rollout`) are not part of the design. The syncer's authoritative fragments decide the start point. To collect the quorum of islands, use the existing K-of-M and grace τ of the syncer.
    - The syncer starts processes by the roles that the algorithm declares. The role list is derived from `execution.needs_critic` (no new field). The launcher generates one syncer command per role.
    - Closing uses the 7 reason codes of P5: `close(reason=LeaveReason.FINISHED)`. Full-parameter runs use the legacy-mode syncer, which has no LEAVE message, and only the event `rl_sync_closed` is written.
    - The full-parameter session contract hash switches to `backend_identity("ports")`. Old test runs cannot be resumed (P0-D5).
    - In this phase, full-parameter does not support `--rl-island-scheduling elastic` (the syncer elastic mode does not handle fragment frames). Selection rejects it explicitly.
    - Known risk: the Miles actor RPC that full-parameter needs is not in the ports pin `64b591a4`. Dense runs use the Miles source that `YETO_FULL_PARAMETER_MILES_SOURCE_ROOT` points to. Before running on GPU, get the user's confirmation of the porting plan.

## Risks / Trade-offs

- Phase 1 needs three minimum-card GPU runs for the baseline and the regression. The budget comes from the GPU ledger. Before each run on GPU, write the review conclusion as the memory gpu-review-before-launch requires.
- Moving `generate_rollout` first and splitting later leaves a large module (the function body and its private helpers) for a short time. But it removes the legacy directory and its pin, which matches the PM starting point. The split continues in phase 4.
- The bridge merge removes about 1000 lines. If the parity test shows that the two delta computations differ (for example the merge_alpha handling in decoupled), stop.
- Generating the config schema moves the definitions of 106 flags, so the diff is large. The comparison of the generated argument list is the safeguard.
- In the local test set, exclude `mismatch_tape` and `upstream_parse_args`, which start a local Ray.

## Migration Plan

Phase 0 (first half of this week) → phase 1 (main part of this week, with three GPU regression runs) → phase 2 (end of this week or start of next week) → phases 3 and 4 (rolling). For each PR: run the local safe test set, and write the pass/fail counts in the English PR description. The log removal PR must merge before #238 merges.

## Open Questions

- How much of `generate_rollout` can be replaced with existing ports features. After the move, mark it section by section.
- Whether `G1_DECLARED` really contradicts `MILES_DECLARED`. Run a diff before phase 4 starts.
- Where the mismatch criterion is copied in the three verl run paths. Check before phase 4 starts.
- `rlf/contracts.group_advantages` subtracts the mean only and does not divide by std. Is this intended (used by the gate) or an omission? Ask the user before the merge.
