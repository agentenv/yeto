# Proposal

## Why

On 2026-10-12 the PM gave two hard requirements: run logs must not enter the git repository, and code duplication must be reduced. The PM's duplication list covers seven areas: engine, island-side sync, syncer, resume and checkpoint, algorithms, rewards and harness, and config and shared utilities. The PM also set the starting point. First delete the legacy engine (about 5.4k lines, one pin, one bundle). Next merge the two bridges into one, and define weights and the staleness limit only once. Then merge the three or four resume mechanisms into one abstraction. This is the only delivery line of this week (`infra-drafts/SESSION23/P0-code-quality.md`, `P6-repo-hygiene.md`). The Ascend RL work runs on native verl first and does not take time from this line.

## What Changes

The work has five phases in the order the PM set. Each phase has several independent PRs. The facts that the main agent verified by reading code on 2026-10-12 are in design.md. Where they differ from the PM list, the code is the reference. For example, there are 46 forwarding stubs, not 18. The `co_names` probe targets the legacy writer, not an engine check. These differences do not change the cleanup goal.

**Phase 0: repo hygiene and a smaller reference surface** (zero behavior change)
- Remove run logs from git: run `git rm --cached` on the 87 tracked log-type files (about 2.6MB, mostly in `openspec/changes/*/run-logs`, `*/evidence/` and `tests/`). Add `run-logs/`, `*.stderr`, `*.stdout` and `*metrics*.json` to `.gitignore`. Remove the `!` allowlist entries that let logs in. **BREAKING**: tests or docs that depend on these committed logs must point to a path outside the repo, or be deleted.
- Delete the 46 module-level forwarding stubs (18 in `yeto/rl/`, 28 in `yeto/rl/engine/miles_adapter/`). Change the whole repo to import directly. **BREAKING**: the old paths can no longer be imported. `python -m yeto.rl.learner` becomes a formal module.
- Give the pin constants and the fork lineage one source (six pin files, three hand-copied lineages).
- Make the lease default one value (syncer: 30 seconds changes to 300 seconds).

**Phase 1: keep only one Miles engine** (the PM's starting point)
- Delete `adapters/miles/legacy/engine.py` (2289 lines) with its pin, image and setup path. `island_entry.py` keeps only the ports path.
- Precondition: move the legacy-only abilities to ports. These are `generate_rollout` (about 1120 lines, the largest item), `set_current_published_policy_identity`, compaction record parsing, and the DSV4 expert full-parameter runtime load. Clear the reject entries at `engine/selection.py:54-67` one by one (SAO, dense-full, DSV4).
- Delete the `__code__.co_names` runtime probe at `island_entry.py:656`.
- Delete the second decoupled boundary loop in legacy (`legacy/engine.py:2174-2216`).

**Phase 2: one bridge, one definition of weights and staleness**
- Merge `rl/bridge.py` (StrictRlBridge/ElasticRlBridge) and `engine/bridges.py` (StrictAvgSync/ElasticAvgSync) into one. Write the delta math once. First add a cross-check test between StrictRlBridge and StrictAvgSync, then merge.
- Define the merge weight only once: `elastic.rs:483` calls `merge.rs`. Replace the copy at Python `island_ledger.py:160` with a cross-check test lock, or delete it.
- Add a cross-check assertion between the Rust staleness limit `MAX_SAMPLE_OUTER_LAG` and the Python `max_outer_lag`.
- Keep only one critic sync path: delete `TwoRoleStrictAvg` (no production caller). Delete the unused role-lane design in `local_learner.py`.
- Write the weight, staleness and closing rules of the three sync modes (strict / decoupled / elastic) as one table in code comments and in the spec. Do not keep them scattered.

**Phase 3: one resume abstraction**
- Merge `--rl-elastic` (controller store), cross-island elastic (syncer local checkpoint), `engine/resume.ResumeController` and the legacy `--spot` completed-groups volume into one definition of a "consistent cut point" and one resume controller. The legacy one is deleted with Phase 1.
- Merge the four checkpoint stores (CheckpointStore, BucketMountStore, ModalVolumeStore, CriticCheckpointStore) into one interface with backends. Merge the three save paths (InFlightSaver, handle_notice on reclaim, round_cut) into one.

**Phase 4: converge algorithms, rewards and config**
- Algorithms: merge `rl/algorithms/` (the standalone PPOTrainer with its own GAE) into `rl/algos/`. Merge our three GRPO group-normalization copies into one (the copies in Miles and verl are upstream code. Do not change them. Add a cross-check). Merge `loss_reference.py` and `fp32_reference.py` into one. Merge the three policy-age modules into one with backend adapters. Merge the two staleness checks in the driver and the governor into one. Make the capability declarations `MILES_DECLARED` / `G1_DECLARED` / verl `CONFIG_GOLDEN` consistent.
- Rewards and harness: move the reward definition point from six places to one place, `rewards/registry`. Merge the two trust paths (TB2 and RLF) and the two finish functions into one path. Merge the four Codex agent entries into one main entry with thin wrappers, and merge generate into one file. Move the separate pin of `ssh_harness.py` into pins. Merge the three contract/identity modules (`rl/contracts.py`, `rlf/contracts.py`, `backend_identity.py`).
- Config and shared utilities: replace the three-layer pass-through of RL flags with one schema (`yeto/rl/engine/run_flags.py` is the only source. cli, launcher and island_entry are generated from it. Decided on 2026-10-13: do not put it in `run_config.py`). Make the algorithm hash and the island contract hash nested. Merge the three W&B modules, two exporters, two finalisation modules and two env scrubbers into one each.
- Merge the three verl run paths (ports_impl, fully_async_runner, fully_async_round/ports/translate) into one. Wire the mismatch criterion only once.

Each single item in Phase 4 is large and risky. This week the plan goes only up to the end of Phase 2. Phases 3 and 4 are set up as work items but roll week by week. This change does not promise a completion date for them.

## Capabilities

### New Capabilities
- `rl-codebase-consistency`: Codebase consistency constraints. Logs do not enter git. A module has only one import path. There is only one Miles engine. The sync bridge and the weight/staleness definitions exist once. Resume has one abstraction. Algorithms, rewards and config each have one source. Every step causes zero change in training behavior.

### Modified Capabilities
(None. This change does not alter training numerics or scheduling semantics. The requirements in the existing main specs do not change. When Phase 1 deletes legacy, the "reject and go to ports" entries of `rl-engine-selection` are cleared. That is a convergence of implementation detail, not a requirement change.)

## Impact

- Code: almost all subdirectories of `yeto/rl/`, `syncer/src/`, `yeto/cli.py`, `yeto/launcher.py`, `.gitignore`.
- Deleted amount: about 614 lines of forwarding stubs. About 5.4k lines of the legacy engine plus its pin and bundle (about 1.5k lines of abilities must be moved first). About 1000 lines from the bridge merge. About 125 lines of dead code.
- Tests: `test_import_boundaries.py`, `test_rl_engine_pins.py`, `test_rl_inter_island_rust_golden.py`, `test_rl_engine_selection.py`, `test_rl_sao_streaming_runtime.py`, `test_rl_deepseek_v4_expert_full_runtime.py`, `test_rl_m1_dense_full_direct_launch.py`, `test_rl_reward_pipeline_equivalence.py`, `test_rl_fp32_reference.py`, `test_rl_algorithm_capabilities.py`, `test_rl_grpo_knobs.py`, and the new bridge cross-check and weight cross-check tests.
- External: the old import paths and options such as `--rl-engine legacy` are gone. The image has one legacy pin fewer.
- Verification: for Phases 0-2, the full local test set. For Phase 1, after legacy is deleted, SAO / dense-full / DSV4 each need one GPU regression run with the minimum number of cards (per memory gpu-min-card-policy and gpu-review-before-launch). The CI of main was already red. The local safe test set is the reference.
