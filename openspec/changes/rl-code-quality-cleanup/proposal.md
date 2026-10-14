# Proposal

## Why

On 2026-10-12 the PM set two hard requirements. Run logs must not live in the git repository. Duplicated code must be consolidated. The PM's duplication list covers seven areas: engine, island-side sync, syncer, resume and checkpoint, algorithms, reward and harness, and config and shared utilities. The PM also set the starting point. First delete the legacy engine (about 5.4k lines, one pin, one bundle). Then merge the two bridges into one, and define the weight and the staleness limit once. Then merge the three or four resume mechanisms into one abstraction. This is the only delivery line of this week (`infra-drafts/SESSION23/P0-code-quality.md`, `P6-repo-hygiene.md`). Ascend RL runs on native verl first and does not compete for time.

## What Changes

The work follows the PM's order in five phases. Each phase has several independent PRs. The facts that the main agent checked by reading code on 2026-10-12 are in design.md. Where they differ from the PM's list numbers (for example, there are 46 forwarding stubs, not 18, and the `co_names` probe targets the legacy writer, not an engine switch), the code is the reference. The cleanup target does not change.

**Phase 0: repo hygiene and a narrower import surface** (zero behavior change)
- Remove run logs from the repo: `git rm --cached` the 87 tracked log-type files (about 2.6MB, mostly in `openspec/changes/*/run-logs`, `*/evidence/`, `tests/`). Add `run-logs/`, `*.stderr`, `*.stdout`, `*metrics*.json` to `.gitignore`. Remove the `!` allowlist lines that let logs through. **BREAKING**: tests or docs that depend on these committed logs now reference paths outside the repo, or are deleted.
- Delete 46 module-level forwarding stubs (18 in `yeto/rl/`, 28 in `yeto/rl/engine/miles_adapter/`). Change all imports repo-wide to the direct path. **BREAKING**: the old paths can no longer be imported. `python -m yeto.rl.learner` is replaced by a real module.
- Make the pin constants and the fork lineage single-source (six pin files, three hand-copied lineages).
- Unify the lease default (syncer 30 seconds becomes 300 seconds).

**Phase 1: keep only one Miles engine** (the PM's starting point)
- Delete `adapters/miles/legacy/engine.py` (2289 lines) and its pin, image and setup path. `island_entry.py` keeps only the ports path.
- Prerequisite: move the legacy-only capabilities to ports: `generate_rollout` (about 1120 lines, the largest item), `set_current_published_policy_identity`, compaction record parsing, and DSV4 expert-full runtime loading. Clear the rejection items at `engine/selection.py:54-67` one by one (SAO, dense-full, DSV4).
- Delete the `__code__.co_names` runtime probe at `island_entry.py:656`.
- Delete the second copy of the decoupled boundary loop in legacy (`legacy/engine.py:2174-2216`).

**Phase 2: one bridge, one definition of weight and staleness**
- Merge `rl/bridge.py` (StrictRlBridge/ElasticRlBridge) and `engine/bridges.py` (StrictAvgSync/ElasticAvgSync) into one. Write the delta math once. Add a parity test between StrictRlBridge and StrictAvgSync before the merge.
- Define the merge weight once: `elastic.rs:483` calls `merge.rs`. The Python copy at `island_ledger.py:160` is either locked by a parity test or deleted.
- Add parity assertions between the Rust staleness limit `MAX_SAMPLE_OUTER_LAG` and the Python `max_outer_lag`.
- Keep one critic sync path: delete `TwoRoleStrictAvg` (no production caller). Delete the unused role-lane design in `local_learner.py`.
- Write the weight, staleness and finish rules of the three sync modes (strict / decoupled / elastic) as one table in a code comment and in the spec, instead of leaving them scattered.

**Phase 3: one resume abstraction**
- Merge these into one definition of a "consistent cut" and one resume controller: `--rl-elastic` (controller store), inter-island elastic (syncer local checkpoint), `engine/resume.ResumeController`, and the legacy `--spot` completed-groups volume. The legacy one is deleted with phase 1.
- Merge the four checkpoint stores (CheckpointStore, BucketMountStore, ModalVolumeStore, CriticCheckpointStore) into one interface with backends. Merge the three save paths (InFlightSaver, handle_notice on reclaim, round_cut) into one.

**Phase 4: algorithm, reward and config consolidation**
- Algorithms: merge `rl/algorithms/` (standalone PPOTrainer with its own GAE) into `rl/algos/`. Merge our three GRPO group normalization copies into one (the Miles and verl copies are upstream code and stay as they are, with a parity test added). Merge `loss_reference.py` and `fp32_reference.py` into one. Merge the three policy-age modules into one with backend adapters. Merge the two staleness checks in driver and governor into one. Make the capability declarations `MILES_DECLARED` / `G1_DECLARED` / verl `CONFIG_GOLDEN` consistent.
- Reward and harness: move the reward definition point from six places to one, `rewards/registry`. Merge the two trust paths (TB2 and RLF) and the two finish functions into one. Merge the four Codex agent entries into one main entry with thin wrappers, and merge generate into one file. Merge the separate pin in `ssh_harness.py` into pins. Merge the three contract/identity modules (`rl/contracts.py`, `rlf/contracts.py`, `backend_identity.py`).
- Config and shared utilities: replace the three-layer pass-through of RL flags with one schema (`run_config.py` is the only source; cli/launcher/island_entry are generated from it). Make the algorithm hash and the island contract hash nested. Merge the three W&B modules, the two exporters, the two finalisation modules and the two env scrubbers into one each.
- Merge the three verl run paths (ports_impl, fully_async_runner, fully_async_round/ports/translate) into one. Wire the mismatch criterion only once.

Each single item in phase 4 is large and risky. This week's schedule ends when phase 2 is complete. Phases 3 and 4 are created as work items and roll week by week. This change does not commit to a completion date for them.

## Capabilities

### New Capabilities
- `rl-codebase-consistency`: codebase consistency constraints. Logs are not committed. Each module has one import path. There is only one Miles engine. The sync bridge and the weight/staleness definitions exist once. Resume has one abstraction. Algorithms, rewards and config each have one source. Every step has zero training behavior change.

### Modified Capabilities
(None. This change does not alter training numerics or scheduling semantics. The requirements of the existing main specs do not change. When phase 1 deletes legacy, the "reject, use ports" entries of `rl-engine-selection` are cleared. That is an implementation detail converging, not a requirement change.)

## Impact

- Code: almost all subdirectories of `yeto/rl/`, `syncer/src/`, `yeto/cli.py`, `yeto/launcher.py`, `.gitignore`.
- Deleted volume: about 614 lines of forwarding stubs. About 5.4k lines of legacy engine plus pin and bundle (about 1.5k lines of capability must move first). About 1000 lines from the bridge merge. About 125 lines of dead code.
- Tests: `test_import_boundaries.py`, `test_rl_engine_pins.py`, `test_rl_inter_island_rust_golden.py`, `test_rl_engine_selection.py`, `test_rl_sao_streaming_runtime.py`, `test_rl_deepseek_v4_expert_full_runtime.py`, `test_rl_m1_dense_full_direct_launch.py`, `test_rl_reward_pipeline_equivalence.py`, `test_rl_fp32_reference.py`, `test_rl_algorithm_capabilities.py`, `test_rl_grpo_knobs.py`, and the new bridge parity and weight parity tests.
- External: the old import paths and `--rl-engine legacy` style options disappear. The image has one legacy pin fewer.
- Verification: the full local test set for phases 0-2. After legacy is deleted in phase 1, SAO / dense-full / DSV4 each need one minimum-card run on GPU as a regression (per memory gpu-min-card-policy and gpu-review-before-launch). The CI of main is red already. The local safe test set is the reference.
