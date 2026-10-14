# Tasks

Each section is one or more PRs. Commit and PR text use English only. Before each PR merges, run the local safe test set (exclude `mismatch_tape` and `upstream_parse_args`) and write the pass/fail counts in the PR description. Run a parity test before merging any numeric path. If the parity fails, stop. Run on GPU only in 1.2, 1.7b.5 and 1.11 (budget: see COORDINATION P0-G). Use the minimum number of cards and write the review conclusion first.

## 0. Repository hygiene and a smaller reference surface

- [x] 0.1 List the 236 tracked files that match the log patterns with `git ls-files` (about 3.8MB; the jsonl files under `tests/` and `data/` are test inputs and stay). Move the evidence files to `/home/michael/work/infra-drafts/evidence/<change>/`. Leave an `EVIDENCE.md` in each original directory to point to the new place
- [x] 0.2 Run `git rm --cached` on these files. In `.gitignore`, delete the `!` lines that let logs through, and add `run-logs/`, `*.stderr`, `*.stdout`, `*metrics*.json`
- [x] 0.3 Add `tests/test_no_tracked_logs.py`: the number of `git ls-files` entries that match the log patterns is 0. Change the tests that depend on tracked logs to skip, or to read a path outside the repository
- [x] 0.4 (The user cancelled the "before #238" order on 10-13) Submit the log removal as a separate PR #240
- [x] 0.5 Delete the 28 stubs in `yeto/rl/engine/miles_adapter/` and change the about 13 referring files. Add a forbidden rule to `test_import_boundaries.py`. Shrink `import_boundary_allowlist.txt`
- [x] 0.6 Delete the 18 stubs in `yeto/rl/`. Rewrite all references with a script (yeto/, tests/, scripts/, tools/, docker/, docs/, tutorial/, Modal/Sky command strings). Change `python -m yeto.rl.learner` to `python -m yeto.rl.adapters.miles.island_entry`. Add a forbidden rule
- [x] 0.7 Add `MILES_FORK_LINEAGE` and `CRITIC_FORK_LINEAGE` to `pins.py`. Derive `algos/loss_variants.py:72`, `algos/critic_fork.py:47` and `adapters/miles/entry.py:205-236` from them. Merge the separate pin of `ssh_harness.py` into them. `test_rl_engine_pins.py` asserts that the three places are equal
- [x] 0.8 Change the lease default at `syncer/src/main.rs:160` to 300.0, and run `cargo test`

## 1. Keep only one Miles engine

- [x] 1.1 Write the GPU-run review conclusion (code logic, minimum config, collection list) to `infra-drafts/SESSION23/P0-legacy-baseline-review.md`
- [ ] 1.2 Run on GPU to record the baseline (use the code before the change, #238 HEAD `8f952cca`; run dense-full and SAO 2 times each to measure the run-to-run noise; first budget part about $150; show the user the result before going on). Do one minimum-card run each for SAO streaming, dense-full and DSV4 expert full-parameter. Store the raw data in the evidence directory outside the repository (token counts, step counts, loss curves, run manifest)
- [x] 1.3 Move `generate_rollout` (`legacy/engine.py:1166-1349`) and its 25 private helper functions as one module to `adapters/miles/models/rollout_legacy_port.py`. Point `sao_streaming_runtime.py:30-31` to the new place. Run `test_rl_sao_streaming_runtime.py` and `test_rl_sao_spec.py`
- [x] 1.4 Move `set_current_published_policy_identity` (`legacy/engine.py:225`) to the ports publish module
- [x] 1.5 Move the compaction record parsing (`legacy/engine.py:324-358`) to `compaction.py`. Run the compaction tests
- [x] 1.6 Move the DSV4 expert full-parameter runtime loading (`legacy/engine.py:1644`) to `models/deepseek_v4/expert_full_runtime.py`. Run `test_rl_deepseek_v4_expert_full_runtime.py`
- [ ] 1.7 (Merged into 1.7b.1) Dense-full does not go through the legacy engine: in `island_entry.py:1660-1666`, the full-parameter mode points to `full_parameter_dense.create_miles_full_parameter_dense_sync`. It shares only `backend_identity("legacy")` with legacy
- [ ] 1.7b **Bring three capabilities of legacy and of the Miles control loop into ports (decided by the user on 2026-10-13; for the decisions see section 4 of `infra-drafts/SESSION23/COORDINATION.md`, P0-D1…D9 and P0-S).** Target state: a Miles island has only the ports (`IslandDriver`) path, and the `yeto_policy_sync` branch is deleted.
  - [x] 1.7b.0 Read the code before starting to set the scope: `sao_streaming.py` (`MilesFullParameterRoleStream` :282-960, `MilesSaoStreamingPolicySync` :961-1684), `full_parameter_dense.py`, `dense_sweep_wire.py`, `chunked_full_parameter.py`, `engine/ports.py`, `engine/driver.py`, `engine/trainable_state.py`. Write the design section in design.md (Decision 11, design draft `infra-drafts/SESSION23/P0-1.7b.1-design.md`; open questions follow the recommendation; the user authorized continuous work on 10-13)
  - [ ] 1.7b.1 **Unified full-parameter sync (P0-D8)**: add one generic full-parameter Sync to ports. Write the single-role streaming sync once, based on `MilesFullParameterRoleStream`. For multiple roles (actor+critic of SAO/PPO), add the pairing rules on top (paired staging, paired apply, when one role fails both stop together). The dense strict sweep is only a restricted config and is not implemented separately. Keep the weights in fragments and never assemble the whole model in one process (P0-D1). Add no `DenseSweepSync` (P0-D2). The syncer starts processes by the roles that the algorithm declares (the same program, one process per role; the launcher `critic_syncer_command` changes to generate commands from the role list). Remove the duplicates between dense and SAO (keep one copy each of `_validate_optimizer_apply_transition` and `_legacy_identity_sha256`)
  - [ ] 1.7b.1a Unify the policy token as `yeto:<version>:<weight hash>` (P0-D3). The dense publish goes through the publish check of `driver.publish` (check the version and hash of each inference engine one by one, P0-D4)
  - [ ] 1.7b.1b The M1 limits (H=1, start from version 0, fixed island list) are not a design requirement (P0-D7). The generic Sync supports H>1, the syncer τ grace window and merging after the quorum of islands is reached (matches Milestone 4 of `docs/DECOUPLED_DILOCO_SECRLENV_PLAN.md`). For the evidence directory and the v0/final evaluation, use the generic ports mechanism. If there is none, first move the current code as it is and record a separate task
  - [ ] 1.7b.2 DSV4: connect `models/deepseek_v4/` on the ports side (clone LoRA, the expert-full runtime and the telemetry have already moved out in c0fe0367). Expert-full uses the full-parameter path of 1.7b.1. Selection allows `deepseek-v4-flash`, `attention-routed-experts` and `expert_full_count`. Run `test_rl_deepseek_v4_*.py`
  - [ ] 1.7b.3 SAO: use the two-role config of 1.7b.1. Rollout no longer goes through `rollout_legacy_port.generate_rollout`. **Relax "rollout and training versions must be strictly equal" to "the version gap is at most a limit" (P0-S).** Overlap of rollout and training inside an island reuses the Miles native feature (train_async / fully_async / `--max-weight-staleness`) and the verl native feature (fully_async / `staleness_threshold`). yeto gives only the limit. The ports LoRA path follows the same rule. Selection allows the `sao` preset and `--sao*` (`selection_argv.py:35`). Run `test_rl_sao_streaming_runtime.py`, `test_rl_sao_spec.py` and `test_rl_miles_sao_streaming.py`
  - [ ] 1.7b.4 Remove the remaining legacy references: `island_entry.py:25 verify_miles_revision`; `backend_identity("legacy")` (after the change, old test runs cannot be resumed; the user accepted this, P0-D5); `_append_rl_event` at `entry.py:1445,1503,1806` and `sao_streaming.py:1667`; `full_parameter_manifest_probe.py:16 miles_execution_source_sha256`. **Move the data fields in the dense training contract out of the island comparison** (decided by the user on 10-13; P3 depends on it)
  - [ ] 1.7b.5 GPU comparison run: merge it with 1.11. Use the minimum number of cards, write the review first, and pre-register in `infra-drafts/gpu-spend.md` (budget: see P0-G)
- [ ] 1.8 Remove all reject items at `engine/selection.py:54-67` and `selection_argv.py:35`. Change `test_rl_engine_selection.py` to assert that all cases return ports
- [x] 1.9 Delete the `co_names` probe at `island_entry.py:656`. Change the legacy `_append_rl_event` to use `append_record`. Run `test_rl_tape_collector_stream.py`
- [ ] 1.10 Delete the `adapters/miles/legacy/` directory, the legacy pin, the mirror and setup paths, the `--rl-engine legacy` type options, the legacy decoupled boundary loop and the `--spot` completed-groups resume path. Add a rule to `test_import_boundaries.py` that forbids `adapters.miles.legacy`. At the same time, remove the 3 imports of overlay in `launcher.py` (:3336,4148,7871), and lower the allowlist limit from 11 back to 10 or lower (P0-W)
- [ ] 1.11 Regression run on GPU: run SAO, dense-full and DSV4 again with the same config as 1.2. Compare with the baseline within the tolerance. Store the result in the evidence directory and write it in the P0 document

## 2. One bridge, one weight definition and one staleness definition

- [x] 2.1 Delete `rl/bridge.py:881 ElasticRlBridge`, `:981 make_island_bridge` (with its 2 test files) and `critic_state.py:198 TwoRoleStrictAvg` (with its 1 test). Check the unused PPO role-lane code in `local_learner.py`, and delete it
- [x] 2.2 Write the parity test `StrictRlBridge` against `StrictAvgSync` (the same set of deltas, strict mode, element-wise equal). If the parity fails, stop and record it in design.md
- [ ] 2.3 Merge with `engine/bridges.py` as the surviving side. `rl/bridge.py` becomes a thin wrapper or is deleted. Keep only `bridges.py:1097-1125` as the decoupled boundary loop
- [x] 2.4 Change `elastic.rs:483 merge_weight` to call `merge::learner_weight` (the step count is relaxed to u64; the original plan of `u32::try_from` is dropped; keep the zero-value branch). Run `cargo test`
- [x] 2.5 Add parity checks to `test_rl_inter_island_rust_golden.py`: the Python weight copy at `island_ledger.py:160` against the Rust fixture; a regex reads `DEFAULT_MAX_CARRY_LAG` and `MAX_SAMPLE_OUTER_LAG` from `elastic.rs` and compares them with `island_ledger.py:105,185`; assert that the fixture contains the rejected lag=3 case
- [x] 2.6 Write the weight, staleness and closing of the three sync modes (strict / decoupled / elastic) as a table in the `engine/bridges.py` docstring. A test asserts that the table header exists
- [x] 2.7 Keep only `DualStrictAvgSync._critic_catchup` for the critic sync

## 3. One resume abstraction

- [ ] 3.1 Define `ConsistentCut` and the unified `CheckpointStore` interface (four backends implement it: local directory, bucket mount, Modal volume, critic)
- [ ] 3.2 Change the three save paths, `InFlightSaver`, `handle_notice` at reclaim and `round_cut`, to call the same `save(cut)`. Parity: the three triggers write the same format
- [ ] 3.3 Both the `--rl-elastic` controller store and the syncer local checkpoint of cross-island elasticity use this interface. `engine/resume.ResumeController` becomes the only resume controller
- [ ] 3.4 Resume tests: after preemption, periodic and manual triggers, the step count and the island state are consistent

## 4. Algorithm, reward and config consolidation (rolling; one PR per item; parity first)

- [ ] 4.1 GRPO group normalization: run parity on `algos/reward_pipeline.py:400`, `algos/seq_adv.py:332`, `fp32_reference.py:112` and `rlf/contracts.py:67` (the last one does not divide by std; first ask the user whether this is intended). If they agree, merge into one copy. For the two upstream copies (Miles/verl), add a parity test and do not change them
- [ ] 4.2 Merge `loss_reference.py` and `fp32_reference.py` into one. Merge the three policy-age modules into one and add a backend adapter. Merge the staleness checks of driver and governor into one place
- [ ] 4.3 Merge the PPOTrainer and GAE of `rl/algorithms/` into `rl/algos/`
- [ ] 4.4 Diff `MILES_DECLARED` and `G1_DECLARED`. Derive G1 from it or lock it with a test. Add a coverage test for verl `CONFIG_GOLDEN`
- [ ] 4.5 Move the six reward definitions into `rewards/registry`. Merge the two trust paths (TB2 HMAC and RLF), and merge `finish_trusted` and `_finish_rlf` into one. Merge `rl/contracts.py`, `rlf/contracts.py` and `backend_identity.py`
- [ ] 4.6 Merge the four Codex entries into one main entry plus thin wrappers. Merge generate into one file
- [ ] 4.7 RL flag schema: `run_config.py` is the only source and generates the arguments of cli, launcher and island_entry. Switch only when the diff of the argument list before and after generation is zero
- [ ] 4.8 Embed the algorithm hash in the island contract hash. Merge the three W&B modules, two exporters, two finalisations and two env scrubbers, one each

- [ ] 4.9 Merge the three verl run paths into one, and wire the mismatch criterion only once. First check where it is copied

## 5. Closing

- [ ] 5.1 At the end of each phase, update the status table in `infra-drafts/SESSION23/P0-code-quality.md` (implemented / verified / unverified)
- [ ] 5.2 `openspec validate rl-code-quality-cleanup --strict` passes
