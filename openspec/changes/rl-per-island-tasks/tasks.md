# Tasks

## 0. Preconditions

- [x] 0.1 Confirm that 1.10 (remove legacy), 4.7 (flag schema), and 4.8 (hash nesting) of `rl-code-quality-cleanup` are done; if not done, do not start
- [x] 0.2 The user decides Q1-Q5 pending in proposal.md, and the decisions are written into design.md
- [x] 0.3 Read further whether the syncer compares the dense-full profile payload across islands (`local_learner.py:276`), and write the conclusion into "Contract hash" in design.md (P0-1.7b.4 already treats it as compared across islands: training contract v5 removes the training data; this change bumps to v6 and also removes eval, see design.md)

## 1. Parse and validate (local machine)

- [x] 1.1 Add `rl_island_tasks` to the `run_flags.py` schema (the parse logic lives on the launcher side); parse both the string format and the JSON file format, and expand ranges
- [x] 1.2 Validation at launch: island number range, every island has a list, task ids exist; unequal task counts only give a warning; dense-full/SAO allow per-island tasks (depends on P0 1.7b.4 moving data out of the comparison)
- [x] 1.3 Unit tests cover the positive cases and every rejection of 1.1 and 1.2

## 2. Generate and deliver prompt files (local machine)

- [x] 2.1 The launcher calls `write_prompt_data(task_ids=...)` per island, generates the file, and mounts it as `--data` of that island
- [x] 2.2 The run manifest and the `rl_driver_start` event record `task_ids` and the sha256 of the prompt file
- [x] 2.3 At island start, compare the sha256 of the prompt file with the manifest, and refuse if they differ
- [x] 2.4 Dry-run test: `--data` differs in the two island commands, and the manifest fields are complete

## 3. eval

- [x] 3.1 eval uses the full set and runs only on a separate eval island; training islands do not load the eval set (Q3); a test asserts that the start arguments of training islands contain no eval data

## Progress (2026-10-13, P3 sub-agent, branch s23-p3)

- Sections 0-3 are implemented, and local tests pass (new assertions in `tests/test_rl_island_tasks.py`, `tests/test_rl_engine_driver.py`, `tests/test_rl_launcher.py`). Not verified on GPU.
- Two differences from the original plan are in the "Implementation notes" of design.md: the task list is filtered by line from the `--data` JSONL and does not call `write_prompt_data`; with per-island tasks, the launcher rejects eval on training islands.

## 4. Acceptance on GPU (minimum card count plan; this proposal does not open cards)

- [ ] 4.1 Review before running on GPU: write the code logic, the minimum config, and the collection list into `infra-drafts/SESSION23/`
- [ ] 4.2 Plan: Modal, two islands with 1 H100 each (2 cards in total), LoRA ports, strict-avg, small model, 2 tasks per island, 3 rounds; raw events and the manifest are stored in an evidence directory outside the repo. Hand over to P0-1.11 GPU run to run in one merged batch; the same batch may add one full-parameter two-island group (different tasks per island, different eval settings) and check that the syncer does not reject
- [ ] 4.3 Criteria: the `rl_driver_start.island_tasks` of the two islands reports different `task_ids`, and `data_sha256` matches `island_tasks` in the run manifest `run_manifest.json`; the `metadata.task_id` of the rollout samples of each island comes only from the tasks of that island; the cross-island sync check passes; negative case (local machine is enough): after the prompt file on an island is changed, the island refuses to start
