# Tasks

## 0. Preconditions

- [ ] 0.1 Confirm that 1.10 (remove legacy), 4.7 (flag schema), and 4.8 (hash nesting) of `rl-code-quality-cleanup` are done; if not done, do not start
- [ ] 0.2 The user decides Q1-Q5 pending in proposal.md, and the decisions are written into design.md
- [ ] 0.3 Read further whether the syncer compares the dense-full profile payload across islands (`local_learner.py:276`), and write the conclusion into "Contract hash" in design.md

## 1. Parse and validate (local machine)

- [ ] 1.1 Add `rl_island_tasks` to the `run_config.py` schema; parse both the string format and the JSON file format, and expand ranges
- [ ] 1.2 Validation at launch: island number range, every island has a list, task ids exist; unequal task counts only give a warning; dense-full/SAO allow per-island tasks (depends on P0 1.7b.4 moving data out of the comparison)
- [ ] 1.3 Unit tests cover the positive cases and every rejection of 1.1 and 1.2

## 2. Generate and deliver prompt files (local machine)

- [ ] 2.1 The launcher calls `write_prompt_data(task_ids=...)` per island, generates the file, and mounts it as `--data` of that island
- [ ] 2.2 The run manifest and the `rl_driver_start` event record `task_ids` and the sha256 of the prompt file
- [ ] 2.3 At island start, compare the sha256 of the prompt file with the manifest, and refuse if they differ
- [ ] 2.4 Dry-run test: `--data` differs in the two island commands, and the manifest fields are complete

## 3. eval

- [ ] 3.1 eval uses the full set and runs only on a separate eval island; training islands do not load the eval set (Q3); a test asserts that the start arguments of training islands contain no eval data

## 4. Acceptance on GPU (minimum card count plan; this proposal does not open cards)

- [ ] 4.1 Review before running on GPU: write the code logic, the minimum config, and the collection list into `infra-drafts/SESSION23/`
- [ ] 4.2 Plan: Modal, two islands with 1 H100 each (2 cards in total), LoRA ports, strict-avg, small model, 2 tasks per island, 3 rounds; raw events and the manifest are stored in an evidence directory outside the repo
- [ ] 4.3 Criteria: the `rl_driver_start` of the two islands reports different `task_ids`; the rollout samples of each island come only from the tasks of that island; the cross-island sync check passes
