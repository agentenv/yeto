# Tasks

## 0. Preconditions

- [ ] 0.1 Confirm that 2.3 (bridge merge), 3.1-3.3 (resume abstraction), 4.7 (flag schema), and 4.8 (hash nesting) of `rl-code-quality-cleanup` are done; if not done, do not start
- [x] 0.2 The user's decisions on Q1-Q4 (2026-10-13) are written into "Decided" in proposal.md and into design.md
- [ ] 0.3 First step of the work: check whether "progress weighted by compute" in cross-island elasticity is computed correctly on both colocated and fixed-partition islands, and write the conclusion into "Cross-island elasticity check" in design.md
- [ ] 0.4 Check whether the algorithm hash changes with the `with_partitioned_serial` capability set; placement-related capabilities do not enter the algorithm hash (Q4); confirm together with 4.8 at its start, and write the conclusion into "Algorithm resolution" in design.md
- [ ] 0.5 Check that the `rl_driver_start` event in `driver.py` contains a `placement` field

## 1. Per-island values (local machine)

- [ ] 1.1 Add "per-island placement" to the unified parameter definition of the P0 4.7 flag schema, and implement per-island override of `rl_placement`, `rl_standby_gpus`, `rl_island_gpus_per_node`
- [ ] 1.2 Change every launcher function that reads these three values to take a per-island value (`rl_actor_gpus_per_node`, `rl_trainer_shape`, `rl_island_layout`, `rl_min_nodes`, `_prepare_ports_algorithm`, `_prepare_rl_args`, island command assembly)
- [ ] 1.3 A static check like `test_import_boundaries`: the launcher has no more places that read the global `args.rl_placement` directly (except the allowlist)

## 2. Validation (local machine)

- [ ] 2.1 Implement items 1-3 of "Validation" in design.md (under strict, reject mixed placement and explain the reason; allow it under cross-island elasticity)
- [ ] 2.2 Unit tests cover every rejection, and assert that no cloud resource is created

## 3. Records (local machine)

- [ ] 3.1 The run manifest records the placement and the algorithm hash of each island, and records the time per round of each island (general record)
- [ ] 3.2 Dry-run test: when two islands have different placement, the resource shape and command line of each island are correct

## 4. Verification on GPU (Modal 3×H100)

- [ ] 4.1 Review before running on GPU: write the code logic, the minimum config, and the collection list into `infra-drafts/SESSION23/`; pre-register in `/home/michael/work/infra-drafts/gpu-spend.md` (the budget is already part of the second tranche of the $500 approved by the user); do not open cards until the review and the pre-registration are done
- [ ] 4.2 Plan: Modal 3×H100, cross-island elasticity (`--rl-island-scheduling elastic`), 3 rounds. Island 0 colocated with 1 card; island 1 fixed-partition with 2 cards (1 for training, 1 for inference). Small model; raw events and the time per round of each island are stored in an evidence directory outside the repo
- [ ] 4.3 Criteria: the cross-island sync check passes; the `rl_driver_start` of the two islands reports different `placement`; the result of progress weighted by compute agrees with the check in 0.3; under strict, the same config fails at launch and opens no card (this item can be tested locally); adding `--rl-elastic` to the same config fails at launch and opens no card (this item can be tested locally)
