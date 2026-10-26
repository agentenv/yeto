# Tasks

## 0. Prerequisites

- [x] 0.1 Confirm that 2.3 (bridge merge), 3.1-3.3 (recovery abstraction), 4.7 (flag schema) and 4.8 (hash nesting) of `rl-code-quality-cleanup` are done; if not, do not start (2026-10-13: baseline s23-p0-1.7b.4 bf6465bf already contains all four)
- [x] 0.2 The user's decisions on Q1-Q4 (2026-10-13) are written into the proposal.md "Decided" section and design.md
- [x] 0.3 First step of the work: verify that the "capacity-weighted advance" of cross-island elastic is computed correctly on both colocated and fixed-partition islands, and write the conclusion into the design.md section "Cross-island elastic verification" (verified by reading code: neither the advance nor the weight reads the placement; capacity is always 1.0, so "capacity-weighted" is in fact by island count; the new problem is recorded separately)
- [x] 0.4 Verify whether the algorithm hash changes with the `with_partitioned_serial` capability set; placement-related capabilities do not enter the algorithm hash (Q4); confirm together with the start of 4.8; write the conclusion into the design.md section "Algorithm resolution"
- [x] 0.5 Verify that the `rl_driver_start` event in `driver.py` contains the `placement` field (emitted in `handshake()`, verified)

## 1. Per-island values (local)

- [x] 1.1 In the unified parameter definition of the P0 4.7 flag schema (`yeto/rl/engine/run_flags.py`), add "per-island placement", and implement per-island overrides of `rl_placement`, `rl_standby_gpus` and `rl_island_gpus_per_node` (`--rl-island-placement ISLAND:KEY=VALUE`, module `yeto/rl/engine/island_placement.py`)
- [x] 1.2 In the launcher, change all functions that read these three values to take per-island values (`rl_actor_gpus_per_node`, `rl_trainer_shape`, `rl_island_layout`, `rl_min_nodes`, `_prepare_ports_algorithm`, `_prepare_rl_args`, island command assembly)
- [x] 1.3 A static check in the style of `test_import_boundaries`: the launcher no longer has a point that reads the global `args.rl_placement` directly (except the allowlist)

## 2. Checks (local)

- [x] 2.1 Implement items 1-3 of the design.md section "Checks" (under strict, mixed placement is refused with the reason stated; under cross-island elastic it is allowed)
- [x] 2.2 Unit tests cover every refusal and assert that no cloud resource is created

## 3. Records (local)

- [x] 3.1 The run manifest records each island's placement and algorithm hash, and records each island's round time (generic record: driver event `rl_round_wall`)
- [x] 3.2 Dry-run test: with two different placements, the resource shape and the command line of each island are correct

## 4. GPU validation (Modal 3×H100)

- [ ] 4.1 Review before GPU: write the code logic, the minimal configuration and the collection list into `infra-drafts/SESSION23/`; pre-register in `/home/michael/work/infra-drafts/gpu-spend.md` (the budget is included in the second tranche of the $500 the user approved); do not launch cards before the review and the pre-registration are done
- [ ] 4.2 Plan: Modal 3×H100, cross-island elastic (`--rl-island-scheduling elastic`), 3 rounds. Island 0 colocated on 1 card; island 1 fixed-partition on 2 cards (1 for training, 1 for inference). Small model; store the raw events and the per-island round time in an evidence directory outside the repository
- [ ] 4.3 Criteria: the cross-island sync check passes; the two islands report different `placement` in `rl_driver_start`; the result of the capacity-weighted advance matches the verification in 0.3; under strict the same configuration fails at startup and no card is launched (this item can be tested locally); the same configuration plus `--rl-elastic` fails at startup and no card is launched (this item can be tested locally). The two local items are covered by `tests/test_rl_mixed_island_placement.py`
- [x] 4.4 Review addition (local): when per-island placement values exist, the full-parameter training contract does not contain the placement parameters (except the engine card count); unified placement keeps the GPU split check; tests lock this (commit 5710fcd5 and the review rework). This is not validated separately on GPU (the full-parameter preset now requires legacy scheduling, so a mixed run cannot reach it)

### GPU validation plan (handed to P0-1.11 GPU run; this agent does not run on GPU)

- Configuration: Modal 3×H100 (`H100!:N` pins the card type and asserts the GPU name), two islands: `--gpu modal:1xh100,modal:2xh100`; `--rl-island-scheduling elastic`; `--rl-placement colocated --rl-island-placement 1:placement=fixed-partition --rl-rollout-gpus 1`; small model (the same one as the other cases of P0-1.11 GPU run), `--total-steps 3`; `--rl-max-policy-age 0` (do not mix in the P0-S variable).
- Required locally before launching cards: run the same command with `--dry-run`, and check that the island 0 command has no `--rl-placement`, that the island 1 command contains `--rl-placement fixed-partition --rollout-num-gpus 1 --actor-num-gpus-per-node 1`, and that `--rl-expected-algorithm-sha256` is the same on both islands. The same command without `--rl-island-scheduling elastic` must fail at startup. Adding `--rl-elastic` must fail at startup.
- Collection (store in the evidence directory outside the repository): the full event tape of both islands, the syncer tape and the elastic status, `run_manifest.json`, launch.log, and the GPU name of each island.
- Criteria:
  1. The 3 rounds finish, the syncer admits both islands (no identity/contract refusal), and the outer step advances normally in each round.
  2. `rl_driver_start.placement` of the two islands is `colocated` and `fixed-partition` respectively; `island_placements` in `run_manifest.json` matches them.
  3. Every round of both islands has `rl_round_wall`, recording `round_seconds` (record only; do not assume which is faster or a speedup ratio).
  4. In the syncer elastic status, the capacity of both islands is 1.0, and the advance matches the design.md section "Cross-island elastic verification" (advance when arrived islands ≥ θ × island count).
