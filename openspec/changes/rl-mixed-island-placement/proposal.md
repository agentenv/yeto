# Proposal

## Why

PM item 2, confirmed by the user on 2026-10-12 (00-DECISIONS Q2): mix placements per island, with island 1 using colocated and island 2 using fixed-partition. The motive is the mixed mode itself, not mixed card types. Today `--rl-placement` is a global flag, and all islands use one placement. On 2026-10-13 the user decided to implement it this week, and Q1-Q4 are decided (see the "Decided" section).

## What Changes

- `--rl-placement`, `--rl-standby-gpus`, and `--rl-island-gpus-per-node` can be overridden per island. How to write it: wait for the P0 4.7 flag schema, and add "per-island placement" to the unified parameter definition.
- Main target: allow mixed placement under cross-island elasticity (`--rl-island-scheduling elastic`). Under strict (`--rl-island-scheduling legacy`, the default), mixed placement is refused at launch, and the error explains the reason.
- First step of the work: check whether "progress weighted by compute" in cross-island elasticity is computed correctly on both colocated and fixed-partition islands (so far the code has only been read, and this is unverified).
- Change the launcher functions that depend on placement to per-island values: `rl_actor_gpus_per_node` (line 361), `rl_trainer_shape` (379), `rl_island_layout` (458), `rl_min_nodes` (525), `_prepare_ports_algorithm` (1613, 1677), `_prepare_rl_args` (2370, 2670), island command assembly (2262-2266).
- Validation at launch: when any island is colocated, reject `--rl-elastic` (intra-island elasticity, requires fixed-partition, `check_elastic_placement` at `launcher.py:2074`); the same for `--rl-overlap-eval`. Under strict, reject mixed placement at launch.
- The `rl_driver_start` event of each island already has a `placement` field (the P4 draft says the driver already emits it; this reading did not verify it line by line). The run manifest adds the placement of each island.
- The compatibility group is not touched: all islands must still have the same vendor and card type.

## Current state and obstacles from reading code (2026-10-13)

1. **Cross-island hashes do not contain placement** (as in P4 section 5.1; this reading re-checked `backend_identity.py:82-87`: `session_contract_hash` = layout fingerprint + backend identity digest). `ExecutionProfile.contract_hash` (`execution_profile.py:249-251`) contains `execution_mode`. This reading saw it used in the decision records of `pause_audit.py` and saw no cross-island comparison.
2. **Overturned**: P4 section 5.2 item 1 says "elastic scheduling requires all islands to be fixed-partition, so mixed placement cannot be used together with elastic". From reading code: `check_elastic_placement` (`execution_profile.py:103-108`) is called only by the `--rl-elastic` path (`launcher.py:2074`). It is the sampling reconfiguration controller inside an island. Cross-island elastic scheduling is another flag, `--rl-island-scheduling elastic` (`cli.py:488`). This reading found no placement check for it in the launcher. So "cross-island elastic + mixed placement" is not blocked by this check. Whether something else blocks it, or whether it is feasible in meaning: unverified. This is listed as the first step of the work.
3. **New finding**: a per-island override mechanism already exists, `yeto/island_overrides.py` (`--rl-island-override ISLAND:KEY=VALUE`, with an allowlist, copying args per island, and writing the run manifest and the `rl_island_override` event). But it requires `--rl-negative-test-run` and is used only for negative-test runs. The P4 draft proposed copying the `learner_image_for` style and did not mention this module.
4. **New obstacle**: `_prepare_ports_algorithm` (`launcher.py:1677-1682`) decides by the global placement whether to add the `with_partitioned_serial` capability, and then resolves the algorithm. After mixing, it must resolve per island. P0 task 4.8 plans to embed the algorithm hash in the island contract hash. If the algorithm hash changes with the capability set, then after 4.8 mixed placement would be rejected at the handshake. Whether the algorithm hash changes with the capability set: unverified.
5. The three "must check first" items of P4 section 5.4: `pause_audit.py:79 _AUDITED_MODES` contains both `colocated-serial` and `partitioned-serial`, and it judges by the own profile of each island and does not assume uniformity. grep finds no `placement` or `execution_mode` in `wandb_rl.py` and `engine/recommend.py`, and no global assumption was found. They were not read through function by function.

## Dependencies (P0 `rl-code-quality-cleanup`)

Must be done before this change starts:
- **4.7 RL flag schema**: the three flags must become overridable per island. Add "per-island placement" on the single schema, instead of writing it once in each of three layers.
- **4.8 embed the algorithm hash in the island contract hash**: see obstacle item 4. Placement-related capabilities do not enter the algorithm hash (Q4 decided). Confirm this together when 4.8 starts.
- **2.3 bridge merge**: which bridge the driver inside an island uses affects both driver paths, colocated and fixed-partition (`driver.py:585`, `driver.py:1620`). After the merge, only one set needs testing.
- **Phase 3 resume abstraction (3.1-3.3)**: the two placements save at different times (fixed-partition has standby cards). After `save(cut)` is unified, the resume of a mixed run needs to be verified only once.

Not dependent: phase 1 remove legacy (legacy does not support `--rl-placement`; `launcher.py:1612-1616` already rejects it).

## Decided (user, 2026-10-13)

- **This change is done this week** (it was written as not done).
- **Q1 How to write it**: wait for the P0 4.7 flag schema, and add "per-island placement" to the unified parameter definition. Do not extend `island_overrides.py` and do not add a dedicated flag.
- **Q2 Cross-island elasticity**: cross-island elasticity (`--rl-island-scheduling elastic`) is the main target of this change and allows mixed placement. Under strict (`--rl-island-scheduling legacy`, the default), mixed placement is refused at launch, and the error explains the reason. First step of the work: check whether "progress weighted by compute" in cross-island elasticity is computed correctly on both colocated and fixed-partition islands (unverified). Note: intra-island elasticity `--rl-elastic` (requires fixed-partition) is not the same thing as cross-island elasticity.
- **Q3 strict**: strict is not tested. This change targets only cross-island elasticity. No dedicated record of "fast islands wait for slow islands" is made; the general record of the time per round of each island is kept.
- **Q4 Algorithm hash**: placement-related capabilities do not enter the algorithm hash. P0 4.8 decides it together when it starts.
- **Verification on GPU**: Modal 3×H100, 3 rounds. Island 0 colocated with 1 card; island 1 fixed-partition with 2 cards (1 for training, 1 for inference); run under cross-island elasticity. The budget is already part of the second tranche of the $500 approved by the user. Before opening cards, write the review first and pre-register in `/home/michael/work/infra-drafts/gpu-spend.md`.

## Capabilities

### New Capabilities
- `rl-mixed-island-placement`: Choose the placement per island, including resource calculation, validation at launch, and record rules.

### Modified Capabilities
(None.)

## Impact

- Code: `yeto/cli.py`, `yeto/launcher.py` (the functions listed above), `yeto/rl/engine/run_config.py` (after 4.7).
- Tests: local dry-run tests cover the resource shape, command lines, and validation when two islands have different placement.
- Run on GPU: acceptance needs one 3-card run (Modal 3×H100), see section 4 of tasks.md. Write the review and pre-register before opening cards.
