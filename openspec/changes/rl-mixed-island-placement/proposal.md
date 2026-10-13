# Proposal

## Why

PM item 2. The user confirmed it on 2026-10-12 (00-DECISIONS Q2). Mix placement modes per island: island 1 uses colocated, island 2 uses fixed-partition. The motive is the mixed modes themselves, not mixed card types. Today `--rl-placement` is a global flag, so all islands use one placement. On 2026-10-13 the user decided to implement this change this week and settled Q1-Q4 (see the "Decided" section).

## What Changes

- `--rl-placement`, `--rl-standby-gpus` and `--rl-island-use-gpus-per-node` can be overridden per island. Syntax: add the launcher flag `--rl-island-placement ISLAND:KEY=VALUE` to the P0 4.7 flag schema (implemented, see design.md "Value rules").
- Main goal: allow mixed placement under cross-island elastic (`--rl-island-scheduling elastic`). Under strict (`--rl-island-scheduling legacy`, the default), mixed placement is refused at startup with an error that states the reason.
- First step of the work: verify that the "capacity-weighted advance" of cross-island elastic is computed correctly on both colocated and fixed-partition islands (so far only the code was read; unverified).
- Launcher functions that depend on the placement take per-island values: `rl_actor_gpus_per_node` (line 361), `rl_trainer_shape` (379), `rl_island_layout` (458), `rl_min_nodes` (525), `_prepare_ports_algorithm` (1613, 1677), `_prepare_rl_args` (2370, 2670), island command assembly (2262-2266).
- Startup checks: when any island is colocated, refuse `--rl-elastic` (intra-island elastic, requires fixed-partition, `check_elastic_placement` at `launcher.py:2074`); the same for `--rl-overlap-eval`. Under strict, mixed placement is refused at startup.
- The per-island `rl_driver_start` event already has a `placement` field (the P4 draft says the driver already emits it; not verified line by line in this change). The run manifest also records each island's placement.
- The compatibility group does not change: all islands must still use the same vendor and the same card type.

## Current state and blockers found by reading code (2026-10-13)

1. **The cross-island hash does not include the placement** (follows P4 section 5.1; rechecked in this change at `backend_identity.py:82-87`: `session_contract_hash` = layout fingerprint + backend identity digest). `ExecutionProfile.contract_hash` (`execution_profile.py:249-251`) includes `execution_mode`. In this change it was seen used for the decision records of `pause_audit.py`; no cross-island comparison was seen.
2. **Overturned**: P4 section 5.2 item 1 says "elastic scheduling requires all islands to be fixed-partition, so mixed placement cannot be used with elastic". From the code: `check_elastic_placement` (`execution_profile.py:103-108`) is called only by the `--rl-elastic` path (`launcher.py:2074`). That path is the intra-island sampling reconfiguration controller. Cross-island elastic scheduling is a different flag, `--rl-island-scheduling elastic` (`cli.py:488`). This change found no placement check for it in the launcher. So "cross-island elastic + mixed placement" is not blocked by this check. Whether it is blocked elsewhere, or is semantically feasible: unverified, listed as the first step of the work.
3. **New finding**: a per-island override mechanism already exists in `yeto/island_overrides.py` (`--rl-island-override ISLAND:KEY=VALUE`, allowlist, per-island copy of args, writes the run manifest and the `rl_island_override` event). But it requires `--rl-negative-test-run` and is used only for negative-test runs. The P4 draft proposed copying the `learner_image_for` style and did not mention this module.
4. **New blocker**: `_prepare_ports_algorithm` (`launcher.py:1677-1682`) decides from the global placement whether to add the `with_partitioned_serial` capability, then resolves the algorithm. With mixed placement it must resolve per island. P0 task 4.8 plans to embed the algorithm hash into the island contract hash. If the algorithm hash changed with the capability set, mixed placement would be refused at the handshake after 4.8. Does the algorithm hash change with the capability set? Verified in 4.8: no (the algorithm hash covers only the spec; the capability set is not in the spec).
5. The three "check first" items of P4 section 5.4: `pause_audit.py:79 _AUDITED_MODES` contains both `colocated-serial` and `partitioned-serial`, and it judges by each island's own profile, with no global-consistency assumption. A grep of `wandb_rl.py` and `engine/recommend.py` finds no `placement` or `execution_mode`, so no global assumption was found; these files were not read function by function.

## Dependencies (P0 `rl-code-quality-cleanup`)

These must be complete before the work starts:
- **4.7 RL flag schema**: the three flags become per-island overridable. Add "per-island placement" to the single schema, instead of writing it once in each of three layers.
- **4.8 Algorithm hash embedded in the island contract hash**: see blocker 4. Placement-related capabilities do not enter the algorithm hash (Q4 decided); confirm this together when 4.8 starts.
- **2.3 Bridge merge**: which bridge the island driver uses affects both the colocated and the fixed-partition driver paths (`driver.py:585`, `driver.py:1620`). After the merge, only one set needs testing.
- **Phase 3 recovery abstraction (3.1-3.3)**: the two placements save at different times (fixed-partition has standby cards). After `save(cut)` is unified, recovery of a mixed run needs to be verified only once.

No dependency: phase 1, delete legacy (legacy does not support `--rl-placement`; `launcher.py:1612-1616` already refuses it).

## Decided (user, 2026-10-13)

- **This change is done this week** (it was first written as not done).
- **Q1 syntax**: wait for the P0 4.7 flag schema and add "per-island placement" to the unified parameter definition. Do not extend `island_overrides.py`. Do not add a dedicated flag.
- **Q2 cross-island elastic**: cross-island elastic (`--rl-island-scheduling elastic`) is the main goal of this change and allows mixed placement. Under strict (`--rl-island-scheduling legacy`, the default), mixed placement is refused at startup with an error that states the reason. First step of the work: verify that the "capacity-weighted advance" of cross-island elastic is computed correctly on both colocated and fixed-partition islands (unverified). Note: intra-island elastic `--rl-elastic` (requires fixed-partition) and cross-island elastic are different things.
- **Q3 strict**: strict is not tested. This change targets cross-island elastic only. No dedicated record of "fast islands wait for slow islands". Keep the generic record of per-island round time.
- **Q4 algorithm hash**: placement-related capabilities do not enter the algorithm hash. Decide this together when P0 4.8 starts.
- **GPU validation**: Modal 3×H100, 3 rounds. Island 0 colocated on 1 card; island 1 fixed-partition on 2 cards (1 for training, 1 for inference); run under cross-island elastic. The budget is included in the second tranche of the $500 the user approved. Before launching cards, write the review and pre-register in `/home/michael/work/infra-drafts/gpu-spend.md`.

## Capabilities

### New Capabilities
- `rl-mixed-island-placement`: choose the placement per island, including the resource computation, the startup checks and the recording rules.

### Modified Capabilities
(None.)

## Impact

- Code: `yeto/cli.py`, `yeto/launcher.py` (the functions listed above), `yeto/rl/engine/run_flags.py` (after 4.7).
- Tests: local dry-run tests cover the resource shape, the command line and the checks when two islands use different placements.
- GPU: acceptance needs one 3-card run (Modal 3×H100); see section 4 of tasks.md. Write the review and pre-register before launching cards.
