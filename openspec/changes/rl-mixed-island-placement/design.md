# Design

## Two axes (as in the P4 draft)

- Axis A, sync mode: strict / decoupled / elastic. One syncer speaks only one protocol, so it cannot mix modes per island. This change does not touch it.
- Axis B, placement: colocated / fixed-partition. It affects only the driver inside the island. This change makes it a per-island value.

There are also two easily confused kinds of "elastic":
- `--rl-elastic`: the sampling reconfiguration controller inside an island. It requires fixed-partition (`execution_profile.py:103-108`).
- `--rl-island-scheduling elastic`: cross-island elastic scheduling. It is the main target of this change. This reading found no placement check for it. The default is `legacy` (that is, strict).

## Value rules

The value for each island is taken in this order: per-island override > global flag > default `colocated`. The per-island override is written in the "per-island placement" of the P0 4.7 flag schema (Q1 decided). Every launcher function that reads `rl_placement` is changed to receive learner_id, or to receive a copy of args already resolved per island (`island_overrides.py` already uses the approach "copy args per island").

## Validation

1. When any island is colocated, reject `--rl-elastic` and `--rl-overlap-eval`.
2. When any island is colocated, `--rl-standby-gpus` MUST be 0 for that island (the existing rule at `run_config.py:689` already checks inside the island; the launcher side checks earlier).
3. Under strict (`--rl-island-scheduling legacy`, the default), refuse to start when mixed placement appears. The error explains: strict requires the same placement for all islands, the reason, and how to fix it (Q2 decided). Under cross-island elasticity, mixed placement is allowed, provided that the check of "progress weighted by compute" in the next section passes.
4. A different compatibility group is still rejected by the syncer (`server.rs:1060-1067`). This change does not change it.

## Cross-island elasticity check (first step of the work)

Check whether "progress weighted by compute" is computed correctly on both colocated and fixed-partition islands. So far the code has only been read, and this is unverified. Write the conclusion in this section.

## Algorithm resolution

Change `_prepare_ports_algorithm` to resolve the capability set and the algorithm per island. If the algorithm hashes resolved for the islands differ:
- Before 4.8: they do not enter the cross-island hash, and are recorded in the run manifest.
- After 4.8: placement-related capabilities do not enter the algorithm hash (Q4 decided). Confirm this together when 4.8 starts.

## Risks

- strict is not in the scope of this change (Q3 decided) and is not tested. The per-island time per round is recorded as a general record.
- The launcher has more than 10 places that read `rl_placement` globally. If one place is missed, the resource shape of some island is wrong. The dry-run test asserts the resource shape island by island.
