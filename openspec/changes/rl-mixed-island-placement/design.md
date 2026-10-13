# Design

## Two axes (follows the P4 draft)

- Axis A, sync mode: strict / decoupled / elastic. One syncer speaks one protocol only, so the mode cannot be mixed per island. This change does not touch it.
- Axis B, placement: colocated / fixed-partition. It affects only the intra-island driver. This change makes it per-island.

There are also two "elastic" terms that are easy to confuse:
- `--rl-elastic`: the intra-island sampling reconfiguration controller. It requires fixed-partition (`execution_profile.py:103-108`).
- `--rl-island-scheduling elastic`: cross-island elastic scheduling, the main goal of this change. The default value is `legacy` (that is, strict).

## Value rules

Per-island value order: per-island override > global flag > default `colocated`.

Implementation (2026-10-13, commit c034a506):
- Flag: `--rl-island-placement ISLAND:KEY=VALUE`, repeatable. It exists only at the launcher layer (`cli_flag`). It is defined once, in the placement topic of `yeto/rl/engine/run_flags.py`. Three KEYs: `placement` (maps to `--rl-placement`), `standby-gpus` (`--rl-standby-gpus`), `use-gpus-per-node` (`--rl-island-use-gpus-per-node`; on the island side it is converted to the learner's `--rl-island-gpus-per-node`). All three are neutral concepts, so the Miles `run_flags.py` needs no change.
- Module: `yeto/rl/engine/island_placement.py` (pure functions, does not import launcher) parses, checks, and builds the per-island copy of args (`island_args`) and the record rows.
- Launcher: `rl_island_args(args, m)` returns the copy of args for island m. When there is no per-island value it returns args itself, so the learner command is byte-for-byte unchanged (tested; "byte-for-byte unchanged" refers only to the learner command; the run manifest adds `island_placements` and the dry-run adds `placement` for each island). `_island_task` applies the per-island placement first and then `island_overrides`. So on both the sky path and the Modal path, `--actor-num-*`, `--rl-placement`, `--rollout-num-gpus` and `--rl-standby-gpus` in the learner command take per-island values. The shape check in `_prepare_rl_args`, `rl_min_nodes`, `require_ports_supported`, and the capability check in `_prepare_ports_algorithm` all run per island (deduplicated by distinct value).
- In a mixed run, a colocated island has no dedicated inference cards and no standby cards. It does not inherit the global `--rl-rollout-gpus` and `--rl-standby-gpus` (they are set to None/0 in the copy). Explicitly giving a colocated island `standby-gpus` > 0 is refused.
- Static check (task 1.3): `tests/test_rl_mixed_island_placement.py` scans the launcher with an AST. The functions that read `args.rl_placement`, `rl_standby_gpus`, `rl_island_use_gpus_per_node` or `rollout_num_gpus` must equal the allowlist. A second check asserts that the island task factories (`make_miles_island_task` / `task_factory`) are called only in `_island_task`. Readers of per-island args are `rl_island_spec`, `rl_actor_gpus_per_node`, `rl_trainer_shape`, `rl_island_layout`, `rl_min_nodes` and `_ports_algorithm_flags` (all callers pass the island copy). Readers of global args are `_prepare_sao_streaming` and `_prepare_dense_full_sweep` (both require legacy scheduling, so a mixed run cannot reach them).

## Checks

All checks are in `island_placement.check`, called by `_check_ports_infra_switches` before cards are launched:
1. Under strict (`--rl-island-scheduling legacy`, the default), mixed placement is refused. The error lists each island's placement and hints: "use one placement for all islands, or add `--rl-island-scheduling elastic`". Under cross-island elastic it is allowed.
2. A mixed run with `--rl-elastic` or `--rl-overlap-eval` is refused. The error names the colocated island number. When all islands use the same placement, the existing execution profile checks and errors (`check_elastic_placement`, `check_overlap_eval`) apply, and the behavior does not change.
3. Explicitly giving a colocated island `standby-gpus` > 0 is refused (the check inside the island at `run_config.py:689` does not change).
4. A different compatibility group is still refused by the syncer (`server.rs:1060-1067`). This change does not touch it.

## Cross-island elastic verification (0.3, verified by reading code, not verified on GPU)

Conclusion: neither the advance nor the merge weight of cross-island elastic reads the placement. Colocated islands and fixed-partition islands are computed with the same formula.

1. Advance criterion: the syncer advances when `cap_arrived >= quorum_theta * cap_total` (`syncer/src/elastic.rs:1166-1170`, `:1210`). The capacity of each island comes from the JOIN message.
2. Source of capacity: `ElasticAvgSync._client_config` (`yeto/rl/engine/bridges.py:459-464`) does not pass capacity, and `ElasticClientConfig.capacity` defaults to 1.0 (`yeto/rl/elastic_client.py:517`). So the capacity of every island is 1.0, whatever the placement and the card count.
3. Merge weight: `learner_weight = tokens² / steps` (`syncer/src/merge.rs:31-38`). The island side passes `c_tokens = stats.action_tokens` and `c_steps = local_optimizer_steps` (`bridges.py:610-611`). Both placements go through the same `IslandDriver` → `ElasticAvgSync` path, and the weight does not read the placement.
4. Therefore: mixed placement does not make the advance or the weight wrong.
5. New finding (does not block this change; written to the message board): "capacity-weighted advance" is in fact "advance by island count". Capacity is always 1.0, and the quorum is arrived islands ≥ θ × island count. A fast island's advantage shows only as more rounds submitted per unit time and a larger weight for a delta with more tokens. It does not show in the quorum share. To weight by compute, the island side must report the real capacity. Open a separate change for that.

## Algorithm resolution (0.4)

`_prepare_ports_algorithm` runs one capability check for each placement that appears among the islands (a fixed-partition island adds `with_partitioned_serial`). The algorithm spec is resolved only once, and all islands receive the same `--rl-expected-algorithm-sha256`.

4.8 verified (2026-10-13): the algorithm hash is `AlgorithmSpec.sha256()` and covers only the spec itself. `with_partitioned_serial` changes only the capability set, not the spec, so the hash does not change with the placement (Q4). The test `test_algorithm_hash_ignores_placement_capabilities` locks this. This change adds a test that asserts the algorithm hash is the same in the commands of both islands of a mixed run. Mixed placement is not refused at the handshake because of the placement.

## Full-parameter training contract (review addition, commit 5710fcd5, revised after the review rework)

The P0-1.7b.4 review pointed out that placement parameters enter `shared_miles_argv` of full-parameter training contract v5, so the syncer would refuse two islands with different placements. The rule after the P4 review rework:
1. Only when per-island placement values exist (`--rl-island-placement`), the launcher adds the island flag `--rl-per-island-placement` to every island (an internal Miles island flag, defined in `yeto/rl/adapters/miles/run_flags.py`). In that case `--actor-num-nodes`, `--actor-num-gpus-per-node`, `--num-gpus-per-node`, `--rollout-num-gpus`, `--update-weight-transfer-mode`, `--yeto-placement-map` and the switches `--colocate`, `--offload-train`, `--no-offload-train` do not enter the contract; the contract also records `per_island_placement: true`.
2. A run with unified placement does not add this flag. The placement parameters still enter the contract, and the check "all islands have the same GPU split" stays.
3. `--rollout-num-gpus-per-engine` always enters the contract: it decides the inference engine TP, it may affect sampling numerics, and it has no per-island entry.
Tests lock this: with unified placement, different inference card counts → contracts differ; with per-island placement, a difference only in placement parameters → contracts equal; with per-island placement, different engine card counts → contracts differ. The contract value changes as a whole (new field `per_island_placement`). The schema name is unified as v6 after the P3 merge (P4-Q6, merged in s23-final). v6 contains two changes at once: removing the eval flags (P3) and removing the placement parameters under per-island placement (this change). The combination test `test_island_tasks_and_mixed_placement_combine_per_island` locks that when both are given, the island tasks and placements differ per island and the algorithm hash is the same. At present the full-parameter preset requires legacy scheduling and fixed-partition, so a mixed run cannot reach this code yet. This change prepares for the day it is opened.

## Relation to P0-S (policy-age limit N>0)

When `--rl-max-policy-age N>0`, a fixed-partition island automatically uses Miles fully_async (rollout overlaps training), and a colocated island uses `--partial-rollout` (resume at the cutoff). In a mixed run it is expected that the two islands differ; this is not an error. The test `test_policy_age_overlap_differs_by_island_placement` locks this.

## Records (3.1)

- `run_manifest.json` adds `island_placements`: one row per island, with `placement`, `standby_gpus`, `use_gpus_per_node` and `algorithm_sha256`. Each dry-run island entry gets the same `placement` row.
- Per-island round time (generic record, not split by placement): at the end of each round the driver emits the `rl_round_wall` event (`rollout_id`, `placement`, `round_seconds`, `rollout_seconds`). It is a bookkeeping event like `rl_round_trained`; the R0 comparison test excludes it in the same way. It is emitted only for a round that completes successfully: when the round raises an error (including recovery and cancellation) it is not emitted, and the round time is derived from other events.
- `rl_driver_start` already has a `placement` field (verified in 0.5: emitted in `handshake()` of `driver.py`).

## Risks

- Strict is outside the scope of this change (Q3 decided) and is not tested. Per-island round time is a generic record.
- The launcher has more than 10 places that read `rl_placement` globally. Missing one would give an island the wrong resource shape. An AST static check locks the read points, and dry-run tests assert the command of each island.
