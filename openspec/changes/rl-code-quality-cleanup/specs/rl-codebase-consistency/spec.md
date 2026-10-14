# Spec Delta

## Purpose
This spec sets consistency rules for the yeto codebase. Run logs are not tracked in the repository. Each module has one reference path. There is only one Miles engine. The sync bridge, the merge weight, the staleness limit and the resume cut each have one definition. The algorithm, the reward and the config each have one source. Every step has zero change in training behavior.

## ADDED Requirements

### Requirement: Run logs are not tracked
The repository MUST NOT track run-log files (the `run-logs/` directory, `*.log`, `*.stderr`, `*.stdout`, `*metrics*.json`, tape or events in `*.jsonl` form). `.gitignore` MUST cover these patterns and MUST NOT contain allowlist lines that let them through. Logs that must be kept as evidence SHALL be stored in an evidence directory outside the repository. The repository keeps only a note file that points to the path.

#### Scenario: No tracked logs
- **WHEN** a `git ls-files` scan with the log patterns runs on the repository
- **THEN** the number of matching files is 0

#### Scenario: New logs are not added
- **WHEN** a `.log` file is created under `openspec/changes/<x>/run-logs/` and `git add -A` runs
- **THEN** the file does not enter the staging area

### Requirement: A module has one official reference path
Every module in the `yeto` package SHALL be imported only through its official path. Transitional forwarding stubs MUST NOT exist. The static reference check MUST cover yeto/, tests/, scripts/ and tools/. The check MUST fail on an old path and list the files. The command-line entry `python -m yeto.rl.learner` MUST stay usable and point to the official module.

#### Scenario: Old paths cannot be imported
- **WHEN** any file imports one of the 18 old forwarding modules or `yeto.rl.engine.miles_adapter.*`
- **THEN** the reference check test fails and lists the file and the old path

#### Scenario: The entry command does not change
- **WHEN** it starts with `python -m yeto.rl.learner <argv>`
- **THEN** the behavior is the same as `python -m yeto.rl.adapters.miles.island_entry <argv>`

### Requirement: Only one Miles engine
The Miles backend SHALL have only one engine, ports. The legacy engine module, its pin, its mirror and its setup path MUST be deleted. SAO streaming, dense-full, DeepSeek V4 expert full-parameter and compaction RL MUST be able to run on the ports engine. The engine selection logic MUST NOT have any "cannot use ports" reject item. Code MUST NOT use runtime code-object probing to tell engines or writer versions apart.

#### Scenario: Engine selection has no reject item
- **WHEN** a run starts with the SAO sync preset, a `--sao*` flag, dense-full or a DSV4 recipe
- **THEN** engine selection returns ports and raises no reject error

#### Scenario: The legacy path is gone
- **WHEN** the repository is searched for imports of `adapters.miles.legacy` and for references to the legacy pin
- **THEN** the number of hits is 0, and the mirror list contains no legacy pin

#### Scenario: Capability regression
- **WHEN** SAO streaming, dense-full and DSV4 expert full-parameter each run once with the minimum config
- **THEN** the output matches the reference output recorded before the deletion (token counts, step counts and loss curves are within the set tolerance)

### Requirement: Only one sync bridge
The strict and elastic sync bridges SHALL each have only one implementation, and the delta math SHALL be written once. The decoupled boundary loop SHALL exist in one place. The critic sync SHALL have one path. Before the merge, a parity test of the two old implementations MUST exist. After the merge, this test MUST pass on the new implementation. The weight, staleness and closing rules of the three sync modes (strict, decoupled, elastic) MUST be documented in one place.

#### Scenario: Parity before and after the merge
- **WHEN** the same set of island deltas is fed in strict mode and in elastic mode to the old bridge and to the new bridge
- **THEN** the merge results are equal element by element

The rule table of the three sync modes (identical line by line to the module docstring of `yeto/rl/engine/bridges.py`; `tests/test_rl_sync_mode_table.py` checks it):

| mode | merge weight | staleness | finish |
|---|---|---|---|
| strict (``StrictAvgSync``, ``DualStrictAvgSync`` for the critic) | syncer ``--learner-weight equal`` (launcher RL default); ``tokens2-over-steps`` = ``merge::learner_weight`` when chosen | ``--max-base-lag 0``: every island trains from the exact global version v and pushes against v; quorum = all islands | stop when the committed version reaches ``global_rounds``; ``finish`` checks the final policy equals the committed one |
| decoupled (``DecoupledSync``) | same syncer flags as strict (``equal``, ``--max-base-lag 0``), per fragment | fragments rotate with ``--pipeline`` fragments in flight; each fragment is merged against its own fragment version | stop when the syncer is finalizing or a submission reaches ``total_fragment_steps``; ``learner_budget_steps`` consolidates into one final cut |
| elastic (``ElasticAvgSync``) | ``merge_weight`` = ``merge::learner_weight`` (tokens^2 / steps); a catch-up island has weight 0 on its join version; a late delta is carried with ``carry_gamma^lag`` | a delta with lag > ``DEFAULT_MAX_CARRY_LAG`` (2) is rejected (``carry_lag_exceeded``); samples at most ``MAX_SAMPLE_OUTER_LAG`` (2) outer steps stale; step at quorum ``theta`` (0.75) of capacity, at least ``q_min`` (1) | the syncer answers FINISHED after ``total_steps``, and keeps answering for ``final_grace_s`` (default: the soft deadline) |

#### Scenario: No second implementation
- **WHEN** `StrictRlBridge`, `ElasticRlBridge`, `TwoRoleStrictAvg` and the PPO role-lane code of `local_learner` are searched
- **THEN** the number of hits in production code is 0

### Requirement: Only one full-parameter sync
Full-parameter sync SHALL have only one implementation: the ports `FullParameterSync` plus one single-role streaming sync. Dense, SAO and DSV4 expert-full MUST reuse it through config only, and MUST NOT each write their own transport or coordination code. With multiple roles, staging MUST be paired and apply MUST be paired. When any role fails, all roles MUST close together with the same reason code. The dense strict sweep MUST be a restricted config of this implementation (1 role, H=1, pipeline depth 1, whole-round sweep). The weights MUST stay in fragments, and the whole model MUST NOT be assembled in one process. Every publish MUST use the policy token `yeto:<version>:<weight hash>` and MUST pass the publish check that verifies each inference engine one by one.

#### Scenario: Dense is no longer a separate implementation
- **WHEN** `MilesFullParameterDenseSync`, `DenseSweepWire` and `dense_sweep_session_contract_hash` are searched
- **THEN** the number of hits in production code is 0

#### Scenario: One role fails
- **WHEN** in a two-role config the critic role fails at commit
- **THEN** both the actor role and the critic role are closed with the same reason code, and one `rl_sync_closed` event is written

#### Scenario: A start from version 0 is not required
- **WHEN** the authoritative fragments of the syncer are already at version v>0
- **THEN** the sync session starts from v and raises no error

### Requirement: Merge weight and staleness limit are defined once
The learner merge weight formula SHALL be defined once, in the Rust merge module. The elastic module MUST call it. The Python side MUST NOT keep an independent copy, or the copy MUST be locked by a parity test. The maximum outer-step lag and the maximum carry lag SHALL have automated parity checks between Rust and Python. The test MUST fail when the values differ. The golden fixture MUST contain a case that is rejected because the lag exceeds the limit.

#### Scenario: One side changes a value
- **WHEN** only Rust or only Python changes the maximum lag value or the weight formula
- **THEN** the parity test fails and prints the values of both sides

#### Scenario: Over-limit is rejected
- **WHEN** the golden fixture submits a fragment with lag=3
- **THEN** both sides decide `delta_rejected carry_lag_exceeded`

### Requirement: Only one resume abstraction
The system SHALL have only one definition of the "consistent cut" and only one resume controller. The three resume mechanisms, `--rl-elastic`, cross-island elasticity and `ResumeController`, MUST converge to it (the legacy `--spot` path is deleted together with the legacy engine). Checkpoint storage SHALL be one interface with multiple backends. The save path SHALL be only one.

#### Scenario: Three triggers, one cut
- **WHEN** a save is triggered separately by a preemption notice, a periodic round_cut and a manual resume
- **THEN** the written checkpoints have the same format, and after resume the training step count and the island state are consistent

#### Scenario: Storage backends are replaceable
- **WHEN** the checkpoint backend changes from a local directory to a bucket mount or a Modal volume
- **THEN** the save and resume code does not change, and only the backend config changes

### Requirement: Only one source for the algorithm implementation
GRPO group normalization SHALL have only one implementation in our code (the copies that come with upstream Miles and verl are locked by parity tests). The reference loss SHALL have one copy. policy-age SHALL be one module plus backend adapters. The staleness check SHALL be in one place. The separate PPOTrainer SHALL be merged into the main algorithm package. Capability declarations MUST be consistent between the Miles table, the GRPO knob table and the verl binding table, and MUST NOT be hand-copied mirrors.

#### Scenario: Group normalization is equivalent
- **WHEN** the same set of reward inputs goes through all callers to compute advantages, once with `std_normalize` on and once with it off
- **THEN** the results are equal element by element (the fp32 reference is within tolerance)

#### Scenario: Declaration drift
- **WHEN** a knob in the Miles declaration table changes and the other tables do not
- **THEN** the consistency test fails and names the drifted knob

### Requirement: Reward, contract and pin have one source
Reward definitions SHALL be registered in one place only, the reward registry. Reward trust and finish SHALL have one path. There SHALL be one contract/identity module. The Miles/SGLang pin and the fork lineage SHALL be defined in one pin module only, and all other users MUST derive them from it. The Codex agent SHALL have only one main entry.

#### Scenario: Add a reward
- **WHEN** a new reward function is added
- **THEN** it needs to be registered once in the registry, and both the Miles backend and the verl backend can fetch it by name

#### Scenario: Lineage derivation
- **WHEN** a fork commit is appended in the pin module
- **THEN** the loss variants, the critic fork and the Miles entry contain that commit without any change

### Requirement: Config has one schema
RL flags SHALL have only one schema. The CLI arguments, the launcher pass-through and the island entry arguments MUST be generated or checked from it, and MUST NOT each be hand-written. The algorithm hash and the island contract hash SHALL be nested. W&B, exporter, finalisation and worker env scrubber SHALL each have only one module. The island lease default value SHALL be the same in the syncer, the CLI help and the launcher.

#### Scenario: Add a flag
- **WHEN** a `--rl-*` flag is added to the schema
- **THEN** the CLI, the launcher and the island entry recognize and pass it through without each adding it

#### Scenario: Run the syncer directly
- **WHEN** the syncer starts directly without `--island-lease-s`
- **THEN** the lease seconds equal the value that the launcher passes by default

### Requirement: Zero training behavior change in every step
Every merged PR MUST keep the set of passing tests in the local safe test set from shrinking. A consolidation that touches a numeric path MUST first have a parity test. When the parity fails, the work MUST stop and record the difference. It MUST NOT change values silently.

#### Scenario: Parity mismatch
- **WHEN** two implementations to be merged give different results for the same input
- **THEN** the PR is not merged, and the source of the difference is recorded in design.md
