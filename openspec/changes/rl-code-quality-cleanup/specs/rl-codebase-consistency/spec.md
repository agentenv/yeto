# Spec Delta

## Purpose
Constrain the consistency of the yeto codebase. Run logs do not enter git. Each module has only one import path. There is only one Miles engine. The sync bridge, the merge weight, the staleness limit and the resume cut point each have only one definition. Algorithms, rewards and config each have only one source. Every step causes zero change in training behavior.

## ADDED Requirements

### Requirement: Run logs do not enter git
The repo MUST NOT track run-log-type files (the `run-logs/` directory, `*.log`, `*.stderr`, `*.stdout`, `*metrics*.json`, and tapes or events in `*.jsonl` form). `.gitignore` MUST cover these patterns and MUST NOT contain an allowlist that lets them in. A log that must be kept as evidence SHALL be put in an evidence directory outside the repo. The repo keeps only a note file that points to the path.

#### Scenario: Tracked logs are zero
- **WHEN** a `git ls-files` scan with the log patterns runs on the repo
- **THEN** the number of matching files is 0

#### Scenario: New logs are not added
- **WHEN** a `.log` file is created under `openspec/changes/<x>/run-logs/` and `git add -A` runs
- **THEN** the file does not enter the staging area

### Requirement: A module has only one formal import path
Each module in the `yeto` package SHALL be importable only through its formal path. Transition forwarding stubs MUST NOT exist. The static reference check MUST cover yeto/, tests/, scripts/ and tools/. When it finds an old path, it MUST fail and list the files. The command-line entry `python -m yeto.rl.learner` MUST stay usable and MUST point to the formal module.

#### Scenario: Old paths cannot be imported
- **WHEN** any file imports one of the 18 old forwarding modules or `yeto.rl.engine.miles_adapter.*`
- **THEN** the reference check test fails and lists the file and the old path

#### Scenario: The entry command is unchanged
- **WHEN** it is started as `python -m yeto.rl.learner <argv>`
- **THEN** the behavior is the same as `python -m yeto.rl.adapters.miles.island_entry <argv>`

### Requirement: Only one Miles engine
The Miles backend SHALL have only one engine, ports. The legacy engine module and its pin, image and setup path MUST be deleted. SAO streaming, dense-full, DeepSeek V4 expert full-parameter and compaction RL MUST run on the ports engine. The engine selection logic MUST NOT have any "cannot go to ports" reject entry. The code MUST NOT use a runtime code-object probe to tell engines or writer versions apart.

#### Scenario: Engine selection has no reject entry
- **WHEN** a run starts with the SAO sync preset, a `--sao*` flag, dense-full or a DSV4 recipe
- **THEN** engine selection returns ports and does not raise a reject error

#### Scenario: The legacy path is gone
- **WHEN** the repo is searched for imports of `adapters.miles.legacy` and for references to the legacy pin
- **THEN** there are 0 hits, and the image manifest does not contain the legacy pin

#### Scenario: Capability regression
- **WHEN** SAO streaming, dense-full and DSV4 expert full-parameter each run once with a minimal config
- **THEN** the output matches the reference output recorded before the deletion (token count, step count and loss curve are within the set tolerance)

### Requirement: Only one sync bridge
The strict and the elastic sync bridge SHALL each have only one implementation, and the delta math is written only once. The decoupled boundary loop exists in only one place. The critic sync has only one path. Before the merge, there MUST be a cross-check test of the two old implementations. After the merge, that test MUST pass on the new implementation. The weight, staleness and closing rules of the three sync modes (strict, decoupled, elastic) MUST be documented in one place.

#### Scenario: Cross-check before and after the merge
- **WHEN** the same set of island deltas is fed to the old bridge and the new bridge in strict mode and in elastic mode
- **THEN** the merge results are identical element by element

The rule table of the three sync modes (identical line by line to the module docstring of `yeto/rl/engine/bridges.py`, checked by `tests/test_rl_sync_mode_table.py`):

| mode | merge weight | staleness | finish |
|---|---|---|---|
| strict (``StrictAvgSync``, ``DualStrictAvgSync`` for the critic) | syncer ``--learner-weight equal`` (launcher RL default); ``tokens2-over-steps`` = ``merge::learner_weight`` when chosen | ``--max-base-lag 0``: every island trains from the exact global version v and pushes against v; quorum = all islands | stop when the committed version reaches ``global_rounds``; ``finish`` checks the final policy equals the committed one |
| decoupled (``DecoupledSync``) | same syncer flags as strict (``equal``, ``--max-base-lag 0``), per fragment | fragments rotate with ``--pipeline`` fragments in flight; each fragment is merged against its own fragment version | stop when the syncer is finalizing or a submission reaches ``total_fragment_steps``; ``learner_budget_steps`` consolidates into one final cut |
| island pool (``PoolAvgSync``) | syncer ``--learner-weight`` (``tokens2-over-steps`` by default here); a late delta is carried with ``carry_gamma^lag`` | a delta whose base lags the round by more than ``--rl-sync-max-base-lag`` is rejected (``carry_lag_exceeded``); a round merges at quorum ``theta`` of capacity (at least ``q_min``) or at the soft deadline; islands hold leases and re-JOIN after an expiry | the syncer sends the final cut after ``total_steps`` and waits ``final_grace_s`` for the islands still connected |

#### Scenario: No second implementation
- **WHEN** the repo is searched for `StrictRlBridge`, `ElasticRlBridge`, `TwoRoleStrictAvg` and the PPO role-lane code of `local_learner`
- **THEN** there are 0 hits in production code

### Requirement: Only one full-parameter sync
Full-parameter sync SHALL have only one implementation: the ports `FullParameterSync` plus one single-role streaming sync. Dense, SAO and DSV4 expert-full MUST reuse it through config only. They MUST NOT each write their own transport or coordination code. With several roles, staging MUST be done in pairs and applying MUST be done in pairs. When any role fails, all roles MUST close together with the same reason code. The dense strict sweep MUST be a restricted config of this implementation (1 role, H=1, pipeline depth 1, whole-round sweep). Weights MUST stay in chunks. The code MUST NOT assemble the whole model in a single process. Each publish MUST use the policy token `yeto:<version>:<weight hash>` and MUST go through a publish check that is verified engine by engine.

#### Scenario: Dense no longer has its own implementation
- **WHEN** the repo is searched for `MilesFullParameterDenseSync`, `DenseSweepWire` and `dense_sweep_session_contract_hash`
- **THEN** there are 0 hits in production code

#### Scenario: One role fails
- **WHEN** in a two-role config the critic role fails at commit
- **THEN** both the actor role and the critic role are closed with the same reason code, and one `rl_sync_closed` event is written

#### Scenario: Starting from version 0 is not required
- **WHEN** the authoritative shard of the syncer is already at version v>0
- **THEN** the sync session starts from v and raises no error

### Requirement: The merge weight and the staleness limit are defined once
The learner merge weight formula SHALL be defined only once, in the Rust merge module. The elastic module MUST call it. The Python side MUST NOT keep an independent copy, or the copy MUST be locked by a cross-check test. The maximum outer-step lag and the maximum carry lag SHALL have automated cross-checks on both the Rust side and the Python side. When the values differ, the test MUST fail. The golden fixture MUST include a case where a lag above the limit is rejected.

#### Scenario: One side changes a value
- **WHEN** only Rust or only Python changes the maximum lag value or the weight formula
- **THEN** the cross-check test fails and prints the values of both sides

#### Scenario: Over the limit is rejected
- **WHEN** the golden fixture submits a fragment with lag=3
- **THEN** both sides decide `delta_rejected carry_lag_exceeded`

### Requirement: Only one resume abstraction
The system SHALL have only one definition of a "consistent cut point" and one resume controller. The three resume mechanisms `--rl-elastic`, cross-island elastic and `ResumeController` MUST converge to it (the legacy `--spot` path is deleted with the legacy engine). Checkpoint storage SHALL be one interface with several backends. The save path SHALL be only one.

#### Scenario: Three triggers, one cut point
- **WHEN** a save is triggered separately by a preemption notice, a periodic round_cut and a manual resume
- **THEN** the written checkpoints have the same format, and the training step count and island state are the same after resume

#### Scenario: Storage backend is replaceable
- **WHEN** the checkpoint backend changes from a local directory to a bucket mount or a Modal volume
- **THEN** the save and resume code is unchanged, and only the backend config changes

### Requirement: Algorithm implementation has only one source
GRPO group normalization SHALL have only one implementation in our code (the copies in upstream Miles and verl are locked by a cross-check test). The reference loss has only one copy. Policy-age is one module with backend adapters. The staleness check is in one place only. The standalone PPOTrainer is merged into the main algorithm package. The capability declarations MUST be consistent across the Miles table, the GRPO knob table and the verl binding table. They MUST NOT be hand-copied mirrors.

#### Scenario: Group normalization is equivalent
- **WHEN** the same group of rewards is given to all callers to compute advantages, once with `std_normalize` on and once with it off
- **THEN** the results are identical element by element (the fp32 reference is within tolerance)

#### Scenario: Declaration drift
- **WHEN** one knob in the Miles declaration table is changed and the other tables are not
- **THEN** the consistency test fails and names the drifting knob

### Requirement: Rewards, contracts and pins have only one source
A reward definition SHALL be registered in only one place, the reward registry. Reward trust and finish have only one path. There is only one contract/identity module. The Miles/SGLang pins and the fork lineage are defined only in one pin module, and other users MUST derive from it. The Codex agent SHALL have only one main entry.

#### Scenario: Add a reward
- **WHEN** a new reward function is added
- **THEN** it needs to be registered only once in the registry, and both the Miles and the verl backend can get it by name

#### Scenario: Lineage derivation
- **WHEN** one fork commit is appended in the pin module
- **THEN** the loss variant, the critic fork and the Miles entry contain that commit without any change

### Requirement: Config has only one schema
RL flags SHALL have only one schema. CLI arguments, launcher pass-through and island entry arguments MUST be generated or checked from it. They MUST NOT each hand-write a copy. The algorithm hash and the island contract hash SHALL be nested. W&B, exporter, finalisation and worker env scrubber SHALL each have only one module. The island lease default SHALL be the same in the syncer, the CLI help and the launcher.

#### Scenario: Add a flag
- **WHEN** one `--rl-*` flag is added to the schema
- **THEN** the CLI, the launcher and the island entry recognize and pass it through without separate additions

#### Scenario: Run the syncer directly
- **WHEN** the syncer starts directly without `--island-lease-s`
- **THEN** the lease seconds equal the value that the launcher passes by default

### Requirement: Zero training behavior change at every step
Each merged PR MUST keep the set of passing tests in the local safe test set from shrinking. A convergence that touches a numeric path MUST first have a cross-check test. When the cross-check does not match, the work MUST stop and record the difference. It MUST NOT silently change a value.

#### Scenario: Cross-check mismatch
- **WHEN** two implementations to be merged give different results for the same input
- **THEN** the PR is not merged, and the source of the difference is recorded in design.md
