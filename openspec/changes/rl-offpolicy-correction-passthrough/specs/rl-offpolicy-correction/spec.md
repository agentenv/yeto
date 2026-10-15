# Spec Delta

## Purpose

Correct off-policy samples with decoupled PPO on the backends, limit sample age at admission and consumption, and record every drop. Phase 1 warns on mismatches and does not refuse.

## ADDED Requirements

### Requirement: Correction settings are explicit and passed through
The neutral `CorrectionSpec` SHALL carry `method`, `level`, `lower`, `upper`, `veto`, `batch_normalize`, and `ratio_mask`. When `method` is not `none`, `lower` and `upper` SHALL be given. `ratio_mask` SHALL default to `none`. The adapters SHALL map these fields to backend settings and SHALL NOT compute importance weights in yeto.

#### Scenario: IcePop on Miles
- **WHEN** the spec has `ratio_mask=icepop`, `level=token`, `lower=0.5`, `upper=5.0`
- **THEN** the Miles flags select the IcePop function with the range 0.5 to 5.0

#### Scenario: IcePop on verl
- **WHEN** the same spec runs on verl
- **THEN** the verl config has `rollout_is=token` and `rollout_is_threshold="0.5_5.0"`

### Requirement: Decoupled PPO objective
Every response token SHALL enter the loss. The importance weight SHALL be `π_prox / π_behav`, truncated at `upper`, where `π_behav` is the per-token rollout logprob. The PPO clip anchor SHALL be `π_prox`, recomputed by the trainer. On Miles the adapter SHALL set `--use-tis --tis-clip <upper>` and SHALL refuse `--use-rollout-logprobs` and `--skip-actor-forward-only`. On verl the adapter SHALL write `bypass_mode=False`, `rollout_is=token`, and `rollout_is_threshold=<upper>` explicitly. The manifest and tape SHALL record which weights the backend uses for `π_prox`.

#### Scenario: verl yaml defaults overridden
- **WHEN** a verl island starts from the `fully_async` yaml that sets `bypass_mode: True`
- **THEN** the effective config has `bypass_mode=False`, `rollout_is=token`, and `rollout_is_threshold` equal to `upper`

#### Scenario: Forbidden Miles flag
- **WHEN** a Miles island config contains `--use-rollout-logprobs` or `--skip-actor-forward-only`
- **THEN** the adapter refuses the config and names the flag

#### Scenario: Proximal anchor differs
- **WHEN** a Miles island and a verl island run in one job
- **THEN** each tape records its `prox_anchor`, and the launcher emits one `correction_config_warning` that lists both values

### Requirement: Version gap from the start version
The version gap of a trajectory SHALL equal the current trainer version minus the trajectory start version. An implementation that uses the oldest token version SHALL give the same value.

#### Scenario: Continued trajectory
- **WHEN** a trajectory starts at version 3, continues across updates to versions 4 and 5, and the trainer is at version 6
- **THEN** its gap is 3

### Requirement: Three version limits
Admission SHALL use existing backend mechanisms: Miles buffer capacity with `--max-weight-staleness`, and verl `async_training.staleness_threshold`. In-flight generation SHALL continue across weight updates: Miles partial rollout with `mask_offpolicy_in_partial_rollout` off, and verl `partial_rollout=True`. At consumption, a group with gap over `A_hard` SHALL be dropped whole on both backends. `A_hard` SHALL be the existing `max_policy_age`. The adapters SHALL accept `A_hard ≥ 2`. No token SHALL be masked because of its version.

#### Scenario: Miles partial-rollout mask refused
- **WHEN** a Miles island config enables `mask_offpolicy_in_partial_rollout`
- **THEN** the adapter refuses the config

#### Scenario: verl drop at consumption
- **WHEN** a verl round collects a group with gap `A_hard + 1`
- **THEN** the yeto verl adapter drops the whole group with reason `group_dropped`

#### Scenario: A_hard of 2 accepted
- **WHEN** a run sets `max_policy_age=2`
- **THEN** both adapters accept it and map it to their backend limits

#### Scenario: Defaults keep behavior
- **WHEN** a run does not set new fields
- **THEN** `max_policy_age` keeps its current default and meaning

### Requirement: Phase 1 warns and does not refuse
In phase 1, a correction setting that a backend cannot map, or a correction group that differs between islands, SHALL produce a `correction_config_warning` event and a log warning. It SHALL NOT refuse the launch. The correction group SHALL NOT change `algorithm_sha256`. Refusals of forbidden flags in other requirements are not affected.

#### Scenario: TIS lower bound on verl
- **WHEN** a verl island gets `method=tis` with `lower=0.1`
- **THEN** the launch continues and the event names the field `lower`, the backend verl, and the value it ran with

#### Scenario: Islands differ
- **WHEN** two islands have different `upper` values
- **THEN** each island starts, and the launcher emits one warning that lists both values

#### Scenario: Hash unchanged
- **WHEN** only the correction group changes between two specs
- **THEN** both specs have the same `algorithm_sha256`

### Requirement: Correction settings are recorded
The run manifest and each island tape SHALL contain the full correction group and the backend values it mapped to.

#### Scenario: After a run
- **WHEN** a run ends
- **THEN** the manifest and every island tape contain the correction group, and a reader can compare them without the launch logs

### Requirement: Drop and truncation monitoring
Each training round SHALL emit `offpolicy_gate` with counts per reason, a fixed-bucket response length histogram of dropped trajectories, and the ratio of truncated importance weights by turn position. When the backend gives only trajectory-level versions, the event SHALL mark the field `granularity=trajectory`.

#### Scenario: Round with drops
- **WHEN** a round drops two groups of lengths 300 and 5000 tokens
- **THEN** the `group_dropped` histogram has one count in each bucket that holds these lengths

#### Scenario: verl granularity
- **WHEN** a verl island emits `offpolicy_gate`
- **THEN** the truncation-by-turn field has `granularity=trajectory`

### Requirement: Both backends use their own async mode
The correction settings and version limits SHALL run on Miles `fully_async` and on verl `fully_async_policy`. yeto SHALL NOT add its own async loop for either backend.

#### Scenario: verl island with A_hard 2
- **WHEN** a verl island runs with `max_policy_age=2` and decoupled PPO
- **THEN** it runs through verl `FullyAsyncRollouter` and `FullyAsyncTrainer`, and emits `offpolicy_gate` with the same reasons as a Miles island
