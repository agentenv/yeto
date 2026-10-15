# Spec Delta

## Purpose

Lower and measure the start cost of a trainer, so that a trainer that starts and stops many times is affordable.

## ADDED Requirements

### Requirement: Light actors do not import the training backend
Ray actors that do not train (worker manager, command actor, inference controller) SHALL NOT import `miles.backends.megatron_utils`. The trainer actor SHALL NOT import `modelopt.torch` until a feature that needs it is used.

#### Scenario: Import of the HTTP utilities
- **WHEN** a process imports `miles.utils.http_utils`
- **THEN** `miles.backends.megatron_utils` is not in `sys.modules`

#### Scenario: Import of the bridge
- **WHEN** a process imports `megatron.bridge` and no quantization feature is configured
- **THEN** `modelopt.torch` is not in `sys.modules`

### Requirement: Setup checks packages without importing them
Island setup SHALL check the versions of `peft` and `sglang` against the image manifest. It SHALL NOT import these packages for the check when the manifest exists. When the manifest is missing, setup SHALL fall back to the import check and SHALL emit an event that names the fallback.

#### Scenario: Manifest present
- **WHEN** the image has `/opt/yeto/image-manifest.json` with matching versions
- **THEN** setup passes the check without a Python import of `peft` or `sglang`

#### Scenario: Version mismatch
- **WHEN** the manifest version differs from the pin
- **THEN** setup fails before Ray starts and names the package and both versions

### Requirement: Model prefetch reports failure
The model prefetch command SHALL be a command that exists in the image. When the prefetch fails, the system SHALL emit `model_prefetch_failed` with the exit code. A prefetch failure SHALL NOT stop the run.

#### Scenario: Prefetch fails
- **WHEN** the prefetch command exits with a non-zero code
- **THEN** the event log has `model_prefetch_failed` with that code, and the run continues

### Requirement: Start phases are recorded
Each trainer start SHALL emit `trainer_start_phase` events with the fixed phase names `submitted`, `container_ready`, `setup_done`, `ray_ready`, `actor_import_done`, `actor_init_done`, `first_step_done`, and one summary event with the total start time.

#### Scenario: Normal start
- **WHEN** a trainer starts and finishes its first optimizer step
- **THEN** all seven phases appear in order, and the summary total equals `first_step_done − submitted`

### Requirement: Sampler snapshot mode is gated
The sampler engine snapshot mode SHALL be refused at launch until the weight-update compatibility check has a recorded pass for the pinned SGLang commit.

#### Scenario: No recorded pass
- **WHEN** a run requests snapshot mode and no pass is recorded for the pinned SGLang commit
- **THEN** launch fails with an error that names the missing check
