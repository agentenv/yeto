# Spec Delta

## ADDED Requirements

### Requirement: Choose the placement per island
The launcher SHALL allow each island to choose colocated or fixed-partition placement independently, and SHALL compute the resource shape and the island command by the placement of that island. When no per-island value is given, all islands MUST use the global `--rl-placement`, and behavior is the same as today.

#### Scenario: Two islands with different placement
- **WHEN** island 0 is set to colocated and island 1 is set to fixed-partition, and the run starts under cross-island elasticity (`--rl-island-scheduling elastic`)
- **THEN** the command of island 0 does not contain `--rl-placement fixed-partition`, the command of island 1 contains this argument and its GPU partition, and the cross-island sync handshake does not fail because of different placement

#### Scenario: No per-island value given
- **WHEN** only the global `--rl-placement fixed-partition` is given
- **THEN** all islands start with fixed-partition

### Requirement: Reject mixed placement at launch
When any island is colocated, the launcher MUST reject `--rl-elastic` and `--rl-overlap-eval` before cards are opened. Under strict (`--rl-island-scheduling legacy`, the default), the launcher MUST refuse to start when the placements differ and MUST explain the reason. Under cross-island elasticity, mixed placement MUST be allowed.

#### Scenario: Mixed placement plus the intra-island elasticity controller
- **WHEN** island 0 is colocated and `--rl-elastic` is also given
- **THEN** the launch fails, the error states the colocated island number, and no cloud resource is created

#### Scenario: Mixed placement under strict
- **WHEN** two islands have different placement, and `--rl-island-scheduling` is `legacy` (the default)
- **THEN** the launch fails, the error states that strict requires the same placement for all islands and suggests switching to `--rl-island-scheduling elastic`, and no cloud resource is created

#### Scenario: Mixed placement under cross-island elasticity
- **WHEN** two islands have different placement, and `--rl-island-scheduling elastic` is given
- **THEN** the launch passes validation

### Requirement: Record the placement of each island
The run manifest SHALL record the placement of each island. The `rl_driver_start` event of each island SHALL contain a `placement` field.

#### Scenario: Events can be told apart
- **WHEN** a mixed placement run completes its start
- **THEN** the `placement` fields of the `rl_driver_start` events of the two islands are colocated and fixed-partition respectively
