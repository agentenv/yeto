# Spec Delta

## ADDED Requirements

### Requirement: Choose the placement per island
The launcher SHALL allow each island to choose colocated or fixed-partition placement independently, and SHALL compute the resource shape and the island command from that island's placement. When no per-island value is given, all islands MUST use the global `--rl-placement`, and the learner command is byte-for-byte unchanged (the run manifest and the dry-run output add a placement record for each island).

#### Scenario: Two islands with different placements
- **WHEN** island 0 is set to colocated and island 1 to fixed-partition, and the run starts under cross-island elastic (`--rl-island-scheduling elastic`)
- **THEN** the command of island 0 does not contain `--rl-placement fixed-partition`, the command of island 1 contains that argument and its GPU partition, and the cross-island sync handshake does not fail because the placements differ

#### Scenario: No per-island value given
- **WHEN** only the global `--rl-placement fixed-partition` is given
- **THEN** all islands start as fixed-partition

### Requirement: Startup refusal of mixed placement
In a mixed run where any island is colocated, the launcher MUST refuse `--rl-elastic` and `--rl-overlap-eval` before cards are launched and MUST name the colocated island number (when all islands use the same placement, the existing checks apply). Explicitly giving a colocated island standby cards MUST be refused. Under strict (`--rl-island-scheduling legacy`, the default), different placements MUST be refused at startup with the reason stated. Under cross-island elastic, mixed placement MUST be allowed.

#### Scenario: Mixed placement with the intra-island elastic controller
- **WHEN** island 0 is colocated, island 1 is fixed-partition, and `--rl-elastic` is also given
- **THEN** startup fails, the error names the colocated island number, and no cloud resource is created

#### Scenario: Mixed placement under strict
- **WHEN** the two islands have different placements and `--rl-island-scheduling` is `legacy` (the default)
- **THEN** startup fails, the error states that strict requires one placement for all islands and hints to use `--rl-island-scheduling elastic` instead; no cloud resource is created

#### Scenario: Mixed placement under cross-island elastic
- **WHEN** the two islands have different placements and `--rl-island-scheduling elastic` is given
- **THEN** startup passes the checks

### Requirement: Record each island's placement
The run manifest SHALL record the placement and the algorithm hash of each island. The `rl_driver_start` event of each island SHALL contain the `placement` field. For each successfully completed round, each island SHALL emit the `rl_round_wall` event, which records the placement and the round time.

#### Scenario: Events can be told apart
- **WHEN** a mixed placement run finishes starting
- **THEN** the `placement` fields of the `rl_driver_start` events of the two islands are colocated and fixed-partition respectively

### Requirement: Placement does not enter the cross-island comparison
The placement MUST NOT enter the algorithm hash. When per-island placement values exist, the placement MUST NOT enter the full-parameter training contract hash (except the inference engine card count). In a mixed run, all islands SHALL receive the same algorithm hash.

#### Scenario: Same algorithm hash on both islands
- **WHEN** island 0 is colocated, island 1 is fixed-partition, and the run starts under cross-island elastic
- **THEN** `--rl-expected-algorithm-sha256` is the same in the commands of both islands

#### Scenario: Full-parameter contract does not contain the placement under per-island placement
- **WHEN** the run starts with per-island placement values, and the two islands differ only in placement parameters (trainer shape, inference card count, colocate/offload switches)
- **THEN** the full-parameter training contract hashes of the two islands are the same

#### Scenario: GPU split check is kept under unified placement
- **WHEN** there is no per-island placement value and the two islands have different inference card counts
- **THEN** the full-parameter training contract hashes of the two islands differ

#### Scenario: Engine card count always enters the contract
- **WHEN** the two islands have different `--rollout-num-gpus-per-engine`
- **THEN** the full-parameter training contract hashes of the two islands differ

### Requirement: Per-island placement behavior under the policy-age limit
When `--rl-max-policy-age` N>0, a fixed-partition island SHALL overlap rollout and training (Miles fully_async), and a colocated island SHALL resume at the cutoff (`--partial-rollout`). In a mixed run it is expected that the two islands differ.

#### Scenario: Mixed run with N=1
- **WHEN** island 0 is colocated, island 1 is fixed-partition, and `--rl-max-policy-age 1`
- **THEN** the Miles arguments of island 1 contain `--fully-async`, and those of island 0 contain `--partial-rollout`
