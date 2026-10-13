# Spec Delta

## ADDED Requirements

### Requirement: Reward results carry no key signature
Reward results (TB2 outcome and secrlenv outcome) MUST NOT carry or require an HMAC signature. The reward path MUST NOT read `TBENCH_REWARD_HMAC_KEY`, `TBENCH_REWARD_HMAC_KEY_FILE`, or `SECRLENV_REWARD_HMAC_KEY`. It SHALL run normally when these variables are absent.

#### Scenario: Reward is computed without a reward key
- **WHEN** the environment has no reward key variable and the harness produces a valid outcome
- **THEN** the reward function returns the reward value of that outcome and raises no exception

#### Scenario: An old mac field is ignored
- **WHEN** the metadata carries an old `tbench_trusted_outcome_hmac` field
- **THEN** the reward function ignores that field, and the result is the same as without it

### Requirement: Reward results keep structure and range checks
The reward path SHALL keep checking the outcome schema, field set, status enum, and reward value range. An invalid outcome MUST be rejected.

#### Scenario: An out-of-range reward is rejected
- **WHEN** the reward value of an outcome is outside the allowed range
- **THEN** the reward function rejects the outcome and reports an error

### Requirement: Trajectory evidence hash has no signature
Newly written trajectory evidence SHALL use schema 3, and its outcome hash MUST NOT include a mac. The evidence MUST keep `reward_contract_hash`.

#### Scenario: New evidence is schema 3
- **WHEN** a batch of trajectory evidence is written
- **THEN** the schema_version of the evidence is 3, and the hash input has no mac

### Requirement: Compatibility hashes and island authentication are unchanged
The composition and values of `BackendIdentity`, `session_contract_hash`, `lr_schedule_sha256`, the algorithm hash, `training_contract_hash`, and `reward_contract_hash` MUST NOT change. The island HELLO HMAC and the per-frame HMAC in elastic mode MUST stay.

#### Scenario: Islands with different backends are still refused a merge
- **WHEN** a Miles island and a verl island connect to the same syncer with the same layout
- **THEN** their session_contract_hash values differ, and the syncer rejects the island that connects later

#### Scenario: A wrong island key is still rejected
- **WHEN** an island sends HELLO with a wrong `YETO_ISLAND_HMAC_KEY`
- **THEN** the syncer rejects the connection
