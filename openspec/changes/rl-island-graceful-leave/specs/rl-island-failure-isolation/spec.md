# Spec Delta

## Purpose
Constrain the behavior when an island exits on the elastic path: when the island can still run wrap-up code, it leaves the fleet actively and tells the syncer the broad category of the reason. When an island vanishes silently, the lease timeout removes it.

## ADDED Requirements

### Requirement: An exiting island sends LEAVE with a reason code
When the driver of an elastic island exits, it SHALL send LEAVE to the syncer before the process ends. The reason code SHALL be one of the following, by exit reason: 0 FINISHED, 2 STOPPED, 3 PREEMPTED, 4 INFRA_FAILED, 5 TRAINING_FAILED, 6 CONFIG_MISMATCH. The mapping from exception class to code MUST be defined by the mapping table. Code 1 LEASE_EXPIRED is decided only by the syncer itself. An island MUST NOT send it.

#### Scenario: LEAVE after node loss
- **WHEN** the controller reports node loss and the driver raises a RECOVERY_REQUIRED error
- **THEN** the elastic client sends a LEAVE with reason code 4, and the original error propagates as usual

#### Scenario: Training check failure
- **WHEN** the driver raises PublicationError, PolicyIdentityError, or RoundFailedError
- **THEN** the LEAVE reason code sent is 5

#### Scenario: Incompatible with other islands
- **WHEN** the driver raises CapabilityMismatch, BackendIdentityMismatch, or ManifestMismatch
- **THEN** the LEAVE reason code sent is 6

#### Scenario: User stop
- **WHEN** the island receives KeyboardInterrupt or SystemExit, or the user runs `yeto stop-run`
- **THEN** the LEAVE reason code sent is 2

#### Scenario: Spot reclaim
- **WHEN** the cloud sends a spot reclaim notification
- **THEN** the LEAVE reason code sent is 3

#### Scenario: Normal wrap-up
- **WHEN** the island completes the last round as planned and wraps up
- **THEN** the LEAVE reason code sent is 0

### Requirement: A code expresses only the broad category
A reason code MUST be 1 byte and MUST express only the broad category. The exact exception class name and error message SHALL go to the island's event log and MUST NOT enter the wire protocol. When the syncer receives an unknown code, it SHALL record it as 0 and report no error. The syncer log SHALL record the code name.

#### Scenario: Unknown code
- **WHEN** an old syncer receives reason code 3
- **THEN** it records 0, the LEAVE takes effect as usual, and no error is reported

### Requirement: A LEAVE failure is visible
When a LEAVE send fails, the bridge MUST emit an `elastic_leave_failed` event and MUST NOT replace the original error being propagated with that failure.

#### Scenario: The syncer is already disconnected
- **WHEN** the LEAVE send raises a connection error during a failure exit
- **THEN** the event stream contains `elastic_leave_failed`, and the error propagated by the driver is still the original error

### Requirement: Reason codes agree on both sides
The LEAVE reason code constants in Python and Rust MUST have the same values, and a test MUST lock them.

#### Scenario: Cross-check
- **WHEN** the reason code cross-check test runs
- **THEN** FINISHED=0, LEASE_EXPIRED=1, STOPPED=2, PREEMPTED=3, INFRA_FAILED=4, TRAINING_FAILED=5, CONFIG_MISMATCH=6 agree on both sides

### Requirement: The other islands do not wait for the lease
After an island sends LEAVE, the syncer SHALL remove it from the membership at once. The other islands MUST be able to produce the next round before the soft deadline, without waiting for the lease to expire.

#### Scenario: One of two islands leaves
- **WHEN** one of two running islands leaves for any reason and sends LEAVE
- **THEN** the other island completes the next round before the soft deadline

### Requirement: A silent vanish is removed by lease timeout
When an island vanishes without sending LEAVE (hard kill, whole machine down, network cut), the syncer SHALL remove it after the lease times out and SHALL record it as LEASE_EXPIRED. The default lease of elastic-mode islands MUST be 150 s. The defaults of the syncer `--island-lease-s` and of `--rl-island-lease-s` in the launcher and in `yeto` MUST be equal.

#### Scenario: The three defaults are equal
- **WHEN** the default consistency test runs
- **THEN** all three defaults are 150 s

#### Scenario: Removal on lease timeout
- **WHEN** in a test with a shortened lease, one island stops its heartbeat and sends no LEAVE
- **THEN** the syncer removes that island after the lease times out, records LEASE_EXPIRED, and the other island keeps producing rounds
