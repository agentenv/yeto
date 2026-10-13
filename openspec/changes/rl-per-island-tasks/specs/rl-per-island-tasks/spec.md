# Spec Delta

## ADDED Requirements

### Requirement: Specify training tasks per island
The launcher SHALL accept a task id list per island, and for each island SHALL generate a prompt file that contains only the tasks of that island as the training data of that island. When this flag is not given, all islands MUST use the same data, and behavior is the same as today.

#### Scenario: Two islands with different tasks
- **WHEN** a two-island run starts with `--rl-island-tasks "l0=web1-10,pwn1-10;l1=web11-20,pwn11-20"`
- **THEN** the prompt file of island 0 contains only web1-10 and pwn1-10, and the file of island 1 contains only web11-20 and pwn11-20

#### Scenario: Flag not given
- **WHEN** the run starts without `--rl-island-tasks`
- **THEN** the `--data` of every island is the same

### Requirement: Reject wrong task configuration at launch
The launcher MUST reject the following configurations before cards are opened: an island number out of range, an island with no task list, a task id not in the catalog. When task counts differ between islands, the launcher SHALL give a warning that lists the task count of each island, and SHALL NOT reject (the same for strict and elastic). The dense-full and SAO paths SHALL allow per-island tasks.

#### Scenario: Unequal task counts between islands
- **WHEN** island 0 has 20 tasks and island 1 has 18 tasks (strict-avg or elastic)
- **THEN** the launch continues, and the launcher gives one warning that states the task count of each island

#### Scenario: Unknown task
- **WHEN** the task list contains an id that does not exist in the catalog
- **THEN** the launch fails, and the error lists that id

### Requirement: Record the tasks of each island
The run manifest and the `rl_driver_start` event of each island SHALL record the task id list of that island and the sha256 of the prompt file. At island start, the island MUST compare the sha256 of the local prompt file with the manifest, and MUST refuse to start if they differ.

#### Scenario: Data replaced on resume
- **WHEN** after an island restarts, the sha256 of its local prompt file differs from the manifest
- **THEN** the island refuses to start and reports both sha256 values
