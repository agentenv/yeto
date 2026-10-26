# Spec Delta

## Purpose

Run full-parameter trainer islands for large MoE models on NPU and PPU nodes, with a memory plan that is checked before any card is used.

## ADDED Requirements

### Requirement: Trainer island on NPU and PPU nodes
The system SHALL launch a trainer island on `npu` or `ppu` nodes from the node inventory. The island SHALL write BF16 releases, cuts, and resume state through the same ports as other trainer islands. One trainer island SHALL use one device family.

#### Scenario: Mixed families in one trainer island
- **WHEN** a trainer island configuration lists both `npu` and `ppu` nodes
- **THEN** the launch is refused and the error names both families

### Requirement: Checked memory plan
Before launch, the system SHALL compute per-card HBM and per-node host memory for the configured memory plan, and SHALL refuse the launch if either exceeds the inventory values. The result SHALL be written to the run record and marked as derived.

#### Scenario: Plan does not fit host memory
- **WHEN** the plan offloads optimizer states and the computed host memory per node is larger than the inventory value
- **THEN** the launch is refused and the error shows the computed and the available host memory

### Requirement: W4A8 releases derived from BF16 releases
When a trainer island has a re-quantization template, the system SHALL write one W4A8 release for each BF16 release, in a separate release directory, with the BF16 policy version, the buffer positions, and a record of the BF16 release it came from. A W4A8 sampler SHALL load only complete W4A8 releases.

#### Scenario: Restart after a conversion
- **WHEN** the re-quantizer restarts and the newest BF16 release already has a W4A8 release
- **THEN** no new W4A8 release is written

### Requirement: Card sharing on one site
When the trainer and a sampler share the same nodes, the configuration SHALL state time-shared or node-split mode. In time-shared mode the sampler SHALL release its device memory before each update and SHALL serve the new release after it.

#### Scenario: Time-shared round
- **WHEN** a time-shared round finishes generation
- **THEN** the sampler sleeps, the trainer updates, the quantized release is committed, and the sampler wakes on that release before the next generation
