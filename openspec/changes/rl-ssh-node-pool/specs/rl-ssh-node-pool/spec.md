# Spec Delta

## Purpose

Let yeto launch RL islands on machines that it reaches by SSH, for any device family, with checks before any accelerator process starts.

## ADDED Requirements

### Requirement: Node inventory
The launcher SHALL read nodes from an inventory file. Each node SHALL state an SSH target, a device family (`cuda`, `npu`, or `ppu`), a card model, a card count, an HBM capacity, and a host memory size. A node MAY state a jump host, a fabric, shared mounts, an image, and a declared price.

#### Scenario: Missing device family
- **WHEN** a node in the inventory has no device family
- **THEN** the launch is refused and the error names the node

### Requirement: Device-family preflight
Before any accelerator process starts, the launcher SHALL read the card tool of the node's device family and SHALL refuse the launch if the model, count, or HBM capacity differs from the inventory.

#### Scenario: HBM mismatch on Ascend
- **WHEN** the inventory states 64 GiB and `npu-smi info` reports 32 GiB
- **THEN** the launch is refused and the error shows both values

#### Scenario: Unknown card model
- **WHEN** the card tool reports a model that the device catalog does not know
- **THEN** the launch is refused; the model is not treated as `cuda`

### Requirement: SSH through a jump host
The executor SHALL reach a node through the jump host that the inventory states, and SHALL NOT edit the user's SSH configuration.

#### Scenario: Node behind a jump host
- **WHEN** a node states `proxy_jump: bastion`
- **THEN** every SSH and file copy to that node goes through `bastion`

### Requirement: Network environment by fabric
The island environment SHALL contain the collective-communication variables for the node's device family and fabric. It SHALL NOT disable a fabric that the inventory declares.

#### Scenario: Ascend island on RoCE
- **WHEN** an `npu` node declares fabric `roce` and interface `eth0`
- **THEN** the island environment contains the HCCL interface variable set to `eth0` and no NCCL variables
