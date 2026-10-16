# Proposal

## Why

The S25 goal runs full-parameter RL of GLM5.3-Flash on Ascend 910 NPUs (a local cluster of 4 nodes x 8 cards, 64 GiB, behind a VPN and a jump host) and Alibaba T-Head PPUs (machines that the user opens and reaches by SSH). Both device families host trainer islands and sampler islands. NVIDIA cards are not used in this goal. SkyPilot cannot launch either: it has no Ascend accelerator and no Alibaba Cloud provider (`infra-drafts/SESSION24/T-NPU-gaps.md` §10, `T-PPU-gaps.md` §1). The SkyPilot ssh node pool installs k3s and is a Kubernetes cluster, not plain machines (`openspec/changes/rl-local-cluster-deploy` design D1).

The draft change `rl-local-cluster-deploy` defines a `local` provider (node inventory + SSH + docker) and `--controller here`, but it assumes NVIDIA cards (`nvidia-smi`, nvidia-container-toolkit), no jump host, and the H200 pin in `yeto/rl/ssh_harness.py`. This change extends that design to any device family, so the local cluster and machines that the user opens by hand on a cloud use one path.

## What Changes

- Node inventory file (`--rl-node-inventory <yaml>`): per node the SSH target (with an optional `proxy_jump`), device family (`cuda`, `npu`, `ppu`), card model, card count, HBM capacity in GiB, host memory in GiB (the trainer offload plan depends on it), fabric (`tcp`, `roce`, `hccs`, `ib`), shared mount paths, image source, and an optional declared price per card hour.
- Device-family preflight before any accelerator process starts: `nvidia-smi` for `cuda`, `npu-smi info` for `npu`, `ppu-smi` for `ppu`. A model, count, or HBM mismatch refuses the launch. An unknown card model refuses the launch (today `yeto/gpu_spec.py` `device_type_of` falls back to `cuda`).
- SSH executor: prepare, deploy, start, status, stop, collect, with `ProxyJump` support. It reuses `yeto/rl/ssh_harness.py` and removes its H200 pin.
- Collective-communication environment by device family in `yeto/launch/topology.py` (NCCL for `cuda`; HCCL for `npu`; ACCL-P under NCCL names for `ppu`), driven by the inventory `fabric`, not by cloud names.
- The head runs on the submitting machine (`--controller here` from `rl-local-cluster-deploy`); the inventory gives the address that nodes use to reach it.

## Non-goals

- No SkyPilot path for these nodes, no Kubernetes or Slurm backend.
- No cloud API that opens or closes Huawei or Alibaba machines.
- No change to Modal, Nebius, Verda, or AWS launch behavior.
- No training logic; the trainer island is change `rl-npu-ppu-trainer`. This change only launches and checks nodes.
- The `cuda` family keeps its parser, but this goal does not test it on cards.

## Capabilities

### New Capabilities
- `rl-ssh-node-pool`: inventory, device-family preflight, SSH executor, and network environment for nodes that yeto reaches by SSH.

### Modified Capabilities
- None. This change builds on the draft `rl-local-cluster-provider` capability of `rl-local-cluster-deploy`; if that draft is merged first, this change becomes a delta on it.

## Impact

- Code: `yeto/gpu_spec.py`, `yeto/hw/catalog.py`, `yeto/rl/ssh_harness.py`, `yeto/launch/topology.py`, a new inventory module under `yeto/launch/`.
- Tests: CPU tests with stub `npu-smi`/`ppu-smi`/`nvidia-smi` output. One smoke on one local node.
- Risk: the jump host and VPN add latency and a single point of failure for the head-to-node path.
