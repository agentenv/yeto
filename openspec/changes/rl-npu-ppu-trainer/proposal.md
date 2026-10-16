# Proposal

## Why

The S25 goal (user decision, 2026-10-16) trains GLM5.3-Flash (`glm5_next`, about 314B parameters, MoE, KDA and DSA attention) with full parameters on Ascend 910 NPUs and on Alibaba PPUs. yeto has no trainer island for either: its Megatron learner is CUDA-only (`yeto/megatron/learner.py` sets `torch.cuda` and `nccl`), the Miles path is CUDA-only, and verl has no RL run on NPU cards yet (`infra-drafts/SESSION24/T-NPU-gaps.md` §0, §5).

The GLM work in `glm-rl` (contract `glm_rl/CONTRACT.md`) chose FSDP2 + HF `glm5_next` + Ulysses under verl, because verl has no `glm5_next` and Megatron lacks KDA and mHC. It trained only LoRA on a frozen base, and on 8 x H200 the first update step ran out of memory (T-GLM2 hand-off §5). Full-parameter training needs far more memory: with BF16 weights and gradients, FP32 master weights, and FP32 Adam states (16 bytes per parameter), 314B parameters need about 5.0 TB (derived, not measured). The local cluster has 32 x 64 GiB, about 2.2 TB.

Stage order (user decision, 2026-10-16, later the same day): the PPU machines (8 nodes x 8 cards, 144 GB per card, about 9.2 TB of HBM) are the trainer island; the Ascend cluster (32 x 64 GiB) serves only as a sampler island in this goal. The NPU trainer island and its offload plans are deferred, not dropped.

## What Changes

- Trainer island on `npu` and `ppu` nodes, launched through `rl-ssh-node-pool`, running verl with the FSDP2 engine and HF `glm5_next`, full parameters. It reuses the GLM fixes: rank-0 load (`glm_rl/verl_ext/rank0_load.py`) and the native layout copy (`glm_rl/model/native_layout.py`).
- Memory plan by configuration: FSDP2 full sharding plus optimizer-state offload to host memory (plan B), or full offload of parameters, gradients, and optimizer (plan C). An estimator reports per-card HBM and per-node host memory for each plan before launch and refuses a plan that does not fit the inventory.
- Card split between trainer and sampler on one site: time-shared (the sampler sleeps during the update) or node-split, chosen by configuration.
- The trainer writes BF16 releases through `ReleasingPublisher`, and the quantized-release step of `rl-hetero-sampler-engines` derives W4A8 releases.

## Non-goals

- No single trainer island that spans NPU and PPU (different collective libraries).
- No MindSpeed or Megatron path for `glm5_next` in this change.
- No new kernels for KDA or DSA; if a family lacks them, this change reports it and stops (task N7.0).
- No change to the importance-weight computation (backend side, S24 decision Q4).

## Capabilities

### New Capabilities
- `rl-npu-ppu-trainer`: full-parameter trainer islands on NPU and PPU nodes, with a checked memory plan.

### Modified Capabilities
- None in `openspec/specs/`.

## Impact

- Code (neutral, in yeto): `yeto/rl/fullparam/memory.py` (estimator), `yeto/rl/fullparam/fsdp2.py` (FSDP2 plan, verl overrides), `yeto/launch/ssh_trainer_island.py` (island over SSH nodes), `yeto/rl/quant/release.py` (BF16 release -> W4A8 release). The GLM-specific trainer entry stays in glm-rl.
- Images: an NPU trainer image (torch_npu, verl, transformers with `glm5_next`); a PPU trainer image based on the official training image.
- Host memory: plan B needs about 3.8 TB of host memory across the trainer nodes, plan C about 5.0 TB (derived). The local nodes' host memory is not known yet.
