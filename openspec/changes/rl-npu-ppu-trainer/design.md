## Context

- Model: GLM5.3-Flash, `glm5_next`, about 314B parameters; BF16 source 598.5 GiB (measured file size; about 321B parameters at 2 bytes, the gap is not explained).
- GLM stack (glm-rl): verl 0.9.1, FSDP2, HF `glm5_next` (transformers 5.16.1), Ulysses sequence parallel. Rank-0 load and native layout fixes passed on CUDA. Full-size training health was never measured.
- NPU: verl lists FSDP on Ascend as supported (`infra-drafts/VERL-ASCEND-DESIGN.md` §1); `glm5_next` is not in its model table. yeto's own torch learner passed on 910B4 (one node, 8 cards, HCCL).
- PPU: CUDA compatibility layer; official training image with megatron-core 0.14/0.15; no official verl guide.
- Kernels: whether the HF `glm5_next` KDA and DSA code needs CUDA or Triton kernels, and whether NPU or PPU builds exist, is not verified.

## Decisions

### D1. FSDP2 + HF `glm5_next`, same as the GLM contract
Reason: verl has no `glm5_next` model; MindSpeed and Megatron lack KDA and mHC (inferred, not verified for MindSpeed). Changing the parallel framework would add a model port before the first run.

### D2. Memory plans as configuration, checked before launch
Per parameter (derived): BF16 weights 2 B, BF16 gradients 2 B, FP32 master 4 B, Adam m and v 8 B; 16 B in total, about 5.02 TB for 314B.
- Plan A, all on device: about 157 GB per card on 32 cards; does not fit 64 GiB. Not offered.
- Plan B, weights and gradients sharded on device, master and Adam on host with a host optimizer step: about 39 GB per card on 32 cards (52 GB on 24 cards) plus activations; about 3.77 TB host memory.
- Plan C, full offload: device holds the current layers and activations; about 5.02 TB host memory.
- Option D (combined with B or C): BF16 or 8-bit optimizer states, or no FP32 master; 8 to 12 B per parameter. Numerical effect on RL is not known; off by default.
The estimator adds activations from sequence length and Ulysses degree as a configured margin, because no measurement exists. It refuses a plan whose host or device total exceeds the inventory.

Implemented estimate (`yeto/rl/fullparam/memory.py`): FSDP2 keeps an FP32 parameter shard when it computes in BF16, so plan A is 4 B parameters + 4 B reduced gradients + 8 B Adam per parameter over the shard group. It adds the gathered largest layer (prefetch copies in BF16 plus its unsharded gradient), the layer inputs under full recomputation (mHC keeps `hc_mult` streams), the FP32 logits, a configured working-set margin, and the dense DSA terms of the HF path (`[T, T]` mask and the indexer scores, not divided by Ulysses). Plan B (host optimizer step with BF16 weights on the card) is estimated but refused at launch: neither FSDP2 nor verl runs it.

### D3. One site, two ways to share cards
- Time-shared: all cards train, then the W4A8 sampler wakes and generates. No overlap; the trainer gets every card.
- Node-split: fixed nodes for the sampler. Overlap of rollout and training; fewer trainer cards.
Stage 1 starts with time-shared plus plan B; the probe (N7.4) decides.

### D4. Same island contract on NPU and PPU
The trainer island exposes the same ports as other trainers (release, cut, resume). Device family appears only in the image, the launch environment, and the collective backend (`hccl` for `npu`, NCCL names over ACCL-P for `ppu`).

### D5. SSH island mechanics
The island is one `torchrun` per node, started over SSH in a new session; node 0's log stream carries the trainer events to the head; release kills each node's process group through a PID file whose path is derived from the run id, island name, and incarnation (so a restarted head can stop an orphan). Node 0 also runs the re-quantizer as a side process: each BF16 release in `policy-releases/` becomes a W4A8 release in `policy-releases-w4a8/` with the same commit order, and one more pass runs after the trainer ends. The head reads the store by object key and counts shard records the trainer has not merged yet (N3.6).

### D6. Ulysses for `glm5_next`
verl's Ulysses patch replaces only `_flash_attention_forward`; `glm5_next` has no flash-attention path and its KDA layers need their own sequence split. Ulysses above 1 is refused for such a model until a KDA/DSA Ulysses patch exists.

### Non-goals and rejected alternatives
- Rejected: one training island across NPU and PPU.
- Rejected: LoRA training (user chose full parameters).

## Risks
- Missing KDA or DSA training kernels on NPU or PPU block stage 1 or stage 3.
- Host memory may not fit plan B or C; then the trainer needs more nodes.
- Host optimizer steps over 3.77 TB of state may make one step very slow; not measured.
- Sequence length (the GLM contract uses calls of up to 262144 tokens) may make activations the largest item; not estimated.
