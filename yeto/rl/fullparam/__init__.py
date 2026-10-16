"""Full-parameter training islands (S25 N7, change rl-npu-ppu-trainer).

* :mod:`.memory`: per-card and per-node memory estimate of one FSDP2 training
  step; the launcher refuses a plan that does not fit before any SSH (N7.1).
* :mod:`.fsdp2`: the FSDP2 full-parameter configuration (sharding, mixed
  precision, offload, Ulysses checks) and one training step on any HF causal
  LM; the verl overrides of the same plan (N7.2).

Device-neutral: nothing here imports a vendor runtime (torch_npu, PPU SDK).
The device family enters only through the collective backend name.
"""
