# Tasks

Codes follow `infra-drafts/SESSION25/GOAL.md` §3 (v2). CPU-first tasks need no NPU or PPU. Effort is in sub-agent person-days and is an estimate.

## N7. Trainer island

- [ ] N7.0 Read the transformers `glm5_next` modeling code; list the kernels that KDA, DSA, and mHC use (Triton, CUDA, plain torch) and whether NPU and PPU builds exist. Acceptance: table with source lines. CPU-first: yes. Estimate: 1–2 d
- [x] N7.1 Memory estimator for plans B, C, and option D (design D2); refuses plans that exceed the inventory. Acceptance: CPU tests; outputs marked derived. CPU-first: yes. Estimate: 1–1.5 d. Done: `yeto/rl/fullparam/memory.py`, tests/test_rl_fullparam_memory.py.
- [x] N7.2 Full-parameter verl FSDP2 configuration with optimizer offload switch and Ulysses; remove LoRA-only assumptions. Acceptance: CPU test with 2 gloo ranks on `glm_rl/model/mini.py`, one step with finite loss. CPU-first: yes. Estimate: 3–4 d. Done: `yeto/rl/fullparam/fsdp2.py`; 2 gloo ranks, one step on a tiny `glm5_next` (needs transformers with `glm5_next`; skipped otherwise). Open: Ulysses for KDA/DSA (design D6); the glm-rl trainer entry that applies these overrides.
- [x] N7.3 yeto trainer island entry on SSH nodes: BF16 releases through `ReleasingPublisher`, cuts, resume (design D4). Acceptance: CPU tests with a fake trainer. CPU-first: yes. Estimate: 2–3 d. Done: `yeto/launch/ssh_trainer_island.py` (design D5), `yeto/rl/quant/release.py`; tests/test_rl_ssh_trainer_island.py. Open: the launcher does not yet hold back an SSH island from `--gpu` (N1.3); `runtime: docker` nodes refused (N0.2b).
- [ ] N0.2b NPU trainer image (torch_npu, verl, transformers with `glm5_next`); pins and in-image assertions. Acceptance: assertions pass on one local node. CPU-first: Dockerfile and pins yes. Estimate: 1–2 d
- [ ] N9.0 PPU single-card probe script (mini GLM with KDA/DSA/mHC, 2k/8k/32k forward + backward, peak memory, block times, fla/triton check): `infra-drafts/SESSION25/probes/ppu_mini_glm_probe.py`, CPU dry run only. Needs: one PPU card.
- [ ] N7.4 Stage 1 NPU probe: mini model, then a few layers, then full size one step; record per-card HBM peak, host memory peak, step time. Acceptance: full-size step completes with finite loss and gradient norm. Cards: local cluster. Estimate: 3–5 d
- [ ] N5.1 Stage 1 RL loop: NPU trainer + NPU W4A8 sampler; 3 rounds, then 20 rounds as the same-family control. Acceptance: `GOAL.md` §6 criteria 1–4. Cards: local cluster. Estimate: 2–3 d

## N9. PPU trainer and fleet acceptance

- [ ] N9.1 Stage 3 PPU trainer probe, as N7.4. Needs: PPU machines. Estimate: 3–5 d
- [ ] N9.2 Stage 3 cross runs: PPU trainer + NPU sampler; NPU trainer + PPU sampler. Acceptance: `GOAL.md` §6 criteria 1–4. Estimate: 2–3 d
- [ ] N9.3 Stage 4 NPU + PPU fleet RL, 20 rounds. Acceptance: `GOAL.md` §6, all criteria. Estimate: 2–3 d
