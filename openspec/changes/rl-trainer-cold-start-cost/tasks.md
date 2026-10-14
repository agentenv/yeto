# Tasks

Codes follow the phase codes of `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §8 (L0–L8). This change holds L0, L1, L2, and L8. Effort is in sub-agent person-days and is an estimate. "GPU: no" means local CPU tests only.

## L0. Start phase events

- [ ] L0.1 Emit `trainer_start_phase` for the seven phases in design D4, and one summary event. Acceptance: CPU test with a fake launcher asserts names, order, and total. GPU: no. Estimate: 0.5–1 d
- [ ] L0.2 Write the phases into the run manifest summary, so that the scheduler can read `T_s` from the last run. Acceptance: CPU test reads `T_s` back. GPU: no. Estimate: 0.5 d

## L2. Software fixes (no new GPU feature)

- [ ] L2.1 Miles fork: light actors do not import `miles.backends.megatron_utils`. Acceptance: CPU test on `sys.modules` (spec "Light actors"); existing Miles CPU tests pass. GPU: no. Estimate: 1–2 d
- [ ] L2.2 Miles fork: `megatron.bridge` imports `modelopt` lazily. Acceptance: CPU test on `sys.modules`; `-X importtime` before and after recorded in the evidence directory. GPU: no. Estimate: 1 d
- [ ] L2.3 Setup reads the image manifest instead of importing `peft` and `sglang`; fallback with an event. Acceptance: CPU tests for present, missing, and mismatched manifest. GPU: no. Estimate: 0.5 d
- [ ] L2.4 Add `peft` 0.20.0 to the ports image, new tag, update pins, regenerate golden samples by `hash-migration.md`. Acceptance: image build log; pin tests pass. GPU: no (CPU image build). Estimate: 0.5–1 d
- [ ] L2.5 Replace the `huggingface-cli download` prefetch (`yeto/launcher.py:4166`, `:4824` on `d6c58fb8`) and emit `model_prefetch_failed`. Acceptance: CPU test of the generated command and of the failure event. GPU: no. Estimate: 0.5 d
- [ ] L2.6 One GPU smoke run with L2.1–L2.5: same shape as the 10-14 test 1 (1×H100, Qwen3.5-0.8B). Acceptance: all seven phases present; actor import time and setup time compared with `coldstart-gpu-review.md` §6.1 (target: lower; exact target not set). GPU: yes, review before launch. Estimate: 0.5 d + about 1 GPU-hour (estimate, about $5 at $5.3/h, the rate in §6.5)

## L1. Measure what is still unknown

- [ ] L1.1 Measure a separate trainer island (no colocated rollout), with optimizer state load, in two steps. Step 1: Qwen3.5-4B on 2 cards, TP2. Step 2: decide from step 1 whether to measure Qwen3.8-27B on 8 cards. Step 2 needs the shared filesystem grown to 1024 GiB and a full `torch_dist` checkpoint conversion first. Acceptance: phase events and load time per phase written to the evidence directory. GPU: yes, review before launch. Estimate: 1 d + step 1 about $5–8 (estimate); step 2 not estimated, needs user budget approval

## L8. Snapshot work

- [ ] L8.1 SGLang snapshot compatibility check (design D5): flags on vs off, two RL weight updates, compare text and weight checksum. Acceptance: pass or fail record for the pinned SGLang commit. GPU: yes, 1×H100. Estimate: 1 d + under 1 GPU-hour (estimate)
- [ ] L8.2 Only if L8.1 passes: opt-in snapshot mode for sampler engines, refused without a recorded pass. Acceptance: CPU test of the refusal; GPU run with at least one restore. GPU: yes. Estimate: 1–2 d
- [ ] L8.3 Trainer snapshot: research only, no task in this change.
