# Design

## Context

The measurements are in `infra-drafts/SESSION23/coldstart-gpu-review.md` §6 (2026-10-14, 1×H100, about $5–8 total, estimate not checked against the bill). The cost breakdown of the earlier runs is in `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §1. All numbers below come from these two files unless marked "estimate".

Container setup steps (original image / prebuilt image), seconds: Miles check 1.3/1.3; peft 32.5/27.5; sglang check 17.8/19.2; ray start 13.2/10.9; island entry import 2.6/2.5; model pull 7.2/0.5; dataset 4.2/3.1; placement group 4.5/4.1.

## Goals / Non-Goals

Goals:
- Lower the trainer start cost `T_s` with low-risk software fixes.
- Make `T_s` a measured value per run, so the scheduler can use it.
- Prepare a fast start path for sampler engines.

Non-goals:
- Trainer GPU snapshots in production.
- Warm pools that hold idle GPUs (replaced by convertible nodes in `rl-trainer-start-stop`).
- Large-model (27B) weight and optimizer load time. This is not measured yet; task L1.1 measures Qwen3.5-4B first and decides on 27B from that result.

## Decisions

### D1. Fix imports before anything else
The import cost is per actor process and is paid again on every start. A lazy import is a code-only fix with no GPU need to build. Expected saving: about 25–30 s per light actor (estimate from `-X importtime`, not measured on the Miles branch).

Rule: the fix must not change behavior. A CPU test asserts that importing `miles.utils.http_utils` does not load `miles.backends.megatron_utils`, and that importing `megatron.bridge` does not load `modelopt.torch`.

### D2. Setup checks read the image manifest
The ports image already writes `/opt/yeto/image-manifest.json`. Setup compares the package versions in the manifest with the pins. It does not import the packages. If the manifest is missing, setup falls back to the import check and emits an event. Expected saving: about 45 s (estimate: peft 27 s + sglang 18 s).

### D3. Prefetch must work or say it failed
Replace `huggingface-cli download` with `hf download`, or with a Python call to `huggingface_hub.snapshot_download`. The implementer picks the one that exists in the image and records the choice. On failure, emit `model_prefetch_failed` with the exit code. The run continues; Miles pulls the model later as now.

### D4. Start phase events
Emit one `trainer_start_phase` event per phase with a name and a monotonic time. Phase names are fixed: `submitted`, `container_ready`, `setup_done`, `ray_ready`, `actor_import_done`, `actor_init_done`, `first_step_done`. The scheduler computes `T_s` as `first_step_done − submitted`. A summary event carries the total.

### D5. SGLang snapshot is opt-in and gated
Snapshot mode needs `--enable-memory-saver --enable-weights-cpu-backup`. Before snapshot mode can be turned on, a GPU check must pass: start an engine with these flags, apply two RL weight updates, and compare the generated text and the weight checksum with an engine that runs without the flags. Until the check passes, the option is refused at launch.

Hit rate is low (3 of 8 in the test; Modal documents 2–3 snapshots per GPU type). So a cold start is still the normal case. The scheduler must not assume the snapshot start time.

### D6. Trainer snapshots stay research
Only 1 GPU with the NCCL group kept passed. Re-init failed with `Call to bind failed: Cannot assign requested address`. The multi-GPU trainer is the target, so this path is not used.

## Risks

- The Miles import fix can break a hidden import side effect. Mitigation: run the existing Miles CPU tests and one GPU smoke run (L2.6).
- A new image tag changes pins and golden samples (see `yeto-framework-decoupling/hash-migration.md`). Mitigation: regenerate the golden samples in the same task, as the hash migration document requires.
