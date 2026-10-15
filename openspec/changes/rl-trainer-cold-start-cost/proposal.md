# Proposal

## Why

The next goal makes the trainer start and stop many times (see `rl-trainer-start-stop`). Each start has a cost. The ski-rental stop rule and the backlog threshold `Q_on` both scale with this cost. So the start cost must go down first.

Measured on 2026-10-14, 1×H100 80GB on Modal, Qwen3.5-0.8B (source: `infra-drafts/SESSION23/coldstart-gpu-review.md` §6, and `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §1):

1. From submit to the first generation: about 8.4 min with a cached image, about 13 min on the first image pull.
2. Loading the 0.8B weights takes about 1 s. Most of the start time is software initialization, not weight loading.
3. Each Ray actor process imports the full set of heavy libraries: 31–48 s for one process. Six actors that import at the same time take 72–78 s wall time.
   - Light actors (RayWorkerManager, CommandActor, InferenceController) import the full `miles.backends.megatron_utils` package through `miles.utils.http_utils → logging_utils → audit_utils.event_logger.models → miles.backends.megatron_utils.ft.types`. This costs about 28–31 s.
   - `MegatronTrainRayActor` imports `megatron.bridge` in about 28–33 s. Of this, `modelopt.torch` is about 21–30 s.
4. The island setup imports `peft` (about 27 s) and `sglang` (about 18 s) only to check that they exist. A prebuilt image does not remove this cost.
5. The setup prefetch command `huggingface-cli download` fails in the current image (the CLI is deprecated). The command line ends with `|| true`, so the failure is silent and the prefetch has no effect (`yeto/launcher.py:4166`, `:4824` on `d6c58fb8`).
6. SGLang GPU memory snapshot works: restore to ready in 9.7–16.6 s against 195–292 s cold. Only 3 of 8 calls restored. It needs `--enable-memory-saver --enable-weights-cpu-backup`. The effect of these flags on the RL weight update path is not verified.
7. Trainer GPU snapshot (Megatron + NCCL) works only for 1 GPU with the NCCL group kept. Re-init of NCCL after restore failed. Multi-GPU is not verified.

## What Changes

- Fix the Miles import chain (highest priority). Light actors must not import `miles.backends.megatron_utils`. `megatron.bridge` imports `modelopt` lazily. The change is in the Miles fork; yeto pins the new commit.
- The island setup does not import heavy libraries to check them. It reads the image manifest `/opt/yeto/image-manifest.json` instead. `peft` 0.20.0 goes into the ports image (new image tag).
- Replace the failed `huggingface-cli download` prefetch with a command that works in the image. A prefetch failure becomes a visible event, not a silent `|| true`.
- Add start phase events (submit, container ready, setup steps, Ray ready, actor import, actor init, first train step). The start cost becomes a measured number per run. The trainer scheduler reads it as `T_s`.
- SGLang snapshot for sampler engines: first verify that memory-saver and CPU weight backup are compatible with the RL weight update path. Only after that, add an opt-in snapshot mode for sampler engines.
- Do not depend on trainer GPU snapshots. Keep it as a research item only.
- Do not copy yeto into the image. The prebuilt image saved only about 14 s (83 → 69 s), and every yeto change would need a rebuild.

## Capabilities

### New Capabilities
- `rl-trainer-cold-start`: start phase events, setup checks without heavy imports, a prefetch that reports failure, and an opt-in sampler snapshot mode gated by a compatibility check.

### Modified Capabilities
<!-- None. -->

## Impact

- Miles fork (`agentenv/miles`, ports chain): import chain of `miles.utils.http_utils` and `megatron.bridge`. New Miles commit and pins.
- `yeto/launcher.py`: setup checks (`:3379`, `:3401` on `d6c58fb8`), prefetch (`:4166`, `:4824`), phase events.
- Ports image build: add `peft` 0.20.0 and a manifest entry. New image tag.
- `yeto/rl/adapters/miles/` SGLang engine launch: opt-in snapshot flags.
- No change to algorithm hashes or tape fields of training.
- Depends on nothing in the other two changes. It can start now.
