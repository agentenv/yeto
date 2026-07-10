# Diffusion Adapters

Adapters are for diffusion models whose training or sampling contract cannot be
handled by Yeto's generic Diffusers learner. The default path should remain the
LM-style path: resolve an alias or raw Hugging Face id, load it with Diffusers,
discover the trainable denoiser, attach LoRA, build conditioning from dataset
rows, run the generic scheduler/noise/loss loop, and reuse the normal DiLoCo
sync path.

## Boundary

Prefer the generic learner when the difference is exposed by public Diffusers
interfaces:

- pipeline components and config fields;
- scheduler methods and prediction/noise conventions;
- `encode_prompt()` and denoiser `forward()` signatures;
- latent, prompt, mask, id, guidance, or packed-sequence tensor shapes;
- helper methods provided by the pipeline.

Use an adapter when the model needs behavior that cannot be inferred reliably
from those public interfaces:

- custom loading that `DiffusionPipeline.from_pretrained()` cannot perform;
- private media, text, audio, or multimodal encoding;
- conditioning layouts that are model-specific and not declared by signatures;
- a custom loss, timestep contract, or native rows-to-loss training step;
- custom artifact save/load or sampling behavior.

In short: if the fix is a reusable Diffusers capability, put it in the generic
learner. If it is model semantics, put it in an adapter.

## Adapter Shapes

Start from the smallest hook set that solves the model:

- load-only: implement `load_pipeline()` when loading is custom but the generic
  training loop still applies;
- preparation: add `prepare_model()` or `trainable_module_items()` for custom
  freezing, LoRA attachment, or trainable module discovery;
- encoding: add `encode_latents()` or `encode_prompt_embeds()` when feature
  construction is custom but scheduler/denoiser/loss remain generic;
- full-step: add `training_step()` or `compute_loss()` when the model owns the
  whole batch-to-loss flow;
- artifact/sampling: add save/load/sample hooks only when generic PEFT/Diffusers
  artifact handling is not enough.

`base.py` documents the supported hook protocol. `template.py` contains
copyable examples for the modes above. `nava.py` is a real adapter example, but
it is not the standard shape every future adapter should copy.

Stable requirements:

- trainable module names and trainable parameter keys must be deterministic
  across learners, restarts, and syncer-checkpoint export;
- loss denominators must be scalar tensors on the training device;
- adapters must not start launchers, syncers, storage uploads, or cross-learner
  communication. Yeto owns orchestration and DiLoCo sync.
