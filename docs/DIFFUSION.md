# Diffusion island backend

## Why

The causal-LM learner cannot train image and video generators: diffusion
pipelines have VAEs, one or more text encoders, scheduler-specific noise
contracts, model-specific conditioning, and denoisers whose inputs may be
spatial tensors or packed token sequences.

`yeto.diffusion.learner` provides that task-specific inner loop while keeping
the existing Yeto fleet and synchronization model. Each island trains its own
diffusion pipeline with PyTorch; the Rust syncer only sees deterministic
fragments of the trainable tensors.

The backend is selected with `--model-kind diffusion`. Diffusion aliases select
it automatically. A raw Hugging Face repository id must use the flag
explicitly because unknown ids default to the causal-LM learner.

## Design: generic Diffusers first

The default path loads a repository with `DiffusionPipeline.from_pretrained()`
and derives the training contract from public Diffusers interfaces:

- pipeline components and denoiser attributes;
- `encode_prompt()` and denoiser `forward()` signatures;
- scheduler timesteps, sigmas, `scale_noise()`, or `add_noise()`;
- VAE and pipeline packing/unpacking helpers;
- latent, mask, id, size, guidance, and temporal-conditioning shapes.

This keeps model aliases as repository shortcuts rather than switches that
select separate hard-coded trainers. Reusable behavior belongs in the generic
learner. `--diffusion-adapter module:factory` is reserved for model semantics
that cannot be inferred from public interfaces, such as NAVA's custom
audio/video pipeline.

## Architecture

```text
yeto launch --model-kind diffusion ...
        head VM
          |-- Rust syncer and checkpoint              unchanged
          `-- fleet controller                        unchanged
                |
                `-- learner island: one torchrun job
                      yeto.diffusion.learner
                        |-- load Diffusers pipeline or external adapter
                        |-- freeze VAE/text encoders and attach denoiser LoRA
                        |-- rows -> latents + conditioning
                        |-- scheduler noise -> denoiser -> flow-matching loss
                        |-- AdamW inner optimization with DDP/FSDP2
                        `-- canonical trainable tensors
                              -> fragments -> SyncerClient -> Rust syncer
```

Only island rank 0 communicates with the syncer. Returned fragments are
broadcast to the other ranks before all ranks apply the same local/global
blend. The fragment protocol, q4 transport, RDA merge, HeLoCo correction,
outer optimizer, event tape, and checkpoint format are shared with the torch
causal-LM backend.

## Trainable boundary

The generic learner freezes every torch module in `pipe.components`, then
discovers denoisers under `transformer`, `transformer_2`, `unet`, or `model`.
This covers standard UNets, DiTs, and dual-denoiser pipelines such as Wan2.2.

| setting | behavior |
|---|---|
| `--tuning lora` | Attach PEFT LoRA to every discovered denoiser; this is the primary supported path. |
| `--lora-targets auto` | Adapt common attention and MLP projection names. |
| `--lora-targets attention` | Restrict adapters to attention projections. |
| `--lora-targets all-linear` | Ask PEFT to adapt every eligible linear layer. |
| `--shard ddp` | Replicate the base inside the island and explicitly all-reduce LoRA gradients. |
| `--shard fsdp` | Use FSDP2 to shard the frozen base while keeping LoRA tensors replicated and name-stable. |

FSDP2 requires CUDA and a torch build that provides composable
`fully_shard`. Full-parameter tuning is exposed for experiments, but a
syncer-connected FSDP full-tuning run is rejected because the trainable
parameters are sharded. LoRA is the benchmarked asynchronous path.

After wrapping, parameter names are normalized and converted into the same
`{canonical_name: tensor}` mapping on every rank and island. `build_layout`,
`pack_fragment`, `apply_fragment`, and `SyncerClient` are then reused without
diffusion-specific wire behavior.

## Data contract

`--data` accepts a Hugging Face dataset id, a local JSON/JSONL/Parquet or
`save_to_disk` path, or a cloud URI mounted by the launcher. Relative media
and tensor paths in a local manifest are resolved from the manifest directory.

Media and text conditioning are independent choices. Each row needs one media
source and one conditioning source; raw media can use cached text embeddings,
and cached latents can use raw prompts.

| dimension | mode | required row field | optional fields and flags |
|---|---|---|---|
| media | raw image | `image` | Override with `--image-column`. |
| media | raw video | `video` | Override with `--video-column`; `frames`, `height`, and `width` metadata are optional. Use `--num-frames` and `--fps` for a fixed profile. |
| media | cached latents | `latents` | Enable `--cache-latents`; override with `--latent-column`. `latent_num_frames`, `latent_height`, and `latent_width` are optional. |
| conditioning | raw prompt | `prompt` | Override with `--prompt-column`. |
| conditioning | cached text | `prompt_embeds` | Enable `--cache-text-embeds`; override standard fields with `--text-embeds-column`, `--text-attention-mask-column`, and `--pooled-text-embeds-column`. Model-specific tensors are signature-dependent. |

Raw images may be PIL values, dataset byte/path objects, or file paths. Raw
videos may be a video file, a directory of ordered frames, or a list of image
values. Cached tensors may be inline tensors/lists or `.pt`, `.pth`, and `.npy`
paths.

Important shape behavior:

- `--height` and `--width` resize raw media before VAE encoding;
- `--resize-mode stretch` is the default;
- `--resize-mode center-crop` preserves aspect ratio by scaling to fill and
  taking a centered crop;
- `--num-frames` deterministically samples long videos and pads short videos
  with their last frame;
- `--bucket-by-shape` groups rows by `(frames, height, width)` so variable
  shapes never share a micro batch;
- cached latents are not resized or resampled; shape flags only describe the
  intended target profile.

Yeto validates `yeto_diffusion_cache.json` when a cache dataset provides it,
but it does not currently include a cache-precompute command. Cache generation
belongs to the dataset preparation pipeline.

Example manifests:

```json
{"image":"images/0001.png","prompt":"a red chair in a white studio"}
```

```json
{"video":"clips/0001.mp4","prompt":"waves crossing a dark shoreline","frames":49,"height":512,"width":512}
```

## Conditioning and loss

The generic inner step is:

1. Encode raw media through the pipeline VAE, or load cached latents.
2. Call `encode_prompt()` by signature, or load cached text conditioning.
3. Sample scheduler timesteps and noise from the learner's seeded RNG stream.
4. Build the scheduler-specific noisy input and training target.
5. Call the denoiser with signature-matched conditioning and shape fields.
6. Align packed/unpacked prediction and target layouts.
7. Compute element-normalized flow-matching loss and run AdamW.

The denoiser dispatcher supplies common fields such as prompt masks, pooled
embeddings, image/text ids, packed shapes, guidance, crop/size conditioning,
rotary embeddings, FPS, and LTX-style rope interpolation when the model
signature requests them. Multi-denoiser pipelines route samples by timestep
and combine their predictions back into batch order.

`flow_matching` is currently the only accepted loss-function name. The
scheduler may still provide sigma interpolation, epsilon prediction, sample
prediction, or velocity targets. `--diffusion-loss-weighting` supports `linear`,
`sigma`, `snr`, and `min-snr` in addition to the default unweighted loss.

## Reproducibility and autobatching

`--diffusion-seed` controls LoRA initialization, row order, timestep sampling,
noise, and loader RNG streams. Each `(learner_id, island_rank)` maps to one
stable logical rank, so matching topologies can reproduce the same logical
streams. The seed does not force deterministic CUDA kernels.

`--micro-batch-size auto` probes real forward/backward steps and chooses the
largest batch that fits. With shape bucketing it probes up to eight distinct
shapes and uses the smallest successful result. Gradient accumulation is then
rebalanced to preserve the requested effective batch as closely as possible.

## Launching

Install the launcher and diffusion dependencies:

```bash
pip install "yeto[launcher,diffusion] @ ."
```

Image LoRA example:

```bash
yeto launch \
  --gpu aws:1xa100@us-west-2 \
  --model sd35 --model-kind diffusion \
  --data ./image-train.jsonl \
  --height 512 --width 512 --resize-mode center-crop \
  --lora-r 16 --lora-targets auto \
  --diffusion-seed 17
```

Cross-region video example:

```bash
yeto launch \
  --gpu aws:4xa100@us-east-1,aws:4xa100@us-west-2 \
  --model ltx-video \
  --data ./video-train.jsonl \
  --shard fsdp \
  --height 512 --width 512 --resize-mode center-crop \
  --num-frames 49 --fps 9.3 --bucket-by-shape \
  --diffusion-seed 17
```

Diffusion launches currently require an explicit `--gpu` fleet. The automatic
cost/TFLOPs shape planner models causal-LM memory and does not size diffusion
pipelines yet. Diffusion islands use the torch backend; Megatron and MLX are
not selectable for this learner.

## Artifacts, export, and sampling

A learner's saved adapter is its local state. The authoritative DiLoCo result
is the Rust syncer's checkpoint. Rebuild the exact trainable layout and export
the merged adapter with the same model, LoRA, fragment, and external-adapter
settings used for training:

```bash
yeto-diffusion-export \
  --checkpoint yeto-state.ckpt \
  --model ltx-video \
  --lora-r 16 --lora-alpha 32 --lora-targets auto \
  --fragments 8 --fragment-pattern binpack \
  --output-dir merged-ltx-lora
```

The artifact includes `yeto_diffusion_adapter.json`, which records the base
model, trainable modules, cache contract, loss recipe, LoRA recipe, and export
provenance. Standard modules are saved as PEFT/Diffusers adapter directories;
external adapters may provide their own save/load hooks.

Sample locally:

```bash
yeto-diffusion-sample \
  --adapter-dir merged-ltx-lora \
  --prompt "waves crossing a dark shoreline" \
  --num-frames 49 --fps 9 \
  --output sample-frames
```

Or run sampling as a self-cleaning SkyPilot task:

```bash
yeto sample-diffusion \
  --gpu aws:1xa100@us-west-2 \
  --adapter-dir merged-ltx-lora \
  --prompt "waves crossing a dark shoreline" \
  --output ./samples
```

Both samplers also accept a prompt dataset for batch generation.

## External adapter boundary

Use `--diffusion-adapter` only when the generic pipeline cannot express the
model contract. Adapters are duck-typed and may implement the smallest needed
hook set:

- pipeline loading or model preparation;
- trainable module or parameter discovery;
- latent or text/audio conditioning encoders;
- a complete rows-to-loss training step;
- artifact save/load and generation behavior.

Trainable names must remain deterministic across learners, restarts, and
checkpoint export. Adapters must not start syncers, launch infrastructure,
upload artifacts, or communicate between learners. See the
[adapter guide](../yeto/diffusion/adapters/README.md) and
`yeto/diffusion/adapters/template.py`. NAVA is the in-tree example that
requires this boundary.

### Protenix

Protenix is exposed through `yeto.diffusion.adapters.protenix` because its
AF3-style structure diffusion stack is not an image/video Diffusers pipeline.
The adapter can construct the native Protenix model/loss stack for prebatched
Protenix rows, while Yeto owns data distribution, gradient accumulation, and
DiLoCo synchronization. MSA/template search and Protenix feature construction
should happen before Yeto training.

Install the optional dependency under Python 3.11+ and point at an optional
checkpoint:

```bash
pip install "yeto[diffusion-protenix] @ ."
export YETO_PROTENIX_MODEL_NAME=protenix_base_default_v1.0.0
export YETO_PROTENIX_CHECKPOINT=/path/to/protenix/checkpoint
```

Then launch with the external adapter:

```bash
yeto launch \
  --model protenix --model-kind diffusion \
  --data /path/to/protenix-ready-rows.jsonl \
  ...
```

`--model protenix` and `--model protenix-v2` default to
`yeto.diffusion.adapters.protenix:make_adapter`; pass `--diffusion-adapter`
only to override the built-in adapter.

Each Yeto row must contain one complete pre-collated Protenix batch via
`protenix_batch`, the three native keys `input_feature_dict`, `label_dict`, and
`label_full_dict`, or a `protenix_batch_path` / `batch_path` pointing to a
`torch.save`d batch. Keep `--micro-batch-size 1` for native prebatched rows
unless your row already contains a larger Protenix batch.

To produce those rows from a Protenix training environment, run:

```bash
yeto-protenix-export-batch \
  --model-name protenix_base_default_v1.0.0 \
  --output-dir /path/to/yeto-protenix-batches \
  --batch-count 8 \
  --arg-str "--dtype bf16 --diffusion_batch_size 1 --train_crop_size 384 --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 --data.test_sets recentPDB_1536_sample384_0925"
```

The command writes `batches/batch-*.pt` plus `yeto_protenix_rows.jsonl`.

For custom Protenix APIs or on-the-fly feature construction, set
`YETO_PROTENIX_WRAPPER=my_project.protenix_yeto`. The wrapper must provide
`load_pipeline(args, device, model_name=None, checkpoint_path=None)` and return
an object with `model.named_parameters()` or `trainable_params()`, plus
`training_step(batch, global_step=...)` or `compute_loss(...)`. If the wrapper
exposes `build_batch(rows, args, device)`, the adapter calls it before the
training step.

## Validation status

- Unit coverage lives in `tests/test_diffusion.py`,
  `tests/test_diffusion_export.py`, and `tests/test_diffusion_sample.py`.
- Raw-image LoRA train/backward/save/reload has been exercised on Flux Schnell,
  Ideogram4, and Stable Diffusion 3.5.
- Raw-video LoRA has been exercised on LTX-Video and Wan2.1; Wan2.1 14B and
  dual-denoiser Wan2.2 have completed 8-GPU FSDP2 validation.
- The NAVA external adapter has completed GPU train/save/reload validation.
- Held-out quality and equal-hardware synchronization are separate concerns;
  use [DIFFUSION_BENCHMARK.md](DIFFUSION_BENCHMARK.md) for that experiment.

## Current limitations

- The backend remains experimental and LoRA-focused.
- Diffusion launch requires an explicit fleet and the torch island backend.
- Only `flow_matching` is accepted as the generic loss-function name.
- Syncer-connected FSDP full-parameter training is unsupported.
- Cache generation is external to Yeto.
