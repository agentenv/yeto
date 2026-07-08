# Task backends and diffusion components

Yeto's fleet controller, Rust syncer, fragment layout, checkpointing, and
artifact delivery are task-agnostic. Task-specific code lives behind a backend
interface so adding a new training family does not add one-off branches to the
launcher.

## Dispatch model

- `--task lm` is the default backend and preserves the existing language-model
  fine-tuning path.
- `--task diffusion` selects the generic diffusion learner. A diffusion run must
  also select a `--component` that knows how to construct the concrete model,
  data loader, trainable parameter set, optimizer schedule, and export format.
- Structured model aliases may infer the backend. Today `--model nava` expands
  to `--task diffusion --component nava` plus component defaults from
  `yeto/models.py`.
- Unstructured model ids still default to `--task lm`, so existing LM commands
  continue to work.

The parser is built in two phases: first it reads `--task` and `--model` to
choose the backend, then it asks that backend to add its own CLI flags. This is
why LM-only flags such as `--loss-function`, `--train-on`, and
`--micro-batch-size` live in `yeto/backends/lm.py`, while diffusion-only flags
such as `--component-root`, `--base-checkpoint`, and `--adapter` live in
`yeto/backends/diffusion.py`.

## Backend responsibilities

A task backend implements `TaskBackend` in `yeto/backends/base.py`:

- add launch/export CLI args for that task,
- normalize structured model defaults after parsing,
- validate task-specific required fields,
- build one learner island `sky.Task`,
- choose optional runtime images,
- expose the learner artifact directory for delivery,
- export a syncer checkpoint into a task-specific artifact.

The LM backend also owns island engines (`torch` and `megatron`). Engines are
an LM-only axis for now: they change the intra-island trainer while keeping the
same LM data, loss, adapter, and DiLoCo sync semantics.

## Diffusion components

The diffusion backend owns provisioning and the generic DiLoCo loop. A
component implements `DiffusionComponent` in `yeto/components/base.py` and owns
only framework-specific behavior:

- resolve component paths/config,
- build the runtime pipeline,
- choose and expose trainable parameters,
- build the data loader,
- run one training step,
- build the learning-rate scheduler,
- save/export the final artifact.

`yeto/diffusion/learner.py` stays generic: it wraps the component model with
DDP/FSDP, builds the fragment layout, connects to the syncer, pushes/pulls
fragments, saves learner state, and writes layout metadata. It must not import
component packages directly.

## NAVA as a component

NAVA support is implemented as the built-in `nava` diffusion component, not as
a top-level task backend.

Typical launch forms:

```bash
# Shorthand via the structured model spec.
YETO_NAVA_BASE_CHECKPOINT=/models/nava/base.safetensors \
YETO_NAVA_ROOT=/opt/NAVA \
yeto launch \
  --gpu aws:8xa100@us-east-2,runpod:8xh100@CA \
  --model nava \
  --data /data/nava/train.jsonl \
  --adapter lora \
  --lora-r 16

# Equivalent explicit form.
yeto launch \
  --task diffusion \
  --component nava \
  --gpu aws:8xa100@us-east-2 \
  --component-root /opt/NAVA \
  --base-checkpoint /models/nava/base.safetensors \
  --data /data/nava/train.jsonl
```

Runtime defaults:

- `YETO_NAVA_ROOT` supplies a local NAVA checkout when `--component-root` is
  omitted. If neither is provided, the learner image must already contain the
  NAVA package importable by Python.
- `YETO_NAVA_BASE_CHECKPOINT` supplies `--base-checkpoint` for `--model nava`.
- `YETO_NAVA_LEARNER_IMAGE` supplies a runtime image for NAVA dependencies.
- `configs/nava.yaml` is the default component config relative to the component
  root.
- The default adapter is LoRA with `--lora-targets mmdit-all-linear`.

The `--data` value is passed to the component. Yeto core does not define a
NAVA-specific dataset conversion or dataset layout. Prepare data in the format
the selected component/runtime expects before launching.

## Export

LM export keeps the existing shape:

```bash
yeto-export --model qwen35-9b --checkpoint yeto-state.ckpt --output-dir out/
```

Diffusion export dispatches through the selected component:

```bash
yeto-export \
  --model nava \
  --checkpoint yeto-state.ckpt \
  --base-checkpoint /models/nava/base.safetensors \
  --output-dir out/ \
  --format lora
```

The syncer checkpoint may include layout metadata written by the learner. Export
uses it to validate task, component, trainable policy, fragment layout, tensor
names, shapes, and merge modes before applying fragments.

## Shape planner

`yeto shape`, `--budget`, and `--flops` remain LM-only. Diffusion/component
runs require an explicit `--gpu` fleet until there is a separate sizing model
for those components.
