# Causal LoRA adapter lifecycle

Yeto supports two ways to start causal-LM LoRA training from an existing
adapter, followed by a safe deployment export:

- `--resume-from` continues from a Yeto adapter only when its immutable base
  model, dataset, trust setting, and recorded training recipe match the new
  run. Any drift fails before model loading or GPU training.
- `--branch-from` starts a new lineage from a safe PEFT adapter. It permits an
  intentional data or recipe change, while still requiring the same base
  model and compatible LoRA rank/alpha.
- `yeto merge` safely folds the adapter into its immutable base and writes a
  standard Hugging Face model in configurable SafeTensors shards.

Both training modes load the adapter as trainable. They do not merge it into
the base before training.

This lifecycle currently applies to the causal-LM SFT path. Miles RL rejects
these flags rather than silently ignoring a parent adapter.
## Resume versus checkpoint recovery

Adapter resume and distributed checkpoint recovery solve different problems.
The syncer checkpoint that every launch maintains automatically restores an
interrupted run's coordinator state. `--resume-from` starts a new run at the
supplied LoRA weights with a fresh optimizer and scheduler. It is useful for
extending a completed training stage, but it is not bit-identical mid-step
recovery.

Use `--branch-from` when changing data, sequence length, loss/masking,
normalization, optimizer settings, batching, kernels, or the fragment/merge
recipe. The loaded adapter's LoRA structure is inherited. Rank and alpha
describe the existing tensors and therefore cannot be changed by either mode;
create a new adapter to change its structure.

## Launch examples

Continue the exact recorded recipe from a local Yeto artifact:

```bash
yeto launch \
  --gpu aws:8xh200@us-east-2 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data org/training-data \
  --resume-from ./artifacts/stage-1 \
  --tuning lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --shard ddp
```

Start a new stage from the same weights with a changed dataset or recipe:

```bash
yeto launch \
  --gpu aws:8xh200@us-east-2 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data org/new-training-data \
  --branch-from ./artifacts/stage-1 \
  --tuning lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --shard ddp
```

Local adapter directories are hashed before cloud spend and mounted onto each
learner. Cloud object-store sources require the expected digest explicitly:

```bash
yeto launch ... \
  --branch-from s3://my-bucket/adapters/stage-1 \
  --adapter-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Only SafeTensors PEFT adapters are accepted. Hub adapter IDs are not accepted
directly yet; download the adapter to a local directory first.

## Merge and shard for deployment

Merge a local adapter into the base recorded in `yeto_provenance.json`:

```bash
yeto merge \
  --adapter-dir ./artifacts/stage-2 \
  --output-dir ./artifacts/stage-2-merged \
  --device cuda \
  --dtype bf16 \
  --max-shard-size 2GB
```

For a reviewed legacy PEFT adapter without Yeto provenance, pass the base
explicitly with `--model` and optionally `--model-revision`. The output
directory must not already exist. Yeto calls PEFT's safe merge, writes only
SafeTensors model weights, passes `--max-shard-size` to the Hugging Face save
path, records the parent adapter digest and export settings, and atomically
publishes the completed directory.

The shard size limits each model weight file; it does not divide the model
across GPUs at inference time. Runtime tensor/pipeline sharding remains the
responsibility of the inference system.

## Artifact lineage

Training outputs include a `training_recipe` object in
`yeto_provenance.json`. Child outputs also include `parent_adapter` with the
resume/branch mode, SHA-256 digest, source model and dataset provenance,
source recipe, and adapter configuration. Merged exports record the same
parent digest plus the safe-merge, SafeTensors, and maximum-shard settings.

The directory digest covers relative file names and bytes and rejects symbolic
links. This catches mutable or inconsistent adapter contents across launch and
training boundaries.
