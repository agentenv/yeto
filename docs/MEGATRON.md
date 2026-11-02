# Megatron-Core island backend

## Why

The torch backend (`yeto.learner`, FSDP2/DDP) trains any bf16 model that
fits an island, but it shards a MoE base as flat parameters and all-gathers
a full transformer block's experts every layer. For a 1T+-parameter MoE
(DeepSeek-V4-Pro class) that is memory-feasible only across a fragile
multi-node island and is throughput-poor. The industry answer is **expert
parallelism**: shard whole experts across ranks and route tokens to them with
all-to-all, keeping the block-quantized/large expert tensors intact and the
all-gather off the hot path.

## Decision: native Megatron-Core, not a wrapper

yeto's learner must hook the training loop at outer-step boundaries to
pack/push/pull/apply LoRA adapter fragments to the syncer. ms-swift and NeMo
own their trainers and assume synchronous DDP-mean gradient sync, so DiLoCo
would be an injected callback fighting their loop. Native Megatron-Core keeps
the loop ours — it is "yeto.learner, with Megatron-Core parallelism under the
frozen base instead of FSDP2." We vendor/adapt (Apache-2.0, with attribution)
the painful pieces from Megatron-Bridge / ms-swift:

- **HF → Megatron-Core weight conversion** (Megatron-Bridge `AutoBridge`) for
  deepseek_v3/v4 and qwen3_moe.
- **LoRA-on-MoE adapter specs** (`share_expert_adapters`, adapter module
  classes) so adapters land on attention/dense projections, not routed
  experts.

## Architecture

```
yeto launch --island-backend megatron --gpu aws:3x8xB200@us-east-2 \
    --expert-parallel 24 --tensor-parallel 1 --pipeline-parallel 1 ...
        head (syncer + fleet controller)      [unchanged]
          └── island: 3 nodes x 8 B200, one torchrun job
                yeto.megatron.learner
                  ├── init parallel state: EP=24, TP=1, PP=1
                  ├── build mcore MoE model, import HF bf16 weights
                  ├── freeze base, attach LoRA (attention/dense only)
                  ├── inner loop: get_forward_backward_func + DistributedOptimizer
                  └── DiLoCo sync at outer-step boundaries  [reuses yeto sync]
```

The **DiLoCo bridge is backend-agnostic**: the syncer only ever moves the
LoRA adapters, which are ordinary replicated tensors (attention/dense adapters
are replicated across EP ranks — they are not expert weights). So the megatron
learner builds a `{canonical_name: adapter_tensor}` dict and reuses yeto's
existing `build_layout` / `pack_fragment` / `SyncerClient` unchanged. Only the
model construction, weight import, adapter attachment, inner step, and adapter
enumeration are Megatron-specific.

## Parallelism mapping (from the EP research)

- **Experts** → EP (shard whole experts; keep expert-tensor-parallel = 1 so a
  quantized/large expert tensor is never split within itself).
- **Attention / dense / embeddings** → TP (and PP for depth) or replicate.
- Keep EP × TP inside one 8-GPU NVLink domain where possible.
- Since we LoRA (tiny trainable state, no base optimizer states), memory is
  dominated by the frozen weights, so lower PP / no TP than a full-training
  recipe usually suffices.

## Status

Implemented:

- `--island-backend {torch,megatron}` selector and
  `--expert/tensor/pipeline-parallel` (EP defaults to filling the island).
- The Megatron island task: `yeto.megatron.learner` entrypoint. With
  `--island-backend megatron` the image defaults to `MEGATRON_IMAGE` (an NGC NeMo
  container that ships torch, Transformer Engine, megatron-core and
  megatron-bridge as one consistent stack); the container-path setup installs
  only yeto's pure-Python dependencies with `--no-deps`.
- `yeto/megatron/learner.py`: parallel-state init, `AutoBridge` HF→mcore import,
  `LoRA(share_expert_adapters=True)` attach and freeze, mcore DDP +
  `DistributedOptimizer`, `get_forward_backward_func` inner loop, adapter
  enumeration (`linear_in`/`linear_out`), and the DiLoCo sync reusing yeto's
  `build_layout`/`pack_fragment`/`apply_fragment`/`SyncerClient` with the torch
  learner's counter and α-blend semantics.

Verified:

- The trainer imports and advances through `megatron.core`, Transformer Engine,
  mcore DDP and the optimizer, up to the `AutoBridge` import.
- A pip install outside a container does not work: Transformer Engine needs the
  prebuilt `transformer-engine-cu12` wheel (the `transformer-engine[pytorch]`
  source build cannot see torch), and `megatron-bridge` has no prebuilt wheel
  and its source build fails the same way. Hence the container default.

Not verified:

- A full training run. Open items: `AutoBridge.to_megatron_model` /
  `save_hf_pretrained` signatures, the mcore `GPTModel` forward arguments
  (position_ids / labels), and the DiLoCo bridge under real EP.
- NGC registry authentication on the island host (`nvcr.io/nvidia/nemo` needs a
  docker login with an NGC API key).
- The host GPU driver: a container shares the host driver, and B200 needs an
  open-kernel driver >= 570 on the sky docker host image.
- Passing the host NVMe into the container (the model download currently uses
  the container disk).
- Whether the container's bridge-pinned transformers version suits newer
  architectures.

Not implemented:

- EP-aware sizing in the shape planner.
- TP>1 / PP>1 adapter gather (refused with a clear error; TP shards
  `linear_in`/`linear_out`, PP splits adapters across stages).
