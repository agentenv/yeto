# A100 causal training kernels

Yeto's causal-LM learner has an opt-in A100 kernel lane. The native path is
the default; optional kernels never install or activate unless requested.

The supported controls are:

| control | values | default | meaning |
|---|---|---|---|
| `--attention-backend` | `auto`, `sdpa`, `flash-attn-2` | `auto` | Hugging Face attention implementation |
| `--kernel-backend` | `native`, `liger` | `native` | model layers and fused SFT loss |

An explicit request is fail-closed. A missing or differently-versioned
dependency, an unsupported model implementation, FlashAttention with FP32,
a custom loss on the Liger path, or fractional token weights produces an
error instead of silently selecting another implementation.

The optional dependencies are pinned and excluded from ordinary CPU/dev
installs:

- `liger-kernel==0.8.0`
- `flash-attn==2.8.3`

Remote learner setup installs either package only when its corresponding flag
is selected. FlashAttention is built after the CUDA-matched PyTorch wheel is
installed and uses `--no-build-isolation` so the build can see that wheel.

For example, an existing launch can opt into Liger while keeping SDPA:

```bash
yeto launch \
  --gpu aws:8xa100@us-east-2 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data org/chat-traces \
  --attention-backend sdpa \
  --kernel-backend liger
```

## Loss contracts

The native SFT path uses `torch.nn.functional.cross_entropy`, returns a local
token sum, and preserves arbitrary nonnegative per-token weights. Custom and
pickled losses continue to receive materialized logits exactly as before.

The Liger path uses `AutoLigerKernelForCausalLM` with fused linear
cross-entropy. Binary token masks are converted to labels with `-100` at
ignored positions. `num_items_in_batch=1` makes the pinned implementation
return the required local token sum without materializing logits. Fractional
weights and non-built-in losses must use `--kernel-backend native`.

This lane is scoped to BF16 and FP32 on A100. FlashAttention 2 is BF16-only;
FP32 measurements use SDPA. It does not enable FP8 or kernels tied to newer
GPU generations.

## Standalone benchmark

The benchmark does not create, modify, or terminate cloud resources. Run it
on an existing 8-GPU A100 node from the repository root:

```bash
pip install -e .
pip install 'liger-kernel==0.8.0'
pip install 'ninja==1.13.0' 'packaging==25.0'
MAX_JOBS=8 pip install --no-build-isolation 'flash-attn==2.8.3'

torchrun --standalone --nproc_per_node=8 \
  scripts/benchmark_a100_kernels.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision main \
  --dtype bf16 \
  --micro-batch-size 2 \
  --seq-len 1024 \
  --warmup-steps 5 \
  --steps 20 \
  --output a100-kernel-benchmark.json
```

The ordered comparison is:

1. Native layers with SDPA and PyTorch fused cross-entropy—the parity reference.
2. Native layers with FlashAttention 2 and PyTorch fused cross-entropy, when installed and supported.
3. Liger layers/fused loss with SDPA, when installed and supported.
4. Liger layers/fused loss with FlashAttention 2, when both are available.

Before timing, every available variant must match the reference loss and every
element of every trainable parameter gradient. Each variant then applies one
identical AdamW update and must also match every element of every resulting
parameter delta. Reference gradients and deltas are kept in host memory and
compared in bounded chunks. A mismatch is fatal and produces no performance
result for that variant. Unavailable optional packages or model implementations
are recorded as skipped in JSON.

For each passing variant, the JSON report contains:

- model/DDP setup time;
- first parity forward/backward compile time and first optimizer-step time;
- p50, p95, and mean synchronized step time;
- global raw and target tokens per second;
- maximum peak allocated and reserved memory per GPU across ranks;
- loss and gradient parity errors and tolerances;
- GPU, CUDA, PyTorch, dependency-version, and command configuration metadata.

The harness resolves `--revision` (default `main`) through the Hub before any
model load, then gives every rank the returned immutable commit SHA. Both the
requested revision and resolved SHA are written to JSON; no timed run uses an
unrecorded moving model revision.

One invocation is one timing trial. For three independent trials on the same
node, use distinct output files and trial indices:

```bash
for trial in 1 2 3; do
  torchrun --standalone --nproc_per_node=8 \
    scripts/benchmark_a100_kernels.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --revision <40-character-commit-sha> \
    --trial-index "$trial" \
    --output "a100-kernel-benchmark-trial-${trial}.json"
done
```

Each report states `timing_trials_in_record: 1`; aggregate records by
`trial.index`, not by treating the measured steps inside one invocation as
independent trials.

Use an explicit micro-batch for comparisons. If a variant OOMs, reduce the
micro-batch uniformly and rerun the whole matrix; do not compare variants at
different effective batches.
