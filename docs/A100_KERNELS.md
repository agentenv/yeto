# A100 causal training kernels

Yeto's causal-LM learner has an opt-in A100 kernel lane. The native path is
the default; optional kernels never install or activate unless requested.

The supported controls are:

| control | values | default | meaning |
|---|---|---|---|
| `--attention-backend` | `auto`, `sdpa`, `flash-attn-2` | `auto` | Hugging Face attention implementation |
| `--kernel-backend` | `native`, `liger` | `native` | SFT loss implementation; model layers remain native; fused requires LoRA/DDP |

An explicit request is fail-closed. A missing or differently-versioned
dependency, an unsupported model implementation, FlashAttention with FP32,
a custom loss on the Liger path, fractional token weights, or any drift in the
pinned Qwen2 apply-function controls produces an error instead of silently
selecting another implementation.

The kernel dependencies and the fused lane's PEFT compatibility point are
pinned. Kernel packages remain excluded from ordinary CPU/dev installs:

- `liger-kernel==0.8.0`
- `peft==0.19.1` for the fused-loss LoRA lane (native modes retain the normal
  project-wide PEFT compatibility range)
- `flash-attn==2.8.3`

Remote learner setup installs each lane's exact packages only when its
corresponding flag is selected. FlashAttention is built after the CUDA-matched
PyTorch wheel is installed and uses `--no-build-isolation` so the build can see
that wheel.

For example, an existing Qwen2/Qwen2.5 launch can opt into the isolated fused
linear cross-entropy loss while keeping SDPA and every model layer native:

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

The Liger path supports only Hugging Face `model_type="qwen2"` models (Qwen2
and Qwen2.5 in the pinned Transformers release). Yeto first loads the ordinary
`AutoModelForCausalLM`, then calls the pinned package's public
`apply_liger_kernel_to_qwen2` function with exactly:

```python
rope=False
cross_entropy=False
fused_linear_cross_entropy=True
rms_norm=False
swiglu=False
model=model
```

The call occurs before PEFT wraps the base model. Yeto requires the resulting
root binding to reference the exact `liger_kernel.transformers.model.qwen2`
fused forward shipped by the pinned package; another callable with a compatible
signature is rejected. Around the call, Yeto attests the Qwen2 module and class
bindings, functional cross-entropy, every concrete module instance and unique
module class present in the model, mutable built-in containers, config-like
objects (including root `config` and `generation_config` plus nested config
aliases), and custom Python objects with inspectable `__dict__` state. Opaque
Python leaves are covered by binding identity rather than an unqualified claim
about inaccessible internal state.

Every registered parameter, parameter gradient, and buffer is attested.
Gradient coverage distinguishes `None` from a tensor and includes exact tensor
identity. Tensor Python attributes and backward/post-accumulate hook bindings
and structures are included. Tensor attestation covers object identity, shape,
dtype, device, layout, stride, storage offset, data pointer, storage size,
`requires_grad`, mutation version, and a strong content digest. Content SHA-256
is computed in
bounded 4 MiB logical-order chunks; Yeto never retains a second copy of model
or gradient contents. CPU RNG state, all visible CUDA generator states when
CUDA is initialized, deterministic-algorithm mode, float32 matmul precision,
and relevant cuDNN, CUDA matmul, and SDPA backend flags are also captured. The
only accepted change across these explicitly attested surfaces is the exact
pinned fused forward bound to that one base-model instance.

Rejection is transactional where state is safely reversible: Yeto restores
the Qwen2 module/class bindings, every captured module-class namespace,
functional cross-entropy, model namespaces and exact module classes, original
gradient bindings, RNG states, and backend flags, then repeats the complete
attestation against the pre-apply snapshot. In-place tensor, gradient,
container, config, or inspectable-object contents cannot in general be
reconstructed without retaining duplicate contents. If any such state does
not match after rollback, `KernelIsolationError` marks the process poisoned.
Every `KernelIsolationError` is fatal for in-process continuation because it is
raised only after third-party code began executing; a verified rollback is
reported as evidence but is not permission for the learner or benchmark to
load another model or run another arm in that process.

The production-approved fused-loss envelope is currently narrower than the
native learner: it requires `--tuning lora --shard ddp`, a frozen and
unadapted `lm_head`, and FP32 for every trainable adapter parameter. The learner
and benchmark call the same post-PEFT validator, so target-selection drift
cannot silently adapt the output head. This lane requires exact PEFT 0.19.1;
the native learner's broader PEFT compatibility remains unchanged. Full tuning
and FSDP fail before launch/model loading and must use
`--kernel-backend native` until each has
separate real-CUDA loss, gradient, optimizer-delta, and distributed parity
evidence. This restriction is independent of whether a configuration might
appear to run; unsupported evidence profiles are never inferred safe.

Binary token masks are converted to labels with `-100` at ignored positions.
`num_items_in_batch=1` makes the pinned implementation return the required
local token sum without materializing logits, and `accum_dtype=torch.float32`
explicitly requests FP32 loss accumulation. `skip_logits=True` makes logit
elision explicit; materialized logits are still rejected. Fractional weights,
non-built-in losses, non-Qwen2 model types, wrapped/prepatched models, and
quantized bases must use `--kernel-backend native`.

This lane is scoped to BF16 and FP32 on A100. FlashAttention 2 is BF16-only;
FP32 measurements use SDPA. It does not enable FP8 or kernels tied to newer
GPU generations.

## Standalone benchmark

The benchmark does not create, modify, or terminate cloud resources. Run it
on an existing 8-GPU A100 node from the repository root:

```bash
pip install -e .
pip install 'liger-kernel==0.8.0' 'peft==0.19.1'
pip install 'ninja==1.13.0' 'packaging==25.0'
MAX_JOBS=8 pip install --no-build-isolation 'flash-attn==2.8.3'

torchrun --standalone --nproc_per_node=8 \
  scripts/benchmark_a100_kernels.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision main \
  --dtype bf16 \
  --tuning lora \
  --micro-batch-size 2 \
  --seq-len 1024 \
  --warmup-steps 5 \
  --steps 20 \
  --output a100-kernel-benchmark.json
```

The benchmark defaults to the production fine-tuning profile: LoRA rank 16,
alpha 32, and `--lora-targets auto`. Target resolution is shared with the
learner (`all-linear` for dense models and attention-only for MoE models), and
PEFT is applied before distributed setup. As in the learner, LoRA leaves the
frozen base unwrapped and manually averages only trainable adapter gradients;
DDP is used only by the explicit full-tuning profile. The frozen base remains
BF16, trainable adapters must remain FP32, and `lm_head` must remain frozen and
unadapted. Use `--tuning full` only with explicitly selected native-only
variants as a separately reported profile; the fused-loss arm rejects it. Do
not combine full-tuning evidence with default LoRA trials. The optimizer is
ordinary AdamW with the learner's default learning rate, `3e-4`, followed by
gradient clipping at norm `1.0`.

The publishable default matrix is component-isolated:

1. Native layers with SDPA and PyTorch cross-entropy—the parity reference.
2. Native layers with FlashAttention 2 and PyTorch cross-entropy, when installed and supported.
3. Native layers with SDPA and the instance-only Liger fused-linear-CE loss,
   only for the production LoRA/DDP profile when installed and supported.

The benchmark report records `layer_backend`, `loss_backend`, and
`loss_implementation` independently. It does not publish arms that change both
model layers and loss, nor an arm that combines FlashAttention with the fused
loss. Those combinations cannot attribute a parity or speed difference to one
component. To run only the fused-loss candidate and its automatically included
native-SDPA reference, pass `--variants fused-linear-ce-sdpa`.

Before parity, native SDPA performs exactly one controlled deterministic LoRA
update. This makes both `lora_A` and `lora_B` nonzero; the harness verifies and
records their nonzero coverage. That single native trainable state becomes the
anchor. Candidates never warm themselves: the exact native anchor is restored
into every model and verified by SHA-256 before any witness.

Every arm, including native SDPA, then runs the same two-witness self-repeat
procedure. Each witness starts at the exact anchor with a fresh optimizer, the
same batch, and the same RNG seed, and restores the anchor after observing its
parameter delta. The self-repeat must pass, and each candidate's first witness
must independently match the native reference. After parity, the anchor is
restored and verified once more and another fresh optimizer is constructed for
timing. No parity or controlled-warmup update leaks into timed model state.

Model and adapter construction use separate recorded initialization seeds, so
backend-specific model loading cannot perturb PEFT's adapter RNG stream. The
constructed trainable state and the warmed parity anchor are hashed separately;
a layout or digest mismatch is fatal before timing.

For both witnesses and timing, each rank contributes a token-sum loss. Target
counts are summed across the island, each local loss is scaled by
`world_size / global_target_tokens`, replicated LoRA gradients are manually
averaged, and trainable gradients are clipped at `1.0`. This produces the exact
island-global target-token mean despite unequal target counts between ranks.
The standalone benchmark deliberately fixes one microbatch per optimizer step:
it does not model the learner's multi-microbatch gradient accumulation or
learning-rate scheduler, and its report must not be treated as evidence about
those separate mechanisms.

Every available variant must match the reference loss and every element of
every trainable parameter gradient. Each applies one identical AdamW update
and must also match every resulting trainable parameter delta. Reference
gradients and deltas are kept in host memory and compared in bounded chunks.
The comparison never stops at the first failure: checked counts, first and
worst failing tensors, whole-model relative L2 error, cosine similarity,
allclose violation fraction, norms, scales, and nonfinite counts cover every
compatible finite tensor element. Structural coverage, finiteness, and numeric
scope are reported separately, so finite compatible-subset metrics remain
available even if other keys, shapes, or values fail. Numerically unevaluable
fields are JSON `null`, never `NaN` or infinity, and report serialization uses
strict RFC JSON.

The report also records actual and reference update norms, maximum magnitude,
nonzero count, and nonzero fraction. If BF16 quantization rounds every observed
update away, the delta witness is marked `not_meaningful` and cannot satisfy
the parity gate. Parameter deltas use their own scale-appropriate tolerances
(`--parameter-delta-rtol 5e-2` and `--parameter-delta-atol 1e-8` by default),
not the much larger gradient absolute tolerance. Any reference-control or
candidate mismatch is fatal and produces no performance result. Unavailable
optional packages or model implementations are recorded as skipped.

Each rank contributes a compact parity diagnostic. Rank zero records every
rank, every failing rank, per-rank reasons, the worst failing rank and reason,
and global worst/minimum metrics. A global failure is therefore never reported
alongside only rank-zero's local zero errors. Top-level `planned_variants`,
`completed_variants`, and `status` (`passed`, `failed`, or `incomplete`) define
the evidence boundary; fatal reports also include an explicit phase and reason.

For each passing variant, the JSON report contains:

- total model setup and correctness-validation time;
- first parity forward/backward compile time and first optimizer-step time;
- p50, p95, and mean synchronized step time;
- global raw and target tokens per second;
- maximum peak allocated and reserved memory per GPU across ranks;
- independent loss, gradient, and parameter-delta statuses plus full streamed
  parity and update-sensitivity diagnostics;
- tuning mode, requested and resolved adapter configuration, trainable dtype
  counts, deterministic initialization seed, and trainable-state SHA-256;
- GPU, CUDA, PyTorch, Transformers, PEFT, Accelerate, dependency-version,
  source-provenance, and command configuration metadata.

The harness resolves `--revision` (default `main`) through the Hub before any
model load, then gives every rank the returned immutable commit SHA. Both the
requested revision and resolved SHA are written to JSON; no timed run uses an
unrecorded moving model revision. Schema version 2 records the benchmark script
SHA-256, git object ID, dirty state, provenance source, and library versions. A
clean tree can be identified by its git object ID; a dirty tree cannot, so its
report explicitly sets `clean_commit_exact: false` and the script hash identifies
only the harness file, not every modified source file.

Sky-synchronized workdirs often omit `.git`. Source provenance is required, so
such a launch must pass both validated overrides; omitting either one is fatal:

```bash
export YETO_GIT_SHA=<40-or-64-character-git-object-id>
export YETO_GIT_DIRTY=false

torchrun --standalone --nproc_per_node=8 \
  scripts/benchmark_a100_kernels.py \
  --revision <40-character-model-commit-sha> \
  --output a100-kernel-benchmark.json
```

Set `YETO_GIT_DIRTY=true` if the synchronized source includes uncommitted
changes. For a Sky YAML, put the same two values in its `envs` mapping. The
harness validates the object ID and boolean rather than silently accepting
missing or malformed provenance.

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
