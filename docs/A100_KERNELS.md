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
pinned Qwen2 source hash, forward signature, or fused-loss primitive contract
produces an error instead of silently selecting another implementation.

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
and Qwen2.5 in the pinned Transformers release). Before PEFT wraps the ordinary
`AutoModelForCausalLM`, Yeto verifies `liger-kernel==0.8.0`, imports
`liger_kernel.transformers.model.qwen2.lce_forward`, and requires the installed
Qwen2 source module to match SHA-256
`6df24ecc3d259e84cd833a62d9e08aea2db278ddc97af14fc27dc61b9259e6ed`.
It verifies the `labels`, `use_cache`, `skip_logits`, and variadic-keyword
forward contract plus explicit `accum_dtype` support on the fused primitive.
Yeto then performs one local
`model.forward = types.MethodType(lce_forward, model)` assignment. It never
calls the package-wide `apply_liger_kernel_to_qwen2` callback, so that callback
has no authority to rewrite layer classes or process globals in this lane.

Bounded before/after attestation surrounds that direct assignment to detect
accidental state or API drift. It covers Qwen2 module/class bindings,
functional cross-entropy, every concrete module instance and unique module
class present in the model, mutable built-in containers, config-like objects
(including root `config` and `generation_config` plus nested config aliases),
and custom Python objects with inspectable `__dict__` state. Opaque Python
leaves and the forward function are checked for binding identity; identity is
a drift invariant, not proof of semantic integrity.

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
only intended change across these explicitly attested surfaces is the
hash-locked fused forward binding on that one base-model instance.

Rejection is transactional where state is safely reversible: Yeto restores
the Qwen2 module/class bindings, every captured module-class namespace,
functional cross-entropy, model namespaces and exact module classes, original
gradient bindings, RNG states, and backend flags, then repeats the complete
attestation against the pre-binding snapshot. In-place tensor, gradient,
container, config, or inspectable-object contents cannot in general be
reconstructed without retaining duplicate contents. If any such state does
not match after rollback, `KernelIsolationError` marks the process poisoned.
Every `KernelIsolationError` is fatal for in-process continuation because it is
raised only after direct binding began; a verified rollback is
reported as evidence but is not permission for the learner or benchmark to
load another model or run another arm in that process.

This attestation is deliberately not a sandbox against a malicious or
compromised Python dependency, import-time code, concurrent mutation, native
extensions, runtime function `__code__` tampering, modified builtins, or
arbitrary inherited framework globals outside the stated surfaces. Runtime
enforcement covers the exact package version and approved Qwen2 source-module
SHA-256. Supply-chain trust for the remaining dependency, import, and runtime
surface must come from the trusted build and provisioning pipeline; attestation
provides bounded accidental-drift detection inside that boundary.

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
PEFT is applied before any DDP wrapper is constructed. As in the learner, LoRA
leaves the frozen base unwrapped and manually averages only trainable adapter
gradients; DDP is used only by the explicit full-tuning profile. The frozen base remains
BF16, trainable adapters must remain FP32, and `lm_head` must remain frozen and
unadapted. Use `--tuning full` only with explicitly selected native-only
variants as a separately reported profile; the fused-loss arm rejects it. Do
not combine full-tuning evidence with default LoRA trials. The optimizer is
ordinary AdamW with the learner's default learning rate, `3e-4`, followed by
gradient clipping at norm `1.0`.

Publishable benchmark invocations are deliberately limited to one selected
reference plus at most one candidate. `--reference-variant` defaults to
`native-sdpa`; when `--variants` is omitted (or empty), only that reference runs.
Supplying one candidate name through `--variants` automatically prepends the
selected reference. `all` and comma-separated multi-candidate matrices are
rejected. This keeps every JSON record independently interpretable and prevents
a later arm from inheriting unreported process-global backend state.

Every report records `attention_backend`, `internal_sdpa_backend`,
`layer_backend`, `loss_backend`, and `loss_implementation` independently. A
loss-implementation comparison is accepted only when the reference and
candidate have identical attention, internal-SDPA, and layer backends. This
prevents an attention change from contaminating fused-loss attribution. Use
`native-sdpa-math` versus `fused-linear-ce-sdpa-math` for the correctness
comparison. Only after forced cuDNN has its own attribution, determinism, and
three-seed parity evidence should `native-sdpa-cudnn` versus
`fused-linear-ce-sdpa-cudnn` be used for the performance comparison. No
publishable arm changes both the high-level attention implementation and the
loss implementation.

The output path is also part of the evidence boundary. Before model loading or
benchmark work, rank zero atomically reserves a path that does not already exist
and every rank receives the reservation result. A pre-existing file, directory,
or symlink is refused. A git-tracked path is also refused when its worktree file
is deleted, so recreating and excluding the generated sentinel cannot hide a
tracked deletion. The harness never silently overwrites an earlier report.
The reservation is a strict schema-version-4 JSON sentinel with
`status: incomplete` and `report_complete: false`. A normal final publication
atomically replaces it. Thus an unexpected early failure or a failed final
write cannot expose an earlier passing report under the new run's output path.
Use a fresh output name for every invocation. When that generated path is inside
the repository, only the reserved output itself is excluded from the subsequent
git dirty-state query; the exact excluded relative path is recorded, and every
other tracked or untracked worktree change still marks the source dirty.

If final serialization, replacement, or directory syncing fails, rank zero
makes a separate best-effort attempt to restore an incomplete schema-version-4
sentinel derived from the original reservation. The sentinel records the
publication error and resets `final_report_written` to false. Rank zero then
reads the visible target back and broadcasts whether the expected reservation
is visibly non-passing. A successful benchmark therefore requires all three:
a complete passing JSON, a successful collective publication decision, and a
zero process exit. JSON status alone is never sufficient evidence.

The native PyTorch SDPA variants are:

| variant | internal selector | meaning |
|---|---|---|
| `native-sdpa` | `auto` | backward-compatible reference with every public PyTorch SDPA backend enabled |
| `native-sdpa-flash` | `flash` | exact `SDPBackend.FLASH_ATTENTION` selector |
| `native-sdpa-math` | `math` | exact `SDPBackend.MATH` selector |
| `native-sdpa-efficient` | `efficient` | exact `SDPBackend.EFFICIENT_ATTENTION` selector |
| `native-sdpa-cudnn` | `cudnn` | exact `SDPBackend.CUDNN_ATTENTION` selector |

The other candidate names are `native-flash-attn-2`,
`fused-linear-ce-sdpa-math`, and `fused-linear-ce-sdpa-cudnn`. A forced
internal backend can also be the reference, so its self-repeat and timing do
not depend on the automatic selector passing.
For a standalone forced-cuDNN run whose parity witness uses the complete timing
shape:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/benchmark_a100_kernels.py \
  --reference-variant native-sdpa-cudnn \
  --micro-batch-size 2 \
  --seq-len 1024 \
  --parity-micro-batch-size 2 \
  --parity-seq-len 1024 \
  --revision <40-character-model-commit-sha> \
  --output a100-native-sdpa-cudnn-standalone.json
```

The explicit parity batch and sequence arguments above are important when the
claim being tested is production-shape self-repeat. A forced-math standalone
run changes only the reference name to `native-sdpa-math`.

For comparison against the default automatic reference, use the forced backend
as the optional candidate instead:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/benchmark_a100_kernels.py \
  --variants native-sdpa-flash \
  --revision <40-character-model-commit-sha> \
  --output a100-native-sdpa-flash.json
```

The ordered comparison within any record is therefore:

1. The explicitly selected parity reference (automatic native SDPA by default).
2. Zero or one explicitly selected candidate.

### Internal SDPA selection and attribution

Each arm holds a public `torch.nn.attention.sdpa_kernel` context for its entire
lifetime, including model setup, parity, attribution, timing, and cleanup. Arm
activation and restoration are explicit all-rank gates; activating a new arm
never closes a previous arm implicitly. Every rank contributes its before,
expected-active, active, pre-restoration, after, restoration, and error
evidence. A selector mutation observed at the end of an otherwise passing arm
invalidates that arm even when the outer context subsequently restores it. Rank
zero cannot serialize a passing record until every rank has restored the selector,
and one final all-rank inactivity audit runs before report serialization. The
harness snapshots all four public CUDA enable flags before entry, verifies the
exact active flag set, and verifies exact restoration on exit. A leak is fatal,
removes any timing metrics from that arm, and remains fatal even if an emergency
repair restores the original flags. No private dispatcher or
implementation-detail selector is used; the mapping is compatible with the
public PyTorch 2.5.1 API.

Parity alone does not prove which fused attention operator executed. After
parity passes and before timing starts, the harness runs two untimed attribution
probes: one with the parity shape and one with the timing shape. A temporary
recorder observes every call to the public functional SDPA entry point and
retains no tensors. It records every unique input signature without truncation:
query/key/value/mask shapes, strides, dtypes, devices, layouts, gradient and
contiguity state, storage offsets, dropout, causal and GQA flags, explicit
scale, grad/autocast state, call counts, and the result of each public
`can_use_*_attention(SDPAParams(...))` capability predicate.

At the same time, `torch.profiler` records the actual primary aten operator:
Flash, math, memory-efficient, or cuDNN SDPA. The profiler's generic call count,
primary-backend call count, and functional recorder count must match exactly.
The forward probe has an explicit ATen allowlist containing only the generic
SDPA entry point and those four primary forward operators. Any other ATen
attention operator—including an unknown extension, a second wrapper, or a
backward operator—is fatal. CUDA kernel-event names remain recorded as evidence
but are not confused with ATen dispatch operators.
An exact selector must agree with the observed operator and be eligible for
every unique input signature. Because aggregate profiler events do not provide
a trustworthy per-call signature/operator mapping, the automatic reference is
deliberately stricter: each probe must observe exactly one recognized backend,
and that backend must be eligible for every recorded signature. Mixed automatic
dispatch fails closed. Every rank contributes its full evidence; normalized
input signatures, observed backends, primary operator counts, and relevant
model-state inputs must agree across all ranks. Missing calls, an unknown
operator, mixed-backend automatic dispatch, mixed-rank dispatch, selector
disagreement, or incomplete profiler coverage is fatal and produces no timing
metrics.

Each attribution probe is a rank-local, grad-enabled forward through the
unwrapped model. Its caught failure region contains no backward pass, optimizer,
DDP operation, manual gradient all-reduce, barrier, or other distributed
collective. Once every rank has exited that local region, one common evidence
gather applies the all-rank gate. The ordinary parity witnesses separately cover
backward gradients and optimizer updates.

This guarded probe does not imply end-to-end recovery from arbitrary distributed
faults. Parity and timed training necessarily execute NCCL collectives. An
unexpected device, process, or communication fault inside one of those
collectives is fail-stop: the launcher/process-group failure is authoritative,
and a complete failure JSON is not guaranteed. Before durable final replacement,
the reserved non-passing sentinel remains authoritative. If the result broadcast
or launcher fails after durable replacement, a complete JSON may remain visible,
but it is invalid without the successful collective decision and zero process
exit. The process group uses a bounded 300-second timeout by default;
experiments may set `--distributed-timeout-seconds` between 1 and 3600. This
timeout configuration is recorded in the report and does not add work to
measured steps. The harness does not claim collective rollback, retry, or
continued timing after such a fault.

The probe restores the exact warmed trainable anchor, every named buffer, and
both CPU and local-CUDA default RNG states. Frozen-parameter and
registered-buffer evidence covers registration names, local object and module
identity, registration order, tensor and module aliasing, qualified object type,
device, layout, stride, storage offset, storage size and identity, dtype, shape,
gradient state, and exact logical bytes. Buffer evidence additionally covers the
complete persistent-versus-nonpersistent registration sets. Buffer value
mutation during a forward is
permitted only when the original registrations, objects, metadata, aliases,
persistence, and values can all be restored. A frozen-parameter mutation or
replacement is fatal. Device indices and storage pointers are retained for exact
local verification and normalized only in the separate cross-rank digest;
storage sizes, types, and alias relationships must still agree across ranks.
Exact local verification retains and compares the actual Python class objects
for modules, parameters, and tensors. Serialized and cross-rank evidence uses
stable qualified class-name strings because class-object identity is necessarily
process-local.
Arbitrary unregistered Python attributes cannot be enumerated reliably; the
supported contract therefore requires a model invoked with `use_cache=False`
not to mutate unregistered
Python-side cache state. Such model implementations need a model-specific state
adapter before their results are publishable.

The timing-shape attribution forward executes before performance timing and can
populate exact-shape SDPA/cuDNN plan and allocator caches. Steady-state p50,
p95, mean, and throughput metrics remain intentionally warm measurements. The
first timed full update is therefore reported as
`first_post_attribution_training_step_seconds`; it includes forward, backward,
distributed gradient work, clipping, and the optimizer application. It is not
cold-start or first-use compilation latency.

Before parity, the selected reference performs exactly one controlled
deterministic LoRA update. This makes both `lora_A` and `lora_B` nonzero; the
harness verifies and records their nonzero coverage. That single reference
trainable state becomes the anchor. Candidates never warm themselves: the exact
reference anchor is restored into every model and verified by SHA-256 before
any witness.

Every arm, including the selected reference, then runs the same two-witness
self-repeat procedure. Each witness starts at the exact anchor with a fresh
optimizer, the same batch, and the same RNG seed, and restores the anchor after
observing its parameter delta. The self-repeat must pass, and each candidate's
first witness must independently match the selected reference. After parity,
the anchor is restored and verified once more and another fresh optimizer is
constructed for timing. No parity or controlled-warmup update leaks into timed
model state.

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

Every selected candidate must match the reference loss and every element of
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
strict RFC JSON. Rank zero writes through an fsynced temporary file followed by
an atomic replacement and an fsync of the containing directory, then broadcasts
the publication result. Every rank exits with failure if strict serialization or
the write fails; no rank waits in a post-write barrier that another rank can
miss. This guarantee applies to handled final publication failures, not to a
complete failure report after an unexpected fault inside an in-flight
distributed collective.

The result broadcast occurs after durable final replacement. If that broadcast
fails, rank zero cannot safely roll back a report that may already have been
observed, and the complete JSON can remain visible. This is why evidence
acceptance requires the collective/launcher outcome in addition to JSON fields.

No user-space protocol can guarantee visible recovery when the persistent
filesystem itself continues rejecting writes, renames, syncs, or reads. In that
case the collective result reports that non-passing recovery was not verified,
the process exits unsuccessfully, and any visible target must be treated as
untrusted regardless of its JSON status. The recovery attempt and read-back
verification narrow the post-replacement failure window; they do not claim
durability or rollback through a persistent storage failure.

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
- first parity forward/backward compile time and first post-attribution full
  training-step time (explicitly warm, not cold-start latency);
- p50, p95, and mean synchronized step time;
- global raw and target tokens per second;
- maximum peak allocated and reserved memory per GPU across ranks;
- independent loss, gradient, and parameter-delta statuses plus full streamed
  parity and update-sensitivity diagnostics;
- parity-shape and timing-shape SDPA input signatures, public selector
  eligibility, profiler operator counts, selector/operator agreement, and
  complete per-rank attribution evidence;
- per-rank before/active/after public SDPA flag snapshots, activation errors,
  exact restoration status, and the process-finalization inactivity audit;
- exact trainable/frozen-parameter and named-buffer restoration evidence plus
  the explicit unregistered-Python-state support boundary;
- tuning mode, requested and resolved adapter configuration, trainable dtype
  counts, deterministic initialization seed, and trainable-state SHA-256;
- GPU, CUDA, PyTorch, Transformers, PEFT, Accelerate, dependency-version,
  source-provenance, and command configuration metadata.

The harness resolves `--revision` (default `main`) through the Hub before any
model load, then gives every rank the returned immutable commit SHA. Both the
requested revision and resolved SHA are written to JSON; no timed run uses an
unrecorded moving model revision. Schema version 4 includes the publishable
selectable-reference/single-candidate contract and mandatory two-shape,
all-rank SDPA attribution evidence. It also records the explicit all-rank
selector lifecycle, strict forward-only ATen allowlist, exact registered-state
scope, normalized cross-rank state digests, collective atomic publication, and
the full-training-step timing name. It also distinguishes the non-passing output
reservation from a complete final report and records the bounded fail-stop
process-group contract without claiming collective recovery. It records the
benchmark script SHA-256, git object ID, dirty state, provenance source, and
library versions. A
clean tree can be identified by its git object ID; a dirty tree cannot, so its
report explicitly sets `clean_commit_exact: false` and the script hash
identifies only the harness file, not every modified source file.

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
    --variants native-sdpa-flash \
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
