# Ascend NPU islands (Huawei 910x)

Ascend cards run the ordinary `yeto.learner` — there is no separate backend
module. `torch_npu` registers an `npu` device type into torch, so the only
thing standing between the existing learner and an Ascend card was a handful
of calls that torch spells per-vendor rather than per-device. Those live in
`yeto/accel.py`.

## Why no peer backend

`yeto.mlx.learner` exists because MLX is a different tensor library: it needs
its own LoRA implementation (`yeto/mlx/lora.py`) purely to reproduce peft's
names, shapes and flatten order. Ascend needs none of that — it runs the same
`transformers`, the same `peft`, the same `yeto.data` and the same
`yeto.losses`. Forking a second copy of a 1300-line training loop to change
six function names would buy nothing and cost a permanent merge burden.

So the diff is: one new module plus the call sites that used to name CUDA.

## What `yeto/accel.py` owns

torch has no device-generic API for seeding, memory queries or the caching
allocator; `torch.cuda` and `torch.npu` just happen to spell them the same.
`accel` resolves the vendor namespace once and exposes what the learner needs:

| call | CUDA | Ascend | CPU |
| --- | --- | --- | --- |
| `dist_backend()` | `nccl` | `hccl` | `gloo` |
| `detect()` | `torch.cuda.set_device` | `torch.npu.set_device` | — |
| `manual_seed_all()` | `torch.cuda.*` | `torch.npu.*` | no-op |
| `mem_get_info()` | `torch.cuda.*` | `torch.npu.*` | `None` |
| `empty_cache()` | `torch.cuda.*` | `torch.npu.*` | no-op |
| `oom_error()` | `torch.cuda.OutOfMemoryError` | `torch.npu.OutOfMemoryError` | — |
| `fork_rng()` | CUDA generators | `device_type="npu"` | CPU only |

`register_backends()` imports `torch_npu`, which is what registers both
`torch.npu` and the `npu` device type; `torch.device("npu")` raises a bare
`Expected one of cpu, cuda, ...` before it runs, so `detect` imports first and
turns a missing extension into a message that names `torch_npu`.

## Initialization parity, and what it rests on

Every island must start from bit-identical trainable parameters — the syncer
cannot distinguish a mismatched LoRA init from a local update, and it would
corrupt the first merge. Two things have to agree for that, and only the first
one is free.

The random draw is free, because peft is applied while the model is still on
the CPU:

```python
model = get_peft_model(model, lora)   # kaiming lora_A off the CPU generator
...
model.to(device)                      # only now does the card see anything
```

The seeding at `learner.py` is `torch.manual_seed(args.seed)` on the CPU
generator, so the draw never touches an accelerator and cannot diverge between
families. `accel.manual_seed_all` seeds the device generator afterwards for
dropout, which is deliberately *not* required to match across islands.

What is **not** free is the dtype that draw is rounded into.
`load_model_and_tokenizer` selects fp32 when `--shard fsdp --tuning full` or
the device is not an accelerator, and bf16 otherwise; peft builds `lora_A`
against the base layer, so the init materializes in whichever dtype that branch
chose. Measured on a 910B4 (Qwen3-0.6B, same seed, sha256 over all 392
trainable tensors):

| load | fingerprint |
| --- | --- |
| NPU, bf16 base | `1c0dbb873c0bfbe5…` |
| CPU, bf16 base forced | `1c0dbb873c0bfbe5…` |
| CPU, fp32 base (the default off-accelerator) | `be129b1b18fc6643…` |

The device is irrelevant; the dtype branch is decisive. CUDA and Ascend both
take the bf16 branch, so a CUDA island and an Ascend island agree bit for bit —
which is the case this change exists to support. A CPU island, or one running
`--shard fsdp --tuning full`, does not agree with them. That is a pre-existing
property of the dtype rule rather than something Ascend introduced, but it is
the first thing to check if an initial merge ever looks wrong.

The layout side is equally free: `build_layout` packs fragments by name, and
the `layout_fingerprint` in HELLO covers merge modes, FQNs, order, numel and
shape (see [PROTOCOL.md](PROTOCOL.md)). Same `transformers` + same `peft` ⇒
same fingerprint, so a mismatch fails loudly at the handshake instead of
silently merging misaligned tensors.

The Rust syncer needed no changes at all: `syncer/src/merge.rs` operates on
flat f32 CPU buffers and never learns which accelerator produced them.

## What runs and what is refused

Refusals are load-time, not silent degradations — a run that starts must be
numerically comparable to a CUDA run of the same flags. ✅ below means "reaches
the card instead of being rejected", which is a claim about the code path, not
about hardware evidence; see [Validation status](#validation-status) for what has
actually been executed on a 910x.

| feature | on Ascend | where |
| --- | --- | --- |
| LoRA tuning, `--shard ddp` | ✅ | — |
| bf16 base + fp32 adapters | ✅ | `load_model_and_tokenizer` |
| SDPA attention (`--attention-backend auto`/`sdpa`) | ✅ | — |
| `--micro-batch-size auto` OOM probing | ✅ | `autobatch.py` |
| `--gradient-checkpointing auto` | ✅ | via `accel.mem_get_info` |
| multi-card islands under `torchrun` | ✅ (hccl) | `setup_distributed` |
| `--base-quantization nf4` | ❌ refused | bitsandbytes is CUDA-only |
| `--kernel-backend liger` | ❌ refused | Triton; `validate_kernel_request` |
| `--attention-backend flash-attn-2` | ❌ refused | `attention_load_kwargs` |
| `--shard fsdp` | ❌ refused | no validated sharding evidence |

The three CUDA-only kernel paths already refused every non-CUDA device before
this change, so they reject Ascend with their existing messages and needed no
new code. Only the FSDP message was reworded, since "cannot shard on cpu" is
no longer the only way to reach it.

## Running a pure-Ascend fleet

SkyPilot cannot provision Huawei Cloud or on-prem Ascend, so `yeto launch` is
not the entry point here: run the syncer and the learners directly.

```bash
# once, on any machine that can build Rust
cargo build --release --manifest-path syncer/Cargo.toml

# coordinator — M islands, quorum K
./syncer/target/release/yeto-syncer \
  --learners 2 --quorum 2 --total-steps 64 --port 29400

# each Ascend node (one island per node; 8 cards -> 8 ranks)
torchrun --nproc-per-node 8 -m yeto.learner \
  --model lfm25-1b --data org/chat-traces \
  --syncer <syncer-host>:29400 --learner-id 0 --num-learners 2 \
  --shard ddp --tuning lora
```

`--device` is not needed: `accel.detect` finds the card and binds this rank to
`LOCAL_RANK`. Pass `--device npu` only to force the family on a node that also
has an NVIDIA card. Learners dial out to the syncer, so Ascend nodes behind
NAT need no inbound rule — only outbound TCP to the syncer's port.

Two flags are worth knowing before the first run:

- **`--assistant-mask-mode legacy`** is required for any tokenizer whose chat
  template lacks `{% generation %}`, and Qwen3 is one of them. Without it
  `yeto.data` raises `ExactAssistantMaskError` rather than guess where the
  assistant span begins. This is not Ascend-specific — it bites identically on
  CUDA — but it is the first thing a Qwen run hits.
- **`ASCEND_RT_VISIBLE_DEVICES`** restricts which of the container's cards this
  process may touch, the way `CUDA_VISIBLE_DEVICES` does. Do not confuse it with
  `ASCEND_VISIBLE_DEVICES`, the Docker-runtime knob that decides which cards are
  passed into the container in the first place.

### Version matrix

`torch_npu` ships one build per torch minor version and per CANN release. Pick
the `torch` / `torch_npu` / CANN triple first; there is no combination to
discover later. bf16 is required — the training path loads the base in bf16,
which rules out pre-910B silicon.

`pyproject.toml` declares `torch>=2.4,<2.9`, and that upper bound is worth
reading in context: its stated reason is that cu13x wheels need an NVIDIA driver
>=580 while cloud AMIs ship 535. That is a CUDA wheel-availability constraint,
not a torch API one, so it says nothing about Ascend. The stack actually
exercised below sits above the bound and needed no shim anywhere in
`yeto/accel.py` or the learner:

| component | verified value |
| --- | --- |
| image | `quay.io/ascend/vllm-ascend:v0.23.0rc1` |
| arch | aarch64 (Kunpeng) |
| Python | 3.12.13 |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post2 |
| CANN, in container | 9.0.1 |
| CANN / driver, on host | 8.3.RC1 / 25.3.rc1 |
| transformers / peft | 5.13.0 / 0.19.1 |

Container CANN may differ from host CANN, as it does here: only the host
*driver* is shared into the container, and 25.3.rc1 lists C19–C23 as compatible.
So the toolkit version to match is the image's, not the node's. Running from
source (`PYTHONPATH=<repo>`) keeps pip from enforcing the declared torch bound;
relaxing that bound for Ascend would be defensible but is not done here.

## Not implemented

Deliberately out of scope for this change, listed so the boundary is explicit:

- **Heterogeneous NVIDIA + Ascend fleets.** The wire protocol and fragment
  layout already permit it — that was the point of keeping one learner — but
  a mixed run needs the `--external-learners` join path generalized beyond its
  hardcoded `python -m yeto.mlx.learner`, plus grace-window tuning for the
  step-time gap between families (`--grace-tau`, `--sync-interval-steps`).
  Nothing here forecloses it.
- **`npu_fusion_attention`.** Ascend's fused attention kernel would need a new
  `--attention-backend` value and an NPU analogue of the CUDA-flag attestation
  in `causal_kernels.py`. Until then Ascend runs SDPA.
- **FSDP on NPU**, **Megatron/MindSpeed**, **the diffusion backend**, and the
  **`yeto shape` ILP planner** (which has no Ascend offering catalog). Ascend
  islands are always declared explicitly.

These three fail in three different ways, which is worth knowing before pointing
one of them at a 910x. `--shard fsdp` is refused at load with a message naming
the device. `yeto.megatron.learner` raises, because it hardcodes
`torch.cuda.set_device` and `backend="nccl"`. `yeto.diffusion.learner` does
neither: its device selection is an `elif torch.cuda.is_available()` chain, so on
an Ascend node it silently falls through to the CPU and `gloo` and trains at
unusable speed without warning. That is pre-existing behaviour on any non-CUDA
host rather than something introduced here, and it was left alone to keep this
change inside its stated boundary — but it does contradict the load-time-refusal
principle above, and it is the one gap worth closing first if diffusion on
Ascend ever becomes interesting.

## Validation status

- ✅ Unit tests (`tests/test_accel.py`): family resolution and priority,
  `hccl`/`nccl`/`gloo` selection, `LOCAL_RANK` binding, the missing-`torch_npu`
  message, per-device seeding/memory/allocator/OOM dispatch, and that
  `fork_rng` names the family only off CUDA (torch <2.7 has no such kwarg).
  The Ascend namespace is stubbed, since `torch.device("npu")` cannot be
  constructed without the extension.
- ✅ Existing suites unaffected (`tests/test_autobatch.py`,
  `tests/test_learner_units.py`, `tests/test_causal_kernels.py`).
- ✅ **On hardware**: 8× Ascend 910B4-1 (64 GB, aarch64), in the container stack
  tabulated above. Confirmed there:
  - `torch_npu` registers the family, `torch.npu.device_count()` reports 8, and
    `accel.detect` binds a rank to its card.
  - `load_model_and_tokenizer` places Qwen3-0.6B on the NPU with a bf16 base and
    fp32 adapters — 392 trainable tensors, names and shapes identical to a CPU
    load, weights bit-identical at matching dtype (see the parity table above).
  - SDPA runs on the NPU across every shape the learner asks of it: forward and
    backward, bf16 and fp32, `is_causal` and an explicit additive mask, batch 1
    and 8, `--seq-len` 512 and 1024. `npu_fusion_attention` is not needed to
    train.

  Parity was checked directly, by hashing trainable tensors from a CPU load and
  an NPU load of the same seed. `scripts/check_name_parity.py` is not the tool
  for this: it compares torch against MLX and imports `mlx`, so it cannot run on
  an Ascend node at all.

  Everything above ran single-process with `--syncer none`. No part of the wire
  protocol — HELLO, the `layout_fingerprint` handshake, PUSH/PULL, merging — has
  been exercised from an Ascend learner.
- ⛔ **The training loop itself has not completed a run.** Two attempts stopped
  early, neither of them inside the accelerator abstraction:
  1. A co-tenant `vllm serve` held all 8 cards at 97% HBM, leaving ~345 MB free.
     The shortage surfaced *inside* SDPA as `aclnnFlashAttentionScore` rather
     than as a clean OOM. Worth remembering: an acl operator error on a busy
     card usually means memory, not an unsupported operator — check `npu-smi`
     before suspecting the op.
  2. `ExactAssistantMaskError` from Qwen3's chat template; fixed by
     `--assistant-mask-mode legacy` as described above.
- ⛔ **Still unverified.** Once cards are free, run in this order:
  1. Single-card smoke to completion: loss decreasing, then adapters saved and
     re-loaded through `PeftModel.from_pretrained`.
  2. Multi-card under `torchrun --nproc-per-node 8`. This is the first real
     exercise of `hccl`, through `dist.all_reduce` (gradient averaging, token
     counts) and `dist.broadcast_object_list` (metadata to non-zero ranks) — the
     two collectives most likely to need attention, and the largest remaining
     risk in this change.
  3. `--micro-batch-size auto` on an idle card, so the probe's OOM path is
     measured against real free memory instead of a co-tenant's leftovers.
  4. Two-node run with `--quorum 2`; the event tape must show both islands as
     responders on every merge.
