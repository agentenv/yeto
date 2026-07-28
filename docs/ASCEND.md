# Ascend NPU islands (Huawei 910x)

Ascend cards run the ordinary torch learners: `yeto.learner` for causal LMs and
`yeto.diffusion.learner` for diffusion. There is no separate Ascend backend
module. `torch_npu` registers an `npu` device type into torch, so the vendor
calls shared by both learners live in `yeto/accel.py`.

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
| causal-LM `--micro-batch-size auto` OOM probing | ✅ | `autobatch.py` |
| `--gradient-checkpointing auto` | ✅ | via `accel.mem_get_info` |
| multi-card islands under `torchrun` | ✅ (hccl) | `setup_distributed` |
| Diffusers LoRA, raw images/video, `--shard ddp` | ✅ | `yeto.diffusion.learner` |
| Diffusers BnB/NF4 pipelines | ✅ | `yeto.diffusion.quantization` |
| diffusion `--shard fsdp` | ✅ (FSDP2/hccl) | `maybe_wrap_for_distributed` |
| diffusion adapter reload and image/video sampling | ✅ | `yeto.diffusion.sample` |
| causal-LM `--base-quantization nf4` | ❌ refused | `validate_base_quantization` |
| `--kernel-backend liger` | ❌ refused | Triton; `validate_kernel_request` |
| `--attention-backend flash-attn-2` | ❌ refused | `attention_load_kwargs` |
| causal-LM `--shard fsdp` | ❌ refused | no validated sharding evidence |

The causal-LM NF4, Liger, and FlashAttention paths still reject every non-CUDA
device. Diffusion quantization is a separate Diffusers load path: it is enabled
only when the installed bitsandbytes explicitly reports NPU support. The
compatibility context supplies Diffusers 0.39's missing NPU device map during
load and restores its original quantizer hooks afterwards.

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
| transformers / peft / diffusers / bitsandbytes | 5.13.0 / 0.19.1 / 0.39.0 / 0.50.0 |

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
- **Causal-LM FSDP on NPU**, **Megatron/MindSpeed**, and the **`yeto shape` ILP
  planner** (which has no Ascend offering catalog). Ascend islands are always
  declared explicitly.

`yeto.learner --shard fsdp` is refused at load with a message naming the
device. `yeto.megatron.learner` raises, because it still hardcodes
`torch.cuda.set_device` and `backend="nccl"`. In contrast,
`yeto.diffusion.learner --shard fsdp` accepts CUDA or NPU accelerators and has
two-card FSDP2/HCCL evidence below. Generic Diffusers image, video, quantized,
DDP, and FSDP paths have now been exercised on Ascend. Executable external
diffusion adapters remain a separate, unverified boundary on this hardware.

## Validation status

- ✅ Unit and regression tests (2026-07-28): 148 passed in the Ascend container
  for the diffusion/accelerator/learner-focused suites (8 launcher-only tests
  deselected because that image does not contain SkyPilot); 726 passed and 4
  skipped for the complete Python suite on the development host; all 56 Rust
  syncer tests passed. Coverage includes selected-family backend resolution,
  explicit-device rank binding, accelerator-aware process-group setup, NPU
  diffusion dtype/RNG dispatch, and the NPU loss-metric dtype.
- ✅ **Single-card training** on an Ascend 910B4-1:
  - Qwen3-0.6B completed three fixed-micro-batch optimizer steps, and the real
    auto-batch probe selected micro-batch 4 and then completed training. The
    saved LoRA adapter reloaded and produced finite bf16 NPU logits with shape
    `(1, 4, 151936)`.
  - Qwen3-8B completed two optimizer steps with explicit `--device npu`: 504
    trainable tensors (87.3 MB) and 241 target tokens. Its saved adapter also
    reloaded through `PeftModel.from_pretrained`; an independent NPU forward
    produced finite bf16 logits with shape `(1, 4, 151936)`.
  - TinyLlama-1.1B completed three optimizer steps on the English
    `alpaca-gpt4-data-en` dataset. Its 308-tensor adapter reloaded and produced
    finite bf16 NPU logits with shape `(1, 4, 32000)`.
  - Phi-3.5-mini completed two optimizer steps on Databricks Dolly-15k. Its
    256-tensor adapter reloaded and produced finite bf16 NPU logits with shape
    `(1, 4, 32064)`.
- ✅ **Standard diffusion training and sampling:** SD 1.5 trained a 256-tensor
  UNet LoRA for two optimizer steps on real Pokemon BLIP caption/image rows at
  128×128. The adapter reloaded through `yeto.diffusion.sample` on NPU and
  generated a valid 128×128 RGB PNG. A separate two-card HCCL run completed two
  optimizer steps per rank; its saved 256 tensors (797,184 values) were all
  finite.
- ✅ **Diffusion FSDP2:** a fresh two-card SD 1.5 LoRA run completed two
  optimizer steps per rank through FSDP2/HCCL. All 797,184 values in its 256
  adapter tensors were finite and 797,181 were nonzero. The FSDP artifact then
  reloaded on NPU and generated a valid 128×128 image. Evidence is retained at
  `extended-diffusion/fsdp-sd15` under the validation root below.
- ✅ **Quantized video diffusion:** `diffusers/CogVideoX-5b-nf4` at commit
  `8c165c88c289ab9f35cbbd5c4e084ef1a0665edc` loaded with 341 `Linear4bit`
  modules and 341 `Params4bit` tensors, remained quantized after NPU placement,
  and trained for two optimizer steps on two real MP4/caption rows from
  `svjack/Lelouch_Vi_Britannia_FramePack_First_Last_Frame_Video_Captioned` at
  commit `851e88358fad2aad65576418d0ca368fa0b0a99c`. Its 336 LoRA tensors contain
  4,128,768 finite values, of which 4,128,737 are nonzero. Reloading that adapter
  through `yeto.diffusion.sample` produced a decodable 8-frame 64×64 RGB MP4
  (SHA256
  `145f6fba56e478494e1f4ea0956200ca7a3eec309165da6f0d5e4040b3f64646`).
  Eight frames is the exact native profile here: with this Diffusers/VAE pair a
  five-frame request also decodes to the next eight-frame temporal block, while
  an eight-frame request produces exactly eight. Evidence is retained at
  `extended-diffusion/cogvideox-nf4` under the validation root below.
- ✅ **Eight-card HCCL and syncer E2E** on all 8 cards of the same host. A fresh
  post-fix `torchrun --nproc-per-node 8` smoke completed two optimizer steps.
  The full run completed 34 local steps and four outer merges, wrote checkpoint,
  final marker, final state and event tape, and completed `FINAL_FRAGMENT` /
  `FINAL_ACK` for every rank. All 392 LoRA tensors exported from the syncer
  checkpoint were exactly equal to the learner's final saved adapter (maximum
  absolute difference 0.0).
- ✅ **Two logical islands** were exercised concurrently as two independent
  four-card HCCL groups with syncer quorum 2. They completed 95 and 52 local
  steps respectively; every one of the four event-tape rounds recorded exactly
  responders `[0, 1]`, and the two final adapters were bit-identical (maximum
  absolute difference 0.0).

Hardware validation exposed one blocking bug and two initialization hazards:
HCCL rejects float64 `all_reduce`, so NPU loss telemetry now accumulates and
reduces in float32; explicit accelerator devices without an index now bind to
`LOCAL_RANK`; and device selection/binding now precedes process-group creation,
whose backend and `device_id` come from that selected device. Backend
registration was also made genuinely idempotent when `torch.npu` is already
registered. The final eight-card smoke completed without the earlier
process-group device/barrier warning.

The extended diffusion check found that its learner and sampler still selected
only CUDA or CPU, so an Ascend run could silently train on CPU. Both now use the
same accelerator detection and rank binding as the causal-LM learner; diffusion
also selects bf16 on NPU, seeds the NPU RNG, uses NPU autocast, and creates HCCL
process groups from the selected device. Real-model testing then exposed two
additional upstream-contract gaps. Diffusers 0.39 rejects BnB quantization
unless CUDA/XPU/MPS is present even when bitsandbytes supports NPU, so Yeto now
applies a load-scoped NPU gate and device map. CogVideoX's VAE returns
channels-first video latents while its pipeline prepares frames-first latents,
so Yeto now infers the expected layout through the public `prepare_latents()`
interface and aligns it without model-name switches.

The retained hardware logs and artifacts are under
`/workspace/yeto-ascend-validation-20260728` in the container, with syncer-side
artifacts under `/root/yeto-ascend-validation-20260728` on the bastion.

- ⛔ **Still unverified:** executable external diffusion adapters have not been
  trained and reloaded on Ascend. Their attestation and hook behavior remains
  covered by unit tests and non-Ascend validation only.
- ⛔ **Still unverified:** there was only one physical Ascend host available.
  The two-island run validates independent HCCL groups and the complete wire
  protocol, but it does not validate HCCL or learner-to-syncer networking across
  two physical Ascend nodes. A real multi-node run remains required before
  claiming that topology.
