# MLX island backend (Apple silicon)

## Why

Macs with Apple silicon are abundant idle capacity: unified memory holds
models that would need multiple consumer GPUs, and MLX trains LoRA at
useful throughput (an M1 Ultra does ~5k tok/s on LFM2.5-230M at seq 256).
The MLX backend makes a Mac a first-class learner island so a fleet can mix
macOS and NVIDIA learners in one run — cross-Mac training, or a Mac joining
a CUDA fleet over the WAN.

## Design: same sync, different tensor engine

`yeto.mlx.learner` is a peer of `yeto.learner` (torch) and
`yeto.megatron.learner`. The DiLoCo bridge is backend-agnostic: the syncer
only moves the LoRA adapter fragments, so the backend's whole job is to
produce the same `{canonical_name: tensor}` view the torch learner would.

The cross-backend contract lives in `yeto/mlx/lora.py`:

- **peft-shaped adapters.** Our own `LoRALinear` (not mlx-lm's, which stores
  transposed factors under different names): `lora_A` is `(r, in)`
  kaiming-init, `lora_B` is `(out, r)` zeros, forward adds
  `(alpha/r)·x Aᵀ Bᵀ` — bit-compatible with peft in name, shape, flatten
  order and scale.
- **Canonical FQNs.** MLX tree path `model.layers.N…q_proj.lora_A` maps to
  peft's `base_model.model.model.layers.N…q_proj.lora_A.default.weight`.
  `build_layout` packs fragments BY NAME, so identical names+numels ⇒
  identical fragments on every learner, whatever the backend.
- **Same target selection.** The attention-projection regex and the
  all-linear/MoE-auto rules are imported from `yeto.learner`, driven by the
  HF config, so both backends adapt the same linears.
- **Same data.** The HF tokenizer (not mlx-lm's wrapper) feeds `yeto.data`,
  so blocks and loss masks are identical to a CUDA learner's; the loss is
  `yeto.losses.sft_loss` re-derived in MLX (weighted per-token CE).
- **torch on the wire.** Adapters are megabytes, so step-boundary
  mx→numpy→torch copies are noise; `pack_fragment`/`quantize_q4`/
  `SyncerClient` are reused unchanged.

Verify parity for any new architecture before a heterogeneous run:

    python scripts/check_name_parity.py --model <alias-or-hf-id>

(For LFM2.5-230M: 164 trainable tensors, names and shapes identical.)

## Joining a fleet from a Mac

Cloud learners are provisioned by `yeto launch`; a Mac joins manually into a
reserved slot:

```bash
# submitting machine: reserve one external slot
yeto launch --gpu aws:1xA10G@us-west-2 --external-learners 1 \
  --model lfm25-1b --data org/chat-traces --quorum 2 ...

# the launch log prints the join command for each reserved slot; on the Mac:
pip install "yeto[mlx] @ ." && \
python -m yeto.mlx.learner --model lfm25-1b --data org/chat-traces \
  --syncer <syncer-ip>:29400 --learner-id 1 --num-learners 2
```

The syncer's port is already public (`ports=[SYNCER_PORT]` on the syncer
cluster / head VM) and it waits for all `--learners` before starting, so the
run begins when the Mac dials in. Learner reconnection/backoff applies to the
Mac exactly as to cloud learners.

Notes:

- Single-process only: one Mac = one island (no torchrun). LoRA tuning only.
- `--loss-function cross_entropy` only (custom/pickled losses are torch
  callables; the MLX learner computes its loss in MLX).
- transformers 5.x: mlx-lm ≤ 0.31 fails to import under transformers 5
  (string-keyed tokenizer registration); `yeto.mlx.learner.import_mlx_lm`
  works around it. `mlx_config_shim` re-flattens `rope_parameters` for
  configs written by transformers ≥ 4.54.

## Validation status

- ✅ Unit tests (`tests/test_mlx_backend.py`): peft math/shape parity of the
  LoRA layer, canonical naming, fragment pack/write round-trip.
- ✅ Name parity vs torch/peft on LFM2.5-230M (`scripts/check_name_parity.py`).
- ✅ Local smoke on an M1 Ultra: real syncer + MLX learner, LFM2.5-230M,
  12 outer steps, adapters saved and re-loaded into torch/peft
  (`PeftModel.from_pretrained` on the MLX-written directory), generation
  reflects the fine-tune.
- ✅ Heterogeneous local smoke: torch (CPU) learner 0 + MLX learner 1 on one
  syncer, quorum 2 — all 8 outer merges had both learners as responders with
  token-weighted RDA (the MLX side contributed ~28× the steps), both applied
  broadcasts and exited cleanly.
- ✅ Mac + AWS G-instance cross run over the WAN (2026-07-04): syncer VM +
  g5 spot learner in us-west-2 (`--external-learners 1`), this Mac joined as
  MLX learner 1 on LFM2.5-1.2B. All 16 outer merges had BOTH learners as
  responders (event tape), with token-weighted RDA reflecting the ~160×
  token-rate gap between the islands. Shakedown fixes that run produced:
  the syncer is now built ON the syncer VM for non-x86-Linux submitters
  (`SYNCER_REMOTE_BUILD` — a Mac-built arm64 binary is an Exec format error
  on the VM), and single-file `--data` mounts keep their extension
  (`datasource._mount_target`) so learners can detect the format.
  Operational note: this AWS account had zero on-demand G quota, so the run
  used spot in us-west-2 (the only region with spot G quota) after
  `--retry-until-up` rode out InsufficientInstanceCapacity.
