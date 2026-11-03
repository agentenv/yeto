# MLX island backend (Apple silicon)

The MLX learner accepts `--model-revision` and `--data-revision`. Remote model
snapshots are materialized at the resolved model commit before `mlx-lm` sees a
path, while the matching Hugging Face config and tokenizer use that same
commit. `--trust-remote-code` remains off unless explicitly requested. See
[PROVENANCE.md](PROVENANCE.md) for the shared artifact and loading policy.

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

- Unit tests (`tests/test_mlx_backend.py`) cover peft math/shape parity of the
  LoRA layer, canonical naming and the fragment pack/write round-trip;
  `scripts/check_name_parity.py` checks name parity against torch/peft.
- Verified locally on Apple silicon: an MLX learner against a real syncer, with
  the saved adapter reloading in torch/peft (`PeftModel.from_pretrained`), and a
  mixed torch (CPU) + MLX fleet at quorum 2 where every merge has both learners
  as responders.
- Verified over the WAN: a Mac joining a cloud fleet as an external MLX learner
  (`--external-learners 1`) next to a cloud GPU learner, both responders in every
  merge. A submitter that is not x86 Linux builds the syncer on the syncer VM
  (`SYNCER_REMOTE_BUILD`).
