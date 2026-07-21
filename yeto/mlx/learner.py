"""MLX island learner — a peer of yeto.learner for Apple-silicon machines,
speaking the exact same DiLoCo adapter sync to the Rust syncer.

The base model runs under MLX (unified memory, Metal); the wire path reuses
yeto's torch primitives unchanged (pack_fragment/apply_fragment/SyncerClient
operate on CPU tensors — adapters are megabytes, so the mx<->torch copies at
step boundaries are noise). Data/tokenization reuse yeto.data with the HF
tokenizer, so an MLX learner consumes the same blocks, with the same loss
masking, as a CUDA learner on the same run.

Cross-backend contract (what lets a Mac and an NVIDIA island share a syncer):
the adapter registry maps every trainable tensor to the peft FQN, shape and
flatten order the torch learner would report (see yeto/mlx/lora.py), so
build_layout produces identical fragments on both sides.

Single-process only: one Mac is one island (no torchrun/world). Tuning is
LoRA-only for now — full tuning would sync the frozen base, which the
Mac-to-CUDA dtype story does not need yet.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

from ..protocol import add_syncer_security_arguments

log = logging.getLogger("mlx-learner")


def parse_args(argv=None):
    p = argparse.ArgumentParser("yeto.mlx.learner")
    p.add_argument("--model", required=True, help="HF id or yeto/models.py alias")
    p.add_argument("--data", required=True)
    p.add_argument("--syncer", required=True, help="host:port or 'none' (local-only run)")
    p.add_argument("--learner-id", type=int, default=0)
    add_syncer_security_arguments(p)
    p.add_argument("--num-learners", type=int, default=1)
    p.add_argument("--loss-function", default="cross_entropy")
    p.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    p.add_argument(
        "--assistant-mask-mode",
        choices=["native", "legacy"],
        default="native",
        help="assistant-only masking: tokenizer-native exact mask or explicit legacy format",
    )
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument(
        "--micro-batch-size",
        default="1",
        help="int; 'auto' (the launcher default for CUDA learners) maps to 1",
    )
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--fragments", type=int, default=8)
    p.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    p.add_argument("--merge-alpha", type=float, default=0.5)
    p.add_argument("--wire-dtype", choices=["bf16", "f32", "q4"], default="bf16")
    p.add_argument("--wan-streams", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--tokenize", choices=["stream", "preload"], default="stream")
    p.add_argument("--stream-workers", type=int, default=0, help="accepted for launcher flag parity; tokenization runs inline")
    p.add_argument("--shard", default="ddp", help="accepted for flag parity; a Mac island is single-process")
    p.add_argument("--max-local-steps", type=int, default=1_000_000)
    p.add_argument("--output-dir", default="checkpoints/out")
    return p.parse_args(argv)


def import_mlx_lm():
    """Import mlx_lm under transformers 5.

    mlx-lm (<= 0.31 at least) registers its NewlineTokenizer at import time
    with a transformers-4-style string key, which transformers 5 rejects —
    the whole package fails to import. That tokenizer is for mlx-community
    audio models yeto never loads, so tolerate the failed registration.
    """
    import sys

    if "mlx_lm" in sys.modules:
        return sys.modules["mlx_lm"]
    from transformers import AutoTokenizer

    orig = AutoTokenizer.register

    def _tolerant_register(*a, **k):
        try:
            return orig(*a, **k)
        except (AttributeError, TypeError, ValueError):
            pass

    AutoTokenizer.register = _tolerant_register
    try:
        import mlx_lm
    finally:
        AutoTokenizer.register = orig
    return mlx_lm


def load_model_and_tokenizer(args):
    """MLX model (bf16, unified memory) + the HF tokenizer.

    The tokenizer is transformers', not mlx-lm's TokenizerWrapper, so
    yeto.data tokenizes IDENTICALLY to the torch learner — same blocks,
    same loss masks, regardless of backend.
    """
    mlx_lm = import_mlx_lm()
    from transformers import AutoConfig, AutoTokenizer

    from ..learner import _from_pretrained_offline_first
    from ..models import resolve

    model_id = resolve(args.model)
    hf_config = _from_pretrained_offline_first(AutoConfig, model_id, trust_remote_code=True)
    model, _ = mlx_lm.load(model_id, model_config=mlx_config_shim(hf_config))
    tokenizer = _from_pretrained_offline_first(AutoTokenizer, model_id, trust_remote_code=True)
    return model, tokenizer, hf_config


def mlx_config_shim(hf_config) -> dict:
    """Config keys mlx-lm's model args expect but newer HF configs moved.

    transformers >= 4.54 nests rope settings under ``rope_parameters`` in
    config.json (AutoConfig back-fills a flat attribute, but mlx-lm reads the
    raw file); mlx-lm model args still want a flat ``rope_theta``. The
    returned dict is merged over config.json by mlx_lm.load(model_config=...);
    duplicating an already-flat key is harmless."""
    rope = getattr(hf_config, "rope_parameters", None) or {}
    shim = {}
    if isinstance(rope, dict) and "rope_theta" in rope:
        shim["rope_theta"] = rope["rope_theta"]
    return shim


def torch_adapters(model, registry) -> dict[str, torch.Tensor]:
    """Snapshot the trainable adapters as {peft FQN: f32 CPU tensor}.

    A copy, not a view: pack/blend read it at a step boundary and
    write_fragment pushes merged values back into the MLX tree explicitly.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.trainable_parameters()))
    out = {}
    for cname, info in registry.items():
        arr = flat[info.path]
        if arr.dtype != mx.float32:
            arr = arr.astype(mx.float32)
        out[cname] = torch.from_numpy(np.array(arr))
    return out


def write_fragment(model, frag, flat: torch.Tensor, registry) -> None:
    """Overwrite the fragment's adapters in the MLX tree from a flat f32
    torch tensor (the mx analogue of tensor_io.apply_fragment)."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    updates = []
    off = 0
    for name, numel in frag.tensors:
        info = registry[name]
        chunk = flat[off : off + numel].reshape(info.shape).contiguous().numpy()
        updates.append((info.path, mx.array(chunk).astype(info.dtype)))
        off += numel
    model.update(tree_unflatten(updates))


def _micro_batches(args, tokenizer):
    """Endless (input_ids, weights) micro-batches as stacked CPU tensors.

    Reuses yeto.data directly (no DataLoader): MLX trains in-process, so
    worker fan-out buys nothing a Mac needs yet.
    """
    from ..data import StreamingPackedBlocks, build_packed_dataset

    mbs = args.micro_batch_size

    if args.tokenize == "stream":
        dataset = StreamingPackedBlocks(
            args.data,
            tokenizer,
            args.learner_id,
            args.num_learners,
            args.seq_len,
            args.max_rows,
            train_on=args.train_on,
            assistant_mask_mode=args.assistant_mask_mode,
        )

        def gen():
            ids_buf, w_buf = [], []
            for ids, weights in dataset:  # infinite
                ids_buf.append(ids)
                w_buf.append(weights)
                if len(ids_buf) == mbs:
                    yield torch.stack(ids_buf), torch.stack(w_buf)
                    ids_buf, w_buf = [], []

        return gen()

    dataset = build_packed_dataset(
        args.data,
        tokenizer,
        args.learner_id,
        args.num_learners,
        args.seq_len,
        args.max_rows,
        train_on=args.train_on,
        assistant_mask_mode=args.assistant_mask_mode,
    )
    log.info("dataset ready: %d blocks of %d tokens", len(dataset), args.seq_len)

    def gen():
        import random

        order = list(range(len(dataset)))
        rng = random.Random(0)
        while True:
            rng.shuffle(order)
            for i in range(0, len(order) - mbs + 1, mbs):
                pick = order[i : i + mbs]
                yield (
                    torch.stack([dataset[j][0] for j in pick]),
                    torch.stack([dataset[j][1] for j in pick]),
                )

    return gen()


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s mlx-learner{args.learner_id} %(levelname)s %(message)s",
    )
    if args.micro_batch_size == "auto":
        args.micro_batch_size = 1
    args.micro_batch_size = int(args.micro_batch_size)
    if args.tuning != "lora":
        raise NotImplementedError("the MLX backend supports --tuning lora only for now")
    if args.loss_function != "cross_entropy":
        raise NotImplementedError(
            "the MLX backend computes its loss in MLX; custom/pickled torch "
            "losses are torch-only — use --loss-function cross_entropy"
        )
    if not 0.0 <= args.merge_alpha < 1.0:
        raise ValueError(f"--merge-alpha must be in [0, 1), got {args.merge_alpha}")

    import mlx.core as mx

    from .lora import attach_lora

    log.info("loading model %s (mlx, %s)", args.model, args.tuning)
    model, tokenizer, hf_config = load_model_and_tokenizer(args)
    registry = attach_lora(model, hf_config, args.lora_targets, args.lora_r, args.lora_alpha)
    model.train()
    mx.eval(model.parameters())

    from ..fragments import build_layout
    from ..protocol import (
        DTYPE_BF16,
        DTYPE_F32,
        DTYPE_Q4,
        SyncerClient,
        bulk_dtype,
        syncer_security_from_args,
    )
    from ..tensor_io import (
        fragment_flat,
        pack_fragment,
        pack_tensor,
        quantize_q4,
        unpack_fragment,
    )

    layout = build_layout(
        [(n, int(np.prod(i.shape))) for n, i in registry.items()],
        args.fragments,
        args.fragment_pattern,
        named_shapes={n: tuple(int(dim) for dim in info.shape) for n, info in registry.items()},
    )
    total = sum(int(np.prod(i.shape)) for i in registry.values())
    log.info(
        "%d trainable tensors -> %d fragments (%.1f MB total)",
        len(registry),
        layout.num_fragments,
        total * 2 / 1e6,
    )

    wire_dtype = {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "q4": DTYPE_Q4}[args.wire_dtype]
    client = None
    if args.syncer != "none":
        host, port = args.syncer.rsplit(":", 1)
        run_id, tls, allow_insecure_loopback = syncer_security_from_args(args)
        client = SyncerClient(
            (host, int(port)),
            args.learner_id,
            layout,
            wire_dtype,
            args.wan_streams,
            run_id=run_id,
            tls=tls,
            allow_insecure_loopback=allow_insecure_loopback,
        )
        client.start()
        log.info("connected to syncer at %s", args.syncer)
        if args.learner_id == 0:
            snap = torch_adapters(model, registry)
            for fid, frag in enumerate(layout.fragments):
                client.send_init(fid, pack_fragment(frag, snap, bulk_dtype(wire_dtype)))
            log.info("sent INIT_PARAMS for %d fragments", layout.num_fragments)

    run_inner_loop(
        args, model, registry, layout, tokenizer, client,
        fragment_flat=fragment_flat, pack_tensor=pack_tensor,
        quantize_q4=quantize_q4,
        unpack_fragment=unpack_fragment,
        bulk_dtype=bulk_dtype, dtype_q4=DTYPE_Q4,
    )

    save_adapters(args, model, registry, tokenizer)
    if client is not None:
        client.close()


def run_inner_loop(
    args, model, registry, layout, tokenizer, client,
    *, fragment_flat, pack_tensor, quantize_q4, unpack_fragment,
    bulk_dtype, dtype_q4,
):
    """MLX inner AdamW steps + the torch learner's exact sync semantics:
    counters advance every step, pulls are answered once c_steps >= 1,
    broadcasts α-blend into the local adapters and reset that fragment."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    from ..protocol import DTYPE_F32

    def loss_fn(mdl, ids, weights):
        # Same math as yeto.losses.sft_loss: next-token logprobs, weighted
        # SUM over trained tokens, normalized by the micro-batch's own
        # trained-token count (clamped to 1).
        logits = mdl(ids)
        logprobs = nn.log_softmax(logits[:, :-1, :].astype(mx.float32), axis=-1)
        targets = ids[:, 1:]
        tlp = mx.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
        w = weights[:, 1:]
        trained = mx.maximum(w.sum(), 1.0)
        return -(tlp * w).sum() / trained, trained

    value_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.inner_lr, weight_decay=args.weight_decay)
    batches = _micro_batches(args, tokenizer)

    steps_total = 0
    tokens_total = 0
    steps_at_reset = [0] * layout.num_fragments
    tokens_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments
    pending_pulls: list = []
    global_step = 0
    tokens_per_inner_step = args.micro_batch_size * args.grad_accum * args.seq_len
    anchors: list[torch.Tensor | None] | None = None
    if client is not None:
        anchors = [None] * layout.num_fragments

    shutdown = False
    t_last = time.monotonic()
    while not shutdown and steps_total < args.max_local_steps:
        grads_acc = None
        loss_val = 0.0
        for _ in range(args.grad_accum):
            ids_t, w_t = next(batches)
            ids = mx.array(ids_t.numpy().astype(np.int32))
            weights = mx.array(w_t.numpy())
            (loss, trained), grads = value_and_grad(model, ids, weights)
            grads_acc = (
                grads if grads_acc is None else tree_map(lambda a, b: a + b, grads_acc, grads)
            )
            mx.eval(grads_acc, loss)
            loss_val = loss.item()
        if args.grad_accum > 1:
            grads_acc = tree_map(lambda g: g / args.grad_accum, grads_acc)
        grads_acc, _ = optim.clip_grad_norm(grads_acc, 1.0)
        opt.learning_rate = args.inner_lr * min(1.0, (steps_total + 1) / max(1, args.warmup_steps))
        opt.update(model, grads_acc)
        mx.eval(model.parameters(), opt.state)
        steps_total += 1
        tokens_total += tokens_per_inner_step

        if steps_total % 10 == 0:
            dt = time.monotonic() - t_last
            t_last = time.monotonic()
            log.info(
                "local_step=%d global_step=%d loss/token=%.4f (%.2f s/step, %.0f tok/s)",
                steps_total,
                global_step,
                loss_val,  # loss_fn already normalizes per trained token
                dt / 10,
                10 * tokens_per_inner_step / max(dt, 1e-9),
            )

        # --- fragment sync at the step boundary (never blocks) ---
        # Broadcasts apply BEFORE pulls are answered (same ordering as
        # yeto.learner): the pipelined syncer's pull opening a fragment's
        # next round (control stream) can overtake the broadcast that
        # closed its previous one (data streams); answering first would
        # push a stale base_version. Applying first resets the fragment's
        # counters, so the self-clock defers the answer one step.
        if client is None:
            continue
        client.check_health()
        if client.finalizing.is_set():
            manifest, broadcasts = client.wait_for_final_fragments()
            for update in broadcasts:
                fid = update.fragment_id
                flat = unpack_fragment(
                    layout.fragments[fid],
                    update.data,
                    DTYPE_F32,
                )
                # Finalization is a raw overwrite: normal delayed-application
                # blending must not leak into the saved adapter.
                write_fragment(model, layout.fragments[fid], flat, registry)
            mx.eval(model.parameters())
            client.acknowledge_finalization(manifest)
            global_step = max(global_step, manifest.global_step)
            shutdown = True
            break
        snap = None  # torch view of the adapters, built lazily per boundary
        for bc in client.drain_updates():
            fid = bc.fragment_id
            flat = unpack_fragment(layout.fragments[fid], bc.data, bulk_dtype(client.dtype))
            if anchors is not None:
                anchors[fid] = flat.clone()
            if args.merge_alpha > 0:
                snap = snap if snap is not None else torch_adapters(model, registry)
                local = fragment_flat(layout.fragments[fid], snap)
                flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
            write_fragment(model, layout.fragments[fid], flat, registry)
            steps_at_reset[fid] = steps_total
            tokens_at_reset[fid] = tokens_total
            fragment_versions[fid] = bc.version
            global_step = max(global_step, bc.version)
        snap = None  # writes above invalidate the lazy snapshot
        pending_pulls.extend(client.drain_pulls())
        still_pending = []
        for pull in pending_pulls:
            fid = pull.fragment_id
            c_steps = steps_total - steps_at_reset[fid]
            if c_steps < 1:
                still_pending.append(pull)
                continue
            c_tokens = tokens_total - tokens_at_reset[fid]
            snap = snap if snap is not None else torch_adapters(model, registry)
            anchor = anchors[fid] if anchors is not None else None
            if anchor is None:
                still_pending.append(pull)
                continue
            delta = fragment_flat(layout.fragments[fid], snap) - anchor
            if client.dtype == dtype_q4:
                payload = quantize_q4(delta)
            else:
                payload = pack_tensor(delta, client.dtype)
            client.push_fragment(
                fid,
                pull.global_step,
                pull.round_attempt,
                fragment_versions[fid],
                steps_total,
                c_steps,
                c_tokens,
                payload,
            )
        pending_pulls = still_pending
        shutdown = client.shutdown.is_set()
    if client is not None and not client.finalized.is_set():
        raise RuntimeError(
            "MLX learner stopped before authoritative finalization; "
            "refusing to save local parameters"
        )
    mx.eval(model.parameters())
    log.info("inner loop done at local_step=%d global_step=%d", steps_total, global_step)


def save_adapters(args, model, registry, tokenizer) -> None:
    """Write a peft-loadable adapter directory (adapter_model.safetensors
    keys are the canonical FQNs minus peft's '.default' adapter marker)."""
    import json

    from safetensors.torch import save_file

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    snap = torch_adapters(model, registry)
    state = {n.replace(".default.weight", ".weight"): t.contiguous() for n, t in snap.items()}
    save_file(state, os.path.join(save_dir, "adapter_model.safetensors"))
    from ..models import resolve

    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": resolve(args.model),
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        # module paths minus the trailing .lora_A/.lora_B, deduplicated
        "target_modules": sorted({i.path.rsplit(".", 2)[-2] for i in registry.values()}),
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    tokenizer.save_pretrained(save_dir)
    log.info("saved adapters to %s", save_dir)


if __name__ == "__main__":
    main()
