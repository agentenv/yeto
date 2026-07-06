"""Generic diffusion learner with Yeto fragment synchronization.

Concrete diffusion frameworks are loaded through ``yeto.components``. The sync
loop is task-agnostic: components provide pipeline/data/loss hooks, while this
module owns DDP/FSDP wrapping, fragment layout, async sync, and learner output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ..components import component_names, default_component, get_component
from ..fragments import build_layout
from ..layout_metadata import build_layout_metadata
from ..protocol import DTYPE_BF16, DTYPE_F32, DTYPE_Q4, SyncerClient, bulk_dtype
from ..tensor_io import (
    apply_fragment,
    fragment_flat,
    pack_fragment,
    quantize_q4,
    unpack_fragment,
)

log = logging.getLogger("diffusion-learner")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Yeto generic diffusion learner")
    p.add_argument("--component", choices=component_names(), default=default_component())
    p.add_argument("--syncer", required=True, help="host:port, or none for local baseline")
    p.add_argument("--learner-id", type=int, required=True)
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument("--component-root", default=None)
    p.add_argument("--component-config", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--data-format", choices=["jsonl", "list"], default="jsonl")
    p.add_argument("--modality", default="text_to_av")
    p.add_argument("--adapter", choices=["lora", "full", "regex"], default="lora")
    p.add_argument("--trainable-regex", default=None)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-targets", default=None)
    p.add_argument("--merge-avg-regex", default=r"(^|\.)(bias|norm|modulation|scale|shift)(\.|$)")
    p.add_argument("--init-timeout", type=float, default=1800.0)
    p.add_argument("--shard", choices=["ddp", "fsdp"], default="fsdp")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--max-local-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--io-workers", type=int, default=None)
    p.add_argument("--disable-ema", action="store_true")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--learner-state-dir", default=None)
    p.add_argument("--fragments", type=int, default=32)
    p.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    p.add_argument("--merge-alpha", type=float, default=0.5)
    p.add_argument("--wire-dtype", choices=["bf16", "f32", "q4"], default="bf16")
    p.add_argument("--wan-streams", type=int, default=8)
    p.add_argument("--output-dir", default="checkpoints/diffusion-yeto-out")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def setup_distributed() -> tuple[int, int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def allreduce_grads(params, world: int) -> None:
    if world <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def maybe_wrap_model(component, runtime, params: dict[str, torch.Tensor], args, rank: int, world: int, device):
    model = component.get_model(runtime)
    if args.shard == "fsdp":
        if device.type != "cuda":
            raise RuntimeError("diffusion --shard fsdp requires CUDA")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools

        wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        kwargs = {"auto_wrap_policy": wrap_policy, "use_orig_params": True, "device_id": device}
        if args.adapter == "lora":
            # Keep adapters replicated and stable while sharding the frozen base.
            kwargs["ignored_states"] = list(params.values())
        else:
            kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
        component.set_model(runtime, FSDP(model, **kwargs))
        wrapped = dict(component.trainable_params(runtime))
        if args.adapter == "lora" and set(wrapped) != set(params):
            diff = sorted(set(wrapped) ^ set(params))[:8]
            raise RuntimeError(f"FSDP changed diffusion LoRA trainable names; layout would diverge: {diff}")
        return wrapped
    if world > 1:
        component.set_model(
            runtime,
            torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            ),
        )
        return dict(component.trainable_params(runtime))
    return params


def _as_batch(maybe_batch):
    return maybe_batch[0] if isinstance(maybe_batch, list) and len(maybe_batch) == 1 else maybe_batch


def _cpu_data_state(batch):
    state = batch.get("data_state") if isinstance(batch, dict) else None
    if state is None:
        return None
    if isinstance(state, list) and len(state) == 1:
        state = state[0]
    return state.detach().cpu() if hasattr(state, "detach") else state


def save_learner_state(args, opt, sched, counters, layout_meta, batch) -> None:
    state_dir = Path(args.learner_state_dir or Path(args.output_dir) / "learner_state")
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "learner_id": args.learner_id,
        "local_step": counters[0],
        "global_step": counters[6],
        "layout_hash": layout_meta.get("layout_hash"),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if hasattr(sched, "state_dict") else None,
        "data_state": _cpu_data_state(batch),
    }
    tmp = state_dir / "learner_state.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(state_dir / "learner_state.pt")
    (state_dir / "learner_id.json").write_text(
        json.dumps(
            {
                "learner_id": args.learner_id,
                "local_step": counters[0],
                "global_step": counters[6],
                "layout_hash": layout_meta.get("layout_hash"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_sync_boundary(args, params, layout, client, rank, world, device, counters, anchors):
    steps_total, units_total, steps_at_reset, units_at_reset, fragment_versions, pending_pulls, global_step = counters
    actions = []
    shutdown = False
    if rank == 0 and client is not None:
        client.check_health()
        pending_pulls.extend(client.drain_pulls())
        still_pending = []
        for pull in pending_pulls:
            fid = pull.fragment_id
            c_steps = steps_total - steps_at_reset[fid]
            if c_steps < 1:
                still_pending.append(pull)
                continue
            c_units = units_total - units_at_reset[fid]
            if anchors is not None:
                delta = fragment_flat(layout.fragments[fid], params).cpu() - anchors[fid]
                payload = quantize_q4(delta)
            else:
                payload = pack_fragment(layout.fragments[fid], params, client.dtype)
            client.push_fragment(
                fid,
                pull.global_step,
                fragment_versions[fid],
                steps_total,
                c_steps,
                c_units,
                payload,
            )
        pending_pulls = still_pending
        for bc in client.drain_updates():
            flat = unpack_fragment(layout.fragments[bc.fragment_id], bc.data, bulk_dtype(client.dtype))
            if anchors is not None:
                anchors[bc.fragment_id] = flat.clone()
            actions.append((bc.fragment_id, bc.version, flat))
        shutdown = client.shutdown.is_set()

    if world > 1:
        meta = [(f, v) for f, v, _ in actions] if rank == 0 else None
        box = [meta, shutdown]
        dist.broadcast_object_list(box, src=0)
        meta, shutdown = box
        if rank != 0:
            actions = [(f, v, torch.empty(layout.fragments[f].numel, dtype=torch.float32)) for f, v in meta]
        for fid, version, flat in actions:
            flat = flat.to(device)
            dist.broadcast(flat, src=0)
            if args.merge_alpha > 0:
                local = fragment_flat(layout.fragments[fid], params)
                flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
            apply_fragment(layout.fragments[fid], flat, params)
            if rank == 0:
                steps_at_reset[fid] = steps_total
                units_at_reset[fid] = units_total
                fragment_versions[fid] = version
            global_step = max(global_step, version)
    else:
        for fid, version, flat in actions:
            flat = flat.to(device)
            if args.merge_alpha > 0:
                local = fragment_flat(layout.fragments[fid], params)
                flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
            apply_fragment(layout.fragments[fid], flat, params)
            steps_at_reset[fid] = steps_total
            units_at_reset[fid] = units_total
            fragment_versions[fid] = version
            global_step = max(global_step, version)

    counters[:] = [steps_total, units_total, steps_at_reset, units_at_reset, fragment_versions, pending_pulls, global_step]
    return shutdown


def wait_for_initial_sync(args, params, layout, client, rank: int, world: int, device):
    if args.syncer == "none":
        return [0] * layout.num_fragments, 0
    seen: set[int] = set()
    seen_versions = [0] * layout.num_fragments
    global_step = 0
    deadline = time.monotonic() + float(args.init_timeout)
    while True:
        actions = []
        shutdown = False
        error = None
        done = False
        if rank == 0:
            try:
                client.check_health()
                for bc in client.drain_updates():
                    flat = unpack_fragment(layout.fragments[bc.fragment_id], bc.data, bulk_dtype(client.dtype))
                    actions.append((bc.fragment_id, bc.version, flat))
                    seen.add(bc.fragment_id)
                shutdown = client.shutdown.is_set()
                done = len(seen) == layout.num_fragments
                if shutdown and not done:
                    error = "syncer shut down before initial diffusion state broadcast completed"
                elif not done and time.monotonic() > deadline:
                    error = f"timed out waiting for initial diffusion state broadcast ({len(seen)}/{layout.num_fragments} fragments)"
            except BaseException as e:  # noqa: BLE001 - rank0 broadcasts the failure
                error = repr(e)

        if world > 1:
            meta = [(f, v) for f, v, _ in actions] if rank == 0 else None
            box = [meta, shutdown, done, error]
            dist.broadcast_object_list(box, src=0)
            meta, shutdown, done, error = box
            if rank != 0:
                actions = [(f, v, torch.empty(layout.fragments[f].numel, dtype=torch.float32)) for f, v in meta]

        if error:
            raise RuntimeError(error)

        if world > 1:
            for fid, version, flat in actions:
                flat = flat.to(device)
                dist.broadcast(flat, src=0)
                apply_fragment(layout.fragments[fid], flat, params)
                seen_versions[fid] = version
                global_step = max(global_step, version)
        else:
            for fid, version, flat in actions:
                apply_fragment(layout.fragments[fid], flat.to(device), params)
                seen_versions[fid] = version
                global_step = max(global_step, version)

        if done:
            if rank == 0:
                log.info("initial diffusion global state applied (%d fragments)", layout.num_fragments)
            return seen_versions, global_step
        time.sleep(0.1)


def main(argv=None) -> None:
    args = parse_args(argv)
    rank, world, local_rank = setup_distributed()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s diff{args.learner_id}.r{rank} %(levelname)s %(message)s")
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    component = get_component(args.component)
    component.resolve_paths(args)
    cfg = component.load_config(args)
    runtime = component.build_pipeline(args, cfg, device)
    adapter_config = component.configure_trainables(runtime, args)
    params = dict(component.trainable_params(runtime))
    if not params:
        raise RuntimeError(f"diffusion component {args.component!r} has no trainable parameters")
    params = maybe_wrap_model(component, runtime, params, args, rank, world, device)
    layout = build_layout(
        [(n, p.numel()) for n, p in params.items()],
        args.fragments,
        args.fragment_pattern,
        avg_name_regex=args.merge_avg_regex,
    )
    layout_meta = build_layout_metadata(
        task="diffusion",
        layout=layout,
        params=params,
        backend_version="diffusion-yeto-v1",
        component=args.component,
        component_version=f"{args.component}-component-v1",
        base_checkpoint=args.base_checkpoint,
        trainable_policy=args.adapter,
        trainable_regex=args.trainable_regex,
        adapter=(
            {
                "type": "lora",
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "targets": args.lora_targets,
            }
            if args.adapter == "lora"
            else {"type": args.adapter}
        ),
        merge_avg_regex=args.merge_avg_regex,
    )
    if rank == 0:
        log.info(
            "diffusion/%s trainables: %d tensors -> %d fragments (%.2f MB bf16 wire)",
            args.component,
            len(params),
            layout.num_fragments,
            sum(p.numel() for p in params.values()) * 2 / 1e6,
        )

    wire_dtype = {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "q4": DTYPE_Q4}[args.wire_dtype]
    client = None
    if rank == 0 and args.syncer != "none":
        host, port = args.syncer.rsplit(":", 1)
        client = SyncerClient(
            (host, int(port)),
            args.learner_id,
            layout,
            wire_dtype,
            args.wan_streams,
            layout_metadata=layout_meta,
        )
        client.start()
        if args.learner_id == 0:
            for fid, frag in enumerate(layout.fragments):
                client.send_init(fid, pack_fragment(frag, params, bulk_dtype(wire_dtype)))
            log.info("sent INIT_PARAMS for %d fragments", layout.num_fragments)

    initial_versions, initial_global_step = wait_for_initial_sync(args, params, layout, client, rank, world, device)
    loader = component.build_dataloader(args, cfg, runtime, rank, world)
    opt = torch.optim.AdamW(
        params.values(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    sched = component.build_scheduler(opt, cfg)

    counters = [
        0,
        0,
        [0] * layout.num_fragments,
        [0] * layout.num_fragments,
        initial_versions,
        [],
        initial_global_step,
    ]
    anchors = [fragment_flat(frag, params).cpu() for frag in layout.fragments] if rank == 0 and client is not None and wire_dtype == DTYPE_Q4 else None
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    max_steps = int(cfg["max_steps"])
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    log_every = int(cfg.get("log_every", 20))

    component.get_model(runtime).train()
    opt.zero_grad(set_to_none=True)
    accum = 0
    loss_acc = 0.0
    t_last = time.monotonic()
    shutdown = False
    while not shutdown and counters[0] < max_steps:
        for maybe_batch in loader:
            batch = _as_batch(maybe_batch)
            loss, _logs = component.training_step(runtime, batch, global_step=counters[6])
            (loss / grad_accum).backward()
            loss_acc += float(loss.detach().cpu())
            accum += 1
            if accum < grad_accum:
                continue
            accum = 0
            if args.shard == "fsdp" and args.adapter == "lora":
                allreduce_grads(params.values(), world)
            torch.nn.utils.clip_grad_norm_(params.values(), max_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            counters[0] += 1
            counters[1] += component.batch_units(batch, world)
            if rank == 0 and counters[0] % log_every == 0:
                dt = max(1e-9, time.monotonic() - t_last)
                t_last = time.monotonic()
                log.info(
                    "local_step=%d global_step=%d loss=%.4f units/s=%.1f",
                    counters[0],
                    counters[6],
                    loss_acc / log_every,
                    counters[1] / dt,
                )
                loss_acc = 0.0
            shutdown = run_sync_boundary(args, params, layout, client, rank, world, device, counters, anchors)
            if rank == 0 and args.save_every > 0 and counters[0] % args.save_every == 0:
                save_learner_state(args, opt, sched, counters, layout_meta, batch)
            if shutdown or counters[0] >= max_steps:
                break
        if not getattr(getattr(loader, "dataset", None), "is_cycle", False):
            break

    if rank == 0:
        layout_meta["local_steps"] = counters[0]
        layout_meta["global_step"] = counters[6]
        out = Path(args.output_dir)
        component.save_artifact(runtime, args, out, params, adapter_config, layout_meta)
        if client is not None:
            client.close()
        log.info("saved diffusion Yeto output to %s", out)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
