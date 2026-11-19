"""Cluster preflight benchmarks that run inside the trainer container.

This file is shipped verbatim to every node by ``yeto preflight``
(:mod:`yeto.launch.preflight.cluster`) and must stay self-contained: the
standard library, plus ``torch`` for the ``collectives`` mode. It never imports
yeto, so an image built from another yeto revision runs it unchanged.

Modes:

``ports``
    Listen on TCP ports, then dial the same ports on every peer and print one
    JSON line ``@@yeto-bench {"mode": "ports", ...}``. Every node runs it at
    the same time, so each listener is up while the peers dial it. A port that
    is already taken on this node is reported as ``busy`` (a stale Ray head
    or trainer still holds it), and the dial results show what a peer reached.

``collectives``
    Run under ``torchrun`` on every node: NCCL ``all_reduce``, ``all_to_all``
    and a rail-aligned cross-node ``sendrecv`` (every GPU sends to the GPU of
    the same local index on the next node), each checked for correctness once
    and timed over several sizes. Rank 0 prints one ``@@yeto-bench`` JSON line
    per op and size with the bandwidths in nccl-tests terms (``algbw`` = bytes
    / time; ``busbw`` = algbw x 2(n-1)/n for all_reduce, x (n-1)/n for
    all_to_all, x 1 for sendrecv), in GB/s (1e9 bytes per second).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from typing import Any

MARKER = "@@yeto-bench"


def emit(record: dict) -> None:
    print(f"{MARKER} {json.dumps(record, sort_keys=True)}", flush=True)


# ---------------------------------------------------------------------- ports
def _listen(port: int, stop: threading.Event, bound: dict) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.listen(64)
    except OSError as exc:
        bound[port] = f"busy: {exc.strerror or exc}"
        sock.close()
        return
    bound[port] = "listening"
    sock.settimeout(0.2)
    while not stop.is_set():
        try:
            conn, _ = sock.accept()
        except socket.timeout:  # noqa: UP041 - an alias of TimeoutError only from Python 3.10
            continue
        except OSError:
            break
        conn.close()
    sock.close()


def dial(host: str, port: int, timeout: float) -> str:
    """``ok``, ``refused`` (something on the path rejects), ``timeout``
    (a firewall that drops) or ``error: ...``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "ok"
    except ConnectionRefusedError:
        return "refused"
    except socket.timeout:  # noqa: UP041 - an alias of TimeoutError only from Python 3.10
        return "timeout"
    except OSError as exc:
        return f"error: {exc.strerror or exc}"


def run_ports(ports: list[int], peers: list[str], *, wait_s: float, hold_s: float,
              dial_timeout_s: float, me: str) -> dict:
    stop = threading.Event()
    bound: dict[int, str] = {}
    threads = [threading.Thread(target=_listen, args=(p, stop, bound), daemon=True) for p in ports]
    for t in threads:
        t.start()
    start = time.monotonic()
    time.sleep(wait_s)
    dials = {peer: {str(p): dial(peer, p, dial_timeout_s) for p in ports} for peer in peers}
    remaining = hold_s - (time.monotonic() - start)
    if remaining > 0:
        time.sleep(remaining)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    return {"mode": "ports", "node": me, "listen": {str(p): bound.get(p, "unknown") for p in ports},
            "dial": dials}


# ---------------------------------------------------------------- collectives
def busbw_factor(op: str, world: int) -> float:
    """nccl-tests bus-bandwidth factor of ``op`` over ``world`` ranks."""
    if world <= 1:
        return 1.0
    if op == "all_reduce":
        return 2.0 * (world - 1) / world
    if op == "all_to_all":
        return (world - 1) / world
    if op == "sendrecv":
        return 1.0
    raise ValueError(f"unknown op {op!r}")


def parse_size(text: str) -> int:
    """``64M`` / ``1G`` / ``4096`` -> bytes (binary units)."""
    text = text.strip().upper().rstrip("B")
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if text and text[-1] in scale:
        return int(float(text[:-1]) * scale[text[-1]])
    return int(text)


def run_collectives(ops: list[str], sizes: list[int], *, iters: int, warmup: int, backend: str = "nccl",
                    local_world: int | None = None) -> int:
    """``backend`` ``gloo`` runs the same ops on CPU tensors (tests of this script);
    ``local_world`` overrides the ranks per node (default: ``LOCAL_WORLD_SIZE``)."""
    import torch
    import torch.distributed as dist

    cuda = backend == "nccl"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def barrier() -> None:
        dist.barrier(device_ids=[local_rank]) if cuda else dist.barrier()

    t_init = time.monotonic()
    dist.init_process_group(backend, **({"device_id": device} if cuda else {}))
    world, rank = dist.get_world_size(), dist.get_rank()
    local_world = int(local_world or os.environ.get("LOCAL_WORLD_SIZE") or
                      (torch.cuda.device_count() if cuda else world))
    nodes = max(1, world // max(1, local_world))
    # the first collective builds the communicator: its time is the NCCL init cost
    probe = torch.ones(1, device=device)
    dist.all_reduce(probe)
    sync()
    init_s = time.monotonic() - t_init
    nccl = ".".join(str(v) for v in torch.cuda.nccl.version()) if cuda else backend
    if rank == 0:
        emit({"mode": "collectives", "event": "init", "world": world, "nodes": nodes,
              "local_world": local_world, "nccl": nccl, "torch": torch.__version__,
              "init_s": round(init_s, 3)})
    failed = 0
    peer_next = (rank + local_world) % world
    peer_prev = (rank - local_world) % world

    def check(op: str) -> bool:
        n = 1024 * world
        if op == "all_reduce":
            x = torch.full((n,), float(rank + 1), device=device)
            dist.all_reduce(x)
            return bool(torch.all(x == world * (world + 1) / 2).item())
        if op == "all_to_all":
            chunk = n // world
            x = (torch.arange(world, device=device, dtype=torch.float32) + rank * 1000).repeat_interleave(chunk)
            y = torch.empty_like(x)
            dist.all_to_all_single(y, x)
            want = (torch.arange(world, device=device, dtype=torch.float32) * 1000 + rank).repeat_interleave(chunk)
            return bool(torch.equal(y, want))
        if op == "sendrecv":
            x = torch.full((n,), float(rank), device=device)
            y = torch.empty_like(x)
            for req in dist.batch_isend_irecv([dist.P2POp(dist.isend, x, peer_next),
                                               dist.P2POp(dist.irecv, y, peer_prev)]):
                req.wait()
            return bool(torch.all(y == float(peer_prev)).item())
        raise ValueError(op)

    for op in ops:
        if op == "sendrecv" and nodes < 2:
            continue
        ok = check(op)
        flag = torch.tensor([0.0 if ok else 1.0], device=device)
        dist.all_reduce(flag)
        correct = flag.item() == 0
        for size in sizes:
            numel = max(world, size // 4 // world * world)
            x = torch.zeros(numel, dtype=torch.float32, device=device)
            y = torch.empty_like(x)

            def step(op: str = op, x: Any = x, y: Any = y) -> None:
                if op == "all_reduce":
                    dist.all_reduce(x)
                elif op == "all_to_all":
                    dist.all_to_all_single(y, x)
                else:
                    for req in dist.batch_isend_irecv([dist.P2POp(dist.isend, x, peer_next),
                                                       dist.P2POp(dist.irecv, y, peer_prev)]):
                        req.wait()

            for _ in range(warmup):
                step()
            sync()
            barrier()
            t0 = time.perf_counter()
            if cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            for _ in range(iters):
                step()
            if cuda:
                end.record()
                sync()
                per_iter = start.elapsed_time(end) / 1000.0 / iters
            else:
                per_iter = (time.perf_counter() - t0) / iters
            # the slowest rank sets the collective's time
            elapsed = torch.tensor([per_iter], device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            seconds = float(elapsed.item())
            nbytes = numel * 4
            algbw = nbytes / seconds / 1e9 if seconds > 0 else 0.0
            if rank == 0:
                emit({"mode": "collectives", "event": "result", "op": op, "bytes": nbytes, "world": world,
                      "nodes": nodes, "time_us": round(seconds * 1e6, 1), "algbw_gbps": round(algbw, 2),
                      "busbw_gbps": round(algbw * busbw_factor(op, world), 2), "correct": correct})
        if not correct:
            failed += 1
    barrier()
    dist.destroy_process_group()
    if rank == 0:
        emit({"mode": "collectives", "event": "done", "failed_ops": failed})
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    ports = sub.add_parser("ports", help="listen on ports and dial the same ports on the peers")
    ports.add_argument("--ports", required=True, help="comma-separated TCP ports")
    ports.add_argument("--peers", default="", help="comma-separated peer addresses")
    ports.add_argument("--me", default=socket.gethostname(), help="this node's label in the output")
    ports.add_argument("--wait", type=float, default=5.0, help="seconds to wait before dialing")
    ports.add_argument("--hold", type=float, default=15.0, help="seconds to keep listening")
    ports.add_argument("--dial-timeout", type=float, default=3.0)
    coll = sub.add_parser("collectives", help="NCCL collective bandwidth (run under torchrun)")
    coll.add_argument("--ops", default="all_reduce,all_to_all,sendrecv")
    coll.add_argument("--sizes", default="16M,128M,1G")
    coll.add_argument("--iters", type=int, default=20)
    coll.add_argument("--warmup", type=int, default=5)
    coll.add_argument("--backend", choices=("nccl", "gloo"), default="nccl", help="gloo: CPU tensors (tests)")
    coll.add_argument("--local-world", type=int, default=None, help="ranks per node (default LOCAL_WORLD_SIZE)")
    args = parser.parse_args(argv)
    if args.mode == "ports":
        record = run_ports([int(p) for p in args.ports.split(",") if p.strip()],
                           [p for p in args.peers.split(",") if p.strip()],
                           wait_s=args.wait, hold_s=args.hold, dial_timeout_s=args.dial_timeout, me=args.me)
        emit(record)
        return 0
    return run_collectives([o.strip() for o in args.ops.split(",") if o.strip()],
                           [parse_size(s) for s in args.sizes.split(",") if s.strip()],
                           iters=args.iters, warmup=args.warmup, backend=args.backend,
                           local_world=args.local_world)


if __name__ == "__main__":
    sys.exit(main())
