#!/usr/bin/env python3
"""m5 cross-node NCCL all-reduce probe: 2 nodes x 4 local ranks (world 8), torch.distributed NCCL init over
TCP sockets (NCCL_IB_DISABLE=1), all_reduce latency per message size (5 warmup + 20 timed iterations, cuda
synchronized). Run on every node: python3 s1m5nccl.py <node_rank> <nnodes> <master_addr> <port> <out.json>
Rank 0 writes {ok, world, nodes, init_s, results:[{bytes, lat_us, algbw_gbps, busbw_gbps}], error}."""
import json, os, sys, time, traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [4 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20]


def worker(local, node_rank, nnodes, addr, port, out):
    per = torch.cuda.device_count()
    rank, world = node_rank * per + local, nnodes * per
    res = {"ok": False, "world": world, "nodes": nnodes, "results": [], "error": None}
    try:
        os.environ.update(MASTER_ADDR=addr, MASTER_PORT=str(port))
        torch.cuda.set_device(local)
        t0 = time.time()
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=__import__("datetime").timedelta(seconds=300))
        x = torch.ones(1, device="cuda"); dist.all_reduce(x); torch.cuda.synchronize()
        res["init_s"] = round(time.time() - t0, 3)
        res["sum_check"] = float(x.item()) == float(world)
        for size in SIZES:
            t = torch.ones(size // 4, dtype=torch.float32, device="cuda")
            for _ in range(5):
                dist.all_reduce(t)
            torch.cuda.synchronize(); dist.barrier()
            t1 = time.perf_counter()
            for _ in range(20):
                dist.all_reduce(t)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t1) / 20
            alg = size / lat / 1e9
            res["results"].append({"bytes": size, "lat_us": round(lat * 1e6, 1), "algbw_gbps": round(alg, 3),
                                   "busbw_gbps": round(alg * 2 * (world - 1) / world, 3)})
        res["ok"] = bool(res["sum_check"])
        dist.destroy_process_group()
    except Exception:
        res["error"] = traceback.format_exc()[-2000:]
    if rank == 0 or res["error"]:
        path = out if rank == 0 else f"{out}.rank{rank}.err"
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    node_rank, nnodes, addr, port, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    mp.spawn(worker, args=(node_rank, nnodes, addr, port, out), nprocs=torch.cuda.device_count(), join=True)
