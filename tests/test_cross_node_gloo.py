"""cross_node_gloo: groups spanning "nodes" go to gloo; staged collectives and
P2P keep their semantics. CPU-only: 4 gloo ranks, 2 ranks per simulated node."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from tests.free_ports import free_port  # noqa: E402
from yeto.swift_megatron import cross_node_gloo as X  # noqa: E402

WORLD, LOCAL = 4, 2


def test_node_mapping():
    assert X.node_of(5, 4) == 1
    assert X.spans_nodes([0, 1], 2) is False
    assert X.spans_nodes([1, 2], 2) is True


def _worker(rank, port, q):
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), YETO_CROSS_NODE_LOCAL_WORLD=str(LOCAL))
        dist.init_process_group("gloo", rank=rank, world_size=WORLD)
        X.install()
        intra = [dist.new_group([n * LOCAL + i for i in range(LOCAL)]) for n in range(WORLD // LOCAL)]
        pps = [dist.new_group([i + n * LOCAL for n in range(WORLD // LOCAL)]) for i in range(LOCAL)]
        mine_intra, pp = intra[rank // LOCAL], pps[rank % LOCAL]
        assert id(pp) in X._cross and id(mine_intra) not in X._cross

        t = torch.full((3,), float(rank))
        dist.all_reduce(t)  # WORLD is redirected to the gloo world group
        assert t.tolist() == [6.0] * 3

        t = torch.full((2,), float(rank))
        dist.all_reduce(t, op=dist.ReduceOp.AVG, group=pp)
        peer = (rank + LOCAL) % WORLD
        assert t.tolist() == [(rank + peer) / 2] * 2

        out = torch.empty(4)
        dist.all_gather_into_tensor(out, torch.full((2,), float(rank)), group=pp)
        lo, hi = sorted((rank, peer))
        assert out.tolist() == [lo, lo, hi, hi]

        rs = torch.empty(2)
        dist.reduce_scatter_tensor(rs, torch.arange(4, dtype=torch.float32) + rank, group=pp)
        r = pp.rank()
        assert rs.tolist() == [2 * (2 * r) + rank + peer, 2 * (2 * r + 1) + rank + peer]

        objs = [None] * WORLD
        dist.all_gather_object(objs, rank)
        assert objs == list(range(WORLD))

        x, y = torch.full((5,), float(rank)), torch.empty(5)
        ops = [dist.P2POp(dist.isend, x, peer, pp), dist.P2POp(dist.irecv, y, peer, pp)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        assert y.tolist() == [float(peer)] * 5

        y2 = torch.empty(5)
        if rank < LOCAL:
            a, b = dist.isend(x, peer, group=pp), dist.irecv(y2, peer, group=dist.group.WORLD)
        else:
            b, a = dist.irecv(y2, peer, group=pp), dist.isend(x, peer, group=dist.group.WORLD)
        a.wait()
        b.wait()
        assert y2.tolist() == [float(peer)] * 5
        dist.barrier()
        dist.destroy_process_group()
        q.put((rank, "ok"))
    except Exception as e:  # pragma: no cover - reported to the parent
        q.put((rank, repr(e)))


def test_cross_node_collectives_and_p2p():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    res = dict(q.get(timeout=120) for _ in range(WORLD))
    for p in procs:
        p.join(30)
    assert res == {r: "ok" for r in range(WORLD)}
