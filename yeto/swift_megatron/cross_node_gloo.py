"""Route cross-node process groups through gloo with host staging.

Use case: a pipeline-parallel island built from two (or more) Ascend nodes that
have no RoCE between them.  HCCL cannot form a communicator across such nodes
(EI0015 on the first collective), but PP only needs one activation P2P per
micro-batch per direction at the stage boundary, which plain TCP handles.

With this module installed (``YETO_CROSS_NODE_GLOO=1``):

* ``torch.distributed.new_group`` creates every group whose ranks span more
  than one node with the gloo backend; groups inside one node keep the default
  (HCCL) backend, so TP/EP/DP traffic stays on HCCS.
* Collectives and P2P ops issued on such a group -- or on the default/WORLD
  group when the job spans nodes -- are redirected to the gloo group and run on
  host copies of the device tensors; results are copied back to the device.

Rank-to-node mapping assumes torchrun's contiguous layout
(``rank // LOCAL_WORLD_SIZE``).  Load as an ms-swift external plugin before
Megatron builds its parallel state.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

logger = logging.getLogger(__name__)

_ENV = "YETO_CROSS_NODE_GLOO"
_installed = False
_orig: dict = {}
_cross: dict = {}  # id(group) -> group, gloo groups spanning nodes
_world_gloo = None
stats = {"p2p_bytes": 0, "coll_bytes": 0}


def _local_world() -> int:
    return int(os.environ.get("YETO_CROSS_NODE_LOCAL_WORLD") or os.environ.get("LOCAL_WORLD_SIZE") or 8)


def node_of(rank: int, local_world: int | None = None) -> int:
    return rank // (local_world or _local_world())


def spans_nodes(ranks, local_world: int | None = None) -> bool:
    return len({node_of(r, local_world) for r in ranks}) > 1


def _job_spans_nodes() -> bool:
    return dist.is_initialized() and spans_nodes(range(dist.get_world_size()))


def _get_world_gloo():
    global _world_gloo
    if _world_gloo is None:
        _world_gloo = _orig["new_group"](backend="gloo")
        _cross[id(_world_gloo)] = _world_gloo
    return _world_gloo


def _redirect(group):
    """Return the gloo group to use instead of ``group``, or None to keep it."""
    if group is None or group is c10d.GroupMember.WORLD:
        return _get_world_gloo() if _job_spans_nodes() else None
    return group if id(group) in _cross else None


def _new_group(ranks=None, timeout=None, backend=None, pg_options=None, **kwargs):
    if _job_spans_nodes():
        _get_world_gloo()  # all ranks reach the first new_group together
        rr = range(dist.get_world_size()) if ranks is None else ranks
        wants_device = backend is None or "gloo" not in str(backend).lower()
        if spans_nodes(list(rr)) and wants_device:
            kwargs.pop("device_id", None)
            g = _orig["new_group"](ranks, timeout=timeout, backend="gloo", **kwargs)
            if g is not None and g is not c10d.GroupMember.NON_GROUP_MEMBER:
                _cross[id(g)] = g
            return g
    return _orig["new_group"](ranks, timeout=timeout, backend=backend, pg_options=pg_options, **kwargs)


class _Done:
    """Work handle for an op that already completed synchronously."""

    def __init__(self, finish=None):
        self._finish = finish

    def wait(self, timeout=None):
        if self._finish is not None:
            f, self._finish = self._finish, None
            with torch.no_grad():  # recv buffers may be leaves that require grad
                f()
        return True

    def is_completed(self):
        return self._finish is None

    def is_success(self):
        return True

    def exception(self):
        return None

    def synchronize(self):
        self.wait()

    def get_future(self):
        fut = torch.futures.Future()
        self.wait()
        fut.set_result(None)
        return fut


class _Pending(_Done):
    """Wraps an async gloo work; on wait, copies host results back to device."""

    def __init__(self, work, finish=None):
        super().__init__(finish)
        self._work = work

    def wait(self, timeout=None):
        if self._work is not None:
            w, self._work = self._work, None
            w.wait()
        return super().wait()

    def is_completed(self):
        return self._work is None and self._finish is None


def _host(t):
    if t is None or t.device.type == "cpu":
        return t
    return t.detach().to("cpu")


def _bytes(ts):
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def _staged_collective(name):
    orig = _orig[name]
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        try:
            ba = sig.bind(*args, **kwargs)
        except TypeError:
            return orig(*args, **kwargs)
        g = _redirect(ba.arguments.get("group"))
        if g is None:
            return orig(*args, **kwargs)
        a = dict(ba.arguments)
        a["group"] = g
        async_op = bool(a.pop("async_op", False))
        a.pop("device_ids", None)
        _run_on_host(name, a)
        return _Done() if async_op else None

    return wrapper


def _avg_fix(a):
    op = a.get("op", dist.ReduceOp.SUM)
    if op == dist.ReduceOp.AVG:
        a["op"] = dist.ReduceOp.SUM
        return True
    return False


@torch.no_grad()
def _run_on_host(name, a):
    o = _orig
    g = a["group"]
    if name == "barrier":
        o["barrier"](group=g)
    elif name in ("all_reduce", "reduce", "broadcast"):
        t = a["tensor"]
        h = _host(t)
        stats["coll_bytes"] += _bytes([h])
        avg = _avg_fix(a) if name != "broadcast" else False
        a["tensor"] = h
        o[name](**a)
        if avg:
            h.div_(g.size())
        if h is not t:
            t.copy_(h)
    elif name in ("all_gather_into_tensor", "_all_gather_base"):
        out, inp = a["output_tensor"], a["input_tensor"]
        hin = _host(inp).contiguous()
        parts = [torch.empty_like(hin) for _ in range(g.size())]
        stats["coll_bytes"] += _bytes([hin]) * g.size()
        o["all_gather"](parts, hin, group=g)
        out.copy_(torch.cat([p.reshape(-1) for p in parts]).reshape(out.shape))
    elif name in ("reduce_scatter_tensor", "_reduce_scatter_base"):
        out, inp = a["output"], a["input"]
        h = _host(inp).clone()
        avg = _avg_fix(a)
        stats["coll_bytes"] += _bytes([h])
        o["all_reduce"](h, op=a.get("op", dist.ReduceOp.SUM), group=g)
        if avg:
            h.div_(g.size())
        r = g.rank()
        flat = h.reshape(-1)
        n = out.numel()
        out.copy_(flat[r * n:(r + 1) * n].reshape(out.shape))
    elif name == "all_gather":
        outs, t = a["tensor_list"], a["tensor"]
        ht = _host(t)
        houts = [torch.empty_like(ht) for _ in outs]
        stats["coll_bytes"] += _bytes([ht]) * len(outs)
        o["all_gather"](houts, ht, group=g)
        for d, s in zip(outs, houts):
            d.copy_(s)
    elif name == "all_to_all_single":
        out, inp = a["output"], a["input"]
        hout = torch.empty(out.shape, dtype=out.dtype)
        o["all_to_all_single"](hout, _host(inp), a.get("output_split_sizes"), a.get("input_split_sizes"), group=g)
        out.copy_(hout)
    else:  # pragma: no cover - guarded by _COLLECTIVES
        raise NotImplementedError(name)


def _isend(tensor, dst=None, group=None, tag=0, group_dst=None):
    g = _redirect(group)
    if g is None:
        return _orig["isend"](tensor, dst, group=group, tag=tag, group_dst=group_dst)
    h = _host(tensor).contiguous()
    stats["p2p_bytes"] += _bytes([h])
    w = _orig["isend"](h, dst, group=g, tag=tag, group_dst=group_dst)
    return _Pending(w, finish=lambda: h)  # keeps h alive until wait


def _irecv(tensor, src=None, group=None, tag=0, group_src=None):
    g = _redirect(group)
    if g is None:
        return _orig["irecv"](tensor, src, group=group, tag=tag, group_src=group_src)
    h = torch.empty(tensor.shape, dtype=tensor.dtype)
    stats["p2p_bytes"] += _bytes([h])
    w = _orig["irecv"](h, src, group=g, tag=tag, group_src=group_src)
    return _Pending(w, finish=lambda: tensor.copy_(h))


def _send(tensor, dst=None, group=None, tag=0, group_dst=None):
    if _redirect(group) is None:
        return _orig["send"](tensor, dst, group=group, tag=tag, group_dst=group_dst)
    _isend(tensor, dst, group=group, tag=tag, group_dst=group_dst).wait()


def _recv(tensor, src=None, group=None, tag=0, group_src=None):
    if _redirect(group) is None:
        return _orig["recv"](tensor, src, group=group, tag=tag, group_src=group_src)
    w = _irecv(tensor, src, group=group, tag=tag, group_src=group_src)
    w.wait()
    return src


def _batch_isend_irecv(p2p_op_list):
    if not any(_redirect(op.group) is not None for op in p2p_op_list):
        return _orig["batch_isend_irecv"](p2p_op_list)
    reqs = []
    for op in p2p_op_list:
        peer = op.peer
        if op.op in (_isend, _orig["isend"]):
            reqs.append(_isend(op.tensor, peer, group=op.group, tag=op.tag))
        else:
            reqs.append(_irecv(op.tensor, peer, group=op.group, tag=op.tag))
    return reqs


def _object_coll(name):
    orig = _orig[name]
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        g = _redirect(ba.arguments.get("group"))
        if g is not None:
            ba.arguments["group"] = g
            ba.arguments.pop("device", None)
        return orig(*ba.args, **ba.kwargs)

    return wrapper


_COLLECTIVES = (
    "all_reduce", "reduce", "broadcast", "barrier", "all_gather", "all_gather_into_tensor",
    "_all_gather_base", "reduce_scatter_tensor", "_reduce_scatter_base", "all_to_all_single",
)
_OBJECT = ("all_gather_object", "broadcast_object_list", "gather_object", "scatter_object_list")
_P2P = {"isend": _isend, "irecv": _irecv, "send": _send, "recv": _recv, "batch_isend_irecv": _batch_isend_irecv}


def _set(name, fn):
    setattr(c10d, name, fn)
    if hasattr(dist, name):
        setattr(dist, name, fn)


def install() -> bool:
    """Patch torch.distributed in place. Idempotent."""
    global _installed
    if _installed:
        return False
    names = ("new_group",) + _COLLECTIVES + _OBJECT + tuple(_P2P)
    for n in names:
        if hasattr(c10d, n):
            _orig[n] = getattr(c10d, n)
    _set("new_group", _new_group)
    for n in _COLLECTIVES:
        if n in _orig:
            _set(n, _staged_collective(n))
    for n in _OBJECT:
        if n in _orig:
            _set(n, _object_coll(n))
    for n, fn in _P2P.items():
        _set(n, fn)
    _installed = True
    if _job_spans_nodes():
        _get_world_gloo()
    logger.warning("[cross_node_gloo] installed (local_world=%d)", _local_world())
    print(f"[cross_node_gloo] installed local_world={_local_world()}", flush=True)
    return True


def cross_groups():
    return list(_cross.values())


if os.environ.get(_ENV) == "1":
    install()
