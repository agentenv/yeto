"""Full-parameter DiLoCo sync of an ms-swift Megatron island (yeto/swift_megatron/sync.py).

Megatron-Core does not run on CPU, so the islands here are gloo process groups
whose model chunks carry exactly the attributes the sync reads from a Megatron
model: pipeline-local ``decoder.layers.<k>`` with a global ``layer_number``,
TP-sharded parameters (``tensor_model_parallel`` / ``partition_dim``), expert
parameters (``allreduce = False``) under containers with
``local_expert_indices`` (SequentialMLP and TEGroupedMLP naming), a frozen
vision projection, and a DistributedOptimizer leaf whose inner optimizer is a
precision-aware ``HybridDeviceOptimizer`` (FP32 host masters behind
``param_to_fp32_param``), the production optimizer.

Every full tensor is a pure function of its global name, so each rank cuts
its shard out of the same full value and a gathered tensor can be checked
against it.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import socket
import struct
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from free_ports import free_port
from yeto.swift_megatron import sync as sm

ROOT = Path(__file__).resolve().parent.parent
H, V, F, E, LAYERS = 8, 12, 6, 4, 4


# ----------------------------------------------------------------------------- pure helpers
def test_global_names_cover_layers_and_both_expert_namings():
    layers = sm._global_layer_prefixes(_chunk(tp=1, pp=1, tp_size=2, pp_size=2))
    experts = sm._expert_modules(_chunk(tp=1, pp=1, tp_size=2, pp_size=2))
    name = sm.global_parameter_name
    assert name("decoder.layers.0.mixer.in_proj.weight", layers, experts) == "decoder.layers.2.mixer.in_proj.weight"
    assert (name("decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight", layers, experts)
            == "decoder.layers.2.mlp.experts.local_experts.3.linear_fc1.weight")
    assert (name("decoder.layers.1.mlp.experts.linear_fc1.weight0", layers, experts)
            == "decoder.layers.3.mlp.experts.linear_fc1.weight2")
    assert name("output_layer.weight", layers, experts) == "output_layer.weight"
    assert name("decoder.layers.0.mlp.router.weight", layers, experts) == "decoder.layers.2.mlp.router.weight"


def test_syncer_address_and_hmac_key_file(tmp_path):
    assert sm._syncer_spec("tcp://10.0.0.1:29400") == "10.0.0.1:29400"
    assert sm._syncer_spec("a:1, tcp://b:2") == "a:1,b:2"
    key = tmp_path / "hmac.key"
    key.write_text("s3cret\n")
    assert sm._read_hmac_key(str(key)) == b"s3cret"  # same encoding as YETO_ISLAND_HMAC_KEY
    assert sm._read_hmac_key(None) is None


def test_a_sharded_syncer_is_refused_before_connecting():
    with pytest.raises(ValueError, match="sharded syncers"):
        sm.MegatronDiLoCoSync([], None, syncer="a:1,b:2", learner_id=0, num_learners=1, fragments=1,
                              sync_interval_steps=1)


def test_chunks_are_bounded_and_keep_order():
    sizes = {"a": 3, "b": 3, "c": 10, "d": 1, "e": 1}
    assert sm.chunk_names(list(sizes), sizes.__getitem__, 6) == [["a", "b"], ["c"], ["d", "e"]]


# ----------------------------------------------------------------------------- fake Megatron island
def test_heartbeat_thread_beats_between_steps_and_stops_on_a_stall():
    class Beats:
        def __init__(self):
            self.steps = []

        def heartbeat(self, step):
            self.steps.append(step)

    client, progress = Beats(), [time.monotonic()]
    beat = sm._Heartbeat(client, 0.02, 0.3, lambda: 7, lambda: progress[0])
    deadline = time.monotonic() + 5
    while len(client.steps) < 3:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert set(client.steps) == {7}
    time.sleep(0.5)  # no progress for longer than the stall limit: silence
    quiet = len(client.steps)
    time.sleep(0.2)
    assert len(client.steps) == quiet
    progress[0] = time.monotonic()
    while len(client.steps) == quiet:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    beat.stop()
    assert not beat.thread.is_alive()


class _LinkClient:
    """The client surface ``_PushSender`` and ``_ready_pushes`` use: a link
    generation the test bumps, and a push result it chooses."""

    def __init__(self, result=True):
        self.connection_generation, self.result, self.pushes, self.pulls = 1, result, [], []

    def push_fragment_parts(self, *head_and_parts):
        if isinstance(self.result, BaseException):
            raise self.result
        self.pushes.append(head_and_parts[:-1])
        return self.result

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls


@pytest.mark.parametrize("outcome", ["not-queued", "torso-lost", "socket-error"])
def test_push_sender_treats_an_undelivered_push_as_a_drop(outcome):
    """``push_fragment_parts`` returns False when the link is down (nothing
    queued) and raises PartialMessageGenerationLost when the connection died
    mid-message: neither reaches the syncer, so the anchor is released
    instead of blocking that fragment's later pulls (or failing the island)."""

    from yeto.protocol import PartialMessageGenerationLost

    result = {"not-queued": False,
              "torso-lost": PartialMessageGenerationLost(connection_generation=1, connection_epoch=1,
                                                         queued_chunks=1, operation="PUSH_FRAGMENT"),
              "socket-error": ConnectionResetError("reset")}[outcome]
    stats = {"pushed": 0, "push_bytes": 0, "dropped_pushes": 0}
    dropped, sent = [], []
    sender = sm._PushSender(_LinkClient(result), stats, lambda *a: dropped.append(a), sent.append)
    sender.submit((2, 9, 1, 7, 40, 3, 300), parts=[b"\0" * 8])
    assert sender.wait_idle(5)
    sender.stop()
    assert dropped == [(2, 7, 9)] and sent == [2]
    assert stats["dropped_pushes"] == 1 and stats["pushed"] == 0
    sender.raise_error()  # not an island failure


def _bare_sync(client, fragments=2):
    from yeto.diloco_sync import DiLoCoSyncState

    sync = object.__new__(sm.MegatronDiLoCoSync)
    sync.client, sync.sender, sync.interval = client, None, 1
    sync.state = DiLoCoSyncState.create(fragments, track_anchors=False)
    sync.state.fragment_versions = [5] * fragments
    sync.anchored = [True] * fragments
    sync.spent, sync.spent_link, sync.pushed_from = [None] * fragments, [None] * fragments, [None] * fragments
    sync.iteration, sync.units_total, sync.stats = 20, 100, {"deferred_pushes": 0}
    return sync


def test_a_spent_anchor_is_released_once_the_island_redials():
    """A PUSH queued on a connection that later died may never have reached
    the syncer, and a restarted syncer has lost the round it was carried
    into; after the redial the next pull of that fragment is answered from
    the same anchor (the syncer refuses it if it does hold that delta)."""

    from yeto.protocol import PullRequest

    client = _LinkClient()
    sync = _bare_sync(client)
    sync.spent[0] = (5, 7)  # pushed from anchor 5 for round 7, on link 1
    sync._push_sent(0)
    assert sync.spent_link[0] == 1
    client.pulls = [PullRequest(0, 9, 1)]  # another round of the same anchor
    assert sync._ready_pushes() == []  # same link: the syncer has the delta
    assert sync.state.pending_pulls == [PullRequest(0, 9, 1)]
    client.connection_generation = 2  # the link was replaced
    assert [p[:3] for p in sync._ready_pushes()] == [(0, 9, 1)]
    assert sync.spent == [None, None] and sync.stats["released_anchors"] == 1
    # A record from the process the island resumed from: released likewise.
    sync.spent[1], sync.spent_link[1] = (5, 4), sm._PREVIOUS_PROCESS
    client.pulls = [PullRequest(1, 10, 1)]
    assert [p[:3] for p in sync._ready_pushes()] == [(1, 10, 1)]


def test_a_dropped_push_frees_its_anchor_on_the_same_link():
    from yeto.protocol import PullRequest

    client = _LinkClient()
    sync = _bare_sync(client)
    sync.spent[0] = (5, 7)
    sync._push_sent(0)
    sync._push_dropped(0, 5, 7)
    client.pulls = [PullRequest(0, 9, 1)]
    assert [p[:3] for p in sync._ready_pushes()] == [(0, 9, 1)]


def _full(name: str, shape) -> torch.Tensor:
    """The initial full value of a global tensor (bf16-exact, so INIT is lossless)."""

    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    g = torch.Generator().manual_seed(seed)
    return (1 + 0.25 * torch.randn(*shape, generator=g)).to(torch.bfloat16).float()


def _update(name: str, shape, island: int, step: int) -> torch.Tensor:
    """One inner step's update of a global tensor (a function of position only)."""

    seed = int.from_bytes(hashlib.sha256(f"{name}/{island}/{step}".encode()).digest()[:4], "little")
    g = torch.Generator().manual_seed(seed)
    sign = 1.0 if island == 0 else -1.0
    return sign * 1e-3 * (1 + torch.rand(*shape, generator=g))


class _Module(torch.nn.Module):
    pass


def _param(full_name, full_shape, *, tp, tp_size, part_dim=-1, expert=False, trainable=True):
    full = _full(full_name, full_shape)
    if part_dim >= 0:
        size = full_shape[part_dim] // tp_size
        full = full.narrow(part_dim, tp * size, size)
    p = torch.nn.Parameter(full.to(torch.bfloat16).clone(), requires_grad=trainable)
    if part_dim >= 0:
        p.tensor_model_parallel, p.partition_dim, p.partition_stride = True, part_dim, 1
    if expert:
        p.allreduce = False
    p._full_name, p._full_shape = full_name, tuple(full_shape)
    return p


def _chunk(*, tp: int, pp: int, tp_size: int, pp_size: int) -> torch.nn.Module:
    """One rank's model chunk.  EP spans the TP ranks with ETP 1 (each TP rank
    holds E / tp_size whole experts), as in the production recipe."""

    root = _Module()
    per_stage = LAYERS // pp_size
    if pp == 0:
        root.embedding = _Module()
        root.embedding.word_embeddings = _Module()
        root.embedding.word_embeddings.weight = _param("embedding.word_embeddings.weight", (V, H), tp=tp,
                                                       tp_size=tp_size, part_dim=0)
        root.vision_model = _Module()
        root.vision_model.proj = _Module()
        root.vision_model.proj.weight = _param("vision_model.proj.weight", (H, H), tp=tp, tp_size=tp_size,
                                               trainable=False)
    root.decoder = _Module()
    root.decoder.layers = torch.nn.ModuleList()
    local_experts = E // tp_size
    for k in range(per_stage):
        g = pp * per_stage + k
        layer = _Module()
        layer.layer_number = g + 1
        prefix = f"decoder.layers.{g}"
        layer.mixer = _Module()
        layer.mixer.in_proj = _Module()
        layer.mixer.in_proj.weight = _param(f"{prefix}.mixer.in_proj.weight", (2 * F, H), tp=tp,
                                            tp_size=tp_size, part_dim=0)
        layer.mixer.out_proj = _Module()
        layer.mixer.out_proj.weight = _param(f"{prefix}.mixer.out_proj.weight", (H, F), tp=tp,
                                             tp_size=tp_size, part_dim=1)
        layer.norm = _Module()
        layer.norm.weight = _param(f"{prefix}.norm.weight", (H,), tp=tp, tp_size=tp_size)
        layer.mlp = _Module()
        layer.mlp.local_expert_indices = [tp * local_experts + e for e in range(local_experts)]  # MoELayer
        layer.mlp.router = _Module()
        layer.mlp.router.weight = _param(f"{prefix}.mlp.router.weight", (E, H), tp=tp, tp_size=tp_size)
        layer.mlp.router.register_buffer("expert_bias", torch.zeros(E))  # no master: not synced
        experts = _Module()
        experts.local_expert_indices = [tp * local_experts + e for e in range(local_experts)]
        if g % 2 == 0:  # SequentialMLP
            experts.local_experts = torch.nn.ModuleList()
            for e in range(local_experts):
                mlp = _Module()
                mlp.linear_fc1 = _Module()
                mlp.linear_fc1.weight = _param(
                    f"{prefix}.mlp.experts.local_experts.{experts.local_expert_indices[e]}.linear_fc1.weight",
                    (F, H), tp=0, tp_size=1, expert=True)
                experts.local_experts.append(mlp)
        else:  # TEGroupedMLP
            experts.linear_fc1 = _Module()
            for e in range(local_experts):
                setattr(experts.linear_fc1, f"weight{e}", _param(
                    f"{prefix}.mlp.experts.linear_fc1.weight{experts.local_expert_indices[e]}",
                    (F, H), tp=0, tp_size=1, expert=True))
        layer.mlp.experts = experts
        root.decoder.layers.append(layer)
    if pp == pp_size - 1:
        root.decoder.final_layernorm = _Module()
        root.decoder.final_layernorm.weight = _param("decoder.final_layernorm.weight", (H,), tp=tp,
                                                     tp_size=tp_size)
        root.output_layer = _Module()
        root.output_layer.weight = _param("output_layer.weight", (V, H), tp=tp, tp_size=tp_size, part_dim=0)
    return root


def _global_shapes(tp_size: int, pp_size: int) -> dict[str, tuple[int, ...]]:
    """Every trainable global tensor of the island and its full shape."""

    return {p._full_name: p._full_shape
            for tp in range(tp_size) for pp in range(pp_size)
            for p in _chunk(tp=tp, pp=pp, tp_size=tp_size, pp_size=pp_size).parameters() if p.requires_grad}


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Hybrid:
    """megatron-core ``HybridDeviceOptimizer`` as ``yeto.rl.megatron_masters`` reads it."""

    def __init__(self, shards):
        self.param_groups = [{"params": shards}]
        self.param_update_in_fp32 = True
        self.param_to_fp32_param = {s: s.detach().clone().float() for s in shards}
        self.state = {s: {"exp_avg": torch.full((s.numel(),), 0.5), "exp_avg_sq": torch.full((s.numel(),), 0.25)}
                      for s in shards}


class _DistOpt:
    """A DistributedOptimizer leaf (precision-aware, CPU offload): this DP rank
    owns a contiguous range of the flattened local parameters."""

    def __init__(self, params, *, dp_rank, dp_size, dp_group):
        self.params = list(params)
        total = sum(p.numel() for p in self.params)
        per = -(-total // dp_size)
        lo, hi = dp_rank * per, min((dp_rank + 1) * per, total)
        param_map, shards, self.model_param_group_index_map = {}, [], {}
        self.owned = {}
        offset = 0
        for p in self.params:
            s, e = max(lo, offset) - offset, min(hi, offset + p.numel()) - offset
            if e > s:
                param_map[p] = {"param": _Range(s, e)}
                self.model_param_group_index_map[p] = (0, len(shards))
                shards.append(p.detach().view(-1)[s:e])
                self.owned[id(p)] = (s, e, len(shards) - 1)
            offset += p.numel()
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True, bf16=True,
                                      main_params_dtype=torch.float32, optimizer_cpu_offload=True,
                                      store_param_remainders=True)
        self.data_parallel_group = dp_group
        self.optimizer = _Hybrid(shards)

    def master(self, p):
        s, e, i = self.owned[id(p)]
        return self.optimizer.param_to_fp32_param[self.optimizer.param_groups[0]["params"][i]]

    def step(self, local_updates):
        """AdamW on the host FP32 copy, then the copy-back hook into the bf16 shard."""

        for p in self.params:
            if id(p) not in self.owned:
                continue
            s, e, i = self.owned[id(p)]
            shard = self.optimizer.param_groups[0]["params"][i]
            master = self.optimizer.param_to_fp32_param[shard]
            master.add_(local_updates[id(p)].reshape(-1)[s:e])
            shard.copy_(master)
        # The bf16 copy outside this rank's range stays stale until the next
        # forward's param all-gather (overlap_param_gather): the sync must not read it.


def _local_update(p, island, step, tp, tp_size):
    full = _update(p._full_name, p._full_shape, island, step)
    if getattr(p, "tensor_model_parallel", False):
        size = p._full_shape[p.partition_dim] // tp_size
        full = full.narrow(p.partition_dim, tp * size, size)
    return full


def _init_dist(rank, world, port):
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    return dist


def _run(target, world, *args, timeout=240):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=target, args=(rank, world, *args)) for rank in range(world)]
    for p in procs:
        p.start()
    return procs


def _join(procs, timeout, out_dir):
    deadline = time.monotonic() + timeout
    for p in procs:
        p.join(max(1.0, deadline - time.monotonic()))
    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        p.kill()
    errors = sorted(Path(out_dir).glob("error-*.txt"))
    detail = "\n".join(e.read_text() for e in errors)
    assert not alive, f"{len(alive)} worker(s) hung\n{detail}"
    assert all(p.exitcode == 0 for p in procs), f"worker exit codes {[p.exitcode for p in procs]}\n{detail}"


def _guard(out_dir, tag, fn):
    try:
        fn()
    except BaseException:
        Path(out_dir, f"error-{tag}.txt").write_text(f"[{tag}]\n{traceback.format_exc()}")
        raise


# ----------------------------------------------------------------------------- scripted client (TP2 x DP2)
class _ScriptedClient:
    """The SyncerClient surface the sync uses, driven by a per-iteration script."""

    def __init__(self, layout, script):
        import threading

        self.layout, self.script = layout, script
        self.finalizing, self.shutdown, self.finalized = threading.Event(), threading.Event(), threading.Event()
        self.inits, self.pushes, self.calls = {}, [], 0

    def start(self):
        pass

    def close(self):
        pass

    def check_health(self):
        self.calls += 1

    def send_init_parts(self, fid, parts):
        self.inits[fid] = b"".join(bytes(p) for p in parts)
        return True

    def drain_updates(self):
        return self.script.get(("bcast", self.calls), [])

    def drain_pulls(self):
        return self.script.get(("pull", self.calls), [])

    def push_fragment_parts(self, fid, step, attempt, base, local_step, c_steps, c_units, parts):
        self.pushes.append((fid, step, attempt, base, local_step, c_steps, c_units, b"".join(bytes(p) for p in parts)))
        return True


def _flat(layout, fid, values):
    return torch.cat([values[name].reshape(-1) for name, _ in layout.fragments[fid].tensors])


def _scripted_worker(rank, world, port, out_dir):
    _guard(out_dir, f"scripted-{rank}", lambda: _scripted_body(rank, world, port, out_dir))


def _scripted_body(rank, world, port, out_dir):
    from yeto.protocol import BcastFragment, PullRequest
    from yeto.tensor_io import pack_tensor, unpack_fragment

    dist = _init_dist(rank, world, port)
    tp, dp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[dp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=0, tp_size=2, pp_size=1)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=dp, dp_size=2, dp_group=dp_groups[tp])
    full_shapes = _global_shapes(2, 1)
    globals_g = {name: (_full(name, shape) + 0.125).to(torch.bfloat16).float()
                 for name, shape in full_shapes.items()}
    holder = {}
    alpha = 0.5

    def factory(addrs, learner_id, layout, wire, streams, **kwargs):
        assert addrs == "127.0.0.1:1234" and len(kwargs["session_contract_hash"]) == 32
        script = {}
        script[("bcast", 1)] = [BcastFragment(fid, 7, pack_tensor(_flat(layout, fid, globals_g), 2))
                                for fid in range(layout.num_fragments)]
        # Pulls of every fragment arrive with the broadcast (so must wait for two steps)
        script[("pull", 1)] = [PullRequest(fid, 8, 1) for fid in range(layout.num_fragments)]
        holder["client"] = client = _ScriptedClient(layout, script)
        return client

    frozen_before = chunk.vision_model.proj.weight.detach().clone()
    syncer = sm.MegatronDiLoCoSync([chunk], opt, syncer="tcp://127.0.0.1:1234", learner_id=0, num_learners=2,
                                   fragments=3, sync_interval_steps=2, merge_alpha=alpha, chunk_numel=40,
                                   parallel_context=ctx, client_factory=factory)
    names = set(syncer.layout.tensor_names())
    assert names == set(full_shapes) and "vision_model.proj.weight" not in names
    assert syncer.layout.fragments[0].tensors[0][0] in ("embedding.word_embeddings.weight", "output_layer.weight")
    if rank == 0:  # INIT is the current FP32 masters, gathered over TP and DP
        client = holder["client"]
        for fid, frag in enumerate(syncer.layout.fragments):
            got = unpack_fragment(frag, client.inits[fid], 2)
            want = _flat(syncer.layout, fid, {n: _full(n, s) for n, s in full_shapes.items()})
            assert torch.equal(got, want), f"INIT fragment {fid}"

    masters_before = {id(p): opt.master(p).clone() for p in trainable if id(p) in opt.owned}
    adam_before = {k: {n: t.clone() for n, t in v.items()} for k, v in opt.optimizer.state.items()}
    updates_total = {name: torch.zeros(shape) for name, shape in full_shapes.items()}
    applied = {}
    snapshots = {}  # iteration -> updates since the merge, at the end of that step
    for it in (1, 2, 3, 4, 5):
        opt.step({id(p): _local_update(p, 0, it, tp, 2) for p in trainable})
        for name, shape in full_shapes.items():
            if it >= 2:  # updates after the merge at iteration 1
                updates_total[name] += _update(name, shape, 0, it)
        if it == 1:
            pre_merge = {name: _full(name, shape) + _update(name, shape, 0, 1)
                         for name, shape in full_shapes.items()}
        snapshots[it] = {n: t.clone() for n, t in updates_total.items()}
        syncer.after_optimizer_step(it, 100 * it)
        if rank == 0:  # the PUSH streams in the background; let it land before the next step
            assert syncer.sender.wait_idle(10)
        if it == 1:
            # merged with alpha: master = a * local + (1 - a) * global, bf16 copy = master
            for p in trainable:
                full = alpha * pre_merge[p._full_name] + (1 - alpha) * globals_g[p._full_name]
                if getattr(p, "tensor_model_parallel", False):
                    size = p._full_shape[p.partition_dim] // 2
                    full = full.narrow(p.partition_dim, tp * size, size)
                applied[p._full_name] = full
                if id(p) in opt.owned:
                    s, e, _ = opt.owned[id(p)]
                    assert torch.allclose(opt.master(p), full.reshape(-1)[s:e], atol=1e-6), p._full_name
                    assert not torch.equal(opt.master(p), masters_before[id(p)])
                assert torch.equal(p.detach(), full.to(torch.bfloat16)), p._full_name
            assert syncer.anchored == [True] * syncer.num_fragments
            assert [e[1] for e in syncer.events] == ["apply"] * syncer.num_fragments
        if it == 2:  # one step since the merge: the pulls wait (H = 2)
            assert not any(e[1] == "push" for e in syncer.events)
    pushes = [e for e in syncer.events if e[1] == "push"]
    # decoupled: one fragment per boundary, the PUSH returns at once
    assert [(e[0], e[2]) for e in pushes] == [(3 + f, f) for f in range(syncer.num_fragments)]
    assert torch.equal(chunk.vision_model.proj.weight, frozen_before)
    for key, state in opt.optimizer.state.items():  # Adam moments kept
        for n, t in state.items():
            assert torch.equal(t, adam_before[key][n])
    if rank == 0:
        client = holder["client"]
        assert len(client.pushes) == syncer.num_fragments
        for fid, step, attempt, base, local_step, c_steps, c_units, payload in client.pushes:
            it = 3 + fid
            assert (step, attempt, base, local_step, c_steps, c_units) == (
                8, 1, 7, it, it - 1, sum(100 * k for k in range(2, it + 1)))
            # PUSH = local - raw anchor (the global, not the blend)
            local = {n: alpha * pre_merge[n] + (1 - alpha) * globals_g[n] + snapshots[it][n] for n in full_shapes}
            want = _flat(syncer.layout, fid, local) - _flat(syncer.layout, fid, globals_g)
            got = unpack_fragment(syncer.layout.fragments[fid], payload, 2)
            assert torch.allclose(got, want.to(torch.bfloat16).float(), atol=1e-3), fid
    syncer.close()
    syncer.close()
    dist.destroy_process_group()


@pytest.mark.timeout(240)
def test_tp2_dp2_island_reads_masters_writes_masters_and_pushes_raw_anchor_deltas(tmp_path):
    procs = _run(_scripted_worker, 4, free_port(), str(tmp_path))
    _join(procs, 200, tmp_path)


def _initial_cut_worker(rank, world, port, out_dir):
    _guard(out_dir, f"initial-cut-{rank}", lambda: _initial_cut_body(rank, world, port, out_dir))


def _initial_cut_body(rank, world, port, out_dir):
    from yeto.protocol import BcastFragment
    from yeto.tensor_io import pack_tensor

    dist = _init_dist(rank, world, port)
    tp, dp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[dp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=0, tp_size=2, pp_size=1)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=dp, dp_size=2, dp_group=dp_groups[tp])
    full_shapes = _global_shapes(2, 1)
    globals_g = {name: (_full(name, shape) + 0.125).to(torch.bfloat16).float()
                 for name, shape in full_shapes.items()}
    holder = {}

    def factory(addrs, learner_id, layout, wire, streams, **kwargs):
        # The initial cut trickles in: fragment 0 at the first poll, the rest later.
        cut = [BcastFragment(fid, 0, pack_tensor(_flat(layout, fid, globals_g), 2))
               for fid in range(layout.num_fragments)]
        holder["client"] = client = _ScriptedClient(layout, {("bcast", 1): cut[:1], ("bcast", 4): cut[1:]})
        return client

    syncer = sm.MegatronDiLoCoSync([chunk], opt, syncer="tcp://127.0.0.1:1234", learner_id=1, num_learners=2,
                                   fragments=3, sync_interval_steps=1, chunk_numel=40,
                                   parallel_context=ctx, client_factory=factory, initial_cut_timeout=60)
    # Installed before the first step, fragment by fragment, as raw globals.
    assert syncer.anchored == [True] * syncer.num_fragments
    assert [(e[0], e[1], e[3]) for e in syncer.events] == [(0, "apply", 0)] * syncer.num_fragments
    for p in trainable:
        full = globals_g[p._full_name]
        if getattr(p, "tensor_model_parallel", False):
            size = p._full_shape[p.partition_dim] // 2
            full = full.narrow(p.partition_dim, tp * size, size)
        if id(p) in opt.owned:
            s, e, _ = opt.owned[id(p)]
            assert torch.equal(opt.master(p), full.reshape(-1)[s:e]), p._full_name
        assert torch.equal(p.detach(), full.to(torch.bfloat16)), p._full_name
    if rank == 0:
        assert holder["client"].calls >= 4 and not holder["client"].inits  # learner 1 sends no INIT
    # The boundary finds nothing left to apply.
    syncer.after_optimizer_step(1, 10)
    assert syncer.stats["applied"] == syncer.num_fragments
    syncer.close()
    dist.destroy_process_group()


@pytest.mark.timeout(240)
def test_initial_cut_is_installed_fragment_by_fragment_before_training(tmp_path):
    procs = _run(_initial_cut_worker, 4, free_port(), str(tmp_path))
    _join(procs, 200, tmp_path)


def _resume_worker(rank, world, port, out_dir):
    _guard(out_dir, f"resume-{rank}", lambda: _resume_body(rank, world, port, out_dir))


def _resume_body(rank, world, port, out_dir):
    """An island checkpoints after three steps and restarts from it; the
    syncer's rejoin catch-up must not undo the island's local progress."""

    from yeto.diloco_sync import shifted_resume_value
    from yeto.protocol import BcastFragment
    from yeto.tensor_io import pack_tensor

    dist = _init_dist(rank, world, port)
    tp, dp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[dp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=0, tp_size=2, pp_size=1)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=dp, dp_size=2, dp_group=dp_groups[tp])
    full_shapes = _global_shapes(2, 1)
    old_g = {n: (_full(n, s) + 0.125).to(torch.bfloat16).float() for n, s in full_shapes.items()}
    new_g = {n: (_full(n, s) - 0.25).to(torch.bfloat16).float() for n, s in full_shapes.items()}
    cut = {}  # fid -> (version, values) of the next sync's catch-up

    def factory(addrs, learner_id, layout, wire, streams, **kwargs):
        bcasts = [BcastFragment(fid, v, pack_tensor(_flat(layout, fid, values), 2))
                  for fid, (v, values) in sorted(cut.items())]
        return _ScriptedClient(layout, {("bcast", 1): bcasts})

    def make(resume_dir=None, resume_iteration=None):
        return sm.MegatronDiLoCoSync([chunk], opt, syncer="tcp://127.0.0.1:1234", learner_id=1, num_learners=2,
                                     fragments=3, sync_interval_steps=100, chunk_numel=40, parallel_context=ctx,
                                     client_factory=factory, initial_cut_timeout=60, resume_dir=resume_dir,
                                     resume_iteration=resume_iteration)

    def shard(full, p):
        if getattr(p, "tensor_model_parallel", False):
            size = p._full_shape[p.partition_dim] // 2
            full = full.narrow(p.partition_dim, tp * size, size)
        return full

    ckpt, bare = Path(out_dir, "checkpoint-3"), Path(out_dir, "checkpoint-3-bare")
    # The first run: the initial cut (version 4), three local steps, a checkpoint.
    cut.update({fid: (4, old_g) for fid in range(3)})
    first = make()
    for it in (1, 2, 3):
        opt.step({id(p): _local_update(p, 1, it, tp, 2) for p in trainable})
        first.after_optimizer_step(it, 10)
    assert first.save_state(str(ckpt), iteration=3, save_anchors=True)
    assert first.save_state(str(bare), iteration=3, save_anchors=False)
    fid_of = {name: fid for fid, frag in enumerate(first.layout.fragments) for name, _ in frag.tensors}
    first.close()
    if rank == 0:
        record = json.loads((ckpt / sm.RESUME_STATE_FILE).read_text())
        assert record["versions"] == [4, 4, 4] and record["anchors"] and record["iteration"] == 3
        assert record["steps_at_reset"] == [0, 0, 0] and record["units_total"] == 30
        assert not json.loads((bare / sm.RESUME_STATE_FILE).read_text())["anchors"]
    saved_master = {id(p): opt.master(p).clone() for p in trainable if id(p) in opt.owned}
    saved_param = {id(p): p.detach().clone() for p in trainable}

    def restart():  # the trainer loads the checkpoint again
        for p in trainable:
            if id(p) in opt.owned:
                opt.master(p).copy_(saved_master[id(p)])
            p.data.copy_(saved_param[id(p)])

    def check(sync, expect):  # expect(p) -> the full local FP32 value or None (= the checkpoint, bit for bit)
        assert sync.anchored == [True] * 3
        for p in trainable:
            want = expect(p)
            if want is None:
                if id(p) in opt.owned:
                    assert torch.equal(opt.master(p), saved_master[id(p)]), p._full_name
                assert torch.equal(p.detach(), saved_param[id(p)]), p._full_name
                continue
            if id(p) in opt.owned:
                s, e, _ = opt.owned[id(p)]
                assert torch.equal(opt.master(p), want.reshape(-1)[s:e]), p._full_name
                assert torch.equal(p.detach().reshape(-1)[s:e], want.reshape(-1)[s:e].to(torch.bfloat16))

    def check_anchors(sync, values):
        for p in trainable:
            assert torch.equal(sync.anchors[id(p)].float(), shard(values[p._full_name], p)), p._full_name

    # (a) The syncer is still at the checkpoint's anchor version: anchor only.
    restart()
    cut.update({fid: (4, old_g) for fid in range(3)})
    same = make(str(ckpt), 3)
    check(same, lambda p: None)
    check_anchors(same, old_g)
    assert same.resumed == {0: "anchor", 1: "anchor", 2: "anchor"}
    same.after_optimizer_step(4, 10)
    assert same.state.steps_at_reset == [0, 0, 0], "the H gate restarted with the island"
    assert same.units_total == 40
    same.close()

    # (b) The syncer moved on: global_new + (local - anchor_old), exactly.
    restart()
    cut.update({fid: (9, new_g) for fid in range(3)})
    moved = make(str(ckpt), 3)
    for p in trainable:
        if id(p) in opt.owned:
            s, e, _ = opt.owned[id(p)]
            want = shifted_resume_value(shard(new_g[p._full_name], p).reshape(-1)[s:e], saved_master[id(p)],
                                        shard(old_g[p._full_name], p).reshape(-1)[s:e])
            assert torch.equal(opt.master(p), want), p._full_name
            assert torch.equal(p.detach().reshape(-1)[s:e], want.to(torch.bfloat16)), p._full_name
            assert not torch.equal(want, shard(new_g[p._full_name], p).reshape(-1)[s:e])
    check_anchors(moved, new_g)
    assert set(moved.resumed.values()) == {"shift"} and not moved._resume_anchors
    moved.after_optimizer_step(4, 10)
    assert moved.state.steps_at_reset == [0, 0, 0] and moved.state.fragment_versions == [9, 9, 9]
    moved.close()

    # (c) Without saved anchors a moved fragment takes the global; an unmoved one is kept
    # (a record without the contract, as written by hand for an older checkpoint, is accepted).
    if rank == 0:
        record = json.loads((bare / sm.RESUME_STATE_FILE).read_text())
        del record["contract"]
        (bare / sm.RESUME_STATE_FILE).write_text(json.dumps(record))
    restart()
    cut.update({0: (4, old_g), 1: (9, new_g), 2: (9, new_g)})
    mixed = make(str(bare), 3)
    check(mixed, lambda p: None if fid_of[p._full_name] == 0 else shard(new_g[p._full_name], p))
    assert mixed.resumed == {0: "anchor", 1: "overwrite", 2: "overwrite"}
    mixed.after_optimizer_step(4, 10)
    assert mixed.state.steps_at_reset == [0, 3, 3]
    mixed.close()

    # (d) A cold start (no record, or a record of another iteration) takes the global.
    for resume in ((None, None), (str(ckpt), 7)):
        restart()
        cut.update({fid: (9, new_g) for fid in range(3)})
        cold = make(*resume)
        check(cold, lambda p: shard(new_g[p._full_name], p))
        assert cold.resumed == {} and cold.resume_record is None
        cold.close()
    dist.destroy_process_group()


@pytest.mark.timeout(240)
def test_resumed_island_keeps_its_checkpoint_through_the_rejoin_catch_up(tmp_path):
    procs = _run(_resume_worker, 4, free_port(), str(tmp_path))
    _join(procs, 200, tmp_path)


def _catch_up_worker(rank, world, port, out_dir):
    _guard(out_dir, f"catch-up-{rank}", lambda: _catch_up_body(rank, world, port, out_dir))


def _catch_up_body(rank, world, port, out_dir):
    """A running island redials and gets the syncer's catch-up: a replay of
    its anchor changes nothing (bit for bit), a rewound or newer catch-up of
    a fragment it did not push from keeps its local progress (global_new +
    (local - anchor)), a newer one of a fragment it pushed from its anchor is
    installed as an ordinary merge."""

    from yeto.diloco_sync import shifted_resume_value
    from yeto.protocol import BcastFragment, PullRequest
    from yeto.tensor_io import pack_tensor

    dist = _init_dist(rank, world, port)
    tp, dp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[dp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=0, tp_size=2, pp_size=1)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=dp, dp_size=2, dp_group=dp_groups[tp])
    full_shapes = _global_shapes(2, 1)
    old_g = {n: (_full(n, s) + 0.125).to(torch.bfloat16).float() for n, s in full_shapes.items()}
    new_g = {n: (_full(n, s) - 0.25).to(torch.bfloat16).float() for n, s in full_shapes.items()}

    def factory(addrs, learner_id, layout, wire, streams, **kwargs):
        assert kwargs["accept_rewound_catch_up"] is True
        client = _ScriptedClient(layout, script)
        script[("layout",)] = layout
        return client

    def bcast(fid, version, values, catch_up):
        layout = script[("layout",)]
        return BcastFragment(fid, version, pack_tensor(_flat(layout, fid, values), 2), catch_up=catch_up)

    plan = {1: [(fid, 4, old_g, False) for fid in range(3)],  # the initial cut
            3: "pull",                                         # iteration 2: a pull of fragment 2
            5: [(0, 4, old_g, False), (1, 2, new_g, True), (2, 9, new_g, True)],
            6: [(0, 9, new_g, True)]}

    class _Script(dict):  # call -> scripted pulls / broadcasts (packed once the layout is known)
        def get(self, key, default=None):
            kind, call = key
            entry = plan.get(call)
            if kind == "pull":
                return [PullRequest(2, 7, 1)] if entry == "pull" else []
            if isinstance(entry, list):
                return [bcast(*row) for row in entry]
            return []

    script = _Script()
    sync = sm.MegatronDiLoCoSync([chunk], opt, syncer="tcp://127.0.0.1:1234", learner_id=1, num_learners=2,
                                 fragments=3, sync_interval_steps=1, chunk_numel=40, parallel_context=ctx,
                                 client_factory=factory, initial_cut_timeout=60)
    fid_of = {name: fid for fid, frag in enumerate(sync.layout.fragments) for name, _ in frag.tensors}
    for it in (1, 2, 3):
        opt.step({id(p): _local_update(p, 1, it, tp, 2) for p in trainable})
        sync.after_optimizer_step(it, 10)
    if rank == 0:
        assert sync.sender.wait_idle(10) and sync.stats["pushed"] == 1 and sync.pushed_from == [None, None, 4]
    master = {id(p): opt.master(p).clone() for p in trainable if id(p) in opt.owned}
    param = {id(p): p.detach().clone() for p in trainable}

    def shard(full, p):
        if getattr(p, "tensor_model_parallel", False):
            size = p._full_shape[p.partition_dim] // 2
            full = full.narrow(p.partition_dim, tp * size, size)
        return full

    def owned(p, full):
        s, e, _ = opt.owned[id(p)]
        return shard(full, p).reshape(-1)[s:e]

    sync.after_optimizer_step(4, 10)  # call 5: no local step since the snapshot
    for p in trainable:
        fid = fid_of[p._full_name]
        if fid == 0:  # a replay of its anchor: untouched
            assert torch.equal(p.detach(), param[id(p)]), p._full_name
            if id(p) in opt.owned:
                assert torch.equal(opt.master(p), master[id(p)])
            assert torch.equal(sync.anchors[id(p)].float(), shard(old_g[p._full_name], p))
            continue
        assert torch.equal(sync.anchors[id(p)].float(), shard(new_g[p._full_name], p)), p._full_name
        if id(p) not in opt.owned:
            continue
        if fid == 1:  # rewound to version 2: rebased exactly
            want = shifted_resume_value(owned(p, new_g[p._full_name]), master[id(p)], owned(p, old_g[p._full_name]))
            assert not torch.equal(want, owned(p, new_g[p._full_name]))
        else:  # newer, pushed from its anchor: the ordinary install
            want = owned(p, new_g[p._full_name])
        assert torch.equal(opt.master(p), want), (fid, p._full_name)
        assert torch.equal(p.detach().reshape(-1)[opt.owned[id(p)][0]:opt.owned[id(p)][1]], want.to(torch.bfloat16))
    assert sync.state.fragment_versions == [4, 2, 9]
    assert sync.state.steps_at_reset[1] == 0 and sync.state.steps_at_reset[2] == 4, sync.state.steps_at_reset
    before = {id(p): opt.master(p).clone() for p in trainable if id(p) in opt.owned and fid_of[p._full_name] == 0}
    sync.after_optimizer_step(5, 10)  # call 6: a newer catch-up of fragment 0, never pushed: rebased
    for p in trainable:
        if fid_of[p._full_name] == 0 and id(p) in opt.owned:
            want = shifted_resume_value(owned(p, new_g[p._full_name]), before[id(p)], owned(p, old_g[p._full_name]))
            assert torch.equal(opt.master(p), want), p._full_name
    if rank == 0:
        assert sync.stats["confirmed"] == 1 and sync.stats["rebased"] == 2, sync.stats
    sync.close()
    dist.destroy_process_group()


@pytest.mark.timeout(240)
def test_running_island_keeps_its_progress_through_a_catch_up(tmp_path):
    procs = _run(_catch_up_worker, 4, free_port(), str(tmp_path))
    _join(procs, 200, tmp_path)


def _resumed_init_worker(rank, world, port, out_dir):
    _guard(out_dir, f"resumed-init-{rank}", lambda: _resumed_init_body(rank, world, port, out_dir))


def _resumed_init_body(rank, world, port, out_dir):
    """Learner 0 resumed from a checkpoint with a record: an initialized
    syncer answers its HELLO with the catch-up, so no INIT_PARAMS is sent;
    a syncer that sends nothing within the probe gets INIT_PARAMS, streamed
    chunk by chunk, byte-identical to a cold start's."""

    from yeto.protocol import BcastFragment
    from yeto.tensor_io import pack_tensor

    dist = _init_dist(rank, world, port)
    tp, dp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[dp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=0, tp_size=2, pp_size=1)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=dp, dp_size=2, dp_group=dp_groups[tp])
    full = {n: (_full(n, s) + 0.125).to(torch.bfloat16).float() for n, s in _global_shapes(2, 1).items()}
    clients, cut = [], {"calls": ()}

    def factory(addrs, learner_id, layout, wire, streams, **kwargs):
        script = {("bcast", c): [BcastFragment(fid, 4, pack_tensor(_flat(layout, fid, full), 2)) for fid in range(3)]
                  for c in cut["calls"]}
        clients.append(_ScriptedClient(layout, script))
        return clients[-1]

    def make(resume_dir=None, **kw):
        return sm.MegatronDiLoCoSync([chunk], opt, syncer="tcp://127.0.0.1:1234", learner_id=0, num_learners=2,
                                     fragments=3, sync_interval_steps=100, chunk_numel=40, parallel_context=ctx,
                                     client_factory=factory, resume_dir=resume_dir, resume_iteration=3, **kw)

    cut["calls"] = (1,)
    first = make(initial_cut_timeout=60)  # a cold start: INIT, then the initial cut
    ckpt = Path(out_dir, "checkpoint-3")
    first.save_state(str(ckpt), iteration=3)
    first.close()
    cold_init = dict(clients[-1].inits) if rank == 0 else {}
    if rank == 0:
        assert sorted(cold_init) == [0, 1, 2]
    resumed = make(str(ckpt), initial_cut_timeout=60, init_probe_s=30)  # the catch-up arrives: no INIT
    assert resumed.anchored == [True] * 3 and resumed.resumed == {0: "anchor", 1: "anchor", 2: "anchor"}
    if rank == 0:
        assert clients[-1].inits == {}
    resumed.close()
    cut["calls"] = ()
    started = time.monotonic()
    fresh = make(str(ckpt), initial_cut_timeout=1, init_probe_s=0.5)  # nothing arrives: INIT after the probe
    assert time.monotonic() - started >= 0.5
    if rank == 0:  # the island's current masters (the installed cut), streamed by chunk
        layout = clients[-1].layout
        assert clients[-1].inits == {fid: bytes(pack_tensor(_flat(layout, fid, full), fresh.bulk)) for fid in range(3)}
    fresh.close()
    dist.destroy_process_group()


@pytest.mark.timeout(240)
def test_a_resumed_learner_zero_sends_init_only_when_the_syncer_needs_it(tmp_path):
    procs = _run(_resumed_init_worker, 4, free_port(), str(tmp_path))
    _join(procs, 200, tmp_path)


# ----------------------------------------------------------------------------- two islands, real syncer (TP2 x PP2)
def build_syncer() -> Path:
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.cargo/bin'}:{env.get('PATH', '')}"
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True, env=env, capture_output=True)
    assert binary.exists()
    return binary


def read_final_state(path: Path) -> list[torch.Tensor]:
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 0)
    offset, out = 4, []
    for _ in range(count):
        (numel,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        out.append(torch.frombuffer(bytearray(data[offset:offset + 4 * numel]), dtype=torch.float32).clone())
        offset += 4 * numel
    assert offset == len(data)
    return out


def _island_worker(rank, world, island, dist_port, syncer_port, out_dir, drop_at, opts=None):
    _guard(out_dir, f"island{island}-{rank}",
           lambda: _island_body(rank, world, island, dist_port, syncer_port, out_dir, drop_at, opts or {}))


def _island_body(rank, world, island, dist_port, syncer_port, out_dir, drop_at, opts):
    dist = _init_dist(rank, world, dist_port)
    tp, pp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[pp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=pp, tp_size=2, pp_size=2)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=0, dp_size=1, dp_group=singles[rank])
    syncer = sm.MegatronDiLoCoSync([chunk], opt, syncer=f"127.0.0.1:{syncer_port}", learner_id=island,
                                   num_learners=2, fragments=3, sync_interval_steps=2,
                                   wire_dtype=opts.get("wire", "bf16"),
                                   chunk_numel=64, parallel_context=ctx, final_cut_payload="full",
                                   heartbeat_s=opts.get("heartbeat_s", 30.0))
    it = 0
    dropped = False
    started = time.monotonic()
    stall_at, stall_s = opts.get("stall_at"), opts.get("stall_s", 0.0)
    step_s = opts.get("step_s", 0.01)
    timeline = []  # (iteration, wall time after the boundary, boundary seconds, applied so far)
    while not syncer.finished:
        if time.monotonic() - started > 150:
            raise TimeoutError(f"island {island} not finalized after {it} steps")
        it += 1
        opt.step({id(p): _local_update(p, island, it, tp, 2) for p in trainable})
        if rank == 0 and drop_at is not None and not dropped and syncer.stats["pushed"] >= drop_at:
            for sock in list(getattr(syncer.client, "_socks", [])):  # a WAN drop: the client redials
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            dropped = True
        if it == stall_at:  # a slow / stalled island: no boundary for stall_s
            time.sleep(stall_s)
        t0 = time.monotonic()
        syncer.after_optimizer_step(it, 64)
        timeline.append((it, time.monotonic(), time.monotonic() - t0, syncer.stats["applied"]))
        time.sleep(step_s)
    for p in trainable:  # the bf16 copy equals the installed FP32 master
        if id(p) in opt.owned:
            assert torch.equal(p.detach().reshape(-1), opt.master(p).to(torch.bfloat16)), p._full_name
    names = syncer.layout.tensor_names()
    values = syncer._export_chunk(names, delta=False)
    if rank == 0:
        torch.save({"names": names, "values": dict(zip(names, values, strict=True)),
                    "events": list(syncer.events), "stats": dict(syncer.stats), "dropped": dropped,
                    "timeline": timeline, "lease_expiries": syncer.client.lease_expiries,
                    "fragments": [[n for n, _ in f.tensors] for f in syncer.layout.fragments]},
                   Path(out_dir, f"island{island}.pt"))
    syncer.close()
    dist.destroy_process_group()


@pytest.mark.timeout(300)
def test_two_tp2_pp2_islands_merge_to_the_syncer_final_state_through_a_connection_drop(tmp_path):
    binary = build_syncer()
    from process_groups import group_popen

    port = free_port()
    final_state = tmp_path / "final.bin"
    total_steps = 15
    proc = group_popen([str(binary), "--port", str(port), "--learners", "2", "--quorum", "2",
                        "--grace-ms", "300", "--total-steps", str(total_steps), "--outer-lr", "0.7",
                        "--outer-momentum", "0.9", "--final-state", str(final_state)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs = []
    try:
        procs += _run(_island_worker, 4, 0, free_port(), port, str(tmp_path), None)
        procs += _run(_island_worker, 4, 1, free_port(), port, str(tmp_path), 2)
        _join(procs, 240, tmp_path)
        assert proc.wait(timeout=60) == 0, "syncer exited nonzero"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-4000:])

    islands = [torch.load(tmp_path / f"island{i}.pt") for i in range(2)]
    assert islands[1]["dropped"], "the connection drop never happened"
    names = islands[0]["names"]
    assert names == islands[1]["names"] and "vision_model.proj.weight" not in names
    for name in names:  # both islands hold the same final global, in FP32 masters
        assert torch.equal(islands[0]["values"][name], islands[1]["values"][name]), name
    final = read_final_state(final_state)
    for fid, frag_names in enumerate(islands[0]["fragments"]):
        flat = torch.cat([islands[0]["values"][n].reshape(-1) for n in frag_names])
        assert torch.equal(flat, final[fid]), f"fragment {fid} differs from the syncer's final state"
    # The model moved: the final global is not the initial value.
    moved = sum(float((islands[0]["values"][n] - _full(n, islands[0]["values"][n].shape)).abs().sum())
                for n in names)
    assert moved > 0
    for island in islands:  # fragment rotation: every fragment pushed and merged more than once
        pushes = [e[2] for e in island["events"] if e[1] == "push"]
        applies = [e[2] for e in island["events"] if e[1] == "apply"]
        for fid in range(len(island["fragments"])):
            assert pushes.count(fid) >= 2 and applies.count(fid) >= 2, (fid, pushes, applies)
        versions = {}
        for _it, kind, fid, version in island["events"]:
            if kind == "apply":
                assert version >= versions.get(fid, -1), "fragment versions went backwards"
                versions[fid] = version


def test_write_masters_also_sets_a_distinct_hdo_host_copy():
    """core 0.18.2 makes the offloaded host copy the FP32 master itself; a
    build that kept them apart would copy the stale host copy back over the
    shard after the next step, so the write covers both."""

    from yeto.rl import megatron_masters as mm

    params = [torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))]
    opt = _DistOpt(params, dp_rank=0, dp_size=1, dp_group=None)
    shard = opt.optimizer.param_groups[0]["params"][0]
    opt.optimizer.gpu_params_map_cpu_copy = {shard: torch.zeros(4)}
    target = torch.full((4,), 2.5)
    mm.write_masters(opt, params, [target])
    assert torch.equal(opt.master(params[0]), target)
    assert torch.equal(opt.optimizer.gpu_params_map_cpu_copy[shard], target)
    assert torch.equal(params[0].detach(), target.to(torch.bfloat16))


DECOUPLED_SYNCER_FLAGS = [  # the RL async full-parameter preset (yeto/launch/rl_args.py), test-sized grace
    "--quorum", "1", "--max-base-lag", "1", "--learner-weight", "tokens2-over-steps",
    "--pipeline", "2", "--outer-lr", "1.0", "--outer-momentum", "0.0",
    # A late-carried delta answers its round, so a quorum-1 round never waits
    # for the stalled island just because the fast island's anchor was spent.
    "--carry-answers-round",
]


@pytest.mark.timeout(300)
def test_decoupled_quorum_one_keeps_the_fast_island_merging_while_the_other_stalls(tmp_path):
    """quorum 1 + max_base_lag 1 + int4 tensor-block pushes: while island 1
    makes no boundary for several seconds, island 0 keeps training without a
    blocking boundary and its fragment rounds keep merging; island 1 then
    resumes with a stale base (accepted with lag or refused by max_base_lag),
    nothing waits for it, and both islands install the same final cut."""

    binary = build_syncer()
    from process_groups import group_popen

    port = free_port()
    final_state = tmp_path / "final.bin"
    proc = group_popen([str(binary), "--port", str(port), "--learners", "2", "--grace-ms", "200",
                        "--terminal-grace-ms", "2000", "--total-steps", "40", "--final-state", str(final_state),
                        *DECOUPLED_SYNCER_FLAGS],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs = []
    stall_s = 4.0
    try:
        procs += _run(_island_worker, 4, 0, free_port(), port, str(tmp_path), None,
                      {"wire": "int4", "step_s": 0.05})
        procs += _run(_island_worker, 4, 1, free_port(), port, str(tmp_path), None,
                      {"wire": "int4", "step_s": 0.05, "stall_at": 6, "stall_s": stall_s})
        _join(procs, 240, tmp_path)
        assert proc.wait(timeout=60) == 0, "syncer exited nonzero"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        if proc.poll() is None:
            proc.kill()
        log = proc.stdout.read() if proc.stdout else ""
        (tmp_path / "syncer.log").write_text(log)
        print(log[-4000:])

    fast, slow = (torch.load(tmp_path / f"island{i}.pt") for i in range(2))
    stalled_from = next(t for it, t, _d, _a in slow["timeline"] if it == 5)
    stalled_to = next(t for it, t, _d, _a in slow["timeline"] if it == 6)
    assert stalled_to - stalled_from >= stall_s
    during = [row for row in fast["timeline"] if stalled_from < row[1] < stalled_to]
    assert len(during) >= 10, "the fast island stopped stepping while the other stalled"
    assert max(d for _it, _t, d, _a in during) < 1.0, "a boundary blocked on the stalled island"
    merged_during = during[-1][3] - during[0][3]
    resumed_pushes = [e for e in slow["events"] if e[1] == "push" and e[0] > 6]
    assert resumed_pushes, "the resumed island never pushed again"
    for name in fast["names"]:
        assert torch.equal(fast["values"][name], slow["values"][name]), name
    final = read_final_state(final_state)
    for fid, frag_names in enumerate(fast["fragments"]):
        assert torch.equal(torch.cat([fast["values"][n].reshape(-1) for n in frag_names]), final[fid])
    # A late-carried delta counts towards its round's quorum, so the fast
    # island's rounds keep closing while the other island stalls.
    assert merged_during >= 3, merged_during


@pytest.mark.timeout(300)
@pytest.mark.parametrize("stop", ["sigterm", "sigkill"])
def test_islands_keep_merging_across_a_syncer_restart(tmp_path, stop):
    """The syncer is stopped while both islands keep training and answering
    the pulls they already hold, then comes back with --resume on the same
    port.  ``sigterm``: it writes its checkpoint and exits.  During the
    outage a PUSH is not queued at all (``push_fragment_parts`` returns
    False) and is a drop; a PUSH that went out on the old connection holds an
    anchor whose round the restarted syncer no longer has, and is released
    once the island redials.  ``sigkill``: a crash; the syncer resumes from
    its last periodic checkpoint, older than what the islands hold, and its
    catch-up rewinds them (rebased onto it) instead of being dropped while
    every push from the lost versions is refused as a future base.  Both
    islands must reach the restarted syncer's final cut."""

    import signal

    binary = build_syncer()
    from process_groups import group_popen

    port = free_port()
    final_state = tmp_path / "final.bin"
    ckpt = tmp_path / "syncer.ckpt"
    flags = [str(binary), "--port", str(port), "--learners", "2", "--grace-ms", "200",
             "--terminal-grace-ms", "2000", "--total-steps", "30", "--final-state", str(final_state),
             "--checkpoint-path", str(ckpt), "--checkpoint-every", "4", "--shutdown-timeout-s", "5",
             *DECOUPLED_SYNCER_FLAGS]
    logs: list[str] = []

    def start(extra):
        proc = group_popen(flags + extra, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs.append("")
        return proc

    def read(i, proc):  # the log file written so far
        logs[i] += proc.stdout.read()
        return logs[i]

    first = start([])
    second = None
    procs = []
    try:
        procs += _run(_island_worker, 4, 0, free_port(), port, str(tmp_path), None, {"step_s": 0.05})
        procs += _run(_island_worker, 4, 1, free_port(), port, str(tmp_path), None, {"step_s": 0.05})
        deadline = time.monotonic() + 120
        # sigkill: past a periodic checkpoint (every 4 rounds), not on one.
        while read(0, first).count("outer step") < (8 if stop == "sigterm" else 10):
            assert time.monotonic() < deadline, "the first syncer never merged enough rounds"
            assert first.poll() is None, "the first syncer exited early"
            time.sleep(0.05)
        if stop == "sigterm":
            first.send_signal(signal.SIGTERM)
            assert first.wait(timeout=60) == 0, "SIGTERM did not end in a clean checkpoint"
        else:
            first.kill()
            first.wait(timeout=60)
        assert ckpt.exists()
        time.sleep(3.0)  # the outage: both islands keep stepping
        second = start(["--resume"])
        _join(procs, 240, tmp_path)
        assert second.wait(timeout=60) == 0, "the restarted syncer exited nonzero"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        for i, proc in enumerate((first, second)):
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                read(i, proc)
                (tmp_path / f"syncer{i}.log").write_text(logs[i])
        print(logs[-1][-4000:])

    islands = [torch.load(tmp_path / f"island{i}.pt") for i in range(2)]
    for name in islands[0]["names"]:
        assert torch.equal(islands[0]["values"][name], islands[1]["values"][name]), name
    final = read_final_state(final_state)
    for fid, frag_names in enumerate(islands[0]["fragments"]):
        assert torch.equal(torch.cat([islands[0]["values"][n].reshape(-1) for n in frag_names]), final[fid])
    assert "outer step" in logs[1], "the restarted syncer merged nothing"
    if stop == "sigkill":
        assert sum(i["stats"].get("rebased", 0) for i in islands) > 0, "no island was rewound"
    print(json.dumps([i["stats"] for i in islands]))


@pytest.mark.timeout(300)
@pytest.mark.parametrize("heartbeat_s", [0.3, 0.0])
def test_heartbeats_keep_a_stalled_island_inside_its_lease(tmp_path, heartbeat_s):
    """--island-lease-s 2 with an island that makes no boundary for 4 s:
    rank 0's heartbeat thread keeps its lease; without heartbeats the syncer
    expires it (the island redials as a re-JOIN)."""

    binary = build_syncer()
    from process_groups import group_popen

    port = free_port()
    final_state = tmp_path / "final.bin"
    proc = group_popen([str(binary), "--port", str(port), "--learners", "2", "--grace-ms", "200",
                        "--terminal-grace-ms", "2000", "--total-steps", "30", "--final-state", str(final_state),
                        "--island-lease-s", "2", *DECOUPLED_SYNCER_FLAGS],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs = []
    try:
        procs += _run(_island_worker, 4, 0, free_port(), port, str(tmp_path), None,
                      {"step_s": 0.05, "heartbeat_s": heartbeat_s})
        procs += _run(_island_worker, 4, 1, free_port(), port, str(tmp_path), None,
                      {"step_s": 0.05, "stall_at": 6, "stall_s": 4.0, "heartbeat_s": heartbeat_s})
        _join(procs, 240, tmp_path)
        assert proc.wait(timeout=60) == 0, "syncer exited nonzero"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        if proc.poll() is None:
            proc.kill()
        log = proc.stdout.read() if proc.stdout else ""
        (tmp_path / "syncer.log").write_text(log)
    islands = [torch.load(tmp_path / f"island{i}.pt") for i in range(2)]
    for name in islands[0]["names"]:
        assert torch.equal(islands[0]["values"][name], islands[1]["values"][name]), name
    expired = '"reason":"lease_expired"' in log or islands[1]["lease_expiries"] > 0
    assert expired == (heartbeat_s == 0), (islands[1]["lease_expiries"], heartbeat_s)
