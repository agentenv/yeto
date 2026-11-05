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
    assert sm._syncer_addrs("tcp://10.0.0.1:29400") == [("10.0.0.1", 29400)]
    assert sm._syncer_addrs("a:1, tcp://b:2") == [("a", 1), ("b", 2)]
    key = tmp_path / "hmac.key"
    key.write_text("s3cret\n")
    assert sm._read_hmac_key(str(key)) == b"s3cret"  # the YETO_ISLAND_HMAC_KEY encoding
    assert sm._read_hmac_key(None) is None


def test_chunks_are_bounded_and_keep_order():
    sizes = {"a": 3, "b": 3, "c": 10, "d": 1, "e": 1}
    assert sm.chunk_names(list(sizes), sizes.__getitem__, 6) == [["a", "b"], ["c"], ["d", "e"]]


# ----------------------------------------------------------------------------- fake Megatron island
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
        assert addrs == [("127.0.0.1", 1234)] and len(kwargs["session_contract_hash"]) == 32
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
    for it in (1, 2, 3):
        opt.step({id(p): _local_update(p, 0, it, tp, 2) for p in trainable})
        for name, shape in full_shapes.items():
            if it >= 2:  # updates after the merge at iteration 1
                updates_total[name] += _update(name, shape, 0, it)
        if it == 1:
            pre_merge = {name: _full(name, shape) + _update(name, shape, 0, 1)
                         for name, shape in full_shapes.items()}
        syncer.after_optimizer_step(it, 100 * it)
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
    assert [e[2] for e in pushes] == list(range(syncer.num_fragments)) and all(e[0] == 3 for e in pushes)
    assert torch.equal(chunk.vision_model.proj.weight, frozen_before)
    for key, state in opt.optimizer.state.items():  # Adam moments kept
        for n, t in state.items():
            assert torch.equal(t, adam_before[key][n])
    if rank == 0:
        client = holder["client"]
        assert len(client.pushes) == syncer.num_fragments
        for fid, step, attempt, base, local_step, c_steps, c_units, payload in client.pushes:
            assert (step, attempt, base, local_step, c_steps, c_units) == (8, 1, 7, 3, 2, 500)
            # PUSH = local - raw anchor (the global, not the blend)
            local = {n: alpha * pre_merge[n] + (1 - alpha) * globals_g[n] + updates_total[n] for n in full_shapes}
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


def _island_worker(rank, world, island, dist_port, syncer_port, out_dir, drop_at):
    _guard(out_dir, f"island{island}-{rank}",
           lambda: _island_body(rank, world, island, dist_port, syncer_port, out_dir, drop_at))


def _island_body(rank, world, island, dist_port, syncer_port, out_dir, drop_at):
    dist = _init_dist(rank, world, dist_port)
    tp, pp = rank % 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    singles = [dist.new_group([r]) for r in range(world)]
    ctx = sm.ParallelContext(rank, world, tp_groups[pp], tp, 2, singles[rank], 0, 1)
    chunk = _chunk(tp=tp, pp=pp, tp_size=2, pp_size=2)
    trainable = [p for p in chunk.parameters() if p.requires_grad]
    opt = _DistOpt(trainable, dp_rank=0, dp_size=1, dp_group=singles[rank])
    syncer = sm.MegatronDiLoCoSync([chunk], opt, syncer=f"127.0.0.1:{syncer_port}", learner_id=island,
                                   num_learners=2, fragments=3, sync_interval_steps=2, wire_dtype="bf16",
                                   chunk_numel=64, parallel_context=ctx, final_cut_payload="full")
    it = 0
    dropped = False
    started = time.monotonic()
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
        syncer.after_optimizer_step(it, 64)
        time.sleep(0.01)
    for p in trainable:  # the bf16 copy equals the installed FP32 master
        if id(p) in opt.owned:
            assert torch.equal(p.detach().reshape(-1), opt.master(p).to(torch.bfloat16)), p._full_name
    names = syncer.layout.tensor_names()
    values = syncer._export_chunk(names, delta=False)
    if rank == 0:
        torch.save({"names": names, "values": dict(zip(names, values, strict=True)),
                    "events": list(syncer.events), "stats": dict(syncer.stats), "dropped": dropped,
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
    print(json.dumps([i["stats"] for i in islands]))


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
