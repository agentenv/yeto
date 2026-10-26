"""Same-shape cut under DistributedOptimizer with PP2 (DP1/DP2) and EP2 (dense DP2, expert DP1).

CPU protocol test, NOT GPU evidence. Every rank is simulated in one process
with duck-typed Megatron DistOpt leaves (``gbuf_ranges``, main-param shards,
Adam moments) and the REAL fork-M5 helpers (``dp_invariant_state`` at the
Miles image pin and at the stage-1 branch s25-distopt, read from a local Miles
checkout; skipped when absent).
save -> fresh rebuild -> restore -> re-export must be element-wise equal, and
the full FP32 masters gathered for the republish (state_plugin.full_masters)
must equal the pre-cut ones. A different DP/EP layout is refused.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles import cut_plugin as cp
from yeto.rl.adapters.miles import state_plugin as sp

DPS_PATH = "miles/backends/megatron_utils/lora/dp_invariant_state.py"
# The fork-M5 helpers at the Miles image pin, and at the stage-1 fork branch that
# carries the CPU-offload (HybridDeviceOptimizer) state (S25 N7.1d, not pushed).
DPS_SOURCES = {
    "m5-c35702e": (os.environ.get("YETO_MILES_CHECKOUT", "/home/michael/work/miles-m3"), "c35702e"),
    "s25-distopt": (os.environ.get("YETO_MILES_DISTOPT_CHECKOUT", "/home/michael/work/miles-next"), "s25-distopt"),
    # E1: + export_named_optimizer_state(device=None) on the image pin 44d41efd (not pushed)
    "e1-device-export": (os.environ.get("YETO_MILES_DISTOPT_CHECKOUT", "/home/michael/work/miles-next"),
                         "yeto/s26-e1-device-export"),
    # S26 E6 fork branch: side-effect-free reads plus the ported CPU-offload state
    "s26-e6-fp8": (os.environ.get("YETO_MILES_E6_CHECKOUT", "/home/michael/work/miles-s26-e6-fp8"), "ed5f64fd7"),
}


def _load_dps(checkout, ref):
    try:
        src = subprocess.run(["git", "-C", checkout, "show", f"{ref}:{DPS_PATH}"],
                             capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    spec = importlib.util.spec_from_loader(f"_yeto_test_dps_{ref.replace('-', '_')}", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(src, DPS_PATH, "exec"), module.__dict__)  # noqa: S102 - pinned local source
    return module


_LOADED = {label: _load_dps(*source) for label, source in DPS_SOURCES.items()}
DPS = None  # set per test by the ``dps`` fixture


@pytest.fixture(params=sorted(DPS_SOURCES))
def dps(request, monkeypatch):
    module = _LOADED[request.param]
    if module is None:
        pytest.skip(f"Miles checkout with the fork-M5 helpers at {DPS_SOURCES[request.param]} not available")
    monkeypatch.setattr(sys.modules[__name__], "DPS", module)
    return module


needs_dps = pytest.mark.usefixtures("dps")


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Leaf:
    """One DistOpt instance: owns ``[lo, hi)`` of the flattened params of its DP group."""

    def __init__(self, params, *, dp_rank, dp_size, group, pad=0):
        sizes = [p.numel() for p in params]
        total = sum(sizes) + pad
        shard = -(-total // dp_size)
        lo, hi = dp_rank * shard, min((dp_rank + 1) * shard, total)
        param_map, mains, off = {}, [], 0
        for p, n in zip(params, sizes, strict=True):
            s, e = max(lo, off) - off, min(hi, off + n) - off
            if e > s:
                param_map[p] = {"param": _Range(s, e)}
                mains.append(p.detach().float().reshape(-1)[s:e].clone())
            off += n
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(param_map)}
        self.buffers = [SimpleNamespace(param_index_map={p: None for p in params})]
        self.data_parallel_group = group
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.optimizer = SimpleNamespace(param_groups=[{"params": mains, "lr": 0.01, "betas": (0.9, 0.95)}],
                                         state={})
        self._ranges = {id(p): (r["param"].start, r["param"].end) for p, r in param_map.items()}

    def _main(self, p):
        g, i = self.model_param_group_index_map[p]
        return self.optimizer.param_groups[g]["params"][i]

    def _get_main_param_and_optimizer_states(self, p):
        main = self._main(p)
        st = self.optimizer.state.setdefault(main, {})  # Megatron indexes a defaultdict
        return {"param": main, **{k: st[k] for k in ("exp_avg", "exp_avg_sq") if k in st}}

    def _set_main_param_and_optimizer_states(self, p, tensors):
        main = self._main(p)
        st = self.optimizer.state[main]
        for key, value in tensors.items():
            if key == "param":
                main.copy_(value)
            elif key in st:
                st[key].copy_(value)

    def _init_optimizer_states_with_dummy_values(self):
        for main in self.optimizer.param_groups[0]["params"]:
            self.optimizer.state[main] = {"exp_avg": torch.zeros_like(main), "exp_avg_sq": torch.zeros_like(main)}

    def train(self, seed):
        g = torch.Generator().manual_seed(seed)
        for main in self.optimizer.param_groups[0]["params"]:
            main.add_(torch.randn(main.shape, generator=g))
            self.optimizer.state[main] = {"exp_avg": torch.randn(main.shape, generator=g),
                                          "exp_avg_sq": torch.rand(main.shape, generator=g)}


class _Backend(cp.MilesCutBackend):
    def __init__(self, coord):
        self._coord = coord

    def coord(self):
        return dict(self._coord)

    def is_adapter(self, name):
        return "lora" in name

    def _dps(self):
        return DPS

    def megatron_counters(self):
        return {}


class _Sched:
    def __init__(self):
        self.num_steps = 0

    def state_dict(self):
        return {"num_steps": self.num_steps, "max_lr": 0.01}

    def load_state_dict(self, state):
        self.num_steps += state["num_steps"]


def _module(named):
    m = torch.nn.Module()
    for name, t in named:
        m.register_parameter(name, torch.nn.Parameter(t.clone().to(torch.bfloat16)))
    return m


def _tensors(seed, shapes):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g) for s in shapes]


def _world(layout):
    """Freshly built ranks. layout: 'pp2dp1' | 'pp2dp2' | 'ep2' (dense DP2, expert DP1)."""
    ranks = []
    if layout.startswith("pp2"):
        dp_size = 1 if layout == "pp2dp1" else 2
        for pp in range(2):
            init = _tensors(100 + pp, ((3, 5), (7,), (2, 4)))
            for dp in range(dp_size):
                model = _module([(f"lora_{i}", t) for i, t in enumerate(init)])
                params = list(model.parameters())
                leaf = _Leaf(params, dp_rank=dp, dp_size=dp_size, group=("dp", pp))
                coord = {"global_rank": pp * dp_size + dp, "tp": 0, "pp": pp, "dp": dp, "dp_size": dp_size,
                         "tp_size": 1, "pp_size": 2, "cp_size": 1, "ep_size": 1}
                ranks.append(_rank(model, leaf, coord, pp=2))
    else:
        dense = _tensors(7, ((3, 5), (7,), (2, 4)))
        for ep in range(2):
            expert = _tensors(20 + ep, ((4, 3), (5,)))
            model = _module([(f"lora_d{i}", t) for i, t in enumerate(dense)]
                            + [(f"experts_lora_{i}", t) for i, t in enumerate(expert)])
            params = list(model.parameters())
            d = _Leaf(params[:3], dp_rank=ep, dp_size=2, group=("dense-dp", (0, 1)))
            e = _Leaf(params[3:], dp_rank=0, dp_size=1, group=("expert-dp", (ep,)))
            coord = {"global_rank": ep, "tp": 0, "pp": 0, "dp": ep, "dp_size": 2, "tp_size": 1, "pp_size": 1,
                     "cp_size": 1, "ep_size": 2, "ep": ep, "etp_size": 1, "edp": 0, "edp_size": 1}
            ranks.append(_rank(model, SimpleNamespace(chained_optimizers=[d, e]), coord, ep=2))
    return ranks


def _rank(model, optimizer, coord, *, pp=1, ep=1):
    args = SimpleNamespace(fp16=False, use_distributed_optimizer=True, pipeline_model_parallel_size=pp,
                           expert_model_parallel_size=ep, tensor_model_parallel_size=1)
    return SimpleNamespace(args=args, model=[model], optimizer=optimizer, opt_param_scheduler=_Sched(),
                           _yeto_cut_backend=_Backend(coord))


def _leaves(rank):
    return list(getattr(rank.optimizer, "chained_optimizers", None) or [rank.optimizer])


def _train(ranks):
    for r, rank in enumerate(ranks):
        for k, leaf in enumerate(_leaves(rank)):
            leaf.train(1000 + 10 * r + k)
        rank.opt_param_scheduler.num_steps = 8


def _gathered_masters(ranks):
    """Full FP32 masters per rank via state_plugin.full_masters, group sums simulated in one process."""
    record = {}
    for rank in ranks:
        calls = [0]

        def rec(flat, leaf, calls=calls):
            calls[0] += 1
            record.setdefault((leaf.data_parallel_group, calls[0]), []).append(flat.clone())

        sp.full_masters(rank.optimizer, list(rank.model[0].parameters()), reduce=rec)
    out = []
    for rank in ranks:
        n = [0]

        def red(flat, leaf, n=n):
            n[0] += 1
            flat.copy_(sum(record[(leaf.data_parallel_group, n[0])]))

        out.append(sp.full_masters(rank.optimizer, list(rank.model[0].parameters()), reduce=red))
    return out


def _export(rank):
    named = [(n, p) for n, p in rank.model[0].named_parameters()]
    return rank._yeto_cut_backend.export_optimizer(rank.optimizer, named)


def _save(ranks, tmp_path):
    files = []
    for rank in ranks:
        s = cp.save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
        assert "refused" not in s, s
        files.append(s)
    return files


@needs_dps
@pytest.mark.parametrize("layout", ["pp2dp1", "pp2dp2", "ep2"])
def test_save_restore_reexport_is_elementwise_equal(tmp_path, layout):
    ranks = _world(layout)
    _train(ranks)
    before = [_export(r) for r in ranks]
    masters = _gathered_masters(ranks)
    files = _save(ranks, tmp_path)
    if layout == "ep2":  # each EP rank keeps its own expert params (same names, different values)
        assert not torch.equal(masters[0][3], masters[1][3])
    fresh = _world(layout)
    for rank in fresh:
        out = cp.restore_cut_shard(rank, directory=str(tmp_path), files=files, cut_id="c1")
        assert "refused" not in out, out
        assert "diff" not in out, out
    for rank, want in zip(fresh, before, strict=True):
        got = _export(rank)
        assert cp.optimizer_ranges(got) == cp.optimizer_ranges(want)
        assert cp.state_digest(got) == cp.state_digest(want)
        for name, e in want["entries"].items():
            for key, t in e["tensors"].items():
                assert torch.equal(got["entries"][name]["tensors"][key], t), (name, key)
    for got, want in zip(_gathered_masters(fresh), masters, strict=True):
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b)


@needs_dps
@pytest.mark.parametrize("layout", ["pp2dp2", "ep2"])
def test_resume_manifest_samples_each_rank_own_ranges_and_survives_the_cut(tmp_path, layout):
    # N7.6 under the sharded optimizer (N7.1d): every rank records the moments of the
    # parameters it owns a range of, and the restored rank reproduces them exactly.
    from yeto.rl.engine.resume_manifest import OptimizerResumeState, restore_problems, ResumeManifest

    ranks = _world(layout)
    _train(ranks)
    files = _save(ranks, tmp_path)
    saved = {f["path"]: OptimizerResumeState.from_dict(f["resume"]) for f in files}
    for rank, f in zip(ranks, files, strict=True):
        owned = set(_export(rank)["entries"])
        assert saved[f["path"]].moments and set(saved[f["path"]].moments) <= owned
    fresh = _world(layout)
    restored = {}
    for rank in fresh:
        out = cp.restore_cut_shard(rank, directory=str(tmp_path), files=files, cut_id="c1")
        assert "refused" not in out, out
        restored[out["path"]] = OptimizerResumeState.from_dict(out["resume"])

    def manifest(states):
        return ResumeManifest(global_step=1, data_watermark=8, scheduler_increment=8, ranks=states)

    assert restore_problems(manifest(saved), manifest(restored)) == []
    # a rank that records another rank's moments (here: none of its own) is caught
    first, last = sorted(saved)[0], sorted(saved)[-1]
    swapped = {**restored, first: dataclasses.replace(restored[first], moments=restored[last].moments)}
    assert saved[first].moments != saved[last].moments
    assert any(first in p for p in restore_problems(manifest(saved), manifest(swapped)))


@needs_dps
def test_ep_layout_change_is_refused(tmp_path):
    ranks = _world("ep2")
    _train(ranks)
    files = _save(ranks, tmp_path)
    fresh = _world("ep2")
    fresh[0]._yeto_cut_backend._coord["ep_size"] = 4
    out = cp.restore_cut_shard(fresh[0], directory=str(tmp_path), files=files, cut_id="c1")
    assert "same-shape" in out["refused"] and out["refusal_kind"] == "refused"


@needs_dps
def test_dp_layout_change_is_refused(tmp_path):
    ranks = _world("pp2dp2")
    _train(ranks)
    files = _save(ranks, tmp_path)
    fresh = _world("pp2dp1")
    out = cp.restore_cut_shard(fresh[0], directory=str(tmp_path), files=files, cut_id="c1")
    assert "refused" in out


@needs_dps
def test_changed_shard_ranges_are_refused_before_any_write(tmp_path):
    ranks = _world("pp2dp2")
    _train(ranks)
    files = _save(ranks, tmp_path)
    fresh = _world("pp2dp2")
    rank = fresh[0]
    model = rank.model[0]
    rank.optimizer = _Leaf(list(model.parameters()), dp_rank=0, dp_size=2, group=("dp", 0), pad=6)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = cp.restore_cut_shard(rank, directory=str(tmp_path), files=files, cut_id="c1")
    assert "ranges differ" in out["refused"]
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p.detach())


def test_local_states_cover_with_fillers():
    t = torch.arange(3.0)
    state = {"format": "f", "entries": {"a": {"numel": 10, "shape": (10,), "start": 4, "end": 7,
                                              "tensors": {"param": t}, "scalars": {}, "hyper": {}},
                                        "b": {"numel": 3, "shape": (3,), "start": 0, "end": 3,
                                              "tensors": {"param": t}, "scalars": {}, "hyper": {}}}}
    states = cp.local_states(state)
    spans = sorted((n, e["start"], e["end"]) for s in states for n, e in s["entries"].items())
    assert spans == [("a", 0, 4), ("a", 4, 7), ("a", 7, 10), ("b", 0, 3)]
    assert all(len(s["entries"]) and s["format"] == "f" for s in states)
    assert cp.range_problems(state, state) == []
    other = {"entries": {"a": {**state["entries"]["a"], "start": 5}}}
    assert len(cp.range_problems(state, other)) == 2


# --------------------------------------------------------------------------
# E1: background cuts copy their state device-to-device first
# --------------------------------------------------------------------------


@needs_dps
@pytest.mark.parametrize("layout", ["pp2dp1", "pp2dp2", "ep2"])
def test_device_export_is_the_fork_host_export_once_on_the_host(layout):
    ranks = _world(layout)
    _train(ranks)
    for rank in ranks:
        named = list(rank.model[0].named_parameters())
        host = DPS.export_named_optimizer_state(rank.optimizer, named)
        device = cp.to_host(cp._device_export(DPS)(rank.optimizer, named))
        assert cp.state_digest(device) == cp.state_digest(host)
        assert device["format"] == host["format"] and list(device["entries"]) == list(host["entries"])
        for name, e in host["entries"].items():
            got = device["entries"][name]
            assert {k: v for k, v in got.items() if k not in ("tensors", "scalars")} == \
                {k: v for k, v in e.items() if k not in ("tensors", "scalars")}
            for part in ("tensors", "scalars"):
                assert list(got[part]) == list(e[part])
                for key, t in e[part].items():
                    assert got[part][key].dtype == t.dtype and torch.equal(got[part][key], t), (name, key)


def test_device_export_uses_the_fork_device_argument_when_it_has_one():
    calls = []

    def export(optimizer, named, *, device="cpu"):
        calls.append(device)
        return {"entries": {}}

    cp._device_export(SimpleNamespace(export_named_optimizer_state=export))("opt", [])
    assert calls == [None]


@needs_dps
@pytest.mark.parametrize("staged", [False, True], ids=["resident", "sleeps_after_cut"])
@pytest.mark.parametrize("layout", ["pp2dp2", "ep2"])
def test_background_device_capture_writes_the_inline_shard(tmp_path, layout, staged, monkeypatch):
    """Same shard (file bytes, digests, N7.6 resume state) from the device path as from the
    inline host path, although the trainer trains on before the writer copies to the host."""
    ranks = _world(layout)
    _train(ranks)
    inline = [cp.save_cut_shard(r, directory=str(tmp_path / "inline"), cut_id="c1") for r in ranks]
    monkeypatch.setattr(cp, "_device_copy_plan", lambda actor, named: (True, "ok"))
    if staged:  # a sleeping trainer's copies go through the staging buffer (CPU stand-ins too)
        real_stage = cp._stage
        monkeypatch.setattr(cp, "_sleeps_after_cut", lambda actor: True)
        monkeypatch.setattr(cp, "_stage", lambda snap: real_stage(snap, include_cpu=True))
    device = []
    for r, rank in enumerate(ranks):  # one pending write per cut id per process: rank by rank
        started = cp.save_cut_shard(rank, directory=str(tmp_path / "device"), cut_id="c1", background=True)
        assert started["capture"] == {"mode": "device", "reason": "ok"}, started
        for k, leaf in enumerate(_leaves(rank)):  # the next step changes the live state first
            leaf.train(2000 + 10 * r + k)
        rank.opt_param_scheduler.num_steps = 16
        device.append(cp.finish_cut_shard(rank, cut_id="c1"))
    assert [d.pop("writer") for d in device] == [{"mode": "process"}] * len(ranks)
    assert cp._STAGING.free.is_set() and (cp._STAGING.buffer is not None or not staged)
    assert device == inline
    for name in sorted(p.name for p in (tmp_path / "inline").iterdir()):
        assert (tmp_path / "device" / name).read_bytes() == (tmp_path / "inline" / name).read_bytes()


def test_background_capture_falls_back_to_the_host_and_says_why(tmp_path, monkeypatch):
    from tests.rl_cut_fakes import make_rank, train_step

    rank = make_rank(0)
    train_step(rank, torch.randn(3, 6, generator=torch.Generator().manual_seed(7)))
    inline = cp.save_cut_shard(rank, directory=str(tmp_path / "inline"), cut_id="c1")
    monkeypatch.setattr(cp, "_device_copy_plan", lambda actor, named: (False, "low_device_memory(free=1,need=2)"))
    started = cp.save_cut_shard(rank, directory=str(tmp_path / "low"), cut_id="c1", background=True)
    assert started["capture"] == {"mode": "host", "reason": "low_device_memory(free=1,need=2)"}
    assert _without_writer(cp.finish_cut_shard(rank, cut_id="c1")) == inline

    real = cp._snapshot

    def oom(actor, backend, named, *, on_device=False):
        if on_device:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(actor, backend, named)

    monkeypatch.setattr(cp, "_device_copy_plan", lambda actor, named: (True, "ok"))
    monkeypatch.setattr(cp, "_snapshot", oom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    started = cp.save_cut_shard(rank, directory=str(tmp_path / "oom"), cut_id="c1", background=True)
    assert started["capture"] == {"mode": "host", "reason": "device_oom"}
    assert _without_writer(cp.finish_cut_shard(rank, cut_id="c1")) == inline


def _without_writer(summary):
    assert summary.pop("writer")["mode"] == "process", summary
    return summary


def _one_rank_cut(tmp_path, name, *, background):
    import random

    import numpy as np

    from tests.rl_cut_fakes import make_rank, train_step

    random.seed(5), np.random.seed(5), torch.manual_seed(5)  # the shard holds the global RNG state
    rank = make_rank(0)
    train_step(rank, torch.randn(3, 6, generator=torch.Generator().manual_seed(7)))
    if not background:
        return cp.save_cut_shard(rank, directory=str(tmp_path / name), cut_id="c1")
    cp.save_cut_shard(rank, directory=str(tmp_path / name), cut_id="c1", background=True)
    return cp.finish_cut_shard(rank, cut_id="c1")


def test_a_dead_writer_process_falls_back_to_the_thread_and_says_why(tmp_path, monkeypatch):
    """The writer process dies before it writes (its start fails): this rank's thread
    writes the same shard and the summary records the fallback."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    inline = _one_rank_cut(tmp_path, "inline", background=False)
    dead = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                               initializer=os._exit, initargs=(3,))
    monkeypatch.setattr(cp, "_WRITER", [dead])
    got = _one_rank_cut(tmp_path, "dead", background=True)
    writer = got.pop("writer")
    assert writer["mode"] == "thread" and "writer process failed" in writer["fallback"], writer
    assert got == inline
    assert cp._WRITER == []  # the broken pool is dropped; the next cut starts a new one
    assert (tmp_path / "dead" / got["path"]).read_bytes() == (tmp_path / "inline" / got["path"]).read_bytes()


def test_a_partial_file_of_a_dead_writer_is_removed_before_the_fallback(tmp_path, monkeypatch):
    inline = _one_rank_cut(tmp_path, "inline", background=False)
    target = tmp_path / "partial"
    target.mkdir()
    (target / (inline["path"] + ".tmp")).write_bytes(b"half")
    (target / inline["path"]).write_bytes(b"whole but unreported")

    class Broken:
        def submit(self, *a, **k):
            raise RuntimeError("writer died")

        def shutdown(self, **k):
            pass

    monkeypatch.setattr(cp, "_WRITER", [Broken()])
    got = _one_rank_cut(tmp_path, "partial", background=True)
    assert got.pop("writer") == {"mode": "thread", "fallback": "writer process failed: RuntimeError('writer died')"}
    assert got == inline and not (target / (inline["path"] + ".tmp")).exists()


def test_the_writer_process_can_be_turned_off(tmp_path, monkeypatch):
    inline = _one_rank_cut(tmp_path, "inline", background=False)
    monkeypatch.setenv(cp.WRITER_PROCESS_ENV, "0")
    got = _one_rank_cut(tmp_path, "thread", background=True)
    assert got.pop("writer") == {"mode": "thread", "fallback": "disabled"}
    assert got == inline


def test_device_copy_plan_needs_cuda_and_can_be_turned_off(monkeypatch):
    named = [("lora_a", torch.nn.Parameter(torch.zeros(2)))]
    assert cp._device_copy_plan(None, named) == (False, "not_cuda")
    sleeper = SimpleNamespace(args=SimpleNamespace(offload_train=True))  # staged, below
    assert cp._device_copy_plan(sleeper, named) == (False, "not_cuda")
    monkeypatch.setenv(cp.DEVICE_COPY_ENV, "0")
    assert cp._device_copy_plan(None, named) == (False, "disabled")


def test_pack_hands_the_state_over_in_one_buffer_and_unpacks_it_exactly():
    shared = torch.arange(6.0)
    state = {"a": torch.randn(3, 4), "b": [torch.ones(2, dtype=torch.bfloat16), torch.tensor(7)],
             "view": torch.arange(10.0)[2:5], "twice": (shared, shared), "empty": torch.zeros(0), "n": 3}
    skeleton, layout, flat = cp._pack(state)
    assert len(layout) == 3  # a, b[0], b[1]; the view, the shared storage and the empty one are not packed
    assert flat.is_shared() and skeleton["n"] == 3
    back = cp._unpack(skeleton, layout, flat)
    assert cp.state_digest(back) == cp.state_digest(state)
    for key in ("a", "view", "empty"):
        assert back[key].dtype == state[key].dtype and torch.equal(back[key], state[key])
    assert back["b"][0].untyped_storage().data_ptr() != flat.untyped_storage().data_ptr()  # own copies


def test_host_background_writes_the_inline_shard_from_one_host_copy(tmp_path):
    """S27 G-EL peer snapshots (``background="host"``): the state is copied to the
    host at the step boundary and a thread of the rank writes it; the shard, its
    summary and its bytes equal the inline save's. ``poll_cut_shard`` never waits."""
    import time

    from tests.rl_cut_fakes import make_rank, train_step

    rank = make_rank(0)
    train_step(rank, torch.randn(3, 6, generator=torch.Generator().manual_seed(7)))
    assert cp.poll_cut_shard(rank, cut_id="c1") == {"done": True, "missing": True}
    inline = cp.save_cut_shard(rank, directory=str(tmp_path / "inline"), cut_id="c1")
    started = cp.save_cut_shard(rank, directory=str(tmp_path / "host"), cut_id="c1", background=cp.HOST_BACKGROUND)
    assert started["pending"] == "c1"
    assert started["capture"] == {"mode": "host", "reason": "host_background"}
    deadline = time.monotonic() + 30
    while not cp.poll_cut_shard(rank, cut_id="c1")["done"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    summary = cp.finish_cut_shard(rank, cut_id="c1")
    assert summary.pop("writer") == {"mode": "host_thread"}
    assert summary == inline
    for name in sorted(p.name for p in (tmp_path / "inline").iterdir()):
        assert (tmp_path / "host" / name).read_bytes() == (tmp_path / "inline" / name).read_bytes()
