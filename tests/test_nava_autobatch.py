"""NAVA auto micro-batch probe: doubling, OOM stop, explicit/CPU short-circuits."""
from types import SimpleNamespace

import torch

from yeto.nava import autobatch_nava


def _args(bs):
    return SimpleNamespace(nava_batch_size=bs)


def _opt():
    return SimpleNamespace(zero_grad=lambda set_to_none=True: None)


def _no_cuda_side_effects(monkeypatch):
    # The probe caps VRAM while probing and clears cache between sizes; on a
    # CUDA-less box these are no-ops we stub so tests exercise pure probe logic.
    monkeypatch.setattr(autobatch_nava, "_set_mem_fraction", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_explicit_size_skips_probe(monkeypatch):
    # An explicit int never probes; resolve echoes cfg["batch_size"]
    # (which _apply_config_overrides already set from the flag).
    def boom(*a):
        raise AssertionError("must not probe")

    monkeypatch.setattr(autobatch_nava, "_probe_once", boom)
    got = autobatch_nava.resolve_nava_micro_batch(
        _args(2), None, None, None, _opt(), {"batch_size": 3}, SimpleNamespace(type="cpu"), 1, 0
    )
    assert got == 3


def test_cpu_returns_one_without_probe(monkeypatch):
    monkeypatch.setattr(autobatch_nava, "_probe_once",
                        lambda *a: (_ for _ in ()).throw(AssertionError))
    got = autobatch_nava.resolve_nava_micro_batch(
        _args("auto"), None, None, None, _opt(), {"batch_size": 1}, SimpleNamespace(type="cpu"), 1, 0
    )
    assert got == 1


def test_probe_doubles_until_oom_and_keeps_last_passing(monkeypatch):
    tried = []

    def probe(build_loader, pipe, params, opt, cfg, size, gs, world):
        tried.append(size)
        if size > 4:
            raise torch.cuda.OutOfMemoryError("boom")

    monkeypatch.setattr(autobatch_nava, "_probe_once", probe)
    _no_cuda_side_effects(monkeypatch)
    got = autobatch_nava.resolve_nava_micro_batch(
        _args("auto"), None, None, None, _opt(), {"batch_size": 1}, SimpleNamespace(type="cuda"), 1, 0
    )
    assert got == 4
    assert tried == [1, 2, 4, 8]


def test_oom_at_one_falls_back_to_one(monkeypatch):
    def probe(*a):
        raise torch.cuda.OutOfMemoryError("boom")

    monkeypatch.setattr(autobatch_nava, "_probe_once", probe)
    _no_cuda_side_effects(monkeypatch)
    got = autobatch_nava.resolve_nava_micro_batch(
        _args("auto"), None, None, None, _opt(), {"batch_size": 1}, SimpleNamespace(type="cuda"), 1, 0
    )
    assert got == 1  # max(1, best); a size-1 OOM still trains (surfaces at step 0)


def test_non_oom_error_stops_probe(monkeypatch):
    def probe(build_loader, pipe, params, opt, cfg, size, gs, world):
        if size >= 4:
            raise RuntimeError("dataset exploded")

    monkeypatch.setattr(autobatch_nava, "_probe_once", probe)
    _no_cuda_side_effects(monkeypatch)
    got = autobatch_nava.resolve_nava_micro_batch(
        _args("auto"), None, None, None, _opt(), {"batch_size": 1}, SimpleNamespace(type="cuda"), 1, 0
    )
    assert got == 2  # keeps the last size that passed before the non-OOM failure


def test_ceiling_caps_the_result(monkeypatch):
    # Everything fits, but the ceiling stops runaway doubling.
    tried = []
    monkeypatch.setenv("YETO_NAVA_MAX_MICRO_BATCH", "4")
    monkeypatch.setattr(autobatch_nava, "_probe_once",
                        lambda *a: tried.append(a[5]))  # a[5] == size
    _no_cuda_side_effects(monkeypatch)
    got = autobatch_nava.resolve_nava_micro_batch(
        _args("auto"), None, None, None, _opt(), {"batch_size": 1}, SimpleNamespace(type="cuda"), 1, 0
    )
    assert got == 4          # never exceeds the ceiling
    assert tried == [1, 2, 4]  # stops probing at the ceiling


def test_probe_reserves_headroom_then_restores(monkeypatch):
    # The probe caps VRAM (<1.0) while probing, then restores the full card (1.0).
    calls = []
    monkeypatch.setenv("YETO_NAVA_PROBE_MEM_FRACTION", "0.8")
    monkeypatch.setattr(autobatch_nava, "_set_mem_fraction",
                        lambda device, frac: calls.append(frac))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(autobatch_nava, "_probe_once",
                        lambda *a: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x"))
                        if a[5] > 2 else None)
    got = autobatch_nava.resolve_nava_micro_batch(
        _args("auto"), None, None, None, _opt(), {"batch_size": 1}, SimpleNamespace(type="cuda"), 1, 0
    )
    assert got == 2
    assert calls[0] == 0.8   # reserved headroom during probing
    assert calls[-1] == 1.0  # restored the whole card for training
