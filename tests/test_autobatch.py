"""Auto micro-batch resolution: probe doubling, consensus, rebalance."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto import autobatch


def _args(mb="auto", seq_len=2048, grad_accum=4):
    return SimpleNamespace(micro_batch_size=mb, seq_len=seq_len, grad_accum=grad_accum)


class Tok:
    def __len__(self):
        return 1000


def test_int_or_auto():
    assert autobatch.int_or_auto("auto") == "auto"
    assert autobatch.int_or_auto("8") == 8


def test_explicit_size_skips_probe(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("probe must not run")

    monkeypatch.setattr(autobatch, "_probe_once", boom)
    got = autobatch.resolve_micro_batch_size(
        _args(mb=3), None, {}, None, Tok(), SimpleNamespace(type="cuda"), 1
    )
    assert got == 3


def test_cpu_returns_one_without_probe(monkeypatch):
    monkeypatch.setattr(autobatch, "_probe_once", lambda *a: (_ for _ in ()).throw(AssertionError))
    got = autobatch.resolve_micro_batch_size(
        _args(), None, {}, None, Tok(), SimpleNamespace(type="cpu"), 1
    )
    assert got == 1


def test_probe_doubles_until_oom_and_keeps_last_passing(monkeypatch):
    sizes = []

    def probe(model, params, opt, seq_len, vocab, device, mb):
        sizes.append(mb)
        if mb >= 8:
            raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(autobatch, "_probe_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    opt = SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    got = autobatch.resolve_micro_batch_size(
        _args(), None, {}, opt, Tok(), SimpleNamespace(type="cuda"), 1
    )
    assert got == 4
    assert sizes == [1, 2, 4, 8]  # stops at first failure


def test_oom_at_one_is_a_clear_error(monkeypatch):
    def probe(*a, **k):
        raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(autobatch, "_probe_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    opt = SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    with pytest.raises(RuntimeError, match="does not fit even micro-batch 1"):
        autobatch.resolve_micro_batch_size(
            _args(), None, {}, opt, Tok(), SimpleNamespace(type="cuda"), 1
        )


def test_rebalance_grad_accum_preserves_effective_batch():
    # mb 1 -> accum unchanged; probed 8 with accum 4 -> 1 (never below 1);
    # probed 2 with accum 4 -> 2 (effective tokens preserved).
    assert autobatch.rebalance_grad_accum(4, 1) == 4
    assert autobatch.rebalance_grad_accum(4, 2) == 2
    assert autobatch.rebalance_grad_accum(4, 8) == 1
    assert autobatch.rebalance_grad_accum(1, 64) == 1
