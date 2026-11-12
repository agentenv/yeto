"""A drain-mode weight switch is bounded by reload_max_defer_s (k31-s1m sweep: a drain
waited 2684 s on one agent-timeout trial with the engine idle)."""

from __future__ import annotations

import pytest

from yeto.rl.engine import agentic_round as ar
from yeto.rl.engine.engine_sampler import DEFAULT_RELOAD_MAX_DEFER_S


@pytest.mark.parametrize(("drain", "defer", "want"), [
    (3600.0, 300.0, 300.0), (120.0, 300.0, 120.0), (None, 300.0, 300.0), (3600.0, None, 3600.0),
    (None, None, None), (300.0, 300.0, 300.0)])
def test_the_effective_drain_limit_is_the_smaller_bound(drain, defer, want):
    assert ar.effective_drain_timeout(drain, defer) == want


def test_reload_max_defer_reads_the_default_and_an_explicit_null():
    assert ar.reload_max_defer_s({}) == DEFAULT_RELOAD_MAX_DEFER_S
    assert ar.reload_max_defer_s({"reload_max_defer_s": 300}) == 300.0
    assert ar.reload_max_defer_s({"reload_max_defer_s": None}) is None


def test_a_drain_timeout_above_the_reload_bound_is_warned():
    spec = {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 3600}}
    warning = ar.drain_timeout_warning(spec)
    assert warning is not None and "3600 s" in warning and "300 s" in warning
    unset = ar.drain_timeout_warning({"agentic": {"weight_switch": "drain"}})
    assert unset is not None and "unset" in unset and f"{DEFAULT_RELOAD_MAX_DEFER_S:g} s" in unset
    auto = ar.drain_timeout_warning({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600}})
    assert auto is not None  # auto resolves to drain on an engine without live updates


@pytest.mark.parametrize("spec", [
    {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 300}},
    {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 60}},
    {"reload_max_defer_s": None, "agentic": {"weight_switch": "drain", "drain_timeout_s": 3600}},
    {"reload_max_defer_s": 300, "agentic": {"weight_switch": "live", "drain_timeout_s": 3600}},
    {"reload_max_defer_s": 300}])
def test_no_warning_when_the_drain_is_within_the_bound_or_not_used(spec):
    assert ar.drain_timeout_warning(spec) is None
