"""The sampler layer of the breaker registry (yeto.breakers.layers.sampler): its
declarations load, its defaults are the sampler's code constants, and a spec
resolves to exactly the values the sampler read from it before the registry."""

from __future__ import annotations

import subprocess
import sys

import pytest

from yeto import breakers
from yeto.breakers.layers import sampler as layer
from yeto.breakers.registry import default_registry
from yeto.rl.engine.sampler_breakers import config_dispense_limits, dispense_limits, spec_breakers

SAMPLER = {b.name: b for b in layer.BREAKERS}


def test_the_layer_loads_with_unique_sampler_names():
    registry = default_registry()
    for name in SAMPLER:
        assert name in registry and registry.get(name).layer == "sampler"
    invariants = [b for b in layer.BREAKERS if b.kind == "invariant"]
    assert {b.name for b in invariants} == {"sampler.token_alignment", "sampler.compaction_startup"}
    assert all(not b.can_disable for b in invariants)


def test_the_layer_defaults_are_the_code_constants():
    from yeto.rl.agentic.types import AgentBudget
    from yeto.rl.engine import (agentic_round, engine_sampler, incarnation_lease, infra_streak, sampler_loop,
                                store_dispense, vram_guard)

    assert layer.LOAD_FAILURES_N == engine_sampler.EngineSamplerConfig.max_consecutive_load_failures
    assert layer.MAX_READ_OUTAGE_S == sampler_loop.DEFAULT_MAX_READ_OUTAGE_S
    assert layer.MAX_DAMAGED_LATEST_READS == sampler_loop.DEFAULT_MAX_DAMAGED_LATEST_READS
    assert layer.WRITE_DEFER_MAX_S == engine_sampler.DEFAULT_WRITE_DEFER_MAX_S
    assert layer.MAX_UNREAD_S == incarnation_lease.DEFAULT_MAX_UNREAD_S
    assert layer.RELOAD_MAX_DEFER_S == engine_sampler.DEFAULT_RELOAD_MAX_DEFER_S
    assert layer.ABANDON_GRACE_S == agentic_round.ABANDON_GRACE_S
    assert layer.INFRA_RETRIES == agentic_round.DEFAULT_INFRA_RETRIES
    assert layer.TRIAL_WALL_CLOCK_S == AgentBudget.wall_clock_s
    assert layer.DISPENSE_MAX_REDO == store_dispense.DEFAULT_MAX_REDO
    assert layer.DISPENSE_LEASE_S == store_dispense.DEFAULT_LEASE_S
    assert layer.VRAM_HIGH_FRAC == vram_guard.DEFAULT_HIGH_FRAC
    assert layer.VRAM_REALERT_S == vram_guard.DEFAULT_REALERT_S
    assert layer.INFRA_STREAK_N == infra_streak.DEFAULT_MAX_N
    assert layer.INFRA_STREAK_COOLDOWN_S == infra_streak.DEFAULT_COOLDOWN_S


def test_the_dispense_lease_and_redo_are_one_entry_each_with_three_aliases():
    for name, key, flag in (("sampler.dispense_lease", "lease_s", "--dispense-lease-s"),
                            ("sampler.dispense_redo", "max_redo", "--dispense-max-redo")):
        aliases = {(a.surface, a.path) for a in SAMPLER[name].aliases}
        assert aliases == {("spec", f"prompts.dispense.{key}"), ("pool_dispense", key), ("pool_cli_flag", flag)}


def test_the_pool_cli_dispense_flags_exist():
    from yeto.launch.nc import cli

    flags = []
    parser = __import__("argparse").ArgumentParser()
    cli._add_sampler_override_args(parser)
    for action in parser._actions:
        flags += action.option_strings
    for b in layer.BREAKERS:
        for alias in b.aliases:
            if alias.surface == "pool_cli_flag":
                assert alias.path in flags, alias.path


def test_the_registry_imports_no_heavy_module():
    code = ("import sys; from yeto.breakers.registry import default_registry; default_registry(); "
            "import yeto.rl.engine.sampler_breakers; "
            "print(sorted(m for m in ('torch', 'numpy', 'ray', 'miles', 'modal', 'boto3') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ------------------------------------------------------------------ no change for a spec without breakers
def _legacy(spec):
    """What the sampler read from ``spec`` before the registry (engine_sampler_entry,
    agentic_round.round_factory, store_dispense.open as of the commit before)."""
    agentic = spec.get("agentic") or {}
    lease = spec.get("incarnation_lease", True)
    guard = spec.get("vram_guard") if isinstance(spec.get("vram_guard"), dict) else {}
    dispense = (spec.get("prompts") or {}).get("dispense") or {}
    unread = (lease if isinstance(lease, dict) else {}).get("max_unread_s", 60.0)
    return {
        "sampler.drain.reload_max_defer_s": (None if "reload_max_defer_s" in spec and spec["reload_max_defer_s"] is None
                                             else float(spec.get("reload_max_defer_s", 600.0))),
        "sampler.load_failures.max_consecutive_n": int(spec.get("max_consecutive_load_failures", 3)),
        "sampler.write_defer.limit_s": (float(spec["write_defer_max_s"]) if spec.get("write_defer_max_s") is not None
                                        else 600.0),
        "sampler.incarnation_lease.enabled": not (lease is False or lease is None),
        "sampler.incarnation_lease.max_unread_s": None if unread is None else float(unread),
        "sampler.vram_guard.high_frac_ratio": float(guard.get("high_frac", 0.97)),
        "sampler.vram_guard.realert_s": float(guard.get("realert_s", 600.0)),
        "sampler.drain.timeout_s": (None if agentic.get("drain_timeout_s") is None
                                    else float(agentic["drain_timeout_s"])),
        "sampler.drain.abandon_grace_s": float(agentic.get("abandon_grace_s", 120.0)),
        "sampler.min_samples.min_n": int(agentic.get("min_samples", 2)),
        "sampler.infra_retries.max_n": int(agentic.get("infra_retries", 2)),
        "sampler.dispense_lease.lease_s": float(dispense.get("lease_s", 900.0)),
        "sampler.dispense_redo.max_n": int(dispense.get("max_redo", 2)),
        "sampler.store_read_outage.limit_s": 1800.0,
        "sampler.store_read_outage.max_damaged_latest_n": 120,
    }


SPECS = [
    {},
    {"reload_max_defer_s": None, "write_defer_max_s": None, "incarnation_lease": False},
    {"reload_max_defer_s": 2400, "max_consecutive_load_failures": "5", "write_defer_max_s": 0,
     "incarnation_lease": {"max_unread_s": None, "poll_s": 10}, "vram_guard": {"high_frac": 0.9}},
    {"incarnation_lease": True, "vram_guard": True, "agentic": {"drain_timeout_s": 2400, "abandon_grace_s": 0,
                                                                "min_samples": 4, "infra_retries": 0}},
    {"incarnation_lease": None, "agentic": {"drain_timeout_s": None},
     "prompts": {"dispense": {"backend": "store", "lease_s": 1800, "max_redo": 6}}},
    {"incarnation_lease": {"max_unread_s": 120}, "prompts": {"dispense": {"backend": "store", "max_redo": 0}}},
]


@pytest.mark.parametrize("spec", SPECS)
def test_a_spec_without_breakers_resolves_to_what_the_sampler_read_before(spec):
    resolved = spec_breakers(spec)
    for key, want in _legacy(spec).items():
        got = resolved.enabled(key.rsplit(".", 1)[0]) if key.endswith(".enabled") else resolved.get(key)
        assert got == want and type(got) is type(want), (key, got, want)


def test_the_dispense_limits_of_a_spec_and_of_a_bare_config_agree():
    spec = SPECS[4]
    assert dispense_limits(spec_breakers(spec)) == config_dispense_limits(spec["prompts"]["dispense"]) == (1800.0, 6)
    assert config_dispense_limits({"backend": "store"}) == (900.0, 2)


def test_a_breakers_section_in_the_spec_sets_the_sampler_entries():
    spec = {"breakers": {"sampler": {"write_defer": {"limit_s": 900}, "infra_streak": {"action": "pause"}}}}
    resolved = spec_breakers(spec)
    assert resolved.get("sampler.write_defer.limit_s") == 900
    assert resolved.entry("sampler.infra_streak").action == "pause"


def test_a_section_that_disagrees_with_a_spec_key_is_refused():
    spec = {"write_defer_max_s": 300, "breakers": {"sampler": {"write_defer": {"limit_s": 900}}}}
    with pytest.raises(breakers.BreakerConfigError, match="write_defer"):
        spec_breakers(spec)


@pytest.mark.parametrize("spec", [{"max_consecutive_load_failures": 0}, {"agentic": {"min_samples": "two"}},
                                  {"breakers": {"sampler": {"drain": {"enabled": False}}}}])
def test_a_bad_value_is_refused(spec):
    with pytest.raises(ValueError):
        spec_breakers(spec)
