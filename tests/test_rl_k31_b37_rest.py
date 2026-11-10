"""yeto follow-ups of Miles k31/b3-b7 (plan B1-B3-B5-B6-B7 section 8, items 2-6).

* runtime statistics are read by Miles' own kinds through rollout_runtime_stats.take;
* the agent suspend/resume/abort hooks take Miles' immutable custom-function view;
* the k31 B4 96437f37 describe_cells shape (no workers_lost state) classifies as no_workers;
* the over-sampling tally Miles publishes reaches the rollout handle;
* --rl-host-sync-free / --no-rl-host-sync-free reaches the Miles argv.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
import typing
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import carry_over, pins
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata
from yeto.rl.engine import proc_state as ps

from test_rl_algo_supp_metrics import STATS as OVER_SAMPLING
from test_rl_miles_adapter_rollout import H, group


@pytest.fixture
def k31_channel(monkeypatch):
    """A ``miles.rollout.rollout_runtime_stats`` with Miles' kinds; returns its store."""
    store: dict[str, dict] = {}
    monkeypatch.setitem(sys.modules, "miles", types.ModuleType("miles"))
    monkeypatch.setitem(sys.modules, "miles.rollout", types.ModuleType("miles.rollout"))
    registry = types.ModuleType("miles.rollout.rollout_runtime_stats")
    registry.publish = lambda kind, stats: store.__setitem__(kind, dict(stats))  # type: ignore[attr-defined]
    registry.take = lambda kind: store.pop(kind, None)  # type: ignore[attr-defined]
    sys.modules["miles.rollout"].rollout_runtime_stats = registry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "miles.rollout.rollout_runtime_stats", registry)
    return registry


# ------------------------------------------------------------------ 2. statistics via take(kind)


def test_yeto_kinds_are_miles_stats_kinds():
    """The kinds yeto asks for are the StatsKind literal of the installed Miles."""
    stats = pytest.importorskip("miles.rollout.rollout_runtime_stats")
    assert set(typing.get_args(stats.StatsKind)) == set(ps.RUNTIME_STATS_KINDS)
    assert carry_over.SUSPEND_STATS_KIND in ps.RUNTIME_STATS_KINDS
    assert carry_over.RESUME_STATS_KIND in ps.RUNTIME_STATS_KINDS


def test_legacy_attribute_names_map_to_kinds():
    for kind in ps.RUNTIME_STATS_KINDS:
        assert ps.runtime_stats_kind(kind) == kind
        assert ps.runtime_stats_kind(ps.legacy_runtime_stats_attr(kind)) == kind
    assert ps.legacy_runtime_stats_attr("abort_discard") == "rollout_abort_discard_stats"


def _extract(args, tmp_path, kept):
    hook.record_trained_groups(args, kept)
    os.environ[hook.META_SINK_ENV] = f"dir:{tmp_path}"
    try:
        hook.extract_rollout_metadata(args, kept, SimpleNamespace(sample_group_index=6))
    finally:
        os.environ.pop(hook.META_SINK_ENV)
    return json.loads((tmp_path / "rollout-3.json").read_text())


def test_rollout_metadata_reads_the_k31_abort_and_over_sampling_tallies(tmp_path, k31_channel):
    """Miles k31 abort() / MetricGatherer publish before the all-samples hook runs;
    the hook takes them by kind. A stale legacy attribute is ignored and kept."""
    kept = [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])]
    k31_channel.publish("abort_discard", {"groups": 4, "samples": 8, "response_tokens": 1234,
                                          "unknown_groups": 1})
    k31_channel.publish("over_sampling", OVER_SAMPLING)
    stale = {"groups": 99, "samples": 99, "response_tokens": 99, "unknown_groups": 0}
    args = SimpleNamespace(rollout_abort_discard_stats=dict(stale))
    payload = _extract(args, tmp_path, kept)
    h = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    assert (h.aborted_in_flight_groups, h.aborted_in_flight_trajectories,
            h.aborted_in_flight_tokens, h.aborted_in_flight_unknown_groups) == (4, 8, 1234, 1)
    assert payload["over_sampling"] == OVER_SAMPLING and h.over_sampling == OVER_SAMPLING
    assert args.rollout_abort_discard_stats == stale  # never read, never reset
    assert k31_channel.take("abort_discard") is None and k31_channel.take("over_sampling") is None


def test_rollout_metadata_without_a_k31_tally_has_no_over_sampling(tmp_path, k31_channel):
    """Partial rollout publishes no abort tally and a rollout with nothing submitted no
    over-sampling tally: unknown, no field, and no fallback to a stale attribute."""
    kept = [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])]
    args = SimpleNamespace(rollout_over_sampling_stats=dict(OVER_SAMPLING))
    payload = _extract(args, tmp_path, kept)
    assert "over_sampling" not in payload


def test_suspend_fields_read_the_k31_suspend_and_resume_tallies(k31_channel):
    k31_channel.publish("suspend", {"suspended_groups": 2, "suspended_done_groups": 1,
                                    "over_age_cancelled_groups": 0, "over_age_cancelled_samples": 0,
                                    "over_age_cancelled_tokens": 0, "over_age_unknown_groups": 0,
                                    "engines_idle_confirmed": 1})
    k31_channel.publish("resume", {"resumed_groups": 3, "resumed_done_groups": 1})
    args = SimpleNamespace(rollout_suspend_stats={"suspended_groups": 77})  # stale, ignored
    fields = carry_over.suspend_fields(args, [[SimpleNamespace(metadata={})]])
    assert fields["suspended_groups"] == 2 and fields["suspended_done_groups"] == 1
    assert fields["resumed_groups"] == 3 and fields["engines_idle_confirmed"] == 1
    assert args.rollout_suspend_stats == {"suspended_groups": 77}
    assert carry_over.suspend_fields(args, []) == {}  # taken once


# ------------------------------------------------------------------ 3. hooks on the custom-function view


class _StrictView:
    """Like Miles' ImmutableNamespace from compute_custom_function_config: frozen.
    Any attribute read or write fails the test, so a hook must not depend on it."""

    def __getattr__(self, name):
        raise AssertionError(f"the hook read {name!r} from the custom-function view")

    def __setattr__(self, name, value):
        raise AssertionError(f"the hook wrote {name!r} on the custom-function view")


def test_suspend_capable_agent_hooks_take_the_immutable_view(monkeypatch):
    """Miles k31 call_agent_hook / call_agent_abort_hook pass
    compute_custom_function_config(args, agent_function), not the raw args. The
    Harbor hooks read their relay settings from the environment (the B0 owner
    rule: yeto's own settings are not Miles args), so the view is never touched."""
    import importlib

    from yeto.rl.adapters.miles import policy_age

    for path in policy_age.SUSPEND_CAPABLE_AGENTS:
        module = importlib.import_module(path.rpartition(".")[0])
        for name in ("suspend", "resume", "abort"):
            assert asyncio.iscoroutinefunction(getattr(module, name)), (path, name)

    from yeto.rl.adapters.miles import harbor_agent as mod

    events = []

    class Gate:
        def close(self):
            events.append("close")

        def open(self):
            events.append("open")

    monkeypatch.setattr(mod, "turn_gate", lambda: Gate())
    monkeypatch.setattr(mod, "check_suspend_config", lambda environ=None: None)
    monkeypatch.delenv(mod.TEST_HOLD_EXTRA_ENV, raising=False)
    view = _StrictView()
    asyncio.run(mod.suspend(view))
    asyncio.run(mod.resume(view))
    asyncio.run(mod.abort(view))
    assert events == ["close", "open"]


# ------------------------------------------------------------------ 4. workers_lost on k31 B4


def test_k31_b4_describe_cells_has_no_workers_lost_state():
    """Miles k31/b4 96437f37 (before k31/b4-lost efc83590) RayWorkerManager.describe_cells
    reports unbound / stopped / running and no lost_workers: on such a pin the watchdog
    keeps its no_workers classification instead of failing. The efc83590 shape is
    covered in test_rl_k31_workers_lost."""
    from yeto.rl.adapters.miles.elastic_wiring import _lost_workers_of

    class _Remote:
        def __init__(self, fn):
            self.remote = fn

    class _Ray:
        def get(self, ref, timeout=None):
            return ref

    k31_entry = {"cell_id": "c3", "alias": None, "pool_id": "rollout", "deferred": False,
                 "state": "stopped", "generation": 2, "pg_name": "rollout", "pg_slot_offset": 0,
                 "bundles": [3], "gpu_ids": [3]}
    manager = type("M", (), {"describe_cells": _Remote(lambda pool_ids=None: {"c3": k31_entry})})()
    assert _lost_workers_of(manager, _Ray(), "c3", 1.0) is None


# ------------------------------------------------------------------ 6. --rl-host-sync-free


def test_island_flag_reaches_the_miles_argv(tmp_path, monkeypatch):
    from test_agentic_stage3_island_argv import _island_argv, _miles_argv, island_entry

    on = island_entry.parse_args(_island_argv(tmp_path))
    assert on.rl_host_sync_free is True
    argv = _miles_argv(on, tmp_path, monkeypatch)
    assert "--no-host-sync-free" not in argv and "--host-sync-free" not in argv
    off = island_entry.parse_args(_island_argv(tmp_path, "--no-rl-host-sync-free"))
    assert off.rl_host_sync_free is False
    argv_off = _miles_argv(off, tmp_path, monkeypatch)
    assert argv_off.count("--no-host-sync-free") == 1
    assert [a for a in argv_off if a != "--no-host-sync-free"] == argv


def test_launcher_parses_forwards_and_checks_the_flag(monkeypatch):
    from yeto import launcher as L
    from yeto.cli import parse_args
    from yeto.rl.adapters.miles import run_config_rules

    base = ["--training-mode", "rl", "--gpu", "modal:1xh100", "--model", "Qwen/Qwen3-0.6B",
            "--data", "zhuzilin/gsm8k"]
    args = parse_args(base)
    assert args.rl_host_sync_free is True
    assert "host-sync-free" not in L._ports_infra_flags(args)[1]
    off = parse_args([*base, "--no-rl-host-sync-free"])
    assert off.rl_host_sync_free is False
    assert " --no-rl-host-sync-free" in L._ports_infra_flags(off)[1]

    run_config_rules.check_host_sync_free(True)
    run_config_rules.check_host_sync_free(False)  # the k31/b3-b7 head has the switch
    for older in (pins.MILES_K31_B2_COMMIT, pins.MILES_PORTS_V1_COMMIT):
        monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", older)
        run_config_rules.check_host_sync_free(True)  # the default never needs it
        with pytest.raises(ValueError, match="--no-rl-host-sync-free needs Miles"):
            run_config_rules.check_host_sync_free(False)
