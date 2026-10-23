"""S27 G-AG diagnostics: host facts, named Miles group calls, push time split."""

from __future__ import annotations

import asyncio
import json

import pytest

from yeto.rl import host_info
from yeto.rl.adapters.miles.models import chunked_full_parameter as adapter_module
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    TimedGroup,
    skip_resident_supported,
    timed_group,
    transport_supports,
)


def test_the_host_report_is_json_and_names_the_facts():
    report = host_info.host_report()
    json.dumps(report)
    for key in ("cpu_model", "cpu_count", "cpus_usable", "mem_total_gib", "cgroup_cpu_limit",
                "cgroup_memory_limit_gib", "numa_nodes", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "torch_threads", "torch_interop_threads"):
        assert key in report
    assert report["cpu_count"] and report["mem_total_gib"]


def test_the_rank_report_never_raises(monkeypatch):
    monkeypatch.setattr(host_info, "host_report", lambda: 1 / 0)
    assert "ZeroDivisionError" in host_info.rank_host_report(object())["error"]


def test_rank_reports_print_per_rank_and_survive_a_failing_group(capsys):
    class Group:
        async def run_plugin(self, path, kwargs):
            assert path == host_info.RANK_HOST_REPORT and kwargs == {}
            return [host_info.rank_host_report(None), {"error": "x"}]

    class Broken:
        async def run_plugin(self, path, kwargs):
            raise RuntimeError("cell down")

    asyncio.run(host_info.report_rank_hosts(Group(), "actor"))
    asyncio.run(host_info.report_rank_hosts(Broken(), "actor"))
    asyncio.run(host_info.report_rank_hosts(object(), "actor"))  # no run_plugin: nothing
    err = capsys.readouterr().err
    assert "[yeto-host] actor rank 0 {" in err and "[yeto-host] actor rank 1" in err
    assert "rank report failed: RuntimeError: cell down" in err


class _Group:
    def __init__(self):
        self.calls = []
        self.flag = 7

    async def apply_full_parameter_chunked_cut(self, cut, *, skip_resident_fragments=False):
        self.calls.append(("apply", cut, skip_resident_fragments))
        return (1,)

    async def full_parameter_optimizer_states(self):
        raise ValueError("proof failed")

    def sync_method(self, value):
        return value + 1


def test_the_timed_group_logs_each_call_and_keeps_signatures(capsys):
    inner = _Group()
    group = timed_group(inner, "actor")
    assert isinstance(group, TimedGroup) and timed_group(group, "actor") is group
    assert group.flag == 7 and group.sync_method(1) == 2
    assert asyncio.run(group.apply_full_parameter_chunked_cut("cut", skip_resident_fragments=True)) == (1,)
    with pytest.raises(ValueError, match="proof failed"):
        asyncio.run(group.full_parameter_optimizer_states())
    err = capsys.readouterr().err
    assert "miles op=apply_full_parameter_chunked_cut role=actor elapsed_s=" in err and "ok=true" in err
    assert "miles op=full_parameter_optimizer_states role=actor" in err and "ok=false" in err
    assert inner.calls == [("apply", "cut", True)]
    # Feature detection reads the wrapped signature.
    assert skip_resident_supported(group) is skip_resident_supported(inner) is True
    assert transport_supports(group, "apply_full_parameter_chunked_cut", "skip_resident_fragments", "x")
    assert getattr(group, "materialize_full_parameter_fragments", None) is None


def test_the_timed_group_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(adapter_module.OP_TIMING_ENV, "0")
    inner = _Group()
    assert timed_group(inner, "actor") is inner


def test_the_round_event_reports_each_boundarys_push_time_split():
    from types import SimpleNamespace

    from yeto.rl.engine.full_parameter_sync import FullParameterSync

    actor = SimpleNamespace(push_timing_totals={"count": 2, "delta_s": 3.0, "total_s": 5.0})
    sync = FullParameterSync(SimpleNamespace(streams={"actor": actor}), groups={"actor": None},
                             run=None, trajectories=None)
    assert sync._push_timing_fields() == {"sync/push_count": 2, "sync/push_delta_s": 3.0, "sync/push_total_s": 5.0}
    actor.push_timing_totals.update(count=5, delta_s=4.5, total_s=9.0)
    assert sync._push_timing_fields() == {"sync/push_count": 3, "sync/push_delta_s": 1.5, "sync/push_total_s": 4.0}
    assert FullParameterSync(SimpleNamespace(streams={"actor": SimpleNamespace()}), groups={"actor": None},
                             run=None, trajectories=None)._push_timing_fields() == {}
