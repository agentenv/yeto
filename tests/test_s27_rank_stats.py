"""S27 G-AP: the per-rank time split of Miles' full-parameter operations."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from yeto.rl.adapters.miles.full_parameter import rank_stats
from yeto.rl.adapters.miles.full_parameter.rank_stats import RANK_STATS, rank_apply_stats, report_rank_stats


def test_the_plugin_returns_a_copy_of_the_rank_stats():
    actor = SimpleNamespace(_external_full_parameter_apply_stats={"commit": {"seconds": 1.5}})
    stats = rank_apply_stats(actor)
    assert stats == {"commit": {"seconds": 1.5}}
    stats["commit"]["seconds"] = 0
    assert actor._external_full_parameter_apply_stats["commit"]["seconds"] == 1.5
    assert rank_apply_stats(SimpleNamespace()) == {}


class _Group:
    def __init__(self, results=None, error=None):
        self.results, self.error, self.calls = results, error, []

    async def run_plugin(self, path, kwargs):
        self.calls.append((path, kwargs))
        if self.error is not None:
            raise self.error
        return self.results


def test_one_line_per_rank_with_the_phases_of_the_operation(capsys):
    group = _Group([{"prepare": {"seconds": 1.23456, "read_fragments": 3}, "commit": {"seconds": 9.0},
                     "export": {"seconds": 4.0}}, {}])
    asyncio.run(report_rank_stats(group, "apply", "actor"))
    assert group.calls == [(RANK_STATS, {})]
    lines = capsys.readouterr().err.splitlines()
    assert lines == ['[yeto-full-parameter] rank stats op=apply role=actor rank=0 '
                     '{"commit": {"seconds": 9.0}, "prepare": {"read_fragments": 3, "seconds": 1.235}}']


def test_a_failure_or_a_group_without_plugins_never_raises(capsys, monkeypatch):
    asyncio.run(report_rank_stats(_Group(error=RuntimeError("cell lost")), "materialize", "actor"))
    assert "rank stats after materialize failed: RuntimeError: cell lost" in capsys.readouterr().err
    asyncio.run(report_rank_stats(object(), "export", "actor"))
    monkeypatch.setenv(rank_stats.RANK_STATS_ENV, "0")
    group = _Group([{"export": {"seconds": 1.0}}])
    asyncio.run(report_rank_stats(group, "export", "actor"))
    assert group.calls == [] and capsys.readouterr().err == ""
