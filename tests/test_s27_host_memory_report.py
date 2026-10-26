"""S27 G-EZ3: host memory around each verl training step (``YETO_HOST_MEMORY``).

The 120B verl island on npu1 used about 150-175 GiB per node more than its
parameters, gradients and Adam state (G-EZ2). ``report`` logs every worker's
RSS by kind and its node's meminfo before and after each step; ``trim`` also
runs ``release_host_memory`` (full GC + ``malloc_trim(0)``, S27 G-EY) after
each step and after each export.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.engine import host_memory as hm

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")


def test_mode_is_off_unless_set_and_refuses_unknown_values():
    assert hm.host_memory_mode({}) is None
    assert hm.host_memory_mode({hm.HOST_MEMORY_ENV: "0"}) is None
    assert hm.host_memory_mode({hm.HOST_MEMORY_ENV: " Report "}) == "report"
    assert hm.host_memory_mode({hm.HOST_MEMORY_ENV: "trim"}) == "trim"
    with pytest.raises(ValueError, match="expected one of"):
        hm.host_memory_mode({hm.HOST_MEMORY_ENV: "yes"})


@linux
def test_readers_give_mib_by_kind():
    process, node = hm.process_memory(), hm.node_memory()
    assert {"VmRSS", "RssAnon", "RssFile", "RssShmem"} <= set(process)
    assert {"MemTotal", "MemAvailable", "AnonPages", "Shmem"} <= set(node)
    assert 0 < process["VmRSS"] < node["MemTotal"]
    assert hm.process_memory(pid=2 ** 31 - 1) == {}  # no such process: empty, not an error


@linux
def test_worker_report_trims_only_when_asked():
    plain = fp.remote_host_memory(None, False)
    assert "release" not in plain and plain["process"]["VmRSS"] > 0 and plain["node"]["MemTotal"] > 0
    trimmed = fp.remote_host_memory(None, True)
    assert trimmed["release"]["malloc_trim"] is True and trimmed["release"]["rss_anon_mib_after"] is not None


def test_summary_is_per_node_with_the_lowest_available():
    def report(host, rank, available, anon, trimmed=None):
        out = {"host": host, "rank": rank, "pid": 100 + rank,
               "process": {"VmRSS": anon + 2, "RssAnon": anon, "RssFile": 2, "RssShmem": 0},
               "node": {"MemTotal": 1000, "MemAvailable": available, "Shmem": 5}}
        if trimmed is not None:
            out["release"] = {"rss_anon_mib_before": anon + trimmed, "rss_anon_mib_after": anon,
                              "gc_s": 0.5, "trim_s": 1.0}
        return out

    summary = hm.summarize_by_node([report("n3", 9, 300, 100, 7), report("n2", 0, 500, 120, 3),
                                    report("n3", 8, 290, 110, 1)])
    n3 = summary["nodes"]["n3"]
    assert n3["workers"] == 2 and n3["mem_available_mib"] == 290 and n3["mem_total_mib"] == 1000
    assert n3["workers_rssanon_mib"] == 210 and n3["workers_vmrss_mib"] == 214 and n3["trimmed_mib"] == 8
    assert n3["trim_s_max"] == 1.5
    assert summary["nodes"]["n2"]["mem_available_mib"] == 500
    assert [(w["host"], w["rank"]) for w in summary["workers"]] == [("n2", 0), ("n3", 8), ("n3", 9)]


class _Island:
    """The parts of ``FullyAsyncIsland`` that ``FullyAsyncTrainerGroup.train_step`` uses."""

    learner_id = 0
    full_group = None
    layout_hash = "0" * 64

    def __init__(self, tmp_path):
        from pathlib import Path

        self.out = Path(tmp_path)
        self.calls, self.events = [], []
        self.last_metrics, self.last_criteria = {}, {}

    def call(self, name, *args, timeout=None):
        assert name == "yeto_train_step"
        self.calls.append(name)
        return {"metrics": {"actor/grad_norm": 1.0, "actor/pg_loss": 0.1}, "criteria": {}, "param_version": 1}

    def call_workers(self, fn, *args):
        self.calls.append((fn.__name__, *args))
        return [fn(None, *args)]

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def append_jsonl(self, name, record):
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def _step(tmp_path, monkeypatch, mode):
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncTrainerGroup
    from yeto.rl.engine.ports import RolloutBatchHandle

    if mode is None:
        monkeypatch.delenv(hm.HOST_MEMORY_ENV, raising=False)
    else:
        monkeypatch.setenv(hm.HOST_MEMORY_ENV, mode)
    island = _Island(tmp_path)
    group = SimpleNamespace(sample_ids=("s0", "s1"), token_count=8)
    batch = RolloutBatchHandle(rollout_id=3, policy_version=0, policy_hash="a" * 64, groups=(group,), completed=1,
                               aborted=0, payload=None)
    FullyAsyncTrainerGroup(island).train_step(batch)
    return island


@linux
def test_train_step_reports_around_the_step_and_trims_after_it(tmp_path, monkeypatch):
    island = _step(tmp_path, monkeypatch, "trim")
    assert island.calls == [("remote_host_memory", False), "yeto_train_step", ("remote_host_memory", True)]
    records = [json.loads(line) for line in (tmp_path / "verl-host-memory-0.jsonl").read_text().splitlines()]
    assert [(r["phase"], r["release"], r["rollout_id"]) for r in records] == [
        ("before_step", False, 3), ("after_step", True, 3)]
    assert all(len(r["nodes"]) == 1 and r["driver"]["VmRSS"] > 0 for r in records)
    assert [e for e, _ in island.events].count("rl_host_memory") == 2


@linux
def test_train_step_reports_without_trimming_and_is_silent_when_off(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    island = _step(tmp_path / "a", monkeypatch, "report")
    assert island.calls == [("remote_host_memory", False), "yeto_train_step", ("remote_host_memory", False)]
    island = _step(tmp_path / "b", monkeypatch, None)
    assert island.calls == ["yeto_train_step"]
    assert not (tmp_path / "b" / "verl-host-memory-0.jsonl").exists()


@linux
def test_export_release_trims_the_host_only_in_trim_mode(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() and not (hasattr(torch, "npu") and torch.npu.is_available()):
        # remote_release_memory needs a device allocator; stand one in
        class _Alloc:
            @staticmethod
            def memory_allocated():
                return 0

            memory_reserved = memory_allocated

            @staticmethod
            def empty_cache():
                return None

            synchronize = empty_cache

        monkeypatch.setattr(torch, "cuda", _Alloc, raising=False)
    monkeypatch.setenv(hm.HOST_MEMORY_ENV, "trim")
    assert fp.remote_release_memory(None)["host"]["malloc_trim"] is True
    assert "host" not in fp.remote_release_memory(None, False)  # a per-file release of the export
    monkeypatch.setenv(hm.HOST_MEMORY_ENV, "report")
    assert "host" not in fp.remote_release_memory(None)
