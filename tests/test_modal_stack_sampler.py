"""PeriodicStackSampler and allow_any_ptracer (S27 RAY-SEGV R5): opt-in periodic py-spy ring in the tape dir."""
import json
import os
import stat
import subprocess
import sys

import pytest

from yeto import modal_runner
from yeto.modal_runner import PeriodicStackSampler, allow_any_ptracer, python_pids, stack_sample_interval_s


def _fake_proc(root, pids):
    for pid, cmd in pids.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    (root / "self").mkdir()


def _fake_py_spy(tmp_path):
    exe = tmp_path / "py-spy"
    exe.write_text("#!/bin/sh\necho \"Thread main (stack of pid $3) args $*\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


ISLAND = "/opt/sglang/bin/python3 -m yeto.rl.adapters.miles.island_entry --learner-id 0"


def test_interval_and_from_env_are_opt_in(tmp_path):
    assert stack_sample_interval_s({}) is None
    assert stack_sample_interval_s({"YETO_STACK_SAMPLE_S": ""}) is None
    assert stack_sample_interval_s({"YETO_STACK_SAMPLE_S": "0"}) is None
    assert stack_sample_interval_s({"YETO_STACK_SAMPLE_S": "45"}) == 45.0
    assert PeriodicStackSampler.from_env({}, str(tmp_path), 0) is None
    s = PeriodicStackSampler.from_env({"YETO_STACK_SAMPLE_S": "30", "YETO_STACK_SAMPLE_KEEP": "3",
                                       "YETO_STACK_SAMPLE_NATIVE": "1"}, str(tmp_path), 2)
    assert (s.interval_s, s.keep, s.native, s.nonblocking, s.match, s.rank) == (
        30.0, 3, True, False, "island_entry", 2)
    with pytest.raises(ValueError):
        PeriodicStackSampler(str(tmp_path), 0, interval_s=10, keep=0)


def test_command_flags(tmp_path):
    s = PeriodicStackSampler(str(tmp_path), 0, interval_s=1, py_spy="/x/py-spy")
    assert s.command(7) == ["/x/py-spy", "dump", "--pid", "7"]
    s = PeriodicStackSampler(str(tmp_path), 0, interval_s=1, py_spy="/x/py-spy", native=True, nonblocking=True)
    assert s.command(7) == ["/x/py-spy", "dump", "--pid", "7", "--native", "--nonblocking"]


def test_targets_are_the_matching_python_processes(tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, {11: "ray::Actor.train", 12: ISLAND, 13: "/bin/bash -c island_entry"})
    assert [p for p, _ in python_pids(str(proc))] == [11, 12]
    s = PeriodicStackSampler(str(tmp_path / "out"), 0, interval_s=1, py_spy="x", proc=str(proc))
    assert [p for p, _ in s.targets()] == [12]
    s = PeriodicStackSampler(str(tmp_path / "out"), 0, interval_s=1, py_spy="x", proc=str(proc), match="ray::")
    assert [p for p, _ in s.targets()] == [11]


def test_ring_keeps_the_newest_rounds_and_indexes_every_round(tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, {12: ISLAND, 13: "/usr/bin/python3 -m other"})
    out = tmp_path / "out"
    s = PeriodicStackSampler(str(out), 1, interval_s=1, keep=2, py_spy=_fake_py_spy(tmp_path), proc=str(proc))
    recs = [s.sample() for _ in range(5)]
    assert sorted(os.listdir(out)) == ["stack-samples-rank1-00.txt", "stack-samples-rank1-01.txt",
                                       "stack-samples-rank1.jsonl"]
    assert [r["slot"] for r in recs] == [0, 1, 0, 1, 0]
    index = [json.loads(line) for line in open(out / "stack-samples-rank1.jsonl")]
    assert [r["round"] for r in index] == [1, 2, 3, 4, 5]
    assert index[-1]["pids"] == [12] and index[-1]["rc"] == [0]
    text = open(out / "stack-samples-rank1-00.txt").read()
    assert text.startswith("# round 5 ") and "stack of pid 12" in text and "pid 13" not in text


def test_missing_py_spy_is_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_runner, "_which", lambda cmd: None)
    proc = tmp_path / "proc"
    _fake_proc(proc, {12: ISLAND})
    s = PeriodicStackSampler(str(tmp_path / "out"), 0, interval_s=1, proc=str(proc))
    assert s.py_spy is None
    rec = s.sample()
    assert rec["rc"] == ["no py-spy"]
    assert "py-spy not found" in open(tmp_path / "out" / "stack-samples-rank0-00.txt").read()


def test_loop_survives_a_failed_round(tmp_path, capsys):
    s = PeriodicStackSampler(str(tmp_path), 0, interval_s=0.001, py_spy="x")
    calls = []

    def sample():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        s.stop()
        return {}

    s.sample = sample
    s._loop()
    assert len(calls) == 2 and "round failed: OSError('disk full')" in capsys.readouterr().out


def test_allow_any_ptracer_only_when_sampling(tmp_path):
    assert allow_any_ptracer({}) is None
    # prctl changes the calling process: run it in a child interpreter
    code = ("from yeto.modal_runner import allow_any_ptracer; "
            "print('RESULT', allow_any_ptracer({'YETO_STACK_SAMPLE_S': '5'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout
    if os.path.exists("/proc/sys/kernel/yama/ptrace_scope"):
        assert "RESULT True" in out and "PR_SET_PTRACER_ANY set" in out
    else:  # no Yama: the prctl is refused, the helper reports it instead of raising
        assert "RESULT False" in out


def test_host_map_probe_is_opt_in_and_rank0_only(tmp_path):
    from yeto.modal_runner import start_host_map_probe

    assert start_host_map_probe({}, 0, str(tmp_path)) is None
    assert start_host_map_probe({"YETO_HOST_MAP_PROBE_PAGES": "0"}, 0, str(tmp_path)) is None
    assert start_host_map_probe({"YETO_HOST_MAP_PROBE_PAGES": "1000"}, 1, str(tmp_path)) is None
    assert os.listdir(tmp_path) == []


def test_host_map_probe_runs_in_the_background_and_logs_the_exit(tmp_path):
    from yeto.modal_runner import start_host_map_probe

    thread = start_host_map_probe({"YETO_HOST_MAP_PROBE_PAGES": "2000"}, 0, str(tmp_path))
    thread.join(timeout=60)
    lines = open(tmp_path / "host-map-probe-rank0.log").read().splitlines()
    assert lines[0] == "MAPPROBE start pages=2000"
    assert lines[-2].startswith("MAPPROBE ok touched=2000 ") and lines[-1] == "MAPPROBE exit rc=0"
    assert sum(1 for line in lines if line.startswith("MAPPROBE touched=")) == 8
