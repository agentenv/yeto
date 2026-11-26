"""Resident ssh: trainer island (yeto.launch.ssh_resident) over a fake SSH cluster:
start, liveness watch, one rank's failure, a normal end and release, stop, and a
restarted launcher reattaching (or refusing) through the island's journal."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.cost.ledger import ResourceLedger
from yeto.hw import card_tools
from yeto.launch import ssh_resident
from yeto.launch.fleet import DONE, FleetController, FixedRosterIslandAbandoned
from yeto.launch.node_inventory import Node
from yeto.launch.ssh_resident import SshIslandOps, dry_run_record, rank_layout
from yeto.launch.ssh_trainer_island import (
    CUDA_RDMA_DOCKER_ARGS,
    NODE_STATE_MARK,
    SshIslandError,
    SshOps,
    SshTrainerSpec,
    _write_files,
    container_name,
    kill_script,
    parse_node_state,
    remote_paths,
    start_script,
    status_script,
)
from yeto.modal_runner import RoutingOps
from yeto.rl import event_echo

INC = re.compile(r"trainer-islands/[^/\"]+/[^/\"]+-i(\d+)")


# ------------------------------------------------------------------ the fake SSH cluster
class FakeStream:
    """One SSH stream (a start or a follow): its lines until the node's process ends
    or the link drops; ``poll`` is the ssh exit code."""

    def __init__(self, lines=()):
        self.q: queue.Queue = queue.Queue()
        for line in lines:
            self.q.put(line)
        self.rc = None
        self.stdout = self._lines()

    def _lines(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item

    def push(self, line):
        self.q.put(line)

    def close(self, rc=0):
        if self.rc is None:
            self.rc = rc
            self.q.put(None)

    def poll(self):
        return self.rc

    def terminate(self):
        self.close(-15)


class FakeCluster(SshOps):
    """SSH nodes in memory. Per node and incarnation: state (running / exited / lost),
    exit code, log lines and container; the scripts are recognized by their marks."""

    def __init__(self, names, *, docker=True):
        self.runs = {n: {} for n in names}
        self.streams = {n: [] for n in names}
        self.docker = docker
        self.unreachable: set[str] = set()
        self.kill_rc: dict[str, int] = {}
        self.calls: list[tuple] = []

    @staticmethod
    def _k(script):
        return int(INC.search(script).group(1))

    def start(self, node, script):
        if node.name in self.unreachable:
            s = FakeStream()
            s.close(255)
            return s
        k, runs = self._k(script), self.runs[node.name]
        if "setsid bash -c" in script:
            self.calls.append(("start", node.name, k))
            runs[k] = {"state": "running", "rc": None, "log": [f"started {node.name} i{k}\n"],
                       "container": "running" if self.docker else "none", "script": script}
        else:
            self.calls.append(("follow", node.name, k))
        run = runs.get(k)
        stream = FakeStream(list(run["log"]) if run else [])
        if run is None or run["state"] != "running":
            stream.close(0 if run else 1)
        else:
            self.streams[node.name].append((k, stream))
        return stream

    def run(self, node, script, *, timeout=300.0):
        done = lambda rc, out="", err="": subprocess.CompletedProcess([], rc, stdout=out, stderr=err)  # noqa: E731
        if node.name in self.unreachable:
            return done(255, "", "ssh: connect to host: Connection timed out")
        if script == "uname -m":
            return done(0, "x86_64\n")
        if script == "true":  # the reachability probe before a later start
            return done(0)
        # the preflight's card check (a status call carries the node health check's nvidia-smi too)
        if ("nvidia-smi" in script and NODE_STATE_MARK not in script) or script.startswith("docker pull"):
            return done(0, "ok\n")
        k, runs = self._k(script), self.runs[node.name]
        run = runs.get(k)
        if NODE_STATE_MARK in script:
            self.calls.append(("status", node.name, k))
            if run is None:
                return done(0, f"{NODE_STATE_MARK} state=absent rc= container=none\n")
            rc = "" if run["rc"] is None else run["rc"]
            return done(0, f"{NODE_STATE_MARK} state={run['state']} rc={rc} container={run['container']}\n")
        if "kill -TERM" in script:
            self.calls.append(("kill", node.name, k))
            rc = self.kill_rc.get(node.name, 0)
            if run is not None and rc == 0:
                if run["state"] == "running":
                    run.update(state="exited", rc=143)
                run["container"] = "none"
                self._close(node.name, k, 0)
            return done(rc, "", "" if rc == 0 else "[yeto] kill: container still exists")
        if script.startswith("cat "):
            return done(0, "".join(run["log"]) if run else "")
        raise AssertionError(f"unexpected script on {node.name}: {script[:80]}")

    def fetch(self, node, script, target, *, timeout):
        """The ~/yeto-output copy on release (ssh_node_storage): nothing to copy here."""
        self.calls.append(("fetch", node.name))
        if node.name in self.unreachable:
            raise SshIslandError(f"copy from {node.name} ended with rc 255")
        return 0

    # -- test controls ------------------------------------------------------------
    def say(self, name, k, line):
        self.runs[name][k]["log"].append(line)
        for kk, s in self.streams[name]:
            if kk == k and s.rc is None:
                s.push(line)

    def _close(self, name, k, rc):
        for kk, s in self.streams[name]:
            if kk == k:
                s.close(rc)

    def exit(self, name, k, rc):
        self.runs[name][k].update(state="exited", rc=rc, container="exited")
        self._close(name, k, 0)

    def vanish(self, name, k):  # OOM kill / reboot: no exit code
        self.runs[name][k].update(state="lost", rc=None, container="none")
        self._close(name, k, 255)

    def drop(self, name):  # the SSH link drops; the process runs on
        for _k, s in self.streams[name]:
            s.close(255)

    def count(self, kind, k=None):
        return sum(1 for c in self.calls if c[0] == kind and (k is None or c[2] == k))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def b300(i: int, **kw) -> Node:
    base = dict(device_family="cuda", cpu_arch="x86_64", card="B300", cards_per_node=8, hbm_gib=288.0,
                host_memory_gib=2048.0, fabric="ib", interface="bond0", address=f"198.51.100.{10 + i}",
                runtime="docker", image="ghcr.io/o/yeto-miles-k31@sha256:" + "a" * 64,
                docker_args=CUDA_RDMA_DOCKER_ARGS, mounts={"/mnt/nvme/yeto": "/root/.cache"},
                price_per_card_hour=2.0)
    return Node(name=f"b300-{i}", host=f"ops@192.0.2.{10 + i}", **(base | kw))


def spec(n=3, tmp_path: Path | None = None, **kw) -> SshTrainerSpec:
    files = {}
    if tmp_path is not None:
        sel = tmp_path / "selection.json"
        sel.write_text('{"schema": 1}')
        files = {"~/yeto-rl-expert-selection.json": str(sel)}
    return SshTrainerSpec(nodes=tuple(b300(i) for i in range(n)), run_id="run1", workdir="/opt/yeto",
                          run_script="echo island", env={"YETO_X": "1"}, files=files, **kw)


@pytest.fixture(autouse=True)
def _cards(monkeypatch, tmp_path):
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    # the run's resource ledger (SshTrainerIslands records every incarnation there)
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))


def make_ops(tmp_path, cluster, *, clock=None, lines=None, **kw):
    sinks = lines if lines is not None else []
    return SshIslandOps(run_id="run1", journal_dir=tmp_path / "runs" / "run1" / "ssh-islands",
                        events_dir=tmp_path / "events", ops=cluster, clock=clock or Clock(),
                        sink_for=lambda name: sinks.append, log=lambda m: None, **kw)


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


NAMES = ["b300-0", "b300-1", "b300-2"]


# ------------------------------------------------------------------ start
def test_start_runs_incarnation_0_on_every_node_without_the_scheduler_control(tmp_path):
    cluster, lines = FakeCluster(NAMES), []
    ops = make_ops(tmp_path, cluster, lines=lines)
    ops.add("isl", spec(tmp_path=tmp_path))
    assert ops.start("isl") == 0
    assert [c[1] for c in cluster.calls if c[0] == "start"] == NAMES
    script = cluster.runs["b300-0"][0]["script"]
    assert "YETO_TRAINER_CONTROL" not in script and "YETO_TRAINER_INCARNATION" not in script
    assert "export YETO_X=1" in script and "export SKYPILOT_NODE_RANK=0" in script
    assert json.loads(ops.jobs_path("isl").read_text()) == {"0": "ssh:isl-i0"}
    assert wait_for(lambda: lines == ["started b300-0 i0\n"])  # node 0's stream reaches the sink
    assert str(ops.job_status("isl", 0)) == "RUNNING" and ops.cluster_up("isl")
    rows = ResourceLedger(tmp_path / "ledger.jsonl").open_resources()
    assert [(r.kind, r.id) for r in rows] == [("ssh_trainer", "run1/isl-i0")]
    assert [n["name"] for n in rows[0].attrs["nodes"]] == NAMES and rows[0].attrs["usd_per_hour"] == 3 * 8 * 2.0
    assert rows[0].attrs["gpus"] == 24 and rows[0].attrs["docker"] is True


# ------------------------------------------------------------------ the watch
def test_rank0_exit_0_is_success_and_down_releases_every_node(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.exit("b300-1", 0, 0)
    cluster.exit("b300-2", 0, 0)
    cluster.exit("b300-0", 0, 0)
    assert wait_for(lambda: all(p.poll() is not None for p in ops.islands["isl"].job.procs))
    assert str(ops.job_status("isl", 0)) == "SUCCEEDED"
    assert cluster.count("kill") == 0  # a finished island is released by the teardown
    assert ops.down("isl") is True
    assert cluster.count("kill", 0) == 3
    assert not ResourceLedger(tmp_path / "ledger.jsonl").open_resources()


@pytest.mark.parametrize("rank", [1, 2, 0])
def test_one_rank_exiting_non_zero_fails_the_island_and_stops_every_node(tmp_path, rank):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.exit(NAMES[rank], 0, 3)
    assert wait_for(lambda: ops.islands["isl"].job.procs[rank].poll() is not None)
    status = ops.job_status("isl", 0)
    assert str(status) == "FAILED" and status.is_terminal()
    assert f"rank {rank} ({NAMES[rank]}) exited with code 3" in ops.islands["isl"].reason
    assert cluster.count("kill", 0) == 3  # the whole island goes at once
    assert all(r[0]["state"] != "running" for r in cluster.runs.values())
    assert not ops.cluster_up("isl")


def test_a_process_gone_without_exit_code_fails_the_island(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.vanish("b300-2", 0)
    assert wait_for(lambda: ops.islands["isl"].job.procs[2].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert "gone without an exit code (lost" in ops.islands["isl"].reason


def test_a_node_that_stops_answering_fails_the_island_after_the_health_checks(tmp_path):
    cluster, clock = FakeCluster(NAMES), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=60.0, unreachable_polls=3)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.unreachable.add("b300-1")  # its stream hangs (no keepalive answer yet)
    for i in range(2):
        clock.t += 61
        assert str(ops.job_status("isl", 0)) == "RUNNING"
    clock.t += 61
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert "rank 1 (b300-1) does not answer over SSH (3 checks in a row)" in ops.islands["isl"].reason


def test_a_dropped_stream_is_followed_again_while_the_node_runs(tmp_path):
    cluster, lines = FakeCluster(NAMES), []
    ops = make_ops(tmp_path, cluster, lines=lines)
    ops.add("isl", spec())
    ops.start("isl")
    assert wait_for(lambda: lines)
    cluster.drop("b300-0")
    assert wait_for(lambda: ops.islands["isl"].job.procs[0].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "RUNNING"
    assert ("follow", "b300-0", 0) in cluster.calls
    cluster.say("b300-0", 0, event_echo.format_record({"event": "x", "time_unix": 1.0}) + "\n")
    assert wait_for(lambda: any(event_echo.PREFIX in line for line in lines))
    cluster.exit("b300-0", 0, 0)
    assert wait_for(lambda: ops.islands["isl"].job.procs[0].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "SUCCEEDED"


def test_a_worker_ending_cleanly_while_rank0_runs_fails_after_the_grace(tmp_path):
    cluster, clock = FakeCluster(NAMES), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, worker_exit_grace_s=120.0, health_every_s=0)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.exit("b300-2", 0, 0)
    assert wait_for(lambda: ops.islands["isl"].job.procs[2].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "RUNNING"
    clock.t += 121
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert "rank(s) [2] ended (exit 0) while rank 0 still runs" in ops.islands["isl"].reason


def test_strict_failure_markers_are_read_from_node0_stream(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.say("b300-0", 0, "[yeto-rl-strict-failure] bad layout\n")
    assert wait_for(lambda: ops.rl_strict_failure("isl", 0) == "[yeto-rl-strict-failure] bad layout")


# ------------------------------------------------------------------ relaunch / stop
def test_relaunch_releases_the_failed_incarnation_then_starts_the_next(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.exit("b300-1", 0, 1)
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert ops.relaunch(None, "isl") == 1
    assert cluster.count("start", 1) == 3
    assert str(ops.job_status("isl", 0)) == "FAILED"  # a stale job id
    assert str(ops.job_status("isl", 1)) == "RUNNING"
    assert json.loads(ops.jobs_path("isl").read_text()) == {"0": "ssh:isl-i0", "1": "ssh:isl-i1"}
    assert [r.id for r in ResourceLedger(tmp_path / "ledger.jsonl").open_resources()] == ["run1/isl-i1"]


def test_no_new_incarnation_while_the_old_one_is_not_confirmed_stopped(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.kill_rc["b300-2"] = 4  # its container survives docker rm -f
    assert ops.relaunch(None, "isl") is None
    assert cluster.count("start", 1) == 0
    assert ops.down("isl") is False  # the teardown reports it unverified
    cluster.kill_rc.clear()
    assert ops.down("isl") is True


def test_user_stop_releases_a_running_island(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    assert ops.down("isl") is True
    assert cluster.count("kill", 0) == 3
    assert all(r[0]["state"] == "exited" and r[0]["container"] == "none" for r in cluster.runs.values())
    assert ops.down("isl") is True and cluster.count("kill", 0) == 3  # idempotent, no second round


# ------------------------------------------------------------------ restarted launcher
def _crash(ops, name="isl"):
    """The launcher process dies: its lock goes, its streams end; the nodes run on."""
    rec = ops.islands[name]
    rec.lock.close()
    for proc in rec.job.procs:
        proc.terminate()


def test_a_restarted_launcher_reattaches_to_the_running_incarnation(tmp_path):
    cluster, lines = FakeCluster(NAMES), []
    first = make_ops(tmp_path, cluster)
    first.add("isl", spec())
    first.start("isl")
    cluster.say("b300-0", 0, event_echo.format_record({"event": "rl_learner_step", "time_unix": 1.0}) + "\n")
    _crash(first)
    again = make_ops(tmp_path, cluster, lines=lines)
    again.add("isl", spec())
    assert again.start("isl") == 0
    assert cluster.count("start") == 3 and cluster.count("follow", 0) == 3  # no second start
    assert wait_for(lambda: any("rl_learner_step" in line for line in lines))  # the log replays
    assert str(again.job_status("isl", 0)) == "RUNNING"
    cluster.exit("b300-0", 0, 0)
    assert wait_for(lambda: again.islands["isl"].job.procs[0].poll() is not None)
    assert str(again.job_status("isl", 0)) == "SUCCEEDED"


def test_a_restarted_launcher_follows_a_finished_incarnation_to_its_success(tmp_path):
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path, cluster)
    first.add("isl", spec())
    first.start("isl")
    _crash(first)
    for name in NAMES:
        cluster.exit(name, 0, 0)
    again = make_ops(tmp_path, cluster)
    again.add("isl", spec())
    assert again.start("isl") == 0 and cluster.count("start") == 3
    assert wait_for(lambda: all(p.poll() is not None for p in again.islands["isl"].job.procs))
    assert str(again.job_status("isl", 0)) == "SUCCEEDED"


def test_a_restarted_launcher_releases_a_broken_incarnation_and_starts_the_next(tmp_path):
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path, cluster)
    first.add("isl", spec())
    first.start("isl")
    _crash(first)
    cluster.vanish("b300-1", 0)
    again = make_ops(tmp_path, cluster)
    again.add("isl", spec())
    assert again.start("isl") == 1
    assert cluster.count("kill", 0) == 3 and cluster.count("start", 1) == 3
    assert cluster.runs["b300-0"][0]["state"] == "exited"  # the survivors of i0 were stopped first


def test_a_restarted_launcher_refuses_to_start_when_a_node_does_not_answer(tmp_path):
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path, cluster)
    first.add("isl", spec())
    first.start("isl")
    _crash(first)
    cluster.unreachable.add("b300-2")
    again = make_ops(tmp_path, cluster)
    again.add("isl", spec())
    with pytest.raises(SshIslandError, match=r"nodes \['b300-2'\] do not answer; refusing to start another copy"):
        again.start("isl")
    assert cluster.count("start") == 3 and cluster.count("kill") == 0


def test_a_second_launcher_of_the_same_island_is_refused(tmp_path):
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path, cluster)
    first.add("isl", spec())
    first.start("isl")
    second = make_ops(tmp_path, cluster)
    second.add("isl", spec())
    with pytest.raises(SshIslandError, match="supervised by another launcher"):
        second.start("isl")
    assert cluster.count("start") == 3


# ------------------------------------------------------------------ fleet controller
class _SkySide:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        time.sleep(0.01)
        self.t += s
        if self.t > 36000:
            raise AssertionError("controller did not end")


class _SyncThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()

    def join(self, timeout=None):
        pass


def _controller(ops, name="isl", **kw):
    return FleetController(learners={name: (ops.islands[name].spec, 0)}, syncer=None,
                           sky_ops=RoutingOps(_SkySide(), None, ssh_ops=ops), poll_interval=30.0,
                           recover_timeout=kw.pop("recover_timeout", 600.0), thread_cls=_SyncThread,
                           fixed_roster=True, **kw)


def test_the_fleet_relaunches_a_failed_rank_then_ends_with_the_trainer(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.exit("b300-2", 0, 1)  # incarnation 0: rank 2 crashes

    def finish_i1():
        if 1 in cluster.runs["b300-0"]:
            for name in NAMES:
                cluster.exit(name, 1, 0)

    ctl = _controller(ops, on_relaunch=lambda name, job: finish_i1())
    codes = ctl.run()
    assert ctl.learners["isl"]["state"] == DONE and "SUCCEEDED" in codes["isl"]
    assert cluster.count("start", 0) == 3 and cluster.count("start", 1) == 3
    assert cluster.count("kill", 0) == 3


def test_the_fleet_gives_up_on_an_island_failing_again_and_releases_it(tmp_path):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    cluster.exit("b300-1", 0, 1)

    def crash_again(name, job):
        cluster.exit("b300-1", int(job), 1)

    ctl = _controller(ops, on_relaunch=crash_again)
    with pytest.raises(FixedRosterIslandAbandoned):
        ctl.run()
    last = ops.islands["isl"].incarnation
    assert last >= 1 and all(cluster.count("kill", k) == 3 for k in range(last + 1))


def test_the_stop_flag_ends_the_fleet_and_down_stops_the_island(tmp_path):
    from yeto.launch.fleet import RunStopRequested

    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    flag = tmp_path / "STOP"
    flag.write_text("stop")
    with pytest.raises(RunStopRequested):
        _controller(ops, stop_flag=str(flag)).run()
    assert ops.down("isl") and cluster.count("kill", 0) == 3


# ------------------------------------------------------------------ node scripts (local bash)
def test_status_script_reports_running_exited_lost_and_absent(tmp_path):
    paths = remote_paths("r", "t", 0)
    state = tmp_path / ".yeto/trainer-islands/r/t-i0"

    def report():
        out = subprocess.run(["env", f"HOME={tmp_path}", "bash", "-c", status_script(paths)],
                             capture_output=True, text=True, check=True).stdout
        return parse_node_state(out)

    assert report().state == "absent"
    state.mkdir(parents=True)
    assert report().state == "lost"
    (state / "pgid").write_text(str(os.getpgid(0)))  # a live process group: this test's
    assert report().running
    (state / "rc").write_text("3\n")
    assert report() == ssh_resident.NodeState("exited", 3, "none")
    assert parse_node_state("garbage") is None


FAKE_DOCKER = """#!/bin/bash
# a docker CLI over one state file: absent / running / stuck (rm -f cannot remove it)
st=$(cat "$DOCKER_STATE" 2>/dev/null || echo absent)
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  container) [ "$st" != absent ] ;;
  exec) [ "$st" = running ] && [ "$STOP_ON_TERM" = 1 ] && echo exited > "$DOCKER_STATE"; exit 0 ;;
  inspect) [ "$st" = running ] && echo true || echo false ;;
  rm) [ "$st" = stuck ] || echo absent > "$DOCKER_STATE" ;;
esac
"""


@pytest.mark.parametrize("initial,stop_on_term,rc", [("running", "1", 0), ("running", "0", 0),
                                                    ("stuck", "0", 4), ("absent", "0", 0)])
def test_kill_script_stops_the_container_gracefully_and_verifies_it_is_gone(tmp_path, initial, stop_on_term, rc):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "docker").write_text(FAKE_DOCKER)
    (bindir / "docker").chmod(0o755)
    (tmp_path / "state").write_text(initial)
    env = {**os.environ, "HOME": str(tmp_path), "PATH": f"{bindir}:{os.environ['PATH']}",
           "DOCKER_STATE": str(tmp_path / "state"), "DOCKER_LOG": str(tmp_path / "log"),
           "STOP_ON_TERM": stop_on_term}
    script = kill_script(remote_paths("r", "t", 0), 2, container="yeto-r-t-i0-n0")
    done = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert done.returncode == rc, done.stderr
    calls = (tmp_path / "log").read_text().splitlines()
    if initial != "absent":
        assert calls[1] == "exec yeto-r-t-i0-n0 bash -c kill -TERM -1"  # the processes get TERM first
        assert "rm -f yeto-r-t-i0-n0" in calls
    if rc == 4:
        assert "still exists" in done.stderr


def test_shipped_files_decode_on_the_node_and_mount_read_only(tmp_path):
    s = spec(tmp_path=tmp_path)
    paths = remote_paths("run1", "isl", 0)
    script = start_script(s, 1, {}, paths, container=container_name("run1", "isl", 0, 1))
    assert ('/files/0:/root/yeto-rl-expert-selection.json:ro"' in script)
    from yeto.launch.ssh_trainer_island import _file_payloads, _pin_state_home

    snippet = _write_files(_file_payloads(s), _pin_state_home(paths))
    subprocess.run(["env", f"HOME={tmp_path}", "bash", "-c", f"__yeto_state_home=$HOME\n{snippet}"], check=True)
    shipped = tmp_path / ".yeto/trainer-islands/run1/isl-i0/files/0"
    assert shipped.read_text() == '{"schema": 1}' and (shipped.stat().st_mode & 0o777) == 0o600


def test_file_mounts_must_stay_under_home_or_be_absolute():
    with pytest.raises(SshIslandError, match="absolute or under ~/"):
        SshTrainerSpec(nodes=(b300(0),), run_id="r", workdir="/w", run_script="x", files={"rel/x": "/tmp/x"})


# ------------------------------------------------------------------ dry run, cleanup, cost
def test_rank_layout_of_6x8_pp3_puts_stage_s_on_nodes_2s_and_2s_plus_1():
    s = SshTrainerSpec(nodes=tuple(b300(i) for i in range(6)), run_id="r", workdir="/w", run_script="x")
    layout = rank_layout(s, tensor_parallel=8, pipeline_parallel=3, context_parallel=2)
    assert layout["world"] == 48 and not layout["warnings"]
    assert layout["pp_stage_nodes"] == {str(st): [f"b300-{2 * st}", f"b300-{2 * st + 1}"] for st in range(3)}
    assert [n["global_ranks"] for n in layout["nodes"]] == [f"{8 * i}-{8 * i + 7}" for i in range(6)]


def test_rank_layout_follows_the_node_ip_order_of_the_trainer():
    nodes = [b300(i, address=a) for i, a in enumerate(["198.51.100.9", "198.51.100.10", "198.51.100.2"])]
    s = SshTrainerSpec(nodes=tuple(nodes), run_id="r", workdir="/w", run_script="x")
    layout = rank_layout(s, tensor_parallel=8, pipeline_parallel=3)
    assert [n["trainer_position"] for n in layout["nodes"]] == [1, 2, 0]  # numeric, not string order
    assert layout["pp_stage_nodes"] == {"0": ["b300-2"], "1": ["b300-0"], "2": ["b300-1"]}
    assert "orders nodes by IP" in layout["warnings"][0]


def test_dry_run_record_prints_each_node_docker_run_without_secret_values(tmp_path):
    s = SshTrainerSpec(nodes=tuple(b300(i) for i in range(2)), run_id="run1", workdir="/opt/yeto",
                       run_script="echo hi", env={"YETO_RL_STORE_S3_SECRET_ACCESS_KEY": "s3cr3t"},
                       files={"~/sel.json": str(tmp_path / "missing.json")})
    rec = dry_run_record(SimpleNamespace(tensor_parallel=8, pipeline_parallel=2), s, "isl")
    run0 = rec["nodes"][0]["docker_run"]
    assert run0.startswith("docker run --rm --name yeto-run1-isl-i0-n0 --network host --ipc host --gpus all")
    assert "-e YETO_RL_STORE_S3_SECRET_ACCESS_KEY" in run0 and "s3cr3t" not in json.dumps(rec)
    assert "/files/0:/root/sel.json:ro" in run0 and "bash -c <island script: 7 bytes" in run0
    assert rec["rank_layout"]["pp_stage_nodes"] == {"0": ["b300-0"], "1": ["b300-1"]}
    assert any("missing.json is not a readable file" in w for w in rec["warnings"])
    assert any("workdir /opt/yeto is under no inventory mount" in w for w in rec["warnings"])


def test_cleanup_stops_an_island_whose_launcher_died_from_its_ledger_row(tmp_path):
    from yeto.cost import cleanup
    from yeto.launch.ssh_trainer_ledger import cleanup_release

    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec())
    ops.start("isl")
    _crash(ops)  # the launcher dies: the row stays open, the nodes run on
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")

    class Ops:
        ssh_ops = cluster

    actions = cleanup.cleanup(ledger, Ops())
    assert [a.ok for a in actions] == [True] and cluster.count("kill", 0) == 3
    assert not ledger.open_resources()
    assert all(r[0]["state"] == "exited" for r in cluster.runs.values())
    assert cleanup_release  # the ledger kind's own release (ssh_trainer_ledger)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_follow_script_is_valid_bash():
    from yeto.launch.ssh_trainer_island import follow_script

    subprocess.run(["bash", "-n"], input=follow_script(remote_paths("r", "t", 2)), text=True, check=True)


# ------------------------------------------------------------------ code, image and environment
FAKE_BOTO3 = '''
import io, json, os


class _Client:
    def __init__(self, kw):
        self.kw = kw

    def get_object(self, Bucket, Key):
        with open(os.environ["FAKE_S3_LOG"], "a") as fh:
            fh.write(json.dumps({"bucket": Bucket, "key": Key, **self.kw}) + "\\n")
        with open(os.path.join(os.environ["FAKE_S3_ROOT"], Bucket, Key), "rb") as fh:
            return {"Body": io.BytesIO(fh.read())}


class session:
    class Session:
        def __init__(self, **kw):
            self.kw = {k: v for k, v in kw.items() if k != "aws_secret_access_key"}

        def client(self, name, endpoint_url=None, region_name=None, config=None):
            return _Client({**self.kw, "endpoint": endpoint_url})
'''


def _tiny_pack(tmp_path):
    """A run code pack of a git tree that holds the pack's own unpacker."""
    from yeto import codepack

    repo = tmp_path / "repo"
    (repo / "yeto").mkdir(parents=True)
    src = Path(codepack.__file__).parent
    for name in ("__init__.py", "codepack.py"):
        (repo / "yeto" / name).write_text((src / name).read_text())
    git = ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "pack"], check=True)
    return codepack.build_pack(repo)


def test_every_node_fetches_the_run_code_pack_from_the_store_and_unpacks_it_verified(tmp_path):
    import sys

    from yeto import codepack
    from yeto.launch.ssh_trainer_island import PACK_FETCH_PY

    pack, data = _tiny_pack(tmp_path)
    store = tmp_path / "s3" / "bkt" / "runs" / "r1" / "code_packs"
    store.mkdir(parents=True)
    (store / f"{pack.digest}.tar").write_bytes(data)
    fake = tmp_path / "fake"
    (fake / "boto3").mkdir(parents=True)
    (fake / "boto3" / "__init__.py").write_text(FAKE_BOTO3)
    (fake / "botocore").mkdir()
    (fake / "botocore" / "__init__.py").write_text("")
    (fake / "botocore" / "config.py").write_text("class Config:\n    def __init__(self, **kw):\n        pass\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "PYTHONPATH": str(fake),
           "FAKE_S3_ROOT": str(tmp_path / "s3"), "FAKE_S3_LOG": str(tmp_path / "s3.log"),
           "YETO_RL_STORE_S3_ACCESS_KEY_ID": "kid", "YETO_RL_STORE_S3_SECRET_ACCESS_KEY": "secret",
           "YETO_RL_STORE_S3_ENDPOINT_URL": "https://r2.example"}
    uri = f"s3://bkt/runs/r1/code_packs/{pack.digest}.tar"
    argv = [sys.executable, "-c", PACK_FETCH_PY, uri, pack.digest, "~/.cache/yeto/codepacks"]
    done = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    tree = tmp_path / "home/.cache/yeto/codepacks" / pack.digest
    assert codepack.verify_tree(tree, pack.digest) == pack.digest  # what the island checks
    (log,) = [json.loads(line) for line in (tmp_path / "s3.log").read_text().splitlines()]
    assert log == {"bucket": "bkt", "key": f"runs/r1/code_packs/{pack.digest}.tar", "aws_access_key_id": "kid",
                   "aws_session_token": None, "endpoint": "https://r2.example"}
    (store / f"{pack.digest}.tar").unlink()
    assert subprocess.run(argv, env=env).returncode == 0  # unpacked already: no second download
    bad = subprocess.run([*argv[:4], "0" * 64, argv[5]], env={**env, "HOME": str(tmp_path / "h2")},
                         capture_output=True, text=True)
    assert bad.returncode != 0


def test_the_node_job_links_sky_workdir_to_the_pack_and_starts_in_the_container_home():
    s = SshTrainerSpec(nodes=(b300(0), b300(1)), run_id="r", workdir="", run_script="cd ~/sky_workdir && run",
                       code_pack=("s3://bkt/runs/r/code_packs/" + "c" * 64 + ".tar", "c" * 64))
    script = start_script(s, 1, {}, remote_paths("r", "t", 0), container="c1")
    assert "-w /root " in script and "python3 -I -c" in script
    from yeto.launch.ssh_trainer_island import _code_link

    link = _code_link(s)
    assert f'ln -sfn "$HOME/.cache/yeto/codepacks/{"c" * 64}" "$HOME/sky_workdir"' in link
    subprocess.run(["bash", "-n"], input=link, text=True, check=True)
    with pytest.raises(SshIslandError, match="workdir .* or a run code pack"):
        SshTrainerSpec(nodes=(b300(0),), run_id="r", workdir="", run_script="x")


def _code_args(**kw):
    base = dict(training_mode="rl", rl_code_pack="auto", rl_checkpoint_store="s3://bkt/runs/r",
                rl_ssh_trainer_workdir=None, code_pack=None, rl_image="docker:ghcr.io/o/i@sha256:" + "a" * 64)
    return SimpleNamespace(**(base | kw))


def test_code_source_prefers_the_run_code_pack_then_the_release_image_then_a_workdir():
    from yeto.codepack import CodePack
    from yeto.launch.ssh_trainer_island import code_source, island_code_pack

    assert code_source(_code_args()) == "code-pack"
    assert island_code_pack(_code_args()) == ("s3://bkt/runs/r/code_packs/<digest>.tar", "<digest>")  # planned
    stored = CodePack(commit="x", dirty=False, patch_sha="p", digest="d" * 64, pack_uri="s3://bkt/runs/r/cp.tar")
    assert island_code_pack(_code_args(code_pack=stored)) == ("s3://bkt/runs/r/cp.tar", "d" * 64)
    assert code_source(_code_args(rl_code_pack="release")) == "release"
    assert code_source(_code_args(rl_checkpoint_store="oss://b/r")) is None  # auto packs, but not to s3
    assert code_source(_code_args(rl_code_pack="off", rl_ssh_trainer_workdir="/opt/yeto")) == "workdir"


def test_the_trainer_image_is_the_run_image_and_an_inventory_image_must_agree():
    from yeto.launch.ssh_trainer_island import run_image

    ref = "ghcr.io/o/i@sha256:" + "a" * 64
    assert run_image(_code_args(), [b300(0, image=None), b300(1, image=None)]) == ref
    assert run_image(_code_args(), [b300(0, image=ref)]) == ref
    with pytest.raises(SshIslandError, match=r"nodes \['b300-1'\] is not the run's --rl-image"):
        run_image(_code_args(), [b300(0, image=ref), b300(1, image="ghcr.io/o/other:1")])
    assert run_image(_code_args(), [b300(0, runtime="rootfs", docker_args=())]) is None


def test_an_overlay_rejected_pinned_image_runs_as_its_repack_on_ssh_nodes():
    """The K3.1 full trainer image of the pins (2de5d796) has a layer Docker 29 cannot
    unpack; an ssh island runs its overlay-safe repack (same files and Miles commit),
    whether the run names it by --rl-image or only the inventory does."""
    from yeto.launch.ssh_trainer_island import run_image
    from yeto.rl.adapters.miles import pins

    old, new = next(iter(pins.OVERLAY_SAFE_DIGESTS.items()))
    rejected, repack = f"ghcr.io/o/yeto-miles-k31@sha256:{old}", f"ghcr.io/o/yeto-miles-k31@sha256:{new}"
    args = _code_args(rl_image=f"docker:{rejected}")
    assert run_image(args, [b300(0, image=None), b300(1, image=None)]) == repack
    # an inventory may name either digest of the same image
    assert run_image(args, [b300(0, image=rejected), b300(1, image=repack)]) == repack
    with pytest.raises(SshIslandError, match="is not the run's --rl-image"):
        run_image(args, [b300(0, image="ghcr.io/o/other@sha256:" + "b" * 64)])
    # no --rl-image: the inventory's one image is mapped the same way
    assert run_image(_code_args(rl_image=None), [b300(0, image=rejected), b300(1, image=rejected)]) == repack
    assert run_image(_code_args(rl_image=None), [b300(0, image="ghcr.io/o/i:1")]) is None
    # the repack keeps the Miles commit the pins check
    assert pins.image_miles_commit(repack) == pins.image_miles_commit(rejected) is not None


def test_check_refuses_an_inventory_that_does_not_match_the_island_before_any_ssh(tmp_path):
    from yeto.gpu_spec import parse_gpu_spec

    args = _code_args(rl_node_inventory_nodes=[b300(i, image=None) for i in range(2)], rl_trainer_scheduler="off",
                      rl_trainer_island=None, rl_node_class=[], rl_island_sample_role=[], rl_sample_role="trainer")
    ssh_resident.check(args, parse_gpu_spec("ssh:2x8xb300@bm"))
    with pytest.raises(SystemExit, match="has 3 node"):
        ssh_resident.check(args, parse_gpu_spec("ssh:3x8xb300@bm"))
    with pytest.raises(SystemExit, match="code cannot reach"):
        ssh_resident.check(_code_args(**{**vars(args), "rl_code_pack": "off"}), parse_gpu_spec("ssh:2x8xb300@bm"))


def test_island_env_reaches_every_island_and_modal_env_still_comes_last():
    from yeto.launch.modal_islands import parse_island_env, parse_modal_env

    args = SimpleNamespace(island_env=["MILES_K31_INDEXER_QUERY_CHUNK=1024", "YETO_MILES_TRAIN_EXTRA_ARGV=--a b=c"])
    assert parse_island_env(args) == {"MILES_K31_INDEXER_QUERY_CHUNK": "1024", "YETO_MILES_TRAIN_EXTRA_ARGV": "--a b=c"}
    with pytest.raises(ValueError, match="--island-env takes KEY=VALUE; item 1"):
        parse_island_env(SimpleNamespace(island_env=["novalue"]))
    with pytest.raises(ValueError, match="not an environment variable name"):
        parse_island_env(SimpleNamespace(island_env=["BAD-NAME=1"]))
    assert parse_modal_env(["A=1"]) == {"A": "1"}
