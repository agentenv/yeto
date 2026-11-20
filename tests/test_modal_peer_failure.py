"""A failed rank>0 container of a multi-node Modal island fails the island at once.

mfu16 a0 (2026-11-18): rank 1's run script exited 1 at 13:20:45 (model fetch), rank 0
kept starting (Ray head, a 16-GPU placement group) and the island's Modal call (rank
0's) ended only when an outside guard stopped the app at 13:22:53; unguarded, the
2400 s Modal timeout. Rank 1 now reports to rank 0's PeerFailureRelay, which stops
rank 0's island script so rank 0 raises with the peer's reason."""

from __future__ import annotations

import signal
import sys
import threading
import types

import pytest

from yeto import modal_runner as mr


def _cfg(**kw) -> mr.ModalIslandConfig:
    base = dict(app_name="yeto-run", learner_id=0, training_mode="rl", gpu="B300", gpus_per_node=8,
                num_nodes=2, run_script="echo run", envs={"SYNCER_ADDR": "none"})
    base.update(kw)
    return mr.ModalIslandConfig(**base)


def _cluster(monkeypatch, rank: int):
    modal = types.ModuleType("modal")
    modal.get_cluster_info = lambda: types.SimpleNamespace(
        rank=rank, container_ips=["fdaa::1", "fdaa::2"], container_ipv4_ips=["127.0.0.1", "127.0.0.2"])
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setattr(mr, "visible_gpu_names", lambda: ["NVIDIA B300"] * 8)
    monkeypatch.setattr(mr, "narrow_ephemeral_ports", lambda: "kept")
    monkeypatch.setattr(mr, "hostmem_interval_s", lambda cfg: None)  # no sampler thread writing /root


def test_relay_records_the_first_failure_and_stops_the_island_once():
    stops = []
    relay = mr.PeerFailureRelay(port=0, host="127.0.0.1", stop=lambda: stops.append(1)).start()
    try:
        assert mr.send_peer_failure("127.0.0.1", relay.port, 1, "island 0 rank 1 exited with 1",
                                    attempts=1, sleep=lambda s: None)
        assert mr.send_peer_failure("127.0.0.1", relay.port, 2, "later", attempts=1, sleep=lambda s: None)
    finally:
        relay.stop()
    assert relay.failure == {"rank": 1, "reason": "island 0 rank 1 exited with 1"} and stops == [1]
    assert relay.failure_message(0) == ("island 0 rank 0 stopped: peer rank 1 failed: "
                                        "island 0 rank 1 exited with 1")


def test_send_gives_up_when_rank0_never_listens():
    sleeps = []
    assert not mr.send_peer_failure("127.0.0.1", 1, 1, "x", timeout_s=0.2, attempts=3, sleep=sleeps.append)
    assert sleeps == [2.0, 2.0]  # no pause after the last attempt


def test_failed_worker_rank_reports_to_rank0(monkeypatch):
    _cluster(monkeypatch, rank=1)
    sent = []
    monkeypatch.setattr(mr, "send_peer_failure", lambda host, port, rank, reason: sent.append((host, port, rank, reason)))
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 1)
    with pytest.raises(RuntimeError, match="rank 1 exited with 1"):
        mr.island_main(_cfg().to_json())
    assert sent == [("127.0.0.1", mr.PEER_FAILURE_PORT_DEFAULT, 1, "island 0 rank 1 exited with 1")]


def test_worker_rank_gpu_check_failure_is_reported_too(monkeypatch):
    _cluster(monkeypatch, rank=1)
    monkeypatch.setattr(mr, "visible_gpu_names", lambda: ["NVIDIA H100"] * 8)
    sent = []
    monkeypatch.setattr(mr, "send_peer_failure", lambda *a: sent.append(a))
    with pytest.raises(RuntimeError):
        mr.island_main(_cfg(gpu_exact=True).to_json())
    assert len(sent) == 1 and sent[0][2] == 1


def test_successful_worker_and_single_node_report_nothing(monkeypatch):
    _cluster(monkeypatch, rank=1)
    sent = []
    monkeypatch.setattr(mr, "send_peer_failure", lambda *a: sent.append(a))
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 0)
    assert mr.island_main(_cfg().to_json()) == 0
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 2)
    with pytest.raises(RuntimeError, match="exited with 2"):
        mr.island_main(_cfg(num_nodes=1).to_json())
    assert sent == []


def test_rank0_stops_and_raises_with_the_peer_reason(monkeypatch):
    _cluster(monkeypatch, rank=0)
    relays = []

    class Relay(mr.PeerFailureRelay):
        def __init__(self, *, port):
            super().__init__(port=0, host="127.0.0.1")
            relays.append(self)

    stopped = threading.Event()
    monkeypatch.setattr(mr, "PeerFailureRelay", Relay)
    monkeypatch.setattr(mr, "stop_children", stopped.set)

    def run_script(cmd, env=None):
        # the learner is still starting when rank 1 reports; the relay stops it
        assert mr.send_peer_failure("127.0.0.1", relays[0].port, 1, "island 0 rank 1 exited with 1",
                                    attempts=1, sleep=lambda s: None)
        assert stopped.wait(5)
        return -signal.SIGTERM

    monkeypatch.setattr(mr.subprocess, "call", run_script)
    with pytest.raises(RuntimeError, match="rank 0 stopped: peer rank 1 failed: island 0 rank 1 exited with 1"):
        mr.island_main(_cfg().to_json())
    assert relays[0]._stop.is_set()  # the relay closes with the island


def test_rank0_refuses_to_start_its_script_after_a_peer_failed(monkeypatch):
    _cluster(monkeypatch, rank=0)

    class Relay(mr.PeerFailureRelay):
        def __init__(self, *, port):
            super().__init__(port=0, host="127.0.0.1", stop=lambda: None)
            self.failure = {"rank": 1, "reason": "setup exited with 7"}

    monkeypatch.setattr(mr, "PeerFailureRelay", Relay)
    calls = []
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)
    with pytest.raises(RuntimeError, match="peer rank 1 failed: setup exited with 7"):
        mr.island_main(_cfg(setup_script="echo setup").to_json())
    assert calls == []


def test_child_pids_walks_the_process_tree(tmp_path):
    def proc(pid, ppid, name="x"):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} ({name}) S {ppid} 1 1 0\n")

    proc(10, 1)
    proc(11, 10, "bash -lc (x)")  # a ')' in the name
    proc(12, 11)
    proc(13, 12)
    proc(20, 1)
    (tmp_path / "self").mkdir()
    assert sorted(mr.child_pids(10, str(tmp_path))) == [11, 12, 13]
    assert mr.child_pids(10, str(tmp_path / "missing")) == []


def test_stop_children_terms_then_kills_what_is_left():
    alive = {11, 12}
    sent = []

    def kill(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGTERM and pid == 11:
            alive.discard(11)
        if sig == signal.SIGKILL:
            alive.discard(pid)

    clock = iter(range(1000))
    first = mr.stop_children(grace_s=0.0, pids=lambda: sorted(alive), kill=kill, sleep=lambda s: next(clock))
    assert first == [11, 12]
    assert sent == [(11, signal.SIGTERM), (12, signal.SIGTERM), (12, signal.SIGKILL)]
