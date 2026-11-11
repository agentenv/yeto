"""A sampler outlives its trainer: with no trainer reading (backlog at the cap) it
waits, never exits, and keeps its event tape moving so the NC pool's stall check
does not restart or replace it."""

from __future__ import annotations

from types import SimpleNamespace

from yeto.launch.pool import status as el
from yeto.rl.engine import sampler_loop as sl


class Follower:
    def __init__(self):
        self.last_seen = SimpleNamespace(buffer_shards=None, buffer_selected=0, seq=1, policy_hash="h",
                                         released_at=0.0)
        self.version = 3
        self.release = object()
        self.releases = SimpleNamespace()
        self.live_update = False

    def refresh(self):
        return None


class Buffer:
    def __init__(self, n):
        self.n = n

    def refresh(self):
        pass

    def __len__(self):
        return self.n


def test_a_sampler_with_no_trainer_waits_and_heartbeats(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(sl.time, "monotonic", lambda: now[0])
    events, generated = [], []
    sleeps = [0]

    def sleep(s):
        now[0] += s
        sleeps[0] += 1

    loop = sl.SamplerLoop(follower=Follower(), buffer=Buffer(10), generate=generated.append, num_rollouts=None,
                          max_backlog_groups=4, emit=lambda name, **f: events.append((name, f)), poll_s=10.0,
                          sleep=sleep, should_stop=lambda: sleeps[0] >= 90)  # 15 min without a trainer
    assert loop.run(0) == 0 and generated == []
    names = [n for n, _ in events]
    assert names.count("rl_sampler_backlog_full") == 1
    beats = [f for n, f in events if n == "rl_sampler_waiting"]
    assert len(beats) >= 14 and beats[-1]["reason"] == "backlog_full" and beats[-1]["backlog_groups"] == 10
    assert names[-1] == "rl_sampler_stopped"  # only should_stop ends it


def test_a_heartbeating_waiting_sampler_is_healthy_for_the_pool():
    """The pool sees a fresh tape (heartbeat) on a running container: healthy, not crashed."""
    cfg = el.ElasticConfig(stall_s=600)
    probe = el.Probe(node=el.PROBE_OK, sampler="running", exit_code=None, joined=True, progress_age_s=60)
    verdict = el.classify(el.LeaseStatus("active", 1e12), probe, now=0.0, unreachable_since=None, cfg=cfg,
                          joined=True)
    assert verdict.kind == el.HEALTHY
