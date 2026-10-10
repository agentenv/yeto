"""rl-spot-cost-saving phase 2 (tasks 5.1 + code of 5.x) on CPU: one on-demand anchor
island + droppable spot islands."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.cloud import droppable as dr
from yeto.cloud import preemption as pre
from yeto.rl.engine.island_ledger import CrossIslandLedger, DeltaEntry


def _args(**kw):
    base = dict(training_mode="rl", rl_island_scheduling="elastic", spot=False, rl_q_min=None,
                rl_checkpoint_store=None, rl_island_role=["1:droppable", "2:droppable"])
    base.update(kw)
    return SimpleNamespace(**base)


# --- 5.1 admission -------------------------------------------------------------------------


def test_roles_default_anchor_and_manifest():
    roles = dr.admit(_args(), ["nebius", "aws", "modal"])
    assert roles == {0: "anchor", 1: "droppable", 2: "droppable"}
    m = dr.manifest_fields(roles)["island_roles"]
    assert m["0"] == {"role": "anchor", "billing": "on_demand", "rejoin": "checkpoint_store"}
    assert m["1"] == {"role": "droppable", "billing": "spot", "rejoin": "syncer_base_from_anchor"}


def test_no_roles_is_a_noop():
    assert dr.admit(_args(rl_island_role=None), ["aws"]) == {}
    assert dr.manifest_fields({}) == {}


def test_legacy_refused_with_spec_reason():
    with pytest.raises(ValueError, match="可丢弃岛只在 elastic 下支持"):
        dr.admit(_args(rl_island_scheduling=None), ["nebius", "aws", "aws"])
    with pytest.raises(ValueError, match="可丢弃岛只在 elastic 下支持"):
        dr.admit(_args(rl_island_scheduling="legacy"), ["nebius", "aws", "aws"])


def test_needs_an_anchor():
    with pytest.raises(ValueError, match="at least one anchor"):
        dr.admit(_args(rl_island_role=["0:droppable", "1:droppable"]), ["aws", "aws"])


def test_modal_cannot_be_anchor():
    with pytest.raises(ValueError, match="always preemptible"):
        dr.admit(_args(rl_island_role=["1:droppable"]), ["modal", "aws"])


def test_global_spot_refused():
    with pytest.raises(ValueError, match="drop --spot"):
        dr.admit(_args(spot=True), ["nebius", "aws", "aws"])


def test_q_min_must_fit_the_anchors():
    with pytest.raises(ValueError, match="--rl-q-min 2 > 1 anchor"):
        dr.admit(_args(rl_q_min=2), ["nebius", "aws", "aws"])
    assert dr.admit(_args(rl_q_min=2, rl_island_role=["2:droppable"]), ["nebius", "nebius", "aws"])


def test_verda_anchor_cut_not_on_local_volume():
    with pytest.raises(ValueError, match="Verda"):
        dr.admit(_args(rl_checkpoint_store="/mnt/ckpt", rl_island_role=["1:droppable"]),
                 ["verda", "aws"])
    roles = dr.admit(_args(rl_checkpoint_store="s3://b/run", rl_island_role=["1:droppable"]),
                     ["verda", "aws"])
    assert roles[0] == "anchor"


def test_droppable_on_unknown_cloud_refused():
    with pytest.raises(ValueError, match="spot support unknown"):
        dr.admit(_args(rl_island_role=["1:droppable"]), ["nebius", "nowhere"])


@pytest.mark.parametrize("item", ["x:droppable", "5:droppable", "1:train", "1"])
def test_bad_items(item):
    with pytest.raises(ValueError):
        dr.parse_roles([item], 3)


def test_island_given_twice():
    with pytest.raises(ValueError, match="twice"):
        dr.parse_roles(["1:anchor", "1:droppable"], 3)


# --- per-island args: billing and no archive resume ----------------------------------------


def test_role_args_droppable_is_spot_without_store():
    args = _args(rl_checkpoint_store="s3://b/run")
    d = dr.role_args(args, "droppable")
    assert d.spot is True and d.rl_checkpoint_store is None
    assert not dr.island_keeps_spot_volume(d)  # no per-island completed-groups volume
    a = dr.role_args(args, "anchor")
    assert a.spot is False and a.rl_checkpoint_store == "s3://b/run"
    assert dr.role_args(args, None) is args
    assert args.rl_checkpoint_store == "s3://b/run" and args.spot is False  # original untouched
    assert dr.island_keeps_spot_volume(SimpleNamespace(spot=True))  # legacy --spot unchanged


def test_role_env_roundtrip():
    env = dr.role_env("droppable", "aws", "us-east-1")
    doc = dr.env_role(env)
    assert doc == {"role": "droppable", "cloud": "aws", "region": "us-east-1"}
    assert dr.role_env(None, "aws", None) == {}
    assert dr.env_role({}) is None and dr.env_role({dr.ROLE_ENV: "nope"}) is None


def test_island_task_carries_role(monkeypatch):
    from yeto import launcher

    seen = {}

    class Task:
        envs: dict = {}

        def update_envs(self, env):
            self.envs = {**self.envs, **env}

    def factory(iargs, spec, m, n, addr):
        seen["args"] = iargs
        return Task()

    spec = SimpleNamespace(cloud="aws", region="us-east-2")
    args = _args(rl_checkpoint_store="s3://b/run", rl_negative_test_run=False)
    task = launcher._island_task(factory, args, spec, 1, 3, "x:1", {}, "droppable")
    assert seen["args"].spot is True and seen["args"].rl_checkpoint_store is None
    assert json.loads(task.envs[dr.ROLE_ENV])["role"] == "droppable"
    task = launcher._island_task(factory, args, spec, 0, 3, "x:1", {}, None)
    assert seen["args"] is args and dr.ROLE_ENV not in task.envs


# --- reclaim: event role field (#180), LEAVE, nothing saved ---------------------------------


def test_aws_droppable_reclaim_leaves_and_writes_role():
    events, left = [], []
    env = dr.role_env("droppable", "aws", "us-east-1")
    poller = dr.install_reclaim_listener("1", leave=lambda: left.append(1) or True,
                                         emit=lambda e, **f: events.append((e, f)),
                                         environ=env, start=False)
    assert isinstance(poller, pre.AwsMetadataPoller)
    poller.fetch = lambda: (200, json.dumps({"action": "terminate"}))
    poller.poll_once()
    assert left == [1]
    name, ev = events[0]
    assert name == pre.RECLAIM_EVENT and ev["role"] == "droppable" and ev["billing"] == "spot"
    assert ev["saved"] is False and ev["leave_confirmed"] is True


def test_modal_droppable_reclaim_handler():
    events = []
    env = dr.role_env("droppable", "modal", None)
    h = dr.install_reclaim_listener("2", leave=lambda: True, emit=lambda e, **f: events.append(f),
                                    environ=env, start=False)
    assert isinstance(h, pre.ModalExitHandler)
    ev = h.handle(source="test")
    assert ev["role"] == "droppable" and ev["cloud"] == "modal" and ev["leave_confirmed"] is True


def test_listener_only_on_droppable():
    noop = dict(leave=lambda: True, emit=lambda *a, **k: None, start=False)
    assert dr.install_reclaim_listener("0", environ=dr.role_env("anchor", "aws", None), **noop) is None
    assert dr.install_reclaim_listener("0", environ={}, **noop) is None
    # Verda droppable: no notice path (lease expiry); Nebius: see test_nebius_reclaim.py
    assert dr.install_reclaim_listener("1", environ=dr.role_env("droppable", "verda", None),
                                       **noop) is None


def test_entry_wiring_noop_without_role():
    from yeto.rl.adapters.miles.entry import _wire_droppable_reclaim

    assert _wire_droppable_reclaim(object(), SimpleNamespace(emit=None), environ={}) is None


# --- every droppable island reclaimed at once: the anchor alone keeps training ---------------


def _delta(island, led):
    return DeltaEntry(island, led.outer_version, 4, led.published[led.outer_version], 100, 4)


def test_all_droppable_reclaimed_anchor_keeps_rounds_going():
    """Python mirror of the syncer ledger (the Rust syncer applies the same P4 rule)."""
    led = CrossIslandLedger(mode="elastic", theta=0.75, quorum_min=1)
    for i in ("anchor", "d1", "d2"):
        led.join(i, now=0.0)
    for i in ("anchor", "d1", "d2"):
        led.submit(_delta(i, led))
    assert led.try_advance(timed_out=False) is not None
    v = led.outer_version
    # both spot islands reclaimed with notice -> LEAVE
    led.leave("d1", reason="spot_reclaim")
    led.leave("d2", reason="spot_reclaim")
    for _ in range(3):
        led.submit(_delta("anchor", led))
        assert led.try_advance(timed_out=False) is not None
    assert led.outer_version == v + 3


def test_all_droppable_vanish_without_notice_round_closes_on_timeout():
    led = CrossIslandLedger(mode="elastic", theta=0.75, quorum_min=1)
    for i in ("anchor", "d1", "d2"):
        led.join(i, now=0.0)
    led.submit(_delta("anchor", led))
    assert led.try_advance(timed_out=False) is None   # 1/3 < theta, still waiting
    step = led.try_advance(timed_out=True)            # soft deadline: anchor alone is enough
    assert step is not None and sorted(step["absent"]) == ["d1", "d2"]


def test_droppable_rejoin_takes_the_current_base_with_zero_weight():
    led = CrossIslandLedger(mode="elastic", theta=0.5, quorum_min=1)
    led.join("anchor", now=0.0)
    for _ in range(2):
        led.submit(_delta("anchor", led))
        led.try_advance(timed_out=False)
    ev = led.join("d1", now=1.0)
    assert ev["catch_up"] is True  # syncer sends the base the anchor advanced
    d = _delta("d1", led)
    assert led.weight_of(d) == 0.0  # first round after JOIN on the current base: no weight


# --- 5.2 CPU part: after a reclaim LEAVE the still-running island must not rejoin -----------


def test_client_after_leave_never_rejoins_on_not_member_errors():
    """The syncer answers a left island's delta/heartbeat with "not a member
    (rejoin required)". The droppable island LEAVEs on reclaim and is killed by
    the cloud later; until then it must not JOIN again by itself (the rejoin
    comes from the relaunched island, catch_up from the syncer base)."""
    from yeto.rl.elastic_client import JoinAck, ElasticClientConfig, ElasticIslandClient

    sent = []
    c = ElasticIslandClient(ElasticClientConfig(syncer_addr=("x", 1), island_id=2), b"k")
    c.ack = JoinAck(1, 2, 3, 1, bytes(32), False)
    c.sock = object()
    c._send = sent.append
    c.leave()
    assert type(sent[-1]).__name__ == "Leave"
    c._on_error("2 is not a member (rejoin required)")
    assert not c._rejoin.is_set()
    assert c.rejoin() is True and c.rejoins == 0
    c.check()
    assert [type(m).__name__ for m in sent] == ["Leave"]


# --- 5.2: the Modal reclaim signal reaches the learner (grandchild) ---------------------


def test_island_script_forwards_reclaim_signal_to_group_on_droppable_only():
    import json as _json
    import signal as _signal

    from yeto import modal_runner as mr

    class P:
        pid = 4242

        def __init__(self, cmd, env, **kw):
            self.kw = kw

        def wait(self):
            return 0

    made, handlers, killed = [], {}, []

    def popen(cmd, env, **kw):
        made.append(P(cmd, env, **kw))
        return made[-1]

    class Sig:
        SIGINT, SIGTERM = _signal.SIGINT, _signal.SIGTERM

        @staticmethod
        def signal(num, fn):
            handlers[num] = fn

    env = {"YETO_ISLAND_ROLE": _json.dumps({"role": "droppable", "cloud": "modal"})}
    assert mr.run_island_script(["x"], env, popen=popen, signal_mod=Sig,
                                killpg=lambda pid, n: killed.append((pid, n))) == 0
    assert made[-1].kw == {"start_new_session": True}
    handlers[_signal.SIGINT](_signal.SIGINT, None)
    handlers[_signal.SIGTERM](_signal.SIGTERM, None)
    assert killed == [(4242, _signal.SIGINT), (4242, _signal.SIGTERM)]

    handlers.clear()
    anchor = {"YETO_ISLAND_ROLE": _json.dumps({"role": "anchor", "cloud": "nebius"})}
    assert mr.run_island_script(["x"], anchor, popen=popen, signal_mod=Sig) == 0
    assert made[-1].kw == {} and handlers == {}
    assert mr.run_island_script(["x"], {}, popen=popen, signal_mod=Sig) == 0 and handlers == {}
