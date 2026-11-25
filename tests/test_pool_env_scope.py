"""The run config's ``env`` around the provider calls of a pool driven in the ``yeto run
up`` process (``controller: "in_process"``), yeto.launch.pool.env_scope: the provider sees
the run's variables, and the process environment outside a pool call stays the shell's."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import pytest

from tests.test_rollout_entry import _NoThread, _two_pools
from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute
from yeto.launch.pool import controller as ctl
from yeto.launch.pool.env_scope import EnvOverlayConflict, EnvScopedOps, scoped_environ
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.rl import up

WATCHED = ("AWS_PROFILE", "YETO_RL_STORE_S3_PROFILE")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setenv("AWS_PROFILE", "shell")
    monkeypatch.delenv("YETO_RL_STORE_S3_PROFILE", raising=False)

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


class SeeingProvider(FakeProvider):
    """Records the process environment each order and release saw."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen: list[tuple[str, dict]] = []

    def _see(self, what):
        self.seen.append((what, {k: os.environ.get(k) for k in WATCHED}))

    def order(self, slot):
        self._see("order")
        return super().order(slot)

    def release(self, slot, ticket):
        self._see("release")
        return super().release(slot, ticket)


def test_in_process_pool_provider_calls_see_the_run_config_env(tmp_path):
    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process", env={"AWS_PROFILE": "x", "YETO_RL_STORE_S3_PROFILE": "x-store"})
    clock = Clock()
    providers = {name: SeeingProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        c.tick()
        # outside a pool call this process keeps the shell's environment
        assert os.environ.get("AWS_PROFILE") == "shell" and "YETO_RL_STORE_S3_PROFILE" not in os.environ
    finally:
        c.stop("test")
    assert c not in ctl.active()
    for p in providers.values():
        assert [w for w, _ in p.seen] == ["order", "release"]
        assert all(env == {"AWS_PROFILE": "x", "YETO_RL_STORE_S3_PROFILE": "x-store"} for _, env in p.seen), p.seen
    assert os.environ.get("AWS_PROFILE") == "shell" and "YETO_RL_STORE_S3_PROFILE" not in os.environ


def test_in_process_pool_without_a_run_env_sees_the_shell(tmp_path):
    run_cfg, _, _ = _two_pools(tmp_path)
    clock = Clock()
    provider = SeeingProvider("nc", clock)
    ops = {"nc": ProviderPoolOps(provider, FakeRoute(), FakeFencer()),
           "modal": ProviderPoolOps(FakeProvider("modal", clock), FakeRoute(), FakeFencer())}
    c = up.Run(dict(run_cfg, controller="in_process")).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        c.tick()
    finally:
        c.stop("test")
    assert provider.seen and all(env == {"AWS_PROFILE": "shell", "YETO_RL_STORE_S3_PROFILE": None}
                                 for _, env in provider.seen)


def test_pool_environ_is_what_a_pool_process_would_differ_in(tmp_path, monkeypatch):
    run_cfg, _, _ = _two_pools(tmp_path)
    monkeypatch.setenv("YETO_RL_AUTO_BACKLOG_GROUPS", "99")  # inherited: run up drops it from the pools' env
    run = up.Run(dict(run_cfg, env={"AWS_PROFILE": "x", "YETO_NC_ACCOUNTS": "1,2,3,4"}))
    overlay = run.pool_environ()
    assert overlay["AWS_PROFILE"] == "x" and "YETO_NC_ACCOUNTS" not in overlay  # unchanged: not in the overlay
    assert overlay["YETO_RL_AUTO_BACKLOG_GROUPS"] is None
    assert {k: v for k, v in overlay.items() if v is not None} == {
        k: v for k, v in run.env.items() if os.environ.get(k) != v}


def test_scoped_environ_restores_and_counts_overlapping_scopes(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "shell")
    monkeypatch.setenv("GONE", "here")
    overlay = {"AWS_PROFILE": "x", "NEW": "n", "GONE": None}
    with scoped_environ(overlay):
        assert os.environ["AWS_PROFILE"] == "x" and os.environ["NEW"] == "n" and "GONE" not in os.environ
        with scoped_environ(overlay):  # nested: applied once
            pass
        assert os.environ["AWS_PROFILE"] == "x"  # the outer scope still holds it
    assert os.environ["AWS_PROFILE"] == "shell" and "NEW" not in os.environ and os.environ["GONE"] == "here"
    with scoped_environ(None), scoped_environ({}):
        assert os.environ["AWS_PROFILE"] == "shell"


def test_scoped_environ_holds_while_a_background_call_runs(monkeypatch):
    """A long call on another thread (a background sampler start) keeps the overlay after
    the tick that started it has left its scope; the last one out restores the shell's."""
    monkeypatch.setenv("AWS_PROFILE", "shell")
    entered, release, seen = threading.Event(), threading.Event(), []

    class Ops:
        start_in_background = True

        def start(self):
            entered.set()
            release.wait(5)
            seen.append(os.environ.get("AWS_PROFILE"))

    ops = EnvScopedOps(Ops(), {"AWS_PROFILE": "x"})
    assert ops.start_in_background is True and getattr(ops, "probe", None) is None
    with scoped_environ({"AWS_PROFILE": "x"}):  # the tick
        thread = threading.Thread(target=ops.start)
        thread.start()
        assert entered.wait(5)
    assert os.environ["AWS_PROFILE"] == "x"  # the start still runs
    release.set()
    thread.join(5)
    assert seen == ["x"] and os.environ["AWS_PROFILE"] == "shell"




def _two_slots(tmp_path, environs, providers):
    from tests.test_nc_pool_run import _cfg as _plain_cfg  # noqa: PLC0415

    slots = {}
    for name, environ in environs.items():
        d = tmp_path / name
        d.mkdir()
        cfg, path = _plain_cfg(d, accounts=[1], shard_base=name)
        slots[name] = ctl.nc_pool(cfg, path, ops=ProviderPoolOps(providers[name], FakeRoute(), FakeFencer()),
                                  environ=environ)
    return slots


class _Holding(FakeProvider):
    """``order`` records the watched variables, then (``hold``) stays inside the call
    until ``release`` is set and records them again."""

    def __init__(self, name, clock, *, hold=False):
        super().__init__(name, clock)
        self.hold, self.seen = hold, []
        self.inside, self.release = threading.Event(), threading.Event()

    def order(self, slot):
        self.seen.append({k: os.environ.get(k) for k in ("AWS_PROFILE", "ONLY_A", "ONLY_B")})
        self.inside.set()
        if self.hold:
            assert self.release.wait(5)
            time.sleep(0.05)
            self.seen.append({k: os.environ.get(k) for k in ("AWS_PROFILE", "ONLY_A", "ONLY_B")})
        return super().order(slot)


ENVIRONS = pytest.mark.parametrize("environs", [
    {"a": {"AWS_PROFILE": "profile-a"}, "b": {"AWS_PROFILE": "profile-b"}},  # one key, two values
    {"a": {"AWS_PROFILE": "same", "ONLY_A": "1"}, "b": {"AWS_PROFILE": "same", "ONLY_B": "2"}},  # key sets differ
], ids=["same-keys", "different-keys"])


@ENVIRONS
def test_two_pool_threads_with_different_envs_never_see_each_others(tmp_path, monkeypatch, environs):
    """Two pools of one process, each with its own environment, called at once from two
    threads (per-pool workers): the second call waits for the first to leave and then
    runs with its own values; the first sees its own values throughout."""
    monkeypatch.delenv("ONLY_A", raising=False)
    monkeypatch.delenv("ONLY_B", raising=False)
    clock = Clock()
    providers = {"a": _Holding("a", clock, hold=True), "b": _Holding("b", clock)}
    slots = _two_slots(tmp_path, environs, providers)
    try:
        a = threading.Thread(target=lambda: slots["a"].pool.ops.order(1))
        a.start()
        assert providers["a"].inside.wait(5)
        b = threading.Thread(target=lambda: slots["b"].pool.ops.order(1))
        b.start()
        assert not providers["b"].inside.wait(0.2)  # b waits while a's overlay is active
        providers["a"].release.set()
        a.join(5)
        b.join(5)
        assert not a.is_alive() and not b.is_alive()
    finally:
        for slot in slots.values():
            slot.lock.close()
    own_a = {"ONLY_A": None, "ONLY_B": None, **environs["a"]}
    own_b = {"ONLY_A": None, "ONLY_B": None, **environs["b"]}
    assert providers["a"].seen == [own_a, own_a]
    assert providers["b"].seen == [own_b]
    assert os.environ.get("AWS_PROFILE") == "shell" and "ONLY_A" not in os.environ and "ONLY_B" not in os.environ


@ENVIRONS
def test_a_call_whose_wait_runs_out_never_reaches_the_provider(monkeypatch, environs):
    """A background call (a start or release submitted in a tick that has ended) while
    another overlay stays active: after its wait it raises EnvOverlayConflict (variable
    names, never values) and the provider is never called with the other values."""
    monkeypatch.delenv("ONLY_A", raising=False)
    monkeypatch.delenv("ONLY_B", raising=False)
    provider = _Holding("b", Clock())
    ops = EnvScopedOps(ProviderPoolOps(provider, FakeRoute(), FakeFencer()), environs["b"], wait_s=0.05)
    with scoped_environ(environs["a"]):
        result = []
        t = threading.Thread(target=lambda: result.append(_raises(lambda: ops.order(1))))
        t.start()
        t.join(5)
    assert provider.seen == [] and provider.orders == []
    error = result[0]
    assert isinstance(error, EnvOverlayConflict) and "another environment overlay is active" in str(error)
    assert "profile-" not in str(error) and "same" not in str(error).split("differ:")[1]
    ops.order(1)  # the other overlay has left: the call runs with its own values
    assert provider.seen == [{"ONLY_A": None, "ONLY_B": None, **environs["b"]}]


def _raises(fn):
    try:
        fn()
    except BaseException as error:  # noqa: BLE001
        return error
    return None


def test_pools_with_the_same_env_run_their_calls_at_once(tmp_path):
    """Every pool of one run has the same overlay: calls from three threads are inside
    the provider at the same time and each sees the run's values."""
    clock = Clock()
    environ = {"AWS_PROFILE": "x"}
    providers = {name: _Holding(name, clock, hold=True) for name in ("a", "b", "c")}
    slots = _two_slots(tmp_path, {name: environ for name in providers}, providers)
    try:
        threads = [threading.Thread(target=slots[name].pool.ops.order, args=(1,)) for name in providers]
        for t in threads:
            t.start()
        assert all(p.inside.wait(5) for p in providers.values())  # all three inside at once
        for p in providers.values():
            p.release.set()
        for t in threads:
            t.join(5)
        assert not any(t.is_alive() for t in threads)
    finally:
        for slot in slots.values():
            slot.lock.close()
    for p in providers.values():
        assert [s["AWS_PROFILE"] for s in p.seen] == ["x", "x"]
    assert os.environ.get("AWS_PROFILE") == "shell"


def test_scoped_environ_refuses_any_other_overlay_while_one_is_active(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "shell")
    with scoped_environ({"AWS_PROFILE": "a", "K": "1"}):
        for other in ({"AWS_PROFILE": "b"}, {"AWS_PROFILE": "a"}, {"AWS_PROFILE": "a", "K": "1", "M": "2"}):
            with pytest.raises(EnvOverlayConflict):  # another value, a subset, a superset
                with scoped_environ(other):
                    pass
        assert os.environ["AWS_PROFILE"] == "a" and os.environ["K"] == "1"
        with scoped_environ({"K": "1", "AWS_PROFILE": "a"}):  # the same mapping: nests
            assert os.environ["AWS_PROFILE"] == "a"
    assert os.environ["AWS_PROFILE"] == "shell" and "K" not in os.environ
    with scoped_environ({"AWS_PROFILE": "b"}):  # nothing active any more: accepted
        assert os.environ["AWS_PROFILE"] == "b"


def test_scoped_environ_restores_on_an_exception_and_keeps_an_outside_change(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "shell")
    monkeypatch.delenv("OTHER", raising=False)
    with pytest.raises(KeyboardInterrupt):
        with scoped_environ({"AWS_PROFILE": "x"}):
            raise KeyboardInterrupt
    assert os.environ["AWS_PROFILE"] == "shell"
    with scoped_environ({"AWS_PROFILE": "x", "OTHER": "o"}):
        os.environ["OTHER"] = "set meanwhile"
    assert os.environ["AWS_PROFILE"] == "shell" and os.environ["OTHER"] == "set meanwhile"
    with scoped_environ({"AWS_PROFILE": "y"}):  # nothing left active by the earlier scopes
        assert os.environ["AWS_PROFILE"] == "y"


def test_scoped_environ_applies_all_or_nothing(monkeypatch):
    """A variable that cannot be set (a NUL byte) undoes the ones set before it and
    leaves no overlay active."""
    monkeypatch.setenv("AWS_PROFILE", "shell")
    monkeypatch.delenv("FIRST", raising=False)
    with pytest.raises(ValueError):
        with scoped_environ({"FIRST": "1", "AWS_PROFILE": "x", "BAD": "a\0b"}):
            raise AssertionError("not entered")
    assert "FIRST" not in os.environ and os.environ["AWS_PROFILE"] == "shell" and "BAD" not in os.environ
    with scoped_environ({"AWS_PROFILE": "z"}):
        assert os.environ["AWS_PROFILE"] == "z"
    assert os.environ["AWS_PROFILE"] == "shell"


# -- the controller: a conflicting overlay never ends a tick loop, a worker or a stop ------------

def _events(slot, name):
    return [e for e in slot.pool.events.events if e["event"].endswith(name)]


def test_a_conflicting_overlay_skips_that_pools_tick_and_the_others_tick_on(tmp_path):
    clock = Clock()
    providers = {"a": SeeingProvider("a", clock), "b": SeeingProvider("b", clock)}
    slots = _two_slots(tmp_path, {"a": {"AWS_PROFILE": "a"}, "b": None}, providers)
    c = ctl.RolloutController(slots, thread_cls=_NoThread, log=lambda _m: None)
    try:
        with scoped_environ({"AWS_PROFILE": "someone-else"}):  # another caller's overlay is active
            c.tick()
        assert providers["a"].orders == [] and len(providers["b"].orders) == 1  # b ticked on
        assert len(_events(slots["a"], "tick_failed")) == 1 and "EnvOverlayConflict" in \
            _events(slots["a"], "tick_failed")[0]["error"]
        assert c.ticks == 1 and c.check() is None
        c.tick()  # the conflict is gone: pool a ticks with its own overlay
        assert len(providers["a"].orders) == 1 and providers["a"].seen[-1][1]["AWS_PROFILE"] == "a"
    finally:
        c.stop("test")
    assert c not in ctl.active()


def test_a_stop_drains_with_the_pools_own_env_once_the_other_overlay_leaves(tmp_path):
    """Pool b stops while pool a's (another) overlay is active: the drain waits for it to
    leave and b's release sees b's values, never a's."""
    clock = Clock()
    providers = {"b": SeeingProvider("b", clock)}
    slots = _two_slots(tmp_path, {"b": {"AWS_PROFILE": "b"}}, providers)
    c = ctl.RolloutController(slots, thread_cls=_NoThread, log=lambda _m: None, drain_wait_s=0)
    c.tick()
    holding, done = threading.Event(), threading.Event()

    def pool_a_call():
        with scoped_environ({"AWS_PROFILE": "a"}):
            holding.set()
            done.wait(0.3)  # pool a's call lasts a while

    t = threading.Thread(target=pool_a_call)
    t.start()
    assert holding.wait(5)
    left = c.stop("test", join_s=10)
    t.join(5)
    assert left == {"b": []} and providers["b"].released == ["b-t0"]
    assert [env["AWS_PROFILE"] for what, env in providers["b"].seen if what == "release"] == ["b"]
    assert not _events(slots["b"], "env_overlay_conflict")
    assert os.environ.get("AWS_PROFILE") == "shell"


def test_a_stop_whose_wait_runs_out_leaves_the_members_to_run_down(tmp_path, capsys):
    """The other overlay stays past the stop's wait: no provider call with its values;
    the members are reported left (CRITICAL, a critical event, the billing banner) for
    `yeto run down`, and the stop still ends and closes every pool."""
    clock = Clock()
    providers = {"a": SeeingProvider("a", clock), "b": SeeingProvider("b", clock)}
    slots = _two_slots(tmp_path, {"a": {"AWS_PROFILE": "a"}, "b": {"AWS_PROFILE": "a"}}, providers)
    banners = []
    for name, slot in slots.items():
        slot.on_close = lambda left, name=name: banners.append((name, [m.ticket for m in left]))
    logs = []
    c = ctl.RolloutController(slots, thread_cls=_NoThread, log=logs.append, drain_wait_s=0)
    c.tick()
    with scoped_environ({"AWS_PROFILE": "someone-else"}):
        left = c.stop("test", join_s=0.1)
    assert {name: [m.ticket for m in members] for name, members in left.items()} == {"a": ["a-t0"], "b": ["b-t0"]}
    assert not any(p.released for p in providers.values())
    assert [what for what, _ in providers["a"].seen] == ["order"]  # no release under another overlay
    assert sorted(banners) == [("a", ["a-t0"]), ("b", ["b-t0"])] and c not in ctl.active()
    for slot in slots.values():
        events = _events(slot, "env_overlay_conflict")
        assert [(e["during"], e.get("critical")) for e in events] == [("drain", True)]
    assert any("CRITICAL" in line and "not drained" in line for line in logs)


def test_a_conflicting_overlay_never_kills_or_restarts_a_pool_worker(tmp_path):
    """Per-pool workers under the supervisor: ticks refused by a conflicting overlay are
    tick_failed events; the worker keeps running (no worker_restarted) and orders once
    the conflict is gone."""
    clock = Clock()
    providers = {"a": SeeingProvider("a", clock)}
    slots = _two_slots(tmp_path, {"a": {"AWS_PROFILE": "a"}}, providers)
    c = ctl.RolloutController(slots, log=lambda _m: None, interval=0.02, pool_workers=True, drain_wait_s=0)
    conflict = scoped_environ({"AWS_PROFILE": "someone-else"})
    conflict.__enter__()
    try:
        c.start()
        deadline = time.monotonic() + 5
        while len(_events(slots["a"], "tick_failed")) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        conflict.__exit__(None, None, None)
    try:
        deadline = time.monotonic() + 5
        while not providers["a"].orders and time.monotonic() < deadline:
            time.sleep(0.01)
        assert providers["a"].orders and providers["a"].seen[0][1]["AWS_PROFILE"] == "a"
        assert len(_events(slots["a"], "tick_failed")) >= 3 and not _events(slots["a"], "worker_restarted")
        assert slots["a"].restarts == 0 and c.check() is None
    finally:
        c.stop("test")
    assert os.environ.get("AWS_PROFILE") == "shell"


def test_a_conflicting_overlay_never_stops_a_detach(tmp_path):
    clock = Clock()
    providers = {"a": SeeingProvider("a", clock)}
    slots = _two_slots(tmp_path, {"a": {"AWS_PROFILE": "a"}}, providers)
    c = ctl.RolloutController(slots, thread_cls=_NoThread, log=lambda _m: None)
    c.tick()
    with scoped_environ({"AWS_PROFILE": "someone-else"}):
        out = c.detach("test")
    assert out == {"a": ["a-t0"]} and slots["a"].closed and c not in ctl.active()
    assert [e["during"] for e in _events(slots["a"], "env_overlay_conflict")] == ["detach"]
    assert not providers["a"].released  # a detach leaves the machine running


def test_ticks_skipped_in_a_row_on_a_conflict_reach_check_and_the_run_breach(tmp_path):
    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process", env={"AWS_PROFILE": "x"})
    clock = Clock()
    ops = {name: ProviderPoolOps(SeeingProvider(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    run = up.Run(cfg)
    c = run.start_controller(ops=ops, thread_cls=_NoThread)
    try:
        with scoped_environ({"AWS_PROFILE": "someone-else"}):
            for _ in range(ctl.ENV_CONFLICT_TICKS_N - 1):
                c.tick()
            assert c.check() is None
            c.tick()
        why = c.check()
        assert why and "not driven" in why and "nc" in why and "modal" in why
        assert "not driven" in (run.breach() or "")  # the run layer's controller_exit
        for slot in c.slots.values():
            assert [e.get("critical") for e in _events(slot, "env_overlay_conflict")] == [True]
        c.tick()  # the conflict is gone: the pools tick again and check() is clear
        assert c.check() is None
    finally:
        c.stop("test")


def test_members_a_timed_out_stop_left_stay_billed_and_run_down_releases_them_with_the_pools_env(
        tmp_path, monkeypatch, capsys):
    """End to end: a stop whose wait for another overlay runs out releases nothing and
    leaves the members billed in the pool's state (never ended, never counted as
    released); the pool-stop that `yeto run down` runs (a child process with the run's
    environment) then releases them, under the pool's own AWS_PROFILE."""
    from tests.test_nc_pool_run import _cfg as _plain_cfg  # noqa: PLC0415
    from yeto.launch.nc import cli as nc_cli, pool_run  # noqa: PLC0415

    clock = Clock()
    provider = SeeingProvider("b", clock)
    d = tmp_path / "b"
    d.mkdir()
    cfg, path = _plain_cfg(d, accounts=[1], shard_base="b")
    slot = ctl.nc_pool(cfg, path, ops=ProviderPoolOps(provider, FakeRoute(), FakeFencer()),
                       environ={"AWS_PROFILE": "b"})
    c = ctl.RolloutController({"b": slot}, thread_cls=_NoThread, log=lambda _m: None, drain_wait_s=0)
    c.tick()
    ticket = slot.pool.live()[0].ticket
    with scoped_environ({"AWS_PROFILE": "a"}):  # pool a's overlay stays past the stop's wait
        left = c.stop("test", join_s=0.05)
    assert [m.ticket for m in left["b"]] == [ticket] and provider.released == []
    out = capsys.readouterr().out
    assert "STILL BILLING" in out and "yeto run down --config <run.json>" in out  # not a pool-stop from a shell
    doc = json.loads(nc_cli.pool_state_path(cfg).read_text())
    assert [m["ticket"] for m in pool_run.billed_members(doc)] == [ticket]  # still billed: not ended
    assert provider.machines[ticket]["state"] == "active"
    # `yeto run down`: `python -m yeto.launch.nc pool-stop --config <pool.json>` with env=Run.env
    monkeypatch.setenv("AWS_PROFILE", "b")
    assert pool_run.stop_pool(cfg, path, ops=ProviderPoolOps(provider, FakeRoute(), FakeFencer()),
                              sleep=lambda s: None) == 0
    assert provider.released == [ticket]
    assert [env["AWS_PROFILE"] for what, env in provider.seen if what == "release"] == ["b"]
    assert not pool_run.billed_members(json.loads(nc_cli.pool_state_path(cfg).read_text()))
