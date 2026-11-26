"""An on-site control plane moved to another host (own-cluster-fleet-stability 5.2): ``yeto
run up`` keeps the part of its state another host needs in the run store (a whitelist: no
process of the host, no key), one writer at a time (a writer, a sequence number, an
exclusive claim of each number), written from a background thread; a run up on another
host starts from it, and the one it replaced is fenced (exit 76). The spend continues:
never from zero, never twice."""

from __future__ import annotations

import ast
import json
import os
import threading
import time
from pathlib import Path

import pytest

from test_fleet_island_launches import example_doc
from test_fleet_service import (
    KEY,
    _env,  # noqa: F401 - autouse: the run's env, restored after each test
    onsite_config,
)

from yeto.launch import fleet_syncer
from yeto.rl import up, up_state_mirror
from yeto.rl.engine.object_store import DirectoryObjectStore


def make_run(tmp_path, root: str, objects, config=None) -> up.Run:
    run = up.Run(up.load_config(str(config or onsite_config(tmp_path))), state_root=tmp_path / root)
    run._objects = lambda: objects  # noqa: SLF001 - the run store
    return run


def priced(tmp_path):
    doc = json.loads(onsite_config(tmp_path).read_text())
    doc["syncer"] = {"usd_per_h": 10.0}  # the on-site syncer priced, so its hours count
    path = tmp_path / "priced.json"
    path.write_text(json.dumps(doc))
    return path


def mirror(objects) -> dict:
    return json.loads(objects.read(up.STATE_MIRROR))


def events(run) -> list[dict]:
    path = run.dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_the_mirror_holds_the_whitelisted_state_only(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    run = make_run(tmp_path, "host-a", objects)
    (run.dir / "island_hmac.key").write_text(KEY)  # never read for the mirror
    run.state.update({"run_t0": 100.0, "trainer_t0": 200.0, "steps": {"first_group": "t"},
                      "procs": {"trainer": {"pid": 4242, "log": "/x/trainer.log"}},
                      "island_hmac_key": KEY, "fleet_island_stops": {"1": {"why": "breaker"}},
                      "retired_pools": [{"name": "p0", "config": "/c/p0.json", "usd_per_h": 3.0, "shard_base": "s",
                                         "removed_at": "t", "pool": {"provider": {"env": {"SECRET": "v"}}}}]})
    run.start_state_mirror(objects, host="host-a")
    raw = objects.read(up.STATE_MIRROR).decode()
    doc = json.loads(raw)
    assert set(doc["state"]) == {"run_t0", "trainer_t0", "steps", "fleet_island_stops", "retired_pools"}
    assert KEY not in raw and "procs" not in raw and "4242" not in raw
    # a removed pool: its billing fields only, never its pool config (a provider's environment)
    assert doc["state"]["retired_pools"] == [{"name": "p0", "config": "/c/p0.json", "usd_per_h": 3.0,
                                              "shard_base": "s", "removed_at": "t"}]
    assert "SECRET" not in raw and "provider" not in raw
    assert doc["writer"]["host"] == "host-a" and doc["seq"] == 1
    assert run.env[up.JOURNAL_FROM_STORE_ENV] == "1"


def _state_keys(paths) -> set[str]:
    """Every literal key run up's code reads or writes in its state: ``<x>.state["k"]``,
    ``.state.get/setdefault/pop("k")``, ``"k" in <x>.state`` and the ``flag_stop`` keys."""
    keys = set()

    def is_state(node):
        return isinstance(node, ast.Attribute) and node.attr == "state"

    for path in paths:
        for node in ast.walk(ast.parse(Path(path).read_text())):
            if isinstance(node, ast.Subscript) and is_state(node.value) and isinstance(node.slice, ast.Constant):
                keys.add(node.slice.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and is_state(node.func.value)
                  and node.func.attr in ("get", "setdefault", "pop") and node.args
                  and isinstance(node.args[0], ast.Constant)):
                keys.add(node.args[0].value)
            elif (isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant)
                  and any(is_state(c) for c in node.comparators)):
                keys.add(node.left.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "flag_stop" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)):
                keys.add(node.args[1].value)
    return {k for k in keys if isinstance(k, str)}


def test_every_state_key_is_mirrored_or_excluded_by_name():
    """A new run up state field must be put in one of the lists (a test, not a review, notices)."""
    root = Path(up.__file__).parent
    keys = _state_keys([root / "up.py", root / "up_fleet_runtime.py", root / "up_fleet.py"])
    assert keys, "the scan found nothing"
    assert keys - up_state_mirror.KEYS - up_state_mirror.EXCLUDED == set()
    assert not up_state_mirror.KEYS & up_state_mirror.EXCLUDED
    assert "procs" in up_state_mirror.EXCLUDED and "fleet_stop" in up_state_mirror.KEYS


def test_a_run_up_without_an_on_site_control_plane_writes_nothing_to_the_store(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(example_doc(tmp_path)))  # per_island on the store, no syncer service
    run = make_run(tmp_path, "host-a", objects, path)
    run.state["run_t0"] = 1.0
    run.start_state_mirror(objects)
    run.save()
    assert objects.read(up.STATE_MIRROR) is None and up.JOURNAL_FROM_STORE_ENV not in run.env
    assert getattr(run, "_state_mirror", None) is None and run.control_fenced() is None


def test_a_dry_run_starts_no_mirror_and_never_touches_the_store(tmp_path, monkeypatch):
    from yeto import cli

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(up.Run, "_objects", lambda self, *a, **kw: pytest.fail("a dry run touched the run store"))
    started = []
    monkeypatch.setattr(up.Run, "start_state_mirror", lambda self, *a, **kw: started.append(1))
    for name in ("preflight", "cluster_preflight"):
        monkeypatch.setattr(up.Run, name, lambda self, *a, **kw: True)
    monkeypatch.setattr(up.Run, "log_fleet_plan", lambda self: None)
    args = cli.build_parser().parse_args(["run", "up", "--config", str(onsite_config(tmp_path)), "--dry-run"])
    assert up.main(args) == 0 and started == []


def test_a_new_host_starts_from_the_mirror_without_the_old_host_s_processes(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    a.state.update({"run_t0": 1000.0, "trainer_t0": 1100.0, "steps": {"first_group": "t", "trainer": "t"},
                    "ups": [{"pid": os.getpid(), "start": 1000.0, "seen": 5000.0, "end": None}],
                    "procs": {"trainer": {"pid": os.getpid()}}, "fleet_island_stops": {"1": {"why": "x"}}})
    a.save()
    a.mirror_state(force=True)
    at = mirror(objects)["at"]
    b = make_run(tmp_path, "host-b", objects)  # another host: no state directory of this run
    assert b.state == {}
    before = time.time()
    b.start_state_mirror(objects, host="host-b")
    assert b.state["run_t0"] == 1000.0 and b.state["trainer_t0"] == 1100.0
    assert b.state["fleet_island_stops"] == {"1": {"why": "x"}} and b.state["steps"]["first_group"] == "t"
    assert "procs" not in b.state  # a PID of host A means nothing here (it could be any process)
    # the lost host's period ends at the mirror's time plus its write interval, never past now
    end = b.state["ups"][0]["end"]
    assert min(at + up.STATE_MIRROR_EVERY_S, before) <= end <= min(at + up.STATE_MIRROR_EVERY_S, time.time())
    assert b.state["restored_from"]["host"] == "host-a" and b.state["restored_from"]["seq"] == 2
    assert json.loads(b.state_path.read_text())["run_t0"] == 1000.0
    assert mirror(objects)["writer"]["host"] == "host-b" and mirror(objects)["seq"] == 3
    again = make_run(tmp_path, "host-b", objects)  # this host's own state now: never restored over
    assert again.restore_state(mirror(objects), host="host-b",
                               local=up_state_mirror.read_local(again.dir / up_state_mirror.LOCAL_FILE)) is False


def test_a_restarted_host_whose_state_is_older_than_another_host_s_mirror_restores_it(tmp_path):
    """Host A ran, host B took over (and wrote the mirror after it), host A comes back: its local
    state is older than the mirror, which another host wrote. Restored from the mirror; the local
    state is kept as state.json.stale."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.state.update({"run_t0": 1000.0, "steps": {"first_group": "a"}})
    a.save()
    a.start_state_mirror(objects, host="host-a")
    b = make_run(tmp_path, "host-b", objects)
    b.start_state_mirror(objects, host="host-b")
    b.state["steps"]["trainer"] = "b"  # what happened on host B after it took over
    b.save()
    b.mirror_state(force=True)
    a2 = make_run(tmp_path, "host-a", objects)
    assert a2.state["steps"] == {"first_group": "a"}  # its own, older state
    a2.start_state_mirror(objects, host="host-a")
    assert a2.state["steps"] == {"first_group": "a", "trainer": "b"} and a2.state["restored_from"]["host"] == "host-b"
    stale = a2.state_path.with_name("state.json.stale")
    assert json.loads(stale.read_text())["steps"] == {"first_group": "a"}
    assert a2.state["restored_from"]["stale"] == str(stale)
    assert mirror(objects)["writer"]["host"] == "host-a"  # it writes on, from after host B's number


def test_a_run_up_that_another_host_replaced_is_fenced_never_overwrites_and_exits_76(tmp_path, monkeypatch):
    from test_fleet_service import recording_controller
    from yeto import cli
    from yeto.launch.pool import controller as ctl

    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "host-a")
    monkeypatch.setattr(up.Run, "_objects", lambda self, *a, **kw: objects)
    monkeypatch.setattr("socket.gethostname", lambda: "host-a")
    for name in ("preflight", "cluster_preflight", "write_fleet_plan", "sandbox_prepare"):
        monkeypatch.setattr(up.Run, name, lambda self, *a, **kw: True)
    monkeypatch.setattr(up.Run, "log_fleet_plan", lambda self: None)
    monkeypatch.setattr(up.signal, "signal", lambda *a: None)  # not this test process's handlers
    made, seen = [], {}
    before = list(ctl._ACTIVE)

    def pool(self):
        self.controller = recording_controller(True)  # it would drain on exit: a fenced run up must not
        made.append(self.controller)

    def drive(run, args):
        b = make_run(tmp_path, "host-b", objects)
        b.start_state_mirror(objects, host="host-b")  # another host took the run over
        seen["b"] = mirror(objects)
        run.state.setdefault("steps", {})["late"] = "a"
        run.save()
        run.mirror_state(force=True)  # host A writes again: it finds host B's newer number
        seen["after"] = mirror(objects)
        seen["fenced"] = run.control_fenced()
        monkeypatch.setattr(up.time, "sleep", lambda s: pytest.fail("the monitor went on after the fence"))
        run.monitor(poll_s=0.0)  # its monitor: the breakers stop at once
        return 0

    monkeypatch.setattr(up.Run, "pool", pool)
    monkeypatch.setattr(up, "_drive", drive)
    args = cli.build_parser().parse_args(["run", "up", "--config", str(onsite_config(tmp_path))])
    try:
        assert up.main(args) == up.CONTROL_FENCED_EXIT == 76
    finally:
        with ctl._ACTIVE_LOCK:
            ctl._ACTIVE[:] = before
    assert seen["after"] == seen["b"]  # never overwritten
    assert "written by host host-b" in seen["fenced"]
    (controller,) = made
    assert controller.drain_on_exit is False and [c for c in controller.calls if c[0] == "stop"] == []
    run = make_run(tmp_path, "host-a", objects)
    assert [e for e in events(run) if e.get("event") == "control_fenced"]
    from yeto.rl import fleet_service

    assert up.CONTROL_FENCED_EXIT == fleet_service.CONTROL_FENCED_EXIT in fleet_service.CONTROL_PREVENT
    assert not (tmp_path / "host-a" / "r" / "onsite-control.ended").exists()


def test_a_fenced_control_host_s_trainer_worker_forwards_no_fleet_stop(tmp_path):
    from test_fleet_island_runup import FakeController
    from test_fleet_trainer_worker import FakeLaunches

    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    launches = FakeLaunches()
    worker = a.attach_trainer_worker(FakeController(), objects, ops=launches)
    launches.start("run-i0", [])
    worker._seen.add("run-i0")
    (a.dir / "STOP").write_text("stop")  # the run's STOP on the old host
    make_run(tmp_path, "host-b", objects).start_state_mirror(objects, host="host-b")
    a.mirror_state(force=True)
    assert a.mirror_fenced() and worker.fenced()
    worker.step()
    assert launches.stops == [] and not objects.list("fleet/trainers/0/STOP")  # nothing forwarded
    with pytest.raises(up.ControlFenced):
        a.breach()


def test_the_spend_continues_on_the_new_host_and_never_counts_twice(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    config = priced(tmp_path)
    now = time.time()
    a = make_run(tmp_path, "host-a", objects, config)
    a.state.update({"run_t0": now - 7200, "trainer_t0": now - 7000,
                    "ups": [{"pid": os.getpid(), "start": now - 7200, "seen": now, "end": None}]})
    a.save()
    a.start_state_mirror(objects, host="host-a")
    spent_a = a.spend_usd()
    assert spent_a > 0
    hour_a = a.fleet_rt().spend(now + 3600).total_usd - a.fleet_rt().spend(now).total_usd
    # host A is lost now; the control plane starts on host B from the mirror
    b = make_run(tmp_path, "host-b", objects, config)
    b.start_state_mirror(objects, host="host-b")
    assert b.fleet_rt().spend(now).total_usd == pytest.approx(a.fleet_rt().spend(now).total_usd, rel=1e-4)
    b.begin_up()
    hour_b = b.fleet_rt().spend(now + 3600).total_usd - b.fleet_rt().spend(now).total_usd
    assert hour_b == pytest.approx(hour_a, rel=1e-3)  # one host's hour, not two (A's period ended)
    assert hour_b > 0


def test_the_on_site_syncer_s_hours_come_from_its_heartbeat(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    run = make_run(tmp_path, "host-b", objects, priced(tmp_path))
    run.start_state_mirror(objects, host="host-b")
    assert run.syncer_hours(0.5) == 0.5  # no heartbeat yet: the launch's hours
    fleet_syncer.write_service_status(objects, host="10.0.0.5", state="running", key=None, generation=3,
                                      since=1000.0, billed_s=3600.0, clock=lambda: 4600.0)
    run._syncer_hours = None  # noqa: SLF001 - read again now
    assert run.syncer_hours(0.5) == pytest.approx(2.0)  # an hour before, and this generation's hour
    assert fleet_syncer.service_seconds({"since": 10.0, "at": 70.0, "billed_s": 5.0}) == 65.0
    assert fleet_syncer.service_seconds({"at": 70.0}) is None


def test_a_mirror_that_does_not_read_refuses_the_start(tmp_path):
    class Broken(DirectoryObjectStore):
        def read(self, key):
            raise OSError("store down")

    run = make_run(tmp_path, "host-b", Broken(tmp_path / "store"))
    with pytest.raises(up.RunUpRefused, match="did not read"):
        run.start_state_mirror(Broken(tmp_path / "store"), host="host-b")
    assert not run.state_path.exists()


def test_two_saves_inside_the_interval_mirror_once_a_forced_one_always(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    clock = {"t": 100.0}
    m = up_state_mirror.StateMirror(objects, run="r", host="h", seq=0, local_path=tmp_path / "local.json",
                                    monotonic=lambda: clock["t"], background=False, log=lambda m: None)
    assert m.submit({"t0": 1.0}) is True and m.writes == 1
    clock["t"] += 10
    assert m.submit({"t0": 2.0}) is False and m.writes == 1  # throttled
    assert m.submit({"t0": 3.0}, force=True) is True and m.writes == 2  # a stop: at once
    clock["t"] += up_state_mirror.EVERY_S
    assert m.submit({"t0": 4.0}) is True and m.writes == 3
    assert mirror(objects)["state"] == {"t0": 4.0} and mirror(objects)["seq"] == 3
    assert json.loads((tmp_path / "local.json").read_text())["seq"] == 3


def test_a_store_that_hangs_never_holds_the_monitor(tmp_path):
    release = threading.Event()

    class Hangs(DirectoryObjectStore):
        def read(self, key):
            release.wait(30)
            return super().read(key)

    logs: list[str] = []
    m = up_state_mirror.StateMirror(Hangs(tmp_path / "store"), run="r", host="h", seq=0,
                                    local_path=tmp_path / "local.json", timeout_s=0.3, log=logs.append)
    started = time.monotonic()
    assert m.submit({"t0": 1.0}, force=True) is True  # a forced write waits at most its timeout
    assert m.submit({"t0": 2.0}, force=True) is True
    assert time.monotonic() - started < 3.0
    assert any("did not answer within" in line for line in logs)
    release.set()
    deadline = time.monotonic() + 10
    while m.writes < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert m.writes >= 1 and m.fenced is None  # written once the store answered


def test_a_starting_writer_skips_numbers_claimed_by_a_writer_that_never_wrote(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    objects.create(up_state_mirror.marker_key(1), b'{"host": "gone", "nonce": "x"}')  # claimed, never written
    m = up_state_mirror.StateMirror(objects, run="r", host="h", seq=0, local_path=tmp_path / "l.json",
                                    background=False, log=lambda m: None)
    assert m.submit({"t0": 1.0}) is True and m.fenced is None and mirror(objects)["seq"] == 2
    objects.create(up_state_mirror.marker_key(3), b'{"host": "other", "nonce": "y"}')  # a live writer's claim
    assert m.submit({"t0": 2.0}, force=True) is True and "host other" in m.fenced
    assert mirror(objects)["seq"] == 2  # not written over


def test_only_an_on_site_control_plane_s_trainer_worker_takes_islands_over(tmp_path):
    from test_fleet_island_runup import FakeController
    from test_fleet_trainer_worker import FakeLaunches

    objects = DirectoryObjectStore(tmp_path / "store")
    onsite = make_run(tmp_path, "onsite", objects)
    worker = onsite.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
    assert worker.takeover is True and worker.takeover_margin_s == 60.0
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(example_doc(tmp_path)))
    other = DirectoryObjectStore(tmp_path / "store2")
    plain = make_run(tmp_path, "plain", other, path)
    assert plain.attach_trainer_worker(FakeController(), other, ops=FakeLaunches()).takeover is False


def test_a_newer_number_another_writer_wrote_is_never_written_over(tmp_path):
    """The mirror says another writer wrote a higher number (its claims of the numbers between
    gone): this writer is fenced by the mirror itself, before it claims its next number."""
    objects = DirectoryObjectStore(tmp_path / "store")
    m = up_state_mirror.StateMirror(objects, run="r", host="host-a", seq=0, local_path=tmp_path / "l.json",
                                    background=False, log=lambda m: None)
    assert m.submit({"t0": 1.0}) is True and mirror(objects)["seq"] == 1
    newer = {"schema": up_state_mirror.SCHEMA, "run": "r", "host": "host-b", "seq": 5, "at": 1.0, "state": {},
             "writer": {"host": "host-b", "nonce": "other"}}
    objects.write(up.STATE_MIRROR, json.dumps(newer).encode())
    assert m.submit({"t0": 2.0}, force=True) is True and "host host-b" in m.fenced
    assert mirror(objects) == newer and objects.read(up_state_mirror.marker_key(2)) is None
    assert m.submit({"t0": 3.0}, force=True) is False  # fenced: nothing more


def test_a_stop_a_breaker_and_the_end_reach_the_store_at_once(tmp_path, monkeypatch):
    from yeto import runs

    objects = DirectoryObjectStore(tmp_path / "store")
    run = make_run(tmp_path, "host-a", objects)
    run.start_state_mirror(objects, host="host-a")
    forced: list[str] = []
    real = run.mirror_state

    def spy(*, force=False):
        if force:
            import traceback

            forced.append(traceback.extract_stack(limit=2)[0].name)
        return real(force=force)

    run.mirror_state = spy
    run.end_up()
    monkeypatch.setattr(runs, "request_stop", lambda name: (_ for _ in ()).throw(OSError("no flag")))
    run.trainer_down = lambda why: None
    run.flag_stop("run.starvation", "starvation", "STARVATION", "no groups", grace_s=0.0)
    run.state["t0"] = time.time() - 7200
    monkeypatch.setattr(run, "breaker_param", lambda name, key: 1.0 if name == "run.wall_clock" else None)
    assert run.breach() == "wall clock"
    assert {"end_up", "flag_stop", "breach", "record"} <= set(forced)
    assert mirror(objects)["state"]["breaker"]["why"] == "wall clock"  # not left to the 30 s interval


# -- second review: conditional writes, a writer cut off from the store, the last state at exit -------

def _mirror(objects, tmp_path, **kw):
    kw.setdefault("background", False)
    kw.setdefault("log", lambda m: None)
    return up_state_mirror.StateMirror(objects, run="r", host=kw.pop("host", "host-a"), seq=kw.pop("seq", 0),
                                       local_path=tmp_path / "l.json", **kw)


def test_the_mirror_is_replaced_only_over_what_was_read(tmp_path):
    """Another write between the read and this one (L1): read again and written over it when
    it is not newer, fenced when it is another writer's higher number."""
    objects = DirectoryObjectStore(tmp_path / "store")
    m = _mirror(objects, tmp_path)
    assert m.submit({"t0": 1.0}) is True and mirror(objects)["seq"] == 1
    real = objects.replace_if
    calls = []

    def replace_if(key, data, expected):
        calls.append(key)
        if len(calls) == 1:  # a stale writer's late write lands right before this one
            objects.write(key, json.dumps({**mirror(objects), "state": {"late": 1}}).encode())
        return real(key, data, expected)

    objects.replace_if = replace_if
    assert m.submit({"t0": 2.0}, force=True) is True and m.fenced is None
    assert len(calls) == 2 and mirror(objects)["state"] == {"t0": 2.0} and mirror(objects)["seq"] == 2
    calls.clear()

    def newer(key, data, expected):  # another writer with a higher number wrote meanwhile
        calls.append(key)
        objects.write(key, json.dumps({"schema": up_state_mirror.SCHEMA, "run": "r", "host": "host-b", "seq": 9,
                                       "at": 1.0, "state": {}, "writer": {"host": "host-b", "nonce": "b"}}).encode())
        return False

    objects.replace_if = newer
    assert m.submit({"t0": 3.0}, force=True) is True and "host host-b" in m.fenced
    assert mirror(objects)["seq"] == 9 and len(calls) == 1  # never written over


def test_the_first_mirror_is_an_exclusive_create(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    real = objects.create
    seen = []

    def create(key, data):
        if key == up.STATE_MIRROR and not seen:
            seen.append(1)
            real(key, json.dumps({"schema": up_state_mirror.SCHEMA, "run": "r", "host": "host-b", "seq": 4, "at": 1.0,
                                  "state": {}, "writer": {"host": "host-b", "nonce": "b"}}).encode())
            return False
        return real(key, data)

    objects.create = create
    m = _mirror(objects, tmp_path)
    m.submit({"t0": 1.0})
    assert "host host-b" in m.fenced and mirror(objects)["writer"]["host"] == "host-b"


def test_a_claim_whose_answer_was_lost_is_this_writer_s(tmp_path):
    """The exclusive create of the next number went through but its answer was lost (M2):
    the claim reads back as this writer's, the write goes on, no fence."""
    objects = DirectoryObjectStore(tmp_path / "store")
    real = objects.create

    def create(key, data):
        made = real(key, data)
        return False if key.startswith(up_state_mirror.STATE_MIRROR_SEQ) else made

    objects.create = create
    m = _mirror(objects, tmp_path)
    assert m.submit({"t0": 1.0}) is True and m.fenced is None and mirror(objects)["seq"] == 1
    assert m.submit({"t0": 2.0}, force=True) is True and m.fenced is None and mirror(objects)["seq"] == 2


def test_a_writer_cut_off_from_the_store_is_fenced_after_the_lease_and_margin(tmp_path):
    """No write confirmed for the records lock's lease plus the takeover margin while it tries
    (M3): fenced, as by another host; a writer that only had nothing to write is not."""
    class Down(DirectoryObjectStore):
        down = False

        def read(self, key):
            if self.down:
                raise OSError("partitioned")
            return super().read(key)

    objects = Down(tmp_path / "store")
    clock = {"t": 1000.0}
    fenced = []
    m = _mirror(objects, tmp_path, monotonic=lambda: clock["t"], stale_after_s=180.0, on_fenced=fenced.append)
    assert m.submit({"t0": 1.0}) is True and m.writes == 1
    clock["t"] += 3600  # nothing to write for an hour: not stale
    assert m.stale() is None
    objects.down = True
    assert m.submit({"t0": 2.0}, force=True) is True and m.writes == 1  # tried, not confirmed
    clock["t"] += 179
    assert m.stale() is None
    clock["t"] += 2
    assert "no write of run up's state" in m.stale() and m.fenced == m.stale() and len(fenced) == 1
    assert m.submit({"t0": 3.0}, force=True) is False
    objects = Down(tmp_path / "store2")
    m2 = _mirror(objects, tmp_path, monotonic=lambda: clock["t"], stale_after_s=180.0)
    m2.submit({"t0": 1.0})
    objects.down = True
    m2.submit({"t0": 2.0}, force=True)
    objects.down = False
    clock["t"] += 100
    m2.submit({"t0": 3.0}, force=True)  # confirmed again in time
    clock["t"] += 100
    assert m2.stale() is None and m2.fenced is None


def test_run_up_s_stale_limit_is_the_records_lease_plus_the_takeover_margin(tmp_path):
    from yeto import launcher

    assert up_state_mirror.RECORDS_LOCK_TTL_S == launcher.ISLAND_RECORDS_LOCK_TTL_S
    objects = DirectoryObjectStore(tmp_path / "store")
    run = make_run(tmp_path, "host-a", objects)
    run.start_state_mirror(objects, host="host-a")
    assert run._state_mirror.stale_after_s == launcher.ISLAND_RECORDS_LOCK_TTL_S + 60.0  # noqa: SLF001
    run._state_mirror.fenced = None  # noqa: SLF001
    run._state_mirror._tried_at, run._state_mirror._ok_at = 1.0, 0.0  # noqa: SLF001 - tried, never confirmed
    run._state_mirror.monotonic = lambda: 1.0 + 181.0  # noqa: SLF001
    with pytest.raises(up.ControlFenced, match="no write of run up's state"):
        run.raise_if_control_fenced()


def test_the_nc_watch_after_a_breach_stops_once_another_host_drives(tmp_path, monkeypatch):
    objects = DirectoryObjectStore(tmp_path / "store")
    run = make_run(tmp_path, "host-a", objects)
    run.start_state_mirror(objects, host="host-a")
    rounds = []
    monkeypatch.setattr(run, "pool_members", lambda: rounds.append(1) or [{"state": "running"}])
    monkeypatch.setattr(run, "spend_usd", lambda: 1.0)
    monkeypatch.setattr(run, "breaker_param", lambda name, key: None)

    def sleep(s):
        if len(rounds) == 2:  # another host takes the run over after the second round
            make_run(tmp_path, "host-b", objects).start_state_mirror(objects, host="host-b")
            run.mirror_state(force=True)

    monkeypatch.setattr(up.time, "sleep", sleep)
    with pytest.raises(up.ControlFenced):
        run.watch_nc_after_breach("budget", poll_s=0.0)
    assert len(rounds) == 2  # checked at the top of every round


def test_the_last_throttled_state_reaches_the_store_at_exit(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    clock = {"t": 100.0}
    m = _mirror(objects, tmp_path, monotonic=lambda: clock["t"])
    m.submit({"t0": 1.0})
    m.submit({"t0": 2.0})  # throttled
    assert mirror(objects)["state"] == {"t0": 1.0}
    assert m.flush() is True and mirror(objects)["state"] == {"t0": 2.0}
    assert m.flush() is False  # nothing kept back any more
    run = make_run(tmp_path, "host-a", objects)
    assert run.flush_state_mirror() is False  # no mirror: nothing


def test_a_mirror_of_this_very_host_is_never_restored_over_its_state(tmp_path):
    """The mirror is newer than this host's sequence file but this host wrote it (Me): its own
    state stays."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.state.update({"run_t0": 1.0, "steps": {"first_group": "local"}})
    a.save()
    doc = {"schema": up_state_mirror.SCHEMA, "run": a.run, "host": "host-a", "seq": 7, "at": time.time(),
           "writer": {"host": "host-a", "nonce": "old"}, "state": {"steps": {"first_group": "mirror"}}}
    assert a.restore_state(doc, host="host-a", local={"run": a.run, "seq": 2}) is False
    assert a.state["steps"] == {"first_group": "local"}
    assert a.restore_state({**doc, "writer": {"host": "host-b", "nonce": "b"}, "host": "host-b"}, host="host-a",
                           local={"run": a.run, "seq": 2}) is True


def test_an_old_mirror_ends_the_lost_host_s_period_at_its_time_not_now(tmp_path):
    """The lost host's open period ends at the mirror's time plus the write interval (Mx),
    however long ago that was: the hours since are no one's."""
    objects = DirectoryObjectStore(tmp_path / "store")
    b = make_run(tmp_path, "host-b", objects)
    at = time.time() - 5000
    doc = {"schema": up_state_mirror.SCHEMA, "run": b.run, "host": "host-a", "seq": 3, "at": at,
           "writer": {"host": "host-a", "nonce": "a"},
           "state": {"ups": [{"start": at - 1000, "end": None}], "run_t0": at - 1000}}
    assert b.restore_state(doc, host="host-b", local={}) is True
    assert b.state["ups"][0]["end"] == pytest.approx(at + up.STATE_MIRROR_EVERY_S)


def test_the_syncer_service_s_earlier_generations_are_carried_over(tmp_path, monkeypatch):
    """Each generation's heartbeat carries the seconds of the generations before it (Mw)."""
    from yeto.rl import fleet_service

    objects = DirectoryObjectStore(tmp_path / "store")
    fleet_syncer.write_service_status(objects, host="h1", state="running", key=None, generation=1,
                                      since=1000.0, billed_s=500.0, clock=lambda: 1600.0)
    assert fleet_syncer.service_seconds(fleet_syncer.read_service_status(objects)) == 1100.0
    beat = fleet_service.Heartbeat(objects, "h2", None, generation=2,
                                   billed_s=fleet_syncer.service_seconds(fleet_syncer.read_service_status(objects)))
    beat.beat()
    doc = fleet_syncer.read_service_status(objects)
    assert doc["billed_s"] == 1100.0 and doc["since"] == pytest.approx(time.time(), abs=60)


def test_a_restarted_syncer_service_carries_the_earlier_generations_seconds(tmp_path, monkeypatch):
    from test_fleet_service import run_service

    objects = DirectoryObjectStore(tmp_path / "store")
    fleet_syncer.write_service_status(objects, host="10.0.0.5", state="running", key=None, generation=1,
                                      since=1000.0, billed_s=500.0, clock=lambda: 1600.0)  # its last beat
    rc, seen, _syncer, objects, _logs = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 0 and seen["before"]["billed_s"] == 1100.0  # 500 s before, then 600 s of generation 1
