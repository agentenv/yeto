"""An on-site control plane's run up that must not drive the run (own-cluster-fleet-stability
5.2): an old control host that comes back while the run store does not answer starts nothing
and writes nothing (its unit tries again; the store's sequence numbers decide once it reads),
and a run up that another host refuses never fences the host that drives the run."""

from __future__ import annotations

import json

import pytest

from test_fleet_service import _env, onsite_config  # noqa: F401 - autouse: the run's env, restored after each test
from test_onsite_state_mirror import make_run, mirror

from yeto.rl import fleet_service, up
from yeto.rl.engine.object_store import DirectoryObjectStore


def _main(tmp_path, monkeypatch, objects, host: str, *, machine_id: str | None = None, hostname: str | None = None,
          **fakes):
    from yeto import cli
    from yeto.rl import up_state_mirror

    id_path = tmp_path / f"machine-id-{host}"  # this host's own: never the test machine's
    if machine_id is not None:
        id_path.write_text(machine_id + "\n")
    monkeypatch.setattr(up_state_mirror, "MACHINE_ID_PATH", str(id_path))
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / host)
    monkeypatch.setattr(up.Run, "_objects", lambda self, *a, **kw: objects)
    monkeypatch.setattr("socket.gethostname", lambda: hostname or host)
    monkeypatch.setattr(up.Run, "log_fleet_plan", lambda self: None)
    monkeypatch.setattr(up.signal, "signal", lambda *a: None)
    for name, value in fakes.items():
        monkeypatch.setattr(up.Run, name, value)
    args = cli.build_parser().parse_args(["run", "up", "--config", str(onsite_config(tmp_path))])
    return up.main(args)


def test_an_old_control_host_that_comes_back_while_the_store_is_down_starts_nothing(tmp_path, monkeypatch):
    """The mirror does not read: refused before any preflight, pool or launch (exit 2), nothing
    written; exit 2 is a refusal the control unit tries again (every 30 s), never a fence, so
    the host that drives the run is not touched and this one starts once the store reads."""
    class Down(DirectoryObjectStore):
        def read(self, key):
            raise OSError("the run store does not answer")

        def write(self, key, data):
            pytest.fail(f"a refused run up wrote {key}")

        def create(self, key, data):
            pytest.fail(f"a refused run up created {key}")

    started = []
    rc = _main(tmp_path, monkeypatch, Down(tmp_path / "store"), "host-a",
               preflight=lambda self, *a, **kw: started.append("preflight") or True,
               pool=lambda self: started.append("pool"), trainer=lambda self: started.append("trainer"))
    assert rc == 2 and started == []
    assert 2 not in fleet_service.CONTROL_PREVENT  # the unit tries again, without a start limit
    assert up.CONTROL_FENCED_EXIT in fleet_service.CONTROL_PREVENT


def _passes(**more):
    """Every check before ordering passes (no cluster, no sandbox); ``more`` replaces any."""
    return {"preflight": lambda self, *a, **kw: True, "cluster_preflight": lambda self, *a, **kw: True,
            "write_fleet_plan": lambda self: True, "sandbox_prepare": lambda self: True, **more}


def _seq_claims(objects) -> list[str]:
    return sorted(str(k) for k in objects.list("fleet/control/up-state-seq/"))


@pytest.mark.parametrize("refusal", ["preflight", "cluster_preflight", "write_fleet_plan", "sandbox_prepare",
                                     "forced_stop", "lock"])
def test_a_run_up_that_is_refused_never_fences_the_host_that_drives(tmp_path, monkeypatch, refusal):
    """H1: a run up on host B refused before it orders (any check, or the forced stop of the
    state it restored, or another run up holding its lock) takes no sequence number and writes
    nothing to the run store: host A drives on, never fenced."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    if refusal == "forced_stop":
        a.state.setdefault("steps", {})["nc_forced_stop"] = {"at": "2026-11-26T00:00:00Z", "why": "budget"}
    a.start_state_mirror(objects, host="host-a")  # host A drives the run
    before, claims = mirror(objects), _seq_claims(objects)
    fakes = _passes(pool=lambda self: pytest.fail("a refused run up ordered"))
    if refusal in fakes:
        fakes[refusal] = lambda self, *a, **kw: False
    if refusal == "lock":
        fakes["lock_up"] = lambda self, role="up": 4242
    rc = _main(tmp_path, monkeypatch, objects, "host-b", **fakes)
    assert rc == 2  # host B refused: nothing ordered
    assert mirror(objects) == before and _seq_claims(objects) == claims  # no number taken, nothing written
    a.save()
    a.mirror_state(force=True)
    assert a.control_fenced() is None  # host A drives on
    assert mirror(objects)["writer"]["host"] == "host-a"


def test_a_run_up_takes_its_number_only_past_its_checks(tmp_path, monkeypatch):
    """H1: the mirror is read at the start (state restored) and written first right before the
    pools start: the preflight ran on the restored state, and nothing reached the store before."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.state["run_t0"] = 123.0
    a.start_state_mirror(objects, host="host-a")
    seen = {}

    def preflight(self, *a, **kw):
        seen["run_t0"] = self.state.get("run_t0")  # restored before the checks
        seen["mirror_host"] = mirror(objects)["writer"]["host"]  # not written yet
        return True

    def pool(self):
        seen["at_pool"] = mirror(objects)["writer"]["host"]
        raise up.RunUpRefused("stop here")

    rc = _main(tmp_path, monkeypatch, objects, "host-b", **_passes(preflight=preflight, pool=pool))
    assert rc == 2
    assert seen == {"run_t0": 123.0, "mirror_host": "host-a", "at_pool": "host-b"}
    a.save()
    a.mirror_state(force=True)
    assert a.control_fenced() is not None  # B went on to order: A is fenced


def test_a_mirror_another_host_wrote_during_the_checks_is_restored_again(tmp_path, monkeypatch):
    """H1: host A writes the mirror while host B runs its checks: B reads it again before its
    first write, restores the newer state (keeping its own steps), takes the number past A's."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.state.update(spend_usd_estimate=1.0, breaker={"why": "an old trip"})
    a.start_state_mirror(objects, host="host-a")
    b = make_run(tmp_path, "host-b", objects)  # host B's own older state: kept aside at the first restore
    b.state["spend_usd_estimate"] = 0.5
    b.save()
    seen = {}

    def preflight(self, *args, **kw):
        a.state["spend_usd_estimate"] = 7.0  # host A goes on meanwhile
        a.state.pop("breaker")
        a.save()
        a.mirror_state(force=True)
        self.mark("preflight", ok=True)
        return True

    def pool(self):
        disk = json.loads(self.state_path.read_text())
        seen.update(disk_spend=disk.get("spend_usd_estimate"), disk_breaker=disk.get("breaker"),
                    disk_steps=sorted(disk.get("steps") or {}))
        seen.update(spend=self.state.get("spend_usd_estimate"), steps=sorted(self.state.get("steps") or {}),
                    seq=mirror(objects)["seq"], host=mirror(objects)["writer"]["host"])
        raise up.RunUpRefused("stop here")

    a_seq = None

    def sandbox(self):
        nonlocal a_seq
        a_seq = mirror(objects)["seq"]
        return True

    assert _main(tmp_path, monkeypatch, objects, "host-b",
                 **_passes(preflight=preflight, sandbox_prepare=sandbox, pool=pool)) == 2
    assert seen["spend"] == 7.0 and "preflight" in seen["steps"]
    assert seen["disk_spend"] == 7.0 and seen["disk_breaker"] is None and "preflight" in seen["disk_steps"]
    stale = json.loads((b.dir / "state.json.stale").read_text())
    assert stale["spend_usd_estimate"] == 0.5  # this host's own state, not the first restore's copy
    assert seen["host"] == "host-b" and seen["seq"] > a_seq


def test_a_forced_stop_another_host_wrote_during_the_checks_refuses(tmp_path, monkeypatch):
    """H1: the state restored again carries a forced stop host A recorded meanwhile: host B is
    refused before its first write (no number taken), as at the start."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")

    def preflight(self, *args, **kw):
        a.mark("nc_forced_stop", why="budget")
        a.mirror_state(force=True)
        return True

    claims = {}

    def sandbox(self):
        claims["before"] = _seq_claims(objects)
        return True

    rc = _main(tmp_path, monkeypatch, objects, "host-b",
               **_passes(preflight=preflight, sandbox_prepare=sandbox, pool=lambda self: pytest.fail("ordered")))
    assert rc == 2 and _seq_claims(objects) == claims["before"]
    a.mirror_state(force=True)
    assert a.control_fenced() is None


def test_a_mirror_that_stops_reading_before_the_first_write_refuses(tmp_path, monkeypatch):
    """H1: the store answered at the start, not right before the pools: refused (exit 2, the
    unit tries again), nothing written."""
    objects = DirectoryObjectStore(tmp_path / "store")
    make_run(tmp_path, "host-a", objects).start_state_mirror(objects, host="host-a")
    before = mirror(objects)

    def sandbox(self):
        def down(key):
            raise OSError("the run store does not answer")

        monkeypatch.setattr(objects, "read", down)
        return True

    rc = _main(tmp_path, monkeypatch, objects, "host-b",
               **_passes(sandbox_prepare=sandbox, pool=lambda self: pytest.fail("ordered")))
    monkeypatch.undo()
    assert rc == 2 and mirror(objects) == before


# -- M4: a fenced run up hands its island launches over ----------------------------------------

def _registered(tmp_path, monkeypatch, objects):
    from yeto import runs
    from yeto.rl.island_launches import fleet_prefix

    run = make_run(tmp_path, "host-a", objects)
    base = fleet_prefix(run.cfg["fleet"]["trainer_argv"])
    for k, gpu in ((0, "ssh:2x8xb300"), (1, "modal:8xb200"), (2, "aws:2xh100@us-east-2")):
        runs.create_run(f"{base}-i{k}", {"rl_fleet_role": "island", "rl_fleet_learner_id": k, "gpu": gpu})
    return run, base


def test_a_fenced_run_up_asks_its_ssh_and_modal_island_launches_to_hand_over(tmp_path, monkeypatch):
    """M4: run up fenced (exit 76) writes the hand-off flag of this host's island launches an
    other host can take over (ssh, Modal); a sky island's launch keeps supervising it here."""
    from yeto import runs
    from yeto.rl import up

    objects = DirectoryObjectStore(tmp_path / "store")
    _run, base = _registered(tmp_path, monkeypatch, objects)

    def pool(self):
        raise up.ControlFenced("host host-b drives the run now")

    assert _main(tmp_path, monkeypatch, objects, "host-a", **_passes(pool=pool)) == up.CONTROL_FENCED_EXIT
    assert [runs.handoff_flag_path(f"{base}-i{k}").exists() for k in (0, 1, 2)] == [True, True, False]
    assert "host host-b drives the run now" in runs.handoff_flag_path(f"{base}-i0").read_text()


def test_a_run_up_that_ends_otherwise_asks_no_launch_to_hand_over(tmp_path, monkeypatch):
    from yeto import runs
    from yeto.rl import up

    objects = DirectoryObjectStore(tmp_path / "store")
    _run, base = _registered(tmp_path, monkeypatch, objects)

    def pool(self):
        raise up.RunUpRefused("a pool refused")

    assert _main(tmp_path, monkeypatch, objects, "host-a", **_passes(pool=pool)) == 2
    assert not any(runs.handoff_flag_path(f"{base}-i{k}").exists() for k in (0, 1, 2))


# -- REV-890c: the first write must be confirmed; a fenced run up stops nothing more ----------------

def test_a_first_write_that_loses_to_a_write_landing_during_it_refuses_the_start(tmp_path, monkeypatch):
    """H1, counterexample 1: host A writes right after host B's last read before its first
    write (between that read and the one inside the write): B's first write loses. B is
    refused (exit 2, nothing ordered), it fences no one, and A drives on."""
    from yeto.rl import up_state_mirror

    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    real = up_state_mirror.StateMirror.start

    def a_writes_first(self, state):
        a.state["spend_usd_estimate"] = 3.0
        a.mirror_state(force=True)  # host A's write lands now
        return real(self, state)

    monkeypatch.setattr(up_state_mirror.StateMirror, "start", a_writes_first)
    rc = _main(tmp_path, monkeypatch, objects, "host-b", **_passes(pool=lambda self: pytest.fail("ordered")))
    assert rc == 2
    assert mirror(objects)["writer"]["host"] == "host-a" and mirror(objects)["state"]["spend_usd_estimate"] == 3.0
    assert not [e for e in _events(tmp_path / "host-b") if e.get("event") == "control_fenced"]  # no fence
    a.mirror_state(force=True)
    assert a.control_fenced() is None


def test_a_first_write_the_store_does_not_take_refuses_the_start(tmp_path, monkeypatch):
    """H1, counterexample 2: the store fails host B's first write (OSError): B is refused
    (exit 2; its unit tries again) instead of ordering with no write of its own. Before,
    B's reconciler then claimed the fleet, A was fenced by it, and A's next keepalive
    fenced B: no host drove the run."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    before = mirror(objects)
    real_create = objects.create
    jitter = {"n": 0}

    def create(key, data):
        if "up-state-seq" in str(key) and jitter["n"] == 0:
            jitter["n"] += 1
            raise OSError("the run store dropped the connection")
        return real_create(key, data)

    def sandbox(self):
        monkeypatch.setattr(objects, "create", create)  # only for host B's first write
        return True

    rc = _main(tmp_path, monkeypatch, objects, "host-b",
               **_passes(sandbox_prepare=sandbox, pool=lambda self: pytest.fail("ordered")))
    monkeypatch.setattr(objects, "create", real_create)
    assert rc == 2 and jitter["n"] == 1 and mirror(objects) == before
    a.mirror_state(force=True)
    assert a.control_fenced() is None  # host A drives on


def _events(state_root) -> list[dict]:
    out = []
    for path in state_root.rglob("events.jsonl"):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def _b_drives(tmp_path, objects):
    b = make_run(tmp_path, "host-b", objects)
    b.start_state_mirror(objects, host="host-b")
    return b


@pytest.mark.parametrize("where", ["flag_stop", "breach", "record"])
def test_a_run_up_fenced_by_its_forced_write_stops_and_tears_down_nothing(tmp_path, monkeypatch, where):
    """M2: host A's stop or breaker trip finds host B's newer state at its forced mirror write:
    A raises ControlFenced at once. No STOP flag, no trainer down, no breaker record goes on."""
    from yeto import runs

    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    _b_drives(tmp_path, objects)  # host B took the run over
    downs, flags = [], []
    a.trainer_down = lambda why: downs.append(why)
    monkeypatch.setattr(runs, "request_stop", lambda name: flags.append(name))
    with pytest.raises(up.ControlFenced) as fenced:
        if where == "flag_stop":
            a.flag_stop("run.starvation", "starvation", "STARVATION", "no groups", grace_s=0.0)
        elif where == "breach":
            a.state["t0"] = __import__("time").time() - 7200
            monkeypatch.setattr(a, "breaker_param", lambda name, key: 1.0 if name == "run.wall_clock" else None)
            monkeypatch.setattr(a, "raise_if_control_fenced", _after_first(a.raise_if_control_fenced))
            monkeypatch.setattr(a.fleet_rt(), "record", lambda event, island=None: ())  # no fleet stop: the trip's own write
            a.breach()
        else:
            a.fleet_rt().record({"event": "rl_breaker", "name": "run.budget", "state": "trip",
                                 "stop_reason": "budget", "scope": None})
    assert fenced.value.handoff  # another writer drives the run: the island launches hand over
    assert downs == [] and flags == []
    assert mirror(objects)["writer"]["host"] == "host-b"


def _after_first(check):
    """``check`` skipped once (the breaker's own check at its start), then real: the fence is
    found by the forced write of the trip."""
    calls = {"n": 0}

    def wrapped():
        calls["n"] += 1
        if calls["n"] > 1:
            check()

    return wrapped


def test_a_run_up_cut_off_from_the_store_keeps_its_island_launches_supervising(tmp_path, monkeypatch):
    """L1: a stale fence (no write confirmed for the limit, no successor known) ends run up
    (exit 76) but writes no HANDOFF: the island launches keep supervising their islands."""
    from yeto import runs

    objects = DirectoryObjectStore(tmp_path / "store")
    _run, base = _registered(tmp_path, monkeypatch, objects)

    def pool(self):
        raise up.ControlFenced("no write of run up's state was confirmed", handoff=False)

    assert _main(tmp_path, monkeypatch, objects, "host-a", **_passes(pool=pool)) == up.CONTROL_FENCED_EXIT
    assert not any(runs.handoff_flag_path(f"{base}-i{k}").exists() for k in (0, 1, 2))


def test_the_fence_s_cause_decides_the_hand_off(tmp_path):
    """L1: fenced by another writer's newer number: hand off; only cut off from the store
    (stale): not."""
    clock = {"t": 0.0}
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    _b_drives(tmp_path, objects)
    with pytest.raises(up.ControlFenced) as by_writer:
        a.mirror_state_or_fence()
    assert by_writer.value.handoff

    c = make_run(tmp_path, "host-c", DirectoryObjectStore(tmp_path / "store-c"))
    c.start_state_mirror(c._objects(), host="host-c")  # noqa: SLF001
    mirror_c = c._state_mirror  # noqa: SLF001
    mirror_c.monotonic = lambda: clock["t"]
    mirror_c._ok_at = 0.0  # noqa: SLF001
    mirror_c._tried_at = 1.0  # noqa: SLF001 - a write tried and never confirmed since
    clock["t"] = 1.0 + mirror_c.stale_after_s
    with pytest.raises(up.ControlFenced) as stale:
        c.raise_if_control_fenced()
    assert not stale.value.handoff and mirror_c.fenced_by is None


def test_a_run_up_fenced_by_the_fleet_writes_no_throttled_state_at_its_exit(tmp_path, monkeypatch):
    """M09: run up fenced by another writer of the fleet (its mirror itself not fenced) does
    not flush the state its throttle kept back: the host that drives the run writes it."""
    objects = DirectoryObjectStore(tmp_path / "store")

    def pool(self):
        self.state["spend_usd_estimate"] = 9.0
        self.save()  # mirrored at most every 30 s: kept back
        raise up.ControlFenced("fleet/applied.json is written by another controller now")

    assert _main(tmp_path, monkeypatch, objects, "host-b", **_passes(pool=pool)) == up.CONTROL_FENCED_EXIT
    assert mirror(objects)["writer"]["host"] == "host-b"
    assert mirror(objects)["state"].get("spend_usd_estimate") != 9.0  # never written by the fenced run up


def test_two_hosts_of_one_host_name_are_told_by_their_machine_ids(tmp_path, monkeypatch):
    """L5: hosts A and B are both named ``ubuntu``: B restores A's newer state (its own older
    state kept aside) because their machine ids differ; a host with A's machine id finds its
    own state there and restores nothing."""
    objects = DirectoryObjectStore(tmp_path / "store")
    ids = {}

    def a_pool(self):
        self.state["spend_usd_estimate"] = 4.0
        self.mirror_state(force=True)
        ids["a"] = self.control_host_id
        raise up.RunUpRefused("host A stops here")

    assert _main(tmp_path, monkeypatch, objects, "host-a", hostname="ubuntu", machine_id="a" * 32,
                 **_passes(pool=a_pool)) == 2
    seen = {}

    def b_preflight(self, *args, **kw):  # the state restored at the start; refused: nothing written
        seen.update(spend=self.state.get("spend_usd_estimate"), restored=self.state.get("restored_from"),
                    id=self.control_host_id)
        return False

    for root, mid in (("host-a2", "a" * 32), ("host-b", "b" * 32)):  # A's own machine first: the mirror is A's
        own = make_run(tmp_path, root, objects)
        own.state["spend_usd_estimate"] = 0.5  # its own older state
        own.save()
        seen.clear()
        assert _main(tmp_path, monkeypatch, objects, root, hostname="ubuntu", machine_id=mid,
                     **_passes(preflight=b_preflight)) == 2
        if root == "host-b":  # another machine of the same name: A's newer state is restored
            assert seen["spend"] == 4.0 and seen["restored"]["host"] == "ubuntu"
            assert seen["id"] and seen["id"] != ids["a"]
        else:  # host A's own machine: its own state, nothing restored
            assert seen["spend"] == 0.5 and not seen["restored"] and seen["id"] == ids["a"], seen


@pytest.mark.parametrize("where", ["stop_island", "on_trainer_start"])
def test_a_fenced_run_up_stops_no_island_and_rewrites_no_fleet_stop(tmp_path, monkeypatch, where):
    """M2, the fleet's own stops: an island stop (its STOP, its pool-stop) and a new trainer
    launch's fleet stops check the fence first or right after their forced write."""
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")
    _b_drives(tmp_path, objects)
    cmds = []
    a._cmd = lambda *args, **kw: cmds.append(args) or 0  # noqa: SLF001
    with pytest.raises(up.ControlFenced):
        if where == "stop_island":
            a.fleet_rt().stop_island(0, "run.starvation", "no groups")
        else:
            a.fleet_rt().on_trainer_start()
    assert cmds == []  # no pool-stop
    assert not (objects.read("fleet/trainers/0/STOP") or b"")


def test_the_trainer_worker_of_run_up_is_told_by_this_host_s_machine_id(tmp_path, monkeypatch):
    """L5: the trainer worker records and compares the machine id run up read at its start."""
    from test_fleet_island_runup import FakeController
    from test_fleet_trainer_worker import FakeLaunches

    objects = DirectoryObjectStore(tmp_path / "store")
    seen = {}

    def pool(self):
        worker = self.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
        seen.update(host=worker.host, host_id=worker.host_id, run=self.control_host_id)
        raise up.RunUpRefused("stops here")

    assert _main(tmp_path, monkeypatch, objects, "host-a", hostname="ubuntu", machine_id="c" * 32,
                 **_passes(pool=pool)) == 2
    assert seen["host"] == "ubuntu" and seen["host_id"] == seen["run"] and seen["host_id"].startswith("machine-id:")
    assert "c" * 32 not in seen["host_id"]  # a digest, never the id itself
