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


def _main(tmp_path, monkeypatch, objects, host: str, **fakes):
    from yeto import cli

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / host)
    monkeypatch.setattr(up.Run, "_objects", lambda self, *a, **kw: objects)
    monkeypatch.setattr("socket.gethostname", lambda: host)
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
