"""An on-site control plane's run up that must not drive the run (own-cluster-fleet-stability
5.2): an old control host that comes back while the run store does not answer starts nothing
and writes nothing (its unit tries again; the store's sequence numbers decide once it reads),
and a run up that another host refuses never fences the host that drives the run."""

from __future__ import annotations

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


@pytest.mark.xfail(strict=True, reason="H1 (REV-890b): run up still claims the mirror before its refusals; "
                                       "moved after the preflight on the rebase onto #888")
def test_a_run_up_that_is_refused_never_fences_the_host_that_drives(tmp_path, monkeypatch):
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects, host="host-a")  # host A drives the run
    before = mirror(objects)
    rc = _main(tmp_path, monkeypatch, objects, "host-b", preflight=lambda self, *a, **kw: False)
    assert rc == 2  # host B refused (its preflight failed): nothing ordered
    assert mirror(objects) == before  # it neither took a number nor wrote the mirror
    a.save()
    a.mirror_state(force=True)
    assert a.control_fenced() is None  # host A drives on
