"""An on-site control plane moved to another host (own-cluster-fleet-stability 5.2): ``yeto
run up`` keeps the part of its state another host needs in the run store (a whitelist: no
process of the host, no key), and a run up on a host without the run's state directory
starts from it. The spend continues: never from zero, never twice."""

from __future__ import annotations

import json
import os
import time

import pytest

from test_fleet_island_launches import example_doc
from test_fleet_service import (
    KEY,
    _env,  # noqa: F401 - autouse: the run's env, restored after each test
    onsite_config,
)

from yeto.rl import up
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


def test_the_mirror_holds_the_whitelisted_state_only(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    run = make_run(tmp_path, "host-a", objects)
    (run.dir / "island_hmac.key").write_text(KEY)  # never read for the mirror
    run.state.update({"run_t0": 100.0, "trainer_t0": 200.0, "steps": {"first_group": "t"},
                      "procs": {"trainer": {"pid": 4242, "log": "/x/trainer.log"}},
                      "island_hmac_key": KEY, "fleet_island_stops": {"1": {"why": "breaker"}}})
    run.start_state_mirror(objects)
    raw = objects.read(up.STATE_MIRROR).decode()
    doc = json.loads(raw)
    assert set(doc["state"]) == {"run_t0", "trainer_t0", "steps", "fleet_island_stops"}
    assert KEY not in raw and "procs" not in raw and "4242" not in raw
    assert set(doc["state"]) <= up.STATE_MIRROR_KEYS and "procs" not in up.STATE_MIRROR_KEYS
    assert run.env[up.JOURNAL_FROM_STORE_ENV] == "1"


def test_a_run_up_without_an_on_site_control_plane_writes_nothing_to_the_store(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(example_doc(tmp_path)))  # per_island on the store, no syncer service
    run = make_run(tmp_path, "host-a", objects, path)
    run.state["run_t0"] = 1.0
    run.start_state_mirror(objects)
    run.save()
    assert objects.read(up.STATE_MIRROR) is None and up.JOURNAL_FROM_STORE_ENV not in run.env
    assert getattr(run, "_state_mirror", None) is None


def test_a_new_host_starts_from_the_mirror_without_the_old_host_s_processes(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    a = make_run(tmp_path, "host-a", objects)
    a.start_state_mirror(objects)
    a.state.update({"run_t0": 1000.0, "trainer_t0": 1100.0, "steps": {"first_group": "t", "trainer": "t"},
                    "ups": [{"pid": os.getpid(), "start": 1000.0, "seen": 5000.0, "end": None}],
                    "procs": {"trainer": {"pid": os.getpid()}}, "fleet_island_stops": {"1": {"why": "x"}}})
    a.save()
    a.mirror_state(force=True)
    b = make_run(tmp_path, "host-b", objects)  # another host: no state directory of this run
    assert b.state == {}
    b.start_state_mirror(objects)
    assert b.state["run_t0"] == 1000.0 and b.state["trainer_t0"] == 1100.0
    assert b.state["fleet_island_stops"] == {"1": {"why": "x"}} and b.state["steps"]["first_group"] == "t"
    assert "procs" not in b.state  # a PID of host A means nothing here (it could be any process)
    assert b.state["ups"][0]["end"] == 5000.0  # the lost host's period ends at its last sight
    assert b.state["restored_from"]["host"]
    assert json.loads(b.state_path.read_text())["run_t0"] == 1000.0
    again = make_run(tmp_path, "host-b", objects)  # a local state now: never restored over
    assert again.restore_state(objects) is False


def test_the_spend_continues_on_the_new_host_and_never_counts_twice(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    config = priced(tmp_path)
    now = time.time()
    a = make_run(tmp_path, "host-a", objects, config)
    a.state.update({"run_t0": now - 7200, "trainer_t0": now - 7000,
                    "ups": [{"pid": os.getpid(), "start": now - 7200, "seen": now, "end": None}]})
    a.save()
    a.start_state_mirror(objects)
    spent_a = a.spend_usd()
    assert spent_a > 0
    hour_a = a.fleet_rt().spend(now + 3600).total_usd - a.fleet_rt().spend(now).total_usd
    # host A is lost now; the control plane starts on host B from the mirror
    b = make_run(tmp_path, "host-b", objects, config)
    b.start_state_mirror(objects)
    assert b.fleet_rt().spend(now).total_usd == pytest.approx(a.fleet_rt().spend(now).total_usd)  # not from zero
    b.begin_up()
    hour_b = b.fleet_rt().spend(now + 3600).total_usd - b.fleet_rt().spend(now).total_usd
    assert hour_b == pytest.approx(hour_a, rel=1e-3)  # one host's hour, not two (A's period ended)
    assert hour_b > 0


def test_a_mirror_that_does_not_read_refuses_the_start(tmp_path):
    class Broken(DirectoryObjectStore):
        def read(self, key):
            raise OSError("store down")

    run = make_run(tmp_path, "host-b", Broken(tmp_path / "store"))
    with pytest.raises(up.RunUpRefused, match="did not read"):
        run.start_state_mirror(Broken(tmp_path / "store"))
    assert not run.state_path.exists()


def test_only_an_on_site_control_plane_s_trainer_worker_takes_islands_over(tmp_path):
    from test_fleet_island_runup import FakeController
    from test_fleet_trainer_worker import FakeLaunches

    objects = DirectoryObjectStore(tmp_path / "store")
    onsite = make_run(tmp_path, "onsite", objects)
    assert onsite.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches()).takeover is True
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(example_doc(tmp_path)))
    other = DirectoryObjectStore(tmp_path / "store2")
    plain = make_run(tmp_path, "plain", other, path)
    assert plain.attach_trainer_worker(FakeController(), other, ops=FakeLaunches()).takeover is False
