"""fleet.jsonl names a fleet island by its learner id and is copied into the run store.

* ``island_id``: an island launch of a fleet launched island by island has one island,
  island 0 of its launch (``<prefix>-l0-<cloud>``); its records carry the fleet learner id
  (``--rl-fleet-learner-id``), the id the syncer tape uses. Every other launch keeps the
  ``-l<m>`` index of the name (as on main).
* the store copy (``dashboard.fleet_store``): chunks under ``fleet/log/<lane>/<folder>/``,
  one folder per launcher process; a failing store is a WARN and never reaches the log
  writer; the dashboard and ``fleet_store merge`` fold every folder, each record once.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from yeto.dashboard import fleet as fleet_mod
from yeto.dashboard import fleet_store
from yeto.dashboard.fleet import FleetLog, close_open_islands, from_env
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all
from yeto.dashboard.store_tapes import StoreTapes
from yeto.launch import manifest
from yeto.rl.engine.object_store import DirectoryObjectStore

ISLAND = "run-i1-l0-ssh"  # the one island of island 1's launch


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- island_id ---------------------------------------------------------------------------------

def test_a_fleet_island_launch_names_its_island_by_learner_id(tmp_path):
    log = from_env(tmp_path / "fleet.jsonl", {ISLAND: SimpleNamespace()}, learner_ids={ISLAND: 3})
    log.event("island_ready", ISLAND)
    log.cost_tick()
    log.event("island_node_lost", ISLAND, node="b300-1", rank=1)
    ready, tick, lost = _records(tmp_path / "fleet.jsonl")
    assert ready["island_id"] == 3 and lost["island_id"] == 3
    assert tick["islands"][0]["island_id"] == 3
    assert "learner_id" not in ready and "learner_id" not in lost  # no new field: island_id carries it


def test_without_a_learner_id_the_name_index_stays(tmp_path):
    log = from_env(tmp_path / "fleet.jsonl", {"run-l0-aws": SimpleNamespace(), "run-l1-aws": SimpleNamespace()})
    log.event("island_ready", "run-l1-aws")
    log.event("rollout_event", "pool-a", pool_event="pool_up")  # a pool: no index in its name
    ready, pool = _records(tmp_path / "fleet.jsonl")
    assert ready["island_id"] == 1 and pool["island_id"] is None
    assert fleet_mod.island_id_of("run-l3-aws") == 3 and fleet_mod.island_id_of("pool") is None


def test_the_record_shape_is_main_s_for_a_single_island_launch(tmp_path):
    """The customer launch (one trainer island, no fleet role): byte-identical records."""
    clock = Clock()
    meta = {"cloud": "modal", "region": None, "gpu": "H100", "gpus": 8}
    new = FleetLog(tmp_path / "a.jsonl", {"k31-l0-modal": meta}, prices={}, clock=clock)
    new.event("island_ready", "k31-l0-modal")
    new.cost_tick()
    want = [{"event": "island_ready", "time_unix": 1000.0, "island": "k31-l0-modal", "island_id": 0,
             "cloud": "modal", "region": None, "gpu": "H100", "gpus": 8, "price_key": None}]
    assert _records(tmp_path / "a.jsonl")[:1] == want
    assert list(_records(tmp_path / "a.jsonl")[1]["islands"][0]) == ["island", "island_id", "rate_usd_h", "wall_h",
                                                                    "cost_usd", "running"]


def test_a_bool_or_negative_learner_id_is_not_a_learner_id(tmp_path):
    log = from_env(tmp_path / "f.jsonl", {ISLAND: SimpleNamespace(), "x-l2-aws": SimpleNamespace()},
                   learner_ids={ISLAND: True, "x-l2-aws": -1})
    assert log.island_id(ISLAND) == 0 and log.island_id("x-l2-aws") == 2


def test_teardown_close_keeps_the_learner_id(tmp_path):
    log = from_env(tmp_path / "fleet.jsonl", {ISLAND: SimpleNamespace()}, learner_ids={ISLAND: 2})
    log.event("island_ready", ISLAND)
    assert close_open_islands(tmp_path / "fleet.jsonl", "teardown") == [ISLAND]
    assert _records(tmp_path / "fleet.jsonl")[-1]["island_id"] == 2


def _launch_args(tmp_path, *, role=None, learner=None):
    return SimpleNamespace(cluster_prefix="run-i1", rl_fleet_role=role, rl_fleet_learner_id=learner,
                           rl_fleet_learners=None if learner is None else 4)


def test_the_launch_passes_its_fleet_learner_id(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    log = manifest._dashboard_fleet_log(_launch_args(tmp_path, role="island", learner=1),  # noqa: SLF001
                                        {ISLAND: SimpleNamespace()}, {ISLAND: ("job", None)})
    assert log.island_id(ISLAND) == 1 and log.uploader is None  # no store: local only, as before
    plain = manifest._dashboard_fleet_log(_launch_args(tmp_path), {"run-i1-l1-aws": SimpleNamespace()},  # noqa: SLF001
                                          {"run-i1-l1-aws": ("job", None)})
    assert plain.island_id("run-i1-l1-aws") == 1 and plain.uploader is None


def test_per_island_fleet_logs_fold_into_their_learner_islands(tmp_path):
    """Two island launches (learner 0 and 1), each island 0 of its launch: the dashboard
    shows two islands under the syncer tape's learner ids, not one island 0."""
    paths = []
    for k in (0, 1):
        path = tmp_path / f"i{k}" / "fleet.jsonl"
        log = from_env(path, {f"run-i{k}-l0-ssh": SimpleNamespace()}, learner_ids={f"run-i{k}-l0-ssh": k})
        log.clock = Clock(1000.0 + k)
        log.event("island_ready", f"run-i{k}-l0-ssh")
        paths.append(str(path))
    reducer = Reducer()
    load_all(reducer, paths)
    assert {iid: reducer.islands[iid]["name"] for iid in ("0", "1")} == {"0": "run-i0-l0-ssh", "1": "run-i1-l0-ssh"}
    assert reducer.islands["1"]["ready_ts"] == 1001.0


# -- the store copy ----------------------------------------------------------------------------

def _store_keys(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.jsonl"))


def test_the_store_copy_has_every_line_under_its_lane(tmp_path):
    store = tmp_path / "store"
    objects = DirectoryObjectStore(str(store))
    log = from_env(tmp_path / "fleet.jsonl", {ISLAND: SimpleNamespace()}, learner_ids={ISLAND: 1})
    log.uploader = fleet_store.start_uploader(objects, log.path, learner_id=1, interval_s=3600.0)
    log.event("island_ready", ISLAND)
    log.event("island_node_lost", ISLAND, node="b300-1")
    assert log.close() is True and log.uploader is None
    keys = _store_keys(store)
    assert len(keys) == 1 and keys[0].startswith("fleet/log/l1/") and keys[0].endswith("/00000000.jsonl")
    folder = keys[0].split("/")[3]
    assert fleet_store.incarnation().split("-p")[0] == folder.split("-p")[0]  # <host>-p<pid>-<UTC>
    from yeto.dashboard.store_tapes import launch_started
    assert launch_started(folder) is not None
    assert (store / keys[0]).read_bytes() == (tmp_path / "fleet.jsonl").read_bytes()


def test_a_whole_fleet_launch_uses_the_fleet_lane():
    assert fleet_store.lane(None) == "fleet" and fleet_store.lane(True) == "fleet" and fleet_store.lane(0) == "l0"


class FailingObjects:
    def __init__(self):
        self.writes = 0

    def write(self, key, data):
        self.writes += 1
        raise OSError("store down")


def test_a_failing_store_is_a_warn_and_never_reaches_the_writer(tmp_path):
    objects, lines = FailingObjects(), []
    up = fleet_store.start_uploader(objects, tmp_path / "fleet.jsonl", learner_id=0, interval_s=3600.0,
                                    log=lines.append)
    log = from_env(tmp_path / "fleet.jsonl", {ISLAND: SimpleNamespace()})
    log.uploader = up
    for _ in range(3):
        log.event("island_ready", ISLAND)
        up.flush()
    assert len(_records(tmp_path / "fleet.jsonl")) == 3  # the local file is whole
    assert objects.writes == 3 and len(lines) == 1 and "WARN" in lines[0] and "store down" in lines[0]
    assert log.close(timeout_s=0.2) is False  # bounded, never raises


def test_the_warn_is_rate_limited():
    lines = []
    warn = fleet_store._Warn(lines.append)  # noqa: SLF001
    for _ in range(25):
        warn("[tape-upload] fleet.jsonl: OSError: down")
    assert len(lines) == 3 and "(1 failure(s))" in lines[0] and "(20 failure(s))" in lines[2]


def test_an_upload_interval_of_0_uploads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "0")
    assert fleet_store.start_uploader(DirectoryObjectStore(str(tmp_path)), tmp_path / "f.jsonl") is None


def test_an_uploader_that_cannot_start_is_a_warn(tmp_path, monkeypatch):
    lines = []

    def boom(*a, **kw):
        raise RuntimeError("no thread")

    monkeypatch.setattr("yeto.rl.engine.tape_upload.TapeUploader.start", boom)
    assert fleet_store.start_uploader(object(), tmp_path / "f.jsonl", interval_s=1.0, log=lines.append) is None
    assert lines and "WARN" in lines[0]


def test_the_launch_starts_the_copy_only_with_a_store(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    store = tmp_path / "store"
    log = manifest._dashboard_fleet_log(_launch_args(tmp_path, role="island", learner=2),  # noqa: SLF001
                                        {ISLAND: SimpleNamespace()}, {ISLAND: ("job", None)},
                                        objects=DirectoryObjectStore(str(store)))
    assert log.uploader is not None and log.uploader.island == "l2"
    log.event("island_ready", ISLAND)
    assert log.close() is True
    assert [k.split("/")[2] for k in _store_keys(store)] == ["l2"]


# -- readers -----------------------------------------------------------------------------------

def _two_hosts(tmp_path):
    """A control plane that moved: host A's launcher (island ready, node lost), then host B's
    (ready again, a spare in); host B's local file also holds A's lines (a restored run
    directory), its store folder only its own. Host B's folder sorts first by name."""
    store = tmp_path / "store"
    objects = DirectoryObjectStore(str(store))
    a = [{"event": "island_ready", "time_unix": 100.0, "island": ISLAND, "island_id": 1},
         {"event": "island_node_lost", "time_unix": 150.0, "island": ISLAND, "island_id": 1, "node": "b300-1"},
         {"event": "island_lost", "time_unix": 160.0, "island": ISLAND, "island_id": 1}]
    b = [{"event": "island_ready", "time_unix": 400.0, "island": ISLAND, "island_id": 1},
         {"event": "island_node_replaced", "time_unix": 401.0, "island": ISLAND, "island_id": 1, "spare": "b300-3"}]
    text = lambda rows: "".join(json.dumps(r) + "\n" for r in rows).encode()  # noqa: E731
    objects.write("fleet/log/l1/hosta-p7-20261127T010000Z/00000000.jsonl", text(a[:2]))
    objects.write("fleet/log/l1/hosta-p7-20261127T010000Z/00000001.jsonl", text(a[2:]))
    # host B sorts before host A by name; its folder started later
    objects.write("fleet/log/l1/a-hostb-p9-20261127T020000Z/00000000.jsonl", text(b))
    local = tmp_path / "fleet.jsonl"
    local.write_bytes(text(a + b))
    return objects, a, b, local


def test_the_dashboard_folds_every_folder_once_in_start_order(tmp_path):
    objects, a, b, local = _two_hosts(tmp_path)
    reducer = Reducer()
    load_all(reducer, [str(local)])
    store = fleet_store.store_readers(StoreTapes(objects, label="store"))
    assert store.pump(reducer) == 0  # every store record is already folded from the local file
    isl = reducer.islands["1"]
    assert isl["fleet_state"] == "ready" and isl["lost_ts"] == 160.0 and isl["closed_s"] == pytest.approx(60.0)
    assert reducer.duplicates == len(a) + len(b)
    only_store = Reducer()
    assert fleet_store.store_readers(StoreTapes(objects, label="store")).pump(only_store) == len(a) + len(b)
    o = only_store.islands["1"]
    assert o["fleet_state"] == "ready" and o["ready_ts"] == 100.0 and o["open_ts"] == 400.0
    assert {r["event"] for r in only_store.fleet_records} == {"island_ready", "island_lost"}
    assert sum(1 for e in only_store.events if e["type"] == "island_node_lost") == 1


def test_the_store_readers_report_both_readers(tmp_path):
    objects, *_ = _two_hosts(tmp_path)
    store = fleet_store.store_readers(StoreTapes(objects, label="store"))
    store.pump(Reducer())
    status = store.status()
    assert status["label"] == "store" and status["sources"] == 2 and status["failing"] == 0
    assert store.objects is objects and len(store.sources) == 2


class BrokenList(DirectoryObjectStore):
    def list(self, prefix, **kw):
        if prefix.startswith("fleet/log"):
            raise OSError("list refused")
        return super().list(prefix, **kw)


def test_a_fleet_log_read_failure_shows_in_the_store_status(tmp_path):
    store = fleet_store.store_readers(StoreTapes(BrokenList(str(tmp_path)), label="store"))
    assert store.pump(Reducer()) == 0
    status = store.status()
    assert status["failing"] == 1 and "list refused" in status["last_error"]


def test_merge_writes_one_time_ordered_log_without_duplicates(tmp_path, capsys):
    objects, a, b, local = _two_hosts(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(a[0]) + "\nnot json\n")
    out = tmp_path / "merged.jsonl"
    assert fleet_store.main(["merge", "--store", str(tmp_path / "store"), "--fleet-log", str(local),
                             "--fleet-log", str(other), "--out", str(out)]) == 0
    assert _records(out) == a + b
    assert fleet_store.main(["merge", "--fleet-log", str(other)]) == 0
    assert capsys.readouterr().out == json.dumps(a[0]) + "\n"


def test_merge_keeps_an_untimed_record_after_its_predecessor():
    rows = [{"event": "x", "time_unix": 5.0}, {"event": "untimed"}, {"event": "y", "time_unix": 1.0}]
    assert [r["event"] for r in fleet_store.merge_records([rows])] == ["y", "x", "untimed"]


def test_merge_fails_when_the_store_cannot_be_listed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("yeto.dashboard.store_tapes.open_objects", lambda store: BrokenList(str(tmp_path)))
    assert fleet_store.main(["merge", "--store", "s3://example/run"]) == 2
    assert "list refused" in capsys.readouterr().err


def test_a_dashboard_without_store_has_no_store_reader():
    from yeto.dashboard import cli

    assert cli._store(SimpleNamespace(store=None)) is None  # noqa: SLF001


def test_the_dashboard_store_reads_the_fleet_logs(tmp_path):
    from yeto.dashboard import cli

    objects, *_ = _two_hosts(tmp_path)
    store = cli._store(SimpleNamespace(store=str(tmp_path / "store"), store_poll=0))  # noqa: SLF001
    reducer = Reducer()
    assert store.pump(reducer) == 5 and "1" in reducer.islands


def test_the_writer_and_the_copy_run_concurrently(tmp_path):
    """Lines written while chunks upload: the store ends up with exactly the local bytes."""
    store = tmp_path / "store"
    log = from_env(tmp_path / "fleet.jsonl", {ISLAND: SimpleNamespace()})
    up = fleet_store.start_uploader(DirectoryObjectStore(str(store)), log.path, interval_s=0.001)
    log.uploader = up
    stop = threading.Event()

    def write():
        for i in range(200):
            log.event("island_ready", ISLAND, n=i)
        stop.set()

    t = threading.Thread(target=write)
    t.start()
    t.join()
    assert log.close() is True
    data = b"".join((store / k).read_bytes() for k in _store_keys(store))
    assert data == log.path.read_bytes()
