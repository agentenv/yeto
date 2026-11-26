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
from yeto.dashboard.reducer import _FLEET_LIFE, Reducer, _new_island
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
    assert fleet_store.incarnation().split("-p")[0] == folder.split("-p")[0]  # <host>-p<pid>-<nonce>-<UTC>
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
    args = _launch_args(tmp_path, role="island", learner=2)
    args.rl_checkpoint_store = str(store)
    log = manifest._dashboard_fleet_log(args, {ISLAND: SimpleNamespace()}, {ISLAND: ("job", None)},  # noqa: SLF001
                                        objects=DirectoryObjectStore(str(store)))
    assert log.uploader is not None and log.uploader.island == "l2"
    log.event("island_ready", ISLAND)
    assert log.close() is True
    assert [k.split("/")[2] for k in _store_keys(store)] == ["l2"]


# -- readers -----------------------------------------------------------------------------------

def _text(rows) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


A_ROWS = [{"event": "island_ready", "time_unix": 100.0, "island": ISLAND, "island_id": 1},
          {"event": "island_node_lost", "time_unix": 150.0, "island": ISLAND, "island_id": 1, "node": "b300-1"},
          {"event": "island_lost", "time_unix": 160.0, "island": ISLAND, "island_id": 1}]
B_ROWS = [{"event": "island_ready", "time_unix": 400.0, "island": ISLAND, "island_id": 1},
          {"event": "island_node_replaced", "time_unix": 401.0, "island": ISLAND, "island_id": 1, "spare": "b300-3"}]


def _two_hosts(tmp_path, *, a_stamp="20261127T010000Z", b_stamp="20261127T020000Z"):
    """A control plane that moved: host A's launcher (island ready, node lost, island lost),
    then host B's (ready again, a spare in). Host B's local file has only its own lines; the
    store has both folders. Host B's folder sorts first by name."""
    store = tmp_path / "store"
    objects = DirectoryObjectStore(str(store))
    objects.write(f"fleet/log/l1/hosta-p7-0a0a0a0a-{a_stamp}/00000000.jsonl", _text(A_ROWS[:2]))
    objects.write(f"fleet/log/l1/hosta-p7-0a0a0a0a-{a_stamp}/00000001.jsonl", _text(A_ROWS[2:]))
    objects.write(f"fleet/log/l1/a-hostb-p9-0b0b0b0b-{b_stamp}/00000000.jsonl", _text(B_ROWS))
    local = tmp_path / "fleet.jsonl"
    local.write_bytes(_text(B_ROWS))
    return objects, A_ROWS, B_ROWS, local


def _lifecycle(isl):
    return {k: isl[k] for k in ("fleet_state", "ready_ts", "last_ready_ts", "stop_ts", "lost_ts", "open_ts",
                                "closed_s")}


RUNNING_AGAIN = {"fleet_state": "ready", "ready_ts": 100.0, "last_ready_ts": 400.0, "stop_ts": None,
                 "lost_ts": 160.0, "open_ts": 400.0, "closed_s": pytest.approx(60.0)}


def test_the_moved_host_s_island_is_running_with_the_lost_host_s_history(tmp_path):
    """Local file of host B only (its run directory), store with A and B: the island runs
    (B's ready is the newest record), A's ready -> lost interval is billed, each record once."""
    objects, a, b, local = _two_hosts(tmp_path)
    reducer = Reducer()
    load_all(reducer, [str(local)])
    store = fleet_store.store_readers(StoreTapes(objects, label="store"))
    assert store.pump(reducer) == len(a)  # A's lines are new; B's are folded already
    assert _lifecycle(reducer.islands["1"]) == RUNNING_AGAIN
    assert reducer.duplicates == len(b)
    assert sum(1 for e in reducer.events if e["type"] == "island_node_lost") == 1
    only_store = Reducer()
    assert fleet_store.store_readers(StoreTapes(objects, label="store")).pump(only_store) == len(a) + len(b)
    assert _lifecycle(only_store.islands["1"]) == RUNNING_AGAIN
    assert {r["event"] for r in only_store.fleet_records} == {"island_ready", "island_lost"}


def test_a_folder_stamp_two_hours_off_does_not_change_the_state(tmp_path):
    """Host B's clock 2 h behind for its folder stamp: B's folder is read before A's; the
    records' own times still decide (the newest record, B's ready, sets the state)."""
    objects, a, b, local = _two_hosts(tmp_path, a_stamp="20261127T030000Z", b_stamp="20261127T010000Z")
    reducer = Reducer()
    assert fleet_store.store_readers(StoreTapes(objects, label="store")).pump(reducer) == len(a) + len(b)
    assert _lifecycle(reducer.islands["1"]) == RUNNING_AGAIN


LIFE = [{"event": "island_ready", "time_unix": 10.0, "island": ISLAND, "island_id": 1, "gpu": "B300"},
        {"event": "island_lost", "time_unix": 20.0, "island": ISLAND, "island_id": 1},
        {"event": "island_ready", "time_unix": 30.0, "island": ISLAND, "island_id": 1, "gpu": "H100"},
        {"event": "island_stop", "time_unix": 50.0, "island": ISLAND, "island_id": 1, "reason": "teardown"}]
HOST = ["host-a", "host-a", "host-b", "host-b"]  # the lost host wrote the first two, the new one the rest


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [2, 3, 0, 1], [2, 0, 3, 1], [2, 0, 1, 3], [0, 2, 3, 1]])
def test_the_lifecycle_does_not_depend_on_the_read_order(order):
    """Two sources (hosts) read in any interleaving, each in its own order."""
    want = Reducer()
    for r in LIFE:
        want.feed(dict(r))
    got = Reducer()
    for k in order:
        got.feed(dict(LIFE[k]), source=HOST[k])
    assert _lifecycle(got.islands["1"]) == _lifecycle(want.islands["1"])
    assert _lifecycle(want.islands["1"])["closed_s"] == pytest.approx(30.0)
    assert got.islands["1"]["gpu"] == "H100"  # the newest record's shape


def _main_fold(rows):
    """origin/main's fold of one source: every record in write order."""
    isl = {k: _new_island("1")[k] for k in _FLEET_LIFE}
    for r in rows:
        Reducer._fold_fleet_lifecycle(isl, r["event"], r.get("time_unix"))  # noqa: SLF001
    return isl


@pytest.mark.parametrize("source", [None, "fleet.jsonl"])
def test_a_clock_stepped_back_in_one_file_folds_in_write_order(source):
    """NTP steps the launcher host's clock back between ready and lost: one file, written in
    order. The island is lost, as on main (not ready and billed on)."""
    rows = [{"event": "island_ready", "time_unix": 100.0, "island": ISLAND, "island_id": 1, "gpu": "B300"},
            {"event": "island_lost", "time_unix": 90.0, "island": ISLAND, "island_id": 1},
            {"event": "island_ready", "time_unix": 95.0, "island": ISLAND, "island_id": 1, "gpu": "H100"},
            {"event": "island_stop", "time_unix": 99.0, "island": ISLAND, "island_id": 1}]
    for n in range(1, len(rows) + 1):
        reducer = Reducer()
        for r in rows[:n]:
            reducer.feed(dict(r), **({} if source is None else {"source": source}))
        assert _lifecycle(reducer.islands["1"]) == _lifecycle(_main_fold(rows[:n])), n
    assert reducer.islands["1"]["fleet_state"] == "stop" and reducer.islands["1"]["gpu"] == "H100"
    lost = Reducer()
    for r in rows[:2]:
        lost.feed(dict(r), source="fleet.jsonl")
    assert lost.islands["1"]["fleet_state"] == "lost" and lost.islands["1"]["open_ts"] is None
    # another host's older interval read after it: replayed, each record at its own time
    for r in ({"event": "island_ready", "time_unix": 50.0, "island": ISLAND, "island_id": 1},
              {"event": "island_lost", "time_unix": 60.0, "island": ISLAND, "island_id": 1}):
        lost.feed(dict(r), source="old-host")
    assert lost.islands["1"]["fleet_state"] == "lost" and lost.islands["1"]["lost_ts"] == 90.0
    assert lost.islands["1"]["closed_s"] == pytest.approx(10.0) and lost.islands["1"]["ready_ts"] == 50.0


def test_a_late_copy_is_folded_in_at_its_place_and_the_rest_replayed():
    """A record of another source before several folded ones: the state from there on is the
    time-ordered fold (the replay starts at the record, not at the island's first one)."""
    rows = [{"event": e, "time_unix": t, "island": ISLAND, "island_id": 1}
            for e, t in (("island_ready", 10.0), ("island_lost", 20.0), ("island_ready", 30.0),
                         ("island_lost", 40.0), ("island_ready", 60.0))]
    want = Reducer()
    for r in rows:
        want.feed(dict(r))
    for late in range(len(rows)):
        got = Reducer()
        for k, r in enumerate(rows):
            if k != late:
                got.feed(dict(r), source="new-host")
        got.feed(dict(rows[late]), source="old-host")
        assert _lifecycle(got.islands["1"]) == _lifecycle(want.islands["1"]), late
    assert want.islands["1"]["closed_s"] == pytest.approx(20.0) and want.islands["1"]["open_ts"] == 60.0


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


# -- folders of one host ---------------------------------------------------------------------------

def test_two_launchers_of_one_host_pid_and_second_get_two_folders():
    from yeto.dashboard.store_tapes import launch_started

    one = fleet_store.incarnation(host="h", pid=1, now=1795700000.0)
    two = fleet_store.incarnation(host="h", pid=1, now=1795700000.0)
    assert one != two and one.startswith("h-p1-") and two.startswith("h-p1-")
    assert launch_started(one) == launch_started(two) == 1795700000.0  # the start still ends the name
    assert fleet_store.incarnation(host="h", pid=1, now=1795700000.0, nonce="ab") == "h-p1-ab-20261126T133320Z"


def test_a_quick_restart_s_chunks_are_read_after_the_first_launcher_s(tmp_path):
    """A launcher restarted on the same host in the same second writes seq 0 again, under its
    own folder: a dashboard that already consumed the first folder's chunks reads it (one
    cursor per folder)."""
    objects = DirectoryObjectStore(str(tmp_path / "store"))
    first, second = (fleet_store.incarnation(host="h", pid=1, now=1795700000.0) for _ in range(2))
    rows = [{"event": "island_ready", "time_unix": 10.0 + i, "island": ISLAND, "island_id": 1} for i in range(3)]
    objects.write(f"fleet/log/l1/{first}/00000000.jsonl", _text(rows[:1]))
    objects.write(f"fleet/log/l1/{first}/00000001.jsonl", _text(rows[1:2]))
    reader = fleet_store.log_reader(objects)
    reducer = Reducer()
    assert reader.pump(reducer) == 2
    objects.write(f"fleet/log/l1/{second}/00000000.jsonl", _text(rows[2:]))
    assert reader.pump(reducer) == 1 and len(reader.cursors) == 2
    assert reducer.islands["1"]["last_ready_ts"] == 12.0


# -- a reused run directory ------------------------------------------------------------------------

def test_the_copy_starts_where_this_run_s_lines_start(tmp_path):
    path = tmp_path / "runs" / "r" / "fleet.jsonl"
    assert fleet_store.copy_start(path, "s3://store/a", run=1.0) == 0  # a new run directory
    path.write_bytes(_text(A_ROWS))
    size = path.stat().st_size
    assert fleet_store.copy_start(path, "s3://store/a", learner_id=1, run=1.0) == 0  # the run restarted
    assert fleet_store.read_copy_record(path)["learner_id"] == 1
    assert fleet_store.copy_start(path, "s3://store/a", run=2.0) == size  # another run, same store: from the end
    assert fleet_store.copy_start(path, "s3://store/b", run=2.0) == size  # another store: from the end
    path.write_bytes(_text(A_ROWS + B_ROWS))
    assert fleet_store.copy_start(path, "s3://store/b", run=2.0) == size  # that run's start is kept
    assert fleet_store.copy_start(path, "s3://store/b") == path.stat().st_size  # no run identity: the end
    other = tmp_path / "old" / "fleet.jsonl"
    other.parent.mkdir()
    other.write_bytes(_text(A_ROWS))  # written before the copy record existed
    assert fleet_store.copy_start(other, "s3://store/a", run=1.0) == other.stat().st_size


def _new_run(name="run-i1"):
    """A run of ``name`` in the (test's) run registry: its identity, new each time."""
    from yeto import runs

    runs.create_run(name, {})
    started = fleet_store.run_identity(name)
    assert started is not None
    return started


def _launch(tmp_path, store=None, *, environ=None, monkeypatch=None):
    """One island launch of learner 1 (``store``: it copies its fleet.jsonl there), one record."""
    args = _launch_args(tmp_path, role="island", learner=1)
    args.rl_checkpoint_store = None if store is None else str(store)
    log = manifest._dashboard_fleet_log(args, {ISLAND: SimpleNamespace()}, {ISLAND: ("job", None)},  # noqa: SLF001
                                        objects=None if store is None else DirectoryObjectStore(str(store)))
    log.event("island_ready", ISLAND)
    return log


def _uploaded(store) -> bytes:
    return b"".join((store / k).read_bytes() for k in _store_keys(store)) if store.exists() else b""


def _folders(store) -> list[bytes]:
    """Each launcher folder's bytes, in a fixed order (folder names carry a random nonce)."""
    out: dict[str, bytes] = {}
    for k in _store_keys(store) if store.exists() else ():
        folder = k.rsplit("/", 1)[0]
        out[folder] = out.get(folder, b"") + (store / k).read_bytes()
    return sorted(out.values())


def test_yeto_down_after_a_launch_without_a_copy_leaves_the_earlier_store_alone(tmp_path, monkeypatch):
    """run 1 copies into S1; run 2 of the name keeps its fleet.jsonl local and its launcher is
    killed; yeto down closes run 2's island: its island_stop stays out of S1."""
    from yeto import cli

    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    s1 = tmp_path / "s1"
    _new_run()
    assert _launch(tmp_path, s1).close() is True
    before = _uploaded(s1)
    _new_run()
    log = _launch(tmp_path)  # killed: never closed
    assert fleet_store.read_copy_record(log.path) is None  # the record names no store any more
    assert cli._close_fleet_islands("run-i1") == [ISLAND]  # noqa: SLF001
    assert _records(log.path)[-1]["event"] == "island_stop" and _uploaded(s1) == before


def test_yeto_down_does_not_use_another_run_s_record(tmp_path, monkeypatch):
    """A record left by an earlier run of the name (its launch never voided it): yeto down of
    this run copies nothing into that run's store."""
    from yeto import cli

    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    s1 = tmp_path / "s1"
    path = tmp_path / "run-i1" / "fleet.jsonl"
    path.parent.mkdir()
    fleet_store.copy_start(path, str(s1), learner_id=1, run=_new_run())
    _new_run()
    path.write_bytes(_text(B_ROWS[:1]))  # this run's open island
    monkeypatch.setattr("yeto.launch.island_lifecycle.store_objects", lambda uri: DirectoryObjectStore(uri))
    assert cli._close_fleet_islands("run-i1") == [ISLAND]  # noqa: SLF001
    assert not s1.exists()


def test_a_launch_without_a_copy_between_two_copies_into_one_store_stays_local(tmp_path, monkeypatch):
    """S2, a launch that keeps its lines local, S2 again: the middle run's lines never reach S2."""
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    s2 = tmp_path / "s2"
    _new_run()
    assert _launch(tmp_path, s2).close() is True
    first = _folders(s2)
    _new_run()
    middle = _launch(tmp_path)
    middle.event("island_stop", ISLAND)
    assert middle.close() is True
    size = middle.path.stat().st_size
    _new_run()
    third = _launch(tmp_path, s2)
    assert third.close() is True
    assert _folders(s2) == sorted([*first, third.path.read_bytes()[size:]])


def test_another_run_s_record_for_the_same_store_starts_at_the_end(tmp_path, monkeypatch):
    """The record of an earlier run for this store, not voided (that run's launch was older
    or died first): a new run starts at the file's end all the same."""
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    s2 = tmp_path / "s2"
    path = tmp_path / "run-i1" / "fleet.jsonl"
    path.parent.mkdir()
    fleet_store.copy_start(path, str(s2), learner_id=1, run=_new_run())
    path.write_bytes(_text(A_ROWS))  # lines of runs that are not the next one
    _new_run()
    log = _launch(tmp_path, s2)
    assert log.close() is True
    assert _uploaded(s2) == log.path.read_bytes()[len(_text(A_ROWS)):]


def test_a_restarted_launcher_of_the_run_goes_on_from_the_run_s_start(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    store = tmp_path / "s"
    _new_run()
    assert _launch(tmp_path, store).close() is True
    again = _launch(tmp_path, store)  # the same run (no new registry entry)
    assert fleet_store.read_copy_record(again.path)["start"] == 0 and again.close() is True


def test_a_launch_with_the_copy_turned_off_voids_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    store = tmp_path / "s"
    _new_run()
    assert _launch(tmp_path, store).close() is True
    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "0")
    log = _launch(tmp_path, store)
    assert log.uploader is None and fleet_store.read_copy_record(log.path) is None
    assert json.loads(fleet_store.copy_record_path(log.path).read_text())["start"] == log.path.stat().st_size \
        - len(_text([_records(log.path)[-1]]))


def test_a_launch_without_a_record_writes_none(tmp_path, monkeypatch):
    """A local-only launch of a run directory that never had a copy (one trainer island, the
    customer path): no fleet.store.json appears."""
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    _new_run()
    log = _launch(tmp_path)
    assert log.close() is True and not fleet_store.copy_record_path(log.path).exists()


def test_a_reused_run_directory_uploads_only_this_run_s_lines(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / run)
    (tmp_path / "run-i1").mkdir()
    old = _text([{"event": "island_ready", "time_unix": 1.0, "island": ISLAND, "island_id": 0}])
    (tmp_path / "run-i1" / "fleet.jsonl").write_bytes(old)  # an earlier run of this name
    store = tmp_path / "store"
    args = _launch_args(tmp_path, role="island", learner=1)
    args.rl_checkpoint_store = str(store)
    log = manifest._dashboard_fleet_log(args, {ISLAND: SimpleNamespace()}, {ISLAND: ("job", None)},  # noqa: SLF001
                                        objects=DirectoryObjectStore(str(store)))
    log.event("island_ready", ISLAND)
    assert log.close() is True
    uploaded = b"".join((store / k).read_bytes() for k in _store_keys(store))
    assert uploaded == log.path.read_bytes()[len(old):] and b'"time_unix": 1.0' not in uploaded


# -- yeto down -------------------------------------------------------------------------------------

def _down_run(tmp_path, monkeypatch, *, record=True, store_objects=None):
    from yeto import cli

    monkeypatch.setattr(cli.runs, "run_dir", lambda run: tmp_path / run)
    path = tmp_path / "r" / "fleet.jsonl"
    path.parent.mkdir()
    path.write_bytes(_text(B_ROWS[:1]))  # the launcher was killed: the island is still open
    store = tmp_path / "store"
    cli.runs.create_run("r", {})
    if record:
        fleet_store.copy_start(path, str(store), learner_id=1, run=fleet_store.run_identity("r"))
    if store_objects is not None:
        monkeypatch.setattr("yeto.launch.island_lifecycle.store_objects", lambda uri: store_objects)
    return cli, path, store


def test_yeto_down_copies_the_island_stop_lines_into_the_store(tmp_path, monkeypatch):
    cli, path, store = _down_run(tmp_path, monkeypatch)
    assert cli._close_fleet_islands("r") == [ISLAND]  # noqa: SLF001
    keys = _store_keys(store)
    assert len(keys) == 1 and keys[0].startswith("fleet/log/l1/")
    (rec,) = [json.loads(line) for line in (store / keys[0]).read_text().splitlines()]
    assert rec["event"] == "island_stop" and rec["island_id"] == 1


def test_yeto_down_of_a_local_only_run_copies_nothing(tmp_path, monkeypatch, capsys):
    cli, path, store = _down_run(tmp_path, monkeypatch, record=False)
    assert cli._close_fleet_islands("r") == [ISLAND]  # noqa: SLF001
    assert not store.exists() and "WARN" not in capsys.readouterr().err
    assert fleet_store.upload_appended(path, 0) is True  # no copy record: nothing to copy, not a failure


def test_yeto_down_with_a_failing_store_still_closes_the_islands(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fleet_store, "FINAL_UPLOAD_S", 0.3)  # the bounded final upload, short here
    cli, path, store = _down_run(tmp_path, monkeypatch, store_objects=FailingObjects())
    assert cli._close_fleet_islands("r") == [ISLAND]  # noqa: SLF001
    assert _records(path)[-1]["event"] == "island_stop"
    assert "WARN" in capsys.readouterr().err


# -- arguments and status --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 0.0, -1.0])
def test_an_explicit_interval_must_be_positive(tmp_path, bad):
    with pytest.raises(ValueError, match="interval_s must be > 0"):
        fleet_store.start_uploader(DirectoryObjectStore(str(tmp_path)), tmp_path / "f.jsonl", interval_s=bad)


class _Status:
    def __init__(self, **status):
        self._status = {"label": "x", "sources": 1, "requests": 1, "errors": 0, "failing": 0, "last_error": None,
                        "last_error_at": None, "last_ok_at": None, "polled": True, **status}
        self.objects = object()

    def status(self):
        return dict(self._status)

    def pump(self, reducer):
        return 0


def test_the_store_status_is_ok_since_the_oldest_clean_poll():
    both = fleet_store.StoreReaders(_Status(last_ok_at=20.0), _Status(last_ok_at=10.0))
    assert both.status()["last_ok_at"] == 10.0
    one_never = fleet_store.StoreReaders(_Status(last_ok_at=20.0), _Status(last_ok_at=None))
    assert one_never.status()["last_ok_at"] is None
    failed = fleet_store.StoreReaders(_Status(last_error="old", last_error_at=1.0, errors=1),
                                      _Status(last_error="new", last_error_at=5.0, errors=2, failing=1))
    status = failed.status()
    assert (status["last_error"], status["last_error_at"], status["errors"], status["failing"]) == ("new", 5.0, 3, 1)
    assert fleet_store.StoreReaders(_Status(), _Status(polled=False)).status()["polled"] is False
