"""``yeto run up`` runtime of an island fleet, island by island (:mod:`yeto.rl.up_fleet_runtime`):
the first-group wait, starvation and release stall per island, the data end per island,
the fleet's spend and budget, per-island trips (``FleetStops``) and the restart init.
Every island's ``trainer<k>/`` is a local directory store."""

from __future__ import annotations

import json
import types

import pytest

from tests.test_run_up_fleet import _cfg, _write, files  # noqa: F401
from yeto.breakers.events import clear_stop_reason, current_stop_reason
from yeto.rl import up
from yeto.rl.engine.object_store import DirectoryObjectStore

STORE = "s3://b/runs/r"
MIN = 60.0


class Clock:
    def __init__(self):
        self.t = 1_000_000.0
        self.hooks = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s
        for hook in list(self.hooks):
            hook(self.t)


@pytest.fixture
def fleet_run(files, monkeypatch, tmp_path):  # noqa: F811
    """A two-island fleet (_cfg: island 0 east = i0-vllm + i0-sglang, island 1 west =
    i1-sglang) on a local store; ``build(**edits)`` returns the Run."""
    clock = Clock()
    monkeypatch.setattr(up, "time", types.SimpleNamespace(time=clock.time, monotonic=clock.time, sleep=clock.sleep,
                                                         gmtime=lambda *a: __import__("time").gmtime(*a),
                                                         strftime=lambda *a: __import__("time").strftime(*a)))
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    root = tmp_path / "bucket"
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()

    def objects_of(self, store):
        rel = str(store)[len(STORE):].strip("/")
        return DirectoryObjectStore(root / rel) if rel else DirectoryObjectStore(root)

    monkeypatch.setattr(up.Run, "_objects", lambda self: DirectoryObjectStore(root))
    monkeypatch.setattr(up.Run, "_objects_of", objects_of)
    monkeypatch.setattr("yeto.launch.nc.pool_run.pool_process", lambda cfg: {"alive": True})
    clear_stop_reason()

    def build(cfg=None, **extra):
        cfg = cfg or _cfg(files, **extra)
        cfg.setdefault("env", {})["YETO_NC_STATE_DIR"] = str(nc_dir)
        run = up.Run(up.load_config(_write(files, cfg)))
        run.clock = clock
        run.root = root
        run.nc_dir = nc_dir
        run.calls = []
        run._cmd = lambda argv, *, log, timeout=None: run.calls.append(list(argv)) or 0
        return run

    yield build
    clear_stop_reason()


def _store(run, k):
    return DirectoryObjectStore(run.root / f"trainer{k}")


def _groups(run, k, shard, n, *, end=False, start=0):
    from yeto.rl.engine.buffer_payload import encode_group
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
    from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

    writer = FileSampleBuffer(_store(run, k), batch_size=1, index=shard_index(shard))
    for i in range(start, start + n):
        ref = writer.write_payload(f"{shard}-{i}", encode_group([{"i": i}]))
        writer.append(SampleIndexEntry(f"{shard}-{i}", 1, ref, 8, float(i + 1)))
    if end:
        writer.mark_data_end()


def _pool_state(run, shard_base, *, finished=False, members=()):
    doc = {"members": list(members)}
    if finished:
        doc["finished"] = "sampler exited 0 on the data end"
    (run.nc_dir / f"pool-{shard_base}.json").write_text(json.dumps(doc))


def _ledger(run, *, slots, trained=(), running=()):
    """The fleet's shared task ledger: ``trained`` slots settled, ``running`` = [(slot, holder)]."""
    from yeto.rl.engine import store_dispense as sd

    ledger = sd.ledger_objects(DirectoryObjectStore(run.root / "fleet" / "dispense"))
    ledger.write("task-dispense/plan.json", json.dumps({"slots": slots, "rows": slots}).encode())
    for p in trained:
        ledger.write(f"task-dispense/done/{sd._pkey(p)}.trained", b"{}")
    for p, holder in running:
        ledger.write(f"task-dispense/claims/{sd._pkey(p)}/{sd._akey(0)}",
                     json.dumps({"holder": holder, "lease_s": 1e9}).encode())


def _events(run):
    path = run.dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _out(capsys):
    return [line.split(" [rl up] ", 1)[1] for line in capsys.readouterr().out.splitlines() if " [rl up] " in line]


def _stops(run):
    return [c for c in run.calls if "pool-stop" in c]


# -- the first group (2.6) -------------------------------------------------------------------

def test_the_first_group_wait_waits_only_for_the_island_without_one(fleet_run, capsys):
    run = fleet_run(first_group={"timeout_s": 3600, "poll_s": 60})
    _groups(run, 0, "vl-i0-0", 1)
    polls = []
    run.clock.hooks.append(lambda t: polls.append(t) or (len(polls) == 3 and _groups(run, 1, "sg-i1-0", 2)))
    assert run.wait_first_group() is None
    out = _out(capsys)
    waits = [line for line in out if line.startswith("first group: waiting")]
    assert waits == ["first group: waiting for island 1 (west) (island 0 (east) has a group)"]  # logged once
    assert "first group: island 0 (east): 1 object(s) in s3://b/runs/r/trainer0" in out
    assert "first group: island 1 (west): 2 object(s) in s3://b/runs/r/trainer1" in out
    assert len(polls) == 3
    assert run.state["steps"]["first_group"]["islands"] == [0, 1]
    assert run.state["steps"]["first_group"]["objects"] == 3
    assert run.buffer_objects() == 0  # the run store's root holds no buffer: why the wait is per island


def test_an_island_past_its_first_group_timeout_stops_alone(fleet_run, capsys):
    run = fleet_run(first_group={"timeout_s": 300, "poll_s": 60})
    _groups(run, 0, "sg-i0-1", 1)
    assert run.wait_first_group() is None  # island 0 has a group: the fleet's trainers start
    assert [c[-1] for c in _stops(run)] == [run.pools[2].config]  # island 1's pool only
    trips = [e for e in _events(run) if e.get("name") == "run.first_group"]
    assert len(trips) == 1 and trips[0]["scope"] == {"island": "1", "island_name": "west"}
    assert trips[0]["stop_reason"] == "breaker:run.first_group"
    assert run.fleet_rt().stops().stop_reason(1) == "breaker:run.first_group"
    assert run.fleet_rt().stops().stop_reason(0) is None
    assert run.state["fleet_stops"]["islands"][1]["stop_reason"] == "breaker:run.first_group"
    assert run.state["fleet_island_stops"]["1"]["pools"] == ["i1-sglang"]
    assert current_stop_reason() is None  # an island's trip is not the run up's stop reason
    assert any(line.startswith("CRITICAL fleet_island_stop: island 1 (west): run.first_group") for line in _out(capsys))


def test_no_island_with_a_group_ends_the_wait_as_before(fleet_run):
    run = fleet_run(first_group={"timeout_s": 120, "poll_s": 60})
    assert run.wait_first_group() == "first group timed out"
    assert run.fleet_all_stopped() == "breaker:run.first_group"
    assert len(_stops(run)) == 3  # both islands' pools


def test_a_restarted_run_up_keeps_the_islands_an_earlier_trip_stopped(fleet_run):
    run = fleet_run(first_group={"timeout_s": 120, "poll_s": 60})
    _groups(run, 0, "sg-i0-0", 1)
    run.wait_first_group()
    again = up.Run(run.cfg)
    assert again.fleet_rt().stops().stop_reason(1) == "breaker:run.first_group"
    assert again.fleet_rt().active() == [0]


# -- starvation and release stall per island (2.6, 3.5 wiring) --------------------------------

def test_a_starving_island_stops_alone_and_the_fleet_goes_on(fleet_run, capsys):
    run = fleet_run(breakers={"run": {"starvation": {"alert_s": 1800, "stop_s": 3600, "unhealthy_s": 600,
                                                     "check_s": 60}}})
    run.state["procs"] = {"trainer": {"pid": 1, "log": "/nonexistent"}}
    _pool_state(run, "vl-i0", members=[{"ticket": "a", "state": "ready"}])
    _pool_state(run, "sg-i0", members=[{"ticket": "b", "state": "ready"}])
    _pool_state(run, "sg-i1", members=[])  # island 1: no sampler at all
    n = 0
    for _ in range(80):
        n += 1
        _groups(run, 0, "vl-i0-0", 1, start=n)  # island 0 keeps writing
        assert run.starvation_check(run.clock.t) is None
        run.clock.t += 60
    rt = run.fleet_rt()
    assert rt.stops().stop_reason(1) == "breaker:run.starvation" and rt.stops().stop_reason(0) is None
    assert run.fleet_all_stopped() is None
    assert [c[-1] for c in _stops(run)] == [run.pools[2].config]  # island 1's pool only
    out = _out(capsys)
    assert any(line.startswith("CRITICAL trainer_starvation_alert: island 1 (west):") for line in out)
    assert not any("island 0 (east)" in line and "starvation" in line for line in out)
    warns = [e for e in _events(run) if e["name"] == "run.starvation" and e["state"] == "warn"]
    assert [e["scope"]["island"] for e in warns] == ["1"]


def test_every_island_stopped_ends_the_launch(fleet_run, monkeypatch):
    run = fleet_run()
    rt = run.fleet_rt()
    for k in (0, 1):
        rt.trip_island(k, "run.starvation", value=7200, limit=3600, why="no group")
    assert run.fleet_all_stopped() == "breaker:run.starvation"
    flagged = []
    monkeypatch.setattr(run, "flag_stop", lambda name, key, event, why, **kw: flagged.append((name, key, event, why)))
    run.fleet_stop(run.fleet_all_stopped())
    assert flagged == [("run.starvation", "fleet_stop", "fleet_all_islands_stopped",
                        "every island stopped (island 0 (east): breaker:run.starvation, island 1 (west): "
                        "breaker:run.starvation)")]


def test_a_release_stall_is_judged_per_island(fleet_run, capsys):
    run = fleet_run(breakers={"run": {"release_stall": {"limit_s": 1800, "check_s": 60, "action": "stop"}}})
    run.state["procs"] = {"trainer": {"pid": 1, "started": "x"}}
    _store(run, 0).write("policy-releases/LATEST.json", json.dumps({"seq": 0, "path": "r0-v0"}).encode())
    _store(run, 1).write("policy-releases/LATEST.json", json.dumps({"seq": 0, "path": "r0-v0"}).encode())
    for i in range(1, 60):
        _store(run, 0).write("policy-releases/LATEST.json", json.dumps({"seq": i, "path": f"r{i}-v{i}"}).encode())
        assert run.release_stall_check(run.clock.t) is None
        run.clock.t += 60
    rt = run.fleet_rt()
    assert rt.stops().stop_reason(1) == "breaker:run.release_stall" and rt.stops().stop_reason(0) is None
    assert any(line.startswith("CRITICAL trainer_release_stall_alert: island 1 (west):") for line in _out(capsys))
    assert not run.latest_release()[0]  # the run store's root holds no release


# -- the data end per island (3.3) -------------------------------------------------------------

def test_one_island_running_out_of_data_first(fleet_run, capsys):
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    run = fleet_run()
    _ledger(run, slots=6, trained=range(5), running=[(5, "sg-i1-0")])  # island 1 still holds slot 5
    _groups(run, 0, "vl-i0-0", 3, end=True)
    _groups(run, 1, "sg-i1-0", 2)
    for base in ("vl-i0", "sg-i0"):
        _pool_state(run, base, finished=True, members=[{"ticket": "t", "state": "gone"}])
    _pool_state(run, "sg-i1", members=[{"ticket": "u", "state": "ready"}])
    assert not run.seal_data_end()
    rt = run.fleet_rt()
    assert rt.ended(0) and not rt.ended(1) and "data_ended" not in run.state
    assert sorted(run.state["fleet_data_ended"]["0"]["sealed"]) == ["sg-i0-0", "sg-i0-1", "vl-i0-1", "vl-i0-2"]
    trainer0 = FileSampleBuffer(_store(run, 0), batch_size=1, merge_shards=True,
                                expected_shards=run.fleet_plan().islands[0].buffer_shards())
    trainer1 = FileSampleBuffer(_store(run, 1), batch_size=1, merge_shards=True,
                                expected_shards=run.fleet_plan().islands[1].buffer_shards())
    trainer0.refresh()
    trainer1.refresh()
    assert trainer0.data_ended and len(trainer0) == 3
    assert not trainer1.data_ended  # island 1 goes on
    assert rt.active() == [1]
    assert any(line.startswith("data end: island 0 (east): every pool of it finished") for line in _out(capsys))
    # island 1's pool finishes its last slot: the fleet's data ended
    _ledger(run, slots=6, trained=range(6))
    _pool_state(run, "sg-i1", finished=True, members=[{"ticket": "u", "state": "gone"}])
    assert run.seal_data_end()
    trainer1.refresh()
    assert trainer1.data_ended and run.state["data_ended"]["islands"] == 2
    assert any(line.startswith("data end: every island's data ended") for line in _out(capsys))


def test_an_island_holding_a_slot_of_the_shared_ledger_is_not_ended(fleet_run, capsys):
    run = fleet_run()
    _ledger(run, slots=3, trained=range(2), running=[(2, "sg-i0-1")])  # island 0's own sampler holds it
    for base in ("vl-i0", "sg-i0"):
        _pool_state(run, base, finished=True)
    _pool_state(run, "sg-i1", members=[{"ticket": "u", "state": "ready"}])
    assert not run.seal_data_end() and not run.fleet_rt().ended(0)
    assert any("CRITICAL data_end_incomplete: island 0 (east)" in line for line in _out(capsys))
    _ledger(run, slots=4, trained=range(2))  # queued / unclaimed slots: nothing is settled for anyone
    assert not run.fleet_rt().settled(0)


def test_island_ledgers_settle_each_island_alone(fleet_run, files):  # noqa: F811
    from yeto.rl.engine import store_dispense as sd

    cfg = _cfg(files)
    cfg["fleet"]["dispense"] = "island"
    run = fleet_run(cfg)
    assert [run.dispense_ledger(i)[0] for i in range(3)] == [f"{STORE}/trainer0"] * 2 + [f"{STORE}/trainer1"]
    # one ledger per island (dispense: island): island 0's settled, island 1's not
    for k, (slots, done) in enumerate(((2, 2), (2, 1))):
        ledger = sd.ledger_objects(_store(run, k))
        ledger.write("task-dispense/plan.json", json.dumps({"slots": slots, "rows": slots}).encode())
        for p in range(done):
            ledger.write(f"task-dispense/done/{sd._pkey(p)}.trained", b"{}")
    assert run.fleet_rt().settled(0) and not run.fleet_rt().settled(1)


def test_an_island_with_a_miles_sampler_seals_only_its_pools(fleet_run, files, capsys):  # noqa: F811
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    cfg = _cfg(files)
    cfg["islands"][1]["samplers"].insert(0, {"kind": "miles", "name": "miles", "gpu": "modal:1x1xl40s",
                                             "usd_per_h": 1.5})
    run = fleet_run(cfg)
    west = run.fleet_plan().islands[1]
    (miles,) = west.miles_samplers
    assert west.buffer_shards() == [miles.shard, "sg-i1-0", "sg-i1-1"]
    _ledger(run, slots=2, trained=range(2))
    _pool_state(run, "sg-i1", finished=True)
    _groups(run, 1, miles.shard, 1)  # the Miles sampler island still writes
    assert not run.seal_data_end()
    assert run.state["fleet_pools_ended"]["1"]["sealed"] == ["sg-i1-0", "sg-i1-1"]
    assert not run.fleet_rt().ended(1) and 1 in run.fleet_rt().active()
    reader = FileSampleBuffer(_store(run, 1), batch_size=1, merge_shards=True, expected_shards=west.buffer_shards())
    reader.refresh()
    assert not reader.data_ended and len(reader) == 1  # the Miles shard goes on
    assert any("keep feeding its trainer" in line for line in _out(capsys))
    assert "sg-i1-0" in run.data_end_shards() and miles.shard not in run.data_end_shards()
    assert not run.seal_data_end()  # sealed once: nothing logged or written again
    assert not any(line.startswith("data end:") for line in _out(capsys))
    # a trip of that island stops its pools; its Miles sampler keeps feeding its trainer
    run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="x")
    assert any("Miles sampler island(s) i1-miles are part of the launch" in line for line in _out(capsys))


def test_the_data_end_shards_are_every_islands(fleet_run):
    run = fleet_run()
    assert run.data_end_shards() == ["vl-i0-0", "vl-i0-1", "vl-i0-2", "sg-i0-0", "sg-i0-1", "sg-i1-0", "sg-i1-1"]
    _pool_state(run, "vl-i0", finished=True)
    _pool_state(run, "sg-i0", finished=True)
    assert not run.pools_data_ended()  # island 1's pool has no state
    _pool_state(run, "sg-i1", finished=True)
    assert run.pools_data_ended()


# -- spend and budget (3.4) --------------------------------------------------------------------

def test_the_fleet_spend_counts_pools_each_islands_trainer_and_the_syncer(fleet_run):
    run = fleet_run(syncer={"usd_per_h": 0.5}, budget={"usd": 1000, "wall_h": 100, "nc_usd_per_h": 0})
    now = run.clock.t
    run.state.update({"run_t0": now - 7200, "trainer_t0": now - 3600,
                      "ups": [{"pid": 1, "start": now - 7200, "seen": now, "end": None}]})
    _pool_state(run, "vl-i0", members=[{"state": "ready", "granted": now - 7200}])   # 2 h x $4
    _pool_state(run, "sg-i0", members=[{"state": "ready", "granted": now - 3600}])   # 1 h x $2
    _pool_state(run, "sg-i1", members=[{"state": "gone", "granted": now - 7200, "charged_to": now - 5400}])  # 0.5 h x $2
    spend = run.fleet_rt().spend()
    assert spend.pools_usd == pytest.approx(8 + 2 + 1)
    assert spend.islands_usd == pytest.approx((8.0, 8.0))  # one launch: each island's trainer 1 h
    assert spend.syncer_usd == pytest.approx(0.5)
    assert run.spend_usd() == pytest.approx(27.5)
    assert run.status()["fleet"]["spend"] == {"usd": 27.5, "pools_usd": 11.0, "islands_usd": [8.0, 8.0],
                                              "syncer_usd": 0.5}


def test_the_budget_stops_the_whole_fleet(fleet_run):
    run = fleet_run(syncer={"usd_per_h": 1.0}, budget={"usd": 20, "wall_h": 100, "nc_usd_per_h": 0})
    now = run.clock.t
    run.state.update({"t0": now - 3600, "run_t0": now - 3600, "trainer_t0": now - 3600,
                      "ups": [{"pid": 1, "start": now - 3600, "seen": now, "end": None}]})
    for base in ("vl-i0", "sg-i0", "sg-i1"):
        _pool_state(run, base, members=[{"state": "ready", "granted": now - 3600}])
    why = run.breach(trainer=False)
    assert why == "budget $25.00"  # pools 8 + trainers 16 + syncer 1
    assert run.state["breaker"]["stop_reason"] == "breaker:run.budget"
    rt = run.fleet_rt()
    assert rt.stops().fleet_stop_reason == "breaker:run.budget"
    assert [rt.stops().stop_reason(k) for k in (0, 1)] == ["breaker:run.budget"] * 2
    (event,) = [e for e in _events(run) if e["name"] == "run.budget"]
    assert event["stop_reason"] == "breaker:run.budget" and event["value"] == 25.0 and event["limit"] == 20
    assert '"islands_usd": [8.0, 8.0]' in event["detail"] and '"syncer_usd": 1.0' in event["detail"]
    assert run.limit_breach(why)  # on_breach: the launch (every island and the syncer) stops


# -- breakers.pools under the compiled names (3.4) ---------------------------------------------

def test_breakers_pools_take_the_compiled_pool_names(fleet_run):
    run = fleet_run(breakers={"pools": {"i1-sglang": {"engine_stall": {"limit_s": 1200}}}})
    assert run.fleet_preflight() is None
    # the override reaches island 1's copy of the pool only (its merged pool config)
    assert run.pools[2].config.endswith("pools/i1-sglang.json")
    assert json.dumps(run.pools[2].load()).count("1200") == 1 and "1200" not in json.dumps(run.pools[1].load())
    run = fleet_run(breakers={"pools": {"sglang": {"engine_stall": {"limit_s": 1200}}}})
    problem = run.fleet_preflight()
    assert problem.startswith("breakers.pools.sglang: no pool of the fleet is named 'sglang'; a fleet's pools run as "
                              "i<k>-<name> (this one as i0-sglang / i1-sglang)")


def test_island_thresholds_come_from_each_islands_resolve(fleet_run):
    run = fleet_run(breakers={"run": {"first_group": {"timeout_s": 900}}})
    rt = run.fleet_rt()
    assert [rt.resolved(k).get("run.first_group.timeout_s") for k in (0, 1)] == [900, 900]


# -- the restart init (2.6) --------------------------------------------------------------------

def test_a_round_cuts_off_restart_names_the_islands_holding_a_release(fleet_run, monkeypatch, capsys):
    from yeto.rl.engine import policy_release

    run = fleet_run()
    run.cfg["trainer"]["argv"] += ["--rl-round-cuts", "off"]
    run.state["steps"] = {"trainer": {"at": "x"}}
    seen = []

    def records(objects):
        seen.append(objects.root.name)
        return [types.SimpleNamespace(derived_from=None, policy_hash="h")] if objects.root.name == "trainer1" else []

    monkeypatch.setattr(policy_release, "records_in_objects", records)
    argv = run.trainer_argv()
    assert "--rl-init-from-release" not in argv and "--rl-ledger-set-aside-before" in argv
    assert seen == ["trainer0", "trainer1"]
    out = _out(capsys)
    assert any(line.startswith("WARNING: trainer restart with --rl-round-cuts off: island 1 (west) hold(s) policy "
                               "releases") for line in out)
    assert run.state["steps"]["trainer_init_from_release"]["islands_with_release"] == ["island 1 (west)"]


# -- status --------------------------------------------------------------------------------------

def test_status_shows_each_island(fleet_run, monkeypatch):
    run = fleet_run()
    monkeypatch.setattr(up.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="{}", stderr="",
                                                                                     returncode=0))
    _groups(run, 1, "sg-i1-0", 2)
    run.fleet_rt().trip_island(0, "run.starvation", value=1, limit=1, why="x")
    fleet = run.status()["fleet"]
    assert [(i["index"], i["name"], i["buffer_objects"], i["stop_reason"]) for i in fleet["islands"]] == [
        (0, "east", 0, "breaker:run.starvation"), (1, "west", 2, None)]
    assert fleet["stop_reason"] is None


# -- integration: the documented two-island example through run up's drive loop --------------

def _example(tmp_path):
    from tests.test_run_up_fleet import EXAMPLE

    text = EXAMPLE.read_text()
    for placeholder, value in (("<bucket>", "b"), ("<run>", "r"), ("<store.env>", str(tmp_path / "store.env")),
                               ("<sampler-spec.vllm.json>", str(tmp_path / "vllm.json")),
                               ("<sampler-spec.sglang.json>", str(tmp_path / "sglang.json")),
                               ("<sampler-image>", "registry/sampler@sha256:" + "0" * 64),
                               ("<modal-secret>", "store-secret"), ("<volume>", "weights")):
        text = text.replace(placeholder, value)
    cfg = json.loads(text)
    cfg["first_group"] = {"timeout_s": 4800, "poll_s": 60}
    cfg["syncer"] = {"usd_per_h": 0.5}
    return cfg


def test_the_two_island_example_runs_island_by_island_through_the_monitor(fleet_run, tmp_path, monkeypatch, capsys):
    """Two islands' pools write shards and groups under trainer0/ and trainer1/, the
    trainers release there, island 0 runs out of data first, then island 1: run up's
    first-group wait and monitor judge each island in its own sub-directory, and the
    spend counts every pool, both island trainers and the syncer."""
    run = fleet_run(_example(tmp_path))
    plan = run.fleet_plan()
    assert [i.name for i in plan.islands] == ["mixed", "sglang"]
    assert [p.shard_base for i in plan.islands for p in i.pools] == ["r-vllm-i0", "r-sglang-i0", "r-sglang-i1"]
    clock = run.clock
    t_start = clock.t
    for base in ("r-vllm-i0", "r-sglang-i0", "r-sglang-i1"):
        _pool_state(run, base, members=[{"ticket": f"{base}-t", "state": "ready", "granted": t_start}])
    _groups(run, 0, "r-vllm-i0-0", 2)  # island 0 sampled first; island 1's pool is still starting
    alive = {"trainer": True}
    monkeypatch.setattr(up, "_alive", lambda pid: alive["trainer"])
    monkeypatch.setattr(up.Run, "dashboard", lambda self: None)

    def trainer(self):
        self.state["procs"] = {"trainer": {"pid": 4242, "log": str(tmp_path / "trainer.log"), "started": "s"}}
        self.state.setdefault("trainer_t0", clock.t)
        self.mark("trainer")

    monkeypatch.setattr(up.Run, "trainer", trainer)
    monkeypatch.setattr(up.Run, "on_trainer_finished", lambda self: True)
    seen: list[str] = []
    step = {"n": 0}

    def timeline(t):
        step["n"] += 1
        n = step["n"]
        if n == 1:  # first-group wait: island 1's first group
            _groups(run, 1, "r-sglang-i1-0", 1)
        elif n == 2:  # monitor: both islands sample and release
            _groups(run, 0, "r-sglang-i0-0", 2)
            _groups(run, 1, "r-sglang-i1-0", 2, start=1)
            for k in (0, 1):
                _store(run, k).write("policy-releases/LATEST.json", json.dumps({"seq": 1, "path": "r1-v1"}).encode())
            _ledger(run, slots=10, trained=range(6), running=[(6, "r-sglang-i1-0"), (7, "r-sglang-i1-0")])
        elif n == 3:  # island 0 ran out of data: its pools finished; island 1 still holds two slots
            _ledger(run, slots=10, trained=range(8), running=[(8, "r-sglang-i1-0"), (9, "r-sglang-i1-0")])
            _groups(run, 0, "r-vllm-i0-0", 0, end=True)  # its sampler ended on the data end itself
            for base in ("r-vllm-i0", "r-sglang-i0"):
                _pool_state(run, base, finished=True,
                            members=[{"ticket": f"{base}-t", "state": "gone", "granted": t_start, "charged_to": t}])
        elif n == 4:
            seen.append("island0-ended" if run.fleet_rt().ended(0) and not run.fleet_rt().ended(1) else "?")
            _groups(run, 1, "r-sglang-i1-1", 2)
        elif n == 5:  # island 1 finished its slots: the fleet's data ended
            _ledger(run, slots=10, trained=range(10))
            _pool_state(run, "r-sglang-i1", finished=True,
                        members=[{"ticket": "r-sglang-i1-t", "state": "gone", "granted": t_start, "charged_to": t}])
        elif n == 7:  # the syncer's final version: the launch ends SUCCEEDED
            (tmp_path / "trainer.log").write_text("[launcher] run finished: SUCCEEDED (exit code 0)\n")
            alive["trainer"] = False

    clock.hooks.append(timeline)
    assert up._drive(run, types.SimpleNamespace(trainer_now=False, no_monitor=False)) == 0
    out = _out(capsys)
    # the first group, island by island
    assert "first group: waiting for island 1 (sglang) (island 0 (mixed) has a group)" in out
    assert run.state["fleet_first_group"].keys() == {"0", "1"}
    # the data end, island by island: island 0 first, then the fleet
    assert seen == ["island0-ended"]
    ended = run.state["fleet_data_ended"]
    assert sorted(ended["0"]["sealed"]) == sorted(set(plan.islands[0].buffer_shards()) - {"r-vllm-i0-0"})
    assert sorted(ended["1"]["sealed"]) == sorted(plan.islands[1].buffer_shards())
    assert run.state["data_ended"]["islands"] == 2
    lines = [line for line in out if line.startswith("data end:")]
    assert lines[0].startswith("data end: island 0 (mixed)") and lines[1].startswith("data end: island 1 (sglang)")
    assert lines[2].startswith("data end: every island's data ended")
    # no island starved, none stopped, no budget trip
    assert not any("starvation" in line or "fleet_island_stop" in line for line in out)
    assert run.fleet_rt().stops().running() == (0, 1) and _stops(run) == []
    # the spend: pools (node hours x their price) + 2 island trainers x $8/h + the syncer $0.5/h
    spend = run.fleet_rt().spend(clock.t)
    hours = run.trainer_hours(clock.t)
    pools = sum(run.node_hours(i, clock.t) * price for i, price in enumerate((4.0, 2.0, 2.0)))
    assert spend.total_usd == pytest.approx(pools + hours * 16.0 + hours * 0.5)
    assert spend.islands_usd == pytest.approx((hours * 8.0, hours * 8.0)) and hours > 0
    assert run.spend_usd() == pytest.approx(spend.total_usd)
