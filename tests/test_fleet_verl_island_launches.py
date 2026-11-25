"""verl trainer islands through the fleet's launches (rl-fleet-verl-islands 1.1, 3.2).

* The example fleet on verl: its compiled synced-group trainer launch passes ``yeto
  launch --dry-run`` in process, each trainer reading its own ``trainer<k>/`` and told to
  release per learner; launched island by island, every island launch agrees with the
  syncer launch on the session.
* The trainer worker over verl island launches (fake launches, local store): an island
  whose launch ends with the island stop exit (a deterministic stop, exit 11) is failed
  for good and never started again, while the other island keeps running untouched; a
  killed launch is relaunched alone. Inside the island launch, its controller records the
  stop in ``fleet/trainers/<k>/state.json`` and orders nothing again.
No cloud, no GPU.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from test_fleet_island_launches import example_doc
from test_fleet_trainer_worker import FakeLaunches
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.config_errors import ISLAND_STOP_EXIT
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool import desired as dz
from yeto.rl import island_launches as il
from yeto.rl import trainer_worker as tw
from yeto.rl import up
from yeto.rl.engine.object_store import DirectoryObjectStore

LAYOUT = "e" * 64
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _island_env(monkeypatch, tmp_path):
    # hermetic: the offline layout check reads no model config from this host's HF cache, so
    # the launches go on unverified (test_fleet_verl_layout_preflight checks the layout)
    from yeto.launch import fragment_preflight

    monkeypatch.setattr(fragment_preflight, "local_hf_config", lambda model, revision: None)
    monkeypatch.setenv(fragment_preflight.LAYOUT_MODE_ENV, "warn")
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-verl-launch-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")


def verl_doc(tmp_path, **kw) -> dict:
    doc = example_doc(tmp_path, **kw)
    argv = doc["fleet"]["trainer_argv"]
    argv[argv.index("--rl-parameter-layout-sha256") + 1] = LAYOUT  # verl plans its layout offline
    argv += ["--rl-backend", "verl"]
    # verl's TIS clamps the upper side only: the example's two-sided TIS spec without its lower bound
    spec = json.loads((ROOT / argv[argv.index("--rl-algorithm-spec") + 1]).read_text())
    spec["correction"]["lower"] = 0.0
    (tmp_path / "algorithm-spec.verl.json").write_text(json.dumps(spec))
    argv[argv.index("--rl-algorithm-spec") + 1] = str(tmp_path / "algorithm-spec.verl.json")
    return doc


def _run(tmp_path, monkeypatch, **kw):
    root = tmp_path / "bucket"
    monkeypatch.setattr(up.Run, "_objects", lambda self: DirectoryObjectStore(root))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(verl_doc(tmp_path, **kw)))
    return up.Run(up.load_config(str(path)), state_root=tmp_path / "state"), DirectoryObjectStore(root)


# -- 1.1: the launches of a verl fleet ---------------------------------------------------------

def test_the_synced_group_trainer_launch_of_a_verl_fleet_passes_its_dry_run(tmp_path, monkeypatch, capsys):
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    run, _ = _run(tmp_path, monkeypatch, per_island=False)
    assert run.fleet_problem() is None
    plan = run.fleet_plan()
    assert plan.sync["backend"] == "verl" and [i.name for i in plan.islands] == ["mixed", "sglang"]
    args = parse_args([str(a) for a in run.cfg["trainer"]["argv"]])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.controller = "local"
    _prepare_rl_args(args)
    out = dry_run_plan(args)
    trainers = [i for i in out["island_requests"] if i.get("sample_role") == "trainer"]
    assert len(trainers) == 2
    for k, island in enumerate(trainers):
        tokens = shlex.split(island["learner_command"].split("-m yeto.rl.adapters.verl.island_entry", 1)[1])
        assert tokens[tokens.index("--rl-sample-store") + 1].endswith(f"/trainer{k}")
        assert tokens[tokens.index("--rl-sample-lanes") + 1] == "1"
        assert tokens[tokens.index("--rl-release-mode") + 1] == "per_learner"
    # the island's own cadence reaches its verl island (island 0 of the example: every 2)
    first = shlex.split(trainers[0]["learner_command"])
    assert first[first.index("--rl-release-every") + 1] == "2"


def test_every_island_launch_of_a_per_island_verl_fleet_agrees_with_the_syncer_launch(tmp_path, monkeypatch):
    run, _ = _run(tmp_path, monkeypatch)
    assert run.per_island_fleet() and run.fleet_problem() is None
    assert run.fleet_session_problem() is None
    members = il.trainer_members(run.cfg, run.fleet_plan())
    assert [m["learner_id"] for m in members] == [0, 1]
    for m in members:
        argv = il.launch_argv(m, syncer={"addr": "SYNCER:29400", "syncer_epoch": 0})
        assert argv[argv.index("--rl-backend") + 1] == "verl"
        assert argv[argv.index("--rl-fleet-learner-id") + 1] == str(m["learner_id"])


# -- 3.2: a verl island's deterministic stop under the trainer worker ----------------------------

def _worker(tmp_path, monkeypatch):
    run, _ = _run(tmp_path, monkeypatch)
    members = il.trainer_members(run.cfg, run.fleet_plan())
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": members}, expect=0)
    il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=1, learners=2, max_learners=4, run="r")
    launches = FakeLaunches()
    w = tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=lambda m: None)
    return w, launches, objects, members


VERL_STOP_LOG = ("[yeto-island] trainer.sample_wait: no new group in the buffer for 900 s\n"
                 "[yeto-island] island stop: exit 11 (deterministic; the launcher does not relaunch it)\n"
                 "(RayWorkerManager) teardown line")


def test_a_verl_island_stopped_for_good_leaves_the_other_island_running(tmp_path, monkeypatch):
    w, launches, objects, members = _worker(tmp_path, monkeypatch)
    prefixes = {m["learner_id"]: m["prefix"] for m in members}
    w.step()
    assert sorted(p for p, _ in launches.starts) == sorted(prefixes.values())
    for _prefix, argv in launches.starts:
        assert argv[argv.index("--rl-backend") + 1] == "verl"
    launches.end(prefixes[1], ISLAND_STOP_EXIT, log=VERL_STOP_LOG)
    for _ in range(3):
        w.step()
    # island 1: failed for good, never started again; its state in the run store
    assert launches.started(prefixes[1]) == 1
    record = w.load(1)
    assert record["state"] == lifecycle.FAILED and record["decision"]["action"] == lifecycle.FAIL
    assert record["decision"]["cause"] == lifecycle.BREAKER  # a deterministic stop, not an infrastructure end
    assert record["ended"]["exit_code"] == ISLAND_STOP_EXIT
    # the worker's own record of island 1's launch, beside the island's state.json (which its
    # launch writes: the next test)
    stored = json.loads(objects.read(f"{lifecycle.STATE_DIR}/1/{tw.LAUNCH_RECORD}"))
    assert stored["state"] == lifecycle.FAILED and stored["ended"]["exit_code"] == ISLAND_STOP_EXIT
    # island 0: running, never stopped or started again (the syncer merges on without island 1)
    assert launches.alive(prefixes[0]) and launches.started(prefixes[0]) == 1
    assert prefixes[0] not in launches.stops
    assert w.load(0)["state"] == tw.RUNNING
    assert json.loads(objects.read(f"{lifecycle.STATE_DIR}/0/{tw.LAUNCH_RECORD}"))["state"] == tw.RUNNING


def test_a_killed_verl_island_launch_is_relaunched_alone(tmp_path, monkeypatch):
    w, launches, _objects, members = _worker(tmp_path, monkeypatch)
    prefixes = {m["learner_id"]: m["prefix"] for m in members}
    w.step()
    launches.end(prefixes[0], 1, log="ray.exceptions.RayActorError: The actor died unexpectedly")
    w.step()
    assert launches.started(prefixes[0]) == 2 and launches.started(prefixes[1]) == 1
    assert w.load(0)["relaunches"] == 1 and w.load(1)["relaunches"] == 0
    assert launches.stops == []


def test_the_verl_island_launch_records_its_stop_in_the_run_store_and_is_not_relaunched(tmp_path):
    """Inside the island launch (one island, ``--island-lifecycle on``): the verl island
    process ends on a trainer breaker's stop (it printed the island stop marker and exited
    11); the launch's controller gives it up without ordering it again and records the
    decision in ``fleet/trainers/<k>/state.json``, which the trainer worker and the
    dashboard read. A killed island (no marker) is relaunched instead."""
    from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
    from yeto.config_errors import ISLAND_STOP_MARKER
    from yeto.launch.fleet import FleetController, IslandsFailedForGood

    def launch(k, ops):
        objects = DirectoryObjectStore(tmp_path / "bucket")
        store = lifecycle.IslandStateStore(objects, log=lambda m: None)
        name = f"r-i{k}"
        ops.status_seq.setdefault(SYNCER, [RUNNING])
        ctl = FleetController(learners={name: (f"task-{name}", 1)}, syncer=(SYNCER, "task-syncer", 1),
                              sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                              recorder=lifecycle.FleetRecorder(store, {name: lifecycle.IslandRef(k, "modal")}),
                              elastic=True, deterministic_stop=True)
        return ctl, objects, name

    ops = FakeOps()
    ctl, objects, name = launch(1, ops)
    ops.status_seq[name] = [RUNNING, FAILED]
    ops.relaunch_results[name] = ["j2"]
    ops.rl_strict_failure = lambda n, job: (f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: no new group in the "
                                             "buffer for 900 s")
    with pytest.raises(IslandsFailedForGood):
        ctl.run()  # the island launch ends: its process exits with the island stop exit
    assert ops.relaunch_calls == []
    stored = json.loads(objects.read(lifecycle.state_key(1)))
    assert stored["decision"]["action"] == lifecycle.FAIL and stored["decision"]["cause"] == lifecycle.BREAKER
    assert stored["state"] in (lifecycle.FAILED, "abandoned")
    assert objects.read(lifecycle.state_key(0)) is None  # the other island's record is its own launch's

    ops = FakeOps()
    ctl, objects, name = launch(0, ops)
    ops.status_seq[name] = [RUNNING, FAILED]
    ops.relaunch_results[name] = ["j2"]
    ops.after_relaunch[name] = [RUNNING, SUCCEEDED]
    ctl.run()
    assert ops.relaunch_calls == [name]
    assert json.loads(objects.read(lifecycle.state_key(0)))["state"] == "done"
