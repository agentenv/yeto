"""A fleet launched island by island (``fleet.launch: per_island``, ``yeto.rl.island_launches``):
the compile makes one syncer launch and one launch per trainer island, every island its
learner id, its own sub-store and shards, the fleet's roster limit; ``yeto launch
--dry-run`` accepts each. The default (synced group) compile is unchanged."""

from __future__ import annotations

import json
import types
import shlex
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from yeto.rl import fleet, island_launches, up
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fleet" / "run.two-islands.json"


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


def example_doc(tmp_path, *, per_island: bool = True, max_islands: int | None = 4) -> dict:
    text = EXAMPLE.read_text()
    for placeholder, value in (("<bucket>", "b"), ("<run>", "r"), ("<store.env>", str(tmp_path / "store.env")),
                               ("<sampler-spec.vllm.json>", str(tmp_path / "spec.json")),
                               ("<sampler-spec.sglang.json>", str(tmp_path / "spec.json")),
                               ("<sampler-image>", "registry/sampler@sha256:" + "0" * 64),
                               ("<modal-secret>", "store-secret"), ("<volume>", "weights")):
        text = text.replace(placeholder, value)
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    (tmp_path / "spec.json").write_text(json.dumps({"sampler_id": "x", "sampling": {"n": 8}}))
    doc = json.loads(text)
    if per_island:
        doc["fleet"]["launch"] = "per_island"
        if max_islands is not None:
            doc["fleet"]["max_islands"] = max_islands
        doc["controller"] = "in_process"
        doc.setdefault("rollout", {})["control_plane"] = "store"
    return doc


def example(tmp_path, **kw) -> dict:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(example_doc(tmp_path, **kw)))
    return up.load_config(str(path))


def _prepared(argv: list[str]):
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args

    args = parse_args(argv)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    return args


def _task(args):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task
    from yeto.launcher import _island_overrides, _island_task

    specs = parse_gpu_spec(args.gpu)
    return _island_task(make_miles_island_task, args, specs[0], 0,
                        sample_roles.syncer_learner_count(args, len(specs)), "10.0.0.1:29400",
                        _island_overrides(args, specs))


def test_synced_group_compile_is_unchanged_without_per_island(tmp_path):
    base = example(tmp_path, per_island=False)
    plan = fleet.plan_of(base)
    assert not plan.per_island and "launch" not in plan.to_json()
    assert "--rl-fleet-role" not in base["trainer"]["argv"]


def test_per_island_compiles_a_syncer_launch_and_one_launch_per_island(tmp_path):
    cfg = example(tmp_path)
    plan = fleet.plan_of(cfg)
    assert plan.per_island and plan.max_islands == 4
    argv = cfg["trainer"]["argv"]
    assert argv[-6:] == ["--rl-fleet-role", "syncer", "--rl-fleet-learners", "2", "--rl-sync-max-learners", "4"]
    members = island_launches.trainer_members(cfg, plan)
    assert [m["learner_id"] for m in members] == [0, 1]
    assert [m["prefix"] for m in members] == ["r-i0", "r-i1"]
    for m, island in zip(members, plan.islands):
        a = m["argv"]
        assert a[a.index("--rl-fleet-learner-id") + 1] == str(island.learner)
        assert a[a.index("--rl-island-buffer-shards") + 1] == "0:" + ",".join(island.pool_shards())
        assert "--gpu" not in a and m["gpu"] == island.gpu
    assert "--rl-island-release-every" in members[0]["argv"] and "--rl-island-release-every" not in members[1]["argv"]


def test_max_islands_below_the_declared_islands_is_refused(tmp_path):
    with pytest.raises(ValueError, match="max_islands"):
        example(tmp_path, max_islands=1)


def test_the_syncer_launch_dry_run_starts_the_syncer_for_the_fleet(tmp_path):
    from yeto.launcher import dry_run_plan

    cfg = example(tmp_path)
    args = _prepared([*cfg["trainer"]["argv"], "--controller", "local"])
    assert sample_roles.fleet_role(args) == "syncer"
    launch = dry_run_plan(args)
    command = launch["role_syncers"][0]["command"]
    assert "--learners 2 " in command and "--max-learners 4" in command


def test_each_island_launch_is_learner_k_on_its_own_sub_store(tmp_path):
    from yeto.launcher import dry_run_plan

    cfg = example(tmp_path)
    plan = fleet.plan_of(cfg)
    for member, island in zip(island_launches.trainer_members(cfg, plan), plan.islands):
        argv = island_launches.launch_argv(member, syncer={"addr": "10.0.0.1:29400", "syncer_epoch": 3})
        args = _prepared(argv)
        args.dry_run = True
        assert sample_roles.fleet_learner(args) == (island.learner, 2)
        launch = dry_run_plan(args)
        (request,) = launch["island_requests"]
        assert request["sample_substore"] == island.substore
        assert request["buffer_shards"] == island.buffer_shards()
        tail = request["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{"$SYNCER_ADDR": "10.0.0.1:29400", "$LEARNER_ID": str(island.learner)}.get(t, t)
                  for t in shlex.split(tail)]
        parsed = island_entry.parse_args(tokens)
        task = _task(args)
        assert task.envs["LEARNER_ID"] == str(island.learner)
        assert int(parsed.learner_id) == island.learner and int(parsed.num_learners) == 2
        assert str(parsed.rl_sample_store).endswith(f"/{island.substore}")
        assert int(parsed.sync_max_learners) == 4


def test_an_island_launch_past_the_roster_limit_is_refused(tmp_path):
    cfg = example(tmp_path)
    plan = fleet.plan_of(cfg)
    member = dict(island_launches.trainer_members(cfg, plan)[0])
    argv = [a if a != "0" or i == 0 else a for i, a in enumerate(member["argv"])]
    k = argv.index("--rl-fleet-learner-id")
    argv[k + 1] = "4"
    with pytest.raises(ValueError, match="--rl-fleet-learner-id K in 0..3"):
        _prepared(island_launches.launch_argv({**member, "argv": argv}, syncer={"addr": "h:1"}))


def test_fleet_options_without_the_role_are_refused(tmp_path):
    cfg = example(tmp_path, per_island=False)
    with pytest.raises(ValueError, match="need --rl-fleet-role"):
        _prepared([*cfg["trainer"]["argv"], "--controller", "local", "--rl-fleet-learners", "2"])


def test_the_syncer_launch_keeps_no_island_records():
    """The syncer launch names every declared trainer in --gpu but runs none: it never
    records, clears the STOP of, or adopts an island (each island launch keeps its own)."""
    from yeto import launcher

    syncer = types.SimpleNamespace(rl_fleet_role="syncer", island_lifecycle="on", training_mode="rl")
    assert not launcher.island_lifecycle_enabled(syncer, [])
    island = types.SimpleNamespace(rl_fleet_role="island", rl_fleet_learner_id=1, rl_fleet_learners=2,
                                   island_lifecycle="on", training_mode="rl")
    assert launcher.island_lifecycle_enabled(island, [])


def test_the_syncer_grace_goes_to_the_syncer_launch_only(tmp_path):
    doc = example_doc(tmp_path)
    doc["fleet"]["trainer_argv"] += ["--rl-fleet-syncer-grace-s", "900"]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    cfg = up.load_config(str(path))
    plan = fleet.plan_of(cfg)
    syncer = cfg["trainer"]["argv"]
    assert syncer[syncer.index("--rl-fleet-syncer-grace-s") + 1] == "900"
    for m in island_launches.trainer_members(cfg, plan):
        assert "--rl-fleet-syncer-grace-s" not in m["argv"]


def test_island_k_s_launch_reads_and_clears_only_its_own_record_and_stop(tmp_path, monkeypatch):
    """A fleet island launch (``--rl-fleet-role island``, learner 2) plans its island
    records under its learner id: it reads record 2 (not record 0, its position in its
    own --gpu), clears only STOP 2 at a new start, and records island 2."""
    from yeto import launcher
    from yeto.launch import island_lifecycle as lc
    from yeto.rl.engine.object_store import DirectoryObjectStore

    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "s"), log=lambda m: None)
    states.write(0, "done", name="r-i0-l0", source="sky", incarnation=0)
    states.request_stop(0, "island 0 is being removed", by="trainer worker")
    states.request_stop(2, "an earlier run's STOP", by="operator")
    read = []
    real = states.read_all
    monkeypatch.setattr(states, "read_all", lambda ids, **kw: read.append(list(ids)) or real(ids, **kw))
    args = types.SimpleNamespace(rl_fleet_role="island", rl_fleet_learner_id=2, rl_fleet_learners=2,
                                 rl_sync_max_learners=4)
    recorder, adoption, kept = launcher._plan_island_records(
        args, [types.SimpleNamespace(cloud="aws")], ["r-i2-l0"], states, None, relaunch=False, ssh_names=set(),
        events_dir=None, echo_names=set(), held_name=None, syncer_cluster=None)
    assert read == [[2]] and adoption == {} and kept is None
    assert states.stop_requested(2) is None  # its own earlier STOP is void
    assert states.stop_requested(0) is not None  # island 0's STOP untouched
    recorder.record("r-i2-l0", "starting", reason="test")
    assert states.read(2)["state"] == "starting" and states.read(0)["state"] == "done"


def test_per_island_launches_take_no_roster_option_of_a_synced_group(tmp_path, monkeypatch):
    """The synced-group compile adds its roster's membership options to the group argv
    (a lease; a count quorum like --rl-sync-quorum 1 would merge a round on the first
    island). A per-island fleet's syncer and island launches take none of them: they
    resolve their own membership, the capacity quorum theta 1.0 (every island of the pool
    answers each round), so the fleet is neither refused nor merged on one island."""
    from yeto.launcher import dry_run_plan

    injected = ["--rl-sync-island-lease-s", "150", "--rl-sync-quorum", "1"]
    monkeypatch.setattr(fleet, "roster_of", lambda *a, **k: fleet.Roster(False, 1, 150.0, None, tuple(injected)))
    group = example(tmp_path, per_island=False)["trainer"]["argv"]
    assert " ".join(injected) in " ".join(group)  # the synced group takes them
    (tmp_path / "pi").mkdir()
    cfg = example(tmp_path / "pi")
    plan = fleet.plan_of(cfg)
    members = island_launches.trainer_members(cfg, plan)
    for argv in [cfg["trainer"]["argv"], *(m["argv"] for m in members)]:
        assert "--rl-sync-quorum" not in argv and "--rl-sync-island-lease-s" not in argv
    syncer = _prepared([*cfg["trainer"]["argv"], "--controller", "local"])
    assert syncer.rl_sync_quorum is None and syncer.rl_sync_quorum_theta == 1.0
    command = dry_run_plan(syncer)["role_syncers"][0]["command"]
    assert "--quorum-theta 1.0" in command and "--quorum 1 " in command  # theta, not a count of one
    island = _prepared(island_launches.launch_argv(members[0], syncer={"addr": "10.0.0.1:29400"}))
    assert island.rl_sync_quorum is None
    # fleet.island_lease_s still reaches the syncer and every island; a fixed roster is refused
    for name, value, check in (("island_lease_s", 300, None), ("fixed_roster", True, "fixed_roster")):
        (tmp_path / name).mkdir()
        doc = example_doc(tmp_path / name)
        doc["fleet"][name] = value
        path = tmp_path / name / "run.json"
        path.write_text(json.dumps(doc))
        if check:
            with pytest.raises(ValueError, match=check):
                up.load_config(str(path))
            continue
        cfg = up.load_config(str(path))
        plan = fleet.plan_of(cfg)
        for argv in [cfg["trainer"]["argv"], *(m["argv"] for m in island_launches.trainer_members(cfg, plan))]:
            assert argv[argv.index("--rl-sync-island-lease-s") + 1] == "300"
