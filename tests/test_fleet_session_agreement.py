"""A fleet launched island by island: the syncer launch and every island launch (the
run config's islands, an island added at run time, a moved island) compile the same
syncer semantic profile and the same HELLO-bound session flags, so no island is refused
at HELLO for a profile the syncer does not run ("HELLO syncer semantic profile does not
match the running server"). Each launch derives them from its own argv: one --gpu entry
for an island launch, every declared trainer's for the syncer launch."""

from __future__ import annotations

import json
import re

import pytest

from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture
from test_fleet_island_launches import _prepared, example_doc
from yeto.launch.rl_island_task import _rl_trainable_island_flags
from yeto.launch.syncer import rl_syncer_profile
from yeto.launch.topology import rl_island_count
from yeto.rl import fleet, fleet_islands, island_launches, up
from yeto.rl.engine.sample_roles import syncer_learner_count

RECORD = {"addr": "10.0.0.9:29400", "syncer_epoch": 0}


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


def _session(args) -> tuple[dict, str]:
    """The launch's syncer semantic profile (every field) and its HELLO-bound island flags
    (learner id masked: the one field that is the island's own)."""
    learners = syncer_learner_count(args, rl_island_count(args))
    profile = rl_syncer_profile(args, learners)
    flags = _rl_trainable_island_flags(args, 0, learners)
    return dict(profile.__dict__), re.sub(r"--learner-id \d+", "", flags)


def _islands(cfg) -> dict[str, list[str]]:
    plan = fleet.plan_of(cfg)
    out = {m["name"]: island_launches.launch_argv(m, syncer=RECORD)
           for m in island_launches.trainer_members(cfg, plan)}
    if int(plan.max_islands) > len(plan.islands):  # room for an island added while it trains
        entry = dict(cfg["islands"][-1]["samplers"][0], name="sglang-north")
        added, _ = fleet_islands.compile_island(cfg, plan, k=len(plan.islands), name="north", gpu="modal:1x2xh100",
                                                usd_per_h=8.0, samplers=[entry])
        out["north (added)"] = island_launches.launch_argv(added, syncer=RECORD)
    moved = island_launches.trainer_members(cfg, plan)[-1]
    out[f"{moved['name']} (moved)"] = island_launches.launch_argv(
        {**moved, "gpu": "aws:2xh100@us-east-2"}, incarnation=1, syncer=RECORD)
    return out


@pytest.mark.parametrize("islands, max_islands, nesterov", [(2, 4, True), (1, 3, True), (1, None, False)])
def test_every_island_launch_compiles_the_syncer_launch_s_profile_and_session(tmp_path, islands, max_islands,
                                                                              nesterov):
    """The outer optimizer follows the fleet plan's static size (fleet.max_islands, else
    the declared islands), never the islands running now: two islands, or one planned to
    grow to three, run the outer Nesterov (lr 0.7, momentum 0.9) on every launch, an
    island added later too; one island with no room for more runs the identity merge
    (1.0, 0.0)."""
    doc = example_doc(tmp_path, max_islands=max_islands)
    doc["islands"] = doc["islands"][:islands]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    cfg = up.load_config(str(path))
    syncer_args = _prepared([*cfg["trainer"]["argv"], "--controller", "local"])
    syncer_profile, syncer_flags = _session(syncer_args)
    assert syncer_profile["learners"] == islands
    assert ("north (added)" in _islands(cfg)) == (max_islands is not None)
    assert syncer_profile["outer_momentum"] == (pytest.approx(0.9) if nesterov else 0.0)
    assert syncer_profile["outer_lr"] == (pytest.approx(0.7) if nesterov else 1.0)
    for name, argv in _islands(cfg).items():
        island_profile, island_flags = _session(_prepared(argv))
        diff = {k: (v, island_profile.get(k)) for k, v in syncer_profile.items() if island_profile.get(k) != v}
        assert not diff, f"island {name}: (syncer, island) {diff}"
        assert island_flags == syncer_flags, name


def _secret_cfg(tmp_path):
    """The two-island example with the fleet's store Secret for its Modal islands (and the
    fragments its model's layout has: a sky island's launch checks them before ordering)."""
    doc = example_doc(tmp_path)
    argv = doc["fleet"]["trainer_argv"]
    argv[argv.index("--fragments") + 1] = "9"
    argv += ["--modal-store-secret", "store-secret"]
    path = tmp_path / "secret.json"
    path.write_text(json.dumps(doc))
    return up.load_config(str(path))


def test_two_islands_a_third_added_and_one_moved_to_aws_agree_and_carry_only_their_cloud_s_flags(tmp_path):
    from yeto.rl import fleet_session

    cfg = _secret_cfg(tmp_path)
    plan = fleet.plan_of(cfg)
    members = island_launches.trainer_members(cfg, plan)
    entry = dict(cfg["islands"][-1]["samplers"][0], name="sglang-north")
    added, _ = fleet_islands.compile_island(cfg, plan, k=2, name="north", gpu="modal:1x2xh100", usd_per_h=8.0,
                                            samplers=[entry])
    moved = {**members[0], "gpu": "aws:1xh100@us-west-2"}
    for case in (members, [*members, added], [moved, *members[1:], added]):
        fleet_session.check(cfg["trainer"]["argv"], case)  # raises on any disagreement
    for member in (*members, added):
        assert "--modal-store-secret" in island_launches.launch_argv(member, syncer=RECORD)
    aws = island_launches.launch_argv(moved, incarnation=1, syncer=RECORD)
    assert "--modal-store-secret" not in aws and "store-secret" not in aws
    _prepared(aws)  # the AWS island's launch takes its options (no Modal Secret to attach)


def test_a_disagreeing_island_fails_the_run_up_preflight_and_its_add(tmp_path, monkeypatch):
    """An island whose launch compiles another profile (here an outer lr of its own) is
    refused at the preflight (dry run too) with the differing field named, and so is an
    `yeto fleet island add/move` of it."""
    from test_fleet_island_runup import FakeController, build
    from test_fleet_trainer_worker import FakeLaunches
    from yeto.launch.pool import desired as dz

    run, objects = build(tmp_path, monkeypatch)
    run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
    run.adopt_desired()
    assert run.fleet_session_problem() is None
    store = dz.DesiredStore(objects)
    store.update(lambda doc: dz.trainers(doc)["sglang"]["argv"].extend(["--rl-sync-outer-lr", "0.5"]))
    run.adopt_desired()
    problem = run.fleet_session_problem()
    assert problem and "island sglang (learner 1)" in problem and "'outer_lr'" in problem
    logs = []
    monkeypatch.setattr(up, "_log", logs.append)
    assert run.preflight() is False  # `yeto run up` and its dry run: nothing is ordered
    assert any("do not agree on the syncer session" in m and "'outer_lr'" in m for m in logs)
    member = dz.trainers(store.read())["sglang"]
    with pytest.raises(ValueError, match="outer_lr"):
        fleet_islands.session_check(run, store.read(), member)
    clash = {**dz.trainers(store.read())["mixed"], "name": "late"}
    with pytest.raises(fleet_islands.IslandChangeError, match="learner id 0 of island late"):
        fleet_islands.session_check(run, store.read(), clash)


def test_a_move_of_an_island_whose_launch_disagrees_is_refused(fleet_store, capsys):
    from yeto.launch.pool import desired as dz

    cfg, objects, store, run_cli, _ = fleet_store
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "6", "--dry-run") == 0
    store.update(lambda doc: dz.trainers(doc)["sglang"]["argv"].extend(["--rl-sync-outer-lr", "0.5"]))
    capsys.readouterr()
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "6", "--dry-run") == 2
    err = capsys.readouterr().err
    assert "do not agree on the syncer session" in err and "'outer_lr'" in err


def test_an_island_off_modal_launches_without_any_modal_only_flag(tmp_path):
    """--modal-store-secret NAME, --modal-gpu-name NAME and --modal-gpu-exact (no value)
    are dropped from an island launch moved or added off Modal, each with its own values
    only; a Modal island keeps them."""
    argv = ["--modal-gpu-exact", "--rl-release-every", "0:2", "--modal-gpu-name", "H100",
            "--modal-store-secret", "store-secret", "--modal-gpu-name=H100"]
    assert island_launches.own_cloud_argv(argv, "aws:1xh100@us-west-2") == ["--rl-release-every", "0:2"]
    assert island_launches.own_cloud_argv(argv, "modal:1x2xh100") == argv
    cfg = _secret_cfg(tmp_path)
    member = island_launches.trainer_members(cfg, fleet.plan_of(cfg))[0]
    member = {**member, "argv": [*member["argv"], "--modal-gpu-exact", "--modal-gpu-name", "H100"]}
    on_modal = island_launches.launch_argv(member, syncer=RECORD)
    assert "--modal-gpu-exact" in on_modal and "--modal-gpu-name" in on_modal
    aws = island_launches.launch_argv({**member, "gpu": "aws:1xh100@us-west-2"}, incarnation=1, syncer=RECORD)
    assert not {"--modal-gpu-exact", "--modal-gpu-name", "--modal-store-secret"} & set(aws)
    _prepared(aws)  # the moved island's launch takes its options


@pytest.mark.parametrize("broken", ["missing", "not json"])
def test_an_algorithm_spec_the_launches_cannot_read_is_a_preflight_refusal(tmp_path, monkeypatch, broken):
    from test_fleet_island_runup import build

    run, _ = build(tmp_path, monkeypatch)
    spec = tmp_path / "spec.algorithm.json"
    if broken == "not json":
        spec.write_text("{not json")
    argv = run.cfg["trainer"]["argv"]
    argv[argv.index("--rl-algorithm-spec") + 1] = str(spec)
    for m in run.cfg["fleet"]["trainer_argv"], *(i["trainer"].get("argv") or [] for i in run.cfg["islands"]):
        if "--rl-algorithm-spec" in m:
            m[m.index("--rl-algorithm-spec") + 1] = str(spec)
    problem = run.fleet_session_problem()  # a refusal, not an exception out of the preflight
    assert problem and "the syncer launch refuses its options" in problem
    logs = []
    monkeypatch.setattr(up, "_log", logs.append)
    assert run.preflight() is False
    assert any("refuses its options" in m for m in logs)


@pytest.mark.parametrize("where", ["syncer launch", "island launch"])
def test_an_os_error_while_a_launch_reads_its_files_is_a_refusal(tmp_path, where):
    """The launch readers turn their file errors into ValueError today; an OSError that
    still gets through (a reader that does not, a permission error) is the same refusal."""
    from yeto.launcher import _prepare_rl_args
    from yeto.rl import fleet_session

    cfg = _secret_cfg(tmp_path)
    calls = []

    def prepare(args):
        calls.append(args)
        if (where == "syncer launch") == (len(calls) == 1):
            raise PermissionError(13, "Permission denied", "spec.algorithm.json")
        _prepare_rl_args(args)

    members = island_launches.trainer_members(cfg, fleet.plan_of(cfg))
    with pytest.raises(ValueError, match="refuses its options: PermissionError") as raised:
        fleet_session.check(cfg["trainer"]["argv"], members, prepare=prepare)
    assert (where == "syncer launch") == ("the syncer launch" in str(raised.value))
