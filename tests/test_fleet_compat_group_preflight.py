"""An island moved or added to a card of another compat group, under strict identity, is
refused before anything is ordered: its session contract binds the compat group (same
layout, another session_contract_hash), so the running syncer would refuse its HELLO only
after the machine booted and pulled its image. The refusal names both compat groups and
the two ways out (a card of the fleet's group, or an explicit --rl-compat-group /
--rl-identity relaxed). `yeto run up` checks the islands of its desired state the same way."""

from __future__ import annotations

import pytest

from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture
from test_fleet_session_agreement import RECORD, _secret_cfg
from yeto.launch.pool import desired as dz
from yeto.rl import fleet, fleet_session, island_launches

H200 = "modal:1x2xh200"  # a fleet_store island's launch dry-runs on Modal (no sky fragment check)


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


def _members(tmp_path):
    cfg = _secret_cfg(tmp_path)
    return cfg, island_launches.trainer_members(cfg, fleet.plan_of(cfg))


def test_an_island_on_another_compat_group_is_refused_with_both_groups_and_the_fixes(tmp_path):
    cfg, members = _members(tmp_path)
    moved = {**members[0], "gpu": "nebius:2xh200@us-central1"}
    with pytest.raises(ValueError) as raised:
        fleet_session.check(cfg["trainer"]["argv"], [moved, *members[1:]])
    text = str(raised.value)
    assert "compat_group nvidia-h200" in text and "compat_group nvidia-h100" in text
    assert "session_contract_mismatch" in text
    assert "--rl-compat-group" in text and "--rl-identity relaxed" in text
    # Same card group elsewhere (another cloud): agrees.
    fleet_session.check(cfg["trainer"]["argv"], [{**members[0], "gpu": "aws:2xh100@us-west-2"}, *members[1:]])


@pytest.mark.parametrize("relax", [["--rl-compat-group", "nvidia-h100"], ["--rl-identity", "relaxed"]])
def test_an_explicit_compat_group_or_relaxed_identity_admits_the_other_card(tmp_path, relax):
    cfg, members = _members(tmp_path)
    cfg["trainer"]["argv"].extend(relax)
    members = [{**m, "argv": [*m["argv"], *relax]} for m in members]
    moved = {**members[0], "gpu": "nebius:2xh200@us-central1"}
    fleet_session.check(cfg["trainer"]["argv"], [moved, *members[1:]])


def test_an_island_whose_identity_mode_differs_is_refused(tmp_path):
    cfg, members = _members(tmp_path)
    relaxed = {**members[0], "argv": [*members[0]["argv"], "--rl-identity", "relaxed"]}
    with pytest.raises(ValueError, match="--rl-identity relaxed, island sglang .* runs --rl-identity strict"):
        fleet_session.check(cfg["trainer"]["argv"], [members[1], relaxed])


def test_the_running_session_is_compared_with_but_not_reported_for(tmp_path):
    """``running``: the launches holding the fleet's session (a moved island's old machines,
    the other islands) are compared with the island written, never refused themselves."""
    cfg, members = _members(tmp_path)
    old = members[0]
    moved = {**old, "gpu": "nebius:2xh200@us-central1"}
    with pytest.raises(ValueError, match="compat_group nvidia-h200, the fleet's session holds compat_group "
                                         "nvidia-h100 \\(island\\(s\\) mixed-old\\)"):
        fleet_session.check(cfg["trainer"]["argv"], [moved], running=[{**old, "name": "mixed-old"}])
    # A running launch that does not compile is left out (with a note), not a refusal.
    fleet_session.check(cfg["trainer"]["argv"], [old], running=[{**old, "name": "broken", "gpu": "nowhere:9xq"}])


def test_a_move_to_h200_is_refused_at_its_dry_run_before_any_order(fleet_store, capsys):  # noqa: F811
    cfg, objects, store, run_cli, _ = fleet_store
    before = store.read()["generation"]
    assert run_cli("move", "--name", "sglang", "--gpu", H200, "--usd-per-h", "4",
                   "--dry-run") == 2
    err = capsys.readouterr().err
    assert "fleet island: refused" in err
    assert "compat_group nvidia-h200" in err and "compat_group nvidia-h100" in err
    assert "--rl-compat-group" in err and "--rl-identity relaxed" in err
    assert run_cli("move", "--name", "sglang", "--gpu", H200, "--usd-per-h", "4") == 2
    assert store.read()["generation"] == before  # nothing written, nothing ordered
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4", "--dry-run") == 0


def test_an_add_on_another_card_is_refused(fleet_store, capsys):  # noqa: F811
    cfg, objects, store, run_cli, samplers = fleet_store
    assert run_cli("add", "--name", "north", "--gpu", H200, "--usd-per-h", "4",
                   "--samplers", str(samplers), "--dry-run") == 2
    err = capsys.readouterr().err
    assert "island north" in err and "compat_group nvidia-h200" in err


def test_the_only_island_moved_to_another_card_is_refused_once_the_fleet_has_a_syncer(fleet_store, capsys):  # noqa: F811
    """One trainer island left: no other island holds the session, but its old machines
    joined it; with a syncer running the move to another card group is refused too."""
    cfg, objects, store, run_cli, _ = fleet_store
    assert run_cli("remove", "--name", "mixed") == 0
    capsys.readouterr()
    assert run_cli("move", "--name", "sglang", "--gpu", H200, "--usd-per-h", "4",
                   "--dry-run") == 0  # no syncer yet: no session to join
    island_launches.write_syncer_record(objects, addr=RECORD["addr"], syncer_epoch=0, learners=2,
                                        max_learners=4, run="r")
    capsys.readouterr()
    assert run_cli("move", "--name", "sglang", "--gpu", H200, "--usd-per-h", "4",
                   "--dry-run") == 2
    err = capsys.readouterr().err
    assert "sglang (its machines before the move)" in err and "compat_group nvidia-h100" in err


def test_an_unreadable_syncer_record_is_never_taken_for_no_syncer(fleet_store, capsys, monkeypatch):  # noqa: F811
    """A store error while reading the syncer record: the real move is refused with the
    reason (nothing written); the dry run warns and compares with the previous machines
    as if a syncer ran, so the move of the only island to another card group is refused
    there too, while a move within the group passes its dry run."""
    cfg, objects, store, run_cli, _ = fleet_store
    assert run_cli("remove", "--name", "mixed") == 0

    def unreadable(_objects):
        raise OSError("store unavailable")

    monkeypatch.setattr(island_launches, "read_syncer_record", unreadable)
    generation = store.read()["generation"]
    capsys.readouterr()
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4") == 2
    err = capsys.readouterr().err
    assert "syncer record could not be read (OSError: store unavailable)" in err and "retry" in err
    assert store.read()["generation"] == generation
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4", "--dry-run") == 0
    assert "WARNING: the fleet's syncer record could not be read" in capsys.readouterr().err
    assert run_cli("move", "--name", "sglang", "--gpu", H200, "--usd-per-h", "4", "--dry-run") == 2
    err = capsys.readouterr().err
    assert "sglang (its machines before the move)" in err and "compat_group nvidia-h200" in err


def test_a_syncer_record_that_does_not_parse_is_unreadable_not_missing(fleet_store, capsys):  # noqa: F811
    cfg, objects, store, run_cli, _ = fleet_store
    assert run_cli("remove", "--name", "mixed") == 0
    objects.write(island_launches.SYNCER_RECORD, b"{not json")
    capsys.readouterr()
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4") == 2
    assert "does not parse or names no address" in capsys.readouterr().err
    assert run_cli("move", "--name", "sglang", "--gpu", H200, "--usd-per-h", "4", "--dry-run") == 2
    assert "sglang (its machines before the move)" in capsys.readouterr().err


def test_a_strict_syncer_refuses_relaxed_islands_on_two_compat_groups(tmp_path):
    """The syncer compares compat groups unless it runs --rl-identity relaxed itself: two
    relaxed islands on different cards are refused by a strict syncer launch."""
    cfg, members = _members(tmp_path)
    relaxed = [{**m, "argv": [*m["argv"], "--rl-identity", "relaxed"]} for m in members]
    moved = {**relaxed[0], "gpu": "nebius:2xh200@us-central1"}
    with pytest.raises(ValueError) as raised:
        fleet_session.check(cfg["trainer"]["argv"], [moved, *relaxed[1:]])
    text = str(raised.value)
    assert "compat_group nvidia-h200" in text and "compat_group nvidia-h100" in text
    assert "session_contract_mismatch" in text and "runs --rl-identity" not in text  # not the mode check


def test_run_up_preflight_refuses_islands_on_two_compat_groups(tmp_path, monkeypatch):
    from test_fleet_island_runup import FakeController, build
    from test_fleet_trainer_worker import FakeLaunches
    from yeto.rl import up

    run, objects = build(tmp_path, monkeypatch)
    run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
    run.adopt_desired()
    assert run.fleet_session_problem() is None
    dz.DesiredStore(objects).update(lambda doc: dz.trainers(doc)["sglang"].update(gpu=H200))
    run.adopt_desired()
    problem = run.fleet_session_problem()
    assert problem and "island sglang (learner 1): compat_group nvidia-h200" in problem
    logs = []
    monkeypatch.setattr(up, "_log", logs.append)
    assert run.preflight() is False


def test_an_island_of_another_trainer_shape_is_refused_naming_both_shapes(tmp_path):
    """The shape-bound full-parameter layout names each tensor by its rank's
    TP/PP/EP/CP/DP position: an island on more GPUs of the same card binds another layout."""
    cfg, members = _members(tmp_path)
    wider = {**members[0], "gpu": "modal:1x4xh100"}
    with pytest.raises(ValueError) as raised:
        fleet_session.check(cfg["trainer"]["argv"], [*members[1:], wider])
    text = str(raised.value)
    assert "trainer shape TP1 PP1 EP1 CP1 DP4" in text and "trains on TP1 PP1 EP1 CP1 DP2" in text
    assert "--rl-sync-layout shape-free" in text
    fleet_session.check(cfg["trainer"]["argv"], [*members[1:], {**members[0], "gpu": "aws:2xh100@us-west-2"}])
