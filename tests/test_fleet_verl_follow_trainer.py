"""Samplers that follow their own trainer by backend capability (rl-fleet-verl-islands
2.2): ``--rl-sampler-follow trainer`` with external sampler pools and ``--rl-release-every
N`` ask the backend's ``per_learner_release`` capability, not its name; sampler islands
(decoupled groups, the sampler stop count, follow-trainer sampler islands) ask
``sampler_islands_follow``, which only Miles declares: a verl fleet's trainers are
followed by external sampler pools only. The same external-pool scenarios run on Miles
and on verl and must decide the same, item by item; a verl trainer whose pools follow it
is told to release its own versions (``--rl-release-mode per_learner``). No cloud, no GPU."""

from __future__ import annotations

import re
import shlex
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_external_pool_follow import SHARDS, TRAINERS, TWO
from yeto.rl.engine import backend_identity as bi
from yeto.rl.engine import backends, sample_roles


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-verl-test-key")


def _ns(**over):
    base = dict(rl_backend="miles", rl_sampler_follow="trainer", rl_initial_adapter=None, rl_sample_role="trainer",
                rl_release_every=1, rl_release_skip_busy=False, rl_max_policy_age=2, tuning="full",
                rl_release_inline=False, rl_sync_preset="full-parameter")
    return SimpleNamespace(**{**base, **over})


def _decide(call):
    try:
        out = call()
    except ValueError as error:
        # the refusal, without the backend's own name (the verdict is compared)
        return ("refused", re.sub(r"--rl-backend (miles|verl)( does not| releases)", r"--rl-backend X\2",
                                  str(error)).split(";")[0])
    return ("ok", out)


def _events(backend: str) -> list[tuple[str, object]]:
    """Every decision of the four follow-trainer sites for one backend."""
    out = []
    # decoupled synced group without following samplers: refused on both, before the backend
    out.append(("decoupled, merged samplers", _decide(lambda: sample_roles.check_decoupled_group(
        _ns(rl_backend=backend, rl_sampler_follow="merged")))))
    # the follow check of synced trainers fed by external pools
    def pools():
        args = _follow(_ns(rl_backend=backend))
        args.rl_island_buffer_shards = ["0:a", "1:b"]
        return sample_roles.check_sampler_follow(args, 2)

    out.append(("external pools follow", _decide(pools)))
    # release cadence of a trainer whose pools follow it
    follow = SimpleNamespace(**vars(_ns(rl_backend=backend)))
    for every, age in ((1, 2), (2, 2), (2, 3), (3, 2), (4, 4)):
        args = _ns(rl_backend=backend, rl_release_every=every, rl_max_policy_age=age)
        out.append((f"release every {every}, age {age}", _decide(
            lambda a=args: _cadence(_follow(a)))))
    out.append(("release every 2 on a trainer whose samplers do not follow it",
                "refused" if backend == "verl" else _decide(lambda: _cadence(
                    _ns(rl_backend=backend, rl_release_every=2, rl_sampler_follow="merged")))[0]))
    del follow
    return out


def _cadence(args):
    return sample_roles._check_release_cadence(args, sample_roles.TRAINER)  # noqa: SLF001


def _follow(args):
    """``args`` as a synced trainer fed by external pools that follow it."""
    args.syncer = "10.0.0.1:29400"
    args.rl_single_island_no_sync = False
    args.rl_island_buffer_shards = ["0:a"]
    return args


def test_miles_and_verl_decide_the_same_where_both_release_per_learner(monkeypatch):
    monkeypatch.setattr(sample_roles, "follows_trainer", lambda args: args.rl_sampler_follow == "trainer")
    miles, verl = _events("miles"), _events("verl")
    assert [name for name, _ in miles] == [name for name, _ in verl]
    for (name, a), (_, b) in zip(miles, verl):
        if name == "release every 2 on a trainer whose samplers do not follow it":
            continue  # verl honours N only per learner (release_cadence is Miles'); asserted below
        assert a == b, name
    assert dict(miles)["release every 3, age 2"][0] == "refused"  # the policy-age floor holds for both
    assert dict(verl)["release every 2, age 3"] == ("ok", None)


def test_a_backend_that_does_not_release_per_learner_is_refused_at_every_site(monkeypatch):
    real = backends.capabilities
    monkeypatch.setattr(backends, "capabilities",
                        lambda name=None: real(name) - {bi.PER_LEARNER_RELEASE} if name == "verl" else real(name))
    with pytest.raises(ValueError, match=r"--rl-sampler-follow trainer needs a backend whose trainers release their "
                                         r"own local versions \(per_learner_release: --rl-backend miles only\); "
                                         r"--rl-backend verl does not"):
        sample_roles.check_sampler_follow(_follow(_ns(rl_backend="verl")), 2)
    with pytest.raises(ValueError, match="--rl-backend miles \\(release_cadence\\)"):
        _cadence(_follow(_ns(rl_backend="verl", rl_release_every=2)))


def test_verl_skip_busy_and_a_merged_cadence_stay_refused():
    with pytest.raises(ValueError, match="no --rl-release-skip-busy; --rl-backend verl releases every version"):
        _cadence(_follow(_ns(rl_backend="verl", rl_release_skip_busy=True, rl_max_policy_age=3)))
    with pytest.raises(ValueError, match="--rl-sampler-follow trainer"):
        _cadence(_ns(rl_backend="verl", rl_release_every=2, rl_sampler_follow="merged"))


# -- the launch: the islands' flags -----------------------------------------------------------

def _island_flags(backend, *extra):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    flags = (("--rl-backend", "verl") if backend == "verl" else ()) + ("--inner-lr", "1e-6")
    args = _launcher_args("ports", (*TRAINERS, *SHARDS, *flags, *extra), gpu=TWO)
    args.controller = "local"
    _prepare_rl_args(args)
    plan = dry_run_plan(args)
    out = []
    for island in plan["island_requests"]:
        tokens = shlex.split(island["learner_command"].split("island_entry", 1)[1])
        picked = {}
        for flag in ("--rl-sample-store", "--rl-sample-lanes", "--rl-release-every", "--rl-release-mode",
                     "--rl-max-policy-age"):
            if flag in tokens:
                picked[flag] = tokens[tokens.index(flag) + 1]
        out.append((island.get("sample_role"), picked))
    return plan, out


def test_the_same_fleet_launch_on_miles_and_verl_gives_each_trainer_its_own_store(capsys):
    miles_plan, miles = _island_flags("miles", "--rl-island-release-every", "1:2")
    verl_plan, verl = _island_flags("verl", "--rl-island-release-every", "1:2")
    assert [role for role, _ in miles] == [role for role, _ in verl] == ["trainer", "trainer"]
    for (_, m), (_, v) in zip(miles, verl):
        # the same store, lane, cadence and age; verl is told to release its own versions
        assert {k: x for k, x in v.items() if k != "--rl-release-mode"} == m
        assert v["--rl-release-mode"] == "per_learner" and "--rl-release-mode" not in m
    assert [m["--rl-sample-store"].rsplit("/", 1)[1] for _, m in verl] == ["trainer0", "trainer1"]
    assert [m["--rl-sample-lanes"] for _, m in verl] == ["1", "1"]
    assert "--rl-release-every" not in verl[0][1] and verl[1][1]["--rl-release-every"] == "2"
    assert miles_plan["outer_sync"] and verl_plan["outer_sync"]


def test_verl_without_per_learner_releases_cannot_follow_its_trainer(monkeypatch):
    real = backends.capabilities
    monkeypatch.setattr(backends, "capabilities",
                        lambda name=None: real(name) - {bi.PER_LEARNER_RELEASE} if name == "verl" else real(name))
    with pytest.raises(ValueError, match="--rl-sampler-follow trainer needs a backend whose trainers release"):
        _island_flags("verl")


def _group(backend: str, *extra: str):
    from yeto.launcher import _prepare_rl_args

    four = ",".join(["modal:1xh100"] * 4)
    flags = (("--rl-backend", "verl") if backend == "verl" else ()) + ("--inner-lr", "1e-6")
    args = _launcher_args("ports", (*TRAINERS, *flags, "--rl-island-sample-role", "2:sampler",
                                    "--rl-island-sample-role", "3:sampler", *extra), gpu=four)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def test_miles_sampler_islands_stop_with_their_own_trainer(capsys):
    from yeto.launcher import dry_run_plan

    args = _group("miles")
    assert sample_roles.sampler_stop_trainers(args, 4) == {2: 1, 3: 1}
    until = []
    for island in dry_run_plan(args)["island_requests"]:
        tokens = shlex.split(island["learner_command"].split("island_entry", 1)[1])
        flag = "--rl-sampler-until-trainers-end"
        until.append((island.get("sample_role"), tokens[tokens.index(flag) + 1] if flag in tokens else None,
                      "--rl-release-mode" in tokens))
    assert until == [("trainer", None, False), ("trainer", None, False), ("sampler", "1", False),
                     ("sampler", "1", False)]


def test_verl_sampler_islands_are_refused_its_fleet_takes_external_pools_only():
    """A verl sampler island runs a fixed round count and names no trainer (its engine
    spec reads neither the stop count nor the trainer it feeds), so a synced group whose
    samplers are verl sampler islands is refused, follow-trainer or decoupled; external
    sampler pools (the island fleet) follow verl trainers."""
    external = "follows its trainers with external sampler pools only"
    with pytest.raises(ValueError, match=r"--rl-sampler-follow trainer with sampler islands .* "
                                         r"\(sampler_islands_follow: --rl-backend miles only\); "
                                         rf"--rl-backend verl {external}"):
        _group("verl")
    with pytest.raises(ValueError, match=rf"a decoupled synced group's sampler islands need .*--rl-backend verl "
                                         rf"{external}"):
        sample_roles.check_decoupled_group(_ns(rl_backend="verl"))
    sample_roles.check_decoupled_group(_ns(rl_backend="miles"))


def test_the_sampler_stop_count_asks_the_sampler_islands_capability(monkeypatch):
    args = _group("miles")
    assert sample_roles.sampler_stop_trainers(args, 4) == {2: 1, 3: 1}
    # the same group as verl would see it: no stop count (its sampler islands do not stop with it)
    assert sample_roles.sampler_stop_trainers(SimpleNamespace(**{**vars(args), "rl_backend": "verl"}), 4) == {}
    real = backends.capabilities
    monkeypatch.setattr(backends, "capabilities",
                        lambda name=None: real(name) - {bi.SAMPLER_ISLANDS_FOLLOW} if name == "miles" else real(name))
    assert sample_roles.sampler_stop_trainers(args, 4) == {}
    assert sample_roles.describe_sampler_stop(args, 4) is None


def test_verl_declares_both_its_fleet_capabilities():
    """Both the fleet admission (fleet_full_parameter) and the per-learner release
    (per_learner_release) are verl's: a merge that kept one side would refuse every verl
    fleet, or every verl follow-trainer launch."""
    verl = backends.capabilities("verl")
    assert {bi.FLEET_FULL_PARAMETER, bi.PER_LEARNER_RELEASE} <= verl
    assert not verl & {bi.FLEET_PARTIAL_EXPERTS, bi.RELEASE_CADENCE, bi.COMPACTION_SEGMENTS,
                       bi.SAMPLER_ISLANDS_FOLLOW}
    assert bi.SAMPLER_ISLANDS_FOLLOW in backends.capabilities("miles")
