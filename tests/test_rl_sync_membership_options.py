"""Membership options (--rl-sync-quorum-theta, ...) for every synced RL preset. No Ray.

strict-avg, decoupled and full-parameter with asynchronous rounds take the
same --rl-sync-* membership options. Each set option reaches the role
syncer as the syncer flag of the same name; the islands get the pool switch,
the syncer epoch and the lease. Unset, the syncer and island commands stay
the previous ones. The full-parameter sweep, SAO, --experimental-rl-sync,
several syncer shards and critics are refused.
"""

from __future__ import annotations

import shlex

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_algorithm_provenance import _launcher_args as _provenance_launcher_args


def _launcher_args(engine, extra=(), gpu="aws:1xa100@us-east-1"):
    """These tests pin the strict control (--rl-sync-mode strict); the async
    defaults are covered in test_rl_sync_mode.py."""
    return _provenance_launcher_args(engine, ("--rl-sync-mode", "strict", *extra), gpu=gpu)

TWO_ISLANDS = "modal:2xh100,modal:2xh100"
THREE_ISLANDS = "modal:2xh100,modal:2xh100,modal:2xh100"
DECOUPLED = ("--rl-sync-preset", "decoupled", "--fragments", "4", "--pipeline", "2",
             "--local-rl-rounds-per-sync", "3")
FULL_PARAMETER = ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
                  "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
                  "--fragments", "4")
FULL_ASYNC = FULL_PARAMETER + ("--rl-sync-full-parameter-rounds", "async")
PRESETS = pytest.mark.parametrize("preset", [(), DECOUPLED, FULL_ASYNC],
                                  ids=["strict-avg", "decoupled", "full-parameter-async"])
# A count quorum below the islands lets rounds merge without an island.
QUORUM = ("--rl-sync-quorum", "2")
MEMBERSHIP = ("--rl-sync-q-min", "2", "--rl-sync-soft-deadline-s", "300",
              "--rl-sync-carry-gamma", "0.5", "--rl-sync-island-lease-s", "60",
              "--rl-sync-final-grace-s", "45")
SYNCER_MEMBERSHIP = (" --q-min 2 --carry-gamma 0.5 --soft-deadline-s 300 --island-lease-s 60.0"
                     " --syncer-epoch 0 --final-grace-s 45")
ISLAND_POOL = " --rl-island-pool --rl-syncer-epoch 0 --rl-sync-island-lease-s 60.0"


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")


def _plan(extra=(), gpu=THREE_ISLANDS):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _syncer(plan) -> str:
    [actor] = plan["role_syncers"]
    return actor["command"]


def _islands(plan) -> list[str]:
    return [island["learner_command"] for island in plan["island_requests"]]


@PRESETS
def test_unset_options_keep_every_command(preset):
    _, plan = _plan(preset + QUORUM)
    for flag in ("--quorum-theta", "--q-min", "--carry-gamma", "--soft-deadline-s",
                 "--island-lease-s", "--max-learners", "--final-grace-s", "--syncer-epoch"):
        assert flag not in _syncer(plan)
    assert all("--rl-island-pool" not in command and "--rl-sync-island-lease-s" not in command
               for command in _islands(plan))


@PRESETS
def test_membership_options_add_only_their_syncer_and_island_flags(preset):
    _, baseline = _plan(preset + QUORUM)
    args, plan = _plan(preset + QUORUM + MEMBERSHIP)
    command = _syncer(plan)
    assert SYNCER_MEMBERSHIP in command
    assert command.replace(SYNCER_MEMBERSHIP, "", 1) == _syncer(baseline)
    for pooled, fixed in zip(_islands(plan), _islands(baseline), strict=True):
        assert ISLAND_POOL in pooled
        assert pooled.replace(ISLAND_POOL, "", 1) == fixed
    assert args.quorum == 2


@PRESETS
def test_quorum_theta_replaces_the_count_quorum(preset):
    args, plan = _plan(preset + ("--rl-sync-quorum-theta", "0.75"))
    tokens = shlex.split(_syncer(plan))
    assert tokens[tokens.index("--quorum") + 1] == "1" and args.quorum == 1
    assert " --quorum-theta 0.75 --island-lease-s 150.0 --syncer-epoch 0" in _syncer(plan)
    with pytest.raises(ValueError, match="replaces --rl-sync-quorum"):
        _plan(preset + ("--rl-sync-quorum-theta", "0.75") + QUORUM)


@pytest.mark.parametrize("preset", [(), DECOUPLED], ids=["strict-avg", "decoupled"])
def test_max_learners_reaches_the_syncer(preset):
    _, plan = _plan(preset + QUORUM + ("--rl-sync-max-learners", "5"))
    assert " --island-lease-s 150.0 --max-learners 5 --syncer-epoch 0" in _syncer(plan)
    with pytest.raises(ValueError, match="at least the 3 islands"):
        _plan(preset + QUORUM + ("--rl-sync-max-learners", "2"))


@PRESETS
def test_options_need_rounds_that_may_merge_without_an_island(preset):
    with pytest.raises(ValueError, match="merge without some island"):
        _plan(preset + ("--rl-sync-island-lease-s", "30"))


def test_pool_island_configs_carry_the_lease_and_epoch():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.island_entry import _island_pool_membership

    assert _island_pool_membership(SimpleNamespace(rl_island_pool=False)) == {}
    assert _island_pool_membership(SimpleNamespace(
        rl_island_pool=True, rl_sync_island_lease_s=60.0, rl_syncer_epoch=3)) == {
        "island_lease_s": 60.0, "syncer_epoch": 3}
    assert _island_pool_membership(SimpleNamespace(
        rl_island_pool=True, rl_sync_island_lease_s=None, rl_syncer_epoch=None)) == {
        "island_lease_s": 150.0, "syncer_epoch": 0}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (FULL_PARAMETER + ("--rl-sync-island-lease-s", "30"),
         "membership options need --rl-sync-full-parameter-rounds async"),
        (FULL_ASYNC + QUORUM + ("--rl-sync-max-learners", "4"),
         "--rl-sync-max-learners does not apply to full-parameter"),
        (("--rl-sync-quorum-theta", "1.5"), r"in \(0, 1\]"),
        (QUORUM + ("--rl-sync-q-min", "0"), "q-min must be >= 1"),
    ],
    ids=["full-parameter-sweep", "full-parameter-max-learners", "theta-range", "q-min"],
)
def test_refusals(extra, message):
    from yeto.launcher import _prepare_rl_args, syncer_command

    args = _launcher_args("ports", tuple(extra), gpu=THREE_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)
        syncer_command(args, 3)


@pytest.mark.parametrize(
    ("fields", "critic", "message"),
    [
        (dict(syncer_shards=2), False, "no --syncer-shards > 1"),
        (dict(experimental_rl_sync=True), False, "apply to --rl-sync-preset strict-avg, decoupled"),
        (dict(rl_sync_preset="sao-streaming-full"), False,
         "apply to --rl-sync-preset strict-avg, decoupled"),
        ({}, True, "do not support a critic"),
    ],
    ids=["syncer-shards", "experimental", "sao", "critic"],
)
def test_presets_without_a_pool_session_are_refused(monkeypatch, fields, critic, message):
    from types import SimpleNamespace

    from yeto.launch import rl_args

    monkeypatch.setattr(rl_args, "rl_needs_critic", lambda args: critic)
    args = SimpleNamespace(**{"training_mode": "rl", "rl_sync_island_lease_s": 30.0,
                              "rl_sync_preset": "strict-avg", "experimental_rl_sync": False,
                              "syncer_shards": None, **fields})
    with pytest.raises(ValueError, match=message):
        rl_args._check_island_pool(args, num_islands=2)


def test_role_stream_lease_needs_asynchronous_rounds():
    from dataclasses import replace

    from test_rl_full_parameter_sync import _role_stream_config

    strict = _role_stream_config("actor", 29500)
    with pytest.raises(ValueError, match="island lease needs asynchronous rounds"):
        replace(strict, island_lease_s=30.0)
    pooled = replace(strict, partial_quorum=True, island_lease_s=30.0, syncer_epoch=2)
    assert pooled.island_lease_s == 30.0
