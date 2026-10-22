"""``--rl-peer-recovery`` in the launcher: syncer and island flags, refusals,
and the relaunch environment (CPU, dry-run plans only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import ASYNC, FULL, TWO_ISLANDS, _flag, _plan
from yeto.launch.fleet import FleetController
from yeto.launch.peer_recovery import PEER_RECOVER_ENV, relaunch_envs

RECOVERY = ASYNC + ("--rl-peer-recovery", "--rl-checkpoint-store", "/shared/run-store",
                    "--rl-recovery-bound-steps", "8", "--rl-recovery-max-donor-tries", "2",
                    "--rl-syncer-epoch", "4")


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "peer-recovery-test-key")


def test_peer_recovery_reaches_the_syncer_and_every_island():
    args, plan = _plan(RECOVERY)
    command = plan["role_syncers"][0]["command"]
    assert {flag: _flag(command, flag) for flag in (
        "--syncer-epoch", "--recovery-bound-steps", "--recovery-max-donor-tries")} == {
        "--syncer-epoch": "4", "--recovery-bound-steps": "8", "--recovery-max-donor-tries": "2"}
    assert "--recovery-donor-timeout-s" not in command
    for island in plan["island_requests"]:
        island_command = island["learner_command"]
        assert _flag(island_command, "--rl-syncer-epoch") == "4"
        assert _flag(island_command, "--rl-peer-recovery-store") == "/shared/run-store"
        assert "--rl-peer-recover " not in island_command, "a first launch never recovers"


def test_explicit_store_and_ttl():
    _, plan = _plan(ASYNC + ("--rl-peer-recovery", "--rl-peer-recovery-store", "~/nfs/snapshots",
                             "--rl-peer-recovery-ttl-s", "900"))
    island_command = plan["island_requests"][0]["learner_command"]
    assert _flag(island_command, "--rl-peer-recovery-store") == "~/nfs/snapshots"
    assert _flag(island_command, "--rl-peer-recovery-ttl-s") == "900.0"
    assert _flag(plan["role_syncers"][0]["command"], "--syncer-epoch") == "0"


def test_without_peer_recovery_commands_are_unchanged():
    _, plan = _plan(ASYNC)
    assert "--syncer-epoch" not in plan["role_syncers"][0]["command"]
    assert "--rl-peer-recovery-store" not in plan["island_requests"][0]["learner_command"]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (FULL + ("--rl-peer-recovery", "--rl-checkpoint-store", "/s"), "full-parameter-rounds async"),
        (ASYNC + ("--rl-peer-recovery",), "--rl-peer-recovery-store or --rl-checkpoint-store"),
        (ASYNC + ("--rl-recovery-bound-steps", "4"), "need --rl-peer-recovery"),
        (ASYNC + ("--rl-peer-recovery-store", "/s"), "need --rl-peer-recovery"),
        (RECOVERY + ("--rl-recovery-max-donor-tries", "0"), "must be >= 1"),
        (ASYNC + ("--rl-syncer-epoch", "2"), "need --rl-island-scheduling elastic"),
        (RECOVERY + ("--syncer-shards", "2"), "one syncer process"),
    ],
)
def test_refusals(extra, message):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


class _Task:
    def __init__(self):
        self.envs = {}

    def update_envs(self, envs):
        self.envs.update(envs)


def test_relaunched_island_gets_the_recover_env_and_the_syncer_does_not():
    assert relaunch_envs(SimpleNamespace(rl_peer_recovery=False)) == {}
    envs = relaunch_envs(SimpleNamespace(rl_peer_recovery=True))
    controller = FleetController(learners={}, syncer=None, sky_ops=None, poll_interval=1.0,
                                 recover_timeout=60.0, island_relaunch_envs=envs)
    sky_task, modal_config = _Task(), SimpleNamespace(envs={"A": "1"})
    controller._add_relaunch_envs(sky_task)
    controller._add_relaunch_envs(modal_config)
    assert sky_task.envs == {PEER_RECOVER_ENV: "1"}
    assert modal_config.envs == {"A": "1", PEER_RECOVER_ENV: "1"}


def test_island_entry_reads_the_recover_env(tmp_path):
    from yeto.rl.adapters.miles.island_entry import PEER_RECOVER_ENV as ISLAND_ENV
    from yeto.rl.adapters.miles.island_entry import peer_recovery_options

    assert ISLAND_ENV == PEER_RECOVER_ENV
    args = SimpleNamespace(rl_peer_recovery_store=str(tmp_path), rl_peer_recovery_ttl_s=None)
    assert peer_recovery_options(args, environ={}).recover is False
    options = peer_recovery_options(args, environ={PEER_RECOVER_ENV: "1"})
    assert options.recover and options.ttl_s == 3600.0 and options.store == str(tmp_path)
    assert peer_recovery_options(SimpleNamespace(), environ={}) is None
    with pytest.raises(ValueError, match="need --rl-peer-recovery-store"):
        peer_recovery_options(SimpleNamespace(rl_peer_recover=True), environ={})
