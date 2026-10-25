"""``--rl-peer-recovery`` in the launcher: syncer and island flags, refusals,
and the relaunch environment (CPU, dry-run plans only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import ASYNC, FULL, TWO_ISLANDS, _flag, _launcher_args, _plan
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


POOL = ("--rl-sync-quorum-theta", "0.75", "--rl-sync-max-base-lag", "2", "--rl-sync-island-lease-s", "45")


def test_island_pool_takes_peer_recovery_with_one_epoch_flag():
    """The pool decides who fills a slot; peer recovery is that island's state source."""
    args, plan = _plan(RECOVERY + POOL)
    assert args.rl_island_state_source == "peer"
    command = plan["role_syncers"][0]["command"].split()
    assert command.count("--syncer-epoch") == 1, "pool and recovery share one syncer epoch"
    assert command[command.index("--syncer-epoch") + 1] == "4"
    assert command[command.index("--island-lease-s") + 1] == "45.0"
    assert command[command.index("--recovery-bound-steps") + 1] == "8"
    for island in plan["island_requests"]:
        words = island["learner_command"].split()
        assert words.count("--rl-syncer-epoch") == 1
        assert words[words.index("--rl-syncer-epoch") + 1] == "4"
        assert "--rl-island-pool" in words
        assert _flag(island["learner_command"], "--rl-peer-recovery-store") == "/shared/run-store"
    assert relaunch_envs(args) == {PEER_RECOVER_ENV: "1"}


def test_island_pool_with_catch_up_source_has_no_recovery():
    args, plan = _plan(ASYNC + POOL + ("--rl-island-state-source", "catch-up"))
    assert args.rl_peer_recovery is False
    assert "--recovery-bound-steps" not in plan["role_syncers"][0]["command"]
    assert "--rl-peer-recovery-store" not in plan["island_requests"][0]["learner_command"]
    assert relaunch_envs(args) == {}


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
        (ASYNC + ("--rl-syncer-epoch", "2"), "needs island pool options"),
        (RECOVERY + ("--syncer-shards", "2"), "one syncer process"),
        (RECOVERY + ("--rl-island-state-source", "catch-up"), "contradicts --rl-peer-recovery"),
        (ASYNC + ("--no-rl-peer-recovery", "--rl-island-state-source", "peer"),
         "contradicts --no-rl-peer-recovery"),
        (ASYNC + ("--rl-island-state-source", "cut"), "own round cuts"),
        (ASYNC + ("--rl-island-state-source", "peer"), "--rl-peer-recovery-store or --rl-checkpoint-store"),
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
    assert options.timeout_s == 3600.0
    longer = SimpleNamespace(rl_peer_recovery_store=str(tmp_path), rl_peer_recovery_ttl_s=7200.0)
    assert peer_recovery_options(longer, environ={}).timeout_s == 7200.0, "a recovery waits as long as the TTL"
    assert peer_recovery_options(SimpleNamespace(), environ={}) is None
    with pytest.raises(ValueError, match="need --rl-peer-recovery-store"):
        peer_recovery_options(SimpleNamespace(rl_peer_recover=True), environ={})


def test_fault_injection_turns_a_relaunch_into_a_catch_up_rejoin(tmp_path, capsys):
    import json

    from yeto.rl.adapters.miles.island_entry import peer_recovery_options
    from yeto.rl.engine.driver import FAULT_INJECTION_ENV, MODAL_INCARNATION_ENV

    fault = tmp_path / "fault.json"
    fault.write_text(json.dumps({"exit_at_round": 5, "exit_islands": [1], "exit_rejoined_after_rounds": 9,
                                 "catch_up_incarnations": [2]}))
    args = SimpleNamespace(rl_peer_recovery_store=str(tmp_path), rl_peer_recovery_ttl_s=None)
    env = {PEER_RECOVER_ENV: "1", FAULT_INJECTION_ENV: str(fault)}
    assert peer_recovery_options(args, environ={**env, MODAL_INCARNATION_ENV: "1"}).recover is True
    options = peer_recovery_options(args, environ={**env, MODAL_INCARNATION_ENV: "2"})
    assert options.recover is False and options.store == str(tmp_path)
    assert "re-run 2 rejoins by catch-up" in capsys.readouterr().err
