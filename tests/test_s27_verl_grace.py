"""S27 G-EN: the syncer's grace floor, arrival quantile and terminal grace on a
verl full-parameter launch (VERL-PARITY #5, #6; s27-sync-crit c2045a36, in
s27-next).  They are syncer options: a verl launch sends them to the syncer and
binds them into the syncer profile; the verl island's own argv changes only by
that profile hash (the same rule as Miles).  CPU only.
"""
from __future__ import annotations

import shlex

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import _prepared, _verl_args

PROFILE_FLAG = "--syncer-profile-sha256"


def _plan(extra=()):
    from yeto.launcher import dry_run_plan

    args = _prepared(_verl_args("--rl-sync-mode", "async", *extra))
    return args, dry_run_plan(args)


def _flag(command: str, flag: str):
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def _without_profile(command: str) -> list[str]:
    tokens = shlex.split(command)
    at = tokens.index(PROFILE_FLAG)
    return tokens[:at] + tokens[at + 2:]


def test_a_verl_launch_sends_the_grace_options_to_its_syncer_and_profile_only():
    from yeto import launcher

    args, plan = _plan(("--rl-sync-grace-min-ms", "15000", "--rl-sync-grace-arrival-quantile", "0.9",
                        "--rl-sync-terminal-grace-ms", "45000"))
    [actor] = plan["role_syncers"]
    assert _flag(actor["command"], "--grace-min-ms") == "15000"
    assert _flag(actor["command"], "--grace-arrival-quantile") == "0.9"
    assert _flag(actor["command"], "--terminal-grace-ms") == "45000"
    profile = launcher.rl_syncer_profile(args, 2)
    assert (profile.grace_min_ms, profile.grace_arrival_quantile, profile.terminal_grace_ms) == (15000, 0.9, 45000)

    base_args, base = _plan()
    [base_actor] = base["role_syncers"]
    assert "--grace-min-ms" not in base_actor["command"]
    assert _flag(base_actor["command"], "--terminal-grace-ms") == "120000"  # the async default, as on Miles
    assert launcher.rl_syncer_profile(base_args, 2).sha256 != profile.sha256
    for island, base_island in zip(plan["island_requests"], base["island_requests"], strict=True):
        assert _flag(island["learner_command"], PROFILE_FLAG) == profile.sha256
        assert _without_profile(island["learner_command"]) == _without_profile(base_island["learner_command"])
