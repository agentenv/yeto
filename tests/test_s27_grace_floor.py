"""S27 G-CG: grace floor of the syncer (``--grace-min-ms``,
``--grace-arrival-quantile``) and its launcher options.

With a deep fragment pipeline a round opens long before any push, so the
adaptive grace window (step time minus time to quorum) shrinks to a few
seconds while the islands deliver the same fragment 10-30 s apart.  The floor
waits for that gap.  It enters the semantic profile only when on, so every
existing hash and syncer command stays unchanged.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time

import numpy as np
import pytest
import test_rl_late_carry_real_syncer as late
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401 - fixtures
from test_rl_full_parameter_async import CARGO, needs_cargo
from test_rl_outer_merge_real_syncer import make_rig, syncer_binary  # noqa: F401 - fixtures
from test_s27_sync_boundary import ROOT

from yeto.syncer_profile import SYNCER_SEMANTIC_PROFILE_SCHEMA, SyncerSemanticProfile

# The Rust vector (server.rs GRACE_FLOOR_VECTOR) and the base vector of
# semantic_profile_hash_matches_the_python_canonical_vector.
BASE_VECTOR = "85a50d0e4fd5103b81a2295536fca0e5ad96cd50531fb1396208af8459be571e"
GRACE_FLOOR_VECTOR = "6f9104e302f894575fe2203cfe91a93ff37d16e37b4e75c4367b060d13748d12"
TERMINAL_GRACE_VECTOR = "6f40c5c1c02e6c63b99bf5aa77fab924fed928a42a9e0d88811583a19fc1aa39"

_BASE = {
    "schema": SYNCER_SEMANTIC_PROFILE_SCHEMA, "learners": 2, "quorum": 2, "grace_ms": 0,
    "grace_gamma": 0.8, "grace_tau": 2.0, "pipeline": 2, "min_round_interval_ms": 0,
    "sync_interval_steps": 24.0, "delta_correction": "none", "quorum_timeout_s": 900,
    "final_ack_timeout_s": 900, "total_steps": 8, "policy_sweep_fragments": None, "outer_lr": 0.7,
    "outer_momentum": 0.9, "checkpoint_enabled": True, "checkpoint_every": 1, "resume": True,
    "mark_final_checkpoint": True, "learner_budget_steps": None, "max_base_lag": 0,
    "learner_weight": "equal", "require_profile_binding": True,
}


def test_the_grace_floor_hash_matches_the_rust_vector_and_off_keeps_the_base():
    assert SyncerSemanticProfile.from_mapping(_BASE).sha256 == BASE_VECTOR
    off = SyncerSemanticProfile.from_mapping({**_BASE, "grace_min_ms": 0, "grace_arrival_quantile": 0.0})
    assert off.sha256 == BASE_VECTOR
    on = SyncerSemanticProfile.from_mapping({**_BASE, "grace_min_ms": 15000, "grace_arrival_quantile": 0.9})
    assert on.sha256 == GRACE_FLOOR_VECTOR
    only_quantile = SyncerSemanticProfile.from_mapping({**_BASE, "grace_arrival_quantile": 0.9})
    assert only_quantile.sha256 not in (BASE_VECTOR, GRACE_FLOOR_VECTOR)


def test_the_terminal_grace_hash_matches_the_rust_vector():
    assert SyncerSemanticProfile.from_mapping({**_BASE, "terminal_grace_ms": 0}).sha256 == BASE_VECTOR
    on = SyncerSemanticProfile.from_mapping({**_BASE, "terminal_grace_ms": 120000})
    assert on.sha256 == TERMINAL_GRACE_VECTOR


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_the_profile_refuses_a_quantile_outside_zero_one(quantile):
    with pytest.raises(ValueError, match="grace_arrival_quantile"):
        SyncerSemanticProfile.from_mapping({**_BASE, "grace_arrival_quantile": quantile})


def _plan(extra=()):
    from test_rl_full_parameter_async import ASYNC
    from test_rl_full_parameter_async import _plan as plan

    return plan(ASYNC + tuple(extra))


def _flag(command: str, flag: str):
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def test_the_launcher_passes_the_floor_to_syncer_profile_and_islands():
    from yeto import launcher

    args, plan = _plan(("--rl-sync-grace-min-ms", "15000", "--rl-sync-grace-arrival-quantile", "0.9"))
    [actor] = plan["role_syncers"]
    assert _flag(actor["command"], "--grace-min-ms") == "15000"
    assert _flag(actor["command"], "--grace-arrival-quantile") == "0.9"
    profile = launcher.rl_syncer_profile(args, 2)
    assert (profile.grace_min_ms, profile.grace_arrival_quantile) == (15000, 0.9)
    for island in plan["island_requests"]:
        assert _flag(island["learner_command"], "--syncer-profile-sha256") == profile.sha256

    # Off (the default): no flag, and the profile hash of the run without the options.
    base_args, base = _plan()
    [base_actor] = base["role_syncers"]
    assert "--grace-min-ms" not in base_actor["command"]
    assert "--grace-arrival-quantile" not in base_actor["command"]
    base_profile = launcher.rl_syncer_profile(base_args, 2)
    assert (base_profile.grace_min_ms, base_profile.grace_arrival_quantile) == (0, 0.0)
    assert base_profile.sha256 != profile.sha256
    assert base_actor["command"] == actor["command"].replace(
        " --grace-min-ms 15000 --grace-arrival-quantile 0.9", "")


@pytest.mark.parametrize(("extra", "terminal"), [
    ((), "120000"),  # the async mode's default with full-parameter async rounds
    (("--rl-sync-terminal-grace-ms", "45000"), "45000"),
    (("--rl-sync-terminal-grace-ms", "0"), None),
])
def test_the_async_mode_holds_last_rounds_open_by_default(extra, terminal):
    from yeto import launcher

    args, plan = _plan(("--rl-sync-mode", "async", *extra))
    [actor] = plan["role_syncers"]
    assert _flag(actor["command"], "--terminal-grace-ms") == terminal
    profile = launcher.rl_syncer_profile(args, 2)
    assert profile.terminal_grace_ms == (int(terminal) if terminal else 0)
    for island in plan["island_requests"]:
        assert _flag(island["learner_command"], "--syncer-profile-sha256") == profile.sha256


def test_the_strict_mode_and_the_test_plans_have_no_terminal_grace():
    _args, default = _plan()
    assert "--terminal-grace-ms" not in default["role_syncers"][0]["command"]
    _args, strict = _plan(("--rl-sync-mode", "strict"))
    assert "--terminal-grace-ms" not in strict["role_syncers"][0]["command"]


@pytest.mark.parametrize(("extra", "message"), [
    (("--rl-sync-grace-arrival-quantile", "0"), "must be in \\(0, 1\\]"),
    (("--rl-sync-grace-arrival-quantile", "1.5"), "must be in \\(0, 1\\]"),
    (("--rl-sync-grace-min-ms", "-1"), "must be >= 0"),
    (("--rl-sync-terminal-grace-ms", "-1"), "must be >= 0"),
])
def test_the_launcher_refuses_invalid_floor_values(extra, message):
    with pytest.raises((ValueError, SystemExit), match=message):
        _plan(extra)


def test_the_strict_sweep_refuses_the_floor():
    from test_rl_full_parameter_async import FULL
    from test_rl_full_parameter_async import _plan as plan

    with pytest.raises(ValueError, match="--rl-sync-grace-min-ms"):
        plan(FULL + ("--rl-sync-grace-min-ms", "15000"))


@needs_cargo
def test_the_syncer_refuses_a_quantile_outside_zero_one():
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    result = subprocess.run(
        [str(ROOT / "syncer/target/debug/yeto-syncer"), "--port", "0", "--learners", "2",
         "--total-steps", "4", "--grace-arrival-quantile", "1.5",
         "--allow-unauthenticated-islands"],
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode != 0
    assert "--grace-arrival-quantile must be in [0, 1]" in result.stderr


# -- real syncer: the adaptive window collapses, the floor still waits -----------


class _StepReportingIsland(late.ReactiveIsland):
    """A reactive island (answers each PULL ``work_s`` later) that reports
    ``STEPS_PER_PUSH`` local steps per push: the syncer's step time is then a
    quarter of the time a round waits for its first push, so the adaptive
    window has no slack (E1: rounds open ~34 s before a push, ~27 s steps)."""

    STEPS_PER_PUSH = 4

    def run(self):
        try:
            while not self.stop.is_set():
                pulls = self.client.drain_pulls()
                if not pulls:
                    time.sleep(0.005)
                    continue
                time.sleep(self.work_s)
                self.client.heartbeat(self.rounds)
                for update in self.client.drain_updates():
                    fid = update.fragment_id
                    self.version[fid] = max(self.version[fid], update.version)
                self.rounds += self.STEPS_PER_PUSH
                for pull in pulls:
                    fid = pull.fragment_id
                    self.client.push_fragment(fid, pull.global_step, pull.round_attempt, self.version[fid],
                                              self.rounds, 1, 10,
                                              late._f32(np.full(late.NUMEL, 0.01 * (self.index + 1))))
        except Exception as error:  # noqa: BLE001 - surfaced by the test
            if not self.stop.is_set():
                self.error = error


@needs_cargo
@pytest.mark.parametrize(("options", "least_direct", "most_direct"), [
    ((), 0, 4),  # control: the adaptive window is ~0 s, island 1 is ~0.2 s late
    (("--rl-sync-grace-min-ms", "600"), 15, 16),
    (("--rl-sync-grace-arrival-quantile", "0.9"), 10, 16),  # warm-up: a few late gaps first
])
def test_the_floor_waits_for_a_peer_when_the_adaptive_window_collapses(
        make_rig, monkeypatch, options, least_direct, most_direct):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "grace-floor-test-key")
    rounds = 16
    rig = make_rig(2, *late.ASYNC_K1, "--rl-sync-grace-ms", "5000", *options, total_steps=rounds + 2)
    rig.start(late._initial(rig.layout))
    stop = threading.Event()
    islands = [_StepReportingIsland(rig, 0, 0.5, stop), _StepReportingIsland(rig, 1, 0.7, stop)]
    for island in islands:
        island.start()
    deadline = time.monotonic() + 120 + rounds * 4
    try:
        while time.monotonic() < deadline:
            records = late._merge_records(rig)
            if all(t in records for t in range(1, rounds + 1)):
                break
            for island in islands:
                assert island.error is None, island.error
            time.sleep(0.1)
        else:
            raise TimeoutError(f"syncer merged only {sorted(late._merge_records(rig))}")
    finally:
        stop.set()
        for island in islands:
            island.join(timeout=5)
    records = [late._merge_records(rig)[t] for t in range(1, rounds + 1)]
    direct = sum(any(r["id"] == 1 for r in rec["responders"]) for rec in records)
    grace = [rec["grace_ms"] for rec in records]
    print(f"{options or 'no floor'}: island 1 direct in {direct}/{rounds} rounds; grace_ms {grace}")
    assert least_direct <= direct <= most_direct, (direct, grace)


@needs_cargo
@pytest.mark.parametrize("terminal_ms", ["0", "20000"])
def test_the_terminal_grace_merges_a_late_delta_of_a_last_round(make_rig, monkeypatch, terminal_ms):
    """P=2 fragments, one RL round: steps 1 and 2 are both last rounds, with
    no grace (--rl-sync-grace-ms 0). Island 1's delta for step 1 comes 0.5 s
    after island 0's. Without the terminal grace step 1 merges island 0
    alone, island 1's delta waits for a round that never comes and the final
    cut leaves it out (logged); with it step 1 merges both deltas."""
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "terminal-grace-test-key")
    rig = make_rig(2, *late.ASYNC_K1, "--rl-sync-grace-ms", "0",
                   "--rl-sync-terminal-grace-ms", terminal_ms, total_steps=1)
    assert "--total-steps 2" in rig.command
    initial = late._initial(rig.layout)
    rig.start(initial)
    ref = late.Reference(rig.layout, initial, late.DEFAULT_LR, late.DEFAULT_MU)
    d = late._deltas(rig.layout, 0, 11, 2)
    e = late._deltas(rig.layout, 1, 12, 2)

    rig.push(0, 1, d[0], base=0, c_steps=1, c_tokens=4)
    time.sleep(0.5)
    rig.push(1, 1, d[1], base=0, c_steps=1, c_tokens=4)
    if terminal_ms == "0":
        want = ref.step(0, [d[0]], [1.0])
    else:
        want = ref.step(0, [d[0], d[1]], [late._weight(1, 4), late._weight(1, 4)])
    late._check(rig.bcast(0, 0, 1), want)
    rig.push(0, 2, e[0], base=0, c_steps=1, c_tokens=4)
    rig.push(1, 2, e[1], base=0, c_steps=1, c_tokens=4)
    deadline = time.monotonic() + 30  # the last step is not broadcast: the final cut carries it
    while 2 not in late._merge_records(rig) and time.monotonic() < deadline:
        time.sleep(0.05)
    records = late._merge_records(rig)
    responders = [sorted(r["id"] for r in records[t]["responders"]) for t in (1, 2)]
    rig.close()
    if terminal_ms == "0":
        # Step 2 races the same way (island 1 pushes right after island 0).
        assert responders[0] == [0], responders
        assert "late pushes left out of the final cut: no later round" in rig.output
    else:
        assert responders == [[0, 1], [0, 1]], responders
        assert records[1]["grace_ms"] == 20000
        assert "late pushes left out of the final cut" not in rig.output
