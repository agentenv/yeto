from __future__ import annotations

import copy

import pytest

from yeto.rl import tbench_outcome

_REWARD_KEY_ENV = (
    "TBENCH_REWARD_HMAC_KEY",
    "TBENCH_REWARD_HMAC_KEY_FILE",
    "SECRLENV_REWARD_HMAC_KEY",
)


@pytest.fixture(autouse=True)
def _no_reward_keys(monkeypatch):
    for name in _REWARD_KEY_ENV:
        monkeypatch.delenv(name, raising=False)


def _metadata(**overrides):
    values = {
        "task_id": "configure-git-webserver",
        "sample_id": "baseline:configure-git-webserver:r0",
        "episode_id": "openenv-0123456789abcdef",
        "status": "completed",
        "reward": 1.0,
        "verifier": tbench_outcome.TEST_SH_VERIFIER,
        "testsh_rc": 0,
    }
    values.update(overrides)
    return tbench_outcome.build_outcome_metadata(**values)


def test_test_sh_and_native_outcomes_validate_without_any_key():
    metadata = _metadata()
    assert set(metadata) == {tbench_outcome.OUTCOME_KEY}
    outcome, reward = tbench_outcome.verified_outcome(metadata)
    assert reward == 1.0
    assert outcome["benchmark"] == "terminal-bench-2.1"
    assert outcome["sample_id"] == "baseline:configure-git-webserver:r0"

    native = _metadata(
        reward=0.0,
        verifier=tbench_outcome.NATIVE_VERIFIER,
        testsh_rc=None,
        status="max_turns",
    )
    assert tbench_outcome.verify_outcome(native) == 0.0


def test_timeout_cannot_claim_a_verdict():
    timeout = _metadata(
        status="timeout",
        reward=0.0,
        verifier=tbench_outcome.TIMEOUT_VERIFIER,
        testsh_rc=None,
    )
    assert tbench_outcome.verify_outcome(timeout) == 0.0

    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="claims a verifier verdict"
    ):
        _metadata(status="timeout", reward=0.0, testsh_rc=0)


def test_inconsistent_and_open_outcomes_are_rejected():
    metadata = _metadata()
    changed = copy.deepcopy(metadata)
    changed[tbench_outcome.OUTCOME_KEY]["reward"] = 0.0
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="pass bit differs"
    ):
        tbench_outcome.verified_outcome(changed)

    changed = copy.deepcopy(metadata)
    changed[tbench_outcome.OUTCOME_KEY]["unexpected"] = "untrusted"
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="fields are not closed"
    ):
        tbench_outcome.verified_outcome(changed)

    with pytest.raises(tbench_outcome.UntrustedTBenchOutcome, match="no Terminal-Bench"):
        tbench_outcome.verified_outcome({})


def test_out_of_range_reward_is_rejected():
    metadata = _metadata()
    metadata[tbench_outcome.OUTCOME_KEY]["reward"] = 0.5
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="finite and binary"
    ):
        tbench_outcome.verified_outcome(metadata)


def test_legacy_mac_field_is_ignored():
    metadata = _metadata()
    legacy = {**metadata, "tbench_trusted_outcome_hmac": "not-a-mac"}
    assert tbench_outcome.verified_outcome(legacy) == tbench_outcome.verified_outcome(
        metadata
    )


def test_reward_key_env_is_ignored(monkeypatch):
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", "short")
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY_FILE", "relative.key")
    assert tbench_outcome.verify_outcome(_metadata()) == 1.0
