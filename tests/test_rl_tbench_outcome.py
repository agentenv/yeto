from __future__ import annotations

import copy

import pytest

from yeto.rl import tbench_outcome


def _signed(**overrides):
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
    return tbench_outcome.build_signed_metadata(**values)


def test_signed_test_sh_and_native_outcomes_verify(monkeypatch):
    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "t" * 48)
    metadata = _signed()
    outcome, reward = tbench_outcome.verified_outcome(metadata)
    assert reward == 1.0
    assert outcome["benchmark"] == "terminal-bench-2.1"
    assert outcome["sample_id"] == "baseline:configure-git-webserver:r0"

    native = _signed(
        reward=0.0,
        verifier=tbench_outcome.NATIVE_VERIFIER,
        testsh_rc=None,
        status="max_turns",
    )
    assert tbench_outcome.verify_outcome(native) == 0.0


def test_timeout_is_signed_but_cannot_claim_a_verdict(monkeypatch):
    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "t" * 48)
    timeout = _signed(
        status="timeout",
        reward=0.0,
        verifier=tbench_outcome.TIMEOUT_VERIFIER,
        testsh_rc=None,
    )
    assert tbench_outcome.verify_outcome(timeout) == 0.0

    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="claims a verifier verdict"
    ):
        _signed(status="timeout", reward=0.0, testsh_rc=0)


def test_forgery_extra_fields_and_wrong_key_fail_closed(monkeypatch):
    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "t" * 48)
    metadata = _signed()
    forged = copy.deepcopy(metadata)
    forged[tbench_outcome.OUTCOME_KEY]["reward"] = 0.0
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="pass bit differs"
    ):
        tbench_outcome.verified_outcome(forged)

    forged = copy.deepcopy(metadata)
    forged[tbench_outcome.OUTCOME_KEY]["unexpected"] = "untrusted"
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="fields are not closed"
    ):
        tbench_outcome.verified_outcome(forged)

    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "x" * 48)
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="signature mismatch"
    ):
        tbench_outcome.verified_outcome(metadata)


def test_dedicated_key_is_required(monkeypatch):
    monkeypatch.delenv(tbench_outcome.HMAC_ENV, raising=False)
    monkeypatch.delenv(tbench_outcome.HMAC_FILE_ENV, raising=False)
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="missing or outside"
    ):
        _signed()


def test_private_key_file_is_supported_without_direct_env(tmp_path, monkeypatch):
    monkeypatch.delenv(tbench_outcome.HMAC_ENV, raising=False)
    key_file = tmp_path / "tbench.key"
    key_file.write_text("f" * 48 + "\n")
    key_file.chmod(0o400)
    monkeypatch.setenv(tbench_outcome.HMAC_FILE_ENV, str(key_file))
    assert tbench_outcome.verify_outcome(_signed()) == 1.0

    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "d" * 48)
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="mutually exclusive"
    ):
        _signed()


def test_key_file_rejects_relative_symlink_public_and_oversized_paths(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(tbench_outcome.HMAC_ENV, raising=False)
    real = tmp_path / "real.key"
    real.write_bytes(b"f" * 48)
    real.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(real)

    for value, message in (
        ("relative.key", "absolute and non-symlink"),
        (str(link), "absolute and non-symlink"),
    ):
        monkeypatch.setenv(tbench_outcome.HMAC_FILE_ENV, value)
        with pytest.raises(tbench_outcome.UntrustedTBenchOutcome, match=message):
            _signed()

    real.chmod(0o644)
    monkeypatch.setenv(tbench_outcome.HMAC_FILE_ENV, str(real))
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="private and regular"
    ):
        _signed()

    real.write_bytes(b"f" * 4097)
    real.chmod(0o600)
    with pytest.raises(
        tbench_outcome.UntrustedTBenchOutcome, match="outside its size bound"
    ):
        _signed()
