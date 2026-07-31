#!/usr/bin/env python3
"""Test the CyberGym submission contract."""

import hashlib
import pytest

from yeto.rl.envs.cybergym_env import CyberGymEnv


def test_checksum():
    """Test checksum computation."""
    task_id = "arvo:10400"
    agent_id = "yeto_agent"
    salt = "CyberGym"
    expected = hashlib.sha256(f"{task_id}{agent_id}{salt}".encode('utf-8')).hexdigest()
    actual = CyberGymEnv.compute_checksum(task_id, agent_id, salt)
    assert actual == expected, f"Checksum mismatch: {actual} != {expected}"


def test_reward_semantics():
    """Test reward logic."""
    assert CyberGymEnv.compute_reward(0) == -1.0
    assert CyberGymEnv.compute_reward(300) == -1.0
    assert CyberGymEnv.compute_reward(-1) == -1.0   # missing code
    assert CyberGymEnv.compute_reward(None) == -1.0   # safety
    for code in (1, 2, 100, 137, 139, 255):
        assert CyberGymEnv.compute_reward(code) == 1.0


def test_server_error_is_not_used_as_training_reward(monkeypatch):
    """Infrastructure errors must abort rather than look like failed PoCs."""

    class Response:
        status_code = 500
        text = '{"detail":"No such image: n132/arvo:47101-vul"}'

    env = CyberGymEnv(task_ids=["arvo:47101"])
    env.reset()
    env._server_checked = True
    monkeypatch.setattr(
        "yeto.rl.envs.cybergym_env.requests.post",
        lambda *args, **kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match="HTTP 500.*No such image"):
        env.step("test")


@pytest.mark.integration
def test_connectivity():
    """Integration test: check that the server is reachable (optional)."""
    import requests
    try:
        resp = requests.options("http://127.0.0.1:8666/submit-vul", timeout=5)
        assert resp.status_code < 500, f"Server returned {resp.status_code}"
    except requests.exceptions.ConnectionError:
        pytest.skip("CyberGym server not running – skipping connectivity test")
