"""CyberGym submission contract for the experimental local RL path."""

import hashlib

import pytest

from yeto.rl.envs.cybergym_env import CyberGymEnv


def test_checksum():
    task_id = "arvo:10400"
    agent_id = "yeto_agent"
    salt = "CyberGym"
    expected = hashlib.sha256(f"{task_id}{agent_id}{salt}".encode()).hexdigest()
    assert CyberGymEnv.compute_checksum(task_id, agent_id, salt) == expected


def test_reward_semantics():
    assert CyberGymEnv.compute_reward(0) == -1.0
    assert CyberGymEnv.compute_reward(300) == -1.0
    assert CyberGymEnv.compute_reward(-1) == -1.0
    assert CyberGymEnv.compute_reward(None) == -1.0
    for code in (1, 2, 100, 137, 139, 255):
        assert CyberGymEnv.compute_reward(code) == 1.0


def test_server_error_is_not_used_as_training_reward(monkeypatch):
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
