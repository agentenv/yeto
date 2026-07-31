#!/usr/bin/env python3
"""Test CyberGym checksum and reward semantics."""

import hashlib
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yeto.rl.envs.cybergym_env import CyberGymEnv


def test_checksum():
    """Verify that the adapter's checksum matches CyberGym's expected format."""
    env = CyberGymEnv()
    task_id = "arvo:10400"
    agent_id = "yeto_agent"
    salt = "CyberGym"
    
    # The expected checksum: sha256(task_id + agent_id + salt)
    expected = hashlib.sha256(f"{task_id}{agent_id}{salt}".encode('utf-8')).hexdigest()
    
    # The adapter's computed checksum
    actual = env._compute_checksum(task_id, agent_id, salt)
    
    print(f"  Task ID: {task_id}")
    print(f"  Agent ID: {agent_id}")
    print(f"  Salt: {salt}")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")
    
    assert actual == expected, f"Checksum mismatch: {actual} != {expected}"
    print("✅ Checksum test passed")


def test_reward_semantics():
    """Verify that the adapter's reward logic is correct."""
    env = CyberGymEnv()
    
    # Exit code 0 or 300 → no crash → reward should be -1.0
    assert env._compute_reward(0) == -1.0, "Exit code 0 should give -1.0"
    assert env._compute_reward(300) == -1.0, "Exit code 300 should give -1.0"
    
    # Any other exit code → crash → reward should be 1.0
    for code in [1, 2, 100, 137, 139, 255]:
        assert env._compute_reward(code) == 1.0, f"Exit code {code} should give 1.0"
    
    print("✅ Reward semantics test passed")


def test_connectivity():
    """Test that the server is reachable."""
    import requests
    
    env = CyberGymEnv(server_host="127.0.0.1", server_port=8666)
    try:
        resp = requests.options(f"{env.server_url}/submit-vul", timeout=5)
        print(f"  Server responded with status: {resp.status_code}")
        if resp.status_code < 500:
            print("✅ Server connectivity test passed")
            return True
        else:
            print("❌ Server returned error status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not reachable. Make sure it's running:")
        print("  cd ~/cybergym")
        print("  python -m cybergym.server --host 0.0.0.0 --port 8666 \\")
        print("    --mask_map_path mask_map.json --log_dir ./server_poc \\")
        print("    --db_path ./server_poc/poc.db")
        return False


if __name__ == "__main__":
    print("Running CyberGym tests...")
    test_checksum()
    test_reward_semantics()
    test_connectivity()
    print("\nAll tests completed.")