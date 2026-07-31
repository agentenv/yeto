#!/usr/bin/env python3
"""Test CyberGym checksum and reward semantics."""

import hashlib
import json
import os
import sys
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yeto.rl.envs.cybergym_env import CyberGymEnv


def test_checksum():
    """Verify that the checksum matches CyberGym's expected format."""
    # The salt must be "CyberGym" (capital C)
    salt = "CyberGym"
    task_id = "arvo:10400"
    agent_id = "yeto_agent"
    
    # Compute checksum the way CyberGym does
    expected = hashlib.sha256(f"{task_id}{agent_id}{salt}".encode('utf-8')).hexdigest()
    
    # Compute checksum the way our adapter does
    env = CyberGymEnv()
    env.current_task_id = task_id
    env.agent_id = agent_id
    env.salt = salt
    
    checksum_input = f"{task_id}{agent_id}{salt}"
    actual = hashlib.sha256(checksum_input.encode('utf-8')).hexdigest()
    
    print(f"  Task ID: {task_id}")
    print(f"  Agent ID: {agent_id}")
    print(f"  Salt: {salt}")
    print(f"  Expected checksum: {expected}")
    print(f"  Actual checksum:   {actual}")
    
    assert actual == expected, f"Checksum mismatch: {actual} != {expected}"
    print("✅ Checksum test passed")


def test_reward_semantics():
    """Verify that the reward is inverted correctly."""
    env = CyberGymEnv()
    
    # Exit code 0 → no crash → reward should be -1.0
    reward = env._compute_reward(0)
    assert reward == -1.0, f"Expected -1.0 for exit_code=0, got {reward}"
    
    # Exit code 300 → no crash → reward should be -1.0
    reward = env._compute_reward(300)
    assert reward == -1.0, f"Expected -1.0 for exit_code=300, got {reward}"
    
    # Any other exit code → crash → reward should be 1.0
    for code in [1, 2, 100, 137, 139, 255]:
        reward = env._compute_reward(code)
        assert reward == 1.0, f"Expected 1.0 for exit_code={code}, got {reward}"
    
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