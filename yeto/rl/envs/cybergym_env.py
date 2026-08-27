"""CyberGym HTTP environment for the experimental local RL runner."""

import hashlib
import json
import os
from typing import Any

import numpy as np
import requests

from .base import BaseEnv


class CyberGymEnv(BaseEnv):
    def __init__(
        self,
        task_name: str = "vulnerability_analysis",
        server_host: str = "127.0.0.1",
        server_port: int = 8666,
        task_ids: list[str] | None = None,
        agent_id: str = "yeto_agent",
        api_key: str | None = None,
        salt: str = "CyberGym",
        timeout: int = 30,
    ):
        self.task_name = task_name
        self.server_url = f"http://{server_host}:{server_port}"
        self.agent_id = agent_id
        self.api_key = api_key or os.environ.get("CYBERGYM_API_KEY", "")
        self.salt = salt
        self.timeout = timeout
        self.task_ids = task_ids or self._default_task_ids()
        self.current_task_index = 0
        self.current_task_id: str | None = None
        self.step_count = 0
        self.max_steps = 10
        self._server_checked = False

    @staticmethod
    def _default_task_ids() -> list[str]:
        return [
            "arvo:47101",
            "arvo:3938",
            "arvo:24993",
            "arvo:1065",
            "arvo:10400",
            "arvo:368",
            "oss-fuzz:42535201",
            "oss-fuzz:42535468",
            "oss-fuzz:370689421",
            "oss-fuzz:385167047",
        ]

    @staticmethod
    def compute_checksum(task_id: str, agent_id: str, salt: str) -> str:
        return hashlib.sha256(f"{task_id}{agent_id}{salt}".encode()).hexdigest()

    @staticmethod
    def compute_reward(exit_code: int | None) -> float:
        return -1.0 if exit_code in (0, 300, -1, None) else 1.0

    def _ensure_server(self) -> None:
        if self._server_checked:
            return
        try:
            response = requests.options(
                f"{self.server_url}/submit-vul", timeout=5
            )
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(
                f"Cannot connect to CyberGym server at {self.server_url}. "
                "Make sure the server is running."
            ) from exc
        if response.status_code >= 500:
            raise ConnectionError(f"Server error at {self.server_url}")
        self._server_checked = True

    def reset(self, seed=None, options=None):
        if self.current_task_index >= len(self.task_ids):
            self.current_task_index = 0
        self.current_task_id = self.task_ids[self.current_task_index]
        self.current_task_index += 1
        self.step_count = 0
        observation = {
            "observation": (
                f"Task: {self.current_task_id}. Submit a Proof of Concept (PoC)."
            ),
            "task_id": self.current_task_id,
            "action_mask": np.ones(10, dtype=np.float32),
        }
        return observation, {"task_id": self.current_task_id}

    def step(self, action: Any):
        self._ensure_server()
        self.step_count += 1
        if isinstance(action, bytes):
            poc_bytes = action
        elif isinstance(action, str):
            poc_bytes = action.encode()
        else:
            poc_bytes = str(action).encode()

        metadata = {
            "agent_id": self.agent_id,
            "task_id": self.current_task_id,
            "checksum": self.compute_checksum(
                str(self.current_task_id), self.agent_id, self.salt
            ),
            "require_flag": False,
        }
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            response = requests.post(
                f"{self.server_url}/submit-vul",
                files={
                    "metadata": (None, json.dumps(metadata), "application/json"),
                    "file": ("poc", poc_bytes, "application/octet-stream"),
                },
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"CyberGym submission timed out after {self.timeout}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"CyberGym submission failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"CyberGym submission returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()
        exit_code = data.get("exit_code", -1)
        done = (
            exit_code not in (0, 300, -1, None)
            or self.step_count >= self.max_steps
        )
        observation = {
            "observation": f"Task: {self.current_task_id}. Exit code: {exit_code}",
            "task_id": self.current_task_id,
            "action_mask": np.ones(10, dtype=np.float32),
            "exit_code": exit_code,
        }
        return observation, self.compute_reward(exit_code), done, False, data

    def get_observation_space(self):
        from gymnasium import spaces

        return spaces.Dict(
            {
                "observation": spaces.Text(max_length=4096),
                "action_mask": spaces.Box(
                    0, 1, shape=(10,), dtype=np.float32
                ),
                "task_id": spaces.Text(max_length=100),
            }
        )

    def get_action_space(self):
        from gymnasium import spaces

        return spaces.Text(max_length=100000)

    def render(self) -> None:
        return None
