"""Environment interface used by the local PPO runner."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``."""

    @abstractmethod
    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply an action using the Gymnasium step contract."""

    @abstractmethod
    def get_observation_space(self):
        """Return the observation space."""

    @abstractmethod
    def get_action_space(self):
        """Return the action space."""

    @abstractmethod
    def render(self) -> None:
        """Render the environment when supported."""
