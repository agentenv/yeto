from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

class BaseEnv(ABC):
    """Abstract base class for RL environments in Yeto."""
    
    @abstractmethod
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment. Returns (observation, info)."""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step. Returns (observation, reward, done, truncated, info)."""
        pass
    
    @abstractmethod
    def get_observation_space(self):
        """Return the observation space (gymnasium.Space)."""
        pass
    
    @abstractmethod
    def get_action_space(self):
        """Return the action space (gymnasium.Space)."""
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render the environment."""
        pass