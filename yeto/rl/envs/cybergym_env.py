import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnv

class CyberGymEnv(BaseEnv):
    """Gymnasium wrapper for CyberGym environments."""
    
    def __init__(self, task_name: str = "vulnerability_analysis", **kwargs):
        self.task_name = task_name
        self.kwargs = kwargs
        self._env = None
        self._init_env()
    
    def _init_env(self):
        """Lazy initialization of the CyberGym environment."""
        try:
            # CyberGym may be installed as a package or cloned locally
            from cybergym import CyberGymEnv as CyberGymBase
            self._env = CyberGymBase(task=self.task_name, **self.kwargs)
        except ImportError:
            raise ImportError(
                "CyberGym not installed. Install with:\n"
                "  git clone https://github.com/sunblaze-ucb/cybergym.git\n"
                "  cd cybergym && pip install -e '.[dev]'"
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._env is None:
            self._init_env()
        obs, info = self._env.reset(seed=seed, options=options)
        return self._format_observation(obs), info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._env is None:
            self._init_env()
        obs, reward, done, truncated, info = self._env.step(action)
        return self._format_observation(obs), float(reward), bool(done), bool(truncated), info
    
    def _format_observation(self, obs) -> Dict[str, Any]:
        """Convert CyberGym observation to a dict with text and mask."""
        # Adjust based on CyberGym's actual observation format
        if isinstance(obs, dict):
            return obs
        return {"observation": str(obs), "action_mask": np.ones(10, dtype=np.float32)}
    
    def get_observation_space(self):
        # Define based on CyberGym's actual observation space
        return spaces.Dict({
            "observation": spaces.Text(max_length=4096),
            "action_mask": spaces.Box(0, 1, shape=(10,), dtype=np.float32)
        })
    
    def get_action_space(self):
        # Define based on CyberGym's actual action space
        return spaces.Discrete(10)
    
    def render(self) -> None:
        if self._env is not None:
            self._env.render()