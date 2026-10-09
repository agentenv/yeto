"""Framework-neutral rewards, group filters and their registry (decoupling group 3).

Nothing here imports a training framework or a backend adapter
(``tests/test_import_boundaries.py``).  Backend adapters wrap these functions
(Miles: ``yeto.rl.adapters.miles.rewards``).
"""

from yeto.rl.rewards.types import (
    STATUSES,
    FilterDecision,
    NeutralGroupFilter,
    NeutralReward,
    RewardResult,
    Trajectory,
)

__all__ = [
    "STATUSES",
    "FilterDecision",
    "NeutralGroupFilter",
    "NeutralReward",
    "RewardResult",
    "Trajectory",
]
