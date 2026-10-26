"""The ``yeto_grpo`` advantage estimator: verl GRPO without placeholder rows (S25 N10.3b).

verl cannot drop one sample of a batch, so ``HarborAgentLoop`` returns an
ABORTED trajectory as a placeholder: one token with loss mask 0 and reward 0.
verl's GRPO would still count it in its prompt group's mean and std.  This
estimator moves every row without a loss token to a group of its own
(:func:`yeto.rl.agentic.groups.isolate_rows`), then calls verl's own GRPO
function unchanged.  A one-row group has mean 0 and std 1 in verl, so the
placeholder's advantage is ``0 * mask = 0`` and its siblings' baseline is
theirs alone.  verl is not modified: the estimator goes through verl's
registry (``core_algos.register_adv_est``) and is selected with
``algorithm.adv_estimator=yeto_grpo``.

"No loss token" is the test, not the ABORTED flag: verl passes the estimator
only rewards, mask and group ids.  A row whose whole response is masked
(an ABORTED placeholder, or a sequence a rollout correction rejected in full)
trains nothing, so leaving its reward out of the baseline is correct for it too.

verl's multi-trajectory GRPO path (``compute_advantage_for_multi_trajectories``)
only runs for the built-in ``grpo`` name; ``HarborAgentLoop`` returns one
output per sample, so that path is not needed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from verl.trainer.ppo import core_algos

from yeto.rl.agentic.groups import isolate_rows

from .run_config_rules import AGENTIC_ADV_ESTIMATOR as YETO_GRPO


@core_algos.register_adv_est(YETO_GRPO)
def compute_yeto_grpo_advantage(token_level_rewards: Any, response_mask: Any, index: np.ndarray,
                                epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool | None = None,
                                config: Any = None, **_: Any) -> tuple[Any, Any]:
    """verl GRPO over ``index`` with every all-masked row isolated (module doc)."""
    if norm_adv_by_std_in_grpo is None:  # the registry path passes only the config
        norm_adv_by_std_in_grpo = True if config is None else bool(config.get("norm_adv_by_std_in_grpo", True))
    empty = (response_mask.sum(dim=-1) == 0).tolist()
    groups = np.empty(len(empty), dtype=object)  # filled one by one: numpy would unpack the tuples
    for row, group in enumerate(isolate_rows(list(index), empty)):
        groups[row] = group
    return core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=response_mask, index=groups,
        epsilon=epsilon, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo, config=config)
