"""Miles classes for the SecRLEnv same-task retry (P0-1.7b.4-Q1).

Loaded by Miles in the rollout process (``--data-source-path`` and
``--rollout-function-path``, set by :func:`.secrlenv_retry.rollout_argv`);
the contract and its logic live in :mod:`.secrlenv_retry`.
"""

from __future__ import annotations

from typing import Any

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.rollout.inference_rollout.inference_rollout_train import abort

from .secrlenv_retry import (
    RETRY_STATS_ATTR,
    TERMINAL_ERRORS,
    SecRLEnvTaskRetry,
    raise_after_terminal_cleanup,
    retry_metrics,
)


class SecRLEnvRetryDataSource(SecRLEnvTaskRetry, RolloutDataSourceWithBuffer):
    """Miles' buffered data source with the reserved same-task retry slot."""


class SecRLEnvRolloutFn(InferenceRolloutFn):
    """Miles' rollout function; on a terminal SecRLEnv error it drains first.

    The training round also reports the retry counters as rollout metrics.
    """

    async def _call_train(self, input: Any) -> Any:
        try:
            output = await super()._call_train(input)
        except TERMINAL_ERRORS as error:
            await raise_after_terminal_cleanup(
                error,
                # Miles' end-of-round abort: stop every engine request and
                # call the agent's abort hook (tears down local episodes).
                abort=lambda: abort(self.state, set(), input.rollout_id),
                require_clean=_require_no_episode_residue,
            )
            raise  # unreachable: the cleanup always raises
        stats = getattr(self.state.args, RETRY_STATS_ATTR, None)
        metrics = getattr(output, "metrics", None)
        if isinstance(metrics, dict):
            metrics.update(retry_metrics(stats))
        return output


def _require_no_episode_residue() -> None:
    from yeto.rl.harness.codex.agent import require_no_episode_residue

    require_no_episode_residue()
