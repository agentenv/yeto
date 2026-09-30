"""``TrainerGroup`` port over the upstream actor ``TrainGroup`` (task 3.3).

``train_step`` hands the opaque ``RolloutDataPack`` straight to
``actor_model.train`` (data plane never enters yeto), reads the step's grad
norm via the state plugin, and always releases the rollout object-store refs
(upstream ``remove_rollout_data_refs``). The receipt is built from handle
metadata only; ``base_policy_hash`` is the published ``policy_tensor_hash``
(the hash inside the policy token). The group must consist of exactly one cell
(D5); the adapter never touches cells itself.

``step_metrics()`` is the driver's grad_norm source for the per-round
gradient invariant. Only ``grad_norm`` is reported (read by the state plugin);
loss/lr are not surfaced by upstream ``actor_model.train`` and stay ``None``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any

from yeto.rl.contracts import LocalStepReceipt

from ..ports import RolloutBatchHandle
from . import LoopRunner
from .rollout import policy_token, require_policy_tokens
from .state_plugin import GRAD_NORM


class TrainStepError(RuntimeError):
    pass


def batch_hash(batch: RolloutBatchHandle) -> str:
    """Content identity of the trained batch, computed from metadata only."""

    payload = {
        "rollout_id": batch.rollout_id,
        "policy_version": batch.policy_version,
        "policy_hash": batch.policy_hash,
        "groups": [
            {
                "group_id": g.group_id,
                "sample_ids": list(g.sample_ids),
                "policy_token": g.policy_token,
                "reward_mean": None if math.isnan(g.reward_mean) else g.reward_mean,
                "reward_std": None if math.isnan(g.reward_std) else g.reward_std,
                "token_count": g.token_count,
            }
            for g in batch.groups
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _outcome_ok(output: Any) -> bool:
    outcome = getattr(output, "outcome", output)
    return str(getattr(outcome, "name", outcome)).lower() == "normal"


def _default_release(args: Any, data_pack: Any) -> None:
    from miles.utils.data import remove_rollout_data_refs

    remove_rollout_data_refs(args, data_pack)


class MilesTrainerGroup:
    def __init__(
        self,
        *,
        args: Any,
        actor_model: Any,
        learner_id: int,
        learner_generation: int,
        parameter_layout_hash: Callable[[], str],
        algorithm: str = "grpo",
        release_refs: Callable[[Any, Any], None] | None = None,
        check_policy_tokens: bool = True,
        runner: LoopRunner | None = None,
    ) -> None:
        self._args = args
        self._actor = actor_model
        self._learner_id = learner_id
        self._generation = learner_generation
        self._layout_hash = parameter_layout_hash
        self._algorithm = algorithm
        self._release = release_refs or _default_release
        self._check_tokens = check_policy_tokens
        self._run = (runner or LoopRunner()).run
        self.last_grad_norm: float | None = None
        self.last_outputs: list[Any] | None = None

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        if batch.payload is None:
            raise TrainStepError("rollout batch handle carries no engine payload")
        self.last_grad_norm = None
        try:
            if self._check_tokens:  # spec: reject before training
                require_policy_tokens(batch, policy_token(batch.rollout_id, batch.policy_hash))
            outputs = self._run(self._actor.train(batch.rollout_id, batch.payload))
            outputs = list(outputs or [])
            self.last_outputs = outputs
            if len(outputs) != 1:
                raise TrainStepError(
                    f"expected one train output from the single-cell group, got {len(outputs)}"
                )
            succeeded = all(_outcome_ok(o) for o in outputs)
            if succeeded:
                norms = [float(n) for n in self._run(self._actor.run_plugin(GRAD_NORM, {}))]
                if not norms or any(not math.isfinite(n) for n in norms):
                    raise TrainStepError(f"non-finite grad norm {norms}")
                self.last_grad_norm = max(norms)
        finally:
            self._release(self._args, batch.payload)
        steps = int(self._args.num_steps_per_rollout) if succeeded else 0
        return LocalStepReceipt(
            algorithm=self._algorithm,
            learner_id=self._learner_id,
            learner_generation=self._generation,
            base_policy_version=batch.policy_version,
            base_policy_hash=batch.policy_hash,
            input_batch_hash=batch_hash(batch),
            trajectory_ids=tuple(
                f"r{batch.rollout_id}:{g.group_id}:{s}" for g in batch.groups for s in g.sample_ids
            ),
            trained_tokens=sum(g.token_count for g in batch.groups),
            optimizer_steps=steps,
            optimizer_step_succeeded=succeeded,
            parameter_layout_hash=self._layout_hash(),
        )

    def step_metrics(self):
        """Telemetry of the last ``train_step`` (NaN grad_norm if it failed)."""

        from ..driver import TrainStepMetrics  # lazy: driver imports torch

        norm = self.last_grad_norm
        return TrainStepMetrics(grad_norm=math.nan if norm is None else float(norm))

    def onload(self) -> None:
        # Upstream wake_up asserts --offload-train; without it the actor stays resident.
        if getattr(self._args, "offload_train", False):
            self._run(self._actor.onload())

    def offload(self) -> None:
        if getattr(self._args, "offload_train", False):
            self._run(self._actor.offload())
        else:
            self._run(self._actor.clear_memory())
