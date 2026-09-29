"""``TrainerGroup`` port over the upstream actor ``TrainGroup`` (task 3.3).

``train_step`` hands the opaque ``RolloutDataPack`` straight to
``actor_model.train`` (data plane never enters yeto), reads the step's grad
norm via the state plugin, and always releases the rollout object-store refs
(upstream ``remove_rollout_data_refs``). The receipt is built from handle
metadata only; ``base_policy_hash`` is the published ``policy_tensor_hash``
(the hash inside the policy token). The group must consist of exactly one cell
(D5); the adapter never touches cells itself.

``step_metrics()`` is the driver's grad_norm source for the per-round
gradient invariant. ``grad_norm`` and ``applied_lrs`` (the LR each optimizer
step applied, read before the scheduler advances) come from the state plugin;
loss and Miles' logged post-step lr are not surfaced by upstream
``actor_model.train`` and stay ``None``.
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
from .state_plugin import APPLIED_LRS, GRAD_NORM, STEP_LOSSES


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


MASKED_FRACTION_KEYS = ("masked_fraction", "train/masked_fraction")


def masked_fraction(outputs: Any) -> float | None:
    """Masked-token fraction from the train outputs' metrics, else None.

    Upstream Miles does not report it by default; a masking mechanism's
    change wires its metric to one of ``MASKED_FRACTION_KEYS``
    (rl-algorithm-capabilities D6). Unknown (None) keeps the R0 rule.
    """

    from ..algorithm import valid_masked_fraction

    values = []
    for output in outputs or ():
        metrics = getattr(output, "metrics", output)
        if not isinstance(metrics, dict):
            continue
        for key in MASKED_FRACTION_KEYS:
            if metrics.get(key) is not None:
                values.append(valid_masked_fraction(metrics[key]))
                break
    # Only non-bool real numbers in [0, 1]; anything else -> unknown (None).
    if not values or any(v is None for v in values):
        return None
    return min(values)  # conservative: fully masked only if every cell is


# Estimators whose masked fraction comes from the per-step clip fraction
# (rl-algo-seq-and-adv D2: GSPO clips whole sequences).
CLIPFRAC_MASKED_ESTIMATORS = frozenset({"gspo"})


def clipfrac_masked_fraction(step_losses: list[dict[str, Any]]) -> float | None:
    """GSPO ``masked_fraction`` via ``yeto.rl.algos.seq_adv.clipfrac_from_losses``.

    Optional import: without the rl-algo-seq-and-adv module the value stays
    None (unknown keeps the R0 gradient rule).
    """
    try:
        from yeto.rl.algos.seq_adv import clipfrac_from_losses
    except ImportError:
        return None
    tokens = [s.get("loss_tokens") for s in step_losses]
    return clipfrac_from_losses(step_losses, None if None in tokens else tokens)


# Miles loss-dict keys that are train/rollout mismatch diagnostics
# (rl-algo-mismatch-correction; alignment A5 labels them per round).
MISMATCH_KEY_MARKERS = ("train_rollout_kl", "tis", "mis_", "ois", "mismatch", "opsm")


def _has_corrections(spec: Any) -> bool:
    if spec is None:
        return False
    try:
        from yeto.rl.algos.mismatch_correction import selected_corrections
    except ImportError:
        return False
    return bool(selected_corrections(spec))


def mean_step_metrics(step_losses: list[dict[str, Any]] | None) -> dict[str, float]:
    """Per-key mean over the round's optimizer steps of Miles' loss-dict scalars."""
    sums: dict[str, list[float]] = {}
    for step in step_losses or ():
        for key, value in (step.get("metrics") or {}).items():
            sums.setdefault(key, []).append(float(value))
    return {k: sum(v) / len(v) for k, v in sums.items()}


def mismatch_metrics(round_metrics: dict[str, float]) -> dict[str, float]:
    return {
        k: v for k, v in sorted(round_metrics.items())
        if any(m in k.removeprefix("train/") for m in MISMATCH_KEY_MARKERS)
    }


def correction_masked_fraction(spec: Any, round_metrics: dict[str, float]) -> float | None:
    """1a ``masked_fraction_from_metrics`` (optional import; absent -> None)."""
    try:
        from yeto.rl.algos.mismatch_correction import masked_fraction_from_metrics
    except ImportError:
        return None
    return masked_fraction_from_metrics(spec, round_metrics)


def _mean_clipfrac(step_losses: list[dict[str, Any]] | None) -> float | None:
    values = [s.get("pg_clipfrac") for s in step_losses or ()]
    if not values or any(v is None for v in values):
        return None
    return sum(values) / len(values)


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
        spec: Any = None,
    ) -> None:
        self._spec = spec
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
        self.last_applied_lrs: tuple[float, ...] | None = None
        self.last_masked_fraction: float | None = None
        self.last_step_losses: list[dict[str, Any]] | None = None
        self.last_outputs: list[Any] | None = None

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        if batch.payload is None:
            raise TrainStepError("rollout batch handle carries no engine payload")
        self.last_grad_norm = None
        self.last_applied_lrs = None
        self.last_masked_fraction = None
        self.last_step_losses = None
        try:
            if self._check_tokens:  # spec: reject before training
                require_policy_tokens(batch, policy_token(batch.rollout_id, batch.policy_hash))
            outputs = self._run(self._actor.train(batch.rollout_id, batch.payload))
            outputs = list(outputs or [])
            self.last_outputs = outputs
            self.last_masked_fraction = masked_fraction(outputs)
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
                self.last_applied_lrs = self._applied_lrs()
                corrections = _has_corrections(self._spec)
                # The receipt label is the role family ("grpo"); the estimator
                # comes from the spec (review R1: gating on the label never
                # collected GSPO clip fractions).
                estimator = getattr(self._spec, "advantage_estimator", self._algorithm)
                # Clip fraction / mismatch diagnostics are collected when a
                # mechanism needs them (R0 GRPO keeps its RPC set unchanged).
                if estimator in CLIPFRAC_MASKED_ESTIMATORS or corrections:
                    self.last_step_losses = self._step_losses()
                    round_metrics = mean_step_metrics(self.last_step_losses)
                    if self.last_masked_fraction is None and corrections:
                        self.last_masked_fraction = correction_masked_fraction(
                            self._spec, round_metrics
                        )
                    if (
                        self.last_masked_fraction is None
                        and estimator in CLIPFRAC_MASKED_ESTIMATORS
                    ):
                        self.last_masked_fraction = clipfrac_masked_fraction(
                            self.last_step_losses
                        )
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

    def _applied_lrs(self) -> tuple[float, ...]:
        per_rank = [list(v) for v in self._run(self._actor.run_plugin(APPLIED_LRS, {}))]
        steps = int(self._args.num_steps_per_rollout)
        if not per_rank or any(len(v) != steps for v in per_rank):
            raise TrainStepError(
                f"expected {steps} applied learning rates per rank, got {per_rank}"
            )
        if any(v != per_rank[0] for v in per_rank[1:]):
            raise TrainStepError(f"ranks disagree on the applied learning rates {per_rank}")
        return tuple(float(x) for x in per_rank[0])

    def algorithm_metrics(self) -> dict[str, float]:
        """Mismatch diagnostics of the last round (empty when not collected)."""
        steps = getattr(self, "last_step_losses", None)
        return mismatch_metrics(mean_step_metrics(steps)) if steps else {}

    def _step_losses(self) -> list[dict[str, Any]]:
        # Only the last pipeline stage records losses; take the first rank that did.
        per_rank = [list(v) for v in self._run(self._actor.run_plugin(STEP_LOSSES, {}))]
        return next((v for v in per_rank if v), [])

    def step_metrics(self):
        """Telemetry of the last ``train_step`` (NaN grad_norm if it failed)."""

        from ..driver import TrainStepMetrics  # lazy: driver imports torch

        norm = self.last_grad_norm
        return TrainStepMetrics(
            grad_norm=math.nan if norm is None else float(norm),
            applied_lrs=self.last_applied_lrs,
            masked_fraction=self.last_masked_fraction,
            clip_fraction=_mean_clipfrac(getattr(self, "last_step_losses", None)),
        )

    def onload(self) -> None:
        # Upstream wake_up asserts --offload-train; without it the actor stays resident.
        if getattr(self._args, "offload_train", False):
            self._run(self._actor.onload())

    def offload(self) -> None:
        if getattr(self._args, "offload_train", False):
            self._run(self._actor.offload())
        else:
            self._run(self._actor.clear_memory())
