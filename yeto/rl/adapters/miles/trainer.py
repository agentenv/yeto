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
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

from yeto.rl.contracts import LocalStepReceipt

from yeto.rl.engine.cut import CutContext  # noqa: F401 - re-exported (moved to the core, decoupling 2.2)
from yeto.rl.engine.ports import RolloutBatchHandle
from . import LoopRunner
from .rollout import policy_token, require_policy_tokens
from .state_plugin import (
    APPLIED_LRS,
    CRITIC_RECORDERS,
    CRITIC_STATE_SUMMARY,
    GRAD_NORM,
    STEP_LOSSES,
)


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


def receipt_trajectory_ids(batch: RolloutBatchHandle) -> tuple[str, ...]:
    """One id per trained trajectory, in batch order.

    CompactionRL trains one sample per segment; the segments of one rollout
    keep its sample index, so ``g.sample_ids`` repeats it (s19-compaction-g1-
    20261010g: "local-step receipt contains duplicate trajectories"). The
    segments are one trajectory: repeated ids collapse to the first. The
    sample-level ``sample_ids`` (batch hash, trained_sample_indices check,
    pooled reward) stay unchanged; trained_tokens still sums every segment.
    """

    return tuple(dict.fromkeys(
        f"r{batch.rollout_id}:{g.group_id}:{s}" for g in batch.groups for s in g.sample_ids
    ))


MASKED_FRACTION_KEYS = ("masked_fraction", "train/masked_fraction")


def masked_fraction(outputs: Any) -> float | None:
    """Masked-token fraction from the train outputs' metrics, else None.

    Upstream Miles does not report it by default; a masking mechanism's
    change wires its metric to one of ``MASKED_FRACTION_KEYS``
    (rl-algorithm-capabilities D6). Unknown (None) keeps the R0 rule.
    """

    from yeto.rl.engine.algorithm import valid_masked_fraction

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
# Policy-loss variants whose gradient rule reads the clip fraction
# (rl-algo-loss-variants D5: GMPO, global gmpo_clip_num / gmpo_clip_den).
# CISPO keeps gradients on clipped tokens and SAPO never clips: not listed.
CLIPFRAC_LOSS_VARIANTS = frozenset({"gmpo"})


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
    if spec is None or getattr(spec, "correction", None) is None:
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


# Miles loss-dict KL keys, in preference order (first present wins).
KL_METRIC_KEYS = ("mean_kl", "kl_loss", "kl", "ppo_kl", "train_rollout_kl")


def _metric(round_metrics: dict[str, float], *names: str) -> float | None:
    for name in names:
        for key in (name, f"train/{name}"):
            if key in round_metrics:
                return round_metrics[key]
    return None


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


def _default_release_outputs(outputs: Any) -> None:
    from miles.utils.data import remove_train_output_refs

    remove_train_output_refs(outputs)


# rl-algo-critic-family 3.2: critic loss-dict scalars surfaced per round
# (Miles value_loss_function: value_loss, value_clipfrac; explained_variance is
# added by the state plugin's value-metrics recorder).
CRITIC_METRIC_KEYS = ("value_loss", "value_clipfrac", "explained_variance")


def critic_round_metrics(step_losses: list[dict[str, Any]] | None,
                         grad_norm: float | None) -> dict[str, float]:
    means = mean_step_metrics(step_losses)
    out = {}
    for key in CRITIC_METRIC_KEYS:
        value = _metric(means, key)  # bare or "train/"-prefixed loss-dict key
        if value is not None:
            out[f"critic/{key}"] = value
    if grad_norm is not None:
        out["critic/grad_norm"] = float(grad_norm)
    return out


def _own_storage(value: Any) -> Any:
    """A tensor that owns exactly its own bytes, so pickling it ships only them.

    Pickle (and Ray's serializer) writes a tensor's whole storage, not the view:
    the critic channel hands over views into one flat fp32 buffer
    (``tensors_from_flat_owned``), so each of ~300 views shipped the whole
    ~2.4 GB buffer to the critic rank -- hundreds of GiB in the driver and a
    hang in ``import_critic_state`` (s19-ppo-g3-20261009a)."""

    import torch

    if not isinstance(value, torch.Tensor):
        return value
    value = value.detach()
    if value.is_contiguous() and value.storage_offset() == 0 and \
            value.untyped_storage().nbytes() == value.numel() * value.element_size():
        return value
    return value.clone(memory_format=torch.contiguous_format)


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
        critic_model: Any = None,
        release_outputs: Callable[[Any], None] | None = None,
    ) -> None:
        self._spec = spec
        # rl-algo-critic-family 3.1: shared actor/critic PPO (Miles train.py
        # order: critic.train -> critic.offload -> actor.train(external_data)).
        self._critic = critic_model
        self._release_outputs = release_outputs or _default_release_outputs
        self.last_critic_metrics: dict[str, float] = {}
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
        # fleet-dashboard 1.1: every drained step-loss record of the last round
        # (``last_step_losses`` stays gated on the mechanisms that consume it).
        self.last_round_step_losses: list[dict[str, Any]] | None = None
        self.optimizer_steps_total = 0

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        if batch.payload is None:
            raise TrainStepError("rollout batch handle carries no engine payload")
        self.last_grad_norm = None
        self.last_applied_lrs = None
        self.last_masked_fraction = None
        self.last_step_losses = None
        self.last_round_step_losses = None
        self.last_critic_metrics = {}
        critic_outputs = None
        try:
            self._reshard_guard(batch)
            if self._check_tokens:  # spec: reject before training
                require_policy_tokens(batch, policy_token(batch.rollout_id, batch.policy_hash))
            critic_outputs = self._train_critic(batch) if self._critic is not None else None
            if critic_outputs is None:
                outputs = self._run(self._actor.train(batch.rollout_id, batch.payload))
            else:
                outputs = self._run(self._actor.train(
                    batch.rollout_id, batch.payload, external_data=critic_outputs))
            outputs = list(outputs or [])
            self.last_outputs = outputs
            self.last_masked_fraction = masked_fraction(outputs)
            # Upstream returns one output per train WORKER of the single cell
            # (TrainerController.train flattens cell -> worker results); the
            # single-cell invariant itself is checked at startup.
            workers = int(getattr(self._args, "actor_num_nodes", 1) or 1) * int(
                getattr(self._args, "actor_num_gpus_per_node", 1) or 1
            )
            if len(outputs) != workers:
                raise TrainStepError(
                    f"expected {workers} train outputs (one per worker of the single-cell group), "
                    f"got {len(outputs)}"
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
                # rl-algo-loss-variants D5: GMPO needs pg_clipfrac, passed as
                # TrainStepMetrics.clip_fraction (never as masked_fraction,
                # which corrections may fill with their own mask).
                clipfrac_variant = (
                    getattr(getattr(self._spec, "loss", None), "policy_loss_variant", None)
                    in CLIPFRAC_LOSS_VARIANTS
                )
                # Drain the per-step loss records every successful round, even
                # when no mechanism consumes them: otherwise they accumulate
                # without bound and a later save_cut refuses the cut as "not
                # drained" (integ-s2 review finding 1).
                step_losses = self._step_losses()
                self.last_round_step_losses = step_losses
                if estimator in CLIPFRAC_MASKED_ESTIMATORS or corrections or clipfrac_variant:
                    self.last_step_losses = step_losses
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
            try:
                if critic_outputs:
                    self._release_outputs(critic_outputs)
            finally:
                self._release(self._args, batch.payload)
        steps = int(self._args.num_steps_per_rollout) if succeeded else 0
        self.optimizer_steps_total += steps
        return LocalStepReceipt(
            algorithm=self._algorithm,
            learner_id=self._learner_id,
            learner_generation=self._generation,
            base_policy_version=batch.policy_version,
            base_policy_hash=batch.policy_hash,
            input_batch_hash=batch_hash(batch),
            trajectory_ids=receipt_trajectory_ids(batch),
            trained_tokens=sum(g.token_count for g in batch.groups),
            optimizer_steps=steps,
            optimizer_step_succeeded=succeeded,
            parameter_layout_hash=self._layout_hash(),
        )

    def _workers(self) -> int:
        return int(getattr(self._args, "actor_num_nodes", 1) or 1) * int(
            getattr(self._args, "actor_num_gpus_per_node", 1) or 1
        )

    def _train_critic(self, batch: RolloutBatchHandle) -> list[Any]:
        """One critic round before the actor (rl-algo-critic-family 3.1/3.2).

        The critic shares the actor's GPUs (Miles placement_group.py:320) and
        is offloaded before the actor trains when ``offload_train`` (forced for
        shared PPO, arguments.py:3708-3716). Its outputs carry the values the
        actor's GAE needs (``external_data``); the caller releases them.
        """

        critic = self._critic
        self._run(critic.run_plugin(CRITIC_RECORDERS, {}))
        outputs = list(self._run(critic.train(batch.rollout_id, batch.payload)) or [])
        try:
            if len(outputs) != self._workers():
                raise TrainStepError(
                    f"expected {self._workers()} critic train outputs (one per worker), "
                    f"got {len(outputs)}"
                )
            if not all(_outcome_ok(o) for o in outputs):
                raise TrainStepError("critic train step did not finish normally")
            norms = [float(n) for n in self._run(critic.run_plugin(GRAD_NORM, {}))]
            per_rank = [list(v) for v in self._run(critic.run_plugin(STEP_LOSSES, {}))]
            step_losses = next((v for v in per_rank if v), [])
            self.last_critic_metrics = critic_round_metrics(
                step_losses, max(norms) if norms else None)
            if getattr(self._args, "offload_train", False):
                self._run(critic.offload())
        except BaseException:
            self._release_outputs(outputs)
            raise
        return outputs

    def critic_round_receipt(self, rollout_id: int):
        """rl-algo-critic-family 4.1/4.4: the critic record of the round just trained.

        Every rank reports its critic parameter specs and weight hash; the
        layout hash covers all ranks' specs (rank-tagged), the weight hash all
        ranks' hashes. The value head is Miles' critic ``output_layer``
        (model_provider.py:340-341).
        """

        if self._critic is None:
            return None
        import hashlib

        from yeto.rl.critic_state import CriticRoundReceipt, critic_layout_hash

        summaries = sorted(self._run(self._critic.run_plugin(CRITIC_STATE_SUMMARY, {})),
                           key=lambda s: s["rank"])
        specs = [(f"r{s['rank']}:{name}", shape, dtype)
                 for s in summaries for name, shape, dtype in s["specs"]]
        bins = self._value_head_bins()
        heads = [name for name, shape, _ in specs
                 if name.endswith("output_layer.weight") and shape and shape[0] == bins]
        critic = getattr(self._spec, "critic", None)
        param_mode = getattr(critic, "param_mode", None) or "full"
        weights = hashlib.sha256(
            "".join(s["weights_sha256"] for s in summaries).encode()).hexdigest()
        metrics = self.last_critic_metrics or {}
        return CriticRoundReceipt(
            rollout_id=rollout_id,
            actor_layout_hash=self._layout_hash(),
            critic_layout_hash=critic_layout_hash(
                specs, value_head=heads[0] if heads else "", param_mode=param_mode,
                value_bins=bins),
            critic_param_mode=param_mode,
            critic_init=getattr(critic, "init", None) or "copy_actor_backbone",
            critic_init_sha256=getattr(self._args, "yeto_rl_critic_init_sha256", None),
            critic_weights_sha256=weights,
            value_loss=metrics.get("critic/value_loss"),
            explained_variance=metrics.get("critic/explained_variance"),
        )

    def _value_head_bins(self) -> int:
        """Rows of the critic value head: 1 (scalar / MSE) or Miles' ``value_num_bins``
        when ``value_loss_type='classification'`` (HL-Gauss / two-hot; fork
        model_provider.py ``_value_head_output_size``)."""
        if getattr(self._args, "value_loss_type", "mse") == "classification":
            return int(getattr(self._args, "value_num_bins", 51) or 51)
        return 1

    # -- rl-algo-critic-family 4.2.2 / 4.3: critic tensors and critic cut ---------------

    def critic_layout(self) -> str:
        """critic_layout_hash over every critic rank's trainable specs (same as the receipt)."""
        receipt = self.critic_round_receipt(0)  # only the layout hash is read
        if receipt is None:
            raise TrainStepError("no critic in this trainer")
        return receipt.critic_layout_hash

    def export_critic_state(self) -> dict[str, Any]:
        """All critic ranks' full-parameter tensors (fp32 CPU), keyed ``r<rank>:<index>:<name>``."""
        from .state_plugin import EXPORT_CRITIC_TENSORS
        from yeto.rl.critic_state import critic_weights_sha256

        if self._critic is None:
            raise TrainStepError("no critic in this trainer")
        out: dict[str, Any] = {}
        for part in self._run(self._critic.run_plugin(EXPORT_CRITIC_TENSORS, {})):
            if critic_weights_sha256(part["tensors"]) != part["weights_sha256"]:
                raise TrainStepError(f"critic rank {part['rank']} export fails its hash")
            out.update({f"r{part['rank']}:{k}": v for k, v in part["tensors"].items()})
        if not out:
            raise TrainStepError("critic exported no tensors")
        return out

    def import_critic_state(self, tensors: Mapping[str, Any]) -> str:
        """Write ``export_critic_state``-shaped tensors back to every critic rank; each
        rank re-hashes what it wrote. Returns the combined content hash."""
        from .state_plugin import IMPORT_CRITIC_TENSORS
        from yeto.rl.critic_state import CriticStateError, critic_weights_sha256

        if self._critic is None:
            raise TrainStepError("no critic in this trainer")
        by_rank: dict[int, dict[str, Any]] = {}
        for key, value in tensors.items():
            prefix, _, rest = key.partition(":")
            if not prefix.startswith("r") or not rest:
                raise CriticStateError(f"critic tensor key {key!r} has no rank prefix")
            by_rank.setdefault(int(prefix[1:]), {}).setdefault("tensors", {})[rest] = _own_storage(value)
        for entry in by_rank.values():
            entry["sha256"] = critic_weights_sha256(entry["tensors"])
        results = list(self._run(self._critic.run_plugin(IMPORT_CRITIC_TENSORS, {"by_rank": by_rank})))
        refused = [r["refused"] for r in results if "refused" in r]
        if refused:
            raise CriticStateError("critic write-back refused: " + "; ".join(sorted(set(refused))))
        if sorted(r["rank"] for r in results) != sorted(by_rank):
            raise CriticStateError(f"critic ranks {sorted(r['rank'] for r in results)} != {sorted(by_rank)}")
        return critic_weights_sha256(dict(tensors))

    def _save_critic_cut(self, directory: Any, round_id: int) -> dict[str, Any] | None:
        if self._critic is None:
            return None
        from .state_plugin import SAVE_CRITIC_CUT

        ranks = sorted((dict(r) for r in self._run(self._critic.run_plugin(
            SAVE_CRITIC_CUT, {"directory": str(directory), "round_id": int(round_id)}))),
            key=lambda r: r["rank"])
        if not ranks:
            raise TrainStepError("critic saved no cut shard")
        # the pointer records the critic round next to the actor's policy_version
        return {"round": int(round_id), "directory": "critic",
                "ranks": [{k: r[k] for k in ("rank", "weights_sha256", "optimizer_sha256")} for r in ranks]}

    def _restore_critic_cut(self, directory: Any, manifest: Any) -> None:
        from yeto.rl.engine.cut import CutError

        pointer = (manifest.runtime or {}).get("critic")
        if self._critic is None:
            if pointer is not None:
                raise CutError("the cut carries critic state but this trainer has no critic")
            return
        if pointer is None:
            raise CutError("this trainer has a critic but the cut carries no critic state")
        actor_round = int(manifest.progress.policy_version)
        if int(pointer["round"]) != actor_round:
            raise CutError(f"cut actor round {actor_round} != critic round {pointer['round']}; "
                           "actor and critic must come from the same committed round")
        from .state_plugin import RESTORE_CRITIC_CUT

        results = {r["rank"]: dict(r) for r in self._run(self._critic.run_plugin(
            RESTORE_CRITIC_CUT, {"directory": str(directory), "actor_round": actor_round,
                                 "critic_round": int(pointer["round"])}))}
        refused = [r["refused"] for r in results.values() if "refused" in r]
        if refused:
            raise CutError("critic restore refused: " + "; ".join(sorted(set(refused))))
        saved = {r["rank"]: r["weights_sha256"] for r in pointer["ranks"]}
        got = {rank: r["weights_sha256"] for rank, r in results.items()}
        if got != saved:
            raise CutError(f"restored critic weights differ from the cut (ranks {sorted(saved)} vs {sorted(got)})")

    def _reshard_guard(self, batch: RolloutBatchHandle) -> None:
        """After a DP change: refuse a batch the fork would split on its unscheduled path (4.6 review M2)."""
        plan = getattr(self, "_reshard_plan", None)
        if plan is None:
            return
        from .cut_plugin import TRAIN_PARALLEL_CONFIG
        from .reshard import batch_guard_problems

        configs = [dict(c) for c in self._run(self._actor.run_plugin(TRAIN_PARALLEL_CONFIG, {}))]
        # One sample per rollout on the GRPO ports path: the fork's rollout id falls back to the sample index.
        rollout_indices = [i for i, _ in enumerate(s for g in batch.groups for s in g.sample_ids)]
        problems = batch_guard_problems(plan, rank_configs=configs, rollout_indices=rollout_indices,
                                        steps=int(getattr(self._args, "num_steps_per_rollout", 1) or 1))
        if problems:
            raise TrainStepError("batch refused after a DP change: " + "; ".join(problems))

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

    def round_metrics(self) -> dict[str, float]:
        """1.1: per-key round mean of every Miles loss-dict scalar (empty if none);
        a critic run adds ``critic/*`` (value_loss, explained_variance ...)."""
        out = mean_step_metrics(getattr(self, "last_round_step_losses", None))
        out.update(getattr(self, "last_critic_metrics", None) or {})
        return out

    def _step_losses(self) -> list[dict[str, Any]]:
        # Only the last pipeline stage records losses; take the first rank that did.
        per_rank = [list(v) for v in self._run(self._actor.run_plugin(STEP_LOSSES, {}))]
        return next((v for v in per_rank if v), [])

    def step_metrics(self):
        """Telemetry of the last ``train_step`` (NaN grad_norm if it failed)."""

        from yeto.rl.engine.driver import TrainStepMetrics  # lazy: driver imports torch

        norm = self.last_grad_norm
        means = self.round_metrics()
        lrs = getattr(self, "last_applied_lrs", None)
        return TrainStepMetrics(
            grad_norm=math.nan if norm is None else float(norm),
            loss=_metric(means, "loss"),
            pg_loss=_metric(means, "pg_loss"),
            mean_kl=_metric(means, *KL_METRIC_KEYS),
            ess_ratio=_metric(means, "ess_ratio"),
            lr=float(lrs[-1]) if lrs else None,
            train_step=getattr(self, "optimizer_steps_total", None) if norm is not None else None,
            applied_lrs=self.last_applied_lrs,
            masked_fraction=self.last_masked_fraction,
            clip_fraction=self._clip_fraction(),
        )

    def _clip_fraction(self) -> float | None:
        """Mean ``pg_clipfrac``; for GMPO the global num/den clip fraction.

        rl-algo-loss-variants D5 (fork 5c1b49eb): GMPO's gradient rule needs
        sum(gmpo_clip_num) / sum(gmpo_clip_den) over the round, not the
        per-sequence-mean ``pg_clipfrac`` (which stays in the step metrics).
        """
        step_losses = getattr(self, "last_step_losses", None)
        loss = getattr(getattr(self, "_spec", None), "loss", None)
        if getattr(loss, "policy_loss_variant", None) in CLIPFRAC_LOSS_VARIANTS:
            try:
                from yeto.rl.algos.loss_variants import gmpo_clip_fraction
            except ImportError:
                return None
            return gmpo_clip_fraction(step_losses)
        return _mean_clipfrac(step_losses)

    @property
    def publish_offloaded(self) -> bool:
        """Upstream train.py order: ``offload_train()`` (actor ``sleep``) runs *before*
        ``update_weights`` and the engines' ``onload_kv``; ``update_weights`` reads the
        host backups while asleep. The driver offloads before publishing when set."""
        return bool(getattr(self._args, "colocate", False)) and bool(
            getattr(self._args, "offload_train", False))

    def onload(self) -> None:
        # Upstream wake_up asserts --offload-train; without it the actor stays resident.
        if getattr(self._args, "offload_train", False):
            self._run(self._actor.onload())

    def offload(self) -> None:
        if getattr(self._args, "offload_train", False):
            self._run(self._actor.offload())
        else:
            self._run(self._actor.clear_memory())

    # ------------------------------------------------------------------
    # ReconfigurationCut (rl-infra-spec 4.2): explicit, off the default
    # checkpoint path. The driver calls these only at a quiescent cut.
    # ------------------------------------------------------------------

    def layout(self) -> dict[str, int]:
        return trainer_layout(self._args)

    @contextmanager
    def retained_payloads(self):
        """E2 harness (plan-v3 G-4.3): train the SAME frozen batch in two arms.

        Inside the block ``train_step`` does not release the rollout refs; the
        caller releases them once with :meth:`release_payload` afterwards.
        """
        release = self._release
        self._release = lambda _args, _payload: None
        try:
            yield
        finally:
            self._release = release

    def release_payload(self, batch: RolloutBatchHandle) -> None:
        self._release(self._args, batch.payload)

    def rebind_args(self, args: Any) -> None:
        """Follow the Miles args of the rebuilt trainer (4.6/4.7: another DP size / bundle set)."""
        layout = trainer_layout(args)  # validates world % (tp*pp*cp)
        self._args = args
        plan = getattr(self, "_reshard_plan", None)
        if plan is not None and layout != dict(plan.target):
            # back on another layout (e.g. REBUILD_OLD / restore_source): the DP-change batch guard
            # of the abandoned target no longer applies (review L-1); restore_cut_resharded sets it again.
            self._reshard_plan = None

    def save_cut(self, *, epoch: int, context: "CutContext") -> str:
        """Write every rank's shard, then commit the manifest; returns the cut id.

        Refuses (``CutError``) an incomplete cut: missing optimizer/RNG on any
        rank, scheduler progress that disagrees with the driver's local step,
        a non-zero ``carried_over``/ready-unconsumed ledger, an unsettled outer
        commit or a missing data cursor.
        """
        from yeto.rl.engine.cut import CutError, CutFile, CutManifest, commit_manifest, context_problems, cut_dir
        from .cut_plugin import SAVE_CUT_SHARD, config_problems

        cut_id = context.cut_id
        runtime = self._runtime(context)
        # Everything that does not need the shards is refused before any rank writes.
        problems = context_problems(
            cut_id=cut_id, progress=context.progress, algorithm=context.algorithm, data=context.data,
            ledger=context.ledger, outer=context.outer, runtime=runtime,
            in_flight=tuple(getattr(context, "in_flight", ()) or ()),
            max_policy_age=int((context.ledger or {}).get("max_policy_age", 0) or 0),
        ) + config_problems(self._args)
        if problems:
            raise CutError("refusing to save a cut: " + "; ".join(problems))
        directory = cut_dir(context.root, cut_id)
        summaries = [
            dict(s) for s in self._run(
                self._actor.run_plugin(SAVE_CUT_SHARD, {"directory": str(directory), "cut_id": cut_id})
            )
        ]
        refused = [f"[{s.get('refusal_kind', 'refused')}] {s['refused']}" for s in summaries if "refused" in s]
        if refused:
            raise CutError("rank refused the cut: " + "; ".join(sorted(set(refused))))
        expected = trainer_workers(self._args)
        if len(summaries) != expected:
            raise TrainStepError(f"expected {expected} cut shards (one per rank), got {len(summaries)}")
        # has_optimizer_state means: every adapter tensor of each (tp, pp) has
        # optimizer state on at least one of its DP ranks (DistOpt ranges).
        covered: dict[tuple[int, int], tuple[set[str], set[str]]] = {}
        for s in summaries:
            key = (s["coord"]["tp"], s["coord"]["pp"], s["coord"].get("ep", 0))
            adapters, optimized = covered.setdefault(key, (set(), set()))
            adapters.update(s.get("adapter_names") or ())
            optimized.update(s.get("optimizer_names") or ())
        for key, (adapters, optimized) in sorted(covered.items()):
            if not adapters or not adapters <= optimized:
                missing = sorted(adapters - optimized)[:4]
                for s in summaries:
                    if (s["coord"]["tp"], s["coord"]["pp"], s["coord"].get("ep", 0)) == key:
                        s["has_optimizer_state"] = False
                problems.append(f"tp{key[0]}/pp{key[1]}/ep{key[2]}: no optimizer state for {missing or 'any adapter'}")
        if problems:
            raise CutError("refusing an incomplete cut: " + "; ".join(problems))
        critic = self._save_critic_cut(directory / "critic", context.progress.policy_version)
        if critic is not None:  # 4.3; absent for every critic-free trainer (manifest unchanged)
            runtime = {**runtime, "critic": critic}
        manifest = CutManifest(
            cut_id=cut_id,
            epoch=int(epoch),
            runtime=runtime,
            progress=context.progress,
            algorithm=context.algorithm,
            data=context.data,
            ledger=context.ledger,
            outer=context.outer,
            in_flight=tuple(getattr(context, "in_flight", ()) or ()),
            files=tuple(
                CutFile(s["path"], s["sha256"], int(s["bytes"]), s["coord"]) for s in summaries
            ),
            rank_summaries=tuple(
                {k: s[k] for k in ("path", "scheduler_samples", "has_optimizer_state", "has_rng",
                                   "state_digest", "rng_digest", "components", "train_state_digest",
                                   "weight_version") if k in s}
                for s in summaries
            ),
        )
        if problems:
            raise CutError("refusing an incomplete cut: " + "; ".join(problems))
        commit_manifest(context.root, manifest, check_files=context.shared_filesystem)
        return cut_id

    def _runtime(self, context: "CutContext") -> dict[str, Any]:
        return {
            "backend_fingerprint": context.backend_fingerprint,
            "layout": self.layout(),
            "rng_policy": "exact",
            "precision": "fp16" if getattr(self._args, "fp16", False) else (
                "bf16" if getattr(self._args, "bf16", False) else "fp32"),
            "distributed_optimizer": bool(getattr(self._args, "use_distributed_optimizer", False)),
            "shard_schema": "yeto.cut_shard/v1",
        }

    def reshard_problems(self, plan: Any, *, args: Any = None, spec: Any = None,
                         spec_sha256: str | None = None, certified: Any = None) -> list[str]:
        """``CuttableTrainer.reshard_problems``: the Miles DP-change checks (:func:`.reshard.reshard_problems`)."""
        from .reshard import reshard_problems

        return reshard_problems(plan, args=args, spec=spec, spec_sha256=spec_sha256, certified=certified)

    def actual_layout(self) -> dict[str, int]:
        """Layout read back from the running ranks (not from args)."""
        from .cut_plugin import RANK_COORDS

        coords = [dict(c) for c in self._run(self._actor.run_plugin(RANK_COORDS, {}))]
        if not coords:
            raise TrainStepError("trainer reported no ranks")
        first = coords[0]
        return {
            "world": len(coords),
            "tp": int(first.get("tp_size", 1)),
            "pp": int(first.get("pp_size", 1)),
            "cp": int(first.get("cp_size", 1)),
            "ep": int(first.get("ep_size", 1)),
            "dp": int(first.get("dp_size", 1)),
        }

    def restore_cut(self, cut_id: str, *, epoch: int, root: str, expect: Any,
                    shared_filesystem: bool = True) -> Any:
        """Verify the cut against the running trainer, load it on every rank and check the digests.

        After loading, each rank re-exports its state; the digest of adapter +
        optimizer (FP32 main, moments, step, hyper-parameters) + scheduler +
        Megatron counters, and the RNG digest, must equal the saved ones.

        Only for a freshly built trainer (its LR scheduler must be at 0:
        Megatron ``load_state_dict`` adds the saved progress). ANY exception
        raised here leaves the trainer in an unknown state: the caller treats
        it as RECOVERY_REQUIRED and never resumes training on it.
        """
        from yeto.rl.engine.cut import CutError, cut_dir, verify_cut
        from .cut_plugin import RESTORE_CUT_SHARD

        self._reshard_plan = None  # exact same-shape restore: no DP change to guard (review L-1)

        manifest = verify_cut(root, cut_id, expect, check_files=shared_filesystem)
        if manifest.epoch > epoch:
            raise CutError(f"cut epoch {manifest.epoch} is newer than the restoring epoch {epoch}")
        actual = self.actual_layout()
        if dict(manifest.runtime["layout"]) != actual:
            raise CutError(f"cut layout {manifest.runtime['layout']} != running trainer layout {actual}")
        if not shared_filesystem and manifest.runtime.get("distributed_optimizer") and actual["dp"] > 1:
            # A DistOpt rank merges every DP shard of its (tp, pp): they must all be readable here.
            raise CutError("DistributedOptimizer with DP>1 needs a shared cut filesystem")
        files = [f.to_dict() for f in manifest.files]
        results = [
            dict(r) for r in self._run(
                self._actor.run_plugin(
                    RESTORE_CUT_SHARD,
                    {"directory": str(cut_dir(root, cut_id)), "files": files, "cut_id": cut_id},
                )
            )
        ]
        refused = [f"[{r.get('refusal_kind', 'refused')}] {r['refused']}" for r in results if "refused" in r]
        if refused:
            # every rank refused before writing, or some ranks wrote: the
            # caller treats any restore_cut error as RECOVERY_REQUIRED either way
            raise CutError("rank refused the restore: " + "; ".join(sorted(set(refused))))
        saved = {s["path"]: s for s in manifest.rank_summaries}
        if len(results) != len(saved) or {r["path"] for r in results} != set(saved):
            raise CutError(f"restored shards {sorted(r['path'] for r in results)} != cut {sorted(saved)}")
        for r in results:
            s = saved[r["path"]]
            for key in ("scheduler_samples", "state_digest", "rng_digest"):
                if r[key] != s[key]:
                    saved_c, now_c = s.get("components") or {}, r.get("components") or {}
                    differ = sorted(k for k in set(saved_c) | set(now_c) if saved_c.get(k) != now_c.get(k))
                    raise CutError(f"{r['path']}: restored {key} differs from the cut; "
                                   f"cut->reexport {r.get('optimizer_cut_vs_reexport')}; "
                                   f"cut->after_load {r.get('optimizer_cut_vs_after_load')}; "
                                   f"after_load->reexport {r.get('optimizer_after_load_vs_reexport')}; "
                                   f"rank diff {r.get('diff')}; "
                                   f"differing components ({len(differ)}): {differ[:40]}")
        self._restore_critic_cut(cut_dir(root, cut_id) / "critic", manifest)
        return manifest

    def restore_cut_resharded(self, cut_id: str, *, epoch: int, root: str, expect: Any, plan: Any,
                              certified: Any = None, shared_filesystem: bool = True) -> dict[str, Any]:
        """Load a cut written at ``plan.source`` DP into this freshly built trainer at ``plan.target`` DP (4.6).

        ``expect`` describes the run with the SOURCE layout (the cut's). The
        edge is refused before any rank writes when :func:`.reshard.reshard_problems`
        finds anything (layout other than DP changes, batch/loss normalization,
        uncertified algorithm, unsupported precision/optimizer) or the ranks
        cannot read every DP shard. After loading, every rank's re-exported
        optimizer ranges must equal the gathered cut, the gathered full-state
        digest must agree across the DP ranks of each (tp, pp), and the RNG
        source of every rank must match the recorded mapping. As with
        :meth:`restore_cut`, ANY exception means RECOVERY_REQUIRED.
        """
        from dataclasses import replace

        from yeto.rl.engine.cut import CutError, cut_dir, verify_cut
        from .cut_plugin import RANK_COORDS, RESTORE_RESHARDED_SHARD
        from .reshard import ReshardRefused, reshard_problems, rng_mapping
        if getattr(self, "_critic", None) is not None:  # rl-algo-critic-family 4.3 (critic excludes elastic)
            raise CutError("a resharded (DP-change) restore with a critic is not supported")

        problems = reshard_problems(plan, args=self._args, spec=self._spec, certified=certified)
        if not shared_filesystem:
            problems.append("a DP change needs a shared cut filesystem (every rank reads all DP shards)")
        if problems:
            raise ReshardRefused("DP edge refused: " + "; ".join(problems))
        manifest = verify_cut(root, cut_id, replace(expect, layout=dict(plan.source)), check_files=True)
        if manifest.epoch > epoch:
            raise CutError(f"cut epoch {manifest.epoch} is newer than the restoring epoch {epoch}")
        coords = [dict(c) for c in self._run(self._actor.run_plugin(RANK_COORDS, {}))]
        actual = self.actual_layout()
        if actual != dict(plan.target):
            raise CutError(f"running trainer layout {actual} != planned target {dict(plan.target)}")
        mapping = rng_mapping(plan, coords, self._args)
        files = [f.to_dict() for f in manifest.files]
        results = [
            dict(r) for r in self._run(
                self._actor.run_plugin(
                    RESTORE_RESHARDED_SHARD,
                    {"directory": str(cut_dir(root, cut_id)), "files": files, "cut_id": cut_id,
                     "source_dp": int(plan.source["dp"]), "rng_policy": plan.rng_policy},
                )
            )
        ]
        if len(results) != int(plan.target["world"]):
            raise CutError(f"{len(results)} ranks restored, target world is {plan.target['world']}")
        by_group: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for r in results:
            if r["scheduler_samples"] != manifest.progress.scheduler_samples:
                raise CutError(f"rank {r['coord']}: scheduler at {r['scheduler_samples']} samples after restore")
            by_group.setdefault((r["coord"]["tp"], r["coord"]["pp"]), []).append(r)
        for key, group in sorted(by_group.items()):
            if len({r["full_state_digest"] for r in group}) != 1:
                raise CutError(f"tp{key[0]}/pp{key[1]}: DP ranks gathered different states")
            covered = set().union(*(set(r["optimizer_names"]) for r in group))
            if not set(group[0]["adapter_names"]) <= covered:
                raise CutError(f"tp{key[0]}/pp{key[1]}: optimizer ranges do not cover every adapter")
        expected = {(m["coord"]["tp"], m["coord"]["pp"], m["coord"]["dp"]): m["source"] for m in mapping}
        for r in results:
            c = r["coord"]
            if expected.get((c["tp"], c["pp"], c["dp"])) != r["rng"]:
                raise CutError(f"rank {c}: RNG {r['rng']} does not follow the recorded mapping")
        self._reshard_plan = plan  # every later batch is guarded (batch_guard_problems)
        return {"manifest": manifest, "plan": plan.to_dict(), "rng_mapping": mapping,
                "full_state_digests": {f"tp{k[0]}_pp{k[1]}": g[0]["full_state_digest"]
                                       for k, g in sorted(by_group.items())},
                "ranks": results}


def trainer_workers(args: Any) -> int:
    return int(getattr(args, "actor_num_nodes", 1) or 1) * int(getattr(args, "actor_num_gpus_per_node", 1) or 1)


def trainer_layout(args: Any) -> dict[str, int]:
    """TP/PP/CP/EP and the derived DP of the single trainer cell."""
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    ep = int(getattr(args, "expert_model_parallel_size", 1) or 1)
    world = trainer_workers(args)
    if world % (tp * pp * cp):
        raise TrainStepError(f"world size {world} is not divisible by tp*pp*cp={tp * pp * cp}")
    return {"world": world, "tp": tp, "pp": pp, "cp": cp, "ep": ep, "dp": world // (tp * pp * cp)}
