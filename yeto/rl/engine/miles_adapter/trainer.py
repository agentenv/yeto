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
from dataclasses import dataclass
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
            self._reshard_guard(batch)
            if self._check_tokens:  # spec: reject before training
                require_policy_tokens(batch, policy_token(batch.rollout_id, batch.policy_hash))
            outputs = self._run(self._actor.train(batch.rollout_id, batch.payload))
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

    def rebind_args(self, args: Any) -> None:
        """Follow the Miles args of the rebuilt trainer (4.6/4.7: another DP size / bundle set)."""
        trainer_layout(args)  # validates world % (tp*pp*cp)
        self._args = args

    def save_cut(self, *, epoch: int, context: "CutContext") -> str:
        """Write every rank's shard, then commit the manifest; returns the cut id.

        Refuses (``CutError``) an incomplete cut: missing optimizer/RNG on any
        rank, scheduler progress that disagrees with the driver's local step,
        a non-zero ``carried_over``/ready-unconsumed ledger, an unsettled outer
        commit or a missing data cursor.
        """
        from ..cut import CutError, CutFile, CutManifest, commit_manifest, context_problems, cut_dir
        from .cut_plugin import SAVE_CUT_SHARD, config_problems

        cut_id = context.cut_id
        runtime = self._runtime(context)
        # Everything that does not need the shards is refused before any rank writes.
        problems = context_problems(
            cut_id=cut_id, progress=context.progress, algorithm=context.algorithm, data=context.data,
            ledger=context.ledger, outer=context.outer, runtime=runtime,
        ) + config_problems(self._args)
        if problems:
            raise CutError("refusing to save a cut: " + "; ".join(problems))
        directory = cut_dir(context.root, cut_id)
        summaries = [
            dict(s) for s in self._run(
                self._actor.run_plugin(SAVE_CUT_SHARD, {"directory": str(directory), "cut_id": cut_id})
            )
        ]
        expected = trainer_workers(self._args)
        if len(summaries) != expected:
            raise TrainStepError(f"expected {expected} cut shards (one per rank), got {len(summaries)}")
        # has_optimizer_state means: every adapter tensor of each (tp, pp) has
        # optimizer state on at least one of its DP ranks (DistOpt ranges).
        covered: dict[tuple[int, int], tuple[set[str], set[str]]] = {}
        for s in summaries:
            key = (s["coord"]["tp"], s["coord"]["pp"])
            adapters, optimized = covered.setdefault(key, (set(), set()))
            adapters.update(s.get("adapter_names") or ())
            optimized.update(s.get("optimizer_names") or ())
        for key, (adapters, optimized) in sorted(covered.items()):
            if not adapters or not adapters <= optimized:
                missing = sorted(adapters - optimized)[:4]
                for s in summaries:
                    if (s["coord"]["tp"], s["coord"]["pp"]) == key:
                        s["has_optimizer_state"] = False
                problems.append(f"tp{key[0]}/pp{key[1]}: no optimizer state for {missing or 'any adapter'}")
        manifest = CutManifest(
            cut_id=cut_id,
            epoch=int(epoch),
            runtime=runtime,
            progress=context.progress,
            algorithm=context.algorithm,
            data=context.data,
            ledger=context.ledger,
            outer=context.outer,
            files=tuple(
                CutFile(s["path"], s["sha256"], int(s["bytes"]), s["coord"]) for s in summaries
            ),
            rank_summaries=tuple(
                {k: s[k] for k in ("path", "scheduler_samples", "has_optimizer_state", "has_rng",
                                   "state_digest", "rng_digest")}
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
        from ..cut import CutError, cut_dir, verify_cut
        from .cut_plugin import RESTORE_CUT_SHARD

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
        saved = {s["path"]: s for s in manifest.rank_summaries}
        if len(results) != len(saved) or {r["path"] for r in results} != set(saved):
            raise CutError(f"restored shards {sorted(r['path'] for r in results)} != cut {sorted(saved)}")
        for r in results:
            s = saved[r["path"]]
            for key in ("scheduler_samples", "state_digest", "rng_digest"):
                if r[key] != s[key]:
                    raise CutError(f"{r['path']}: restored {key} differs from the cut")
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

        from ..cut import CutError, cut_dir, verify_cut
        from .cut_plugin import RANK_COORDS, RESTORE_RESHARDED_SHARD
        from .reshard import ReshardRefused, reshard_problems, rng_mapping

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


@dataclass(frozen=True)
class CutContext:
    """What the caller (driver/executor) contributes to a cut besides the trainer shards."""

    root: str
    cut_id: str
    backend_fingerprint: str
    progress: Any  # yeto.rl.engine.cut.CutProgress
    algorithm: Any  # yeto.rl.engine.cut.AlgorithmIdentity
    data: Mapping[str, Any]  # rollout data cursor
    ledger: Mapping[str, Any]
    outer: Mapping[str, Any]
    shared_filesystem: bool = True
