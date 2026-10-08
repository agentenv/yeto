"""Trainer DP change and in-pool role transfer (rl-infra-spec 4.7; E3). Import-light.

The E1 :class:`~yeto.rl.engine.controller.IslandController` owns the
transaction, the journal and the epochs; it hands ``trainer-dp`` and
``role-transfer`` edges to this module (interface in
``infra-drafts/patches/infra-e3-controller-v3.patch``):

* :func:`plan_trainer_edge` -- pure validation, called from
  ``IslandController.plan``: the edge must be certified in the attestation
  for the running ``algorithm_spec_sha256`` (A4), keep TP/PP/CP/EP, the pool
  size and the engine shape, move exactly the GPUs that change role, and
  pass the trainer's ``reshard_problems`` (:class:`~yeto.rl.engine.ports.CuttableTrainer`;
  Miles: ``miles_adapter.reshard.reshard_problems``). Nothing is written.
* :class:`TrainerTransition` -- executed at a quiescent safe point, phases of
  design D4 recorded through the controller's journal::

      QUIESCING      (rollout->trainer only) fence + drain the engines on the moved GPUs
      CUT            save_cut at the old layout; ``trainer_cut`` record (cut id, manifest digest)
      TRANSFERRING   stop the drained engines; dispose + rebuild the trainer on the
                     target bundles/DP (fork-M6) and restore the cut resharded (4.6)
      INITIALIZING   (trainer->rollout only) start engines on the freed GPUs
      VERIFYING      republish the SAME policy to new members (ACK), the trainer's
                     exported policy hash equals the cut's, serving members/layout
                     equal the target, GBS/data cursor unchanged
      COMMITTED      durable epoch CAS by the controller (single commit point)
      RESUMING -> SUCCEEDED

  Failure before the trainer is disposed: undo the engine drain/stop ->
  ``CANCELLED``/``REBUILT_OLD``. After: the rebuild helper falls back once
  to the old shape from the same cut (``REBUILD_OLD``); the stopped engines
  are restarted and republished; anything else -> ``RECOVERY_REQUIRED``.
  After COMMITTED there is no rollback.
* :func:`recovery_decision` -- controller restart: what to do with an
  open trainer transaction found in the journal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import RecoveryRequired
from .reshard_plan import ReshardPlan

TRAINER_EDGE_KINDS = frozenset({"trainer-dp", "role-transfer"})


class TrainerEdgeRejected(ValueError):
    """The trainer edge is refused before any side effect."""


class TrainerTransitionFailed(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        super().__init__(f"{phase}: {message}")
        self.phase = phase


@dataclass(frozen=True)
class TrainerEdgePlan:
    source: str
    target: str
    kind: str
    expected_config_epoch: int
    reshard: ReshardPlan
    direction: str  # trainer_to_rollout | rollout_to_trainer | standby
    moved_gpus: tuple[str, ...]
    add_engines: int
    remove_engines: int
    algorithm_spec_sha256: str
    source_trainer_gpus: tuple[str, ...] = ()
    target_trainer_gpus: tuple[str, ...] = ()
    # rollout->trainer: the engines that serve on moved_gpus (from the pool's member->GPU map)
    remove_members: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "kind": self.kind,
                "expected_config_epoch": self.expected_config_epoch, "reshard": self.reshard.to_dict(),
                "direction": self.direction, "moved_gpus": list(self.moved_gpus),
                "add_engines": self.add_engines, "remove_engines": self.remove_engines,
                "algorithm_spec_sha256": self.algorithm_spec_sha256,
                "source_trainer_gpus": list(self.source_trainer_gpus),
                "target_trainer_gpus": list(self.target_trainer_gpus),
                "remove_members": list(self.remove_members)}


def _layout(cfg: Any) -> dict[str, int]:
    d = cfg.dims
    return {"world": int(cfg.trainer), "tp": d["tp"], "pp": d["pp"], "cp": d["cp"], "ep": d["ep"],
            "dp": int(cfg.data_parallel)}


def plan_trainer_edge(
    *,
    configs: Mapping[str, Any],
    attestation: Any,
    source: str,
    target: str,
    expected_config_epoch: int,
    spec: Any,
    args: Any,
    global_batch_size: int,
    micro_batch_size: int,
    member_gpus: Mapping[str, Any] | None = None,
    trainer: Any = None,
) -> TrainerEdgePlan:
    """Validate a trainer edge; raise :class:`TrainerEdgeRejected` with every problem.

    ``member_gpus`` (serving engine member -> its GPU ids, from the pool) is
    required when engines must be removed: the removed engines are exactly
    those serving on the moved GPUs, never chosen by name order.
    ``trainer`` supplies the backend feasibility check through its optional
    ``reshard_problems`` method; without one the edge is refused.
    """
    problems: list[str] = []
    if source not in configs or target not in configs:
        raise TrainerEdgeRejected(f"unknown config in {source}->{target}")
    src, dst = configs[source], configs[target]
    edges = [k for k in getattr(attestation, "certified_edges", ()) if k[0] == source and k[1] == target]
    kinds = sorted({k[2] for k in edges} & TRAINER_EDGE_KINDS)
    if not kinds:
        raise TrainerEdgeRejected(f"edge {source}->{target} is not a certified trainer edge")
    kind = kinds[0] if len(kinds) == 1 else "role-transfer"
    sha = spec.sha256() if spec is not None and callable(getattr(spec, "sha256", None)) else None
    certified = attestation.algorithms_for((source, target, kind))
    if src.dims != dst.dims:
        problems.append("TP/PP/CP/EP change (only DP may change)")
    if src.rollout_engine_gpus != dst.rollout_engine_gpus:
        problems.append("engine shape changes")
    if src.total != dst.total:
        problems.append("pool size changes (pool changes are pool transactions)")
    if src.trainer == dst.trainer:
        problems.append("trainer size does not change")
    d_trainer = dst.trainer - src.trainer
    d_rollout = dst.rollout - src.rollout
    d_standby = dst.standby - src.standby
    if kind == "role-transfer":
        if d_standby != 0 or d_trainer != -d_rollout:
            problems.append("role transfer must move GPUs only between trainer and rollout")
        direction = "trainer_to_rollout" if d_trainer < 0 else "rollout_to_trainer"
    else:
        if d_rollout != 0 or d_trainer != -d_standby:
            problems.append("trainer-dp edge must move GPUs only between trainer and standby")
        direction = "standby"
    engine_gpus = int(src.rollout_engine_gpus) or 1
    if d_rollout % engine_gpus:
        problems.append(f"rollout change {d_rollout} is not a multiple of the engine size {engine_gpus}")
    src_t = tuple((src.placement or {}).get("trainer") or ())
    dst_t = tuple((dst.placement or {}).get("trainer") or ())
    if not src_t or not dst_t:
        problems.append("trainer edges need explicit GPU placement for both configs (M1 bundle map)")
    moved = tuple(sorted(set(src_t) ^ set(dst_t)))
    if src_t and dst_t:
        if len(moved) != abs(d_trainer):
            problems.append(f"placement moves {len(moved)} GPUs, the trainer changes by {abs(d_trainer)}")
        if not (set(src_t) <= set(dst_t) or set(dst_t) <= set(src_t)):
            problems.append("trainer GPU sets must be nested (the kept GPUs stay trainer GPUs)")
    remove_members: tuple[str, ...] = ()
    n_remove = max(0, -d_rollout) // engine_gpus
    if n_remove:
        remove_members, problem = members_on_gpus(member_gpus, moved)
        if problem:
            problems.append(problem)
        elif len(remove_members) != n_remove:
            problems.append(f"{len(remove_members)} engines serve on the moved GPUs {list(moved)}, "
                            f"the edge removes {n_remove}")
    reshard = ReshardPlan(_layout(src), _layout(dst), int(global_batch_size), int(micro_batch_size))
    check = getattr(trainer, "reshard_problems", None)
    if callable(check):
        problems += check(reshard, args=args, spec=spec, spec_sha256=sha, certified=certified)
    else:
        problems.append("the trainer does not advertise reshard_problems; DP changes are not supported")
    if problems:
        raise TrainerEdgeRejected(f"trainer edge {source}->{target} refused: " + "; ".join(problems))
    return TrainerEdgePlan(
        source=source, target=target, kind=kind, expected_config_epoch=int(expected_config_epoch),
        reshard=reshard, direction=direction, moved_gpus=moved,
        add_engines=max(0, d_rollout) // engine_gpus, remove_engines=max(0, -d_rollout) // engine_gpus,
        algorithm_spec_sha256=str(sha), source_trainer_gpus=src_t, target_trainer_gpus=dst_t,
        remove_members=remove_members,
    )


def members_on_gpus(member_gpus: Mapping[str, Any] | None, gpus: tuple[str, ...]) -> tuple[tuple[str, ...], str | None]:
    """Engines whose GPUs lie in ``gpus``; an engine straddling them and other GPUs is refused."""
    if member_gpus is None:
        return (), "the pool reports no member->GPU map; engines to remove cannot be chosen"
    wanted = set(gpus)
    chosen, covered = [], set()
    for member, owned in sorted(member_gpus.items()):
        owned = set(owned or ())
        if not owned:
            return (), f"engine {member} reports no GPU"
        if owned <= wanted:
            chosen.append(member)
            covered |= owned
        elif owned & wanted:
            return (), f"engine {member} straddles moved and kept GPUs"
    if covered != wanted:
        return (), f"moved GPUs {sorted(wanted - covered)} are not served by any engine"
    return tuple(chosen), None


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


class TrainerOps(Protocol):
    """What the transition needs from the trainer side (E2/E3 adapters)."""

    def save_cut(self, *, epoch: int, cut_id: str) -> str: ...  # quiescent cut at the current layout
    # rebuild on the target (fallback: old shape from the same cut); returns RebuildResult-like
    def resize(self, plan: TrainerEdgePlan, cut_id: str, *, epoch: int) -> Any: ...
    # rebuild the SOURCE layout on the source bundles and restore the cut exactly (no further fallback)
    def restore_source(self, plan: TrainerEdgePlan, cut_id: str, *, epoch: int) -> Any: ...
    def policy_hash(self) -> str: ...  # hash of the policy the trainer exports now
    def actual_layout(self) -> Mapping[str, int]: ...
    def data_cursor(self) -> Mapping[str, int]: ...


@dataclass
class TransitionResult:
    phase: str  # SUCCEEDED-ready (COMMIT pending) | CANCELLED | REBUILT_OLD | RECOVERY_REQUIRED
    target_members: frozenset[str] = frozenset()
    cut_id: str | None = None
    error: str | None = None
    rebuild: Any = None
    records: list[dict[str, Any]] = field(default_factory=list)


READY_TO_COMMIT = "READY_TO_COMMIT"


class TrainerTransition:
    """Runs one trainer edge up to the commit point; the controller commits (epoch CAS) or fails.

    ``record(kind, **fields)`` appends to the controller's journal;
    ``fork_call(op, members, fn)`` is the controller's fork-mirrored
    membership call (``fn(epoch)``); ``pool``/``publisher`` are the E1 ports.
    """

    def __init__(self, *, plan: TrainerEdgePlan, tx_id: str, epoch: int, trainer: TrainerOps,
                 pool: Any, publisher: Any, published_state: Any, published_version: int,
                 record: Callable[..., Any], fork_call: Callable[[str, frozenset[str], Callable[[int], Any]], Any],
                 fork_epoch: Callable[[], int], drain_deadline: float) -> None:
        self.plan = plan
        self.tx_id = tx_id
        self.epoch = int(epoch)
        self.trainer = trainer
        self.pool = pool
        self.publisher = publisher
        self.state = published_state
        self.version = published_version
        self._record = record
        self._fork_call = fork_call
        self._fork_epoch = fork_epoch
        self.drain_deadline = drain_deadline

    def _phase(self, phase: str, **fields: Any) -> None:
        self._record("phase", tx_id=self.tx_id, phase=phase, edge=self.plan.kind, **fields)

    def run(self) -> TransitionResult:
        plan = self.plan
        old_members = frozenset(self.pool.members())
        cut_hash = self.state.policy_tensor_hash()
        cursor = dict(self.trainer.data_cursor())
        removed: frozenset[str] = frozenset()
        again_gpus = None
        if plan.remove_engines:
            removed = frozenset(plan.remove_members)
            current = getattr(self.pool, "member_gpus", None)
            again_gpus = current() if callable(current) else None
            again, problem = members_on_gpus(again_gpus, plan.moved_gpus)
            if problem or frozenset(again) != removed or not removed <= old_members:
                return TransitionResult("CANCELLED", error="engines on the moved GPUs changed since planning: "
                                                           f"{problem or sorted(again)} vs {sorted(removed)}")
            self._phase("QUIESCING", remove=sorted(removed), moved_gpus=list(plan.moved_gpus))
            if not self.pool.drain(removed, self.drain_deadline):
                self.pool.undrain(removed)
                return TransitionResult("CANCELLED", error="drain did not finish before T_drain")
        # ---- CUT (non-destructive: the old trainer keeps running until resize disposes it) ----
        cut_id = f"{self.tx_id}-cut"
        try:
            self.trainer.save_cut(epoch=self.epoch, cut_id=cut_id)
        except Exception as exc:  # noqa: BLE001 - nothing destructive happened yet
            if removed:
                self.pool.undrain(removed)
            return TransitionResult("CANCELLED", error=f"save_cut failed: {exc}")
        self._record("trainer_cut", tx_id=self.tx_id, cut_id=cut_id, cut_epoch=self.epoch, policy_hash=cut_hash,
                     layout=dict(plan.reshard.source), data_cursor=cursor)
        # ---- TRANSFERRING (destructive) ----
        self._phase("TRANSFERRING", rollback_boundary="trainer_dispose", cut_id=cut_id)
        added: frozenset[str] = frozenset()
        try:
            if removed:
                self._fork_call("stop", removed, lambda e: self.pool.remove_engines(removed, epoch=e))
                # fork F-R1 order (rollout->trainer): drain -> stop_cells -> unbind_cell ->
                # trainer M6; the stopped cells release their bundles to the trainer.
                self._removed_gpus = {m: tuple(g) for m, g in (again_gpus or {}).items()
                                      if m in removed}
                unbind = getattr(self.pool, "unbind_members", None)
                if callable(unbind):
                    unbind(removed)
                    self._record("engines_unbound", tx_id=self.tx_id, members=sorted(removed))
            result = self.trainer.resize(plan, cut_id, epoch=self.epoch)
            self._record("trainer_rebuilt", tx_id=self.tx_id, outcome=result.outcome,
                         attempts=list(getattr(result, "attempts", ())))
            if result.outcome == "REBUILD_OLD":
                return self._old_engines_back(old_members, cut_id, result,
                                              "trainer rebuilt in the old shape after a failure")
            if result.outcome != "RESTORED":
                raise TrainerTransitionFailed("TRANSFERRING", f"unexpected rebuild outcome {result.outcome}")
            # ---- INITIALIZING ----
            if plan.add_engines:
                self._phase("INITIALIZING")
                added = frozenset(self.pool.plan_add(plan.add_engines))
                if len(added) != plan.add_engines or added & old_members:
                    raise TrainerTransitionFailed("INITIALIZING", f"pool offers {sorted(added)}")
                self._record("add_intent", tx_id=self.tx_id, members=sorted(added))
                bind = getattr(self.pool, "bind_members", None)
                if plan.kind == "role-transfer":
                    # The new engines run on the GPUs the trainer released: bind the stopped,
                    # startup-declared cells to them first (fork-M6 set_pg_view + rebind_cell;
                    # E1 pool verb, see evidence/infra-e3/plan.md section 5).
                    if not callable(bind):
                        raise TrainerTransitionFailed("INITIALIZING", "pool cannot bind engines to freed GPUs")
                    bind(added, plan.moved_gpus)
                self._fork_call("start", added,
                                lambda e: self.pool.add_engines(plan.add_engines, epoch=e, members=added))
            # ---- VERIFYING ----
            self._phase("VERIFYING")
            problems = []
            if self.trainer.policy_hash() != cut_hash:
                problems.append("trainer policy hash differs from the cut")
            if dict(self.trainer.actual_layout()) != dict(plan.reshard.target):
                problems.append(f"trainer layout {dict(self.trainer.actual_layout())} != target")
            if dict(self.trainer.data_cursor()) != cursor:
                problems.append("data cursor moved during the transition")
            if added:
                pub = self.publisher.publish_members(self.state, added, epoch=self._fork_epoch(),
                                                     token_rollout_id=self.version)
                if pub.manifest.target_policy_hash != cut_hash or frozenset(pub.members) != added:
                    problems.append("new engines did not acknowledge the current policy")
            target_members = (old_members - removed) | added
            if frozenset(self.pool.members()) != target_members:
                problems.append(f"serving {sorted(self.pool.members())} != {sorted(target_members)}")
            if problems:
                raise TrainerTransitionFailed("VERIFYING", "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, RecoveryRequired):
                self._record("trainer_recovery_required", tx_id=self.tx_id, error=str(exc))
                return TransitionResult("RECOVERY_REQUIRED", cut_id=cut_id, error=str(exc))
            return self._rollback(old_members, removed, added, cut_id, str(exc))
        return TransitionResult(READY_TO_COMMIT, target_members, cut_id, rebuild=result)

    # -- failure paths ------------------------------------------------------------
    def _rollback(self, old_members, removed, added, cut_id, error) -> TransitionResult:
        """After the trainer was rebuilt on the target: back to the old shape from the SAME cut."""
        self._phase("REBUILD_OLD", error=error)
        try:
            if added and frozenset(self.pool.members()) & added:
                self._fork_call("stop", added, lambda e: self.pool.remove_engines(added, epoch=e))
            back = self.trainer.restore_source(self.plan, cut_id, epoch=self.epoch)
            if back.outcome not in ("RESTORED", "REBUILD_OLD"):  # both: source shape running
                raise TrainerTransitionFailed("REBUILD_OLD", f"old trainer shape not restored: {back.outcome}")
        except Exception as exc:  # noqa: BLE001
            self._record("trainer_recovery_required", tx_id=self.tx_id, error=str(exc))
            return TransitionResult("RECOVERY_REQUIRED", cut_id=cut_id, error=f"{error}; rollback: {exc}")
        return self._old_engines_back(old_members, cut_id, back, error)

    def _old_engines_back(self, old_members, cut_id, result, error) -> TransitionResult:
        try:
            missing = old_members - frozenset(self.pool.members())
            if missing:
                # unbound by the forward path: bind them back to the GPUs they had
                previous = getattr(self, "_removed_gpus", {}) or {}
                bind = getattr(self.pool, "bind_members", None)
                if previous and callable(bind):
                    for member in sorted(missing & set(previous)):
                        bind(frozenset({member}), previous[member])
                self._fork_call("start", missing,
                                lambda e: self.pool.add_engines(len(missing), epoch=e, members=missing))
                pub = self.publisher.publish_members(self.state, missing, epoch=self._fork_epoch(),
                                                     token_rollout_id=self.version)
                if frozenset(pub.members) != missing:
                    raise TrainerTransitionFailed("REBUILD_OLD", "old engines did not acknowledge")
            still = old_members & frozenset(self.pool.members())
            if still:
                self.pool.undrain(still)
            if frozenset(self.pool.members()) != old_members:
                raise TrainerTransitionFailed("REBUILD_OLD", "old engine set not restored")
            if dict(self.trainer.actual_layout()) != dict(self.plan.reshard.source):
                raise TrainerTransitionFailed("REBUILD_OLD", "trainer is not back in the old layout")
        except Exception as exc:  # noqa: BLE001
            self._record("trainer_recovery_required", tx_id=self.tx_id, error=str(exc))
            return TransitionResult("RECOVERY_REQUIRED", cut_id=cut_id, error=f"{error}; {exc}")
        return TransitionResult("REBUILT_OLD", old_members, cut_id, error=error, rebuild=result)


# --------------------------------------------------------------------------
# Controller restart
# --------------------------------------------------------------------------


def recovery_decision(records: list[Mapping[str, Any]], committed_config: str | None,
                      committed_tx: str | None) -> dict[str, Any]:
    """What a restarted controller does with the last trainer transaction in the journal.

    * no open trainer tx -> ``none``;
    * open tx without ``trainer_cut`` -> ``resume_old`` (nothing destructive: the
      trainer was never disposed; drained engines are undrained by E1 replay);
    * ``trainer_cut`` recorded, epochs not committed by this tx -> ``restore_old``:
      rebuild the SOURCE layout and restore that cut exactly;
    * epochs committed by this tx (COMMITTED, crash before SUCCEEDED) ->
      ``restore_target``: rebuild the TARGET layout and restore the cut resharded
      (no training ran after the cut: training resumes only after SUCCEEDED);
    * anything that does not fit (cut missing after a destructive phase) ->
      ``recovery_required``.

    Boundaries: this is a HINT, not an automatic recovery. It assumes the journal
    is complete up to the crash (records are fsynced before each step) and that
    no training ran on the new layout (true: training resumes only after
    SUCCEEDED). It does not reconcile the outer protocol (D5: if accepted
    PUSH/BCAST and the cut cannot be matched uniquely, the whole run recovers
    globally), does not check that the cut files still exist, and does not
    undo a half-done fork start/stop (E1 journal replay owns that). ``cut_epoch``
    is the config epoch the cut was saved at; restore with it.
    """
    terminal = {"SUCCEEDED", "CANCELLED", "REBUILT_OLD", "RECOVERY_REQUIRED"}
    txs: dict[str, dict[str, Any]] = {}
    for r in records:
        tx = r.get("tx_id")
        if tx is None:
            continue
        info = txs.setdefault(tx, {"phases": [], "cut": None, "trainer": False, "plan": None})
        if r.get("kind") == "trainer_cut":
            info["cut"] = r
            info["trainer"] = True
        elif r.get("kind") == "phase":
            info["phases"].append(r["phase"])
            if r.get("edge") in TRAINER_EDGE_KINDS:
                info["trainer"] = True
        elif r.get("kind") == "request":
            info["plan"] = r.get("plan")
    open_txs = [(tx, i) for tx, i in txs.items() if i["trainer"] and not terminal & set(i["phases"])]
    if not open_txs:
        return {"action": "none"}
    tx, info = open_txs[-1]
    destructive = {"TRANSFERRING", "INITIALIZING", "VERIFYING", "REBUILD_OLD", "COMMITTED"}
    if info["cut"] is None:
        if destructive & set(info["phases"]):
            return {"action": "recovery_required", "tx_id": tx, "reason": "destructive phase without a cut"}
        return {"action": "resume_old", "tx_id": tx}
    cut_id = info["cut"]["cut_id"]
    cut_epoch = info["cut"].get("cut_epoch")
    if cut_epoch is None:
        return {"action": "recovery_required", "tx_id": tx, "reason": "trainer_cut record lacks cut_epoch"}
    action = "restore_target" if committed_tx == tx else "restore_old"
    return {"action": action, "tx_id": tx, "cut_id": cut_id, "cut_epoch": int(cut_epoch),
            "config": committed_config}
