"""``IslandController``: single-transaction in-island reconfiguration (rl-infra-spec 3.2-3.7).

Design D1/D4/D6 and f-design §1.3. The controller lives inside the learner,
is owned by :class:`~yeto.rl.engine.driver.IslandDriver` and runs a
transaction only when the driver hands it a safe point (3.1). It never talks
to cells: every engine action goes through the E1 port verbs
(:class:`~yeto.rl.engine.ports.ElasticRolloutPool`,
:class:`~yeto.rl.engine.ports.MemberPublisher`,
:class:`~yeto.rl.engine.ports.ReconfigurablePlacement`).

Scope (E1): only ``rollout-only`` edges (trainer untouched; engines are added
or removed inside the fixed pool). Every other edge kind is rejected by
:meth:`IslandController.plan`, as is any edge the capability attestation
(PR #66 format, task 1.6) does not certify, an unknown config, a fingerprint
mismatch or a profile the pause audit (1.5) does not allow.

Entry points (D4): :meth:`plan` (pure), :meth:`request` (idempotent on
``request_id``; same id with another body, a stale ``expected_epoch`` or a
second concurrent transaction are refused), :meth:`status` (answers after a
lost commit acknowledgement from the journal), :meth:`cancel`. Local command
files (:class:`CommandInbox`) forward requests from SSH/Sky; there is no
network endpoint.

Transaction phases (D4): VALIDATING -> WAIT_SAFE -> QUIESCING -> TRANSFERRING
-> INITIALIZING -> VERIFYING -> COMMITTED -> RESUMING -> SUCCEEDED; failures end
in CANCELLED (nothing destructive happened), REBUILD_OLD -> REBUILT_OLD (the old
engine set was restored and serves the same policy), or RECOVERY_REQUIRED
(state uncertain or the recovery budget is spent: the driver stops consuming
data and the run fails through the existing failure path).

The journal (:mod:`.journal`) is the only authority for the config epoch and
for the membership epoch the fork must be at; the fork epoch is a mirror that
is reconciled at startup (:meth:`IslandController.open`) and never overwrites
the journal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .execution_profile import ExecutionProfile, ReadinessSnapshot, quiescent_cut_blockers
from .driver import RebuildNotStarted
from .journal import EpochState, Journal, read_epochs, read_journal
from .pause_audit import DEFAULT_MARGIN, DEFAULT_QUORUM_TIMEOUT_S, PAUSABLE_PHASE, pause_decision
from .ports import ElasticRolloutPool, MemberPublisher, PlacementDescription, ReconfigurablePlacement
try:  # E3 (4.7) module; absent until infra-e3 is integrated -> trainer edges refused
    from .trainer_transition import (
        READY_TO_COMMIT,
        TRAINER_EDGE_KINDS,
        TrainerEdgeRejected,
        TrainerTransition,
        plan_trainer_edge,
        recovery_decision,
    )
except ImportError:  # pragma: no cover - depends on the integrated tree
    READY_TO_COMMIT = "READY_TO_COMMIT"
    TRAINER_EDGE_KINDS = frozenset({"trainer-dp", "role-transfer"})
    TrainerTransition = plan_trainer_edge = None

    class TrainerEdgeRejected(ValueError):
        pass

    def recovery_decision(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"action": "none"}

# D4 phase names.
VALIDATING = "VALIDATING"
WAIT_SAFE = "WAIT_SAFE"
QUIESCING = "QUIESCING"
TRANSFERRING = "TRANSFERRING"
INITIALIZING = "INITIALIZING"
VERIFYING = "VERIFYING"
COMMITTED = "COMMITTED"
RESUMING = "RESUMING"
SUCCEEDED = "SUCCEEDED"
CANCELLED = "CANCELLED"
REBUILD_OLD = "REBUILD_OLD"
REBUILT_OLD = "REBUILT_OLD"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
STORE_MANIFEST = "STORE-MANIFEST.json"  # Q4 (C5): written last by sync_checkpoint_store
LAYOUT_KEYS = ("tp", "pp", "cp", "ep", "trainer", "nodes", "gpus_per_node", "bundle_map",
               "cross_node_tp", "cross_node_engine_tp")
# 4.4: same-shape trainer rebuild behind the ports (cut saved, trainer being
# disposed/rebuilt/restored). A restart that finds it cannot tell whether the
# trainer was restored: RECOVERY_REQUIRED (4.5).
REBUILDING_TRAINER = "REBUILDING_TRAINER"
# 3.7 restart recovery (evidence/infra-e1/recovery-design.md): pseudo phase of the
# recovery pseudo-transaction (``rec-*``) whose fork_op records reuse the
# transaction machinery; never a transaction phase.
RECOVERING = "RECOVERING"
TERMINAL = frozenset({SUCCEEDED, CANCELLED, REBUILT_OLD, RECOVERY_REQUIRED})
# Phases after which the old engine set can no longer simply be resumed.
DESTRUCTIVE = frozenset({TRANSFERRING, INITIALIZING, VERIFYING, REBUILD_OLD, REBUILDING_TRAINER})
TRAINER_REBUILD = "trainer-rebuild"
# Test-only learner kill at a phase (plan.md E1-D ⑤ COMMITTED, ⑥ QUIESCING).
KILL_AT_ENV = "YETO_RL_TEST_KILL_LEARNER_AT"
KILL_EXIT_CODE = 86

E1_EDGE_KINDS = frozenset({"rollout-only"})


class Rejected(ValueError):
    """A plan/request is refused before any side effect."""


class TransactionFailed(RuntimeError):
    def __init__(self, phase: str, message: str, *, cause: str | None = None) -> None:
        super().__init__(f"{phase}: {message}")
        self.phase = phase
        self.cause = cause  # failure class on the REBUILD_OLD record (None -> "other")


class RecoveryRequired(RuntimeError):
    """The island is in RECOVERY_REQUIRED: stop consuming data (D4)."""


class RebuildRefused(RuntimeError):
    """A trainer rebuild was refused BEFORE the trainer was touched (e.g. the
    cut could not be saved): the transaction is CANCELLED, training goes on."""


@dataclass(frozen=True)
class Timeouts:
    """``T_*`` of design D4 (seconds). Configuration values, set from measured P99."""

    safe_point: float = 600.0  # a request not reaching a usable safe point in time is cancelled
    drain: float = 120.0
    init: float = 600.0
    verify: float = 300.0
    recovery: float = 900.0  # budget of REBUILD_OLD beyond the transaction deadline
    retry_interval: float = 1.0


@dataclass(frozen=True)
class Plan:
    request_body_hash: str
    source: str
    target: str
    kind: str
    expected_config_epoch: int
    source_engines: int
    target_engines: int
    expected_pause_s: float
    pause_budget_s: float | None
    profile_hash: str | None
    trainer: Mapping[str, Any] | None = None  # E3 (4.7): TrainerEdgePlan.to_dict()

    @property
    def add(self) -> int:
        return max(0, self.target_engines - self.source_engines)

    @property
    def remove(self) -> int:
        return max(0, self.source_engines - self.target_engines)


def island_layout(cfg: Any, topology: tuple[int, int],
                  given: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalized parallel layout of a multi-node island (Q4, C5): ``tp/pp/cp/ep``
    (startup config dims), ``trainer`` GPUs, ``nodes`` x ``gpus_per_node`` and the
    role -> logical bundle ``bundle_map`` (None = leading-bundle layout). ``given``
    overrides any key (the entry passes the Megatron args / placement it launched)."""
    dims = dict(getattr(cfg, "dims", {}) or {})
    layout: dict[str, Any] = {d: int(dims.get(d, 1) or 1) for d in ("tp", "pp", "cp", "ep")}
    layout["trainer"] = int(getattr(cfg, "trainer", 0) or 0)
    layout["nodes"], layout["gpus_per_node"] = int(topology[0]), int(topology[1])
    layout["bundle_map"] = None
    layout["cross_node_tp"] = int(bool(getattr(cfg, "allow_cross_node_tp", False)))
    layout["cross_node_engine_tp"] = int(bool(getattr(cfg, "allow_cross_node_engine_tp", False)))
    for key, value in dict(given or {}).items():
        if key not in LAYOUT_KEYS:
            raise Rejected(f"unknown layout key {key!r} (known: {LAYOUT_KEYS})")
        if key == "bundle_map":
            layout[key] = (None if value is None
                           else {str(k): [int(b) for b in v] for k, v in sorted(dict(value).items())})
        else:
            layout[key] = int(value)
    return layout


def layout_diff(base: Mapping[str, Any], now: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """``{key: (baseline, current)}`` for every LAYOUT_KEYS entry that differs."""
    def norm(v: Any) -> Any:
        if isinstance(v, Mapping):
            return {str(k): [int(b) for b in vv] for k, vv in sorted(v.items())}
        return None if v is None else int(v)
    return {k: (base.get(k), now.get(k)) for k in LAYOUT_KEYS if norm(base.get(k)) != norm(now.get(k))}


def _failure_detail(exc: BaseException) -> dict[str, Any]:
    """Publication cause + disagreeing engine ids of a failed step (``other`` when the
    exception is not a structured PublicationError)."""
    cause = getattr(exc, "cause", None)
    return {"cause": getattr(cause, "value", cause) or "other",
            "inconsistent_engines": list(getattr(exc, "engine_ids", ()) or ())}


def request_body(target: str, expected_epoch: int, deadline_s: float) -> dict[str, Any]:
    return {"target": str(target), "expected_config_epoch": int(expected_epoch),
            "deadline_s": float(deadline_s)}


def rebuild_request_body(expected_epoch: int, deadline_s: float) -> dict[str, Any]:
    return {"kind": TRAINER_REBUILD, "expected_config_epoch": int(expected_epoch),
            "deadline_s": float(deadline_s)}


def body_hash(body: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(body), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class _Tx:
    tx_id: str
    request_id: str
    body: dict[str, Any]
    plan: Plan
    deadline_wall: float
    phase: str = VALIDATING
    safe_points_seen: int = 0
    added: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()
    cancel_requested: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IslandStatus:
    """f-design §1.1 (subset used by E1)."""

    health: str  # RUNNING | RECONFIGURING | RECOVERY_REQUIRED
    config_id: str | None
    config_epoch: int
    fork_membership_epoch: int
    members: tuple[str, ...]
    tx_id: str | None
    tx_phase: str | None
    tx_deadline: float | None
    admission_open: bool
    fork_incomplete: Any = None
    last_result: Mapping[str, Any] | None = None
    recovery: Mapping[str, Any] | None = None  # 3.7 restart recovery awaiting confirm
    incarnation: Mapping[str, Any] | None = None


class IslandController:
    def __init__(
        self,
        *,
        state_dir: str | Path,
        configs: Mapping[str, Any],  # name -> elastic_benchmark.capabilities.ResourceConfig
        attestation: Any,  # elastic_benchmark.capabilities.Attestation
        profile: ExecutionProfile | None,
        initial_config: str,
        runtime_fingerprint: str | None = None,
        timeouts: Timeouts = Timeouts(),
        expected_pause_s: Callable[[Plan], float] | None = None,
        quorum_timeout_s: float = DEFAULT_QUORUM_TIMEOUT_S,
        pause_margin: float = DEFAULT_MARGIN,
        idle_flow_timeout_s: float | None = None,
        budget_mode: bool = False,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        on_watchdog: Callable[[str, str], None] | None = None,
        inbox: "CommandInbox | None" = None,
        trainer_rebuilder: Callable[..., Mapping[str, Any]] | None = None,
        trainer_edges: Callable[[], Mapping[str, Any]] | None = None,
        max_recovery_attempts: int = 3,
        topology: tuple[int, int] | None = None,
        node_probe: Callable[[], Any] | None = None,
        layout: Mapping[str, Any] | None = None,
        checkpoint_store: str | Path | None = None,
    ) -> None:
        if initial_config not in configs:
            raise Rejected(f"initial config {initial_config!r} is unknown")
        self.state_dir = Path(state_dir).expanduser()
        self.configs = dict(configs)
        self.initial_config = initial_config
        self.attestation = attestation
        self.profile = profile
        self.runtime_fingerprint = runtime_fingerprint
        self.timeouts = timeouts
        self._expected_pause = expected_pause_s
        self.quorum_timeout_s = quorum_timeout_s
        self.pause_margin = pause_margin
        self.idle_flow_timeout_s = idle_flow_timeout_s
        self.budget_mode = budget_mode
        self.finalizing: int | None = None  # rollout id at which finalization began (3.8)
        self._wall = wall_clock
        self._sleep = sleep
        self._on_watchdog = on_watchdog
        self.inbox = inbox
        # 4.4: ``trainer_rebuilder(driver, *, epoch, cut_id) -> Mapping`` saves the
        # cut, rebuilds the trainer behind the ports and restores it (via
        # ``IslandDriver.rebuild_trainer``). Raises RebuildRefused before the
        # trainer is touched; any other exception is RECOVERY_REQUIRED.
        self.trainer_rebuilder = trainer_rebuilder
        self._record_lock = threading.RLock()
        import os as _os

        self._exit = _os._exit  # test seam for KILL_AT_ENV
        # Serializes "watchdog decides + kills" with "phase change" and with the
        # last check before the commit CAS (review F2): once the watchdog fired
        # the transaction cannot commit; once the commit began it cannot kill.
        self._watchdog_lock = threading.RLock()
        self._committing = False
        # E3 (4.7): None keeps trainer edges refused. Otherwise returns
        # {"spec", "args", "global_batch_size", "micro_batch_size", "ops"} where
        # ops is a trainer_transition.TrainerOps (miles_adapter.trainer_resize.MilesTrainerOps).
        self._trainer_edges = trainer_edges
        # rl-multinode-island Q4 (C5): an off-island copy of the state dir (journal,
        # epochs, cuts, ledger) kept by sync_checkpoint_store() after every commit
        # point; a learner starting on an EMPTY state dir (machine replaced) restores
        # it from there first. None = node0-local state only.
        self.checkpoint_store = (Path(checkpoint_store).expanduser()
                                 if checkpoint_store not in (None, "") else None)
        self.last_store_sync: dict[str, Any] | None = None
        restored = self._restore_from_checkpoint_store()
        self.journal = Journal(self.state_dir / "reconfig", wall_clock=wall_clock)
        if self.journal.epochs.config_id is None:
            self.journal.compare_and_swap(
                expected_config_epoch=0,
                new=EpochState(config_epoch=0, config_id=initial_config),
            )
        self._tx: _Tx | None = None
        self._by_request: dict[str, dict[str, Any]] = {}
        self._fork_epoch = self.journal.epochs.fork_membership_epoch
        self._last_op: tuple[str, tuple[str, ...]] | None = None
        self._pending_retry: tuple[str, tuple[str, ...]] | None = None
        self.admission_open = True
        self.last_outcome: dict[str, Any] | None = None
        self.recovery_required: str | None = None
        # 3.7 restart recovery (recovery-design.md §3): the membership of the
        # committed config was rebuilt at open() and awaits confirm_recovery()
        # after the driver's first full publication; admission stays closed.
        self.recovery_pending: dict[str, Any] | None = None
        if int(max_recovery_attempts) < 1:
            raise Rejected("max_recovery_attempts must be >= 1")
        self.max_recovery_attempts = int(max_recovery_attempts)
        # rl-multinode-island D9: (nodes, gpus_per_node) of the island and a probe
        # returning the alive node ids (or {node_id: gpus} mapping). None = single node.
        self.topology: tuple[int, int] | None = None
        self._topology_checked = False
        self._node_probe: Callable[[], Any] | None = None
        self.set_topology(topology, node_probe, layout=layout)
        self._recovery_unverified = 0  # consecutive planned recoveries never verified
        self._recovery_open_id: str | None = None  # planned, not verified/failed/superseded
        import uuid as _uuid

        self.incarnation = {"pid": _os.getpid(), "id": _uuid.uuid4().hex[:12],
                            "wall_time": self._wall()}
        self._watchdog_fired = threading.Event()
        self._replay()
        if restored is not None:
            self._record("checkpoint_store", tx_id=None, action="restore", store=str(self.checkpoint_store),
                         incarnation=self.incarnation["id"], **restored)

    # ------------------------------------------------------------------ journal
    def close(self) -> None:
        self.journal.close()

    def _replay(self) -> None:
        open_txs: dict[str, dict[str, Any]] = {}
        for r in self.journal.records:
            kind = r["kind"]
            if kind == "request":
                self._by_request[r["request_id"]] = {"tx_id": r["tx_id"], "body": r["body"],
                                                     "body_hash": r["body_hash"], "plan": r["plan"],
                                                     "deadline_wall": r["deadline_wall"]}
                open_txs[r["tx_id"]] = r
            elif kind == "fork_op":
                if r["status"] == "done":
                    self._fork_epoch = int(r["result_fork_epoch"])
                    self._last_op = (r["op"], tuple(r["cells"]))
                    self._pending_retry = None
                elif r["status"] == "incomplete":
                    self._pending_retry = (r["retry_op"], tuple(r["retry_cells"]))
            elif kind == "phase" and r["phase"] in TERMINAL:
                open_txs.pop(r["tx_id"], None)
                if r["phase"] == RECOVERY_REQUIRED:
                    self.recovery_required = r.get("error") or "recovery required"
            elif kind == "recovery":
                status = r.get("status")
                if status == "planned":
                    self._recovery_unverified += 1
                    self._recovery_open_id = r["tx_id"]
                elif status == "verified":
                    self._recovery_unverified = 0
                    self._recovery_open_id = None
                elif status in ("failed", "superseded"):
                    self._recovery_open_id = None
        self._fork_epoch = max(self._fork_epoch, self.journal.epochs.fork_membership_epoch)
        self._open_after_restart = list(open_txs)

    def _record(self, kind: str, **fields: Any) -> dict[str, Any]:
        # the watchdog thread journals too: serialize appends within the process
        with self._record_lock:
            return self.journal.append(kind, **fields)

    def _maybe_test_kill(self, phase: str) -> None:
        """TEST ONLY (plan.md E1-D ⑤⑥): ``YETO_RL_TEST_KILL_LEARNER_AT=<PHASE>``
        hard-kills the learner (``os._exit``) right after a transaction journals PHASE, once
        per state dir (marker file), so the restarted learner goes on."""
        import os

        wanted = os.environ.get(KILL_AT_ENV)
        if not wanted or wanted != phase:
            return
        marker = self.state_dir / f"test-killed-at-{phase}"
        if marker.exists():
            return
        marker.write_text("killed\n", encoding="utf-8")
        import sys

        print(f"[yeto] TEST INJECTION {KILL_AT_ENV}={phase}: killing the learner",
              file=sys.stderr, flush=True)
        self._exit(KILL_EXIT_CODE)

    def watchdog_target_cells(self) -> list[str]:
        """The target generation a fired watchdog may kill (3.7/H2): cells this
        transaction is starting/verifying. Never old members, never a drain."""
        tx = self._tx
        if tx is None or tx.phase not in (INITIALIZING, VERIFYING):
            return []
        if tx.added:
            return sorted(tx.added)
        # E3 trainer edges journal their new engines as add_intent
        intent = next((r for r in reversed(self.journal.records)
                       if r["kind"] == "add_intent" and r.get("tx_id") == tx.tx_id), None)
        return sorted(intent["members"]) if intent else []

    def record_test_event(self, driver: Any, event: str, **fields: Any) -> None:
        """TEST ONLY: a structured record of a test injection (``test_injection``) or
        hold (``test_hold``) on BOTH the journal and the driver's tape, tagged with the
        running transaction (id, phase) and a ``ts``. The journal's own record kind is
        ``event``, so the injection's ``kind`` is stored as ``injection_kind`` there."""
        tx = self._tx
        full = {"ts": time.time(), "tx_id": None if tx is None else tx.tx_id,
                "phase": None if tx is None else tx.phase, **fields}
        journal_fields = dict(full)
        if "kind" in journal_fields:
            journal_fields["injection_kind"] = journal_fields.pop("kind")
        self._record(event, **journal_fields)
        emit = getattr(driver, "emit", None)
        if callable(emit):
            emit(event, **full)

    def _block_state(self, tx_id: str) -> str:
        """``not_reached`` / ``blocked`` / ``released`` of this transaction's injected
        update_weights block, from its ``test_injection`` records."""
        last = None
        for r in self.journal.records:
            if (r["kind"] == "test_injection" and r.get("injection_kind") == "block_update"
                    and r.get("tx_id") == tx_id):
                last = r
        if last is None or not last.get("applied"):
            return "not_reached"
        return "released" if last.get("released_ts") is not None else "blocked"

    def _watchdog_classification(self, tx: _Tx, phase: str) -> dict[str, Any]:
        """Which situation a fired watchdog was in (the deadline itself is unchanged):
        ``INJECTION_NOT_REACHED`` (still INITIALIZING, or the block was never reached),
        ``FIRED_ON_BLOCKED_UPDATE`` (stuck at the injected block) or
        ``FIRED_AFTER_BLOCK_RELEASED`` (the block ended, the real update was running)."""
        state = self._block_state(tx.tx_id)
        reached = state != "not_reached"
        if phase == INITIALIZING or not reached:
            cls = "INJECTION_NOT_REACHED"
        elif state == "blocked":
            cls = "FIRED_ON_BLOCKED_UPDATE"
        else:
            cls = "FIRED_AFTER_BLOCK_RELEASED"
        return {"injection_reached": reached, "classification": cls}

    def record_watchdog_action(self, tx_id: str, **fields: Any) -> None:
        self._record("watchdog_action", tx_id=tx_id, **fields)

    def watchdog_unresolved(self, tx_id: str, phase: str, errors: Any) -> None:
        """The watchdog action could not kill (all of) the target generation. The step
        stays blocked on engines nobody manages, so the old-set rebuild cannot be
        trusted: enter RECOVERY_REQUIRED at once (admission closed, the learner stops
        consuming); the blocked step's own ``_rebuild_old`` then re-raises it."""
        tx = self._tx
        if tx is None or tx.tx_id != tx_id or self.recovery_required:
            return
        self._enter_recovery(tx_id, cause="watchdog_unresolved",
                             error=f"watchdog could not kill the target generation in {phase}: "
                                    f"{list(errors)[:2]!r}")

    # ------------------------------------------------------------- node topology (D9)
    def set_topology(self, topology: tuple[int, int] | None,
                     node_probe: Callable[[], Any] | None = None, *,
                     layout: Mapping[str, Any] | None = None) -> None:
        if topology is not None:
            nodes, per = int(topology[0]), int(topology[1])
            if nodes < 1 or per < 1:
                raise Rejected(f"topology {topology!r} must be (nodes >= 1, gpus_per_node >= 1)")
            topology = (nodes, per)
        if topology != getattr(self, "topology", None):
            self._topology_checked = False  # refuse_partial_island() ran for this topology
        self.topology = topology
        self._node_probe = node_probe
        # Q4 (C5): the island's parallel layout; defaults come from the startup config
        self.layout: dict[str, Any] | None = None
        if topology is not None:
            self.layout = island_layout(self.configs[self.initial_config], topology, layout)

    # ------------------------------------------------------ parallel layout (Q4, C5)
    def layout_baseline(self) -> dict[str, Any] | None:
        """The ``layout`` of the last ``topology`` record written by an EARLIER
        incarnation (what the journal's cuts/epochs were produced with), or None."""
        mine = self.incarnation["id"]
        for r in reversed(self.journal.records):
            if (r.get("kind") == "topology" and r.get("layout") and r.get("incarnation") != mine
                    and r.get("layout_accepted", True)):  # a refused incarnation sets no baseline
                return dict(r["layout"])
        return None

    def layout_rejection(self) -> str | None:
        """Why this incarnation's layout (tp/pp/cp/ep, trainer shape, bundle map) differs
        from the journal baseline (None when equal, or without topology/baseline). A
        rebuild from a consistent checkpoint must be same-shape: a Megatron cut written
        under another parallel layout is not restorable without conversion."""
        if self.layout is None:
            return None
        base = self.layout_baseline()
        if base is None:
            return None
        diff = layout_diff(base, self.layout)
        if not diff:
            return None
        return "layout changed: " + ", ".join(f"{k} {b!r} -> {n!r}" for k, (b, n) in diff.items())

    def _alive_nodes(self) -> tuple[list[Any], dict[Any, int] | None]:
        """``(alive node ids, {node: gpus} or None)`` from the probe (raises on failure)."""
        alive = self._node_probe() if self._node_probe is not None else None
        if alive is None:
            return [], None
        if isinstance(alive, Mapping):
            return list(alive), {k: int(v) for k, v in alive.items()}
        return list(alive), None

    def topology_rejection(self) -> str | None:
        """Why the island's alive nodes do not match its declared topology (None when
        they do, or when there is no topology/probe): fewer alive nodes than declared,
        or a node exposing another GPU count (same-shape rule, hostnames may change)."""
        if self.topology is None or self._node_probe is None:
            return None
        nodes, per = self.topology
        try:
            alive, gpus = self._alive_nodes()
        except Exception as exc:  # noqa: BLE001 - unknown node state: do not act
            return f"node probe failed: {exc!r}"
        if len(alive) < nodes:
            return f"{len(alive)} of {nodes} island nodes alive ({sorted(map(str, alive))})"
        if gpus is not None:
            bad = {str(n): g for n, g in gpus.items() if g != per}
            if bad:
                return f"nodes {bad} do not expose {per} GPUs (topology {nodes}x{per})"
        return None

    def refuse_partial_island(self) -> str | None:
        """Startup precondition (D9, tasks 3.3): one ``topology`` journal record (declared
        nodes/gpus_per_node, alive nodes, incarnation); on a partial island enter
        RECOVERY_REQUIRED and return why, else None. Runs once per incarnation: the entry
        preflight calls it right after the Ray connect (before any placement group is
        created, which would hang on a dead node) and :meth:`open` then skips it. No-op
        (None) without a topology."""
        if self.topology is None:
            return None
        self._topology_checked = True
        try:
            alive: Any = [str(n) for n in self._alive_nodes()[0]]
        except Exception as exc:  # noqa: BLE001
            alive = f"probe failed: {exc!r}"
        layout_why = self.layout_rejection()  # against the baseline BEFORE this record
        self._record("topology", tx_id=None, nodes=self.topology[0], gpus_per_node=self.topology[1],
                     alive=alive, incarnation=self.incarnation["id"], layout=self.layout,
                     layout_accepted=layout_why is None, layout_error=layout_why)
        why = self.topology_rejection()
        if why is not None:
            # D9: never touch the fork's membership epoch on a partial island; the
            # restart precondition fails before any restore/start/stop call.
            self._enter_recovery(None, f"island topology: {why}; recovery refused on fewer nodes")
            return why
        if layout_why is not None:
            # Q4 (C5): fail closed for this incarnation (admission closed, the entry
            # exits) but, like gpu_pool, NOT a journal terminal: the operator relaunches
            # with the same cfg/recipe (same shape) and the island resumes; no flag
            # overrides this, a rebuild must be same-shape.
            self.recovery_required = (f"{layout_why}; a rebuild from this journal must keep "
                                      "the parallel layout of the committed checkpoint")
            self.admission_open = False
            return layout_why
        self.sync_checkpoint_store("topology")
        return None

    def check_nodes(self) -> str | None:
        """Driver poll (D9): any node loss -> ``node_lost`` journal record and the
        RECOVERY_REQUIRED terminal state (admission closed). Returns the error, or
        None while every declared node is alive. Idempotent once entered."""
        if self.recovery_required:
            return self.recovery_required
        why = self.topology_rejection()
        if why is None:
            return None
        alive: Any = None
        try:
            alive = [str(n) for n in self._alive_nodes()[0]]
        except Exception:  # noqa: BLE001
            pass
        self._record("node_lost", tx_id=None, topology=list(self.topology or ()), alive=alive, error=why)
        self._enter_recovery(None, f"node_lost: {why}; the island does not run on fewer nodes")
        return self.recovery_required

    # ----------------------------------------------- GPU pool binding (Q6, 2026-10-04)
    def gpu_pool_baseline(self) -> tuple[tuple[str, ...], ...] | None:
        """Per-node GPU uuids bound by the last accepted ``gpu_pool`` journal record
        (an earlier incarnation), or None when this island never journaled a pool."""
        for r in reversed(self.journal.records):
            if r.get("kind") == "gpu_pool" and r.get("accepted") and r.get("uuids"):
                return tuple(tuple(str(u) for u in node) for node in r["uuids"])
        return None

    def record_gpu_pool(self, result: Any, *, source: str, accept_rebind: bool,
                        roles: Mapping[str, str] | None = None) -> str | None:
        """Startup precondition (Q6): journal one ``gpu_pool`` record for this incarnation
        (observed per-node uuids, what they were compared against, rebind + mapping,
        diffs, accepted) and, when the reconciliation refused the pool, close this
        incarnation (``recovery_required = "gpu_pool: ..."``, no journal phase record so a
        restart with ``--rl-elastic-accept-rebind`` may bind the new pool) and return
        why; None when accepted.
        ``source`` names the declared side (``cfg`` / ``journal`` / ``cfg+journal`` /
        ``none``)."""
        self._record("gpu_pool", tx_id=None, incarnation=self.incarnation["id"],
                     uuids=[list(node) for node in result.observed], source=source,
                     accept_rebind=bool(accept_rebind), rebind=bool(result.rebind),
                     mapping=dict(result.mapping), diffs=list(result.diffs),
                     roles=dict(roles or {}),  # 2026-10-04 v2: uuid -> role (one role per uuid)
                     accepted=bool(result.ok), error=result.error)
        if result.ok:
            self.sync_checkpoint_store("gpu_pool")
            return None
        # Fail closed for this incarnation (admission closed, the entry exits), but NOT
        # a journal terminal: unlike node loss, a changed pool is bound by an explicit
        # restart with --rl-elastic-accept-rebind (ruling 2026-10-04), which the replay
        # of a RECOVERY_REQUIRED phase record would forbid forever.
        self.recovery_required = (f"gpu_pool: {result.error}; this incarnation refuses to "
                                  "start on an unreconciled GPU pool")
        self.admission_open = False
        return result.error

    # ------------------------------------------------ checkpoint store (Q4, C5)
    def _restore_from_checkpoint_store(self) -> dict[str, Any] | None:
        """Before the journal opens: with a store holding a complete copy
        (``STORE-MANIFEST.json`` written last) and NO journal in the local state dir
        (fresh machine), copy the store into the state dir. A local journal always
        wins (an in-place restart is at least as new as its last sync; dropping
        local records could hide a RECOVERY_REQUIRED terminal)."""
        store = self.checkpoint_store
        if store is None:
            return None
        local_journal = self.state_dir / "reconfig" / "journal.jsonl"
        if local_journal.exists():
            return None
        manifest = store / STORE_MANIFEST
        if not manifest.is_file():
            return None
        info = json.loads(manifest.read_text(encoding="utf-8"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for child in store.iterdir():
            if child.name == STORE_MANIFEST:
                continue
            target = self.state_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
        return {"restored_from": {k: info.get(k) for k in ("incarnation", "reason", "config_epoch",
                                                              "wall_time", "seq")}}

    def sync_checkpoint_store(self, reason: str) -> bool:
        """Copy the state dir (journal, epochs, cuts, ledger, inbox...) to the store and
        write ``STORE-MANIFEST.json`` last (a reader trusts the copy only with it). Runs
        under the record lock so no journal line is torn by a concurrent append. A
        failing sync is logged + remembered (``last_store_sync``), never fatal: the
        store is a durability aid, the island's truth stays the local journal."""
        store = self.checkpoint_store
        if store is None:
            return False
        with self._record_lock:
            info = {"incarnation": self.incarnation["id"], "reason": reason, "wall_time": self._wall(),
                    "config_epoch": self.journal.epochs.config_epoch,
                    "seq": int(self.journal.records[-1].get("seq", 0)) if self.journal.records else 0,
                    "layout": self.layout, "state_dir": str(self.state_dir)}
            try:
                store.mkdir(parents=True, exist_ok=True)
                for child in self.state_dir.iterdir():
                    if child.name == STORE_MANIFEST:
                        continue
                    target = store / child.name
                    if child.is_dir():
                        shutil.copytree(child, target, dirs_exist_ok=True,
                                        ignore=shutil.ignore_patterns("journal.lock"))
                    elif child.name != "journal.lock":
                        shutil.copy2(child, target)
                tmp = store / (STORE_MANIFEST + ".tmp")
                tmp.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
                tmp.replace(store / STORE_MANIFEST)
            except Exception as exc:  # noqa: BLE001 - never fail the commit on the copy
                self.last_store_sync = {**info, "ok": False, "error": repr(exc)}
                logging.getLogger(__name__).warning(
                    "checkpoint store sync (%s) to %s failed: %r", reason, store, exc)
                return False
            self.last_store_sync = {**info, "ok": True}
            return True

    def set_on_watchdog(self, handler: Callable[[str, str], None] | None) -> None:
        self._on_watchdog = handler

    def set_trainer_edges(self, provider: Callable[[], Mapping[str, Any]] | None) -> None:
        """E3 (4.7): wired by compose_island once the trainer/pool exist."""
        self._trainer_edges = provider

    def _phase(self, tx: _Tx, phase: str, **fields: Any) -> None:
        with self._watchdog_lock:  # the watchdog reads tx.phase under this lock
            tx.phase = phase
        self._record("phase", tx_id=tx.tx_id, request_id=tx.request_id, phase=phase,
                     config_epoch=self.journal.epochs.config_epoch,
                     fork_epoch=self._fork_epoch, **fields)
        if phase in (COMMITTED, SUCCEEDED):
            self.sync_checkpoint_store(phase)  # Q4 (C5): after the durable CAS / cut commit
        self._maybe_test_kill(phase)

    # ------------------------------------------------------------------ startup
    def open(self, pool: ElasticRolloutPool) -> IslandStatus:
        """Reconcile the fork's membership mirror with the journal (f-design §1.3 rule 3)."""
        self._pool = pool
        if self.topology is not None:
            if not self._topology_checked:  # not already done by the entry preflight
                self.refuse_partial_island()
            if self.recovery_required:
                return self.inspect()
        status = dict(pool.membership_status())
        fork_epoch = int(status.get("epoch", 0))
        expected = self._fork_epoch
        last_issued = next((r for r in reversed(self.journal.records) if r["kind"] == "fork_op"), None)
        if fork_epoch == expected:
            pass
        elif fork_epoch == 0 and expected > 0:
            incomplete = list(self._pending_retry) if self._pending_retry else None
            if incomplete:
                incomplete = [incomplete[0], list(incomplete[1])]
            last_op = [self._last_op[0], list(self._last_op[1])] if self._last_op else None
            try:
                pool.restore_membership(epoch=expected, incomplete=incomplete, last_op=last_op,
                                        expected_current_epoch=0)
            except Exception as exc:  # noqa: BLE001 - CAS lost: never overwrite the journal
                self._enter_recovery(None, f"restoring fork membership epoch failed: {exc}")
            else:
                self._record("reconcile", action="restore_membership_state", epoch=expected)
        elif (
            fork_epoch == expected + 1
            and last_issued is not None
            and last_issued["status"] == "issued"
            and int(last_issued["expected_fork_epoch"]) == expected
        ):
            # the op was applied but its answer was lost: complete the journal record
            self._record("fork_op", tx_id=last_issued["tx_id"], op=last_issued["op"],
                         cells=last_issued["cells"], expected_fork_epoch=expected,
                         result_fork_epoch=fork_epoch, status="done", recovered=True)
            self._fork_epoch = fork_epoch
            self._last_op = (last_issued["op"], tuple(last_issued["cells"]))
        else:
            self._enter_recovery(None, f"fork membership epoch {fork_epoch} disagrees with the "
                                       f"journal ({expected})")
        epochs = self.journal.epochs
        rebuild_after_restart: list[tuple[str, str]] = []
        for tx_id in self._open_after_restart:
            request = next(r for r in self.journal.records
                           if r["kind"] == "request" and r["tx_id"] == tx_id)
            phases = [r["phase"] for r in self.journal.iter_tx(tx_id) if r["kind"] == "phase"]
            if epochs.last_tx_id == tx_id:
                # the durable CAS happened: the transaction committed (3.7 "commit后恢复")
                self._record("phase", tx_id=tx_id, request_id=request["request_id"],
                             phase=SUCCEEDED, config_epoch=epochs.config_epoch,
                             fork_epoch=self._fork_epoch, recovered_after_restart=True)
            elif not DESTRUCTIVE.intersection(phases):
                self._record("phase", tx_id=tx_id, request_id=request["request_id"],
                             phase=CANCELLED, config_epoch=epochs.config_epoch,
                             fork_epoch=self._fork_epoch, error="learner restarted before release")
            elif (request["body"].get("kind", "rollout-only") == "rollout-only"
                  and REBUILDING_TRAINER not in phases):
                # 3.7 "release 后重建旧 rollout" (recovery-design.md §3.2, D1): the
                # committed members are the old set; the recovery below rebuilds
                # them and the transaction ends REBUILT_OLD.
                rebuild_after_restart.append((tx_id, request["request_id"]))
            else:
                # E3: the trainer was rebuilt from startup args; the hint names the cut
                # a manual recovery would restore (trainer_transition.recovery_decision).
                hint = recovery_decision(list(self.journal.iter_tx(tx_id)), epochs.config_id,
                                         epochs.last_tx_id)
                if hint.get("action") not in (None, "none"):
                    self._record("trainer_recovery_hint", tx_id=tx_id, **hint)
                self._enter_recovery(tx_id, "learner restarted after release, before commit")
        self._open_after_restart = []
        if self.recovery_required is None:
            members = frozenset(pool.members())
            committed = frozenset(epochs.members)
            if committed and members != committed:
                self._recover_membership(pool, committed, members, rebuild_after_restart)
            elif committed:
                self._finish_rebuilt_after_restart(rebuild_after_restart)
            elif not epochs.members:
                self.journal.compare_and_swap(
                    expected_config_epoch=epochs.config_epoch,
                    new=EpochState(epochs.config_epoch, epochs.config_id, self._fork_epoch,
                                   tuple(sorted(members)), epochs.last_tx_id),
                )
        return self.inspect()

    # ------------------------------------------------------------------ restart recovery (3.7)
    def _finish_rebuilt_after_restart(self, txs: list[tuple[str, str]]) -> None:
        epochs = self.journal.epochs
        for tx_id, request_id in txs:
            self._record("phase", tx_id=tx_id, request_id=request_id, phase=REBUILT_OLD,
                         config_epoch=epochs.config_epoch, fork_epoch=self._fork_epoch,
                         recovered_after_restart=True,
                         error="learner restarted after release, before commit: the committed "
                               "(old) members were rebuilt")

    def _recovery_precondition(self, pool: ElasticRolloutPool, committed: frozenset[str],
                               actual: frozenset[str]) -> str | None:
        """Why the committed config cannot be recovered by rollout cell start/stop
        (recovery-design.md §2): None when it can."""
        epochs = self.journal.epochs
        topology_why = self.topology_rejection()  # D9: never recover onto fewer nodes
        if topology_why is not None:
            return f"island topology: {topology_why}"
        layout_why = self.layout_rejection()  # Q4: never recover onto another layout
        if layout_why is not None:
            return layout_why
        cfg = self.configs.get(epochs.config_id)
        if cfg is None:
            return f"committed config {epochs.config_id!r} is unknown"
        initial = self.configs[self.initial_config]
        if cfg.trainer != initial.trainer or cfg.dims != initial.dims:
            return (f"committed config {cfg.name!r} has another trainer shape than the startup "
                    f"config {initial.name!r}: the trainer was rebuilt from startup args")
        if cfg.rollout_engine_gpus != initial.rollout_engine_gpus:
            return f"committed config {cfg.name!r} changes the engine shape"
        states = getattr(pool, "member_states", None)
        if callable(states):
            try:
                described = dict(states() or {})
            except Exception as exc:  # noqa: BLE001 - unknown cell state: do not act
                return f"member states unavailable: {exc!r}"
            unknown = sorted(committed - set(described))
            if unknown:
                return f"committed members {unknown} are not cells the fork declares"
            unbound = sorted(m for m in committed - actual
                             if (described[m] or {}).get("state") == "unbound")
            if unbound:
                return f"committed members {unbound} are bound to no bundle (role transfer binding lost)"
        return None

    def _recover_membership(self, pool: ElasticRolloutPool, committed: frozenset[str],
                            actual: frozenset[str], rebuild_after_restart: list[tuple[str, str]]) -> None:
        """Rebuild the committed rollout membership after a restart (recovery-design.md §3).

        Difference-based (target = epochs.json members, never a replay of past
        transactions), so a repeated run is idempotent. Fork calls go through
        ``_fork_call`` under a ``rec-*`` pseudo transaction, so a crash in the
        middle is replayed by ``_replay``/``open`` like any transaction's. Ends
        with ``recovery_pending`` set and admission closed: the driver's first
        full publication then covers the rebuilt members and
        :meth:`confirm_recovery` verifies before admission reopens. Any
        failure, timeout or spent budget is RECOVERY_REQUIRED (design (a))."""
        epochs = self.journal.epochs
        if self._recovery_open_id is not None:
            self._record("recovery", tx_id=self._recovery_open_id, status="superseded",
                         by=self.incarnation["id"])
            self._recovery_open_id = None
        if self._recovery_unverified >= self.max_recovery_attempts:
            self._enter_recovery(None, f"recovery budget spent: {self._recovery_unverified} "
                                       f"consecutive recoveries were never verified "
                                       f"(max {self.max_recovery_attempts})")
            return
        why = self._recovery_precondition(pool, committed, actual)
        if why is not None:
            self._enter_recovery(None, f"serving members {sorted(actual)} differ from the committed "
                                       f"members {sorted(committed)} and cannot be recovered: {why}")
            return
        rid = f"rec-{epochs.config_epoch}-{self._fork_epoch}-{self.incarnation['id']}"
        deadline = self._wall() + self.timeouts.recovery
        tx = _Tx(rid, "", {}, None, deadline, phase=RECOVERING)  # type: ignore[arg-type]
        self._recovery_unverified += 1
        self._recovery_open_id = rid
        self._record("recovery", tx_id=rid, status="planned", attempt=self._recovery_unverified,
                     incarnation=dict(self.incarnation), config_id=epochs.config_id,
                     config_epoch=epochs.config_epoch, fork_epoch=self._fork_epoch,
                     target=sorted(committed), actual=sorted(actual),
                     stop=sorted(actual - committed), start=sorted(committed - actual),
                     deadline_wall=deadline)
        self.admission_open = False
        self._watchdog_fired.clear()
        try:
            passes = 0
            while True:
                self._check_deadline(tx, "recovery")
                # a half-failed op (fork ``incomplete``) is retried first, like in a transaction
                self._retry_incomplete(tx, recovery=False)
                now = frozenset(pool.members())
                extra, missing = now - committed, committed - now
                if not extra and not missing:
                    break
                passes += 1
                if passes > 3:
                    raise TransactionFailed(RECOVERING, f"members {sorted(now)} did not converge to "
                                                        f"{sorted(committed)} after {passes - 1} passes")
                try:
                    if extra:
                        self._fork_call(tx, "stop", extra,
                                        lambda e, m=extra: pool.remove_engines(m, epoch=e))
                    if missing:
                        self._fork_call(tx, "start", missing,
                                        lambda e, m=missing: pool.add_engines(len(m), epoch=e, members=m))
                except TransactionFailed:
                    raise
                except Exception:  # noqa: BLE001
                    if self.recovery_required or self._pending_retry is None:
                        raise  # a plain failure (e.g. no resources): not retried
                    # half failed: the fork accepts only the retry of that op (next pass)
            self._check_deadline(tx, "recovery mirror")
            self._sync_fork_mirror()
        except Exception as exc:  # noqa: BLE001 - every failure is RECOVERY_REQUIRED
            self._record("recovery", tx_id=rid, status="failed", error=str(exc),
                         fork_epoch=self._fork_epoch)
            self._recovery_open_id = None
            if self.recovery_required is None:
                self._enter_recovery(rid, f"membership recovery failed: {exc}")
            return
        self._record("recovery", tx_id=rid, status="membership_restored",
                     fork_epoch=self._fork_epoch, members=sorted(committed))
        self._finish_rebuilt_after_restart(rebuild_after_restart)
        self.recovery_pending = {"recovery_id": rid, "members": committed, "deadline_wall": deadline,
                                 "config_id": epochs.config_id, "config_epoch": epochs.config_epoch}

    def confirm_recovery(self, driver: Any) -> None:
        """Release after a restart recovery (recovery-design.md §4): called by the
        driver right after its first full publication. Verifies the rollout cells
        (members, fork epoch, every serving engine's policy token, router
        admission) and the trainer side (layout read back from the ranks, no
        unconsumed batches) before admission reopens; any failed check is
        RECOVERY_REQUIRED and raises :class:`RecoveryRequired`."""
        pending = self.recovery_pending
        if pending is None:
            return
        if self.recovery_required:
            raise RecoveryRequired(self.recovery_required)
        rid = pending["recovery_id"]
        target: frozenset[str] = pending["members"]
        checks: dict[str, Any] = {}
        failures: list[str] = []
        pool = driver.rollout
        if self._wall() > pending["deadline_wall"]:
            failures.append("recovery ran past T_recovery before verification")
        if self.topology is not None:
            topology_why = self.topology_rejection()
            checks["nodes"] = "ok" if topology_why is None else topology_why
            if topology_why is not None:
                failures.append(f"island topology: {topology_why}")
            layout_why = self.layout_rejection()
            checks["layout"] = "ok" if layout_why is None else layout_why
            if layout_why is not None:
                failures.append(layout_why)
        members = frozenset(pool.members())
        checks["members"] = sorted(members)
        if members != target:
            failures.append(f"serving members {sorted(members)} differ from the committed "
                            f"members {sorted(target)}")
        try:
            status = dict(pool.membership_status())
        except Exception as exc:  # noqa: BLE001
            status = {"epoch": None, "error": repr(exc)}
        checks["fork_epoch"] = status.get("epoch")
        if status.get("epoch") != self._fork_epoch or status.get("incomplete"):
            failures.append(f"fork membership status {status} disagrees with the journal epoch "
                            f"{self._fork_epoch}")
        state = getattr(driver, "published_state", None)
        version = getattr(driver, "published_version", None)
        checks["published_version"] = version
        if state is None or version is None:
            failures.append("no published policy")
        else:
            verify = getattr(driver.publisher, "verify_serving_policy", None)
            if callable(verify):
                try:
                    verify(epoch=self._fork_epoch, token_rollout_id=version, state=state)
                    checks["policy_token"] = "verified"
                except Exception as exc:  # noqa: BLE001
                    checks["policy_token"] = "failed"
                    failures.append(f"serving engines do not all report the published policy: {exc}")
            else:
                checks["policy_token"] = "unavailable"
        states = getattr(pool, "member_states", None)
        if callable(states):
            try:
                described = dict(states() or {})
                not_admitted = sorted(m for m in target if not (described.get(m) or {}).get("serving")
                                      or (described.get(m) or {}).get("awaiting_admission"))
                checks["router"] = {"not_admitted": not_admitted}
                if not_admitted:
                    failures.append(f"members {not_admitted} are not serving/admitted at the router")
            except Exception as exc:  # noqa: BLE001
                checks["router"] = "failed"
                failures.append(f"member states unavailable: {exc!r}")
        else:
            checks["router"] = "unavailable"
        cfg = self.configs.get(self.journal.epochs.config_id)
        layout_fn = getattr(getattr(driver, "trainer", None), "actual_layout", None)
        if cfg is not None and callable(layout_fn):
            try:
                layout = dict(layout_fn())
            except Exception as exc:  # noqa: BLE001
                checks["trainer_layout"] = "failed"
                failures.append(f"trainer layout unreadable: {exc!r}")
            else:
                checks["trainer_layout"] = layout
                dims = cfg.dims
                bad = {d: (layout.get(d), dims[d]) for d in ("tp", "pp", "cp", "ep")
                       if d in layout and int(layout[d]) != int(dims[d])}
                if "world" in layout and int(layout["world"]) != int(cfg.trainer):
                    bad["world"] = (layout["world"], cfg.trainer)
                if bad:
                    failures.append(f"trainer layout {bad} (actual, committed) differs")
        else:
            checks["trainer_layout"] = "unavailable"
        ledger = getattr(driver, "ledger", None)
        if ledger is not None:
            unconsumed = list(ledger.unconsumed())
            checks["unconsumed_batches"] = unconsumed
            if unconsumed:
                failures.append(f"unconsumed batches {unconsumed} at the restart point")
        if failures:
            self._record("recovery", tx_id=rid, status="failed", checks=checks, errors=failures)
            self._recovery_open_id = None
            self.recovery_pending = None
            self._enter_recovery(rid, "recovery verification failed: " + "; ".join(failures))
            raise RecoveryRequired(self.recovery_required)
        self._record("recovery", tx_id=rid, status="verified", checks=checks,
                     fork_epoch=self._fork_epoch, members=sorted(target))
        self._recovery_unverified = 0
        self._recovery_open_id = None
        self.recovery_pending = None
        self.admission_open = True

    def _enter_recovery(self, tx_id: str | None, error: str, *, cause: str = "other") -> None:
        """Island-level RECOVERY_REQUIRED (``scope=island``, ``request_id=None``), plus a
        request-level terminal record for the transaction's request when ``tx_id`` is a
        request transaction that has no terminal record yet (``scope=request``,
        ``request_id=<id>``, ``cause``, ``island_record_seq`` -> the island record).

        Revision (2026-10-02 user ruling, E1-D ④ chain 6r2): the island record alone left the
        request (dn1) without a terminal record of its own; both are written now, with the
        same error/epochs, so the request's status, the journal replay and the GPU judge
        agree on one terminal state. ``cause`` names the failure class
        (``stop_retry_deadline``, ``rebuild_old_failed``, ``watchdog_unresolved``, ...)."""
        self.recovery_required = error
        self.admission_open = False
        epochs = {"config_epoch": self.journal.epochs.config_epoch, "fork_epoch": self._fork_epoch}
        island = self._record("phase", tx_id=tx_id, request_id=None, phase=RECOVERY_REQUIRED,
                              scope="island", cause=cause, error=error, **epochs)
        request_id = self._request_of(tx_id)
        if request_id is None or self._has_terminal(tx_id):
            return
        tx = self._tx if self._tx is not None and self._tx.tx_id == tx_id else None
        deadline_wall = (tx.deadline_wall if tx is not None
                         else self._by_request.get(request_id, {}).get("deadline_wall"))
        if tx is not None:
            with self._watchdog_lock:
                tx.phase = RECOVERY_REQUIRED
        self.last_outcome = {"tx_id": tx_id, "phase": RECOVERY_REQUIRED, "error": error, "cause": cause}
        self._record("phase", tx_id=tx_id, request_id=request_id, phase=RECOVERY_REQUIRED,
                     scope="request", cause=cause, error=error,
                     island_record_seq=island.get("seq"), deadline_wall=deadline_wall,
                     recovery_deadline_wall=(None if deadline_wall is None
                                             else deadline_wall + self.timeouts.recovery),
                     **epochs)

    def _request_of(self, tx_id: str | None) -> str | None:
        if tx_id is None:
            return None
        if self._tx is not None and self._tx.tx_id == tx_id:
            return self._tx.request_id
        return next((rid for rid, r in self._by_request.items() if r["tx_id"] == tx_id), None)

    def _has_terminal(self, tx_id: str) -> bool:
        return any(r["kind"] == "phase" and r.get("tx_id") == tx_id and r.get("request_id") is not None
                   and r["phase"] in TERMINAL for r in self.journal.records)

    # ------------------------------------------------------------------ queries
    def inspect(self) -> IslandStatus:
        epochs = self.journal.epochs
        tx = self._tx
        last = next((r for r in reversed(self.journal.records)
                     if r["kind"] == "phase" and r["phase"] in TERMINAL), None)
        health = ("RECOVERY_REQUIRED" if self.recovery_required
                  else "RECOVERING" if self.recovery_pending is not None
                  else "RECONFIGURING" if tx is not None else "RUNNING")
        return IslandStatus(
            health=health, config_id=epochs.config_id, config_epoch=epochs.config_epoch,
            fork_membership_epoch=self._fork_epoch, members=epochs.members,
            tx_id=tx.tx_id if tx else None, tx_phase=tx.phase if tx else None,
            tx_deadline=tx.deadline_wall if tx else None, admission_open=self.admission_open,
            fork_incomplete=list(self._pending_retry) if self._pending_retry else None,
            last_result=last,
            recovery=None if self.recovery_pending is None else {
                "recovery_id": self.recovery_pending["recovery_id"],
                "members": sorted(self.recovery_pending["members"])},
            incarnation=dict(self.incarnation),
        )

    def status(self, request_id: str) -> dict[str, Any]:
        return request_status(self.journal.records, request_id)

    # ------------------------------------------------------------------ plan
    def plan(self, target: str, expected_epoch: int, *, deadline_s: float = 0.0) -> Plan:
        body = request_body(target, expected_epoch, deadline_s)
        epochs = self.journal.epochs
        if self.recovery_required:
            raise Rejected(f"island is RECOVERY_REQUIRED: {self.recovery_required}")
        if self.recovery_pending is not None:
            raise Rejected(f"island is recovering ({self.recovery_pending['recovery_id']}): "
                           "requests are refused until the recovery is verified")
        if expected_epoch != epochs.config_epoch:
            raise Rejected(f"expected config epoch {expected_epoch}, current is {epochs.config_epoch}")
        source = epochs.config_id
        if target not in self.configs:
            raise Rejected(f"unknown target config {target!r}")
        if target == source:
            raise Rejected(f"target {target!r} is the current config")
        att = self.attestation
        if att is None or att.runtime_fingerprint is None:
            raise Rejected("no capability attestation: no transition is certified")
        if self.runtime_fingerprint is not None and att.runtime_fingerprint != self.runtime_fingerprint:
            raise Rejected("attestation fingerprint differs from the running engine")
        edges = [k for k in att.certified_edges if k[0] == source and k[1] == target]
        if not edges:
            raise Rejected(f"edge {source}->{target} is not certified")
        kinds = {k[2] for k in edges}
        if kinds & TRAINER_EDGE_KINDS and not kinds & E1_EDGE_KINDS:
            return self._plan_trainer(source, target, expected_epoch, deadline_s)
        if not kinds & E1_EDGE_KINDS:
            raise Rejected(f"edge {source}->{target} kinds {sorted(kinds)} are not E1 "
                           "(rollout-only) edges")
        src, dst = self.configs[source], self.configs[target]
        if src.trainer != dst.trainer or src.dims != dst.dims:
            raise Rejected("rollout-only edge changes the trainer")
        if src.rollout_engine_gpus != dst.rollout_engine_gpus:
            raise Rejected("rollout-only edge changes the engine shape")
        from .multinode import engine_replica_delta_rejection

        why = engine_replica_delta_rejection(src.rollout, dst.rollout, src.rollout_engine_gpus)
        if why is not None:  # ruling 2026-10-04 v2: scale by whole engine replicas only
            raise Rejected(f"rollout-only edge {source}->{target}: {why}")
        if src.total != dst.total:
            raise Rejected("edge changes the pool size (pool changes are pool transactions)")
        if self.profile is None or self.profile.execution_mode == "colocated-serial":
            raise Rejected("rollout reconfiguration needs a partitioned profile")
        source_engines = src.rollout // src.rollout_engine_gpus
        target_engines = dst.rollout // dst.rollout_engine_gpus
        if target_engines < 1:
            raise Rejected("target has no rollout engine")
        plan = Plan(
            request_body_hash=body_hash(body), source=source, target=target, kind="rollout-only",
            expected_config_epoch=expected_epoch, source_engines=source_engines,
            target_engines=target_engines, expected_pause_s=0.0, pause_budget_s=None,
            profile_hash=self.profile.contract_hash,
        )
        pause_s = float(self._expected_pause(plan)) if self._expected_pause else float(deadline_s)
        decision = self._pause_decision(PAUSABLE_PHASE, pause_s)
        if not decision.allowed:
            raise Rejected(f"pause not allowed: {decision.reason}")
        from dataclasses import replace

        return replace(plan, expected_pause_s=pause_s, pause_budget_s=decision.budget_s)

    def _pause_decision(self, outer_phase: str, pause_s: float) -> Any:
        return pause_decision(
            self.profile, outer_phase=outer_phase, expected_pause_s=pause_s,
            budget_mode=self.budget_mode, quorum_timeout_s=self.quorum_timeout_s,
            margin=self.pause_margin, idle_flow_timeout_s=self.idle_flow_timeout_s,
        )

    def enter_finalization(self, *, rollout_id: int) -> list[str]:
        """3.8/X6: the run is finalizing (stop boundary reached). Cancel the
        pending request (journaled) and reject every later one. Returns the
        cancelled request ids."""
        cancelled = []
        if self.finalizing is None:
            self.finalizing = int(rollout_id)
            self._record("finalization", rollout_id=int(rollout_id))
        tx = self._tx
        if tx is not None and tx.phase in (VALIDATING, WAIT_SAFE):
            self._finish(tx, CANCELLED, error="finalization refuses reconfiguration",
                         finalization_rollout_id=int(rollout_id))
            cancelled.append(tx.request_id)
        return cancelled
    def _trainer_plan(self, source: str, target: str, expected_epoch: int):
        if self._trainer_edges is None or plan_trainer_edge is None:
            raise Rejected(f"edge {source}->{target} is not E1 (rollout-only) and trainer edges "
                           "are not enabled (E3)")
        ctx = self._trainer_edges()
        pool = getattr(self, "_pool", None)
        probe = getattr(pool, "member_gpus", None)
        try:
            member_gpus = probe() if callable(probe) else None
        except Exception:  # noqa: BLE001 - unknown map: the edge refuses it (plan stays side-effect free)
            member_gpus = None
        try:
            return plan_trainer_edge(
                member_gpus=member_gpus,
                configs=self.configs, attestation=self.attestation, source=source, target=target,
                expected_config_epoch=expected_epoch, spec=ctx["spec"], args=ctx["args"],
                global_batch_size=ctx["global_batch_size"], micro_batch_size=ctx["micro_batch_size"])
        except TrainerEdgeRejected as exc:
            raise Rejected(str(exc)) from exc

    def _plan_trainer(self, source: str, target: str, expected_epoch: int, deadline_s: float) -> Plan:
        """E3 (4.7): trainer-dp / role-transfer edges; validation in trainer_transition (no side effect)."""
        if self.profile is None or self.profile.execution_mode == "colocated-serial":
            raise Rejected("trainer reconfiguration needs a partitioned profile")
        tplan = self._trainer_plan(source, target, expected_epoch)
        src, dst = self.configs[source], self.configs[target]
        body = request_body(target, expected_epoch, deadline_s)
        plan = Plan(
            request_body_hash=body_hash(body), source=source, target=target, kind=tplan.kind,
            expected_config_epoch=expected_epoch,
            source_engines=src.rollout // src.rollout_engine_gpus,
            target_engines=dst.rollout // dst.rollout_engine_gpus,
            expected_pause_s=0.0, pause_budget_s=None, profile_hash=self.profile.contract_hash,
            trainer=tplan.to_dict(),
        )
        pause_s = float(self._expected_pause(plan)) if self._expected_pause else float(deadline_s)
        decision = self._pause_decision(PAUSABLE_PHASE, pause_s)
        if not decision.allowed:
            raise Rejected(f"pause not allowed: {decision.reason}")
        from dataclasses import replace

        return replace(plan, expected_pause_s=pause_s, pause_budget_s=decision.budget_s)

    # ------------------------------------------------------------------ request / cancel
    def request(self, request_id: str, target: str, expected_epoch: int,
                deadline_s: float) -> dict[str, Any]:
        if not request_id or not isinstance(request_id, str):
            raise Rejected("request_id must be a non-empty string")
        body = request_body(target, expected_epoch, deadline_s)
        digest = body_hash(body)
        known = self._by_request.get(request_id)
        if known is not None:
            if known["body_hash"] != digest:
                raise Rejected(f"request {request_id!r} was already used with another body")
            return self.status(request_id)  # idempotent: same answer, no second transaction
        if self.finalizing is not None:
            raise Rejected("finalization refuses reconfiguration "
                           f"(finalizing since rollout {self.finalizing})")
        if self._tx is not None:
            raise Rejected(f"transaction {self._tx.tx_id} is in progress (one per island)")
        if not deadline_s or deadline_s <= 0:
            raise Rejected("deadline_s must be positive")
        plan = self.plan(target, expected_epoch, deadline_s=deadline_s)
        tx_id = f"tx-{self.journal.epochs.config_epoch}-{digest[:12]}-{request_id}"
        deadline_wall = self._wall() + float(deadline_s)
        self._record("request", request_id=request_id, tx_id=tx_id, body=body, body_hash=digest,
                     plan=asdict(plan), deadline_wall=deadline_wall)
        self._by_request[request_id] = {"tx_id": tx_id, "body": body, "body_hash": digest,
                                        "plan": asdict(plan), "deadline_wall": deadline_wall}
        self._tx = _Tx(tx_id, request_id, body, plan, deadline_wall)
        self._phase(self._tx, VALIDATING, plan=asdict(plan))
        return self.status(request_id)

    def rebuild_plan(self, expected_epoch: int, *, deadline_s: float) -> Plan:
        """4.4: a same-shape trainer rebuild at the current config (no edge, no
        membership change); refused like any pause the audit does not allow."""
        body = rebuild_request_body(expected_epoch, deadline_s)
        epochs = self.journal.epochs
        if self.recovery_required:
            raise Rejected(f"island is RECOVERY_REQUIRED: {self.recovery_required}")
        if self.recovery_pending is not None:
            raise Rejected("island is recovering: requests are refused until the recovery is verified")
        if self.trainer_rebuilder is None:
            raise Rejected("no trainer rebuilder is wired on this island")
        if expected_epoch != epochs.config_epoch:
            raise Rejected(f"expected config epoch {expected_epoch}, current is {epochs.config_epoch}")
        if self.profile is None or self.profile.execution_mode == "colocated-serial":
            raise Rejected("trainer rebuild needs a partitioned profile")
        current = self.configs[epochs.config_id]
        engines = current.rollout // current.rollout_engine_gpus
        pause_s = float(deadline_s)
        decision = self._pause_decision(PAUSABLE_PHASE, pause_s)
        if not decision.allowed:
            raise Rejected(f"pause not allowed: {decision.reason}")
        return Plan(
            request_body_hash=body_hash(body), source=epochs.config_id, target=epochs.config_id,
            kind=TRAINER_REBUILD, expected_config_epoch=expected_epoch, source_engines=engines,
            target_engines=engines, expected_pause_s=pause_s, pause_budget_s=decision.budget_s,
            profile_hash=self.profile.contract_hash,
        )

    def request_trainer_rebuild(self, request_id: str, expected_epoch: int,
                                deadline_s: float) -> dict[str, Any]:
        """Journal a trainer-rebuild request (idempotent by request id + body)."""
        if not request_id or not isinstance(request_id, str):
            raise Rejected("request_id must be a non-empty string")
        body = rebuild_request_body(expected_epoch, deadline_s)
        digest = body_hash(body)
        known = self._by_request.get(request_id)
        if known is not None:
            if known["body_hash"] != digest:
                raise Rejected(f"request {request_id!r} was already used with another body")
            return self.status(request_id)
        if self.finalizing is not None:
            raise Rejected("finalization refuses reconfiguration "
                           f"(finalizing since rollout {self.finalizing})")
        if self._tx is not None:
            raise Rejected(f"transaction {self._tx.tx_id} is in progress (one per island)")
        if not deadline_s or deadline_s <= 0:
            raise Rejected("deadline_s must be positive")
        plan = self.rebuild_plan(expected_epoch, deadline_s=deadline_s)
        tx_id = f"tx-{self.journal.epochs.config_epoch}-{digest[:12]}-{request_id}"
        deadline_wall = self._wall() + float(deadline_s)
        self._record("request", request_id=request_id, tx_id=tx_id, body=body, body_hash=digest,
                     plan=asdict(plan), deadline_wall=deadline_wall)
        self._by_request[request_id] = {"tx_id": tx_id, "body": body, "body_hash": digest,
                                        "plan": asdict(plan), "deadline_wall": deadline_wall}
        self._tx = _Tx(tx_id, request_id, body, plan, deadline_wall)
        self._phase(self._tx, VALIDATING, plan=asdict(plan))
        return self.status(request_id)

    def _execute_trainer_rebuild(self, tx: _Tx, driver: Any, snapshot: ReadinessSnapshot) -> str:
        epochs = self.journal.epochs
        cut_id = f"rb-{epochs.config_epoch}-{tx.plan.request_body_hash[:12]}"
        unconsumed = [] if driver.ledger is None else driver.ledger.unconsumed()
        if driver.published_state is None or unconsumed:
            self._finish(tx, CANCELLED, error="no published policy or unconsumed batches at the cut")
            return CANCELLED
        self.admission_open = False
        self._phase(tx, REBUILDING_TRAINER, cut_id=cut_id, rollout_id=snapshot.rollout_id,
                    optimizer_step=snapshot.optimizer_step,
                    published_version=driver.published_version,
                    policy_hash=driver.published_state.policy_tensor_hash())
        try:
            self._check_deadline(tx, "trainer rebuild")
        except TransactionFailed as exc:  # nothing touched yet
            self._finish(tx, CANCELLED, error=str(exc))
            return CANCELLED
        try:
            result = dict(self.trainer_rebuilder(driver, epoch=epochs.config_epoch, cut_id=cut_id)
                          or {})
        except (RebuildRefused, RebuildNotStarted) as exc:
            # review F3: every refusal raised before the trainer is touched
            self._finish(tx, CANCELLED, error=f"trainer rebuild refused: {exc}")
            return CANCELLED
        except BaseException as exc:  # noqa: BLE001 - trainer state unknown
            self._enter_recovery(tx.tx_id, f"trainer rebuild failed: {exc!r}")
            self._tx = None
            raise RecoveryRequired(self.recovery_required) from exc
        if self._watchdog_fired.is_set():
            # the rebuild returned after the absolute deadline: outcome is
            # correct but late; journal it (4.5 bounded recovery is the caller's)
            result["late"] = True
        self._finish(tx, SUCCEEDED, rebuild=result, cut_id=cut_id)
        return SUCCEEDED

    def cancel(self, request_id: str) -> str:
        """``cancelled`` / ``recovery_started`` / ``already_committed`` / ``unknown`` (D4)."""
        known = self._by_request.get(request_id)
        if known is None:
            return "unknown"
        tx = self._tx
        if tx is None or tx.request_id != request_id:
            final = self.status(request_id).get("phase")
            if final == SUCCEEDED:
                return "already_committed"
            return "cancelled" if final == CANCELLED else "recovery_started"
        if tx.phase in (VALIDATING, WAIT_SAFE):
            self._finish(tx, CANCELLED, error="cancelled by request")
            return "cancelled"
        tx.cancel_requested = True  # acted on at the next step boundary of the executor
        return "recovery_started" if tx.phase in DESTRUCTIVE else "cancelled"

    def poll_commands(self) -> list[dict[str, Any]]:
        return self.inbox.poll(self) if self.inbox is not None else []

    def has_pending(self) -> bool:
        return self._tx is not None or self.recovery_pending is not None

    def _finish(self, tx: _Tx, phase: str, **fields: Any) -> None:
        # what the driver puts on the tape's rl_reconfiguration (error / cause / engines)
        self.last_outcome = {"tx_id": tx.tx_id, "phase": phase, **{
            k: v for k, v in fields.items() if k in ("error", "cause", "inconsistent_engines")}}
        self._phase(tx, phase, **fields)
        self._tx = None
        self.admission_open = self.recovery_required is None

    # ------------------------------------------------------------------ execution
    def _remaining(self, tx: _Tx, *, recovery: bool = False) -> float:
        limit = tx.deadline_wall + (self.timeouts.recovery if recovery else 0.0)
        return limit - self._wall()

    def _check_deadline(self, tx: _Tx, what: str) -> None:
        if self._remaining(tx) <= 0 or self._watchdog_fired.is_set():
            raise TransactionFailed(tx.phase, f"transaction deadline passed during {what}")
        if tx.cancel_requested:
            raise TransactionFailed(tx.phase, "cancelled by request")

    def _arm_watchdog(self, tx: _Tx) -> threading.Timer:
        self._watchdog_fired.clear()
        self._committing = False

        def fire() -> None:
            with self._watchdog_lock:
                if self._committing or self._tx is not tx:
                    self._record("watchdog", tx_id=tx.tx_id, phase=tx.phase, target_cells=[],
                                 note="deadline passed after the commit point; nothing killed",
                                 **self._watchdog_classification(tx, tx.phase))
                    return
                self._watchdog_fired.set()
                phase = tx.phase
                self._record("watchdog", tx_id=tx.tx_id, phase=phase,
                             target_cells=self.watchdog_target_cells(),
                             note="absolute transaction deadline passed while a step was running",
                             **self._watchdog_classification(tx, phase))
                if self._on_watchdog is not None:
                    try:
                        self._on_watchdog(tx.tx_id, phase)
                    except Exception as exc:  # noqa: BLE001 - journaled; the step result decides
                        self._record("watchdog_action", tx_id=tx.tx_id, error=repr(exc))

        timer = threading.Timer(max(0.0, self._remaining(tx)), fire)
        timer.daemon = True
        timer.start()
        return timer

    def run_at_safe_point(self, driver: Any, snapshot: ReadinessSnapshot, *,
                          outer_phase: str = PAUSABLE_PHASE) -> str | None:
        """Called by the driver at a round-boundary safe point; returns the final phase.

        ``outer_phase`` is the sync session's phase here (3.8): the pause is
        re-decided with it, so finalization / a stop round / a non-audited
        phase cancels the request instead of pausing the fleet.
        """
        if self.recovery_required:
            raise RecoveryRequired(self.recovery_required)
        if self.recovery_pending is not None:
            self._enter_recovery(self.recovery_pending["recovery_id"],
                                 "safe point reached before the restart recovery was confirmed")
            self.recovery_pending = None
            raise RecoveryRequired(self.recovery_required)
        tx = self._tx
        if tx is None:
            return None
        tx.safe_points_seen += 1
        if self._remaining(tx) <= 0:
            self._finish(tx, CANCELLED, error="deadline passed before a safe point")
            return CANCELLED
        # VALIDATING again at the safe point: epochs/profile may have moved.
        try:
            if tx.plan.kind == TRAINER_REBUILD:
                self.rebuild_plan(tx.plan.expected_config_epoch,
                                  deadline_s=float(tx.body["deadline_s"]))
            else:
                self.plan(tx.plan.target, tx.plan.expected_config_epoch,
                          deadline_s=float(tx.body["deadline_s"]))
        except Rejected as exc:
            self._finish(tx, CANCELLED, error=f"revalidation failed: {exc}")
            return CANCELLED
        decision = self._pause_decision(outer_phase, tx.plan.expected_pause_s)
        self._record("pause_decision", tx_id=tx.tx_id, rollout_id=snapshot.rollout_id,
                     outer_phase=outer_phase, allowed=decision.allowed, reason=decision.reason,
                     budget_s=decision.budget_s, stalls_peers=decision.stalls_peers,
                     expected_pause_s=tx.plan.expected_pause_s,
                     quorum_timeout_s=self.quorum_timeout_s, margin=self.pause_margin,
                     idle_flow_timeout_s=self.idle_flow_timeout_s)
        if not decision.allowed:
            self._finish(tx, CANCELLED, error=f"pause not allowed: {decision.reason}",
                         outer_phase=outer_phase)
            return CANCELLED
        self._phase(tx, WAIT_SAFE, safe_point_rollout_id=snapshot.rollout_id)
        # WAIT_SAFE covers step/outer/publication state; in-flight requests and
        # tool waits are what QUIESCING fences and drains (D4/D6).
        from dataclasses import replace as _replace

        boundary = _replace(snapshot, active_requests=0, tool_wait=0, unfinished_trajectories=0)
        blockers = self.profile and quiescent_cut_blockers(self.profile, boundary)
        if blockers:
            waited = self._wall() - (tx.deadline_wall - float(tx.body["deadline_s"]))
            if waited >= self.timeouts.safe_point:
                self._finish(tx, CANCELLED, error="no quiescent safe point: " + "; ".join(blockers))
                return CANCELLED
            self._record("safe_point_blocked", tx_id=tx.tx_id, blockers=blockers,
                         rollout_id=snapshot.rollout_id)
            return WAIT_SAFE
        timer = self._arm_watchdog(tx)
        try:
            if tx.plan.kind == TRAINER_REBUILD:
                return self._execute_trainer_rebuild(tx, driver, snapshot)
            return self._execute(tx, driver, snapshot)
        finally:
            timer.cancel()

    # -- fork-mirrored membership operations ---------------------------------------
    def _fork_call(self, tx: _Tx, op: str, members: frozenset[str],
                   call: Callable[[int], Any]) -> None:
        expected = self._fork_epoch
        cells = sorted(members)
        self._record("fork_op", tx_id=tx.tx_id, op=op, cells=cells,
                     expected_fork_epoch=expected, status="issued")
        try:
            call(expected)
        except BaseException as exc:
            status = {}
            try:
                status = dict(self._pool.membership_status())
            except Exception:  # noqa: BLE001 - reported below as unknown
                status = {"epoch": None}
            incomplete = status.get("incomplete")
            if incomplete:
                self._pending_retry = (incomplete[0], tuple(incomplete[1]))
                self._record("fork_op", tx_id=tx.tx_id, op=op, cells=cells,
                             expected_fork_epoch=expected, status="incomplete",
                             retry_op=incomplete[0], retry_cells=list(incomplete[1]),
                             reason=status.get("incomplete_reason"), error=str(exc))
            elif status.get("epoch") == expected + 1:
                # the op committed in the fork and a later part failed (e.g. the
                # start committed, waiting for tracking timed out): journal it as done
                self._fork_epoch = expected + 1
                self._last_op = (op, tuple(cells))
                self._record("fork_op", tx_id=tx.tx_id, op=op, cells=cells,
                             expected_fork_epoch=expected, result_fork_epoch=expected + 1,
                             status="done", error_after_commit=str(exc))
            else:
                self._record("fork_op", tx_id=tx.tx_id, op=op, cells=cells,
                             expected_fork_epoch=expected, status="failed",
                             fork_epoch=status.get("epoch"), error=str(exc))
                if status.get("epoch") not in (None, expected):
                    # the fork moved although the call failed: journal cannot explain it
                    self._enter_recovery(tx.tx_id, f"fork epoch moved to {status.get('epoch')} "
                                                   f"on a failed {op}")
            raise
        status = dict(self._pool.membership_status())
        result = int(status["epoch"])
        if result != expected + 1:
            self._record("fork_op", tx_id=tx.tx_id, op=op, cells=cells,
                         expected_fork_epoch=expected, status="failed", fork_epoch=result,
                         error="unexpected fork epoch after the call")
            self._enter_recovery(tx.tx_id, f"{op} left fork epoch {result}, expected {expected + 1}")
            raise TransactionFailed(tx.phase, "fork epoch disagrees with the journal")
        self._fork_epoch = result
        self._last_op = (op, tuple(cells))
        self._pending_retry = None
        self._record("fork_op", tx_id=tx.tx_id, op=op, cells=cells, expected_fork_epoch=expected,
                     result_fork_epoch=result, status="done")

    def _retry_incomplete(self, tx: _Tx, *, recovery: bool) -> None:
        """3.3a/3.7: only the same half-failed op is retried, until the deadline."""
        while self._pending_retry is not None:
            op, cells = self._pending_retry
            if self._remaining(tx, recovery=recovery) <= 0:
                raise TransactionFailed(tx.phase, f"retry of {op} {list(cells)} ran past the deadline",
                                        cause=f"{op}_retry_deadline")
            members = frozenset(cells)
            try:
                if op == "stop":
                    self._fork_call(tx, "stop", members,
                                    lambda e: self._pool.remove_engines(members, epoch=e))
                else:
                    self._fork_call(tx, "start", members,
                                    lambda e: self._pool.add_engines(len(members), epoch=e,
                                                                     members=members))
            except TransactionFailed:
                raise
            except Exception:  # noqa: BLE001 - retried until the deadline
                if self.recovery_required:
                    raise
                self._sleep(self.timeouts.retry_interval)

    def _execute(self, tx: _Tx, driver: Any, snapshot: ReadinessSnapshot) -> str:
        if tx.plan.kind in TRAINER_EDGE_KINDS:
            return self._execute_trainer(tx, driver, snapshot)
        pool: ElasticRolloutPool = driver.rollout
        publisher: MemberPublisher = driver.publisher
        self._pool = pool
        plan = tx.plan
        old_members = frozenset(pool.members())
        committed = frozenset(self.journal.epochs.members)
        if committed and old_members != committed:
            self._enter_recovery(tx.tx_id, "serving members differ from the committed members")
            self._tx = None
            raise RecoveryRequired(self.recovery_required)
        if len(old_members) != plan.source_engines:
            self._finish(tx, CANCELLED, error=f"{len(old_members)} engines serve, the source "
                                               f"config has {plan.source_engines}")
            return CANCELLED
        state = driver.published_state
        token_rollout = driver.published_version
        cut = {"rollout_id": snapshot.rollout_id, "policy_token": driver.expected_token,
               "published_version": token_rollout,
               "policy_hash": state.policy_tensor_hash() if state is not None else None,
               "optimizer_step": snapshot.optimizer_step,
               "unconsumed_batches": [] if driver.ledger is None else driver.ledger.unconsumed()}
        if state is None or cut["unconsumed_batches"]:
            self._finish(tx, CANCELLED, error="no published policy or unconsumed batches at the cut")
            return CANCELLED
        if plan.remove:
            removed = frozenset(sorted(old_members)[-plan.remove:])
        else:
            removed = frozenset()
        tx.removed = removed
        # ---- QUIESCING: admission fence + drain (3.3, D6) ----
        self.admission_open = False
        self._phase(tx, QUIESCING, cut=cut, remove=sorted(removed))
        try:
            self._check_deadline(tx, "quiesce")
            drained = self._quiesce(tx, pool, removed)
            if not drained:
                raise TransactionFailed(QUIESCING, "drain did not finish before T_drain")
            self._check_deadline(tx, "quiesce")
        except TransactionFailed as exc:
            if removed:
                try:
                    pool.undrain(removed)
                except Exception as undo:  # noqa: BLE001
                    # The cancel is only safe if the cordoned members serve again: with
                    # them still cordoned the old routing is NOT restored, so reopening
                    # admission would run the fleet short. Never CANCELLED here.
                    self._record("undrain_failed", tx_id=tx.tx_id, error=str(undo))
                    self._enter_recovery(
                        tx.tx_id, f"{exc}; undrain of {sorted(removed)} failed: {undo!r}")
                    self._tx = None
                    raise RecoveryRequired(self.recovery_required) from undo
            self._finish(tx, CANCELLED, error=str(exc))
            return CANCELLED
        # ---- destructive part ----
        try:
            if removed:
                self._phase(tx, TRANSFERRING, rollback_boundary="release", release=sorted(removed))
                try:
                    self._fork_call(tx, "stop", removed,
                                    lambda e: pool.remove_engines(removed, epoch=e))
                except TransactionFailed:
                    raise
                except Exception:  # noqa: BLE001 - half-failed stop: retry the same stop
                    self._retry_incomplete(tx, recovery=False)
                    if frozenset(pool.members()) & removed:
                        raise TransactionFailed(TRANSFERRING, "stop failed and was not retried")
            if plan.add:
                self._phase(tx, INITIALIZING)
                added = pool.plan_add(plan.add)
                if len(added) != plan.add or added & old_members:
                    raise TransactionFailed(INITIALIZING, f"pool offers {sorted(added)} for "
                                                          f"{plan.add} new engines")
                tx.added = added
                self._record("add_intent", tx_id=tx.tx_id, members=sorted(added))
                self._check_deadline(tx, "engine start")
                try:
                    self._fork_call(tx, "start", added,
                                    lambda e: pool.add_engines(plan.add, epoch=e, members=added))
                except TransactionFailed:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if self._last_op == ("start", tuple(sorted(added))):
                        tx.extra["started"] = True  # committed, then failed: stop them again
                        raise TransactionFailed(INITIALIZING, f"started engines unusable: {exc}") from exc
                    if self._pending_retry is None:
                        # the fork rolled the start back: nothing started
                        tx.added = frozenset()
                        raise TransactionFailed(INITIALIZING, f"start failed: {exc}") from exc
                    raise TransactionFailed(INITIALIZING, f"start half failed: {exc}") from exc
                tx.extra["started"] = True
                self._check_deadline(tx, "engine start")
                self._phase(tx, VERIFYING)
                result = publisher.publish_members(state, added, epoch=self._fork_epoch,
                                                   token_rollout_id=token_rollout)
                if (result.manifest.target_policy_hash != cut["policy_hash"]
                        or frozenset(result.members) != added):
                    raise TransactionFailed(VERIFYING, "member publication acknowledged "
                                                       f"{sorted(result.members)} / other policy")
                self._record("weight_admission", tx_id=tx.tx_id, members=sorted(added),
                             policy_token=cut["policy_token"],
                             manifest_hash=result.manifest.target_manifest_hash)
                self._emit_member_publication(driver, tx, result, token_rollout,
                                              phase=VERIFYING)
            else:
                self._phase(tx, VERIFYING)
            self._check_deadline(tx, "verify")
            target_members = (old_members - removed) | tx.added
            serving = frozenset(pool.members())
            if serving != target_members:
                raise TransactionFailed(VERIFYING, f"serving {sorted(serving)}, expected "
                                                   f"{sorted(target_members)}")
            commit_check = getattr(publisher, "verify_serving_policy", None)
            if callable(commit_check):
                commit_check(epoch=self._fork_epoch, token_rollout_id=token_rollout,
                             state=state)
            if tx.cancel_requested:
                raise TransactionFailed(VERIFYING, "cancelled by request")
        except TransactionFailed as exc:
            return self._rebuild_old(tx, driver, old_members, str(exc), **_failure_detail(exc))
        except RecoveryRequired:
            raise
        except Exception as exc:  # noqa: BLE001 - any engine failure: restore the old set
            return self._rebuild_old(tx, driver, old_members, f"{type(exc).__name__}: {exc}",
                                     **_failure_detail(exc))
        # ---- COMMITTED: the single commit point is the durable CAS ----
        # Review F2: re-check the watchdog under its lock; a fired watchdog may
        # have killed the new cells after the last verify, so never commit then.
        with self._watchdog_lock:
            fired = self._watchdog_fired.is_set()
            if not fired:
                self._committing = True
                new_epoch = self._commit(tx, plan.target, target_members)
        if fired:
            return self._rebuild_old(tx, driver, old_members,
                                     "watchdog fired before the commit point")
        self._phase(tx, COMMITTED, members=sorted(target_members))
        self._emit_membership(driver, tx, new_epoch, target_members,
                              kind="up" if plan.add else "down", rollout_id=snapshot.rollout_id)
        # ---- RESUMING ----
        self._phase(tx, RESUMING)
        placement = driver.placement
        if isinstance(placement, ReconfigurablePlacement):
            current = placement.describe()
            gpus = self.configs[plan.target].placement or {}
            rollout_gpus = tuple(gpus.get("rollout") or current.rollout_gpus)
            try:
                placement.reconfigure(
                    PlacementDescription(current.kind, current.trainer_gpus, rollout_gpus,
                                         dict(current.extra)),
                    epoch=new_epoch,
                )
            except Exception as exc:  # noqa: BLE001 - committed: never roll back blindly
                self._enter_recovery(tx.tx_id, f"placement bookkeeping failed after commit: {exc}")
                self._tx = None
                raise RecoveryRequired(self.recovery_required) from exc
        driver.config_epoch = new_epoch
        self._finish(tx, SUCCEEDED, config_epoch_to=new_epoch)
        return SUCCEEDED

    # -- the ONLY transaction commit point (durable epoch CAS), shared by the
    # E1 rollout edges and the E3 trainer edges ----------------------------------
    def _commit(self, tx: _Tx, target: str, members: Any, *,
                cut_id: str | None = None) -> int:
        """Durable CAS. A failed CAS (EpochConflict, fsync/rename error) leaves
        the new members/trainer running with the commit durable or not: never a
        silent continue -> RECOVERY_REQUIRED. For a trainer edge (``cut_id``)
        the recovery hint is decided from the durable epochs file (E3 v3):
        ``last_tx_id == tx`` -> ``restore_target`` (committed=True); another
        tx -> ``restore_old`` (committed=False); epochs unreadable ->
        ``recovery_required`` (committed=None)."""
        epochs = self.journal.epochs
        new_epoch = epochs.config_epoch + 1
        try:
            self.journal.compare_and_swap(
                expected_config_epoch=epochs.config_epoch,
                new=EpochState(new_epoch, target, self._fork_epoch, tuple(sorted(members)),
                               tx.tx_id),
            )
        except Exception as exc:  # noqa: BLE001 - EpochConflict, I/O error
            if cut_id is not None:
                try:
                    durable = read_epochs(self.state_dir / "reconfig")
                except Exception as read_exc:  # noqa: BLE001 - unknown: no restore hint
                    durable, why = None, f"epochs unreadable: {read_exc!r}"
                else:
                    why = f"durable last_tx_id={durable.last_tx_id}"
                # three states (E3 v3): durable commit / not committed / unknown
                committed = None if durable is None else durable.last_tx_id == tx.tx_id
                action = ("recovery_required" if committed is None
                          else "restore_target" if committed else "restore_old")
                self._record("trainer_recovery_hint", tx_id=tx.tx_id, action=action,
                             committed=committed, cut_id=cut_id, cut_epoch=epochs.config_epoch,
                             config=target if committed else epochs.config_id,
                             reason=f"commit CAS failed: {exc!r}; {why}")
            self._enter_recovery(tx.tx_id, f"commit CAS failed: {exc!r}")
            self._tx = None
            raise RecoveryRequired(self.recovery_required) from exc
        return new_epoch

    def _sync_fork_mirror(self) -> None:
        """REBUILT_OLD: config epoch unchanged; only the fork membership mirror moved."""
        epochs = self.journal.epochs
        if epochs.fork_membership_epoch != self._fork_epoch:
            self.journal.compare_and_swap(
                expected_config_epoch=epochs.config_epoch,
                new=EpochState(epochs.config_epoch, epochs.config_id, self._fork_epoch,
                               epochs.members, epochs.last_tx_id),
            )

    def _execute_trainer(self, tx: _Tx, driver: Any, snapshot: ReadinessSnapshot) -> str:
        """E3 (4.7): run the trainer edge to the commit point, then commit here (single CAS)."""
        pool: ElasticRolloutPool = driver.rollout
        self._pool = pool
        plan = tx.plan
        tplan = self._trainer_plan(plan.source, plan.target, plan.expected_config_epoch)
        committed = frozenset(self.journal.epochs.members)
        if committed and frozenset(pool.members()) != committed:
            self._enter_recovery(tx.tx_id, "serving members differ from the committed members")
            self._tx = None
            raise RecoveryRequired(self.recovery_required)
        state = driver.published_state
        if state is None or (driver.ledger is not None and driver.ledger.unconsumed()):
            self._finish(tx, CANCELLED, error="no published policy or unconsumed batches at the cut")
            return CANCELLED
        self.admission_open = False
        ctx = self._trainer_edges()
        transition = TrainerTransition(
            plan=tplan, tx_id=tx.tx_id, epoch=self.journal.epochs.config_epoch, trainer=ctx["ops"],
            pool=pool, publisher=driver.publisher, published_state=state,
            published_version=driver.published_version,
            record=lambda kind, **f: self._trainer_record(tx, kind, **f),
            fork_call=lambda op, members, call: self._fork_call(tx, op, members, call),
            fork_epoch=lambda: self._fork_epoch,
            drain_deadline=self._wall() + min(self.timeouts.drain, self._remaining(tx)),
        )
        result = transition.run()
        if result.phase == CANCELLED:
            self._finish(tx, CANCELLED, error=result.error)
            return CANCELLED
        if result.phase == RECOVERY_REQUIRED:
            self._enter_recovery(tx.tx_id, result.error or "trainer transition failed")
            self._tx = None
            raise RecoveryRequired(self.recovery_required)
        if result.phase == REBUILT_OLD:
            self._sync_fork_mirror()
            self._finish(tx, REBUILT_OLD, error=result.error)
            return REBUILT_OLD
        assert result.phase == READY_TO_COMMIT, result.phase
        with self._watchdog_lock:
            fired = self._watchdog_fired.is_set()
            if not fired:
                self._committing = True
                new_epoch = self._commit(tx, plan.target, result.target_members,
                                         cut_id=result.cut_id)
        if fired:
            # the watchdog may have killed the new engines after the transition's
            # last check: never commit; the trainer already runs the target layout
            self._record("trainer_recovery_hint", tx_id=tx.tx_id, action="restore_old",
                         cut_id=result.cut_id, cut_epoch=self.journal.epochs.config_epoch,
                         config=self.journal.epochs.config_id,
                         reason="watchdog fired before the commit point")
            self._enter_recovery(tx.tx_id, "watchdog fired before the trainer edge commit")
            self._tx = None
            raise RecoveryRequired(self.recovery_required)
        self._phase(tx, COMMITTED, members=sorted(result.target_members), cut_id=result.cut_id)
        self._phase(tx, RESUMING)
        placement = driver.placement
        gpus = self.configs[plan.target].placement or {}
        if isinstance(placement, ReconfigurablePlacement):
            current = placement.describe()
            try:
                move = getattr(placement, "reconfigure_trainer", None)
                if not callable(move):
                    raise RuntimeError("placement cannot record a trainer GPU change")
                move(PlacementDescription(current.kind, tuple(gpus.get("trainer") or ()),
                                          tuple(gpus.get("rollout") or ()), dict(current.extra)),
                     epoch=new_epoch)
            except Exception as exc:  # noqa: BLE001 - committed: never roll back blindly
                self._enter_recovery(tx.tx_id, f"placement bookkeeping failed after commit: {exc}")
                self._tx = None
                raise RecoveryRequired(self.recovery_required) from exc
        driver.config_epoch = new_epoch
        self._finish(tx, SUCCEEDED, config_epoch_to=new_epoch)
        return SUCCEEDED

    def _trainer_record(self, tx: _Tx, kind: str, **fields: Any) -> None:
        if kind == "phase":
            with self._watchdog_lock:  # the watchdog reads tx.phase under this lock
                tx.phase = fields["phase"]  # the fork mirror and cancel() read the current phase
            fields.setdefault("config_epoch", self.journal.epochs.config_epoch)
            fields.setdefault("fork_epoch", self._fork_epoch)
        self._record(kind, request_id=tx.request_id, **fields)

    def _quiesce(self, tx: _Tx, pool: ElasticRolloutPool, removed: frozenset[str]) -> bool:
        """3.3: admission is fenced; wait for engine requests AND tool waits to reach 0.

        ``active_requests == 0`` does not imply no trajectory waits on a tool
        (D6); while ``tool_wait > 0`` the old routing is kept and nothing is
        released. A timeout cancels the switch; nothing is aborted or replayed.
        """
        budget = min(self.timeouts.drain, self._remaining(tx))
        deadline = self._wall() + budget
        if removed and not pool.drain(removed, deadline):
            return False
        probe = getattr(pool, "trajectory_load", None)
        while True:
            load = probe() if callable(probe) else None
            if load is None:
                # unknown load: only the serial round boundary proves quiescence
                # (generate returned, D6 round-boundary drain); any other mode fails closed
                return self.profile is not None and self.profile.execution_mode == "partitioned-serial"
            if "blockers" in load:  # tool_wait.drain_blockers: unknown counts fail closed
                blockers = list(load["blockers"])
            else:
                blockers = [f"{k}={load.get(k)}" for k in ("active_requests", "tool_wait")
                            if load.get(k) is None or int(load[k]) > 0]
            active, tool_wait = load.get("active_requests"), load.get("tool_wait")
            if not blockers:
                return True
            if self._wall() >= deadline:
                self._record("drain_timeout", tx_id=tx.tx_id, active_requests=active,
                             tool_wait=tool_wait, blockers=blockers)
                return False
            self._sleep(self.timeouts.retry_interval)

    def _emit_member_publication(self, driver: Any, tx: _Tx, result: Any, rollout_id: Any, *,
                                 phase: str) -> None:
        """Tape record of a member-scoped publication (3.4a/3.5 ``publish_members``):
        the already-published policy loaded into ``members`` (new or restarted
        engines) at a reconfiguration, so the engines serving the next generation
        are auditable on the tape before the next full ``rl_publication``."""
        emit = getattr(driver, "emit", None)
        if not callable(emit):
            return
        manifest = result.manifest
        serving = sorted(frozenset(self._pool.members()) | frozenset(result.members))
        emit(
            "rl_member_publication",
            policy_version=rollout_id,
            tx_id=tx.tx_id,
            phase=phase,
            **{
                "rl/policy_token": getattr(driver, "expected_token", None),
                "sync/publication_payload_bytes": manifest.payload_bytes,
                "sync/publication_payload_hash": manifest.payload_hash,
                "sync/publication_manifest_hash": manifest.target_manifest_hash,
                "sync/publication_members": sorted(result.members),
                "sync/serving_members": serving,
            },
        )

    def _emit_membership(self, driver: Any, tx: _Tx, config_epoch: int, members: Any, *,
                         kind: str, rollout_id: int) -> None:
        """Tape record of a committed scale-up/-down: the membership the next generation
        (rollout ``rollout_id``, generated right after this safe point) is served by.
        Carries no payload: it is NOT a publication (``rl_member_publication`` is only
        written when a member publication really ran)."""
        emit = getattr(driver, "emit", None)
        if callable(emit):
            emit("rl_membership", config_epoch=config_epoch, members=sorted(members),
                 tx_id=tx.tx_id, kind=kind, round=rollout_id)

    def _rebuild_old(self, tx: _Tx, driver: Any, old_members: frozenset[str], error: str, *,
                     cause: str = "other", inconsistent_engines: Any = ()) -> str:
        """E1 failure path (D5): restore the old engine count and republish the same policy.

        ``cause`` / ``inconsistent_engines`` (publication cause enum, disagreeing engine ids)
        go on the REBUILD_OLD and REBUILT_OLD journal records."""
        pool = self._pool
        if self.recovery_required:
            self._tx = None
            raise RecoveryRequired(self.recovery_required)
        detail = {"cause": cause, "inconsistent_engines": sorted(inconsistent_engines)}
        # 2026-10-02 ruling (E1-D ④): exactly one bounded REBUILD_OLD after the transaction
        # deadline; its limit is deadline_wall + T_recovery, the transaction deadline itself
        # is never reset or extended, and a second failure is RECOVERY_REQUIRED at once.
        self._phase(tx, REBUILD_OLD, error=error, deadline_wall=tx.deadline_wall,
                    recovery_deadline_wall=tx.deadline_wall + self.timeouts.recovery, **detail)
        try:
            self._retry_incomplete(tx, recovery=True)
            if tx.extra.get("started"):
                stray = tx.added
                self._fork_call(tx, "stop", stray, lambda e: pool.remove_engines(stray, epoch=e))
                tx.extra["started"] = False
                self._retry_incomplete(tx, recovery=True)
            missing = old_members - frozenset(pool.members())
            if missing:
                self._fork_call(tx, "start", missing,
                                lambda e: pool.add_engines(len(missing), epoch=e, members=missing))
                result = driver.publisher.publish_members(
                    driver.published_state, missing, epoch=self._fork_epoch,
                    token_rollout_id=driver.published_version)
                if frozenset(result.members) != missing:
                    raise TransactionFailed(REBUILD_OLD, "old engines did not acknowledge")
                self._emit_member_publication(driver, tx, result, driver.published_version,
                                              phase=REBUILD_OLD)
            still = tx.removed & frozenset(pool.members())
            if still:
                pool.undrain(still)
            if frozenset(pool.members()) != old_members:
                raise TransactionFailed(REBUILD_OLD, "old engine set not restored")
            if self._remaining(tx, recovery=True) <= 0:
                raise TransactionFailed(REBUILD_OLD, "recovery budget spent")
        except Exception as exc:  # noqa: BLE001
            self._enter_recovery(tx.tx_id, f"rebuild of the old engine set failed: {exc}",
                                 cause="rebuild_old_failed")
            self._tx = None
            raise RecoveryRequired(self.recovery_required) from exc
        self._sync_fork_mirror()
        self._finish(tx, REBUILT_OLD, error=error, **detail)
        return REBUILT_OLD


# ---------------------------------------------------------------------- status helpers
def request_status(records: Any, request_id: str) -> dict[str, Any]:
    request = next((r for r in records if r["kind"] == "request"
                    and r["request_id"] == request_id), None)
    if request is None:
        return {"request_id": request_id, "known": False}
    phases = [r for r in records if r["kind"] == "phase" and r.get("tx_id") == request["tx_id"]]
    last = phases[-1] if phases else None
    return {
        "request_id": request_id,
        "known": True,
        "tx_id": request["tx_id"],
        "target": request["body"].get("target"),
        "kind": request["body"].get("kind", "rollout-only"),
        "phase": last["phase"] if last else VALIDATING,
        "terminal": bool(last and last["phase"] in TERMINAL),
        "config_epoch": last["config_epoch"] if last else None,
        "error": last.get("error") if last else None,
    }


class CommandInbox:
    """Restricted command files (D4 manual entry): ``<dir>/<request_id>.<verb>.json``.

    Verbs: ``request`` (``{"target", "expected_config_epoch", "deadline_s"}``),
    ``rebuild`` (4.4 same-shape trainer rebuild: ``{"kind", "expected_config_epoch",
    "deadline_s"}``) and ``cancel`` (``{}``). The learner polls at safe points; answers are written
    as ``<request_id>.status.json`` (atomic rename). Files are consumed.
    """

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)

    def submit(self, request_id: str, verb: str, body: Mapping[str, Any]) -> Path:
        if verb not in ("request", "cancel", "rebuild") or "/" in request_id or not request_id:
            raise ValueError("bad command")
        path = self.dir / f"{request_id}.{verb}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(body), sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    def poll(self, controller: IslandController) -> list[dict[str, Any]]:
        answers = []
        for path in (sorted(self.dir.glob("*.request.json")) + sorted(self.dir.glob("*.rebuild.json"))
                     + sorted(self.dir.glob("*.cancel.json"))):
            request_id, verb = path.name[: -len(".json")].rsplit(".", 1)
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
                if verb == "request":
                    answer = controller.request(request_id, body["target"],
                                                int(body["expected_config_epoch"]),
                                                float(body["deadline_s"]))
                elif verb == "rebuild":
                    answer = controller.request_trainer_rebuild(
                        request_id, int(body["expected_config_epoch"]), float(body["deadline_s"]))
                else:
                    answer = {"request_id": request_id, "cancel": controller.cancel(request_id)}
            except (Rejected, KeyError, ValueError, TypeError) as exc:
                answer = {"request_id": request_id, "rejected": str(exc)}
            path.unlink()
            self.write_status(request_id, answer)
            answers.append(answer)
        return answers

    def write_status(self, request_id: str, answer: Mapping[str, Any]) -> None:
        out = self.dir / f"{request_id}.status.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(answer), sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(out)


def main(argv: list[str] | None = None) -> int:
    """``python -m yeto.rl.engine.controller {request,cancel,status} ...`` (read-only on the journal)."""
    import argparse

    parser = argparse.ArgumentParser(prog="yeto.rl.engine.controller")
    parser.add_argument("--state-dir", required=True)
    sub = parser.add_subparsers(dest="verb", required=True)
    req = sub.add_parser("request")
    req.add_argument("request_id")
    req.add_argument("--target", required=True)
    req.add_argument("--expected-epoch", type=int, required=True)
    req.add_argument("--deadline-s", type=float, required=True)
    reb = sub.add_parser("rebuild-trainer")
    reb.add_argument("request_id")
    reb.add_argument("--expected-epoch", type=int, required=True)
    reb.add_argument("--deadline-s", type=float, required=True)
    can = sub.add_parser("cancel")
    can.add_argument("request_id")
    st = sub.add_parser("status")
    st.add_argument("request_id", nargs="?")
    args = parser.parse_args(argv)
    state = Path(args.state_dir).expanduser()
    inbox = CommandInbox(state / "inbox")
    if args.verb == "request":
        inbox.submit(args.request_id, "request", request_body(args.target, args.expected_epoch,
                                                              args.deadline_s))
    elif args.verb == "rebuild-trainer":
        inbox.submit(args.request_id, "rebuild",
                      rebuild_request_body(args.expected_epoch, args.deadline_s))
    elif args.verb == "cancel":
        inbox.submit(args.request_id, "cancel", {})
    else:
        records = read_journal(state / "reconfig")
        epochs = read_epochs(state / "reconfig")
        out: dict[str, Any] = {"epochs": epochs.to_dict()}
        if args.request_id:
            out["request"] = request_status(records, args.request_id)
        print(json.dumps(out, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
