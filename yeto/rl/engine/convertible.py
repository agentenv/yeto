"""Convertible nodes: rollout by default, trainer when the scheduler says so.

rl-trainer-start-stop design D7 (Q5), tasks L7.1–L7.3. Neutral core: node
classes, the conversion port, the P0-S cut rule for in-flight trajectories, the
rollout capacity of the node pool, and the scheduler that converts nodes.
No backend names here; the Miles/SGLang port is
:mod:`yeto.rl.adapters.miles.convertible`.

Node classes (D7):

* ``trainer``: on-demand, fast interconnect, trainer precision;
* ``rollout``: may be spot, another region, lower precision;
* ``convertible``: a trainer-class node that does rollout by default. A spot node
  or a node with a rollout precision other than the trainer's is refused.

Conversion rollout -> trainer (in this order):

1. the rollout engine stops taking requests; the in-flight trajectories are cut
   by the P0-S rule (:func:`apply_cut_rule`): finished segments go back to the
   buffer, unfinished trajectories are dropped and regenerated (P0-S-Q1 ③);
2. the rollout engine releases its GPU memory;
3. the trainer onloads on the same GPUs.

Conversion trainer -> rollout: the trainer offloads, the rollout engine takes its
memory back and loads the published version, the weight checksum is compared
with the published checksum, then the engine takes requests again.

The scheduler never converts the last rollout-capable node, and it counts the
rollout capacity the conversion takes away. The idle time of a convertible node
is the time it does neither rollout nor training.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .trainer_lifecycle import BacklogStats, LifecycleState, SampleIndexEntry, Start, Stop, TrainerScheduler

CONVERSION_EVENT = "node_conversion"
CONVERSION_DECISION_KIND = "convertible_scheduler_decision"


# ---------------------------------------------------------------------------
# L7.1 Node classes
# ---------------------------------------------------------------------------
class NodeClass(str, enum.Enum):
    ROLLOUT = "rollout"
    TRAINER = "trainer"
    CONVERTIBLE = "convertible"


class NodeClassError(ValueError):
    """A node configuration breaks the class rule; the message names the node."""


@dataclass(frozen=True)
class NodeSpec:
    """One node of the run. ``precision`` is the rollout precision of the node
    (None: the trainer precision). ``capacity`` is its rollout capacity in any
    common unit (e.g. trajectories per second); only ratios are used."""

    name: str
    node_class: NodeClass
    spot: bool = False
    precision: str | None = None
    capacity: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise NodeClassError("node name is empty")
        if self.capacity <= 0:
            raise NodeClassError(f"node {self.name}: capacity must be > 0, got {self.capacity}")

    @property
    def does_rollout(self) -> bool:
        return self.node_class is not NodeClass.TRAINER


def validate_nodes(nodes: Iterable[NodeSpec], *, trainer_precision: str) -> None:
    """Refuse a convertible node that is spot or has another rollout precision."""
    seen: set[str] = set()
    for node in nodes:
        if node.name in seen:
            raise NodeClassError(f"node {node.name}: name is used twice")
        seen.add(node.name)
        if node.node_class is not NodeClass.CONVERTIBLE:
            continue
        if node.spot:
            raise NodeClassError(
                f"node {node.name}: a spot node cannot be convertible "
                "(only trainer-class nodes: on-demand, trainer precision)")
        if node.precision is not None and node.precision != trainer_precision:
            raise NodeClassError(
                f"node {node.name}: rollout precision {node.precision} is not the trainer "
                f"precision {trainer_precision}; only trainer-class nodes can be convertible")


def parse_node_spec(text: str) -> NodeSpec:
    """``NAME=CLASS[:spot][:PRECISION][:capacity=X]``, e.g. ``island1=convertible``,
    ``island2=rollout:spot:fp8``."""
    name, sep, rest = text.partition("=")
    if not sep:
        raise NodeClassError(f"node spec {text!r}: expected NAME=CLASS[:spot][:PRECISION]")
    head, *options = rest.split(":")
    try:
        node_class = NodeClass(head)
    except ValueError:
        raise NodeClassError(f"node {name}: class {head!r} is not one of "
                             f"{[c.value for c in NodeClass]}") from None
    spot, precision, capacity = False, None, 1.0
    for option in options:
        if option == "spot":
            spot = True
        elif option.startswith("capacity="):
            capacity = float(option.removeprefix("capacity="))
        elif option:
            precision = option
    return NodeSpec(name.strip(), node_class, spot=spot, precision=precision, capacity=capacity)


# ---------------------------------------------------------------------------
# L7.2 P0-S cut rule for in-flight trajectories
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InflightTrajectory:
    """A trajectory on the node when it stops taking requests. ``finished`` are
    its segments that ended before the stop (each a complete sample with its
    behavior version); ``unfinished`` is true when a part was still generating."""

    trajectory_id: str
    finished: tuple[SampleIndexEntry, ...] = ()
    unfinished: bool = True


@dataclass(frozen=True)
class CutOutcome:
    returned: tuple[str, ...] = ()  # sample ids appended to the buffer
    dropped: tuple[str, ...] = ()  # trajectory ids to regenerate


class SampleSink(Protocol):
    def append(self, entry: SampleIndexEntry) -> None: ...


def apply_cut_rule(inflight: Iterable[InflightTrajectory], buffer: SampleSink) -> CutOutcome:
    """P0-S cut: finished segments go to the buffer; the unfinished part is dropped
    and the sampler regenerates the trajectory (P0-S-Q1 ③). A trajectory whose
    segments all finished is not dropped."""
    returned: list[str] = []
    dropped: list[str] = []
    for trajectory in inflight:
        for entry in trajectory.finished:
            buffer.append(entry)
            returned.append(entry.sample_id)
        if trajectory.unfinished:
            dropped.append(trajectory.trajectory_id)
    return CutOutcome(tuple(returned), tuple(dropped))


# ---------------------------------------------------------------------------
# L7.2 Conversion port and node
# ---------------------------------------------------------------------------
class ConversionPort(Protocol):
    """Mechanism on one convertible node. The node calls the verbs in the order of
    the module docstring; the port does not decide anything."""

    def stop_requests(self) -> list[InflightTrajectory]: ...
    def release_rollout(self) -> None: ...
    def onload_trainer(self) -> None: ...
    def offload_trainer(self) -> None: ...
    def resume_rollout(self, version: int) -> str | None: ...  # weight checksum after load
    def accept_requests(self) -> None: ...


class Role(str, enum.Enum):
    ROLLOUT = "rollout"
    TO_TRAINER = "to_trainer"
    TRAINER = "trainer"
    TO_ROLLOUT = "to_rollout"
    FAILED = "failed"


class ConversionError(RuntimeError):
    pass


class LastRolloutNodeError(ConversionError):
    """The conversion would leave no node doing rollout."""


@dataclass(frozen=True)
class ConversionReceipt:
    node: str
    direction: str  # "to_trainer" | "to_rollout"
    seconds: float
    phases: Mapping[str, float]
    cut: CutOutcome | None = None
    version: int | None = None
    weight_checksum: str | None = None

    def event(self) -> dict[str, Any]:
        out: dict[str, Any] = {"event": CONVERSION_EVENT, "node": self.node,
                               "direction": self.direction, "seconds": self.seconds,
                               "phases": dict(self.phases)}
        if self.cut is not None:
            out["returned"] = len(self.cut.returned)
            out["dropped"] = len(self.cut.dropped)
        if self.version is not None:
            out["version"] = self.version
            out["weight_checksum"] = self.weight_checksum
        return out


class ConvertibleNode:
    """Role state of one convertible node, plus its idle time (L7.3): the time in
    ``TRAINER`` role while it does not train. Rollout time is never idle."""

    def __init__(self, spec: NodeSpec, port: ConversionPort, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if spec.node_class is not NodeClass.CONVERTIBLE:
            raise NodeClassError(f"node {spec.name}: class {spec.node_class.value} is not convertible")
        self.spec = spec
        self.port = port
        self._clock = clock
        self.role = Role.ROLLOUT
        self._training = False
        self._idle_s = 0.0
        self._idle_from: float | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    # ------------------------------------------------------------ idle (L7.3)
    def set_training(self, active: bool) -> None:
        """The trainer on this node trains (True) or waits for samples (False)."""
        self._training = active
        self._update_idle()

    def idle_seconds(self) -> float:
        now = self._clock()
        return self._idle_s + (now - self._idle_from if self._idle_from is not None else 0.0)

    def _update_idle(self) -> None:
        idle = self.role is Role.TRAINER and not self._training
        now = self._clock()
        if idle and self._idle_from is None:
            self._idle_from = now
        elif not idle and self._idle_from is not None:
            self._idle_s += now - self._idle_from
            self._idle_from = None

    def _set_role(self, role: Role) -> None:
        self.role = role
        if role is not Role.TRAINER:
            self._training = False
        self._update_idle()

    # ------------------------------------------------------------ conversion
    def to_trainer(self, buffer: SampleSink) -> ConversionReceipt:
        self._require(Role.ROLLOUT, "to_trainer")
        self._set_role(Role.TO_TRAINER)
        phases: dict[str, float] = {}
        t0 = self._clock()
        try:
            inflight = self._timed(phases, "stop_requests", self.port.stop_requests)
            cut = apply_cut_rule(inflight or (), buffer)
            self._timed(phases, "release_rollout", self.port.release_rollout)
            self._timed(phases, "onload_trainer", self.port.onload_trainer)
        except BaseException as exc:
            self._set_role(Role.FAILED)
            raise ConversionError(f"node {self.name}: rollout -> trainer failed: {exc}") from exc
        self._set_role(Role.TRAINER)
        return ConversionReceipt(self.name, "to_trainer", self._clock() - t0, phases, cut=cut)

    def to_rollout(self, version: int, *, expected_checksum: str | None = None) -> ConversionReceipt:
        """Back to rollout on published ``version``. With ``expected_checksum``
        (the published checksum of that version), a different engine checksum
        fails the conversion and the engine does not take requests."""
        self._require(Role.TRAINER, "to_rollout")
        self._set_role(Role.TO_ROLLOUT)
        phases: dict[str, float] = {}
        t0 = self._clock()
        try:
            self._timed(phases, "offload_trainer", self.port.offload_trainer)
            checksum = self._timed(phases, "resume_rollout", lambda: self.port.resume_rollout(version))
            if expected_checksum is not None and checksum != expected_checksum:
                raise ConversionError(f"weight checksum {checksum} != published {expected_checksum} "
                                      f"for version {version}")
            self._timed(phases, "accept_requests", self.port.accept_requests)
        except BaseException as exc:
            self._set_role(Role.FAILED)
            raise ConversionError(f"node {self.name}: trainer -> rollout failed: {exc}") from exc
        self._set_role(Role.ROLLOUT)
        return ConversionReceipt(self.name, "to_rollout", self._clock() - t0, phases,
                                 version=version, weight_checksum=checksum)

    def _require(self, role: Role, verb: str) -> None:
        if self.role is not role:
            raise ConversionError(f"node {self.name}: {verb} needs role {role.value}, "
                                  f"node is {self.role.value}")

    def _timed(self, phases: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
        t = self._clock()
        try:
            return fn()
        finally:
            phases[name] = self._clock() - t


# ---------------------------------------------------------------------------
# L7.3 Rollout capacity of the pool
# ---------------------------------------------------------------------------
class NodePool:
    """All nodes of the run. Fixed rollout and trainer nodes have a fixed role;
    convertible nodes carry their :class:`ConvertibleNode` role."""

    def __init__(self, specs: Iterable[NodeSpec], convertible: Iterable[ConvertibleNode] = ()) -> None:
        self.specs = {s.name: s for s in specs}
        self.convertible = {n.name: n for n in convertible}
        for name, node in self.convertible.items():
            if self.specs.get(name, node.spec) != node.spec:
                raise NodeClassError(f"node {name}: convertible node spec differs from the pool spec")
            self.specs[name] = node.spec

    def doing_rollout(self) -> list[str]:
        out = []
        for name, spec in self.specs.items():
            node = self.convertible.get(name)
            if node is not None:
                if node.role is Role.ROLLOUT:
                    out.append(name)
            elif spec.node_class is NodeClass.ROLLOUT:
                out.append(name)
        return out

    def total_capacity(self) -> float:
        """Rollout capacity with every rollout-capable node doing rollout."""
        return sum(s.capacity for s in self.specs.values() if s.does_rollout)

    def capacity(self, without: Iterable[str] = ()) -> float:
        skip = set(without)
        return sum(self.specs[n].capacity for n in self.doing_rollout() if n not in skip)

    def check_conversion(self, names: Sequence[str]) -> float:
        """Rollout capacity lost by converting ``names`` to trainer. Raises
        :class:`LastRolloutNodeError` if no node would be left doing rollout."""
        rolling = set(self.doing_rollout())
        for name in names:
            if name not in self.convertible or name not in rolling:
                raise ConversionError(f"node {name}: not a convertible node doing rollout")
        if not rolling - set(names):
            raise LastRolloutNodeError(
                f"converting {sorted(names)} leaves no node doing rollout "
                "(the last rollout-capable node is never converted)")
        return self.capacity() - self.capacity(without=names)

    def in_role(self, role: Role) -> list[ConvertibleNode]:
        return [n for n in self.convertible.values() if n.role is role]


# ---------------------------------------------------------------------------
# Node selection: which convertible nodes become the trainer (GOAL §7.1 item 14)
# ---------------------------------------------------------------------------
# Training speed rank of a GPU type (higher trains faster). An unknown type ranks 0.
GPU_TRAIN_RANK: Mapping[str, int] = {"B300": 6, "B200": 5, "H200": 4, "H100": 3, "A100": 2,
                                     "L40S": 1}


def gpu_train_rank(gpu: str | None) -> int:
    """Rank of a GPU type name (``H100``, ``H100!``, ``H100-80GB``); 0 when unknown."""
    if not gpu:
        return 0
    head = str(gpu).upper().rstrip("!").split("-")[0].split(":")[0]
    return GPU_TRAIN_RANK.get(head, 0)


@dataclass(frozen=True)
class NodeProfile:
    """What the selection knows of one node. Declared: ``spot``, ``precision``
    (None: the trainer precision), ``gpu`` type, ``gpus`` on the node,
    ``gpu_memory_gb`` per GPU, network ``domain`` (e.g. ``modal`` or
    ``aws/us-east-1``). Measured (None: not reported yet): ``inflight`` trajectories
    on the node and its rollout ``capacity`` (groups per second)."""

    name: str
    spot: bool = False
    precision: str | None = None
    gpu: str | None = None
    gpus: int = 1
    gpu_memory_gb: float | None = None
    domain: str | None = None
    inflight: int | None = None
    capacity: float | None = None


@dataclass(frozen=True)
class TrainerNeeds:
    """What a trainer on one node needs. ``memory_gb``: trainer state per GPU (None:
    not checked); ``gpus``: GPUs one trainer node must hold for its TP/DP layout;
    ``domains``: network domains of the running trainer and of the parameter source
    (the store holding the newest cut)."""

    precision: str
    memory_gb: float | None = None
    gpus: int = 1
    domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedNode:
    """A node that passed the hard conditions, with its sort key (lower first) and
    the value of each level, for the journal."""

    name: str
    key: tuple[Any, ...]
    levels: Mapping[str, Any]


def hard_condition_failure(node: NodeProfile, needs: TrainerNeeds) -> str | None:
    """Level 1 (hard conditions): on-demand, trainer precision, trainer state fits
    in GPU memory. Returns the first failed condition, or None."""
    if node.spot:
        return "spot"
    if node.precision is not None and node.precision != needs.precision:
        return f"precision {node.precision} != trainer {needs.precision}"
    if needs.memory_gb is not None and node.gpu_memory_gb is not None \
            and node.gpu_memory_gb < needs.memory_gb:
        return f"gpu memory {node.gpu_memory_gb:g} GB < trainer state {needs.memory_gb:g} GB"
    return None


def rank_nodes(nodes: Iterable[NodeProfile], needs: TrainerNeeds
               ) -> tuple[list[RankedNode], dict[str, str]]:
    """Order the candidate nodes by the five selection levels (GOAL §7.1 item 14):

    1. hard conditions (:func:`hard_condition_failure`); a failing node is refused;
    2. same network domain as the running trainer or the parameter source first;
    3. enough GPUs on the node for the trainer layout first, then the faster GPU type
       (:data:`GPU_TRAIN_RANK`: B200/H200 before H100);
    4. fewer in-flight trajectories first (less work lost by the conversion);
    5. smaller rollout capacity first (less rollout lost after the conversion).

    A level that is not known (no domain declared, nothing measured yet) sorts after
    the known values of that level. The name breaks the last tie. Returns the ranked
    nodes and ``{name: reason}`` of the refused ones."""
    ranked: list[RankedNode] = []
    refused: dict[str, str] = {}
    domains = set(needs.domains)
    for node in nodes:
        reason = hard_condition_failure(node, needs)
        if reason is not None:
            refused[node.name] = reason
            continue
        near = None if node.domain is None or not domains else node.domain in domains
        fits = node.gpus >= needs.gpus
        rank = gpu_train_rank(node.gpu)
        key = (0 if near else 1 if near is False else 2,
               0 if fits else 1, -rank,
               math.inf if node.inflight is None else int(node.inflight),
               math.inf if node.capacity is None else float(node.capacity),
               node.name)
        ranked.append(RankedNode(node.name, key, {
            "near_trainer": near, "fits_layout": fits, "gpu": node.gpu, "gpu_rank": rank,
            "inflight": node.inflight, "capacity": node.capacity}))
    ranked.sort(key=lambda r: r.key)
    return ranked, refused


@dataclass(frozen=True)
class Selection:
    """The group a conversion takes (empty: hold, see ``note``), the ranking it
    came from, the refused nodes, and the rollout capacity it takes away."""

    group: tuple[str, ...]
    ranking: tuple[RankedNode, ...]
    refused: Mapping[str, str]
    lost_capacity: float = 0.0
    note: str = ""

    def record(self) -> dict[str, Any]:
        return {"group": list(self.group), "note": self.note, "lost_capacity": self.lost_capacity,
                "ranking": [{"node": r.name, **dict(r.levels)} for r in self.ranking],
                "refused": dict(self.refused)}


def select_group(candidates: Iterable[NodeProfile], needs: TrainerNeeds, *, group_size: int,
                 rollout_nodes: Iterable[str]) -> Selection:
    """The best ``group_size`` convertible nodes by :func:`rank_nodes`.

    ``candidates`` are the convertible nodes doing rollout; ``rollout_nodes`` are all
    nodes doing rollout (convertible or not). Holds when fewer than ``group_size``
    candidates pass the hard conditions, or when the group would leave no node doing
    rollout (the last rollout node is never converted)."""
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    candidates = list(candidates)
    ranked, refused = rank_nodes(candidates, needs)
    if len(ranked) < group_size:
        return Selection((), tuple(ranked), refused,
                         note=f"hold:only {len(ranked)} eligible convertible nodes doing rollout")
    group = tuple(r.name for r in ranked[:group_size])
    if not set(rollout_nodes) - set(group):
        return Selection((), tuple(ranked), refused, note="hold:last_rollout_node")
    by_name = {c.name: c for c in candidates}
    lost = sum(by_name[n].capacity or 0.0 for n in group)
    return Selection(group, tuple(ranked), refused, lost_capacity=lost)


# ---------------------------------------------------------------------------
# L7.3 Scheduler over convertible nodes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConversionDecision:
    to_trainer: tuple[str, ...] = ()
    to_rollout: tuple[str, ...] = ()
    lost_capacity: float = 0.0  # rollout capacity the decision takes away
    capacity_after: float = 0.0
    note: str = ""


@dataclass
class ConversionScheduler:
    """Turns the start/stop decisions of a :class:`TrainerScheduler` (e.g. S1) into
    conversions of ``group_size`` convertible nodes (a multi-GPU trainer needs a
    group). Start converts a group to trainer; Stop converts it back to rollout.
    The ski-rental stop of S1 counts only the time the converted group waits in
    ``TRAINER`` role with an empty backlog, i.e. the time it does neither.

    A start that would convert the last rollout-capable node holds instead. Each
    decision carries the rollout capacity it takes away (negative: gives back)
    and the capacity after it, and is journaled with the pool capacity. The group
    is chosen by :func:`select_group` (five levels, GOAL §7.1 item 14); the
    ranking goes into the journal record."""

    inner: TrainerScheduler
    pool: NodePool
    group_size: int = 1
    record: Callable[..., None] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    # selection (GOAL §7.1 item 14): what a trainer needs, and each node's profile
    needs: TrainerNeeds = TrainerNeeds(precision="bf16")
    profiles: Callable[[str], NodeProfile | None] | None = None
    last_selection: Selection | None = None

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    def lifecycle_state(self) -> LifecycleState:
        if self.pool.in_role(Role.TO_TRAINER) or self.pool.in_role(Role.TO_ROLLOUT):
            return LifecycleState.WARMING  # a transition: the inner scheduler holds
        return LifecycleState.TRAINING if self.pool.in_role(Role.TRAINER) else LifecycleState.STOPPED

    def decide(self, backlog: BacklogStats,
               prices: Mapping[str, float] | None = None) -> ConversionDecision:
        state = self.lifecycle_state()
        self.last_selection = None
        decision = self.inner.decide(backlog, state, prices)
        out = ConversionDecision(capacity_after=self.pool.capacity())
        if isinstance(decision, Start):
            out = self._plan_to_trainer()
        elif isinstance(decision, Stop):
            names = tuple(n.name for n in self.pool.in_role(Role.TRAINER))
            back = sum(self.pool.specs[n].capacity for n in names)
            out = ConversionDecision(to_rollout=names, lost_capacity=-back,
                                     capacity_after=self.pool.capacity() + back,
                                     note=f"stop:{decision.reason.name}")
        if self.record is not None and (out.to_trainer or out.to_rollout or out.note):
            self.record(CONVERSION_DECISION_KIND, tx_id=None,
                        inputs={"lifecycle_state": state.value, "batches": backlog.batches,
                                "samples": backlog.samples, "capacity": self.pool.capacity(),
                                "total_capacity": self.pool.total_capacity()},
                        decision={"to_trainer": list(out.to_trainer),
                                  "to_rollout": list(out.to_rollout),
                                  "lost_capacity": out.lost_capacity,
                                  "capacity_after": out.capacity_after, "note": out.note},
                        **({"selection": self.last_selection.record()}
                           if self.last_selection is not None
                           and (out.to_trainer or out.note.startswith("hold:")) else {}))
        return out

    def _plan_to_trainer(self) -> ConversionDecision:
        candidates = [self._profile(n.name) for n in self.pool.in_role(Role.ROLLOUT)]
        selection = select_group(candidates, self.needs, group_size=self.group_size,
                                 rollout_nodes=self.pool.doing_rollout())
        self.last_selection = selection
        if not selection.group:
            return ConversionDecision(capacity_after=self.pool.capacity(), note=selection.note)
        lost = self.pool.check_conversion(selection.group)
        return ConversionDecision(to_trainer=selection.group, lost_capacity=lost,
                                  capacity_after=self.pool.capacity() - lost)

    def _profile(self, name: str) -> NodeProfile:
        """The node's profile (``profiles``), else one from its spec (declared
        capacity, nothing measured)."""
        profile = self.profiles(name) if self.profiles is not None else None
        if profile is not None:
            return profile
        spec = self.pool.specs[name]
        return NodeProfile(name, spot=spec.spot, precision=spec.precision, capacity=spec.capacity)

    def apply(self, decision: ConversionDecision, buffer: SampleSink, *, version: int | None = None,
              expected_checksum: str | None = None) -> list[ConversionReceipt]:
        """Run the conversions of ``decision`` and keep their events."""
        receipts: list[ConversionReceipt] = []
        for name in decision.to_trainer:
            receipts.append(self.pool.convertible[name].to_trainer(buffer))
        for name in decision.to_rollout:
            if version is None:
                raise ConversionError("to_rollout needs the published version")
            receipts.append(self.pool.convertible[name].to_rollout(
                version, expected_checksum=expected_checksum))
        self.events.extend(r.event() for r in receipts)
        return receipts
