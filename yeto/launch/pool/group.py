"""Group orders: ``n`` machines of one shape on one fabric, all of them or none.

A trainer island of several nodes is useless until every node is up, and every node
that is up bills. A group order therefore succeeds only when all ``n`` nodes are
active; when any node fails, times out or cannot be ordered, every node of the group
that was opened is released (confirmed) before the order reports its failure, and
the bill covers only the time each node was actually open.

The provider contract (:mod:`.provider`) gains one optional method::

    order_group(n, shape, fabric) -> [(slot, ticket), ...]

It *places* the group and returns its tickets. A provider whose one order already
covers ``n`` nodes (a sky multi-node cluster, a Modal ``clustered(size, rdma)``
function) returns one ticket; an inventory returns one ticket per node it picked. It
raises :class:`GroupOrderFailed` (after releasing what it placed) or any error when
nothing usable was placed. Waiting for the group to come up and the all-or-none
release are :func:`order_group`'s, the same for every provider. A provider without
``order_group`` is grouped node by node: :func:`order_group` orders ``n`` single
machines on free slots and applies the same rule.

``fabric`` is the network the nodes must share: ``tcp`` (no RDMA), ``ib``, ``roce``,
``efa`` or ``rdma`` (an RDMA network the platform picks: Modal's clustered RDMA, sky's
``network_tier: best``). A provider declares what it can give in ``fabrics`` (absent:
``("tcp",)``); an order for a fabric it does not declare is refused before anything is
ordered. Any fabric serves a ``tcp`` order.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

FABRICS = ("tcp", "ib", "roce", "efa", "rdma")
RDMA_FABRICS = frozenset({"ib", "roce", "efa", "rdma"})
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_POLL_S = 10.0


class GroupOrderFailed(RuntimeError):
    """The group did not come up whole. ``released``: tickets released (confirmed);
    ``unreleased``: tickets whose release failed (the caller's cleanup must retry them:
    they may still bill); ``members``: every member with its open span."""

    def __init__(self, message: str, *, released: Sequence[str] = (), unreleased: Sequence[str] = (),
                 members: Sequence[GroupMember] = ()):
        super().__init__(message)
        self.released = list(released)
        self.unreleased = list(unreleased)
        self.members = list(members)


@dataclass(frozen=True)
class GroupShape:
    """What every node of the group is: ``gpus_per_node`` cards of ``gpu`` (``region``:
    a placement the provider must honour, None: its own choice)."""

    gpu: str
    gpus_per_node: int
    region: str | None = None

    @classmethod
    def of(cls, value: Any) -> GroupShape:
        """A :class:`GroupShape` from itself or from a ``gpu_spec.ClusterSpec``."""
        if isinstance(value, GroupShape):
            return value
        return cls(str(value.gpu), int(value.gpus_per_node), getattr(value, "region", None))


@dataclass
class GroupMember:
    """One ticket of a group: ``nodes`` machines (more than one: a native group order),
    opened at ``opened_at``, active from ``ready_at``, closed at ``closed_at`` (released by
    the group, or read ended)."""

    slot: Any
    ticket: str
    nodes: int = 1
    opened_at: float = 0.0
    ready_at: float | None = None
    closed_at: float | None = None
    end: str | None = None

    def open_s(self, now: float) -> float:
        """Seconds this ticket was open (billed), up to ``now`` while it still is."""
        end = self.closed_at if self.closed_at is not None else now
        return max(0.0, end - self.opened_at)


@dataclass
class Group:
    """A group that came up whole: its members (``native``: one order covers the nodes)."""

    n: int
    shape: GroupShape
    fabric: str
    members: list[GroupMember] = field(default_factory=list)
    native: bool = False

    @property
    def tickets(self) -> list[str]:
        return [m.ticket for m in self.members]

    def node_hours(self, now: float) -> float:
        """Node hours the group has been open (the bill at a per-node rate)."""
        return sum(m.nodes * m.open_s(now) for m in self.members) / 3600.0


def fabrics_of(provider: Any) -> tuple[str, ...]:
    return tuple(getattr(provider, "fabrics", None) or ("tcp",))


def fabric_problem(provider: Any, fabric: str) -> str | None:
    """Why ``provider`` cannot give ``fabric`` (None: it can)."""
    if fabric not in FABRICS:
        return f"fabric {fabric!r} is unknown (known: {list(FABRICS)})"
    if fabric == "tcp":
        return None
    have = fabrics_of(provider)
    if fabric in have or (fabric == "rdma" and RDMA_FABRICS & set(have)):
        return None
    return f"provider {getattr(provider, 'kind', type(provider).__name__)} gives {list(have)}, not {fabric}"


def free_slots(provider: Any, slots: Sequence[Any] | None) -> list[Any]:
    """The slots of ``slots`` (default: the provider's own ``slots``) that hold nothing now."""
    pool = list(slots if slots is not None else getattr(provider, "slots", ()) or ())
    return [s for s in pool if provider.account_free(s)]


def order_group(provider: Any, n: int, shape: Any, fabric: str = "tcp", *, slots: Sequence[Any] | None = None,
                timeout_s: float = DEFAULT_TIMEOUT_S, poll_s: float = DEFAULT_POLL_S, attempts: int = 1,
                clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                on_event: Callable[[str, dict], None] | None = None) -> Group:
    """Order ``n`` nodes of ``shape`` on ``fabric`` from ``provider``: all active, or all
    released and :class:`GroupOrderFailed`. ``slots``: the slots a node-by-node group may
    use (default: the provider's free slots). ``attempts``: whole-group orders before the
    failure goes up; a provider with offers moves to its next offer after a failed group
    (a native ``group_failed(ticket)`` cools the offer down). ``on_event(name, fields)``
    sees ``group_ordered`` / ``group_ready`` / ``group_failed``."""
    n = int(n)
    if n < 1:
        raise ValueError(f"a group has at least one node, not {n}")
    shape = GroupShape.of(shape)
    problem = fabric_problem(provider, fabric)
    if problem is not None:
        raise GroupOrderFailed(f"group of {n} x {shape.gpus_per_node} {shape.gpu}: {problem}; nothing ordered")
    emit = on_event or (lambda _name, _fields: None)
    last: GroupOrderFailed | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            return _order_once(provider, n, shape, fabric, slots=slots, timeout_s=timeout_s, poll_s=poll_s,
                               clock=clock, sleep=sleep, emit=emit)
        except GroupOrderFailed as error:
            last = error
            if error.unreleased or attempt + 1 >= max(1, int(attempts)):
                raise  # a node that may still bill: no further order on top of it
    raise last  # pragma: no cover - the loop returns or raises


def _order_once(provider: Any, n: int, shape: GroupShape, fabric: str, *, slots, timeout_s, poll_s, clock, sleep,
                emit) -> Group:
    members: list[GroupMember] = []
    native = getattr(provider, "order_group", None)
    try:
        if native is not None:
            placed = list(native(n, shape, fabric))
            per = n // len(placed) if placed and n % len(placed) == 0 else 1
            members = [GroupMember(slot, str(ticket), per, clock()) for slot, ticket in placed]
        else:
            free = free_slots(provider, slots)
            if len(free) < n:
                raise GroupOrderFailed(f"group of {n}: only {len(free)} free slot(s) {free}; nothing ordered")
            for slot in free[:n]:
                ticket = provider.order(slot)  # a raise: the members ordered so far are released below
                members.append(GroupMember(slot, str(ticket), 1, clock()))
    except GroupOrderFailed as error:  # a native order released what it placed (or names what it could not)
        released, unreleased = list(error.released), list(error.unreleased)
        more_released, more_unreleased = release_group(provider, members, clock=clock)
        emit("group_failed", {"reason": str(error), "released": released + more_released,
                              "unreleased": unreleased + more_unreleased})
        raise GroupOrderFailed(str(error), released=released + more_released,
                               unreleased=unreleased + more_unreleased, members=members) from error
    except BaseException as error:  # also an interrupt: every opened member is released, then it goes up
        failure = _fail(provider, members, f"group order of {n} node(s) failed: {error}", clock, emit)
        if not isinstance(error, Exception):
            raise
        raise failure from error
    if sum(m.nodes for m in members) != n:
        raise _fail(provider, members, f"group order of {n} node(s) placed {sum(m.nodes for m in members)}",
                    clock, emit)
    emit("group_ordered", {"n": n, "tickets": [m.ticket for m in members], "fabric": fabric})
    deadline = clock() + float(timeout_s)
    try:
        while True:
            for m in members:
                if m.ready_at is not None:
                    continue
                status = provider.status(m.slot, m.ticket)
                if status.state == "active":
                    m.ready_at = clock()
                elif status.state == "ended":
                    # the reason only: an ended read (a sky cluster stuck in INIT, a STOPPED
                    # one) may still be a machine that bills, so it is released like the rest
                    m.end = str(status.end_reason or "ended")
                    raise GroupOrderFailed(f"{m.ticket} ended ({m.end}) before the group was up")
            if all(m.ready_at is not None for m in members):
                group = Group(n, shape, fabric, members, native=native is not None)
                emit("group_ready", {"n": n, "tickets": group.tickets})
                return group
            if clock() >= deadline:
                waiting = [m.ticket for m in members if m.ready_at is None]
                raise GroupOrderFailed(f"{waiting} not active after {timeout_s:g} s")
            sleep(poll_s)
    except BaseException as error:  # a failed status call, an ended member, a timeout, an interrupt
        failure = _fail(provider, members, f"group of {n}: {error}", clock, emit)
        for m in members:
            group_failed = getattr(provider, "group_failed", None)
            if group_failed is not None:
                try:
                    group_failed(m.ticket)  # the offer it was ordered from cools down
                except Exception:  # noqa: BLE001 - a courtesy to the next order
                    pass
        if not isinstance(error, Exception):
            raise
        raise failure from error


def release_group(provider: Any, members: Sequence[GroupMember], *,
                  clock: Callable[[], float] = time.time) -> tuple[list[str], list[str]]:
    """Release every member that this group has not released yet; ``(released,
    unreleased)``. A member read ended is released too: the read is a reason, not proof
    that the machine is gone (the provider's release confirms that)."""
    released, unreleased = [], []
    for m in members:
        if m.closed_at is not None:
            continue
        try:
            provider.release(m.slot, m.ticket)
        except Exception:  # noqa: BLE001 - reported: the caller's cleanup retries it
            unreleased.append(m.ticket)
            continue
        m.closed_at = clock()
        m.end = m.end or "released"
        released.append(m.ticket)
    return released, unreleased


def release_placed(provider: Any, placed: Sequence[tuple[Any, str]], error: BaseException) -> GroupOrderFailed:
    """For a native ``order_group`` that failed after placing ``placed``: release each
    (confirmed) and return the :class:`GroupOrderFailed` to raise from ``error``; a release
    that fails is named in ``unreleased`` and never hides ``error``."""
    released, unreleased = [], []
    for slot, ticket in placed:
        try:
            provider.release(slot, ticket)
            released.append(str(ticket))
        except Exception:  # noqa: BLE001 - reported, the original error stays the cause
            unreleased.append(str(ticket))
    why = f"group order failed: {error}"
    if unreleased:
        why += f"; release FAILED for {unreleased} (they may still bill: clean them up)"
    return GroupOrderFailed(why, released=released, unreleased=unreleased)


def _fail(provider: Any, members: list[GroupMember], why: str, clock, emit) -> GroupOrderFailed:
    released, unreleased = release_group(provider, members, clock=clock)
    if unreleased:
        why += f"; release FAILED for {unreleased} (they may still bill: clean them up)"
    emit("group_failed", {"reason": why, "released": released, "unreleased": unreleased})
    return GroupOrderFailed(why, released=released, unreleased=unreleased, members=members)
