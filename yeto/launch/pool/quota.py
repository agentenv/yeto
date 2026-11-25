"""Account-level quotas shared by every pool of a run or fleet (capacity broker, 6.4).

Some limits belong to an account, not to a pool: the boat account's active sandboxes
and starts per minute / hour, the National Compute accounts (one machine each), a
Daytona organisation's sandbox quota. Every pool that draws on one gets a share, and
the shares never add up to more than the account holds. Pure functions only; the
callers write the shares where the pools read them (``rollout.pools.with_boat_shares``
for boat; the broker caps a pool's node count with :func:`split_nodes` /
:func:`split_accounts`).

* :func:`split` - one per-unit value per pool (a sampler process holds its share
  alone): the limit left after the values a pool sets itself, in proportion to each
  pool's weight; every unit gets at least its weight, or :class:`QuotaShort`.
* :func:`split_nodes` - a node count per pool (Daytona): floors first, nodes a pool
  holds now are kept, the rest round robin up to each pool's want.
* :func:`split_accounts` - disjoint account sets per pool (National Compute): the
  protected accounts (``pool_run.protected_accounts`` of every pool config: 2 and 3
  always, plus each config's ``protected_accounts``) are never granted, accounts a pool
  holds now stay with it.
* :func:`transition` - adding or removing a pool at run time: processes started
  under the old split keep their old share until they restart, so a pool that grows
  gets only the headroom the account has now; the sum never passes the limit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

class QuotaShort(ValueError):  # noqa: N818 - a ValueError, as the run config's other refusals
    """The account cannot give every unit its minimum. ``need``: what the free units need at
    least; ``left``: what the limit leaves after the values pools set themselves (``used``)."""

    def __init__(self, resource: str, limit: int, need: int, used: int, left: int) -> None:
        self.resource, self.limit, self.need, self.used, self.left = resource, limit, need, used, left
        super().__init__(f"{resource}: the account holds {limit}, pools set {used} themselves and the "
                         f"other pools need at least {need}")


@dataclass(frozen=True)
class Claim:
    """One pool's draw on a per-unit limit: ``units`` processes each hold a share alone;
    ``weight`` is both the per-unit minimum and the proportional weight; ``fixed`` is a
    per-unit value the pool sets itself (kept, and counted against the limit)."""

    units: int
    weight: int = 1
    fixed: int | None = None


def split(resource: str, limit: int, claims: Mapping[str, Claim]) -> dict[str, int]:
    """``{pool: per-unit share}`` for every claim without a ``fixed`` value: ``left * weight
    // need`` where ``left`` is the limit after the fixed values and ``need`` the free units'
    total weight. ``sum(units * share) + sum(units * fixed) <= limit`` always. Raises
    :class:`QuotaShort` when the free units cannot each get their weight, or the fixed
    values alone pass the limit."""
    fixed = {n: c for n, c in claims.items() if c.fixed is not None}
    free = {n: c for n, c in claims.items() if c.fixed is None and c.units > 0}
    used = sum(int(c.fixed) * c.units for c in fixed.values())  # type: ignore[arg-type]
    left = int(limit) - used
    need = sum(c.units * c.weight for c in free.values())
    if left < need or left < 0:
        raise QuotaShort(resource, int(limit), need, used, left)
    return {n: left * c.weight // max(1, need) for n, c in free.items()}


@dataclass(frozen=True)
class NodeWant:
    """A pool's draw on a node-count quota: ``want`` nodes, at least ``floor``; ``held``
    nodes it runs now (kept: a quota split never takes a running machine away)."""

    want: int
    floor: int = 0
    held: int = 0


@dataclass
class NodeShares:
    grants: dict[str, int] = field(default_factory=dict)
    short: dict[str, int] = field(default_factory=dict)  # want - grant, > 0 only

    def total(self) -> int:
        return sum(self.grants.values())


def split_nodes(resource: str, limit: int, wants: Mapping[str, NodeWant]) -> NodeShares:
    """``grants`` per pool with ``sum <= limit``: what every pool holds now first (more held
    than the limit raises :class:`QuotaShort`: the account is already over), then the floors,
    then one node at a time round robin (pool name order) up to each want."""
    held = {n: min(max(0, w.held), max(0, w.want, w.held)) for n, w in wants.items()}
    if sum(held.values()) > limit:
        raise QuotaShort(resource, int(limit), sum(held.values()), 0, int(limit))
    grants = dict(held)
    left = int(limit) - sum(grants.values())
    for goal in ("floor", "want"):
        moved = True
        while left > 0 and moved:
            moved = False
            for name in sorted(wants):
                if left <= 0:
                    break
                if grants[name] < max(0, getattr(wants[name], goal)):
                    grants[name] += 1
                    left -= 1
                    moved = True
    short = {n: w.want - grants[n] for n, w in wants.items() if w.want > grants[n]}
    return NodeShares(grants, short)


@dataclass
class AccountShares:
    grants: dict[str, tuple[int, ...]] = field(default_factory=dict)
    short: dict[str, int] = field(default_factory=dict)
    free: tuple[int, ...] = ()


def split_accounts(allowed: Iterable[int], wants: Mapping[str, int], *,
                   held: Mapping[str, Iterable[int]] | None = None,
                   configs: Mapping[str, Mapping[str, object]] | None = None) -> AccountShares:
    """Disjoint account sets per pool (one machine per account): an account protected by
    any pool's config (``configs``: pool -> pool config; ``pool_run.protected_accounts``,
    so 2 and 3 with no config at all) is never granted; accounts a pool holds now stay
    with it (an account two pools hold, or a held account outside ``allowed`` or
    protected, raises ValueError); the rest go round robin (pool name order, lowest
    account first) up to each pool's want."""
    from yeto.launch.nc.pool_run import protected_accounts  # noqa: PLC0415

    protected: set[int] = set(protected_accounts({}))
    for cfg in (configs or {}).values():
        protected |= protected_accounts(dict(cfg))
    guard = protected
    pool = sorted({int(a) for a in allowed} - guard)
    grants: dict[str, list[int]] = {n: [] for n in wants}
    owner: dict[int, str] = {}
    for name, accounts in sorted((held or {}).items()):
        for a in sorted({int(x) for x in accounts}):
            if a in guard:
                raise ValueError(f"pool {name!r} holds protected account {a}")
            if a not in pool:
                raise ValueError(f"pool {name!r} holds account {a}, which is not allowed")
            if a in owner:
                raise ValueError(f"account {a} is held by pools {owner[a]!r} and {name!r}")
            owner[a] = name
            grants.setdefault(name, []).append(a)
    free = [a for a in pool if a not in owner]
    moved = True
    while free and moved:
        moved = False
        for name in sorted(wants):
            if free and len(grants[name]) < max(0, int(wants[name])):
                grants[name].append(free.pop(0))
                moved = True
    short = {n: int(w) - len(grants[n]) for n, w in wants.items() if int(w) > len(grants[n])}
    return AccountShares({n: tuple(sorted(a)) for n, a in grants.items()}, short, tuple(free))


def transition(limit: int, target: Mapping[str, int], held: Mapping[str, int]) -> dict[str, int]:
    """What each pool may hold now while the account moves from the old split to a new one
    (``target``, e.g. after a pool was added or removed). ``held``: the totals the pools'
    running processes hold now (a process keeps the share it started with until it
    restarts; a removed pool, absent from ``target``, holds its share until its processes
    end). A pool over its new target gets its target (it drops there as its processes
    restart); the headroom ``limit - sum(held)`` goes to the pools under their target, in
    proportion to what they miss. So ``sum(max(result, held)) + removed pools' held <=
    limit`` holds at every step."""
    grant = {n: max(0, int(t)) for n, t in target.items()}
    have = {n: max(0, int(v)) for n, v in held.items()}
    headroom = max(0, int(limit) - sum(have.values()))
    missing = {n: t - have.get(n, 0) for n, t in grant.items() if t > have.get(n, 0)}
    total_missing = sum(missing.values())
    for n, m in missing.items():
        grant[n] = have.get(n, 0) + min(m, headroom * m // total_missing)
    return grant
