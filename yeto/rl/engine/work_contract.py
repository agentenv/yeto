"""WorkContract: the one task-dispense and membership contract of yeto rollout.

Two implementations hand rows to rollout islands:

* :mod:`yeto.rl.engine.store_dispense` (``R2StoreDispense``): a ledger in the
  run's object store, used by samplers that share nothing but the bucket (the
  R2 data path, async GRPO). It is the **reference implementation**;
* ``syncer/src/dispense.rs`` + ``syncer/src/members.rs``: the syncer's task
  dispenser and member table (the syncer data path, decoupled DiLoCo and spot
  samplers), reached over the TASK_* and MEMBER_RETIRE frames
  (:mod:`yeto.rl.engine.task_dispense`).

Both must pass the same conformance cases
(``tests/fixtures/work_contract_cases.json``; the Python driver is
``tests/test_work_contract.py``, the Rust one ``syncer/src/work_contract.rs``).
This module states the contract; :class:`WorkContract` is the operation set a
controller or a sampler runtime may rely on, whichever data path it runs on.

Terms
-----

*Item*: one unit of work, identified by its **slot** (store ledger) or
**position** (syncer); the two numberings agree for one plain stream: slot ``p``
of pass ``k`` is row ``order(k)[p mod rows]``. An item is never identified by
its row: with ``passes > 1`` two items of one claim may name the same row
(an epoch boundary of a small data set). A consumer that names things after
the row (a prompt id, a recorder session, a buffer group key) must make the
repeat distinct: the sampler round does
(``yeto.rl.engine.engine_sampler.unique_prompt_ids``: ``row<r>-p<slot>``).

*Member*: one island process, ``(holder, incarnation)``. The holder is stable
across restarts of the island (sampler id, learner id); the incarnation is new
for every process. A member is **live** while it renews its lease, **retired**
once it was superseded, expired, fenced or retired by the controller. Retired is
final for that incarnation.

Operations (pre-condition, effect, idempotency, concurrency)
------------------------------------------------------------

``join(member)``
    Admit a process. A new incarnation of a holder **supersedes** the current
    one at once: the old incarnation is retired and its open claims are
    released (not counted toward ``max_redo``). Joining the same incarnation
    again is a reconnect and keeps its claims. A retired incarnation cannot
    join (``fenced``).

``claim(member, n) -> items | WAIT | END | FENCED``
    Up to ``n`` items. Returned items (released, requeued, expired, fenced)
    come before never-claimed ones. Every item is held by at most one live
    member at a time (store: exclusive create of the attempt; syncer: one
    mutex). ``WAIT``: nothing now, but items held by live members may come
    back. ``END`` (data end): every item of every pass is settled (trained,
    dropped or exhausted) and none is held. ``FENCED``: the member is retired;
    it takes nothing more and must drop what it holds.

``done(member, items)`` (``trained``) and ``drop(member, items)``
    Settle items for good: they are never handed out again. Settling twice is
    a no-op for the store (create-once keys) and, for the syncer, a resent
    request with the same ``seq`` gets the same answer; an item the member no
    longer holds is refused / ignored ("not held"), never settled by it.

``fail(member, items)`` (``infra`` / ``requeue``)
    The item ended without a result through no fault of the policy (trials
    failed on infrastructure, a stale group). Counted toward ``max_redo``; the
    item goes back to the front, or is settled ``exhausted`` when its counted
    failures exceed ``max_redo``.

``release(member, items)``
    Give items back unconsumed (closing, unwritten). Not counted.

``beat(member)``
    Renew the lease. A member that renews nothing for ``lease_s`` **expires**:
    its open claims end ``expired`` at the next operation of any member,
    **counted** toward ``max_redo`` (a task that kills its sampler must not
    loop for ever), and the member is retired. Before another member ends the
    claim, the holder itself may still settle it (the store reads the time
    only at scans); the syncer ends it at the very next request, the holder's
    own included. Neither ever hands an item to two live members.

``fence(member)``
    The data plane learns that the member's machine was replaced (store: its
    incarnation lease names another incarnation, ``fence_shard``). Its open
    claims are released at the next scan of any member, not counted, without
    waiting for the lease; the member is fenced on its next operation.

``retire(member, reason) -> released``
    The controller removes a member now (scale-down, evict, preemption seen
    by the provider). Pre-condition: none (an unknown or already retired
    member is a no-op that returns 0; ``incarnation=None`` retires every
    incarnation of the holder that holds a claim, store; the current one,
    syncer). Effect, at once: the member is retired on every channel, every
    open claim is **released** (back to the front, not counted, so the next
    claim of any member takes it; nothing waits for in-flight work: a group
    the retired member still finishes is refused by ``still_holds`` / the
    ``fenced`` answer and by the incarnation gate). Idempotent. A graceful
    drain is the island's own business: stop it (its ``close()`` releases what
    it holds), then retire it.

``still_holds(member, item) -> bool``
    The member may still deliver the item's result: it is live, the item's
    claim is its own and not ended. A sampler checks it before it writes a
    group.

Totals (``trained``, ``dropped``, ``exhausted``) count items; both
implementations report them.

Allowed differences (not part of the contract)
----------------------------------------------

* The order of returned items among themselves (store: ascending slot;
  syncer: queue front). Both put them before fresh items.
* The store's lease is per claim (each claim object is beaten), the syncer's
  per member (any request renews all its blocks). A member that beats keeps
  everything alive in both.
* Expiry of a holder nobody else observed (above).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

ClaimStatus = Literal["items", "wait", "end", "fenced"]
SETTLED_KINDS = ("trained", "dropped", "exhausted")
ITEM_END_KINDS = ("trained", "dropped", "infra", "release", "expired")
COUNTED_ENDS = ("infra", "expired")  # toward max_redo
RETIRE_REASONS = ("scale_down", "evicted", "preempted", "operator")


@dataclass(frozen=True)
class Member:
    """One island process."""

    holder: str
    incarnation: str


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    items: tuple[int, ...] = ()


@runtime_checkable
class WorkContract(Protocol):
    """The operations of the contract (module doc) on one data plane."""

    def join(self, member: Member) -> int:
        """Admit ``member``; the open claims released from the incarnation it supersedes."""

    def claim(self, member: Member, n: int) -> ClaimResult: ...

    def done(self, member: Member, items: Sequence[int]) -> int:
        """Settle ``trained``; the number actually settled by this call."""

    def drop(self, member: Member, items: Sequence[int]) -> int: ...

    def fail(self, member: Member, items: Sequence[int]) -> int: ...

    def release(self, member: Member, items: Sequence[int]) -> int: ...

    def beat(self, member: Member) -> None: ...

    def still_holds(self, member: Member, item: int) -> bool: ...

    def retire(self, holder: str, incarnation: str | None = None, reason: str = "scale_down") -> int:
        """Retire now; the number of open claims released."""

    def totals(self) -> dict[str, int]:
        """``{"trained", "dropped", "exhausted"}`` item counts."""
