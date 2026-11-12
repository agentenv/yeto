"""Invariant of a store read outage: no sampler leaves, no task in flight is released.

A sampler's reads of the shared store are polls (the incarnation lease, the
handoff record, the dispenser ledger, ``LATEST``, its own buffer index); its
lease renewals and group writes are writes. While every read (GET / LIST /
HEAD) fails and writes still work, for up to ``max_outage_s`` (default
:data:`DEFAULT_MAX_READ_OUTAGE_S`, 10 minutes):

* no sampler ends on it (no ``rl_sampler_leave`` with reason ``INFRA_FAILED``);
* no task in flight goes back to the dispenser: no attempt that was open when
  the outage began is ended ``release`` / ``expired`` / ``infra`` during it (the
  renewals are writes, so the claims stay alive; a group that is ready is held
  back, ``rl_sampler_write_deferred``, and written once the store answers);
* afterwards every sampler goes on (a claim, a settled group or a round after
  the outage).

A fence or a retire by the controller during the outage releases claims on
purpose; pass the shards it named as ``fenced`` and they are left out (their
tapes; their slots too with ``holders``, ``{slot: sampler}`` of the claims open
when the outage began).

An outage of reads **and** writes is a different case: the renewals fail too,
so a claim runs out once the outage is longer than its dispense lease
(``prompts.dispense.lease_s``); the check then only holds for outages shorter
than that lease (``writes_failed=True`` leaves the release rule to the caller's
``max_outage_s``).

:func:`read_outage_problems` is the check (a list of problems, empty when the
invariant holds); a fault simulation calls it after every outage it injects::

    problems = read_outage_problems(tape_records, start=t0, end=t1,
                                    ledger_before=keys0, ledger_after=keys1)

``tape_records``: the samplers' tape records (``time_unix``, ``event``,
``sampler_id``); ``ledger_before`` / ``ledger_after``: the dispenser listing
(``objects.list_meta(namespace + "/")[0]`` or plain keys) at the start of the
outage and after the recovery.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_MAX_READ_OUTAGE_S = 600.0
DEFAULT_GRACE_S = 5.0  # after the outage: polls that were already failing may report late
LEAVE_EVENT = "rl_sampler_leave"
# the sampler gave a task in flight back, or found it taken
RELEASE_EVENTS = ("rl_dispense_release", "rl_dispense_requeue", "rl_dispense_lost", "rl_dispense_expired")
PROGRESS_EVENTS = ("rl_dispense_claim", "rl_dispense_settle", "rl_sampler_engine_round", "rl_sampler_round",
                   "rl_agentic_group")
END_EVENTS = ("rl_sampler_data_end", "rl_sampler_data_ended", "rl_sampler_stopped")


def _keys(listing: Iterable[Any]) -> list[tuple[str, float]]:
    return [(item, 0.0) if isinstance(item, str) else (str(item[0]), float(item[1])) for item in listing]


def released_in_flight(ledger_before: Iterable[Any], ledger_after: Iterable[Any],
                       namespace: str = "task-dispense") -> dict[int, tuple[int, str]]:
    """``{slot: (attempt, kind)}`` of the attempts open in ``ledger_before`` that
    ``ledger_after`` shows ended without a trained group."""
    from yeto.rl.engine.store_dispense import parse_ledger  # noqa: PLC0415

    before, after = parse_ledger(_keys(ledger_before), namespace), parse_ledger(_keys(ledger_after), namespace)
    out = {}
    for slot, state in sorted(before.items()):
        attempt = state.open_attempt()
        if attempt is None:
            continue
        kind = after.get(slot, state).fails.get(attempt)
        if kind is not None:
            out[slot] = (attempt, kind)
    return out


def read_outage_problems(records: Iterable[Mapping[str, Any]], *, start: float, end: float,
                         ledger_before: Iterable[Any] | None = None, ledger_after: Iterable[Any] | None = None,
                         namespace: str = "task-dispense", max_outage_s: float = DEFAULT_MAX_READ_OUTAGE_S,
                         grace_s: float = DEFAULT_GRACE_S, fenced: Iterable[str] = (),
                         holders: Mapping[int, str] | None = None, expect_progress: bool = True) -> list[str]:
    """The ways the tapes / the ledger break the invariant (module doc) for the
    outage ``start .. end`` (``time_unix``); [] when it holds. An outage longer
    than ``max_outage_s`` is outside the invariant: [] as well."""
    if float(end) - float(start) > float(max_outage_s):
        return []
    skip = {str(s) for s in fenced}
    records = [r for r in records if str(r.get("sampler_id")) not in skip]
    window = [r for r in records if float(start) <= float(r.get("time_unix") or 0.0) <= float(end) + float(grace_s)]
    problems = []
    for r in window:
        who = r.get("sampler_id")
        if r.get("event") == LEAVE_EVENT and str(r.get("reason")) == "INFRA_FAILED":
            problems.append(f"sampler {who} left INFRA_FAILED during the outage: {str(r.get('error'))[:200]}")
        elif r.get("event") in RELEASE_EVENTS:
            problems.append(f"sampler {who} gave up slot {r.get('slot')} attempt {r.get('attempt')} during the "
                            f"outage ({r.get('event')}, {r.get('state') or r.get('reason')})")
    if ledger_before is not None and ledger_after is not None:
        for slot, (attempt, kind) in released_in_flight(ledger_before, ledger_after, namespace).items():
            if str((holders or {}).get(slot)) in skip:
                continue
            problems.append(f"slot {slot} attempt {attempt} was in flight when the outage began and ended {kind}")
    if expect_progress:
        seen_before = {str(r.get("sampler_id")) for r in records if float(r.get("time_unix") or 0.0) < float(start)}
        ended = {str(r.get("sampler_id")) for r in records if r.get("event") in END_EVENTS}
        after = {str(r.get("sampler_id")) for r in records
                 if float(r.get("time_unix") or 0.0) > float(end) and r.get("event") in PROGRESS_EVENTS}
        for who in sorted(seen_before - after - ended):
            problems.append(f"sampler {who} did nothing after the outage (no claim, group or round on its tape)")
    return problems
