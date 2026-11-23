"""A live island that barely contributes to the syncer's merges (``diloco_island_starved``).

Derived from the syncer tape alone: every merge record names the trainers it
expected, those that responded, and those it waited its grace for
(``missed_grace``). An island that keeps missing merges while it is live trains on
stale fragments; the usual cause is a slow downlink (the syncer broadcasts each
merged fragment in f32), for example a syncer in another region than the
island. The rule says how often the island responded, how long the syncer
waited for it, and how far its fragment versions are behind.

The numbers below are this module's defaults; ``alerts_diloco.evaluate`` calls
:func:`starved` with the merge rows of ``topology_diloco.merge_rows``.
"""

from __future__ import annotations

from typing import Any

from .alerts import _alert

WINDOW_MERGES = 8  # the last merges an island was expected in
MIN_MERGES = 4  # judged once it was expected in at least this many
STARVED_SHARE = 0.25  # responded to at most this share of them


def _ids(values) -> set[str]:
    out = set()
    for v in values or []:
        out.add(str(v.get("id") if isinstance(v, dict) else v))
    return out


def participation(merges: list[dict], iid: str, window: int = WINDOW_MERGES) -> dict[str, Any]:
    """The island's share of its last ``window`` merges: expected, responded,
    waited out (``missed_grace``) and the grace the syncer waited for it."""
    mine = []
    for m in merges:
        rec = m.get("rec") or {}
        expected = _ids(rec.get("expected"))
        if expected and iid not in expected:
            continue
        mine.append(rec)
    mine = mine[-window:]
    responded = [r for r in mine if iid in _ids(r.get("responded") or r.get("responders"))]
    missed = [r for r in mine if iid in _ids(r.get("missed_grace"))]
    waited = sum(float(r.get("grace_ms") or 0) / 1000 for r in missed)
    return {"expected": len(mine), "responded": len(responded), "missed_grace": len(missed), "waited_s": waited}


def _versions_behind(t: dict, versions: list[int]) -> str:
    vec = t.get("versions")
    if not vec or not versions:
        return ""
    return f"; fragment versions {list(vec)} vs syncer {list(versions)}"


def starved(d: dict, now: float, merges: list[dict]) -> list[dict]:
    if not d.get("running") or d.get("kind") == "sft":
        return []
    out = []
    for t in d.get("trainers") or []:
        if not t.get("live"):
            continue  # trainer_lost / roster cover an island that is down
        iid = str(t["id"])
        p = participation(merges, iid)
        if p["expected"] < MIN_MERGES or p["responded"] > STARVED_SHARE * p["expected"]:
            continue
        sev = 0 if p["responded"] == 0 and p["missed_grace"] == p["expected"] else 1
        bcast = t.get("last_bcast") or {}
        wire = ""
        if bcast.get("bytes"):
            wire = f"; last BCAST {bcast['bytes'] / 1e6:.0f} MB"
            if bcast.get("queue_s") is not None:
                wire += f", queued {bcast['queue_s']:.0f}s"
        waited = (f"; the syncer waited out its grace for it {p['missed_grace']} time(s), {p['waited_s']:.0f}s in all"
                  if p["missed_grace"] else "")
        out.append(_alert(
            sev, "diloco_island_starved",
            f"island {iid} starved: responded to {p['responded']} of its last {p['expected']} merges",
            f"it trains on stale fragments{waited}{_versions_behind(t, d.get('versions') or [])}{wire}. "
            "The syncer broadcasts every merged fragment in f32: on a slow link (another region than the "
            "syncer) an island cannot fetch each version in time. Pin the islands and the syncer to one "
            "region (@region, --syncer-region) or use --rl-sync-bcast-delta-full-every",
            island=iid))
    return out
