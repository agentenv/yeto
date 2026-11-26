"""A trainer island that moved or restarted has one store tape folder per launch
(``tapes/trainer/l<k>-<host>-<UTC start>/``). By name, the new launch's folder can sort
before the old one (``l0-computeinstance-...`` < ``l0-modal-...``), and an old launch's tail
can arrive in a later poll. Either way the newest launch's step, outer sweep and fragment
versions are the island's: an older launch never overwrites them. An island with one launch
reads exactly as before."""

from __future__ import annotations

import json
import logging

from test_dashboard_diloco import D, diloco, lines

from yeto.dashboard import store_tapes
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all
from yeto.dashboard.store_tapes import StoreTapes, _launch_ordered, launch_started
from yeto.rl.engine.object_store import DirectoryObjectStore

G5 = D / "g5-fullparam"
OLD = "l0-modal-20261125T224821Z"  # started first, sorts last by name
NEW = "l0-computeinstance-a-20261125T230755Z"


def _trainer0(r: Reducer) -> dict:
    t = next(x for x in diloco(r)["trainers"] if x["id"] == "0")
    return {k: t[k] for k in ("step", "outer", "versions", "hash")}


def _baseline() -> dict:
    r = Reducer(run="g5")
    load_all(r, [str(G5 / "yeto-tape.jsonl"), str(G5 / "fleet.jsonl"), str(G5 / "events")])
    return _trainer0(r)


def _split() -> tuple[list[dict], list[dict]]:
    """Island 0's records, split in time into an older and a newer launch."""
    recs = sorted(lines(G5 / "events" / "g5-v1-a-l0-modal.jsonl"), key=lambda x: x.get("time_unix") or 0)
    half = len(recs) // 2
    return recs[:half], recs[half:]


def _write(root, launch: str, recs: list[dict]) -> None:
    path = root / "tapes" / "trainer" / launch / "00000001.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in recs))


def _others() -> Reducer:
    r = Reducer(run="g5")
    load_all(r, [str(G5 / "yeto-tape.jsonl"), str(G5 / "fleet.jsonl"), str(G5 / "events" / "g5-v1-a-l1-modal.jsonl")])
    return r


def test_a_moved_trainer_shows_its_newest_launch_whatever_the_folder_names(tmp_path):
    want = _baseline()
    old, new = _split()
    _write(tmp_path, OLD, old)
    _write(tmp_path, NEW, new)
    r = _others()
    StoreTapes(DirectoryObjectStore(str(tmp_path))).pump(r)
    assert _trainer0(r) == want
    assert want["step"] is not None and want["versions"]


def test_an_older_launchs_tail_read_in_a_later_poll_does_not_overwrite_the_newest(tmp_path):
    want = _baseline()
    old, new = _split()
    r = _others()
    tapes = StoreTapes(DirectoryObjectStore(str(tmp_path)))
    _write(tmp_path, NEW, new)
    tapes.pump(r)
    assert _trainer0(r) == want
    _write(tmp_path, OLD, old)  # the old launch's records arrive a poll later
    assert tapes.pump(r) > 0
    assert _trainer0(r) == want


def _keys(*paths: str) -> list[str]:
    return [f"tapes/{p}.jsonl" for p in paths]


def test_an_islands_launches_are_read_in_the_order_they_started():
    keys = _keys(f"trainer/{OLD}/00000001", f"trainer/{OLD}/00000002", f"trainer/{NEW}/00000001",
                 "syncer/host-20261125T224000Z/00000001", "s-0/x1/00000001")
    assert _launch_ordered(keys, "tapes/") == _keys(
        "s-0/x1/00000001", "syncer/host-20261125T224000Z/00000001",
        f"trainer/{OLD}/00000001", f"trainer/{OLD}/00000002", f"trainer/{NEW}/00000001")
    assert launch_started(NEW) - launch_started(OLD) == 19 * 60 + 34


def test_launches_started_in_one_second_read_by_name():
    a, b = "l0-b-20261125T224821Z", "l0-a-20261125T224821Z"
    keys = _keys(f"trainer/{a}/00000001", f"trainer/{b}/00000001")
    assert _launch_ordered(keys, "tapes/") == sorted(keys)


def test_a_malformed_stamp_reads_by_name_and_says_so_once(caplog, monkeypatch):
    monkeypatch.setattr(store_tapes, "_UNDATED_LOGGED", set())
    keys = _keys(f"trainer/{OLD}/00000001", "trainer/l0-x-20261399T999999Z/00000001")
    assert launch_started("l0-x-20261399T999999Z") is None
    with caplog.at_level(logging.DEBUG, logger="yeto.dashboard.store_tapes"):
        assert _launch_ordered(keys, "tapes/") == sorted(keys)
        assert _launch_ordered(keys, "tapes/") == sorted(keys)
    assert [r.getMessage() for r in caplog.records].count(
        "store tapes: island trainer has incarnation folders without a start stamp (['l0-x-20261399T999999Z']): "
        "read by name") == 1


def test_one_launch_per_island_reads_exactly_as_before():
    keys = _keys(f"trainer/{NEW}/00000002", f"trainer/{NEW}/00000001", "trainer-x/l0-h-20261125T224821Z/00000001",
                 "s-0/x1/00000001", "s-0/x2/00000001", "notatape")
    assert _launch_ordered(keys, "tapes/") == sorted(keys)
    mixed = keys + _keys(f"trainer/{OLD}/00000001")  # another island's order is untouched by the reorder
    out = _launch_ordered(mixed, "tapes/")
    assert [k for k in out if "/trainer/" not in k] == [k for k in sorted(mixed) if "/trainer/" not in k]
