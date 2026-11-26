"""A trainer island that moved or restarted has one store tape folder per launch
(``tapes/trainer/l<k>-<host>-<UTC start>/``). By name, the new launch's folder can sort
before the old one (``l0-computeinstance-...`` < ``l0-modal-...``), and an old launch's tail
can arrive in a later poll. Either way the newest launch's step, outer sweep and fragment
versions are the island's: an older launch never overwrites them. An island with one launch
reads exactly as before."""

from __future__ import annotations

import json
import logging

import pytest

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


def _with_metrics(r: Reducer) -> list[int]:
    """Island 0's rounds the Trainer rounds table draws: those with a measured metric."""
    return [x["round"] for x in r.round_records(r.islands["0"])
            if any(x.get(k) is not None for k in ("reward", "kl_tr", "kl", "tok_s"))]


class _Unpaged:
    """An object API without paged listings (StoreTapes lists everything every poll)."""

    def __init__(self, objects):
        self._objects = objects

    def __getattr__(self, name):
        if name == "list_page":
            raise AttributeError(name)
        return getattr(self._objects, name)


@pytest.mark.parametrize("paged", [True, False])
def test_the_store_reads_the_older_launch_first_so_its_rounds_and_events_stay(tmp_path, paged):
    """The ingest order itself: the reducer folds a superseded launch's late rounds into no
    table (a stopped launch's late records are not the island's current state), so the old
    launch must be read before the new one. Its rounds keep their metrics in the Trainer
    rounds table, and the event stream ends with the newest launch's records."""
    old, new = _split()
    _write(tmp_path, OLD, old)
    _write(tmp_path, NEW, new)
    r = _others()
    objects = DirectoryObjectStore(str(tmp_path))
    StoreTapes(objects if paged else _Unpaged(objects)).pump(r)
    want = Reducer(run="g5")
    load_all(want, [str(G5 / "yeto-tape.jsonl"), str(G5 / "fleet.jsonl"), str(G5 / "events")])
    rounds = _with_metrics(r)
    old_rounds = sorted({x["rollout_id"] for x in old if x.get("event") == "rl_round_trained"})
    assert old_rounds and set(old_rounds) <= set(rounds)  # the old launch's rounds are in the table
    assert rounds == _with_metrics(want)  # as one launch read in time order
    launches = [e["source"].rsplit("/", 1)[-1] for e in r.events if e["source"].endswith(("/" + OLD, "/" + NEW))]
    assert launches[0] == OLD and launches[-1] == NEW
    assert launches == [OLD] * launches.count(OLD) + [NEW] * launches.count(NEW)  # the stream ends with the newest launch


def _ranked(launches: dict, events: list[dict]) -> list[str]:
    from types import SimpleNamespace

    from yeto.dashboard.topology_diloco import _launch_rank

    rank = _launch_rank(SimpleNamespace(islands={"0": {"launches": launches}}), "0")
    return [e["id"] for e in sorted(events, key=rank)]


def _ev(eid: str, ts: float, launch: str | None = None) -> dict:
    return {"id": eid, "ts": ts, "source": f"store:trainer/{launch}" if launch else "events/l0.jsonl"}


def test_a_record_of_no_launch_goes_with_the_newest_launch_started_by_its_time():
    """A launcher-side record (the echoed tape, no launch folder) between the two launches is
    the old launch's; one after the new launch started is the new one's; one before any
    launch goes with the first."""
    t0, t1 = launch_started(OLD), launch_started(NEW)
    launches = {OLD: {"first_ts": t0 + 5}, NEW: {"first_ts": t1 + 5}}
    events = [_ev("new", t1 + 50, NEW), _ev("late", t1 + 60), _ev("mid", t0 + 60), _ev("old", t0 + 50, OLD),
              _ev("early", t0 - 100)]
    assert _ranked(launches, events) == ["mid", "old", "early", "new", "late"]


def test_a_launch_is_ordered_by_its_folder_stamp_before_its_first_record():
    """A restarted trainer uploads its whole local tape again under its new launch, so the new
    folder's first record can be older than the old folder's: the folder stamp decides. A
    folder without a stamp falls back to its first record."""
    t0, t1 = launch_started(OLD), launch_started(NEW)
    launches = {OLD: {"first_ts": t1 + 30}, NEW: {"first_ts": t0 - 30}}
    events = [_ev("new", t1 + 50, NEW), _ev("old", t0 + 50, OLD)]
    assert _ranked(launches, events) == ["old", "new"]
    undated = "l0-x-host"
    launches = {OLD: {"first_ts": t0 + 5}, undated: {"first_ts": t0 - 30}}
    assert _ranked(launches, [_ev("old", t0 + 50, OLD), _ev("x", t1, undated)]) == ["x", "old"]


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


def test_the_trainers_outer_sweep_and_the_syncers_global_step_are_told_apart(tmp_path):
    """The trainer card counts complete sweeps it applied (every fragment merged at least N
    times); the syncer card counts global steps (one fragment merge each) and names the sweep
    its last merge is in. At 400 px, or where the long forms would not fit beside each other
    (a long run's "of 1510 · 120 merges"), both keep a short form."""
    from test_dashboard_diloco import _smoke, load

    wide = _smoke(tmp_path, load("run-b"))
    texts = wide["texts"]
    assert "outer sweep 6" in texts and "global step 24" in texts and not any(t.startswith("outer v") for t in texts)
    assert "24 merges · in sweep 6" in texts  # P 4: global step 24 closes sweep 6
    assert "Local step / outer sweep" in wide["nodes"]
    assert "outer sweep: every fragment merged at least that many times" in wide["nodes"]
    assert "Syncer global step / local steps" in wide["kpis"] and "outer sweep 6" in wide["kpis"]
    (tmp_path / "n").mkdir()
    narrow = _smoke(tmp_path / "n", load("run-b"), width=400)["texts"]
    assert "sweep 6" in narrow and "step 24" in narrow and "24 merges · sweep 6" in narrow
    assert "outer sweep 6" not in narrow and "global step 24" not in narrow
    (tmp_path / "s").mkdir()
    crowded = _smoke(tmp_path / "s", load("sft-nemotron"))["texts"]
    assert "step 210" in crowded and "global step 210" not in crowded  # the long run's card keeps "step"
