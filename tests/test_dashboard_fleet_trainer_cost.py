"""A fleet launched island by island whose trainer island has no tape on the dashboard (a
backend that does not upload its trainer tape, an island added or moved that never wrote one):
its launch records still bill it in the cost table, as run.budget does, so "fleet spend" never
reads "trainers $n/a"; when nothing bills a trainer, the reason is said."""

from __future__ import annotations

import re

import pytest
import test_dashboard_fleet_na as fleet_na
from test_dashboard_fleet_islands import fleet_records, plan_doc
from test_dashboard_trainer_islands import _store

from yeto.dashboard import cost, fleet_islands
from yeto.dashboard.reducer import Reducer
from yeto.rl.engine.object_store import DirectoryObjectStore


def _fleet(tmp_path, *, spend: dict | None, no_tape: tuple[int, ...] = (), per_island: bool = True,
           edit=None) -> Reducer:
    """The two-island fleet fixture (islands i0, i1, and i2 added then removed by the desired
    state), without the trainer tape of the islands in ``no_tape``; ``spend``: each island's
    (hours, usd, fallback) from its launch records; ``edit(rec)``: a record to feed instead
    (None: dropped)."""
    r = Reducer(run="fleet-qa")
    for rec in fleet_records():
        if rec.get("island_id") in no_tape:
            continue
        rec = rec if edit is None else edit(rec)
        if rec is not None:
            r.feed(rec)
    plan = plan_doc()
    assert r.load_fleet_plan(dict(plan, launch="per_island") if per_island else plan, "test")
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path)))))
    if spend is not None:
        r.trainer_spend = {"at": r.max_ts, "islands": spend}
    return r


def _rows(r: Reducer) -> tuple[dict, dict]:
    view = cost.cost_view(r, r.max_ts + 30)
    return {row["id"]: row for row in view["islands"]}, view


SPEND = {0: (1.0, 3.0, False), 1: (0.5, 2.0, False), 2: (0.25, 0.7, True)}


def test_a_trainer_island_without_a_tape_is_billed_by_its_launch_records(tmp_path):
    r = _fleet(tmp_path, spend=SPEND, no_tape=(1,))
    rows, view = _rows(r)
    for k, (hours, usd, _fallback) in SPEND.items():
        row = rows[str(k)]
        assert row["kind"] == "trainer" and row["island"] == k and row["hours"] == hours
        assert row["cost_usd"] == pytest.approx(usd) and row["priced"] and "launch records" in row["rate_source"]
    assert "declared price" in rows["2"]["rate_source"]  # its hours without a price
    assert view["fleet"]["trainers_usd"] == pytest.approx(5.7) and "trainers_note" not in view["fleet"]


def test_record_only_trainer_rows_do_not_move_the_syncers_clock(tmp_path):
    with_tape, _ = _rows(_fleet(tmp_path / "a", spend={0: SPEND[0]}, no_tape=(1,)))
    added, _ = _rows(_fleet(tmp_path / "b", spend=SPEND, no_tape=(1,)))
    assert added["syncer"]["hours"] == pytest.approx(with_tape["syncer"]["hours"])
    assert added["syncer"]["cost_usd"] == pytest.approx(with_tape["syncer"]["cost_usd"])


def _later_without_ready(rec):
    """Island 1's tape without its ``island_ready`` (no cost row of its own: no ready stamp or
    cloud), 15 minutes later than island 0's."""
    if rec.get("island_id") != 1:
        return rec
    return None if rec["event"] == "island_ready" else dict(rec, time_unix=rec["time_unix"] + 900)


def test_a_record_row_over_a_tape_without_island_ready_leaves_the_syncers_clock(tmp_path):
    """Island 1's trainer has a tape here (later than island 0's) but no cost row: its row comes
    from its launch records, and its tape's events do not stretch the syncer's hours."""
    r = _fleet(tmp_path / "a", spend=SPEND, edit=_later_without_ready)
    assert r.islands["1"]["last_event_ts"] > r.islands["0"]["last_event_ts"]  # it would move the clock
    rows, view = _rows(r)
    assert rows["1"]["kind"] == "trainer" and rows["1"]["hours"] == 0.5 and "launch records" in rows["1"]["rate_source"]
    island0_only, _ = _rows(_fleet(tmp_path / "b", spend=SPEND, no_tape=(1,)))
    assert rows["syncer"]["hours"] == pytest.approx(island0_only["syncer"]["hours"])
    assert view["fleet"]["trainers_usd"] == pytest.approx(5.7) and "trainers_uncounted" not in view["fleet"]


def _ready_only(rec):
    """Island 1's tape with its ``island_ready`` only (no event names it a trainer yet)."""
    return rec if rec.get("island_id") != 1 or rec["event"] == "island_ready" else None


def _no_ready(rec):
    """Island 1's tape without its ``island_ready``: its events, nothing to bill its hours from."""
    return None if rec.get("island_id") == 1 and rec["event"] == "island_ready" else rec


def test_a_trainer_island_known_by_its_ready_stamp_only_is_billed_as_a_trainer(tmp_path):
    rows, view = _rows(_fleet(tmp_path, spend=None, per_island=False, edit=_ready_only))
    assert rows["1"]["kind"] == "trainer" and "fleet plan trainer_usd_per_h" in rows["1"]["rate_source"]
    assert view["fleet"]["trainers_usd"] == pytest.approx(rows["0"]["cost_usd"] + rows["1"]["cost_usd"])


def test_a_trainer_tape_without_hours_says_no_hours_and_the_total_says_it_is_low(tmp_path):
    r = _fleet(tmp_path, spend=None, per_island=False, edit=_no_ready)
    rows, view = _rows(r)
    fl = view["fleet"]
    assert "1" not in rows and fl["trainers_usd"] == pytest.approx(rows["0"]["cost_usd"])
    assert fl["trainers_uncounted"] == 2
    assert fl["trainers_note"] == ("island 1: no hours yet (its trainer tape has no ready stamp or launch record to "
                                   "bill from); island 2: no trainer tape or launch record here")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", fleet_na._page(tmp_path, r)["costBox"]))
    assert re.search(r"\+ trainers \$\d+\.\d\d \(2 trainer islands not counted: island 1: no hours yet", text)
    assert "no trainer tape or launch record here; the total is low)" in text


def _launch_not_ready(rec):
    """Island 1 launched (``island_launch``: its cloud and card) and never ready: a priced
    trainer row with no billed hours."""
    if rec.get("island_id") == 1 and rec["event"] == "island_ready":
        return dict(rec, event="island_launch")
    return rec


def test_a_priced_trainer_row_without_hours_says_no_hours(tmp_path):
    rows, view = _rows(_fleet(tmp_path, spend=None, per_island=False, edit=_launch_not_ready))
    assert rows["1"]["kind"] == "trainer" and rows["1"]["hours"] is None and rows["1"]["cost_usd"] is None
    assert rows["1"]["rate_usd_h"] is not None  # priced by the plan
    fl = view["fleet"]
    assert fl["trainers_usd"] == pytest.approx(rows["0"]["cost_usd"]) and fl["trainers_uncounted"] == 2
    assert fl["trainers_note"].startswith("island 1: no hours yet")


def test_a_sampler_row_is_never_taken_for_a_trainer_island(tmp_path):
    """The fallback to the learner id (a trainer the tape does not name yet) skips a row whose
    island is a sampler that happens to carry that id."""
    def sampler_one(rec):
        rec = _ready_only(rec)
        if rec is not None and rec.get("sampler_id") == "qa-a-i1-0":
            rec = dict(rec, sampler_id="1")
        return rec

    r = _fleet(tmp_path, spend=None, per_island=False, edit=sampler_one)
    r.feed({"event": "island_ready", "time_unix": r.max_ts, "island": "sampler-one", "island_id": 1,
            "cloud": "modal", "gpu": "L40S", "gpus": 1})
    assert r.async_.roles.get("1") == "sampler"
    rows, view = _rows(r)
    assert rows["1"].get("kind") != "trainer" and view["fleet"]["trainers_uncounted"] == 2


def test_no_tape_and_no_record_says_why_the_trainers_are_not_priced(tmp_path):
    r = _fleet(tmp_path, spend={}, no_tape=(0, 1))
    rows, view = _rows(r)
    assert not [x for x in rows.values() if x.get("kind") == "trainer"]
    assert view["fleet"]["trainers_usd"] is None
    assert view["fleet"]["trainers_note"] == "island 0, 1, 2: no trainer tape or launch record here"
    text = re.sub(r"<[^>]*>", " ", fleet_na._page(tmp_path, r)["costBox"])
    assert "trainers not priced (island 0, 1, 2: no trainer tape or launch record here)" in text
    assert "$n/a" not in text and not re.search(r"\bn/a\b", text.split("fleet spend")[1].split(".")[0])


def test_a_fleet_whose_trainers_have_tapes_keeps_its_rows(tmp_path):
    """Every trainer with a tape and no launch records (a fleet launched as one run): its row
    priced by the plan, as before; an island the desired state added without a tape is named,
    and the sum says it is low by it."""
    r = _fleet(tmp_path, spend=None, per_island=False)
    rows, view = _rows(r)
    assert rows["0"]["kind"] == "trainer" and "fleet plan trainer_usd_per_h" in rows["0"]["rate_source"]
    assert "2" not in rows and view["fleet"]["trainers_usd"] is not None
    assert view["fleet"]["trainers_uncounted"] == 1
    assert view["fleet"]["trainers_note"] == "island 2: no trainer tape or launch record here"
