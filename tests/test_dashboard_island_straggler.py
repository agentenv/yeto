"""A trainer island the syncer has merged no push of for N merge cycles (``island_straggler``
in fleet.jsonl, ``straggler`` in its record) on the dashboard: the event stream shows the
straggler events, the trainer card carries a straggler badge until it is cleared, and the
events leave the island's lifecycle state alone."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_dashboard_diloco import SMOKE
from test_dashboard_fleet_islands import T0, fleet_reducer

from yeto.dashboard.export import export_html
from yeto.launch import island_straggler as st

# the topology page's smoke, also printing the event stream
LEGEND = 'legend: mk("topoLg").innerHTML'
EV_SMOKE = SMOKE.replace(LEGEND, LEGEND + ', events: mk("evs").innerHTML')
assert EV_SMOKE != SMOKE


def _flagged(k: int = 1, at: float = T0 + 50) -> dict:
    return {"event": st.EVENT, "time_unix": at, "island": f"fleet-qa-l{k}-modal", "island_id": k,
            "action": st.WAIT, "missed_cycles": 3, "merges": 12, "fragments": 4, "newest_step": 40,
            "last_merged_step": 28, "absent": True}


def _cleared(k: int = 1, at: float = T0 + 60) -> dict:
    return {"event": st.CLEARED_EVENT, "time_unix": at, "island": f"fleet-qa-l{k}-modal", "island_id": k,
            "missed_cycles": 0, "reason": "a push of it is merged again"}


def _reducer(*extra: dict):
    r = fleet_reducer()
    for rec in extra:
        r.feed(rec)
    return r


def _smoke(tmp: Path, r) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    tmp.mkdir(parents=True, exist_ok=True)
    page = export_html(r, tmp / "p.html", generated_at=r.max_ts)
    (tmp / "smoke.js").write_text(EV_SMOKE)
    res = subprocess.run([node, str(tmp / "smoke.js"), str(page), "1200"], capture_output=True, text=True,
                         timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _cards(r) -> dict:
    return {c["id"]: c for c in r.overview(now=r.max_ts)["islands"]}


def test_a_straggler_event_marks_the_card_and_keeps_its_lifecycle_state():
    plain = _cards(_reducer())
    flagged = _cards(_reducer(_flagged()))
    assert flagged["1"]["straggler"] == {"action": "wait", "missed_cycles": 3, "newest_step": 40,
                                         "last_merged_step": 28, "at": T0 + 50}
    assert flagged["0"]["straggler"] is None and plain["1"]["straggler"] is None
    assert flagged["1"]["status"] == plain["1"]["status"]  # not a lifecycle state
    assert _cards(_reducer(_flagged(), _cleared()))["1"]["straggler"] is None


def test_an_unavailable_watch_is_no_mark():
    rec = {"event": st.UNAVAILABLE_EVENT, "time_unix": T0 + 50, "island": "fleet-qa-l1-modal", "island_id": 1,
           "reason": "the run store is not an object store"}
    cards = _cards(_reducer(_flagged(), rec))
    assert cards["1"]["straggler"]["missed_cycles"] == 3  # still flagged: unavailable says nothing of it


def test_the_trainer_card_draws_a_straggler_badge_until_cleared(tmp_path):
    flagged = _smoke(tmp_path / "a", _reducer(_flagged()))
    assert "straggler" in flagged["texts"]  # the topology's trainer card
    # Islands now: trainer 1's row says straggler in warn colour in place of "ok", once
    assert flagged["nodes"].count('<span class="wn">straggler</span>') == 1
    shown = re.sub(r"<[^>]+>", "", flagged["nodes"])  # the row's visible text
    assert "H100 × 2 · straggler" in shown and "ok · straggler" not in shown
    assert "straggler" not in flagged["matrix"]  # not in the Fragments rows
    for r in (_reducer(), _reducer(_flagged(), _cleared())):
        out = _smoke(tmp_path / str(len(r.fleet_records)), r)
        assert "straggler" not in out["texts"] and "straggler" not in out["nodes"]


def test_the_island_record_s_mark_wins_over_the_events(tmp_path):
    """A per-island fleet: the island's record says what the controller holds now."""
    from test_dashboard_trainer_islands import _state

    from yeto.dashboard import fleet_islands

    r = _reducer(_flagged())
    state = _state(tmp_path)
    rec = {**state["trainers"][1]["record"], "launch_state": "running", "life_state": "running",
           "straggler": None}
    state["trainers"][1].update(record=rec, now=fleet_islands.island_state(rec, state["trainers"][1]))
    r.load_fleet_state(state)
    assert _cards(r)["1"]["island_now"]["straggler"] is None
    out = _smoke(tmp_path / "rec", r)
    assert "straggler" not in out["texts"] and "straggler" not in out["nodes"]


def test_the_controller_s_mark_reaches_the_card_through_the_island_records(tmp_path):
    """The production path: the controller writes the mark to the island's record in the run
    store, ``read_fleet_state`` reads it back (``_trainer_records``) and the card shows it."""
    from test_island_straggler import _controller, _join, _step

    from yeto.dashboard import fleet_islands
    from yeto.rl.engine.object_store import DirectoryObjectStore

    ctl, _ops, _store, tape, _log = _controller(tmp_path)
    tape.pending += [_join(0), _join(1), _step(1, [0, 1], fragments=1), _step(2, [0], fragments=1, absent=[1])]
    ctl._poll_stragglers()  # island 1 missed one cycle: reported, its record carries the mark

    def card_of():  # a fresh read of the run store, as the dashboard's poll does
        r = _reducer()  # no island_straggler event: only the record can tell
        assert r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(tmp_path / "store"))))
        return r, _cards(r)["1"]

    r, card = card_of()
    mark = card["island_now"]["straggler"]
    assert mark["action"] == st.WAIT and mark["missed_cycles"] == 1 and mark["last_merged_step"] == 1
    assert mark["newest_step"] == 2 and mark["absent"] is True and isinstance(mark["at"], float)
    assert (_cards(r)["0"]["island_now"] or {}).get("straggler") is None  # island 0: no mark
    out = _smoke(tmp_path / "flagged", r)
    assert out["nodes"].count('<span class="wn">straggler</span>') == 1
    tape.pending.append(_step(3, [0, 1], fragments=1))
    ctl._poll_stragglers()  # a push of it is merged again: cleared in its record
    r, card = card_of()
    assert card["island_now"]["straggler"] is None
    out = _smoke(tmp_path / "cleared", r)
    assert "straggler" not in out["texts"] and "straggler" not in out["nodes"]


def test_the_event_stream_shows_the_straggler_events(tmp_path):
    end = fleet_reducer().max_ts  # the newest events: the stream shows the latest 15
    events = _smoke(tmp_path, _reducer(_flagged(at=end + 10), _cleared(at=end + 20)))["events"]
    assert "<b>island_straggler</b>" in events and "missed_cycles=3" in events and "action=wait" in events
    assert "<b>island_straggler_cleared</b>" in events and "reason=a push of it is merged again" in events
    cleared = events.split("<b>island_straggler_cleared</b>", 1)[1].split("</div>", 1)[0]
    assert "missed_cycles" not in cleared  # its reason only
    assert 'title="lag"' in events and 'title="back"' in events
    assert "<b>island_straggler</b> trainer 1 action=wait missed_cycles=3" in events
