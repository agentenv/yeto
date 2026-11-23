"""diloco_island_starved: a live island that barely responds to the syncer's merges.

Derived from the merge records of the syncer tape (expected / responded /
missed_grace / grace_ms). The starved pattern is the one of a two-island
full-parameter run whose second island sat behind a cross-region link: it
responded to one merge in eight and the syncer waited out its grace for it on
the others (fixture g5-fullparam, replayed merge by merge). A run whose islands
miss merges now and then (fixture run-a) raises nothing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace


from dashboard_helpers import FX

from yeto.dashboard import alerts_diloco, island_starvation as st, sources, topology_diloco
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all


def _trainer(iid: str, **kw) -> dict:
    t = {"id": iid, "live": True, "versions": [9, 10, 11, 8], "last_bcast": {"bytes": 590e6, "queue_s": 41.0}}
    t.update(kw)
    return t


def _view(**kw) -> dict:
    d = {"kind": "island", "running": True, "N": 1, "versions": [9, 10, 11, 8],
         "trainers": [_trainer("0"), _trainer("1", versions=[5, 2, 3, 4])]}
    d.update(kw)
    return d


def _merge(step: int, responded, missed=(), expected=(0, 1), grace_ms=132000) -> dict:
    rec = {"step": step, "fragment": (step - 1) % 4, "time_unix": 1000.0 + step, "expected": list(expected),
           "responded": list(responded), "missed_grace": list(missed), "grace_ms": grace_ms}
    return {"step": step, "frag": rec["fragment"], "ts": rec["time_unix"], "shard": 0, "rec": rec}


def _starved_run() -> list[dict]:
    """Island 1 responds to merge 4 only of steps 1..8; the syncer waits its grace on the rest."""
    return [_merge(s, [0, 1] if s == 4 else [0], [] if s == 4 else [1]) for s in range(1, 9)]


def test_the_starved_island_is_named_with_its_numbers():
    got = st.starved(_view(), 1100.0, _starved_run())
    assert len(got) == 1
    a = got[0]
    assert (a["rule"], a["island"], a["sev"]) == ("diloco_island_starved", "1", 1)
    assert "responded to 1 of its last 8 merges" in a["title"]
    assert "7 time(s), 924s" in a["detail"]
    assert "[5, 2, 3, 4] vs syncer [9, 10, 11, 8]" in a["detail"]
    assert "590 MB" in a["detail"] and "--syncer-region" in a["detail"]


def test_an_island_that_never_responds_and_is_always_waited_for_is_critical():
    merges = [_merge(s, [0], [1]) for s in range(1, 9)]
    assert st.starved(_view(), 1100.0, merges)[0]["sev"] == 0


def test_healthy_and_intermittent_islands_raise_nothing():
    healthy = [_merge(s, [0, 1]) for s in range(1, 9)]
    assert st.starved(_view(), 1100.0, healthy) == []
    # every other merge: 50% share, above the threshold
    flaky = [_merge(s, [0, 1] if s % 2 else [0], [] if s % 2 else [1]) for s in range(1, 9)]
    assert st.starved(_view(), 1100.0, flaky) == []


def test_only_the_last_window_counts():
    recovered = _starved_run() + [_merge(s, [0, 1]) for s in range(9, 17)]
    assert st.starved(_view(), 1100.0, recovered) == []


def test_too_few_merges_down_islands_finished_runs_and_sft_raise_nothing():
    few = _starved_run()[-3:]
    assert st.starved(_view(), 1100.0, few) == []
    down = _view(trainers=[_trainer("0"), _trainer("1", live=False)])
    assert st.starved(down, 1100.0, _starved_run()) == []
    assert st.starved(_view(running=False), 1100.0, _starved_run()) == []
    assert st.starved(_view(kind="sft"), 1100.0, _starved_run()) == []


def test_merges_that_do_not_expect_the_island_are_skipped():
    # the island left the roster: later merges expect only island 0
    left = _starved_run()[:2] + [_merge(s, [0], expected=(0,)) for s in range(3, 20)]
    assert st.participation(left, "1")["expected"] == 2
    assert st.starved(_view(), 1100.0, left) == []


def test_responders_records_without_a_responded_list():
    merges = []
    for s in range(1, 9):
        m = _merge(s, [], [1])
        m["rec"].pop("responded")
        m["rec"]["responders"] = [{"id": 0, "contribution": 1.0}]
        merges.append(m)
    p = st.participation(merges, "1")
    assert (p["expected"], p["responded"], p["missed_grace"]) == (8, 0, 8)
    assert st.participation(merges, "0")["responded"] == 8


def test_the_real_run_a_merges_raise_nothing():
    """run-a: island 0 waited out the grace on 4 of the first 8 merges; a half share is not starvation."""
    r = Reducer(run="run-a")
    load_all(r, [str(p) for p in sources.discover([str(FX / "diloco" / "run-a")])])
    merges = topology_diloco.merge_rows(r, r.max_ts, 1)
    first = st.participation(merges[:8], "0")
    assert (first["expected"], first["responded"], first["missed_grace"]) == (8, 4, 4)
    view = _view(trainers=[_trainer("0"), _trainer("1")])
    for end in range(st.MIN_MERGES, len(merges) + 1):
        assert st.starved(view, r.max_ts, merges[:end]) == [], end


def _replay(src: Path, cut: float, tmp: Path) -> list[tuple[str, int]]:
    """The starvation alerts of the run as its tapes stood at ``cut`` (every record later dropped)."""
    for p in src.rglob("*.jsonl"):
        q = tmp / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        keep = [x for x in p.read_text().splitlines() if x.strip() and json.loads(x).get("time_unix", 0) <= cut]
        q.write_text("".join(x + "\n" for x in keep))
    r = Reducer(run=src.name)
    load_all(r, [str(p) for p in sources.discover([str(tmp)])])
    cards = r.overview(now=cut)["islands"]
    return [(a["island"], a["sev"]) for a in alerts_diloco.evaluate(r, cut, cards)
            if a["rule"] == "diloco_island_starved"]


def test_the_real_cross_region_run_starves_island_1_only(tmp_path):
    """g5-fullparam: island 1 sat behind a cross-region link, responded to merge 1 only and
    was waited for on merges 2..12. It warns from merge 4 (1 of 4) and is critical from
    merge 9 (none of its last 8, each waited out); island 0 never alerts."""
    src = FX / "diloco" / "g5-fullparam"
    merges = [json.loads(x) for x in (src / "yeto-tape.jsonl").read_text().splitlines() if x.strip()]
    assert [m["step"] for m in merges] == list(range(1, 13))
    for m in merges:
        got = _replay(src, m["time_unix"], tmp_path / f"at-{m['step']}")
        want = [] if m["step"] < 4 else [("1", 1)] if m["step"] < 9 else [("1", 0)]
        assert got == want, m["step"]


def test_the_real_run_a_replay_raises_nothing(tmp_path):
    src = FX / "diloco" / "run-a"
    merges = [json.loads(x) for x in (src / "yeto-tape.jsonl").read_text().splitlines() if x.strip()]
    for m in merges:
        assert _replay(src, m["time_unix"], tmp_path / f"at-{m['step']}") == [], m["step"]


def test_evaluate_runs_the_rule_on_the_syncer_merges(monkeypatch):
    view = _view()
    monkeypatch.setattr(topology_diloco, "current", lambda reducer, cards, now: copy.deepcopy(view))
    monkeypatch.setattr(alerts_diloco, "RULES", ())  # the other rules read fields this view leaves out
    reducer = SimpleNamespace(merges=[m["rec"] for m in _starved_run()])
    got = [a for a in alerts_diloco.evaluate(reducer, 1100.0, []) if a["rule"] == "diloco_island_starved"]
    assert [a["island"] for a in got] == ["1"]
