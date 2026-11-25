"""A run with a syncer (trainer islands) where one island ended before its first round: the
drawing, the Fragments table, the Islands table and the island details show a dash with the
reason (or one line saying how it ended) and the KPI cards a sentence, never a bare "n/a".
A run with one trainer keeps its "n/a" (``test_dashboard_fleet_na`` checks those pages)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import test_dashboard_fleet_na as fleet_na
from test_dashboard_diloco import D, _smoke, lines, load

from yeto.dashboard import sources
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

L1 = "dlrl-qwen4b-1123-b-l1-modal.jsonl"


def _text(html: str) -> str:
    return re.sub(r"<[^>]*>", " ", html)


@pytest.fixture
def never_trained(tmp_path: Path) -> Reducer:
    """Run B with island 1's trainer tape cut after its start (hardware, engine, driver start):
    the island launched, then ended (fleet.jsonl's island_stop) before its first round."""
    dst = tmp_path / "run-b"
    shutil.copytree(D / "run-b", dst)
    tape = dst / "events" / L1
    tape.write_text("".join(json.dumps(x) + "\n" for x in lines(tape)[:4] if x.get("event") != "rl_driver_phase"))
    r = Reducer(run="run-b")
    load_all(r, [str(p) for p in sources.discover([str(dst), str(dst / "events")])])
    return r


def _page(tmp_path: Path, r: Reducer) -> dict:
    """The fleet n/a smoke page, with the island details (``islands``) as well."""
    smoke = fleet_na.SMOKE.replace('"evs","rNodes"]', '"evs","rNodes","islands"]')
    assert smoke != fleet_na.SMOKE
    orig, fleet_na.SMOKE = fleet_na.SMOKE, smoke
    try:
        return fleet_na._page(tmp_path, r)
    finally:
        fleet_na.SMOKE = orig


def test_a_trainer_that_never_stepped_is_drawn_with_dashes(tmp_path, never_trained):
    out = _smoke(tmp_path, never_trained)
    texts = out["texts"]
    for want in ("step —", "round —", "no snapshot yet", "step 29", "snapshot e5fd32df · lr 1e-5"):
        assert want in texts, want
    assert not any(re.search(r"\bn/a\b", t) for t in texts)
    matrix = _text(out["matrix"])
    assert re.search(r"Trainer 1\s+H100 · step —", matrix) and "n/a" not in matrix


def test_an_island_that_never_trained_is_one_line_in_the_islands_table(tmp_path, never_trained):
    out = _page(tmp_path, never_trained)
    nodes = out["rNodes"]
    row = next(tr for tr in re.findall(r"<tr[^>]*>.*?</tr>", nodes) if "trainer 1" in tr)
    assert "done · ended before its first round" in row and 'colspan="' in row
    assert row.count("<td") == 2  # the island, and the one line
    assert "n/a" not in _text(nodes)
    # the island that trained keeps its full row
    assert any("trainer 0" in tr and tr.count("<td") > 5 for tr in re.findall(r"<tr[^>]*>.*?</tr>", nodes))


def test_island_details_dash_what_the_island_did_not_report(tmp_path, never_trained):
    details = _page(tmp_path, never_trained)["islands"]
    assert "phase / policy" in details and "n/a" not in _text(details)
    assert 'title="not reported by this island yet">—</span>' in details


def test_syncer_run_kpi_cards_say_what_is_missing(tmp_path):
    out = _page(tmp_path, load("run-a"))
    for key in ("rKpis", "rCostK"):
        assert not re.search(r"\bn/a\b", _text(out[key])), key
    assert '<div class="kt">MFU</div><div class="kv">not on the tapes</div>' in out["rCostK"]
