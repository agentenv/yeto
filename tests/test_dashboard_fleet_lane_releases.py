"""A fleet's policy releases live on its island lanes (``trainer<k>/policy-releases/``), not at
the run store's root: the topology's store node reads each lane's ``LATEST.json`` instead of
saying "no release" while every island released. When a run has lanes, the lanes win over the
release events on the tapes (a lane whose tape is not here released all the same). A run
without lanes draws as before."""

from __future__ import annotations

import json
import re

import test_dashboard_topology as tt
from test_dashboard_fleet_islands import plan_doc

from yeto.dashboard import fleet_islands
from yeto.dashboard.reducer import Reducer
from yeto.rl.engine.object_store import DirectoryObjectStore


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def _store(tmp_path, latest: dict) -> DirectoryObjectStore:
    store = tmp_path / "store"
    (store / "fleet").mkdir(parents=True)
    (store / "fleet" / "plan.json").write_text(json.dumps(plan_doc()))
    for lane, doc in latest.items():
        _latest(tmp_path, lane, doc)
    return DirectoryObjectStore(str(store))


def _latest(tmp_path, lane: str, doc) -> None:
    path = tmp_path / "store" / lane / "policy-releases" / "LATEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))


def _plan(tmp_path, latest: dict) -> tuple[fleet_islands.PlanFile, Reducer, Clock]:
    clock = Clock()
    r = Reducer()
    pf = fleet_islands.PlanFile([], _store(tmp_path, latest), clock=clock)
    pf.pump(r)
    return pf, r, clock


def test_each_lanes_newest_release_is_read_at_the_plan_poll(tmp_path):
    pf, r, clock = _plan(tmp_path, {"trainer0": {"version": 16, "released_at": 1790000100.5, "key": "r16-v16"},
                                    "trainer1": {"version": 14}})
    pf.pump_tapes(r)
    assert r.lane_releases == {"trainer0": 16, "trainer1": 14}  # the version only
    _latest(tmp_path, "trainer1", {"version": 15})
    assert pf.pump_lane_releases(r) is False and r.lane_releases["trainer1"] == 14  # not before the poll
    clock.t += pf.interval_s
    assert pf.pump_lane_releases(r) is True and r.lane_releases["trainer1"] == 15


def test_a_lane_that_cannot_be_read_keeps_its_last_version(tmp_path):
    pf, r, clock = _plan(tmp_path, {"trainer0": {"version": 16}, "trainer1": {"version": 14}})
    assert pf.pump_lane_releases(r) is True and r.lane_releases == {"trainer0": 16, "trainer1": 14}
    for bad in ("{not json", {"version": True}, {"version": -1}, {"version": 16.5}, {"version": "17"}, []):
        _latest(tmp_path, "trainer0", bad)
        clock.t += pf.interval_s
        assert pf.pump_lane_releases(r) is True and r.lane_releases == {"trainer0": 16, "trainer1": 14}, bad
    (tmp_path / "store" / "trainer0" / "policy-releases" / "LATEST.json").unlink()  # gone this time
    clock.t += pf.interval_s
    assert pf.pump_lane_releases(r) is True and r.lane_releases["trainer0"] == 16
    _latest(tmp_path, "trainer0", {"version": 17})
    clock.t += pf.interval_s
    assert pf.pump_lane_releases(r) is True and r.lane_releases["trainer0"] == 17


def test_a_never_read_lane_with_a_bad_document_is_left_out(tmp_path):
    pf, r, _ = _plan(tmp_path, {"trainer0": "{not json", "trainer1": {"version": True}})
    assert pf.pump_lane_releases(r) is True and r.lane_releases == {}


def test_version_tolerance(tmp_path):
    """``16.0`` is 16; a negative, fractional, boolean or string version is not a release."""
    pf, r, clock = _plan(tmp_path, {"trainer0": {"version": 16.0}, "trainer1": {"version": -3}})
    assert pf.pump_lane_releases(r) is True and r.lane_releases == {"trainer0": 16}
    assert type(r.lane_releases["trainer0"]) is int
    _latest(tmp_path, "trainer1", {"version": 0})  # the first release of a lane is v0
    clock.t += pf.interval_s
    assert pf.pump_lane_releases(r) is True and r.lane_releases == {"trainer0": 16, "trainer1": 0}


def test_no_lanes_no_read(tmp_path):
    r = Reducer()
    pf = fleet_islands.PlanFile([], None, clock=Clock())
    assert pf.pump_lane_releases(r) is False and getattr(r, "lane_releases", None) is None


def _unreleased() -> Reducer:
    """Two samplers and a trainer that trained, with no release at the store's root."""
    recs = [tt.sampler_start("b300-0", tt.T0, "NVIDIA B300", "nvfp4", "vllm"),
            {"event": "rl_driver_start", "island_id": 0, "time_unix": tt.T0}]
    for k in range(6):
        recs += [tt.group("b300-0", tt.T0 + 20 + 25 * k, 0), tt.sampler_round("b300-0", tt.T0 + 20 + 25 * k, k, 0)]
    recs += tt.trainer_step(tt.T0 + 10, 0)
    r = Reducer(run="topo")
    for rec in sorted(recs, key=lambda x: x["time_unix"]):
        r.feed(rec)
    return r


# the smoke page with a namespaced SVG, so the hover <title> the panel adds is drawn and read back
SMOKE_NS = tt.SMOKE.replace("createElementNS: () => { const n = node(null);",
                            "createElementNS: (ns, tag) => { const n = node(null); n.namespaceURI = ns; n.tag = tag;")
assert SMOKE_NS != tt.SMOKE


def _smoke(tmp_path, r, width=1200) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    saved, tt.SMOKE = tt.SMOKE, SMOKE_NS
    try:
        return tt._smoke(tmp_path, r, tt.NOW, width=width)
    finally:
        tt.SMOKE = saved


def _lanes(versions: list[int]) -> dict:
    return {f"trainer{k}": v for k, v in enumerate(versions)}


def test_the_store_node_shows_each_lanes_release_instead_of_no_release(tmp_path):
    before = _smoke(tmp_path / "a", _unreleased())["texts"]
    assert "no release" in before and "no release yet" in before
    r = _unreleased()
    r.lane_releases = {"trainer1": 14, "trainer0": 16, "trainer2": 64}
    assert tt._view(r)["lanes"] == {"trainer0": 16, "trainer1": 14, "trainer2": 64}
    texts = _smoke(tmp_path / "b", r)["texts"]
    assert "v14–v64" in texts and "trainer0/ v16 · trainer1/ v14 · trainer2/ v64" in texts
    assert "trainer0=v16, trainer1=v14, trainer2=v64" in texts  # the hover list
    assert "no release" not in texts and "no release yet" not in texts
    narrow = _smoke(tmp_path / "d", r, width=400)["texts"]  # three lanes do not fit beside the store at 400 px
    assert "v14–v64 · 3 lanes" in narrow and "trainer0=v16, trainer1=v14, trainer2=v64" in narrow
    r.lane_releases = {"trainer0": 16, "trainer1": 16}
    assert "LATEST v16" in _smoke(tmp_path / "c", r)["texts"]  # every lane at one version


def test_the_lanes_win_over_the_tape_release(tmp_path):
    """The tapes saw v16 at the root; a lane's LATEST.json says v64: the store shows v64."""
    r = tt.run([tt.release(tt.T0 + 400, 16, delta=True, upload_s=10.0, selected=8)])
    assert tt._view(r)["latest"] == 16
    assert "LATEST v16" in _smoke(tmp_path / "a", r)["texts"]  # no lanes: as before
    r.lane_releases = {"trainer0": 64}
    texts = _smoke(tmp_path / "b", r)["texts"]
    assert "LATEST v64" in texts and "trainer0/ v64" in texts
    assert "LATEST v16" not in texts and not any(t.startswith("last release") for t in texts)


def test_lanes_sort_by_number(tmp_path):
    r = _unreleased()
    r.lane_releases = {"trainer10": 30, "trainer2": 20, "trainer1": 10}
    assert list(tt._view(r)["lanes"]) == ["trainer1", "trainer2", "trainer10"]
    texts = _smoke(tmp_path, r)["texts"]
    assert "trainer1/ v10 · trainer2/ v20 · trainer10/ v30" in texts
    assert "trainer1=v10, trainer2=v20, trainer10=v30" in texts


def test_a_line_that_does_not_fit_falls_back_to_the_range_and_the_lane_count(tmp_path):
    for n in (6, 8, 12):
        versions = [14 + 50 * k // (n - 1) for k in range(n)]  # v14 .. v64
        r = _unreleased()
        r.lane_releases = _lanes(versions)
        hover = ", ".join(f"trainer{k}=v{v}" for k, v in enumerate(versions))
        for width in (1200, 400):
            texts = _smoke(tmp_path / f"{n}-{width}", r, width=width)["texts"]
            assert f"v14–v64 · {n} lanes" in texts, (n, width, texts)
            assert hover in texts, (n, width)
            assert not any(re.match(r"trainer\d+/ v", t) for t in texts), (n, width)
    r = _unreleased()
    r.lane_releases = {f"trainer{k}": 16 for k in range(12)}
    assert "v16 · 12 lanes" in _smoke(tmp_path / "same", r)["texts"]  # one version: no "LATEST" twice
