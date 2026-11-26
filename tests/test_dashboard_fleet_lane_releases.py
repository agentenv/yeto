"""A fleet's policy releases live on its island lanes (``trainer<k>/policy-releases/``), not at
the run store's root: the topology's store node reads each lane's ``LATEST.json`` instead of
saying "no release" while every island released. A run without lanes draws as before."""

from __future__ import annotations

import json

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
        path = store / lane / "policy-releases" / "LATEST.json"
        path.parent.mkdir(parents=True)
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return DirectoryObjectStore(str(store))


def test_each_lanes_newest_release_is_read_at_the_plan_poll(tmp_path):
    objects = _store(tmp_path, {"trainer0": {"version": 16, "released_at": 1790000100.5, "key": "r16-v16"},
                                "trainer1": {"version": 14}})
    clock = Clock()
    r = Reducer()
    pf = fleet_islands.PlanFile([], objects, clock=clock)
    pf.pump(r)
    pf.pump_tapes(r)
    assert r.lane_releases == {"trainer0": {"version": 16, "released_at": 1790000100.5},
                               "trainer1": {"version": 14, "released_at": None}}
    (tmp_path / "store" / "trainer1" / "policy-releases" / "LATEST.json").write_text(json.dumps({"version": 15}))
    assert pf.pump_lane_releases(r) is False and r.lane_releases["trainer1"]["version"] == 14  # not before the poll
    clock.t += pf.interval_s
    assert pf.pump_lane_releases(r) is True and r.lane_releases["trainer1"]["version"] == 15


def test_an_unreadable_or_foreign_latest_is_left_out(tmp_path):
    objects = _store(tmp_path, {"trainer0": "{not json", "trainer1": {"version": True}})
    r = Reducer()
    pf = fleet_islands.PlanFile([], objects, clock=Clock())
    pf.pump(r)
    assert pf.pump_lane_releases(r) is True and r.lane_releases == {}


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


def test_the_store_node_shows_each_lanes_release_instead_of_no_release(tmp_path):
    for sub in "abcd":
        (tmp_path / sub).mkdir()
    before = tt._smoke(tmp_path / "a", _unreleased(), tt.NOW)["texts"]
    assert "no release" in before and "no release yet" in before
    r = _unreleased()
    r.lane_releases = {"trainer1": {"version": 14, "released_at": None}, "trainer0": {"version": 16, "released_at": None},
                       "trainer2": {"version": 64, "released_at": None}}
    assert tt._view(r)["lanes"] == {"trainer0": 16, "trainer1": 14, "trainer2": 64}
    texts = tt._smoke(tmp_path / "b", r, tt.NOW)["texts"]
    assert "v14–v64" in texts and "trainer0/ v16 · trainer1/ v14 · trainer2/ v64" in texts
    assert "no release" not in texts and "no release yet" not in texts
    narrow = tt._smoke(tmp_path / "d", r, tt.NOW, width=400)["texts"]
    assert "v14–v64" in narrow and "lanes v16 v14 v64" in narrow  # fits beside the store at 400 px
    r.lane_releases = {"trainer0": {"version": 16, "released_at": None}, "trainer1": {"version": 16, "released_at": None}}
    assert "LATEST v16" in tt._smoke(tmp_path / "c", r, tt.NOW)["texts"]  # every lane at one version


def test_a_root_release_still_wins_over_the_lanes(tmp_path):
    r = tt.run()
    r.lane_releases = {"trainer0": {"version": 9, "released_at": None}}
    texts = tt._smoke(tmp_path, r, tt.NOW)["texts"]
    assert "LATEST v2" in texts and not any(t.startswith("trainer0/ v9") for t in texts)
