"""A fleet's live membership on the dashboard: pools ``yeto fleet apply`` added and removed after
the plan, retired shards and the samplers' leaves. A removed pool's sampler is shown as left
(not stale, no heartbeat alert) and its pool is named by its pool name and provider."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_dashboard_fleet_islands import STORE, T0, fleet_records, plan_doc
from test_dashboard_fleet_na import SMOKE

from yeto.dashboard import fleet_islands
from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.topology_diloco import _fleet_part
from yeto.rl.engine.object_store import DirectoryObjectStore

ADD = "qa-add-i0-0"  # the sampler of the pool fleet apply added (i0-add-l4), then removed
ADDED_AT, REMOVED_AT, RETIRED_AT = T0 + 150, T0 + 400, T0 + 420


def _applied(seq: int, at: float, *, with_add: bool) -> dict:
    members = [{"kind": "sampler_pool", "name": p["name"],
                "pool": {"provider": p["provider"], "shard_base": p["shard_base"], "usd_per_h": 2.0,
                         "run_prefix": f"fleet-qa/trainer{i['learner']}", "sampler": {"island": i["learner"]}}}
               for i in plan_doc()["islands"] for p in i["pools"]]
    if with_add:
        members.append({"kind": "sampler_pool", "name": "i0-add-l4",
                        "pool": {"provider": "modal", "shard_base": "qa-add-i0", "usd_per_h": 1.245,
                                 "run_prefix": "fleet-qa/trainer0", "sampler": {"island": 0}}})
    return {"schema": "yeto.fleet.applied/v1", "seq": seq, "generation": seq, "at": at, "members": members,
            "removing": []}


def _store(tmp_path: Path, *, leave: dict | None = None, member: str | None = "gone") -> Path:
    """The run store after fleet apply added i0-add-l4 (generation 2) and removed it (3): its pool
    state and member table retire its shard; its sampler's member record has ``leave`` (none: the
    sampler was released before it wrote one); ``member``: the pool's state of the machine serving the
    shard (gone: released; None: not listed)."""
    root = tmp_path / "store"
    w = lambda rel, doc: (root / rel).parent.mkdir(parents=True, exist_ok=True) or (root / rel).write_text(  # noqa: E731
        json.dumps(doc))
    w("fleet/plan.json", plan_doc())
    w("fleet/applied/00000001.json", _applied(1, T0, with_add=False))
    w("fleet/applied/00000002.json", _applied(2, ADDED_AT, with_add=True))
    w("fleet/applied/00000003.json", _applied(3, REMOVED_AT, with_add=False))
    w("fleet/pools/i0-add-l4/state.json", {"format": 2, "stopped": "removed from the fleet's desired state",
                                           "retired_shards": [ADD],
                                           "members": [{"shard": ADD, "state": member}] if member else []})
    w("trainer0/fleet/members/i0-add-l4.json", {"schema": "yeto.fleet.member-table/v1", "pool": "i0-add-l4",
                                                "shard_base": "qa-add-i0", "epoch": 2, "shards": [ADD],
                                                "retired": [ADD], "updated_at": RETIRED_AT})
    rec = {"sampler": ADD, "incarnation": "add-x1", "joined_at": ADDED_AT + 10, "left_at": None, "leave_reason": None,
           "detail": {}}
    w(f"trainer0/sample-buffer/members/{ADD}/add-x1.json", {**rec, **(leave or {})})
    return root


def _reducer(*, leave_event: str | None = None, rounds_after: int = 0) -> Reducer:
    """The fleet fixture plus the added pool's sampler: three rounds, then nothing (its machine
    was released when its pool was removed)."""
    r = Reducer(run="fleet-qa")
    recs = fleet_records()
    recs.append({"event": "rl_sampler_start", "sampler_id": ADD, "time_unix": ADDED_AT + 5, "island": 0,
                 "store": f"{STORE}/trainer0", "provenance": {"card_model": "L4", "engine": "sglang", "precision": "bf16",
                                                              "topology": "tp1"}})
    recs.append({"event": "rl_sampler_join", "sampler_id": ADD, "time_unix": ADDED_AT + 10, "sampler": ADD,
                 "incarnation": "add-x1"})
    for i in range(3):
        recs.append({"event": "rl_sampler_round", "sampler_id": ADD, "time_unix": ADDED_AT + 60 + 40 * i,
                     "rollout_id": i, "groups": 2, "policy_version": 1, "generate_s": 25.0})
    for i in range(rounds_after):  # still sampling after its shard retired (a drain, or a failed release)
        recs.append({"event": "rl_sampler_round", "sampler_id": ADD, "time_unix": RETIRED_AT + 60 * (i + 1),
                     "rollout_id": 3 + i, "groups": 2, "policy_version": 1, "generate_s": 25.0})
    if leave_event:
        recs.append({"event": "rl_sampler_leave", "sampler_id": ADD, "time_unix": RETIRED_AT - 5, "sampler": ADD,
                     "incarnation": "add-x1", "reason": leave_event})
    for rec in sorted(recs, key=lambda x: x.get("time_unix", 0.0)):
        r.feed(rec)
    assert r.load_fleet_plan(plan_doc(), "test")
    return r


def _card(r: Reducer, sid: str, now: float) -> dict:
    return next(c for c in r.overview(now=now)["islands"] if c["id"] == sid)


def _hb(r: Reducer, sid: str, now: float) -> list[dict]:
    return [a for a in r.overview(now=now)["alerts"] if a["rule"] == "heartbeat" and a.get("island") == sid]


LATE = T0 + 100 + 60 * 10 + 900  # long after the added sampler's last round: a quiet sampler is stale here


def test_without_membership_a_removed_pools_sampler_is_stale():
    """What the page showed before: no leave on its tape, no membership read."""
    r = _reducer()
    assert _card(r, ADD, LATE)["status"] in ("stale", "lost") and _hb(r, ADD, LATE)


def test_a_retired_shard_is_left_not_stale_and_raises_no_heartbeat_alert(tmp_path):
    r = _reducer()
    state = fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path))))
    pool = state["pools"]["i0-add-l4"]
    assert pool["removed_at"] == REMOVED_AT and pool["provider"] == "modal" and pool["retired"] == [ADD]
    assert state["shards"][ADD]["retired"] and state["shards"][ADD]["incarnation"] == "add-x1"
    assert r.load_fleet_state(state) and not r.load_fleet_state(state)
    card = _card(r, ADD, LATE)
    assert card["status"] == "left" and card["left"]["reason"] == "pool removed"
    assert "i0-add-l4" in card["left"]["detail"] and not _hb(r, ADD, LATE)


@pytest.mark.parametrize("leave, status, reason", [
    ({"left_at": RETIRED_AT - 3, "leave_reason": "POOL_REMOVED", "detail": {"left_by": "pool"}}, "left", "POOL_REMOVED"),
    ({"left_at": RETIRED_AT - 3, "leave_reason": "FENCED"}, "left", "FENCED"),
    ({"left_at": RETIRED_AT - 3, "leave_reason": "FINISHED"}, "done", None),
])
def test_a_leave_in_the_member_record_names_its_reason(tmp_path, leave, status, reason):
    """The pool writes POOL_REMOVED for a sampler released before it could; the sampler's own
    FENCED; FINISHED is a finished sampler."""
    r = _reducer()
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path, leave=leave)))))
    card = _card(r, ADD, LATE)
    assert card["status"] == status and not _hb(r, ADD, LATE)
    if reason:
        assert card["left"]["reason"] == reason
        assert ("written by its pool" in card["left"]["detail"]) == (reason == "POOL_REMOVED")


@pytest.mark.parametrize("reason, status", [("FENCED", "left"), ("POOL_REMOVED", "left"), ("PREEMPTED", "left"),
                                            ("FINISHED", "done")])
def test_the_samplers_own_leave_on_its_tape_ends_it(reason, status):
    """rl_sampler_leave on the sampler's tape: no store membership needed."""
    r = _reducer(leave_event=reason)
    card = _card(r, ADD, LATE)
    assert card["status"] == status and not _hb(r, ADD, LATE)
    if status == "left":
        assert card["left"]["reason"] == reason and card["left"]["source"] == "sampler leave"


def test_a_leave_of_an_older_incarnation_does_not_end_a_restarted_sampler(tmp_path):
    r = _reducer()
    r.feed({"event": "rl_sampler_join", "sampler_id": ADD, "time_unix": LATE - 30, "sampler": ADD,
            "incarnation": "add-x2"})
    r.feed({"event": "rl_sampler_round", "sampler_id": ADD, "time_unix": LATE - 10, "rollout_id": 9, "groups": 2,
            "policy_version": 2, "generate_s": 25.0})
    leave = {"left_at": RETIRED_AT - 3, "leave_reason": "FENCED"}
    st = fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path, leave=leave))))
    st["shards"][ADD]["retired"] = False  # only the old incarnation's leave: the new one runs on
    r.load_fleet_state(st)
    assert _card(r, ADD, LATE)["status"] == "ok"


@pytest.mark.parametrize("member", [None, "draining", "ready"])
def test_a_retired_sampler_that_still_writes_is_not_left(tmp_path, member):
    """The pool retires the shard before it drains: a sampler still writing rounds after it (its
    drain, or a release that failed) runs, shown draining, never left."""
    r = _reducer(rounds_after=12)
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path, member=member)))))
    now = RETIRED_AT + 60 * 12 + 20
    card = _card(r, ADD, now)
    assert card["status"] not in ("left", "done") and card["left"] is None
    assert card["pool_removed"]["reason"] == "pool removed" and card["pool_removed"]["wrote_after"]
    alerts = [a for a in r.overview(now=now)["alerts"] if a["rule"] == "pool_removed_running"]
    assert alerts and alerts[0]["island"] == ADD and "still writes" in alerts[0]["detail"]


def test_a_quiet_retired_sampler_whose_machine_the_pool_still_holds_runs_and_alerts(tmp_path):
    """Quiet since its shard retired, but the pool still holds its machine (release failed):
    still billed, so not left; within the grace it is draining, past it an alert."""
    r = _reducer()
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path, member="draining")))))
    soon = RETIRED_AT + 60
    card = _card(r, ADD, soon)
    assert card["status"] != "left" and card["pool_removed"]["member_state"] == "draining"
    assert not [a for a in r.overview(now=soon)["alerts"] if a["rule"] == "pool_removed_running"]
    late = [a for a in r.overview(now=LATE)["alerts"] if a["rule"] == "pool_removed_running"]
    assert _card(r, ADD, LATE)["status"] != "left" and late and late[0]["sev"] == 1
    assert "still holds its machine (draining)" in late[0]["detail"]
    later = RETIRED_AT + 4 * 300 + 60  # well past the drain: critical
    assert [a["sev"] for a in r.overview(now=later)["alerts"] if a["rule"] == "pool_removed_running"] == [0]


def test_a_released_sampler_is_left_even_with_a_round_after_its_retirement(tmp_path):
    """The pool released its machine (member gone): nothing of it runs, whatever its clock wrote."""
    r = _reducer(rounds_after=1)
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path, member="gone")))))
    card = _card(r, ADD, LATE)
    assert card["status"] == "left" and "released the machine" in card["left"]["detail"]


def test_a_member_record_leave_with_rounds_after_it_is_not_believed(tmp_path):
    r = _reducer(rounds_after=3)
    leave = {"left_at": RETIRED_AT - 3, "leave_reason": "POOL_REMOVED", "detail": {"left_by": "pool"}}
    r.load_fleet_state(fleet_islands.read_fleet_state(
        DirectoryObjectStore(str(_store(tmp_path, leave=leave, member=None)))))
    card = _card(r, ADD, RETIRED_AT + 200)
    assert card["status"] != "left" and card["pool_removed"]["reason"] == "POOL_REMOVED"


def test_the_samplers_own_leave_with_rounds_after_it_is_not_believed():
    r = _reducer(leave_event="FENCED", rounds_after=3)
    assert _card(r, ADD, RETIRED_AT + 200)["status"] != "left"


def test_the_lane_is_the_last_trainer_segment_of_the_run_prefix():
    assert fleet_islands._lane({"run_prefix": "team/run/trainer1"}) == "trainer1"  # noqa: SLF001
    assert fleet_islands._lane({"run_prefix": "run/trainer12/"}) == "trainer12"  # noqa: SLF001
    assert fleet_islands._lane({"run_prefix": "run", "learner": 2}) == "trainer2"  # noqa: SLF001


class _Counting(DirectoryObjectStore):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.reads: list[str] = []

    def read(self, key):  # noqa: ANN001, ANN201
        self.reads.append(key)
        return super().read(key)


def test_settled_membership_is_not_read_again(tmp_path):
    """A generation is folded once; a removed pool whose machine was released is not read again."""
    objects = _Counting(str(_store(tmp_path)))
    cache: dict = {}
    first = fleet_islands.read_fleet_state(objects, cache)
    n_first = len(objects.reads)
    objects.reads.clear()
    again = fleet_islands.read_fleet_state(objects, cache)
    assert again == first
    assert not [k for k in objects.reads if k.startswith("fleet/applied/")]
    assert not [k for k in objects.reads if "i0-add-l4" in k or ADD in k]
    assert len(objects.reads) < n_first
    # a live pool keeps being read: its state and member table may change
    assert any(k.startswith("fleet/pools/") for k in objects.reads)


def test_a_still_held_shard_is_read_again(tmp_path):
    objects = _Counting(str(_store(tmp_path, member="draining")))
    cache: dict = {}
    fleet_islands.read_fleet_state(objects, cache)
    objects.reads.clear()
    fleet_islands.read_fleet_state(objects, cache)
    assert "fleet/pools/i0-add-l4/state.json" in objects.reads


def test_a_new_generation_is_folded_on_the_next_read(tmp_path):
    root = _store(tmp_path)
    cache: dict = {}
    objects = DirectoryObjectStore(str(root))
    assert fleet_islands.read_fleet_state(objects, cache)["pools"]["i0-add-l4"]["removed_at"] == REMOVED_AT
    (root / "fleet/applied/00000004.json").write_text(json.dumps(_applied(4, REMOVED_AT + 900, with_add=True)))
    st = fleet_islands.read_fleet_state(objects, cache)
    assert st["pools"]["i0-add-l4"]["removed_at"] is None  # added back


def test_a_pools_shards_retired_event_ends_its_sampler():
    r = _reducer()
    r.feed({"event": "rl_pool_shards_retired", "shards": [ADD], "t": RETIRED_AT})
    card = _card(r, ADD, LATE)
    assert card["status"] == "left" and "retired" in card["left"]["reason"] and not _hb(r, ADD, LATE)


def test_an_applied_pool_is_named_with_its_provider_and_drawn_as_removed(tmp_path):
    r = _reducer()
    before = fleet_islands.resolve(r)
    assert before["pool_of"][ADD] == "qa-add-i0"  # no plan pool names it: its shard base, no provider
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path)))))
    fleet = fleet_islands.resolve(r)
    assert fleet["pool_of"][ADD] == "i0-add-l4"
    pool = next(p for p in fleet["islands"][0]["pools"] if p["name"] == "i0-add-l4")
    assert pool["provider"] == "modal" and pool["usd_per_h"] == 1.245 and pool["shard_base"] == "qa-add-i0"
    assert pool["removed"] == {"at": REMOVED_AT, "why": "removed from the fleet's desired state"}
    drawn = next(p for p in _fleet_part(fleet, r.async_)["islands"][0]["pools"] if p["name"] == "i0-add-l4")
    assert drawn["provider"] == "modal" and drawn["removed"]["at"] == REMOVED_AT
    # the plan's pools are not removed
    assert not any(p.get("removed") for p in fleet["islands"][0]["pools"] if p["name"] != "i0-add-l4")


def test_plan_file_reads_the_membership_with_the_plan(tmp_path):
    root = _store(tmp_path)
    r = _reducer()
    pf = fleet_islands.PlanFile([], DirectoryObjectStore(str(root)))
    pf.pump(r)
    assert r.fleet_state is not None and _card(r, ADD, LATE)["status"] == "left"


def test_no_applied_membership_reads_as_none(tmp_path):
    (tmp_path / "s").mkdir()
    assert fleet_islands.read_fleet_state(DirectoryObjectStore(str(tmp_path / "s"))) is None


# -- the page -----------------------------------------------------------------------------
def _page(tmp_path: Path, r: Reducer) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(r, tmp_path / "p.html", generated_at=r.max_ts)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE.replace('"costBox","evs","rNodes"]', '"costBox","evs","rNodes","rlDist"]'))
    res = subprocess.run([node, str(script), str(page), ""], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_a_starting_sampler_has_no_bare_na_in_the_distribution_and_tables(tmp_path):
    """A sampler that has started but not written a round yet: dashes that say why, never n/a."""
    r = _reducer()
    r.feed({"event": "rl_sampler_start", "sampler_id": "qa-v-i0-1", "time_unix": r.max_ts - 20, "island": 0,
            "store": f"{STORE}/trainer0", "provenance": {"card_model": "H100", "engine": "vllm"}})
    r.feed({"event": "rl_sampler_join", "sampler_id": "qa-v-i0-1", "time_unix": r.max_ts - 10, "sampler": "qa-v-i0-1",
            "incarnation": "v1-x1"})
    assert _card(r, "qa-v-i0-1", r.max_ts)["status"] == "starting"
    out = _page(tmp_path, r)
    for key in ("rlDist", "aSamplers", "rNodes"):
        text = re.sub(r"<[^>]*>", " ", out[key])
        assert "n/a" not in text, key
    assert "no samples in the window yet" in out["rlDist"]


def test_the_releases_loaded_by_tags_wrap_inside_their_cell():
    css = (Path(fleet_islands.__file__).parent / "static" / "index.html").read_text()
    js = (Path(fleet_islands.__file__).parent / "static" / "app.js").read_text()
    assert re.search(r"\.lts\{display:flex;flex-wrap:wrap", css)
    assert "'<div class=\"lts\">'+sids.map(function(id){return loadTag(" in js
