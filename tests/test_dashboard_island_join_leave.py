"""Islands join and leave by themselves, as on the SFT page: an ended run shows 0
active (pool exit / lost tape follow / teardown), roles come from the records
(never from a file name), and a tape with no trainer or sampler record is marked
unrecognized instead of becoming a starting trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from yeto.dashboard.export import export_view
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

FX = Path(__file__).parent / "fixtures" / "dashboard" / "lora_ended"
T0 = 1_790_000_000.0
ACTIVE = ("ok", "waiting", "starting", "stale")


def _cards(r, now):
    return {c["id"]: c for c in r.overview(now=now)["islands"]}


@pytest.fixture()
def ended(tmp_path):
    """The real tapes of a stopped run (sanitized): a Modal trainer island known only
    from fleet.jsonl, one NC sampler mirrored by hand (its mirror lost the ssh
    follow), the NC pool tape. The launcher's teardown.json closes the run dir."""
    os.utime(FX / "run" / "teardown.json", (T0 + 3000, T0 + 3000))
    r = Reducer(run="run")
    load_all(r, [str(FX / "run"), str(FX / "mirror"), str(FX / "pool")])
    return r


def test_ended_run_has_no_active_island_and_no_fake_trainer(ended):
    now = ended.max_ts + 600
    cards = _cards(ended, now)
    assert set(cards) == {"0", "run-1"}  # no island named after the mirror file
    assert cards["run-1"]["role"] == "sampler"
    assert cards["run-1"]["status"] in ("left", "lost") and "tape source lost" in cards["run-1"]["left"]["reason"]
    assert cards["0"]["status"] == "stopped"  # teardown
    assert not [c for c in cards.values() if c["status"] in ACTIVE]
    ov = ended.overview(now=now)
    assert not [a for a in ov["alerts"] if a["rule"] in ("startup", "startup_step", "source_lost")]
    assert ov["global_status"]["level"] != "bad"


def test_ended_run_offline_export_also_ended(ended):
    view = export_view(ended, generated_at=0.0)  # now = the newest record
    cards = {c["id"]: c for c in view["overview"]["islands"]}
    assert cards["run-1"]["status"] in ("left", "lost") and cards["0"]["status"] == "stopped"


def test_source_lost_names_the_mirrored_tapes_island(tmp_path):
    tape = tmp_path / "nc-sampler-1.jsonl"
    tape.write_text(json.dumps({"event": "rl_sampler_round", "sampler_id": "s9", "rollout_id": 0, "groups": 1,
                                "generate_s": 10.0, "time_unix": T0}) + "\n")
    (tmp_path / "nc-sampler-1.jsonl.source.jsonl").write_text(json.dumps(
        {"event": "dashboard_source_lost", "island_id": "nc-sampler-1", "error": "closed", "time_unix": T0 + 5}) + "\n")
    r = Reducer()
    load_all(r, [str(tmp_path)])
    assert set(r.islands) == {"s9"}
    assert _cards(r, T0 + 10)["s9"]["status"] == "lost"


def test_tape_with_no_role_events_is_unrecognized_not_a_trainer():
    r = Reducer()
    r.feed({"event": "something_else", "island_id": "x", "time_unix": T0})
    c = _cards(r, T0 + 7200)["x"]
    assert c["role"] == "unrecognized" and c["status"] == "unknown"
    assert not [a for a in r.overview(now=T0 + 7200)["alerts"] if a["island"] == "x"]


@pytest.mark.parametrize("event,status", [("nc_elastic_released", "left"), ("nc_elastic_shard_fenced", "left"),
                                          ("nc_elastic_evicted", "lost"), ("nc_elastic_expired", "lost"),
                                          ("nc_elastic_node_lost", "lost"), ("nc_elastic_sampler_finished", "done")])
def test_pool_exit_events_end_the_shards_sampler(event, status):
    r = Reducer()
    r.feed({"t": T0, "event": "nc_elastic_member_ready", "shard": "s1", "ticket": "tk"})
    r.feed({"event": "rl_sampler_round", "sampler_id": "s1", "rollout_id": 0, "groups": 4, "generate_s": 30.0,
            "time_unix": T0 + 30})
    r.feed({"t": T0 + 40, "event": event, "shard": "s1", "ticket": "tk", "why": "test"})
    assert _cards(r, T0 + 50)["s1"]["status"] == status


def test_ticket_release_without_shard_and_pool_stopped():
    r = Reducer()
    r.feed({"t": T0, "event": "nc_elastic_ordered", "shard": "s1", "ticket": "tk"})
    r.feed({"event": "rl_sampler_round", "sampler_id": "s1", "rollout_id": 0, "groups": 4, "generate_s": 30.0,
            "time_unix": T0 + 30})
    r.feed({"t": T0 + 40, "event": "nc_elastic_released", "ticket": "tk", "why": "drain"})
    assert _cards(r, T0 + 50)["s1"]["status"] == "left"
    r2 = Reducer()
    r2.feed({"event": "rl_sampler_round", "sampler_id": "s2", "rollout_id": 0, "groups": 4, "generate_s": 30.0,
             "time_unix": T0})
    r2.feed({"t": T0 + 10, "event": "nc_elastic_pool_stopped", "why": "done"})
    assert _cards(r2, T0 + 20)["s2"]["status"] == "left"


def test_replacement_after_exit_is_alive_again():
    r = Reducer()
    r.feed({"t": T0, "event": "nc_elastic_evicted", "shard": "s1", "ticket": "old"})
    r.feed({"t": T0 + 60, "event": "nc_elastic_member_ready", "shard": "s1", "ticket": "new"})
    r.feed({"event": "rl_sampler_round", "sampler_id": "s1", "rollout_id": 5, "groups": 4, "generate_s": 30.0,
            "time_unix": T0 + 120})
    assert _cards(r, T0 + 130)["s1"]["status"] == "ok"


def test_silent_sampler_is_stale_then_lost():
    r = Reducer()
    for k in range(3):
        r.feed({"event": "rl_sampler_round", "sampler_id": "s1", "rollout_id": k, "groups": 4,
                "generate_s": 60.0, "time_unix": T0 + 60 * k})
    last = T0 + 120
    gap = _cards(r, last)["s1"]["stale_after_s"]
    assert gap >= 300 + 60  # max(3 x p50 round, 300 s floor) + slack
    assert _cards(r, last + gap - 10)["s1"]["status"] == "ok"
    assert _cards(r, last + gap + 10)["s1"]["status"] == "stale"
    assert _cards(r, last + 3 * gap + 10)["s1"]["status"] == "lost"


def test_sft_style_join_and_leave_mid_run():
    """Two samplers run; a third joins mid-run and one leaves (pool released): the
    island set and the active count follow, without a restart of the page."""
    r = Reducer()
    for k in range(4):
        for s in ("a", "b"):
            r.feed({"event": "rl_sampler_round", "sampler_id": s, "rollout_id": k, "groups": 2,
                    "generate_s": 30.0, "time_unix": T0 + 30 * k})
    assert {i for i, c in _cards(r, T0 + 100).items() if c["status"] in ACTIVE} == {"a", "b"}
    r.feed({"event": "rl_sampler_start", "sampler_id": "c", "time_unix": T0 + 110})
    r.feed({"event": "rl_sampler_round", "sampler_id": "c", "rollout_id": 0, "groups": 2, "generate_s": 30.0,
            "time_unix": T0 + 140})
    r.feed({"t": T0 + 141, "event": "nc_elastic_released", "shard": "a", "why": "drain"})
    for s in ("b", "c"):
        r.feed({"event": "rl_sampler_round", "sampler_id": s, "rollout_id": 9, "groups": 2, "generate_s": 30.0,
                "time_unix": T0 + 170})
    cards = _cards(r, T0 + 175)
    assert {i for i, c in cards.items() if c["status"] in ACTIVE} == {"b", "c"}
    assert cards["a"]["status"] == "left"
